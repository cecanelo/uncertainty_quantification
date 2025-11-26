# uncertainty_quantification/scripts/train_aux.py
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


# ------------------------- I/O and config -------------------------

def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Preprocessing (from saved base meta) -------------------------

def _to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def load_preproc_meta(meta_path: Path) -> Dict:
    meta = json.loads(meta_path.read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'. Did you point to the base run's file?")
    return meta

def build_features_from_meta(csv_path: Path, meta: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Recreate the exact feature matrix X and target vector y^t using the encoders
    and target transform stored by the base run. This keeps AuxUE strictly post‑hoc.
    """
    df = pd.read_csv(csv_path)
    if "lat" in df.columns:
        df["lat"] = df["lat"].map(_to_float)
    if "long" in df.columns:
        df["long"] = df["long"].map(_to_float)

    enc = meta["encoders"]
    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols = meta.get("onehot_cols", [])
    hash_cols = meta.get("hash_cols", [])
    hash_dims = meta.get("hash_dims", {})

    feats = []

    # numeric (standardize with saved stats)
    for c in numeric_cols:
        stats = enc["num"][c]
        mean = float(stats["mean"]); std = float(stats["std"]) if stats["std"] > 1e-12 else 1.0
        x = df[c].astype(float).to_numpy()
        feats.append(((x - mean) / std).reshape(-1, 1))

    # one-hot with saved levels
    for c in onehot_cols:
        levels = enc["oh_levels"][c]
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), len(levels)), dtype=np.float32)
        idx = {lv: i for i, lv in enumerate(levels)}
        for i, val in enumerate(s):
            j = idx.get(val)
            if j is not None:
                mat[i, j] = 1.0
        feats.append(mat)

    # hashed high-card columns
    import hashlib
    def _hash_bucket(val: str, n_features: int) -> int:
        h = hashlib.md5(val.encode("utf-8")).hexdigest()
        return int(h, 16) % n_features

    for c in hash_cols:
        n = int(hash_dims[c])
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), n), dtype=np.float32)
        for i, val in enumerate(s):
            j = _hash_bucket(f"{c}={val}", n)
            mat[i, j] = 1.0
        feats.append(mat)

    X = np.concatenate(feats, axis=1).astype(np.float32)

    # target transform from base
    target_col = "price"
    y = df[target_col].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "log1p")
    if tmode == "log1p":
        y = np.log1p(y)

    return X, y.reshape(-1, 1).astype(np.float32), {"target_mode": tmode}


# ------------------------- Predictions: schema and loading -------------------------

def read_base_preds(csv_path: Path) -> pd.DataFrame:
    """
    Expects columns:
      - row_idx  : int position in the original CSV used for the base model
      - y_true_t : target in the same transformed space as the base model trained (e.g., log1p)
      - y_pred_t : base model prediction in that space
    You can add extra columns; they will be ignored here.
    """
    df = pd.read_csv(csv_path)
    # allow fallback if user named columns slightly differently
    if "row_idx" not in df.columns:
        if "idx" in df.columns:
            df = df.rename(columns={"idx": "row_idx"})
        else:
            raise ValueError(f"{csv_path} must have a 'row_idx' column.")
    if "y_true_t" not in df.columns or "y_pred_t" not in df.columns:
        raise ValueError(f"{csv_path} must have 'y_true_t' and 'y_pred_t' columns.")
    return df[["row_idx", "y_true_t", "y_pred_t"]].copy()


# ------------------------- Binning (balanced discretization of squared residuals) -------------------------

def make_balanced_bins_sq_resid(y_true_t: np.ndarray, y_pred_t: np.ndarray, K: int) -> np.ndarray:
    """
    Return cutpoints (length K+1) for K equal-count bins based on squared residuals of TRAIN split.
    """
    sq = (y_true_t - y_pred_t) ** 2
    qs = np.linspace(0.0, 1.0, K + 1)
    cuts = np.quantile(sq, qs)
    # tiny nudges to ensure strict ordering
    eps = 1e-12
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + eps
    return cuts

def assign_bins_sq_resid(y_true_t: np.ndarray, y_pred_t: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    sq = (y_true_t - y_pred_t) ** 2
    # bins: 0..K-1
    return np.clip(np.digitize(sq, cutpoints[1:-1], right=True), 0, len(cutpoints) - 2)


# ------------------------- Models -------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], activation: str = "relu", dropout: float = 0.1, batchnorm: bool = True):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(activation.lower(), nn.ReLU)
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        if len(self.net) == 0:
            return x
        return self.net(x)

class AleatoricNet(nn.Module):
    """Predict Laplace scale b(x) > 0 for residual r = y_true_t - y_pred_t (zero-mean Laplace)."""
    def __init__(self, in_dim: int, hidden: list[int], activation: str = "relu", dropout: float = 0.1, batchnorm: bool = True):
        super().__init__()
        self.trunk = MLP(in_dim, hidden, activation, dropout, batchnorm)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        b = torch.nn.functional.softplus(self.head(h)) + 1e-6
        return b

class DidoDirichletNet(nn.Module):
    """Predict Dirichlet concentration α(x) > 0 for K bins."""
    def __init__(self, in_dim: int, hidden: list[int], K: int, activation: str = "relu", dropout: float = 0.1, batchnorm: bool = True):
        super().__init__()
        self.trunk = MLP(in_dim, hidden, activation, dropout, batchnorm)
        self.head = nn.Linear(self.trunk.out_dim, K)

    def forward(self, x):
        h = self.trunk(x)
        # α = softplus(logits) + 1 keeps α_k > 0 and avoids degeneracy near 0
        alpha = torch.nn.functional.softplus(self.head(h)) + 1.0
        return alpha


# ------------------------- Losses -------------------------

def laplace_zero_mean_nll(b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    b = torch.clamp(b, min=1e-8)
    return (torch.abs(r) / b + torch.log(2.0 * b)).mean()

def dirichlet_expected_nll(alpha: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
    """
    E_{y~Cat(π)}[-log π_k] under Dirichlet posterior α.
    For one-hot target k, this reduces to ψ(S) - ψ(α_k).
    """
    S = alpha.sum(dim=1, keepdim=True)
    digamma_S = torch.digamma(S)
    # gather α_k for each sample
    a_k = alpha.gather(1, k_idx.view(-1, 1))
    digamma_a_k = torch.digamma(a_k)
    return (digamma_S - digamma_a_k).mean()

def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(α) || Dir(1)).
    """
    K = alpha.shape[1]
    S = alpha.sum(dim=1)
    term1 = torch.lgamma(S) - torch.lgamma(torch.tensor(K, dtype=alpha.dtype, device=alpha.device))
    term2 = - torch.lgamma(alpha).sum(dim=1)  # + sum log Γ(1) = 0
    term3 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S).unsqueeze(1))).sum(dim=1)
    kl = term1 + term2 + term3
    return kl.mean()


# ------------------------- Training -------------------------

def make_split_tensors(X, y_t, preds_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns:
      X_split [N, D], r_split [N, 1], k_idx [N], idx_list [N] (row_idx)
    """
    idx = preds_df["row_idx"].astype(int).to_numpy()
    y_true_t = preds_df["y_true_t"].astype(np.float32).to_numpy().reshape(-1, 1)
    y_pred_t = preds_df["y_pred_t"].astype(np.float32).to_numpy().reshape(-1, 1)

    # Safety: check that df-based y_t and file-based y_true_t match if desired (tolerate small float noise)
    # Users can comment this out if the CSV is already the source of truth.
    if y_t is not None:
        diff = np.nanmean(np.abs(y_t[idx] - y_true_t))
        if diff > 1e-4:
            print(f"[warn] Provided y_true_t in predictions differs from CSV-derived transform by ~{diff:.4f}")

    r = (y_true_t - y_pred_t).astype(np.float32)

    Xs = torch.tensor(X[idx], dtype=torch.float32)
    rs = torch.tensor(r, dtype=torch.float32)
    return Xs, rs, idx

def build_bins_from_train(train_preds_df: pd.DataFrame, K: int) -> np.ndarray:
    cuts = make_balanced_bins_sq_resid(
        train_preds_df["y_true_t"].to_numpy().astype(np.float32),
        train_preds_df["y_pred_t"].to_numpy().astype(np.float32),
        K
    )
    return cuts

def assign_bins_for_split(preds_df: pd.DataFrame, cutpoints: np.ndarray) -> torch.Tensor:
    k = assign_bins_sq_resid(
        preds_df["y_true_t"].to_numpy().astype(np.float32),
        preds_df["y_pred_t"].to_numpy().astype(np.float32),
        cutpoints
    )
    return torch.tensor(k, dtype=torch.long)


def run_epoch(
    X: torch.Tensor, r: torch.Tensor, k_idx: torch.Tensor,
    net_ale: AleatoricNet, net_dido: DidoDirichletNet,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    lambda_kl: float, w_ale: float, batch_size: int, grad_clip: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Returns: total, nll_ale, nll_dir, kl
    """
    N = X.shape[0]
    order = torch.randperm(N, device=device) if optimizer is not None else torch.arange(N, device=device)
    total_losses = []; ale_losses = []; dir_losses = []; kl_losses = []

    net_ale.train() if optimizer is not None else net_ale.eval()
    net_dido.train() if optimizer is not None else net_dido.eval()

    for start in range(0, N, batch_size):
        idx = order[start:start + batch_size]
        xb = X[idx].to(device, non_blocking=True)
        rb = r[idx].to(device, non_blocking=True)
        kb = k_idx[idx].to(device, non_blocking=True)

        with torch.set_grad_enabled(optimizer is not None):
            b = net_ale(xb)        # [B,1]
            alpha = net_dido(xb)   # [B,K]
            loss_ale = laplace_zero_mean_nll(b, rb)
            loss_dir = dirichlet_expected_nll(alpha, kb)
            loss_kl = dirichlet_kl_to_uniform(alpha)
            loss_total = w_ale * loss_ale + loss_dir + lambda_kl * loss_kl

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(list(net_ale.parameters()) + list(net_dido.parameters()), grad_clip)
                optimizer.step()

        total_losses.append(loss_total.detach().item())
        ale_losses.append(loss_ale.detach().item())
        dir_losses.append(loss_dir.detach().item())
        kl_losses.append(loss_kl.detach().item())

    return float(np.mean(total_losses)), float(np.mean(ale_losses)), float(np.mean(dir_losses)), float(np.mean(kl_losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist config snapshot
    with open(outdir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed = int(cfg.get("seed", 42)); set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and str(cfg.get("training", {}).get("device", "cuda")).lower() == "cuda") else "cpu")

    # Load base meta and rebuild features in the same space as the base model
    base_meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = load_preproc_meta(base_meta_path)
    X, y_t, _ = build_features_from_meta(Path(cfg["data"]["csv_path"]).resolve(), meta)
    in_dim = int(X.shape[1])

    # Load base predictions per split (strictly post-hoc)
    preds_train = read_base_preds(Path(cfg["base_preds"]["train_csv"]).resolve())
    preds_val   = read_base_preds(Path(cfg["base_preds"]["val_csv"]).resolve())

    # Build K bins from TRAIN, then assign bins for train/val
    K = int(cfg["auxue"].get("K", 32))
    cutpoints = build_bins_from_train(preds_train, K)
    k_train = assign_bins_for_split(preds_train, cutpoints)
    k_val   = assign_bins_for_split(preds_val,   cutpoints)

    # Split tensors (features X, residual r)
    X_tr, r_tr, idx_tr = make_split_tensors(X, y_t, preds_train)
    X_va, r_va, idx_va = make_split_tensors(X, y_t, preds_val)

    # Models
    mcfg = cfg["model"]
    net_ale = AleatoricNet(
        in_dim=in_dim,
        hidden=mcfg.get("ale_hidden_dims", [512, 256]),
        activation=mcfg.get("activation", "relu"),
        dropout=float(mcfg.get("dropout", 0.1)),
        batchnorm=False,
    ).to(device)

    net_dido = DidoDirichletNet(
        in_dim=in_dim,
        hidden=mcfg.get("dido_hidden_dims", [512, 256]),
        K=K,
        activation=mcfg.get("activation", "relu"),
        dropout=float(mcfg.get("dropout", 0.1)),
        batchnorm=False,
    ).to(device)

    # Optimizer
    tcfg = cfg["training"]
    opt = optim.Adam(
        list(net_ale.parameters()) + list(net_dido.parameters()),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    # Training loop
    epochs = int(tcfg.get("epochs", 20))
    bs = int(tcfg.get("batch_size", 4096))
    grad_clip = float(tcfg.get("grad_clip", 0.0))

    lambda_kl = float(cfg["auxue"].get("lambda_kl", 1e-3))
    w_ale = float(cfg["auxue"].get("w_ale", 1.0))

    metrics_csv = outdir / "metrics.csv"
    write_header = not metrics_csv.exists()
    best_val = float("inf"); best_epoch = -1

    for epoch in range(1, epochs + 1):
        tr_tot, tr_ale, tr_dir, tr_kl = run_epoch(
            X_tr, r_tr, k_train, net_ale, net_dido, opt, device, lambda_kl, w_ale, bs, grad_clip
        )
        va_tot, va_ale, va_dir, va_kl = run_epoch(
            X_va, r_va, k_val, net_ale, net_dido, None, device, lambda_kl, w_ale, bs
        )

        if va_tot < best_val:
            best_val, best_epoch = va_tot, epoch

        # CSV log (Optuna pruner watches 'val_loss')
        with open(metrics_csv, "a", newline="") as mf:
            w = csv.writer(mf)
            if write_header:
                w.writerow(["epoch","split","loss","nll_ale","nll_dir","kl","val_loss","objective"])
                write_header = False
            w.writerow([epoch, "train", f"{tr_tot:.6f}", f"{tr_ale:.6f}", f"{tr_dir:.6f}", f"{tr_kl:.6f}", "", ""])
            w.writerow([epoch, "val",   f"{va_tot:.6f}", f"{va_ale:.6f}", f"{va_dir:.6f}", f"{va_kl:.6f}",
                        f"{va_tot:.6f}", f"{va_tot:.6f}"])

        print(f"[auxue] epoch {epoch}/{epochs} "
              f"train: total={tr_tot:.4f} ale={tr_ale:.4f} dir={tr_dir:.4f} kl={tr_kl:.4f} | "
              f"val: total={va_tot:.4f} ale={va_ale:.4f} dir={va_dir:.4f} kl={va_kl:.4f}")

    # Save artifacts for HPO and later inference
    with open(outdir / "metrics.json", "w") as f:
        json.dump({
            "objective_name": "val_total",
            "objective": float(best_val),
            "best_epoch": int(best_epoch),
            "val_loss": float(best_val)
        }, f, indent=4)

    with open(outdir / "auxue_cutpoints.json", "w") as f:
        json.dump({"K": K, "cutpoints": cutpoints.tolist()}, f, indent=4)

    torch.save({"state_dict": net_ale.state_dict()}, outdir / "model_ale.pt")
    torch.save({"state_dict": net_dido.state_dict()}, outdir / "model_dido.pt")


if __name__ == "__main__":
    main()
