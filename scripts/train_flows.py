# uncertainty_quantification/scripts/train_flows.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml

# ----------------------------- util -----------------------------
def _load_cfg(p: str | Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

# ----------------------- flow construction ----------------------
def _build_flow(cond_dim: int,
                transform: str = "affine",
                hidden_features: int = 256,
                num_layers: int = 6,
                actnorm: bool = True,
                num_bins: int = 8):
    """
    Conditional flow p(z | x) for scalar z with x as context.
    - transform: "affine" (MAF) or "spline" (RQs spline-AR)
    - features=1 because z is 1D
    Notes:
      * No permutations are inserted since D=1.
      * ActNorm can stabilize deeper stacks.
    """
    try:
        from nflows.flows import Flow
        from nflows.distributions import StandardNormal
        from nflows.transforms import CompositeTransform
        from nflows.transforms.autoregressive import (
            MaskedAffineAutoregressiveTransform,
        )
        from nflows.transforms.spline import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform as SplineAR,
        )
        from nflows.transforms.normalization import ActNorm as NFActNorm
    except ImportError as e:
        raise SystemExit(
            "train_flows.py needs `nflows`. Install it, e.g.: pip install nflows"
        ) from e

    layers = []
    for _ in range(num_layers):
        if transform.lower() == "spline":
            layers.append(
                SplineAR(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=cond_dim,
                    num_bins=num_bins,
                    tails="linear",
                )
            )
        else:
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=cond_dim,
                )
            )
        if actnorm:
            layers.append(NFActNorm(features=1))

    transform = CompositeTransform(layers)
    base = StandardNormal([1])
    return Flow(transform, base)

# ---------------------- post-hoc data helpers -------------------
def _load_preproc_meta(meta_path: Path) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'.")
    return meta

def _to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def _build_features_from_meta(csv_path: Path, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "lat" in df.columns:  df["lat"]  = df["lat"].map(_to_float)
    if "long" in df.columns: df["long"] = df["long"].map(_to_float)

    enc = meta["encoders"]
    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols  = meta.get("onehot_cols", [])
    hash_cols    = meta.get("hash_cols", [])
    hash_dims    = meta.get("hash_dims", {})

    feats = []

    # numeric
    for c in numeric_cols:
        stats = enc["num"][c]
        mean = float(stats["mean"])
        std = float(stats["std"]) if stats["std"] > 1e-12 else 1.0
        x = df[c].astype(float).to_numpy()
        feats.append(((x - mean) / std).reshape(-1, 1))

    # one-hot
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
    for c in hash_cols:
        n = int(hash_dims[c])
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), n), dtype=np.float32)
        for i, val in enumerate(s):
            h = hashlib.md5(f"{c}={val}".encode("utf-8")).hexdigest()
            j = int(h, 16) % n
            mat[i, j] = 1.0
        feats.append(mat)

    X = np.concatenate(feats, axis=1).astype(np.float32)

    # target transform (consistent with base run)
    y = df["price"].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "log1p")
    if tmode == "log1p":
        y = np.log1p(y)

    return X, y.reshape(-1, 1).astype(np.float32)

def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_idx" not in df.columns or "y_true_t" not in df.columns:
        raise ValueError(f"{path} must include 'row_idx' and 'y_true_t' columns.")
    return df

def _build_split_tensors(role: str,
                         X_all: np.ndarray,
                         df_preds: pd.DataFrame,
                         standardize: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build tensors for a split:
      X_split: features from base run encoders
      z_split: residual target, role-dependent
    """
    idx = df_preds["row_idx"].astype(int).to_numpy()

    if role == "aleatoric":
        required = {"mu_nll_t", "sigma_nll_t"}
        if not required.issubset(df_preds.columns):
            raise ValueError(f"aleatoric flow needs columns {required} in predictions CSV.")
        z = df_preds["y_true_t"].to_numpy() - df_preds["mu_nll_t"].to_numpy()
        if standardize:
            z = z / np.maximum(df_preds["sigma_nll_t"].to_numpy(), 1e-8)

    elif role == "epistemic":
        required = {"mu_mc_t", "sigma_nll_t"}
        if not required.issubset(df_preds.columns):
            raise ValueError(f"epistemic flow needs columns {required} in predictions CSV.")
        z = df_preds["y_true_t"].to_numpy() - df_preds["mu_mc_t"].to_numpy()
        if standardize:
            z = z / np.maximum(df_preds["sigma_nll_t"].to_numpy(), 1e-8)

    else:
        raise ValueError("nf.role must be 'aleatoric' or 'epistemic'.")

    Xs = torch.tensor(X_all[idx], dtype=torch.float32)
    zs = torch.tensor(z.reshape(-1, 1).astype(np.float32), dtype=torch.float32)
    return Xs, zs

# ----------------------------- train ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = str(cfg.get("training", {}).get("device", "cuda")).lower()
    device = torch.device("cuda" if (torch.cuda.is_available() and device_str == "cuda") else "cpu")

    # --- Rebuild features with base encoders, load predictions per split
    base_meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = _load_preproc_meta(base_meta_path)

    X_all, _ = _build_features_from_meta(Path(cfg["data"]["csv_path"]).resolve(), meta)
    cond_dim = int(X_all.shape[1])

    role = str(cfg["nf"].get("role", "aleatoric")).lower()
    standardize = bool(cfg["nf"].get("standardize", True))

    preds_train = _read_preds(Path(cfg["base_preds"]["train_csv"]).resolve())
    preds_val   = _read_preds(Path(cfg["base_preds"]["val_csv"]).resolve())

    X_tr, z_tr = _build_split_tensors(role, X_all, preds_train, standardize)
    X_va, z_va = _build_split_tensors(role, X_all, preds_val,   standardize)

    from torch.utils.data import TensorDataset, DataLoader
    bs = int(cfg.get("training", {}).get("batch_size", 1024))
    nw = int(cfg.get("training", {}).get("num_workers", 4))
    pm = bool(cfg.get("training", {}).get("pin_memory", True))

    train_loader = DataLoader(TensorDataset(X_tr, z_tr), batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(TensorDataset(X_va, z_va), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    # --- Build flow
    nf_cfg = cfg.get("nf", {})
    flow = _build_flow(
        cond_dim=cond_dim,
        transform=nf_cfg.get("transform", "affine"),
        hidden_features=int(nf_cfg.get("hidden_features", 256)),
        num_layers=int(nf_cfg.get("num_layers", 6)),
        actnorm=bool(nf_cfg.get("actnorm", True)),
        num_bins=int(nf_cfg.get("num_bins", 8)),
    ).to(device)

    opt = optim.Adam(
        flow.parameters(),
        lr=float(cfg.get("training", {}).get("lr", 5e-4)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.0)),
    )
    epochs = int(cfg.get("training", {}).get("epochs", 50))

    # --- CSV logging compatible with Optuna worker pruning
    csv_path = outdir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "split", "nll", "val_loss", "objective"])

    def _nll_epoch(loader, train: bool) -> float:
        losses = []
        if train:
            flow.train()
        else:
            flow.eval()
        with torch.set_grad_enabled(train):
            for xb, zb in loader:
                xb = xb.to(device, non_blocking=True)
                zb = zb.to(device, non_blocking=True)
                log_prob = flow.log_prob(inputs=zb, context=xb)  # log p(z | x)
                loss = -log_prob.mean()
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    best_val = float("inf")
    best_state = None

    for e in range(1, epochs + 1):
        tr = _nll_epoch(train_loader, True)
        va = _nll_epoch(val_loader, False)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in flow.state_dict().items()}

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([e, "train", f"{tr:.6f}", "", ""])
            w.writerow([e, "val",   f"{va:.6f}", f"{va:.6f}", f"{va:.6f}"])

        print(f"[nf-{role}] epoch {e}/{epochs} train_nll={tr:.4f} val_nll={va:.4f}")

    # persist best
    if best_state is not None:
        torch.save({"flow_state_dict": best_state}, outdir / "model.pt")

    # HPO-readable metrics
    with open(outdir / "metrics.json", "w") as f:
        json.dump(
            {"objective_name": "val_nll", "objective": float(best_val), "val_loss": float(best_val)},
            f, indent=4,
        )

    # Keep the exact config used + quick run meta
    _save_yaml(cfg, outdir / "used_config.yaml")
    run_meta = {
        "role": role,
        "cond_dim": int(cond_dim),
        "nf": {
            "transform": nf_cfg.get("transform", "affine"),
            "num_layers": int(nf_cfg.get("num_layers", 6)),
            "hidden_features": int(nf_cfg.get("hidden_features", 256)),
            "actnorm": bool(nf_cfg.get("actnorm", True)),
            "num_bins": int(nf_cfg.get("num_bins", 8)),
        },
        "device": str(device),
        "seed": int(seed),
    }
    with open(outdir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=4)

if __name__ == "__main__":
    main()
