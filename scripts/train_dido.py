#!/usr/bin/env python3
"""
Train a post-hoc Dirichlet AuxUE (DIDO) on residual bins of a base regressor.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from data import DataConfig, _prepare_frame


# ---------------------- small utilities ----------------------
def save_yaml(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _load_preproc_meta(meta_path: Path) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'.")
    return meta


def _build_features_from_meta(csv_path: Path, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    dc = DataConfig(
        csv_path=str(csv_path),
        target_col=meta.get("target_col", "price"),
        target_transform=meta.get("target", {}).get("mode", "none"),
    )
    df = _prepare_frame(dc)

    enc = meta["encoders"]
    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols = meta.get("onehot_cols", [])
    hash_cols = meta.get("hash_cols", [])
    hash_dims = meta.get("hash_dims", {})

    feats: List[np.ndarray] = []

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

    # hashed
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

    y = df[dc.target_col].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "none")
    if tmode == "log1p":
        y = np.log1p(y)

    return X, y.reshape(-1, 1).astype(np.float32)


def _read_base_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"y_true", "y_pred_det"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {needed}")
    if "id" not in df.columns and "row_index" not in df.columns and "row_idx" not in df.columns:
        raise ValueError(f"{path} must include an id/row_index/row_idx column.")
    return df


def _fit_bins(residuals: np.ndarray, K: int) -> np.ndarray:
    qs = np.linspace(0, 1, K + 1)
    edges = np.quantile(residuals, qs)
    if np.unique(edges).size < K + 1:
        lo, hi = residuals.min(), residuals.max()
        edges = np.linspace(lo, hi, K + 1)
    return edges


def _assign_bins(residuals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # np.digitize with interior edges
    bins = np.digitize(residuals, edges[1:-1], right=False)
    # clamp just in case numerical drift
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins.astype(int)


class DirichletNet(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else (nn.GELU if activation.lower() == "gelu" else nn.Tanh)
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return self.softplus(raw) + 1e-6  # ensure positivity


def dirichlet_categorical_nll(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target: (B,) int64
    sum_alpha = alpha.sum(dim=1)
    idx = alpha[torch.arange(alpha.size(0)), target]
    return torch.log(sum_alpha) - torch.log(idx)


def submit_slurm(cfg_path: Path, outdir_raw: str, cfg: dict) -> None:
    slurm = cfg.get("slurm", {}) or {}
    partition = slurm.get("partition", "TEST")
    time_str = slurm.get("time", "01:00:00")
    mem_gb = slurm.get("mem_gb", 30)
    cpus = slurm.get("cpus", 2)
    gpus = slurm.get("gpus", 0)
    job_name_cfg = slurm.get("job_name", "dido")
    conda_env = slurm.get("conda_env", "thesis")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = job_name_cfg
    gres_line = f"#SBATCH --gres=gpu:{gpus}" if gpus and gpus > 0 else ""

    outdir_str = (
        str(outdir_raw)
        .replace("{ts}", ts)
        .replace("{jobid}", "${SLURM_JOB_ID}")
        .replace("{job}", job_name)
    )

    repo_root = Path(__file__).resolve().parents[1]
    logs_root = (repo_root / "logs" / "train").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    # Keep log names aligned with the configured job_name; avoid double "dido_" prefixes
    log_stub = f"{job_name}_{ts}" if job_name else f"dido_{ts}"
    out_log = logs_root / f"{log_stub}_%j.out"
    err_log = logs_root / f"{log_stub}_%j.err"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
{gres_line}
#SBATCH --time={time_str}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

cd "{repo_root}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate {conda_env}
echo "[env] host=$(hostname) date=$(date)"
python scripts/train_dido.py --config "{cfg_path}" --outdir "{outdir_str}" --mode local
"""
    print("[train_dido][slurm] sbatch script:\n")
    print(script)
    res = subprocess.run(["sbatch"], input=script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def _ensure_preds(split: str, base_preds_cfg: dict, auto_create: bool, base_cfg_path: Path, base_model_dir: Path) -> Path:
    key = f"{split}_csv"
    if key in base_preds_cfg and base_preds_cfg[key]:
        p = Path(base_preds_cfg[key]).resolve()
        if p.exists():
            return p
    if not auto_create:
        raise FileNotFoundError(f"Base preds for split '{split}' not found and auto_create=False")
    if base_cfg_path is None or base_model_dir is None:
        raise FileNotFoundError("base_run.config_path or base_run.model_dir missing for auto_create")
    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    evals_root = base_cfg.get("io", {}).get("evals_root", "outputs/evals")
    # Prefer the actual model_dir name (matches eval_regression convention), fall back to job_name
    run_tag = base_model_dir.name if base_model_dir is not None else base_cfg.get("slurm", {}).get("job_name", "run")

    # run eval_regression to generate preds_<split>.csv
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "eval_regression.py"),
        "--config",
        str(base_cfg_path),
        "--outdir",
        str(base_model_dir),
        "--split",
        split,
    ]
    print(f"[dido] Auto-creating base preds for split '{split}' via eval_regression.py")
    subprocess.run(cmd, check=True)
    preds_path = Path(evals_root) / run_tag / f"preds_{split}.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Expected preds at {preds_path} after auto-create.")
    return preds_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--outdir", required=False, help="Output directory (default: io.outdir in config)")
    ap.add_argument("--mode", choices=["local", "slurm"], default="local", help="local or slurm submission")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_outdir = args.outdir or cfg.get("io", {}).get("outdir")
    if not raw_outdir:
        raise SystemExit("Please specify an outdir via --outdir or io.outdir in the config.")

    if args.mode == "slurm":
        submit_slurm(Path(args.config).resolve(), raw_outdir, cfg)
        return

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobid_env = os.environ.get("SLURM_JOB_ID", "NA")
    job_name = cfg.get("slurm", {}).get("job_name", "dido")
    outdir = Path(
        str(raw_outdir)
        .replace("{ts}", ts)
        .replace("{jobid}", jobid_env)
        .replace("{job}", job_name)
    ).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_time = time.perf_counter()
    start_iso = datetime.utcnow().isoformat() + "Z"

    base_meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = _load_preproc_meta(base_meta_path)

    X_all, _ = _build_features_from_meta(Path(cfg["data"]["csv_path"]).resolve(), meta)
    feature_dim = X_all.shape[1]

    # Map external IDs (stored in preproc_meta.json) back to row indices so we can
    # index into X_all even when IDs are not 0..N-1.
    id_to_row_index = None
    if "id_values" in meta:
        id_values = np.asarray(meta["id_values"], dtype=int)
        id_to_row_index = {int(value): int(idx) for idx, value in enumerate(id_values)}

    base_preds_cfg = cfg.get("base_preds", {}) or {}
    auto_create = bool(base_preds_cfg.get("auto_create", False))
    base_run_cfg = cfg.get("base_run", {}) or {}
    base_cfg_path = Path(base_run_cfg.get("config_path", "")) if base_run_cfg.get("config_path") else None
    base_model_dir = Path(base_run_cfg.get("model_dir", "")) if base_run_cfg.get("model_dir") else None

    preds_train_path = _ensure_preds("train", base_preds_cfg, auto_create, base_cfg_path, base_model_dir)
    preds_val_path = _ensure_preds("val", base_preds_cfg, auto_create, base_cfg_path, base_model_dir)

    preds_train = _read_base_preds(preds_train_path)
    preds_val = _read_base_preds(preds_val_path)

    def _get_ids(df: pd.DataFrame) -> np.ndarray:
        """
        Return row indices into X_all for the given predictions frame.
        Prefer explicit row index columns, but fall back to mapping external IDs
        via preproc_meta.id_values when only an 'id' column is present.
        """
        if "row_idx" in df.columns:
            return df["row_idx"].to_numpy(dtype=int)
        if "row_index" in df.columns:
            return df["row_index"].to_numpy(dtype=int)
        if "id" in df.columns:
            if id_to_row_index is None:
                raise ValueError(
                    "Base prediction file contains 'id' but preproc_meta.json is "
                    "missing 'id_values'; cannot map IDs to row indices."
                )
            ids = df["id"].to_numpy()
            row_indices: List[int] = []
            for raw_id in ids:
                key = int(raw_id)
                if key not in id_to_row_index:
                    raise IndexError(
                        f"ID {key} from base predictions not found in preproc_meta.id_values."
                    )
                row_indices.append(id_to_row_index[key])
            return np.asarray(row_indices, dtype=int)
        raise ValueError(
            "Base prediction file must contain one of 'row_idx', 'row_index', or 'id' columns."
        )

    y_true_tr = preds_train["y_true"].to_numpy()
    y_pred_tr = preds_train["y_pred_det"].to_numpy()
    y_true_va = preds_val["y_true"].to_numpy()
    y_pred_va = preds_val["y_pred_det"].to_numpy()

    use_abs = bool(cfg.get("binning", {}).get("use_abs_error", True))
    include_mu = bool(cfg.get("binning", {}).get("include_mu_as_feature", True))
    res_tr = np.abs(y_true_tr - y_pred_tr) if use_abs else (y_true_tr - y_pred_tr)
    res_va = np.abs(y_true_va - y_pred_va) if use_abs else (y_true_va - y_pred_va)

    K = int(cfg.get("binning", {}).get("K", 10))
    edges = _fit_bins(res_tr, K)
    bins_tr = _assign_bins(res_tr, edges)
    bins_va = _assign_bins(res_va, edges)

    ids_tr = _get_ids(preds_train).astype(int)
    ids_va = _get_ids(preds_val).astype(int)

    X_tr_full = X_all[ids_tr]
    X_va_full = X_all[ids_va]

    if include_mu:
        X_tr_full = np.concatenate([X_tr_full, y_pred_tr.reshape(-1, 1).astype(np.float32)], axis=1)
        X_va_full = np.concatenate([X_va_full, y_pred_va.reshape(-1, 1).astype(np.float32)], axis=1)
        feature_dim = X_tr_full.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("training", {}).get("device", "cuda") == "cuda" else "cpu")
    # Keep base tensors on CPU; move to device inside the training loop to avoid CUDA issues with DataLoader workers.
    X_tr_t = torch.tensor(X_tr_full, dtype=torch.float32)
    X_va_t = torch.tensor(X_va_full, dtype=torch.float32)
    y_tr_bins = torch.tensor(bins_tr, dtype=torch.long)
    y_va_bins = torch.tensor(bins_va, dtype=torch.long)

    bs = int(cfg.get("training", {}).get("batch_size", 1024))
    nw = int(cfg.get("training", {}).get("num_workers", 2))
    pm = bool(cfg.get("training", {}).get("pin_memory", True))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_bins),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_va_t, y_va_bins),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
    )

    model_cfg = cfg.get("model", {})
    hidden = model_cfg.get("hidden_dims", [256, 128])
    act = model_cfg.get("activation", "relu")
    dropout = float(model_cfg.get("dropout", 0.0))

    model = DirichletNet(feature_dim, hidden, K, activation=act, dropout=dropout).to(device)
    opt = optim.Adam(
        model.parameters(),
        lr=float(cfg.get("training", {}).get("lr", 1e-3)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.0)),
    )

    epochs = int(cfg.get("training", {}).get("epochs", 20))
    patience = int(cfg.get("training", {}).get("patience", 5))
    eval_after_train = bool(cfg.get("training", {}).get("eval_after_train", False))

    csv_path = outdir / "metrics.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "split", "nll"])

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    def eval_loader(loader, train: bool) -> float:
        losses = []
        model.train(train)
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                alpha = model(xb)
                loss = dirichlet_categorical_nll(alpha, yb).mean()
                if train:
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    for epoch in range(1, epochs + 1):
        train_nll = eval_loader(train_loader, True)
        val_nll = eval_loader(val_loader, False)

        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train", train_nll])
            w.writerow([epoch, "val", val_nll])

        print(f"[dido] epoch {epoch}/{epochs} train_nll={train_nll:.4f} val_nll={val_nll:.4f}")

        if val_nll < best_val - 1e-6:
            best_val = val_nll
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"[dido] Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), outdir / "model.pt")

    end_time = time.perf_counter()
    duration = end_time - start_time

    metrics = {
        "best_val_nll": best_val,
        "best_epoch": best_epoch,
        "train_nll_last": train_nll,
        "val_nll_last": val_nll,
        # HPO expects an objective field; use validation NLL (lower is better)
        "objective": best_val,
        "val_loss": best_val,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    bin_summary = {
        "edges": edges.tolist(),
        "counts_train": np.bincount(bins_tr, minlength=K).tolist(),
        "counts_val": np.bincount(bins_va, minlength=K).tolist(),
    }
    (outdir / "bin_summary.json").write_text(json.dumps(bin_summary, indent=2))

    trial_number = os.environ.get("OPTUNA_TRIAL_NUMBER")

    run_meta = {
        "head_type": "dido",
        "binning": {
            "K": K,
            "use_abs_error": use_abs,
            "include_mu_as_feature": include_mu,
            "edges": edges.tolist(),
        },
        "base_run": {
            "config_path": str(base_cfg_path) if base_cfg_path else None,
            "model_dir": str(base_model_dir) if base_model_dir else None,
            "preds_train": str(preds_train_path),
            "preds_val": str(preds_val_path),
        },
        "feature_dim": feature_dim,
        "timing": {
            "start_utc": start_iso,
            "duration_sec": duration,
        },
        "optuna": {"trial_number": trial_number} if trial_number is not None else {},
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    # Save resolved config for provenance
    resolved_cfg_path = outdir / "used_config.yaml"
    save_yaml(cfg, resolved_cfg_path)

    print(f"[dido] Saved model and metrics to: {outdir}")

    # Optional evaluation hook
    if eval_after_train:
        try:
            print("[dido] Running eval_dido.py after training...")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[0] / "eval_dido.py"),
                    "--dido-outdir",
                    str(outdir),
                    "--split",
                    "test",
                ],
                check=True,
            )
        except Exception as e:
            print(f"[dido] eval_dido.py failed: {e}")


if __name__ == "__main__":
    main()
