#!/usr/bin/env python3
"""
Evaluate a trained normalizing flow on a held-out split (e.g., test).

What it does
------------
- Ensures a flow_<split>.csv exists (auto-creates via eval_regression.py when configured).
- Rebuilds features with the base run's preproc_meta and raw CSV.
- Loads the trained flow and computes per-sample log p(z | x) where
    z = (y_true_t - mu_t) / scale_t.
- Writes per-row outputs (including original-scale targets/preds) and summary metrics.

Inputs (from NF config)
-----------------------
- base_artifacts.preproc_meta_path : path to base run preproc_meta.json
- data.csv_path                    : raw CSV used in base run
- base_run.config_path             : base train config (for auto-create)
- base_run.outdir                  : base run dir with model.pt (for auto-create)
- base_preds.test_csv (optional)   : path to flow_<split>.csv; if missing and auto_create=true, it will be created
- flow_dumps.auto_create           : if true, create flow_<split>.csv when missing
- flow_dumps.output_dir            : where to save auto-created flow CSVs (default: save_dir)

CLI
---
python scripts/eval_flows.py \
  --config configs/train_flows_laplace.yaml \
  --outdir outputs/flows/laplace_nf_run \
  --split test
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from data import inverse_target

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _load_preproc_meta(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'.")
    return meta


def _to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan


def _build_features_from_meta(csv_path: Path, meta: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
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
    # numeric
    for c in numeric_cols:
        stats = enc["num"][c]
        mean = float(stats["mean"])
        std = float(stats["std"]) if stats["std"] > 1e-12 else 1.0
        x = df[c].astype(float).to_numpy()
        feats.append(((x - mean) / std).reshape(-1, 1))

    # onehot
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

    # target transform from base
    y = df[meta.get("target_col", "price")].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "log1p")
    if tmode == "log1p":
        y = np.log1p(y)

    return X, y.reshape(-1, 1).astype(np.float32), {"df": df, "y_meta": {"mode": tmode}}


def _delta_sigma_orig(head_type: str, sigma_z: np.ndarray | None, mu_orig: np.ndarray, target_meta: Dict[str, str]) -> np.ndarray:
    n = mu_orig.shape[0]
    if sigma_z is None:
        return np.full(n, np.nan, dtype=float)
    sigma_z = sigma_z.reshape(-1)
    if head_type == "point":
        return np.full(n, np.nan, dtype=float)
    mode = (target_meta or {}).get("mode", "none").lower()
    if mode == "log1p":
        return (mu_orig + 1.0) * sigma_z
    return sigma_z


def _ensure_flow_csv(split: str, cfg: dict, flow_outdir: Path) -> Path:
    base_preds = cfg.get("base_preds", {}) or {}
    key = f"{split}_csv"
    if key in base_preds:
        p = Path(base_preds[key]).resolve()
        if p.exists():
            return p
    flow_cfg = cfg.get("flow_dumps", {}) or {}
    if not flow_cfg.get("auto_create", False):
        raise FileNotFoundError(f"Flow CSV for split '{split}' not found and auto_create is False.")

    base_run = cfg.get("base_run", {}) or {}
    base_cfg_path = Path(base_run.get("config_path", ""))
    base_outdir = Path(base_run.get("outdir", ""))
    if not base_cfg_path.exists() or not base_outdir.exists():
        raise FileNotFoundError("base_run.config_path or base_run.outdir missing/invalid for auto_create")

    flow_outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_regression.py"),
        "--config",
        str(base_cfg_path),
        "--outdir",
        str(base_outdir),
        "--split",
        split,
        "--flow-dump",
        "--flow-outdir",
        str(flow_outdir),
    ]
    print(f"[eval_flows] Auto-creating flow CSV for split '{split}' via eval_regression.py")
    subprocess.run(cmd, check=True)
    p = flow_outdir / f"flow_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Expected flow dump not found at {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained flow on a held-out split")
    ap.add_argument("--config", required=True, help="NF config YAML (same schema as train_flows)")
    ap.add_argument("--outdir", required=True, help="NF run directory (contains model.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate")
    ap.add_argument("--save-dir", default=None, help="Where to write eval outputs (default: outdir/eval_<split>)")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    outdir = Path(args.outdir).resolve()
    if not (outdir / "model.pt").exists():
        raise FileNotFoundError(f"Flow checkpoint model.pt not found in {outdir}")

    meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = _load_preproc_meta(meta_path)

    data_csv = Path(cfg["data"]["csv_path"]).resolve()
    X_all, y_tr_all, extra = _build_features_from_meta(data_csv, meta)
    y_meta = extra["y_meta"]
    df = extra["df"]

    flow_outdir_cfg = cfg.get("flow_dumps", {}) or {}
    flow_outdir = Path(flow_outdir_cfg.get("output_dir") or (outdir / "flow_dumps")).resolve()
    flow_csv = _ensure_flow_csv(args.split, cfg, flow_outdir)

    df_flow = pd.read_csv(flow_csv)
    idx_col = "id" if "id" in df_flow.columns else "row_idx"
    required = {idx_col, "y_true_t", "mu_t", "scale_t", "head_type"}
    if not required.issubset(df_flow.columns):
        raise ValueError(f"{flow_csv} missing required columns {required}")

    idx = df_flow[idx_col].astype(int).to_numpy()
    y_true_t = df_flow["y_true_t"].to_numpy()
    mu_t = df_flow["mu_t"].to_numpy()
    scale_t = df_flow["scale_t"].to_numpy()
    head_type = str(df_flow["head_type"].iloc[0]).lower()

    z = (y_true_t - mu_t) / np.maximum(scale_t, 1e-8)

    Xs = torch.tensor(X_all[idx], dtype=torch.float32)
    zs = torch.tensor(z.reshape(-1, 1).astype(np.float32), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train_flows import _build_flow  # reuse constructor
    nf_cfg = cfg.get("nf", {})
    cond_dim = int(X_all.shape[1])
    flow = _build_flow(
        cond_dim=cond_dim,
        transform=nf_cfg.get("transform", "affine"),
        hidden_features=int(nf_cfg.get("hidden_features", 256)),
        num_layers=int(nf_cfg.get("num_layers", 6)),
        actnorm=bool(nf_cfg.get("actnorm", True)),
        num_bins=int(nf_cfg.get("num_bins", 8)),
    ).to(device)
    state = torch.load(outdir / "model.pt", map_location=device)
    flow.load_state_dict(state["flow_state_dict"])
    flow.eval()

    with torch.no_grad():
        xb = Xs.to(device)
        zb = zs.to(device)
        log_prob = flow.log_prob(inputs=zb, context=xb)  # [N]
        log_prob_np = log_prob.cpu().numpy().reshape(-1)
    mean_nll = float(-log_prob_np.mean())

    y_true_orig = inverse_target(y_true_t.reshape(-1, 1), y_meta).reshape(-1)
    mu_orig = inverse_target(mu_t.reshape(-1, 1), y_meta).reshape(-1)
    sigma_ale_orig = _delta_sigma_orig(head_type, scale_t, mu_orig, y_meta)

    save_dir = Path(args.save_dir).resolve() if args.save_dir else (outdir / f"eval_{args.split}")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / f"flow_eval_{args.split}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_idx",
            "split",
            "head_type",
            "y_true_orig",
            "y_pred_base_orig",
            "sigma_base_orig",
            "z",
            "log_prob_z",
        ])
        for rid, yt_o, mu_o, sig_o, z_i, lp in zip(idx, y_true_orig, mu_orig, sigma_ale_orig, z, log_prob_np):
            writer.writerow([
                int(rid),
                args.split,
                head_type,
                float(yt_o),
                float(mu_o),
                float(sig_o) if np.isfinite(sig_o) else np.nan,
                float(z_i),
                float(lp),
            ])

    metrics = {
        "split": args.split,
        "n": int(len(idx)),
        "mean_nll": mean_nll,
        "head_type": head_type,
    }
    with (save_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[eval_flows] split={args.split} n={len(idx)} mean_nll={mean_nll:.4f}")
    print(f"[eval_flows] wrote rows to: {out_csv}")


if __name__ == "__main__":
    main()
