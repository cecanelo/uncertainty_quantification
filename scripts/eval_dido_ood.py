#!/usr/bin/env python3
"""
Evaluate a trained DIDO AuxUE model on an OOD CSV.

This mirrors eval_dido.py but:
  * takes an OOD CSV instead of a split name,
  * auto-creates base predictions on that OOD CSV via eval_regression_ood.py,
  * builds features with the same preproc_meta.json as in training,
  * runs the trained DIDO network to produce vacuity/entropy/strength.

Example:

  python scripts/eval_dido_ood.py \\
    --dido-outdir outputs/trainings/training_dido_lpl_m/dido_lpl_m_... \\
    --ood-csv datasets/craigslist_ood_geo_fl_tx.csv \\
    --dataset-label geo
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml

from data import DataConfig, _prepare_frame
# Allow both package and script-level import
try:
    from .config_resolver import load_and_resolve_config  # type: ignore
except Exception:
    from config_resolver import load_and_resolve_config  # type: ignore
from eval_dido import _load_preproc_meta, DirichletNet


def _build_features_from_meta_ood(csv_path: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build features for an OOD CSV using the same preprocessing pipeline
    (DataConfig + _prepare_frame + encoders) as in eval_dido/train_dido.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a DIDO model on an OOD CSV.")
    ap.add_argument("--dido-outdir", required=True, help="Directory containing trained DIDO model.pt")
    ap.add_argument("--ood-csv", required=True, help="OOD CSV to evaluate")
    ap.add_argument("--dataset-label", required=True, help="Dataset label (used in filenames)")
    args = ap.parse_args()

    dido_dir = Path(args.dido_outdir).resolve()
    cfg_path = dido_dir / "used_config.yaml"
    run_meta_path = dido_dir / "run_meta.json"
    model_path = dido_dir / "model.pt"

    if not cfg_path.exists() or not model_path.exists():
        raise SystemExit("DIDO run is missing used_config.yaml or model.pt")

    cfg = load_and_resolve_config(cfg_path)
    run_meta = json.loads(run_meta_path.read_text()) if run_meta_path.exists() else {}

    bin_cfg = run_meta.get("binning", cfg.get("binning", {}))
    K = int(bin_cfg.get("K", 10))
    include_mu = bool(
        bin_cfg.get(
            "include_mu_as_feature",
            cfg.get("binning", {}).get("include_mu_as_feature", True),
        )
    )

    base_meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = _load_preproc_meta(base_meta_path)

    ood_csv = Path(args.ood_csv).resolve()
    if not ood_csv.exists():
        raise FileNotFoundError(f"OOD CSV not found: {ood_csv}")

    # Build OOD features using the same preprocessing as the base/DIDO training.
    X_ood, _ = _build_features_from_meta_ood(ood_csv, meta)

    # Create base predictions on this OOD CSV via eval_regression_ood.py.
    base_run = cfg.get("base_run", {}) or {}
    base_cfg_path = Path(base_run.get("config_path", "")).expanduser().resolve()
    base_model_dir = Path(base_run.get("model_dir", "")).expanduser().resolve()
    if not base_cfg_path.exists() or not base_model_dir.exists():
        raise SystemExit("DIDO config base_run.config_path or base_run.model_dir is missing/invalid.")

    base_cfg = load_and_resolve_config(base_cfg_path)
    evals_root = base_cfg.get("io", {}).get("evals_root", "outputs/evals")
    run_tag_base = base_model_dir.name

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "eval_regression_ood.py"),
        "--config",
        str(base_cfg_path),
        "--outdir",
        str(base_model_dir),
        "--ood-csv",
        str(ood_csv),
        "--dataset-label",
        args.dataset_label,
    ]
    print(f"[dido-ood] Running eval_regression_ood.py for base preds on OOD set '{args.dataset_label}'")
    subprocess.run(cmd, check=True)

    preds_path = Path(evals_root) / run_tag_base / f"ood_{args.dataset_label}_test.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Base OOD preds not found at {preds_path}")

    preds_df = pd.read_csv(preds_path)
    if "y_pred_det" not in preds_df.columns:
        raise ValueError(f"{preds_path} missing y_pred_det column.")
    mu_pred = preds_df["y_pred_det"].to_numpy()

    if X_ood.shape[0] != mu_pred.shape[0]:
        raise ValueError(
            f"Mismatch between OOD features (n={X_ood.shape[0]}) and base preds (n={mu_pred.shape[0]})."
        )

    # IDs to write out: prefer 'id' column from OOD preds, else row index
    if "id" in preds_df.columns:
        ids_out = preds_df["id"].to_numpy()
    else:
        ids_out = np.arange(X_ood.shape[0], dtype=int)

    if include_mu:
        X_ood = np.concatenate([X_ood, mu_pred.reshape(-1, 1).astype(np.float32)], axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    hidden = model_cfg.get("hidden_dims", [256, 128])
    act = model_cfg.get("activation", "relu")
    dropout = float(model_cfg.get("dropout", 0.0))
    model = DirichletNet(X_ood.shape[1], hidden, K, activation=act, dropout=dropout).to(device)

    # Be robust to accidental mis-pointing at a non-DIDO run (e.g., an NF
    # model with a flow_state_dict). In that case we skip this run but keep
    # the overall batch alive.
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "flow_state_dict" in state and "net.0.weight" not in state:
        print(
            f"[dido-ood][warn] model.pt at {model_path} appears to be a flow checkpoint "
            "(contains 'flow_state_dict'); skipping this DIDO OOD evaluation."
        )
        return

    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        alpha = model(torch.tensor(X_ood, dtype=torch.float32, device=device))
        strength = alpha.sum(dim=1)
        mean_probs = alpha / strength[:, None]
        strength_np = strength.cpu().numpy()
        mean_probs_np = mean_probs.cpu().numpy()
        entropy = -np.sum(mean_probs_np * np.log(mean_probs_np + 1e-12), axis=1)
        vacuity = K / strength_np

    # Determine base head type (laplace/gauss) as in eval_dido.py
    base_head_type = run_meta.get("base_head_type")
    if not base_head_type:
        base_cfg_local = base_cfg
        base_head_type = (base_cfg_local.get("model", {}) or {}).get("head_type", None)
    if not base_head_type:
        base_head_type = "laplace"

    out_root = Path(cfg.get("io", {}).get("evals_root", "outputs/evals"))
    run_tag_dido = dido_dir.name
    eval_dir = out_root / run_tag_dido
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_out = eval_dir / f"dido_ood_{args.dataset_label}.csv"

    with preds_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "split", "head_type", "dido_strength_raw", "dido_vacuity_raw", "dido_entropy_raw"]
        )
        for raw_id, s, v, e in zip(ids_out, strength_np, vacuity, entropy):
            w.writerow([int(raw_id), "test", base_head_type, float(s), float(v), float(e)])

    metrics = {
        "split": "test",
        "dataset": args.dataset_label,
        "mean_strength": float(np.mean(strength_np)),
        "mean_vacuity": float(np.mean(vacuity)),
        "mean_entropy": float(np.mean(entropy)),
    }
    (eval_dir / f"metrics_ood_{args.dataset_label}.json").write_text(json.dumps(metrics, indent=2))

    print(f"[dido-ood] Saved OOD scores to: {preds_out}")


if __name__ == "__main__":
    main()
