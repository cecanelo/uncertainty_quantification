#!/usr/bin/env python3
"""
Evaluate a trained base regression model on one or more OOD CSVs.

Typical usage (single OOD set):

    python scripts/eval_regression_ood.py \\
        --config outputs/trainings/training_lpl_xs/lpl_xs_<ts>/used_config.yaml \\
        --outdir outputs/trainings/training_lpl_xs/lpl_xs_<ts> \\
        --ood-csv datasets/craigslist_ood_geo_fl_tx.csv \\
        --dataset-label geo

Evaluate all *_ood_*.csv files in a directory:

    python scripts/eval_regression_ood.py \\
        --config .../used_config.yaml \\
        --outdir ... \\
        --ood-dir datasets

In this case dataset labels are derived from the filename segment after
\"_ood_\" (e.g. craigslist_ood_geo_fl_tx.csv -> dataset_label=\"geo_fl_tx\").
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from data import DataConfig, _prepare_frame, _apply_encoders, _target_transform, inverse_target
from model_base import MLPRegressor, gaussian_nll, laplace_nll
from train_regression import _metrics_from_batches, _load_cfg


def _load_meta(outdir: Path) -> Dict[str, Any]:
    meta_path = outdir / "preproc_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"preproc_meta.json not found in {outdir}")
    with meta_path.open("r") as f:
        return json.load(f)


def _delta_sigma_orig(
    head_type: str,
    sigma_z: Optional[np.ndarray],
    mu_orig: np.ndarray,
    target_meta: Dict[str, str],
) -> np.ndarray:
    """
    Map transformed-space scale to an approximate original-space std via the delta method.
    Mirrors eval_regression._delta_sigma_orig for Gauss/Laplace heads.
    """
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


def _eval_single_ood(
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    outdir: Path,
    ood_csv: Path,
    dataset_label: str,
) -> None:
    """
    Run the base model on a single OOD CSV and write ood_<label>_test.csv
    under the usual evals_root/run_tag directory.
    """
    # --- 1) Build OOD frame using same schema / transform as training ---
    data_cfg = cfg["data"]
    dc = DataConfig(
        csv_path=str(ood_csv),
        target_col=data_cfg.get("target_col", "price"),
        target_transform=data_cfg.get("target_transform", "log1p"),
    )
    df = _prepare_frame(dc)

    numeric_cols: List[str] = meta["numeric_cols"]
    onehot_cols: List[str] = meta["onehot_cols"]
    hash_cols: List[str] = meta["hash_cols"]
    enc: Dict[str, Any] = meta["encoders"]

    y = df[dc.target_col].astype(float).to_numpy()
    y_tr, y_meta = _target_transform(y, dc.target_transform)
    X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)

    idx = np.arange(len(df), dtype=int)
    if idx.size == 0:
        raise ValueError(f"OOD CSV {ood_csv} has zero usable rows.")

    X_split = torch.tensor(X[idx], dtype=torch.float32)
    y_split = torch.tensor(y_tr[idx], dtype=torch.float32).view(-1, 1)

    # --- 2) Rebuild model and load checkpoint ---
    device = torch.device("cpu")
    in_dim = int(meta["feature_dim"])
    model_cfg = cfg["model"]
    head_type = model_cfg.get("head_type", "point").lower()
    hidden = model_cfg.get("hidden_dims", [512, 256, 128])

    model = MLPRegressor(
        in_dim=in_dim,
        hidden_dims=hidden,
        head_type=head_type,
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_batchnorm=False,
    ).to(device)

    ckpt_path = outdir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint model.pt not found in {outdir}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # --- 3) Inference on all OOD rows ---
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    nll_vals: List[float] = []
    scales: List[np.ndarray] = []

    with torch.no_grad():
        xb = X_split.to(device)
        yb = y_split.to(device)
        out = model(xb)

        mu = out["mu"]
        preds.append(mu.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

        if head_type == "gauss":
            sigma = out["sigma"]
            nll = gaussian_nll(mu, sigma, yb)
            nll_vals.append(float(nll.item()))
            scales.append(sigma.detach().cpu().numpy())
        elif head_type == "laplace":
            b = out["b"]
            nll = laplace_nll(mu, b, yb)
            nll_vals.append(float(nll.item()))
            scales.append(b.detach().cpu().numpy())

    metrics = _metrics_from_batches(
        preds,
        targets,
        head_type,
        nll_values=nll_vals,
        scales=scales if scales else None,
    )

    scale_concat: Optional[np.ndarray] = None
    if head_type in ("gauss", "laplace") and scales:
        scale_concat = np.concatenate(scales, axis=0).reshape(-1)

    mu_concat = np.concatenate(preds, axis=0).reshape(-1)
    yt_concat = np.concatenate(targets, axis=0).reshape(-1)

    mu_orig = inverse_target(mu_concat.reshape(-1, 1), y_meta).reshape(-1)
    yt_orig = inverse_target(yt_concat.reshape(-1, 1), y_meta).reshape(-1)

    ae_orig = np.abs(mu_orig - yt_orig)
    se_orig = (mu_orig - yt_orig) ** 2
    mae_orig = float(np.mean(ae_orig))
    rmse_orig = float(np.sqrt(np.mean(se_orig)))

    print(f"[ood-eval] csv={ood_csv.name} label={dataset_label} n={idx.size} head_type={head_type}")
    print(f"[ood-eval] MAE_orig={mae_orig:.2f}  RMSE_orig={rmse_orig:.2f}")

    # --- 4) Save per-instance predictions ---
    evals_root = cfg.get("io", {}).get("evals_root", "outputs/evals")
    evals_root = Path(evals_root)
    run_tag = outdir.name
    eval_dir = evals_root / run_tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    # IDs for OOD: use 'id' column if present, else row_index
    if "id" in df.columns:
        ids_split = df["id"].to_numpy()[idx]
    else:
        ids_split = idx

    preds_path = eval_dir / f"ood_{dataset_label}_test.csv"

    sigma_z = None
    if scale_concat is not None:
        sigma_z = scale_concat
        if head_type == "laplace":
            sigma_z = np.sqrt(2.0) * sigma_z
    sigma_ale_orig = _delta_sigma_orig(head_type, sigma_z, mu_orig, y_meta)

    header = [
        "id",
        "split",
        "head_type",
        "mc_flag",
        "n_mc",
        "y_true",
        "y_pred_det",
        "y_pred_mc_mean",
        "sigma_ale_raw",
        "sigma_epi_raw",
    ]

    with preds_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key, yt_o_i, mu_o_i, s_ale in zip(ids_split, yt_orig, mu_orig, sigma_ale_orig):
            writer.writerow([
                key,
                "test",      # treated as test split
                head_type,
                0,
                0,
                float(yt_o_i),
                float(mu_o_i),
                float(mu_o_i),  # deterministic => mc mean = det pred
                float(s_ale) if np.isfinite(s_ale) else np.nan,
                np.nan,         # no epistemic here
            ])

    print(f"[ood-eval] Saved OOD predictions to: {preds_path}")


def _derive_label_from_name(path: Path) -> str:
    """
    Derive a dataset label from an OOD filename.
    Example: craigslist_ood_geo_fl_tx.csv -> geo_fl_tx
    """
    stem = path.stem
    if "_ood_" in stem:
        return stem.split("_ood_", 1)[1]
    # fallback: whole stem
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Train config YAML used for the run")
    ap.add_argument("--outdir", required=True, help="Training outdir (contains model.pt, preproc_meta.json)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ood-csv", help="Single OOD CSV to evaluate")
    group.add_argument("--ood-dir", help="Directory containing multiple *_ood_*.csv files")
    ap.add_argument("--dataset-label", help="Label to use for this OOD set (only with --ood-csv)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not outdir.exists():
        raise FileNotFoundError(f"Outdir not found: {outdir}")

    cfg = _load_cfg(str(cfg_path))
    meta = _load_meta(outdir)

    ood_specs: List[Tuple[Path, str]] = []
    if args.ood_csv:
        ood_path = Path(args.ood_csv).resolve()
        if not ood_path.exists():
            raise FileNotFoundError(f"OOD CSV not found: {ood_path}")
        label = args.dataset_label or _derive_label_from_name(ood_path)
        ood_specs.append((ood_path, label))
    else:
        root = Path(args.ood_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"OOD dir not found: {root}")
        csvs = sorted(p for p in root.glob("*.csv") if "_ood_" in p.name)
        if not csvs:
            raise SystemExit(f"No *_ood_*.csv files found under {root}")
        for p in csvs:
            ood_specs.append((p, _derive_label_from_name(p)))

    for ood_csv, label in ood_specs:
        _eval_single_ood(cfg, meta, outdir, ood_csv, label)


if __name__ == "__main__":
    main()

