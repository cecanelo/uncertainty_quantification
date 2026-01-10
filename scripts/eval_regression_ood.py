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
import pandas as pd
import torch
import yaml

# Allow both package and script-level import
try:
    from .config_resolver import load_and_resolve_config  # type: ignore
except Exception:
    from config_resolver import load_and_resolve_config  # type: ignore

try:
    from .data import DataConfig, _prepare_frame, _apply_encoders, _target_transform, inverse_target  # type: ignore
except Exception:
    from data import DataConfig, _prepare_frame, _apply_encoders, _target_transform, inverse_target  # type: ignore

try:
    from .model_base import MLPRegressor, gaussian_nll, laplace_nll  # type: ignore
except Exception:
    from model_base import MLPRegressor, gaussian_nll, laplace_nll  # type: ignore

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
    *,
    use_mc: bool = False,
    mc_samples: int = 0,
    mc_save_samples: bool = False,
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
    mc_mu_samples: List[np.ndarray] = []
    mc_scale_samples: List[np.ndarray] = []

    with torch.no_grad():
        xb = X_split.to(device)
        yb = y_split.to(device)

        if use_mc and mc_samples > 0 and head_type in ("gauss", "laplace"):
            model.train()  # enable dropout for MC
            for _ in range(mc_samples):
                out = model(xb)
                mu = out["mu"]
                mc_mu_samples.append(mu.detach().cpu().numpy())
                if head_type == "gauss":
                    sigma = out["sigma"]
                    mc_scale_samples.append(sigma.detach().cpu().numpy())
                elif head_type == "laplace":
                    b = out["b"]
                    mc_scale_samples.append(b.detach().cpu().numpy())
            model.eval()

            mu_stack = np.stack(mc_mu_samples, axis=0)  # (mc, n, 1)
            mu_mc_mean = mu_stack.mean(axis=0)
            preds.append(mu_mc_mean)
            targets.append(yb.detach().cpu().numpy())

            # Deterministic forward pass (no dropout) for y_pred_det
            out_det = model(xb)
            mu_det = out_det["mu"].detach().cpu().numpy()  # (n,1)

            # aleatoric: mean of scales; epistemic from std of mu
            if mc_scale_samples:
                scales_mean = np.mean(np.stack(mc_scale_samples, axis=0), axis=0)
                scales.append(scales_mean)
            # nll not well-defined for MC mixture -> skip
        else:
            model.eval()
            out = model(xb)

            mu = out["mu"]
            preds.append(mu.detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
            mu_det = mu.detach().cpu().numpy()  # deterministic path (no MC)

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
        nll_values=nll_vals if nll_vals else None,
        scales=scales if scales else None,
    )

    scale_concat: Optional[np.ndarray] = None
    if head_type in ("gauss", "laplace") and scales:
        scale_concat = np.concatenate(scales, axis=0).reshape(-1)

    mu_concat = np.concatenate(preds, axis=0).reshape(-1)
    yt_concat = np.concatenate(targets, axis=0).reshape(-1)

    mu_orig = inverse_target(mu_concat.reshape(-1, 1), y_meta).reshape(-1)
    mu_det_orig = inverse_target(mu_det.reshape(-1, 1), y_meta).reshape(-1)
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

    # Align OOD schema with ID schema
    mc_count = len(mc_mu_samples) if (use_mc and mc_mu_samples) else 0
    mc_mu_orig_stack: Optional[np.ndarray] = None  # (mc, n) in original space
    sigma_epi_orig: Optional[np.ndarray] = None

    # If we ran MC, compute original-space sample stack once so that:
    # - y_pred_mc_mean is the mean of y_pred_mc_* (in original space)
    # - sigma_epi_orig is the stddev of y_pred_mc_* (in original space)
    if mc_count:
        mc_mu_orig_stack = np.stack(
            [inverse_target(m.reshape(-1, 1), y_meta).reshape(-1) for m in mc_mu_samples],
            axis=0,
        )  # (mc, n)
        sigma_epi_orig = mc_mu_orig_stack.std(axis=0)
        mu_orig = mc_mu_orig_stack.mean(axis=0)
        mc_count = int(mc_mu_orig_stack.shape[0])
    if head_type == "point":
        header = ["id", "split", "head_type", "y_true", "y_pred"]
    else:
        header = [
            "id",
            "split",
            "head_type",
            "mc_flag",
            "n_mc",
            "y_true",
            "y_pred_det",
            "y_pred_mc_mean",
            "sigma_ale_orig",
            "sigma_epi_orig",
        ]
        if mc_save_samples and mc_mu_orig_stack is not None:
            for i in range(int(mc_mu_orig_stack.shape[0])):
                header.append(f"y_pred_mc_{i+1}")

    with preds_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if head_type == "point":
            for key, yt_o_i, mu_o_i in zip(ids_split, yt_orig, mu_orig):
                writer.writerow([key, "test", head_type, float(yt_o_i), float(mu_o_i)])
        else:
            for idx_row, (key, yt_o_i, mu_mc_i, s_ale) in enumerate(zip(ids_split, yt_orig, mu_orig, sigma_ale_orig)):
                mu_det_i = float(mu_det_orig[idx_row]) if head_type != "point" else float(mu_mc_i)
                s_e = float(sigma_epi_orig[idx_row]) if sigma_epi_orig is not None else np.nan
                row = [
                    key,
                    "test",
                    head_type,
                    1 if mc_count else 0,
                    mc_count,
                    float(yt_o_i),
                    mu_det_i,
                    float(mu_mc_i),
                    float(s_ale) if np.isfinite(s_ale) else np.nan,
                    s_e,
                ]
                if mc_save_samples and mc_mu_orig_stack is not None:
                    row.extend(float(mc_mu_orig_stack[i][idx_row]) for i in range(mc_mu_orig_stack.shape[0]))
                writer.writerow(row)
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
    ap.add_argument("--use-mc", action="store_true", help="Enable MC sampling for gauss/laplace heads")
    ap.add_argument("--mc-samples", type=int, default=0, help="Number of MC samples (if --use-mc)")
    ap.add_argument("--mc-save-samples", action="store_true", help="If set, embed per-sample MC draws as extra columns")
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
        _eval_single_ood(
            cfg,
            meta,
            outdir,
            ood_csv,
            label,
            use_mc=args.use_mc,
            mc_samples=args.mc_samples,
            mc_save_samples=args.mc_save_samples,
        )


if __name__ == "__main__":
    main()
