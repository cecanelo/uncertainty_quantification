#!/usr/bin/env python3
"""
MC-dropout OOD evaluation for base regression heads (Laplace/Gauss).

Outputs a CSV per OOD set with the same schema as the ID MC files produced
after training:
    id, split, head_type, mc_flag, n_mc,
    y_true, y_pred_det, y_pred_mc_mean,
    sigma_ale_orig, sigma_epi_orig

Example:
python scripts/eval_regression_ood_mc.py \
  --config outputs/trainings/training_lpl_m/lpl_m_*/used_config.yaml \
  --outdir outputs/trainings/training_lpl_m/lpl_m_* \
  --ood-csv datasets/craigslist_ood_geo_fl_tx.csv \
  --dataset-label geo
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
from model_base import MLPRegressor
from train_regression import _load_cfg


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


def _build_features(
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    ood_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    data_cfg = cfg["data"]
    dc = DataConfig(
        csv_path=str(ood_csv),
        target_col=data_cfg.get("target_col", "price"),
        target_transform=data_cfg.get("target_transform", "log1p"),
    )
    df = _prepare_frame(dc)
    y = df[dc.target_col].astype(float).to_numpy()
    y_tr, y_meta = _target_transform(y, dc.target_transform)

    enc = meta["encoders"]
    numeric_cols: List[str] = meta.get("numeric_cols", [])
    onehot_cols: List[str] = meta.get("onehot_cols", [])
    hash_cols: List[str] = meta.get("hash_cols", [])
    X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)

    if "id" in df.columns:
        ids = df["id"].to_numpy()
    else:
        ids = np.arange(len(df), dtype=int)

    return X, y_tr.reshape(-1, 1), {"y_meta": y_meta, "df": df}, ids


def _mc_forward(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                out = model(xb)
                batch_preds.append(out["mu"].detach().cpu().numpy())
            preds.append(np.concatenate(batch_preds, axis=0).reshape(-1))
    return np.stack(preds, axis=0)


def _det_forward(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    head_type: str,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = []
    scales = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            preds.append(out["mu"].detach().cpu().numpy())
            if head_type == "gauss":
                scales.append(out["sigma"].detach().cpu().numpy())
            elif head_type == "laplace":
                scales.append((np.sqrt(2.0) * out["b"]).detach().cpu().numpy())
    mu_z = np.concatenate(preds, axis=0).reshape(-1)
    sigma_z = np.concatenate(scales, axis=0).reshape(-1) if scales else None
    return mu_z, sigma_z


def _eval_single_ood(
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    outdir: Path,
    ood_csv: Path,
    dataset_label: str,
    n_samples: int,
    batch_size: int,
    p_drop_override: Optional[float],
) -> None:
    head_type = str((cfg.get("model", {}) or {}).get("head_type", "point")).lower()
    if head_type == "point":
        print(f"[mc-ood] Skipping point head for {ood_csv.name} (no epistemic to estimate).")
        return

    X, y_tr, aux, ids = _build_features(cfg, meta, ood_csv)
    y_meta = aux["y_meta"]

    device = torch.device("cpu")
    in_dim = int(meta["feature_dim"])
    model_cfg = cfg["model"]
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

    if p_drop_override is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = float(p_drop_override)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    mu_det_z, sigma_det_z = _det_forward(model, loader, head_type, device)
    mc_samples_z = _mc_forward(model, loader, n_samples, device)

    y_true_z = y_tr.reshape(-1)
    y_true_orig = inverse_target(y_true_z.reshape(-1, 1), y_meta).reshape(-1)
    mu_det_orig = inverse_target(mu_det_z.reshape(-1, 1), y_meta).reshape(-1)
    mc_samples_orig = inverse_target(mc_samples_z, y_meta)
    means_orig = mc_samples_orig.mean(axis=0)
    stds_orig = mc_samples_orig.std(axis=0)
    sigma_ale_orig = _delta_sigma_orig(head_type, sigma_det_z, mu_det_orig, y_meta)

    evals_root = cfg.get("io", {}).get("evals_root", "outputs/evals")
    eval_dir = Path(evals_root) / outdir.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"mc_preds_ood_{dataset_label}.csv"

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

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, key in enumerate(ids):
            writer.writerow(
                [
                    key,
                    "test",
                    head_type,
                    1,
                    n_samples,
                    float(y_true_orig[i]),
                    float(mu_det_orig[i]),
                    float(means_orig[i]),
                    float(sigma_ale_orig[i]) if np.isfinite(sigma_ale_orig[i]) else np.nan,
                    float(stds_orig[i]) if np.isfinite(stds_orig[i]) else np.nan,
                ]
            )

    print(f"[mc-ood] Saved MC OOD preds to: {out_path}")


def _derive_label_from_name(path: Path) -> str:
    stem = path.stem
    if "_ood_" in stem:
        return stem.split("_ood_", 1)[1]
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Train config YAML used for the run")
    ap.add_argument("--outdir", required=True, help="Training outdir (contains model.pt, preproc_meta.json)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ood-csv", help="Single OOD CSV to evaluate")
    group.add_argument("--ood-dir", help="Directory containing multiple *_ood_*.csv files")
    ap.add_argument("--dataset-label", help="Label to use for this OOD set (only with --ood-csv)")
    ap.add_argument("--n-samples", type=int, help="Override MC samples; default from config training.mc_eval.n_samples or 50")
    ap.add_argument("--batch-size", type=int, help="Override batch size for inference")
    ap.add_argument("--p-drop", type=float, help="Optional dropout probability override during MC inference")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    cfg = _load_cfg(str(cfg_path))
    meta = _load_meta(outdir)

    mc_cfg = cfg.get("training", {}).get("mc_eval", {}) or {}
    if not mc_cfg.get("enabled", False) and args.n_samples is None:
        print("[mc-ood] mc_eval.enabled is false and no override provided; skipping.")
        return
    n_samples = int(args.n_samples or mc_cfg.get("n_samples", 50))
    batch_size = int(args.batch_size or mc_cfg.get("batch_size", cfg["data"].get("batch_size", 512)))
    p_drop_override = args.p_drop if args.p_drop is not None else mc_cfg.get("p_drop")

    ood_specs: List[Tuple[Path, str]] = []
    if args.ood_csv:
        ood_path = Path(args.ood_csv).resolve()
        label = args.dataset_label or _derive_label_from_name(ood_path)
        ood_specs.append((ood_path, label))
    else:
        root = Path(args.ood_dir).resolve()
        csvs = sorted(p for p in root.glob("*.csv") if "_ood_" in p.name)
        for p in csvs:
            ood_specs.append((p, _derive_label_from_name(p)))

    for ood_csv, label in ood_specs:
        _eval_single_ood(cfg, meta, outdir, ood_csv, label, n_samples, batch_size, p_drop_override)


if __name__ == "__main__":
    main()
