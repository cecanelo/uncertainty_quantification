#!/usr/bin/env python3
"""
Evaluate an ensemble on a single OOD CSV and produce ensemble epistemic/aleatoric
signals in original target units.

Output schema matches the ID ensemble eval:
    id, split, head_type, method, n_members,
    y_true, y_pred_method, y_pred_ens_mean,
    sigma_ale_ens, sigma_epi_ens

Example:
python scripts/eval_ensemble_ood.py \
  --config outputs/trainings/ensembles/training_ensemble_lpl_m/.../used_config.yaml \
  --ensemble-root outputs/trainings/ensembles/training_ensemble_lpl_m/... \
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

# Allow both package and script-level import
try:
    from .config_resolver import load_and_resolve_config  # type: ignore
except Exception:
    from config_resolver import load_and_resolve_config  # type: ignore

from data import DataConfig, _prepare_frame, _apply_encoders, _target_transform, inverse_target
from model_base import MLPRegressor


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _delta_sigma_orig(head_type: str, sigma_z: Optional[np.ndarray], mu_orig: np.ndarray, target_meta: dict) -> np.ndarray:
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


def _load_ood_frame(ood_csv: Path, data_cfg: dict) -> Tuple[np.ndarray, dict, np.ndarray]:
    dc = DataConfig(
        csv_path=str(ood_csv),
        target_col=data_cfg.get("target_col", "price"),
        target_transform=data_cfg.get("target_transform", "log1p"),
    )
    df = _prepare_frame(dc)
    y = df[dc.target_col].astype(float).to_numpy()
    y_tr, y_meta = _target_transform(y, dc.target_transform)
    return y_tr.reshape(-1, 1), y_meta, df


def _member_paths(ensemble_root: Path) -> List[Path]:
    members = sorted([p for p in ensemble_root.iterdir() if p.is_dir() and p.name.startswith("member_")])
    if not members:
        raise FileNotFoundError(f"No member_* folders found in {ensemble_root}")
    return members


def _load_model(member_dir: Path, in_dim: int, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    hidden = model_cfg.get("hidden_dims", [512, 256, 128])
    head_type = model_cfg.get("head_type", "point").lower()
    activation = model_cfg.get("activation", "relu")
    dropout = float(model_cfg.get("dropout", 0.0))

    model = MLPRegressor(
        in_dim=in_dim,
        hidden_dims=hidden,
        head_type=head_type,
        activation=activation,
        dropout=dropout,
        use_batchnorm=False,
    ).to(device)

    ckpt_path = member_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def _run_member_preds(
    model: torch.nn.Module,
    Xs: torch.Tensor,
    head_type: str,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = []
    scales = []
    with torch.no_grad():
        xb = Xs.to(device)
        out = model(xb)
        preds.append(out["mu"].detach().cpu().numpy())
        if head_type == "gauss":
            scales.append(out["sigma"].detach().cpu().numpy())
        elif head_type == "laplace":
            scales.append((np.sqrt(2.0) * out["b"]).detach().cpu().numpy())
    mu_z = np.concatenate(preds, axis=0).reshape(-1)
    sigma_z = np.concatenate(scales, axis=0).reshape(-1) if scales else None
    return mu_z, sigma_z


def _write_csv(
    path: Path,
    ids: np.ndarray,
    split: str,
    head_type: str,
    n_members: int,
    y_true_orig: np.ndarray,
    y_pred_ens_orig: np.ndarray,
    sigma_ale_orig: np.ndarray,
    sigma_epi_orig: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "id",
        "split",
        "head_type",
        "method",
        "n_members",
        "y_true",
        "y_pred_method",
        "y_pred_ens_mean",
        "sigma_ale_ens",
        "sigma_epi_ens",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, mid in enumerate(ids):
            writer.writerow(
                [
                    mid,
                    split,
                    head_type,
                    "ensemble",
                    n_members,
                    float(y_true_orig[i]),
                    float(y_pred_ens_orig[i]),
                    float(y_pred_ens_orig[i]),
                    float(sigma_ale_orig[i]) if np.isfinite(sigma_ale_orig[i]) else np.nan,
                    float(sigma_epi_orig[i]) if np.isfinite(sigma_epi_orig[i]) else np.nan,
                ]
            )


def _derive_label_from_name(path: Path) -> str:
    stem = path.stem
    if "_ood_" in stem:
        return stem.split("_ood_", 1)[1]
    return stem


def evaluate_ood(
    cfg_path: Path,
    ensemble_root: Path,
    ood_csv: Path,
    dataset_label: str,
) -> None:
    base_cfg = load_and_resolve_config(str(cfg_path))
    members = _member_paths(ensemble_root)
    data_cfg = base_cfg["data"]
    model_cfg = base_cfg["model"]
    head_type = model_cfg.get("head_type", "point").lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_tr, y_meta, df = _load_ood_frame(ood_csv, data_cfg)
    ids = df["id"].to_numpy() if "id" in df.columns else np.arange(len(df), dtype=int)

    n_members = len(members)

    member_means = []
    member_scales = []
    for mdir in members:
        meta_path = mdir / "preproc_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"preproc_meta.json not found in {mdir}")
        meta = load_json(meta_path)

        enc = meta["encoders"]
        numeric_cols = meta.get("numeric_cols", [])
        onehot_cols = meta.get("onehot_cols", [])
        hash_cols = meta.get("hash_cols", [])
        X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)
        Xs = torch.tensor(X, dtype=torch.float32)

        in_dim = int(meta["feature_dim"])
        model = _load_model(mdir, in_dim, model_cfg, device)
        mu_z, sigma_z = _run_member_preds(model, Xs, head_type, device)
        member_means.append(mu_z)
        member_scales.append(sigma_z)

    member_means = np.stack(member_means, axis=0)
    member_scales = np.stack(member_scales, axis=0) if member_scales and member_scales[0] is not None else None

    ens_mean_z = member_means.mean(axis=0)
    sigma_epi_z = member_means.std(axis=0)
    if member_scales is not None:
        ale_var_z = (member_scales ** 2).mean(axis=0)
        sigma_ale_z = np.sqrt(ale_var_z)
    else:
        sigma_ale_z = np.full_like(ens_mean_z, np.nan, dtype=float)

    y_true_z = y_tr.reshape(-1)
    y_true_orig = inverse_target(y_true_z.reshape(-1, 1), y_meta).reshape(-1)
    y_pred_orig = inverse_target(ens_mean_z.reshape(-1, 1), y_meta).reshape(-1)
    sigma_ale_orig = _delta_sigma_orig(head_type, sigma_ale_z, y_pred_orig, y_meta)
    sigma_epi_orig = _delta_sigma_orig(head_type, sigma_epi_z, y_pred_orig, y_meta)

    evals_root = base_cfg.get("io", {}).get("evals_root", "outputs/evals")
    evals_root = Path(evals_root)
    dataset = base_cfg.get("dataset") or (base_cfg.get("data") or {}).get("dataset")
    # If ensemble members wrote into outputs/evals/<tag>, redirect to outputs/{dataset}_evals.
    if dataset and "outputs/evals" in evals_root.as_posix():
        evals_root = Path("outputs") / f"{dataset}_evals"
    eval_dir = evals_root / ensemble_root.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"ensemble_preds_ood_{dataset_label}.csv"
    _write_csv(out_path, ids, f"ood_{dataset_label}", head_type, n_members, y_true_orig, y_pred_orig, sigma_ale_orig, sigma_epi_orig)
    print(f"[ensemble-ood] wrote {out_path} (n={len(ids)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble OOD evaluation")
    ap.add_argument("--config", required=True, help="used_config.yaml of the ensemble run")
    ap.add_argument("--ensemble-root", required=True, help="Folder containing member_* subdirs")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ood-csv", help="Single OOD CSV to evaluate")
    group.add_argument("--ood-dir", help="Directory with *_ood_*.csv files")
    ap.add_argument("--dataset-label", help="Dataset label (used for filename when --ood-csv)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    ensemble_root = Path(args.ensemble_root).resolve()
    if args.ood_csv:
        specs = [(Path(args.ood_csv).resolve(), args.dataset_label or _derive_label_from_name(Path(args.ood_csv)))]
    else:
        root = Path(args.ood_dir).resolve()
        csvs = sorted(p for p in root.glob("*.csv") if "_ood_" in p.name)
        specs = [(p, _derive_label_from_name(p)) for p in csvs]

    for ood_csv, label in specs:
        evaluate_ood(cfg_path, ensemble_root, ood_csv, label)


if __name__ == "__main__":
    main()
