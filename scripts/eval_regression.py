#!/usr/bin/env python3
"""
Evaluate a saved base regression model on a given split (default: val).

Usage:
    python scripts/eval_regression.py \
        --config configs/train_base_D8_point.yaml \
        --outdir outputs/base_D8_point \
        --split val
"""

from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import Dict, Any, List

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


def _delta_sigma_orig(head_type: str,
                      sigma_z: np.ndarray | None,
                      mu_orig: np.ndarray,
                      target_meta: Dict[str, str]) -> np.ndarray:
    """
    Map transformed-space scale to an approximate original-space std via the delta method.
    - point: returns NaNs.
    - gauss: sigma_z is already a std in transformed space.
    - laplace: sigma_z should already be a std-like quantity (e.g., sqrt(2) * b_z).
    """
    n = mu_orig.shape[0]
    if sigma_z is None:
        return np.full(n, np.nan, dtype=float)

    sigma_z = sigma_z.reshape(-1)
    if head_type == "point":
        return np.full(n, np.nan, dtype=float)

    mode = (target_meta or {}).get("mode", "none").lower()
    if mode == "log1p":
        # derivative dy/dz = exp(z) = y + 1; approximate with predicted y
        return (mu_orig + 1.0) * sigma_z
    # identity / none
    return sigma_z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Train config YAML used for the run")
    p.add_argument("--outdir", required=True, help="Output directory of the training run")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Which split to evaluate (default: val)")
    p.add_argument("--flow-dump", action="store_true",
                   help="When set, also write flow_<split>.csv (y_true_t, mu_t, scale_t) for gauss/laplace heads.")
    p.add_argument("--flow-outdir", default=None,
                   help="Directory to save flow CSV (default: eval_dir when --flow-dump is set).")
    args = p.parse_args()

    cfg_path = Path(args.config)
    outdir = Path(args.outdir)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not outdir.exists():
        raise FileNotFoundError(f"Outdir not found: {outdir}")

    # --- 1) Load config and preprocessing meta ---
    cfg = _load_cfg(str(cfg_path))
    meta = _load_meta(outdir)

    data_cfg = cfg["data"]
    dc = DataConfig(**data_cfg)

    # --- 2) Load raw dataframe and target, apply target transform ---
    df = _prepare_frame(dc)
    y = df[dc.target_col].astype(float).to_numpy()
    y_tr, y_meta = _target_transform(y, dc.target_transform)

    # --- 3) Recreate features using stored encoders ---
    numeric_cols: List[str] = meta["numeric_cols"]
    onehot_cols: List[str] = meta["onehot_cols"]
    hash_cols: List[str] = meta["hash_cols"]
    enc: Dict[str, Any] = meta["encoders"]

    X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)

    # --- 4) Select the requested split indices ---
    splits = meta["splits"]
    if args.split not in splits:
        raise KeyError(f"Split '{args.split}' not found in meta['splits']")
    idx = np.array(splits[args.split], dtype=int)
    if idx.size == 0:
        raise ValueError(f"Requested split '{args.split}' has zero rows.")

    X_split = torch.tensor(X[idx], dtype=torch.float32)
    y_split = torch.tensor(y_tr[idx], dtype=torch.float32).view(-1, 1)

    # --- 5) Rebuild model and load checkpoint ---
    device = torch.device("cpu")  # evaluation is cheap; CPU is fine
    in_dim = int(meta["feature_dim"])
    model_cfg = cfg["model"]
    hidden = model_cfg.get("hidden_dims", [512, 256, 128])
    head_type = model_cfg.get("head_type", "point").lower()

    model = MLPRegressor(
        in_dim=in_dim,
        hidden_dims=hidden,
        head_type=head_type,
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_batchnorm=False,  # batch norm disabled for eval (and training)
    ).to(device)

    ckpt_path = outdir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint model.pt not found in {outdir}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # --- 6) Run inference on the split ---
    preds, targets = [], []
    nll_vals: List[float] = []
    scales = []  # NEW: per-instance sigma/b for probabilistic heads
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
            scales.append(sigma.detach().cpu().numpy())  # NEW
        elif head_type == "laplace":
            b = out["b"]
            nll = laplace_nll(mu, b, yb)
            nll_vals.append(float(nll.item()))
            scales.append(b.detach().cpu().numpy())      # NEW


    # --- 7) Compute metrics using the same helper as train_regression (transformed scale) ---
    metrics = _metrics_from_batches(
        preds,
        targets,
        head_type,
        nll_values=nll_vals,
        scales=scales if scales else None,  # NEW
    )

    scale_concat = None
    if head_type in ("gauss", "laplace") and scales:
        scale_concat = np.concatenate(scales, axis=0).reshape(-1)

    # --- 8) Basic consistency checks ---
    mu_concat = np.concatenate(preds, axis=0).reshape(-1)
    yt_concat = np.concatenate(targets, axis=0).reshape(-1)

    assert mu_concat.shape == yt_concat.shape, "Prediction and target shapes mismatch"
    assert mu_concat.shape[0] == idx.size, "Number of predictions != split size"
    if not np.all(np.isfinite(mu_concat)):
        raise ValueError("Non-finite values found in predictions")
    if not np.all(np.isfinite(yt_concat)):
        raise ValueError("Non-finite values found in targets")

    # --- 9) Metrics on original target scale ---
    # We already computed y_meta when applying _target_transform earlier.
    # y_meta describes the target transform (e.g., "mode": "log1p").
    mu_orig = inverse_target(mu_concat.reshape(-1, 1), y_meta).reshape(-1)
    yt_orig = inverse_target(yt_concat.reshape(-1, 1), y_meta).reshape(-1)

    ae_orig = np.abs(mu_orig - yt_orig)
    se_orig = (mu_orig - yt_orig) ** 2
    mae_orig = float(np.mean(ae_orig))
    rmse_orig = float(np.sqrt(np.mean(se_orig)))

    # --- 10) Report and save ---
    print(f"[eval] split={args.split} n={idx.size}")
    print(f"[eval] head_type={head_type}")
    print(f"[eval] MAE (transformed)={metrics['mae']:.6f}  RMSE (transformed)={metrics['rmse']:.6f}")
    print(f"[eval] MAE_orig={mae_orig:.2f}  RMSE_orig={rmse_orig:.2f}")
    if "nll" in metrics:
        print(f"[eval] NLL={metrics['nll']:.6f}")
    if "scale_mean" in metrics:
        print(
            f"[eval] mean_scale={metrics['scale_mean']:.6f} "
            f"median_scale={metrics['scale_median']:.6f}"
        )


    out_json = {
        "split": args.split,
        "head_type": head_type,
        "n": int(idx.size),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mae_orig": mae_orig,
        "rmse_orig": rmse_orig,
    }
    if "nll" in metrics:
        out_json["nll"] = float(metrics["nll"])
    if "scale_mean" in metrics:
        out_json["scale_mean"] = float(metrics["scale_mean"])
        out_json["scale_median"] = float(metrics["scale_median"])


    with (outdir / f"eval_metrics_{args.split}.json").open("w") as f:
        json.dump(out_json, f, indent=4)

    # --- 11) Save per-instance predictions in original target space with ID/row_index ---
    # Re-use evals_root/run_tag convention from train_regression
    evals_root = cfg.get("io", {}).get("evals_root", "outputs/evals")
    evals_root = Path(evals_root)
    run_tag = outdir.name
    eval_dir = evals_root / run_tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Derive ID/row_index for this split
    id_col = None
    ids_split = None
    if "id_col" in meta and "id_values" in meta:
        id_col = meta["id_col"]
        ids_all = np.array(meta["id_values"])
        ids_split = ids_all[idx]
    elif "id" in df.columns:
        id_col = "id"
        ids_split = df["id"].to_numpy()[idx]
    else:
        id_col = "row_index"
        ids_split = idx

    preds_path = eval_dir / ("test_preds.csv" if args.split == "test" else f"preds_{args.split}.csv")

    # Aleatoric scale mapped to original units (std-like). For laplace we first turn b_z into std_z.
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
                args.split,
                head_type,
                0,      # mc_flag
                0,      # n_mc
                float(yt_o_i),
                float(mu_o_i),
                float(mu_o_i),  # deterministic eval => mc mean equals det pred
                float(s_ale) if np.isfinite(s_ale) else np.nan,
                np.nan,         # no epistemic component here
            ])

    meta_payload = {
        "head_type": head_type,
        "split": args.split,
        "target_transform": y_meta.get("mode", "none"),
        "mc_flag": 0,
        "n_mc": 0,
    }
    with (eval_dir / "metadata.json").open("w") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"[eval] Saved predictions to: {preds_path}")

    # --- 12) Optional flow-friendly dump for NF residual modeling (gauss/laplace only) ---
    if args.flow_dump and head_type in ("gauss", "laplace") and scale_concat is not None:
        flow_dir = Path(args.flow_outdir) if args.flow_outdir else eval_dir
        flow_dir.mkdir(parents=True, exist_ok=True)
        flow_path = flow_dir / f"flow_{args.split}.csv"
        scale_t = scale_concat.astype(float)
        if head_type == "laplace":
            scale_t = np.sqrt(2.0) * scale_t
        with flow_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "y_true_t", "mu_t", "scale_t", "head_type"])
            for rid, y_t_i, mu_t_i, s_t_i in zip(idx, yt_concat, mu_concat, scale_t):
                writer.writerow([int(rid), float(y_t_i), float(mu_t_i), float(s_t_i), head_type])
        print(f"[eval] Saved flow preds to: {flow_path}")


if __name__ == "__main__":
    main()


