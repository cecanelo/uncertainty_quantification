#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

def _target_transform(y: np.ndarray, mode: str):
    if mode is None or mode == "identity":
        return y
    if mode == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unsupported target_transform mode: {mode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to train config YAML (same used for train_regression)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory of the training run (contains preproc_meta.json)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    outdir = Path(args.outdir)

    # 1) Load config and preproc meta
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    meta_path = outdir / "preproc_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"preproc_meta.json not found in {outdir}")

    with meta_path.open("r") as f:
        meta = json.load(f)

    data_cfg = cfg["data"]
    csv_path = Path(data_cfg["csv_path"])
    target_col = data_cfg["target_col"]
    target_mode = data_cfg.get("target_transform", "identity")

    # 2) Load raw CSV and target
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {csv_path}")

    y = df[target_col].astype(float).to_numpy()
    y_tr = _target_transform(y, target_mode)

    # 3) Use the same splits as training
    splits = meta["splits"]
    i_tr = np.array(splits["train"], dtype=int)
    i_va = np.array(splits["val"], dtype=int)

    y_train = y_tr[i_tr]
    y_val = y_tr[i_va]

    # 4) Baseline: predict train-mean for everyone
    baseline = float(y_train.mean())
    y_pred_val = np.full_like(y_val, fill_value=baseline)

    # 5) Compute MAE and RMSE on val
    residuals = y_val - y_pred_val
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # 6) Print and optionally save to JSON
    print(f"[baseline_mean] train_size={len(i_tr)} val_size={len(i_va)}")
    print(f"[baseline_mean] target_mode={target_mode}")
    print(f"[baseline_mean] train_mean={baseline:.6f}")
    print(f"[baseline_mean] val_MAE={mae:.6f} val_RMSE={rmse:.6f}")

    out_json = {
        "train_size": int(len(i_tr)),
        "val_size": int(len(i_va)),
        "target_mode": target_mode,
        "train_mean": baseline,
        "val_mae": mae,
        "val_rmse": rmse,
    }
    with (outdir / "baseline_mean_metrics.json").open("w") as f:
        json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    main()
