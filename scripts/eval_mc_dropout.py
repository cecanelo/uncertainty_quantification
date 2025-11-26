#!/usr/bin/env python3
"""
Run Monte Carlo dropout evaluation on a trained model.

What it does
------------
- Loads a trained checkpoint and preprocessing metadata.
- Rebuilds features for requested splits (train/val/test) using stored encoders.
- Runs N stochastic forward passes with dropout active.
- Writes per-split CSVs with a unified schema (id, split, head_type, deterministic + MC stats in original units, optional MC samples).

CLI
---
python scripts/eval_mc_dropout.py --config path/to/eval_mc.yaml [--mode local|slurm]

Config (YAML) shape
-------------------
run:
  mode: "slurm"            # "local" or "slurm"
  partition: "TEST"
  time: "00:10:00"
  mem_gb: 30
  cpus: 4
  gpus: 1
  conda_env: "thesis_pascal"
  job_name: "eval_mc"
  logs_dir: "logs"

model:
  run_dir: "<folder containing model.pt and preproc_meta.json>"
  checkpoint: "model.pt"
  preproc_meta: "preproc_meta.json"
  used_config: "used_config.yaml"   # optional; defaults to run_dir/used_config.yaml

data:
  csv_path: "<path to raw CSV>"
  target_col: "price"
  target_transform: "none"          # or "log1p"

mc_dropout:
  enabled: true
  n_samples: 20
  p_drop: null                      # override dropout prob at inference; null keeps trained value
  splits: ["train", "val", "test"]
  batch_size: 512
  sample_count: null                # optional absolute limit per split
  sample_frac: null                 # optional fraction per split (ignored if sample_count set)
  seed: 42

io:
  save_dir: "outputs/evals/{run_tag}/mc_dropout"
  include_samples: true
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from data import _prepare_frame, _apply_encoders, DataConfig, _target_transform, inverse_target
from model_base import MLPRegressor


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _override_dropout(model: torch.nn.Module, p: float) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p


def _delta_sigma_orig(head_type: str,
                      sigma_z: Optional[np.ndarray],
                      mu_orig: np.ndarray,
                      target_meta: dict) -> np.ndarray:
    """
    Map transformed-space scale to an approximate original-space std via the delta method.
    For laplace, sigma_z is expected to be std_z (sqrt(2)*b_z).
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


def _subset_indices(idx: np.ndarray, count: Optional[int], frac: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if count is not None:
        count = max(0, min(len(idx), int(count)))
        if count == len(idx):
            return idx
        sel = rng.choice(idx, size=count, replace=False)
        return np.sort(sel)
    if frac is not None:
        frac = max(0.0, min(1.0, float(frac)))
        k = int(round(frac * len(idx)))
        if k <= 0:
            return idx[:0]
        sel = rng.choice(idx, size=k, replace=False)
        return np.sort(sel)
    return idx


def _build_features(meta: dict, data_cfg: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    dc = DataConfig(**data_cfg)
    df = _prepare_frame(dc)
    y = df[dc.target_col].astype(float).to_numpy()
    y_tr, y_meta = _target_transform(y, dc.target_transform)

    enc = meta["encoders"]
    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols = meta.get("onehot_cols", [])
    hash_cols = meta.get("hash_cols", [])

    X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)
    return X, y_tr.reshape(-1, 1), {"y_meta": y_meta, "df": df}


def _run_mc_passes(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    model.train()  # enable dropout
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                out = model(xb)
                batch_preds.append(out["mu"].detach().cpu().numpy())
            preds.append(np.concatenate(batch_preds, axis=0).reshape(-1))
    return np.stack(preds, axis=0)  # [n_samples, N]


def _deterministic_pass(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    head_type: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run a deterministic forward pass (dropout disabled) to get mean predictions
    and aleatoric scale in transformed space.
    Returns (mu_z, sigma_z_std) where sigma_z_std is std-like (sqrt(2)*b for laplace).
    """
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
                # convert Laplace scale b_z to std-like in transformed space
                scales.append((np.sqrt(2.0) * out["b"]).detach().cpu().numpy())

    mu_z = np.concatenate(preds, axis=0).reshape(-1)
    sigma_z = np.concatenate(scales, axis=0).reshape(-1) if scales else None
    return mu_z, sigma_z


def _crps_empirical(samples: np.ndarray, y_true: np.ndarray) -> float:
    """
    Empirical CRPS for a single predictive sample set and scalar target.
    samples: [S] predictions
    y_true: scalar
    """
    mean_abs = np.mean(np.abs(samples - y_true))
    pair_abs = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    return float(mean_abs - pair_abs)


def _write_unified_csv(
    path: Path,
    *,
    ids: np.ndarray,
    split: str,
    head_type: str,
    mc_flag: int,
    n_mc: int,
    y_true_orig: np.ndarray,
    y_pred_det_orig: np.ndarray,
    y_pred_mc_mean_orig: np.ndarray,
    sigma_ale_orig: np.ndarray,
    sigma_epi_orig: np.ndarray,
    include_samples: bool,
    mc_samples_orig: Optional[np.ndarray],
) -> None:
    """
    Write per-sample rows with the unified schema, optionally including MC samples (original units).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if include_samples and mc_samples_orig is not None:
        header.extend([f"y_pred_mc_{i}" for i in range(1, mc_samples_orig.shape[0] + 1)])

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for j, mid in enumerate(ids):
            row = [
                mid,
                split,
                head_type,
                mc_flag,
                n_mc,
                float(y_true_orig[j]),
                float(y_pred_det_orig[j]) if np.isfinite(y_pred_det_orig[j]) else np.nan,
                float(y_pred_mc_mean_orig[j]),
                float(sigma_ale_orig[j]) if np.isfinite(sigma_ale_orig[j]) else np.nan,
                float(sigma_epi_orig[j]) if np.isfinite(sigma_epi_orig[j]) else np.nan,
            ]
            if include_samples and mc_samples_orig is not None:
                row.extend([float(v) for v in mc_samples_orig[:, j]])
            writer.writerow(row)


def evaluate(cfg: dict, override_mode: Optional[str] = None) -> None:
    run_cfg = cfg.get("run", {}) or {}
    mode = override_mode or run_cfg.get("mode", "local").lower()

    model_cfg = cfg["model"]
    mc_cfg = cfg["mc_dropout"]
    data_cfg = cfg["data"]
    io_cfg = cfg["io"]

    run_dir = Path(model_cfg["run_dir"]).resolve()
    ckpt_path = run_dir / model_cfg.get("checkpoint", "model.pt")
    meta_path = run_dir / model_cfg.get("preproc_meta", "preproc_meta.json")
    used_cfg_name = model_cfg.get("used_config", "used_config.yaml")
    used_cfg_path = Path(used_cfg_name)
    if not used_cfg_path.is_absolute():
        used_cfg_path = (run_dir / used_cfg_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"preproc_meta.json not found: {meta_path}")
    if not used_cfg_path.exists():
        raise FileNotFoundError(f"used_config.yaml not found: {used_cfg_path}")

    preproc_meta = json.loads(meta_path.read_text())
    used_cfg = load_yaml(used_cfg_path)

    # Reuse model config from training
    mcfg = used_cfg["model"]
    hidden = mcfg.get("hidden_dims", [512, 256, 128])
    head_type = mcfg.get("head_type", "point").lower()
    activation = mcfg.get("activation", "relu")
    dropout = float(mcfg.get("dropout", 0.0))
    use_batchnorm = bool(mcfg.get("batchnorm", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build features
    X_all, y_tr, extra = _build_features(preproc_meta, data_cfg)
    df = extra["df"]

    # Prepare ID column
    id_col = preproc_meta.get("id_col")
    ids_all = None
    if id_col and "id_values" in preproc_meta:
        ids_all = np.array(preproc_meta["id_values"])
    elif "id" in df.columns:
        id_col = "id"
        ids_all = df["id"].to_numpy()
    else:
        id_col = "row_index"
        ids_all = np.arange(len(df))

    # Model
    in_dim = int(preproc_meta["feature_dim"])
    model = MLPRegressor(
        in_dim=in_dim,
        hidden_dims=hidden,
        head_type=head_type,
        activation=activation,
        dropout=dropout,
        use_batchnorm=False,  # batch norm disabled for eval/training
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    if mc_cfg.get("p_drop") is not None:
        _override_dropout(model, float(mc_cfg["p_drop"]))

    n_samples = int(mc_cfg.get("n_samples", 20))
    batch_size = int(mc_cfg.get("batch_size", 512))
    sample_count = mc_cfg.get("sample_count")
    sample_frac = mc_cfg.get("sample_frac")
    seed = int(mc_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    splits = mc_cfg.get("splits", ["val"])
    save_dir_tpl = io_cfg.get("save_dir", "outputs/evals/{run_tag}/mc_dropout")
    include_samples = bool(io_cfg.get("include_samples", True))
    run_tag = run_dir.name
    save_dir = Path(save_dir_tpl.format(run_tag=run_tag))

    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    if job_id:
        save_dir = save_dir / f"job_{job_id}"

    save_dir.mkdir(parents=True, exist_ok=True)

    split_counts: Dict[str, int] = {}
    metrics: Dict[str, dict] = {}
    eval_start = time.perf_counter()
    start_dt = datetime.utcnow()

    log_lines: List[str] = []

    def _log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    from torch.utils.data import DataLoader, TensorDataset

    if dropout <= 0:
        _log("[mc-dropout] Warning: model dropout probability is 0. MC predictions may have ~0 epistemic spread.")

    for split in splits:
        split_key = split.lower()
        if split_key not in preproc_meta.get("splits", {}):
            _log(f"[warn] split '{split_key}' not found in preproc_meta; skipping.")
            continue
        idx = np.array(preproc_meta["splits"][split_key], dtype=int)
        idx = _subset_indices(idx, sample_count, sample_frac, rng)
        if idx.size == 0:
            _log(f"[warn] split '{split_key}' has 0 rows after sampling; skipping.")
            continue

        Xs = torch.tensor(X_all[idx], dtype=torch.float32)
        ys = torch.tensor(y_tr[idx], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(Xs, ys),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        # MC passes (transformed space)
        mc_samples = _run_mc_passes(model, loader, n_samples=n_samples, device=device)
        means_z = mc_samples.mean(axis=0)

        # Deterministic pass (dropout disabled) for baseline prediction + aleatoric scale
        mu_det_z, sigma_det_z = _deterministic_pass(model, loader, device, head_type)

        ids_split = ids_all[idx]
        y_true_z = ys.numpy().reshape(-1)

        # Original-space values
        y_meta = extra["y_meta"]
        y_true_orig = inverse_target(y_true_z.reshape(-1, 1), y_meta).reshape(-1)
        mu_det_orig = inverse_target(mu_det_z.reshape(-1, 1), y_meta).reshape(-1)
        mc_samples_orig = inverse_target(mc_samples, y_meta)
        means_orig = mc_samples_orig.mean(axis=0)
        stds_orig = mc_samples_orig.std(axis=0)

        sigma_ale_orig = _delta_sigma_orig(head_type, sigma_det_z, mu_det_orig, y_meta)

        # Metrics remain in transformed space for compatibility
        variances_split = mc_samples.var(axis=0) + 1e-8
        nll = np.mean(
            0.5 * np.log(2 * np.pi * variances_split)
            + ((y_true_z - means_z) ** 2) / (2 * variances_split)
        )
        crps_vals = []
        for j in range(mc_samples.shape[1]):
            crps_vals.append(_crps_empirical(mc_samples[:, j], y_true_z[j]))
        crps_mean = float(np.mean(crps_vals))

        out_path = save_dir / f"mc_preds_{split_key}.csv"
        _write_unified_csv(
            out_path,
            ids=ids_split,
            split=split_key,
            head_type=head_type,
            mc_flag=1,
            n_mc=n_samples,
            y_true_orig=y_true_orig,
            y_pred_det_orig=mu_det_orig,
            y_pred_mc_mean_orig=means_orig,
            sigma_ale_orig=sigma_ale_orig,
            sigma_epi_orig=stds_orig,
            include_samples=include_samples,
            mc_samples_orig=mc_samples_orig if include_samples else None,
        )

        split_counts[split_key] = len(idx)
        metrics[split_key] = {
            "nll": float(nll),
            "crps": crps_mean,
            "n_rows": int(len(idx)),
            "n_samples": int(mc_samples.shape[0]),
            "head_type": head_type,
            "target_transform": y_meta.get("mode", "none"),
        }
        _log(f"[mc-dropout] split={split_key} n={len(idx)} wrote: {out_path}")

    # annotate metrics with global context
    metrics["_meta"] = {
        "head_type": head_type,
        "target_transform": extra["y_meta"].get("mode", "none"),
        "n_samples": n_samples,
    }

    if split_counts:
        _log("[mc-dropout] evaluated rows per split:")
        for sk, n in split_counts.items():
            _log(f"  {sk}: {n}")
    else:
        _log("[mc-dropout] no splits evaluated.")
    elapsed = time.perf_counter() - eval_start
    duration_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    _log(f"[mc-dropout] total_eval_time_sec={elapsed:.2f} ({duration_hms})")

    metrics_path = save_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "start_utc": start_dt.isoformat() + "Z",
        "end_utc": datetime.utcnow().isoformat() + "Z",
        "duration_seconds": elapsed,
        "duration_hms": duration_hms,
        "split_counts": split_counts,
        "splits": splits,
        "params": {
            "p_drop": mc_cfg.get("p_drop"),
            "n_samples": n_samples,
            "batch_size": batch_size,
            "sample_count": sample_count,
            "sample_frac": sample_frac,
            "seed": seed,
        },
        "run": {
            "run_tag": run_tag,
            "job_id": job_id,
            "mode": mode,
            "device": str(device),
            "head_type": head_type,
        },
        "paths": {
            "save_dir": str(save_dir),
            "metrics": str(metrics_path),
        },
        "target_transform": extra["y_meta"].get("mode", "none"),
    }
    metadata_path = save_dir / "run_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    cfg_out_path = save_dir / "eval_config.yaml"
    with cfg_out_path.open("w") as f:
        yaml.safe_dump(cfg, f)

    log_path = save_dir / "run.log"
    with log_path.open("w") as f:
        for line in log_lines:
            f.write(line + "\n")


def submit_slurm(cfg_path: Path, cfg: dict) -> None:
    run_cfg = cfg.get("run", {}) or {}
    partition = run_cfg.get("partition", "TEST")
    time_str = run_cfg.get("time", "00:10:00")
    mem_gb = int(run_cfg.get("mem_gb", 8))
    cpus = int(run_cfg.get("cpus", 2))
    gpus = int(run_cfg.get("gpus", 0))
    conda_env = run_cfg.get("conda_env", "thesis")
    job_name = run_cfg.get("job_name", "eval_mc")
    logs_dir = Path(run_cfg.get("logs_dir", "logs")).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_log = logs_dir / f"{job_name}_{ts}_%j.out"
    err_log = logs_dir / f"{job_name}_{ts}_%j.err"

    script_path = Path(__file__).resolve()
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_str}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

cd "{script_path.parent.parent}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate {conda_env}

echo "[env] host=$(hostname) date=$(date)"
python "{script_path}" --config "{cfg_path}" --mode local
"""
    print("[eval_mc] Submitting sbatch with script:")
    print(sbatch_script)
    res = subprocess.run(["sbatch"], input=sbatch_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="MC Dropout evaluation from a trained checkpoint.")
    ap.add_argument(
        "--config",
        default="configs/eval_mc_dropout.yaml",
        help="Path to eval_mc YAML config (default: configs/eval_mc_dropout.yaml).",
    )
    ap.add_argument("--mode", choices=["local", "slurm"], help="Override run.mode from config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)

    mode = (args.mode or cfg.get("run", {}).get("mode", "local")).lower()
    if mode == "slurm":
        submit_slurm(cfg_path, cfg)
    else:
        evaluate(cfg, override_mode="local")


if __name__ == "__main__":
    main()
