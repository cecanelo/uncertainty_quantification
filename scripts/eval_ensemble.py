#!/usr/bin/env python3
"""
Evaluate an ensemble of trained regression models (point/gauss/laplace).

Outputs per-sample CSVs with ensemble-specific columns (no MC naming):
    id, split, head_type, method=ensemble, n_members,
    y_true, y_pred_method, y_pred_ens_mean,
    sigma_ale_ens, sigma_epi_ens,
    [optional per-member predictions when --include-members]

Behavior:
  - Rebuilds features/targets via DataConfig and stored encoders (preproc_meta.json
    from member_00).
  - For each requested split, runs deterministic forward pass for each member.
  - Aggregates ensemble mean, aleatoric std (mean of member scales), and
    epistemic std (std of member means), mapped to original target units via the
    same delta-method used elsewhere.

Modes:
  - local: run evaluation in-process (default).
  - slurm: submit an sbatch job that re-invokes this script in local mode.

Defaults:
  - If --save-dir is omitted, outputs go to outputs/evals/ensemble_<ensemble_root_basename>/.

Usage example:
  python scripts/eval_ensemble.py \
      --config configs/train_laplace.yaml \
      --ensemble-root outputs/ensembles/laplace_test \
      --splits val test \
      --save-dir outputs/evals/laplace_test_ensemble
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time

import numpy as np
import torch
import yaml

from data import DataConfig, _prepare_frame, _apply_encoders, _target_transform, inverse_target
from model_base import MLPRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    member_preds: Optional[np.ndarray] = None,
):
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
    if member_preds is not None:
        for k in range(member_preds.shape[0]):
            header.append(f"y_pred_member_{k}")

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, mid in enumerate(ids):
            row = [
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
            if member_preds is not None:
                row.extend([float(v) for v in member_preds[:, i]])
            writer.writerow(row)


def evaluate(cfg_path: Path, ensemble_root: Path, splits: List[str], save_dir: Path, include_members: bool) -> None:
    base_cfg = load_yaml(cfg_path)
    members = _member_paths(ensemble_root)
    first_member = members[0]

    meta_path = first_member / "preproc_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"preproc_meta.json not found in {first_member}")
    meta = load_json(meta_path)

    data_cfg = base_cfg["data"]
    model_cfg = base_cfg["model"]
    head_type = model_cfg.get("head_type", "point").lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_all, y_tr_all, extra = _build_features(meta, data_cfg)
    df = extra["df"]
    y_meta = extra["y_meta"]

    # IDs
    if meta.get("id_col") and "id_values" in meta:
        ids_all = np.array(meta["id_values"])
    elif "id" in df.columns:
        ids_all = df["id"].to_numpy()
    else:
        ids_all = np.arange(len(df))

    in_dim = int(meta["feature_dim"])

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, dict] = {}
    start_time = time.perf_counter()
    start_utc = datetime.utcnow().isoformat() + "Z"

    for split in splits:
        split_key = split.lower()
        if split_key not in meta.get("splits", {}):
            print(f"[eval-ensemble] split '{split_key}' not in meta.splits; skipping")
            continue
        idx = np.array(meta["splits"][split_key], dtype=int)
        if idx.size == 0:
            print(f"[eval-ensemble] split '{split_key}' has 0 rows; skipping")
            continue

        Xs = torch.tensor(X_all[idx], dtype=torch.float32)
        y_true_z = y_tr_all[idx].reshape(-1)

        member_means = []
        member_scales = []
        for mdir in members:
            model = _load_model(mdir, in_dim, model_cfg, device)
            mu_z, sigma_z = _run_member_preds(model, Xs, head_type, device)
            member_means.append(mu_z)
            member_scales.append(sigma_z)

        member_means = np.stack(member_means, axis=0)  # [M, N]
        member_scales = np.stack(member_scales, axis=0) if member_scales and member_scales[0] is not None else None

        ens_mean_z = member_means.mean(axis=0)
        if member_scales is not None:
            ale_var_z = (member_scales ** 2).mean(axis=0)
            sigma_ale_z = np.sqrt(ale_var_z)
        else:
            sigma_ale_z = np.full_like(ens_mean_z, np.nan, dtype=float)
        sigma_epi_z = member_means.std(axis=0)

        y_true_orig = inverse_target(y_true_z.reshape(-1, 1), y_meta).reshape(-1)
        y_pred_orig = inverse_target(ens_mean_z.reshape(-1, 1), y_meta).reshape(-1)
        sigma_ale_orig = _delta_sigma_orig(head_type, sigma_ale_z, y_pred_orig, y_meta)
        sigma_epi_orig = _delta_sigma_orig(head_type, sigma_epi_z, y_pred_orig, y_meta)

        ids_split = ids_all[idx]
        member_preds_orig = None
        if include_members:
            member_preds_orig = inverse_target(member_means, y_meta)

        out_csv = save_dir / f"ensemble_preds_{split_key}.csv"
        _write_csv(out_csv, ids_split, split_key, head_type, len(members), y_true_orig, y_pred_orig,
                   sigma_ale_orig, sigma_epi_orig, member_preds_orig)
        print(f"[eval-ensemble] wrote {out_csv} (n={len(idx)})")

        ae = np.abs(y_pred_orig - y_true_orig)
        se = (y_pred_orig - y_true_orig) ** 2
        metrics[split_key] = {
            "mae": float(np.mean(ae)),
            "rmse": float(np.sqrt(np.mean(se))),
            "n": int(len(idx)),
            "head_type": head_type,
            "n_members": len(members),
        }

    duration_sec = time.perf_counter() - start_time
    metrics["_meta"] = {
        "duration_sec": float(duration_sec),
        "start_utc": start_utc,
        "end_utc": datetime.utcnow().isoformat() + "Z",
    }

    with (save_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def submit_slurm(
    cfg_path: Path,
    ensemble_root: Path,
    splits: List[str],
    save_dir: Path,
    include_members: bool,
    base_cfg: dict,
) -> None:
    slurm_cfg = base_cfg.get("slurm", {}) or {}
    partition = slurm_cfg.get("partition", "TEST")
    time_str = slurm_cfg.get("time", "00:30:00")
    mem_gb = int(slurm_cfg.get("mem_gb", 16))
    cpus = int(slurm_cfg.get("cpus", 2))
    gpus = int(slurm_cfg.get("gpus", 0))
    conda_env = slurm_cfg.get("conda_env", "thesis")
    job_name = slurm_cfg.get("job_name", "eval_ensemble")

    logs_root = (REPO_ROOT / "logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_log = logs_root / f"{job_name}_eval_{ts}_%j.out"
    err_log = logs_root / f"{job_name}_eval_{ts}_%j.err"

    split_args = " ".join(splits)
    include_flag = "--include-members" if include_members else ""
    save_dir_str = str(save_dir)

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}_eval",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]
    if gpus > 0:
        script_lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")

    script_lines += [
        f"cd \"{REPO_ROOT}\"",
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        f"conda activate {conda_env}",
        'echo "[env] host=$(hostname) date=$(date)"',
        f"python \"{Path(__file__).resolve()}\" "
        f"--config \"{cfg_path}\" "
        f"--ensemble-root \"{ensemble_root}\" "
        f"--splits {split_args} "
        f"--save-dir \"{save_dir_str}\" "
        f"{include_flag} "
        f"--mode local",
    ]

    sb_script = "\n".join(script_lines) + "\n"
    print("[eval-ensemble][slurm] sbatch script:\n")
    print(sb_script)

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an ensemble of trained regression models")
    ap.add_argument("--config", required=True, help="Base train config used for members")
    ap.add_argument("--ensemble-root", required=True, help="Folder containing member_* subdirs")
    ap.add_argument("--splits", nargs="+", default=["val"], help="Splits to evaluate (train/val/test)")
    ap.add_argument("--save-dir", default=None, help="Where to write outputs (default: outputs/evals/ensemble_<run>)")
    ap.add_argument("--include-members", action="store_true", default=True, help="Include per-member predictions in CSV (default: True)")
    ap.add_argument("--no-include-members", dest="include_members", action="store_false", help="Disable per-member predictions in CSV")
    ap.add_argument("--mode", choices=["local", "slurm"], default="local", help="local: run here; slurm: submit sbatch")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    ensemble_root = Path(args.ensemble_root).resolve()

    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
    else:
        save_dir = (REPO_ROOT / "outputs" / "evals" / f"ensemble_{ensemble_root.name}").resolve()

    if args.mode == "slurm":
        base_cfg = load_yaml(cfg_path)
        submit_slurm(cfg_path, ensemble_root, args.splits, save_dir, args.include_members, base_cfg)
        return

    evaluate(cfg_path, ensemble_root, args.splits, save_dir, args.include_members)


if __name__ == "__main__":
    main()
