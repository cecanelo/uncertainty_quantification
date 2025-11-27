#!/usr/bin/env python3
"""
Train an ensemble of regression models by launching `train_regression.py` once per member.

Supports:
  - local mode: sequentially trains all members in-process.
  - slurm mode: submits an array job, one task per member (like HPO runners).

Config expectations:
  - Base train config (YAML) should include an `ensemble` block:
      ensemble:
        enabled: false
        n_members: 5
  - When run in ensemble mode, we read `n_members` from that block.
  - Each member gets its own seed: base seed + member_idx (applied to cfg["seed"] and data.split_seed).
  - Member outdirs are under the specified ensemble_root: ensemble_root/member_{idx:02d}

Usage examples:
  Local:
    python scripts/train_ensemble.py --config configs/train_laplace.yaml \
        --ensemble-root outputs/ensembles/laplace_test --mode local

  Slurm:
    python scripts/train_ensemble.py --config configs/train_laplace.yaml \
        --ensemble-root outputs/ensembles/laplace_test --mode slurm

Notes:
  - We intentionally leave `train_regression.py` unchanged; this script just orchestrates.
  - The base config is snapshotted into ensemble_root/base_config.yaml for reproducibility.
  - If --ensemble-root is omitted, defaults to outputs/ensembles/<job|cfg>_<timestamp>.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_regression.py"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _member_outdir(root: Path, idx: int) -> Path:
    return root / f"member_{idx:02d}"


def _build_member_config(base_cfg: Dict[str, Any], member_idx: int, seed: int, member_outdir: Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON for simplicity

    cfg["seed"] = int(seed)
    cfg.setdefault("data", {})
    cfg["data"]["split_seed"] = int(seed)

    # Wire IO outdir for preproc/meta and model artifacts
    cfg.setdefault("io", {})
    cfg["io"]["outdir"] = str(member_outdir)

    # Leave eval_after_train as-is (user-controlled)
    return cfg


def _run_member_local(cfg_path: Path, outdir: Path) -> None:
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", str(cfg_path), "--outdir", str(outdir)]
    print(f"[ensemble][local] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _write_manifest(root: Path, base_cfg_path: Path, n_members: int, mode: str, ts: str) -> None:
    manifest = {
        "base_config": str(base_cfg_path),
        "n_members": int(n_members),
        "mode": mode,
        "timestamp": ts,
    }
    with (root / "ensemble_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=4)


def submit_slurm_array(base_cfg_path: Path, ensemble_root: Path, n_members: int, base_cfg: Dict[str, Any]) -> None:
    # Pull slurm defaults from base config
    slurm_cfg = base_cfg.get("slurm", {}) or {}
    partition = slurm_cfg.get("partition", "TEST")
    time_str = slurm_cfg.get("time", "01:00:00")
    mem_gb = int(slurm_cfg.get("mem_gb", 16))
    cpus = int(slurm_cfg.get("cpus", 2))
    gpus = int(slurm_cfg.get("gpus", 0))
    conda_env = slurm_cfg.get("conda_env", "thesis")
    job_name = slurm_cfg.get("job_name", "ensemble")
    logs_root = Path("logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_log = logs_root / f"{job_name}_{ts}_%A_%a.out"
    err_log = logs_root / f"{job_name}_{ts}_%A_%a.err"

    array_spec = f"0-{n_members - 1}"
    max_parallel = int(slurm_cfg.get("max_parallel", n_members))
    if n_members > 1 and 1 <= max_parallel < n_members:
        array_spec = f"{array_spec}%{max_parallel}"

    script_path = Path(__file__).resolve()
    this_script = script_path

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
        f"#SBATCH --array={array_spec}",
    ]
    if gpus > 0:
        lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")

    exports = [
        f"ENSEMBLE_CONFIG={base_cfg_path}",
        f"ENSEMBLE_ROOT={ensemble_root}",
        f"N_MEMBERS={n_members}",
    ]
    lines += [
        f"export {' '.join(exports)}",
        f"cd '{REPO_ROOT}'",
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        f"conda activate {conda_env}",
        'echo "[env] host=$(hostname) date=$(date) array=$SLURM_ARRAY_TASK_ID"',
        f"python '{this_script}' --config '{base_cfg_path}' --ensemble-root '{ensemble_root}' --mode array-worker",
    ]

    sb_script = "\n".join(lines) + "\n"
    print("[ensemble][slurm] sbatch script:\n")
    print(sb_script)

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an ensemble of models using train_regression.py")
    ap.add_argument("--config", required=True, help="Base train config YAML")
    ap.add_argument("--ensemble-root", required=False, help="Root folder for ensemble (default: outputs/ensembles/<job|config>_<ts>)")
    ap.add_argument("--mode", choices=["local", "slurm", "array-worker"], default="local",
                    help="local: run all members here; slurm: submit array; array-worker: internal")
    ap.add_argument("--base-seed", type=int, default=None, help="Override base seed (else uses cfg['seed'])")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    base_cfg = load_yaml(cfg_path)

    if args.ensemble_root:
        ensemble_root = Path(args.ensemble_root).resolve()
    else:
        # Default: outputs/ensembles/<job_name_or_cfgstem>_<ts>
        tag = base_cfg.get("slurm", {}).get("job_name") or cfg_path.stem
        ts_root = datetime.now().strftime("%Y%m%d-%H%M%S")
        ensemble_root = (REPO_ROOT / "outputs" / "ensembles" / f"{tag}_{ts_root}").resolve()
    ensemble_root.mkdir(parents=True, exist_ok=True)

    ens_cfg = base_cfg.get("ensemble", {}) or {}
    n_members = int(ens_cfg.get("n_members", 1))
    if n_members < 1:
        raise SystemExit("ensemble.n_members must be >= 1")

    # Snapshot base config
    snapshot_path = ensemble_root / "base_config.yaml"
    if not snapshot_path.exists():
        save_yaml(base_cfg, snapshot_path)

    # Manifest
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    _write_manifest(ensemble_root, snapshot_path, n_members, args.mode, ts)

    base_seed = args.base_seed if args.base_seed is not None else int(base_cfg.get("seed", 42))

    if args.mode == "slurm":
        submit_slurm_array(cfg_path, ensemble_root, n_members, base_cfg)
        return

    if args.mode == "array-worker":
        task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id_str is None:
            raise SystemExit("array-worker mode requires SLURM_ARRAY_TASK_ID")
        m_idx = int(task_id_str)
        if m_idx < 0 or m_idx >= n_members:
            raise SystemExit(f"Invalid member index {m_idx} for n_members={n_members}")
        seed = base_seed + m_idx
        member_outdir = _member_outdir(ensemble_root, m_idx)
        member_outdir.mkdir(parents=True, exist_ok=True)
        member_cfg = _build_member_config(base_cfg, m_idx, seed, member_outdir)
        member_cfg_path = member_outdir / "member_config.yaml"
        save_yaml(member_cfg, member_cfg_path)
        _run_member_local(member_cfg_path, member_outdir)
        return

    # local mode: loop over all members sequentially
    for m_idx in range(n_members):
        seed = base_seed + m_idx
        member_outdir = _member_outdir(ensemble_root, m_idx)
        member_outdir.mkdir(parents=True, exist_ok=True)
        member_cfg = _build_member_config(base_cfg, m_idx, seed, member_outdir)
        member_cfg_path = member_outdir / "member_config.yaml"
        save_yaml(member_cfg, member_cfg_path)
        _run_member_local(member_cfg_path, member_outdir)


if __name__ == "__main__":
    main()
