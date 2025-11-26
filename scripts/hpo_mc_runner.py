#!/usr/bin/env python3
"""
Submit an Optuna MC Dropout HPO SLURM array from a YAML config.

Usage:
    python scripts/hpo_mc_runner.py --config configs/hpo_mc.yaml --dry-run   # print sbatch
    python scripts/hpo_mc_runner.py --config configs/hpo_mc.yaml            # submit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit MC Dropout HPO as a SLURM array.")
    ap.add_argument("--config", type=str, default="configs/hpo_mc.yaml", help="Path to HPO config YAML.")
    ap.add_argument("--dry-run", action="store_true", help="Print sbatch script without submitting.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)

    study_cfg = cfg.get("study", {})
    budget_cfg = cfg.get("budget", {})
    io_cfg = cfg.get("io", {})
    resources_cfg = cfg.get("resources", {})
    repro_cfg = cfg.get("repro", {})

    study_name = study_cfg.get("name", "mc_dropout")
    n_trials = int(budget_cfg.get("n_trials", 1))
    outputs_root = Path(io_cfg.get("outputs_root", "outputs/optuna")).resolve()
    trial_tpl = io_cfg.get("trial_dir_template", "trial_{trial:05d}")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_root = outputs_root / f"{study_name}_{ts}"
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "trials").mkdir(exist_ok=True)

    if repro_cfg.get("snapshot_hpo_config", True):
        shutil.copy(cfg_path, job_root / "hpo_config_snapshot.yaml")
    base_cfg_path = Path(cfg["base_config"]["eval_config_path"]).resolve()
    if repro_cfg.get("snapshot_eval_config", True):
        shutil.copy(base_cfg_path, job_root / "eval_config_snapshot.yaml")

    manifest = {
        "study": study_cfg,
        "budget": budget_cfg,
        "io": io_cfg,
        "resources": resources_cfg,
        "job_root": str(job_root),
        "ts": ts,
        "trial_dir_template": trial_tpl,
    }
    save_json(manifest, job_root / "run_manifest.json")

    partition = resources_cfg.get("partition", "TEST")
    time_str = resources_cfg.get("time", "01:00:00")
    mem_gb = int(resources_cfg.get("mem_gb", 16))
    cpus = int(resources_cfg.get("cpus_per_task", 2))
    gpus = int(resources_cfg.get("gpus", 0))
    conda_env = resources_cfg.get("conda_env", "thesis")
    job_name = resources_cfg.get("job_name", f"hpo_mc_{study_name}")
    logs_root = Path(resources_cfg.get("logs_root", "logs")).resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    out_log = logs_root / f"{job_name}_{ts}_%A_%a.out"
    err_log = logs_root / f"{job_name}_{ts}_%A_%a.err"
    max_parallel = int(resources_cfg.get("max_parallel", n_trials))

    script_path = Path(__file__).resolve()
    worker_path = script_path.parent / "hpo_mc_worker.py"

    if max_parallel < 1:
        max_parallel = 1
    array_spec = f"0-{n_trials - 1}"
    if n_trials > 1 and max_parallel < n_trials:
        array_spec = f"{array_spec}%{max_parallel}"

    sb_lines = [
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
        sb_lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")

    sb_lines += [
        f'export HPO_CONFIG="{cfg_path}"',
        f'export JOB_ROOT="{job_root}"',
        f'export TRIAL_DIR_TEMPLATE="{trial_tpl}"',
        f'cd "{script_path.parent.parent}"',
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        f"conda activate {conda_env}",
        'echo "[env] host=$(hostname) date=$(date) array=$SLURM_ARRAY_TASK_ID"',
        f'python "{worker_path}"',
    ]
    sb_script = "\n".join(sb_lines) + "\n"

    print("[hpo_mc_runner] sbatch script:\n")
    print(sb_script)

    if args.dry_run:
        print("[hpo_mc_runner] dry run, not submitting.")
        return

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


if __name__ == "__main__":
    main()
