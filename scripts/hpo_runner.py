#!/usr/bin/env python3
"""
Submit an Optuna HPO SLURM array from a single YAML config.

What this script does
---------------------
1) Loads configs/hpo_config.yaml.
2) Creates a job root: outputs/optuna/{study}_{YYYYMMDD-HHMM}/
   ├─ manifest/               # snapshots and run_manifest.json
   └─ trials/                 # per-trial artifacts (created by the worker)
3) Optionally snapshots the HPO YAML and base training YAML.
4) Writes manifest/run_manifest.json with indent=4 for easy diffing.
5) Builds the sbatch command and either:
   - prints it ( --dry-run ), or
   - submits it and prints "Submitted batch job <JOBID>".

Environment passed to the worker
--------------------------------
HPO_CONFIG         -> absolute path to configs/hpo_config.yaml
JOB_ROOT           -> absolute path to outputs/optuna/{study}_{ts}/
TRIAL_DIR_TEMPLATE -> e.g., "trial_{trial:05d}"

The worker reads HPO_CONFIG, pulls the study and search space, merges params
into the base training config, runs train.py, and writes trial outputs.

Usage
-----
Dry run (no submission):
    python scripts/hpo_runner.py --config configs/hpo_config.yaml --dry-run

Submit to SLURM:
    python scripts/hpo_runner.py --config configs/hpo_config.yaml
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import optuna
import yaml


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_study_initialized(study_cfg: dict) -> None:
    """
    Ensure the Optuna study and its backing storage are initialized.

    This runs once on the login node BEFORE the SLURM array is submitted,
    so workers do not race to create the Alembic version table.
    """
    storage = study_cfg["storage"]
    name = study_cfg["name"]
    direction = study_cfg["direction"]

    print(f"[hpo_runner] Ensuring Optuna study exists: name={name}, storage={storage}")
    # This is effectively idempotent: load_if_exists=True makes it a no-op
    # if the study/storage are already initialized.
    optuna.create_study(
        study_name=name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit HPO array from YAML.")
    ap.add_argument("--config", required=True, help="Path to configs/hpo_config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and exit.")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    _ensure_study_initialized(cfg["study"])

    # --- Required fields (fail fast) ---
    study = cfg["study"]
    budget = cfg["budget"]
    resources = cfg["resources"]
    io_cfg = cfg["io"]
    base_cfg = cfg["base_config"]
    repro = cfg.get("repro", {})

    n_trials = budget.get("n_trials")
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise SystemExit("hpo_config.yaml: budget.n_trials must be a positive integer for array-based HPO.")

    # --- Naming and folders ---
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    job_tag = io_cfg["job_name_template"].format(study=study["name"], ts=ts, jobid="NA")
    

    outputs_root = io_cfg["outputs_root"]
    logs_root = io_cfg["logs_root"]
    job_root = os.path.join(outputs_root, job_tag)
    manifest_dir = os.path.join(job_root, "manifest")
    trials_root = os.path.join(job_root, "trials")

    _ensure_dir(manifest_dir)
    _ensure_dir(trials_root)
    _ensure_dir(logs_root)

    # --- Snapshots and manifest ---
    manifest = {
        "timestamp": ts,
        "job_tag": job_tag,
        "config_path": os.path.abspath(args.config),
        "study": study,
        "budget": budget,
        "resources": resources,
        "io": io_cfg,
        "base_config": base_cfg,
        "notes": cfg.get("notes", ""),
    }

    if repro.get("snapshot_hpo_config", False):
        dst = os.path.join(manifest_dir, "hpo_config.yaml")
        shutil.copy2(args.config, dst)
        manifest["hpo_config_sha256"] = _sha256(args.config)

    train_cfg_path = base_cfg.get("train_config_path", "")
    if repro.get("snapshot_base_config", False) and train_cfg_path and os.path.exists(train_cfg_path):
        dst = os.path.join(manifest_dir, Path(train_cfg_path).name)
        shutil.copy2(train_cfg_path, dst)
        manifest["base_config_sha256"] = _sha256(train_cfg_path)

    if repro.get("save_pip_freeze", False):
        try:
            pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            with open(os.path.join(manifest_dir, "env_spec.txt"), "w") as f:
                f.write(pip_out)
        except Exception as e:
            print(f"Warning: pip freeze failed: {e}", file=sys.stderr)

    if repro.get("save_git_commit", False):
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            with open(os.path.join(manifest_dir, "git_commit.txt"), "w") as f:
                f.write(git_commit + "\n")
            manifest["git_commit"] = git_commit
        except Exception as e:
            print(f"Warning: git commit not recorded: {e}", file=sys.stderr)

    with open(os.path.join(manifest_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    # --- sbatch command construction ---
    array_spec = f"0-{n_trials - 1}%{resources['max_parallel']}"
    mem_flag = f"{int(resources['mem_gb'])}G"
    job_name = f"hpo-{job_tag}"
    out_pat = os.path.join(logs_root, f"{job_name}_%A_%a.out")
    err_pat = os.path.join(logs_root, f"{job_name}_%A_%a.err")

    worker_script = "scripts/hpo_optuna_worker.py"  # We will provide this next

    gres = []
    gpus = int(resources.get("gpus", 0))
    if gpus > 0:
        gres = ["--gres", f"gpu:{gpus}"]

    export_env = ",".join([
        f"HPO_CONFIG={os.path.abspath(args.config)}",
        f"JOB_ROOT={os.path.abspath(job_root)}",
        f"TRIAL_DIR_TEMPLATE={io_cfg['trial_dir_template']}",
    ])

    sbatch_cmd = [
        "sbatch",
        "-p", resources["partition"],
        "-c", str(resources["cpus_per_task"]),
        "--mem", mem_flag,
        "-t", resources["time"],
        "-J", job_name,
        "--array", array_spec,
        "-o", out_pat,
        "-e", err_pat,
        "--export", f"ALL,{export_env}",
        *gres,
        worker_script,
    ]

    # --- User-facing summary ---
    print(f"Job root: {job_root}")
    print(f"Logs pattern: {out_pat}")
    print(f"Array: {array_spec}")
    print(f"Worker: {worker_script}")

    if args.dry_run:
        print("DRY RUN — not submitting.")
        print("sbatch command:")
        print(" ".join(sbatch_cmd))
        return

    # --- Submit ---
    try:
        res = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
        print(res.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        raise

    # Parse JOBID from 'Submitted batch job <ID>'
    jobid = ""
    try:
        jobid = res.stdout.strip().split()[-1]
    except Exception:
        pass

    # Save the exact sbatch command and job tag for traceability
    with open(os.path.join(job_root, "sbatch_command.txt"), "w") as f:
        f.write(" ".join(sbatch_cmd) + "\n")
    with open(os.path.join(job_root, "job_tag.txt"), "w") as f:
        f.write(job_tag + "\n")
    if jobid:
        with open(os.path.join(job_root, "jobid.txt"), "w") as f:
            f.write(jobid + "\n")
        # Tell the user where the worker will move the folder (no symlink)
        final_root = os.path.join(
            outputs_root,
            io_cfg["job_name_template"].format(study=study["name"], ts=ts, jobid=jobid)
        )
        with open(os.path.join(job_root, "expected_job_root.txt"), "w") as f:
            f.write(os.path.abspath(final_root) + "\n")
        print(f"Job root (will be renamed by worker): {os.path.abspath(job_root)}")
        print(f"Job root (final, with JOBID)       : {os.path.abspath(final_root)}")




if __name__ == "__main__":
    main()
