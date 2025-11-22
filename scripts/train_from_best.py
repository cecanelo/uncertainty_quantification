#!/usr/bin/env python3
"""
Train a final model from an HPO export (best_hparams.json) and evaluate on test.

Usage
-----
python scripts/train_from_best.py --best-root outputs/optuna/<study>_<ts>_<jobid>/best

What it does
------------
1. Loads {best-root}/best_hparams.json.
2. Finds:
   - trial_dir -> where train_config_merged.yaml lives
   - io.suggested_training_outdir -> where to put the final training run
     (falls back to outputs/training_<job_tag> if missing).
3. Loads train_config_merged.yaml and:
   - sets training.eval_after_train = True  (so test eval runs)
   - ensures io.evals_root = "outputs/evals"
4. Writes a derived config {best-root}/train_config_from_best.yaml.
5. Calls train_regression.py with:
   - --config = that derived config
   - --outdir = suggested training outdir
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train final model from HPO export (best_hparams.json)."
    )
    parser.add_argument(
        "--best-root",
        type=str,
        required=True,
        help="Folder that contains best_hparams.json (e.g. outputs/optuna/<study>/best)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "slurm"],
        default="slurm",
        help="Run locally or submit an sbatch job (default: slurm).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the training seed stored in the merged config (train_config_merged.yaml)"
    )
    parser.add_argument(
        "--name-suffix", type=str, default="",
        help="Optional suffix appended to job name/outdirs (e.g. 's1', 'tryA')."
    )
    parser.add_argument(
        "--fresh-outdir", action="store_true",
        help="If set, always create a timestamped training/evals outdir to avoid overwrites."
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="STUD",
        help="SLURM partition for --mode slurm (e.g. STUD or TEST).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="02:00:00",
        help="SLURM walltime for --mode slurm (e.g. 02:00:00).",
    )
    parser.add_argument(
        "--mem-gb",
        type=int,
        default=32,
        help="SLURM memory in GB for --mode slurm.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="SLURM CPUs per task for --mode slurm.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to request for --mode slurm.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="train_from_best",
        help="SLURM job name for --mode slurm.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="thesis",
        help="Conda environment to activate inside the SLURM job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch script/command without submitting.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs in the merged config (run longer than HPO).",
    )

    args = parser.parse_args()


    best_root = Path(args.best_root).resolve()
    best_json = best_root / "best_hparams.json"
    if not best_json.is_file():
        raise SystemExit(f"best_hparams.json not found at {best_json}")

    payload = json.loads(best_json.read_text())

    # --- Locate trial_dir ---
    trial_dir_str = payload.get("trial_dir")
    if not trial_dir_str:
        raise SystemExit("best_hparams.json is missing 'trial_dir' field.")
    trial_dir = Path(trial_dir_str).resolve()

    # HPO study name, e.g. "base_full_point_adjusted_2_20251118-1952"
    hpo_name = best_root.parent.name

    # Root for final trainings of this HPO:
    #   output/trainings/training_<HPO_NAME>/
    training_root = (Path("outputs") / "trainings" / f"training_{hpo_name}").resolve()
    training_root.mkdir(parents=True, exist_ok=True)



    cfg_path = trial_dir / "train_config_merged.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"train_config_merged.yaml not found at {cfg_path}")

    # --- Load and tweak config in memory ---
    cfg = yaml.safe_load(cfg_path.read_text())

    # --- Overlay best trial hyperparameters from best_hparams.json ---
    # payload["params"] contains keys like "training.lr", "model.hidden_dims", etc.
    best_params = payload.get("params", {}) or {}

    for full_key, val in best_params.items():
        # full_key examples: "training.lr", "model.hidden_dims"
        if "." not in full_key:
            continue
        section, key = full_key.split(".", 1)

        # If user overrides epochs via CLI, let CLI win later
        if section == "training" and key == "epochs" and args.epochs is not None:
            continue

        cfg.setdefault(section, {})
        cfg[section][key] = val

    # --- Optional: override number of epochs from CLI ---
    if args.epochs is not None:
        train_cfg = cfg.get("training", {}) or {}
        prev_epochs = train_cfg.get("epochs")
        train_cfg["epochs"] = int(args.epochs)
        cfg["training"] = train_cfg
        print(f"[train_from_best] Overriding training.epochs: {prev_epochs} -> {args.epochs}")

    # --- Optional: override seed from CLI ---
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        cfg.setdefault("data", {})
        cfg["data"]["split_seed"] = int(args.seed)
        print(f"[train_from_best] Overriding seed + split_seed -> {args.seed}")


    # --- SLURM resources: allow config defaults, overridden by CLI ---
    slurm_cfg = cfg.get("slurm", {}) or {}
    partition = slurm_cfg.get("partition", args.partition)
    time_str  = slurm_cfg.get("time", args.time)
    mem_gb    = int(slurm_cfg.get("mem_gb", args.mem_gb))
    cpus      = int(slurm_cfg.get("cpus", args.cpus))
    gpus      = int(slurm_cfg.get("gpus", args.gpus))
    if args.job_name != "train_from_best":
        job_name = args.job_name
    else:
        job_name = slurm_cfg.get("job_name", args.job_name)
    conda_env = slurm_cfg.get("conda_env", args.conda_env)


    # --- Build a per-run tag (job_name + optional suffix + optional seed + optional timestamp) ---
    parts = [job_name]
    if args.name_suffix:
        parts.append(args.name_suffix)
    if args.seed is not None:
        parts.append(f"s{args.seed}")
    core_tag = "_".join(parts)

    # Single timestamp reused for run_tag and log filenames
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.fresh_outdir:
        run_tag = f"{core_tag}_{ts}"
    else:
        run_tag = core_tag


    # Final training_outdir: root (from best_hparams) + run_tag
    training_outdir = (training_root / run_tag).resolve()

    # --- IO section: make sure training_outdir and evals_root are set consistently ---
    io_section = cfg.get("io", {}) or {}
    io_section["training_outdir"] = str(training_outdir)
    io_section.setdefault("evals_root", "outputs/evals")
    cfg["io"] = io_section


    # Make sure training section exists and enable test evaluation
    train_cfg = cfg.get("training", {}) or {}
    train_cfg["eval_after_train"] = True
    cfg["training"] = train_cfg

    # # If the config defines a training_outdir, let it override the suggested one
    # training_outdir_cfg = io_section.get("training_outdir")
    # if training_outdir_cfg:
    #     training_outdir = Path(training_outdir_cfg).resolve()

    # Write derived config next to best_hparams.json
    derived_cfg_path = best_root / "train_config_from_best.yaml"
    derived_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # --- Launch training ---
    training_outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/train_regression.py",
        "--config",
        str(derived_cfg_path),
        "--outdir",
        str(training_outdir),
    ]

    print(f"[train_from_best] Using best config from: {cfg_path}")
    print(f"[train_from_best] Training outdir: {training_outdir}")
    print(f"[train_from_best] Evals will be under: {cfg['io']['evals_root']}/{training_outdir.name}")

    if args.mode == "local":
        # Run directly on the login node / current shell
        print("[train_from_best] Running locally:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        # Submit as a SLURM job
        repo_root = Path(__file__).resolve().parents[1]
        logs_dir = repo_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Log filenames: training_<job_name>_<timestamp>_<JOBID>.{out,err}
        safe_job_name = job_name.replace(" ", "_")
        log_stub = f"training_{safe_job_name}_{ts}"
        # %j is expanded by SLURM to the job id
        out_log = logs_dir / f"{log_stub}_%j.out"
        err_log = logs_dir / f"{log_stub}_%j.err"

        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_str}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

cd "{repo_root}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate {conda_env}

echo "[env] host=$(hostname) date=$(date)"
echo "[env] RUN_DIR={training_outdir}_${{SLURM_JOB_ID}}"

{sys.executable} scripts/train_regression.py --config "{derived_cfg_path}" --outdir "{training_outdir}_${{SLURM_JOB_ID}}"

"""
        print("[train_from_best] Submitting sbatch with script:")
        print(sbatch_script)

        if args.dry_run:
            print("[train_from_best] --dry-run set; not submitting.")
        else:
            proc = subprocess.run(
                ["sbatch"],
                input=sbatch_script.encode("utf-8"),
                check=True,
                capture_output=True,
            )
            print(proc.stdout.decode("utf-8").strip())



if __name__ == "__main__":
    main()
