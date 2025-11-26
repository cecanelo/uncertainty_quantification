#!/usr/bin/env python3
"""
Optuna-based HPO for MC Dropout evaluation.

Each trial:
- Loads a base eval config.
- Applies sampled params (p_drop, n_samples, optional sample_count).
- Overrides io.save_dir to the trial folder.
- Runs eval_mc_dropout.py (local mode).
- Reads metrics.json and reports the chosen metric (e.g., nll) for the target split.
- Saves trial artifacts (config, overrides, copied metrics/run_metadata).

Usage:
    python scripts/hpo_mc_dropout.py --config configs/hpo_mc.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import optuna
import yaml


# ---------- utilities ---------- #
def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def set_by_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def sample_param(spec: dict, trial: optuna.trial.Trial) -> Any:
    dist = spec["dist"]
    name = spec["name"]
    if dist == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if dist == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if dist == "choice":
        return trial.suggest_categorical(name, spec["choices"])
    if dist == "int_uniform":
        return trial.suggest_int(name, spec["low"], spec["high"])
    raise ValueError(f"Unsupported dist: {dist}")


# ---------- main ---------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna HPO for MC Dropout eval.")
    ap.add_argument("--config", type=str, default="configs/hpo_mc.yaml", help="Path to HPO config YAML.")
    ap.add_argument("--submit", action="store_true", help="Submit this HPO run to SLURM using resources from the config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    hpo_cfg = load_yaml(cfg_path)

    study_cfg = hpo_cfg.get("study", {})
    budget_cfg = hpo_cfg.get("budget", {})
    io_cfg = hpo_cfg.get("io", {})
    base_cfg_path = Path(hpo_cfg["base_config"]["eval_config_path"]).resolve()
    search_space: List[Dict[str, Any]] = hpo_cfg.get("search_space", [])
    objective_cfg = hpo_cfg.get("objective", {})
    repro_cfg = hpo_cfg.get("repro", {})
    resources_cfg = hpo_cfg.get("resources", {})

    study_name = study_cfg.get("name", "mc_dropout")
    storage = study_cfg.get("storage", f"sqlite:///optuna_studies/{study_name}.db")
    direction = study_cfg.get("direction", "minimize")
    resume = bool(study_cfg.get("resume", True))
    n_trials = int(budget_cfg.get("n_trials", 20))
    target_split = objective_cfg.get("split", "val")
    target_metric = objective_cfg.get("metric", "nll")

    outputs_root = Path(io_cfg.get("outputs_root", "outputs/optuna_mc")).resolve()
    trial_tpl = io_cfg.get("trial_dir_template", "trial_{trial:05d}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.submit and "SLURM_JOB_ID" not in os.environ:
        partition = resources_cfg.get("partition", "TEST")
        time_str = resources_cfg.get("time", "01:00:00")
        mem_gb = int(resources_cfg.get("mem_gb", 16))
        cpus = int(resources_cfg.get("cpus_per_task", 2))
        gpus = int(resources_cfg.get("gpus", 0))
        conda_env = resources_cfg.get("conda_env", "thesis")
        job_name = resources_cfg.get("job_name", f"hpo_mc_{study_name}")
        logs_root = Path(resources_cfg.get("logs_root", "logs")).resolve()
        logs_root.mkdir(parents=True, exist_ok=True)
        out_log = logs_root / f"{job_name}_{ts}_%j.out"
        err_log = logs_root / f"{job_name}_{ts}_%j.err"

        script_path = Path(__file__).resolve()
        sb_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem_gb}G",
            f"#SBATCH --time={time_str}",
            f"#SBATCH --output={out_log}",
            f"#SBATCH --error={err_log}",
        ]
        if gpus > 0:
            sb_lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")
        sb_lines += [
            f'cd "{script_path.parent.parent}"',
            'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
            f"conda activate {conda_env}",
            'echo "[env] host=$(hostname) date=$(date)"',
            f'python "{script_path}" --config "{cfg_path}"',
        ]
        sb_script = "\n".join(sb_lines) + "\n"
        print("[hpo_mc] Submitting sbatch with script:")
        print(sb_script)
        res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
        if res.returncode != 0:
            print(res.stdout.decode())
            print(res.stderr.decode(), file=sys.stderr)
            res.check_returncode()
        else:
            print(res.stdout.decode().strip())
        return

    job_root = outputs_root / f"{study_name}_{ts}"
    job_root.mkdir(parents=True, exist_ok=True)

    if repro_cfg.get("snapshot_hpo_config", True):
        shutil.copy(cfg_path, job_root / "hpo_config_snapshot.yaml")
    if repro_cfg.get("snapshot_eval_config", True):
        shutil.copy(base_cfg_path, job_root / "eval_config_snapshot.yaml")

    base_eval_cfg = load_yaml(base_cfg_path)

    pruner_cfg = hpo_cfg.get("pruner", {})
    pruner_name = pruner_cfg.get("name", "none")
    pruner: optuna.pruners.BasePruner
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(**(pruner_cfg.get("params") or {}))
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=resume,
        pruner=pruner,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_dir = job_root / trial_tpl.format(trial=trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(base_eval_cfg)
        overrides: Dict[str, Any] = {}
        for spec in search_space:
            val = sample_param(spec, trial)
            overrides[spec["name"]] = val
            set_by_path(cfg, spec["name"], val)

        eval_save_dir = trial_dir / "eval_outputs"
        set_by_path(cfg, "io.save_dir", str(eval_save_dir))

        trial_cfg_path = trial_dir / "eval_config.yaml"
        save_yaml(cfg, trial_cfg_path)

        run_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "eval_mc_dropout.py"),
            "--config",
            str(trial_cfg_path),
            "--mode",
            "local",
        ]

        res = subprocess.run(run_cmd, capture_output=True, text=True)
        (trial_dir / "stdout.txt").write_text(res.stdout)
        (trial_dir / "stderr.txt").write_text(res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"Eval failed for trial {trial.number} (see stderr.txt)")

        metrics_path = eval_save_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
        with metrics_path.open("r") as f:
            metrics = json.load(f)
        if target_split not in metrics:
            raise KeyError(f"Split '{target_split}' not in metrics.json")
        split_metrics = metrics[target_split]
        if target_metric not in split_metrics:
            raise KeyError(f"Metric '{target_metric}' not in metrics for split '{target_split}'")
        objective_value = float(split_metrics[target_metric])

        save_json(overrides, trial_dir / "trial_overrides.json")
        shutil.copy(metrics_path, trial_dir / "metrics.json")
        meta_path = eval_save_dir / "run_metadata.json"
        if meta_path.exists():
            shutil.copy(meta_path, trial_dir / "run_metadata.json")

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("overrides", overrides)
        return objective_value

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[hpo_mc] finished study={study_name} best_value={study.best_value} best_params={study.best_params}")
    save_json(
        {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "study_name": study_name,
            "storage": storage,
            "direction": direction,
            "n_trials": n_trials,
            "job_root": str(job_root),
        },
        job_root / "best_summary.json",
    )


if __name__ == "__main__":
    main()
