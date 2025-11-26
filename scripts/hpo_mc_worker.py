#!/usr/bin/env python3
"""
One-trial Optuna worker for MC Dropout HPO (SLURM array friendly).

Environment variables (set by the runner):
  HPO_CONFIG          -> path to configs/hpo_mc.yaml
  JOB_ROOT            -> root folder for this HPO job
  TRIAL_DIR_TEMPLATE  -> e.g., "trial_{trial:05d}"
  SLURM_ARRAY_TASK_ID -> provided by Slurm (falls back to 0 if not set)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import optuna
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def set_by_path(d: dict, dotted: str, value: Any) -> dict:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    return d


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


def main() -> None:
    ap = argparse.ArgumentParser(description="MC Dropout HPO worker (one trial).")
    ap.add_argument("--array-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID when testing locally.")
    args = ap.parse_args()

    hpo_config = os.environ.get("HPO_CONFIG")
    job_root = os.environ.get("JOB_ROOT")
    trial_tpl = os.environ.get("TRIAL_DIR_TEMPLATE", "trial_{trial:05d}")
    array_id = args.array_id
    if array_id is None:
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    if not hpo_config or not job_root:
        raise SystemExit("HPO_CONFIG and JOB_ROOT env vars are required.")

    hpo_path = Path(hpo_config).resolve()
    job_root = Path(job_root).resolve()
    cfg = load_yaml(hpo_path)

    study_cfg = cfg["study"]
    base_cfg_path = Path(cfg["base_config"]["eval_config_path"]).resolve()
    search_space: List[Dict[str, Any]] = cfg.get("search_space", [])
    objective_cfg = cfg.get("objective", {"split": "val", "metric": "nll"})
    target_split = objective_cfg.get("split", "val")
    target_metric = objective_cfg.get("metric", "nll")

    # Prepare trial directory
    trials_root = job_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    trial_dir = trials_root / trial_tpl.format(trial=array_id)
    trial_dir.mkdir(parents=True, exist_ok=True)

    base_eval_cfg = load_yaml(base_cfg_path)

    # Study and objective
    pruner_cfg = cfg.get("pruner", {})
    pruner_name = pruner_cfg.get("name", "none")
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(**(pruner_cfg.get("params") or {}))
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_cfg["name"],
        storage=study_cfg["storage"],
        direction=study_cfg["direction"],
        load_if_exists=study_cfg.get("resume", True),
        pruner=pruner,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        cfg_trial = copy.deepcopy(base_eval_cfg)
        overrides: Dict[str, Any] = {}
        for spec in search_space:
            val = sample_param(spec, trial)
            overrides[spec["name"]] = val
            set_by_path(cfg_trial, spec["name"], val)

        eval_save_dir = trial_dir / "eval_outputs"
        set_by_path(cfg_trial, "io.save_dir", str(eval_save_dir))

        trial_cfg_path = trial_dir / "eval_config.yaml"
        save_yaml(cfg_trial, trial_cfg_path)
        save_json(overrides, trial_dir / "trial_overrides.json")

        run_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "eval_mc_dropout.py"),
            "--config",
            str(trial_cfg_path),
            "--mode",
            "local",
        ]
        res = subprocess.run(run_cmd, cwd=trial_dir, capture_output=True, text=True)
        (trial_dir / "stdout.txt").write_text(res.stdout)
        (trial_dir / "stderr.txt").write_text(res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"Eval failed (returncode {res.returncode}). See stderr.txt")

        metrics_path = eval_save_dir / "metrics.json"
        if not metrics_path.exists():
            # eval_mc_dropout appends job_<id> when SLURM_JOB_ID is set; search recursively
            matches = list(eval_save_dir.rglob("metrics.json"))
            if matches:
                metrics_path = matches[0]
            else:
                raise FileNotFoundError(f"metrics.json not found under {eval_save_dir}")
        metrics = json.loads(metrics_path.read_text())
        if target_split not in metrics:
            raise KeyError(f"Split '{target_split}' not in metrics.json")
        split_metrics = metrics[target_split]
        if target_metric not in split_metrics:
            raise KeyError(f"Metric '{target_metric}' not in metrics for split '{target_split}'")

        # Persist metrics and metadata into the trial folder for convenience
        (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        meta_path = eval_save_dir / "run_metadata.json"
        if meta_path.exists():
            (trial_dir / "run_metadata.json").write_text(meta_path.read_text())

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("overrides", overrides)
        return float(split_metrics[target_metric])

    study.optimize(objective, n_trials=1, gc_after_trial=True, catch=(Exception,))
    print(f"[mc_worker] array_id={array_id} done. study={study_cfg['name']} total_trials={len(study.get_trials())}")


if __name__ == "__main__":
    main()
