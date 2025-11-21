#!/usr/bin/env python3
"""
One-trial worker for Optuna + SLURM arrays.

What this script does
---------------------
• Reads HPO config from HPO_CONFIG (env).
• Uses SLURM_ARRAY_TASK_ID to decide this trial's working folder name.
• Connects/resumes Optuna study (from YAML).
• Samples hyperparameters per the YAML search_space.
• Merges samples into base training config (deep merge by dotted keys).
• Writes trial artifacts:
    - trial_overrides.json     (only the sampled params)
    - train_config_merged.yaml (what train.py will read)
    - run_meta.json            (SLURM/job info, assigned Optuna trial no.)
• Runs train.py in the trial directory (cwd), so metrics land there.
• Reads metrics.json and reports objective (val_loss) back to Optuna.

Environment variables
---------------------
HPO_CONFIG:          path to configs/hpo_config.yaml
JOB_ROOT:            job root like outputs/optuna/{study}_{ts}_{jobid}
TRIAL_DIR_TEMPLATE:  e.g., "trial_{trial:05d}"
SLURM_ARRAY_TASK_ID: provided by Slurm (falls back to 0 if not set)

CLI
---
You can override the array id when testing locally:

    python scripts/hpo_optuna_worker.py --array-id 0
"""
from pathlib import Path
import os
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import optuna
import yaml


# ----------------------- small utilities ----------------------- #
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def deep_update(base: dict, patch: dict) -> dict:
    """Deep merge two dicts."""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_by_dotted_key(d: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    """Return a tiny dict like {"a":{"b":{"c":value}}} for "a.b.c"."""
    parts = dotted.split(".")
    out: Dict[str, Any] = {}
    cur = out
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return out


def sample_param(trial: optuna.trial.Trial, spec: dict) -> Any:
    """Map YAML dist spec to an Optuna suggest_* call."""
    name = spec["name"]
    dist = spec["dist"].lower()

    if dist == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)

    if dist in ("uniform", "float"):
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        if step is not None:
            return trial.suggest_float(name, low, high, step=float(step))
        return trial.suggest_float(name, low, high)

    if dist == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = spec.get("step")
        if step is not None:
            return trial.suggest_int(name, low, high, step=int(step))
        return trial.suggest_int(name, low, high)

    if dist == "choice":
        choices = spec["choices"]
        return trial.suggest_categorical(name, choices)

    raise ValueError(f"Unsupported dist: {dist} for param {name}")


def build_sampler(cfg: dict) -> optuna.samplers.BaseSampler:
    s = cfg.get("sampler", {})
    name = s.get("name", "TPE").lower()
    params = s.get("params", {}) or {}
    if name == "tpe":
        return optuna.samplers.TPESampler(**params)
    if name == "random":
        return optuna.samplers.RandomSampler(**params)
    # Default to TPE if unknown
    return optuna.samplers.TPESampler(**params)


def build_pruner(cfg: dict) -> optuna.pruners.BasePruner:
    p = cfg.get("pruner", {})
    name = p.get("name", "none").lower()
    params = p.get("params", {}) or {}
    if name == "median":
        return optuna.pruners.MedianPruner(**params)
    if name == "none":
        return optuna.pruners.NopPruner()
    # Default to no pruning if unknown
    return optuna.pruners.NopPruner()


# ----------------------- main worker logic ----------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID for local tests.")
    args = ap.parse_args()

    # Read required envs
    HPO_CONFIG = os.environ.get("HPO_CONFIG")
    JOB_ROOT = os.environ.get("JOB_ROOT")
    TRIAL_DIR_TEMPLATE = os.environ.get("TRIAL_DIR_TEMPLATE", "trial_{trial:05d}")

    # Canonical job root: just use JOB_ROOT as passed by the runner.
    # We no longer rename folders (that caused duplicate roots with arrays).
    job_root = Path(JOB_ROOT).resolve()
    slurm_jobid = os.environ.get("SLURM_ARRAY_JOB_ID",
                                 os.environ.get("SLURM_JOB_ID", "")).strip()

    # Ensure the directory exists (runner already created it, but be defensive).
    job_root.mkdir(parents=True, exist_ok=True)

    # Expose the resolved root to any children.
    os.environ["JOB_ROOT"] = str(job_root)

    # Always write jobid.txt for downstream tools
    try:
        (job_root / "jobid.txt").write_text(slurm_jobid + "\n")
    except Exception:
        pass

    if not HPO_CONFIG or not JOB_ROOT:
        raise SystemExit("HPO_CONFIG and JOB_ROOT must be set in environment.")

    array_id = args.array_id
    if array_id is None:
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    cfg = load_yaml(HPO_CONFIG)
    study_cfg = cfg["study"]
    base_cfg = cfg["base_config"]
    io_cfg = cfg["io"]

    # ---- Per-job log folder: logs/<jobname>_<jobid>_<ts>/ ----
    logs_root = Path(io_cfg.get("logs_root", "logs")).resolve()
    job_name = os.environ.get("SLURM_JOB_NAME", "job")
    job_id   = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
    task_id  = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    # Build the final folder name by swapping _NA → _<jobid> if present
    desired_name = job_name.replace("_NA_", f"_{job_id}_").replace("_NA", f"_{job_id}")
    logdir = logs_root / desired_name
    logdir.mkdir(parents=True, exist_ok=True)

    # Current files that SLURM already opened (support both '_' and '-' variants)
    candidates_out = [
        logs_root / f"{job_name}_{job_id}_{task_id}.out",
        logs_root / f"{job_name}-{job_id}_{task_id}.out",
    ]
    candidates_err = [
        logs_root / f"{job_name}_{job_id}_{task_id}.err",
        logs_root / f"{job_name}-{job_id}_{task_id}.err",
    ]

    try:
        for p in candidates_out:
            if p.exists():
                p.rename(logdir / p.name)
                break
        for p in candidates_err:
            if p.exists():
                p.rename(logdir / p.name)
                break
        # Convenience pointer
        try:
            (logs_root / "last_job").unlink(missing_ok=True)
        except Exception:
            pass
        try:
            os.symlink(logdir, logs_root / "last_job", target_is_directory=True)
        except Exception:
            pass
    except Exception:
        pass

    # Let child processes reuse the same folder
    os.environ["LOGDIR"] = str(logdir)



    # Folder setup for this trial
    trials_root = job_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    trial_dir_name = TRIAL_DIR_TEMPLATE.format(trial=array_id)
    trial_dir = trials_root / trial_dir_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Optuna study
    sampler = build_sampler(cfg)
    pruner = build_pruner(cfg)
    study = optuna.create_study(
        study_name=study_cfg["name"],
        storage=study_cfg["storage"],
        direction=study_cfg["direction"],
        load_if_exists=study_cfg.get("resume", True),
        sampler=sampler,
        pruner=pruner,
    )

    # Objective for exactly one trial
    def objective(trial: optuna.trial.Trial) -> float:
        # 1) Sample per YAML search_space
        overrides: Dict[str, Any] = {}
        sampled_params: Dict[str, Any] = {}
        for p in cfg["search_space"]:
            val = sample_param(trial, p)
            sampled_params[p["name"]] = val
            # turn "a.b.c" into a nested dict and merge
            overrides = deep_update(overrides, set_by_dotted_key({}, p["name"], val))

        # 2) Prepare merged training config
        base_train = load_yaml(base_cfg["train_config_path"])
        merged = deep_update(base_train, overrides)

        # 3) Persist trial artifacts
        save_json(sampled_params, str(trial_dir / "trial_overrides.json"))
        save_yaml(merged, str(trial_dir / "train_config_merged.yaml"))

        # Keep HPO metadata separate so train.py cannot overwrite it
        hpo_meta = {
            "optuna": {
                "trial_number": trial.number,
                "study_name": study_cfg["name"],
            },
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "array_id": array_id,
                "node": os.uname().nodename,
            },
            "io": {
                "trial_dir": str(trial_dir),
                "logs_root": cfg["io"]["logs_root"],
            },
        }
        save_json(hpo_meta, str(trial_dir / "hpo_meta.json"))


        import time, signal, csv


        repo_root = Path(HPO_CONFIG).resolve().parents[1]
        # New: allow overriding the train script via base_config.train_script_path
        train_script_cfg = base_cfg.get("train_script_path")
        if train_script_cfg:
            train_script = str(Path(train_script_cfg).resolve())
        else:
            train_script = str(repo_root / "scripts" / "train.py")

        cmd = [
            sys.executable,
            train_script,
            "--config", str(trial_dir / "train_config_merged.yaml"),
            "--outdir", ".",
        ]

        proc = subprocess.Popen(cmd, cwd=str(trial_dir))
        metrics_path = trial_dir / "metrics.csv"
        last_epoch_reported = -1

        def _try_report_latest():
            nonlocal last_epoch_reported
            if not metrics_path.exists():
                return
            # Read the last non-empty row from metrics.csv
            try:
                with open(metrics_path, "r", newline="") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    return
                header = rows[0]
                data_rows = [r for r in rows[1:] if any(cell.strip() for cell in r)]
                if not data_rows:
                    return
                last = data_rows[-1]
                colmap = {name: i for i, name in enumerate(header)}
                obj_col = None
                if "objective" in colmap:
                    obj_col = "objective"
                elif "val_loss" in colmap:
                    obj_col = "val_loss"
                else:
                    return
                vloss = float(last[colmap[obj_col]])

                # epoch column is nice-to-have; otherwise step = number of data rows
                step = None
                if "epoch" in colmap:
                    try:
                        step = int(float(last[colmap["epoch"]]))
                    except Exception:
                        step = None
                if step is None:
                    step = len(data_rows)
                if step <= last_epoch_reported:
                    return
                # Report to Optuna pruner
                trial.report(vloss, step=step)
                last_epoch_reported = step
            except Exception:
                # Be forgiving: any parse error just skips this tick
                pass

        # Poll until process exits; if pruner triggers, stop process and prune
        try:
            while True:
                ret = proc.poll()
                _try_report_latest()
                if last_epoch_reported >= 0 and trial.should_prune():
                    # Ask train.py to stop gracefully, then escalate if needed
                    try:
                        proc.send_signal(signal.SIGINT)
                        for _ in range(20):
                            time.sleep(0.2)
                            if proc.poll() is not None:
                                break
                        if proc.poll() is None:
                            proc.terminate()
                            time.sleep(1.0)
                        if proc.poll() is None:
                            proc.kill()
                    finally:
                        import optuna
                        raise optuna.exceptions.TrialPruned(f"Pruned at step={last_epoch_reported}")
                if ret is not None:
                    if ret != 0:
                        raise RuntimeError(f"train.py failed with exit code {ret}")
                    break
                time.sleep(0.5)
        except Exception:
            # Ensure subprocess is not left running
            try:
                if proc.poll() is None:
                    proc.kill()
            finally:
                raise

        # 5) Read metrics.json (source of truth for objective)
        # Prefer an explicit "objective" field if present; fall back to "val_loss".
        mjson = trial_dir / "metrics.json"
        if not mjson.exists():
            raise RuntimeError("metrics.json not found.")
        payload = json.loads(mjson.read_text())
        obj = payload.get("objective", payload.get("val_loss"))
        if obj is None:
            raise RuntimeError("metrics.json missing both 'objective' and 'val_loss'.")
        return float(obj)


    # Optimize exactly one trial in this worker
    study.optimize(objective, n_trials=1, gc_after_trial=True, catch=(Exception,))

    # Friendly print for the SLURM .out
    print(f"[worker] array_id={array_id} done. Study={study_cfg['name']}, trials_total={len(study.get_trials())}")


if __name__ == "__main__":
    main()
