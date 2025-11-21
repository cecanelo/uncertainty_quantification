#!/usr/bin/env python3
"""
Export best hyperparameters and pointers for a completed HPO run.

Inputs
------
--config     Path to the HPO YAML used for the run (configs/hpo_config.yaml)
--job-root   The job root folder that contains trials/, e.g.
             outputs/optuna/mnist_cnn_smoke_v1_YYYYMMDD-HHMM_NA
             or outputs/optuna/local_test_job

Outputs
-------
1) {JOB_ROOT}/{best_dirname}/{best_filename}
2) {latest_pointer_dir}/{best_filename}  with {study} resolved

The JSON includes study name, direction, best value, best trial number,
best params, and pointers to the winning trial directory, metrics, and model.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, Optional
import glob
import optuna
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# REPLACE this function in scripts/hpo_export_best.py

def find_trial_dir_for_optuna_number(trials_root: Path, target_trial_no: int) -> Optional[Path]:
    """
    Prefer exact match on run_meta.optuna.trial_number.
    Fallbacks:
      • If exactly one trial dir exists, return it.
      • Else pick the newest dir that contains metrics.json.
    """
    if not trials_root.exists():
        return None

    trial_dirs = [d for d in trials_root.iterdir() if d.is_dir()]

    # 1) Exact match via run_meta.json (written by worker) or hpo_meta.json (future-proof)
    for d in sorted(trial_dirs):
        for meta_name in ("run_meta.json", "hpo_meta.json"):
            meta_path = d / meta_name
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    optuna_meta = meta.get("optuna", {})
                    if int(optuna_meta.get("trial_number", -1)) == int(target_trial_no):
                        return d
                except Exception:
                    pass  # keep looking

    # 2) Single dir fallback
    if len(trial_dirs) == 1:
        return trial_dirs[0]

    # 3) Newest dir that has metrics (json or csv)
    candidates = []
    for d in trial_dirs:
        if ((d / "metrics.json").exists() or (d / "metrics" / "metrics.json").exists()
            or (d / "metrics.csv").exists() or (d / "metrics" / "metrics.csv").exists()):
            candidates.append((d.stat().st_mtime, d))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return None


def best_artifacts(trial_dir: Path) -> Dict[str, Optional[str]]:
    """
    Return likely artifact paths inside the trial directory.
    """
    metrics_candidates = [trial_dir / "metrics.json", trial_dir / "metrics" / "metrics.json"]
    metrics_path = None
    for p in metrics_candidates:
        if p.exists():
            metrics_path = str(p)
            break

    model_candidates = [trial_dir / "model.pt", trial_dir / "artifacts" / "model.pt"]
    model_path = None
    for p in model_candidates:
        if p.exists():
            model_path = str(p)
            break

    return {
        "metrics_path": metrics_path,
        "model_path": model_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export best hparams for an HPO run.")
    ap.add_argument("--config", required=True, help="Path to configs/hpo_config.yaml")
    ap.add_argument("--job-root", required=False, help="Path to the HPO job root that contains trials/")
    ap.add_argument("--job-id", default=None, help="Optional JOBID to include in outdir name (overrides auto-detection).")
    

    args = ap.parse_args()

    cfg = load_yaml(args.config)
    study_cfg = cfg["study"]
    post = cfg.get("post_run", {})
    io_cfg = cfg.get("io", {})

    study = optuna.load_study(study_name=study_cfg["name"], storage=study_cfg["storage"])
    best = study.best_trial

    job_root_arg = args.job_root
    if not job_root_arg and args.job_id:
        base = Path(io_cfg.get("outputs_root", "outputs/optuna"))
        pattern = f"{study_cfg['name']}_*_{args.job_id}*"  # match <study>_<JOBID>_<ts>
        candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if candidates:
            job_root_arg = str(candidates[0].resolve())
            # Prefer the canonical final path if the runner wrote it
            exp = Path(job_root_arg) / "expected_job_root.txt"
            if exp.exists():
                txt = exp.read_text().strip()
                if txt:
                    job_root_arg = txt
        else:
            # Fallback: scan for jobid.txt across outputs_root
            for d in sorted((p for p in base.iterdir() if p.is_dir()),
                            key=lambda p: p.stat().st_mtime, reverse=True):
                jid = d / "jobid.txt"
                try:
                    if jid.exists() and jid.read_text().strip() == str(args.job_id):
                        job_root_arg = str(d.resolve())
                        break
                except Exception:
                    pass
            if not job_root_arg:
                raise SystemExit(
                    f"Could not find job root for JOBID={args.job_id}. "
                    f"Looked for {base}/{pattern} and via jobid.txt scan."
                )

    if not job_root_arg:
        raise SystemExit("You must provide --job-root or --job-id")


    job_root = Path(job_root_arg).resolve()

    # Derive a suggested training outdir that mirrors the HPO job tag.
    # Example:
    #   job_root  = outputs/optuna/point_head_hpo_20251114-1012_277133
    #   job_tag   = point_head_hpo_20251114-1012_277133
    #   parent of parent = outputs
    #   training_root = outputs/training_point_head_hpo_20251114-1012_277133
    job_tag = job_root.name
    try:
        training_root = job_root.parents[1] / f"training_{job_tag}"
    except IndexError:
        # Fallback if the folder structure is unusual: default to ./outputs
        training_root = Path("outputs") / f"training_{job_tag}"

    trials_root = job_root / "trials"
    trial_dir = find_trial_dir_for_optuna_number(trials_root, best.number)


    job_id = None
    jid_file = job_root / "jobid.txt"
    if jid_file.exists():
        try:
            job_id = jid_file.read_text().strip()
        except Exception:
            pass


    payload = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "job_root": str(job_root),
        "job_id": job_id,    
        "study": {
            "name": study.study_name,
            "direction": study.direction.name.lower(),
        },
        "best_value": float(best.value),
        "best_trial_number": int(best.number),
        "params": dict(best.params),
        "trial_dir": str(trial_dir) if trial_dir else None,
        "artifacts": {},
        "io": {
                    "trial_dir_template": io_cfg.get("trial_dir_template"),
                    # Where you should save the final training run that uses these best HPs
                    "suggested_training_outdir": str(training_root),
                },
        "notes": cfg.get("notes", ""),
    }

    if trial_dir:
        payload["artifacts"] = best_artifacts(trial_dir)

    # Where to write
    best_dirname = post.get("best_dirname", "best")
    best_filename = post.get("best_filename", "best_hparams.json")
    export_best = bool(post.get("export_best", True))

    if export_best:
        out1 = job_root / best_dirname / best_filename
        save_json(payload, out1)

        latest_dir_template = post.get("latest_pointer_dir", "")
        if latest_dir_template:
            latest_dir = Path(latest_dir_template.replace("\\", "/").format(study=study.study_name))
            out2 = latest_dir / best_filename
            save_json(payload, out2)

        # Friendly prints
        print(f"[export] Best trial #{best.number} value={best.value:.6f}")
        print(f"[export] Wrote: {out1}")
        if latest_dir_template:
            print(f"[export] Also wrote: {out2}")

        # Convenience: show how to run the HPO analysis script
        if job_id is not None:
            print("[export] To analyze HPO timings, run:")
            print(f"[export]   python scripts/analyze_hpo_job.py --job-id {job_id}")
        else:
            print("[export] To analyze HPO timings, run:")
            print(f"[export]   python scripts/analyze_hpo_job.py --job-root {job_root}")

    else:
        print("[export] post_run.export_best is false. Nothing written.")



if __name__ == "__main__":
    main()
