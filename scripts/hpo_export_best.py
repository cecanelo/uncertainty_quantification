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
import re
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

def _find_job_root_via_logs(job_id: str, io_cfg: dict) -> Optional[Path]:
    """
    Try to infer the HPO job root from log filenames that contain the JOBID.
    Looks for patterns like hpo-<job_tag>_<JOBID>_<task>.out/err and maps
    job_tag -> outputs_root/job_tag.
    """
    logs_root = Path(io_cfg.get("logs_root", "logs")).resolve()
    outputs_root = Path(io_cfg.get("outputs_root", "outputs/optuna")).resolve()
    if not logs_root.exists():
        return None

    pat = re.compile(r"^(?:hpo-)?(?P<tag>.+?)_" + re.escape(str(job_id)) + r"(?:_|\.|$)")

    try:
        candidates = sorted(logs_root.rglob(f"*{job_id}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        candidates = []

    for p in candidates:
        for nm in (p.name, p.parent.name):
            m = pat.match(nm)
            if not m:
                continue
            job_tag = m.group("tag")
            candidate = outputs_root / job_tag
            if candidate.exists():
                exp = candidate / "expected_job_root.txt"
                if exp.exists():
                    try:
                        txt = exp.read_text().strip()
                        if txt:
                            candidate = Path(txt)
                    except Exception:
                        pass
                return candidate
    return None

def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


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
    trials = study.get_trials(deepcopy=False)
    # Lightweight failure summary to surface OOMs and other issues.
    try:
        from optuna.trial import TrialState
        n_total = len(trials)
        n_complete = sum(t.state == TrialState.COMPLETE for t in trials)
        n_failed = sum(t.state == TrialState.FAIL for t in trials)
        n_pruned = sum(t.state == TrialState.PRUNED for t in trials)
    except Exception:
        n_total = len(trials)
        n_complete = n_failed = n_pruned = 0

    n_oom = 0
    for t in trials:
        attrs = getattr(t, "user_attrs", {}) or {}
        if attrs.get("oom") or attrs.get("failure_kind") == "oom":
            n_oom += 1

    if n_total:
        print(f"[hpo_export_best] Study trials: total={n_total} complete={n_complete} failed={n_failed} pruned={n_pruned}")
    if n_oom:
        print(f"[warn] Detected {n_oom} OOM-killed trials in Optuna metadata for study '{study_cfg['name']}'.")

    try:
        best = study.best_trial
    except Exception as e:
        print(f"[error] No best trial available for study '{study_cfg['name']}': {type(e).__name__}: {e}")
        raise

    # Helper: map optuna trial number -> value for quick lookup
    trial_values = {t.number: t.value for t in study.trials if t.value is not None}
    direction = study.direction.name.lower()

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
            # Extra fallback: infer from log filenames containing JOBID
            if not job_root_arg:
                via_logs = _find_job_root_via_logs(args.job_id, io_cfg)
                if via_logs:
                    job_root_arg = str(via_logs.resolve())
            if not job_root_arg:
                raise SystemExit(
                    f"Could not find job root for JOBID={args.job_id}. "
                    f"Looked for {base}/{pattern}, via jobid.txt scan, and logs."
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
    # Collect available trial directories with their optuna trial numbers (from hpo_meta/run_meta)
    trial_meta = []
    if trials_root.exists():
        for d in sorted(trials_root.iterdir()):
            if not d.is_dir():
                continue
            optuna_no = None
            for meta_name in ("hpo_meta.json", "run_meta.json"):
                meta_path = d / meta_name
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        optuna_no = int(meta.get("optuna", {}).get("trial_number"))
                        break
                    except Exception:
                        pass
            trial_meta.append((d, optuna_no, d.stat().st_mtime))

    chosen_trial_no = int(best.number)
    chosen_value = float(best.value)
    chosen_params = dict(best.params)
    # Maybe switch to the params from the chosen trial (if it differs from study best)
    def _params_for_trial(num: int) -> Optional[dict]:
        for t in study.trials:
            if t.number == num:
                return dict(t.params)
        return None

    trial_dir = find_trial_dir_for_optuna_number(trials_root, chosen_trial_no)
    if trial_dir is None:
        # Fallback: if the global best trial isn't present locally, pick the best trial among
        # the directories we do have (using Optuna values + study direction).
        trial_map = {n: d for d, n, _ in trial_meta if n is not None}
        present_numbers = [n for n in trial_map if n in trial_values]
        def _is_better(a, b):
            return a < b if direction == "minimize" else a > b

        if present_numbers:
            chosen_trial_no = present_numbers[0]
            for n in present_numbers[1:]:
                if _is_better(trial_values[n], trial_values[chosen_trial_no]):
                    chosen_trial_no = n
            trial_dir = trial_map[chosen_trial_no]
            chosen_value = float(trial_values[chosen_trial_no])
            params = _params_for_trial(chosen_trial_no)
            if params is not None:
                chosen_params = params
        elif trial_meta:
            # No values, just pick the newest directory
            newest = sorted(trial_meta, key=lambda x: x[2], reverse=True)[0]
            trial_dir = newest[0]
            if newest[1] is not None and newest[1] in trial_values:
                chosen_trial_no = newest[1]
                chosen_value = float(trial_values[newest[1]])
                params = _params_for_trial(chosen_trial_no)
                if params is not None:
                    chosen_params = params
        # Final guard: direct canonical folder check
        if trial_dir is None:
            cand = trials_root / f"trial_{best.number:05d}"
            if cand.exists():
                trial_dir = cand


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
        # Study best (for reference) and chosen best (usable) may diverge when old trial dirs are missing.
        "best_value": float(chosen_value),
        "best_trial_number": int(chosen_trial_no),
        "study_best_value": float(best.value),
        "study_best_trial_number": int(best.number),
        "params": dict(chosen_params),
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

        # Build a ready-to-run train config for the best trial (mirrors train_from_best.py defaults)
        best_cfg_filename = post.get("best_config_filename", "train_config_from_best.yaml")
        best_cfg_path = None
        try:
            merged_cfg = None
            if trial_dir and (trial_dir / "train_config_merged.yaml").exists():
                merged_cfg = load_yaml(str(trial_dir / "train_config_merged.yaml"))
            else:
                # Fallback: start from base_config if the trial directory is missing
                base_cfg_path = Path(cfg.get("base_config", {}).get("train_config_path", ""))
                if base_cfg_path.exists():
                    merged_cfg = load_yaml(str(base_cfg_path))
            if merged_cfg is not None:
                # Overlay optional post_run.train_overrides (schema matches train config)
                merged_cfg = _deep_merge(merged_cfg, post.get("train_overrides", {}))

                # Re-apply best params defensively (merged config should already contain them)
                for full_key, val in (payload.get("params", {}) or {}).items():
                    if "." not in full_key:
                        continue
                    section, key = full_key.split(".", 1)
                    merged_cfg.setdefault(section, {})
                    if isinstance(merged_cfg.get(section), dict):
                        merged_cfg[section][key] = val

                train_cfg = merged_cfg.get("training", {}) or {}
                train_cfg.setdefault("eval_after_train", True)
                merged_cfg["training"] = train_cfg

                best_cfg_path = out1.parent / best_cfg_filename
                best_cfg_path.parent.mkdir(parents=True, exist_ok=True)
                with open(best_cfg_path, "w") as f:
                    yaml.safe_dump(merged_cfg, f, sort_keys=False)
                payload.setdefault("artifacts", {})
                payload["artifacts"]["train_config"] = str(best_cfg_path)
        except Exception as e:
            print(f"[export] Warning: failed to write best train config: {e}")

        save_json(payload, out1)

        latest_dir_template = post.get("latest_pointer_dir", "")
        if latest_dir_template:
            latest_dir = Path(latest_dir_template.replace("\\", "/").format(study=study.study_name))
            out2 = latest_dir / best_filename
            save_json(payload, out2)

        # Friendly prints
        print(f"[export] Best trial #{best.number} value={best.value:.6f}")
        if payload.get("params"):
            print("[export] Best hyperparameters:")
            for key in sorted(payload["params"].keys()):
                print(f"[export]   {key} = {payload['params'][key]}")
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
