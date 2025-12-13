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
import re
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
        required=False,
        help="Folder that contains best_hparams.json (e.g. outputs/optuna/<study>/best)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Optional SLURM JOBID to locate the HPO job automatically (requires --hpo-config).",
    )
    parser.add_argument(
        "--hpo-config",
        type=str,
        default=None,
        help="HPO config used for the run (needed to resolve job-id when --best-root is omitted).",
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
        help="Create a timestamped training/evals outdir to avoid overwrites (default: True)."
    )
    parser.add_argument(
        "--no-fresh-outdir", dest="fresh_outdir", action="store_false",
        help="Reuse the core tag (no timestamp) for training/evals outdir."
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition for --mode slurm (overrides config when set; otherwise uses config or STUD).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="SLURM walltime for --mode slurm (e.g. 02:00:00).",
    )
    parser.add_argument(
        "--mem-gb",
        type=int,
        default=None,
        help="SLURM memory in GB for --mode slurm.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="SLURM CPUs per task for --mode slurm.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to request for --mode slurm.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="SLURM job name for --mode slurm.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
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

    parser.set_defaults(fresh_outdir=True)

    args = parser.parse_args()

    def _find_job_root_from_logs(job_id: str, io_cfg: dict) -> Path | None:
        logs_root = Path(io_cfg.get("logs_root", "logs")).resolve()
        outputs_root = Path(io_cfg.get("outputs_root", "outputs/optuna")).resolve()
        pat = re.compile(r"^(?:hpo-)?(?P<tag>.+?)_" + re.escape(str(job_id)) + r"(?:_|\.|$)")
        if not logs_root.exists():
            return None
        try:
            candidates = sorted(logs_root.rglob(f"*{job_id}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            candidates = []
        for p in candidates:
            for nm in (p.name, p.parent.name):
                m = pat.match(nm)
                if not m:
                    continue
                tag = m.group("tag")
                candidate = outputs_root / tag
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

    def _resolve_trial_dir(job_root: Path, payload: dict) -> Path | None:
        """Best-effort resolution of a trial directory from job_root/trials."""
        trials_root = job_root / "trials"
        if not trials_root.exists():
            return None
        meta = []
        for d in sorted(trials_root.iterdir()):
            if not d.is_dir():
                continue
            optuna_no = None
            for meta_name in ("hpo_meta.json", "run_meta.json"):
                mp = d / meta_name
                if mp.exists():
                    try:
                        m = json.loads(mp.read_text())
                        optuna_no = int(m.get("optuna", {}).get("trial_number"))
                        break
                    except Exception:
                        pass
            meta.append((d, optuna_no, d.stat().st_mtime))

        target_no = payload.get("best_trial_number") or payload.get("study_best_trial_number")
        if target_no is not None:
            for d, no, _ in meta:
                if no is not None and int(no) == int(target_no):
                    return d
        if meta:
            return sorted(meta, key=lambda x: x[2], reverse=True)[0][0]
        return None

    best_root = None
    if args.best_root:
        best_root = Path(args.best_root).resolve()
    elif args.job_id:
        if not args.hpo_config:
            raise SystemExit("--job-id requires --hpo-config to resolve outputs_root/logs_root")
        hpo_cfg = yaml.safe_load(Path(args.hpo_config).read_text())
        io_cfg = hpo_cfg.get("io", {}) or {}
        outputs_root = Path(io_cfg.get("outputs_root", "outputs/optuna")).resolve()
        study = hpo_cfg.get("study", {}).get("name", "")
        pattern = f"{study}_*_{args.job_id}*"
        candidates = sorted(outputs_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            best_root = candidates[0] / "best"
            exp = candidates[0] / "expected_job_root.txt"
            if exp.exists():
                try:
                    txt = exp.read_text().strip()
                    if txt:
                        best_root = Path(txt) / "best"
                except Exception:
                    pass
        if best_root is None:
            via_logs = _find_job_root_from_logs(str(args.job_id), io_cfg)
            if via_logs:
                best_root = via_logs.resolve() / "best"
        if best_root is None:
            raise SystemExit(f"Could not resolve best-root for JOBID={args.job_id}")
    else:
        raise SystemExit("Provide --best-root or --job-id (with --hpo-config)")

    best_json = best_root / "best_hparams.json"
    if not best_json.is_file():
        raise SystemExit(f"best_hparams.json not found at {best_json}")

    payload = json.loads(best_json.read_text())

    # --- Locate trial_dir ---
    trial_dir = None
    trial_dir_str = payload.get("trial_dir")
    if trial_dir_str:
        trial_dir = Path(trial_dir_str).resolve()
    if trial_dir is None:
        fallback_trial_dir = _resolve_trial_dir(best_root.parent, payload)
        if fallback_trial_dir:
            print(f"[train_from_best] Resolved trial_dir from job root: {fallback_trial_dir}")
            trial_dir = fallback_trial_dir

    # HPO study name from the export payload (preferred over folder name)
    hpo_name = payload.get("study", {}).get("name")
    if not hpo_name:
        # Fallback to folder name if the payload is missing the study info
        hpo_name = best_root.parent.name

    # Root for final trainings of this HPO:
    #   output/trainings/training_<HPO_NAME>/
    training_root = (Path("outputs") / "trainings" / f"training_{hpo_name}").resolve()
    training_root.mkdir(parents=True, exist_ok=True)



    exported_cfg_path = best_root / "train_config_from_best.yaml"
    merged_cfg_path = trial_dir / "train_config_merged.yaml" if trial_dir else None

    # --- Load base config ---
    if exported_cfg_path.is_file():
        cfg = yaml.safe_load(exported_cfg_path.read_text())
        print(f"[train_from_best] Using exported config: {exported_cfg_path}")
        best_params_applied = True
    else:
        if not merged_cfg_path or not merged_cfg_path.is_file():
            raise SystemExit(
                "Could not locate a training config to use. "
                "Expected either train_config_from_best.yaml or a valid trial_dir/train_config_merged.yaml. "
                "Re-run hpo_export_best.py to regenerate the best config."
            )
        cfg = yaml.safe_load(merged_cfg_path.read_text())
        best_params_applied = False

    # --- Overlay best trial hyperparameters if not already baked in ---
    if not best_params_applied:
        best_params = payload.get("params", {}) or {}
        for full_key, val in best_params.items():
            if "." not in full_key:
                continue
            section, key = full_key.split(".", 1)
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
    # CLI overrides take priority when provided; otherwise fall back to config, then a safe default.
    partition = args.partition if args.partition is not None else slurm_cfg.get("partition", "STUD")
    time_str  = args.time if args.time is not None else slurm_cfg.get("time", "02:00:00")
    mem_gb    = int(args.mem_gb) if args.mem_gb is not None else int(slurm_cfg.get("mem_gb", 32))
    cpus      = int(args.cpus) if args.cpus is not None else int(slurm_cfg.get("cpus", 4))
    gpus      = int(args.gpus) if args.gpus is not None else int(slurm_cfg.get("gpus", 0))
    if args.job_name is not None:
        job_name = args.job_name
    else:
        # Prefer the HPO study name as the default job name to keep lineage clear
        job_name = hpo_name
    conda_env = args.conda_env if args.conda_env is not None else slurm_cfg.get("conda_env", "thesis")


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
    final_outdir = f"{training_outdir}_${{SLURM_JOB_ID}}"

    # --- IO section: make sure training_outdir and evals_root are set consistently ---
    io_section = cfg.get("io", {}) or {}
    io_section["training_outdir"] = str(final_outdir)
    io_section.setdefault("evals_root", "outputs/evals")
    cfg["io"] = io_section


    # Make sure training section exists and enable test evaluation
    train_cfg = cfg.get("training", {}) or {}
    train_cfg["eval_after_train"] = True
    cfg["training"] = train_cfg

    # --- Persist resolved slurm overrides back into the derived config ---
    cfg["slurm"] = {
        "partition": partition,
        "time": time_str,
        "mem_gb": mem_gb,
        "cpus": cpus,
        "gpus": gpus,
        "job_name": job_name,
        "conda_env": conda_env,
    }

    # # If the config defines a training_outdir, let it override the suggested one
    # training_outdir_cfg = io_section.get("training_outdir")
    # if training_outdir_cfg:
    #     training_outdir = Path(training_outdir_cfg).resolve()

    # Write the exact config that will be used for training
    derived_cfg_path = best_root / "train_config_to_use.yaml"
    derived_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Decide which trainer to call: regression vs flows vs dido
    is_flow_cfg = "nf" in cfg and ("model" not in cfg or "objective" not in cfg)
    is_dido_cfg = ("binning" in cfg) and ("nf" not in cfg) and ("model" not in cfg or "objective" not in cfg)
    if is_dido_cfg:
        trainer_script = "scripts/train_dido.py"
    elif is_flow_cfg:
        trainer_script = "scripts/train_flows.py"
    else:
        trainer_script = "scripts/train_regression.py"

    # --- Launch training ---
    cmd = [
        sys.executable,
        trainer_script,
        "--config",
        str(derived_cfg_path),
        "--outdir",
        str(final_outdir),
    ]

    print(f"[train_from_best] Using best config from: {derived_cfg_path}")
    print(f"[train_from_best] Training outdir: {final_outdir}")
    print(f"[train_from_best] Evals will be under: {cfg['io']['evals_root']}/{Path(final_outdir).name}")

    if args.mode == "local":
        # Run directly on the login node / current shell
        print("[train_from_best] Running locally:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        # Submit as a SLURM job
        repo_root = Path(__file__).resolve().parents[1]
        logs_dir = repo_root / "logs" / "train"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Log filenames: training_<job_name>_<timestamp>_<JOBID>.{out,err}
        safe_job_name = job_name.replace(" ", "_")
        log_stub = f"training_{safe_job_name}_{ts}"
        # %j is expanded by SLURM to the job id
        out_log = logs_dir / f"{log_stub}_%j.out"
        err_log = logs_dir / f"{log_stub}_%j.err"

        gres_line = f"#SBATCH --gres=gpu:{gpus}" if gpus and gpus > 0 else ""

        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
{gres_line}
#SBATCH --time={time_str}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

cd "{repo_root}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate {conda_env}

echo "[env] host=$(hostname) date=$(date)"
OUTDIR="{final_outdir}"
echo "[env] RUN_DIR=$OUTDIR"
{sys.executable} {trainer_script} --config "{derived_cfg_path}" --outdir "$OUTDIR"
"""

        print("[train_from_best] Submitting sbatch with script:")
        print(sbatch_script)

        if args.dry_run:
            print("[train_from_best] --dry-run set; not submitting.")
        else:
            try:
                proc = subprocess.run(
                    ["sbatch"],
                    input=sbatch_script.encode("utf-8"),
                    check=True,
                    capture_output=True,
                )
                print(proc.stdout.decode("utf-8").strip())
            except subprocess.CalledProcessError as e:
                # Surface sbatch stdout/stderr to help diagnose submission failures
                out = e.stdout.decode("utf-8", errors="ignore") if e.stdout else ""
                err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
                if out.strip():
                    print(out.strip())
                if err.strip():
                    print(err.strip(), file=sys.stderr)
                raise



if __name__ == "__main__":
    main()
