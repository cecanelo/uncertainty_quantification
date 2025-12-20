#!/usr/bin/env python
"""
Run eval_regression_ood.py for many (base_run, OOD-set) combinations.

Two modes:

1) Config-driven (default):
   - Read base_runs + ood_sets from YAML.

2) Auto-discover base runs:
   - Use --auto-discover-base to scan outputs/trainings for runs with '_m_'
     in the path and a used_config.yaml.
   - Optionally restrict by head_type via --head-types point laplace gauss.

Examples
--------

Config-driven, all combos:

  python scripts/run_ood_eval_batch.py \\
      --batch-config configs/ood_eval_base_models.yaml

Dry run:

  python scripts/run_ood_eval_batch.py --dry-run

Auto-discover all *_m_* base runs under outputs/trainings:

  python scripts/run_ood_eval_batch.py \\
      --auto-discover-base \\
      --batch-config configs/ood_eval_base_models.yaml

Auto-discover only Laplace & Gauss heads:

  python scripts/run_ood_eval_batch.py \\
      --auto-discover-base \\
      --head-types laplace gauss \\
      --batch-config configs/ood_eval_base_models.yaml

Filter which OOD sets to use:

  python scripts/run_ood_eval_batch.py \\
      --auto-discover-base \\
      --only-ood geo tail_multi \\
      --batch-config configs/ood_eval_base_models.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def discover_base_runs(
    train_root: Path,
    tag: str = "_m_",
    head_types: set[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan `train_root` (typically outputs/trainings) for run directories
    containing `tag` in their path and a used_config.yaml file.

    If `head_types` is given, only keep runs whose used_config.yaml has
    a head type in that set (best-effort; falls back to including if it
    cannot parse head_type).
    """
    result: Dict[str, Dict[str, Any]] = {}

    # We expect paths like:
    # outputs/trainings/training_lpl_m/lpl_m_YYYYMMDD-HHMMSS_xxxxx/used_config.yaml
    # outputs/trainings/training_gau_m/gau_m_YYYYMMDD-HHMMSS_xxxxx/used_config.yaml
    # outputs/trainings/training_point_m/point_m_.../used_config.yaml
    pattern = "training_*" + tag + f"*/**/*{tag}*/used_config.yaml"
    for cfg_path in train_root.glob(pattern):
        cfg_path = cfg_path.resolve()
        run_dir = cfg_path.parent
        if tag not in str(run_dir):
            continue

        head_type: str | None = None
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception:
            cfg = None

        if isinstance(cfg, dict):
            # Try a few common locations for head_type.
            head_type = (
                cfg.get("head_type")
                or cfg.get("head")
                or (cfg.get("model", {}) or {}).get("head_type")
                or (cfg.get("model", {}) or {}).get("head")
            )
            if isinstance(head_type, dict):
                head_type = head_type.get("type") or head_type.get("name")

        if head_types and head_type and head_type not in head_types:
            # Skip runs with a head_type we don't want.
            continue

        key = run_dir.name  # e.g. 'lpl_m_YYYYMMDD-HHMMSS_xxxxx'
        result[key] = {
            "config": cfg_path,
            "outdir": run_dir,
            "head_type": head_type,
        }

    print(f"[auto-discover] Found {len(result)} '*{tag}*' runs under {train_root}")
    for k, meta in sorted(result.items()):
        print(f"  {k}: head_type={meta.get('head_type')}, cfg={meta['config']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-config",
        type=str,
        default="configs/ood_eval_base_models.yaml",
        help="YAML file describing base runs and OOD datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing them.",
    )
    parser.add_argument(
        "--only-base",
        nargs="*",
        default=None,
        help="Optional list of base_run keys to include (others are skipped).",
    )
    parser.add_argument(
        "--only-ood",
        nargs="*",
        default=None,
        help="Optional list of ood_set keys to include (others are skipped).",
    )
    parser.add_argument(
        "--auto-discover-base",
        action="store_true",
        help="Ignore base_runs from config and scan outputs/trainings for *_m_* runs.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="_m_",
        help=(
            "Substring tag that identifies the run type in directory names, "
            "e.g. '_xs_', '_m_', '_l_'. Used only with --auto-discover-base."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=str,
        default="outputs/trainings",
        help="Root directory containing training_* subfolders.",
    )
    parser.add_argument(
        "--head-types",
        nargs="*",
        choices=["point", "laplace", "gauss"],
        default=None,
        help="When using --auto-discover-base, restrict to these head types (if detectable).",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch_config)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch config not found: {batch_path}")

    cfg = yaml.safe_load(batch_path.read_text()) or {}
    ood_sets = cfg.get("ood_sets", {}) or {}
    options = cfg.get("options", {}) or {}
    split = options.get("split", "test")
    run_tag_cfg = cfg.get("run_tag")

    if not ood_sets:
        raise ValueError("No ood_sets defined in batch config.")

    # --- Base runs: config-driven vs auto-discover ---
    if args.auto_discover_base:
        # Auto-discover in the main trainings root and, if present,
        # also under an "ensembles" subfolder so ensemble models are
        # evaluated as well.
        head_types = set(args.head_types) if args.head_types else None
        base_runs: Dict[str, Dict[str, Any]] = {}

        main_root = Path(args.train_root)
        roots = [main_root]
        ensembles_root = main_root / "ensembles"
        if ensembles_root.is_dir():
            roots.append(ensembles_root)

        for root in roots:
            found = discover_base_runs(
                train_root=root,
                tag=args.run_tag or run_tag_cfg or "_m_",
                head_types=head_types,
            )
            # Later roots override earlier ones on key collisions, which
            # is fine since run names should normally be unique.
            base_runs.update(found)
    else:
        base_runs = cfg.get("base_runs", {}) or {}
        if not base_runs:
            raise ValueError(
                "No base_runs defined in batch config and --auto-discover-base not set."
            )

    # --- Loop over (base_run, ood_set) combinations ---
    for base_key, base_cfg in base_runs.items():
        # Config-driven mode: value is dict with enabled/config/outdir.
        # Auto-discover mode: value is dict with config/outdir/head_type.
        if isinstance(base_cfg, dict) and "enabled" in base_cfg:
            if not base_cfg.get("enabled", True):
                continue
        if args.only_base and base_key not in args.only_base:
            continue

        config_path = Path(base_cfg["config"])
        outdir = Path(base_cfg["outdir"])

        if not config_path.exists():
            print(f"[warn] config for {base_key} does not exist: {config_path}")
            continue
        if not outdir.exists():
            print(f"[warn] outdir for {base_key} does not exist: {outdir}")
            continue

        for ood_key, ood_cfg in ood_sets.items():
            if not ood_cfg.get("enabled", True):
                continue
            if args.only_ood and ood_key not in args.only_ood:
                continue

            csv_path = Path(ood_cfg["csv"])
            if not csv_path.exists():
                print(f"[warn] OOD csv for {ood_key} does not exist: {csv_path}")
                continue

            cmd = [
                "python",
                "scripts/eval_regression_ood.py",
                "--config",
                str(config_path),
                "--outdir",
                str(outdir),
                "--ood-csv",
                str(csv_path),
                "--dataset-label",
                ood_key,
            ]
            print(" ".join(cmd))
            if args.dry_run:
                continue

            subprocess.run(cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
