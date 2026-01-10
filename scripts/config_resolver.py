"""
Config resolver that merges a dataset profile into a run config and injects
dataset tags into output paths.

Usage:
    from config_resolver import load_and_resolve_config
    cfg = load_and_resolve_config("configs/train_base.yaml")
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "configs" / "datasets"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _find_dataset_profile(dataset_key: str) -> Dict[str, Any]:
    """
    Locate a dataset profile by key (matching the `dataset` field) or by path.
    """
    if not dataset_key:
        return {}

    cand_path = Path(dataset_key)
    if cand_path.exists():
        return _load_yaml(cand_path)

    if DATASETS_DIR.is_dir():
        for yml in DATASETS_DIR.glob("*.yaml"):
            prof = _load_yaml(yml)
            if str(prof.get("dataset")) == str(dataset_key):
                return prof

    raise FileNotFoundError(f"Dataset profile not found for key/path: {dataset_key}")


def _tag_path(path_str: Any, dataset: Optional[str]) -> Any:
    if not dataset or not isinstance(path_str, str):
        return path_str
    if "{dataset}" in path_str:
        return path_str.replace("{dataset}", dataset)
    p = Path(path_str)
    # If the path is already dataset-tagged (common when re-loading used_config.yaml),
    # do not tag again.
    if p.name.startswith(f"{dataset}_"):
        return path_str
    return str(p.parent / f"{dataset}_{p.name}")


def load_and_resolve_config(cfg_path: str | Path,
                            dataset_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a run config YAML, merge in the dataset profile (if specified),
    and inject dataset tags into output paths.
    """
    cfg_path = Path(cfg_path)
    cfg = _load_yaml(cfg_path)

    dataset_key = dataset_override or cfg.get("dataset")
    profile = _find_dataset_profile(dataset_key) if dataset_key else {}

    base_data = profile.get("data", {}) if profile else {}
    cfg_data = cfg.get("data", {}) or {}

    # Merge data; when a dataset_key is provided, dataset-profile fields win for core dataset keys.
    merged_data = _deep_merge(base_data, cfg_data)
    if dataset_key:
        merged_data["dataset"] = dataset_key
        # Force profile values for the core dataset-specific keys to avoid legacy paths leaking in.
        dataset_fields = {
            "csv_path",
            "target_col",
            "target_transform",
            "numeric_cols",
            "onehot_cols",
            "hash_cols",
            "hash_dims",
            "drop_cols",
            "drop_unnamed_index_cols",
        }
        for k in dataset_fields:
            if k in base_data:
                merged_data[k] = base_data[k]

    cfg_out = _deep_merge(cfg, {"data": merged_data})
    if dataset_key:
        cfg_out["dataset"] = dataset_key

    # Inject dataset into IO paths to avoid mixing artifacts.
    io_cfg = cfg_out.setdefault("io", {})
    if dataset_key:
        io_cfg.setdefault("dataset", dataset_key)
        for key in ("outdir", "training_outdir", "evals_root"):
            if key in io_cfg:
                io_cfg[key] = _tag_path(io_cfg[key], dataset_key)

    return cfg_out
