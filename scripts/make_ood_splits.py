"""
Utility to construct ID and OOD datasets for the Craigslist experiments.

This script reads the existing cleaned CSV, reconstructs the numeric-tail,
rare-manufacturer, and geographic (FL+TX) OOD masks using the same logic as
in `notebooks/ood_tails_numeric.ipynb`, and writes:

- An ID-only CSV (no numeric tails, no rare manufacturers, no FL/TX rows)
- Separate OOD CSVs for:
    * High mileage-per-year tail
    * High odometer tail
    * Old-year tail (pre-1999)
    * Multi-tail (rows in >= 2 numeric tails)
    * Rare manufacturers
    * Geographic FL+TX

The original cleaned CSV is left untouched; the ID-only CSV is written
alongside it so you can explicitly rename or swap it into configs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data import DataConfig, _prepare_frame


def _compute_numeric_tail_masks(df: pd.DataFrame, year_q_low: float = 0.20, year_q_high: float = 0.80) -> Dict[str, pd.Series]:
    """Compute numeric tail OOD masks (mpy high, odometer high, old year, any, multi)."""
    numeric_cols: List[str] = ["year", "odometer", "mileage_per_year"]
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected numeric columns for tail masks: {missing}")

    q = {col: df[col].quantile([year_q_low, year_q_high]) for col in numeric_cols}

    mask_mpy_high = df["mileage_per_year"] > q["mileage_per_year"].loc[year_q_high]
    mask_odo_high = df["odometer"] > q["odometer"].loc[year_q_high]
    mask_year_old = df["year"] < q["year"].loc[year_q_low]

    mask_any_tail = mask_mpy_high | mask_odo_high | mask_year_old
    mask_multi_tail = (
        mask_mpy_high.astype(int)
        + mask_odo_high.astype(int)
        + mask_year_old.astype(int)
    ) >= 2

    return {
        "mpy_high": mask_mpy_high,
        "odo_high": mask_odo_high,
        "year_old": mask_year_old,
        "any_tail": mask_any_tail,
        "multi_tail": mask_multi_tail,
    }


def _compute_rare_manufacturer_mask(
    df: pd.DataFrame, id_mask_numeric: pd.Series, coverage_target: float = 0.90
) -> pd.Series:
    """
    Compute rare-manufacturer OOD mask within the numeric ID core.

    - coverage_target controls what fraction of rows are covered by "common" manufacturers.
      The remainder (excluding the 'not_available' placeholder) are treated as rare.
    - id_mask_numeric is the complement of numeric tails; we restrict rare OOD to this
      region so it is not also in numeric tails.
    """
    if "manufacturer" not in df.columns:
        raise ValueError("Column 'manufacturer' not found in dataframe.")

    man_counts = df["manufacturer"].value_counts(dropna=False)
    man_table = (
        man_counts.to_frame(name="count")
        .reset_index()
        .rename(columns={"index": "manufacturer"})
    )
    man_table["fraction"] = man_table["count"] / len(df)
    man_table["cum_fraction"] = man_table["fraction"].cumsum()

    common_man = man_table[man_table["cum_fraction"] <= coverage_target]["manufacturer"]
    rare_man = man_table[man_table["cum_fraction"] > coverage_target]["manufacturer"]

    rare_non_placeholder = [
        m for m in rare_man if str(m).lower() != "not_available"
    ]

    mask_rare_all = df["manufacturer"].isin(rare_non_placeholder)
    mask_rare_numeric = mask_rare_all & id_mask_numeric
    return mask_rare_numeric


def _compute_geo_mask_fl_tx(df: pd.DataFrame, id_mask_numeric: pd.Series) -> pd.Series:
    """
    Geographic OOD: rows from FL or TX within the numeric ID core.
    """
    if "state" not in df.columns:
        raise ValueError("Column 'state' not found in dataframe.")

    # Normalize to lowercase strings to avoid missing matches (e.g., "FL", "Tx", etc.).
    normalized_state = df["state"].astype(str).str.lower().str.strip()
    holdout_states = {"fl", "tx"}
    mask_geo_all = normalized_state.isin(holdout_states)
    return mask_geo_all & id_mask_numeric


def build_and_save_splits(
    csv_path: Path, out_dir: Path | None = None, verbose: bool = True
) -> None:
    """
    Build ID/OOD splits from an existing cleaned Craigslist CSV and write them as CSVs.

    Parameters
    ----------
    csv_path:
        Path to the existing cleaned CSV (current training input).
    out_dir:
        Directory to write output CSVs. Defaults to `csv_path.parent`.
    verbose:
        If True, print a short summary of split sizes.
    """
    csv_path = csv_path.expanduser().resolve()
    if out_dir is None:
        out_dir = csv_path.parent
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the same preparation logic as training to keep derived features consistent
    cfg = DataConfig(csv_path=str(csv_path))
    df = _prepare_frame(cfg)

    # --- numeric tails ---
    # Use relaxed quantiles (80/20) to make tail-year less extreme and more overlapping with ID
    tail_masks = _compute_numeric_tail_masks(df, year_q_low=0.20, year_q_high=0.80)
    mask_any_tail = tail_masks["any_tail"]
    mask_multi_tail = tail_masks["multi_tail"]

    # Numeric ID core: no numeric tails
    id_mask_numeric = ~mask_any_tail

    # --- rare manufacturers within numeric ID core ---
    mask_rare_man = _compute_rare_manufacturer_mask(df, id_mask_numeric, coverage_target=0.90)

    # --- geographic OOD: FL + TX within numeric ID core ---
    mask_geo_fl_tx = _compute_geo_mask_fl_tx(df, id_mask_numeric)

    # --- combined ID/OOD masks ---
    # Primary OOD families:
    #   - numeric tails (any_tail)
    #   - rare manufacturers
    #   - geographic FL+TX
    mask_any_ood = mask_any_tail | mask_rare_man | mask_geo_fl_tx
    mask_id_final = ~mask_any_ood

    # Convenience dictionary for output
    ood_sets: Dict[str, pd.Series] = {
        "ood_tail_mpy_high": tail_masks["mpy_high"],
        "ood_tail_odo_high": tail_masks["odo_high"],
        "ood_tail_year_old": tail_masks["year_old"],
        "ood_tail_multi": mask_multi_tail,
        "ood_rare_manufacturers": mask_rare_man,
        "ood_geo_fl_tx": mask_geo_fl_tx,
    }

    # Columns to drop per OOD set to avoid trivial separability
    # (i.e., remove the feature that directly defines the split).
    ood_drop_columns: Dict[str, List[str]] = {
        # Numeric tails: drop the defining numeric columns
        "ood_tail_mpy_high": ["mileage_per_year", "odometer", "year"],
        "ood_tail_odo_high": ["mileage_per_year", "odometer", "year"],
        "ood_tail_year_old": ["mileage_per_year", "odometer", "year"],
        "ood_tail_multi": ["mileage_per_year", "odometer", "year"],
        # Rare manufacturers: drop manufacturer
        "ood_rare_manufacturers": ["manufacturer"],
        # Geo: drop state/region
        # Geo: drop state/region + lat/long to avoid location leakage
        "ood_geo_fl_tx": ["state", "region", "lat", "long", "latitude", "longitude"],
    }

    # --- write CSVs ---
    # ID-only CSV (you can later rename this to craigslist_cleaned.csv in configs)
    id_csv = out_dir / "craigslist_cleaned_id_only.csv"
    df.loc[mask_id_final].to_csv(id_csv, index=False)

    # Each OOD CSV (note: sets may overlap; that's intentional for multi-tail, etc.)
    for name, mask in ood_sets.items():
        out_csv = out_dir / f"craigslist_{name}.csv"
        # Drop columns that would make the split trivial, if present
        drop_cols = [c for c in ood_drop_columns.get(name, []) if c in df.columns]
        df_out = df.loc[mask].drop(columns=drop_cols, errors="ignore")
        df_out.to_csv(out_csv, index=False)

    if verbose:
        n_total = len(df)
        print(f"Source CSV: {csv_path}  (rows={n_total})")
        print(f"ID_final rows: {int(mask_id_final.sum())} ({mask_id_final.mean():.3%}) -> {id_csv.name}")
        for name, mask in ood_sets.items():
            n = int(mask.sum())
            frac = mask.mean()
            drop_cols = ood_drop_columns.get(name, [])
            msg = f"{name}: {n} rows ({frac:.3%})  drop_cols={drop_cols}"
            if n == 0:
                msg += " [warning: mask produced 0 rows]"
            print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ID/OOD splits from a cleaned Craigslist CSV.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("datasets") / "craigslist_cleaned.csv",
        help="Path to the cleaned CSV to split (default: datasets/craigslist_cleaned.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated CSVs (default: same directory as csv-path).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress summary printing.")
    args = parser.parse_args()

    build_and_save_splits(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )
