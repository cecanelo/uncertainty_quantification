from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


EVAL_ROOT_DEFAULT = Path("outputs/evals")


DEFAULT_RUN_PATTERNS: Mapping[str, str] = {
    "point_xs": "point_xs_*",
    "lpl_xs": "lpl_xs_*",
    "gau_xs": "gau_xs_*",
    "esm_lpl_xs": "esm_laplace_lpl_xs_*",
    "esm_gau_xs": "esm_gauss_gau_xs_*",
    "nf_lpl_xs": "nf_lpl_xs_*",
    "nf_gau_xs": "nf_gau_xs_*",
    "dido_lpl_xs": "dido_lpl_xs_*",
    "dido_gau_xs": "dido_gau_xs_*",
}


@dataclass
class AnalysisSettings:
    eval_root: Path = EVAL_ROOT_DEFAULT
    dido_mode: str = "logistic"  # or "raw"
    n_mc_samples: int = 50
    n_ensemble_members: int = 10
    n_nf_samples: int = 200
    large_error_quantile: float = 0.9


def discover_runs(
    eval_root: Path | str = EVAL_ROOT_DEFAULT,
    run_patterns: Mapping[str, str] = DEFAULT_RUN_PATTERNS,
) -> Dict[str, Path]:
    """
    Discover xs runs under eval_root based on glob patterns.
    Returns a mapping from logical run key -> resolved directory.
    If multiple matches exist for a pattern, the lexicographically last
    match is chosen (typically the most recent timestamped run).
    """
    root = Path(eval_root).resolve()
    runs: Dict[str, Path] = {}
    for key, pattern in run_patterns.items():
        matches = sorted(root.glob(pattern))
        if not matches:
            continue
        runs[key] = matches[-1]
    return runs


def _with_common_cols(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
    split_col: str = "split",
    head_type: str,
    aleatoric_source: str,
    epistemic_source: str,
    dataset: str = "id",
    train_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Attach standard metadata columns expected by downstream analysis.
    """
    df = df.copy()
    if id_col != "id":
        df.rename(columns={id_col: "id"}, inplace=True)
    if split_col != "split" and split_col in df.columns:
        df.rename(columns={split_col: "split"}, inplace=True)
    if "split" not in df.columns:
        df["split"] = "test"
    df["head_type"] = head_type
    df["aleatoric_source"] = aleatoric_source
    df["epistemic_source"] = epistemic_source
    df["dataset"] = dataset
    df["train_frac"] = float(train_frac)
    return df


def load_point_run(run_dir: Path) -> pd.DataFrame:
    """
    Load point_xs run: train/val/test preds without uncertainty.
    Expected files: *preds.csv with columns [id, split?, y_true, y_pred].
    """
    run_dir = Path(run_dir)
    rows: List[pd.DataFrame] = []
    for fname in ("train_preds.csv", "val_preds.csv", "test_preds.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Standardize column names. Handle both older point evals (y_pred)
        # and newer eval_regression outputs (y_pred_det/y_pred_mc_mean).
        col_map: Dict[str, str] = {}
        if "y_pred" in df.columns:
            col_map["y_pred"] = "y_pred_mean"
        elif "y_pred_det" in df.columns:
            col_map["y_pred_det"] = "y_pred_mean"
            # drop redundant mc-mean column if present
            if "y_pred_mc_mean" in df.columns:
                df = df.drop(columns=["y_pred_mc_mean"])
        # sigma_ale_raw may or may not be present for point; keep if it is.
        if "sigma_ale_raw" in df.columns:
            col_map["sigma_ale_raw"] = "sigma_ale_orig_raw"
        df.rename(columns=col_map, inplace=True)
        df = _with_common_cols(
            df,
            head_type="point",
            aleatoric_source="none",
            epistemic_source="none",
        )
        df["y_true"] = df["y_true"].astype(float)
        df["y_pred_mean"] = df["y_pred_mean"].astype(float)
        df["sigma_ale_orig_raw"] = np.nan
        df["sigma_epi_orig_raw"] = np.nan
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_lpl_or_gau_run(run_dir: Path, head_type: str) -> pd.DataFrame:
    """
    Load Laplace/Gauss xs base run with optional MC outputs.
    Combines deterministic preds_* and mc_preds_* into a single table.
    """
    run_dir = Path(run_dir)
    rows: List[pd.DataFrame] = []

    # Deterministic preds (no epistemic)
    for fname in ("preds_train.csv", "preds_val.csv", "test_preds.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # unify column names; for deterministic preds we keep only y_pred_det
        # as the mean prediction and drop any redundant mc mean column.
        col_map = {}
        if "y_pred_det" in df.columns:
            col_map["y_pred_det"] = "y_pred_mean"
        # deterministic eval sets mc_mean == det_pred; drop it to avoid
        # duplicate columns after renaming.
        if "y_pred_mc_mean" in df.columns:
            df = df.drop(columns=["y_pred_mc_mean"])
        if "sigma_ale_raw" in df.columns:
            col_map["sigma_ale_raw"] = "sigma_ale_orig_raw"
        df.rename(columns=col_map, inplace=True)

        df = _with_common_cols(
            df,
            head_type=head_type,
            aleatoric_source="analytic",
            epistemic_source="none",
        )
        df["y_true"] = df["y_true"].astype(float)
        df["y_pred_mean"] = df["y_pred_mean"].astype(float)
        if "sigma_ale_orig_raw" in df.columns:
            df["sigma_ale_orig_raw"] = df["sigma_ale_orig_raw"].astype(float)
        else:
            df["sigma_ale_orig_raw"] = np.nan
        df["sigma_epi_orig_raw"] = np.nan
        rows.append(df)

    # MC dropout preds
    for fname in ("mc_preds_train.csv", "mc_preds_val.csv", "mc_preds_test.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col_map = {}
        if "y_pred_mc_mean" in df.columns:
            col_map["y_pred_mc_mean"] = "y_pred_mean"
        if "y_pred_det" in df.columns:
            # keep deterministic as separate column if needed later
            col_map["y_pred_det"] = "y_pred_det"
        if "sigma_ale_orig" in df.columns:
            col_map["sigma_ale_orig"] = "sigma_ale_orig_raw"
        if "sigma_epi_orig" in df.columns:
            col_map["sigma_epi_orig"] = "sigma_epi_orig_raw"
        df.rename(columns=col_map, inplace=True)

        df = _with_common_cols(
            df,
            head_type=head_type,
            aleatoric_source="analytic",
            epistemic_source="mc",
        )
        df["y_true"] = df["y_true"].astype(float)
        df["y_pred_mean"] = df["y_pred_mean"].astype(float)
        if "sigma_ale_orig_raw" in df.columns:
            df["sigma_ale_orig_raw"] = df["sigma_ale_orig_raw"].astype(float)
        else:
            df["sigma_ale_orig_raw"] = np.nan
        if "sigma_epi_orig_raw" in df.columns:
            df["sigma_epi_orig_raw"] = df["sigma_epi_orig_raw"].astype(float)
        else:
            df["sigma_epi_orig_raw"] = np.nan
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_ensemble_run(run_dir: Path, head_type: str) -> pd.DataFrame:
    """
    Load ensemble preds for Laplace/Gauss xs run.
    Uses ensemble_preds_{train,val,test}.csv.
    """
    run_dir = Path(run_dir)
    rows: List[pd.DataFrame] = []
    for fname in ("ensemble_preds_train.csv", "ensemble_preds_val.csv", "ensemble_preds_test.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col_map = {
            "y_pred_ens_mean": "y_pred_mean",
            "sigma_ale_ens": "sigma_ale_orig_raw",
            "sigma_epi_ens": "sigma_epi_orig_raw",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

        df = _with_common_cols(
            df,
            head_type=head_type,
            aleatoric_source="analytic",
            epistemic_source="ensemble",
        )
        df["y_true"] = df["y_true"].astype(float)
        df["y_pred_mean"] = df["y_pred_mean"].astype(float)
        if "sigma_ale_orig_raw" in df.columns:
            df["sigma_ale_orig_raw"] = df["sigma_ale_orig_raw"].astype(float)
        else:
            df["sigma_ale_orig_raw"] = np.nan
        if "sigma_epi_orig_raw" in df.columns:
            df["sigma_epi_orig_raw"] = df["sigma_epi_orig_raw"].astype(float)
        else:
            df["sigma_epi_orig_raw"] = np.nan
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_nf_run(run_dir: Path, head_type: str) -> pd.DataFrame:
    """
    Load NF residual modeling eval for a given head_type.
    Currently expects flow_eval_{split}.csv files.
    """
    run_dir = Path(run_dir)
    rows: List[pd.DataFrame] = []
    for fname in ("flow_eval_train.csv", "flow_eval_val.csv", "flow_eval_test.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Older runs may use row_idx instead of id
        id_col = "id" if "id" in df.columns else "row_idx"
        df = df.rename(columns={id_col: "id"})
        df = _with_common_cols(
            df,
            head_type=head_type,
            aleatoric_source="nf",
            epistemic_source="none",
        )
        df["y_true"] = df["y_true_orig"].astype(float)
        df["y_pred_mean"] = df["y_pred_base_orig"].astype(float)
        df["sigma_ale_orig_raw"] = df["sigma_base_orig"].astype(float)
        df["sigma_epi_orig_raw"] = np.nan
        df["z_raw"] = df["z_raw"].astype(float)
        df["log_prob_z_raw"] = df["log_prob_z_raw"].astype(float)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_dido_run(run_dir: Path) -> pd.DataFrame:
    """
    Load DIDO eval (dido_test.csv etc.).
    Returned frame only contains DIDO columns plus id/split/head_type;
    caller is responsible for joining with base predictions.
    """
    run_dir = Path(run_dir)
    rows: List[pd.DataFrame] = []
    for fname in ("dido_train.csv", "dido_val.csv", "dido_test.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = _with_common_cols(
            df,
            head_type=str(df["head_type"].iloc[0]).lower(),
            aleatoric_source="analytic",
            epistemic_source="dido",
        )
        df["dido_strength_raw"] = df["dido_strength_raw"].astype(float)
        df["dido_vacuity_raw"] = df["dido_vacuity_raw"].astype(float)
        df["dido_entropy_raw"] = df["dido_entropy_raw"].astype(float)
        rows.append(df[["id", "split", "head_type", "aleatoric_source", "epistemic_source",
                        "dataset", "train_frac",
                        "dido_strength_raw", "dido_vacuity_raw", "dido_entropy_raw"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_df_all(runs: Mapping[str, Path], train_frac: float = 1.0) -> pd.DataFrame:
    """
    Build a unified long-format DataFrame with all available xs runs.
    """
    frames: List[pd.DataFrame] = []

    if "point_xs" in runs:
        frames.append(load_point_run(runs["point_xs"]))

    if "lpl_xs" in runs:
        frames.append(load_lpl_or_gau_run(runs["lpl_xs"], head_type="laplace"))

    if "gau_xs" in runs:
        frames.append(load_lpl_or_gau_run(runs["gau_xs"], head_type="gauss"))

    if "esm_lpl_xs" in runs:
        frames.append(load_ensemble_run(runs["esm_lpl_xs"], head_type="laplace"))

    if "esm_gau_xs" in runs:
        frames.append(load_ensemble_run(runs["esm_gau_xs"], head_type="gauss"))

    if "nf_lpl_xs" in runs:
        frames.append(load_nf_run(runs["nf_lpl_xs"], head_type="laplace"))

    if "nf_gau_xs" in runs:
        frames.append(load_nf_run(runs["nf_gau_xs"], head_type="gauss"))

    # DIDO frames are joined later to base predictions, so we only return raw DIDO rows here.
    if "dido_lpl_xs" in runs:
        frames.append(load_dido_run(runs["dido_lpl_xs"]))
    if "dido_gau_xs" in runs:
        frames.append(load_dido_run(runs["dido_gau_xs"]))

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all["train_frac"] = float(train_frac)
    return df_all


def gaussian_nll(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    """
    Mean Gaussian negative log-likelihood for scalar targets.
    """
    sigma = np.clip(sigma, 1e-8, None)
    z = (y_true - y_pred) / sigma
    nll = 0.5 * (z * z + 2.0 * np.log(sigma) + np.log(2.0 * np.pi))
    return float(np.mean(nll))


def fit_alpha_gaussian(
    df_val: pd.DataFrame,
    group_cols: Iterable[str] = ("head_type", "aleatoric_source"),
) -> Dict[Tuple[str, str], float]:
    """
    Fit a scalar alpha per (head_type, aleatoric_source) for Gaussian NLL.
    Assumes epistemic_source == 'none' and sigma_ale_orig_raw > 0.
    Closed-form solution under normality:
      alpha = mean( (err^2) / sigma_raw^2 )
    """
    alpha: Dict[Tuple[str, str], float] = {}
    df = df_val.copy()
    df = df[np.isfinite(df.get("sigma_ale_orig_raw", np.nan))]
    df = df[df["epistemic_source"] == "none"]
    if df.empty:
        return alpha
    for key, g in df.groupby(list(group_cols)):
        y = g["y_true"].to_numpy(dtype=float)
        mu = g["y_pred_mean"].to_numpy(dtype=float)
        s = g["sigma_ale_orig_raw"].to_numpy(dtype=float)
        mask = (s > 0) & np.isfinite(s)
        if not np.any(mask):
            continue
        err2 = (y[mask] - mu[mask]) ** 2
        ratio = err2 / (s[mask] ** 2)
        alpha[key] = float(np.mean(ratio))
    return alpha


def fit_beta_gaussian(
    df_val: pd.DataFrame,
    alpha: Mapping[Tuple[str, str], float],
    group_cols: Iterable[str] = ("head_type", "aleatoric_source", "epistemic_source"),
    beta_grid: Optional[np.ndarray] = None,
) -> Dict[Tuple[str, str, str], float]:
    """
    Fit a scalar beta per (head_type, aleatoric_source, epistemic_source)
    by minimizing Gaussian NLL over the validation slice using a simple grid search.
    """
    if beta_grid is None:
        beta_grid = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 41))

    df = df_val.copy()
    df = df[np.isfinite(df.get("sigma_ale_orig_raw", np.nan))]
    df = df[np.isfinite(df.get("sigma_epi_orig_raw", np.nan))]
    beta: Dict[Tuple[str, str, str], float] = {}
    if df.empty:
        return beta

    for key, g in df.groupby(list(group_cols)):
        head_type, ale_source, epi_source = key
        a_key = (head_type, ale_source)
        a = float(alpha.get(a_key, 1.0))
        y = g["y_true"].to_numpy(dtype=float)
        mu = g["y_pred_mean"].to_numpy(dtype=float)
        s_ale = g["sigma_ale_orig_raw"].to_numpy(dtype=float)
        s_epi = g["sigma_epi_orig_raw"].to_numpy(dtype=float)
        mask = (s_ale > 0) & (s_epi >= 0) & np.isfinite(s_ale) & np.isfinite(s_epi)
        if not np.any(mask):
            continue
        y = y[mask]
        mu = mu[mask]
        s_ale = s_ale[mask]
        s_epi = s_epi[mask]
        s_ale2_cal = a * (s_ale ** 2)

        best_beta = 1.0
        best_nll = float("inf")
        for b in beta_grid:
            sigma2 = s_ale2_cal + b * (s_epi ** 2)
            sigma = np.sqrt(np.maximum(sigma2, 1e-8))
            nll = gaussian_nll(y, mu, sigma)
            if nll < best_nll:
                best_nll = nll
                best_beta = float(b)
        beta[key] = best_beta
    return beta


def add_calibrated_sigmas(
    df: pd.DataFrame,
    alpha: Mapping[Tuple[str, str], float],
    beta: Mapping[Tuple[str, str, str], float],
) -> pd.DataFrame:
    """
    Add calibrated aleatoric, epistemic, and total std columns to df.
    """
    df = df.copy()
    sigma_ale_cal = []
    sigma_epi_cal = []
    sigma_total_cal = []
    for _, row in df.iterrows():
        h = row.get("head_type")
        a_src = row.get("aleatoric_source")
        e_src = row.get("epistemic_source")
        a = float(alpha.get((h, a_src), 1.0))
        b = float(beta.get((h, a_src, e_src), 1.0))
        s_ale_raw = row.get("sigma_ale_orig_raw")
        s_epi_raw = row.get("sigma_epi_orig_raw")
        if not np.isfinite(s_ale_raw):
            s_ale_raw = np.nan
        if not np.isfinite(s_epi_raw):
            s_epi_raw = np.nan
        if np.isnan(s_ale_raw):
            s_ale_c = np.nan
        else:
            s_ale_c = float(np.sqrt(max(a * (s_ale_raw ** 2), 0.0)))
        if np.isnan(s_epi_raw):
            s_epi_c = np.nan
        else:
            s_epi_c = float(np.sqrt(max(b * (s_epi_raw ** 2), 0.0)))
        if np.isnan(s_ale_c) and np.isnan(s_epi_c):
            s_tot = np.nan
        else:
            s_tot2 = 0.0
            if not np.isnan(s_ale_c):
                s_tot2 += s_ale_c ** 2
            if not np.isnan(s_epi_c):
                s_tot2 += s_epi_c ** 2
            s_tot = float(np.sqrt(max(s_tot2, 0.0)))
        sigma_ale_cal.append(s_ale_c)
        sigma_epi_cal.append(s_epi_c)
        sigma_total_cal.append(s_tot)

    df["sigma_ale_orig_cal"] = sigma_ale_cal
    df["sigma_epi_orig_cal"] = sigma_epi_cal
    df["sigma_total_orig_cal"] = sigma_total_cal
    return df


def compute_large_error_threshold(
    df_point: pd.DataFrame,
    quantile: float = 0.9,
) -> float:
    """
    Compute the absolute-error threshold tau from a point baseline.
    """
    if df_point.empty:
        raise ValueError("Point baseline DataFrame is empty; cannot compute threshold.")
    ae = np.abs(df_point["y_true"].to_numpy(dtype=float) - df_point["y_pred_mean"].to_numpy(dtype=float))
    return float(np.quantile(ae, quantile))


def fit_dido_logistic(
    df: pd.DataFrame,
    tau: float,
    max_epochs: int = 500,
    lr: float = 0.05,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Fit a simple logistic regression mapping DIDO scores to P(large_error).
    Returns parameters per (head_type, aleatoric_source).
    """
    params: Dict[Tuple[str, str], Dict[str, float]] = {}
    # Require DIDO rows with base predictions and y_true present
    df_dido = df.copy()
    df_dido = df_dido[df_dido["epistemic_source"] == "dido"]
    cols_needed = {"y_true", "y_pred_mean", "dido_vacuity_raw", "dido_entropy_raw"}
    if not cols_needed.issubset(df_dido.columns):
        return params
    # large error label based on tau from point baseline
    err = np.abs(df_dido["y_true"].to_numpy(dtype=float) - df_dido["y_pred_mean"].to_numpy(dtype=float))
    y = (err > tau).astype(np.float32)
    x_features = df_dido[["dido_vacuity_raw", "dido_entropy_raw"]].to_numpy(dtype=np.float32)
    # optional: include sigma_total_orig_cal if present
    if "sigma_total_orig_cal" in df_dido.columns:
        sigma_tot = df_dido["sigma_total_orig_cal"].to_numpy(dtype=np.float32).reshape(-1, 1)
        x_features = np.concatenate([x_features, sigma_tot], axis=1)
    # group per (head_type, aleatoric_source)
    grouped = df_dido.groupby(["head_type", "aleatoric_source"])
    for (head_type, ale_src), idx in grouped.groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        x = x_features[idx_arr]
        y_g = y[idx_arr]
        if x.shape[0] < 20:
            continue
        device = torch.device("cpu")
        X_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_g.reshape(-1, 1), dtype=torch.float32, device=device)
        n_features = X_t.shape[1]
        w = torch.zeros(n_features, 1, dtype=torch.float32, requires_grad=True, device=device)
        b = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(max_epochs):
            opt.zero_grad()
            logits = X_t @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            loss.backward()
            opt.step()
        w_np = w.detach().cpu().numpy().reshape(-1)
        b_np = float(b.detach().cpu().item())
        key = (str(head_type), str(ale_src))
        params[key] = {
            "bias": b_np,
            "weights": w_np.tolist(),
        }
    return params


def save_analysis_record(
    path: Path,
    settings: AnalysisSettings,
    runs: Mapping[str, Path],
    alpha: Mapping[Tuple[str, str], float],
    beta: Mapping[Tuple[str, str, str], float],
    dido_params: Mapping[Tuple[str, str], Dict[str, float]],
    extra_meta: Optional[Dict[str, object]] = None,
) -> None:
    """
    Save a JSON record of an analysis/calibration run.
    """
    record: Dict[str, object] = {
        "analysis_name": path.stem,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "settings": asdict(settings),
        "runs": {k: str(v) for k, v in runs.items()},
        "alpha": {"|".join(k): float(v) for k, v in alpha.items()},
        "beta": {"|".join(k): float(v) for k, v in beta.items()},
        "dido_params": {"|".join(k): v for k, v in dido_params.items()},
    }
    if extra_meta:
        record["meta"] = extra_meta
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(record, f, indent=2)


def main() -> None:
    """
    Lightweight CLI entry-point: discover xs runs and report what was found.
    Detailed analysis is expected to be driven from notebooks or
    higher-level scripts that import this module.
    """
    runs = discover_runs(EVAL_ROOT_DEFAULT, DEFAULT_RUN_PATTERNS)
    if not runs:
        print("No xs runs discovered under", EVAL_ROOT_DEFAULT)
        return
    print("Discovered xs runs:")
    for key, path in sorted(runs.items()):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
