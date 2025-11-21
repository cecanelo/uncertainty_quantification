# uncertainty_quantification/scripts/data.py
from __future__ import annotations
import json, csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# ---------- simple utilities ----------
def _to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x))
    std = std if std > 1e-12 else 1.0
    return (x - mean) / std, {"mean": mean, "std": std}

def _standardize_apply(x: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    mean = stats["mean"]; std = stats["std"] if stats["std"] > 1e-12 else 1.0
    return (x - mean) / std

def _hash_bucket(s: str, n_features: int) -> int:
    # Stable hash to int in [0, n_features)
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % n_features

@dataclass
class DataConfig:
    csv_path: str
    target_col: str = "price"
    target_transform: str = "log1p"   # "log1p" or "none"
    split_seed: int = 42
    split_fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    # Feature schema
    numeric_cols: Optional[List[str]] = None
    onehot_cols: Optional[List[str]] = None
    hash_cols: Optional[List[str]] = None
    hash_dims: Optional[Dict[str, int]] = None

    # Loader
    batch_size: int = 2048
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True

def _default_schema(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, int], List[str]]:
    # sensible defaults for this dataset
    num = ["year", "odometer", "lat", "long", "age_years", "log_odometer", "mileage_per_year"]

    oh  = ["manufacturer","condition","cylinders","fuel","title_status","transmission",
           "drive","type","paint_color","state"]
    hash_cols = ["region","model"]
    hash_dims = {"region": 512, "model": 4096} 
    # ensure presence
    num = [c for c in num if c in df.columns]
    oh  = [c for c in oh if c in df.columns]
    hash_cols = [c for c in hash_cols if c in df.columns]
    return num, oh, hash_dims, hash_cols

def _prepare_frame(cfg: DataConfig) -> pd.DataFrame:
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # basic coercions
    if "lat" in df.columns:
        df["lat"] = df["lat"].map(_to_float)
    if "long" in df.columns:
        df["long"] = df["long"].map(_to_float)
    # drop rows with missing target
    df = df[df[cfg.target_col].notna()].copy()

    # --- derived features (mileage & age) ---
    try:
        if "year" in df.columns and "odometer" in df.columns:
            cur_year = pd.Timestamp.now().year
            year_num = pd.to_numeric(df["year"], errors="coerce")
            odo_num  = pd.to_numeric(df["odometer"], errors="coerce").fillna(0)
            age = (cur_year - year_num).clip(lower=0)
            df["age_years"] = age
            df["log_odometer"] = np.log1p(odo_num)
            # avoid div-by-zero; mpy=0 when age==0 and mileage==0
            df["mileage_per_year"] = odo_num / np.where(age > 0, age, 1)
    except Exception:
        # be robust to odd schemas; just skip derived features if anything goes wrong
        pass

    return df


def _target_transform(y: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, str]]:
    if mode.lower() == "log1p":
        return np.log1p(y), {"mode": "log1p"}
    return y, {"mode": "none"}

def _target_inverse(y_hat: np.ndarray, meta: Dict[str, str]) -> np.ndarray:
    if meta.get("mode") == "log1p":
        return np.expm1(y_hat)
    return y_hat

def _fit_encoders(train_df: pd.DataFrame,
                  numeric_cols: List[str],
                  onehot_cols: List[str],
                  hash_cols: List[str],
                  hash_dims: Dict[str, int]) -> Dict:
    enc = {"num": {}, "oh_levels": {}, "hash_dims": hash_dims}
    # numeric stats
    for c in numeric_cols:
        x = train_df[c].astype(float).to_numpy()
        _, stats = _standardize_fit(x)
        enc["num"][c] = stats

    # onehot levels from training
    for c in onehot_cols:
        enc["oh_levels"][c] = sorted(pd.Series(train_df[c].astype(str)).unique().tolist())

    return enc

def _apply_encoders(df: pd.DataFrame,
                    numeric_cols: List[str],
                    onehot_cols: List[str],
                    hash_cols: List[str],
                    enc: Dict) -> np.ndarray:
    feats = []

    # numeric
    for c in numeric_cols:
        x = df[c].astype(float).fillna(np.nan).to_numpy()
        feats.append(_standardize_apply(x, enc["num"][c]).reshape(-1, 1))

    # onehot
    for c in onehot_cols:
        levels = enc["oh_levels"][c]
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), len(levels)), dtype=np.float32)
        idx = {lv: i for i, lv in enumerate(levels)}
        for i, val in enumerate(s):
            j = idx.get(val)
            if j is not None:
                mat[i, j] = 1.0
        feats.append(mat)

    # hashed high-card columns (signed hashing to reduce collision bias)
    for c in hash_cols:
        n = int(enc["hash_dims"][c])
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), n), dtype=np.float32)
        for i, val in enumerate(s):
            key = f"{c}={val}"
            j = _hash_bucket(key, n)
            # NEW: use a second hash to choose ±1 sign; helps collisions cancel rather than only add
            sign = -1.0 if (_hash_bucket(key + "#sign", 2) % 2) else 1.0
            mat[i, j] = sign
        feats.append(mat)


    return np.concatenate(feats, axis=1).astype(np.float32)

def _split_indices(n: int, fracs: Tuple[float, float, float], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = fracs
    if abs(a + b + c - 1.0) > 1e-6:
        raise ValueError("split_fracs must sum to 1.0")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(a * n))
    n_val   = int(round(b * n))
    i_train = idx[:n_train]
    i_val   = idx[n_train:n_train+n_val]
    i_test  = idx[n_train+n_val:]
    return i_train, i_val, i_test

def get_dataloaders(train_cfg: Dict, seed: int = 42):
    dc = DataConfig(**train_cfg["data"])
    df = _prepare_frame(dc)

    numeric_cols, onehot_cols, hash_dims, hash_cols = _default_schema(df) if dc.numeric_cols is None else \
        (dc.numeric_cols, dc.onehot_cols or [], dc.hash_dims or {}, dc.hash_cols or [])

    y = df[dc.target_col].astype(float).to_numpy()
    y_tr, y_meta = _target_transform(y, dc.target_transform)

    # --- split first, then fit encoders on TRAIN ONLY ---
    i_tr, i_va, i_te = _split_indices(len(df), dc.split_fracs, dc.split_seed)

    df_tr = df.iloc[i_tr].copy()
    enc = _fit_encoders(df_tr, numeric_cols, onehot_cols, hash_cols, hash_dims)

    X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)
    # --- end change ---

    tensors = lambda ii: (
        torch.tensor(X[ii], dtype=torch.float32),
        torch.tensor(y_tr[ii], dtype=torch.float32).view(-1, 1),
    )
    Xtr, ytr = tensors(i_tr)
    Xva, yva = tensors(i_va)
    Xte, yte = tensors(i_te)

    # stash preprocessing metadata for inverse-transform and later use
    meta = {
        "target": y_meta,
        "encoders": enc,
        "feature_dim": int(X.shape[1]),
        "splits": {"train": i_tr.tolist(), "val": i_va.tolist(), "test": i_te.tolist()},
        "numeric_cols": numeric_cols,
        "onehot_cols": onehot_cols,
        "hash_cols": hash_cols,
        "hash_dims": hash_dims,
    }

    # If an 'id' column exists, record it so evaluation can join on it later
    if "id" in df.columns:
        meta["id_col"] = "id"
        meta["id_values"] = df["id"].tolist()

    outdir = Path(train_cfg["io"].get("outdir", "."))
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "preproc_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    # Optional convenience file: mapping row_index -> split [-> id]
    split_indices_path = outdir / "split_indices.csv"
    with split_indices_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_index", "split"]
        if "id_col" in meta:
            header.append(meta["id_col"])
        writer.writerow(header)
        for split_name, idx_list in meta["splits"].items():
            for idx in idx_list:
                row = [idx, split_name]
                if "id_col" in meta:
                    row.append(meta["id_values"][idx])
                writer.writerow(row)

    def _loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(X, y),
            batch_size=int(dc.batch_size),
            shuffle=bool(shuffle),
            num_workers=int(dc.num_workers),
            pin_memory=bool(dc.pin_memory),
            drop_last=False,
        )
    return _loader(Xtr, ytr, dc.shuffle_train), _loader(Xva, yva, False), _loader(Xte, yte, False), meta



def inverse_target(y_hat: np.ndarray, meta: Dict[str, str]) -> np.ndarray:
    return _target_inverse(y_hat, meta)
