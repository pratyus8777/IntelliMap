from typing import Dict, Any, List
from .utils.io_utils import load_csvs
import pandas as pd
import numpy as np

def dtype_bucket(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series): return "int"
    if pd.api.types.is_float_dtype(series): return "float"
    if pd.api.types.is_bool_dtype(series): return "bool"
    if pd.api.types.is_datetime64_any_dtype(series): return "date"
    return "string"

def numeric_stats(s: pd.Series) -> Dict[str, float]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(s2.mean()), "std": float(s2.std(ddof=0)), "min": float(s2.min()), "max": float(s2.max())}

def string_stats(s: pd.Series) -> Dict[str, Any]:
    s2 = s.dropna().astype(str)
    if s2.empty:
        return {"avg_len": float("nan"), "uniq_ratio": float("nan"), "top_values": []}
    lens = s2.str.len()
    uniq_ratio = s2.nunique(dropna=True) / max(1, len(s2))
    top_values = s2.value_counts().head(20).index.tolist()
    return {"avg_len": float(lens.mean()), "uniq_ratio": float(uniq_ratio), "top_values": top_values}

def profile_all(root: str, files_map: Dict[str,str], sample_rows: int = 1000) -> Dict[str, Dict[str, Dict]]:
    """
    Return nested dict: profiles[logical_table][column] = {meta...}
    """
    tables = load_csvs(root, files_map, nrows=sample_rows)
    profiles: Dict[str, Dict[str, Dict]] = {}
    for tbl, df in tables.items():
        prof_tbl: Dict[str, Dict] = {}
        if df is None or df.empty:
            profiles[tbl] = prof_tbl
            continue
        for col in df.columns:
            s = df[col]
            meta = {
                "dtype_bucket": dtype_bucket(s),
                "null_ratio": float(s.isna().mean()),
                "distinct_count": int(s.nunique(dropna=True)),
            }
            if meta["dtype_bucket"] in ("int","float"):
                meta.update(numeric_stats(s))
            else:
                meta.update(string_stats(s))
            prof_tbl[col] = meta
        profiles[tbl] = prof_tbl
    return profiles
