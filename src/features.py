from typing import Dict
import pandas as pd
import numpy as np
import random
from .utils.io_utils import parse_ddl, load_csvs
from .utils.text_sim import string_similarity, normalize_name
from .profiler import profile_all

def _dtype_family(dtype: str) -> str:
    if not dtype:
        return "other"
    d = dtype.lower()
    if any(k in d for k in ["int", "decimal", "numeric", "float", "double"]):
        return "number"
    if any(k in d for k in ["char", "varchar", "text", "string"]):
        return "text"
    if "date" in d or "time" in d:
        return "date"
    return "other"

def _len_from_dtype(dtype: str) -> int:
    if not dtype:
        return 0
    import re
    m = re.search(r"\((\d+)", dtype)
    return int(m.group(1)) if m else 0

def _extra_data_features(s_vals: pd.Series, t_vals: pd.Series) -> Dict[str, float]:
    feats = {"value_overlap": 0.0, "mean_diff": 0.0, "std_diff": 0.0, "avg_strlen_diff": 0.0}
    try:
        s_vals = s_vals.dropna()
        t_vals = t_vals.dropna()
        if s_vals.empty or t_vals.empty:
            return feats

        s_sample = s_vals.sample(min(200, len(s_vals)), random_state=42)
        t_sample = t_vals.sample(min(200, len(t_vals)), random_state=42)

        # Jaccard similarity of unique values
        s_set, t_set = set(map(str, s_sample)), set(map(str, t_sample))
        feats["value_overlap"] = len(s_set & t_set) / max(1, len(s_set | t_set))

        # Numeric comparison
        if pd.api.types.is_numeric_dtype(s_sample) and pd.api.types.is_numeric_dtype(t_sample):
            feats["mean_diff"] = abs(s_sample.mean() - t_sample.mean())
            feats["std_diff"] = abs(s_sample.std() - t_sample.std())

        # String length comparison
        if s_sample.dtype == "object" or t_sample.dtype == "object":
            s_len = s_sample.astype(str).str.len().mean()
            t_len = t_sample.astype(str).str.len().mean()
            feats["avg_strlen_diff"] = abs(s_len - t_len)
    except Exception:
        pass
    return feats

def _pattern_score(s_vals: pd.Series, t_vals: pd.Series) -> float:
    """Similarity score based on actual data patterns."""
    if s_vals is None or t_vals is None or s_vals.empty or t_vals.empty:
        return 0.0
    try:
        # Numeric similarity
        if pd.api.types.is_numeric_dtype(s_vals) and pd.api.types.is_numeric_dtype(t_vals):
            mean_diff = abs(s_vals.mean() - t_vals.mean())
            std_diff = abs(s_vals.std() - t_vals.std())
            return max(0.0, 1.0 - (mean_diff + std_diff) / (max(s_vals.max(), t_vals.max(), 1) + 1e-5))
        # String similarity by length distribution
        elif s_vals.dtype == "object" or t_vals.dtype == "object":
            s_len = s_vals.astype(str).str.len()
            t_len = t_vals.astype(str).str.len()
            return max(0.0, 1.0 - abs(s_len.mean() - t_len.mean()) / max(s_len.mean(), t_len.mean(), 1))
    except Exception:
        return 0.0
    return 0.0

def _pair_features(s_tbl: str, s_col: str, t_tbl: str, t_col: str,
                   ddl_map, prof_src, prof_tgt,
                   s_df: pd.DataFrame = None, t_df: pd.DataFrame = None) -> Dict[str, float]:
    # Name similarity
    fuzzy = string_similarity(s_col, t_col)
    norm_eq = 1.0 if normalize_name(s_col) == normalize_name(t_col) else 0.0

    # DDL meta
    s_dt = ddl_map.get(s_tbl, {}).get(s_col)
    t_dt = ddl_map.get(t_tbl, {}).get(t_col)
    type_match = 1.0 if s_dt and t_dt and _dtype_family(s_dt) == _dtype_family(t_dt) else 0.0
    len_diff = abs(_len_from_dtype(s_dt) - _len_from_dtype(t_dt))

    # Data-profile meta
    sprof = prof_src.get(s_tbl, {}).get(s_col, {})
    tprof = prof_tgt.get(t_tbl, {}).get(t_col, {})
    null_gap = abs((sprof.get("null_ratio", 0.0) or 0.0) - (tprof.get("null_ratio", 0.0) or 0.0))
    uniq_gap = abs((sprof.get("distinct_count", 0) or 0) - (tprof.get("distinct_count", 0) or 0))

    # Top-values overlap
    s_top = set(sprof.get("top_values", []) or [])
    t_top = set(tprof.get("top_values", []) or [])
    top_overlap = float(len(s_top & t_top) / max(1, len(s_top | t_top))) if (s_top or t_top) else 0.0

    # Extra value-based features
    extra_feats = {}
    if s_df is not None and t_df is not None and s_col in s_df.columns and t_col in t_df.columns:
        extra_feats = _extra_data_features(s_df[s_col], t_df[t_col])

    # Pattern score
    pattern_score = 0.0
    if s_df is not None and t_df is not None and s_col in s_df.columns and t_col in t_df.columns:
        pattern_score = _pattern_score(s_df[s_col], t_df[t_col])

    # Semantic score (can be fuzzy or synonym-based)
    semantic_score = fuzzy  # simple example; can include synonyms if available

    return {
        "fuzzy_name": float(fuzzy),
        "name_exact_norm": float(norm_eq),
        "dtype_compat": float(type_match),
        "len_diff": float(len_diff),
        "null_gap": float(null_gap),
        "uniq_gap": float(uniq_gap),
        "top_overlap": float(top_overlap),
        "value_overlap": extra_feats.get("value_overlap", 0.0),
        "mean_diff": extra_feats.get("mean_diff", 0.0),
        "std_diff": extra_feats.get("std_diff", 0.0),
        "avg_strlen_diff": extra_feats.get("avg_strlen_diff", 0.0),
        "semantic_score": semantic_score,
        "pattern_score": pattern_score
    }

def build_feature_matrix(cfg, synonyms: Dict[str, list], sample_rows: int = 1000, negative_ratio: int = None):
    negative_ratio = cfg.train.negative_ratio if negative_ratio is None else negative_ratio

    ddl_map = parse_ddl(cfg.paths.ddl_path)
    src_tables = load_csvs(cfg.paths.source_root, cfg.paths.source_files, nrows=sample_rows)
    tgt_tables = load_csvs(cfg.paths.target_root, cfg.paths.target_files, nrows=sample_rows)
    prof_src = profile_all(cfg.paths.source_root, cfg.paths.source_files, sample_rows=sample_rows)
    prof_tgt = profile_all(cfg.paths.target_root, cfg.paths.target_files, sample_rows=sample_rows)

    src_cols = [(tbl, col) for tbl, df in src_tables.items() if not df.empty for col in df.columns]
    tgt_cols = [(tbl, col) for tbl, df in tgt_tables.items() if not df.empty for col in df.columns]

    rows, labels = [], []

    # Positives
    for src_key, tgt_list in (synonyms or {}).items():
        try:
            s_tbl, s_col = src_key.split("::", 1)
        except Exception:
            continue
        for tgt_key in tgt_list:
            try:
                t_tbl, t_col = tgt_key.split("::", 1)
            except Exception:
                continue
            feats = _pair_features(s_tbl, s_col, t_tbl, t_col, ddl_map, prof_src, prof_tgt,
                                   src_tables.get(s_tbl), tgt_tables.get(t_tbl))
            rows.append({"source_table": s_tbl, "source_column": s_col,
                         "target_table": t_tbl, "target_column": t_col, **feats})
            labels.append(1)

    # Negatives
    rng = random.Random(cfg.train.random_state)
    neg_needed = int(len(labels) * negative_ratio) if labels else max(100, len(src_cols)*len(tgt_cols)//50)
    for _ in range(neg_needed):
        if not src_cols or not tgt_cols:
            break
        s_tbl, s_col = src_cols[rng.randrange(len(src_cols))]
        t_tbl, t_col = tgt_cols[rng.randrange(len(tgt_cols))]
        src_key, tgt_key = f"{s_tbl}::{s_col}", f"{t_tbl}::{t_col}"
        if (src_key in (synonyms or {}) and tgt_key in (synonyms or {}).get(src_key, [])):
            continue
        feats = _pair_features(s_tbl, s_col, t_tbl, t_col, ddl_map, prof_src, prof_tgt,
                               src_tables.get(s_tbl), tgt_tables.get(t_tbl))
        rows.append({"source_table": s_tbl, "source_column": s_col,
                     "target_table": t_tbl, "target_column": t_col, **feats})
        labels.append(0)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    full = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    feature_cols = ["fuzzy_name", "name_exact_norm", "dtype_compat", "len_diff",
                    "null_gap", "uniq_gap", "top_overlap",
                    "value_overlap", "mean_diff", "std_diff", "avg_strlen_diff",
                    "semantic_score", "pattern_score"]
    X = full[feature_cols].astype(float).fillna(0.0)
    return X, y, full

def build_features(cfg, synonyms, sample_rows: int = 1000):
    return build_feature_matrix(cfg, synonyms, sample_rows)
