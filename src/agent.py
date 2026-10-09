from __future__ import annotations
import os
import json
import yaml
import re
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class MappingAgentConfig:
    def __init__(
        self,
        combined_threshold: float = 0.50,
        ml_threshold: float = 0.40,
        fuzzy_threshold: float = 0.20,
        tfidf_sim_threshold: float = 0.45,
        best_vs_second_margin: float = 0.08,
        token_overlap_min: float = 0.25,
        deduplicate_keep_top: bool = True,
        top_k: int = 3,
        sample_values_n: int = 200,
    ):
        self.combined_threshold = float(combined_threshold)
        self.ml_threshold = float(ml_threshold)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.tfidf_sim_threshold = float(tfidf_sim_threshold)
        self.best_vs_second_margin = float(best_vs_second_margin)
        self.token_overlap_min = float(token_overlap_min)
        self.deduplicate_keep_top = bool(deduplicate_keep_top)
        self.top_k = int(top_k)
        self.sample_values_n = int(sample_values_n)


def load_cfg_from_yaml_full(cfg_path: Optional[str]) -> Tuple[MappingAgentConfig, Dict[str, Any]]:
    
    defaults = MappingAgentConfig()
    if not cfg_path or not os.path.exists(cfg_path):
        return defaults, {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_y = yaml.safe_load(f) or {}
        preds = cfg_y.get("predict", {})
        ft = cfg_y.get("filter_thresholds", {}) or cfg_y.get("predict", {}).get("filter_thresholds", {})
        cfg = MappingAgentConfig(
            combined_threshold = ft.get("combined_min", ft.get("combined_threshold", defaults.combined_threshold)),
            ml_threshold = ft.get("ml_min", defaults.ml_threshold),
            fuzzy_threshold = ft.get("fuzzy_min", defaults.fuzzy_threshold),
            tfidf_sim_threshold = ft.get("tfidf_sim_threshold", defaults.tfidf_sim_threshold),
            best_vs_second_margin = ft.get("best_vs_second_margin", defaults.best_vs_second_margin),
            token_overlap_min = ft.get("token_overlap_min", defaults.token_overlap_min),
            deduplicate_keep_top = ft.get("deduplicate_keep_top", defaults.deduplicate_keep_top),
            top_k = int(preds.get("top_k", defaults.top_k)),
            sample_values_n = int(ft.get("sample_values_n", defaults.sample_values_n)),
        )
        return cfg, cfg_y
    except Exception:
        return defaults, {}

_non_alnum_re = re.compile(r"[^A-Za-z0-9]+")
_camel_re = re.compile(r"[A-Z]?[a-z]+|[0-9]+|[A-Z]+")

def normalize_text(s: Optional[Any]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = _non_alnum_re.sub(" ", s)
    s = " ".join(part.lower() for part in _camel_re.findall(s))
    return " ".join(s.split())

def token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / max(1, len(union))


class TfidfSemantic:
    def __init__(self, ngram_range=(1,2), min_df=1):
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=ngram_range, min_df=min_df)

    def pairwise_max_similarity(self, left_texts: List[str], right_texts: List[str]) -> float:
        left_texts = [normalize_text(x) for x in (left_texts or []) if normalize_text(x)]
        right_texts = [normalize_text(x) for x in (right_texts or []) if normalize_text(x)]
        if not left_texts or not right_texts:
            return 0.0
        docs = left_texts + right_texts
        try:
            tfidf = self.vectorizer.fit_transform(docs)
            n_left = len(left_texts)
            left_vecs = tfidf[:n_left]
            right_vecs = tfidf[n_left:]
            if left_vecs.shape[0] == 0 or right_vecs.shape[0] == 0:
                return 0.0
            sims = linear_kernel(left_vecs, right_vecs)
            return float(sims.max())
        except Exception:
            return 0.0


EMAIL_RE = re.compile(r"^[\w\.\-+%]+@[\w\.\-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[\d\-\s\(\)]{6,}$")
DATE_RE = re.compile(r"^\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$")  # simple
NUMERIC_RE = re.compile(r"^[\d\.\-]+$")

def infer_value_type_from_series(series: pd.Series, sample_n: int = 200) -> Tuple[str, Dict[str,int]]:
    vals = series.dropna().astype(str)
    if vals.empty:
        return "empty", {}
    if len(vals) > sample_n:
        vals = vals.sample(n=sample_n, random_state=1)
    counts = {"email":0, "phone":0, "date":0, "numeric":0, "code_like":0, "text":0}
    for v in vals.iloc[:sample_n]:
        v = v.strip()
        low = v.lower()
        if not v:
            continue
        if EMAIL_RE.match(low):
            counts["email"] += 1
            continue
        if PHONE_RE.match(v):
            if re.search(r"[A-Za-z]", v):
                counts["text"] += 1
            else:
                counts["phone"] += 1
            continue
        if DATE_RE.match(v):
            counts["date"] += 1
            continue
        if NUMERIC_RE.match(v):
            counts["numeric"] += 1
            continue
        if len(v) <= 20 and re.match(r"^[A-Za-z0-9\-\_\/]+$", v):
            counts["code_like"] += 1
            continue
        counts["text"] += 1

    # decide predominant
    predominant = max(counts.items(), key=lambda x: x[1])
    if predominant[1] == 0:
        # fallback to text
        return "text", counts
    # map code_like -> code
    tmap = {"code_like":"code", "numeric":"numeric", "email":"email", "phone":"phone", "date":"date", "text":"text"}
    return tmap.get(predominant[0], "text"), counts


def load_table_column_samples(cfg_raw: Dict[str, Any], table_name: str, column_name: str, n: int = 200) -> List[str]:
    src_root = None
    tgt_root = None
    try:
        src_root = cfg_raw.get("source", {}).get("root")
    except Exception:
        src_root = None
    try:
        tgt_root = cfg_raw.get("target", {}).get("root")
    except Exception:
        tgt_root = None

    # prefer source root then target root; look in both if necessary
    candidate_paths = []
    if src_root:
        candidate_paths.append(os.path.join(src_root, f"{table_name}.csv"))
    if tgt_root:
        candidate_paths.append(os.path.join(tgt_root, f"{table_name}.csv"))

    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p, usecols=[column_name], dtype=str, low_memory=True)
                vals = df[column_name].dropna().astype(str)
                if vals.empty:
                    return []
                if len(vals) > n:
                    vals = vals.sample(n=n, random_state=1)
                return [str(x) for x in vals.iloc[:n]]
            except Exception:
                # column may not exist or read error
                try:
                    # read full and then try
                    df = pd.read_csv(p, dtype=str, low_memory=True)
                    if column_name in df.columns:
                        vals = df[column_name].dropna().astype(str)
                        if vals.empty:
                            return []
                        if len(vals) > n:
                            vals = vals.sample(n=n, random_state=1)
                        return [str(x) for x in vals.iloc[:n]]
                except Exception:
                    continue
    return []

def sample_column_series(cfg_raw: Dict[str, Any], table: str, column: str, n: int = 200) -> pd.Series:
    samples = load_table_column_samples(cfg_raw, table, column, n=n)
    return pd.Series(samples)


class MappingAgent:
    def __init__(self, cfg: Optional[MappingAgentConfig] = None, synonyms: Optional[Dict[str, List[str]]] = None):
        self.cfg = cfg or MappingAgentConfig()
        self.synonyms = synonyms or {}
        self.semantic = TfidfSemantic()

    @staticmethod
    def load_synonyms(path: str) -> Dict[str, List[str]]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = {}
            for k, v in (data.items() if isinstance(data, dict) else []):
                key = normalize_text(k)
                vals = [normalize_text(x) for x in (v or []) if x]
                out[key] = vals
            return out
        except Exception:
            return {}

    def _pick_score(self, row: pd.Series) -> float:
        for c in ("combined_score", "ml_score", "score", "best_score"):
            if c in row and pd.notna(row[c]):
                try:
                    return float(row[c])
                except Exception:
                    continue
        ml = row.get("ml_score", 0.0) or 0.0
        fuzzy = row.get("fuzzy_score", 0.0) or 0.0
        try:
            return float(0.4 * float(ml) + 0.2 * float(fuzzy))
        except Exception:
            return 0.0

    def decide_row(self, row: pd.Series, cfg_raw: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str,Any]]:
        reasons: List[str] = []
        diag: Dict[str,Any] = {}
        keep = True

        score = float(self._pick_score(row))
        diag["score"] = score
        if score >= self.cfg.combined_threshold:
            reasons.append(f"combined_ok:{score:.3f}")
        else:
            reasons.append(f"combined_low:{score:.3f}")

        ml_ok = float(row.get("ml_score", 0.0) or 0.0) >= self.cfg.ml_threshold
        fuzzy_ok = float(row.get("fuzzy_score", 0.0) or 0.0) >= self.cfg.fuzzy_threshold
        if ml_ok and fuzzy_ok:
            reasons.append("ml_fuzzy_ok")
        else:
            if not ml_ok:
                reasons.append(f"ml_low:{float(row.get('ml_score',0.0)):.3f}")
            if not fuzzy_ok:
                reasons.append(f"fuzzy_low:{float(row.get('fuzzy_score',0.0)):.3f}")

        s_col = normalize_text(row.get("source_column", "") or row.get("src_col", "") or "")
        t_col = normalize_text(row.get("target_column", "") or row.get("tgt_col", "") or "")
        overlap = token_overlap_ratio(s_col, t_col)
        reasons.append(f"tok_overlap:{overlap:.3f}")
        diag["token_overlap"] = overlap

        # infer value types from sample values (load CSVs using cfg_raw)
        s_table = row.get("source_table")
        t_table = row.get("target_table")
        s_samples = sample_column_series(cfg_raw, s_table, row.get("source_column"), n=self.cfg.sample_values_n)
        t_samples = sample_column_series(cfg_raw, t_table, row.get("target_column"), n=self.cfg.sample_values_n)
        s_type, s_counts = infer_value_type_from_series(s_samples, sample_n=self.cfg.sample_values_n)
        t_type, t_counts = infer_value_type_from_series(t_samples, sample_n=self.cfg.sample_values_n)
        diag["source_type"] = s_type
        diag["target_type"] = t_type
        diag["source_counts"] = s_counts
        diag["target_counts"] = t_counts
        reasons.append(f"src_type:{s_type}")
        reasons.append(f"tgt_type:{t_type}")

        # if types present and clearly different, mark mismatch
        if s_type != "empty" and t_type != "empty" and s_type != t_type:
            # allow code<->text leniency; but disallow email<->phone or phone<->numeric etc.
            incompatible_pairs = {("email","phone"), ("email","numeric"), ("phone","numeric"), ("phone","date"), ("email","date")}
            if (s_type, t_type) in incompatible_pairs or (t_type, s_type) in incompatible_pairs:
                reasons.append("type_mismatch")
                keep = False

        # tfidf similarity between names + sample text
        left_texts = [s_col] + (s_samples.dropna().astype(str).tolist() if isinstance(s_samples, pd.Series) else [])
        right_texts = [t_col] + (t_samples.dropna().astype(str).tolist() if isinstance(t_samples, pd.Series) else [])
        tfidf_sim = 0.0
        try:
            tfidf_sim = self.semantic.pairwise_max_similarity(left_texts[:10], right_texts[:10])
        except Exception:
            tfidf_sim = 0.0
        diag["tfidf_sim"] = tfidf_sim
        if tfidf_sim >= self.cfg.tfidf_sim_threshold:
            reasons.append(f"tfidf_ok:{tfidf_sim:.3f}")
        else:
            reasons.append(f"tfidf_low:{tfidf_sim:.3f}")
            # if both token overlap and tfidf low and scores low -> reject
            if score < (self.cfg.combined_threshold + 0.05) and overlap < self.cfg.token_overlap_min:
                keep = False

        return bool(keep), reasons, diag

    def find_better_target_column(self, row: pd.Series, cfg_raw: Dict[str, Any]) -> Optional[Dict[str,Any]]:
        t_table = row.get("target_table")
        if not t_table:
            return None
        # locate target table CSV
        target_root = cfg_raw.get("target", {}).get("root")
        if not target_root:
            return None
        tgt_path = os.path.join(target_root, f"{t_table}.csv")
        if not os.path.exists(tgt_path):
            return None
        try:
            tgt_df = pd.read_csv(tgt_path, dtype=str, low_memory=True)
        except Exception:
            try:
                tgt_df = pd.read_csv(tgt_path, low_memory=True)
            except Exception:
                return None
        src_table = row.get("source_table")
        src_col = row.get("source_column")
        # infer source column type
        s_samples = sample_column_series(cfg_raw, src_table, src_col, n=self.cfg.sample_values_n)
        s_type, _ = infer_value_type_from_series(s_samples, sample_n=self.cfg.sample_values_n)
        best_candidate = None
        best_score = -1.0
        # scan columns
        for col in tgt_df.columns:
            t_samples = tgt_df[col].dropna().astype(str)
            if t_samples.empty:
                continue
            # sample
            if len(t_samples) > self.cfg.sample_values_n:
                t_sample = t_samples.sample(n=self.cfg.sample_values_n, random_state=1).astype(str).tolist()
            else:
                t_sample = t_samples.astype(str).tolist()
            t_type, _ = infer_value_type_from_series(pd.Series(t_sample), sample_n=self.cfg.sample_values_n)
            # require same type for strong preference, but allow code/text fuzziness
            type_score = 1.0 if (s_type == t_type and s_type != "text") else 0.6 if s_type == t_type else 0.0
            # token overlap
            overlap = token_overlap_ratio(src_col, col)
            # tfidf sim between src samples/colname and tgt samples/colname
            left_texts = [normalize_text(src_col)] + (s_samples.dropna().astype(str).tolist() if isinstance(s_samples, pd.Series) else [])
            right_texts = [normalize_text(col)] + t_sample[:10]
            try:
                tfidf_sim = self.semantic.pairwise_max_similarity(left_texts[:10], right_texts[:10])
            except Exception:
                tfidf_sim = 0.0
            # combine signals heuristically
            score = 0.5 * type_score + 0.3 * overlap + 0.2 * tfidf_sim
            # boost if exact name match
            if normalize_text(src_col) == normalize_text(col):
                score += 0.3
            if score > best_score:
                best_score = score
                best_candidate = {"target_table": t_table, "target_column": col, "score": score, "overlap": overlap, "tfidf": tfidf_sim, "t_type": t_type}
        # accept candidate if score exceeds a threshold
        if best_candidate and best_candidate["score"] >= 0.5:
            return best_candidate
        return None

# ---------- Integration helpers ----------
def default_agent_from_repo_configs(configs_root: str = "configs") -> MappingAgent:
    cfg_path = os.path.join(configs_root, "config.yaml")
    cfg, cfg_raw = load_cfg_from_yaml_full(cfg_path) if os.path.exists(cfg_path) else (MappingAgentConfig(), {})
    syn_path = cfg_raw.get("synonyms_path", os.path.join(configs_root, "synonyms.json")) if cfg_raw else os.path.join(configs_root, "synonyms.json")
    synonyms = MappingAgent.load_synonyms(syn_path) if syn_path else {}
    return MappingAgent(cfg=cfg, synonyms=synonyms)

def filter_candidates_df(candidates: pd.DataFrame, cfg_path: Optional[str] = None) -> pd.DataFrame:
    """
    Filter the large candidates dataframe (pre-grouping). Keeps plausible candidate rows only.
    """
    cfg, cfg_raw = load_cfg_from_yaml_full(cfg_path) if cfg_path else (MappingAgentConfig(), {})
    agent = MappingAgent(cfg=cfg, synonyms=MappingAgent.load_synonyms(cfg_raw.get("synonyms_path","configs/synonyms.json") if cfg_raw else "configs/synonyms.json"))
    results = []
    for (s_table, s_col), group in candidates.groupby(["source_table", "source_column"]):
        # apply agent decide_row for each row
        rows = []
        for _, r in group.iterrows():
            keep, reasons, diag = agent.decide_row(r, cfg_raw)
            rd = r.to_dict()
            rd["keep"] = bool(keep)
            rd["reasons"] = ";".join(reasons)
            rd.update({"diag_score": diag.get("score"), "diag_tfidf": diag.get("tfidf_sim"), "diag_src_type": diag.get("source_type"), "diag_tgt_type": diag.get("target_type")})
            rows.append(rd)
        out = pd.DataFrame(rows)
        # keep only rows where keep==True
        kept = out[out["keep"]].copy()
        if not kept.empty:
            # deduplicate to top per source
            kept = kept.sort_values(by="diag_score", ascending=False).groupby(["source_table", "source_column"], as_index=False).first()
            results.append(kept)
    if not results:
        return pd.DataFrame(columns=list(candidates.columns) + ["keep", "reasons"])
    out_all = pd.concat(results, ignore_index=True, sort=False)
    return out_all.reset_index(drop=True)

def verify_mappings(mapping_df: pd.DataFrame,
                    alternates_json: Optional[str] = None,
                    cfg_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Post-process the final mapping DataFrame (the one produced by predict.py).
    Performs row-level decide checks and group-level margin enforcement.
    If top mapping appears wrong, attempts to find a better target column in the same target table by inspecting data.
    Returns: (filtered_mapping_df, report_dict)
    """
    cfg, cfg_raw = load_cfg_from_yaml_full(cfg_path) if cfg_path else (MappingAgentConfig(), {})
    synonyms = MappingAgent.load_synonyms(cfg_raw.get("synonyms_path","configs/synonyms.json")) if cfg_raw else {}
    agent = MappingAgent(cfg=cfg, synonyms=synonyms)

    alternates_map: Dict[Tuple[str,str], List[Dict[str,Any]]] = {}
    if alternates_json and os.path.exists(alternates_json):
        try:
            with open(alternates_json, "r", encoding="utf-8") as f:
                alt_list = json.load(f)
            for a in alt_list:
                key = (a.get("source_table"), a.get("source_column"))
                alternates_map[key] = a.get("alternates", [])
        except Exception:
            alternates_map = {}

    df = mapping_df.copy()
    processed_rows = []
    # Precompute per-group sorted alternatives if alternates_map provided
    for (s_tbl, s_col), g in df.groupby(["source_table", "source_column"]):
        g_sorted = g.sort_values(by=["combined_score" if "combined_score" in g.columns else "ml_score"], ascending=False).reset_index()
        if g_sorted.empty:
            continue
        top = g_sorted.iloc[0]
        top_idx = int(top["index"])
        # apply agent on top candidate
        keep_top, reasons_top, diag_top = agent.decide_row(top, cfg_raw)
        rowd_top = top.to_dict()
        rowd_top["keep_agent"] = bool(keep_top)
        rowd_top["reasons_agent"] = ";".join(reasons_top)
        rowd_top["diag"] = diag_top
        # if top rejected, try to find a better column in the same target table
        if not keep_top:
            replacement = agent.find_better_target_column(top, cfg_raw)
            if replacement:
                # replace target column and mark as kept
                rowd_top["target_table"] = replacement["target_table"]
                rowd_top["target_column"] = replacement["target_column"]
                rowd_top["keep_agent"] = True
                rowd_top["reasons_agent"] += f";replaced_by:{replacement['target_column']};rep_score:{replacement['score']:.3f}"
                # update diag
                rowd_top["diag"]["replacement"] = replacement
            else:
                # if no replacement and alternates have a better candidate, check alternates
                alts = alternates_map.get((s_tbl, s_col), [])
                # if alternates exist, see if any alternate passes agent
                alt_kept = False
                for alt in alts:
                    # alt is a dict with target_table/target_column and possibly combined_score
                    fake_row = rowd_top.copy()
                    fake_row["target_table"] = alt.get("target_table")
                    fake_row["target_column"] = alt.get("target_column")
                    # convert to Series
                    fake_series = pd.Series(fake_row)
                    keep_alt, reasons_alt, diag_alt = agent.decide_row(fake_series, cfg_raw)
                    if keep_alt:
                        rowd_top["target_table"] = alt.get("target_table")
                        rowd_top["target_column"] = alt.get("target_column")
                        rowd_top["keep_agent"] = True
                        rowd_top["reasons_agent"] += f";alt_selected:{alt.get('target_column')}"
                        rowd_top["diag"]["alt_diag"] = diag_alt
                        alt_kept = True
                        break
                if not alt_kept:
                    # top remains rejected
                    pass
        # if top kept, ensure margin vs second-best
        if rowd_top.get("keep_agent", False):
            if len(g_sorted) > 1:
                # compute margin
                second = g_sorted.iloc[1]
                top_score = float(top.get("combined_score", top.get("ml_score", 0.0)))
                second_score = float(second.get("combined_score", second.get("ml_score", 0.0)))
                margin = top_score - second_score
                if margin < cfg.best_vs_second_margin:
                    # require stronger evidence (token overlap or tfidf)
                    overlap = token_overlap_ratio(rowd_top.get("source_column",""), rowd_top.get("target_column",""))
                    tfidf_sim = 0.0
                    try:
                        tfidf_sim = agent.semantic.pairwise_max_similarity([normalize_text(rowd_top.get("source_column",""))],
                                                                           [normalize_text(rowd_top.get("target_column",""))])
                    except Exception:
                        tfidf_sim = 0.0
                    if overlap < cfg.token_overlap_min and tfidf_sim < cfg.tfidf_sim_threshold:
                        # treat as rejected unless we can find replacement
                        replacement = agent.find_better_target_column(top, cfg_raw)
                        if replacement:
                            rowd_top["target_column"] = replacement["target_column"]
                            rowd_top["reasons_agent"] += f";replaced_by:{replacement['target_column']}"
                            rowd_top["diag"]["replacement_margin"] = replacement
                        else:
                            rowd_top["keep_agent"] = False
                            rowd_top["reasons_agent"] += ";dropped_due_to_weak_margin"
        processed_rows.append(rowd_top)

    kept = [r for r in processed_rows if r.get("keep_agent", False)]
    kept_df = pd.DataFrame(kept) if kept else pd.DataFrame(columns=df.columns.tolist() + ["keep_agent","reasons_agent","diag"])
    report = {
        "initial_count": len(df),
        "kept_count": len(kept_df),
        "dropped_count": int(len(df) - len(kept_df)),
    }

    return kept_df.reset_index(drop=True), report




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mapping verification agent with data-type checks")
    parser.add_argument("--in", dest="input_csv", required=True)
    parser.add_argument("--out", dest="output_csv", required=True)
    parser.add_argument("--cfg", dest="cfg", default="configs/config.yaml")
    args = parser.parse_args()

    cfg_path = args.cfg
    cfg, cfg_raw = load_cfg_from_yaml_full(cfg_path) if os.path.exists(cfg_path) else (MappingAgentConfig(), {})
    agent = MappingAgent(cfg=cfg, synonyms=MappingAgent.load_synonyms(cfg_raw.get("synonyms_path","configs/synonyms.json") if cfg_raw else "configs/synonyms.json"))
    df = pd.read_csv(args.input_csv)
    out_df, report = verify_mappings(df, alternates_json=None, cfg_path=cfg_path)
    out_df.to_csv(args.output_csv, index=False)
    print("Report:", report)
    print("Wrote:", args.output_csv)
