import argparse
from pathlib import Path
import joblib, yaml
import pandas as pd
from .utils.io_utils import parse_ddl, load_csvs, write_json
from .features import _pair_features

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class C: pass
    c = C()
    c.paths = type("P", (), {
        "ddl_path": cfg["ddl_path"],
        "source_root": cfg["source"]["root"],
        "target_root": cfg["target"]["root"],
        "source_files": cfg["source"]["files"],
        "target_files": cfg["target"]["files"],
    })()
    c.predict = type("Pr", (), cfg["predict"])()
    c.synonyms_path = cfg.get("synonyms_path", "configs/synonyms.json")
    c.source_system = cfg.get("source_system", "Legacy")
    c.target_system = cfg.get("target_system", "LifePro")
    return c

def build_all_candidates(cfg, sample_rows: int = 500):
    ddl = parse_ddl(cfg.paths.ddl_path)
    src = load_csvs(cfg.paths.source_root, cfg.paths.source_files, nrows=sample_rows)
    tgt = load_csvs(cfg.paths.target_root, cfg.paths.target_files, nrows=sample_rows)

    rows = []
    for s_tbl, s_df in src.items():
        if s_df is None or s_df.empty:
            continue
        for s_col in s_df.columns:
            for t_tbl, t_df in tgt.items():
                if t_df is None or t_df.empty:
                    continue
                for t_col in t_df.columns:
                    feats = _pair_features(s_tbl, s_col, t_tbl, t_col, ddl, {}, {},
                                           s_df, t_df)
                    rows.append({
                        "source_table": s_tbl,
                        "source_column": s_col,
                        "target_table": t_tbl,
                        "target_column": t_col,
                        **feats
                    })
    return pd.DataFrame(rows)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    model_path = Path(cfg.predict.model_in)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = joblib.load(model_path)
    clf = bundle.get("classifier")
    feat_cols = bundle.get("feature_columns", [])

    candidates = build_all_candidates(cfg, sample_rows=500)
    if candidates.empty:
        raise RuntimeError("No candidate pairs constructed")

    X = candidates[feat_cols].astype(float).fillna(0.0)
    ml_probs = clf.predict_proba(X)[:, 1]

    candidates["ml_score"] = ml_probs
    candidates["fuzzy_score"] = candidates["fuzzy_name"]

    # Boost ML score using pattern_score
    candidates["combined_score"] = (
        0.4 * candidates["ml_score"] +
        0.2 * candidates["fuzzy_score"] +
        0.2 * candidates.get("semantic_score", 0.0) +
        0.2 * candidates.get("pattern_score", 0.0)
    )

    top_k = int(cfg.predict.top_k)
    out_rows, alternates = [], []
    grouped = candidates.groupby(["source_table", "source_column"])

    for (s_tbl, s_col), grp in grouped:
        g = grp.sort_values("combined_score", ascending=False).reset_index(drop=True)
        if g.empty:
            continue
        alt_list = g.head(top_k).to_dict(orient="records")
        alternates.append({"source_table": s_tbl, "source_column": s_col, "alternates": alt_list})
        best_row = g.iloc[0]
        scores = {
            "ML Score": best_row["ml_score"],
            "Fuzzy Score": best_row["fuzzy_score"],
            "Combined Score": best_row["combined_score"]
        }
        best_type = max(scores, key=lambda k: scores[k])
        out_rows.append({
            "source_system": cfg.source_system,
            "source_table": s_tbl,
            "source_column": s_col,
            "target_system": cfg.target_system,
            "target_table": best_row["target_table"],
            "target_column": best_row["target_column"],
            "ml_score": float(best_row["ml_score"]),
            "fuzzy_score": float(best_row["fuzzy_score"]),
            "combined_score": float(best_row["combined_score"]),
            "best_score": float(scores[best_type]),
            "best_score_type": best_type
        })

    out_df = pd.DataFrame(out_rows).sort_values(["source_table", "source_column"])
    out_df.to_csv(cfg.predict.out_csv, index=False)
    write_json(alternates, cfg.predict.out_json)
    print(f"Wrote suggestions: {cfg.predict.out_csv} and alternates: {cfg.predict.out_json}")

if __name__ == "__main__":
    main()
