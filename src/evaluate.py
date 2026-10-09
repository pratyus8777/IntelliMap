import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(pred_df: pd.DataFrame, synonyms: dict):
    # build gold set
    gold = set()
    for s, targets in (synonyms or {}).items():
        s_tbl, s_col = s.split("::",1)
        for t in targets:
            t_tbl, t_col = t.split("::",1)
            gold.add((s_tbl, s_col, t_tbl, t_col))
    y_true = []
    y_score = []
    for _, r in pred_df.iterrows():
        tup = (r["source_table"], r["source_column"], r["target_table"], r["predicted_target_column"])
        y_true.append(1 if tup in gold else 0)
        y_score.append(r.get("combined_score", r.get("best_score", 0.0)))
    if not y_true:
        return {}
    try:
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
    except Exception:
        auc, ap = None, None
    return {"auc": auc, "ap": ap}
