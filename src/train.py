import argparse
from pathlib import Path
import joblib, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from .utils.io_utils import read_json
from .features import build_features

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class Cfg: pass
    c = Cfg()
    c.paths = type("P", (), {
        "ddl_path": cfg["ddl_path"],
        "source_root": cfg["source"]["root"],
        "target_root": cfg["target"]["root"],
        "source_files": cfg["source"]["files"],
        "target_files": cfg["target"]["files"],
    })()
    c.table_pairs = cfg.get("table_pairs", [])
    c.train = type("T", (), cfg["train"])()
    c.predict = type("Pr", (), cfg["predict"])()
    c.synonyms_path = cfg.get("synonyms_path", "configs/synonyms.json")
    return c

def make_model(random_state=42):
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state)
    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=random_state)
    model = VotingClassifier(estimators=[("rf", rf), ("hgb", hgb)], voting="soft")
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    try:
        synonyms = read_json(cfg.synonyms_path)
    except Exception as e:
        print(f"[WARN] Could not read synonyms.json: {e}")
        synonyms = {}

    X, y, full = build_features(cfg, synonyms, sample_rows=1000)
    if X.empty:
        raise RuntimeError("No training samples generated. Check CSVs and synonyms.json")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.train.test_size, random_state=cfg.train.random_state,
        stratify=y if len(set(y)) > 1 else None
    )

    model = make_model(cfg.train.random_state)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob) if len(set(y_test)) > 1 else float("nan")
    ap = average_precision_score(y_test, prob) if len(set(y_test)) > 1 else float("nan")
    f1 = f1_score(y_test, (prob >= 0.5).astype(int)) if len(set(y_test)) > 1 else float("nan")

    print(f"[EVAL] AUC={auc:.4f} AP={ap:.4f} F1={f1:.4f}")

    outp = Path(cfg.train.model_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": model, "feature_columns": list(X.columns)}, outp)
    print(f"Saved model bundle to: {outp}")

if __name__ == "__main__":
    main()
