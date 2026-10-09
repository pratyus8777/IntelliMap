import os
import pandas as pd
import pickle
import chardet
import yaml


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config into a dict"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_encoding(file_path: str) -> str:
    """Detect encoding of a CSV file using chardet"""
    with open(file_path, "rb") as f:
        raw_data = f.read(50000)  # read first 50KB
    result = chardet.detect(raw_data)
    return result["encoding"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for mapping CSV"""
    if len(df.columns) < 4:
        raise ValueError("[ERROR] initial_mapping.csv must have at least 4 columns")

    rename_map = {
        df.columns[0]: "source_table",
        df.columns[1]: "source_column",
        df.columns[2]: "target_table",
        df.columns[3]: "target_column",
    }

    df = df.rename(columns=rename_map)

    expected = ["source_table", "source_column", "target_table", "target_column"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise KeyError(f"[ERROR] Missing expected columns after renaming: {missing}")

    return df


# ----------------------------------------------------------------------
# Main pretraining pipeline
# ----------------------------------------------------------------------

def main(config_path: str):
    # Load config
    cfg = load_config(config_path)

    initial_mapping = cfg.get("initial_mapping", "data/initial_mapping.csv")
    output_dir = cfg.get("output_dir", "outputs")
    model_dir = cfg.get("model_dir", "models")
    encoding_mode = cfg.get("encoding", "auto")

    if not os.path.exists(initial_mapping):
        raise FileNotFoundError(f"[ERROR] Cannot find {initial_mapping}")

    # Detect encoding if auto
    if encoding_mode == "auto":
        encoding = detect_encoding(initial_mapping)
        print(f"[INFO] Detected encoding: {encoding}")
    else:
        encoding = encoding_mode

    # Load CSV
    df = pd.read_csv(initial_mapping, encoding=encoding)
    print(f"[INFO] Loaded initial_mapping.csv with {len(df)} rows and {len(df.columns)} columns")

    # Normalize columns
    df = normalize_columns(df)

    # ✅ Keep only the first 5 columns
    df = df.iloc[:, :5]

    # Save normalized CSV
    os.makedirs(output_dir, exist_ok=True)
    pretrained_csv = os.path.join(output_dir, "pretrained.csv")
    df.to_csv(pretrained_csv, index=False, encoding="utf-8")
    print(f"[INFO] Saved cleaned mapping (first 5 columns only) to {pretrained_csv}")

    # Save pickle for ML models
    os.makedirs(model_dir, exist_ok=True)
    pretrained_pkl = os.path.join(model_dir, "pretrained.pkl")
    with open(pretrained_pkl, "wb") as f:
        pickle.dump(df, f)
    print(f"[INFO] Saved pretrained.pkl to {pretrained_pkl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain mapping model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    main(args.config)
