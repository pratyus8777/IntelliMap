import os
import json
import pandas as pd
from difflib import SequenceMatcher

def generate_synonyms_json(base_path="data", output_folder="configs"):
    """
    Generates a synonyms.json file mapping source columns to target columns
    using fuzzy matching for similarity.
    """
    os.makedirs(output_folder, exist_ok=True)

    source_folder = os.path.join(base_path, "source")
    target_folder = os.path.join(base_path, "target")

    # List CSVs
    source_csvs = [f for f in os.listdir(source_folder) if f.endswith(".csv")]
    target_csvs = [f for f in os.listdir(target_folder) if f.endswith(".csv")]

    # Function to strip extensions and use as table name
    def table_name(file):
        return os.path.splitext(file)[0]

    synonyms = {}

    # Helper function for column fuzzy matching
    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Build a quick map of target table columns
    target_columns_map = {}
    for tgt_file in target_csvs:
        tgt_path = os.path.join(target_folder, tgt_file)
        try:
            df = pd.read_csv(tgt_path, nrows=5)  # small sample for speed
            target_columns_map[table_name(tgt_file)] = list(df.columns)
        except Exception as e:
            print(f"Warning: could not read {tgt_file}: {e}")
            target_columns_map[table_name(tgt_file)] = []

    # Iterate over source tables
    for src_file in source_csvs:
        src_table = table_name(src_file)
        src_path = os.path.join(source_folder, src_file)
        try:
            src_df = pd.read_csv(src_path, nrows=5)
        except Exception as e:
            print(f"Warning: could not read {src_file}: {e}")
            continue

        for src_col in src_df.columns:
            best_matches = []
            # Compare with all target columns
            for tgt_table, tgt_cols in target_columns_map.items():
                for tgt_col in tgt_cols:
                    if similar(src_col, tgt_col) > 0.7:  # threshold can be adjusted
                        best_matches.append(f"{tgt_table}::{tgt_col}")
            if best_matches:
                synonyms[f"{src_table}::{src_col}"] = best_matches

    # Write JSON file
    output_path = os.path.join(output_folder, "synonyms.json")
    with open(output_path, "w") as f:
        json.dump(synonyms, f, indent=2)

    print(f"Synonyms JSON generated at: {output_path}")
    print(json.dumps(synonyms, indent=2))


if __name__ == "__main__":
    generate_synonyms_json()
