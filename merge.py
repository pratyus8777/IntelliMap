import pandas as pd

# Paths
mapping_csv = "outputs/mapping_suggestions.csv"
pretrained_csv = "outputs/pretrained.csv"
output_csv = "outputs/combined_mapping.csv"

# Load CSVs
mapping_df = pd.read_csv(mapping_csv)
pretrained_df = pd.read_csv(pretrained_csv)

# Key columns for matching
key_cols = ["source_table", "source_column", "target_table", "target_column"]

# Ensure key columns exist in mapping
for col in key_cols:
    if col not in mapping_df.columns:
        raise KeyError(f"Column '{col}' not found in {mapping_csv}")

# Convert key columns to string
mapping_df[key_cols] = mapping_df[key_cols].astype(str)
pretrained_df[key_cols] = pretrained_df.get(key_cols, pd.DataFrame(columns=key_cols)).astype(str)

# Check which source and target columns actually exist
existing_source_cols = set(zip(mapping_df["source_table"], mapping_df["source_column"]))
existing_target_cols = set(zip(mapping_df["target_table"], mapping_df["target_column"]))

def row_exists(row):
    src_exists = (row["source_table"], row["source_column"]) in existing_source_cols
    tgt_exists = (row["target_table"], row["target_column"]) in existing_target_cols
    return src_exists and tgt_exists

# Filter pretrained rows to only those where columns exist
pretrained_df = pretrained_df[pretrained_df.apply(row_exists, axis=1)]

# Mark pretrained rows with best score
pretrained_df = pretrained_df.copy()
pretrained_df["best_score"] = 1.0
pretrained_df["best_score_type"] = "previous knowledge"

# Merge: replace mapping rows if there is an exact match
merged = mapping_df.merge(
    pretrained_df[key_cols + ["best_score", "best_score_type"]],
    on=key_cols,
    how="left",
    suffixes=("", "_pretrained")
)

# Replace scores if pretrained exists
for col in ["best_score", "best_score_type"]:
    merged[col] = merged[f"{col}_pretrained"].combine_first(merged[col])
    merged.drop(columns=[f"{col}_pretrained"], inplace=True)

# Add pretrained rows not already in mapping
existing_keys = set(zip(
    merged["source_table"], merged["source_column"],
    merged["target_table"], merged["target_column"]
))

new_pretrained_rows = pretrained_df[
    ~pretrained_df.apply(lambda r: (r["source_table"], r["source_column"],
                                    r["target_table"], r["target_column"]) in existing_keys, axis=1)
]

# Concatenate final dataframe
final_df = pd.concat([merged, new_pretrained_rows], ignore_index=True)

# Save output
final_df.to_csv(output_csv, index=False)
print(f"Combined mapping saved to: {output_csv}")
