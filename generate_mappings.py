import os
import pandas as pd
from collections import defaultdict

def generate_etl_spec_from_mapping(mapping_csv="outputs/verified_mappings.csv",
                                   source_folder="data/source",
                                   target_folder="data/target",
                                   output_csv="outputs/etl_spec.csv"):
    """
    Generate dynamic ETL spec CSV from mapping_suggestions.csv
    using actual column-level mappings and computing row/column counts.
    """
    # Load mapping
    mapping_df = pd.read_csv(mapping_csv)

    # Group by source_table and target_table
    table_groups = defaultdict(list)
    for _, row in mapping_df.iterrows():
        key = (row['source_table'], row['target_table'])
        table_groups[key].append((row['source_column'], row['target_column']))

    etl_rows = []

    for (src_table, tgt_table), columns in table_groups.items():
        # Compute row and column counts
        src_path = os.path.join(source_folder, f"{src_table}.csv")
        tgt_path = os.path.join(target_folder, f"{tgt_table}.csv")

        try:
            src_df = pd.read_csv(src_path)
            src_rows, src_cols = src_df.shape
        except Exception:
            src_rows, src_cols = None, None

        try:
            tgt_df = pd.read_csv(tgt_path)
            tgt_rows, tgt_cols = tgt_df.shape
        except Exception:
            tgt_rows, tgt_cols = None, None

        # Determine relationship
        if src_rows is not None and tgt_rows is not None:
            relationship = "1:1" if src_rows == tgt_rows else "1:1/1:M (depends on data)"
        else:
            relationship = ""

        # Join_Keys: pick columns ending with _id or primary-like columns
        join_keys_src = [s for s, t in columns if '_id' in s.lower()]
        join_keys_tgt = [t for s, t in columns if '_id' in s.lower()]
        join_keys = " / ".join(join_keys_src) + " → " + " / ".join(join_keys_tgt) if join_keys_src else ""

        # Typical_Field_Map
        field_map = ", ".join([f"{s} → {t}" for s, t in columns])

        etl_rows.append({
            "Legacy_Table": src_table,
            "Legacy_Row_Count": src_rows,
            "Legacy_Column_Count": src_cols,
            "LifePRO_Table": tgt_table,
            "LifePRO_Row_Count": tgt_rows,
            "LifePRO_Column_Count": tgt_cols,
            "Relationship": relationship,
            "Join_Keys": join_keys,
            "Typical_Field_Map": field_map,
            "Transform_Rules": "",
            "Load_Order": "",
            "Dependencies": "",
            "Test_Cases": "",
            
            
        })

    # Convert to DataFrame and save
    output_df = pd.DataFrame(etl_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"ETL spec generated: {output_csv}")
    print(output_df.head(10))

if __name__ == "__main__":
    generate_etl_spec_from_mapping()
