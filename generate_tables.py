import os
import sys
import logging
import pandas as pd
from typing import Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_table_csvs(input_csv: str, output_dir: str) -> Tuple[int, int]:
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logging.error(f"Failed to read {input_csv}: {e}")
        return 0, 0

    required_columns = {"Table_Name", "Typical_Columns"}
    if not required_columns.issubset(df.columns):
        logging.error(f"Missing required columns in {input_csv}. Required: {required_columns}")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)

    created_count = 0
    skipped_count = 0

    existing_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith(".csv")}

    for _, row in df.iterrows():
        table_name = str(row["Table_Name"]).strip()
        if not table_name:
            logging.warning("Skipping row with empty Table_Name")
            continue

        if table_name in existing_files:
            skipped_count += 1
            logging.info(f"Skipping (already exists): {table_name}.csv")
            continue

        
        typical_columns_raw = str(row["Typical_Columns"])

        
        col_list = [c.strip() for c in typical_columns_raw.split(";") if c.strip()]

        
        headers = []
        while col_list:
            headers.append(col_list.pop(0))  # pop first element

        
        if len(headers) < 2:
            headers.append("DUMMY_COLUMN")

        
        table_df = pd.DataFrame(columns=headers)
        file_path = os.path.join(output_dir, f"{table_name}.csv")

        try:
            table_df.to_csv(file_path, index=False)
            created_count += 1
            logging.info(f"Created: {file_path}")
        except Exception as e:
            logging.error(f"Failed to write {file_path}: {e}")

    logging.info(f"Summary for {output_dir} → Created: {created_count}, Skipped: {skipped_count}")
    return created_count, skipped_count


if __name__ == "__main__":
    base_path = "data"
    inputs_and_outputs = [
        (os.path.join(base_path, "source.csv"), os.path.join(base_path, "source")),
        (os.path.join(base_path, "target.csv"), os.path.join(base_path, "target")),
    ]

    total_created, total_skipped = 0, 0
    for csv_file, folder in inputs_and_outputs:
        c, s = create_table_csvs(csv_file, folder)
        total_created += c
        total_skipped += s

    logging.info(f"\nOverall Summary → Created: {total_created}, Skipped: {total_skipped}")
