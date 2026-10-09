import os
import yaml
from difflib import SequenceMatcher

def generate_config_yaml_dynamic(base_path="data"):
    """
    Automatically generates configs/config.yaml with dynamic table names
    and table_pairs in the requested block style format.
    """
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)

    source_folder = os.path.join(base_path, "source")
    target_folder = os.path.join(base_path, "target")

    # List CSVs
    source_csvs = [f for f in os.listdir(source_folder) if f.endswith(".csv")]
    target_csvs = [f for f in os.listdir(target_folder) if f.endswith(".csv")]

    # Generate friendly names: strip extensions
    def friendly_name(filename):
        return os.path.splitext(filename)[0]

    source_files = {friendly_name(f): f for f in source_csvs}
    target_files = {friendly_name(f): f for f in target_csvs}

    # Generate table_pairs by fuzzy matching names
    table_pairs = []
    used_targets = set()

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    for src_name in source_files.keys():
        best_match = None
        highest_score = 0
        for tgt_name in target_files.keys():
            if tgt_name in used_targets:
                continue
            score = similarity(src_name, tgt_name)
            if score > highest_score:
                highest_score = score
                best_match = tgt_name
        if best_match:
            table_pairs.append([src_name, best_match])
            used_targets.add(best_match)

    # Build config dictionary
    config_data = {
        "ddl_path": "DDL.sql",
        "source": {
            "root": source_folder,
            "files": source_files
        },
        "target": {
            "root": target_folder,
            "files": target_files
        },
        "table_pairs": table_pairs,
        "train": {
            "model_out": "models/matcher.pkl",
            "negative_ratio": 3,
            "test_size": 0.2,
            "random_state": 42
        },
        "predict": {
            "model_in": "models/matcher.pkl",
            "top_k": 5,
            "threshold": 0.7,
            "out_csv": "outputs/mapping_suggestions.csv",
            "out_json": "outputs/mapping_suggestions.json"
        }
    }

    # Custom representer to force inner lists to flow style
    class FlowStyleList(list): pass

    yaml.add_representer(
        FlowStyleList,
        lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    )

    # Convert inner lists to FlowStyleList
    config_data["table_pairs"] = [FlowStyleList(pair) for pair in table_pairs]

    # Write YAML with block style outer list
    config_path = os.path.join(configs_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)

    print(f"Config file generated at: {config_path}")
    print("Table pairs:")
    for pair in table_pairs:
        print(f"  - {pair}")


if __name__ == "__main__":
    generate_config_yaml_dynamic()
