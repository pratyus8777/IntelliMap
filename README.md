
# PE_IMAP_1_Production

Legacy-to-target data mapping, ETL planning, and ML-assisted migration automation for insurance and enterprise data modernization projects.

## Overview

This project automates the early and most time-consuming work of a data migration initiative: identifying which legacy columns likely map to which target columns, scoring those mappings, and turning the approved mappings into a usable ETL specification.

The solution is built around a pragmatic pattern used in enterprise conversion projects:

- parse the legacy and target table metadata from CSV or DDL definitions
- profile source and target columns for structure, nullability, value overlap, and naming features
- build a feature set for every possible source-target column pair
- train a machine learning model to rank likely column matches
- allow manual override and synonym control using a configuration file
- export mapping suggestions and a table-level ETL spec for downstream use

This repository is tailored to a legacy insurance system and a target LifePRO-style schema, but the logic is general enough to adapt to other migration domains.

## What the project does

The project supports four major workflows:

1. Schema and metadata normalization
   - reads source and target table definitions from `data/source.csv` and `data/target.csv`
   - creates CSV table skeletons in `data/source/` and `data/target/`
   - optionally parses DDL from `DDL.sql`

2. Feature engineering for data mapping
   - compares column names using string similarity heuristics
   - checks type compatibility and length compatibility
   - compares null ratios, cardinality, and top value overlap
   - looks at data pattern similarity and value overlap

3. ML-based mapping recommendation
   - trains a classifier on positive and negative column-pair examples
   - outputs one best candidate per source column and a ranked list of alternates
   - writes suggestions to CSV and JSON outputs

4. ETL design preparation
   - groups mappings by source and target tables
   - estimates table row/column counts
   - forms a draft ETL field map and table relationship summary
   - saves a migration-ready CSV in `outputs/etl_spec.csv`

## Why this repo exists

In real migration projects, domain teams are usually asked to answer questions like:

- Which target table should this legacy table become?
- Which legacy column corresponds to which target field?
- Which mappings are high confidence versus weak?
- Can we generate an ETL design without manually writing a huge mapping matrix?

This project is designed to reduce that effort by turning the mapping process into a repeatable, scriptable workflow driven by data and model scoring.

## High-level architecture

The repository is structured as a lightweight data-migration pipeline:

- `data/` holds the domain metadata and CSV table definitions
- `configs/config.yaml` defines the source/target systems and table-pair mappings
- `src/features.py` computes feature vectors for column pairs
- `src/train.py` trains the candidate ranking model
- `src/predict.py` scores and selects the best mapping candidates
- `generate_tables.py` creates per-table CSV stubs from metadata files
- `generate_mappings.py` creates an ETL mapping spec from verified or generated mapping outputs
- `dashboard.py` provides a Streamlit exploration UI for reviewing mappings

## Repository structure

```text
PE_IMAP_1_Production/
├─ .venv/                       # local virtual environment
├─ configs/
│  ├─ config.yaml              # source/target mapping config and table pairs
│  └─ synonyms.json            # optional mapping overrides / known-good mappings
├─ data/
│  ├─ source.csv               # legacy domain metadata, table names, and typical columns
│  ├─ target.csv               # target domain metadata, table names, and typical columns
│  ├─ source/                  # generated legacy table CSV skeletons
│  └─ target/                  # generated target table CSV skeletons
├─ src/
│  ├─ features.py              # feature extraction for source-target column pairs
│  ├─ train.py                 # training pipeline for the mapping model
│  ├─ predict.py               # scoring and best-match selection
│  ├─ profiler.py              # column profiling/statistics utilities
│  ├─ evaluate.py              # evaluation helpers
│  ├─ pre_features.py          # preprocessing support utilities
│  ├─ pretrain.py              # training-related helpers
│  ├─ models.py                # model definitions or wrappers
│  ├─ agent.py                 # automation or agent support utilities
│  └─ utils/
│     ├─ io_utils.py          # JSON/CSV/DDL I/O helpers
│     └─ text_sim.py           # name similarity functions
├─ DDL.sql                     # DDL schema reference for legacy/target systems
├─ dashboard.py                # Streamlit dashboard for reviewing matches
├─ generate_tables.py          # creates table CSV skeletons from metadata sheets
├─ generate_mappings.py        # builds ETL mapping spec from the mapping output
├─ config.py                   # table-generation utility based on metadata CSVs
├─ merge.py                    # merge/transformation support
├─ synonyms.py                 # synonym utilities or mapping helpers
├─ docker-compose.yml          # Docker-related project config
├─ dockerfile                  # container build definition
├─ requirements.txt            # project dependencies
├─ models/                     # saved model artifacts (runtime-generated)
├─ outputs/                    # generated CSV/JSON mapping results
├─ logs/                       # runtime logs
├─ catboost_info/              # training logs and catboost metadata
├─ README.md                   # project documentation (this file)
└─ LICENSE                     # if added later
```

## Data model and conventions

The project assumes a table-driven workflow.

### Metadata inputs

The metadata files `data/source.csv` and `data/target.csv` contain rows like this:

```csv
Domain,Table_Name,Description,Primary_Key,Natural_Key_or_FKs,Typical_Columns
PARTY & CUSTOMER,LEG_PARTY,Party master record,party_id,party_id,name_last; name_first; dob; tax_id; gender; created_ts; updated_ts
```

Each row describes:

- `Domain`: business domain or subject area
- `Table_Name`: logical name of the table
- `Description`: human-readable explanation of the table
- `Primary_Key`: primary key column(s)
- `Natural_Key_or_FKs`: natural key or foreign key references
- `Typical_Columns`: semicolon-delimited list of common columns

This metadata is used to create skeleton CSV files for each table under `data/source/` and `data/target/`.

### Configuration model

The main pipeline settings are defined in `configs/config.yaml`.

A representative section looks like this:

```yaml
source:
  root: data\source
  files:
    LEG_ACTIVITY: LEG_ACTIVITY.csv
    LEG_PARTY: LEG_PARTY.csv

target:
  root: data\target
  files:
    LP_ACTIVITY: LP_ACTIVITY.csv
    LP_PARTY: LP_PARTY.csv

table_pairs:
  - [LEG_ACTIVITY, LP_ACTIVITY]
  - [LEG_PARTY, LP_PARTY]
```

This file controls:

- which source and target tables are loaded
- which table pairs are evaluated together
- which DDL file is referenced
- which models and outputs are used in training and prediction

### Synonym overrides

The project also supports mapping overrides through `configs/synonyms.json`.

This file is used to encode known-correct mappings such as:

```json
{
  "LEG_POLICYHOLDER::party_id": ["LP_POLICY_OWNER::party_id"],
  "LEG_PARTY::tax_id": ["LP_PARTY::tax_id"]
}
```

These known-good mappings can be used to anchor the model and reduce noise in highly repetitive domains.

## Installation

Create and activate a local Python environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using PowerShell, activation is usually:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Typical workflow

### 1. Prepare the table metadata

The repository ships with source and target metadata sheets in `data/source.csv` and `data/target.csv`.

If you want to create skeleton table CSVs from those metadata files:

```bash
python config.py
```

This reads the metadata rows and creates one CSV file per table in `data/source/` and `data/target/`.

### 2. Generate base table structure files

This is often the next step when the raw data has not yet been loaded:

```bash
python generate_tables.py
```

This script loops through metadata and writes empty or stub CSV files to the configured data folders.

### 3. Train the mapping model

To train the ranking model:

```bash
python src/train.py --config configs/config.yaml
```

This process:

- loads the configured source and target table metadata
- profiles source and target columns
- builds positive and negative feature examples
- trains a voting ensemble model using scikit-learn components
- saves the trained bundle to the configured output path

The project uses a feature-driven model trained on:

- fuzzy name similarity
- normalized name matches
- dtype compatibility
- null and cardinality gaps
- top-value overlap
- value-pattern similarity
- semantic and pattern scores

### 4. Generate mapping suggestions

After the model is trained, run:

```bash
python src/predict.py --config configs/config.yaml
```

This script:

- loads the trained model bundle
- enumerates candidate column pairs across all configured tables
- computes model and fuzzy scores
- combines them into a `combined_score`
- selects the best candidate for each source column
- writes `outputs/mapping_suggestions.csv` and `outputs/mapping_suggestions.json`

### 5. Review the mappings in the dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

The dashboard allows you to:

- filter by source/target table
- inspect top candidate mappings
- search for matching columns by name
- review best-score metrics
- compare mapping confidence against threshold values

### 6. Produce the ETL specification

Once a mapping set is considered validated, generate the ETL mapping sheet:

```bash
python generate_mappings.py
```

This reads the mapping file and creates a table-level ETL specification CSV such as `outputs/etl_spec.csv`.

The ETL spec includes fields like:

- `Legacy_Table`
- `Legacy_Row_Count`
- `Legacy_Column_Count`
- `LifePRO_Table`
- `LifePRO_Row_Count`
- `LifePRO_Column_Count`
- `Relationship`
- `Join_Keys`
- `Typical_Field_Map`

This is useful as a starting point for migration design documents and handoff to ETL developers.

## Output files

When the pipeline runs successfully, the following outputs are typically created:

```text
outputs/
├─ mapping_suggestions.csv      # ranked per-column mapping recommendations
├─ mapping_suggestions.json      # same data in JSON form
├─ verified_mappings.csv         # curated or approved mapping output used downstream
├─ etl_spec.csv                  # table-to-table ETL mapping summary
└─ ...
```

The project also writes model artifacts such as:

- `models/` containing serialized model bundles
- `catboost_info/` containing training diagnostics if CatBoost is being used in the training stack
- `logs/` containing operational logs

## Configuration reference

### `configs/config.yaml`

This is the main control file for training and prediction. It defines:

- `ddl_path`: path to DDL schema reference
- `source.root`: root directory for legacy data
- `source.files`: mapping of logical names to CSV files
- `target.root`: root directory for target data
- `target.files`: mapping of logical names to target CSV files
- `table_pairs`: table pairs to compare when building domain mappings
- `train`: training hyperparameters such as random state and test split
- `predict`: prediction controls such as `top_k`, `model_in`, and output files

### `configs/synonyms.json`

This file can override weak or ambiguous predictions with human-validated mapping facts. It is especially useful when:

- naming differs significantly across systems
- a source table has a legacy name that does not match the target name
- a domain team already knows the correct mappings

## Practical notes

### Data quality matters

This project is built around heuristics and model scoring. For best results:

- standardize column naming conventions
- ensure source and target table names are consistent
- use representative CSV samples for training
- keep `configs/synonyms.json` updated with verified mappings

### Model behavior

The mapping model is a soft-voting ensemble that combines multiple classifiers and a feature-rich representation of each candidate pair. It is designed to rank likely matches rather than guarantee perfect correctness. Human review remains important for:

- ambiguous many-to-one matches
- renamed business concepts
- tables with partial semantic overlap

### Output validation

Before moving into ETL implementation, review:

- the top-scoring mapping for each source column
- the alternate candidates returned in JSON output
- any mismatches flagged by low confidence or unusual score drift

## Example command flow

A typical end-to-end sequence looks like this:

```bash
python config.py
python generate_tables.py
python src/train.py --config configs/config.yaml
python src/predict.py --config configs/config.yaml
python generate_mappings.py
streamlit run dashboard.py
```

This workflow is ideal for validating migration assumptions before expensive ETL development begins.

## Extending the project

You can extend the project in several ways:

- add more data-profile features in `src/features.py`
- add alternate model families in `src/train.py`
- improve fuzzy matching or semantic name normalization in `src/utils/text_sim.py`
- expand the DDL parsing logic in `src/utils/io_utils.py`
- add domain-specific synonyms or business rules in `configs/synonyms.json`
- add additional dashboards or reporting metrics in `dashboard.py`

## Troubleshooting

### Model says no training samples generated

Common causes:

- the configured source/target table path is wrong
- the CSV files are empty or missing
- `configs/config.yaml` points to nonexistent names
- the metadata files do not include the expected columns

### Mapping outputs look poor

Check:

- whether `configs/synonyms.json` is populated and valid
- whether source and target data sample CSVs are aligned with the logical table names
- if the schema naming conventions are consistent across the source and target systems

### Dashboard fails to load

Ensure:

- `outputs/verified_mappings.csv` exists
- the file contains the expected columns like `source_table`, `source_column`, and `target_table`
- the YAML file in `configs/config.yaml` still points to valid source and target roots

## License

This project does not currently declare a license in the repository. Add a license file if you intend to distribute it publicly or as an internal artifact.

## Project status

This repository is intended as a working data-migration support tool for model-assisted mapping and ETL planning. It is best used as a starting point for a domain-specific migration workflow rather than as a fully generic product framework.

## Contributors and usage

Use this project to:

- accelerate mapping discovery across large legacy schemas
- reduce spreadsheet-driven manual mapping effort
- produce a first-pass ETL design based on measurable data features
- support data profiling and migration validation before coding ETL logic

If you are working in a regulated or enterprise environment, validate the generated mappings with domain owners before using them as final production logic.
