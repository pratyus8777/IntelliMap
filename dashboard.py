import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
import yaml
from collections import defaultdict


CONFIG_PATH = Path("configs/config.yaml")
MAPPING_CSV = Path("outputs/verified_mappings.csv")
ETL_SPEC_CSV = Path("outputs/etl_spec.csv")


if not CONFIG_PATH.exists():
    st.error("config.yaml not found! Please create a config.yaml with system file mappings.")
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


if not MAPPING_CSV.exists():
    st.error(f"{MAPPING_CSV} not found! Run your ML mapping pipeline first.")
    st.stop()

df = pd.read_csv(MAPPING_CSV)


canonical_map = {
    "source_table": ["source_table", "source_table_raw", "src_table", "sourceTable"],
    "source_column": ["source_column", "source_column_raw", "src_column", "sourceColumn"],
    "target_table": ["target_table", "target_table_raw", "tgt_table", "targetTable"],
    "predicted_target_column": ["predicted_target_column", "target_column", "target_column_raw", "predictedTargetColumn", "prediction"],
    "ml_score": ["ml_score", "mlScore"],
    "fuzzy_name_score": ["fuzzy_name_score", "fuzzyScore", "fuzzy_name", "fuzzy_score"],
    "combined_match_score": ["combined_match_score", "combinedScore", "score", "combined_score"],
    "source_system": ["source_system"],
    "target_system": ["target_system"]
}

rename_dict = {}
for canon, alts in canonical_map.items():
    for alt in alts:
        if alt in df.columns:
            rename_dict[alt] = canon
            break
df = df.rename(columns=rename_dict)

# Fill missing system names
df["source_system"] = df.get("source_system", "Legacy").fillna("Legacy")
df["target_system"] = df.get("target_system", "LifePro").fillna("LifePro")


def pick_best(row):
    scores = {
        "ML Score": row.get("ml_score", np.nan),
        "Fuzzy Score": row.get("fuzzy_name_score", np.nan),
        "Combined Score": row.get("combined_match_score", np.nan),
    }
    best_type = max(scores, key=lambda k: scores[k] if pd.notna(scores[k]) else -1)
    return pd.Series({"best_score_val": scores[best_type], "best_score_type": best_type})

df = df.drop(columns=[c for c in df.columns if c in ("best_score_val", "best_score_type", "best_score")], errors="ignore")
best_scores = df.apply(pick_best, axis=1)
df = pd.concat([df.reset_index(drop=True), best_scores.reset_index(drop=True)], axis=1)


st.set_page_config(page_title="IntelliMap Dashboard", layout="wide")
st.title("IntelliMap Dashboard")
st.caption(f"{df['source_system'].iloc[0]} → {df['target_system'].iloc[0]} | Best-score based mapping view")


thresh = st.sidebar.slider("Confidence threshold (best_score ≥ threshold)", 0.0, 1.0, 0.5, 0.01)


tables_source = sorted(df["source_table"].dropna().unique())
tables_target = sorted(df["target_table"].dropna().unique())
src_filter = st.sidebar.multiselect("Filter by source table(s)", tables_source, default=tables_source)
tgt_filter = st.sidebar.multiselect("Filter by target table(s)", tables_target, default=tables_target)
df_filtered = df[df["source_table"].isin(src_filter) & df["target_table"].isin(tgt_filter)].copy()

df_filtered["is_matched"] = df_filtered["best_score_val"] >= thresh

st.subheader("Column Mapping Explorer")
search_col = st.sidebar.text_input("🔍 Search Column (Source/Target):")

with st.expander("🔍 Column Mapping Explorer Results", expanded=False):
    if search_col:
        s_col_exists = "source_column" in df.columns
        p_col_exists = "predicted_target_column" in df.columns

        if not s_col_exists and not p_col_exists:
            st.error("CSV missing both 'source_column' and 'predicted_target_column' columns.")
        else:
            mask = pd.Series(False, index=df.index)
            if s_col_exists:
                mask |= df["source_column"].astype(str).str.contains(search_col, case=False, na=False)
            if p_col_exists:
                mask |= df["predicted_target_column"].astype(str).str.contains(search_col, case=False, na=False)

            bi_matches = df[mask].copy()
            bi_matches = bi_matches.sort_values("source_column") if "source_column" in bi_matches.columns else bi_matches

            if not bi_matches.empty:
                st.write("🔍 Found the following matches:")
                cols_to_show = [
                    "source_system", "source_table", "source_column",
                    "target_system", "target_table", "predicted_target_column",
                    "ml_score", "fuzzy_name_score", "combined_match_score", "best_score_val", "best_score_type"
                ]
                available_cols = [c for c in cols_to_show if c in bi_matches.columns]
                st.dataframe(bi_matches[available_cols], use_container_width=True)

                choices = sorted(
                    set(
                        bi_matches["source_column"].dropna().astype(str).unique().tolist() +
                        bi_matches["predicted_target_column"].dropna().astype(str).unique().tolist()
                    )
                )
                selected_col = st.selectbox("Select a column to view details:", choices)


            table_info = []
            if selected_col in bi_matches.get("source_column", pd.Series([], dtype=object)).values:
                source_table = bi_matches.loc[bi_matches["source_column"] == selected_col, "source_table"].iloc[0]
                mapped_target_col = bi_matches.loc[bi_matches["source_column"] == selected_col, "predicted_target_column"].iloc[0]
                target_table = bi_matches.loc[bi_matches["source_column"] == selected_col, "target_table"].iloc[0]
                src_system = bi_matches.loc[bi_matches["source_column"] == selected_col, "source_system"].iloc[0] if "source_system" in bi_matches.columns else "guidewire"
                tgt_system = bi_matches.loc[bi_matches["source_column"] == selected_col, "target_system"].iloc[0] if "target_system" in bi_matches.columns else "insurenow"
                table_info.append(("Source", source_table, selected_col, src_system))
                table_info.append(("Mapped Target", target_table, mapped_target_col, tgt_system))
            elif selected_col in bi_matches.get("predicted_target_column", pd.Series([], dtype=object)).values:
                target_table = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "target_table"].iloc[0]
                mapped_source_col = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "source_column"].iloc[0]
                source_table = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "source_table"].iloc[0]
                src_system = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "source_system"].iloc[0] if "source_system" in bi_matches.columns else "guidewire"
                tgt_system = bi_matches.loc[bi_matches["predicted_target_column"] == selected_col, "target_system"].iloc[0] if "target_system" in bi_matches.columns else "insurenow"
                table_info.append(("Target", target_table, selected_col, tgt_system))
                table_info.append(("Mapped Source", source_table, mapped_source_col, src_system))

            def resolve_table_path(table_name, system):
                sys_key = "source" if system.lower() in ["legacy","source","guidewire"] else "target"
                root = Path(config[sys_key]["root"])
                files_map = config[sys_key]["files"]
                return root / files_map[table_name] if table_name in files_map else None

            for label, tbl_name, col_name, system in table_info:
                tbl_path = resolve_table_path(tbl_name, system)
                if tbl_path and tbl_path.exists():
                    try:
                        tbl_df = pd.read_csv(tbl_path)
                        if col_name in tbl_df.columns:
                            st.markdown(f"**{label} Column: `{col_name}` in table `{tbl_name}` ({system})**")
                            st.write(f"- Data Type: {tbl_df[col_name].dtype}")
                            st.write(f"- Total Rows: {len(tbl_df)}")
                            sample_vals = tbl_df[col_name].dropna().unique()[:10]
                            st.dataframe(pd.DataFrame({col_name: sample_vals}))
                        else:
                            st.warning(f"Column '{col_name}' not found in {tbl_path.name}.")
                    except Exception as e:
                        st.warning(f"Could not load {tbl_path.name}: {e}")
                else:
                    st.warning(f"No CSV found for table '{tbl_name}' in {system}. Check config.yaml.")
    else:
        st.error("No matches found for your search.")


tables_source = sorted(df["source_table"].dropna().unique())
tables_target = sorted(df["target_table"].dropna().unique())


total_src_cols = len(df_filtered)
matched = int(df_filtered["is_matched"].sum())
not_matched = total_src_cols - matched
match_pct = (matched / total_src_cols * 100.0) if total_src_cols else 0.0
num_source_tables = df_filtered["source_table"].nunique()
num_target_tables = df_filtered["target_table"].nunique()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Source Tables", num_source_tables)
c2.metric("Target Tables", num_target_tables)
c3.metric("Total Source Columns (filtered)", total_src_cols)
c4.metric("Matched (score ≥ threshold)", matched)
c5.metric("Not Matched", not_matched)
c6.metric("% Matched", f"{match_pct:.1f}%")  


col1, spacer1, col2 = st.columns([1, 0.1, 1])

with col1:
    st.subheader("Matched vs Not Matched")
    fig, ax = plt.subplots(figsize=(5,5), facecolor="none")
    ax.set_facecolor("none")
    
    wedges, texts, autotexts = ax.pie(
        [matched, not_matched],
        labels=["Matched", "Not Matched"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={'fontsize': 14}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_color('white')
        text.set_fontsize(14)
        text.set_fontweight('bold')
    
    ax.axis("equal")
    
    st.pyplot(fig, transparent=True)  


import plotly.express as px

with col2:
    st.subheader("Match Rate by Source Table")
    
    by_src = df_filtered.groupby("source_table").agg(
        total=("source_column", "count"),
        matched=("is_matched", "sum")
    ).reset_index()
    by_src["match_rate"] = np.where(by_src["total"] > 0, by_src["matched"] / by_src["total"] * 100, 0)

    hover_texts = []
    for _, row in by_src.iterrows():
        table_name = row["source_table"]
        src_path = Path(config["source"]["root"]) / config["source"]["files"].get(table_name, "")
        if src_path.exists():
            try:
                df_src = pd.read_csv(src_path)
                sample_vals = df_src.head(5).to_dict()
                hover_text = (
                    f"Table: {table_name}<br>"
                    f"Rows: {len(df_src)}<br>"
                    f"Columns: {len(df_src.columns)}<br>"
                    f"Sample:<br>{sample_vals}"
                )
            except:
                hover_text = f"Table: {table_name}<br>Error reading file"
        else:
            hover_text = f"Table: {table_name}<br>File not found"
        hover_texts.append(hover_text)
    
    fig = px.bar(
        by_src,
        x="source_table",
        y="match_rate",
        text=by_src["match_rate"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"hover_info": hover_texts},
        labels={"match_rate": "Match Rate (%)", "source_table": "Source Table"},
        height=500
    )

    fig.update_traces(
        marker_color="#66CCFF",
        marker_line_color='black',
        marker_line_width=1.5,
        textposition='outside',
        hovertemplate='%{customdata[0]}<extra></extra>'
    )

    fig.update_layout(
        yaxis=dict(range=[0, 110]),
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("Low-Confidence Mappings")
if "predicted_target_column" in df_filtered.columns:
    low_conf = df_filtered[~df_filtered["is_matched"] | df_filtered["predicted_target_column"].isna()]
else:
    low_conf = df_filtered[~df_filtered["is_matched"]]
if "best_score_val" in low_conf.columns:
    low_conf["best_score_val"] = low_conf["best_score_val"] * 100  # percentage
st.dataframe(low_conf, use_container_width=True)


st.subheader("Top Matches")
top_matches = df_filtered[df_filtered["is_matched"]].sort_values("best_score_val", ascending=False)
if "best_score_val" in top_matches.columns:
    top_matches["best_score_val"] = top_matches["best_score_val"] * 100  # percentage
st.dataframe(top_matches, use_container_width=True)


st.subheader("Generated Spec")
def resolve_table_path(table_name, system):
    sys_key = "source" if system.lower() in ["legacy","source","guidewire"] else "target"
    root = Path(config[sys_key]["root"])
    files_map = config[sys_key]["files"]
    return root / files_map[table_name] if table_name in files_map else None

def generate_etl(df):
    table_groups = defaultdict(list)
    for _, row in df.iterrows():
        tgt_col = row.get("predicted_target_column", row.get("target_column"))
        table_groups[(row['source_table'], row['target_table'])].append((row['source_column'], tgt_col))
    etl_rows = []
    for (src_tbl, tgt_tbl), cols in table_groups.items():
        src_path = resolve_table_path(src_tbl, "source")
        tgt_path = resolve_table_path(tgt_tbl, "target")
        try: src_df = pd.read_csv(src_path); src_rows, src_cols = src_df.shape
        except: src_rows, src_cols = None, None
        try: tgt_df = pd.read_csv(tgt_path); tgt_rows, tgt_cols = tgt_df.shape
        except: tgt_rows, tgt_cols = None, None
        relationship = "1:1" if src_rows==tgt_rows else "1:1/1:M (depends on data)" if src_rows and tgt_rows else ""
        join_keys_src = [s for s,t in cols if "_id" in s.lower()]
        join_keys_tgt = [t for s,t in cols if t and "_id" in t.lower()]
        join_keys = " / ".join(join_keys_src)+" → "+" / ".join(join_keys_tgt) if join_keys_src else ""
        field_map = ", ".join([f"{s} → {t}" for s,t in cols if t])
        etl_rows.append({
            "Legacy_Table": src_tbl,
            "Legacy_Row_Count": src_rows,
            "Legacy_Column_Count": src_cols,
            "LifePRO_Table": tgt_tbl,
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
    return pd.DataFrame(etl_rows)

etl_df = generate_etl(df_filtered)
st.dataframe(etl_df, use_container_width=True)
buf_etl = io.BytesIO()
etl_df.to_csv(buf_etl, index=False)
st.download_button("Download ETL Spec CSV", data=buf_etl.getvalue(), file_name="etl_spec.csv", mime="text/csv")


st.subheader("All Matches (Low & High Confidence)")

full_display = df.copy()

if "best_score_val" in full_display.columns:
    full_display["best_score_pct"] = (full_display["best_score_val"] * 100).round(2).astype(str) + "%"
else:
    full_display["best_score_pct"] = "0%"

threshold = 70  
full_display["match_confirmed"] = full_display["best_score_val"].apply(
    lambda x: "✅" if pd.notna(x) and x * 100 >= threshold else "⬜"
)

display_cols = [
    "match_confirmed",
    "source_system", "source_table", "source_column",
    "target_system", "target_table", "predicted_target_column",
    "best_score_pct"
]
full_display = full_display[[c for c in display_cols if c in full_display.columns]]


def highlight_row(row):
    if "best_score_val" in row and pd.notna(row["best_score_val"]) and row["best_score_val"] * 100 < threshold:
        return ["background-color: #FF9999"] * len(row)  
    return [""] * len(row)

def highlight_best_score(val):
    return "background-color: #66CC66; color: white; font-weight: bold; text-align: center;"

styled_full = (
    full_display.style
    .apply(highlight_row, axis=1)
    .applymap(highlight_best_score, subset=["best_score_pct"])
)



st.dataframe(styled_full, use_container_width=True, height=500)


buf = io.BytesIO()
full_display.to_csv(buf, index=False)
st.download_button(
    "Download Full Matched Table CSV",
    data=buf.getvalue(),
    file_name="full_matched_table.csv",
    mime="text/csv"
)
