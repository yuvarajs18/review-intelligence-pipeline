"""Review Intelligence Pipeline - read-only dashboard.

Reads only from data/warehouse.db (SQLite) and the pipeline's Parquet
layers - never writes anything back, and never fabricates a number: every
panel either shows real data or a clear "not available yet" message when
the upstream stage that would produce it hasn't run (or, for the alerts
panel, hasn't been implemented yet - that's pipeline/detect.py, the
author's own unfinished work per the working agreement).

Panels:
- Topic trends over time (data/topics/, labeled via config/topic_names.yaml)
- Complaint themes by product (same data, grouped by product)
- Flagged products (an `alerts` table - gracefully degrades if it doesn't
  exist, since pipeline/detect.py's alerting logic isn't implemented yet)
- Model comparison (models/registry.json + models/topics_registry.json -
  the same metrics scripts/train_baseline.py and scripts/fit_topics.py
  already produced, not a separately invented file)
"""

import json
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import yaml

st.set_page_config(page_title="Review Intelligence Pipeline", layout="wide")

CONFIG = yaml.safe_load(Path("config/pipeline.yaml").read_text())
WAREHOUSE_PATH = Path(CONFIG["paths"]["warehouse"])
TOPICS_DIR = Path(CONFIG["paths"]["topics"])
TOPIC_NAMES_PATH = Path("config/topic_names.yaml")
BASELINE_REGISTRY_PATH = Path(CONFIG["model"]["baseline"]["registry_path"])
TOPICS_REGISTRY_PATH = Path(CONFIG["topics"]["registry_path"])


@st.cache_data(ttl=300)
def load_topic_names() -> dict:
    if not TOPIC_NAMES_PATH.exists():
        return {}
    raw = yaml.safe_load(TOPIC_NAMES_PATH.read_text()) or {}
    return {int(k): v for k, v in raw.items()}


@st.cache_data(ttl=300)
def load_topics_data() -> pd.DataFrame:
    if not any(TOPICS_DIR.glob("*/*.parquet")):
        return pd.DataFrame()
    con = duckdb.connect()
    return con.execute(f"select * from read_parquet('{TOPICS_DIR}/*/*.parquet')").fetchdf()


@st.cache_data(ttl=300)
def load_sqlite_table(table: str) -> pd.DataFrame:
    if not WAREHOUSE_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(WAREHOUSE_PATH)
    existing = pd.read_sql_query("select name from sqlite_master where type='table'", conn)
    if table not in existing["name"].values:
        conn.close()
        return pd.DataFrame()
    df = pd.read_sql_query(f"select * from {table}", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_registry(path: str) -> list:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text())


st.title("Review Intelligence Pipeline")
st.caption("Read-only view over data/warehouse.db and the pipeline's Parquet layers.")

tab_topics, tab_products, tab_alerts, tab_models = st.tabs(
    ["Topic trends", "Complaint themes by product", "Flagged products", "Model comparison"]
)

with tab_topics:
    st.subheader("Negative-review topic volume over time")
    topics_df = load_topics_data()
    if topics_df.empty:
        st.info("No data in data/topics/ yet - run `make embed && make topics` first.")
    else:
        topic_names = load_topic_names()
        topics_df = topics_df.copy()
        topics_df["topic_label"] = topics_df["topic_id"].map(lambda t: topic_names.get(t, f"topic_{t}"))
        topics_df["week"] = topics_df["timestamp"].dt.to_period("W").dt.start_time
        weekly = (
            topics_df.groupby(["week", "topic_label"])
            .size()
            .reset_index(name="review_count")
            .pivot(index="week", columns="topic_label", values="review_count")
            .fillna(0)
        )
        st.line_chart(weekly)
        st.caption(
            f"{len(topics_df)} negative reviews across {topics_df['topic_id'].nunique()} "
            "distinct topics (including outliers, topic -1)."
        )

with tab_products:
    st.subheader("Top complaint themes per product")
    topics_df = load_topics_data()
    if topics_df.empty:
        st.info("No data in data/topics/ yet - run `make embed && make topics` first.")
    else:
        topic_names = load_topic_names()
        real_themes = topics_df[topics_df["topic_id"] != -1]  # -1 is BERTopic's outlier bucket, not a theme
        if real_themes.empty:
            st.info("Every negative review currently falls into the outlier bucket (topic -1) - no themes to show yet.")
        else:
            product_topic_counts = (
                real_themes.groupby(["parent_asin", "topic_id"]).size().reset_index(name="count")
            )
            product_topic_counts["topic_label"] = product_topic_counts["topic_id"].map(
                lambda t: topic_names.get(t, f"topic_{t}")
            )
            top_per_product = (
                product_topic_counts.sort_values("count", ascending=False)
                .groupby("parent_asin")
                .head(3)
                .sort_values(["parent_asin", "count"], ascending=[True, False])
            )
            selected_product = st.selectbox("Product (parent_asin)", sorted(real_themes["parent_asin"].unique()))
            st.dataframe(
                top_per_product.loc[top_per_product["parent_asin"] == selected_product, ["topic_label", "count"]],
                hide_index=True,
            )

with tab_alerts:
    st.subheader("Flagged products")
    alerts_df = load_sqlite_table("alerts")
    if alerts_df.empty:
        st.info(
            "No `alerts` table found in data/warehouse.db. This panel reads whatever "
            "pipeline/detect.py's implementation writes there - Stage 6 is the "
            "author's own work and isn't finished yet."
        )
    else:
        st.dataframe(alerts_df, hide_index=True)

with tab_models:
    st.subheader("Baseline sentiment model")
    baseline_entries = load_registry(str(BASELINE_REGISTRY_PATH))
    if not baseline_entries:
        st.info("No models/registry.json found yet - run scripts/train_baseline.py first.")
    else:
        baseline_rows = [
            {
                "sha256": entry["sha256"][:12],
                "trained_at": entry["trained_at"],
                "training_rows": entry["training_rows"],
                **entry["metrics"],
            }
            for entry in baseline_entries
        ]
        st.dataframe(pd.DataFrame(baseline_rows), hide_index=True)

    st.subheader("Topic model")
    topics_entries = load_registry(str(TOPICS_REGISTRY_PATH))
    if not topics_entries:
        st.info("No models/topics_registry.json found yet - run scripts/fit_topics.py first.")
    else:
        topics_rows = [
            {
                "sha256": entry["sha256"][:12],
                "trained_at": entry["trained_at"],
                "training_rows": entry["training_rows"],
                "n_topics": entry["n_topics"],
                "n_outliers": entry["n_outliers"],
            }
            for entry in topics_entries
        ]
        st.dataframe(pd.DataFrame(topics_rows), hide_index=True)
