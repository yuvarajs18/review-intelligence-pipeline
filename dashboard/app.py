"""Review Intelligence Pipeline - read-only dashboard.

Reads only from data/warehouse.db (SQLite) and the pipeline's Parquet
layers - never writes anything back, and never fabricates a number: every
panel either shows real data or a clear "not available yet" message when
the upstream stage that would produce it hasn't run.

Layout: an at-a-glance summary strip up top (answers "what's here" with no
clicking), then three tabs ordered by how actionable they are - Flagged
Products first (the headline output), Topics second (trend + full product
breakdown in one place, no per-product dropdown required), Models last
(registries, least urgent for a quick look).
"""

import json
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import yaml

from bootstrap import ensure_data_built

st.set_page_config(page_title="Review Intelligence Pipeline", layout="wide")

ensure_data_built()

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

# --- At a glance: answers "what's here" before any tab is clicked ---
alerts_df = load_sqlite_table("alerts")
topics_df = load_topics_data()
baseline_entries = load_registry(str(BASELINE_REGISTRY_PATH))
topics_entries = load_registry(str(TOPICS_REGISTRY_PATH))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Flagged products", len(alerts_df) if not alerts_df.empty else 0)
col2.metric("Topics found", topics_entries[-1]["n_topics"] if topics_entries else "-")
col3.metric(
    "Negative reviews analyzed",
    f"{len(topics_df):,}" if not topics_df.empty else 0,
)
col4.metric(
    "Active baseline F1",
    f"{baseline_entries[-1]['metrics']['f1']:.3f}" if baseline_entries else "-",
)
st.divider()

tab_alerts, tab_topics, tab_models = st.tabs(["Flagged Products", "Topics", "Models"])

with tab_alerts:
    st.subheader("Products flagged for degradation or drift")
    if alerts_df.empty:
        st.info(
            "No `alerts` table found in data/warehouse.db yet - run `make detect` first."
        )
    else:
        st.caption(
            f"{len(alerts_df)} alerts: "
            + ", ".join(
                f"{count} {name}"
                for name, count in alerts_df["alert_type"].value_counts().items()
            )
        )
        display_df = alerts_df[["product_id", "week_start", "alert_type", "metric_value"]].sort_values(
            "week_start", ascending=False
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)

with tab_topics:
    st.subheader("Negative-review topic trends")
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

        st.markdown("##### Top complaint theme per product")
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
            top_theme_per_product = (
                product_topic_counts.sort_values("count", ascending=False)
                .groupby("parent_asin")
                .head(1)
                .sort_values("count", ascending=False)
                .rename(columns={"parent_asin": "product_id"})[["product_id", "topic_label", "count"]]
            )
            st.caption("Every product with a complaint theme, sorted by how concentrated that theme is. Click a column header to re-sort.")
            st.dataframe(top_theme_per_product, hide_index=True, use_container_width=True)

with tab_models:
    st.subheader("Baseline sentiment model")
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
        st.dataframe(pd.DataFrame(baseline_rows), hide_index=True, use_container_width=True)

    st.subheader("Topic model")
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
        st.dataframe(pd.DataFrame(topics_rows), hide_index=True, use_container_width=True)
