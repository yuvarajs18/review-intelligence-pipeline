"""Topics: assign daily negative reviews to the pinned, already-fitted
BERTopic model via .transform() only - this stage never fits or refits a
model. See docs/decisions.md for why refitting daily would break topic-id
stability, and scripts/fit_topics.py for the one-off fit step.

Reads pipeline/embed.py's output (data/embedded/), which already carries
precomputed embeddings, so the embedding model is never loaded here.
"""

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from pipeline.features import build_text_column
from pipeline.runs import log_run


def load_active_topic_model(config: dict) -> tuple[object, dict]:
    topics_config = config["topics"]
    active_sha = topics_config["active_model_sha"]
    if not active_sha:
        raise ValueError(
            "config.topics.active_model_sha is not set. Fit a model with "
            "scripts/fit_topics.py, then pin its sha256 in config/pipeline.yaml."
        )
    registry_path = Path(topics_config["registry_path"])
    registry = json.loads(registry_path.read_text())
    matches = [entry for entry in registry if entry["sha256"] == active_sha]
    if not matches:
        raise ValueError(f"No registry entry with sha256={active_sha!r} in {registry_path}")
    entry = matches[-1]
    with open(entry["artifact_path"], "rb") as f:
        topic_model = pickle.load(f)
    return topic_model, entry


def assign_topics(df: pd.DataFrame, topic_model, model_sha256: str) -> pd.DataFrame:
    """Pure w.r.t. I/O: transform-only, never refits. Takes precomputed
    embeddings already attached to df in an `embedding` column."""
    texts = build_text_column(df).tolist()
    embeddings = np.vstack(df["embedding"].to_numpy())
    topic_ids, _ = topic_model.transform(texts, embeddings=embeddings)
    out = df.drop(columns=["embedding"]).copy()
    out["topic_id"] = topic_ids
    out["topic_model_version"] = model_sha256
    return out


def write_partitions(df: pd.DataFrame, topics_dir: Path) -> list[str]:
    """Overwrite-by-date-partition, same idiom as pipeline/ingest.py."""
    dates_written = []
    for date, group in df.groupby(df["timestamp"].dt.date.astype(str)):
        partition_dir = topics_dir / f"date={date}"
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(partition_dir / "part.parquet", index=False)
        dates_written.append(date)
    return sorted(dates_written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    embedded_dir = Path(config["paths"]["embedded"])
    topics_dir = Path(config["paths"]["topics"])
    warehouse_path = Path(config["paths"]["warehouse"])

    con = duckdb.connect()
    embedded = con.execute(f"select * from read_parquet('{embedded_dir}/*/*.parquet')").fetchdf()

    topic_model, entry = load_active_topic_model(config)
    t0 = time.perf_counter()
    assigned = assign_topics(embedded, topic_model, entry["sha256"])
    latency = time.perf_counter() - t0

    dates_written = write_partitions(assigned, topics_dir)

    topic_counts = {int(k): int(v) for k, v in assigned["topic_id"].value_counts().items()}
    log_run(
        warehouse_path,
        stage="topics",
        rows_in=len(embedded),
        latency_seconds=latency,
        model_version=entry["sha256"],
        score_distribution={"topic_counts": topic_counts},
    )

    print(f"rows assigned: {len(assigned)}")
    print(f"partitions written: {dates_written}")
    print(f"topic distribution: {topic_counts}")


if __name__ == "__main__":
    main()
