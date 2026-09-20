"""Embed: sentence embeddings (all-MiniLM-L6-v2) for negative reviews only.

Negative = rating <= 2, the reviewer's own star rating, not a model
prediction - the same proxy-label definition used everywhere else in this
project (see pipeline/features.make_binary_labels). This is available for
both historical training data and daily scored data alike, since a reviewer
submits the rating and the text together; there's nothing to infer here.

Computed once here and reused by both scripts/fit_topics.py (fit) and
pipeline/topics.py (transform), so the embedding model loads and runs once
per pipeline invocation rather than being duplicated across two stages.

Loads with local_files_only=True after HF_HUB_OFFLINE is set, same offline
guarantee as pipeline/score.py's DistilBERT loading.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from pipeline.features import build_text_column
from pipeline.runs import log_run


def filter_negative(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["rating"] <= 2].copy()


def build_embedding_model(config: dict):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    embedding_config = config["model"]["embedding"]
    return SentenceTransformer(
        embedding_config["name"],
        revision=embedding_config["revision"],
        cache_folder=embedding_config["cache_dir"],
        local_files_only=True,
        device="cpu",
    )


def embed_reviews(df: pd.DataFrame, model) -> pd.DataFrame:
    """Pure w.r.t. I/O: takes an already-loaded model, returns a new frame
    with an `embedding` column (one list[float32] per row)."""
    texts = build_text_column(df).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    out = df.copy()
    out["embedding"] = list(embeddings)
    return out


def write_partitions(df: pd.DataFrame, embedded_dir: Path) -> list[str]:
    """Overwrite-by-date-partition, same idiom as pipeline/ingest.py."""
    dates_written = []
    for date, group in df.groupby(df["timestamp"].dt.date.astype(str)):
        partition_dir = embedded_dir / f"date={date}"
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

    scored_dir = Path(config["paths"]["scored"])
    embedded_dir = Path(config["paths"]["embedded"])
    warehouse_path = Path(config["paths"]["warehouse"])

    con = duckdb.connect()
    scored = con.execute(f"select * from read_parquet('{scored_dir}/*/*.parquet')").fetchdf()
    negative = filter_negative(scored)

    model = build_embedding_model(config)
    t0 = time.perf_counter()
    embedded = embed_reviews(negative, model)
    latency = time.perf_counter() - t0

    dates_written = write_partitions(embedded, embedded_dir)

    log_run(
        warehouse_path,
        stage="embed",
        rows_in=len(negative),
        latency_seconds=latency,
        model_name=config["model"]["embedding"]["name"],
        model_version=config["model"]["embedding"]["revision"],
    )

    print(f"rows in (scored): {len(scored)}")
    print(f"negative rows embedded: {len(embedded)}")
    print(f"partitions written: {dates_written}")


if __name__ == "__main__":
    main()
