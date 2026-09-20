"""Score: apply both sentiment models to the validated reviews, side by side.

The baseline (TF-IDF + logistic regression) runs via config.model.baseline
.active_model_sha, pinned deliberately (see docs/decisions.md). DistilBERT
runs via config.model.distilbert.revision, pinned to a specific HF git
revision for the same reason - a pretrained model can drift under a fixed
name just as easily as a self-trained one can.

Both models score every row; neither replaces the other. Both score columns
share the same semantics (P(positive review), thresholded at 0.5) so they're
directly comparable downstream, even though DistilBERT's raw pipeline output
is "confidence in whichever label it predicted" rather than P(positive) -
see attach_distilbert_scores.

DistilBERT loads with local_files_only=True after HF_HUB_OFFLINE is set, so
a scoring run never silently reaches the network - if the model isn't
already cached (via scripts/download_model.py), this fails immediately with
a clear error instead of a slow/flaky retry mid-run.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import yaml

from pipeline.features import build_text_column
from pipeline.runs import log_run


def load_active_baseline(config: dict) -> tuple[object, dict]:
    baseline_config = config["model"]["baseline"]
    active_sha = baseline_config["active_model_sha"]
    if not active_sha:
        raise ValueError(
            "config.model.baseline.active_model_sha is not set. Train a model with "
            "scripts/train_baseline.py, then pin its sha256 in config/pipeline.yaml."
        )
    registry_path = Path(baseline_config["registry_path"])
    registry = json.loads(registry_path.read_text())
    matches = [entry for entry in registry if entry["sha256"] == active_sha]
    if not matches:
        raise ValueError(f"No registry entry with sha256={active_sha!r} in {registry_path}")
    entry = matches[-1]
    model = joblib.load(entry["artifact_path"])
    return model, entry


def build_distilbert_pipeline(config: dict):
    """Loads the pinned revision from the local cache only - never the network."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    distilbert_config = config["model"]["distilbert"]
    common_kwargs = dict(
        pretrained_model_name_or_path=distilbert_config["name"],
        revision=distilbert_config["revision"],
        cache_dir=distilbert_config["cache_dir"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(**common_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(**common_kwargs)
    return hf_pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer, device="cpu")


def score_with_baseline(df: pd.DataFrame, model, model_name: str, model_version: str) -> pd.DataFrame:
    """Pure w.r.t. I/O: takes an already-loaded model, returns a new frame
    with baseline sentiment + model-identity columns attached."""
    text = build_text_column(df)
    positive_proba = model.predict_proba(text)[:, 1]
    out = df.copy()
    out["baseline_sentiment_score"] = positive_proba
    out["baseline_sentiment_label"] = (positive_proba >= 0.5).astype(int)
    out["baseline_model_name"] = model_name
    out["baseline_model_version"] = model_version
    return out


def attach_distilbert_scores(df: pd.DataFrame, raw_outputs: list[dict]) -> pd.DataFrame:
    """raw_outputs is the transformers pipeline's raw output, aligned by
    position with df's rows: [{"label": "POSITIVE"|"NEGATIVE", "score": float}, ...]
    where `score` is confidence in the predicted label, not P(positive).
    Converted here so both models' score columns mean the same thing."""
    positive_proba = [
        row["score"] if row["label"] == "POSITIVE" else 1 - row["score"] for row in raw_outputs
    ]
    out = df.copy()
    out["distilbert_sentiment_score"] = positive_proba
    out["distilbert_sentiment_label"] = [int(p >= 0.5) for p in positive_proba]
    return out


def write_partitions(df: pd.DataFrame, scored_dir: Path) -> list[str]:
    """Overwrite-by-date-partition, same idiom as pipeline/ingest.py."""
    dates_written = []
    for date, group in df.groupby(df["timestamp"].dt.date.astype(str)):
        partition_dir = scored_dir / f"date={date}"
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(partition_dir / "part.parquet", index=False)
        dates_written.append(date)
    return sorted(dates_written)


def _label_distribution(df: pd.DataFrame, score_col: str, label_col: str) -> dict:
    counts = df[label_col].value_counts()
    return {
        "label_counts": {int(label): int(count) for label, count in counts.items()},
        "score_mean": float(df[score_col].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    validated_dir = Path(config["paths"]["validated"])
    scored_dir = Path(config["paths"]["scored"])
    warehouse_path = Path(config["paths"]["warehouse"])

    con = duckdb.connect()
    validated = con.execute(f"select * from read_parquet('{validated_dir}/*/*.parquet')").fetchdf()
    rows_in = len(validated)

    baseline_model, baseline_entry = load_active_baseline(config)
    t0 = time.perf_counter()
    scored = score_with_baseline(validated, baseline_model, baseline_entry["name"], baseline_entry["sha256"])
    baseline_latency = time.perf_counter() - t0

    distilbert_config = config["model"]["distilbert"]
    distilbert_pipe = build_distilbert_pipeline(config)
    texts = build_text_column(validated).tolist()
    t0 = time.perf_counter()
    raw_outputs = distilbert_pipe(
        texts,
        batch_size=distilbert_config["batch_size"],
        truncation="only_first",
        max_length=distilbert_config["max_length"],
    )
    distilbert_latency = time.perf_counter() - t0
    scored = attach_distilbert_scores(scored, raw_outputs)

    dates_written = write_partitions(scored, scored_dir)

    log_run(
        warehouse_path,
        stage="score",
        rows_in=rows_in,
        latency_seconds=baseline_latency,
        model_name=baseline_entry["name"],
        model_version=baseline_entry["sha256"],
        score_distribution=_label_distribution(scored, "baseline_sentiment_score", "baseline_sentiment_label"),
    )
    log_run(
        warehouse_path,
        stage="score",
        rows_in=rows_in,
        latency_seconds=distilbert_latency,
        model_name=distilbert_config["name"],
        model_version=distilbert_config["revision"],
        score_distribution=_label_distribution(scored, "distilbert_sentiment_score", "distilbert_sentiment_label"),
    )

    print(f"rows scored: {rows_in}")
    print(f"partitions written: {dates_written}")
    print(f"baseline:   {rows_in / baseline_latency:.1f} rows/sec ({baseline_latency:.3f}s)")
    print(f"distilbert: {rows_in / distilbert_latency:.1f} rows/sec ({distilbert_latency:.3f}s)")


if __name__ == "__main__":
    main()
