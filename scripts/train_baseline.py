"""Train the TF-IDF + logistic regression sentiment baseline.

Trains on the historical slice strictly before the ingest window (rows with
timestamp < window_start), so training data never overlaps with what
pipeline/score.py scores - the model can't have "seen" what it's later
asked to predict on, mirroring a real production timeline.

Star ratings are a proxy label: 1-2 stars -> negative, 4-5 stars -> positive,
3 stars dropped (see pipeline/features.make_binary_labels).

Not wired into a Makefile target - like scripts/download_data.py, this is a
deliberate one-off step, not a daily pipeline stage. Run it, inspect the
printed metrics and sha256, then decide whether to pin
config.model.active_model_sha to the new version.
"""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.features import build_text_column, make_binary_labels
from pipeline.ingest import load_config, load_raw_reviews, resolve_window


def build_pipeline(vectorizer_config: dict, classifier_config: dict) -> Pipeline:
    vectorizer_config = {**vectorizer_config, "ngram_range": tuple(vectorizer_config["ngram_range"])}
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**vectorizer_config)),
            ("clf", LogisticRegression(**classifier_config)),
        ]
    )


def select_training_rows(df, window_start):
    """Rows strictly before the ingest window, with a usable binary label."""
    train_df = df.loc[df["timestamp"] < window_start].copy()
    train_df["label"] = make_binary_labels(train_df["rating"])
    return train_df.dropna(subset=["label"])


def compute_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def serialize_and_hash(fitted_pipeline: Pipeline) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    joblib.dump(fitted_pipeline, buffer)
    artifact_bytes = buffer.getvalue()
    sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    return artifact_bytes, sha256


def append_registry_entry(registry_path: Path, entry: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    raw_path = Path(config["paths"]["raw_local"]) / Path(config["dataset"]["hf_path"]).name
    model_config = config["model"]["baseline"]

    reference_date = pd.Timestamp(config["window"]["reference_date"])
    window_start, _ = resolve_window(reference_date, config["window"]["lookback_days"])

    df = load_raw_reviews(raw_path)
    train_df = select_training_rows(df, window_start)

    X_text = build_text_column(train_df)
    y = train_df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=model_config["test_size"],
        stratify=y,
        random_state=model_config["classifier"]["random_state"],
    )

    pipeline = build_pipeline(model_config["vectorizer"], model_config["classifier"])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    artifact_bytes, sha256 = serialize_and_hash(pipeline)
    model_path = Path("models") / f"{model_config['name']}-{sha256[:12]}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(artifact_bytes)

    entry = {
        "name": model_config["name"],
        "sha256": sha256,
        "artifact_path": str(model_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_balance": {"positive": int(y.sum()), "negative": int((y == 0).sum())},
        "metrics": metrics,
        "vectorizer_config": model_config["vectorizer"],
        "classifier_config": model_config["classifier"],
    }
    append_registry_entry(Path(model_config["registry_path"]), entry)

    print(f"trained on {len(X_train)} rows, evaluated on {len(X_test)} held-out rows")
    print(f"metrics: {metrics}")
    print(f"sha256: {sha256}")
    print(f"artifact: {model_path}")
    print("to use this model, set model.baseline.active_model_sha in config/pipeline.yaml")


if __name__ == "__main__":
    main()
