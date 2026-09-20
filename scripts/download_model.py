"""One-off local download of every pinned pretrained model this pipeline
uses (DistilBERT sentiment, the MiniLM sentence embedder).

Not wired into any Makefile target: this hits the network once and caches
weights under each model's cache_dir (gitignored). The pipeline stages that
use these models then load with local_files_only=True, so a pipeline run
never touches the network - if a cache is missing or a pinned revision
changed, it fails immediately with a clear error instead of silently
retrying over the network mid-run.
"""

import argparse
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def download_classifier(name: str, revision: str, cache_dir: Path) -> None:
    AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    AutoModelForSequenceClassification.from_pretrained(name, revision=revision, cache_dir=cache_dir)


def download_embedding_model(name: str, revision: str, cache_dir: Path) -> None:
    SentenceTransformer(name, revision=revision, cache_folder=str(cache_dir), device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())["model"]

    distilbert = config["distilbert"]
    download_classifier(distilbert["name"], distilbert["revision"], Path(distilbert["cache_dir"]))
    print(f"Cached {distilbert['name']}@{distilbert['revision']} under {distilbert['cache_dir']}")

    embedding = config["embedding"]
    download_embedding_model(embedding["name"], embedding["revision"], Path(embedding["cache_dir"]))
    print(f"Cached {embedding['name']}@{embedding['revision']} under {embedding['cache_dir']}")


if __name__ == "__main__":
    main()
