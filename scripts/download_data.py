"""One-off local download of the raw review category used by the pipeline.

Not wired into `make ingest`: this hits the network once and caches the
result under paths.raw_local (gitignored). Re-run manually if you want to
switch category or refresh the cache.

Downloads the plain .jsonl file directly via huggingface_hub instead of
`datasets.load_dataset(...)`, because the dataset repo still ships a legacy
Python loading script that current `datasets` releases refuse to execute
("Dataset scripts are no longer supported"). See docs/decisions.md.
"""

import argparse
import shutil
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


def download_category(hf_repo: str, hf_path: str, dest_dir: Path) -> Path:
    cached_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(hf_path).name
    shutil.copyfile(cached_path, dest_path)
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dest_dir = Path(config["paths"]["raw_local"])
    dest_path = download_category(
        hf_repo=config["dataset"]["hf_repo"],
        hf_path=config["dataset"]["hf_path"],
        dest_dir=dest_dir,
    )
    print(f"Downloaded to {dest_path}")


if __name__ == "__main__":
    main()
