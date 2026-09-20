"""Ingest: slice the locally-downloaded review category into a date-partitioned
Parquet lake.

The window/filter/partition logic is pure functions operating on DataFrames
already in memory, so it's unit-testable without touching disk. Only
`load_raw_reviews`, `write_partitions`, and `main` touch the filesystem.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_raw_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, convert_dates=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # "images" is a list<struct<...>>, but pandas/pyarrow infers list<null>
    # for any partition where every review has zero images - a schema
    # mismatch across date partitions that breaks a cross-partition Parquet
    # read. Dropped here since nothing downstream uses it. See docs/decisions.md.
    df = df.drop(columns=["images"])
    return df


def resolve_window(reference_date: pd.Timestamp, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Rolling window: [reference_date - lookback_days, reference_date)."""
    window_end = reference_date
    window_start = reference_date - pd.Timedelta(days=lookback_days)
    return window_start, window_end


def filter_window(df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    """Keep reviews with `timestamp` in [window_start, window_end); tag each with its calendar date."""
    mask = (df["timestamp"] >= window_start) & (df["timestamp"] < window_end)
    out = df.loc[mask].copy()
    out["review_date"] = out["timestamp"].dt.date.astype(str)
    return out


def write_partitions(df: pd.DataFrame, ingested_dir: Path) -> list[str]:
    """Overwrite-by-date-partition: each date's directory is replaced wholesale
    on every run that touches it, so re-running never duplicates rows."""
    dates_written = []
    for date, group in df.groupby("review_date"):
        partition_dir = ingested_dir / f"date={date}"
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["review_date"]).to_parquet(partition_dir / "part.parquet", index=False)
        dates_written.append(date)
    return sorted(dates_written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    raw_path = Path(config["paths"]["raw_local"]) / Path(config["dataset"]["hf_path"]).name
    ingested_dir = Path(config["paths"]["ingested"])

    reference_date = pd.Timestamp(config["window"]["reference_date"])
    lookback_days = config["window"]["lookback_days"]
    window_start, window_end = resolve_window(reference_date, lookback_days)

    df = load_raw_reviews(raw_path)
    windowed = filter_window(df, window_start, window_end)
    dates_written = write_partitions(windowed, ingested_dir)

    print(f"window: [{window_start.date()}, {window_end.date()})")
    print(f"rows ingested: {len(windowed)}")
    print(f"partitions written: {dates_written}")


if __name__ == "__main__":
    main()
