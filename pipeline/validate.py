"""Validate: pandera schema + English-language filter, gated between ingest
and score.

Every ingested row either passes through to data/validated/... or is
quarantined with a reason code in the SQLite `quarantine` table - never
silently dropped. `pipeline/score.py` and `pipeline/aggregate.py` both read
from the validated layer, not the raw ingested one, so quarantining a row
actually excludes it from every downstream stage.
"""

import argparse
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pandera.pandas as pa
import yaml
from langdetect import DetectorFactory, detect

from pipeline.runs import log_run

# langdetect's classifier is otherwise non-deterministic (uses random
# byte-reading for tie-breaking) - fixed here so the same input always
# quarantines the same way.
DetectorFactory.seed = 0

# Priority order when a row fails more than one pandera check: the guide
# asks for one reason code per row, so the first-matching reason here wins.
REASON_PRIORITY = [
    "missing_required_field",
    "rating_out_of_range",
    "empty_text",
    "implausible_timestamp",
]


def build_schema(timestamp_min: pd.Timestamp) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "rating": pa.Column(
                int, pa.Check.in_range(1, 5, error="rating_out_of_range"), nullable=False
            ),
            "title": pa.Column(str, nullable=False),
            "text": pa.Column(
                str,
                pa.Check(lambda s: s.str.strip().str.len() > 0, error="empty_text"),
                nullable=False,
            ),
            "asin": pa.Column(str, nullable=False),
            "parent_asin": pa.Column(str, nullable=False),
            "user_id": pa.Column(str, nullable=False),
            "timestamp": pa.Column(
                pa.DateTime,
                pa.Check(
                    lambda s: (s >= timestamp_min) & (s <= pd.Timestamp.now()),
                    error="implausible_timestamp",
                ),
                nullable=False,
            ),
            "helpful_vote": pa.Column(int, nullable=False),
            "verified_purchase": pa.Column(bool, nullable=False),
        },
        strict=True,
    )


def classify_schema_failures(df: pd.DataFrame, schema: pa.DataFrameSchema) -> dict[int, str]:
    """Runs the schema in lazy mode and returns {row_index: reason_code} for
    every row that failed at least one check. A row failing multiple checks
    gets exactly one reason, per REASON_PRIORITY. Passing rows are absent
    from the returned dict."""
    try:
        schema.validate(df, lazy=True)
        return {}
    except pa.errors.SchemaErrors as exc:
        reasons_by_index: dict[int, set[str]] = {}
        for _, failure in exc.failure_cases.iterrows():
            idx = failure["index"]
            if idx is None or (isinstance(idx, float) and pd.isna(idx)):
                continue  # schema-level failure with no single offending row
            check_name = failure["check"]
            reason = "missing_required_field" if check_name == "not_nullable" else str(check_name)
            reasons_by_index.setdefault(int(idx), set()).add(reason)

        return {
            idx: next((r for r in REASON_PRIORITY if r in reasons), sorted(reasons)[0])
            for idx, reasons in reasons_by_index.items()
        }


def is_likely_non_english(text: str, min_words: int) -> bool:
    """Below min_words, language detection is unreliable - these are treated
    as indeterminate and pass rather than risk misclassifying short but
    legitimate reviews as non-English."""
    if len(text.split()) < min_words:
        return False
    try:
        return detect(text) != "en"
    except Exception:
        # langdetect raises on text with no detectable features (e.g. only
        # punctuation/numbers) - can't prove it's non-English, so let it pass.
        return False


def split_valid_and_quarantined(
    df: pd.DataFrame, schema: pa.DataFrameSchema, min_words: int
) -> tuple[pd.DataFrame, list[dict]]:
    """Pure w.r.t. I/O: returns (valid_rows, quarantine_records). Each
    quarantine record carries a reason_code and the full original row, so a
    quarantined review is always fully inspectable, not just labeled."""
    schema_reasons = classify_schema_failures(df, schema)

    quarantine_records = []
    valid_indices = []
    for idx, row in df.iterrows():
        if idx in schema_reasons:
            reason = schema_reasons[idx]
        elif is_likely_non_english(f"{row['title']} {row['text']}".strip(), min_words):
            reason = "non_english"
        else:
            valid_indices.append(idx)
            continue
        quarantine_records.append({"reason_code": reason, "row": row.to_dict()})

    return df.loc[valid_indices], quarantine_records


def write_valid_partitions(df: pd.DataFrame, validated_dir: Path) -> list[str]:
    """Overwrite-by-date-partition, same idiom as pipeline/ingest.py."""
    dates_written = []
    for date, group in df.groupby(df["timestamp"].dt.date.astype(str)):
        partition_dir = validated_dir / f"date={date}"
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(partition_dir / "part.parquet", index=False)
        dates_written.append(date)
    return sorted(dates_written)


def write_quarantine_records(records: list[dict], warehouse_path: Path) -> None:
    conn = sqlite3.connect(warehouse_path)
    conn.execute(
        """
        create table if not exists quarantine (
            id integer primary key autoincrement,
            quarantined_at text not null,
            reason_code text not null,
            row_json text not null
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "insert into quarantine (quarantined_at, reason_code, row_json) values (?, ?, ?)",
        [(now, r["reason_code"], json.dumps(r["row"], default=str)) for r in records],
    )
    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    ingested_dir = Path(config["paths"]["ingested"])
    validated_dir = Path(config["paths"]["validated"])
    warehouse_path = Path(config["paths"]["warehouse"])
    timestamp_min = pd.Timestamp(config["validation"]["timestamp_min"])
    min_words = config["validation"]["language_check_min_words"]

    con = duckdb.connect()
    ingested = con.execute(f"select * from read_parquet('{ingested_dir}/*/*.parquet')").fetchdf()
    rows_in = len(ingested)

    schema = build_schema(timestamp_min)
    t0 = time.perf_counter()
    valid, quarantine_records = split_valid_and_quarantined(ingested, schema, min_words)
    latency = time.perf_counter() - t0

    dates_written = write_valid_partitions(valid, validated_dir)
    if quarantine_records:
        write_quarantine_records(quarantine_records, warehouse_path)

    reason_counts: dict[str, int] = {}
    for record in quarantine_records:
        reason_counts[record["reason_code"]] = reason_counts.get(record["reason_code"], 0) + 1

    log_run(
        warehouse_path,
        stage="validate",
        rows_in=rows_in,
        rows_quarantined=len(quarantine_records),
        latency_seconds=latency,
        score_distribution={"quarantine_reasons": reason_counts},
    )

    print(f"rows in: {rows_in}")
    print(f"rows passed: {len(valid)}")
    print(f"rows quarantined: {len(quarantine_records)} {reason_counts}")
    print(f"partitions written: {dates_written}")


if __name__ == "__main__":
    main()
