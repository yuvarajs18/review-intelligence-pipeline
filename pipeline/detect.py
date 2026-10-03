"""Detect: product degradation and model/topic drift alerts.

Implemented by the author (Yuvaraj), not scaffolded by Claude Code, per the
working agreement: the detection logic is what he needs to be able to
defend from first principles in an interview, not just from having it pass
tests someone else wrote. tests/test_detect.py was written first, against
the contracts documented below, before any implementation existed.

Two primitives:
- detect_spike: per-series anomaly detection (e.g. a product's weekly
  review count, or its mean rating) - flags points that deviate unusually
  far from the rest of the series, in either direction.
- compute_psi: Population Stability Index between two score distributions
  (e.g. sentiment scores from two different weeks) - the standard way to
  ask "has this distribution shifted enough to worry about drift?"

main() wires both primitives to the warehouse (config.paths.warehouse):
per-product degradation alerts (direction-aware, written idempotently to
an `alerts` table) and model-drift detection (PSI between a chronological
reference/current split of sentiment scores, logged to the `runs` table).
See docs/decisions.md for the z_threshold=2.0 decision.
"""

from collections.abc import Sequence

import pandas as pd
import numpy as np
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import duckdb
from pipeline.runs import log_run

import yaml


def detect_spike(series: pd.Series, z_threshold: float = 3.0) -> pd.Series:
    """Flags points in `series` that are unusual outliers relative to the
    rest of the series, in either direction - e.g. a product's weekly
    complaint count spiking upward, or its mean rating dropping sharply.

    Contract this must satisfy (see tests/test_detect.py, written against
    this contract before any implementation exists):
    - Compute `series`'s own mean and POPULATION standard deviation
      (ddof=0) - i.e. divide the sum of squared deviations by n, not n-1.
    - For each point, compute its z-score: (value - mean) / std.
    - Flag a point (True) when abs(z-score) >= z_threshold; otherwise
      False. Note the >=, not > - a point sitting exactly at the
      threshold counts as a spike.
    - If `series` has zero variance (std == 0 - e.g. a perfectly flat
      series with no signal at all), no point is a spike: return an
      all-False Series rather than raising or dividing by zero.

    Returns a boolean pd.Series aligned with `series`'s index (same
    length, same index labels).
    """
    std = series.std(ddof=0)
    if std == 0:
        return pd.Series(False, index=series.index)

    mean = series.mean()
    z_scores = (series - mean) / std
    return abs(z_scores) >= z_threshold

def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int | Sequence[float] = 10,
) -> float:
    """Population Stability Index (PSI): how much has `current`'s
    distribution shifted away from `reference`'s?

    In plain English: split the score range into bins. Work out what
    percentage of `reference`'s observations fall into each bin, then do
    the same for `current`. For every bin, compare the two percentages: a
    bin whose share barely changed contributes almost nothing; a bin whose
    share moved a lot contributes a lot. Summing those contributions
    across all bins gives one number describing how different the two
    distributions are overall.

    Formula, per bin i, using the natural log:
        term_i = (current_pct_i - reference_pct_i) * ln(current_pct_i / reference_pct_i)
    and PSI = sum(term_i) over all bins.

    Why bins are fixed from `reference`, never recomputed from `current`:
    the bins have to represent a fixed frame of reference so that a real
    shift shows up as observations migrating between fixed buckets. If you
    instead recomputed bin edges from `current`'s own distribution each
    time (e.g. its own quantiles), `current` would - by construction -
    always come out looking roughly evenly spread across its own bins,
    which would hide exactly the shift PSI exists to detect. So: if
    `bins` is an int, derive that many equal-width bin edges from
    `reference`'s own min/max, and reuse those exact edges for `current`.
    If `bins` is already a sequence of edges, use them as given for both.

    Known edge case to handle: a bin with zero observations in either
    distribution makes ln(0) undefined - needs explicit handling (e.g. a
    small epsilon, or excluding/merging that bin) so this never crashes on
    sparse data.

    Common rule of thumb, useful context but not enforced by this
    function: PSI < 0.1 is usually read as "no significant shift", 0.1-0.25
    as "moderate", and > 0.25 as "significant" - deciding what threshold
    should actually raise an alert is a separate judgment call.
    """
    ref_counts, edges = np.histogram(reference, bins)       # Step 1-2: get edges from reference
    cur_counts, _ = np.histogram(current, bins=edges)             # Step 3: reuse `edges`, NOT `bins` again

    reference_pct =  ref_counts/ref_counts.sum()     # Step 4: ref_counts turned into percentages (divide by the total)
    current_pct =  cur_counts/cur_counts.sum()         # Step 4: same idea, for cur_counts

    reference_pct_safe = reference_pct + 1e-8   # Step 5: add the small nudge number
    current_pct_safe = current_pct + 1e-8      # Step 5: same, for current_pct

    terms = (current_pct_safe - reference_pct_safe) * np.log(current_pct_safe / reference_pct_safe)     # Step 6: the formula, using the _safe versions
    return terms.sum()          # Step 6: add all the terms together into one number
    


def main() -> None:
    """CLI entrypoint - reads the warehouse (config.paths.warehouse),
    applies detect_spike / compute_psi to whichever weekly product stats
    and score distributions matter, and writes alerts wherever they should
    live (a new `alerts` table is a natural fit alongside `quarantine` and
    `runs`, but the schema and the alerting rules themselves are a design
    decision to make deliberately, not to inherit from this stub).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    warehouse_path = Path(config["paths"]["warehouse"])
    conn = sqlite3.connect(warehouse_path)
    stats = pd.read_sql_query("select * from product_week_stats", conn)
    conn.close()

    weeks_per_product = stats.groupby("product_id")["week_start"].count()
    eligible_products = weeks_per_product[weeks_per_product >= 3].index.tolist()
    print(f"{len(eligible_products)} products have >= 3 weeks of history")

    alerts = []
    for product_id in eligible_products:
        product_rows = stats[stats["product_id"] == product_id].sort_values("week_start").reset_index(drop=True)

        count_flags = detect_spike(product_rows["review_count"], z_threshold=2.0)
        count_bad = count_flags & (product_rows["review_count"] > product_rows["review_count"].mean())

        rating_flags = detect_spike(product_rows["mean_rating"], z_threshold=2.0)
        rating_bad = rating_flags & (product_rows["mean_rating"] < product_rows["mean_rating"].mean())

        for i in range(len(product_rows)):
            if count_bad[i]:
                alerts.append({
                    "product_id": product_id,
                    "week_start": product_rows["week_start"][i],
                    "alert_type": "complaint_volume_spike",
                    "metric_value": product_rows["review_count"][i],
                })
            if rating_bad[i]:
                alerts.append({
                    "product_id": product_id,
                    "week_start": product_rows["week_start"][i],
                    "alert_type": "rating_drop",
                    "metric_value": product_rows["mean_rating"][i],
                })

    print(f"{len(alerts)} alerts found")
    conn = sqlite3.connect(warehouse_path)
    conn.execute("""
        create table if not exists alerts (
            id integer primary key autoincrement,
            detected_at text not null,
            product_id text not null,
            week_start text not null,
            alert_type text not null,
            metric_value real not null,
            unique(product_id, week_start, alert_type)
        )
    """)
    now = datetime.now(timezone.utc).isoformat()
    for a in alerts:
        conn.execute(
            "insert or ignore into alerts (detected_at, product_id, week_start, alert_type, metric_value) values (?, ?, ?, ?, ?)",
            (now, a["product_id"], a["week_start"], a["alert_type"], float(a["metric_value"])),
        )
    conn.commit()
    conn.close()
    print(f"{len(alerts)} alerts found and written to the alerts table")

    scored_dir = Path(config["paths"]["scored"])
    con = duckdb.connect()
    scored = con.execute(f"select * from read_parquet('{scored_dir}/*/*.parquet')").fetchdf()
    scored = scored.sort_values("timestamp").reset_index(drop=True)

    midpoint = len(scored) // 2
    reference = scored.iloc[:midpoint]
    current = scored.iloc[midpoint:]

    print(f"reference: {len(reference)} rows, {reference['timestamp'].min()} to {reference['timestamp'].max()}")
    print(f"current:   {len(current)} rows, {current['timestamp'].min()} to {current['timestamp'].max()}")

    baseline_psi = compute_psi(reference["baseline_sentiment_score"], current["baseline_sentiment_score"])
    distilbert_psi = compute_psi(reference["distilbert_sentiment_score"], current["distilbert_sentiment_score"])

    print(f"baseline PSI:   {baseline_psi}")
    print(f"distilbert PSI: {distilbert_psi}")

    log_run(
        warehouse_path,
        stage="detect",
        rows_in=len(scored),
        latency_seconds=0.0,
        score_distribution={"baseline_psi": baseline_psi, "distilbert_psi": distilbert_psi},
    )


if __name__ == "__main__":
    main()

