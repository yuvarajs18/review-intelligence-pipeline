"""Aggregate: weekly review count and mean rating per product, via DuckDB,
written to SQLite.

`compute_weekly_aggregate` is the pure/testable part: it takes an already-
populated DuckDB connection and returns a DataFrame, so a unit test can
register an in-memory fixture table and never touch the Parquet lake or the
SQLite warehouse file.
"""

import argparse
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import yaml

WEEKLY_AGGREGATE_SQL = """
    select
        parent_asin as product_id,
        date_trunc('week', timestamp) as week_start,
        count(*) as review_count,
        avg(rating) as mean_rating
    from {table}
    group by 1, 2
    order by 1, 2
"""


def compute_weekly_aggregate(con: duckdb.DuckDBPyConnection, table: str = "reviews") -> pd.DataFrame:
    return con.execute(WEEKLY_AGGREGATE_SQL.format(table=table)).fetchdf()


def write_to_warehouse(df: pd.DataFrame, warehouse_path: Path) -> None:
    """Upsert keyed on (product_id, week_start): re-running the same window
    overwrites those rows instead of duplicating them."""
    conn = sqlite3.connect(warehouse_path)
    conn.execute(
        """
        create table if not exists product_week_stats (
            product_id text not null,
            week_start text not null,
            review_count integer not null,
            mean_rating real not null,
            primary key (product_id, week_start)
        )
        """
    )
    rows = [
        (row.product_id, str(row.week_start.date()), int(row.review_count), float(row.mean_rating))
        for row in df.itertuples()
    ]
    conn.executemany(
        """
        insert or replace into product_week_stats
            (product_id, week_start, review_count, mean_rating)
        values (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    validated_dir = Path(config["paths"]["validated"])
    warehouse_path = Path(config["paths"]["warehouse"])

    con = duckdb.connect()
    con.execute(f"create view reviews as select * from read_parquet('{validated_dir}/*/*.parquet')")
    weekly = compute_weekly_aggregate(con)
    write_to_warehouse(weekly, warehouse_path)

    print(weekly.to_string(index=False))


if __name__ == "__main__":
    main()
