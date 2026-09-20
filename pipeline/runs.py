"""Shared run-logging: every stage that's worth measuring writes one row per
(stage, model) here, to the `runs` table in the SQLite warehouse.

One row per model rather than one wide row per stage invocation, because
throughput/latency is a per-model property (e.g. `score` runs two models
per invocation and they can have very different speeds) - a single row
covering both would hide exactly the comparison the runs table exists for.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CREATE_RUNS_TABLE = """
    create table if not exists runs (
        id integer primary key autoincrement,
        run_at text not null,
        stage text not null,
        model_name text,
        model_version text,
        rows_in integer,
        rows_quarantined integer,
        latency_seconds real,
        rows_per_second real,
        score_distribution text
    )
"""


def log_run(
    warehouse_path: Path,
    stage: str,
    rows_in: int,
    latency_seconds: float,
    model_name: str | None = None,
    model_version: str | None = None,
    rows_quarantined: int | None = None,
    score_distribution: dict | None = None,
) -> None:
    rows_per_second = rows_in / latency_seconds if latency_seconds > 0 else None
    conn = sqlite3.connect(warehouse_path)
    conn.execute(CREATE_RUNS_TABLE)
    conn.execute(
        """
        insert into runs
            (run_at, stage, model_name, model_version, rows_in,
             rows_quarantined, latency_seconds, rows_per_second, score_distribution)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            stage,
            model_name,
            model_version,
            rows_in,
            rows_quarantined,
            latency_seconds,
            rows_per_second,
            json.dumps(score_distribution) if score_distribution is not None else None,
        ),
    )
    conn.commit()
    conn.close()
