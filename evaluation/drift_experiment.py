"""Drift experiment harness - replays N simulated days of the pipeline
normally, then from a configurable day onward injects a different product
category's reviews instead, and dumps per-day score distributions and PSI
(against a fixed pre-injection reference) to CSV.

This is a harness only. It does not compute a verdict, plot anything, or
decide what counts as "drift" - that analysis is the author's to do himself
from the CSV this produces, per the working agreement (this file lives in
evaluation/, which is otherwise off-limits for Claude to write into without
being explicitly asked, as it was for this stage).

"Run the pipeline unchanged" is literal here, not just similar logic: every
simulated day calls the exact same pure functions the production pipeline
uses (filter_window, split_valid_and_quarantined, score_with_baseline,
attach_distilbert_scores). Every simulated day's data stays in memory - this
never touches config/pipeline.yaml or the real data/ingested, data/validated,
data/scored directories, since writing 60+ simulated days into those
overwrite-by-partition directories would corrupt real pipeline state.

Design notes (see docs/decisions.md for the full reasoning):
- A single calendar day is too sparse for a stable distribution in this
  category (checked: median ~4 reviews/day, several zero-review days) - each
  simulated "day" is instead a rolling 30-day window advancing by one day
  per step, same mechanic as pipeline/ingest.py's production window.
- The injected category (Gift_Cards) is not date-aligned to the normal
  category's calendar - a fixed-seed random sample is drawn from its full
  pool each post-injection day instead, since the point is a content/
  distribution shift, not calendar realism.
- pipeline.detect.compute_psi is still NotImplementedError (Stage 6 is the
  author's to implement) - PSI columns read "pending" until that body
  exists, so the rest of this harness is independently runnable now.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

from pipeline.features import build_text_column
from pipeline.ingest import filter_window, load_raw_reviews, resolve_window
from pipeline.score import (
    attach_distilbert_scores,
    build_distilbert_pipeline,
    load_active_baseline,
    score_with_baseline,
)
from pipeline.validate import build_schema, split_valid_and_quarantined

INJECTION_HF_PATH = "raw/review_categories/Gift_Cards.jsonl"
# Fallback sample size for post-injection days if no pre-injection day ever
# produced a non-empty window to measure a "typical" size from.
DEFAULT_INJECTION_SAMPLE_SIZE = 100


def ensure_injection_data(hf_repo: str, hf_path: str, dest_dir: Path) -> Path:
    """One-off local download of the injection category, same mechanism as
    scripts/download_data.py. Skips the download if already cached."""
    dest_path = dest_dir / Path(hf_path).name
    if dest_path.exists():
        return dest_path
    cached_path = hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=hf_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached_path, dest_path)
    return dest_path


def sample_injection_batch(injected_df: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Fixed-seed random sample from the injected category's full pool - not
    date-aligned to the normal category's simulated calendar, since the
    point is a content/distribution shift, not calendar realism."""
    n_rows = min(n_rows, len(injected_df))
    return injected_df.sample(n=n_rows, random_state=seed).copy()


def run_day(
    day_df: pd.DataFrame,
    schema,
    min_words: int,
    baseline_model,
    baseline_entry: dict,
    distilbert_pipe,
    distilbert_config: dict,
) -> pd.DataFrame | None:
    """Runs validate -> score on one simulated day's windowed data, in
    memory, using the exact pipeline functions - no disk I/O, no config
    mutation, no writes to the real data/ directories. Returns None if
    nothing survives validation (an empty window, or everything quarantined)."""
    valid, _ = split_valid_and_quarantined(day_df, schema, min_words)
    if valid.empty:
        return None
    scored = score_with_baseline(valid, baseline_model, baseline_entry["name"], baseline_entry["sha256"])
    texts = build_text_column(valid).tolist()
    raw_outputs = distilbert_pipe(
        texts,
        batch_size=distilbert_config["batch_size"],
        truncation="only_first",
        max_length=distilbert_config["max_length"],
    )
    return attach_distilbert_scores(scored, raw_outputs)


def summarize_distribution(values: pd.Series) -> dict:
    """Plain descriptive statistics - not interpretation, just the numbers
    the CSV is explicitly asked to carry."""
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def compute_psi_or_pending(reference: pd.Series, current: pd.Series):
    """pipeline.detect.compute_psi is still NotImplementedError until Stage
    6's body is filled in - caught here so the replay/injection/distribution
    mechanics in this harness are independently verifiable right now,
    instead of the whole harness being blocked on that."""
    from pipeline.detect import compute_psi

    if reference.empty or current.empty:
        return "pending"
    try:
        return compute_psi(reference, current)
    except NotImplementedError:
        return "pending"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--n-days", type=int, default=60, help="total simulated days to replay")
    parser.add_argument(
        "--injection-day",
        type=int,
        default=30,
        help="0-based day index from which the injected category replaces the normal one",
    )
    parser.add_argument("--lookback-days", type=int, default=30, help="rolling window size per simulated day")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="evaluation/drift_experiment_results.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["paths"]["raw_local"])
    normal_path = raw_dir / Path(config["dataset"]["hf_path"]).name
    injected_path = ensure_injection_data(config["dataset"]["hf_repo"], INJECTION_HF_PATH, raw_dir)

    normal_df = load_raw_reviews(normal_path)
    injected_df = load_raw_reviews(injected_path)

    schema = build_schema(pd.Timestamp(config["validation"]["timestamp_min"]))
    min_words = config["validation"]["language_check_min_words"]
    baseline_model, baseline_entry = load_active_baseline(config)
    distilbert_config = config["model"]["distilbert"]
    distilbert_pipe = build_distilbert_pipeline(config)

    last_real_date = pd.Timestamp(config["window"]["reference_date"])
    start_date = last_real_date - pd.Timedelta(days=args.n_days - 1)

    # Pass 1: build each simulated day's input window (normal or injected).
    typical_window_size = None
    day_windows = []
    for day_index in range(args.n_days):
        reference_date = start_date + pd.Timedelta(days=day_index)
        window_start, window_end = resolve_window(reference_date, args.lookback_days)

        if day_index < args.injection_day:
            category = "normal"
            windowed = filter_window(normal_df, window_start, window_end)
            if typical_window_size is None and len(windowed) > 0:
                typical_window_size = len(windowed)
        else:
            category = "injected"
            windowed = sample_injection_batch(
                injected_df, typical_window_size or DEFAULT_INJECTION_SAMPLE_SIZE, seed=args.seed + day_index
            )
        day_windows.append((day_index, reference_date, category, windowed))

    # Pass 2: validate + score each day once, in memory.
    scored_by_day = {}
    for day_index, reference_date, category, windowed in day_windows:
        scored = run_day(windowed, schema, min_words, baseline_model, baseline_entry, distilbert_pipe, distilbert_config)
        scored_by_day[day_index] = scored
        n_scored = 0 if scored is None else len(scored)
        print(f"day {day_index:3d} ({category:8s}) {reference_date.date()}: {len(windowed)} in window, {n_scored} scored")

    # Fixed reference distribution: pooled from ALL pre-injection days, built
    # once, never recomputed per-day - see pipeline/detect.py's compute_psi
    # docstring for why this matters.
    pre_injection_frames = [
        scored_by_day[i] for i in range(args.injection_day) if scored_by_day.get(i) is not None
    ]
    if pre_injection_frames:
        reference_baseline = pd.concat([f["baseline_sentiment_score"] for f in pre_injection_frames], ignore_index=True)
        reference_distilbert = pd.concat(
            [f["distilbert_sentiment_score"] for f in pre_injection_frames], ignore_index=True
        )
    else:
        reference_baseline = pd.Series(dtype=float)
        reference_distilbert = pd.Series(dtype=float)

    # Pass 3: dump per-day distribution stats and PSI-vs-reference.
    rows = []
    for day_index, reference_date, category, windowed in day_windows:
        scored = scored_by_day[day_index]
        row = {
            "day_index": day_index,
            "reference_date": reference_date.date().isoformat(),
            "category": category,
            "rows_in_window": len(windowed),
            "rows_scored": 0 if scored is None else len(scored),
        }
        if scored is not None and not scored.empty:
            for prefix, col in [("baseline", "baseline_sentiment_score"), ("distilbert", "distilbert_sentiment_score")]:
                for stat_name, stat_value in summarize_distribution(scored[col]).items():
                    row[f"{prefix}_score_{stat_name}"] = stat_value
            row["baseline_psi"] = compute_psi_or_pending(reference_baseline, scored["baseline_sentiment_score"])
            row["distilbert_psi"] = compute_psi_or_pending(reference_distilbert, scored["distilbert_sentiment_score"])
        rows.append(row)

    results = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
