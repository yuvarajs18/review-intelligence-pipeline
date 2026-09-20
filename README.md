# Review Intelligence Pipeline

A scheduled NLP monitoring system over e-commerce reviews: it detects what
customers complain about per product per week, and alerts when a product
degrades or when the model itself drifts.

## Overview

This is a portfolio project (MSc Data Science) built to be fully defensible
in a technical interview - every non-obvious design decision is recorded in
[`docs/decisions.md`](docs/decisions.md) with its context, the options
considered, and the trade-off actually made.

Hard constraints the whole system is built under:
- No paid APIs - all models run locally (`transformers` / `sentence-transformers`).
- CPU only, 8GB RAM assumed. No CUDA.
- Storage: Parquet for data, DuckDB for analytics, SQLite for state. No server DBs.

## Architecture

```
ingest -> validate -> score -> embed -> topics -> aggregate -> detect
```

| Stage | Reads | Writes |
|---|---|---|
| `ingest` | Raw Amazon Reviews 2023 category (local, via `scripts/download_data.py`) | Parquet, partitioned by date, overwrite-by-partition |
| `validate` | Ingested Parquet | Valid rows forward; failures to a `quarantine` table (SQLite) with a reason code - never silently dropped |
| `score` | Validated Parquet | Both a TF-IDF+logistic-regression baseline and DistilBERT score every row, side by side |
| `embed` | Scored Parquet, negative reviews only (`rating <= 2`) | Sentence embeddings (all-MiniLM-L6-v2) |
| `topics` | Embedded Parquet | Topic ID per negative review, via `.transform()` against a model fit once (never refit automatically - see decisions.md) |
| `aggregate` | Validated Parquet, via DuckDB | Review count + mean rating per product per week (SQLite) |
| `detect` | Aggregated stats + score distributions | Degradation/drift alerts - **implemented by the author**, not scaffolded by Claude Code (see `pipeline/detect.py`) |

Every prediction row carries the exact model name + version that produced
it (a training sha256 for self-trained models, a pinned HF git revision for
pretrained ones). Every stage-plus-model run logs latency, throughput, and
a score-distribution summary to a `runs` table.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py     # one-off: raw review category -> data/raw/
python scripts/download_model.py    # one-off: DistilBERT + MiniLM -> models/hf_cache/

python -m scripts.train_baseline    # one-off: trains the sentiment baseline
python -m scripts.fit_topics        # one-off: fits BERTopic once
```

After each training/fitting script, pin the resulting `sha256` in
`config/pipeline.yaml` (`model.baseline.active_model_sha`,
`topics.active_model_sha`) - this is a deliberate step, not automatic, so a
scoring or topic-assignment run only ever changes model when you say so.

## Usage

```bash
make ingest      # slice the raw category into a date-partitioned lake
make validate    # pandera schema + language filter, quarantines failures
make score       # baseline + DistilBERT, side by side
make embed       # sentence embeddings for negative reviews only
make topics      # assign topics via .transform() - never refits
make aggregate   # weekly review count + mean rating per product
make detect      # degradation/drift alerts (author-implemented)
make dashboard   # Streamlit dashboard, read-only over the warehouse
make test        # pytest
```

## Key design decisions

A few of the more interview-relevant ones - full reasoning for each is in
[`docs/decisions.md`](docs/decisions.md):

- **Why BERTopic is never refit automatically** - topic IDs have no
  persistent meaning across separate fits; refitting daily would silently
  break both the hand-written topic labels and any week-over-week drift
  comparison.
- **Why `active_model_sha` (and every pretrained model's revision) is
  pinned in config and bumped deliberately**, never picked up automatically
  from "whatever was trained most recently."
- **Why `stop_words=None` for the baseline's TF-IDF vectoriser** - sklearn's
  default English stop-word list strips negators (`not`, `no`), which
  silently breaks negation detection for sentiment.
- **A real, observed limitation**: the language filter (`langdetect`)
  produced one false positive in testing - see decisions.md for the
  reproduction and why it's an acceptable, auditable trade-off rather than
  a bug that was hidden.

## Results

TKTK - filled in from my own runs, not invented here.

- Baseline (TF-IDF + logistic regression) vs. DistilBERT: accuracy /
  precision / recall / F1, and where they disagree: TKTK
- Topic model: number of topics found, outlier rate, example themes with
  their hand-edited labels: TKTK
- Drift experiment (`evaluation/drift_experiment.py`): PSI trend across the
  injection point, and what it does/doesn't show: TKTK
- Detection (`pipeline/detect.py`): example alerts raised against real data,
  and how the thresholds were chosen: TKTK

## Limitations

- CI's scheduled pipeline run depends on `actions/cache` for trained model
  artifacts (not committed to git); a cache eviction fails the run loudly
  rather than silently retraining - a known gap in a no-paid-services setup,
  not an oversight (see `.github/workflows/pipeline.yml`).
- BERTopic's outlier rate on this category's training population: TKTK.
- The language filter's short-text exemption threshold is a judgment call,
  not a tuned optimum - see decisions.md.

## How this was built

Claude Code did the scaffolding, boilerplate, and staged implementation
work across each pipeline stage - infrastructure, tests, and the
CLAUDE.md-enforced habit of writing down every non-obvious decision as it
was made. `pipeline/detect.py`'s actual detection logic and everything in
`evaluation/` were implemented by hand, deliberately kept out of Claude's
scope, so the parts of this project that most need defending in an
interview are the parts I can actually defend.
