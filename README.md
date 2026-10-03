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

Raw numbers below are from real runs (`models/registry.json`,
`models/topics_registry.json`, `data/warehouse.db`,
`evaluation/drift_experiment_results.csv`) - interpretation and write-up are
still TKTK, left for the author to do himself.

**Baseline (TF-IDF + logistic regression)**, trained on 99,160 reviews,
held out 24,791 for evaluation:

| Metric | Value |
|---|---|
| Accuracy | 0.952 |
| Precision | 0.992 |
| Recall | 0.956 |
| F1 | 0.974 |

TKTK: these are high partly because of the ~12:1 class imbalance in this
category (see decisions.md) - a per-class breakdown would be the more
honest number to lead with in an interview.

**Baseline vs. DistilBERT**, same 2,206-row scored window:

| | Baseline | DistilBERT |
|---|---|---|
| Throughput | 35,805 rows/sec | 33.7 rows/sec |
| Predicted positive | 1,854 | 1,796 |
| Mean score | 0.802 | 0.815 |

Label agreement: 91.0% (2,008/2,206); disagreement: 9.0% (198/2,206).
TKTK: what those 198 disagreements actually look like (sarcasm, mixed
reviews, truncation effects) - not inspected yet.

**Topic model**: 50 topics found; 3,645 of 9,289 training reviews (39.2%)
landed in the outlier bucket (topic `-1`) rather than any cluster. Labels
are hand-edited (see `config/topic_names.yaml`), grounded in each topic's
actual representative documents, not guessed from keyword stems alone -
e.g. `9` -> `cd_skips_during_playback`, `4` -> `missing_promotional_merch_poster_photocard`,
`13` -> `cracked_or_broken_cd_case`.

Labeling surfaced a real fragmentation issue: topics `23`, `32`, and `41`
are all "poor sound/recording quality," split into three separate clusters
instead of one, and `35`/`43` are near-duplicate "Beatles bootleg" warnings
- documented in the label file's header rather than silently merged, since
fixing it means retuning `hdbscan.min_cluster_size`/`min_samples` and
deliberately refitting, not editing labels. Also surfaced: topics `8`, `37`,
and `44` turn out to be a book, a food item, and a garden hose/instrument
part - miscategorized into Digital_Music in the source dataset.

TKTK: whether 39.2% outliers is acceptable or needs retuning, and whether
the fragmented "sound quality" topics are worth merging on the next
deliberate refit.

**Drift experiment** (`evaluation/drift_experiment.py`, 60 simulated days,
Gift_Cards injected from day 30 onward), PSI against a fixed pre-injection
reference:

| | Baseline PSI | DistilBERT PSI |
|---|---|---|
| Pre-injection (days 0-29) | 0.030 - 0.063 (mean 0.040) | 0.012 - 0.295 (mean 0.144) |
| Post-injection (days 30-59) | 0.089 - 0.614 (mean 0.180) | 0.065 - 0.911 (mean 0.454) |
| Day 29 -> day 30 | 0.042 -> 0.118 | 0.210 -> 0.551 |

TKTK: what this does/doesn't show - e.g. whether DistilBERT's much larger
pre-injection PSI range (up to 0.295 with *no* injection yet) means it's
just a noisier signal on this data, or whether the fixed reference window
was too small to give DistilBERT a stable baseline.

**Detection** (`pipeline/detect.py`, run against a 200-day window, 102
products with >= 3 weeks of history, `z_threshold=2.0`):
- 27 alerts: 15 `rating_drop`, 12 `complaint_volume_spike`.
- Model drift (PSI, chronological first-half/second-half split of all 2,206
  scored reviews): baseline 0.028, DistilBERT 0.017 - both well under the
  0.1 "no significant shift" rule of thumb, consistent with both halves
  coming from the same stable historical category rather than a real shift.

TKTK: whether `z_threshold=2.0` is the right sensitivity for production use
(chosen after observing that the default `3.0` missed a real rating-drop
week in testing - this decision doesn't have a `docs/decisions.md` entry
yet and should get one, per this project's own convention), and 2-3 example
alerts worth walking through in an interview.

## Limitations

- CI's scheduled pipeline run depends on `actions/cache` for trained model
  artifacts (not committed to git); a cache eviction fails the run loudly
  rather than silently retraining - a known gap in a no-paid-services setup,
  not an oversight (see `.github/workflows/pipeline.yml`).
- BERTopic's outlier rate on this category's training population is 39.2%
  (3,645/9,289) - whether that's acceptable or needs retuning is TKTK.
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
