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

All numbers below are from the current pipeline state: `window.lookback_days=730`
(2 years), baseline sha `f732ce395686...`, topics sha `d62f4dfa6044...` -
see decisions.md for why the window was widened and what it took to do
that safely (a real train/serve overlap bug, found and fixed, not just a
config edit).

**Baseline (TF-IDF + logistic regression)**, trained on 87,403 reviews,
held out 21,851 for evaluation. The headline numbers in `models/registry.json`
(accuracy 0.951, precision 0.993, recall 0.954, F1 0.973) are already
**positive-class-only** (sklearn's `average="binary"` default) - they look
uniformly strong because the test set is ~92.8% positive. The real,
previously-uncomputed story is the per-class breakdown, reconstructed by
replaying the exact training-time train/test split against the pinned
model (verified: reconstructed test set is 21,851 rows, and the positive-
class numbers match the registry exactly):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (complaint) | 0.605 | 0.914 | 0.728 | 1,575 |
| Positive | 0.993 | 0.954 | 0.973 | 20,276 |

Confusion matrix (rows = true label, columns = predicted):

```
              pred neg   pred pos
true neg         1439        136
true pos          940      19336
```

**Reading it honestly**: recall on the negative class is strong (0.914 -
the model catches 91.4% of genuine complaints), but precision is weak
(0.605 - about 2 in 5 reviews it calls "negative" are actually positive,
940 false alarms vs. 1,439 true ones). This is the direct, expected cost
of training with `class_weight="balanced"` instead of undersampling (see
decisions.md) - it explicitly trades precision for recall on the minority
class, which is the right trade for a degradation-monitoring system
(missing a real complaint costs more than one extra false alarm), but the
opposite of what the 0.951 accuracy headline implies on its own.

This imprecision does **not** contaminate topic modeling - `pipeline/embed.py`
routes reviews into BERTopic by the real star rating (`rating <= 2`), never
by this model's predicted label. It would matter anywhere
`baseline_sentiment_label` is used directly and standalone (e.g. the
dashboard's model panel).

**Baseline vs. DistilBERT**, same 15,003-row scored window:

| | Baseline | DistilBERT |
|---|---|---|
| Throughput | 34,497 rows/sec | 24.0 rows/sec |
| Agreement | 90.8% (13,617/15,003) | |
| Disagreement | 9.2% (1,386/15,003) | |

A real disagreement case, pulled directly from `data/scored/`: *"Excellent
cd, great presentation but too expensive!"* (rating 5) - baseline scores
it 0.84 (positive), DistilBERT scores it 0.00 (negative). Explainable, not
random: the baseline is bag-of-words and sees "excellent"/"great" without
caring where they sit in the sentence; DistilBERT reads in order and
weights the clause it ends on ("too expensive!") more heavily. TKTK: a
systematic look at what the other ~1,385 disagreements look like, beyond
this one hand-picked example.

**Topic model**: 47 topics found; 3,270 of 7,874 training reviews (41.5%)
landed in the outlier bucket (topic `-1`) rather than any cluster. The
training population *shrank* from the previous fit (9,289&rarr;7,874
negative reviews) as a direct, expected consequence of widening the
serving window - see decisions.md.

Labels in `config/topic_names.yaml` are hand-edited for this fit,
grounded in each topic's real representative documents (the previous
hand-editing pass, done against the old 50-topic fit, was discarded
wholesale when the refit assigned entirely new topic IDs - exactly the
"re-edit after every refit" consequence documented in the file's own
header). Fragmentation persists after refitting, it just relocates: topics
`0`/`2` are both terse one-star reviews, `23`/`24` are both "cracked case,"
and `25`/`31`/`44` are now *three* near-duplicate Beatles-bootleg
clusters (up from two in the previous fit). Also still present:
miscategorized items from the source dataset (`4` a book, `37` a food
item, `42` a garden hose).

TKTK: whether 41.5% outliers is acceptable or needs retuning, and whether
the recurring fragmentation (not just this fit's specific topics, but the
*pattern* of fragmentation surviving a refit) suggests `hdbscan`'s
parameters need retuning rather than the training population.

**Drift experiment** (`evaluation/drift_experiment.py`, 60 simulated days,
Gift_Cards injected from day 30 onward), PSI against a fixed pre-injection
reference, rerun against the newly pinned baseline:

| | Baseline PSI | DistilBERT PSI |
|---|---|---|
| Pre-injection (days 0-29) | 0.017 - 0.144 (mean 0.064) | 0.012 - 0.295 (mean 0.144) |
| Post-injection (days 30-59) | 0.073 - 0.520 (mean 0.141) | 0.065 - 0.911 (mean 0.454) |
| Day 29 -> day 30 | 0.041 -> 0.076 | 0.210 -> 0.551 |

DistilBERT's numbers are unchanged from before (it wasn't retrained, only
the baseline and topic model were) - confirms the harness is deterministic
for a fixed model.

TKTK: what this does/doesn't show - e.g. whether DistilBERT's much larger
pre-injection PSI range (up to 0.295 with *no* injection yet) means it's
just a noisier signal on this data, or whether the fixed reference window
was too small to give DistilBERT a stable baseline.

**Detection** (`pipeline/detect.py`, run against the 2-year window, 969
products with >= 3 weeks of history; two different thresholds, not one -
`z_threshold=2.0` for the volume check, `z_threshold=4.0` for the rating
check - rationale for each in decisions.md):
- 66 alerts: 54 `complaint_volume_spike`, 12 `rating_drop` (down from 313
  at `z_threshold=2.0` for both checks - the rating check alone had been
  producing 259 of those 313 alerts, which is why it was split out and
  tightened separately; see decisions.md for that comparison and for what
  widening the window safely required earlier: a real train/serve overlap
  bug, found and fixed, not just a bigger number).
- Model drift (PSI, chronological first-half/second-half split of all
  15,003 scored reviews): baseline 0.0042, DistilBERT 0.0032 - smaller
  than the previous window's 0.028/0.017, consistent with a much larger,
  more stable reference/current split (7,501 rows each vs. 1,103 before)
  producing less sampling noise.
- A real bug, found by inspecting one alert rather than trusting the
  count: `complaint_volume_spike` fired on product `B07D5ZGJF1`, whose
  `mean_rating` is 5.0 on every single week it has - the check only looked
  at review-count volume, never whether those reviews were actually
  negative. Fixed by requiring the week's rating to also be below the
  product's own average; `complaint_volume_spike` alerts dropped
  190&rarr;54 as a direct result (most of the previous ones were this
  exact false-positive pattern).
- That fix then triggered the *same* stale-alerts gap a second time,
  independently (first found when the window widened, not fixed then
  either): after the logic change, the `alerts` table still showed 449
  until manually cleared and repopulated, since `insert or ignore` had no
  way to remove the 136 alerts the new logic no longer produces. Two
  independent real occurrences of this gap - documented in decisions.md,
  still not fixed in `main()` itself.

TKTK: 2-3 example alerts worth walking through in an interview (three are
already walked through step by step in the Pipeline Blueprint artifact,
with hand-verified z-scores).

## Limitations

- CI's `pipeline` job had a 0% success rate across its entire history (30
  runs) until 2026-10-10 - not an occasional cache-eviction event as this
  doc used to claim, but a structural one: nothing in the workflow ever
  *trained* the self-trained artifacts it was trying to restore from cache.
  Fixed by training on a cache miss and verifying the result's sha matches
  the pin before continuing (see `.github/workflows/pipeline.yml` and
  decisions.md) - residual risk is cross-platform floating-point
  nondeterminism causing a false-alarm sha mismatch, which fails loudly by
  design rather than silently retrying.
- BERTopic's outlier rate on this category's training population is 41.5%
  (3,270/7,874) - whether that's acceptable or needs retuning is TKTK.
- The language filter's short-text exemption threshold is a judgment call,
  not a tuned optimum - see decisions.md.
- A refit always resets `config/topic_names.yaml` to raw keyword stems -
  re-labeling it (done twice now, for the 50-topic and 47-topic fits) is a
  manual step with no automation, and will need redoing again on the next
  deliberate refit.

## How this was built

Claude Code did the scaffolding, boilerplate, and staged implementation
work across each pipeline stage - infrastructure, tests, and the
CLAUDE.md-enforced habit of writing down every non-obvious decision as it
was made. `pipeline/detect.py`'s actual detection logic and everything in
`evaluation/` were implemented by hand, deliberately kept out of Claude's
scope, so the parts of this project that most need defending in an
interview are the parts I can actually defend.
