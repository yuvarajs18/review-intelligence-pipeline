# Decisions

## 2026-09-20 — Category: Digital_Music

**Context:** Needed one Amazon Reviews 2023 category small enough to load fully
into memory on an 8GB, CPU-only machine, but with enough real complaint text
to be worth topic modeling later.

**Options considered:** `Musical_Instruments` (rich complaint text, but its
raw `.jsonl` is 1.56GB — checked via the HF repo's file listing before
downloading anything, not from memory of the dataset); `Gift_Cards` /
`Magazine_Subscriptions` / `Subscription_Boxes` (tiny, 9–50MB, but complaints
skew to delivery/redemption rather than product quality); `Digital_Music`
(78.8MB, ~130k reviews, complaints cover audio/media quality and shipping
condition).

**Choice:** `Digital_Music`.

**Consequence:** Everything downstream — weekly volume, topic themes, what
"degradation" looks like in a demo — is scoped to a music-reviews catalogue,
not electronics or apparel. Worth stating explicitly if asked why this
category in an interview.

## 2026-09-20 — Idempotency: overwrite-by-date-partition

**Context:** `make ingest` needs to be safely re-runnable without duplicating
rows in the Parquet lake.

**Options considered:** append + dedupe on a review id; overwrite the whole
date partition on each run that touches it.

**Choice:** overwrite-by-date-partition — each `date=YYYY-MM-DD/` directory
is deleted and rewritten wholesale.

**Consequence:** Simpler than dedup logic, but a partition is only ever as
complete as the last run that wrote it — there is no partial-append story.
Fine for a static historical dataset; would need revisiting for a genuinely
live feed where a day's data could arrive in multiple batches.

## 2026-09-20 — Window: rolling, not fixed

**Context:** `config/pipeline.yaml` needed a way to express which dates a
given run covers.

**Options considered:** a fixed historical date range (reproducible, but
doesn't generalize past a one-off demo); a rolling window relative to a
reference date (generalizes to a real daily cron job).

**Choice:** rolling, sized by `window.lookback_days`, evaluated against
`window.reference_date`.

**Consequence:** Amazon Reviews 2023 is a static historical dump (max
timestamp 2023-09-06), so `reference_date` is pinned in config to that date
rather than left to default to wall-clock `now()` — otherwise every run
would see zero rows. In a real deployment against a live feed, that config
key would be dropped so the pipeline falls back to `datetime.now()`.

## 2026-09-20 — Source access: local `.jsonl` download, not `datasets.load_dataset`

**Context:** `get_dataset_config_names` / `load_dataset` against
`McAuley-Lab/Amazon-Reviews-2023` fails on the installed `datasets==5.0.1`
with `RuntimeError: Dataset scripts are no longer supported` — the repo still
ships a legacy Python loading script, which recent `datasets` releases
refuse to execute.

**Options considered:** pin an older `datasets` version that still executes
loading scripts; bypass `datasets` entirely and pull the plain `.jsonl` files
the same repo also publishes under `raw/review_categories/`, via
`huggingface_hub.hf_hub_download`.

**Choice:** direct `hf_hub_download` of the per-category `.jsonl`.

**Consequence:** No dependency on a deprecated code-execution path, and one
file per category is exactly the granularity the pipeline needs anyway.
Downside: if the repo ever drops the flat `.jsonl` mirror in favor of the
scripted loader only, `scripts/download_data.py` breaks and needs revisiting.

## 2026-09-20 — Baseline: class-imbalance handling and vectoriser settings

**Context:** Digital_Music reviews strictly before the ingest window split
114,662 positive (rating≥4) vs. 9,289 negative (rating≤2) after dropping
6,392 neutral (rating=3) rows — a ~12:1 imbalance.

**Options considered (imbalance):** undersample the majority class to match
the minority (~9,289 each), vs. `class_weight="balanced"` in
`LogisticRegression`, which reweights the loss instead of dropping rows.

**Choice:** `class_weight="balanced"`.

**Consequence:** Keeps the full training set rather than throwing away ~85%
of it, at the cost of predicted probabilities no longer reflecting the true
base rate (fine for the 0.5 threshold used to derive `sentiment_label`, but
worth remembering if `sentiment_score` is ever read as a literal probability
elsewhere).

**Options considered (vectoriser stop words):** sklearn's built-in English
stop-word list, vs. no stop-word removal (`stop_words=None`).

**Choice:** `stop_words=None`. sklearn's default list strips negators
(`not`, `no`, `nor`, `cannot`) before n-grams are formed, which silently
collapses `"not good"` into `"good"` — a known, documented flaw that is
exactly wrong for sentiment. Combined with `ngram_range=(1, 2)`, keeping
negators lets bigrams like `"not_good"` survive into the vocabulary.

## 2026-09-20 — Dropped the `images` field at ingest

**Context:** `pipeline/score.py` failed with a DuckDB `ConversionException`
reading multiple ingested Parquet partitions together: `images` is a
`list<struct<...>>`, but pandas/pyarrow infers `list<null>` for any date
partition where every review happens to have zero images, producing
inconsistent schemas across partition files.

**Options considered:** read partitions with `union_by_name=True` (didn't
resolve this specific conflicting-type case, only missing-column cases);
explicitly enumerate columns in every downstream `read_parquet` call,
excluding `images`; drop `images` once, at ingest.

**Choice:** drop `images` in `pipeline/ingest.load_raw_reviews`, before it
ever reaches Parquet.

**Consequence:** No downstream stage (score, topics, aggregate, detect) uses
review images, so nothing is lost. If a future stage needs them, they'd need
to be re-added with an explicit, consistent Arrow schema rather than letting
pyarrow infer one per partition.

## 2026-09-20 — DistilBERT: pinned revision, forced CPU, head-only truncation

**Context:** Adding `distilbert-base-uncased-finetuned-sst-2-english`
alongside the baseline (Stage 3). Three things needed a deliberate choice
rather than a library default.

**Pinning:** a pretrained model can drift under a fixed name just as a
self-trained one can if not versioned. Pinned to the model's current HF git
revision, `714eb0fa89d2f80546fda750413ed43d93601a13`, recorded in config and
stamped on every scored row as `distilbert_model_version` — the same
"deliberate bump, not silent drift" philosophy as `active_model_sha`.

**Device:** checked current `transformers` docs before assuming anything —
leaving `device` unset makes `Pipeline` auto-select "the first available
accelerator (CUDA, Apple Silicon MPS, XPU...)". This machine has MPS
available (confirmed via `torch.backends.mps.is_available()`), so leaving
`device` unset would have silently violated the "CPU only" hard constraint.
Passed `device="cpu"` explicitly instead of relying on there being no GPU.

**Truncation:** checked the actual data before choosing - 3,141 of 130,434
reviews (2.4%) exceed DistilBERT's 512-token limit, median review is 28
words. Chose head-only truncation (`truncation="only_first", max_length=512`)
over head+tail or chunk-and-average: it only affects a small tail, prior
work on BERT fine-tuning (Sun et al.) shows head truncation performs close
to fancier strategies for classification, and it's far cheaper on CPU.

**Consequence of all three:** offline-after-first-download is enforced
architecturally, not just documented - `pipeline/score.py` loads the model
with `local_files_only=True`, so a missing/stale cache fails immediately
with a clear error rather than a slow or flaky network retry mid-run.
Verified by re-running `make score` with `HTTP_PROXY`/`HTTPS_PROXY` pointed
at an unroutable address; the run succeeded identically.

## 2026-09-20 — Batching kept configurable despite HF's CPU guidance

**Context:** the stage requires batch size to be configurable, but current
`transformers` docs explicitly advise against batching on CPU ("don't batch
inference if you're using a CPU" - no parallelism win, and uneven-length
padding can add overhead instead of savings).

**Choice:** kept `model.distilbert.batch_size` in config anyway, per the
requirement, rather than hardcoding `batch_size=1`.

**Consequence:** the per-model latency/throughput now logged to the `runs`
table (see `pipeline/runs.py`) makes this an empirical question rather than
a guess - if batching turns out not to help on this machine, that's visible
in the data rather than assumed from a general rule of thumb, and the knob
costs nothing to keep.

## 2026-09-20 — DistilBERT score normalized to P(positive)

**Context:** the baseline's `sentiment_score` is `predict_proba` for the
positive class - a stable, comparable meaning. The raw `transformers`
sentiment-analysis pipeline instead returns confidence in *whichever* label
it predicted (a confident NEGATIVE prediction looks like `{"label":
"NEGATIVE", "score": 0.95}`, not a low positive-class score).

**Choice:** `attach_distilbert_scores` converts this to P(positive) before
writing it out: `score if label == "POSITIVE" else 1 - score`.

**Consequence:** both models' `*_sentiment_score` columns now mean the same
thing and are directly comparable (e.g. for later drift/aggregate work).
Using DistilBERT's raw pipeline output as-is would have silently produced a
column that looks like a probability but isn't consistently one.

## 2026-09-20 — Validation: timestamp bounds, downstream input, quarantine storage

**Context:** Stage 4 inserts `pipeline/validate.py` between `ingest` and
`score`, using pandera (`pandera.pandas`, the current import path - checked
via Context7 rather than assumed) for schema checks plus a language filter.

**Timestamp plausibility bounds:** initially considered using
`window.reference_date` as the upper bound, but checked the actual data
first and found 3 legitimate reviews timestamped later on that same
calendar day - `reference_date` parses to midnight, so it would have
wrongly quarantined them. Used `1995-01-01` (before Amazon reviews existed)
as the lower bound and real wall-clock `pd.Timestamp.now()` as the upper
bound instead - `reference_date` is a windowing concept for the pipeline's
replay simulation, not a truth bound on what timestamp is physically
possible for a review to have.

**Downstream input:** `pipeline/score.py` and `pipeline/aggregate.py` now
both read `data/validated/` instead of `data/ingested/`. Otherwise
quarantining would exclude rows from nothing - scores and aggregate stats
would still include garbage rows validation was supposed to catch.

**Quarantine storage:** each quarantined row is stored with its full
original content as JSON, not just a reason code and a foreign key. A
reason code alone ("non_english") isn't enough to audit or dispute a
decision later; the full row is.

## 2026-09-20 — Language filter: langdetect, seeded, with a short-text exemption

**Context:** needed an English-language filter with no paid APIs, CPU only.

**Options considered:** `langdetect` (popular, but non-deterministic by
default - uses random byte-reading internally); `langid` (deterministic
out of the box, no seed needed); `fasttext` language-id (more accurate, but
requires downloading a ~130MB model - heavier than this filter warrants).

**Choice:** `langdetect`, with `DetectorFactory.seed = 0` set for
determinism. Tested both libraries on a few short/ambiguous samples before
choosing: `langid.classify("Muy bueno")` returned `("en", ...)` - wrong -
while `langdetect` correctly returned `"es"`.

**Short-text exemption:** checked the actual data before setting a
threshold - 16% of this category's reviews (20,893/130,434) are under 5
words. Language detection is unreliable at that length, so rows below 5
words skip the check and pass by default rather than risk quarantining a
meaningful chunk of legitimate short English reviews.

**Known limitation, found by testing, not assumed:** even at exactly the
5-word threshold, `langdetect` can still misfire - `detect("mom my mom
loved it!")` returns `"sk"` (Slovak). One review in this run's window
("mom" / "my mom loved it!", 5 words combined) was quarantined as
`non_english` incorrectly. This is a real, observed false positive, not a
hypothetical one. It's quarantined rather than dropped, so it's auditable
and reversible (raise `validation.language_check_min_words` in config
trades fewer boundary false-positives for letting more short non-English
reviews through unchecked - a real trade-off, not a free fix). Left the
threshold at 5 for now since the false-positive rate observed is low (1/88)
and the row is fully recoverable from the quarantine table.

## 2026-09-20 — Topics: fit once, transform-only daily, never refit automatically

**Context:** Stage 5 adds BERTopic over negative reviews. Topic IDs are
just labels assigned during a specific clustering run (UMAP, then HDBSCAN,
then topics numbered roughly by cluster size). Refitting on a new day's
data - even the same history plus one new day - can produce a different
manifold and different clusters, because both algorithms are global and
non-incremental: there's no "topic 3 from yesterday" concept carried into a
fresh fit.

**What would break if refit daily:** (1) topic 3 on Monday and topic 3 on
Tuesday could mean completely different things, since ranking shuffles with
the data; (2) `config/topic_names.yaml`'s hand-written labels would go
stale silently - the pipeline wouldn't error, it would just attach the
wrong label to the wrong cluster; (3) any week-over-week comparison (the
entire premise of Stage 6 detection) would be comparing unrelated
categories, producing false alarms or missing real drift, silently; (4) it
would also waste compute re-running UMAP+HDBSCAN over accumulating history
every day for no benefit.

**Choice:** `scripts/fit_topics.py` fits once, deliberately, registers the
artifact, and `topics.active_model_sha` is pinned in config exactly like
`model.baseline.active_model_sha` - the same "bump it on purpose" pattern
threaded through every model in this project. `pipeline/topics.py` only
ever calls `.transform()` (confirmed via the installed BERTopic's actual
signature: `transform(documents, embeddings=None, images=None)`), which
assigns new documents into the existing topic space without refitting.

**Verified, not just claimed:** ran `make topics` three times in a row on
the same window; the topic distribution was bit-for-bit identical every
time (`{-1: 2, 20: 1, 13: 1, 7: 1}`).

**Consequence:** a real refit (when topics visibly stop representing new
complaint themes well) is a deliberate, versioned, human-reviewed event -
new registry entry, new sha to pin, and `config/topic_names.yaml` needs
re-editing for the new topic ids. It is never an automatic side effect of a
daily run.

## 2026-09-20 — Topics: full negative population instead of a 20-30k sample; pickle serialization

**Context:** the original spec suggested fitting on a 20-30k sample "for
memory." This category has only ~9,289 negative reviews (rating<=2, pre-
window) in total - checked before writing any fitting code.

**Choice:** used the full population (9,289 rows) rather than subsampling;
there's no memory pressure at this size and no reason to throw away data
that doesn't exist in excess.

**Serialization:** BERTopic's own docs recommend `safetensors` for
safety/size, but state it drops the fitted UMAP/HDBSCAN state - `.transform()`
still works after loading but falls back to a different (cosine-similarity)
assignment method that "can yield different results." Chose `pickle`
instead, since stability is this stage's explicit, named requirement, and
this is a locally-produced artifact (not untrusted third-party data), so
pickle's usual security caveat doesn't apply to this threat model.

## 2026-09-20 — "Negative" defined by real star rating, not a model prediction

**Context:** both `scripts/fit_topics.py` (training) and `pipeline/embed.py`
(daily) need to decide what counts as a "negative" review to route into
topic modeling.

**Choice:** `rating <= 2` - the reviewer's own star rating, the same proxy
label used for the baseline (`pipeline/features.make_binary_labels`) -
rather than a sentiment model's predicted label.

**Consequence:** no leakage concern and no training/serving inconsistency
to worry about, because the star rating isn't something being predicted at
serving time either - a reviewer submits their rating and text together, so
it's always available, identically, in both the historical training
population and the daily scored window. Using a model's predicted
`sentiment_label` instead would have introduced a dependency on which
model's prediction (baseline vs. DistilBERT vs. some combination) counts,
for no benefit.

## 2026-09-20 — Detection: signatures and tests only, no implementation

**Context:** per the working agreement, `pipeline/detect.py` is the one
stage implemented by the author, not Claude - the detection logic (what
counts as degradation, what counts as drift, what threshold triggers an
alert) is what needs to be defensible from first principles in an
interview, not just tested.

**What was written:** two function signatures with full docstrings and
type hints, bodies raising `NotImplementedError` - `detect_spike` (z-score
based anomaly detection, contract: population std ddof=0, flag when
`abs(z) >= z_threshold`, all-False on zero variance) and `compute_psi`
(Population Stability Index, natural log, bins fixed from `reference` and
reused for `current`). Plus `tests/test_detect.py`, written against those
documented contracts before any implementation exists.

**Verified before handing off, not just asserted:** independently
recomputed both hand-derived expected values with a throwaway script
(`numpy` mean/std and PSI terms) to confirm the docstring arithmetic in the
tests is actually correct, and confirmed the exact bin edges used
(`[0, 1/3, 2/3, 1.0]`) produce unambiguous bin membership for the chosen
fixture values via `numpy.histogram` - the test isn't just internally
consistent, it matches what a real implementation would compute.

**Consequence:** `make test` now shows 3 failing tests (`NotImplementedError`,
not assertion failures) until the bodies are filled in - this is the
intended, temporary state, not a regression.

## 2026-09-20 — Drift experiment: harness design

**Context:** `evaluation/drift_experiment.py` replays simulated days,
injects a different category from a configurable day onward, and dumps
score distributions + PSI to CSV. Per the working agreement, this is a
harness only - no interpretation, no plotting, no verdict.

**Injection category:** checked actual file sizes on the HF repo before
picking one - `Office_Products` (5.7GB) and `Patio_Lawn_and_Garden` (7.7GB)
are both far too large; `Gift_Cards` (50.2MB) is small and stylistically
distinct from Digital_Music (delivery/redemption complaints vs. audio/media
quality).

**Per-simulated-day granularity:** checked actual daily volume first -
median ~4 reviews/day near the reference date, with several zero-review
days in a 60-day sample. A literal single calendar day is too sparse for a
stable distribution or PSI computation. Each simulated "day" is instead a
rolling 30-day window (matching production's `lookback_days`) that advances
by one calendar day per step, reusing `pipeline.ingest.resolve_window`/
`filter_window` exactly as production does.

**Injection mechanism:** rather than date-aligning Gift_Cards to the
simulated calendar (fragile - the two categories' real coverage don't
necessarily overlap on the same dates), post-injection days draw a
fixed-seed random sample from Gift_Cards' full pool, sized to match the
first pre-injection window's row count (179, in the run that produced
`evaluation/drift_experiment_results.csv`).

**Never touches production data/ directories:** every simulated day's
ingest -> validate -> score runs entirely in memory via the pipeline's own
pure functions (`filter_window`, `split_valid_and_quarantined`,
`score_with_baseline`, `attach_distilbert_scores`). Writing 60+ simulated
days into the real `data/ingested|validated|scored/` directories would
overwrite-by-partition and corrupt actual pipeline state - reusing the pure
functions in memory gives identical logic ("run the pipeline unchanged")
without that risk.

**Fixed PSI reference, built once:** the reference distribution is pooled
from every pre-injection day's scores, built once after all pre-injection
days are scored (not grown incrementally day-by-day, which would make
early days' "reference" partial and inconsistent with later days').
Applied to every day including pre-injection ones, as a noise-floor
baseline - a known simplification (pre-injection days are partially
compared against a reference that includes themselves) rather than a
held-out split, acceptable for a harness whose numbers the author will
inspect and reason about himself, not a statistically rigorous estimator.

**Verified end-to-end:** ran the harness for 60 simulated days (30 normal +
30 injected). PSI columns correctly read `"pending"` for every row, since
`pipeline.detect.compute_psi` is still `NotImplementedError` (Stage 6 is
unfinished) - confirms the harness's own mechanics (replay, injection,
distribution stats, CSV writing) work correctly independent of that
dependency, exactly as designed.

## 2026-10-03 — Two CI bugs: neither job had ever actually run successfully

**Context:** checked `gh run list` while reviewing project state and found
every CI run since the first push had failed - not from the documented
cold-cache limitation (see the CI workflow's own comments), but from two
more basic bugs upstream of it that meant the designed cache/model-pinning
logic had never actually been exercised.

**Bug 1 - `test` job ran bare `pytest`:** without `-m`, the current
directory isn't put on `sys.path` the same way `python -m pytest` does, so
`from pipeline.X import Y` failed with `ModuleNotFoundError: No module
named 'pipeline'` in every test file. This never surfaced locally because
the Makefile's `test` target always runs `$(PYTHON) -m pytest`, but the
workflow's `test` job called `pytest` directly, bypassing the Makefile
entirely.

**Bug 2 - `pipeline` job's `make` targets hardcoded `.venv/bin/python`:**
every `make ingest`/`score`/etc. step failed immediately with
`.venv/bin/python: No such file or directory`, before reaching the
cache-restore or model-loading logic at all. `actions/setup-python`
installs Python directly onto the runner's PATH; it doesn't create a
project-local `.venv/` the way a local dev setup does.

**Fix:** `Makefile`'s `PYTHON = .venv/bin/python` became `PYTHON ?=
.venv/bin/python` (GNU Make's conditional assignment - verified locally
that an environment variable correctly overrides a `?=` default before
relying on it), and the `pipeline` job in the workflow sets `env: {PYTHON:
python}` at the job level. The `test` job's `run: pytest` became `run:
python -m pytest`. Also removed the `detect` step's `continue-on-error`
and its "expected to fail" comment, since `pipeline/detect.py` is now
fully implemented and a real failure there should fail the job, not be
silently swallowed.

**Consequence:** this is a reminder that an untested CI workflow file is
just a YAML file with good intentions - it needed to actually run (and be
checked with `gh run list`, not just read) before its designed failure
mode could be trusted as the real failure mode.

## 2026-10-03 — Detection: z_threshold=2.0 for per-product spike detection

**Context:** `pipeline/detect.py`'s `main()` calls `detect_spike` on each
eligible product's weekly `review_count` and `mean_rating` series.
`detect_spike`'s own signature defaults to `z_threshold=3.0`, but most
products here only have 3-11 weeks of history (widened from a 30-day to a
200-day ingest window specifically to get enough weekly history for this
to be meaningful at all - see the lookback_days entry above).

**Checked against a real case before deciding, not assumed:** product
`B099BQHKYZ`'s 11-week rating series is `[2.0, 1.0, 5.0, 5.0, 3.0, 5.0,
4.5, 3.0, 5.0, 5.0, 5.0]` (mean ≈ 3.95). At the default `z_threshold=3.0`,
nothing in this series gets flagged at all - not even the genuinely bad
`1.0` week - because with only 11 points, a single outlier inflates the
very standard deviation it's being measured against (the same effect
observed in `tests/test_detect.py`'s hand-computed spike fixture, here
working against sensitivity instead of for it). At `z_threshold=1.5-2.0`,
that `1.0` week correctly gets flagged while the milder `2.0`-rating week
does not.

**Choice:** `z_threshold=2.0` for both series in `main()` - more sensitive
than the function's own default, deliberately, to compensate for how short
these per-product histories are.

**Verified against real alerts after the fact, not just the one test
case:** of the 27 alerts produced, several corroborate each other rather
than looking like noise - e.g. product `B084J7CCY4` fires both
`complaint_volume_spike` and `rating_drop` on the *same* week
(`2023-03-27`: review count jumped to 3 against a mean of 1.33, same
week's mean rating dropped to 3.67 against a mean of 4.78) - one real
event producing two consistent signals, not two unrelated coincidences.

**Consequence:** this threshold is tuned for *short* per-product histories
specifically. If the ingest window grows much larger (more weeks per
product), 2.0 may turn out to be too sensitive and produce noisy alerts on
a series that now has enough points for its std to be a stable estimate -
worth revisiting then, not assumed to be correct forever.

## 2026-10-07 — Found and fixed a real train/serve overlap; widened the window to 2 years

**Context:** user feedback that 27 alerts on a 200-day window "feels
unconvincing." Before just widening the window for appearances, checked
whether widening it further was even safe - and found it wasn't, yet.

**The bug:** when `window.lookback_days` was widened 30&rarr;200 for the
Stage 6 work, nobody checked whether that created overlap with the
baseline's own training population. It did: the baseline was trained on
everything before `2023-08-07` (the window boundary *at training time*),
but the 200-day window scored data back to `2023-02-18` - about 169 days
of data the model had already seen during training were also being scored
by it. This directly contradicts the temporal train/serve split documented
as a deliberate decision in Stage 2. It went unnoticed because nothing
automatically checks this invariant - it has to be checked by hand every
time the window changes.

**The fix, not a shortcut:** widened `lookback_days` to 730 (2 years) and
*retrained the baseline and refit the topic model against the new window
boundary* (`window_start=2021-09-06`), so training data is again strictly
"before" whatever gets scored. New shas pinned: baseline
`f732ce395686...`, topics `d62f4dfa6044...`. Checked real row counts
before picking 2 years over 4: 730 days = 15,469 rows (~7 min of DistilBERT
scoring on this hardware); 1,500 days = 35,441 rows (~17 min) - 2 years was
chosen as a reasonable balance of scale vs. iteration time, not the
largest window physically possible.

**Side effect, expected and consistent, not a bug:** the topic model's
training population *shrank* (9,289&rarr;7,874 negative reviews) even
though the overall window grew. Training population is "everything before
window_start," and pushing the serving window further back in time
necessarily shrinks what counts as "before" it. More weekly history for
detection costs some training population for the topic model - a real
trade-off, not free.

**A second bug found as a direct result of the first fix:** after
retraining and rerunning `make detect`, the `alerts` table had 459 rows,
not the 449 the script printed. 27 stale alerts from the *previous* run
(under the old 200-day window) were still in the table; `insert or ignore`
only prevents an exact duplicate `(product_id, week_start, alert_type)`
triple - it has no mechanism to remove an alert that no longer holds once
more historical context changes a product's statistical baseline (a
product's z-score baseline shifts when its available history changes from
3 weeks to 2 years, so a week flagged under the short series can stop
being an outlier under the long one, and vice versa). Cleaned up the stale
rows manually this time (`delete from alerts where detected_at = ...`),
confirmed `make detect` is idempotent at the new scale (449, rerun twice,
stays 449).

**Known gap, not fixed here - this is `pipeline/detect.py`'s own logic to
own:** `main()` does not currently reconcile the `alerts` table when the
detection configuration changes (widened window, retrained model). It only
appends new alerts and ignores exact-duplicate ones. A `main()` that
deleted every alert not present in the current run's result set before
inserting (mirroring how `aggregate.py` already upserts `product_week_stats`
instead of accumulating it forever) would close this gap properly.

**Result:** 969 eligible products (up from 102), 449 real alerts (259
`rating_drop`, 190 `complaint_volume_spike`, up from 27). Verified real,
not assumed: full test suite still passes, `metric_value` is clean `real`
type throughout (no recurrence of the earlier numpy/sqlite3 blob bug),
and detection PSI dropped to 0.0042/0.0032 (baseline/DistilBERT) - smaller
than before (0.028/0.017), consistent with a much larger, more stable
reference/current split (7,501 rows each vs. 1,103 before) producing less
sampling noise.

## 2026-10-09 — `complaint_volume_spike` fired on a product with zero complaints

**Context:** inspecting a real alert from the dashboard - `B07D5ZGJF1`,
`complaint_volume_spike`, metric_value `2`. Checked its full weekly
history: `mean_rating` is `5.0` on every single week, including the
flagged one. The alert fired on a product that has never once had a bad
review.

**The bug:** `count_bad` in `main()` only checked whether `review_count`
spiked above the product's own average - it never checked whether the
reviews behind that spike were actually negative. Any week where volume
jumped relative to a product's own (often very small) baseline got called
a "complaint" spike, even when every review that week was 5 stars. For
`B07D5ZGJF1`, going from its usual 1 review/week to 2 was already a large
*relative* jump on such a small baseline, which was enough to clear
`z_threshold=2.0` regardless of sentiment.

**The fix:** added a third condition to `count_bad` - the same week's
`mean_rating` must also be below that product's own average rating,
alongside the existing volume-spike and above-average-count checks. A
volume spike made entirely of good reviews now correctly produces no
alert.

**A second, real demonstration of the stale-alerts gap** (first found
2026-10-07, still not fixed - see above): after applying the fix, fresh
detection correctly dropped from 449 to 313 alerts (`complaint_volume_spike`
190&rarr;54), but the `alerts` table still showed 449 immediately after
rerunning `make detect` - the 136 now-invalid alerts, including
`B07D5ZGJF1`'s, were still sitting in the table from before the fix, and
`insert or ignore` found nothing new to insert since the 313 correct
alerts were already a subset of the 449 stale ones. Had to `delete from
alerts` and let one fresh run repopulate it from scratch to actually see
the corrected 313. This is the exact same root cause as before, now
triggered by a logic change instead of a config change - two independent
real occurrences of the same gap is a stronger argument for actually
fixing `main()`'s reconciliation than either one alone.

**Consequence:** 313 real alerts (54 `complaint_volume_spike`, 259
`rating_drop`), down from 449. Verified: idempotent across reruns, full
test suite passes, `B07D5ZGJF1` confirmed absent from fresh output.

## 2026-10-09 — Split `rating_flags` to `z_threshold=4.0`, separate from `count_flags`'s 2.0

**Context:** the 2026-10-03 entry above set `z_threshold=2.0` for *both*
series in `main()`. At that setting, `rating_drop` alone produced 259 of
the 313 alerts - the large majority of all alerts, on a check that, unlike
`complaint_volume_spike`, had no secondary corroborating condition (no
volume check, just the rating series flagging itself). That imbalance was
reason enough to reconsider `rating_flags` on its own, separately from
`count_flags`, rather than assume one shared threshold was still right for
both now that the window is 2 years instead of 30-200 days.

**A real mechanical issue surfaced while deciding this, not a design
choice:** across several "run the code" turns, `rating_flags`'s
`z_threshold` was observed at both `2.0` and `4.0` in the committed file
without an explicit edit being made in the session each time - confirmed
via `git show 01686aa:pipeline/detect.py`, which already had `4.0` at a
commit point where the accompanying numbers (313 total, 259 `rating_drop`)
had been verified against `2.0`'s output. That mismatch means the 313/259
figures recorded in the 2026-10-09 entry above describe `z_threshold=2.0`'s
behavior, not necessarily the file's exact state at every point since.
Flagging this plainly rather than quietly reconciling it: the numbers below
are freshly verified against the file's current, actual contents, not
inferred from history.

**Choice, asked directly and answered explicitly:** keep `count_flags` at
`2.0` (unchanged - still tuned for short per-product histories, per the
2026-10-03 entry) and set `rating_flags` to `4.0`, stricter than before.
This is now two different thresholds for two different series, not one
shared value - `mean_rating` series apparently produce enough borderline
z-scores at `2.0` to flag the large majority of eligible products, where
`review_count` does not.

**Verified against a fresh run, alerts table cleared and rebuilt from
scratch (not just a stale-alerts patch):** 66 total alerts - 54
`complaint_volume_spike` (unchanged, since `count_flags` didn't change) and
12 `rating_drop` (down from 259). Confirmed idempotent (`make detect` rerun
twice, count stays 66).

**Consequence:** `rating_drop` alerts are now rare and, by construction,
only fire on genuinely extreme rating swings rather than any one week on
either side of a product's own mean. This trades sensitivity for
precision - a product with a real, moderate rating decline over a few
weeks may no longer surface a `rating_drop` alert at all. Whether `4.0` is
the right trade long-term, versus a per-product-history-length-aware
threshold, is not re-derived here from first principles - it's the
author's explicit choice after seeing both outputs side by side, and is
revisitable the same way `2.0` was revisitable above.

## 2026-10-10 — CI's `pipeline` job never actually succeeded; added cache-miss training

**Context:** a full project audit, checking real state rather than re-reading
docs. `gh run list` across the repo's entire history (30 runs, from the first
commit through today) showed the `pipeline` job failing on every single run,
with the same `FileNotFoundError` on the baseline `.joblib` each time. The
`test` job genuinely passes (the 2026-10-03 CI fix was real), but `pipeline`
never had.

**The bug in the workflow's own documentation, not just the workflow:** the
comment above the `pipeline` job described this as an occasional cold-cache
event - "GitHub evicts it [the cache] after ~7 days unused." That framing was
never checked against the actual run history. The real cause is structural,
not occasional: nothing in the workflow ever *trains* `models/*.joblib` /
`*.pkl` - `scripts/train_baseline.py` and `scripts/fit_topics.py` are
deliberately one-off, never wired into CI - so `actions/cache` was restoring
from a cache slot that had never once been populated, by anyone, in this
repo's history. A 100% failure rate was reported in prose as "occasional."

**The fix:** added two steps to the `pipeline` job, both gated on
`steps.cache.outputs.cache-hit != 'true'` (cache miss only - a cache hit
means the artifacts already exist and were already verified against the pin
by a previous run): train the baseline and fit the topic model using the
exact same scripts a human runs locally, then verify the freshly-produced
artifact's filename (which encodes its sha256) matches
`config/pipeline.yaml`'s pinned `active_model_sha` / topics sha before
continuing. A mismatch fails the job loudly rather than silently scoring with
an undocumented model - preserving the same "no silent model swap" guarantee
Stages 2/5 exist for, while letting CI self-heal on a cold cache instead of
failing forever.

**Known residual risk, deliberately not solved here:** scikit-learn training
is only bit-for-bit deterministic given identical inputs *and* identical
floating-point behavior - different BLAS backends (this repo was developed
on macOS; the runner is Ubuntu) can produce slightly different coefficients
after the same fit, which would change the serialized artifact's bytes and
therefore its sha256, with no real difference in what the model actually
does. This fix can't distinguish that from a genuine problem (e.g. a stale
pin, or training inputs that changed) - it fails loudly in both cases rather
than guessing, and a mismatch needs a human to look at it rather than being
retried automatically.

**Verified locally, not just read:** re-ran the exact verification snippet
against the current pinned shas (`f732ce395686...` baseline,
`d62f4dfa6044...` topics) - both resolve to real, present artifact files.
The new CI steps themselves are only provable by an actual CI run (ubuntu
runner, not this macOS dev machine) - that's the next real check, not
assumed from local success.

**That CI run happened immediately (same day) and the predicted residual
risk occurred for real, not just in theory:** the Ubuntu runner's freshly
trained baseline used the identical 87,403/21,851 train/test split as the
pinned model (same `random_state`, same row counts), but produced different
metrics - accuracy `0.9502539929522676` vs. the pinned model's
`0.9507574024072125` - a small, real floating-point difference, not a data
or logic bug. Different metrics means different coefficients means a
different serialized artifact means a different sha256
(`7b653674e222...` vs. the pinned `f732ce395686...`). The verification step
did exactly its job: failed loudly instead of silently continuing with a
different model. The job is not green, but it is now failing for an honest,
diagnosed reason instead of a confusing one.

**Resolution, same day:** exact-byte-hash equality and "retrain fresh on
different hardware" are fundamentally incompatible, so the check was
loosened from sha equality to a metric tolerance - baseline accuracy/f1
within 0.005 of the pinned entry's; topics within 5 topics and 0.05 outlier
rate of the pinned entry's. The topics side needed its own check since
BERTopic has no single accuracy-like metric: compared `n_topics` and
outlier rate instead, both already recorded in `topics_registry.json`.
Verified against the real numbers this same CI run produced - topics came
back as 48 topics / 3,138 outliers vs. the pinned 47 / 3,270 (training rows
identical, 7,874), both well inside tolerance - confirming UMAP/HDBSCAN
have the same kind of small cross-platform variance as the baseline's
LogisticRegression.

When metrics are within tolerance, the run's in-workspace config (never
committed, never pushed) is repointed at the freshly-trained sha so
`make score` / `make topics` have an artifact that actually exists on this
runner - the git-committed pin stays exactly as the human set it, and
remains authoritative for local/production use. A tolerance breach still
fails the job loudly; this only closes the gap for the ordinary case of
"functionally the same model, different bytes."

## 2026-10-18 — Committed the two pinned model artifacts, for free hosting

**Context:** wanted to put the dashboard on free hosting (Streamlit
Community Cloud / Hugging Face Spaces) so the project has a live link, not
just a repo. `dashboard/app.py` reads `data/warehouse.db` and Parquet under
`data/scored` / `data/topics`, none of which exist on a fresh clone -
`data/` and `*.db` are gitignored on purpose everywhere else in this
project.

**Options considered:**
1. Commit a snapshot of the generated data itself (`data/warehouse.db` +
   the Parquet layers, ~15-20MB) - fastest for the host, but then the
   "live" dashboard is really just replaying a frozen snapshot forever,
   which undercuts the "scheduled, real" framing of the whole project.
2. Re-run the full pipeline on deploy, loading the pinned model artifacts
   rather than retraining them (retraining-on-deploy would hit the exact
   cross-platform floating-point issue documented in the CI entries
   above, except now on a visitor's first page load instead of a
   scheduled CI run - clearly worse).
3. Re-run the full pipeline on deploy AND train the models fresh each
   time - rejected outright: 15+ minutes of cold-start latency for
   whoever opens the app first, for no benefit over loading the pin.

**Choice:** option 2. Added `dashboard/bootstrap.py`
(`ensure_data_built`, wrapped in `st.cache_resource` so it only runs once
per container lifetime, not once per visitor) that detects a missing
`data/warehouse.db` and runs `ingest -> detect` fresh via the same stage
entrypoints `make` uses, loading the two artifacts pinned in
`config/pipeline.yaml` rather than retraining - same "load the pin, don't
guess" rule as every other stage. Carved a narrow exception into
`.gitignore` for exactly those two files (`tfidf_logreg_baseline-
f732ce395686.joblib`, `bertopic-d62f4dfa6044.pkl`, ~49MB total) rather
than committing all of `models/`, so the old, unpinned versions of both
stay out of git.

**Real, mostly-irreversible consequence, stated plainly:** this adds
~49MB of binary model weights to git history permanently (removable later
only via a history rewrite, which breaks every existing clone/fork). The
exception in `.gitignore` is manual and will silently go stale if the pin
in `config/pipeline.yaml` ever changes without a matching update here -
there is no test enforcing the two stay in sync. First page load after any
cold deploy (first visit, or after the host recycles the container) will
be slow - locally, DistilBERT scoring alone took ~9 minutes; a free host's
CPU is an unknown variable in the other direction. Verified locally before
committing: `ensure_data_built()`'s fast path (data already present) adds
zero overhead, and its subprocess/status mechanics were dry-run tested
independently of a full pipeline execution.

**Update, same day - deployed for real, and the live rebuild failed twice,
differently, with no visible error either time:** first deploy produced a
fully blank page (empty `streamlitApp` DOM, confirmed via reading the page
tree directly, not just a screenshot) that never changed across several
checks and a hard reload. Rebooting the app (Manage app -> Reboot)
produced a second, different failure: the page rendered normally this
time, but only the two numbers that come from already-committed config
(47 topics, baseline F1 0.973 - both read straight from
`models/topics_registry.json` / `models/registry.json`, needing no live
run at all) were populated. The numbers that require a real pipeline run
- flagged products, negative reviews analyzed, the `alerts` table itself -
never appeared, with no error, no traceback, and the script confirmed
(via repeated `use_container_width` deprecation warnings in the logs)
to have completed its run multiple times without raising. Streamlit
Cloud's free-tier CPU throttle was active for both attempts (confirmed via
its own "Your app has been throttled" banner).

**Root cause not confirmed - no shell access to the container to inspect
further**, which is itself the real lesson: a live rebuild-on-deploy has
no equivalent of `gh run view --log-failed` the way CI does, so a silent
partial failure here is much harder to diagnose than the same class of
problem was in CI. Rather than keep debugging blind, switched to
committing the actual, real, already-verified data instead of trying to
reproduce it live on a resource-constrained free host: `data/warehouse.db`
(2.1MB) and `data/topics/` (8.9MB, 574 date-partitioned Parquet files,
matching the pipeline's own overwrite-by-partition layout rather than
collapsing it into one file). `dashboard/app.py` reads only these two
paths - not `data/raw`, `data/scored`, or anything else - so this is the
minimum needed, not a wholesale reversal of keeping `data/` gitignored.

**The `.gitignore` negation gotcha, found while doing this:** a blanket
`data/` directory-level ignore makes a later `!data/warehouse.db`
file-level negation silently do nothing - git won't descend into an
already-ignored directory to evaluate per-file exceptions. Fixed by
replacing the blanket rule with explicit per-subdirectory ignores
(`data/raw/`, `data/ingested/`, `data/validated/`, `data/scored/`,
`data/embedded/`), leaving `data/warehouse.db` and `data/topics/`
untouched by any ignore rule at all rather than trying to negate one.
Verified with `git check-ignore` and a real `git add data/` before
trusting it, not assumed from reading the pattern.

**`dashboard/bootstrap.py` is left in place, now as a true fallback
only** - harmless no-op given the committed data, but still real code for
the scenario it was built for (a fork or clone without the committed
snapshot). Its core lesson stands even though the deploy strategy moved
away from relying on it: a live rebuild on a resource-constrained free
host is unobservable in a way a scheduled CI job isn't, which matters more
than the live-vs-snapshot distinction itself.
