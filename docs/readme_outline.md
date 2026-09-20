# README outline

Structure for the top-level README.md. Each section's actual content -
especially any number - comes from the author's own runs, never invented.
Sections marked TKTK in README.md are placeholders to be filled in by hand.

1. **Title + one-line description** - what this is, in one sentence.
2. **Overview** - the problem (per-product weekly complaint monitoring +
   drift detection), who it's for (portfolio/interview context), and the
   hard constraints (no paid APIs, CPU-only, 8GB RAM).
3. **Architecture** - the pipeline stages in order (ingest -> validate ->
   score -> embed -> topics -> aggregate -> detect), what each reads/writes,
   and the storage choices (Parquet / DuckDB / SQLite).
4. **Setup** - venv/dependency install, `scripts/download_data.py`,
   `scripts/download_model.py`, one-time training (`scripts/train_baseline.py`,
   `scripts/fit_topics.py`).
5. **Usage** - the `make` targets, in the order they're meant to run.
6. **Key design decisions** - a short curated list of the 3-5 most
   interview-relevant decisions, each linking to its full entry in
   `docs/decisions.md` rather than duplicating the reasoning here.
7. **Results** - baseline vs. DistilBERT comparison, topic examples, drift
   experiment findings. Every number here is TKTK until filled in from a
   real run - never a placeholder that looks like a real result.
8. **Limitations** - known gaps, honestly stated (e.g. CI's cold-cache
   behavior, the language-filter false positive documented in decisions.md,
   BERTopic's outlier rate).
9. **How this was built** - the Claude Code / human split: scaffolding and
   boilerplate vs. the evaluation and detection logic written by hand. See
   the interview-honesty framing this project was built around.
