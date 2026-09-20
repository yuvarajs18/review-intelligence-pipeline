# Review Intelligence Pipeline
## What this is
A scheduled NLP monitoring system over e-commerce reviews. It detects what
customers complain about per product per week, and alerts when a product
degrades or when the model itself drifts.
This is a portfolio project for an MSc Data Science student. The author must be
able to defend every design decision in a job interview.
## Hard constraints
- No paid APIs. All models run locally via transformers / sentence-transformers.
- CPU only. Assume 8GB RAM. No CUDA.
- Storage: Parquet for data, DuckDB for analytics, SQLite for state. No server DBs.
- Python 3.11+. Dependencies pinned.
## Working agreement
- Explain the approach and wait for approval before writing code for a new stage.
- Prefer stdlib and well-known libraries over clever abstractions.
- Every stage is a module in pipeline/ with a CLI entrypoint and pure functions
  that can be unit tested without touching disk.
- Never write the contents of evaluation/ without being explicitly asked.
  The author writes the evaluation logic himself.
- When a design decision has more than one reasonable option, stop and list the
  options with trade-offs rather than picking silently.
## Documentation requirements
- Every non-obvious decision gets an entry in docs/decisions.md:
  context, options considered, choice, consequence. Two or three sentences.
- Every function that implements a statistical method carries a docstring
  explaining the method in plain English, not just the parameters.
## Conventions
- Config lives in config/pipeline.yaml. No magic numbers in code.
- Every prediction row records the model name and version hash.
- Log to the runs table: timestamp, rows in, rows quarantined, latency,
  model versions, score distribution summary.
## Commands
make ingest | score | topics | aggregate | detect | dashboard | test
Show less












