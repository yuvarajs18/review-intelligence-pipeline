"""Builds the data dashboard/app.py reads, if it doesn't exist yet.

On a fresh clone (e.g. a free hosting deploy, where data/ is gitignored
and therefore empty), this runs the full ingest -> detect pipeline once,
using the two pinned model artifacts committed to the repo
(models/tfidf_logreg_baseline-f732ce395686.joblib,
models/bertopic-d62f4dfa6044.pkl - see .gitignore). It never retrains:
pipeline.score and pipeline.topics only ever load the sha pinned in
config/pipeline.yaml, the same "load the pin, don't guess" rule as every
other stage - matching how a local dev machine or CI sets this up, just
triggered lazily instead of via `make`.

Cached with st.cache_resource so this runs once per container lifetime,
not once per visitor - the first person to open the app after a cold
deploy waits for the full run (DistilBERT scoring alone took ~9 minutes
on the dev machine; a free host's CPU may be slower or faster), everyone
after that gets the cached result until the container restarts.
"""

import subprocess
import sys
from pathlib import Path

import streamlit as st

PIPELINE_STAGES = ["ingest", "validate", "score", "embed", "topics", "aggregate", "detect"]


def _data_exists() -> bool:
    return Path("data/warehouse.db").exists()


def _run(cmd: list[str], status) -> None:
    status.write(f"`{' '.join(cmd)}`")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        status.write(f"**failed** (exit {result.returncode}):\n```\n{result.stderr[-2000:]}\n```")
        raise RuntimeError(f"{' '.join(cmd)} failed - see output above")


@st.cache_resource(show_spinner=False)
def ensure_data_built() -> None:
    if _data_exists():
        return

    with st.status(
        "First load on this deploy: building pipeline data from scratch "
        "(downloads the dataset + pretrained models, then runs every "
        "stage using the pinned baseline/topic model committed to the "
        "repo - this can take several minutes, only happens once).",
        expanded=True,
    ) as status:
        _run([sys.executable, "scripts/download_data.py"], status)
        _run([sys.executable, "scripts/download_model.py"], status)
        for stage in PIPELINE_STAGES:
            _run([sys.executable, "-m", f"pipeline.{stage}"], status)
        status.update(label="Data ready.", state="complete", expanded=False)
