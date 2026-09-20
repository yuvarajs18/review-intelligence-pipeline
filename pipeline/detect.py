"""Detect: product degradation and model/topic drift alerts.

Signatures, docstrings and type hints only - every body raises
NotImplementedError. This is the one stage the author (Yuvaraj) implements
himself, per the working agreement: the detection logic is what he needs to
be able to defend from first principles in an interview, not just from
having it pass tests someone else wrote.

Two primitives:
- detect_spike: per-series anomaly detection (e.g. a product's weekly
  review count, or its mean rating) - flags points that deviate unusually
  far from the rest of the series, in either direction.
- compute_psi: Population Stability Index between two score distributions
  (e.g. sentiment scores from two different weeks) - the standard way to
  ask "has this distribution shifted enough to worry about drift?"

main() is left to wire these primitives to the warehouse (config.paths
.warehouse) and whatever alert output format is decided - also his call.
"""

from collections.abc import Sequence

import pandas as pd


def detect_spike(series: pd.Series, z_threshold: float = 3.0) -> pd.Series:
    """Flags points in `series` that are unusual outliers relative to the
    rest of the series, in either direction - e.g. a product's weekly
    complaint count spiking upward, or its mean rating dropping sharply.

    Contract this must satisfy (see tests/test_detect.py, written against
    this contract before any implementation exists):
    - Compute `series`'s own mean and POPULATION standard deviation
      (ddof=0) - i.e. divide the sum of squared deviations by n, not n-1.
    - For each point, compute its z-score: (value - mean) / std.
    - Flag a point (True) when abs(z-score) >= z_threshold; otherwise
      False. Note the >=, not > - a point sitting exactly at the
      threshold counts as a spike.
    - If `series` has zero variance (std == 0 - e.g. a perfectly flat
      series with no signal at all), no point is a spike: return an
      all-False Series rather than raising or dividing by zero.

    Returns a boolean pd.Series aligned with `series`'s index (same
    length, same index labels).
    """
    raise NotImplementedError


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int | Sequence[float] = 10,
) -> float:
    """Population Stability Index (PSI): how much has `current`'s
    distribution shifted away from `reference`'s?

    In plain English: split the score range into bins. Work out what
    percentage of `reference`'s observations fall into each bin, then do
    the same for `current`. For every bin, compare the two percentages: a
    bin whose share barely changed contributes almost nothing; a bin whose
    share moved a lot contributes a lot. Summing those contributions
    across all bins gives one number describing how different the two
    distributions are overall.

    Formula, per bin i, using the natural log:
        term_i = (current_pct_i - reference_pct_i) * ln(current_pct_i / reference_pct_i)
    and PSI = sum(term_i) over all bins.

    Why bins are fixed from `reference`, never recomputed from `current`:
    the bins have to represent a fixed frame of reference so that a real
    shift shows up as observations migrating between fixed buckets. If you
    instead recomputed bin edges from `current`'s own distribution each
    time (e.g. its own quantiles), `current` would - by construction -
    always come out looking roughly evenly spread across its own bins,
    which would hide exactly the shift PSI exists to detect. So: if
    `bins` is an int, derive that many equal-width bin edges from
    `reference`'s own min/max, and reuse those exact edges for `current`.
    If `bins` is already a sequence of edges, use them as given for both.

    Known edge case to handle: a bin with zero observations in either
    distribution makes ln(0) undefined - needs explicit handling (e.g. a
    small epsilon, or excluding/merging that bin) so this never crashes on
    sparse data.

    Common rule of thumb, useful context but not enforced by this
    function: PSI < 0.1 is usually read as "no significant shift", 0.1-0.25
    as "moderate", and > 0.25 as "significant" - deciding what threshold
    should actually raise an alert is a separate judgment call.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entrypoint - reads the warehouse (config.paths.warehouse),
    applies detect_spike / compute_psi to whichever weekly product stats
    and score distributions matter, and writes alerts wherever they should
    live (a new `alerts` table is a natural fit alongside `quarantine` and
    `runs`, but the schema and the alerting rules themselves are a design
    decision to make deliberately, not to inherit from this stub).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
