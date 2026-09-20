"""Tests for pipeline/detect.py, written against the contract documented in
that module's docstrings BEFORE any implementation exists. All three will
fail with NotImplementedError until the bodies are filled in - that's
expected, not a bug in these tests.

Every expected value below is hand-computed, not just "whatever the code
produces" - see the comments for the arithmetic.
"""

import pandas as pd
import pytest

from pipeline.detect import compute_psi, detect_spike


def test_detect_spike_flat_series_has_no_spikes():
    # Zero variance - the "no signal at all" edge case. Must return
    # all-False rather than raising on a 0/0 z-score.
    series = pd.Series([10.0] * 10)

    flags = detect_spike(series, z_threshold=3.0)

    assert not flags.any()


def test_detect_spike_flags_the_known_outlier():
    # 9 points at 10.0, one at 100.0.
    # mean = (9*10 + 100) / 10 = 19.0
    # population variance = ((9 * (10-19)**2) + (100-19)**2) / 10
    #                      = (9*81 + 6561) / 10 = (729 + 6561) / 10 = 729.0
    # population std = sqrt(729) = 27.0 exactly
    # z-score of the outlier: (100 - 19) / 27 = 81 / 27 = 3.0 exactly
    # z-score of a normal point: (10 - 19) / 27 = -9 / 27 = -1/3
    series = pd.Series([10.0] * 9 + [100.0])

    flags = detect_spike(series, z_threshold=3.0)

    assert flags.tolist() == [False] * 9 + [True]


def test_compute_psi_matches_hand_computation():
    # reference: 40% in bin0 [0, 1/3), 30% in bin1 [1/3, 2/3), 30% in bin2 [2/3, 1]
    reference = pd.Series([0.1] * 40 + [0.5] * 30 + [0.9] * 30)
    # current: 20% / 30% / 50% in the same three bins
    current = pd.Series([0.1] * 20 + [0.5] * 30 + [0.9] * 50)
    bins = [0.0, 1 / 3, 2 / 3, 1.0]

    # PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)) per bin:
    #   bin0: (0.20 - 0.40) * ln(0.20 / 0.40) = -0.20 * ln(0.5)  = 0.13862943611198905
    #   bin1: (0.30 - 0.30) * ln(0.30 / 0.30) =  0.00 * ln(1.0)  = 0.0
    #   bin2: (0.50 - 0.30) * ln(0.50 / 0.30) =  0.20 * ln(5/3)  = 0.10216512475319814
    expected_psi = 0.13862943611198905 + 0.0 + 0.10216512475319814  # == 0.2407945608651872

    psi = compute_psi(reference, current, bins=bins)

    assert psi == pytest.approx(expected_psi, rel=1e-6)
