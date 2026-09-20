import pandas as pd

from pipeline.ingest import filter_window, resolve_window


def test_resolve_window_is_rolling_and_half_open():
    reference = pd.Timestamp("2023-09-06")
    start, end = resolve_window(reference, lookback_days=3)
    assert start == pd.Timestamp("2023-09-03")
    assert end == reference


def test_filter_window_keeps_only_rows_in_range():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-09-01", "2023-09-03", "2023-09-06"]),
            "rating": [5, 1, 3],
        }
    )
    out = filter_window(df, pd.Timestamp("2023-09-02"), pd.Timestamp("2023-09-06"))
    assert list(out["rating"]) == [1]
    assert list(out["review_date"]) == ["2023-09-03"]


def test_filter_window_end_is_exclusive():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-09-06"]),
            "rating": [3],
        }
    )
    out = filter_window(df, pd.Timestamp("2023-09-01"), pd.Timestamp("2023-09-06"))
    assert out.empty
