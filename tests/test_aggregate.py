import duckdb
import pandas as pd

from pipeline.aggregate import compute_weekly_aggregate


def test_compute_weekly_aggregate_counts_and_averages_per_product_week():
    df = pd.DataFrame(
        {
            "parent_asin": ["A", "A", "B"],
            "timestamp": pd.to_datetime(["2023-09-04", "2023-09-05", "2023-09-04"]),  # same ISO week
            "rating": [4, 2, 5],
        }
    )
    con = duckdb.connect()
    con.register("reviews", df)

    result = compute_weekly_aggregate(con, table="reviews")

    a_row = result[result["product_id"] == "A"].iloc[0]
    assert a_row["review_count"] == 2
    assert a_row["mean_rating"] == 3.0

    b_row = result[result["product_id"] == "B"].iloc[0]
    assert b_row["review_count"] == 1
    assert b_row["mean_rating"] == 5.0


def test_compute_weekly_aggregate_splits_different_weeks():
    df = pd.DataFrame(
        {
            "parent_asin": ["A", "A"],
            "timestamp": pd.to_datetime(["2023-09-04", "2023-09-15"]),  # different ISO weeks
            "rating": [4, 2],
        }
    )
    con = duckdb.connect()
    con.register("reviews", df)

    result = compute_weekly_aggregate(con, table="reviews")

    assert len(result) == 2
    assert set(result["review_count"]) == {1, 1}
