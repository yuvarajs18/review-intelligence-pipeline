import pandas as pd

from pipeline.features import build_text_column, make_binary_labels


def test_build_text_column_concatenates_title_and_text():
    df = pd.DataFrame({"title": ["Great"], "text": ["works well"]})
    out = build_text_column(df)
    assert out.iloc[0] == "Great works well"


def test_build_text_column_handles_missing_values():
    df = pd.DataFrame({"title": [None], "text": ["works well"]})
    out = build_text_column(df)
    assert out.iloc[0] == "works well"


def test_make_binary_labels_maps_ratings_and_drops_neutral():
    ratings = pd.Series([1, 2, 3, 4, 5])
    labels = make_binary_labels(ratings)
    assert labels.tolist()[:2] == [0.0, 0.0]
    assert labels.tolist()[3:] == [1.0, 1.0]
    assert pd.isna(labels.iloc[2])
