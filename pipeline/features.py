"""Shared, pure feature/label construction used by both training and scoring.

Kept in one place so the model that gets trained and the model that gets
served always agree on what the input text and the label mean.
"""

import pandas as pd


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Concatenate title + body text.

    Title alone often carries strong sentiment ("Awful", "Do not buy") that
    gets diluted once mixed into a longer body; keeping both gives the
    vectoriser the clearest signal available.
    """
    title = df["title"].fillna("")
    text = df["text"].fillna("")
    return (title + " " + text).str.strip()


def make_binary_labels(ratings: pd.Series) -> pd.Series:
    """Star-rating proxy labels: 1-2 stars -> 0 (negative), 4-5 stars -> 1
    (positive), 3 stars -> NaN (caller drops these rows).

    Star rating is a proxy for sentiment, not sentiment itself - a 3-star
    review is often genuinely mixed rather than mislabeled, so it's dropped
    rather than forced into either class.
    """
    labels = pd.Series(index=ratings.index, dtype="float64")
    labels[ratings <= 2] = 0.0
    labels[ratings >= 4] = 1.0
    return labels
