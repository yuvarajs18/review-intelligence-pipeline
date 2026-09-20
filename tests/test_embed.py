import numpy as np
import pandas as pd

from pipeline.embed import embed_reviews, filter_negative


def test_filter_negative_keeps_only_rating_1_and_2():
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5]})
    out = filter_negative(df)
    assert out["rating"].tolist() == [1, 2]


class _FakeEncoder:
    def encode(self, texts, show_progress_bar=False):
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_embed_reviews_attaches_one_embedding_per_row():
    df = pd.DataFrame({"title": ["bad"], "text": ["did not work"]})
    out = embed_reviews(df, _FakeEncoder())

    assert len(out) == 1
    assert len(out["embedding"].iloc[0]) == 2
