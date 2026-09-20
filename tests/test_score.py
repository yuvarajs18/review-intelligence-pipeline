import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.score import attach_distilbert_scores, score_with_baseline


def _toy_model() -> Pipeline:
    texts = ["great amazing love it", "terrible awful hate it", "great love it", "terrible hate it"]
    labels = [1, 0, 1, 0]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    model.fit(texts, labels)
    return model


def test_score_with_baseline_attaches_sentiment_and_model_identity():
    df = pd.DataFrame(
        {
            "title": ["great", "terrible"],
            "text": ["amazing love it", "awful hate it"],
            "timestamp": pd.to_datetime(["2023-08-10", "2023-08-11"]),
        }
    )
    model = _toy_model()

    out = score_with_baseline(df, model, model_name="toy", model_version="deadbeef")

    assert list(out["baseline_model_name"]) == ["toy", "toy"]
    assert list(out["baseline_model_version"]) == ["deadbeef", "deadbeef"]
    assert out["baseline_sentiment_label"].tolist() == [1, 0]
    assert out["baseline_sentiment_score"].between(0, 1).all()


def test_attach_distilbert_scores_normalizes_to_p_positive():
    df = pd.DataFrame({"text": ["great", "terrible"]})
    # Raw HF pipeline output: "score" is confidence in the predicted label,
    # not P(positive) - a confident NEGATIVE prediction should convert to a
    # low positive-class score, not be left as a high confidence value.
    raw_outputs = [
        {"label": "POSITIVE", "score": 0.95},
        {"label": "NEGATIVE", "score": 0.90},
    ]

    out = attach_distilbert_scores(df, raw_outputs)

    assert out["distilbert_sentiment_score"].tolist() == pytest.approx([0.95, 0.10])
    assert out["distilbert_sentiment_label"].tolist() == [1, 0]
