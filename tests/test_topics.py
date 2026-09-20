import numpy as np
import pandas as pd

from pipeline.topics import assign_topics


class _FakeTopicModel:
    def transform(self, texts, embeddings=None):
        # Deterministic stand-in for BERTopic: assigns topic 0 to everything,
        # just to prove assign_topics wires columns together correctly -
        # this test isn't about clustering quality.
        return [0 for _ in texts], [1.0 for _ in texts]


def test_assign_topics_attaches_topic_id_and_drops_embedding_column():
    df = pd.DataFrame(
        {
            "title": ["bad", "terrible"],
            "text": ["did not work", "broke immediately"],
            "embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
        }
    )

    out = assign_topics(df, _FakeTopicModel(), model_sha256="deadbeef")

    assert "embedding" not in out.columns
    assert out["topic_id"].tolist() == [0, 0]
    assert out["topic_model_version"].tolist() == ["deadbeef", "deadbeef"]
