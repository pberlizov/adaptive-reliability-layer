import numpy as np

from adaptive_reliability_layer.regime import StreamingRegimeEncoder, build_regime_embedding


def test_regime_encoder_recognizes_reinforced_recurring_regime() -> None:
    encoder = StreamingRegimeEncoder(max_prototypes=4)
    embedding = build_regime_embedding(
        np.array([0.9, -0.2, 0.4, 0.1], dtype=np.float64),
        np.array([0.3, 0.2, -0.1], dtype=np.float64),
    )

    descriptor = encoder.register(embedding, step=1)
    encoder.reinforce(descriptor, step=2, reward=0.82, successful=True)

    recurring = encoder.describe(embedding, step=3)
    assert recurring.prototype_index >= 0
    assert recurring.familiar
    assert recurring.recurrence_confidence >= 0.58
    assert recurring.novelty_score <= 0.45


def test_regime_encoder_marks_distant_embedding_as_more_novel() -> None:
    encoder = StreamingRegimeEncoder(max_prototypes=4)
    familiar = build_regime_embedding(
        np.array([1.0, 0.2, -0.3, 0.4], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    )
    novel = build_regime_embedding(
        np.array([-0.8, 0.7, 0.5, -0.6], dtype=np.float64),
        np.array([0.9, -0.7], dtype=np.float64),
    )

    descriptor = encoder.register(familiar, step=1)
    encoder.reinforce(descriptor, step=2, reward=0.78, successful=True)

    novel_descriptor = encoder.describe(novel, step=3)
    assert novel_descriptor.novelty_score > 0.20
    assert novel_descriptor.recurrence_confidence < 0.70
