"""Unit tests for the pure vector maths in app.streaming.embeddings -
pooling, normalising, cosine similarity, and bank matching. None of this
loads a model; `embed_audio` (the one function that does) is exercised
separately via scripts/validate_command_embeddings.py against the real
checkpoint, not here.
"""

import numpy as np
import pytest

from app.streaming.embeddings import (
    best_match,
    cosine_similarity,
    l2_normalize,
    manhattan_similarity,
    mean_pool,
)


def test_mean_pool_collapses_time_axis():
    # 3 time steps, 2 dims: [[1,1],[3,3],[5,5]] -> mean [3,3]
    hidden_states = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    assert mean_pool(hidden_states) == pytest.approx([3.0, 3.0])


def test_mean_pool_keeps_leading_batch_dim():
    hidden_states = np.ones((2, 4, 3))  # (batch, time, dim)
    pooled = mean_pool(hidden_states)
    assert pooled.shape == (2, 3)


def test_l2_normalize_produces_unit_length():
    vector = np.array([3.0, 4.0])  # length 5
    normalized = l2_normalize(vector)
    assert np.linalg.norm(normalized) == pytest.approx(1.0)
    assert normalized == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_is_left_alone():
    zero = np.zeros(4)
    assert np.array_equal(l2_normalize(zero), zero)


def test_cosine_similarity_identical_vectors_is_one():
    vector = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    vector = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, -vector) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def _unit(*values: float) -> np.ndarray:
    return l2_normalize(np.array(values, dtype=np.float64))


def test_best_match_finds_the_closer_label():
    query = _unit(1.0, 0.0, 0.0)
    bank = {
        "delete": [_unit(0.99, 0.01, 0.0)],
        "stop": [_unit(0.0, 1.0, 0.0)],
    }
    match = best_match(query, bank, threshold=0.5)
    assert match is not None
    assert match.label == "delete"
    # best_match scores on Manhattan similarity, not cosine - see
    # embeddings.manhattan_similarity for why (it separates real
    # same-vs-different-command recordings slightly better).
    assert match.score == pytest.approx(manhattan_similarity(query, bank["delete"][0]))


def test_manhattan_similarity_identical_vectors_is_one():
    vector = _unit(1.0, 2.0, 3.0)
    assert manhattan_similarity(vector, vector) == pytest.approx(1.0)


def test_manhattan_similarity_decreases_as_vectors_diverge():
    query = _unit(1.0, 0.0)
    close = _unit(0.99, 0.01)
    far = _unit(0.0, 1.0)
    assert manhattan_similarity(query, close) > manhattan_similarity(query, far)


def test_best_match_uses_best_sample_not_mean():
    # "stop" has one great sample and one terrible one; a mean would drag
    # its score down, but a single clean take should still be recognised.
    query = _unit(1.0, 0.0)
    bank = {
        "stop": [_unit(1.0, 0.0), _unit(-1.0, 0.001)],
        "save": [_unit(0.7, 0.7)],
    }
    match = best_match(query, bank, threshold=0.9)
    assert match is not None
    assert match.label == "stop"
    assert match.score == pytest.approx(1.0)


def test_best_match_returns_none_below_threshold():
    query = _unit(1.0, 0.0)
    bank = {"stop": [_unit(0.0, 1.0)]}
    assert best_match(query, bank, threshold=0.5) is None


def test_best_match_returns_none_for_empty_bank():
    query = _unit(1.0, 0.0)
    assert best_match(query, {}, threshold=0.0) is None


def test_best_match_ignores_labels_with_no_samples():
    query = _unit(1.0, 0.0)
    bank = {"delete": [], "stop": [_unit(1.0, 0.0)]}
    match = best_match(query, bank, threshold=0.5)
    assert match is not None
    assert match.label == "stop"
