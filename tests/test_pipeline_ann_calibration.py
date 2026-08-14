from __future__ import annotations

import numpy as np
import pytest
from usearch.index import Index

from lexicon_mcp.pipeline.ann_calibration import candidate_recall


def test_candidate_recall_is_exact_for_small_separable_fixture() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(128, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = Index(
        ndim=8,
        metric="cos",
        dtype="i8",
        connectivity=16,
        expansion_add=256,
        expansion_search=512,
    )
    index.add(np.arange(128, dtype=np.uint64), vectors)
    values = candidate_recall(vectors, index, queries=20, k=5, fetch=40, seed=7)
    assert np.mean(values) >= 0.90


def test_candidate_recall_rejects_fetch_below_k() -> None:
    vectors = np.eye(4, dtype=np.float32)
    index = Index(ndim=4, metric="cos", dtype="i8")
    index.add(np.arange(4, dtype=np.uint64), vectors)
    with pytest.raises(ValueError, match="fetch must be at least k"):
        candidate_recall(vectors, index, queries=1, k=3, fetch=2, seed=0)
