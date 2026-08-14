"""Exact-cosine candidate recall calibration for semantic indexes."""

from __future__ import annotations

import numpy as np
from usearch.index import Index


def candidate_recall(
    vectors: np.ndarray,
    index: Index,
    *,
    queries: int,
    k: int,
    fetch: int,
    seed: int,
) -> list[float]:
    """Measure whether an overfetched ANN set contains exact top-k rows."""

    if queries < 1 or k < 1 or fetch < k:
        raise ValueError("queries/k must be positive and fetch must be at least k")
    if len(vectors) <= k:
        raise ValueError("calibration corpus must contain more rows than k")
    rng = np.random.default_rng(seed)
    seeds = np.sort(rng.choice(len(vectors), min(queries, len(vectors)), replace=False))
    values = np.asarray(vectors, dtype=np.float32)
    recall: list[float] = []
    for seed_id in seeds:
        query = values[seed_id]
        scores = values @ query
        scores[seed_id] = -np.inf
        exact = set(int(item) for item in np.argpartition(scores, -k)[-k:])
        candidates = index.search(query, min(fetch + 1, len(vectors))).keys
        candidate_set = {int(item) for item in candidates if int(item) != seed_id}
        recall.append(len(exact & candidate_set) / k)
    return recall
