"""Shared internal search budget for Numberbatch ANN reranking."""

from __future__ import annotations

_MIN_CANDIDATES = 256
_CANDIDATE_MULTIPLIER = 10
_MAX_CANDIDATES = 500


def ann_candidate_count(result_limit: int, index_size: int) -> int:
    """Return the bounded HNSW candidate count used before exact reranking."""

    if isinstance(result_limit, bool) or not isinstance(result_limit, int):
        raise TypeError("result_limit must be an integer")
    if isinstance(index_size, bool) or not isinstance(index_size, int):
        raise TypeError("index_size must be an integer")
    if result_limit < 1:
        raise ValueError("result_limit must be positive")
    if index_size < 0:
        raise ValueError("index_size must be non-negative")
    requested = min(
        max(_MIN_CANDIDATES, result_limit * _CANDIDATE_MULTIPLIER, result_limit + 1),
        _MAX_CANDIDATES,
    )
    return min(index_size, requested)
