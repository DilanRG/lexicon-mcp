from __future__ import annotations

import pytest

from lexicon_mcp.runtime.ann_search import ann_candidate_count


@pytest.mark.parametrize(
    ("result_limit", "index_size", "expected"),
    [
        (1, 10_000, 256),
        (20, 10_000, 256),
        (30, 10_000, 300),
        (100, 10_000, 500),
        (20, 17, 17),
        (20, 0, 0),
    ],
)
def test_ann_candidate_count_is_bounded_and_shared(
    result_limit: int, index_size: int, expected: int
) -> None:
    assert ann_candidate_count(result_limit, index_size) == expected


@pytest.mark.parametrize(
    ("result_limit", "index_size", "error"),
    [
        (0, 10, ValueError),
        (1, -1, ValueError),
        (True, 10, TypeError),
        (1, False, TypeError),
    ],
)
def test_ann_candidate_count_rejects_invalid_inputs(
    result_limit: object, index_size: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        ann_candidate_count(result_limit, index_size)  # type: ignore[arg-type]
