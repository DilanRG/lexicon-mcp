from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from usearch.index import Index, MetricKind, ScalarKind

from lexicon_mcp.usearch_compat import index_count, open_index_view


def _saved_index(
    path: Path,
    *,
    dimensions: int = 2,
    metric: str = "cos",
    dtype: str = "i8",
    connectivity: int = 16,
    delete_key: int | None = None,
) -> None:
    vectors: np.ndarray = np.zeros((2, dimensions), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, min(1, dimensions - 1)] = 1.0
    index = Index(
        ndim=dimensions,
        metric=metric,
        dtype=dtype,
        connectivity=connectivity,
    )
    index.add(
        np.asarray([4, 9], dtype=np.uint64),
        vectors,
    )
    if delete_key is not None:
        assert index.remove(delete_key) is True
    index.save(str(path))
    index.reset()


def test_index_count_uses_mmap_metadata_and_never_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(path)
    original_metadata = Index.metadata
    metadata_calls = 0

    def guarded_metadata(path_or_buffer: Any) -> Any:
        nonlocal metadata_calls
        metadata_calls += 1
        assert isinstance(path_or_buffer, mmap.mmap)
        assert not isinstance(path_or_buffer, (str, os.PathLike))
        return original_metadata(path_or_buffer)

    def forbidden_restore(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Index.restore must not be used")

    monkeypatch.setattr(Index, "metadata", staticmethod(guarded_metadata))
    monkeypatch.setattr(Index, "restore", staticmethod(forbidden_restore))

    assert index_count(path, dimensions=2, expected_count=2) == 2
    assert metadata_calls == 1


def test_open_index_view_is_queryable(tmp_path: Path) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(path)

    index = open_index_view(path, dimensions=2, expected_count=2)
    try:
        matches = index.search(np.asarray([1.0, 0.0], dtype=np.float32), 2)
        assert {int(key) for key in matches.keys} == {4, 9}
    finally:
        index.reset()


@pytest.mark.parametrize(
    ("saved_dimensions", "saved_metric", "saved_dtype", "error"),
    (
        (3, "cos", "i8", "dimensions mismatch"),
        (2, "l2sq", "i8", "metric mismatch"),
        (2, "cos", "f32", "dtype mismatch"),
    ),
)
def test_open_index_view_rejects_wrong_embedded_schema_before_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_dimensions: int,
    saved_metric: str,
    saved_dtype: str,
    error: str,
) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(
        path,
        dimensions=saved_dimensions,
        metric=saved_metric,
        dtype=saved_dtype,
    )
    view_calls = 0
    original_view = Index.view

    def guarded_view(self: Index, viewed_path: str) -> None:
        nonlocal view_calls
        view_calls += 1
        original_view(self, viewed_path)

    monkeypatch.setattr(Index, "view", guarded_view)

    with pytest.raises(RuntimeError, match=error):
        open_index_view(path, dimensions=2, expected_count=2)
    assert view_calls == 0


def test_open_index_view_rejects_wrong_expected_count(tmp_path: Path) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(path)

    with pytest.raises(RuntimeError, match="count_present mismatch"):
        open_index_view(path, dimensions=2, expected_count=3)


def test_open_index_view_rejects_deleted_entries(tmp_path: Path) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(path, delete_key=4)

    with pytest.raises(RuntimeError, match="count_deleted must be zero"):
        open_index_view(path, dimensions=2, expected_count=1)


@pytest.mark.parametrize(
    ("field", "tampered_value", "error"),
    (
        ("version", "2.25.0", "version mismatch"),
        ("matrix_included", False, "matrix is not included"),
        (
            "matrix_uses_64_bit_dimensions",
            True,
            "unsupported 64-bit dimensions",
        ),
        ("kind_key", ScalarKind.U32, "key dtype mismatch"),
        (
            "kind_compressed_slot",
            ScalarKind.U64,
            "compressed-slot dtype mismatch",
        ),
    ),
)
def test_open_index_view_rejects_tampered_native_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered_value: object,
    error: str,
) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(path)
    original_metadata = Index.metadata

    def tampered_metadata(path_or_buffer: Any) -> dict[str, Any]:
        metadata = original_metadata(path_or_buffer)
        assert metadata is not None
        result = dict(metadata)
        result[field] = tampered_value
        return result

    monkeypatch.setattr(Index, "metadata", staticmethod(tampered_metadata))

    with pytest.raises(RuntimeError, match=error):
        open_index_view(path, dimensions=2, expected_count=2)


def test_open_index_view_validates_connectivity_after_view(tmp_path: Path) -> None:
    path = tmp_path / "index.usearch"
    _saved_index(path, connectivity=8)

    with pytest.raises(RuntimeError, match="viewed index connectivity mismatch"):
        open_index_view(path, dimensions=2, connectivity=16, expected_count=2)


def test_open_index_view_disables_reverse_key_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.usearch"
    path.write_bytes(b"placeholder")
    constructor: dict[str, object] = {}

    class FakeIndex:
        def __init__(self, **kwargs: object) -> None:
            constructor.update(kwargs)
            self.ndim = 2
            self.connectivity = 16
            self.dtype = ScalarKind.I8
            self.metric = MetricKind.Cos

        @staticmethod
        def metadata(path_or_buffer: object) -> dict[str, object]:
            assert isinstance(path_or_buffer, mmap.mmap)
            return {
                "dimensions": 2,
                "kind_metric": MetricKind.Cos,
                "kind_scalar": ScalarKind.I8,
                "kind_key": ScalarKind.U64,
                "kind_compressed_slot": ScalarKind.U32,
                "count_present": 2,
                "count_deleted": 0,
                "version": "2.26.0",
                "matrix_included": True,
                "matrix_uses_64_bit_dimensions": False,
            }

        def view(self, viewed_path: str) -> None:
            assert Path(viewed_path) == path.resolve()

        def __len__(self) -> int:
            return 2

        def reset(self) -> None:
            pass

    monkeypatch.setattr("usearch.index.Index", FakeIndex)

    index = open_index_view(path, dimensions=2, expected_count=2)
    index.reset()

    assert constructor["enable_key_lookups"] is False
