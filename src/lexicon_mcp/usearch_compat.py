"""Large-file-safe USearch helpers shared by build and runtime code.

USearch 2.26's ``Index.metadata(path)`` helper can fail with ``EINVAL`` for
multi-gigabyte indexes on Windows even though the same index can be viewed and
queried successfully. Native metadata is therefore read from a read-only mmap
buffer, validated against the pinned schema, and followed by an explicit
``view``. Neither path-based metadata nor ``Index.restore`` is used, and the
index is not copied into private memory.
"""

from __future__ import annotations

import mmap
from contextlib import suppress
from pathlib import Path
from typing import Any

_USEARCH_VERSION = "2.26.0"


def _metadata_integer(metadata: dict[str, Any], key: str, path: Path) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"USearch index metadata {key!r} is invalid for {path}: {value!r}"
        )
    return value


def _validated_metadata(
    path: Path,
    *,
    dimensions: int,
    metric_kind: Any,
    scalar_kind: Any,
    key_kind: Any,
    compressed_slot_kind: Any,
    expected_count: int | None,
    index_type: Any,
) -> dict[str, Any]:
    """Read native metadata through a 64-bit-safe read-only mmap."""

    try:
        with path.open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_READ
        ) as mapped:
            raw_metadata = index_type.metadata(mapped)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"USearch index metadata is unreadable for {path}") from exc
    if not isinstance(raw_metadata, dict):
        raise RuntimeError(f"USearch index metadata is missing for {path}")
    metadata: dict[str, Any] = raw_metadata

    if metadata.get("version") != _USEARCH_VERSION:
        raise RuntimeError(
            "USearch index version mismatch for "
            f"{path}: expected {_USEARCH_VERSION}, found {metadata.get('version')!r}"
        )
    if metadata.get("matrix_included") is not True:
        raise RuntimeError(f"USearch index matrix is not included for {path}")
    if metadata.get("matrix_uses_64_bit_dimensions") is not False:
        raise RuntimeError(
            f"USearch index uses unsupported 64-bit dimensions for {path}"
        )
    embedded_dimensions = _metadata_integer(metadata, "dimensions", path)
    if embedded_dimensions != dimensions:
        raise RuntimeError(
            "USearch index dimensions mismatch for "
            f"{path}: expected {dimensions}, found {embedded_dimensions}"
        )
    if metadata.get("kind_metric") != metric_kind:
        raise RuntimeError(
            f"USearch index metric mismatch for {path}: expected cosine"
        )
    if metadata.get("kind_scalar") != scalar_kind:
        raise RuntimeError(f"USearch index dtype mismatch for {path}: expected i8")
    if metadata.get("kind_key") != key_kind:
        raise RuntimeError(f"USearch index key dtype mismatch for {path}: expected u64")
    if metadata.get("kind_compressed_slot") != compressed_slot_kind:
        raise RuntimeError(
            f"USearch index compressed-slot dtype mismatch for {path}: expected u32"
        )

    count_present = _metadata_integer(metadata, "count_present", path)
    count_deleted = _metadata_integer(metadata, "count_deleted", path)
    if count_present < 0:
        raise RuntimeError(
            f"USearch index count_present is negative for {path}: {count_present}"
        )
    if count_deleted != 0:
        raise RuntimeError(
            f"USearch index count_deleted must be zero for {path}: {count_deleted}"
        )
    if expected_count is not None and count_present != expected_count:
        raise RuntimeError(
            "USearch index count_present mismatch for "
            f"{path}: expected {expected_count}, found {count_present}"
        )
    return metadata


def open_index_view(
    path: str | Path,
    *,
    dimensions: int,
    metric: str = "cos",
    dtype: str = "i8",
    connectivity: int = 16,
    expansion_add: int = 256,
    expansion_search: int = 512,
    expected_count: int | None = None,
) -> Any:
    """Validate and view one immutable index without path metadata introspection."""

    index_path = Path(path).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if dimensions < 1:
        raise ValueError("USearch dimensions must be positive")
    if connectivity < 1:
        raise ValueError("USearch connectivity must be positive")
    if expansion_add < 1 or expansion_search < 1:
        raise ValueError("USearch expansion values must be positive")
    if metric != "cos":
        raise ValueError("USearch metric must be 'cos'")
    if dtype != "i8":
        raise ValueError("USearch dtype must be 'i8'")
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError("USearch expected_count must be a non-negative integer")

    from usearch.index import Index, MetricKind, ScalarKind

    metadata = _validated_metadata(
        index_path,
        dimensions=dimensions,
        metric_kind=MetricKind.Cos,
        scalar_kind=ScalarKind.I8,
        key_kind=ScalarKind.U64,
        compressed_slot_kind=ScalarKind.U32,
        expected_count=expected_count,
        index_type=Index,
    )
    metadata_count = int(metadata["count_present"])

    index: Any = Index(
        ndim=dimensions,
        metric=metric,
        dtype=dtype,
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        # Viewing with key lookups enabled makes USearch rebuild a reverse
        # map for every key.  The full global index has 9.16 million keys;
        # search and integrity checks only need returned keys, never key to
        # slot lookup, so avoid that large private allocation.
        enable_key_lookups=False,
    )
    try:
        index.view(str(index_path))
        if int(index.ndim) != dimensions:
            raise RuntimeError(
                f"USearch viewed index dimensions mismatch for {index_path}"
            )
        if int(index.connectivity) != connectivity:
            raise RuntimeError(
                f"USearch viewed index connectivity mismatch for {index_path}"
            )
        if index.dtype != ScalarKind.I8:
            raise RuntimeError(f"USearch viewed index dtype mismatch for {index_path}")
        if index.metric != MetricKind.Cos:
            raise RuntimeError(f"USearch viewed index metric mismatch for {index_path}")
        if len(index) != metadata_count:
            raise RuntimeError(f"USearch viewed index count mismatch for {index_path}")
    except BaseException:
        with suppress(Exception):
            index.reset()
        raise
    return index


def index_count(
    path: str | Path,
    *,
    dimensions: int,
    metric: str = "cos",
    dtype: str = "i8",
    connectivity: int = 16,
    expansion_add: int = 256,
    expansion_search: int = 512,
    expected_count: int | None = None,
) -> int:
    """Return the number of present keys while releasing the mmap promptly."""

    index = open_index_view(
        path,
        dimensions=dimensions,
        metric=metric,
        dtype=dtype,
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        expected_count=expected_count,
    )
    try:
        return len(index)
    finally:
        index.reset()
