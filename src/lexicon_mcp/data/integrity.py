"""Artifact integrity checks used before activation and during verification."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from ..usearch_compat import index_count
from .manifest import Component


class IntegrityError(RuntimeError):
    """One or more installed artifacts fail their release metadata."""


_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_quick_check(path: Path) -> None:
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute("PRAGMA quick_check").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise IntegrityError(f"SQLite quick_check failed for {path}: {exc}") from exc
    if rows != [("ok",)]:
        details = "; ".join(str(row[0]) for row in rows[:10])
        raise IntegrityError(f"SQLite quick_check failed for {path}: {details}")


def sqlite_count(path: Path, table: str) -> int:
    if not _TABLE_RE.fullmatch(table):
        raise IntegrityError(f"unsafe SQLite table identifier: {table!r}")
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            row = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise IntegrityError(f"unable to count {table} in {path}: {exc}") from exc
    if row is None:
        raise IntegrityError(f"unable to count {table} in {path}")
    return int(row[0])


def sqlite_metadata_value(path: Path, key: str) -> str:
    """Read one required metadata value from an immutable SQLite artifact."""

    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise IntegrityError(f"unable to read SQLite metadata from {path}: {exc}") from exc
    if row is None:
        raise IntegrityError(f"SQLite metadata {key!r} is missing from {path}")
    return str(row[0])


def default_semantic_count(path: Path, component: Component) -> int:
    """Read an index/mapping count without loading all vectors into memory."""

    table = component.integrity.get("semantic_table")
    if component.artifact_type == "semantic_index" or path.suffix.lower() == ".usearch":
        try:
            dimensions = int(component.integrity["semantic_dimensions"])
            metric = str(component.integrity["semantic_metric"])
            dtype = str(component.integrity["semantic_dtype"])
            connectivity = int(component.integrity["semantic_connectivity"])
            expansion_add = int(component.integrity["semantic_expansion_add"])
            expansion_search = int(component.integrity["semantic_expansion_search"])
        except KeyError as exc:
            raise IntegrityError(
                f"semantic index schema is missing {exc.args[0]!r}: {path}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise IntegrityError(f"semantic index schema is malformed: {path}") from exc
        if (
            dimensions < 1
            or metric != "cos"
            or dtype != "i8"
            or connectivity != 16
            or expansion_add != 256
            or expansion_search < 512
        ):
            raise IntegrityError(f"semantic index schema is unsupported: {path}")
        try:
            return index_count(
                path,
                dimensions=dimensions,
                metric=metric,
                dtype=dtype,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
            )
        except Exception as exc:
            raise IntegrityError(f"unable to inspect semantic index {path}: {exc}") from exc
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        if table is None:
            table = component.integrity.get("semantic_mapping_table", "semantic_terms")
        return sqlite_count(path, str(table))
    raise IntegrityError(f"no semantic count reader for {component.artifact_type}: {path}")


def verify_component(
    base: Path,
    component: Component,
    *,
    semantic_count_reader: Callable[[Path, Component], int] = default_semantic_count,
    check_cross_component: bool = True,
) -> list[str]:
    """Return human-readable problems for one installed component."""

    path = base.joinpath(*component.path.parts)
    problems: list[str] = []
    try:
        resolved_base = base.resolve()
        resolved = path.resolve()
        if resolved != resolved_base and resolved_base not in resolved.parents:
            return [f"{component.id}: path escapes installed version"]
    except OSError as exc:
        return [f"{component.id}: cannot resolve path: {exc}"]
    if not path.is_file() or path.is_symlink():
        return [f"{component.id}: artifact is missing or not a regular file"]
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return [f"{component.id}: cannot stat artifact: {exc}"]
    if actual_size != component.final_size:
        problems.append(
            f"{component.id}: size mismatch (expected {component.final_size}, got {actual_size})"
        )
        return problems
    try:
        digest = sha256_file(path)
    except OSError as exc:
        return [f"{component.id}: cannot hash artifact: {exc}"]
    if digest != component.final_sha256:
        problems.append(f"{component.id}: SHA-256 mismatch")
        return problems
    if component.integrity.get("sqlite"):
        try:
            sqlite_quick_check(path)
        except IntegrityError as exc:
            problems.append(f"{component.id}: {exc}")
    expected_schema = component.integrity.get("dataset_schema_version")
    if expected_schema is not None:
        try:
            actual_schema = sqlite_metadata_value(path, "schema_version")
        except IntegrityError as exc:
            problems.append(f"{component.id}: {exc}")
        else:
            if actual_schema != str(expected_schema):
                problems.append(
                    f"{component.id}: dataset schema version mismatch "
                    f"(expected {expected_schema}, got {actual_schema!r})"
                )
    expected_count = component.integrity.get("semantic_count")
    actual_count: int | None = None
    if expected_count is not None:
        try:
            actual_count = semantic_count_reader(path, component)
        except IntegrityError as exc:
            problems.append(f"{component.id}: {exc}")
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(f"{component.id}: semantic count failed: {exc}")
        else:
            if actual_count != expected_count:
                problems.append(
                    f"{component.id}: semantic count mismatch "
                    f"(expected {expected_count}, got {actual_count})"
                )
    mapping = component.integrity.get("semantic_mapping")
    if check_cross_component and mapping is not None:
        mapping_path = base.joinpath(*str(mapping).split("/"))
        if not mapping_path.is_file() or mapping_path.is_symlink():
            problems.append(f"{component.id}: semantic mapping is missing")
        else:
            table = str(component.integrity.get("semantic_mapping_table", "semantic_terms"))
            try:
                mapping_count = sqlite_count(mapping_path, table)
            except IntegrityError as exc:
                problems.append(f"{component.id}: {exc}")
            else:
                comparison = expected_count if expected_count is not None else actual_count
                if comparison is not None and mapping_count != comparison:
                    problems.append(
                        f"{component.id}: semantic mapping count mismatch "
                        f"(index {comparison}, mapping {mapping_count})"
                    )
    return problems
