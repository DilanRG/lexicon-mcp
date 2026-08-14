"""Build provenance and deterministic release-manifest generation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from .common import file_sha256, write_json_atomic
from .constants import SOURCE_EXPECTATIONS

_REQUIRED_NOTICES = {
    "OEWN-LICENSE.md",
    "PRINCETON-WORDNET.txt",
    "CC-BY-4.0.txt",
    "CC-BY-SA-4.0.txt",
    "GFDL-1.3.txt",
    "CMUDICT.txt",
}

_COMPRESSION_READ_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class _CompletedPart:
    temporary: Path
    final: Path
    size: int
    sha256: str
    offset: int


class _SplitPartWriter:
    """File-like sink that splits one zstd frame without staging that frame."""

    def __init__(self, package_dir: Path, name_prefix: str, max_part_size: int) -> None:
        self._package_dir = package_dir
        self._name_prefix = name_prefix
        self._max_part_size = max_part_size
        self._stream: BinaryIO | None = None
        self._current_temporary: Path | None = None
        self._part_digest: Any = None
        self._part_size = 0
        self._part_offset = 0
        self._part_number = 0
        self._compressed_digest = hashlib.sha256()
        self._compressed_size = 0
        self._parts: list[_CompletedPart] = []
        self._committed: list[Path] = []
        self._closed = False

    def _open_part(self) -> None:
        if self._part_number > 9999:
            raise RuntimeError(f"component {self._name_prefix} requires more than 10,000 parts")
        name = f"{self._name_prefix}{self._part_number:04d}"
        temporary = self._package_dir / f".{name}.partial"
        # Exclusive creation makes a stale file or concurrent packager fail
        # without overwriting bytes that may belong to another invocation.
        stream = temporary.open("xb")
        self._stream = stream
        self._current_temporary = temporary
        self._part_digest = hashlib.sha256()
        self._part_size = 0

    def _finish_part(self) -> None:
        if self._stream is None:
            return
        stream = self._stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        name = f"{self._name_prefix}{self._part_number:04d}"
        self._parts.append(
            _CompletedPart(
                temporary=self._package_dir / f".{name}.partial",
                final=self._package_dir / name,
                size=self._part_size,
                sha256=self._part_digest.hexdigest(),
                offset=self._part_offset,
            )
        )
        self._part_offset += self._part_size
        self._part_number += 1
        self._stream = None
        self._current_temporary = None
        self._part_digest = None
        self._part_size = 0

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._closed:
            raise ValueError("cannot write to a closed split-part stream")
        view = memoryview(data).cast("B")
        total = len(view)
        position = 0
        while position < total:
            if self._stream is None:
                self._open_part()
            available = self._max_part_size - self._part_size
            chunk = view[position : position + available]
            assert self._stream is not None
            written = self._stream.write(chunk)
            if written != len(chunk):
                raise OSError(
                    f"short write while packaging {self._name_prefix}: "
                    f"expected {len(chunk)}, wrote {written}"
                )
            self._part_digest.update(chunk)
            self._compressed_digest.update(chunk)
            self._part_size += written
            self._compressed_size += written
            position += written
            if self._part_size == self._max_part_size:
                self._finish_part()
        return total

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()

    def finish(self) -> tuple[list[dict[str, Any]], int, str]:
        if self._closed:
            raise ValueError("split-part stream is already closed")
        self._finish_part()
        self._closed = True
        if not self._parts:
            raise RuntimeError(f"compressor emitted no data for {self._name_prefix}")
        parts = [
            {
                "name": item.final.name,
                "size": item.size,
                "sha256": item.sha256,
                "offset": item.offset,
            }
            for item in self._parts
        ]
        return parts, self._compressed_size, self._compressed_digest.hexdigest()

    def commit(self) -> list[Path]:
        if not self._closed:
            raise ValueError("split-part stream must be finished before commit")
        for item in self._parts:
            # A hard link is an atomic no-replace publish within package_dir.
            # Unlike os.replace(), it cannot silently clobber a stale release
            # asset if another process raced the preflight check.
            os.link(item.temporary, item.final)
            item.temporary.unlink()
            self._committed.append(item.final)
        return list(self._committed)

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._current_temporary is not None:
            self._current_temporary.unlink(missing_ok=True)
            self._current_temporary = None
        for item in self._parts:
            item.temporary.unlink(missing_ok=True)
        for path in self._committed:
            path.unlink(missing_ok=True)


def _component_names(
    source_path: Path, dataset_root: Path, dataset_version: str
) -> tuple[str, str, str]:
    relative = source_path.relative_to(dataset_root).as_posix()
    path_token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    clean_name = source_path.name.replace(" ", "-")
    prefix = f"{dataset_version}-{path_token}-{clean_name}.zst.part"
    return path_token, clean_name, prefix


def _refuse_stale_package_outputs(
    package_dir: Path,
    paths: list[Path],
    dataset_root: Path,
    dataset_version: str,
) -> None:
    reserved = {"manifest.json", "manifest.json.partial"}
    prefixes: list[str] = []
    legacy_temporaries: set[str] = set()
    for source_path in paths:
        path_token, clean_name, prefix = _component_names(
            source_path, dataset_root, dataset_version
        )
        prefixes.extend((prefix, f".{prefix}"))
        legacy_temporaries.add(f".{path_token}-{clean_name}.zst.partial")
    collisions = sorted(
        entry.name
        for entry in package_dir.iterdir()
        if entry.name in reserved
        or entry.name in legacy_temporaries
        or any(entry.name.startswith(prefix) for prefix in prefixes)
    )
    if collisions:
        preview = ", ".join(collisions[:5])
        suffix = " ..." if len(collisions) > 5 else ""
        raise FileExistsError(
            f"package directory contains stale or conflicting release outputs: {preview}{suffix}"
        )


def reproducible_timestamp(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    value = (
        dt.datetime.fromtimestamp(epoch, tz=dt.UTC)
        if epoch
        else dt.datetime.now(tz=dt.UTC)
    )
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def source_record(
    source_id: str,
    path: Path,
    *,
    revision: str | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    expected = SOURCE_EXPECTATIONS[source_id]
    return {
        "id": source_id,
        "name": source_id,
        "url": expected["url"],
        "revision": revision or expected["snapshot"],
        "retrieved_at": reproducible_timestamp(retrieved_at),
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
        "license": expected["license"],
    }


def write_sources_lock(path: Path, records: list[dict[str, Any]]) -> None:
    write_json_atomic(
        path,
        {"schema_version": 1, "sources": sorted(records, key=lambda item: item["id"])},
    )


def _component_type(relative: str) -> str:
    if relative.endswith(".usearch"):
        return "semantic_index"
    if relative == "semantic/mapping.sqlite3":
        return "semantic_mapping"
    if relative.endswith(".sqlite3"):
        return "sqlite"
    if relative.endswith(".f16"):
        return "semantic_vectors"
    return "metadata"


def _semantic_counts(dataset_root: Path) -> tuple[int, dict[str, int]]:
    mapping = dataset_root / "semantic" / "mapping.sqlite3"
    if not mapping.exists():
        return 0, {}
    connection = sqlite3.connect(f"file:{mapping.as_posix()}?mode=ro", uri=True)
    total = connection.execute("SELECT COUNT(*) FROM semantic_terms").fetchone()[0]
    indexes = {
        f"semantic/{index_file}": int(term_count)
        for index_file, term_count in connection.execute(
            "SELECT index_file,term_count FROM semantic_languages"
        )
    }
    connection.close()
    return int(total), indexes


def package_dataset(
    dataset_root: Path,
    package_dir: Path,
    *,
    dataset_version: str,
    repository: str,
    tag: str,
    transformation_commit: str,
    base_url: str | None = None,
    max_part_size: int = 1024 * 1024 * 1024,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Compress every artifact directly into independently hashed parts.

    Concatenating a component's ordered parts reconstructs one complete zstd
    frame.  Compression bytes are never staged as a second whole-component
    file, keeping peak packaging storage to the final split release assets.
    """

    if max_part_size < 1 or max_part_size > 1024 * 1024 * 1024:
        raise ValueError("max_part_size must be between 1 byte and 1 GiB")
    if not re.fullmatch(r"[0-9a-f]{40}", transformation_commit):
        raise ValueError("transformation_commit must be a lowercase 40-hex Git commit")
    dataset_root = dataset_root.resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    notices = dataset_root / "notices"
    missing_notices = [
        name for name in sorted(_REQUIRED_NOTICES) if not (notices / "licenses" / name).is_file()
    ]
    if not (notices / "DATA_LICENSES.md").is_file():
        missing_notices.insert(0, "DATA_LICENSES.md")
    if missing_notices:
        raise ValueError(f"dataset is missing required notices: {', '.join(missing_notices)}")
    total_semantic, index_counts = _semantic_counts(dataset_root)
    lock_path = dataset_root / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    available_ids = sorted(item["id"] for item in lock.get("sources", []))
    components: list[dict[str, Any]] = []
    paths = sorted(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and not path.name.endswith(("-wal", "-shm", ".partial"))
        ),
        key=lambda item: item.relative_to(dataset_root).as_posix(),
    )
    _refuse_stale_package_outputs(package_dir, paths, dataset_root, dataset_version)
    generated_paths: list[Path] = []
    try:
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - required runtime dependency
            raise RuntimeError("zstandard is required to package dataset artifacts") from exc
        for source_path in paths:
            relative = source_path.relative_to(dataset_root).as_posix()
            final_size = source_path.stat().st_size
            final_sha = file_sha256(source_path)
            path_token, _clean_name, part_prefix = _component_names(
                source_path, dataset_root, dataset_version
            )
            sink = _SplitPartWriter(package_dir, part_prefix, max_part_size)
            try:
                compressor = zstandard.ZstdCompressor(
                    level=10, threads=0, write_checksum=True
                )
                with source_path.open("rb") as source_stream:
                    compressor.copy_stream(
                        source_stream,
                        cast(BinaryIO, sink),
                        read_size=_COMPRESSION_READ_SIZE,
                    )
                parts, compressed_size, compressed_sha = sink.finish()
                generated_paths.extend(sink.commit())
            except BaseException:
                sink.abort()
                raise
            if sum(int(part["size"]) for part in parts) != compressed_size:
                raise RuntimeError(f"packaged size mismatch for {relative}")
            artifact_type = _component_type(relative)
            integrity: dict[str, Any] = {}
            if relative.endswith(".sqlite3"):
                integrity.update({"sqlite": True, "dataset_schema_version": 2})
            if relative == "semantic/mapping.sqlite3":
                integrity.update(
                    {
                        "semantic_count": total_semantic,
                        "semantic_mapping_table": "semantic_terms",
                    }
                )
            if artifact_type == "semantic_index":
                if relative.endswith("/global.usearch"):
                    integrity.update(
                        {
                            "semantic_count": total_semantic,
                            "semantic_mapping": "semantic/mapping.sqlite3",
                            "semantic_mapping_table": "semantic_terms",
                        }
                    )
                else:
                    integrity["semantic_count"] = index_counts.get(relative, 0)
            if relative == "lexicon.sqlite3":
                component_sources = [
                    item
                    for item in available_ids
                    if item in {"oewn", "conceptnet", "cmudict"}
                    or item.startswith("wiktextract")
                ]
            elif relative.startswith("semantic/"):
                component_sources = [item for item in available_ids if item == "numberbatch"]
            else:
                component_sources = available_ids
            component: dict[str, Any] = {
                "id": f"artifact-{path_token}",
                "artifact_type": artifact_type,
                "path": relative,
                "compression": "zstd",
                "compressed_size": compressed_size,
                "compressed_sha256": compressed_sha,
                "final_size": final_size,
                "final_sha256": final_sha,
                "parts": parts,
                "sources": component_sources,
            }
            if integrity:
                component["integrity"] = integrity
            components.append(component)
        release: dict[str, Any] = {
            "repository": repository,
            "tag": tag,
            "immutable": True,
        }
        if base_url:
            release["base_url"] = base_url.rstrip("/") + "/"
        manifest = {
            "schema_version": 1,
            "dataset_version": dataset_version,
            "profile": "full",
            "transformation_commit": transformation_commit,
            "release": release,
            "created_at": reproducible_timestamp(created_at),
            "sources": lock.get("sources", []),
            "components": components,
        }
        write_json_atomic(package_dir / "manifest.json", manifest)
        return manifest
    except BaseException:
        for path in generated_paths:
            path.unlink(missing_ok=True)
        (package_dir / "manifest.json.partial").unlink(missing_ok=True)
        raise
