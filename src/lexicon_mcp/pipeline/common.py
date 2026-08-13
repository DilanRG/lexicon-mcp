"""Shared streaming, normalization, hashing, and checkpoint primitives."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import os
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, cast


class BinaryReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def __iter__(self) -> Iterator[bytes]: ...


def normalize_term(value: str) -> str:
    """Return the runtime lookup key while preserving display text elsewhere."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def open_binary(path: Path) -> Iterator[BinaryReader]:
    suffix = path.suffix.casefold()
    if suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            yield cast(BinaryReader, stream)
        return
    elif suffix == ".bz2":
        with bz2.open(path, "rb") as stream:
            yield cast(BinaryReader, stream)
        return
    elif suffix in {".xz", ".lzma"}:
        with lzma.open(path, "rb") as stream:
            yield cast(BinaryReader, stream)
        return
    elif suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - dependency error is explicit
            raise RuntimeError("zstandard is required to read .zst sources") from exc
        source = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(source)
        try:
            yield cast(BinaryReader, reader)
        finally:
            reader.close()  # type: ignore[no-untyped-call]
            source.close()
        return
    with path.open("rb") as stream:
        yield cast(BinaryReader, stream)


def iter_text_lines(path: Path) -> Iterator[tuple[int, str]]:
    with open_binary(path) as stream:
        for line_number, raw in enumerate(stream, start=1):
            yield line_number, raw.decode("utf-8", errors="replace").rstrip("\r\n")


def configure_build_db(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")


def finalize_readonly_db(connection: sqlite3.Connection) -> None:
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Checkpoints:
    """Small atomic stage markers; source fingerprints prevent stale reuse."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage)
        return self.directory / f"{safe}.json"

    def complete(self, stage: str, fingerprint: str) -> bool:
        path = self._path(stage)
        if not path.exists():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            value.get("state") == "complete" and value.get("fingerprint") == fingerprint
        )

    def mark(self, stage: str, fingerprint: str, **details: object) -> None:
        write_json_atomic(
            self._path(stage),
            {"stage": stage, "state": "complete", "fingerprint": fingerprint, **details},
        )


def source_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"sha256:{file_sha256(path)};size:{stat.st_size}"


def checked_language(value: object, fallback: str = "und") -> str:
    language = str(value or fallback).strip().replace("_", "-").casefold()
    if not language or not re.fullmatch(r"[a-z0-9]{2,8}(?:-[a-z0-9]{1,8})*", language):
        return fallback
    return language
