"""Strict release-manifest parsing.

The release manifest is the trust boundary for the installer.  Paths and
identifiers are validated here once, before any filesystem mutation begins.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ManifestError(ValueError):
    """The release manifest is malformed or internally inconsistent."""


def safe_relative_path(value: str, *, field: str = "path") -> PurePosixPath:
    """Return a safe archive-style relative path.

    Backslashes are rejected rather than normalized so a manifest has the same
    meaning on Windows and POSIX.  Colons are excluded to prevent drive and ADS
    paths on Windows.
    """

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ManifestError(f"{field} must be a non-empty POSIX relative path")
    if ":" in value:
        raise ManifestError(f"{field} must not contain ':'")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"{field} escapes the dataset root: {value!r}")
    return path


def safe_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ManifestError(f"{field} must match {_ID_RE.pattern}")
    return value


def safe_version(value: Any, *, field: str = "dataset_version") -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ManifestError(f"{field} is not a safe version identifier")
    return value


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _size(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{field} must be a non-negative integer")
    return int(value)


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class Release:
    repository: str
    tag: str
    immutable: bool
    base_url: str | None


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    name: str
    url: str
    revision: str
    retrieved_at: str
    sha256: str
    size: int
    row_count: int | None
    row_digest: str | None
    license: str
    license_url: str | None


@dataclass(frozen=True, slots=True)
class Part:
    name: str | None
    url: str | None
    size: int
    sha256: str
    offset: int


@dataclass(frozen=True, slots=True)
class Component:
    id: str
    artifact_type: str
    path: PurePosixPath
    compression: str
    compressed_size: int
    compressed_sha256: str
    final_size: int
    final_sha256: str
    parts: tuple[Part, ...]
    sources: tuple[str, ...]
    integrity: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_version: str
    profile: str
    release: Release
    created_at: str
    transformation_commit: str
    sources: tuple[Source, ...]
    components: tuple[Component, ...]
    raw: bytes
    sha256: str

    def component(self, component_id: str) -> Component:
        for item in self.components:
            if item.id == component_id:
                return item
        raise KeyError(component_id)


def _parse_release(value: Any, dataset_version: str) -> Release:
    if not isinstance(value, dict):
        raise ManifestError("release must be an object")
    repository = _string(value.get("repository"), "release.repository")
    if repository.startswith(("http://", "https://")):
        raise ManifestError("release.repository must be an owner/repository slug")
    pieces = repository.split("/")
    if len(pieces) != 2 or not all(_ID_RE.fullmatch(piece) for piece in pieces):
        raise ManifestError("release.repository must be an owner/repository slug")
    tag = safe_version(value.get("tag"), field="release.tag")
    if tag != dataset_version:
        raise ManifestError("release.tag must equal dataset_version")
    if value.get("immutable") is not True:
        raise ManifestError("release.immutable must be true")
    base_url = value.get("base_url")
    if base_url is not None:
        base_url = _string(base_url, "release.base_url")
        if not base_url.startswith(("https://", "http://")):
            raise ManifestError("release.base_url must be HTTP(S)")
    return Release(repository, tag, True, base_url)


def _parse_sources(value: Any) -> tuple[Source, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError("sources must be a non-empty array")
    parsed: list[Source] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        field = f"sources[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")
        source_id = safe_identifier(item.get("id"), field=f"{field}.id")
        if source_id in seen:
            raise ManifestError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        license_url = item.get("license_url")
        if license_url is not None:
            license_url = _string(license_url, f"{field}.license_url")
        row_count = item.get("row_count")
        row_digest = item.get("row_digest")
        if row_count is not None:
            row_count = _size(row_count, f"{field}.row_count")
        if row_digest is not None:
            row_digest = _sha256(row_digest, f"{field}.row_digest")
        if (row_count is None) != (row_digest is None):
            raise ManifestError(f"{field}.row_count and row_digest must be set together")
        parsed.append(
            Source(
                id=source_id,
                name=_string(item.get("name"), f"{field}.name"),
                url=_string(item.get("url"), f"{field}.url"),
                revision=_string(item.get("revision"), f"{field}.revision"),
                retrieved_at=_string(item.get("retrieved_at"), f"{field}.retrieved_at"),
                sha256=_sha256(item.get("sha256"), f"{field}.sha256"),
                size=_size(item.get("size"), f"{field}.size"),
                row_count=row_count,
                row_digest=row_digest,
                license=_string(item.get("license"), f"{field}.license"),
                license_url=license_url,
            )
        )
    return tuple(parsed)


def _parse_integrity(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    allowed = {
        "sqlite",
        "semantic_count",
        "semantic_mapping",
        "semantic_mapping_table",
        "semantic_table",
        "archive_member",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ManifestError(f"{field} contains unsupported keys: {sorted(unknown)}")
    result = dict(value)
    if "sqlite" in result and not isinstance(result["sqlite"], bool):
        raise ManifestError(f"{field}.sqlite must be boolean")
    if "semantic_count" in result:
        result["semantic_count"] = _size(result["semantic_count"], f"{field}.semantic_count")
    if "semantic_mapping" in result:
        result["semantic_mapping"] = str(
            safe_relative_path(result["semantic_mapping"], field=f"{field}.semantic_mapping")
        )
    for key in ("semantic_mapping_table", "semantic_table"):
        if key in result:
            table = _string(result[key], f"{field}.{key}")
            if not _TABLE_RE.fullmatch(table):
                raise ManifestError(f"{field}.{key} is not a safe SQLite identifier")
            result[key] = table
    if "archive_member" in result:
        result["archive_member"] = str(
            safe_relative_path(result["archive_member"], field=f"{field}.archive_member")
        )
    return result


def _parse_components(value: Any, source_ids: set[str]) -> tuple[Component, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError("components must be a non-empty array")
    parsed: list[Component] = []
    seen_ids: set[str] = set()
    seen_paths: set[PurePosixPath] = set()
    for index, item in enumerate(value):
        field = f"components[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")
        component_id = safe_identifier(item.get("id"), field=f"{field}.id")
        if component_id in seen_ids:
            raise ManifestError(f"duplicate component id: {component_id}")
        seen_ids.add(component_id)
        path_value = item.get("path")
        if not isinstance(path_value, str):
            raise ManifestError(f"{field}.path must be a string")
        path = safe_relative_path(path_value, field=f"{field}.path")
        if path in seen_paths:
            raise ManifestError(f"duplicate component path: {path}")
        seen_paths.add(path)
        compression = _string(item.get("compression"), f"{field}.compression")
        if compression not in {"none", "zstd", "zip"}:
            raise ManifestError(f"{field}.compression must be none, zstd, or zip")
        compressed_size = _size(item.get("compressed_size"), f"{field}.compressed_size")
        compressed_sha = _sha256(
            item.get("compressed_sha256"), f"{field}.compressed_sha256"
        )
        final_size = _size(item.get("final_size"), f"{field}.final_size")
        final_sha = _sha256(item.get("final_sha256"), f"{field}.final_sha256")
        parts_value = item.get("parts")
        if not isinstance(parts_value, list) or not parts_value:
            raise ManifestError(f"{field}.parts must be a non-empty array")
        parts: list[Part] = []
        next_offset = 0
        for part_index, part in enumerate(parts_value):
            part_field = f"{field}.parts[{part_index}]"
            if not isinstance(part, dict):
                raise ManifestError(f"{part_field} must be an object")
            name = part.get("name")
            url = part.get("url")
            if name is None and url is None:
                raise ManifestError(f"{part_field} requires name or url")
            if name is not None:
                name = str(safe_relative_path(name, field=f"{part_field}.name"))
            if url is not None:
                url = _string(url, f"{part_field}.url")
                if not url.startswith(("https://", "http://")):
                    raise ManifestError(f"{part_field}.url must be HTTP(S)")
            size = _size(part.get("size"), f"{part_field}.size")
            offset = _size(part.get("offset", next_offset), f"{part_field}.offset")
            if offset != next_offset:
                raise ManifestError(f"{part_field}.offset is not contiguous")
            parts.append(
                Part(
                    name=name,
                    url=url,
                    size=size,
                    sha256=_sha256(part.get("sha256"), f"{part_field}.sha256"),
                    offset=offset,
                )
            )
            next_offset += size
        if next_offset != compressed_size:
            raise ManifestError(f"{field}.parts sizes do not equal compressed_size")
        if compression == "none" and (
            compressed_size != final_size or compressed_sha != final_sha
        ):
            raise ManifestError(
                f"{field} uncompressed component must have matching compressed/final metadata"
            )
        source_refs = item.get("sources")
        if not isinstance(source_refs, list) or not source_refs:
            raise ManifestError(f"{field}.sources must be a non-empty array")
        refs: list[str] = []
        for source in source_refs:
            source_id = safe_identifier(source, field=f"{field}.sources[]")
            if source_id not in source_ids:
                raise ManifestError(f"{field} references unknown source {source_id!r}")
            if source_id not in refs:
                refs.append(source_id)
        parsed.append(
            Component(
                id=component_id,
                artifact_type=_string(item.get("artifact_type"), f"{field}.artifact_type"),
                path=path,
                compression=compression,
                compressed_size=compressed_size,
                compressed_sha256=compressed_sha,
                final_size=final_size,
                final_sha256=final_sha,
                parts=tuple(parts),
                sources=tuple(refs),
                integrity=_parse_integrity(item.get("integrity"), f"{field}.integrity"),
            )
        )
    return tuple(parsed)


def parse_manifest(raw: bytes | str) -> DatasetManifest:
    """Parse and validate a canonical schema-v1 dataset manifest."""

    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("manifest root must be an object")
    if value.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    dataset_version = safe_version(value.get("dataset_version"))
    profile = _string(value.get("profile"), "profile")
    if profile != "full":
        raise ManifestError("schema v1 supports only the full profile")
    release = _parse_release(value.get("release"), dataset_version)
    sources = _parse_sources(value.get("sources"))
    components = _parse_components(value.get("components"), {item.id for item in sources})
    transformation_commit = value.get("transformation_commit")
    if not isinstance(transformation_commit, str) or not _COMMIT_RE.fullmatch(
        transformation_commit
    ):
        raise ManifestError("transformation_commit must be a lowercase 40-hex Git commit")
    return DatasetManifest(
        schema_version=1,
        dataset_version=dataset_version,
        profile=profile,
        release=release,
        created_at=_string(value.get("created_at"), "created_at"),
        transformation_commit=transformation_commit,
        sources=sources,
        components=components,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
