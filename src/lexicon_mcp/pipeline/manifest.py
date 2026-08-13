"""Build provenance and deterministic release-manifest generation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

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
    """Split every artifact into independently hashed raw release parts.

    Concatenating a component's ordered parts reconstructs the exact file. Raw
    transport avoids temporary decompression space and makes compressed/final
    hashes identical by construction.
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
    for source_path in paths:
        relative = source_path.relative_to(dataset_root).as_posix()
        final_size = source_path.stat().st_size
        final_sha = file_sha256(source_path)
        path_token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
        clean_name = source_path.name.replace(" ", "-")
        compressed_path = package_dir / f".{path_token}-{clean_name}.zst.partial"
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - required runtime dependency
            raise RuntimeError("zstandard is required to package dataset artifacts") from exc
        compressor = zstandard.ZstdCompressor(level=10, threads=0, write_checksum=True)
        with source_path.open("rb") as source_stream, compressed_path.open("wb") as compressed:
            compressor.copy_stream(source_stream, compressed, read_size=8 * 1024 * 1024)
            compressed.flush()
            os.fsync(compressed.fileno())
        compressed_size = compressed_path.stat().st_size
        compressed_sha = file_sha256(compressed_path)
        parts: list[dict[str, Any]] = []
        offset = 0
        part_number = 0
        with compressed_path.open("rb") as source_stream:
            while offset < compressed_size:
                name = f"{dataset_version}-{path_token}-{clean_name}.zst.part{part_number:04d}"
                destination = package_dir / name
                digest = hashlib.sha256()
                written = 0
                with destination.open("wb") as output:
                    while written < max_part_size:
                        block = source_stream.read(min(8 * 1024 * 1024, max_part_size - written))
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        written += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                parts.append(
                    {"name": name, "size": written, "sha256": digest.hexdigest(), "offset": offset}
                )
                offset += written
                part_number += 1
        compressed_path.unlink()
        if offset != compressed_size:
            raise RuntimeError(f"packaged size mismatch for {relative}")
        artifact_type = _component_type(relative)
        integrity: dict[str, Any] = {}
        if relative.endswith(".sqlite3"):
            integrity["sqlite"] = True
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
                if item in {"oewn", "conceptnet", "cmudict"} or item.startswith("wiktextract")
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
