"""Package built schema-2 packs into a release with its manifest.

Each pack is compressed straight into independently hashed parts, never staged
as a whole compressed file, so packaging a 23 GB corpus costs only the final
release assets in peak storage.

The manifest's pack table is what makes tiering invisible to callers -- it maps
(capability, language) to the components serving it, so a user names a language
and the installer resolves whether that language owns a pack or shares a bundle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

from .common import file_sha256, write_json_atomic
from .manifest import _SplitPartWriter, reproducible_timestamp
from .transform import DATASET_SCHEMA_VERSION, PackResult

_COMPRESSION_READ_SIZE = 8 * 1024 * 1024
DEFAULT_MAX_PART_SIZE = 1_800_000_000


def _component_id(pack_id: str) -> str:
    return f"artifact-{pack_id}"


def _component_path(capability: str, pack_id: str) -> str:
    if capability == "core":
        return "core.sqlite3"
    return f"{capability}/{pack_id}.sqlite3"


def package_packs(
    built: Sequence[PackResult],
    package_dir: Path,
    *,
    dataset_version: str,
    repository: str,
    tag: str,
    transformation_commit: str,
    sources: list[dict[str, Any]],
    source_dataset: dict[str, str],
    created_at: str | None = None,
    base_url: str | None = None,
    max_part_size: int = DEFAULT_MAX_PART_SIZE,
) -> dict[str, Any]:
    """Compress every pack into parts and emit the schema-2 manifest."""

    if not built:
        raise ValueError("a release must contain at least one pack")
    if not re.fullmatch(r"[0-9a-f]{40}", transformation_commit):
        raise ValueError("transformation_commit must be a lowercase 40-hex Git commit")
    if not any(item.pack.capability == "core" for item in built):
        raise ValueError("a release must contain a core pack")

    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - required packaging dependency
        raise RuntimeError("zstandard is required to package release components") from exc

    package_dir.mkdir(parents=True, exist_ok=True)
    source_ids = [item["id"] for item in sources]
    components: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    generated: list[Path] = []

    try:
        for item in sorted(built, key=lambda entry: entry.pack.id):
            pack = item.pack
            component_id = _component_id(pack.id)
            relative = _component_path(pack.capability, pack.id)
            final_size = item.path.stat().st_size
            final_sha = file_sha256(item.path)
            sink = _SplitPartWriter(
                package_dir, f"{dataset_version}-{pack.id}.sqlite3.zst.part", max_part_size
            )
            try:
                compressor = zstandard.ZstdCompressor(level=10, threads=0, write_checksum=True)
                with item.path.open("rb") as stream:
                    compressor.copy_stream(
                        stream, cast(BinaryIO, sink), read_size=_COMPRESSION_READ_SIZE
                    )
                parts, compressed_size, compressed_sha = sink.finish()
                generated.extend(sink.commit())
            except BaseException:
                sink.abort()
                raise

            components.append(
                {
                    "id": component_id,
                    "artifact_type": "lexical_sqlite",
                    "path": relative,
                    "compression": "zstd",
                    "compressed_size": compressed_size,
                    "compressed_sha256": compressed_sha,
                    "final_size": final_size,
                    "final_sha256": final_sha,
                    "parts": parts,
                    "sources": source_ids,
                    "integrity": {
                        "sqlite": True,
                        "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    },
                }
            )
            entry: dict[str, Any] = {
                "id": pack.id,
                "capability": pack.capability,
                "components": [component_id],
            }
            if pack.capability == "core":
                entry["required"] = True
            else:
                entry["languages"] = list(pack.languages)
            packs.append(entry)

        release: dict[str, Any] = {
            "repository": repository,
            "tag": tag,
            "immutable": True,
        }
        if base_url:
            release["base_url"] = base_url.rstrip("/") + "/"
        manifest = {
            "schema_version": 2,
            "dataset_version": dataset_version,
            "transformation_commit": transformation_commit,
            "release": release,
            "created_at": reproducible_timestamp(created_at),
            "source_dataset": source_dataset,
            "sources": sources,
            "components": components,
            "packs": packs,
        }
        write_json_atomic(package_dir / "manifest.json", manifest)
        return manifest
    except BaseException:
        for path in generated:
            path.unlink(missing_ok=True)
        (package_dir / "manifest.json.partial").unlink(missing_ok=True)
        raise


def load_sources(dataset_root: Path) -> list[dict[str, Any]]:
    """Carry the original corpus' source provenance into the transformed release."""

    lock = dataset_root / "sources.lock.json"
    try:
        value = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source provenance: {lock}") from exc
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{lock} declares no sources")
    return sources
