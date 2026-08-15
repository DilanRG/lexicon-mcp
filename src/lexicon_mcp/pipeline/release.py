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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from .common import file_sha256, write_json_atomic
from .manifest import _SplitPartWriter, reproducible_timestamp
from .transform import DATASET_SCHEMA_VERSION, PackResult, SemanticPackResult

_COMPRESSION_READ_SIZE = 8 * 1024 * 1024
DEFAULT_MAX_PART_SIZE = 1_800_000_000


def _component_id(pack_id: str, suffix: str = "") -> str:
    return f"artifact-{pack_id}{suffix}"


def _component_path(capability: str, pack_id: str) -> str:
    if capability == "core":
        return "core.sqlite3"
    return f"{capability}/{pack_id}.sqlite3"


@dataclass(frozen=True, slots=True)
class _Artifact:
    """One file inside a pack, with the manifest metadata it needs."""

    component_id: str
    path: Path
    logical_path: str
    artifact_type: str
    integrity: dict[str, Any]


def _semantic_artifacts(result: SemanticPackResult) -> list[_Artifact]:
    """The three files a semantic language needs, as separate components."""

    pack_id = f"semantic-{result.language}"
    base = f"semantic/{result.language}"
    schema = {
        "semantic_dimensions": result.dimensions,
        "semantic_metric": "cos",
        "semantic_dtype": "i8",
        "semantic_connectivity": 16,
        "semantic_expansion_add": 256,
        "semantic_expansion_search": 512,
    }
    return [
        _Artifact(
            component_id=_component_id(pack_id, "-mapping"),
            path=result.mapping,
            logical_path=f"{base}/mapping.sqlite3",
            artifact_type="semantic_mapping",
            integrity={
                "sqlite": True,
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "semantic_count": result.terms,
                "semantic_mapping_table": "semantic_terms",
            },
        ),
        _Artifact(
            component_id=_component_id(pack_id, "-vectors"),
            path=result.vectors,
            logical_path=f"{base}/vectors.f16",
            artifact_type="semantic_vectors",
            integrity={},
        ),
        _Artifact(
            component_id=_component_id(pack_id, "-index"),
            path=result.index,
            logical_path=f"{base}/{result.index.name}",
            artifact_type="semantic_index",
            integrity={"semantic_count": result.terms, **schema},
        ),
    ]


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
    semantic: Sequence[SemanticPackResult] = (),
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

    entries: list[tuple[str, str, tuple[str, ...], bool, list[_Artifact]]] = []
    for item in sorted(built, key=lambda entry: entry.pack.id):
        entries.append(
            (
                item.pack.id,
                item.pack.capability,
                item.pack.languages,
                item.pack.capability == "core",
                [
                    _Artifact(
                        component_id=_component_id(item.pack.id),
                        path=item.path,
                        logical_path=_component_path(item.pack.capability, item.pack.id),
                        artifact_type="lexical_sqlite",
                        integrity={
                            "sqlite": True,
                            "dataset_schema_version": DATASET_SCHEMA_VERSION,
                        },
                    )
                ],
            )
        )
    for result in sorted(semantic, key=lambda entry: entry.language):
        entries.append(
            (
                f"semantic-{result.language}",
                "semantic",
                (result.language,),
                False,
                _semantic_artifacts(result),
            )
        )

    try:
        for pack_id, capability, languages, required, artifacts in entries:
            pack_components: list[str] = []
            for artifact in artifacts:
                final_size = artifact.path.stat().st_size
                final_sha = file_sha256(artifact.path)
                name = artifact.logical_path.replace("/", "-")
                sink = _SplitPartWriter(
                    package_dir, f"{dataset_version}-{name}.zst.part", max_part_size
                )
                try:
                    compressor = zstandard.ZstdCompressor(
                        level=10, threads=0, write_checksum=True
                    )
                    with artifact.path.open("rb") as stream:
                        compressor.copy_stream(
                            stream, cast(BinaryIO, sink), read_size=_COMPRESSION_READ_SIZE
                        )
                    parts, compressed_size, compressed_sha = sink.finish()
                    generated.extend(sink.commit())
                except BaseException:
                    sink.abort()
                    raise

                component: dict[str, Any] = {
                    "id": artifact.component_id,
                    "artifact_type": artifact.artifact_type,
                    "path": artifact.logical_path,
                    "compression": "zstd",
                    "compressed_size": compressed_size,
                    "compressed_sha256": compressed_sha,
                    "final_size": final_size,
                    "final_sha256": final_sha,
                    "parts": parts,
                    "sources": source_ids,
                }
                if artifact.integrity:
                    component["integrity"] = artifact.integrity
                components.append(component)
                pack_components.append(artifact.component_id)

            entry: dict[str, Any] = {
                "id": pack_id,
                "capability": capability,
                "components": pack_components,
            }
            if required:
                entry["required"] = True
            else:
                entry["languages"] = list(languages)
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
