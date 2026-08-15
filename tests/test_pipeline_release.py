"""Packaging built packs into a schema-2 release."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lexicon_mcp.data.manifest import parse_manifest
from lexicon_mcp.data.selection import resolve
from lexicon_mcp.pipeline.packs import PlannedPack
from lexicon_mcp.pipeline.release import load_sources, package_packs
from lexicon_mcp.pipeline.transform import DATASET_SCHEMA_VERSION, PackResult

SOURCES = [
    {
        "id": "fixture",
        "name": "Test fixture",
        "url": "https://fixtures.invalid/source",
        "revision": "test-1",
        "retrieved_at": "2026-08-16T00:00:00Z",
        "sha256": "0" * 64,
        "size": 0,
        "row_count": None,
        "row_digest": None,
        "license": "CC0-1.0",
    }
]
SOURCE_DATASET = {"dataset_version": "data-v1.1.0", "manifest_sha256": "a" * 64}


def pack_file(tmp_path: Path, name: str) -> Path:
    """A pack carrying the metadata the integrity check reads back."""

    path = tmp_path / "built" / f"{name}.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO metadata VALUES ('schema_version', ?)", (str(DATASET_SCHEMA_VERSION),)
    )
    connection.execute("INSERT INTO metadata VALUES ('pack_id', ?)", (name,))
    connection.commit()
    connection.close()
    return path


def built(tmp_path: Path) -> list[PackResult]:
    plans = [
        PlannedPack("core", "core", (), 0),
        PlannedPack("lexical-en", "lexical", ("en",), 0),
        PlannedPack("lexical-bundle-001", "lexical", ("cy", "gv"), 0),
        PlannedPack("semantic-en", "semantic", ("en",), 0),
    ]
    return [
        PackResult(
            pack=plan,
            path=pack_file(tmp_path, plan.id),
            raw_bytes=0,
            terms=0,
            stubs=0,
            entries=0,
            senses=0,
            relations=0,
            translations=0,
        )
        for plan in plans
    ]


def test_packaging_emits_a_manifest_the_installer_accepts(tmp_path: Path) -> None:
    manifest = package_packs(
        built(tmp_path),
        tmp_path / "release",
        dataset_version="data-v2.0.0",
        repository="DilanRG/lexicon-mcp",
        tag="data-v2.0.0",
        transformation_commit="1" * 40,
        sources=SOURCES,
        source_dataset=SOURCE_DATASET,
        created_at="2026-08-16T00:00:00Z",
    )

    parsed = parse_manifest(
        (tmp_path / "release" / "manifest.json").read_bytes()
    )

    assert parsed.schema_version == 2
    assert manifest["schema_version"] == 2
    assert parsed.languages == ("cy", "en", "gv")
    assert parsed.source_dataset.dataset_version == "data-v1.1.0"
    assert [pack.id for pack in parsed.required_packs()] == ["core"]


def test_packaged_release_resolves_a_language_to_one_component(tmp_path: Path) -> None:
    package_packs(
        built(tmp_path),
        tmp_path / "release",
        dataset_version="data-v2.0.0",
        repository="DilanRG/lexicon-mcp",
        tag="data-v2.0.0",
        transformation_commit="1" * 40,
        sources=SOURCES,
        source_dataset=SOURCE_DATASET,
        created_at="2026-08-16T00:00:00Z",
    )
    parsed = parse_manifest((tmp_path / "release" / "manifest.json").read_bytes())

    selection = resolve(parsed, languages=["gv"], capabilities=["lexical"])

    assert selection.packs == ("core", "lexical-bundle-001")
    assert selection.components == ("artifact-core", "artifact-lexical-bundle-001")


def test_every_pack_is_compressed_into_named_parts(tmp_path: Path) -> None:
    manifest = package_packs(
        built(tmp_path),
        tmp_path / "release",
        dataset_version="data-v2.0.0",
        repository="DilanRG/lexicon-mcp",
        tag="data-v2.0.0",
        transformation_commit="1" * 40,
        sources=SOURCES,
        source_dataset=SOURCE_DATASET,
        created_at="2026-08-16T00:00:00Z",
    )

    for component in manifest["components"]:
        assert component["compression"] == "zstd"
        assert component["integrity"]["dataset_schema_version"] == DATASET_SCHEMA_VERSION
        for part in component["parts"]:
            asset = tmp_path / "release" / part["name"]
            assert asset.is_file()
            assert asset.stat().st_size == part["size"]


def test_component_paths_are_grouped_by_capability(tmp_path: Path) -> None:
    manifest = package_packs(
        built(tmp_path),
        tmp_path / "release",
        dataset_version="data-v2.0.0",
        repository="DilanRG/lexicon-mcp",
        tag="data-v2.0.0",
        transformation_commit="1" * 40,
        sources=SOURCES,
        source_dataset=SOURCE_DATASET,
        created_at="2026-08-16T00:00:00Z",
    )

    paths = {component["id"]: component["path"] for component in manifest["components"]}

    assert paths["artifact-core"] == "core.sqlite3"
    assert paths["artifact-lexical-en"] == "lexical/lexical-en.sqlite3"
    assert paths["artifact-semantic-en"] == "semantic/semantic-en.sqlite3"


def test_a_release_without_a_core_pack_is_refused(tmp_path: Path) -> None:
    packs = [item for item in built(tmp_path) if item.pack.capability != "core"]

    with pytest.raises(ValueError, match="must contain a core pack"):
        package_packs(
            packs,
            tmp_path / "release",
            dataset_version="data-v2.0.0",
            repository="DilanRG/lexicon-mcp",
            tag="data-v2.0.0",
            transformation_commit="1" * 40,
            sources=SOURCES,
            source_dataset=SOURCE_DATASET,
        )


def test_source_provenance_is_carried_from_the_original_corpus(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "sources.lock.json").write_text(
        json.dumps({"sources": SOURCES}), encoding="utf-8"
    )

    assert load_sources(root) == SOURCES

    with pytest.raises(ValueError, match="cannot read source provenance"):
        load_sources(tmp_path / "missing")
