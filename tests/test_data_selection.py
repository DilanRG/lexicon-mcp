"""Schema-2 capability packs and component selection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from lexicon_mcp.data.manifest import ManifestError, parse_manifest
from lexicon_mcp.data.selection import (
    CAPABILITY_NOT_AVAILABLE_FOR_LANGUAGE,
    LANGUAGE_NOT_IN_DATASET,
    SelectionError,
    resolve,
)

VERSION = "data-v2.0.0"


def component(component_id: str, path: str, *, size: int = 1024) -> dict[str, Any]:
    payload_digest = hashlib.sha256(component_id.encode()).hexdigest()
    return {
        "id": component_id,
        "artifact_type": "lexical_sqlite",
        "path": path,
        "compression": "none",
        "compressed_size": size,
        "compressed_sha256": payload_digest,
        "final_size": size,
        "final_sha256": payload_digest,
        "parts": [
            {
                "name": f"{component_id}.part0000",
                "size": size,
                "sha256": payload_digest,
                "offset": 0,
            }
        ],
        "sources": ["fixture"],
        "integrity": {"sqlite": True},
    }


def manifest_bytes(
    *,
    packs: list[dict[str, Any]] | None = None,
    components: list[dict[str, Any]] | None = None,
    schema_version: int = 2,
    extra: dict[str, Any] | None = None,
) -> bytes:
    if components is None:
        components = [
            component("artifact-core", "core.sqlite3", size=10),
            component("artifact-lexical-en", "lexical/en.sqlite3", size=2000),
            component("artifact-lexical-bundle", "lexical/bundle.sqlite3", size=300),
            # A plain SQLite path: the semantic index schema is exercised in
            # tests/test_data_lifecycle.py, and pack selection is agnostic to it.
            component("artifact-semantic-en", "semantic/en.sqlite3", size=500),
        ]
    if packs is None:
        packs = [
            {"id": "core", "capability": "core", "component": "artifact-core"},
            {
                "id": "lexical-en",
                "capability": "lexical",
                "languages": ["en"],
                "component": "artifact-lexical-en",
            },
            {
                "id": "lexical-bundle",
                "capability": "lexical",
                "languages": ["cy", "gv"],
                "component": "artifact-lexical-bundle",
            },
            {
                "id": "semantic-en",
                "capability": "semantic",
                "languages": ["en"],
                "component": "artifact-semantic-en",
            },
        ]
    value: dict[str, Any] = {
        "schema_version": schema_version,
        "dataset_version": VERSION,
        "release": {"repository": "DilanRG/lexicon-mcp", "tag": VERSION, "immutable": True},
        "created_at": "2026-08-16T00:00:00Z",
        "transformation_commit": "1" * 40,
        "sources": [
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
        ],
        "components": components,
        "packs": packs,
    }
    if extra:
        value.update(extra)
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def test_schema_two_manifest_exposes_packs_and_derived_languages() -> None:
    manifest = parse_manifest(manifest_bytes())

    assert manifest.schema_version == 2
    assert manifest.profile == "components"
    assert manifest.languages == ("cy", "en", "gv")
    assert manifest.languages_for("lexical") == ("cy", "en", "gv")
    assert manifest.languages_for("semantic") == ("en",)
    assert [pack.id for pack in manifest.required_packs()] == ["core"]


def test_schema_two_records_transform_provenance() -> None:
    manifest = parse_manifest(
        manifest_bytes(
            extra={
                "source_dataset": {
                    "dataset_version": "data-v1.1.0",
                    "manifest_sha256": "a" * 64,
                }
            }
        )
    )

    assert manifest.source_dataset is not None
    assert manifest.source_dataset.dataset_version == "data-v1.1.0"
    assert manifest.source_dataset.manifest_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [("profile", "full"), ("languages", ["en"])],
)
def test_schema_two_rejects_schema_one_selection_fields(field: str, value: object) -> None:
    with pytest.raises(ManifestError, match="schema 2"):
        parse_manifest(manifest_bytes(extra={field: value}))


def test_one_pack_may_serve_a_language_for_a_capability() -> None:
    packs = [
        {"id": "core", "capability": "core", "component": "artifact-core"},
        {
            "id": "lexical-en",
            "capability": "lexical",
            "languages": ["en"],
            "component": "artifact-lexical-en",
        },
        {
            "id": "lexical-en-again",
            "capability": "lexical",
            "languages": ["en"],
            "component": "artifact-lexical-bundle",
        },
    ]

    with pytest.raises(ManifestError, match="already served"):
        parse_manifest(manifest_bytes(packs=packs))


def test_same_language_may_appear_under_different_capabilities() -> None:
    manifest = parse_manifest(manifest_bytes())

    assert manifest.pack_for("lexical", "en") is not None
    assert manifest.pack_for("semantic", "en") is not None


def test_packs_must_reference_a_known_component_and_include_core() -> None:
    with pytest.raises(ManifestError, match="unknown component"):
        parse_manifest(
            manifest_bytes(
                packs=[
                    {"id": "core", "capability": "core", "component": "artifact-core"},
                    {
                        "id": "lexical-en",
                        "capability": "lexical",
                        "languages": ["en"],
                        "component": "artifact-missing",
                    },
                ]
            )
        )

    with pytest.raises(ManifestError, match="at least one core pack"):
        parse_manifest(
            manifest_bytes(
                packs=[
                    {
                        "id": "lexical-en",
                        "capability": "lexical",
                        "languages": ["en"],
                        "component": "artifact-lexical-en",
                    }
                ]
            )
        )


def test_schema_one_manifests_still_parse() -> None:
    """An installed v1 dataset stays reportable after the v2 upgrade."""

    raw = json.loads(manifest_bytes())
    raw["schema_version"] = 1
    raw["profile"] = "full"
    del raw["packs"]

    manifest = parse_manifest(json.dumps(raw, sort_keys=True).encode())

    assert manifest.schema_version == 1
    assert manifest.profile == "full"
    assert manifest.packs == ()


def test_resolving_one_language_selects_core_and_its_pack() -> None:
    manifest = parse_manifest(manifest_bytes())

    selection = resolve(manifest, languages=["en"], capabilities=["lexical"])

    assert selection.packs == ("core", "lexical-en")
    assert selection.components == ("artifact-core", "artifact-lexical-en")
    assert selection.effective == {"lexical": ("en",)}
    assert selection.unavailable == ()
    assert selection.installed_size == 2010


def test_selecting_a_bundled_language_counts_its_component_once() -> None:
    manifest = parse_manifest(manifest_bytes())

    selection = resolve(manifest, languages=["cy", "gv"], capabilities=["lexical"])

    assert selection.packs == ("core", "lexical-bundle")
    assert selection.effective == {"lexical": ("cy", "gv")}
    assert selection.installed_size == 310


def test_capability_missing_for_a_known_language_is_reported_not_fatal() -> None:
    manifest = parse_manifest(manifest_bytes())

    selection = resolve(manifest, languages=["en", "cy"], capabilities=["lexical", "semantic"])

    assert selection.effective["lexical"] == ("cy", "en")
    assert selection.effective["semantic"] == ("en",)
    assert [item.as_dict() for item in selection.unavailable] == [
        {
            "capability": "semantic",
            "language": "cy",
            "reason": CAPABILITY_NOT_AVAILABLE_FOR_LANGUAGE,
        }
    ]


def test_unknown_language_is_fatal_in_strict_mode_and_reported_otherwise() -> None:
    manifest = parse_manifest(manifest_bytes())

    with pytest.raises(SelectionError, match="does not contain these languages: zz"):
        resolve(manifest, languages=["en", "zz"], capabilities=["lexical"])

    selection = resolve(manifest, languages=["en", "zz"], capabilities=["lexical"], strict=False)

    assert selection.effective["lexical"] == ("en",)
    assert selection.unavailable[0].reason == LANGUAGE_NOT_IN_DATASET


def test_requesting_every_language_expresses_a_full_install() -> None:
    manifest = parse_manifest(manifest_bytes())

    selection = resolve(manifest, languages=None, capabilities=["lexical", "semantic"])

    assert selection.effective["lexical"] == ("cy", "en", "gv")
    assert selection.effective["semantic"] == ("en",)
    assert selection.unavailable == ()
    assert selection.installed_size == 2810


def test_requested_language_tags_are_normalized() -> None:
    manifest = parse_manifest(manifest_bytes())

    selection = resolve(manifest, languages=["EN"], capabilities=["lexical"])

    assert selection.effective["lexical"] == ("en",)


def test_unknown_capability_and_schema_one_selection_are_rejected() -> None:
    manifest = parse_manifest(manifest_bytes())

    with pytest.raises(SelectionError, match="unknown capability"):
        resolve(manifest, languages=["en"], capabilities=["etymology"])

    raw = json.loads(manifest_bytes())
    raw["schema_version"] = 1
    raw["profile"] = "full"
    del raw["packs"]
    schema_one = parse_manifest(json.dumps(raw, sort_keys=True).encode())

    with pytest.raises(SelectionError, match="requires a schema 2 manifest"):
        resolve(schema_one, languages=["en"], capabilities=["lexical"])
