"""Installing, amending and rolling back schema-2 component datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lexicon_mcp.data.component_lifecycle import ComponentLifecycle
from lexicon_mcp.data.lifecycle import DatasetLifecycle, LifecycleError
from lexicon_mcp.data.selection import SelectionError

VERSION = "data-v2.0.0"

PAYLOADS = {
    "artifact-core": ("core.sqlite3", b"core catalogue payload"),
    "artifact-lexical-en": ("lexical/en.sqlite3", b"english lexical payload" * 4),
    "artifact-lexical-fr": ("lexical/fr.sqlite3", b"french lexical payload" * 3),
    "artifact-lexical-bundle": ("lexical/bundle.sqlite3", b"bundled payload" * 2),
    "artifact-semantic-en": ("semantic/en.sqlite3", b"english vectors"),
}

PACKS = [
    {"id": "core", "capability": "core", "component": "artifact-core"},
    {
        "id": "lexical-en",
        "capability": "lexical",
        "languages": ["en"],
        "component": "artifact-lexical-en",
    },
    {
        "id": "lexical-fr",
        "capability": "lexical",
        "languages": ["fr"],
        "component": "artifact-lexical-fr",
    },
    {
        "id": "lexical-bundle-001",
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


class NoNetwork:
    def open(self, url: str, offset: int = 0) -> Any:
        raise AssertionError(f"unexpected network request: {url}")


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """A local schema-2 release directory, installable without a network."""

    package = tmp_path / "release"
    package.mkdir()
    components = []
    for component_id, (path, payload) in PAYLOADS.items():
        digest = hashlib.sha256(payload).hexdigest()
        (package / f"{component_id}.part0000").write_bytes(payload)
        components.append(
            {
                "id": component_id,
                "artifact_type": "lexical_sqlite",
                "path": path,
                "compression": "none",
                "compressed_size": len(payload),
                "compressed_sha256": digest,
                "final_size": len(payload),
                "final_sha256": digest,
                "parts": [
                    {
                        "name": f"{component_id}.part0000",
                        "size": len(payload),
                        "sha256": digest,
                        "offset": 0,
                    }
                ],
                "sources": ["fixture"],
                "integrity": {},
            }
        )
    manifest = {
        "schema_version": 2,
        "dataset_version": VERSION,
        "release": {"repository": "DilanRG/lexicon-mcp", "tag": VERSION, "immutable": True},
        "created_at": "2026-08-16T00:00:00Z",
        "transformation_commit": "1" * 40,
        "source_dataset": {"dataset_version": "data-v1.1.0", "manifest_sha256": "a" * 64},
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
        "packs": PACKS,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return package


@pytest.fixture
def manager(tmp_path: Path) -> ComponentLifecycle:
    fetcher = DatasetLifecycle(
        tmp_path / "data",
        transport=NoNetwork(),
        sleep=lambda _seconds: None,
        safety_margin=0,
    )
    return ComponentLifecycle(tmp_path / "data", fetcher=fetcher)


def test_installing_one_language_fetches_only_its_components(
    manager: ComponentLifecycle, release: Path
) -> None:
    result = manager.install(release, languages=["en"], capabilities=["lexical"])

    assert result["action"] == "installed"
    assert sorted(result["components_fetched"]) == ["artifact-core", "artifact-lexical-en"]
    assert result["effective"] == {"lexical": ["en"]}
    # French and the bundle were never downloaded.
    assert len(list(manager.store.iter_digests())) == 2
    assert manager.verify()["ok"] is True


def test_reinstalling_the_same_selection_changes_nothing(
    manager: ComponentLifecycle, release: Path
) -> None:
    first = manager.install(release, languages=["en"], capabilities=["lexical"])

    second = manager.install(release, languages=["en"], capabilities=["lexical"])

    assert second["action"] == "unchanged"
    assert second["components_fetched"] == []
    assert second["activation_id"] == first["activation_id"]
    assert len(manager.activations()) == 1


def test_adding_a_language_reuses_everything_already_held(
    manager: ComponentLifecycle, release: Path
) -> None:
    """The property the content-addressed store exists to provide."""

    manager.install(release, languages=["en"], capabilities=["lexical"])

    result = manager.add_languages(release, languages=["fr"])

    assert result["components_fetched"] == ["artifact-lexical-fr"]
    assert result["components_reused"] == 2
    assert result["effective"] == {"lexical": ["en", "fr"]}
    assert manager.verify()["ok"] is True


def test_removing_a_language_keeps_its_component_for_instant_rollback(
    manager: ComponentLifecycle, release: Path
) -> None:
    manager.install(release, languages=["en", "fr"], capabilities=["lexical"])
    before = set(manager.store.iter_digests())

    result = manager.remove_languages(release, languages=["fr"])

    assert result["effective"] == {"lexical": ["en"]}
    assert set(manager.store.iter_digests()) == before
    assert manager.active_activation().installed_languages("lexical") == ("en",)


def test_rollback_is_a_pointer_swap(manager: ComponentLifecycle, release: Path) -> None:
    first = manager.install(release, languages=["en"], capabilities=["lexical"])
    manager.add_languages(release, languages=["fr"])

    result = manager.activate(first["activation_id"])

    assert result["action"] == "activated"
    assert manager.active_activation().activation_id == first["activation_id"]
    assert manager.active_activation().installed_languages("lexical") == ("en",)


def test_pruning_reclaims_only_unreferenced_components(
    manager: ComponentLifecycle, release: Path
) -> None:
    first = manager.install(release, languages=["en"], capabilities=["lexical"])
    manager.add_languages(release, languages=["fr"])

    # While the English-only activation is retained, nothing is unreferenced.
    assert manager.prune()["removed"] == []

    manager.forget(first["activation_id"])
    removed = manager.prune()["removed"]

    assert removed == []  # the wider activation still references every component
    assert manager.verify()["ok"] is True


def test_pruning_drops_components_no_activation_still_needs(
    manager: ComponentLifecycle, release: Path
) -> None:
    wide = manager.install(release, languages=["en", "fr"], capabilities=["lexical"])
    narrow = manager.remove_languages(release, languages=["fr"])

    manager.forget(wide["activation_id"])
    result = manager.prune()

    french = hashlib.sha256(PAYLOADS["artifact-lexical-fr"][1]).hexdigest()
    assert result["removed"] == [french]
    assert manager.active_activation().activation_id == narrow["activation_id"]
    assert manager.verify()["ok"] is True


def test_verification_is_scoped_to_what_was_installed(
    manager: ComponentLifecycle, release: Path
) -> None:
    """A subset install is missing most of the release by design."""

    manager.install(release, languages=["en"], capabilities=["lexical"])

    report = manager.verify()

    assert report["ok"] is True
    assert report["components_checked"] == 2


def test_verification_detects_a_component_corrupted_in_the_store(
    manager: ComponentLifecycle, release: Path
) -> None:
    manager.install(release, languages=["en"], capabilities=["lexical"])
    digest = hashlib.sha256(PAYLOADS["artifact-lexical-en"][1]).hexdigest()
    manager.store.path_for(digest).write_bytes(b"rotted")

    report = manager.verify()

    assert report["ok"] is False
    assert report["damaged_components"] == ["artifact-lexical-en"]


def test_a_capability_missing_upstream_is_recorded_on_the_activation(
    manager: ComponentLifecycle, release: Path
) -> None:
    manager.install(release, languages=["en", "cy"], capabilities=["lexical", "semantic"])

    activation = manager.active_activation()

    assert activation.effective["semantic"] == ("en",)
    assert activation.unavailable == (
        {
            "capability": "semantic",
            "language": "cy",
            "reason": "capability_not_available_for_language",
        },
    )


def test_a_failed_fetch_never_moves_the_pointer(
    manager: ComponentLifecycle, release: Path, tmp_path: Path
) -> None:
    manager.install(release, languages=["en"], capabilities=["lexical"])
    stable = manager.active_activation().activation_id
    (release / "artifact-lexical-fr.part0000").write_bytes(b"corrupted asset")

    with pytest.raises(LifecycleError):
        manager.add_languages(release, languages=["fr"])

    assert manager.active_activation().activation_id == stable
    assert manager.verify()["ok"] is True


def test_an_unknown_language_is_refused_before_anything_is_fetched(
    manager: ComponentLifecycle, release: Path
) -> None:
    with pytest.raises(SelectionError, match="does not contain these languages"):
        manager.install(release, languages=["zz"], capabilities=["lexical"])

    assert list(manager.store.iter_digests()) == []
    assert manager.status()["current"] is None


def test_status_reports_a_schema_one_install_as_incompatible(
    manager: ComponentLifecycle, tmp_path: Path
) -> None:
    manager.root.mkdir(parents=True, exist_ok=True)
    (manager.root / "current.json").write_text(
        json.dumps({"schema_version": 1, "version": "data-v1.1.0"}), encoding="utf-8"
    )

    status = manager.status()

    assert status["current"] is None
    assert "schema 1 layout" in status["current_error"]


def test_amending_requires_a_matching_release(
    manager: ComponentLifecycle, release: Path
) -> None:
    manager.install(release, languages=["en"], capabilities=["lexical"])
    other = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    other["created_at"] = "2026-08-17T00:00:00Z"
    (release / "other.json").write_text(json.dumps(other, sort_keys=True), encoding="utf-8")

    with pytest.raises(LifecycleError, match="does not match the activated release"):
        manager.add_languages(release / "other.json", languages=["fr"])


def test_an_install_must_retain_at_least_one_language(
    manager: ComponentLifecycle, release: Path
) -> None:
    manager.install(release, languages=["en"], capabilities=["lexical"])

    with pytest.raises(LifecycleError, match="at least one language"):
        manager.remove_languages(release, languages=["en"])
