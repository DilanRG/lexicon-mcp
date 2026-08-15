"""Routing queries to packs, and reporting precisely why one is unavailable."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from lexicon_mcp.data.activation import (
    Activation,
    ActivationComponent,
    ActivationPack,
)
from lexicon_mcp.data.store import ComponentStore
from lexicon_mcp.pipeline.packs import CORE_PACK_SCHEMA
from lexicon_mcp.runtime.router import (
    CAPABILITY_NOT_INSTALLED,
    INSTALLED,
    LANGUAGE_NOT_INSTALLED,
    NOT_AVAILABLE_UPSTREAM,
    UNKNOWN_LANGUAGE,
    PackRouter,
    RouterError,
)

# language -> (terms, has_semantic, has_pronunciation, has_wordplay)
CATALOGUE = {
    "en": (2_000_000, True, True, True),
    "fr": (1_500_000, True, False, False),
    "cy": (50_000, False, False, False),
    "gv": (11_000, False, False, False),
}


def write_core(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(CORE_PACK_SCHEMA)
    connection.executemany(
        "INSERT INTO language_catalogue VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (language, terms, terms, terms, 0, 0, int(sem), int(pron), int(play))
            for language, (terms, sem, pron, play) in CATALOGUE.items()
        ],
    )
    connection.commit()
    connection.close()
    return path


def write_pack(path: Path, language: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE lexical_terms (term_id INTEGER PRIMARY KEY, language TEXT)")
    connection.execute("INSERT INTO lexical_terms VALUES (1, ?)", (language,))
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def routed(tmp_path: Path) -> PackRouter:
    """An install with English lexical+semantic, and Welsh lexical only."""

    store = ComponentStore(tmp_path / "components")
    staging = tmp_path / "staging"
    files = {
        "artifact-core": write_core(staging / "core.sqlite3"),
        "artifact-lexical-en": write_pack(staging / "en.sqlite3", "en"),
        "artifact-lexical-bundle": write_pack(staging / "bundle.sqlite3", "cy"),
        "artifact-semantic-en": write_pack(staging / "sem-en.sqlite3", "en"),
    }
    components = []
    for component_id, path in files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        store.adopt(path, digest)
        components.append(
            ActivationComponent(
                id=component_id,
                sha256=digest,
                path=f"{component_id}.sqlite3",
                size=size,
            )
        )
    activation = Activation(
        schema_version=2,
        activation_id="a" * 32,
        dataset_version="data-v2.0.0",
        manifest_sha256="b" * 64,
        created_at="2026-08-16T00:00:00Z",
        requested_languages=("en", "cy"),
        requested_capabilities=("lexical", "semantic"),
        effective={"lexical": ("cy", "en"), "semantic": ("en",)},
        unavailable=(),
        components=tuple(components),
        packs=(
            ActivationPack("core", "core", (), "artifact-core"),
            ActivationPack("lexical-en", "lexical", ("en",), "artifact-lexical-en"),
            ActivationPack(
                "lexical-bundle-001", "lexical", ("cy", "gv"), "artifact-lexical-bundle"
            ),
            ActivationPack("semantic-en", "semantic", ("en",), "artifact-semantic-en"),
        ),
    )
    router = PackRouter(activation, store, max_open_packs=2)
    yield router
    router.close()


def test_an_installed_language_routes_to_its_pack(routed: PackRouter) -> None:
    connection = routed.connection_for("lexical", "en")

    assert connection is not None
    assert connection.execute("SELECT language FROM lexical_terms").fetchone()[0] == "en"


def test_a_bundled_language_routes_to_the_bundle(routed: PackRouter) -> None:
    connection = routed.connection_for("lexical", "cy")

    assert connection.execute("SELECT language FROM lexical_terms").fetchone()[0] == "cy"


def test_availability_separates_never_installed_from_never_existed(
    routed: PackRouter,
) -> None:
    """The distinction an empty result cannot express."""

    # Installed and serving.
    assert routed.availability("lexical", "en").reason == INSTALLED

    # In the corpus, simply not installed here: installing more would fix it.
    assert routed.availability("lexical", "fr").reason == LANGUAGE_NOT_INSTALLED

    # Installed lexically, but its semantic pack was not selected.
    assert routed.availability("semantic", "cy").reason == NOT_AVAILABLE_UPSTREAM

    # French has vectors upstream but neither pack is installed here.
    assert routed.availability("semantic", "fr").reason == LANGUAGE_NOT_INSTALLED

    # Not in this dataset at all.
    assert routed.availability("lexical", "zz").reason == UNKNOWN_LANGUAGE


def test_a_capability_can_be_missing_for_an_installed_language(tmp_path: Path) -> None:
    """English is installed lexically; its semantic pack is not."""

    store = ComponentStore(tmp_path / "components")
    staging = tmp_path / "staging"
    core = write_core(staging / "core.sqlite3")
    english = write_pack(staging / "en.sqlite3", "en")
    components = []
    for component_id, path in (("artifact-core", core), ("artifact-lexical-en", english)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        store.adopt(path, digest)
        components.append(
            ActivationComponent(component_id, digest, f"{component_id}.sqlite3", size)
        )
    activation = Activation(
        schema_version=2,
        activation_id="c" * 32,
        dataset_version="data-v2.0.0",
        manifest_sha256="d" * 64,
        created_at="2026-08-16T00:00:00Z",
        requested_languages=("en",),
        requested_capabilities=("lexical",),
        effective={"lexical": ("en",)},
        unavailable=(),
        components=tuple(components),
        packs=(
            ActivationPack("core", "core", (), "artifact-core"),
            ActivationPack("lexical-en", "lexical", ("en",), "artifact-lexical-en"),
        ),
    )

    with PackRouter(activation, store) as router:
        assert router.availability("semantic", "en").reason == CAPABILITY_NOT_INSTALLED
        assert router.connection_for("semantic", "en") is None


def test_uninstalled_language_routes_to_none_rather_than_raising(
    routed: PackRouter,
) -> None:
    assert routed.connection_for("lexical", "fr") is None


def test_language_tags_are_normalized_before_routing(routed: PackRouter) -> None:
    assert routed.connection_for("lexical", "EN") is not None
    assert routed.availability("lexical", "EN").reason == INSTALLED


def test_packs_open_lazily(routed: PackRouter) -> None:
    assert routed.open_pack_count == 0

    routed.connection_for("lexical", "en")

    assert routed.open_pack_count == 1


def test_open_packs_are_capped_and_the_least_recent_is_evicted(
    routed: PackRouter,
) -> None:
    """A full install is ~55 packs; a session touches a handful."""

    routed.connection_for("lexical", "en")
    routed.connection_for("lexical", "cy")
    routed.connection_for("semantic", "en")

    assert routed.open_pack_count == 2

    # The evicted pack reopens cleanly on next use.
    assert routed.connection_for("lexical", "en") is not None
    assert routed.open_pack_count == 2


def test_reusing_a_pack_keeps_it_warm(routed: PackRouter) -> None:
    routed.connection_for("lexical", "en")
    first = routed.connection_for("lexical", "en")

    assert first is routed.connection_for("lexical", "en")
    assert routed.open_pack_count == 1


def test_a_bundle_is_one_connection_for_every_language_it_serves(
    routed: PackRouter,
) -> None:
    welsh = routed.connection_for("lexical", "cy")
    manx = routed.connection_for("lexical", "gv")

    assert welsh is manx
    assert routed.open_pack_count == 1


def test_coverage_reports_what_the_dataset_carries(routed: PackRouter) -> None:
    coverage = routed.coverage

    assert coverage["fr"].has_semantic is True
    assert coverage["cy"].has_semantic is False
    assert coverage["en"].has_wordplay is True
    assert coverage["fr"].offers("lexical") is True
    assert coverage["fr"].offers("wordplay") is False


def test_an_activation_without_a_core_pack_cannot_answer_coverage(
    tmp_path: Path,
) -> None:
    activation = Activation(
        schema_version=2,
        activation_id="e" * 32,
        dataset_version="data-v2.0.0",
        manifest_sha256="f" * 64,
        created_at="2026-08-16T00:00:00Z",
        requested_languages=("en",),
        requested_capabilities=("lexical",),
        effective={},
        unavailable=(),
        components=(ActivationComponent("artifact-x", "0" * 64, "x.sqlite3", 1),),
        packs=(ActivationPack("lexical-en", "lexical", ("en",), "artifact-x"),),
    )

    router = PackRouter(activation, ComponentStore(tmp_path / "components"))

    with pytest.raises(RouterError, match="no core pack"):
        router.coverage  # noqa: B018  (property access is the call under test)


def test_closing_releases_every_open_pack(routed: PackRouter) -> None:
    routed.connection_for("lexical", "en")
    routed.connection_for("lexical", "cy")

    routed.close()

    assert routed.open_pack_count == 0
