"""Multi-hop relation traversal across pack boundaries."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from lexicon_mcp.data.activation import Activation, ActivationComponent, ActivationPack
from lexicon_mcp.data.store import ComponentStore
from lexicon_mcp.pipeline.packs import CORE_PACK_SCHEMA, PlannedPack
from lexicon_mcp.pipeline.schema import create_lexical_schema
from lexicon_mcp.pipeline.transform import build_lexical_pack, build_term_counts
from lexicon_mcp.runtime.pack_queries import relation_rows
from lexicon_mcp.runtime.router import PackRouter
from lexicon_mcp.runtime.traversal import (
    FrontierNode,
    expand_frontier,
    frontier_from_rows,
)

VERSION = "data-v2.0.0"

# en:dog -> fr:chien -> de:hund. The second hop lives in the French pack, which
# is what makes this a routed traversal rather than a single-pack query.
TERMS = {1: ("dog", "en"), 2: ("chien", "fr"), 3: ("hund", "de"), 4: ("puppy", "en")}
RELATIONS = [
    (1, "s1", 1, 2, "s2", 1, 1),  # en -> fr
    (2, "s2", 1, 3, "s3", 1, 1),  # fr -> de, only the French pack has it
    (1, "s1", 1, 4, "s4", 1, 1),  # en -> en
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "lexicon.sqlite3"
    connection = sqlite3.connect(path)
    create_lexical_schema(connection, "data-v1.1.0")
    connection.execute(
        "INSERT INTO provenance VALUES (1,'fixture','CC0-1.0','https://fixtures.invalid')"
    )
    for term_id, (term, language) in TERMS.items():
        connection.execute(
            "INSERT INTO lexical_terms VALUES (?,?,?,?)", (term_id, term, term, language)
        )
        connection.execute(
            "INSERT INTO lexical_entries VALUES (?,?,?,?,?)",
            (f"e{term_id}", term_id, "noun", None, 1),
        )
        connection.execute(
            "INSERT INTO senses VALUES (?,?,?)", (f"s{term_id}", f"e{term_id}", f"gloss {term}")
        )
    for row in RELATIONS:
        connection.execute("INSERT INTO relations VALUES (?,?,?,?,?,?,?)", row)
    connection.commit()
    connection.close()
    return path


def core_pack(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(CORE_PACK_SCHEMA)
    connection.executemany(
        "INSERT INTO language_catalogue VALUES (?,?,?,?,?,?,?,?,?)",
        [(language, 1, 1, 1, 0, 0, 0, 0, 0) for language in ("en", "fr", "de")],
    )
    connection.commit()
    connection.close()
    return path


def install(tmp_path: Path, corpus: Path, languages: list[str]) -> PackRouter:
    """Build a pack per language, store them, and route over the result."""

    counts = build_term_counts(corpus, tmp_path / "counts.sqlite3")
    store = ComponentStore(tmp_path / "components")
    components: list[ActivationComponent] = []
    packs: list[ActivationPack] = []

    core = core_pack(tmp_path / "built" / "core.sqlite3")
    digest = hashlib.sha256(core.read_bytes()).hexdigest()
    size = core.stat().st_size
    store.adopt(core, digest)
    components.append(ActivationComponent("artifact-core", digest, "core.sqlite3", size))
    packs.append(ActivationPack("core", "core", (), ("artifact-core",)))

    for language in languages:
        built = build_lexical_pack(
            corpus,
            counts,
            tmp_path / "built" / f"{language}.sqlite3",
            PlannedPack(f"lexical-{language}", "lexical", (language,), 0),
            dataset_version=VERSION,
        )
        digest = hashlib.sha256(built.path.read_bytes()).hexdigest()
        size = built.path.stat().st_size
        store.adopt(built.path, digest)
        component_id = f"artifact-lexical-{language}"
        components.append(
            ActivationComponent(component_id, digest, f"lexical/{language}.sqlite3", size)
        )
        packs.append(
            ActivationPack(
                f"lexical-{language}", "lexical", (language,), (component_id,)
            )
        )

    activation = Activation(
        schema_version=2,
        activation_id="a" * 32,
        dataset_version=VERSION,
        manifest_sha256="b" * 64,
        created_at="2026-08-16T00:00:00Z",
        requested_languages=tuple(languages),
        requested_capabilities=("lexical",),
        effective={"lexical": tuple(sorted(languages))},
        unavailable=(),
        components=tuple(components),
        packs=tuple(packs),
    )
    return PackRouter(activation, store)


def first_hop(router: PackRouter, word: str, language: str) -> list[dict]:
    connection = router.connection_for("lexical", language)
    return [
        {key: row[key] for key in row.keys()}  # noqa: SIM118
        for row in relation_rows(
            connection, word=word, language=language, relation_code=1, limit=10
        )
    ]


def test_second_hop_is_served_by_the_pack_that_owns_it(
    tmp_path: Path, corpus: Path
) -> None:
    """en -> fr -> de: the French edge lives only in the French pack."""

    with install(tmp_path, corpus, ["en", "fr", "de"]) as router:
        rows = first_hop(router, "dog", "en")
        assert sorted(row["target_term"] for row in rows) == ["chien", "puppy"]

        result = expand_frontier(
            router, frontier_from_rows(rows), relation_code=1, limit=10
        )

    assert result.complete is True
    assert ("hund", "de") in [
        (row["target_term"], row["target_language"]) for row in result.rows
    ]


def test_an_uninstalled_intermediate_is_reported_not_silently_dropped(
    tmp_path: Path, corpus: Path
) -> None:
    """The distinction between 'no path' and 'you do not have French'."""

    with install(tmp_path, corpus, ["en"]) as router:
        rows = first_hop(router, "dog", "en")
        # English still names its French target, from the catalogue.
        assert "chien" in [row["target_term"] for row in rows]

        result = expand_frontier(
            router, frontier_from_rows(rows), relation_code=1, limit=10
        )

    assert result.complete is False
    assert result.unexpanded == [{"language": "fr", "reason": "language_not_installed"}]
    # The English half of the frontier still expanded.
    assert all(row["via_language"] == "en" for row in result.rows)


def test_each_pack_is_queried_once_for_the_whole_frontier(
    tmp_path: Path, corpus: Path
) -> None:
    with install(tmp_path, corpus, ["en", "fr", "de"]) as router:
        frontier = (
            FrontierNode("en", "dog"),
            FrontierNode("en", "puppy"),
            FrontierNode("fr", "chien"),
        )

        expand_frontier(router, frontier, relation_code=1, limit=10)

        # English and French packs; the German pack is never touched.
        assert router.open_pack_count == 2


def test_results_carry_the_node_they_were_reached_through(
    tmp_path: Path, corpus: Path
) -> None:
    with install(tmp_path, corpus, ["en", "fr"]) as router:
        result = expand_frontier(
            router, (FrontierNode("fr", "chien"),), relation_code=1, limit=10
        )

    assert result.rows[0]["via_language"] == "fr"
    assert result.rows[0]["via_word"] == "chien"


def test_the_frontier_deduplicates_repeated_targets(tmp_path: Path, corpus: Path) -> None:
    rows = [
        {"target_language": "fr", "target_normalized": "chien"},
        {"target_language": "fr", "target_normalized": "chien"},
        {"target_language": "de", "target_normalized": "hund"},
    ]

    assert frontier_from_rows(rows) == (
        FrontierNode("fr", "chien"),
        FrontierNode("de", "hund"),
    )


def test_expansion_respects_its_row_limit(tmp_path: Path, corpus: Path) -> None:
    with install(tmp_path, corpus, ["en", "fr", "de"]) as router:
        result = expand_frontier(
            router,
            (FrontierNode("en", "dog"), FrontierNode("fr", "chien")),
            relation_code=1,
            limit=1,
        )

    assert len(result.rows) == 1
