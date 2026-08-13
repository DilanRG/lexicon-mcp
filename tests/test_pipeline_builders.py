from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import numpy as np
import zstandard
from usearch.index import Index

from lexicon_mcp.data.manifest import parse_manifest
from lexicon_mcp.pipeline import BuildInputs, build_full_corpus
from lexicon_mcp.pipeline.common import normalize_term, stable_id
from lexicon_mcp.pipeline.manifest import package_dataset

FIXTURES = Path(__file__).parent / "fixtures" / "build_inputs"


def _build(tmp_path: Path) -> Path:
    output = tmp_path / "fixture-v1"
    result = build_full_corpus(
        BuildInputs(
            oewn=FIXTURES / "oewn.xml",
            wiktextract=(FIXTURES / "kaikki.jsonl",),
            conceptnet=FIXTURES / "conceptnet.tsv",
            numberbatch=FIXTURES / "numberbatch.txt",
            cmudict=FIXTURES / "cmudict.dict",
            notices_dir=FIXTURES / "notices",
        ),
        output,
        tmp_path / "state",
        dataset_version="fixture-v1",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    assert result["ngrams_included"] is False
    return output


def test_normalization_and_ids_are_unicode_stable() -> None:
    assert normalize_term("  CAFÉ\u3000") == "café"
    assert normalize_term("\uff21\uff22\uff23") == "abc"
    assert stable_id("wikt:de", "Schloss", "noun") == stable_id(
        "wikt:de", "Schloss", "noun"
    )


def test_full_fixture_build_has_senses_translations_relations_and_wordplay(tmp_path: Path) -> None:
    root = _build(tmp_path)
    connection = sqlite3.connect(root / "lexicon.sqlite3")
    connection.row_factory = sqlite3.Row

    bank = connection.execute(
        "SELECT sense_id,gloss FROM senses WHERE language='en' AND normalized_word='bank' "
        "AND gloss IS NOT NULL ORDER BY sense_id"
    ).fetchall()
    assert any("financial" in row["gloss"] for row in bank)
    assert any("river" in row["gloss"] or "water" in row["gloss"] for row in bank)
    translated = connection.execute(
        """SELECT s.gloss,t.term FROM senses s JOIN translations t USING(sense_id)
        WHERE s.normalized_word='bank' AND t.target_language='de' ORDER BY t.term"""
    ).fetchall()
    assert {(row["term"], "financial" in row["gloss"]) for row in translated} == {
        ("Bank", True),
        ("Ufer", False),
    }

    lead_ipa = connection.execute(
        """SELECT p.ipa,s.sense_id FROM pronunciations p JOIN senses s USING(sense_id)
        WHERE s.normalized_word='lead' ORDER BY p.position"""
    ).fetchall()
    assert [row["ipa"] for row in lead_ipa] == ["/li\u02d0d/", "/lɛd/"]
    assert all(":unsensed:" in row["sense_id"] for row in lead_ipa)

    for word, language in (
        ("Schloss", "de"),
        ("Gift", "de"),
        ("bright", "en"),
        ("feliz", "es"),
        ("café", "fr"),
        ("猫", "ja"),
        ("سلام", "ar"),
        ("घर", "hi"),
    ):
        assert connection.execute(
            "SELECT 1 FROM senses WHERE normalized_word=? AND language=?",
            (normalize_term(word), language),
        ).fetchone()

    assert connection.execute(
        "SELECT 1 FROM relations WHERE source_normalized='dog' AND relation='hypernym' "
        "AND target_normalized='animal'"
    ).fetchone()
    assert connection.execute(
        "SELECT 1 FROM relations WHERE source_normalized='poodle' AND relation='hypernym' "
        "AND target_normalized='dog'"
    ).fetchone()
    assert connection.execute(
        "SELECT 1 FROM relations WHERE source_normalized='car' AND relation='meronym' "
        "AND target_normalized='wheel'"
    ).fetchone()
    assert connection.execute(
        "SELECT 1 FROM relations WHERE source_normalized='knife' AND relation='used_for' "
        "AND target_normalized='cutting' AND direction='outbound'"
    ).fetchone()
    assert connection.execute(
        "SELECT 1 FROM relations WHERE source_normalized='book' AND relation='at_location' "
        "AND target_normalized='library'"
    ).fetchone()
    assert connection.execute(
        "SELECT 1 FROM relations WHERE source_normalized='important' AND relation='synonym' "
        "AND target_normalized='significant'"
    ).fetchone()

    pronunciations = dict(
        connection.execute(
            "SELECT normalized_word,phonemes FROM pronunciations_words "
            "WHERE normalized_word IN ('cat','bat','knight','night')"
        )
    )
    assert pronunciations["cat"].split()[1:] == pronunciations["bat"].split()[1:]
    assert pronunciations["knight"] == pronunciations["night"]
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_semantic_artifacts_share_dense_global_ids(tmp_path: Path) -> None:
    root = _build(tmp_path)
    semantic = root / "semantic"
    connection = sqlite3.connect(semantic / "mapping.sqlite3")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert metadata["dimensions"] == "4"
    assert metadata["vector_dtype"] == "float16"
    rows = connection.execute(
        "SELECT semantic_id,vector_offset FROM semantic_terms ORDER BY semantic_id"
    ).fetchall()
    assert rows == [(value, value) for value in range(10)]
    languages = dict(connection.execute("SELECT language,term_count FROM semantic_languages"))
    assert languages["en"] == 4
    connection.close()

    vectors = np.memmap(semantic / "vectors" / "global.f16", dtype=np.float16, mode="r")
    assert vectors.size == 40
    global_index = Index.restore(semantic / "indexes" / "global.usearch", view=True)
    assert len(global_index) == 10
    english = Index.restore(semantic / "indexes" / "languages" / "en.usearch", view=True)
    assert len(english) == 4
    matches = english.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 4)
    assert set(int(key) for key in matches.keys).issubset({0, 1, 2, 9})


def test_wiktextract_gzip_input_and_resumed_completed_build(tmp_path: Path) -> None:
    compressed = tmp_path / "kaikki.jsonl.gz"
    with gzip.open(compressed, "wb") as output:
        output.write((FIXTURES / "kaikki.jsonl").read_bytes())
    inputs = BuildInputs(
        oewn=FIXTURES / "oewn.xml",
        wiktextract=(compressed,),
        conceptnet=FIXTURES / "conceptnet.tsv",
        numberbatch=FIXTURES / "numberbatch.txt",
        cmudict=FIXTURES / "cmudict.dict",
        notices_dir=FIXTURES / "notices",
    )
    output = tmp_path / "dataset"
    first = build_full_corpus(
        inputs,
        output,
        tmp_path / "state",
        dataset_version="fixture-v1",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    second = build_full_corpus(
        inputs,
        output,
        tmp_path / "state",
        dataset_version="fixture-v1",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    assert second == first


def test_release_parts_reconstruct_every_component(tmp_path: Path) -> None:
    root = _build(tmp_path)
    package = tmp_path / "release"
    manifest = package_dataset(
        root,
        package,
        dataset_version="fixture-v1",
        repository="DilanRG/lexicon-mcp",
        tag="fixture-v1",
        transformation_commit="0" * 40,
        max_part_size=127,
        created_at="2026-01-01T00:00:00Z",
    )
    assert manifest["schema_version"] == 1
    assert manifest["release"]["immutable"] is True
    for component in manifest["components"]:
        compressed = b"".join((package / part["name"]).read_bytes() for part in component["parts"])
        rebuilt = zstandard.ZstdDecompressor().decompress(
            compressed, max_output_size=component["final_size"]
        )
        assert rebuilt == (root / component["path"]).read_bytes()
        assert all(part["size"] <= 127 for part in component["parts"])
        assert component["compression"] == "zstd"
        assert "/" not in component["id"]
    mapping = next(
        item
        for item in manifest["components"]
        if item["path"] == "semantic/mapping.sqlite3"
    )
    assert mapping["integrity"]["semantic_mapping_table"] == "semantic_terms"
    global_index = next(
        item
        for item in manifest["components"]
        if item["path"] == "semantic/indexes/global.usearch"
    )
    assert global_index["artifact_type"] == "semantic_index"
    assert global_index["integrity"]["semantic_count"] == 10
    assert json.loads((package / "manifest.json").read_text(encoding="utf-8")) == manifest
    parsed = parse_manifest((package / "manifest.json").read_bytes())
    assert parsed.dataset_version == "fixture-v1"
    assert len(parsed.components) == len(manifest["components"])
