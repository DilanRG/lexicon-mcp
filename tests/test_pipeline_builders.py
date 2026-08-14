from __future__ import annotations

import gzip
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zstandard
from usearch.index import Index

import lexicon_mcp.pipeline.orchestrator as orchestrator
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
    assert result["stage_counts"]["wiktextract"]["language_codes"] == 7
    assert result["corpus_floors"]["wiktextract"]["entries"]["passed"] is False
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
        """SELECT s.sense_id,s.gloss FROM senses s
        JOIN lexical_entries e USING(entry_id)
        JOIN lexical_terms w ON w.term_id=e.term_id
        WHERE w.language='en' AND w.normalized_term='bank' AND s.gloss IS NOT NULL
        ORDER BY s.sense_id"""
    ).fetchall()
    assert any("financial" in row["gloss"] for row in bank)
    assert any("river" in row["gloss"] or "water" in row["gloss"] for row in bank)
    translated = connection.execute(
        """SELECT s.sense_id,s.gloss,t.term FROM senses s
        JOIN lexical_entries e USING(entry_id)
        JOIN lexical_terms source_term ON source_term.term_id=e.term_id
        JOIN translations x USING(sense_id)
        JOIN lexical_terms t ON t.term_id=x.target_term_id
        WHERE source_term.normalized_term='bank' AND t.language='de' ORDER BY t.term"""
    ).fetchall()
    assert {(row["term"], row["gloss"]) for row in translated} == {
        ("Bank", "institution"),
        ("Ufer", "edge of a river or lake"),
    }
    assert all(":labeled:" in row["sense_id"] for row in translated)
    assert connection.execute(
        "SELECT rowid FROM wordplay_fts WHERE wordplay_fts MATCH 'ca*'"
    ).fetchone() is not None

    clear_synonyms = connection.execute(
        """SELECT s.sense_id,s.gloss,y.term
        FROM senses s
        JOIN lexical_entries e USING(entry_id)
        JOIN lexical_terms source_term ON source_term.term_id=e.term_id
        JOIN synonyms x USING(sense_id)
        JOIN lexical_terms y ON y.term_id=x.target_term_id
        WHERE source_term.normalized_term='clear' ORDER BY y.term"""
    ).fetchall()
    assert {(row["term"], row["gloss"]) for row in clear_synonyms} == {
        ("obvious", "easy to understand"),
        ("transparent", "allowing light through"),
        ("plain", None),
    }
    assert next(row for row in clear_synonyms if row["term"] == "plain")[
        "sense_id"
    ].startswith("wikt:unsensed:")

    clear_antonyms = connection.execute(
        """SELECT s.sense_id,s.gloss,target_term.term AS target_term
        FROM senses s
        JOIN lexical_entries e USING(entry_id)
        JOIN lexical_terms source_term ON source_term.term_id=e.term_id
        JOIN relations r ON r.source_sense_id=s.sense_id
        JOIN lexical_terms target_term ON target_term.term_id=r.target_term_id
        WHERE source_term.normalized_term='clear' AND r.relation_code=2
        ORDER BY target_term.term"""
    ).fetchall()
    assert {(row["target_term"], row["gloss"]) for row in clear_antonyms} == {
        ("confusing", "easy to understand"),
        ("opaque", "allowing light through"),
        ("murky", None),
    }
    assert next(row for row in clear_antonyms if row["target_term"] == "murky")[
        "sense_id"
    ].startswith("wikt:unsensed:")

    lead_ipa = connection.execute(
        """SELECT p.ipa FROM pronunciations p
        JOIN lexical_entries e USING(entry_id)
        JOIN lexical_terms w ON w.term_id=e.term_id
        WHERE w.normalized_term='lead' ORDER BY p.position"""
    ).fetchall()
    assert [row["ipa"] for row in lead_ipa] == ["/li\u02d0d/", "/lɛd/"]

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
            """SELECT 1 FROM senses s
            JOIN lexical_entries e USING(entry_id)
            JOIN lexical_terms w ON w.term_id=e.term_id
            WHERE w.normalized_term=? AND w.language=?""",
            (normalize_term(word), language),
        ).fetchone()

    assert connection.execute(
        """SELECT 1 FROM relations r
        JOIN lexical_terms s ON s.term_id=r.source_term_id
        JOIN lexical_terms t ON t.term_id=r.target_term_id
        WHERE s.normalized_term='dog' AND r.relation_code=3
        AND t.normalized_term='animal'"""
    ).fetchone()
    assert connection.execute(
        """SELECT 1 FROM relations r
        JOIN lexical_terms s ON s.term_id=r.source_term_id
        JOIN lexical_terms t ON t.term_id=r.target_term_id
        WHERE s.normalized_term='poodle' AND r.relation_code=3
        AND t.normalized_term='dog'"""
    ).fetchone()
    assert connection.execute(
        """SELECT 1 FROM relations r
        JOIN lexical_terms s ON s.term_id=r.source_term_id
        JOIN lexical_terms t ON t.term_id=r.target_term_id
        WHERE s.normalized_term='wheel' AND r.relation_code=6
        AND t.normalized_term='car'"""
    ).fetchone()
    assert connection.execute(
        """SELECT 1 FROM relations r
        JOIN lexical_terms s ON s.term_id=r.source_term_id
        JOIN lexical_terms t ON t.term_id=r.target_term_id
        WHERE s.normalized_term='knife' AND r.relation_code=9
        AND t.normalized_term='cutting' AND r.direction_code=1"""
    ).fetchone()
    assert connection.execute(
        """SELECT 1 FROM relations r
        JOIN lexical_terms s ON s.term_id=r.source_term_id
        JOIN lexical_terms t ON t.term_id=r.target_term_id
        WHERE s.normalized_term='book' AND r.relation_code=11
        AND t.normalized_term='library'"""
    ).fetchone()
    assert connection.execute(
        """SELECT 1 FROM relations r
        JOIN lexical_terms s ON s.term_id=r.source_term_id
        JOIN lexical_terms t ON t.term_id=r.target_term_id
        WHERE s.normalized_term='important' AND r.relation_code=1
        AND t.normalized_term='significant'"""
    ).fetchone()

    pronunciations = dict(
        connection.execute(
            """SELECT t.normalized_term,p.phonemes FROM pronunciations_words p
            JOIN lexical_terms t USING(term_id)
            WHERE t.normalized_term IN ('cat','bat','knight','night')"""
        )
    )
    assert pronunciations["cat"].split()[1:] == pronunciations["bat"].split()[1:]
    assert pronunciations["knight"] == pronunciations["night"]
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert metadata["build.wiktextract.language_codes"] == "7"
    assert metadata["build.conceptnet.source_assertions"] == "4"
    assert metadata["schema_version"] == "2"
    assert connection.execute("PRAGMA page_size").fetchone()[0] == 32768
    indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
    }
    assert {
        "lexical_entries_lookup",
        "relations_source_lookup",
        "pronunciations_words_rhyme",
    } <= indexes
    connection.close()

    build_manifest = json.loads((root / "build-manifest.json").read_text(encoding="utf-8"))
    measured_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert build_manifest["installed_size"] == measured_size
    assert build_manifest["resource_projection"]["total"] < 30 * 1024**3


def test_semantic_artifacts_share_dense_global_ids(tmp_path: Path) -> None:
    root = _build(tmp_path)
    semantic = root / "semantic"
    connection = sqlite3.connect(semantic / "mapping.sqlite3")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert metadata["dimensions"] == "4"
    assert metadata["vector_dtype"] == "float16"
    assert metadata["schema_version"] == "2"
    assert metadata["expansion_add"] == "256"
    assert metadata["expansion_search"] == "512"
    assert metadata["source"] == "ConceptNet Numberbatch 19.08"
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


def test_wiktextract_gzip_input_and_completed_build_is_never_blindly_trusted(
    tmp_path: Path,
) -> None:
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
    assert first["dataset_version"] == "fixture-v1"
    with pytest.raises(FileExistsError, match="refusing to replace existing output"):
        build_full_corpus(
            inputs,
            output,
            tmp_path / "state",
            dataset_version="fixture-v1",
            retrieved_at="2026-01-01T00:00:00Z",
        )


def test_stale_checkpoints_cannot_skip_a_recreated_partial_database(tmp_path: Path) -> None:
    first = _build(tmp_path)
    state = tmp_path / "state"
    import shutil

    shutil.rmtree(first)
    rebuilt = tmp_path / "fixture-v1"
    build_full_corpus(
        BuildInputs(
            oewn=FIXTURES / "oewn.xml",
            wiktextract=(FIXTURES / "kaikki.jsonl",),
            conceptnet=FIXTURES / "conceptnet.tsv",
            numberbatch=FIXTURES / "numberbatch.txt",
            cmudict=FIXTURES / "cmudict.dict",
            notices_dir=FIXTURES / "notices",
        ),
        rebuilt,
        state,
        dataset_version="fixture-v1",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    with sqlite3.connect(rebuilt / "lexicon.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM senses").fetchone()[0] > 0


def test_interrupted_build_reuses_existing_notices_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = BuildInputs(
        oewn=FIXTURES / "oewn.xml",
        wiktextract=(FIXTURES / "kaikki.jsonl",),
        conceptnet=FIXTURES / "conceptnet.tsv",
        numberbatch=FIXTURES / "numberbatch.txt",
        cmudict=FIXTURES / "cmudict.dict",
        notices_dir=FIXTURES / "notices",
    )
    output = tmp_path / "resumed-notices"
    original = orchestrator.build_oewn

    def interrupt(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(orchestrator, "build_oewn", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")

    partial_notices = output.with_name(output.name + ".partial") / "notices"
    assert (partial_notices / "DATA_LICENSES.md").is_file()
    monkeypatch.setattr(orchestrator, "build_oewn", original)
    build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")
    assert (output / "notices" / "DATA_LICENSES.md").is_file()


def test_changed_pipeline_identity_invalidates_semantic_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = BuildInputs(
        oewn=FIXTURES / "oewn.xml",
        wiktextract=(FIXTURES / "kaikki.jsonl",),
        conceptnet=FIXTURES / "conceptnet.tsv",
        numberbatch=FIXTURES / "numberbatch.txt",
        cmudict=FIXTURES / "cmudict.dict",
        notices_dir=FIXTURES / "notices",
    )
    output = tmp_path / "semantic-checkpoint"
    real_replace = orchestrator.os.replace

    def interrupt_final_promotion(source: object, destination: object) -> None:
        if Path(destination) == output:
            raise RuntimeError("simulated final promotion interruption")
        real_replace(source, destination)

    monkeypatch.setattr(orchestrator.os, "replace", interrupt_final_promotion)
    with pytest.raises(RuntimeError, match="final promotion interruption"):
        build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")
    monkeypatch.setattr(orchestrator.os, "replace", real_replace)

    calls = 0
    real_numberbatch = orchestrator.build_numberbatch

    def counted_numberbatch(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return real_numberbatch(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "build_numberbatch", counted_numberbatch)
    monkeypatch.setattr(orchestrator, "_pipeline_identity", lambda: "changed-pipeline")
    build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")

    assert calls == 1


def test_truncated_lexical_rows_cannot_be_resumed_from_external_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = BuildInputs(
        oewn=FIXTURES / "oewn.xml",
        wiktextract=(FIXTURES / "kaikki.jsonl",),
        conceptnet=FIXTURES / "conceptnet.tsv",
        numberbatch=FIXTURES / "numberbatch.txt",
        cmudict=FIXTURES / "cmudict.dict",
        notices_dir=FIXTURES / "notices",
    )
    output = tmp_path / "lexical-checkpoint"
    real_replace = orchestrator.os.replace

    def interrupt_final_promotion(source: object, destination: object) -> None:
        if Path(destination) == output:
            raise RuntimeError("simulated final promotion interruption")
        real_replace(source, destination)

    monkeypatch.setattr(orchestrator.os, "replace", interrupt_final_promotion)
    with pytest.raises(RuntimeError, match="final promotion interruption"):
        build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")
    monkeypatch.setattr(orchestrator.os, "replace", real_replace)

    partial = output.with_name(output.name + ".partial")
    with closing(sqlite3.connect(partial / "lexicon.sqlite3")) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("DELETE FROM senses")
        connection.commit()

    calls = 0
    real_wiktextract = orchestrator.build_wiktextract

    def counted_wiktextract(*args: object, **kwargs: object) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return real_wiktextract(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "build_wiktextract", counted_wiktextract)
    build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")
    assert calls == 1
    with closing(sqlite3.connect(output / "lexicon.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM senses").fetchone()[0] > 0


def test_corrupt_semantic_index_forces_numberbatch_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = BuildInputs(
        oewn=FIXTURES / "oewn.xml",
        wiktextract=(FIXTURES / "kaikki.jsonl",),
        conceptnet=FIXTURES / "conceptnet.tsv",
        numberbatch=FIXTURES / "numberbatch.txt",
        cmudict=FIXTURES / "cmudict.dict",
        notices_dir=FIXTURES / "notices",
    )
    output = tmp_path / "semantic-corruption"
    real_replace = orchestrator.os.replace

    def interrupt_final_promotion(source: object, destination: object) -> None:
        if Path(destination) == output:
            raise RuntimeError("simulated final promotion interruption")
        real_replace(source, destination)

    monkeypatch.setattr(orchestrator.os, "replace", interrupt_final_promotion)
    with pytest.raises(RuntimeError, match="final promotion interruption"):
        build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")
    monkeypatch.setattr(orchestrator.os, "replace", real_replace)

    partial = output.with_name(output.name + ".partial")
    (partial / "semantic" / "indexes" / "global.usearch").write_bytes(b"corrupt")
    calls = 0
    real_numberbatch = orchestrator.build_numberbatch

    def counted_numberbatch(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return real_numberbatch(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "build_numberbatch", counted_numberbatch)
    build_full_corpus(inputs, output, tmp_path / "state", dataset_version="fixture-v1")
    assert calls == 1
    assert len(Index.restore(output / "semantic" / "indexes" / "global.usearch")) == 10


def test_full_corpus_floors_reject_fixture_without_guessing_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The clean GitHub Windows runner exposes ~35 GiB on its temporary drive,
    # below the production 80 GiB peak preflight. This unit test is about the
    # corpus floor report, so isolate it from host disk capacity.
    monkeypatch.setattr(
        orchestrator.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024**3),
    )
    with pytest.raises(RuntimeError, match=r"wiktextract\.entries.*numberbatch\.terms"):
        build_full_corpus(
            BuildInputs(
                oewn=FIXTURES / "oewn.xml",
                wiktextract=(FIXTURES / "kaikki.jsonl",),
                conceptnet=FIXTURES / "conceptnet.tsv",
                numberbatch=FIXTURES / "numberbatch.txt",
                cmudict=FIXTURES / "cmudict.dict",
                notices_dir=FIXTURES / "notices",
            ),
            tmp_path / "full-floor-failure",
            tmp_path / "full-floor-state",
            dataset_version="fixture-v1",
            retrieved_at="2026-01-01T00:00:00Z",
            enforce_corpus_floors=True,
        )


def test_resource_projection_fails_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "no-output"
    state = tmp_path / "no-state"

    def reject() -> dict[str, int]:
        raise RuntimeError("projection rejected")

    monkeypatch.setattr(orchestrator, "assert_size_targets", reject)
    with pytest.raises(RuntimeError, match="projection rejected"):
        build_full_corpus(
            BuildInputs(
                oewn=FIXTURES / "oewn.xml",
                wiktextract=(FIXTURES / "kaikki.jsonl",),
                conceptnet=FIXTURES / "conceptnet.tsv",
                numberbatch=FIXTURES / "numberbatch.txt",
                cmudict=FIXTURES / "cmudict.dict",
                notices_dir=FIXTURES / "notices",
            ),
            output,
            state,
            dataset_version="fixture-v1",
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()
    assert not state.exists()


def test_measured_installed_size_gate_blocks_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "INSTALLED_LIMIT", 1)
    output = tmp_path / "oversized"
    with pytest.raises(RuntimeError, match="installed-size gate failure"):
        build_full_corpus(
            BuildInputs(
                oewn=FIXTURES / "oewn.xml",
                wiktextract=(FIXTURES / "kaikki.jsonl",),
                conceptnet=FIXTURES / "conceptnet.tsv",
                numberbatch=FIXTURES / "numberbatch.txt",
                cmudict=FIXTURES / "cmudict.dict",
                notices_dir=FIXTURES / "notices",
            ),
            output,
            tmp_path / "oversized-state",
            dataset_version="fixture-v1",
        )
    assert not output.exists()


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
    assert all("row_count" in item and "row_digest" in item for item in manifest["sources"])
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
    assert mapping["integrity"]["dataset_schema_version"] == 2
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
