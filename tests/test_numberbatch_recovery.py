from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

import lexicon_mcp.pipeline.numberbatch as numberbatch
import lexicon_mcp.pipeline.orchestrator as orchestrator
from lexicon_mcp.data.locking import InstallationLock, LockBusyError
from lexicon_mcp.pipeline import (
    BuildInputs,
    build_full_corpus,
    recover_full_corpus_from_semantic_partial,
)
from lexicon_mcp.pipeline.common import file_sha256
from lexicon_mcp.usearch_compat import index_count

FIXTURES = Path(__file__).parent / "fixtures" / "build_inputs"


def _inputs() -> BuildInputs:
    return BuildInputs(
        oewn=FIXTURES / "oewn.xml",
        wiktextract=(FIXTURES / "kaikki.jsonl",),
        conceptnet=FIXTURES / "conceptnet.tsv",
        numberbatch=FIXTURES / "numberbatch.txt",
        cmudict=FIXTURES / "cmudict.dict",
        notices_dir=FIXTURES / "notices",
    )


def _recovery_anchors(partial: Path) -> dict[str, str]:
    semantic = (
        partial / "semantic"
        if (partial / "semantic").is_dir()
        else partial / "semantic.partial"
    )
    return {
        "expected_lexicon_sha256": file_sha256(partial / "lexicon.sqlite3"),
        "expected_global_index_sha256": file_sha256(
            semantic / "indexes" / "global.usearch"
        ),
        "expected_global_vectors_sha256": file_sha256(
            semantic / "vectors" / "global.f16"
        ),
    }


def _interrupt_after_global(
    monkeypatch: pytest.MonkeyPatch, semantic: Path
) -> None:
    real_verify = numberbatch.verify_saved_index

    def interrupt(path: Path, expected_count: int, dimensions: int) -> None:
        if Path(path).name == "global.usearch":
            raise RuntimeError("simulated post-global interruption")
        real_verify(Path(path), expected_count, dimensions)

    monkeypatch.setattr(numberbatch, "verify_saved_index", interrupt)
    with pytest.raises(RuntimeError, match="post-global interruption"):
        numberbatch.build_numberbatch(
            FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
        )
    monkeypatch.setattr(numberbatch, "verify_saved_index", real_verify)


def test_post_global_partial_is_validated_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic = tmp_path / "semantic"
    _interrupt_after_global(monkeypatch, semantic)

    partial = tmp_path / "semantic.partial"
    assert (partial / "mapping.sqlite3").is_file()
    assert (partial / "vectors" / "global.f16").stat().st_size == 10 * 4 * 2
    assert index_count(
        partial / "indexes" / "global.usearch", dimensions=4
    ) == 10

    counts = numberbatch.resume_numberbatch_partial(
        FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
    )
    assert counts["terms"] == 10
    assert counts["languages"] == {
        "ar": 1,
        "de": 1,
        "en": 4,
        "es": 1,
        "fr": 1,
        "hi": 1,
        "ja": 1,
    }
    assert not partial.exists()
    assert index_count(semantic / "indexes" / "global.usearch", dimensions=4) == 10


def test_normal_build_fails_closed_on_existing_semantic_partial(
    tmp_path: Path,
) -> None:
    semantic = tmp_path / "semantic"
    partial = tmp_path / "semantic.partial"
    global_path = partial / "indexes" / "global.usearch"
    global_path.parent.mkdir(parents=True)
    sentinel = partial / "resume-me.sentinel"
    sentinel.write_bytes(b"irreplaceable-resume-state")
    global_path.write_bytes(b"existing-global-index")
    before = {sentinel: sentinel.read_bytes(), global_path: global_path.read_bytes()}

    with pytest.raises(FileExistsError, match="validated recovery command"):
        numberbatch.build_numberbatch(
            FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
        )

    assert {path: path.read_bytes() for path in before} == before
    assert set(path.relative_to(partial) for path in partial.rglob("*") if path.is_file()) == {
        Path("resume-me.sentinel"),
        Path("indexes/global.usearch"),
    }


def test_contiguous_offsets_are_required_before_any_shard_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic = tmp_path / "semantic"
    _interrupt_after_global(monkeypatch, semantic)
    mapping = tmp_path / "semantic.partial" / "mapping.sqlite3"
    with sqlite3.connect(mapping) as connection:
        connection.execute(
            "UPDATE semantic_terms SET vector_offset=99 WHERE semantic_id=9"
        )
        connection.commit()

    def must_not_build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("shard construction ran before contiguous-ID validation")

    monkeypatch.setattr(numberbatch, "_build_language_index", must_not_build)
    with pytest.raises(RuntimeError, match="not exact and contiguous"):
        numberbatch.resume_numberbatch_partial(
            FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
        )
    assert (tmp_path / "semantic.partial").is_dir()


def test_language_shard_validation_accepts_exact_keys_in_hnsw_slot_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic = tmp_path / "semantic"
    _interrupt_after_global(monkeypatch, semantic)
    partial = tmp_path / "semantic.partial"
    artifact = partial / "indexes" / "unordered.usearch"
    artifact.write_bytes(b"synthetic index bytes")

    class ValidatedView:
        def reset(self) -> None:
            return None

    class UnorderedKeysIndex:
        def __init__(self, **_kwargs: object) -> None:
            self.keys = np.asarray([0, 2, 9, 1], dtype=np.uint64)

        def __len__(self) -> int:
            return 4

        def view(self, _path: str) -> None:
            return None

        def reset(self) -> None:
            return None

    def validated_view(*_args: object, **_kwargs: object) -> ValidatedView:
        return ValidatedView()

    monkeypatch.setattr(numberbatch, "open_index_view", validated_view)
    monkeypatch.setattr(numberbatch, "Index", UnorderedKeysIndex)
    with closing(sqlite3.connect(partial / "mapping.sqlite3")) as connection:
        assert numberbatch._language_index_artifact_is_valid(
            connection,
            artifact,
            language="en",
            expected_count=4,
            dimensions=4,
        )


def test_interrupted_language_shards_resume_only_after_count_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic = tmp_path / "semantic"
    _interrupt_after_global(monkeypatch, semantic)
    real_build = numberbatch._build_language_index
    first_attempt: list[str] = []

    def interrupt_second(*args: object, **kwargs: object) -> None:
        language = str(kwargs["language"])
        if len(first_attempt) == 1:
            raise RuntimeError("simulated shard interruption")
        real_build(*args, **kwargs)
        first_attempt.append(language)

    monkeypatch.setattr(numberbatch, "_build_language_index", interrupt_second)
    with pytest.raises(RuntimeError, match="shard interruption"):
        numberbatch.resume_numberbatch_partial(
            FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
        )
    assert first_attempt == ["ar"]
    partial = tmp_path / "semantic.partial"
    with closing(sqlite3.connect(partial / "mapping.sqlite3")) as connection:
        assert connection.execute(
            "SELECT language,term_count FROM semantic_languages"
        ).fetchall() == [("ar", 1)]

    resumed_builds: list[str] = []

    def record_rebuild(*args: object, **kwargs: object) -> None:
        resumed_builds.append(str(kwargs["language"]))
        real_build(*args, **kwargs)

    monkeypatch.setattr(numberbatch, "_build_language_index", record_rebuild)
    numberbatch.resume_numberbatch_partial(
        FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
    )
    assert "ar" not in resumed_builds
    assert resumed_builds == ["de", "en", "es", "fr", "hi", "ja"]


def test_valid_saved_partial_shard_is_promoted_without_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic = tmp_path / "semantic"
    _interrupt_after_global(monkeypatch, semantic)
    real_replace = numberbatch.os.replace

    def interrupt_after_save(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.endswith(".usearch.partial")
            and destination_path.parent.name == "languages"
        ):
            raise RuntimeError("simulated interruption after shard save")
        real_replace(source, destination)

    monkeypatch.setattr(numberbatch.os, "replace", interrupt_after_save)
    with pytest.raises(RuntimeError, match="after shard save"):
        numberbatch.resume_numberbatch_partial(
            FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
        )

    partial = tmp_path / "semantic.partial"
    saved = partial / "indexes" / "languages" / "ar.usearch.partial"
    assert saved.is_file()
    with closing(sqlite3.connect(partial / "mapping.sqlite3")) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM semantic_languages"
        ).fetchone()[0] == 0

    monkeypatch.setattr(numberbatch.os, "replace", real_replace)
    real_build = numberbatch._build_language_index
    rebuilt: list[str] = []

    def record_build(*args: object, **kwargs: object) -> None:
        rebuilt.append(str(kwargs["language"]))
        real_build(*args, **kwargs)

    monkeypatch.setattr(numberbatch, "_build_language_index", record_build)
    numberbatch.resume_numberbatch_partial(
        FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
    )
    assert "ar" not in rebuilt
    assert rebuilt == ["de", "en", "es", "fr", "hi", "ja"]


def test_promoted_shard_without_database_row_is_reconciled_without_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic = tmp_path / "semantic"
    _interrupt_after_global(monkeypatch, semantic)
    real_record = numberbatch._record_language_shard
    interrupted = False

    def interrupt_after_replace(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption after shard replace")
        real_record(*args, **kwargs)

    monkeypatch.setattr(numberbatch, "_record_language_shard", interrupt_after_replace)
    with pytest.raises(RuntimeError, match="after shard replace"):
        numberbatch.resume_numberbatch_partial(
            FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
        )

    partial = tmp_path / "semantic.partial"
    promoted = partial / "indexes" / "languages" / "ar.usearch"
    assert promoted.is_file()
    assert not promoted.with_name("ar.usearch.partial").exists()
    with closing(sqlite3.connect(partial / "mapping.sqlite3")) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM semantic_languages"
        ).fetchone()[0] == 0

    monkeypatch.setattr(numberbatch, "_record_language_shard", real_record)
    real_build = numberbatch._build_language_index
    rebuilt: list[str] = []

    def record_build(*args: object, **kwargs: object) -> None:
        rebuilt.append(str(kwargs["language"]))
        real_build(*args, **kwargs)

    monkeypatch.setattr(numberbatch, "_build_language_index", record_build)
    numberbatch.resume_numberbatch_partial(
        FIXTURES / "numberbatch.txt", semantic, "fixture-v1", batch_size=2
    )
    assert "ar" not in rebuilt
    assert rebuilt == ["de", "en", "es", "fr", "hi", "ja"]


@pytest.mark.parametrize("semantic_name", ["semantic.partial", "semantic"])
def test_normal_full_build_preserves_existing_recovery_state(
    tmp_path: Path, semantic_name: str
) -> None:
    output = tmp_path / "fixture-v1"
    partial = output.with_name(output.name + ".partial")
    partial.mkdir()
    lexical = partial / "lexicon.sqlite3"
    lexical.write_bytes(b"preserved lexical bytes")
    (partial / semantic_name).mkdir()
    before = lexical.read_bytes()

    with pytest.raises(FileExistsError, match="validated recovery command"):
        build_full_corpus(
            _inputs(), output, tmp_path / "state", dataset_version="fixture-v1"
        )

    assert lexical.read_bytes() == before
    assert (partial / semantic_name).is_dir()


def test_dedicated_recovery_preserves_lexical_and_reuses_completed_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs()
    output = tmp_path / "fixture-v1"
    state = tmp_path / "state"
    real_verify = numberbatch.verify_saved_index

    def interrupt(path: Path, expected_count: int, dimensions: int) -> None:
        if Path(path).name == "global.usearch":
            raise RuntimeError("simulated full-build post-global interruption")
        real_verify(Path(path), expected_count, dimensions)

    monkeypatch.setattr(numberbatch, "verify_saved_index", interrupt)
    with pytest.raises(RuntimeError, match="full-build post-global interruption"):
        build_full_corpus(inputs, output, state, dataset_version="fixture-v1")
    monkeypatch.setattr(numberbatch, "verify_saved_index", real_verify)

    partial = output.with_name(output.name + ".partial")
    lexical_before = (partial / "lexicon.sqlite3").read_bytes()
    anchors = _recovery_anchors(partial)
    real_floors = orchestrator.evaluate_corpus_floors
    monkeypatch.setattr(
        orchestrator,
        "evaluate_corpus_floors",
        lambda _counts: ({}, ["simulated post-semantic floor failure"]),
    )
    with pytest.raises(RuntimeError, match="post-semantic floor failure"):
        recover_full_corpus_from_semantic_partial(
            inputs,
            output,
            state,
            dataset_version="fixture-v1",
            original_build_commit="a" * 40,
            recovery_commit="b" * 40,
            **anchors,
        )
    assert (partial / "semantic").is_dir()
    assert not (partial / "semantic.partial").exists()
    assert (partial / "lexicon.sqlite3").read_bytes() == lexical_before

    monkeypatch.setattr(orchestrator, "evaluate_corpus_floors", real_floors)

    def must_not_rebuild(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("completed semantic directory was not reused")

    monkeypatch.setattr(numberbatch, "_build_language_index", must_not_rebuild)
    manifest = recover_full_corpus_from_semantic_partial(
        inputs,
        output,
        state,
        dataset_version="fixture-v1",
        enforce_corpus_floors=False,
        original_build_commit="a" * 40,
        recovery_commit="b" * 40,
        **anchors,
    )
    assert output.is_dir()
    assert (output / "lexicon.sqlite3").read_bytes() == lexical_before
    assert manifest["recovery_provenance"]["original_build_commit"] == "a" * 40
    assert manifest["recovery_provenance"]["recovery_commit"] == "b" * 40


def test_recovery_one_writer_lock_rejects_concurrent_invocation(tmp_path: Path) -> None:
    inputs = _inputs()
    output = tmp_path / "fixture-v1"
    state = tmp_path / "state"
    lock_path = state / "fixture-v1" / ".recovery.lock"

    with InstallationLock(lock_path), pytest.raises(LockBusyError):
        recover_full_corpus_from_semantic_partial(
            inputs,
            output,
            state,
            dataset_version="fixture-v1",
            original_build_commit="a" * 40,
            recovery_commit="b" * 40,
            expected_lexicon_sha256="0" * 64,
            expected_global_index_sha256="0" * 64,
            expected_global_vectors_sha256="0" * 64,
        )
    assert not output.exists()
