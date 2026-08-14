from __future__ import annotations

import hashlib
import json
import sqlite3
import tomllib
from pathlib import Path

import pytest

from lexicon_mcp.data.locking import InstallationLock, LockBusyError
from lexicon_mcp.pipeline import notice_amendment
from lexicon_mcp.pipeline.notice_amendment import amend_promoted_dataset_notices

VERSION = "data-v1.0.0"
ORIGINAL_COMMIT = "a" * 40
RECOVERY_COMMIT = "b" * 40
AMENDMENT_COMMIT = "c" * 40
NOTICE_PATHS = (
    "DATA_LICENSES.md",
    "licenses/CC-BY-4.0.txt",
    "licenses/CC-BY-SA-4.0.txt",
    "licenses/CMUDICT.txt",
    "licenses/GFDL-1.3.txt",
    "licenses/OEWN-LICENSE.md",
    "licenses/PRINCETON-WORDNET.txt",
)


@pytest.fixture(autouse=True)
def _stub_usearch_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def count(_path: Path, *, expected_count: int | None = None, **_kwargs: object) -> int:
        assert expected_count is not None
        return expected_count

    monkeypatch.setattr(notice_amendment, "index_count", count)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_notices(root: Path, label: str) -> None:
    for relative in NOTICE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}: {relative}\n", encoding="utf-8")


def _write_manifest_converged(root: Path, manifest: dict[str, object]) -> None:
    path = root / "build-manifest.json"
    for _attempt in range(4):
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        size = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        if manifest["installed_size"] == size:
            return
        manifest["installed_size"] = size
    raise AssertionError("fixture installed size failed to converge")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repository = tmp_path / "repository"
    _write_notices(repository, "new")
    dataset = tmp_path / VERSION
    _write_notices(dataset / "notices", "old")
    artifacts = {
        "lexicon.sqlite3": dataset / "lexicon.sqlite3",
        "global.usearch": dataset / "semantic" / "indexes" / "global.usearch",
        "global.f16": dataset / "semantic" / "vectors" / "global.f16",
    }
    for name, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "global.f16":
            path.write_bytes(bytes(8))
        else:
            path.write_bytes((name + " immutable fixture bytes").encode())
    mapping = dataset / "semantic" / "mapping.sqlite3"
    with sqlite3.connect(mapping) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE lexical_terms(
                term_id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                normalized_term TEXT NOT NULL,
                language TEXT NOT NULL
            );
            CREATE TABLE semantic_terms(
                semantic_id INTEGER PRIMARY KEY,
                term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
                vector_offset INTEGER NOT NULL
            );
            CREATE TABLE semantic_languages(
                language TEXT PRIMARY KEY,
                index_file TEXT NOT NULL,
                term_count INTEGER NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            (
                ("dataset_version", VERSION),
                ("schema_version", "2"),
                ("dimensions", "2"),
                ("term_count", "2"),
                ("vector_dtype", "float16"),
                ("vector_file", "vectors/global.f16"),
                ("index_metric", "cos"),
                ("index_dtype", "i8"),
                ("global_index", "indexes/global.usearch"),
                ("language_count", "2"),
                ("language_index_dir", "indexes/languages"),
                ("connectivity", "16"),
                ("expansion_add", "256"),
                ("expansion_search", "512"),
            ),
        )
        connection.executemany(
            "INSERT INTO lexical_terms VALUES (?,?,?,?)",
            ((1, "cat", "cat", "en"), (2, "Katze", "katze", "de")),
        )
        connection.executemany(
            "INSERT INTO semantic_terms VALUES (?,?,?)",
            ((1, 1, 0), (2, 2, 1)),
        )
        connection.executemany(
            "INSERT INTO semantic_languages VALUES (?,?,?)",
            (
                ("de", "indexes/languages/de.usearch", 1),
                ("en", "indexes/languages/en.usearch", 1),
            ),
        )
    for language in ("de", "en"):
        shard = dataset / "semantic" / "indexes" / "languages" / f"{language}.usearch"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(f"{language} shard fixture".encode())
    anchors = {name: _sha(path) for name, path in artifacts.items()}
    identity = "d" * 64
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": VERSION,
        "profile": "full",
        "pipeline_identity": identity,
        "installed_size": 0,
        "installed_size_limit": 30 * 1024**3,
        "recovery_provenance": {
            "mode": "validated-post-global-semantic-resume",
            "original_build_commit": ORIGINAL_COMMIT,
            "original_pipeline_identity": "e" * 64,
            "recovery_commit": RECOVERY_COMMIT,
            "recovery_pipeline_identity": identity,
            "artifact_sha256": anchors,
            "usearch_version": "2.26.0",
            "uv_lock_sha256": "f" * 64,
        },
    }
    _write_manifest_converged(dataset, manifest)
    return dataset, repository, anchors


def _arguments(anchors: dict[str, str]) -> dict[str, str]:
    return {
        "dataset_version": VERSION,
        "amendment_commit": AMENDMENT_COMMIT,
        "reason": "Correct the audited attribution text",
        "expected_original_build_commit": ORIGINAL_COMMIT,
        "expected_recovery_commit": RECOVERY_COMMIT,
        "expected_lexicon_sha256": anchors["lexicon.sqlite3"],
        "expected_global_index_sha256": anchors["global.usearch"],
        "expected_global_vectors_sha256": anchors["global.f16"],
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_no_transaction_state(dataset: Path) -> None:
    for suffix in ("transaction", "transaction.partial"):
        assert not (dataset.parent / f".{VERSION}.notice-amendment.{suffix}").exists()


def test_notice_amendment_replaces_only_notices_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    before_artifacts = {
        relative: (dataset / relative).read_bytes()
        for relative in (
            "lexicon.sqlite3",
            "semantic/indexes/global.usearch",
            "semantic/vectors/global.f16",
            "semantic/mapping.sqlite3",
            "semantic/indexes/languages/de.usearch",
            "semantic/indexes/languages/en.usearch",
        )
    }
    old_notice_hashes = {
        relative: _sha(dataset / "notices" / relative) for relative in NOTICE_PATHS
    }
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)

    result = amend_promoted_dataset_notices(
        dataset, repository, **_arguments(anchors)
    )

    for relative in NOTICE_PATHS:
        assert (dataset / "notices" / relative).read_bytes() == (
            repository / relative
        ).read_bytes()
    for relative, content in before_artifacts.items():
        assert (dataset / relative).read_bytes() == content
    amendment = result["notice_amendments"][-1]
    assert amendment["amendment_commit"] == AMENDMENT_COMMIT
    assert amendment["reason"] == "Correct the audited attribution text"
    assert amendment["old_notice_sha256"] == old_notice_hashes
    assert amendment["new_notice_sha256"] == {
        relative: _sha(repository / relative) for relative in NOTICE_PATHS
    }
    assert len(amendment["recovery_provenance_sha256"]) == 64
    assert amendment["semantic_artifact_sha256"] == {
        "semantic/indexes/languages/de.usearch": _sha(
            dataset / "semantic" / "indexes" / "languages" / "de.usearch"
        ),
        "semantic/indexes/languages/en.usearch": _sha(
            dataset / "semantic" / "indexes" / "languages" / "en.usearch"
        ),
        "semantic/mapping.sqlite3": _sha(dataset / "semantic" / "mapping.sqlite3"),
    }
    measured = sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file())
    assert result["installed_size"] == measured
    assert json.loads((dataset / "build-manifest.json").read_text()) == result
    _assert_no_transaction_state(dataset)


def test_notice_amendment_refuses_anchor_mismatch_without_mutating_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    before = _tree_bytes(dataset)
    arguments = _arguments(anchors)
    arguments["expected_global_index_sha256"] = "0" * 64
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)

    with pytest.raises(RuntimeError, match=r"anchor mismatch for global\.usearch"):
        amend_promoted_dataset_notices(dataset, repository, **arguments)

    assert _tree_bytes(dataset) == before


def test_notice_amendment_rejects_wrong_recovery_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)
    arguments = _arguments(anchors)
    arguments["expected_recovery_commit"] = "9" * 40

    with pytest.raises(RuntimeError, match="recovery commit mismatch"):
        amend_promoted_dataset_notices(dataset, repository, **arguments)


def test_notice_amendment_has_one_cross_process_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)
    lock = dataset.parent / f".{VERSION}.notice-amendment.lock"

    with InstallationLock(lock), pytest.raises(
        LockBusyError, match="another lexicon-data operation"
    ):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))


def test_notice_amendment_rejects_transients_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    transient = dataset / "semantic" / "mapping.sqlite3-wal"
    transient.write_bytes(b"uncommitted")
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)

    with pytest.raises(RuntimeError, match="transient paths"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))

    assert transient.read_bytes() == b"uncommitted"
    assert (dataset / "notices" / "DATA_LICENSES.md").read_text().startswith("old:")


def test_notice_amendment_rolls_back_notice_swap_and_manifest_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    before = _tree_bytes(dataset)
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)

    def fail_manifest(_path: Path, _manifest: dict[str, object]) -> None:
        raise RuntimeError("simulated manifest failure")

    monkeypatch.setattr(notice_amendment, "_write_manifest_converged", fail_manifest)
    with pytest.raises(RuntimeError, match="simulated manifest failure"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))

    assert _tree_bytes(dataset) == before
    _assert_no_transaction_state(dataset)


@pytest.mark.parametrize(
    "checkpoint",
    (
        "during_transaction_prepare",
        "after_transaction_prepare",
        "after_old_notices_move",
        "after_new_notices_move",
        "after_manifest_update",
        "after_commit_marker",
    ),
)
def test_notice_amendment_recovers_every_durable_crash_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)

    class SimulatedProcessLoss(BaseException):
        pass

    fired = False

    def crash_at(name: str) -> None:
        nonlocal fired
        if name == checkpoint and not fired:
            fired = True
            raise SimulatedProcessLoss(name)

    monkeypatch.setattr(notice_amendment, "_crash_checkpoint", crash_at)
    with pytest.raises(SimulatedProcessLoss, match=checkpoint):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))

    transaction = dataset.parent / f".{VERSION}.notice-amendment.transaction"
    transaction_partial = transaction.with_name(transaction.name + ".partial")
    assert transaction.is_dir() or transaction_partial.is_dir()
    monkeypatch.setattr(notice_amendment, "_crash_checkpoint", lambda _name: None)
    result = amend_promoted_dataset_notices(
        dataset, repository, **_arguments(anchors)
    )

    assert [
        item["amendment_commit"] for item in result["notice_amendments"]
    ] == [AMENDMENT_COMMIT]
    for relative in NOTICE_PATHS:
        assert (dataset / "notices" / relative).read_bytes() == (
            repository / relative
        ).read_bytes()
    assert result["installed_size"] == sum(
        path.stat().st_size for path in dataset.rglob("*") if path.is_file()
    )
    _assert_no_transaction_state(dataset)


def test_notice_recovery_precedes_later_git_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    before = _tree_bytes(dataset)
    permit_git = True

    def validate_git(*_args: object) -> None:
        if not permit_git:
            raise RuntimeError("simulated dirty or moved repository")

    class SimulatedProcessLoss(BaseException):
        pass

    def crash_after_old_notices_move(name: str) -> None:
        if name == "after_old_notices_move":
            raise SimulatedProcessLoss(name)

    monkeypatch.setattr(notice_amendment, "_validated_clean_head", validate_git)
    monkeypatch.setattr(
        notice_amendment,
        "_crash_checkpoint",
        crash_after_old_notices_move,
    )
    with pytest.raises(SimulatedProcessLoss, match="after_old_notices_move"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))

    assert not (dataset / "notices").exists()
    permit_git = False
    monkeypatch.setattr(notice_amendment, "_crash_checkpoint", lambda _name: None)
    with pytest.raises(RuntimeError, match="dirty or moved repository"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))

    assert _tree_bytes(dataset) == before
    _assert_no_transaction_state(dataset)


def test_notice_amendment_rejects_missing_or_unrecorded_language_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)
    (dataset / "semantic" / "indexes" / "languages" / "de.usearch").unlink()

    with pytest.raises(RuntimeError, match="language artifact set mismatch"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))

    extra = dataset / "semantic" / "indexes" / "languages" / "extra.usearch"
    (dataset / "semantic" / "indexes" / "languages" / "de.usearch").write_bytes(
        b"de shard fixture"
    )
    extra.write_bytes(b"unrecorded")
    with pytest.raises(RuntimeError, match="language artifact set mismatch"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))


def test_notice_amendment_rejects_semantic_mapping_partition_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, repository, anchors = _fixture(tmp_path)
    monkeypatch.setattr(notice_amendment, "_validated_clean_head", lambda *_args: None)
    mapping = dataset / "semantic" / "mapping.sqlite3"
    with sqlite3.connect(mapping) as connection:
        connection.execute(
            "UPDATE semantic_languages SET term_count=2 WHERE language='de'"
        )

    with pytest.raises(RuntimeError, match="do not partition all terms"):
        amend_promoted_dataset_notices(dataset, repository, **_arguments(anchors))


def test_semantic_verification_opens_global_and_every_declared_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _repository, _anchors = _fixture(tmp_path)
    calls: list[tuple[str, int]] = []

    def observe(
        path: Path, *, expected_count: int | None = None, **_kwargs: object
    ) -> int:
        assert expected_count is not None
        calls.append((path.relative_to(dataset / "semantic").as_posix(), expected_count))
        return expected_count

    monkeypatch.setattr(notice_amendment, "index_count", observe)

    hashes = notice_amendment._semantic_artifact_hashes(
        dataset, dataset_version=VERSION
    )

    assert calls == [
        ("indexes/global.usearch", 2),
        ("indexes/languages/de.usearch", 1),
        ("indexes/languages/en.usearch", 1),
    ]
    assert set(hashes) == {
        "semantic/mapping.sqlite3",
        "semantic/indexes/languages/de.usearch",
        "semantic/indexes/languages/en.usearch",
    }


def test_clean_head_validation_rejects_dirty_or_different_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        notice_amendment,
        "_git_output",
        lambda _repository, *arguments: (
            b" M DATA_LICENSES.md\n" if arguments[0] == "status" else b"c" * 40 + b"\n"
        ),
    )
    with pytest.raises(RuntimeError, match="clean Git working tree"):
        notice_amendment._validated_clean_head(tmp_path, AMENDMENT_COMMIT)

    monkeypatch.setattr(
        notice_amendment,
        "_git_output",
        lambda _repository, *arguments: (
            b"" if arguments[0] == "status" else b"d" * 40 + b"\n"
        ),
    )
    with pytest.raises(RuntimeError, match="must equal current clean Git HEAD"):
        notice_amendment._validated_clean_head(tmp_path, AMENDMENT_COMMIT)


def test_project_declares_all_software_wheel_license_files() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license-files"] == ["LICENSE", "NOTICE", "DATA_LICENSES.md"]
