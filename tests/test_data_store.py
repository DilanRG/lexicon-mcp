"""Content-addressed component storage."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lexicon_mcp.data.store import ComponentStore, StoreError


def digest_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stage(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    path = tmp_path / "staging" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, digest_of(payload)


def test_adopting_a_component_stores_it_under_its_digest(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")
    staged, digest = stage(tmp_path, "pack.sqlite3", b"lexical payload")

    stored = store.adopt(staged, digest)

    assert stored == store.objects_root / digest[:2] / digest
    assert stored.read_bytes() == b"lexical payload"
    assert not staged.exists()
    assert store.contains(digest)
    assert store.verify(digest) is True


def test_a_component_that_does_not_hash_to_its_digest_is_refused(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")
    staged, _digest = stage(tmp_path, "corrupt.sqlite3", b"tampered")

    with pytest.raises(StoreError, match="hashes to"):
        store.adopt(staged, digest_of(b"expected"))

    assert list(store.iter_digests()) == []


def test_adopting_an_already_stored_component_is_idempotent(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")
    first, digest = stage(tmp_path, "one.sqlite3", b"shared payload")
    store.adopt(first, digest)
    second, _ = stage(tmp_path, "two.sqlite3", b"shared payload")

    stored = store.adopt(second, digest)

    assert stored.read_bytes() == b"shared payload"
    assert not second.exists()
    assert list(store.iter_digests()) == [digest]


def test_two_installs_sharing_a_component_hold_it_once(tmp_path: Path) -> None:
    """The property that makes add-language cheap."""

    store = ComponentStore(tmp_path / "components")
    core, core_digest = stage(tmp_path, "core.sqlite3", b"core")
    english, english_digest = stage(tmp_path, "en.sqlite3", b"english")
    store.adopt(core, core_digest)
    store.adopt(english, english_digest)

    # A later install adds French and re-offers the core it already has.
    french, french_digest = stage(tmp_path, "fr.sqlite3", b"french")
    core_again, _ = stage(tmp_path, "core-again.sqlite3", b"core")
    store.adopt(french, french_digest)
    store.adopt(core_again, core_digest)

    assert sorted(store.iter_digests()) == sorted(
        {core_digest, english_digest, french_digest}
    )
    assert store.total_bytes() == len(b"core") + len(b"english") + len(b"french")


def test_verify_detects_a_component_corrupted_in_place(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")
    staged, digest = stage(tmp_path, "pack.sqlite3", b"good payload")
    stored = store.adopt(staged, digest)

    stored.write_bytes(b"rotted payload")

    assert store.contains(digest) is True
    assert store.verify(digest) is False


def test_pruning_keeps_everything_a_retained_activation_references(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")
    kept, kept_digest = stage(tmp_path, "kept.sqlite3", b"kept")
    dropped, dropped_digest = stage(tmp_path, "dropped.sqlite3", b"dropped")
    store.adopt(kept, kept_digest)
    store.adopt(dropped, dropped_digest)

    removed = store.prune({kept_digest})

    assert removed == (dropped_digest,)
    assert list(store.iter_digests()) == [kept_digest]
    assert store.verify(kept_digest) is True


def test_missing_component_fails_loudly_rather_than_silently(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")

    with pytest.raises(StoreError, match="not in the store"):
        store.open_path(digest_of(b"absent"))


@pytest.mark.parametrize(
    "digest",
    ["", "not-a-digest", "AB" * 32, "ab" * 31, "../escape"],
)
def test_unsafe_digests_are_rejected(tmp_path: Path, digest: str) -> None:
    store = ComponentStore(tmp_path / "components")

    with pytest.raises(StoreError, match="SHA-256 digest"):
        store.path_for(digest)


def test_iteration_ignores_files_that_are_not_stored_components(tmp_path: Path) -> None:
    store = ComponentStore(tmp_path / "components")
    staged, digest = stage(tmp_path, "pack.sqlite3", b"payload")
    store.adopt(staged, digest)
    (store.objects_root / digest[:2] / "stray.tmp").write_bytes(b"junk")
    misfiled = store.objects_root / "zz"
    misfiled.mkdir(parents=True)
    (misfiled / digest).write_bytes(b"misfiled")

    assert list(store.iter_digests()) == [digest]
