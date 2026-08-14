from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zstandard
from usearch.index import Index

from lexicon_mcp.data.integrity import default_semantic_count, verify_component
from lexicon_mcp.data.lifecycle import (
    DatasetLifecycle,
    DownloadError,
    LifecycleError,
    SpaceError,
    active_version,
)
from lexicon_mcp.data.locking import InstallationLock, LockBusyError
from lexicon_mcp.data.manifest import Component, ManifestError, parse_manifest
from lexicon_mcp.data.transport import Response


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def semantic_index_integrity(*, dimensions: int, count: int) -> dict[str, Any]:
    return {
        "semantic_count": count,
        "semantic_dimensions": dimensions,
        "semantic_metric": "cos",
        "semantic_dtype": "i8",
        "semantic_connectivity": 16,
        "semantic_expansion_add": 256,
        "semantic_expansion_search": 512,
    }


def forbid_usearch_path_introspection(monkeypatch: pytest.MonkeyPatch) -> None:
    original_metadata = Index.metadata

    def metadata(source: object, *args: object, **kwargs: object) -> object:
        if isinstance(source, (str, Path)):
            raise AssertionError("USearch path metadata must not be used")
        return original_metadata(source, *args, **kwargs)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("USearch path metadata/restore must not be used")

    monkeypatch.setattr(Index, "metadata", staticmethod(metadata))
    monkeypatch.setattr(Index, "restore", staticmethod(forbidden))


def sqlite_payload(tmp_path: Path, name: str, values: list[str]) -> bytes:
    path = tmp_path / name
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE entries (value TEXT NOT NULL)")
        connection.executemany("INSERT INTO entries VALUES (?)", [(value,) for value in values])
        connection.commit()
    finally:
        connection.close()
    payload = path.read_bytes()
    path.unlink()
    return payload


def manifest_bytes(
    version: str,
    components: dict[str, tuple[str, bytes]],
    *,
    override_sha: dict[str, str] | None = None,
    profile: str = "full",
    languages: list[str] | None = None,
) -> bytes:
    override_sha = override_sha or {}
    items = []
    for component_id, (path, payload) in components.items():
        expected = override_sha.get(component_id, digest(payload))
        items.append(
            {
                "id": component_id,
                "artifact_type": "lexical_sqlite",
                "path": path,
                "compression": "none",
                "compressed_size": len(payload),
                "compressed_sha256": expected,
                "final_size": len(payload),
                "final_sha256": expected,
                "parts": [
                    {
                        "name": f"{component_id}.part0000",
                        "size": len(payload),
                        "sha256": expected,
                        "offset": 0,
                    }
                ],
                "sources": ["fixture"],
                "integrity": {"sqlite": True},
            }
        )
    value = {
        "schema_version": 1,
        "dataset_version": version,
        "profile": profile,
        "release": {
            "repository": "DilanRG/lexicon-mcp",
            "tag": version,
            "immutable": True,
            "base_url": "https://fixtures.invalid/assets/",
        },
        "created_at": "2026-08-14T00:00:00Z",
        "transformation_commit": "1" * 40,
        "sources": [
            {
                "id": "fixture",
                "name": "Test fixture",
                "url": "https://fixtures.invalid/source",
                "revision": "test-1",
                "retrieved_at": "2026-08-14T00:00:00Z",
                "sha256": "0" * 64,
                "size": 0,
                "row_count": None,
                "row_digest": None,
                "license": "CC0-1.0",
            }
        ],
        "components": items,
    }
    if languages is not None:
        value["languages"] = languages
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def write_local_zstd_package(
    package_dir: Path,
    final: bytes,
    *,
    version: str = "data-v1.0.0",
) -> tuple[Path, tuple[bytes, bytes]]:
    compressed = zstandard.ZstdCompressor(level=3, write_checksum=True).compress(final)
    split_at = max(1, len(compressed) // 2)
    part_payloads = (compressed[:split_at], compressed[split_at:])
    part_names = (f"{version}-lexical.part0000", f"{version}-lexical.part0001")
    value = json.loads(manifest_bytes(version, {"lexical": ("lexicon.sqlite3", final)}))
    component = value["components"][0]
    component.update(
        {
            "compression": "zstd",
            "compressed_size": len(compressed),
            "compressed_sha256": digest(compressed),
            "parts": [
                {
                    "name": name,
                    "size": len(payload),
                    "sha256": digest(payload),
                    "offset": sum(len(previous) for previous in part_payloads[:index]),
                }
                for index, (name, payload) in enumerate(zip(part_names, part_payloads, strict=True))
            ],
        }
    )
    package_dir.mkdir(parents=True)
    for name, payload in zip(part_names, part_payloads, strict=True):
        (package_dir / name).write_bytes(payload)
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, part_payloads


class InterruptingBody(io.BytesIO):
    def __init__(self, payload: bytes, cutoff: int) -> None:
        super().__init__(payload)
        self.cutoff = cutoff
        self.interrupted = False

    def read(self, size: int = -1) -> bytes:
        if self.interrupted:
            raise OSError("fixture connection interrupted")
        position = self.tell()
        if position >= self.cutoff:
            self.interrupted = True
            raise OSError("fixture connection interrupted")
        maximum = self.cutoff - position
        return super().read(maximum if size < 0 else min(size, maximum))


class FakeTransport:
    def __init__(self, assets: dict[str, bytes]) -> None:
        self.assets = assets
        self.calls: list[tuple[str, int]] = []
        self.interrupt_once: dict[str, int] = {}

    def open(self, url: str, offset: int = 0) -> Response:
        self.calls.append((url, offset))
        payload = self.assets[url]
        cutoff = self.interrupt_once.pop(url, None)
        if cutoff is not None and offset == 0:
            body: io.BytesIO = InterruptingBody(payload, cutoff)
        else:
            body = io.BytesIO(payload[offset:])
        headers = {"content-range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"}
        return Response(206 if offset else 200, headers, body)


class NoNetworkTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def open(self, url: str, offset: int = 0) -> Response:
        self.calls.append((url, offset))
        raise AssertionError(f"unexpected network request: {url}")


def transport_for(components: dict[str, tuple[str, bytes]]) -> FakeTransport:
    return FakeTransport(
        {
            f"https://fixtures.invalid/assets/{component_id}.part0000": payload
            for component_id, (_path, payload) in components.items()
        }
    )


def lifecycle(root: Path, transport: FakeTransport, **kwargs: Any) -> DatasetLifecycle:
    return DatasetLifecycle(
        root,
        transport=transport,
        sleep=lambda _seconds: None,
        safety_margin=0,
        **kwargs,
    )


def test_install_resume_verify_and_atomic_activation(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    transport = transport_for(components)
    url = "https://fixtures.invalid/assets/lexical.part0000"
    transport.interrupt_once[url] = 256
    manager = lifecycle(tmp_path / "data", transport)
    manifest = parse_manifest(manifest_bytes("data-v1.0.0", components))

    result = manager.install(manifest, profile="full", version="data-v1.0.0")

    assert result["action"] == "installed"
    assert transport.calls == [(url, 0), (url, 256)]
    assert manager.verify()["ok"] is True
    active = active_version(manager.root)
    assert active is not None
    assert active[0] == "data-v1.0.0"
    pointer = json.loads((manager.root / "current.json").read_text())
    assert pointer["path"] == "versions/data-v1.0.0"
    assert pointer["manifest_sha256"] == manifest.sha256
    assert not list((manager.root / "versions").glob(".*.staging"))


def test_english_profile_install_activation_status_and_repair(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "en.db", ["one"]))}
    manager = lifecycle(tmp_path / "data", transport_for(components))
    manifest = parse_manifest(
        manifest_bytes(
            "data-en-v1.0.0",
            components,
            profile="english",
            languages=["en"],
        )
    )

    result = manager.install(manifest, profile="english", version="data-en-v1.0.0")

    assert result["profile"] == "english"
    pointer = json.loads((manager.root / "current.json").read_text())
    assert pointer["profile"] == "english"
    status = manager.status()
    assert status["installed_versions"][0]["profile"] == "english"
    assert manager.repair()["profile"] == "english"


def test_manifest_english_profile_requires_exact_language_scope(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "en.db", ["one"]))}
    parsed = parse_manifest(
        manifest_bytes(
            "data-en-v1.0.0",
            components,
            profile="english",
            languages=["en"],
        )
    )
    assert parsed.profile == "english"
    assert parsed.languages == ("en",)

    with pytest.raises(ManifestError, match=r"requires languages=\['en'\]"):
        parse_manifest(
            manifest_bytes(
                "data-en-v1.0.0",
                components,
                profile="english",
                languages=["en", "de"],
            )
        )


def test_verify_active_checks_pointer_manifest_pin(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    manager = lifecycle(tmp_path / "data", transport_for(components))
    manager.install(
        parse_manifest(manifest_bytes("data-v1.0.0", components)),
        profile="full",
        version="data-v1.0.0",
    )
    pointer_path = manager.root / "current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = "f" * 64
    pointer_path.write_text(json.dumps(pointer))

    report = manager.verify()

    assert report["ok"] is False
    assert report["problems"][0] == "current.json: manifest SHA-256 mismatch"


def test_install_pipeline_style_zstd_component(tmp_path: Path) -> None:
    final = sqlite_payload(tmp_path, "compressed.db", ["compressed"])
    compressed = zstandard.ZstdCompressor(level=3, write_checksum=True).compress(final)
    value = json.loads(
        manifest_bytes("data-v1.0.0", {"lexical": ("lexicon.sqlite3", final)})
    )
    component = value["components"][0]
    component.update(
        {
            "compression": "zstd",
            "compressed_size": len(compressed),
            "compressed_sha256": digest(compressed),
        }
    )
    component["parts"][0].update(
        {"size": len(compressed), "sha256": digest(compressed), "offset": 0}
    )
    transport = FakeTransport(
        {"https://fixtures.invalid/assets/lexical.part0000": compressed}
    )
    manager = lifecycle(tmp_path / "data", transport)

    manager.install(parse_manifest(json.dumps(value)), profile="full", version="data-v1.0.0")

    installed = manager.root / "versions" / "data-v1.0.0" / "lexicon.sqlite3"
    assert installed.read_bytes() == final
    assert manager.verify()["ok"] is True


def test_install_local_package_uses_sibling_parts_without_network(tmp_path: Path) -> None:
    final = sqlite_payload(tmp_path, "local-package.db", ["local package"])
    manifest_path, _parts = write_local_zstd_package(tmp_path / "package", final)
    transport = NoNetworkTransport()
    manager = DatasetLifecycle(
        tmp_path / "data",
        transport=transport,
        sleep=lambda _seconds: None,
        safety_margin=0,
    )

    result = manager.install(
        manifest_path,
        profile="full",
        version="data-v1.0.0",
    )

    installed = manager.root / "versions" / "data-v1.0.0" / "lexicon.sqlite3"
    assert result["action"] == "installed"
    assert installed.read_bytes() == final
    assert manager.verify()["ok"] is True
    assert transport.calls == []


def test_install_local_package_resumes_verified_partial_without_network(
    tmp_path: Path,
) -> None:
    final = sqlite_payload(tmp_path, "local-resume.db", ["resume"])
    manifest_path, parts = write_local_zstd_package(tmp_path / "package", final)
    manifest = parse_manifest(manifest_path.read_bytes())
    transport = NoNetworkTransport()
    manager = DatasetLifecycle(
        tmp_path / "data",
        transport=transport,
        sleep=lambda _seconds: None,
        safety_margin=0,
    )
    component = manifest.components[0]
    part = component.parts[0]
    cache = manager.root / ".downloads" / manifest.dataset_version / component.id
    cache.mkdir(parents=True)
    partial = manager._part_path(cache, 0, part, complete=False)
    partial.write_bytes(parts[0][:17])

    manager.install(manifest_path, profile="full", version=manifest.dataset_version)

    complete = manager._part_path(cache, 0, part, complete=True)
    assert complete.read_bytes() == parts[0]
    assert not partial.exists()
    assert manager.verify()["ok"] is True
    assert transport.calls == []


def test_install_local_package_rejects_corrupt_partial_before_activation(
    tmp_path: Path,
) -> None:
    final = sqlite_payload(tmp_path, "local-corrupt.db", ["corrupt partial"])
    manifest_path, parts = write_local_zstd_package(tmp_path / "package", final)
    manifest = parse_manifest(manifest_path.read_bytes())
    transport = NoNetworkTransport()
    manager = DatasetLifecycle(
        tmp_path / "data",
        transport=transport,
        sleep=lambda _seconds: None,
        safety_margin=0,
    )
    component = manifest.components[0]
    part = component.parts[0]
    cache = manager.root / ".downloads" / manifest.dataset_version / component.id
    cache.mkdir(parents=True)
    partial = manager._part_path(cache, 0, part, complete=False)
    partial.write_bytes(b"\x00" * min(17, len(parts[0])))

    with pytest.raises(DownloadError, match="local release asset SHA-256 mismatch"):
        manager.install(manifest_path, profile="full", version=manifest.dataset_version)

    assert not partial.exists()
    assert not (manager.root / "current.json").exists()
    assert not (manager.root / "versions" / manifest.dataset_version).exists()
    assert transport.calls == []

    manager.install(manifest_path, profile="full", version=manifest.dataset_version)
    assert manager.verify()["ok"] is True
    assert transport.calls == []


def test_failed_update_never_replaces_current(tmp_path: Path) -> None:
    first = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "v1.db", ["v1"]))}
    first_transport = transport_for(first)
    manager = lifecycle(tmp_path / "data", first_transport, retries=2)
    manager.install(
        parse_manifest(manifest_bytes("data-v1.0.0", first)),
        profile="full",
        version="data-v1.0.0",
    )

    second = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "v2.db", ["v2"]))}
    bad_transport = transport_for(second)
    bad_transport.assets["https://fixtures.invalid/assets/lexical.part0000"] = b"x" * len(
        second["lexical"][1]
    )
    manager.transport = bad_transport
    manifest = parse_manifest(manifest_bytes("data-v2.0.0", second))

    with pytest.raises(DownloadError, match="SHA-256"):
        manager.install(manifest, profile="full", version="data-v2.0.0")

    assert active_version(manager.root)[0] == "data-v1.0.0"  # type: ignore[index]
    assert not (manager.root / "versions" / "data-v2.0.0").exists()
    assert manager.verify("data-v1.0.0")["ok"] is True


def test_repair_redownloads_only_damaged_component(tmp_path: Path) -> None:
    components = {
        "lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "lex.db", ["lexical"])),
        "relations": ("relations.sqlite3", sqlite_payload(tmp_path, "rel.db", ["relation"])),
    }
    transport = transport_for(components)
    manager = lifecycle(tmp_path / "data", transport)
    manifest = parse_manifest(manifest_bytes("data-v1.0.0", components))
    manager.install(manifest, profile="full", version="data-v1.0.0")
    version = manager.root / "versions" / "data-v1.0.0"
    healthy_before = (version / "lexicon.sqlite3").read_bytes()
    (version / "relations.sqlite3").write_bytes(b"damaged")
    shutil_cache = manager.root / ".downloads" / "data-v1.0.0" / "relations"
    for cached in shutil_cache.glob("*.part"):
        cached.unlink()
    transport.calls.clear()

    result = manager.repair()

    assert result["repaired"] == ["relations"]
    assert (version / "lexicon.sqlite3").read_bytes() == healthy_before
    assert [url for url, _offset in transport.calls] == [
        "https://fixtures.invalid/assets/relations.part0000"
    ]
    assert manager.verify()["ok"] is True


def test_repair_can_restore_manifest_from_pinned_copy(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    transport = transport_for(components)
    manager = lifecycle(tmp_path / "data", transport)
    manifest = parse_manifest(manifest_bytes("data-v1.0.0", components))
    manager.install(manifest, profile="full", version="data-v1.0.0")
    (manager.root / "versions" / "data-v1.0.0" / "manifest.json").write_text("broken")

    result = manager.repair(manifest_source=manifest)

    assert result["repaired"] == []
    assert result["manifest_restored"] is True
    assert manager.verify()["ok"] is True


def test_rollback_requires_and_verifies_retained_version(tmp_path: Path) -> None:
    root = tmp_path / "data"
    first = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "v1.db", ["v1"]))}
    manager = lifecycle(root, transport_for(first))
    manager.install(
        parse_manifest(manifest_bytes("data-v1.0.0", first)),
        profile="full",
        version="data-v1.0.0",
    )
    with pytest.raises(LifecycleError, match="no previous"):
        manager.rollback()

    second = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "v2.db", ["v2"]))}
    manager.transport = transport_for(second)
    manager.install(
        parse_manifest(manifest_bytes("data-v2.0.0", second)),
        profile="full",
        version="data-v2.0.0",
    )
    result = manager.rollback()

    assert result["version"] == "data-v1.0.0"
    assert active_version(root)[0] == "data-v1.0.0"  # type: ignore[index]
    assert json.loads((root / "current.json").read_text())["previous_version"] == "data-v2.0.0"


def test_one_writer_lock_rejects_concurrent_mutation(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    manager = lifecycle(tmp_path / "data", transport_for(components))
    manifest = parse_manifest(manifest_bytes("data-v1.0.0", components))

    with InstallationLock(manager.lock_path), pytest.raises(LockBusyError):
        manager.install(manifest, profile="full", version="data-v1.0.0")

    assert not (manager.root / "current.json").exists()


def test_free_space_preflight_happens_before_asset_fetch(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    transport = transport_for(components)
    manager = lifecycle(
        tmp_path / "data",
        transport,
        disk_usage=lambda _path: SimpleNamespace(free=1),
    )

    with pytest.raises(SpaceError, match="free bytes"):
        manager.install(
            parse_manifest(manifest_bytes("data-v1.0.0", components)),
            profile="full",
            version="data-v1.0.0",
        )

    assert transport.calls == []
    assert not (manager.root / "current.json").exists()


@pytest.mark.parametrize(
    "bad_path",
    ["../outside.sqlite3", "/absolute.sqlite3", "C:/drive.sqlite3", r"nested\evil.db"],
)
def test_manifest_rejects_unsafe_component_paths(tmp_path: Path, bad_path: str) -> None:
    components = {"lexical": (bad_path, sqlite_payload(tmp_path, "one.db", ["one"]))}
    with pytest.raises(ManifestError):
        parse_manifest(manifest_bytes("data-v1.0.0", components))


def test_manifest_requires_immutable_matching_release_tag(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    value = json.loads(manifest_bytes("data-v1.0.0", components))
    value["release"]["immutable"] = False
    with pytest.raises(ManifestError, match="immutable"):
        parse_manifest(json.dumps(value))
    value["release"]["immutable"] = True
    value["release"]["tag"] = "data-other"
    with pytest.raises(ManifestError, match="equal dataset_version"):
        parse_manifest(json.dumps(value))


def test_manifest_requires_complete_reproducibility_metadata(tmp_path: Path) -> None:
    components = {"lexical": ("lexicon.sqlite3", sqlite_payload(tmp_path, "one.db", ["one"]))}
    value = json.loads(manifest_bytes("data-v1.0.0", components))
    del value["transformation_commit"]
    with pytest.raises(ManifestError, match="transformation_commit"):
        parse_manifest(json.dumps(value))

    value["transformation_commit"] = "1" * 40
    del value["sources"][0]["size"]
    with pytest.raises(ManifestError, match=r"sources\[0\]\.size"):
        parse_manifest(json.dumps(value))

    value["sources"][0]["size"] = 0
    value["sources"][0]["row_count"] = 42
    value["sources"][0]["row_digest"] = None
    with pytest.raises(ManifestError, match="must be set together"):
        parse_manifest(json.dumps(value))


def test_usearch_226_semantic_count_uses_large_file_safe_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "global.usearch"
    index = Index(ndim=2, metric="cos", dtype="i8")
    index.add(np.asarray([1, 2], dtype=np.uint64), np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    index.save(str(path))
    index.reset()
    forbid_usearch_path_introspection(monkeypatch)
    component = Component(
        id="semantic-global",
        artifact_type="semantic_index",
        path=PurePosixPath("global.usearch"),
        compression="none",
        compressed_size=path.stat().st_size,
        compressed_sha256=digest(path.read_bytes()),
        final_size=path.stat().st_size,
        final_sha256=digest(path.read_bytes()),
        parts=(),
        sources=("numberbatch",),
        integrity=semantic_index_integrity(dimensions=2, count=2),
    )

    assert default_semantic_count(path, component) == 2
    path.unlink()
    assert not path.exists()


def test_manifest_requires_complete_strict_semantic_index_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_usearch_path_introspection(monkeypatch)
    value = json.loads(
        manifest_bytes("data-v1.0.0", {"semantic": ("semantic/global.usearch", b"index")})
    )
    component = value["components"][0]
    component["artifact_type"] = "semantic_index"
    component["integrity"] = semantic_index_integrity(dimensions=300, count=1)

    parsed = parse_manifest(json.dumps(value))

    assert parsed.components[0].integrity == semantic_index_integrity(
        dimensions=300, count=1
    )
    missing_count = json.loads(json.dumps(value))
    del missing_count["components"][0]["integrity"]["semantic_count"]
    with pytest.raises(ManifestError, match="semantic_count is required"):
        parse_manifest(json.dumps(missing_count))
    for missing in (
        "semantic_dimensions",
        "semantic_metric",
        "semantic_dtype",
        "semantic_connectivity",
        "semantic_expansion_add",
        "semantic_expansion_search",
    ):
        invalid = json.loads(json.dumps(value))
        del invalid["components"][0]["integrity"][missing]
        with pytest.raises(ManifestError, match="incomplete semantic index schema"):
            parse_manifest(json.dumps(invalid))

    invalid_values: tuple[tuple[str, object, str], ...] = (
        ("semantic_count", 0, "must be a positive integer"),
        ("semantic_dimensions", 0, "must be a positive integer"),
        ("semantic_metric", "l2sq", "must be cos"),
        ("semantic_dtype", "f32", "must be i8"),
        ("semantic_connectivity", 15, "must be 16"),
        ("semantic_expansion_add", 128, "must be 256"),
        ("semantic_expansion_search", 511, "must be at least 512"),
    )
    for field, bad_value, message in invalid_values:
        invalid = json.loads(json.dumps(value))
        invalid["components"][0]["integrity"][field] = bad_value
        with pytest.raises(ManifestError, match=message):
            parse_manifest(json.dumps(invalid))


def test_install_and_verify_semantic_index_avoid_path_introspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "global.usearch"
    index = Index(
        ndim=2,
        metric="cos",
        dtype="i8",
        connectivity=16,
        expansion_add=256,
        expansion_search=512,
    )
    index.add(
        np.asarray([1, 2], dtype=np.uint64),
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    index.save(str(index_path))
    index.reset()
    index_payload = index_path.read_bytes()

    mapping_path = tmp_path / "mapping.sqlite3"
    with sqlite3.connect(mapping_path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '2');
            CREATE TABLE semantic_terms (semantic_id INTEGER PRIMARY KEY);
            INSERT INTO semantic_terms VALUES (1);
            INSERT INTO semantic_terms VALUES (2);
            """
        )
    mapping_payload = mapping_path.read_bytes()
    components = {
        "semantic-mapping": ("semantic/mapping.sqlite3", mapping_payload),
        "semantic-global": ("semantic/indexes/global.usearch", index_payload),
    }
    value = json.loads(manifest_bytes("data-v1.0.0", components))
    mapping_component, index_component = value["components"]
    mapping_component["artifact_type"] = "semantic_mapping"
    mapping_component["integrity"] = {
        "sqlite": True,
        "dataset_schema_version": 2,
        "semantic_count": 2,
        "semantic_mapping_table": "semantic_terms",
    }
    index_component["artifact_type"] = "semantic_index"
    index_component["integrity"] = {
        **semantic_index_integrity(dimensions=2, count=2),
        "semantic_mapping": "semantic/mapping.sqlite3",
        "semantic_mapping_table": "semantic_terms",
    }
    manifest = parse_manifest(json.dumps(value))
    forbid_usearch_path_introspection(monkeypatch)
    manager = lifecycle(tmp_path / "data", transport_for(components))

    result = manager.install(manifest, profile="full", version="data-v1.0.0")
    report = manager.verify()

    assert result["action"] == "installed"
    assert report["ok"] is True


def test_sqlite_dataset_schema_version_integrity_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '2');
            """
        )
    payload = path.read_bytes()
    component = Component(
        id="lexical",
        artifact_type="lexical_sqlite",
        path=PurePosixPath("lexicon.sqlite3"),
        compression="none",
        compressed_size=len(payload),
        compressed_sha256=digest(payload),
        final_size=len(payload),
        final_sha256=digest(payload),
        parts=(),
        sources=("fixture",),
        integrity={"sqlite": True, "dataset_schema_version": 2},
    )
    assert verify_component(tmp_path, component) == []

    incompatible = Component(
        id=component.id,
        artifact_type=component.artifact_type,
        path=component.path,
        compression=component.compression,
        compressed_size=component.compressed_size,
        compressed_sha256=component.compressed_sha256,
        final_size=component.final_size,
        final_sha256=component.final_sha256,
        parts=component.parts,
        sources=component.sources,
        integrity={"sqlite": True, "dataset_schema_version": 3},
    )
    assert verify_component(tmp_path, incompatible) == [
        "lexical: dataset schema version mismatch (expected 3, got '2')"
    ]


def test_dataset_schema_version_manifest_field_must_be_positive() -> None:
    value = json.loads(manifest_bytes("data-v1.0.0", {"lexical": ("lexicon.db", b"db")}))
    value["components"][0]["integrity"]["dataset_schema_version"] = 2
    parsed = parse_manifest(json.dumps(value))
    assert parsed.components[0].integrity["dataset_schema_version"] == 2

    value["components"][0]["integrity"]["dataset_schema_version"] = 0
    with pytest.raises(ManifestError, match="must be a positive integer"):
        parse_manifest(json.dumps(value))
