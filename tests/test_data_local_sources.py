"""Local mirrors, offline installation, and the `fetch` release mirror."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_data_lifecycle import (
    NoNetworkTransport,
    lifecycle,
    manifest_bytes,
    sqlite_payload,
    transport_for,
    write_local_zstd_package,
)

from lexicon_mcp.data.lifecycle import DatasetLifecycle, DownloadError, LifecycleError
from lexicon_mcp.data.manifest import parse_manifest
from lexicon_mcp.data_cli import _manifest_source, _resolved_source, build_parser, main

VERSION = "data-v1.0.0"


def offline_manager(root: Path) -> tuple[DatasetLifecycle, NoNetworkTransport]:
    transport = NoNetworkTransport()
    manager = DatasetLifecycle(
        root,
        transport=transport,
        sleep=lambda _seconds: None,
        safety_margin=0,
    )
    return manager, transport


def rewrite_parts(manifest_path: Path, mutate: object) -> None:
    """Apply *mutate* to every part object in a local package manifest."""

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    for component in value["components"]:
        for part in component["parts"]:
            mutate(part)  # type: ignore[operator]
    manifest_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_install_from_directory_resolves_its_manifest(tmp_path: Path) -> None:
    final = sqlite_payload(tmp_path, "dir-source.db", ["directory source"])
    manifest_path, _parts = write_local_zstd_package(tmp_path / "package", final)
    manager, transport = offline_manager(tmp_path / "data")

    result = manager.install(manifest_path.parent, profile="full", version=VERSION)

    assert result["action"] == "installed"
    assert (manager.root / "versions" / VERSION / "lexicon.sqlite3").read_bytes() == final
    assert manager.verify()["ok"] is True
    assert transport.calls == []


def test_local_install_prefers_disk_over_declared_part_urls(tmp_path: Path) -> None:
    """A local source never reaches the network, even for URL-bearing parts."""

    final = sqlite_payload(tmp_path, "url-parts.db", ["url bearing parts"])
    manifest_path, _parts = write_local_zstd_package(tmp_path / "package", final)
    rewrite_parts(
        manifest_path,
        lambda part: part.update({"url": f"https://fixtures.invalid/{part['name']}"}),
    )
    manager, transport = offline_manager(tmp_path / "data")

    result = manager.install(manifest_path.parent, profile="full", version=VERSION)

    assert result["action"] == "installed"
    assert manager.verify()["ok"] is True
    assert transport.calls == []


def test_local_install_fails_loudly_when_a_part_has_no_asset_name(tmp_path: Path) -> None:
    final = sqlite_payload(tmp_path, "nameless.db", ["nameless part"])
    manifest_path, _parts = write_local_zstd_package(tmp_path / "package", final)

    def drop_name(part: dict[str, object]) -> None:
        part["url"] = f"https://fixtures.invalid/{part['name']}"
        del part["name"]

    rewrite_parts(manifest_path, drop_name)
    manager, transport = offline_manager(tmp_path / "data")

    with pytest.raises(DownloadError, match="cannot be installed from a local source"):
        manager.install(manifest_path.parent, profile="full", version=VERSION)

    assert not (manager.root / "versions" / VERSION).exists()
    assert transport.calls == []


def test_fetch_mirror_round_trips_into_an_offline_install(tmp_path: Path) -> None:
    payload = sqlite_payload(tmp_path, "mirror.db", ["mirror round trip"])
    components = {"lexical": ("lexicon.sqlite3", payload)}
    manifest = parse_manifest(manifest_bytes(VERSION, components))
    mirror = tmp_path / "mirror"
    source_manager = lifecycle(tmp_path / "source-data", transport_for(components))

    report = source_manager.fetch(manifest, profile="full", version=VERSION, dest=mirror)

    assert report["action"] == "fetched"
    assert report["assets_fetched"] == 1
    assert report["assets_skipped"] == 0
    assert report["manifest_sha256"] == manifest.sha256
    # The mirror reproduces the published release layout exactly.
    assert sorted(item.name for item in mirror.iterdir()) == [
        "lexical.part0000",
        "manifest.json",
    ]
    assert (mirror / "manifest.json").read_bytes() == manifest.raw

    target, transport = offline_manager(tmp_path / "data")
    result = target.install(mirror, profile="full", version=VERSION)

    assert result["action"] == "installed"
    assert (target.root / "versions" / VERSION / "lexicon.sqlite3").read_bytes() == payload
    assert target.verify()["ok"] is True
    assert transport.calls == []


def test_fetch_resumes_a_truncated_asset(tmp_path: Path) -> None:
    payload = sqlite_payload(tmp_path, "resume.db", ["resume mirror"])
    components = {"lexical": ("lexicon.sqlite3", payload)}
    manifest = parse_manifest(manifest_bytes(VERSION, components))
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "lexical.part0000.partial").write_bytes(payload[:11])
    transport = transport_for(components)
    manager = lifecycle(tmp_path / "source-data", transport)

    report = manager.fetch(manifest, profile="full", version=VERSION, dest=mirror)

    assert report["assets_fetched"] == 1
    assert (mirror / "lexical.part0000").read_bytes() == payload
    assert not (mirror / "lexical.part0000.partial").exists()
    # The resume requested only the missing byte range.
    assert transport.calls == [
        ("https://fixtures.invalid/assets/lexical.part0000", 11),
    ]


def test_fetch_skips_assets_that_are_already_valid(tmp_path: Path) -> None:
    payload = sqlite_payload(tmp_path, "idempotent.db", ["idempotent mirror"])
    components = {"lexical": ("lexicon.sqlite3", payload)}
    manifest = parse_manifest(manifest_bytes(VERSION, components))
    mirror = tmp_path / "mirror"
    transport = transport_for(components)
    manager = lifecycle(tmp_path / "source-data", transport)

    manager.fetch(manifest, profile="full", version=VERSION, dest=mirror)
    calls_after_first = len(transport.calls)
    report = manager.fetch(manifest, profile="full", version=VERSION, dest=mirror)

    assert report["assets_fetched"] == 0
    assert report["assets_skipped"] == 1
    assert report["bytes_fetched"] == 0
    assert len(transport.calls) == calls_after_first


def test_fetch_never_publishes_a_corrupt_asset_or_manifest(tmp_path: Path) -> None:
    final = sqlite_payload(tmp_path, "corrupt-source.db", ["corrupt source"])
    manifest_path, parts = write_local_zstd_package(tmp_path / "package", final)
    manifest = parse_manifest(manifest_path.read_bytes())
    corrupt = manifest_path.parent / str(manifest.components[0].parts[0].name)
    corrupt.write_bytes(b"\x00" * len(parts[0]))
    mirror = tmp_path / "mirror"
    manager, transport = offline_manager(tmp_path / "data")

    with pytest.raises(DownloadError, match="local release asset SHA-256 mismatch"):
        manager.fetch(manifest_path, profile="full", version=VERSION, dest=mirror)

    assert not (mirror / corrupt.name).exists()
    # No manifest means an interrupted mirror fails loudly instead of looking whole.
    assert not (mirror / "manifest.json").exists()
    assert transport.calls == []


def test_fetch_refuses_to_mirror_a_directory_onto_itself(tmp_path: Path) -> None:
    final = sqlite_payload(tmp_path, "self-mirror.db", ["self mirror"])
    manifest_path, _parts = write_local_zstd_package(tmp_path / "package", final)
    manager, _transport = offline_manager(tmp_path / "data")

    with pytest.raises(LifecycleError, match="same directory"):
        manager.fetch(
            manifest_path,
            profile="full",
            version=VERSION,
            dest=manifest_path.parent,
        )


def test_fetch_does_not_touch_the_dataset_root(tmp_path: Path) -> None:
    payload = sqlite_payload(tmp_path, "untouched.db", ["untouched root"])
    components = {"lexical": ("lexicon.sqlite3", payload)}
    manifest = parse_manifest(manifest_bytes(VERSION, components))
    data_root = tmp_path / "data"
    manager = lifecycle(data_root, transport_for(components))

    manager.fetch(manifest, profile="full", version=VERSION, dest=tmp_path / "mirror")

    assert not (data_root / "versions").exists()
    assert not (data_root / "current.json").exists()
    assert not (data_root / ".install.lock").exists()


def test_cli_rejects_combining_the_new_and_deprecated_source_flags() -> None:
    args = build_parser().parse_args(
        [
            "install",
            "--profile",
            "full",
            "--version",
            VERSION,
            "--from",
            "/mirror",
            "--manifest-url",
            "https://example.invalid/manifest.json",
        ]
    )

    with pytest.raises(LifecycleError, match="cannot be combined"):
        _resolved_source(args)


def test_cli_keeps_the_deprecated_flag_working_with_a_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser().parse_args(
        [
            "install",
            "--profile",
            "full",
            "--version",
            VERSION,
            "--manifest-url",
            "https://example.invalid/manifest.json",
        ]
    )

    assert _resolved_source(args) == "https://example.invalid/manifest.json"
    assert "deprecated" in capsys.readouterr().err


def test_cli_fetch_accepts_a_destination_and_a_local_source() -> None:
    args = build_parser().parse_args(
        [
            "fetch",
            "--profile",
            "english",
            "--version",
            "data-en-v1.0.0",
            "--dest",
            "/media/usb/lexicon",
            "--from",
            "/mnt/release",
        ]
    )

    assert args.command == "fetch"
    assert args.dest == Path("/media/usb/lexicon")
    assert _resolved_source(args) == "/mnt/release"


def test_cli_round_trips_fetch_into_an_offline_install(tmp_path: Path) -> None:
    """The command surface itself mirrors, installs, and verifies offline."""

    final = sqlite_payload(tmp_path, "cli-round-trip.db", ["cli round trip"])
    manifest_path, _parts = write_local_zstd_package(tmp_path / "package", final)
    data_dir = tmp_path / "data"
    mirror = tmp_path / "mirror"
    common = ["--data-dir", str(data_dir)]

    assert (
        main(
            [
                *common,
                "fetch",
                "--profile",
                "full",
                "--version",
                VERSION,
                "--dest",
                str(mirror),
                "--from",
                str(manifest_path.parent),
            ]
        )
        == 0
    )
    assert (mirror / "manifest.json").is_file()

    assert (
        main(
            [
                *common,
                "install",
                "--profile",
                "full",
                "--version",
                VERSION,
                "--from",
                str(mirror),
            ]
        )
        == 0
    )
    assert main([*common, "verify"]) == 0
    assert (data_dir / "versions" / VERSION / "lexicon.sqlite3").read_bytes() == final


def test_local_sources_are_never_reinterpreted_as_templates() -> None:
    source = _manifest_source(
        r"E:\releases\data-v1.0.0", version=VERSION, profile="full"
    )

    assert source == r"E:\releases\data-v1.0.0"
