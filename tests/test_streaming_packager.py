from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest
import zstandard

from lexicon_mcp.pipeline.manifest import package_dataset

REQUIRED_LICENSES = (
    "OEWN-LICENSE.md",
    "PRINCETON-WORDNET.txt",
    "CC-BY-4.0.txt",
    "CC-BY-SA-4.0.txt",
    "GFDL-1.3.txt",
    "CMUDICT.txt",
)


def _dataset(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "dataset"
    licenses = root / "notices" / "licenses"
    licenses.mkdir(parents=True)
    (root / "notices" / "DATA_LICENSES.md").write_text(
        "# Data licenses\n", encoding="utf-8"
    )
    for name in REQUIRED_LICENSES:
        (licenses / name).write_text(f"fixture notice: {name}\n", encoding="utf-8")
    (root / "sources.lock.json").write_text(
        json.dumps({"schema_version": 1, "sources": []}) + "\n",
        encoding="utf-8",
    )
    (root / "build-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "data-v1.0.0",
                "profile": "full",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = random.Random(20260814).randbytes(8192)
    (root / "payload.bin").write_bytes(payload)
    return root, payload


def _package(root: Path, package: Path, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "dataset_version": "data-v1.0.0",
        "repository": "DilanRG/lexicon-mcp",
        "tag": "data-v1.0.0",
        "transformation_commit": "a" * 40,
        "max_part_size": 257,
        "created_at": "2026-08-14T00:00:00Z",
    }
    arguments.update(overrides)
    return package_dataset(root, package, **arguments)


def test_streams_zstd_directly_into_bounded_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, payload = _dataset(tmp_path)
    package = tmp_path / "package"
    opened_in_package: list[str] = []
    original_open = Path.open

    def recording_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.parent == package:
            opened_in_package.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    manifest = _package(root, package)

    component = next(item for item in manifest["components"] if item["path"] == "payload.bin")
    assert len(component["parts"]) > 1
    compressed = bytearray()
    expected_offset = 0
    for part in component["parts"]:
        part_bytes = (package / part["name"]).read_bytes()
        assert part["offset"] == expected_offset
        assert part["size"] == len(part_bytes) <= 257
        assert part["sha256"] == hashlib.sha256(part_bytes).hexdigest()
        expected_offset += len(part_bytes)
        compressed.extend(part_bytes)
    assert expected_offset == component["compressed_size"]
    assert hashlib.sha256(compressed).hexdigest() == component["compressed_sha256"]
    assert (
        zstandard.ZstdDecompressor().decompress(
            compressed, max_output_size=component["final_size"]
        )
        == payload
    )
    assert not any(name.endswith(".zst.partial") for name in opened_in_package)
    assert not any(path.name.endswith(".partial") for path in package.iterdir())


def test_midstream_compression_failure_removes_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _payload = _dataset(tmp_path)
    package = tmp_path / "package"

    class BrokenCompressor:
        def copy_stream(self, source: Any, destination: Any, *, read_size: int) -> None:
            del source, read_size
            destination.write(b"x" * 600)
            raise RuntimeError("fixture compressor failure")

    monkeypatch.setattr(
        zstandard,
        "ZstdCompressor",
        lambda **_kwargs: BrokenCompressor(),
    )
    with pytest.raises(RuntimeError, match="fixture compressor failure"):
        _package(root, package)
    assert list(package.iterdir()) == []


def test_refuses_stale_part_without_overwriting_it(tmp_path: Path) -> None:
    root, _payload = _dataset(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    relative = "payload.bin"
    token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    collision = package / f"data-v1.0.0-{token}-payload.bin.zst.part0000"
    collision.write_bytes(b"do not replace")

    with pytest.raises(FileExistsError, match="stale or conflicting"):
        _package(root, package)
    assert collision.read_bytes() == b"do not replace"
    assert list(package.iterdir()) == [collision]
