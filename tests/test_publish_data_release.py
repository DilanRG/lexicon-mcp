from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.publish_data_release import (
    ReleaseError,
    publish_release,
    stage_release,
    validate_bundle,
)


def _bundle(tmp_path: Path) -> tuple[Path, bytes]:
    package = tmp_path / "release"
    package.mkdir()
    payload = b"verified release payload"
    part_name = "data-v1.0.0-fixture.zst.part0000"
    (package / part_name).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_version": "data-v1.0.0",
        "profile": "full",
        "transformation_commit": "1" * 40,
        "release": {
            "repository": "DilanRG/lexicon-mcp",
            "tag": "data-v1.0.0",
            "immutable": True,
        },
        "created_at": "2026-08-14T00:00:00Z",
        "sources": [
            {
                "id": "fixture",
                "name": "Fixture",
                "url": "https://fixtures.invalid/source",
                "revision": "fixture-1",
                "retrieved_at": "2026-08-14T00:00:00Z",
                "sha256": "0" * 64,
                "size": 0,
                "license": "CC0-1.0",
            }
        ],
        "components": [
            {
                "id": "artifact-fixture",
                "artifact_type": "metadata",
                "path": "fixture.txt",
                "compression": "zstd",
                "compressed_size": len(payload),
                "compressed_sha256": digest,
                "final_size": 1,
                "final_sha256": "2" * 64,
                "parts": [
                    {
                        "name": part_name,
                        "size": len(payload),
                        "sha256": digest,
                        "offset": 0,
                    }
                ],
                "sources": ["fixture"],
            }
        ],
    }
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (package / "manifest.json").write_bytes(raw)
    return package, raw


class FakeGitHub:
    def __init__(self, package: Path) -> None:
        self.package = package
        self.release: dict[str, Any] | None = None
        self.assets: dict[str, dict[str, Any]] = {}
        self.calls: list[list[str]] = []

    def __call__(self, arguments: list[str]) -> str:
        self.calls.append(arguments)
        if arguments[:1] == ["api"]:
            endpoint = arguments[-1]
            if endpoint.endswith("/immutable-releases"):
                return json.dumps({"enabled": True, "enforced_by_owner": False})
            if "/releases/tags/" in endpoint:
                if self.release is None:
                    raise ReleaseError("gh api failed: HTTP 404: release not found")
                return json.dumps(self.release)
            if "/releases/17/assets" in endpoint:
                page = int(endpoint.rsplit("page=", 1)[-1])
                return json.dumps(list(self.assets.values()) if page == 1 else [])
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        if arguments[:2] == ["release", "create"]:
            self.release = {"id": 17, "draft": True, "immutable": False}
            return ""
        if arguments[:2] == ["release", "upload"]:
            for value in arguments[5:]:
                path = Path(value)
                payload = path.read_bytes()
                self.assets[path.name] = {
                    "name": path.name,
                    "size": len(payload),
                    "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                }
            return ""
        if arguments[:2] == ["release", "edit"]:
            assert self.release is not None
            self.release.update({"draft": False, "immutable": True})
            return ""
        raise AssertionError(f"unexpected gh arguments: {arguments}")


def _acceptance(path: Path, bundle_sha: str) -> Path:
    report = {
        "dataset_version": "data-v1.0.0",
        "manifest_sha256": bundle_sha,
        "transformation_commit": "1" * 40,
        "clean_install_ok": True,
        "verify_ok": True,
        "full_corpus_ok": True,
        "ann_ok": True,
        "performance_ok": True,
        "live_stack_ok": True,
        "offline_runtime_ok": True,
        "restart_cycles": 10,
        "active_models": [],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_validate_bundle_hashes_every_asset_and_rejects_corruption(tmp_path: Path) -> None:
    package, _raw = _bundle(tmp_path)
    bundle = validate_bundle(package)

    assert [asset.name for asset in bundle.assets] == [
        "manifest.json",
        "data-v1.0.0-fixture.zst.part0000",
    ]
    (package / "data-v1.0.0-fixture.zst.part0000").write_bytes(b"corrupt")
    with pytest.raises(ReleaseError, match="size mismatch"):
        validate_bundle(package)


def test_stage_is_idempotent_and_uploads_only_missing_verified_assets(
    tmp_path: Path,
) -> None:
    package, _raw = _bundle(tmp_path)
    bundle = validate_bundle(package)
    github = FakeGitHub(package)

    first = stage_release(
        bundle,
        repository="DilanRG/lexicon-mcp",
        tag="data-v1.0.0",
        runner=github,
    )
    second = stage_release(
        bundle,
        repository="DilanRG/lexicon-mcp",
        tag="data-v1.0.0",
        runner=github,
    )

    assert first["uploaded"] == 2
    assert second["uploaded"] == 0
    assert len([call for call in github.calls if call[:2] == ["release", "upload"]]) == 1


def test_publish_requires_exact_acceptance_and_confirms_immutability(tmp_path: Path) -> None:
    package, _raw = _bundle(tmp_path)
    bundle = validate_bundle(package)
    github = FakeGitHub(package)
    stage_release(
        bundle,
        repository="DilanRG/lexicon-mcp",
        tag="data-v1.0.0",
        runner=github,
    )
    bad = _acceptance(tmp_path / "bad.json", "f" * 64)
    with pytest.raises(ReleaseError, match="manifest_sha256"):
        publish_release(
            bundle,
            repository="DilanRG/lexicon-mcp",
            tag="data-v1.0.0",
            acceptance_report=bad,
            runner=github,
        )

    good = _acceptance(tmp_path / "good.json", bundle.manifest.sha256)
    result = publish_release(
        bundle,
        repository="DilanRG/lexicon-mcp",
        tag="data-v1.0.0",
        acceptance_report=good,
        runner=github,
    )
    assert result["immutable"] is True
    assert github.release == {"id": 17, "draft": False, "immutable": True}
