"""Validate, stage, and explicitly publish an immutable data release.

The default operation is local validation only.  ``--stage`` may create a draft
release and upload missing assets.  ``--publish`` is deliberately separate and
requires a complete acceptance report for the exact manifest being published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lexicon_mcp.data.manifest import DatasetManifest, ManifestError, parse_manifest

DEFAULT_REPOSITORY = "DilanRG/lexicon-mcp"
MAX_RELEASE_ASSETS = 1000
UPLOAD_BATCH_SIZE = 16
API_VERSION = "2026-03-10"


class ReleaseError(RuntimeError):
    """A data release is incomplete, inconsistent, or unsafe to publish."""


class CommandRunner(Protocol):
    def __call__(self, arguments: list[str]) -> str: ...


@dataclass(frozen=True, slots=True)
class LocalAsset:
    name: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ReleaseBundle:
    directory: Path
    manifest: DatasetManifest
    assets: tuple[LocalAsset, ...]


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(
    directory: Path,
    *,
    repository: str = DEFAULT_REPOSITORY,
    tag: str = "data-v1.0.0",
) -> ReleaseBundle:
    """Validate every byte that will become a GitHub release asset."""

    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ReleaseError(f"cannot read release manifest {manifest_path}: {exc}") from exc
    try:
        manifest = parse_manifest(raw)
    except ManifestError as exc:
        raise ReleaseError(f"invalid release manifest: {exc}") from exc
    if manifest.release.repository != repository:
        raise ReleaseError(
            f"manifest repository {manifest.release.repository!r} does not match {repository!r}"
        )
    if manifest.release.tag != tag or manifest.dataset_version != tag:
        raise ReleaseError(
            f"manifest version/tag must both equal {tag!r}, got "
            f"{manifest.dataset_version!r}/{manifest.release.tag!r}"
        )

    by_name: dict[str, LocalAsset] = {}
    for component in manifest.components:
        for part in component.parts:
            if part.name is None:
                raise ReleaseError(
                    f"component {component.id} uses an external URL instead of a release asset"
                )
            if "/" in part.name or "\\" in part.name:
                raise ReleaseError(f"GitHub release asset names must be flat: {part.name!r}")
            if part.name in by_name:
                previous = by_name[part.name]
                if previous.size != part.size or previous.sha256 != part.sha256:
                    raise ReleaseError(f"conflicting duplicate release asset {part.name!r}")
                continue
            path = directory / part.name
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise ReleaseError(f"missing release asset {path}: {exc}") from exc
            if not path.is_file() or path.is_symlink():
                raise ReleaseError(f"release asset is not a regular file: {path}")
            if size != part.size:
                raise ReleaseError(
                    f"release asset {part.name} size mismatch: expected {part.size}, got {size}"
                )
            digest = _sha256(path)
            if digest != part.sha256:
                raise ReleaseError(f"release asset {part.name} SHA-256 mismatch")
            by_name[part.name] = LocalAsset(part.name, path, size, digest)

    manifest_asset = LocalAsset(
        "manifest.json", manifest_path, len(raw), hashlib.sha256(raw).hexdigest()
    )
    assets = (manifest_asset, *sorted(by_name.values(), key=lambda item: item.name))
    if len(assets) > MAX_RELEASE_ASSETS:
        raise ReleaseError(
            f"release has {len(assets)} assets; GitHub permits at most {MAX_RELEASE_ASSETS}"
        )
    return ReleaseBundle(directory, manifest, assets)


def run_gh(arguments: list[str]) -> str:
    """Run one GitHub CLI command without invoking a shell."""

    try:
        result = subprocess.run(
            ["gh", *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise ReleaseError("GitHub CLI (gh) is not installed") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ReleaseError(f"gh {' '.join(arguments)} failed: {detail}") from exc
    return result.stdout


def _gh_api_json(runner: CommandRunner, endpoint: str) -> Any:
    output = runner(
        [
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            f"X-GitHub-Api-Version: {API_VERSION}",
            endpoint,
        ]
    )
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"GitHub API returned invalid JSON for {endpoint}") from exc


def _release_or_none(
    runner: CommandRunner, repository: str, tag: str
) -> dict[str, Any] | None:
    try:
        value = _gh_api_json(runner, f"repos/{repository}/releases/tags/{tag}")
    except ReleaseError as exc:
        if "HTTP 404" in str(exc) or "release not found" in str(exc).casefold():
            return None
        raise
    if not isinstance(value, dict):
        raise ReleaseError("GitHub release response must be an object")
    return value


def _remote_assets(
    runner: CommandRunner, repository: str, release_id: int
) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for page in range(1, 12):
        value = _gh_api_json(
            runner,
            f"repos/{repository}/releases/{release_id}/assets?per_page=100&page={page}",
        )
        if not isinstance(value, list):
            raise ReleaseError("GitHub release assets response must be an array")
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ReleaseError("GitHub returned a malformed release asset")
            name = item["name"]
            if name in assets:
                raise ReleaseError(f"GitHub release contains duplicate asset {name!r}")
            assets[name] = item
        if len(value) < 100:
            return assets
    raise ReleaseError("GitHub release asset pagination exceeded the 1,000-asset limit")


def _verify_remote_assets(
    bundle: ReleaseBundle, remote: dict[str, dict[str, Any]]
) -> list[LocalAsset]:
    expected = {asset.name: asset for asset in bundle.assets}
    unexpected = sorted(set(remote) - set(expected))
    if unexpected:
        raise ReleaseError(f"draft release contains unexpected assets: {unexpected}")
    missing: list[LocalAsset] = []
    for name, asset in expected.items():
        item = remote.get(name)
        if item is None:
            missing.append(asset)
            continue
        if item.get("size") != asset.size:
            raise ReleaseError(f"remote asset {name!r} has the wrong size")
        remote_digest = item.get("digest")
        if remote_digest is not None and remote_digest != f"sha256:{asset.sha256}":
            raise ReleaseError(f"remote asset {name!r} has the wrong SHA-256 digest")
    return missing


def stage_release(
    bundle: ReleaseBundle,
    *,
    repository: str,
    tag: str,
    runner: CommandRunner = run_gh,
) -> dict[str, Any]:
    """Create/update a draft release and upload only missing verified assets."""

    immutable = _gh_api_json(runner, f"repos/{repository}/immutable-releases")
    if not isinstance(immutable, dict) or immutable.get("enabled") is not True:
        raise ReleaseError("repository immutable releases must be enabled before staging")
    release = _release_or_none(runner, repository, tag)
    if release is None:
        runner(
            [
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--draft",
                "--target",
                bundle.manifest.transformation_commit,
                "--title",
                f"Lexicon MCP data {tag}",
                "--notes",
                "Full offline corpus. See DATA_LICENSES.md and manifest.json for provenance.",
            ]
        )
        release = _release_or_none(runner, repository, tag)
    if release is None:
        raise ReleaseError("GitHub did not return the newly created draft release")
    if release.get("draft") is not True:
        raise ReleaseError("refusing to alter an already-published release")
    release_id = release.get("id")
    if isinstance(release_id, bool) or not isinstance(release_id, int):
        raise ReleaseError("GitHub draft release has no numeric ID")

    remote = _remote_assets(runner, repository, release_id)
    missing = _verify_remote_assets(bundle, remote)
    for offset in range(0, len(missing), UPLOAD_BATCH_SIZE):
        batch = missing[offset : offset + UPLOAD_BATCH_SIZE]
        runner(
            [
                "release",
                "upload",
                tag,
                "--repo",
                repository,
                *(str(asset.path) for asset in batch),
            ]
        )
    final_remote = _remote_assets(runner, repository, release_id)
    still_missing = _verify_remote_assets(bundle, final_remote)
    if still_missing:
        raise ReleaseError(
            "GitHub draft is still missing assets: "
            + ", ".join(item.name for item in still_missing)
        )
    return {
        "action": "staged",
        "draft": True,
        "repository": repository,
        "tag": tag,
        "release_id": release_id,
        "asset_count": len(bundle.assets),
        "uploaded": len(missing),
        "manifest_sha256": bundle.manifest.sha256,
    }


def _validate_acceptance(path: Path, bundle: ReleaseBundle) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read acceptance report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError("acceptance report must be an object")
    expected = {
        "dataset_version": bundle.manifest.dataset_version,
        "manifest_sha256": bundle.manifest.sha256,
        "transformation_commit": bundle.manifest.transformation_commit,
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise ReleaseError(f"acceptance report {key} does not match the release bundle")
    for key in (
        "clean_install_ok",
        "verify_ok",
        "full_corpus_ok",
        "ann_ok",
        "performance_ok",
        "live_stack_ok",
        "offline_runtime_ok",
    ):
        if value.get(key) is not True:
            raise ReleaseError(f"acceptance report requires {key}=true")
    if value.get("restart_cycles") != 10:
        raise ReleaseError("acceptance report requires restart_cycles=10")
    if value.get("active_models") != []:
        raise ReleaseError("acceptance must finish with active_models=[]")
    return value


def publish_release(
    bundle: ReleaseBundle,
    *,
    repository: str,
    tag: str,
    acceptance_report: Path,
    runner: CommandRunner = run_gh,
) -> dict[str, Any]:
    """Publish a complete draft only after exact-bundle acceptance has passed."""

    _validate_acceptance(acceptance_report, bundle)
    release = _release_or_none(runner, repository, tag)
    if release is None or release.get("draft") is not True:
        raise ReleaseError("the exact data release must exist as a draft before publishing")
    release_id = release.get("id")
    if isinstance(release_id, bool) or not isinstance(release_id, int):
        raise ReleaseError("GitHub draft release has no numeric ID")
    remote = _remote_assets(runner, repository, release_id)
    missing = _verify_remote_assets(bundle, remote)
    if missing:
        raise ReleaseError(
            "draft release is missing assets: " + ", ".join(item.name for item in missing)
        )
    runner(["release", "edit", tag, "--repo", repository, "--draft=false"])
    published = _release_or_none(runner, repository, tag)
    if published is None or published.get("draft") is not False:
        raise ReleaseError("GitHub did not publish the data release")
    if published.get("immutable") is not True:
        raise ReleaseError("published data release is not immutable")
    return {
        "action": "published",
        "draft": False,
        "immutable": True,
        "repository": repository,
        "tag": tag,
        "asset_count": len(bundle.assets),
        "manifest_sha256": bundle.manifest.sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, stage, or explicitly publish a Lexicon MCP data release."
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--tag", default="data-v1.0.0")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage", action="store_true", help="create/update the draft release")
    action.add_argument(
        "--publish", action="store_true", help="publish an accepted complete draft"
    )
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        help="required JSON evidence for --publish",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = validate_bundle(
            args.package_dir, repository=args.repository, tag=args.tag
        )
        if args.publish:
            if args.acceptance_report is None:
                raise ReleaseError("--publish requires --acceptance-report")
            result = publish_release(
                bundle,
                repository=args.repository,
                tag=args.tag,
                acceptance_report=args.acceptance_report,
            )
        elif args.stage:
            result = stage_release(
                bundle, repository=args.repository, tag=args.tag
            )
        else:
            result = {
                "action": "validated",
                "repository": args.repository,
                "tag": args.tag,
                "asset_count": len(bundle.assets),
                "manifest_sha256": bundle.manifest.sha256,
            }
    except (ReleaseError, ManifestError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
