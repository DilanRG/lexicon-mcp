"""Resumable, verified, and rollback-safe dataset installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

from platformdirs import user_data_path

from .integrity import default_semantic_count, sha256_file, verify_component
from .locking import InstallationLock
from .manifest import Component, DatasetManifest, ManifestError, Part, parse_manifest, safe_version
from .transport import Transport, TransportError, UrllibTransport, read_limited


class LifecycleError(RuntimeError):
    """A dataset lifecycle operation could not complete safely."""


class SpaceError(LifecycleError):
    """The target volume lacks enough free space for a verified install."""


class VerificationError(LifecycleError):
    """An installed or staged version did not pass integrity checks."""


class DownloadError(LifecycleError):
    """A release component could not be downloaded and verified."""


def default_data_root() -> Path:
    configured = os.environ.get("LEXICON_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(user_data_path("lexicon-mcp", appauthor=False))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{path} must contain a JSON object")
    return value


def _atomic_replace(source: Path, destination: Path) -> None:
    """Atomically replace a file or rename a directory on the host platform.

    On Windows, Python's ``os.replace`` asks MoveFileEx to replace an existing
    target, which can reject directory renames with Access Denied even when the
    target does not exist.  ``Path.rename`` uses the appropriate same-volume
    directory rename path.  Pointer files retain replace semantics.
    """

    if sys.platform == "win32" and source.is_dir():
        source.rename(destination)
    else:
        os.replace(source, destination)


def local_manifest_path(source: str | Path) -> Path | None:
    """Resolve a local install source to its manifest file, or None for HTTP(S).

    A directory is resolved to the ``manifest.json`` inside it, so a mirror
    produced by ``lexicon-data fetch`` can be installed by pointing at the
    directory itself.
    """

    if not isinstance(source, Path) and str(source).startswith(("http://", "https://")):
        return None
    path = Path(source).expanduser()
    if path.is_dir():
        return path / "manifest.json"
    return path


def _path_within(base: Path, relative: str) -> Path:
    """Resolve an activation path while rejecting traversal and symlinks."""

    if "\\" in relative or ":" in relative:
        raise LifecycleError("activation path is not a safe POSIX relative path")
    pieces = relative.split("/")
    if not pieces or any(piece in {"", ".", ".."} for piece in pieces):
        raise LifecycleError("activation path is not a safe POSIX relative path")
    candidate = base.joinpath(*pieces)
    resolved_base = base.resolve()
    resolved = candidate.resolve()
    if resolved_base not in resolved.parents:
        raise LifecycleError("activation path escapes the dataset root")
    return candidate


def active_version(data_root: Path | None = None) -> tuple[str, Path] | None:
    """Resolve the current installed version without changing any state."""

    root = data_root or default_data_root()
    pointer = root / "current.json"
    if not pointer.is_file():
        return None
    value = _read_json(pointer)
    if value.get("schema_version") != 1:
        raise LifecycleError("current.json has an unsupported schema")
    profile = value.get("profile")
    if profile is not None and profile not in {"full", "english"}:
        raise LifecycleError("current.json has an unsupported dataset profile")
    version = safe_version(value.get("version"), field="current.version")
    expected_path = f"versions/{version}"
    relative = value.get("path")
    if relative != expected_path:
        raise LifecycleError("current.json path does not match its version")
    path = _path_within(root, relative)
    if not path.is_dir() or path.is_symlink():
        raise LifecycleError("active dataset directory is missing or unsafe")
    return version, path


class DatasetLifecycle:
    """Own all explicit dataset mutations; MCP runtime code never constructs it."""

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        transport: Transport | None = None,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        sleep: Callable[[float], None] = time.sleep,
        semantic_count_reader: Callable[[Path, Component], int] = default_semantic_count,
        retries: int = 4,
        safety_margin: int = 64 * 1024 * 1024,
    ) -> None:
        self.root = Path(data_root) if data_root is not None else default_data_root()
        self.transport = transport or UrllibTransport()
        self.disk_usage = disk_usage
        self.sleep = sleep
        self.semantic_count_reader = semantic_count_reader
        self.retries = max(1, retries)
        self.safety_margin = max(0, safety_margin)

    @property
    def lock_path(self) -> Path:
        return self.root / ".install.lock"

    def load_manifest(self, source: str | Path) -> DatasetManifest:
        """Load a local or HTTP(S) release manifest, bounded to 16 MiB."""

        path = local_manifest_path(source)
        if path is not None:
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ManifestError("manifest exceeds 16 MiB")
                raw = path.read_bytes()
            except OSError as exc:
                raise LifecycleError(f"cannot read manifest {path}: {exc}") from exc
        else:
            try:
                with self.transport.open(str(source), 0) as response:
                    if response.status != 200:
                        raise TransportError(f"manifest returned HTTP {response.status}")
                    raw = read_limited(response, 16 * 1024 * 1024)
            except TransportError as exc:
                raise DownloadError(str(exc)) from exc
        return parse_manifest(raw)

    def install(
        self,
        manifest_source: str | Path | DatasetManifest,
        *,
        profile: str,
        version: str,
    ) -> dict[str, Any]:
        if isinstance(manifest_source, DatasetManifest):
            manifest = manifest_source
            local_asset_root: Path | None = None
        else:
            manifest = self.load_manifest(manifest_source)
            local_asset_root = self._local_asset_root(manifest_source)
        self._match_request(manifest, profile=profile, version=version)
        with InstallationLock(self.lock_path):
            self._preflight(manifest)
            target = self._version_path(manifest.dataset_version)
            if target.exists():
                installed = self._load_installed_manifest(target)
                if installed.sha256 != manifest.sha256:
                    raise LifecycleError(
                        f"installed immutable version {version} has a different manifest"
                    )
                report = self._verify_path(target, installed)
                if not report["ok"]:
                    raise VerificationError(
                        "installed version is damaged; run lexicon-data repair"
                    )
                current = self._current_version()
                if current != version:
                    self._activate(installed, previous=current)
                return {
                    "action": "already-installed",
                    "version": version,
                    "profile": profile,
                    "components": len(manifest.components),
                }

            stage = self._staging_path(version, "install")
            try:
                stage.mkdir(parents=True, exist_ok=False)
                (stage / "manifest.json").write_bytes(manifest.raw)
                for component in manifest.components:
                    self._materialize_component(
                        manifest,
                        component,
                        stage,
                        local_asset_root=local_asset_root,
                    )
                report = self._verify_path(stage, manifest)
                if not report["ok"]:
                    raise VerificationError("; ".join(report["problems"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_replace(stage, target)
                self._activate(manifest, previous=self._current_version())
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            return {
                "action": "installed",
                "version": version,
                "profile": profile,
                "components": len(manifest.components),
                "path": str(target),
            }

    def fetch(
        self,
        manifest_source: str | Path | DatasetManifest,
        *,
        profile: str,
        version: str,
        dest: Path | str,
    ) -> dict[str, Any]:
        """Mirror a release into *dest* without touching the dataset root.

        The result reproduces the packaged release layout exactly: one
        ``manifest.json`` beside one file per part, named by its manifest asset
        name.  It installs directly with ``install --from <dest>``, so a
        connected machine can prepare an air-gapped install.  Nothing here
        writes to the data root, activates a version, or takes the install lock.
        """

        destination = Path(dest).expanduser()
        if isinstance(manifest_source, DatasetManifest):
            manifest = manifest_source
            local_asset_root: Path | None = None
        else:
            manifest = self.load_manifest(manifest_source)
            local_asset_root = self._local_asset_root(manifest_source)
        self._match_request(manifest, profile=profile, version=version)
        destination.mkdir(parents=True, exist_ok=True)
        resolved = destination.resolve()
        if local_asset_root is not None and local_asset_root == resolved:
            raise LifecycleError("fetch source and destination are the same directory")
        targets = self._mirror_targets(manifest, resolved)
        self._fetch_preflight(targets, resolved)
        fetched: list[str] = []
        skipped: list[str] = []
        bytes_fetched = 0
        for part, target in targets:
            name = str(part.name)
            if self._valid_file(target, part.size, part.sha256):
                skipped.append(name)
                continue
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise DownloadError(f"unsafe mirror destination: {target}")
            partial = target.with_name(target.name + ".partial")
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise DownloadError(f"unsafe partial mirror path: {partial}")
            if partial.is_file() and partial.stat().st_size > part.size:
                partial.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            # Drop an unverified file first so a failed fetch never leaves one
            # sitting at its final, trusted name.
            target.unlink(missing_ok=True)
            if local_asset_root is not None:
                self._copy_local_asset(local_asset_root, part, partial, target)
            else:
                self._http_download(
                    self._part_url(manifest, part),
                    partial,
                    target,
                    size=part.size,
                    sha256=part.sha256,
                    description=f"release asset {name}",
                )
            fetched.append(name)
            bytes_fetched += part.size
        # The manifest lands last: a mirror interrupted mid-fetch has no
        # manifest, so it fails loudly on install instead of looking complete.
        manifest_path = resolved / "manifest.json"
        temporary = manifest_path.with_name(f".manifest.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(manifest.raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "action": "fetched",
            "version": manifest.dataset_version,
            "profile": manifest.profile,
            "dest": str(resolved),
            "components": len(manifest.components),
            "assets_fetched": len(fetched),
            "assets_skipped": len(skipped),
            "bytes_fetched": bytes_fetched,
            "manifest_sha256": manifest.sha256,
        }

    def materialize(
        self,
        manifest: DatasetManifest,
        component: Component,
        stage: Path,
        *,
        local_asset_root: Path | None = None,
    ) -> Path:
        """Fetch, assemble, decompress and verify one component into *stage*.

        The seam the schema-2 installer fetches through, so both schemas share
        one download path with its resume, retry and integrity behavior.
        """

        self._materialize_component(
            manifest, component, stage, local_asset_root=local_asset_root
        )
        return stage.joinpath(*component.path.parts)

    @staticmethod
    def _mirror_targets(
        manifest: DatasetManifest, destination: Path
    ) -> list[tuple[Part, Path]]:
        """Map every distinct release part onto its mirrored asset path."""

        targets: list[tuple[Part, Path]] = []
        seen: dict[str, str] = {}
        for component in manifest.components:
            for index, part in enumerate(component.parts):
                if part.name is None:
                    raise LifecycleError(
                        f"component {component.id} part {index} has no asset name "
                        "and cannot be mirrored"
                    )
                if part.name == "manifest.json":
                    raise LifecycleError("a release part must not be named manifest.json")
                previous = seen.get(part.name)
                if previous is not None:
                    if previous != part.sha256:
                        raise LifecycleError(
                            f"release asset name is reused with different content: {part.name}"
                        )
                    continue
                seen[part.name] = part.sha256
                targets.append((part, destination.joinpath(*part.name.split("/"))))
        return targets

    def _fetch_preflight(self, targets: list[tuple[Part, Path]], destination: Path) -> None:
        required = 0
        for part, target in targets:
            if self._valid_file(target, part.size, part.sha256):
                continue
            partial = target.with_name(target.name + ".partial")
            present = partial.stat().st_size if partial.is_file() else 0
            required += max(0, part.size - min(present, part.size))
        margin = max(self.safety_margin, required // 20)
        try:
            free = int(self.disk_usage(destination).free)
        except OSError as exc:
            raise SpaceError(f"cannot determine free space for {destination}: {exc}") from exc
        if free < required + margin:
            raise SpaceError(
                f"release mirror requires {required + margin} free bytes "
                f"({required} working + {margin} reserve), but {free} are available"
            )

    def status(self) -> dict[str, Any]:
        """Report pointers and installed manifests without hashing large files."""

        current: dict[str, Any] | None = None
        pointer = self.root / "current.json"
        pointer_error: str | None = None
        if pointer.exists():
            try:
                current = _read_json(pointer)
                active_version(self.root)
            except (LifecycleError, ManifestError) as exc:
                pointer_error = str(exc)
        versions: list[dict[str, Any]] = []
        versions_root = self.root / "versions"
        if versions_root.is_dir():
            for directory in sorted(versions_root.iterdir(), key=lambda item: item.name):
                if not directory.is_dir() or directory.is_symlink():
                    continue
                try:
                    manifest = self._load_installed_manifest(directory)
                except (LifecycleError, ManifestError) as exc:
                    versions.append({"version": directory.name, "manifest_error": str(exc)})
                else:
                    versions.append(
                        {
                            "version": manifest.dataset_version,
                            "profile": manifest.profile,
                            "manifest_sha256": manifest.sha256,
                            "components": len(manifest.components),
                            "active": bool(current and current.get("version") == directory.name),
                        }
                    )
        return {
            "data_root": str(self.root),
            "current": current,
            "current_error": pointer_error,
            "installed_versions": versions,
        }

    def verify(self, version: str | None = None) -> dict[str, Any]:
        active_pointer: dict[str, Any] | None = None
        if version is None:
            active = active_version(self.root)
            if active is None:
                raise LifecycleError("no active dataset is installed")
            version, path = active
            active_pointer = self._pointer_or_none()
        else:
            version = safe_version(version)
            path = self._version_path(version)
            if not path.is_dir():
                raise LifecycleError(f"dataset version is not installed: {version}")
        manifest = self._load_installed_manifest(path)
        report = self._verify_path(path, manifest)
        if active_pointer is not None and active_pointer.get("manifest_sha256") != manifest.sha256:
            report["ok"] = False
            report["problems"].insert(0, "current.json: manifest SHA-256 mismatch")
        if (
            active_pointer is not None
            and active_pointer.get("profile") is not None
            and active_pointer.get("profile") != manifest.profile
        ):
            report["ok"] = False
            report["problems"].insert(0, "current.json: dataset profile mismatch")
        report.update({"version": version, "path": str(path)})
        return report

    def repair(
        self,
        *,
        version: str | None = None,
        manifest_source: str | Path | DatasetManifest | None = None,
    ) -> dict[str, Any]:
        with InstallationLock(self.lock_path):
            if version is None:
                active = active_version(self.root)
                if active is None:
                    raise LifecycleError("no active dataset is installed")
                version, target = active
            else:
                version = safe_version(version)
                target = self._version_path(version)
            if not target.is_dir() or target.is_symlink():
                raise LifecycleError(f"dataset version is not installed: {version}")
            local_asset_root: Path | None = None
            if manifest_source is None:
                manifest = self._load_installed_manifest(target)
            elif isinstance(manifest_source, DatasetManifest):
                manifest = manifest_source
            else:
                manifest = self.load_manifest(manifest_source)
                local_asset_root = self._local_asset_root(manifest_source)
            self._match_request(manifest, profile=manifest.profile, version=version)
            pointer = self._pointer_or_none()
            if pointer and pointer.get("version") == version:
                expected_profile = pointer.get("profile")
                if expected_profile is not None and expected_profile != manifest.profile:
                    raise LifecycleError("repair manifest profile does not match active profile")
                expected_manifest_sha = pointer.get("manifest_sha256")
                if expected_manifest_sha and expected_manifest_sha != manifest.sha256:
                    raise LifecycleError("repair manifest does not match the activated release")
            report = self._verify_path(target, manifest)
            damaged = set(report["damaged_components"])
            manifest_damaged = any(
                problem.startswith("manifest.json:") for problem in report["problems"]
            )
            if not damaged and not manifest_damaged:
                return {
                    "action": "healthy",
                    "version": version,
                    "profile": manifest.profile,
                    "repaired": [],
                }
            if damaged:
                self._preflight(manifest, components=damaged)
            stage = self._staging_path(version, "repair")
            repaired: list[str] = []
            try:
                stage.mkdir(parents=True, exist_ok=False)
                for component in manifest.components:
                    if component.id not in damaged:
                        continue
                    self._materialize_component(
                        manifest,
                        component,
                        stage,
                        local_asset_root=local_asset_root,
                    )
                    source = stage.joinpath(*component.path.parts)
                    destination = target.joinpath(*component.path.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
                    repaired.append(component.id)
                # Restore a damaged manifest only after all repaired artifacts exist.
                manifest_path = target / "manifest.json"
                if manifest_damaged:
                    temporary = target / f".manifest.{uuid.uuid4().hex}.tmp"
                    temporary.write_bytes(manifest.raw)
                    os.replace(temporary, manifest_path)
                final = self._verify_path(target, manifest)
                if not final["ok"]:
                    raise VerificationError("repair incomplete: " + "; ".join(final["problems"]))
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            return {
                "action": "repaired",
                "version": version,
                "profile": manifest.profile,
                "repaired": repaired,
                "manifest_restored": manifest_damaged,
            }

    def rollback(self, version: str | None = None) -> dict[str, Any]:
        with InstallationLock(self.lock_path):
            pointer = self._pointer_or_none()
            if pointer is None:
                raise LifecycleError("no active dataset is installed")
            current = safe_version(pointer.get("version"), field="current.version")
            target_version = version or pointer.get("previous_version")
            if target_version is None:
                raise LifecycleError("there is no previous dataset version to roll back to")
            target_version = safe_version(target_version, field="rollback.version")
            if target_version == current:
                raise LifecycleError("rollback target is already active")
            target = self._version_path(target_version)
            if not target.is_dir() or target.is_symlink():
                raise LifecycleError(f"rollback target is not installed: {target_version}")
            manifest = self._load_installed_manifest(target)
            report = self._verify_path(target, manifest)
            if not report["ok"]:
                raise VerificationError(
                    "rollback target is damaged: " + "; ".join(report["problems"])
                )
            self._activate(manifest, previous=current)
            return {"action": "rolled-back", "version": target_version, "previous": current}

    def _match_request(self, manifest: DatasetManifest, *, profile: str, version: str) -> None:
        version = safe_version(version)
        if profile not in {"full", "english"}:
            raise LifecycleError("dataset profile must be full or english")
        if manifest.profile != profile:
            raise LifecycleError(
                f"manifest profile {manifest.profile!r} does not match requested {profile!r}"
            )
        if manifest.dataset_version != version or manifest.release.tag != version:
            raise LifecycleError(
                f"manifest version {manifest.dataset_version!r} does not match "
                f"requested {version!r}"
            )

    def _preflight(
        self, manifest: DatasetManifest, *, components: set[str] | None = None
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        required = 0
        for component in manifest.components:
            if components is not None and component.id not in components:
                continue
            cache = self._component_cache(manifest.dataset_version, component)
            for index, part in enumerate(component.parts):
                complete = self._part_path(cache, index, part, complete=True)
                partial = self._part_path(cache, index, part, complete=False)
                if self._valid_file(complete, part.size, part.sha256):
                    continue
                partial_size = partial.stat().st_size if partial.is_file() else 0
                required += max(0, part.size - min(partial_size, part.size))
            required += component.final_size
            if component.compression != "none":
                required += component.compressed_size
        margin = max(self.safety_margin, required // 20)
        try:
            free = int(self.disk_usage(self.root).free)
        except OSError as exc:
            raise SpaceError(f"cannot determine free space for {self.root}: {exc}") from exc
        if free < required + margin:
            raise SpaceError(
                f"dataset install requires {required + margin} free bytes "
                f"({required} working + {margin} reserve), but {free} are available"
            )

    def _version_path(self, version: str) -> Path:
        version = safe_version(version)
        return self.root / "versions" / version

    def _staging_path(self, version: str, action: str) -> Path:
        # The temporary directory and final version share one parent.  This is
        # required for a true atomic directory rename and avoids a Windows
        # MoveFileEx edge case across sibling directories.
        return (
            self.root
            / "versions"
            / f".{action}-{safe_version(version)}-{uuid.uuid4().hex}.staging"
        )

    def _component_cache(self, version: str, component: Component) -> Path:
        return self.root / ".downloads" / safe_version(version) / component.id

    @staticmethod
    def _part_path(cache: Path, index: int, part: Part, *, complete: bool) -> Path:
        suffix = "part" if complete else "partial"
        return cache / f"{index:04d}-{part.sha256}.{suffix}"

    def _part_url(self, manifest: DatasetManifest, part: Part) -> str:
        if part.url:
            return part.url
        if part.name is None:
            raise DownloadError("manifest part has neither URL nor name")
        encoded = "/".join(quote(piece, safe="") for piece in part.name.split("/"))
        if manifest.release.base_url:
            return urljoin(manifest.release.base_url.rstrip("/") + "/", encoded)
        repository = manifest.release.repository
        tag = quote(manifest.release.tag, safe="")
        return f"https://github.com/{repository}/releases/download/{tag}/{encoded}"

    @staticmethod
    def _local_asset_root(source: str | Path) -> Path | None:
        """Resolve sibling assets only when the manifest itself is local.

        This allows a clean installation of the exact production manifest
        before its immutable GitHub release is published, and installation from
        a mirror directory produced by ``fetch``.  HTTP manifests retain the
        ordinary release URL behavior.
        """

        path = local_manifest_path(source)
        return None if path is None else path.resolve().parent

    @staticmethod
    def _valid_file(path: Path, size: int, digest: str) -> bool:
        try:
            return (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == size
                and sha256_file(path) == digest
            )
        except OSError:
            return False

    def _download_part(
        self,
        manifest: DatasetManifest,
        component: Component,
        index: int,
        part: Part,
        *,
        local_asset_root: Path | None = None,
    ) -> Path:
        cache = self._component_cache(manifest.dataset_version, component)
        cache.mkdir(parents=True, exist_ok=True)
        complete = self._part_path(cache, index, part, complete=True)
        partial = self._part_path(cache, index, part, complete=False)
        if self._valid_file(complete, part.size, part.sha256):
            return complete
        complete.unlink(missing_ok=True)
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise DownloadError(f"unsafe partial download path: {partial}")
        if partial.is_file() and partial.stat().st_size > part.size:
            partial.unlink()
        if local_asset_root is not None:
            # A local install source is a hard offline guarantee.  Every part is
            # resolved on disk; a part that cannot be is an error rather than a
            # silent fall back to the network.
            return self._copy_local_asset(local_asset_root, part, partial, complete)
        return self._http_download(
            self._part_url(manifest, part),
            partial,
            complete,
            size=part.size,
            sha256=part.sha256,
            description=f"component {component.id} part {index}",
        )

    @staticmethod
    def _copy_local_asset(
        local_asset_root: Path, part: Part, partial: Path, complete: Path
    ) -> Path:
        """Copy one verified release part from a local asset directory."""

        if part.name is None:
            raise DownloadError(
                "local release part has no asset name; this release cannot be "
                "installed from a local source"
            )
        root = local_asset_root.resolve()
        pieces = part.name.split("/")
        cursor = root
        for piece in pieces:
            cursor /= piece
            if cursor.is_symlink():
                raise DownloadError(
                    f"local release asset must not traverse a symlink: {part.name}"
                )
        try:
            source = root.joinpath(*pieces).resolve(strict=True)
        except OSError as exc:
            raise DownloadError(f"local release asset is missing: {part.name}") from exc
        if not source.is_relative_to(root) or not source.is_file():
            raise DownloadError(f"local release asset is missing or unsafe: {part.name}")
        offset = partial.stat().st_size if partial.is_file() else 0
        try:
            with (
                source.open("rb") as input_stream,
                partial.open("ab" if offset else "wb") as output,
            ):
                input_stream.seek(offset)
                received = offset
                while chunk := input_stream.read(1024 * 1024):
                    received += len(chunk)
                    if received > part.size:
                        raise DownloadError("local release asset exceeds its declared size")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except OSError as exc:
            raise DownloadError(f"cannot copy local release asset {part.name}: {exc}") from exc
        if partial.stat().st_size != part.size:
            raise DownloadError(
                f"incomplete local release asset (expected {part.size}, "
                f"got {partial.stat().st_size})"
            )
        if sha256_file(partial) != part.sha256:
            partial.unlink(missing_ok=True)
            raise DownloadError(f"local release asset SHA-256 mismatch: {part.name}")
        os.replace(partial, complete)
        return complete

    def _http_download(
        self,
        url: str,
        partial: Path,
        complete: Path,
        *,
        size: int,
        sha256: str,
        description: str,
    ) -> Path:
        """Resumably fetch one verified payload, retrying with backoff."""

        last_error: BaseException | None = None
        for attempt in range(self.retries):
            offset = partial.stat().st_size if partial.is_file() else 0
            try:
                response = self.transport.open(url, offset)
                with response:
                    if offset and response.status != 206:
                        partial.unlink(missing_ok=True)
                        raise DownloadError("server ignored the HTTP Range request; restarting")
                    if not offset and response.status not in {200, 206}:
                        raise DownloadError(f"release asset returned HTTP {response.status}")
                    if response.status == 206:
                        content_range = response.headers.get("content-range")
                        if not content_range or not content_range.startswith(f"bytes {offset}-"):
                            partial.unlink(missing_ok=True)
                            raise DownloadError("release asset returned an invalid Content-Range")
                    mode = "ab" if offset else "wb"
                    received = offset
                    with partial.open(mode) as output:
                        while chunk := response.body.read(1024 * 1024):
                            received += len(chunk)
                            if received > size:
                                raise DownloadError("release asset exceeds its declared size")
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                actual_size = partial.stat().st_size
                if actual_size != size:
                    raise DownloadError(
                        f"incomplete release part (expected {size}, got {actual_size})"
                    )
                if sha256_file(partial) != sha256:
                    partial.unlink(missing_ok=True)
                    raise DownloadError("release part SHA-256 mismatch")
                os.replace(partial, complete)
                return complete
            except (DownloadError, TransportError, OSError) as exc:
                last_error = exc
                if partial.is_file() and partial.stat().st_size > size:
                    partial.unlink(missing_ok=True)
                if attempt + 1 < self.retries:
                    self.sleep(min(8.0, 0.5 * (2**attempt)))
        raise DownloadError(f"unable to download {description}: {last_error}") from last_error

    def _materialize_component(
        self,
        manifest: DatasetManifest,
        component: Component,
        stage: Path,
        *,
        local_asset_root: Path | None = None,
    ) -> None:
        destination = stage.joinpath(*component.path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise LifecycleError(f"component destination already exists: {component.path}")
        parts = [
            self._download_part(
                manifest,
                component,
                index,
                part,
                local_asset_root=local_asset_root,
            )
            for index, part in enumerate(component.parts)
        ]
        if component.compression == "none":
            self._assemble(parts, destination, component)
        else:
            compressed = stage / ".compressed" / f"{component.id}.payload"
            compressed.parent.mkdir(parents=True, exist_ok=True)
            self._assemble_compressed(parts, compressed, component)
            try:
                if component.compression == "zstd":
                    self._extract_zstd(compressed, destination, component.final_size)
                elif component.compression == "zip":
                    self._extract_zip(compressed, destination, component)
                else:  # guarded by manifest validation
                    raise LifecycleError(f"unsupported compression: {component.compression}")
            finally:
                compressed.unlink(missing_ok=True)
        problems = verify_component(
            stage,
            component,
            semantic_count_reader=self.semantic_count_reader,
            check_cross_component=False,
        )
        if problems:
            destination.unlink(missing_ok=True)
            raise VerificationError("; ".join(problems))

    @staticmethod
    def _assemble(parts: list[Path], destination: Path, component: Component) -> None:
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("xb") as output:
                for part in parts:
                    with part.open("rb") as stream:
                        while chunk := stream.read(4 * 1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        if size != component.compressed_size or digest.hexdigest() != component.compressed_sha256:
            destination.unlink(missing_ok=True)
            raise VerificationError(f"assembled payload mismatch for {component.id}")

    @staticmethod
    def _assemble_compressed(
        parts: list[Path], destination: Path, component: Component
    ) -> None:
        DatasetLifecycle._assemble(parts, destination, component)

    @staticmethod
    def _extract_zstd(source: Path, destination: Path, expected_size: int) -> None:
        try:
            import zstandard

            with source.open("rb") as compressed, destination.open("xb") as output:
                reader = zstandard.ZstdDecompressor().stream_reader(compressed)
                written = 0
                try:
                    while chunk := reader.read(4 * 1024 * 1024):
                        written += len(chunk)
                        if written > expected_size:
                            raise VerificationError(
                                "Zstandard component exceeds its declared final size"
                            )
                        output.write(chunk)
                finally:
                    reader.close()  # type: ignore[no-untyped-call]
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    @staticmethod
    def _extract_zip(source: Path, destination: Path, component: Component) -> None:
        try:
            with zipfile.ZipFile(source) as archive:
                regular = []
                for member in archive.infolist():
                    # Reject symlinks and traversal even though only one member is copied.
                    mode = (member.external_attr >> 16) & 0xF000
                    if mode == 0xA000:
                        raise VerificationError("ZIP release component contains a symlink")
                    name = member.filename
                    if "\\" in name or name.startswith("/") or ".." in Path(name).parts:
                        raise VerificationError("ZIP release component contains an unsafe path")
                    if not member.is_dir():
                        regular.append(member)
                requested = component.integrity.get("archive_member")
                if requested is not None:
                    matches = [member for member in regular if member.filename == requested]
                else:
                    matches = regular
                if len(matches) != 1:
                    raise VerificationError(
                        "ZIP component must contain exactly one selected regular file"
                    )
                if matches[0].file_size != component.final_size:
                    raise VerificationError("ZIP member size does not match manifest")
                with archive.open(matches[0]) as compressed, destination.open("xb") as output:
                    written = 0
                    while chunk := compressed.read(4 * 1024 * 1024):
                        written += len(chunk)
                        if written > component.final_size:
                            raise VerificationError("ZIP member exceeds its declared final size")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    def _load_installed_manifest(self, directory: Path) -> DatasetManifest:
        manifest_path = directory / "manifest.json"
        try:
            return parse_manifest(manifest_path.read_bytes())
        except OSError as exc:
            raise LifecycleError(f"installed manifest is missing: {manifest_path}") from exc

    def _verify_path(
        self, directory: Path, manifest: DatasetManifest
    ) -> dict[str, Any]:
        problems: list[str] = []
        damaged: list[str] = []
        manifest_path = directory / "manifest.json"
        if not self._valid_file(manifest_path, len(manifest.raw), manifest.sha256):
            problems.append("manifest.json: SHA-256 or size mismatch")
        for component in manifest.components:
            component_problems = verify_component(
                directory,
                component,
                semantic_count_reader=self.semantic_count_reader,
                check_cross_component=True,
            )
            if component_problems:
                damaged.append(component.id)
                problems.extend(component_problems)
        return {
            "ok": not problems,
            "components_checked": len(manifest.components),
            "damaged_components": damaged,
            "problems": problems,
        }

    def _pointer_or_none(self) -> dict[str, Any] | None:
        pointer = self.root / "current.json"
        return _read_json(pointer) if pointer.is_file() else None

    def _current_version(self) -> str | None:
        pointer = self._pointer_or_none()
        if pointer is None:
            return None
        return safe_version(pointer.get("version"), field="current.version")

    def _activate(self, manifest: DatasetManifest, *, previous: str | None) -> None:
        target = self._version_path(manifest.dataset_version)
        if not target.is_dir() or target.is_symlink():
            raise LifecycleError("cannot activate a missing or unsafe version directory")
        pointer: dict[str, Any] = {
            "schema_version": 1,
            "version": manifest.dataset_version,
            "profile": manifest.profile,
            "path": f"versions/{manifest.dataset_version}",
            "manifest_sha256": manifest.sha256,
            "activated_at": _utc_now(),
        }
        if previous and previous != manifest.dataset_version:
            pointer["previous_version"] = safe_version(previous)
        _write_atomic(self.root / "current.json", pointer)
