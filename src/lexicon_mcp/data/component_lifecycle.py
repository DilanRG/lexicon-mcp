"""Install, activate and maintain schema-2 component datasets.

The data root holds three things: a content-addressed store of verified
components, a directory of immutable activation records, and a pointer naming
the active one.  Every mutation ends in an atomic pointer swap, so an
interrupted operation leaves the previously active install exactly as it was.

Adding or removing a language is therefore not a reinstall.  It re-resolves the
selection, fetches only components the store lacks, writes a new activation and
swaps the pointer; everything already held is reused, and the previous
activation stays valid for rollback.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .activation import Activation, ActivationError, build_activation, parse_activation
from .lifecycle import (
    DatasetLifecycle,
    LifecycleError,
    SpaceError,
    _atomic_replace,
    _read_json,
    _write_atomic,
    default_data_root,
)
from .locking import InstallationLock
from .manifest import DatasetManifest, safe_identifier, safe_version
from .selection import Selection, resolve
from .store import ComponentStore

POINTER_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ComponentLifecycle:
    """Own every mutation of a schema-2 dataset root."""

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        fetcher: DatasetLifecycle | None = None,
    ) -> None:
        self.root = Path(data_root) if data_root is not None else default_data_root()
        self.fetcher = fetcher or DatasetLifecycle(self.root)
        self.store = ComponentStore(self.root / "components")

    @property
    def lock_path(self) -> Path:
        return self.root / ".install.lock"

    @property
    def pointer_path(self) -> Path:
        return self.root / "current.json"

    @property
    def activations_root(self) -> Path:
        return self.root / "activations"

    def activation_path(self, activation_id: str) -> Path:
        safe = safe_identifier(activation_id, field="activation_id")
        return self.activations_root / f"{safe}.json"

    # ------------------------------------------------------------------ read

    def active_activation(self) -> Activation | None:
        pointer = self._pointer()
        if pointer is None:
            return None
        activation_id = pointer.get("activation_id")
        if not isinstance(activation_id, str):
            raise LifecycleError("current.json does not name an activation")
        return self.read_activation(activation_id)

    def read_activation(self, activation_id: str) -> Activation:
        path = self.activation_path(activation_id)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise LifecycleError(f"activation record is missing: {path}") from exc
        activation = parse_activation(raw)
        if activation.activation_id != activation_id:
            raise LifecycleError(
                f"activation record {path} declares id {activation.activation_id!r}"
            )
        return activation

    def activations(self) -> tuple[Activation, ...]:
        if not self.activations_root.is_dir():
            return ()
        found: list[Activation] = []
        for path in sorted(self.activations_root.glob("*.json")):
            try:
                found.append(parse_activation(path.read_bytes()))
            except (OSError, ActivationError):
                continue
        return tuple(found)

    def status(self) -> dict[str, Any]:
        pointer = self._pointer()
        current: dict[str, Any] | None = None
        error: str | None = None
        if pointer is not None:
            if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
                error = (
                    "this dataset was installed by an earlier release and uses the "
                    "schema 1 layout; reinstall it or pin lexicon-mcp 1.2.x"
                )
            else:
                try:
                    activation = self.active_activation()
                except (LifecycleError, ActivationError) as exc:
                    error = str(exc)
                else:
                    assert activation is not None
                    current = {
                        "activation_id": activation.activation_id,
                        "dataset_version": activation.dataset_version,
                        "components": len(activation.components),
                        "installed_size": activation.installed_size(),
                        "effective": {
                            capability: list(languages)
                            for capability, languages in sorted(activation.effective.items())
                        },
                        "unavailable": [dict(item) for item in activation.unavailable],
                    }
        return {
            "data_root": str(self.root),
            "current": current,
            "current_error": error,
            "activations": [
                {
                    "activation_id": item.activation_id,
                    "dataset_version": item.dataset_version,
                    "components": len(item.components),
                    "active": bool(current and current["activation_id"] == item.activation_id),
                }
                for item in self.activations()
            ],
            "store_bytes": self.store.total_bytes(),
        }

    def verify(self, activation_id: str | None = None) -> dict[str, Any]:
        """Hash every component the activation actually installed.

        Deliberately scoped to the activation rather than the manifest: a
        subset install is missing most of the release by design, and treating
        that as damage would leave every partial install permanently broken.
        """

        activation = (
            self.read_activation(activation_id)
            if activation_id is not None
            else self.active_activation()
        )
        if activation is None:
            raise LifecycleError("no active dataset is installed")
        problems: list[str] = []
        damaged: list[str] = []
        for component in activation.components:
            if not self.store.contains(component.sha256):
                problems.append(f"{component.id}: component is missing from the store")
                damaged.append(component.id)
            elif not self.store.verify(component.sha256):
                problems.append(f"{component.id}: SHA-256 mismatch")
                damaged.append(component.id)
        return {
            "ok": not problems,
            "activation_id": activation.activation_id,
            "components_checked": len(activation.components),
            "damaged_components": damaged,
            "problems": problems,
        }

    # ----------------------------------------------------------------- write

    def install(
        self,
        manifest_source: str | Path | DatasetManifest,
        *,
        languages: Sequence[str] | None,
        capabilities: Sequence[str] = ("lexical",),
        strict: bool = True,
    ) -> dict[str, Any]:
        manifest, local_asset_root = self._load(manifest_source)
        selection = resolve(
            manifest, languages=languages, capabilities=capabilities, strict=strict
        )
        return self._apply(
            manifest,
            selection,
            requested_languages=None if languages is None else list(languages),
            requested_capabilities=list(capabilities),
            local_asset_root=local_asset_root,
        )

    def add_languages(
        self,
        manifest_source: str | Path | DatasetManifest,
        *,
        languages: Sequence[str],
        capabilities: Sequence[str] | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Widen the active selection without refetching what is already held."""

        return self._amend(
            manifest_source,
            add=languages,
            remove=(),
            capabilities=capabilities,
            strict=strict,
        )

    def remove_languages(
        self,
        manifest_source: str | Path | DatasetManifest,
        *,
        languages: Sequence[str],
        capabilities: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Narrow the active selection.

        Components stay in the store so the change is instantly reversible;
        reclaiming their space is an explicit `prune`.
        """

        return self._amend(
            manifest_source,
            add=(),
            remove=languages,
            capabilities=capabilities,
            strict=False,
        )

    def activate(self, activation_id: str) -> dict[str, Any]:
        """Point at a previously installed activation."""

        with InstallationLock(self.lock_path):
            activation = self.read_activation(activation_id)
            report = self.verify(activation_id)
            if not report["ok"]:
                raise LifecycleError(
                    "cannot activate a damaged selection: " + "; ".join(report["problems"])
                )
            previous = self._active_id()
            self._write_pointer(activation, previous=previous)
            return {
                "action": "activated",
                "activation_id": activation.activation_id,
                "previous": previous,
            }

    def prune(self) -> dict[str, Any]:
        """Drop stored components no activation references."""

        with InstallationLock(self.lock_path):
            keep: set[str] = set()
            for activation in self.activations():
                keep.update(activation.digests())
            removed = self.store.prune(keep)
            return {
                "action": "pruned",
                "removed": list(removed),
                "retained": len(keep),
                "store_bytes": self.store.total_bytes(),
            }

    def forget(self, activation_id: str) -> dict[str, Any]:
        """Delete an activation record, refusing to remove the active one."""

        with InstallationLock(self.lock_path):
            if activation_id == self._active_id():
                raise LifecycleError("refusing to forget the active activation")
            path = self.activation_path(activation_id)
            if not path.is_file():
                raise LifecycleError(f"no such activation: {activation_id}")
            path.unlink()
            return {"action": "forgotten", "activation_id": activation_id}

    # --------------------------------------------------------------- helpers

    def _load(
        self, manifest_source: str | Path | DatasetManifest
    ) -> tuple[DatasetManifest, Path | None]:
        if isinstance(manifest_source, DatasetManifest):
            return manifest_source, None
        manifest = self.fetcher.load_manifest(manifest_source)
        return manifest, self.fetcher._local_asset_root(manifest_source)

    def _amend(
        self,
        manifest_source: str | Path | DatasetManifest,
        *,
        add: Sequence[str],
        remove: Sequence[str],
        capabilities: Sequence[str] | None,
        strict: bool,
    ) -> dict[str, Any]:
        active = self.active_activation()
        if active is None:
            raise LifecycleError("no active dataset to amend; install one first")
        manifest, local_asset_root = self._load(manifest_source)
        if manifest.sha256 != active.manifest_sha256:
            raise LifecycleError(
                "amendment manifest does not match the activated release"
            )
        wanted_capabilities = list(
            capabilities if capabilities is not None else active.requested_capabilities
        )
        if active.requested_languages is None and add:
            raise LifecycleError("every language is already installed")
        current = set(active.requested_languages or manifest.languages)
        current.update(add)
        current.difference_update(remove)
        if not current:
            raise LifecycleError("an install must retain at least one language")
        selection = resolve(
            manifest,
            languages=sorted(current),
            capabilities=wanted_capabilities,
            strict=strict,
        )
        return self._apply(
            manifest,
            selection,
            requested_languages=sorted(current),
            requested_capabilities=wanted_capabilities,
            local_asset_root=local_asset_root,
        )

    def _apply(
        self,
        manifest: DatasetManifest,
        selection: Selection,
        *,
        requested_languages: list[str] | None,
        requested_capabilities: list[str],
        local_asset_root: Path | None,
    ) -> dict[str, Any]:
        activation = build_activation(
            manifest,
            selection,
            requested_languages=requested_languages,
            requested_capabilities=requested_capabilities,
            created_at=_utc_now(),
        )
        with InstallationLock(self.lock_path):
            missing = [
                component
                for component in activation.components
                if not self.store.verify(component.sha256)
            ]
            self._preflight(sum(item.size for item in missing))
            fetched: list[str] = []
            if missing:
                stage = self.root / f".staging-{uuid.uuid4().hex}"
                try:
                    stage.mkdir(parents=True, exist_ok=False)
                    for component in missing:
                        path = self.fetcher.materialize(
                            manifest,
                            manifest.component(component.id),
                            stage,
                            local_asset_root=local_asset_root,
                        )
                        self.store.adopt(path, component.sha256)
                        fetched.append(component.id)
                finally:
                    shutil.rmtree(stage, ignore_errors=True)

            previous = self._active_id()
            reused = previous == activation.activation_id
            self._retain_manifest(manifest)
            self._write_activation(activation)
            self._write_pointer(activation, previous=previous)
        return {
            "action": "unchanged" if reused else "installed",
            "activation_id": activation.activation_id,
            "dataset_version": activation.dataset_version,
            "components": len(activation.components),
            "components_fetched": fetched,
            "components_reused": len(activation.components) - len(fetched),
            "installed_size": activation.installed_size(),
            "effective": {
                capability: list(languages)
                for capability, languages in sorted(activation.effective.items())
            },
            "unavailable": [dict(item) for item in activation.unavailable],
            "previous_activation": previous,
        }

    def _preflight(self, required: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        margin = max(64 * 1024 * 1024, required // 20)
        try:
            free = int(shutil.disk_usage(self.root).free)
        except OSError as exc:
            raise SpaceError(f"cannot determine free space for {self.root}: {exc}") from exc
        if free < required + margin:
            raise SpaceError(
                f"install requires {required + margin} free bytes "
                f"({required} payload + {margin} reserve), but {free} are available"
            )

    def manifest_path(self, dataset_version: str) -> Path:
        return self.root / "manifests" / f"{safe_version(dataset_version)}.json"

    def _retain_manifest(self, manifest: DatasetManifest) -> None:
        """Keep the manifest an install came from.

        Releases are immutable, so this is a faithful record of what was
        installed -- and it is what lets provenance be read back without asking
        the caller to supply the release again.
        """

        path = self.manifest_path(manifest.dataset_version)
        if path.is_file() and path.read_bytes() == manifest.raw:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(manifest.raw)
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_activation(self, activation: Activation) -> None:
        path = self.activation_path(activation.activation_id)
        if path.is_file():
            # Activations are immutable and content-addressed; an identical
            # selection reuses the record rather than rewriting it.
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            payload = (
                json.dumps(activation.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode()
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_pointer(self, activation: Activation, *, previous: str | None) -> None:
        pointer: dict[str, Any] = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "activation_id": activation.activation_id,
            "dataset_version": activation.dataset_version,
            "activated_at": _utc_now(),
        }
        if previous and previous != activation.activation_id:
            pointer["previous_activation"] = previous
        _write_atomic(self.pointer_path, pointer)

    def _pointer(self) -> dict[str, Any] | None:
        return _read_json(self.pointer_path) if self.pointer_path.is_file() else None

    def _active_id(self) -> str | None:
        pointer = self._pointer()
        if pointer is None or pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
            return None
        activation_id = pointer.get("activation_id")
        return activation_id if isinstance(activation_id, str) else None
