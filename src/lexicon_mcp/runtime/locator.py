"""Resolve an already-installed, atomically activated dataset.

Schema 1 resolved a directory; schema 2 resolves an activation record and the
content-addressed store behind it. Both are read-only views: nothing here can
download, activate or delete anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from ..data.activation import Activation, ActivationError, parse_activation
from ..data.store import ComponentStore
from .router import PackRouter


@dataclass(frozen=True, slots=True)
class ActiveDataset:
    root: Path
    version: str
    path: Path
    manifest: dict[str, Any]

    @property
    def lexical_database(self) -> Path:
        return self.path / "lexicon.sqlite3"

    @property
    def semantic_directory(self) -> Path:
        return self.path / "semantic"


class DatasetLocator:
    """Read current.json without installing, repairing, or mutating anything."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        configured = root or os.environ.get("LEXICON_DATA_DIR")
        self.root = Path(configured or user_data_dir("lexicon-mcp")).expanduser().resolve()

    def active(self) -> ActiveDataset:
        activation_path = self.root / "current.json"
        try:
            activation = json.loads(activation_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"No active lexicon dataset at {activation_path}. "
                "Install one explicitly with: "
                "lexicon-data install --profile full --version data-v1.0.0. "
                f"LEXICON_DATA_DIR selects the dataset root (currently {self.root})."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read valid activation metadata at {activation_path}"
            ) from exc

        if not isinstance(activation, dict):
            raise RuntimeError("Dataset activation metadata must be a JSON object")
        version = activation.get("version")
        relative_path = activation.get("path")
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError("Dataset activation metadata has no valid version")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise RuntimeError("Dataset activation metadata has no valid path")

        candidate = Path(relative_path)
        dataset_path = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if not dataset_path.is_relative_to(self.root):
            raise RuntimeError("Active dataset path escapes LEXICON_DATA_DIR")
        if not dataset_path.is_dir():
            raise RuntimeError(f"Active dataset directory does not exist: {dataset_path}")

        manifest_path = dataset_path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Active dataset has no manifest: {manifest_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Active dataset has an invalid manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("Dataset manifest must be a JSON object")
        manifest_version = manifest.get("version") or manifest.get("dataset_version")
        if manifest_version is not None and manifest_version != version:
            raise RuntimeError(
                f"Activation version {version!r} does not match "
                f"manifest version {manifest_version!r}"
            )
        profile = manifest.get("profile", "full")
        if profile not in {"full", "english"}:
            raise RuntimeError(f"Dataset manifest has unsupported profile {profile!r}")
        activation_profile = activation.get("profile")
        if activation_profile is not None and activation_profile != profile:
            raise RuntimeError(
                f"Activation profile {activation_profile!r} does not match "
                f"manifest profile {profile!r}"
            )
        if profile == "english" and manifest.get("languages") != ["en"]:
            raise RuntimeError("English dataset manifest must declare languages=['en']")
        lexical = dataset_path / "lexicon.sqlite3"
        if not lexical.is_file():
            raise RuntimeError(f"Active dataset is missing {lexical.name}")
        return ActiveDataset(self.root, version, dataset_path, manifest)


@dataclass(frozen=True, slots=True)
class ActiveComponents:
    """A resolved schema-2 install: an activation plus the store backing it."""

    root: Path
    activation: Activation
    store: ComponentStore

    @property
    def version(self) -> str:
        return self.activation.dataset_version

    def router(self, **options: Any) -> PackRouter:
        return PackRouter(self.activation, self.store, **options)


def _pointer_schema(root: Path) -> int | None:
    pointer = root / "current.json"
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value.get("schema_version") if isinstance(value, dict) else None


class ComponentLocator:
    """Resolve the active schema-2 activation without mutating anything.

    Deliberately reads the activation record and the store directly rather than
    importing the installer: runtime code must never be able to reach an object
    that can download, activate or delete data.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        configured = root or os.environ.get("LEXICON_DATA_DIR")
        self.root = Path(configured or user_data_dir("lexicon-mcp")).expanduser().resolve()

    def active(self) -> ActiveComponents:
        pointer_path = self.root / "current.json"
        schema = _pointer_schema(self.root)
        if schema is None:
            raise RuntimeError(
                f"No active lexicon dataset at {pointer_path}. Install one explicitly, "
                "for example: lexicon-data install --version data-v2.0.0 --languages en. "
                f"LEXICON_DATA_DIR selects the dataset root (currently {self.root})."
            )
        if schema != 2:
            raise RuntimeError(
                f"The dataset at {self.root} uses the schema {schema} layout, which this "
                "release does not serve. Reinstall it with a schema 2 release, or pin "
                "lexicon-mcp 1.2.x to keep using the existing corpus."
            )
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        activation_id = pointer.get("activation_id")
        if not isinstance(activation_id, str) or not activation_id.strip():
            raise RuntimeError("Activation pointer does not name an activation")
        record = self.root / "activations" / f"{activation_id}.json"
        try:
            activation = parse_activation(record.read_bytes())
        except FileNotFoundError as exc:
            raise RuntimeError(f"Active dataset has no activation record: {record}") from exc
        except (OSError, ActivationError) as exc:
            raise RuntimeError(f"Active dataset has an invalid activation: {exc}") from exc
        if activation.activation_id != activation_id:
            raise RuntimeError("Activation record does not match the pointer that names it")

        store = ComponentStore(self.root / "components")
        missing = [
            component.id
            for component in activation.components
            if not store.contains(component.sha256)
        ]
        if missing:
            raise RuntimeError(
                "Active dataset is missing installed components "
                f"({', '.join(sorted(missing))}); run lexicon-data verify"
            )
        return ActiveComponents(self.root, activation, store)
