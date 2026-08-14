"""Resolve an already-installed, atomically activated dataset."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir


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
        lexical = dataset_path / "lexicon.sqlite3"
        if not lexical.is_file():
            raise RuntimeError(f"Active dataset is missing {lexical.name}")
        return ActiveDataset(self.root, version, dataset_path, manifest)
