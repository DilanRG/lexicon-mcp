"""Focused tests for installed-dataset discovery diagnostics."""

import json
from pathlib import Path

import pytest

from lexicon_mcp.runtime.locator import DatasetLocator


def test_missing_dataset_diagnostic_includes_exact_install_command(tmp_path: Path) -> None:
    root = tmp_path / "lexicon-data"

    with pytest.raises(RuntimeError) as error:
        DatasetLocator(root).active()

    message = str(error.value)
    assert "lexicon-data install --profile full --version data-v1.0.0" in message
    assert "LEXICON_DATA_DIR selects the dataset root" in message
    assert str(root.resolve()) in message
    assert str((root / "current.json").resolve()) in message


def test_locator_preserves_english_profile_and_rejects_pointer_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "lexicon-data"
    version = root / "versions" / "data-en-v1.0.0"
    version.mkdir(parents=True)
    (version / "lexicon.sqlite3").write_bytes(b"fixture")
    (version / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "data-en-v1.0.0",
                "profile": "english",
                "languages": ["en"],
            }
        ),
        encoding="utf-8",
    )
    pointer = {
        "schema_version": 1,
        "version": "data-en-v1.0.0",
        "profile": "english",
        "path": "versions/data-en-v1.0.0",
    }
    (root / "current.json").write_text(json.dumps(pointer), encoding="utf-8")

    active = DatasetLocator(root).active()
    assert active.manifest["profile"] == "english"

    pointer["profile"] = "full"
    (root / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Activation profile"):
        DatasetLocator(root).active()
