"""Focused tests for installed-dataset discovery diagnostics."""

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
