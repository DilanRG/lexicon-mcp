"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from release_fixture import write_release


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """A local schema-2 release directory, installable without a network."""

    return write_release(tmp_path / "release")
