"""Verified, offline dataset lifecycle for Lexicon MCP.

Importing this package is deliberately side-effect free.  In particular, it
does not inspect the network, create directories, or acquire installation
locks.  Runtime code may read ``current.json`` through :func:`active_version`,
while all mutations live behind the ``lexicon-data`` CLI.
"""

from .lifecycle import DatasetLifecycle, active_version
from .manifest import DatasetManifest, ManifestError, parse_manifest

__all__ = [
    "DatasetLifecycle",
    "DatasetManifest",
    "ManifestError",
    "active_version",
    "parse_manifest",
]
