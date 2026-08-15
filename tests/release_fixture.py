"""A miniature schema-2 release, shared by the installer and CLI tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VERSION = "data-v2.0.0"

PAYLOADS = {
    "artifact-core": ("core.sqlite3", b"core catalogue payload"),
    "artifact-lexical-en": ("lexical/en.sqlite3", b"english lexical payload" * 4),
    "artifact-lexical-fr": ("lexical/fr.sqlite3", b"french lexical payload" * 3),
    "artifact-lexical-bundle": ("lexical/bundle.sqlite3", b"bundled payload" * 2),
    "artifact-semantic-en": ("semantic/en.sqlite3", b"english vectors"),
}

PACKS = [
    {"id": "core", "capability": "core", "component": "artifact-core"},
    {
        "id": "lexical-en",
        "capability": "lexical",
        "languages": ["en"],
        "component": "artifact-lexical-en",
    },
    {
        "id": "lexical-fr",
        "capability": "lexical",
        "languages": ["fr"],
        "component": "artifact-lexical-fr",
    },
    {
        "id": "lexical-bundle-001",
        "capability": "lexical",
        "languages": ["cy", "gv"],
        "component": "artifact-lexical-bundle",
    },
    {
        "id": "semantic-en",
        "capability": "semantic",
        "languages": ["en"],
        "component": "artifact-semantic-en",
    },
]


def write_release(package: Path) -> Path:
    """Write a local schema-2 release directory installable without a network."""

    package.mkdir(parents=True, exist_ok=True)
    components = []
    for component_id, (path, payload) in PAYLOADS.items():
        digest = hashlib.sha256(payload).hexdigest()
        (package / f"{component_id}.part0000").write_bytes(payload)
        components.append(
            {
                "id": component_id,
                "artifact_type": "lexical_sqlite",
                "path": path,
                "compression": "none",
                "compressed_size": len(payload),
                "compressed_sha256": digest,
                "final_size": len(payload),
                "final_sha256": digest,
                "parts": [
                    {
                        "name": f"{component_id}.part0000",
                        "size": len(payload),
                        "sha256": digest,
                        "offset": 0,
                    }
                ],
                "sources": ["fixture"],
                "integrity": {},
            }
        )
    manifest = {
        "schema_version": 2,
        "dataset_version": VERSION,
        "release": {"repository": "DilanRG/lexicon-mcp", "tag": VERSION, "immutable": True},
        "created_at": "2026-08-16T00:00:00Z",
        "transformation_commit": "1" * 40,
        "source_dataset": {"dataset_version": "data-v1.1.0", "manifest_sha256": "a" * 64},
        "sources": [
            {
                "id": "fixture",
                "name": "Test fixture",
                "url": "https://fixtures.invalid/source",
                "revision": "test-1",
                "retrieved_at": "2026-08-16T00:00:00Z",
                "sha256": "0" * 64,
                "size": 0,
                "row_count": None,
                "row_digest": None,
                "license": "CC0-1.0",
            }
        ],
        "components": components,
        "packs": PACKS,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return package
