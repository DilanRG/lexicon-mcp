"""Immutable activation records.

An activation is the real identity of an install.  The languages a user asked
for are only intent; what was actually resolved, downloaded and verified is the
component set recorded here.  Verification checks this set rather than the whole
manifest, which is what makes a partial install legal instead of permanently
"damaged", and the runtime routes queries through it rather than assuming any
directory layout.

Records are content-addressed by their own selection, so re-resolving the same
request yields the same activation id and simply reuses the existing record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .manifest import DatasetManifest, is_sha256, normalize_language, safe_identifier
from .selection import Selection

ACTIVATION_SCHEMA_VERSION = 2


class ActivationError(ValueError):
    """An activation record is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ActivationComponent:
    id: str
    sha256: str
    path: str
    size: int


@dataclass(frozen=True, slots=True)
class ActivationPack:
    id: str
    capability: str
    languages: tuple[str, ...]
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Activation:
    schema_version: int
    activation_id: str
    dataset_version: str
    manifest_sha256: str
    created_at: str
    requested_languages: tuple[str, ...] | None
    requested_capabilities: tuple[str, ...]
    effective: Mapping[str, tuple[str, ...]]
    unavailable: tuple[Mapping[str, str], ...]
    components: tuple[ActivationComponent, ...]
    packs: tuple[ActivationPack, ...]

    def component(self, component_id: str) -> ActivationComponent:
        for item in self.components:
            if item.id == component_id:
                return item
        raise ActivationError(f"activation has no component {component_id!r}")

    def component_for(self, capability: str, language: str) -> ActivationComponent | None:
        """The component serving *language* for *capability*, if it is installed.

        Returning ``None`` is a real answer, not an error: it is how the runtime
        distinguishes "this language is not installed" from "no results".
        """

        language = normalize_language(language)
        for pack in self.packs:
            if pack.capability == capability and language in pack.languages:
                return self.component(pack.components[0])
        return None

    def digests(self) -> set[str]:
        return {item.sha256 for item in self.components}

    def installed_languages(self, capability: str) -> tuple[str, ...]:
        seen: set[str] = set()
        for pack in self.packs:
            if pack.capability == capability:
                seen.update(pack.languages)
        return tuple(sorted(seen))

    def installed_size(self) -> int:
        return sum(item.size for item in self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "activation_id": self.activation_id,
            "dataset_version": self.dataset_version,
            "manifest_sha256": self.manifest_sha256,
            "created_at": self.created_at,
            "requested": {
                "languages": (
                    None
                    if self.requested_languages is None
                    else list(self.requested_languages)
                ),
                "capabilities": list(self.requested_capabilities),
            },
            "effective": {
                capability: list(languages)
                for capability, languages in sorted(self.effective.items())
            },
            "unavailable": [dict(item) for item in self.unavailable],
            "components": [
                {
                    "id": item.id,
                    "sha256": item.sha256,
                    "path": item.path,
                    "size": item.size,
                }
                for item in self.components
            ],
            "packs": [
                {
                    "id": pack.id,
                    "capability": pack.capability,
                    "languages": list(pack.languages),
                    "components": list(pack.components),
                }
                for pack in self.packs
            ],
        }


def compute_activation_id(
    dataset_version: str, manifest_sha256: str, component_digests: Sequence[str]
) -> str:
    """Derive a stable id from what the activation actually contains.

    Deterministic on purpose: re-requesting the same selection resolves to the
    same id, so an install that changes nothing writes nothing.
    """

    digest = hashlib.sha256()
    digest.update(dataset_version.encode())
    digest.update(b"\0")
    digest.update(manifest_sha256.encode())
    for component in sorted(set(component_digests)):
        digest.update(b"\0")
        digest.update(component.encode())
    return digest.hexdigest()[:32]


def build_activation(
    manifest: DatasetManifest,
    selection: Selection,
    *,
    requested_languages: Sequence[str] | None,
    requested_capabilities: Sequence[str],
    created_at: str,
) -> Activation:
    """Turn a resolved selection into the record that will be activated."""

    components = tuple(
        ActivationComponent(
            id=component.id,
            sha256=component.final_sha256,
            path=str(component.path),
            size=component.final_size,
        )
        for component in (
            manifest.component(component_id) for component_id in selection.components
        )
    )
    selected_packs = set(selection.packs)
    packs = tuple(
        ActivationPack(
            id=pack.id,
            capability=pack.capability,
            languages=pack.languages,
            components=pack.components,
        )
        for pack in manifest.packs
        if pack.id in selected_packs
    )
    return Activation(
        schema_version=ACTIVATION_SCHEMA_VERSION,
        activation_id=compute_activation_id(
            manifest.dataset_version,
            manifest.sha256,
            [component.sha256 for component in components],
        ),
        dataset_version=manifest.dataset_version,
        manifest_sha256=manifest.sha256,
        created_at=created_at,
        requested_languages=(
            None if requested_languages is None else tuple(requested_languages)
        ),
        requested_capabilities=tuple(requested_capabilities),
        effective=dict(selection.effective),
        unavailable=tuple(item.as_dict() for item in selection.unavailable),
        components=components,
        packs=packs,
    )


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActivationError(f"{field} must be a non-empty string")
    return value


def parse_activation(raw: bytes | str | Mapping[str, Any]) -> Activation:
    """Parse and validate an activation record read back from disk."""

    if isinstance(raw, (bytes, str)):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ActivationError(f"activation is not valid JSON: {exc}") from exc
    else:
        value = dict(raw)
    if not isinstance(value, dict):
        raise ActivationError("activation must be an object")
    if value.get("schema_version") != ACTIVATION_SCHEMA_VERSION:
        raise ActivationError(
            f"unsupported activation schema {value.get('schema_version')!r}"
        )

    components: list[ActivationComponent] = []
    component_ids: set[str] = set()
    raw_components = value.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ActivationError("activation must record at least one component")
    for index, item in enumerate(raw_components):
        field = f"components[{index}]"
        if not isinstance(item, dict):
            raise ActivationError(f"{field} must be an object")
        component_id = safe_identifier(item.get("id"), field=f"{field}.id")
        if component_id in component_ids:
            raise ActivationError(f"duplicate activation component: {component_id}")
        component_ids.add(component_id)
        sha256 = item.get("sha256")
        if not is_sha256(sha256):
            raise ActivationError(f"{field}.sha256 must be a lowercase SHA-256 digest")
        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ActivationError(f"{field}.size must be a non-negative integer")
        components.append(
            ActivationComponent(
                id=component_id,
                sha256=str(sha256),
                path=_string(item.get("path"), f"{field}.path"),
                size=size,
            )
        )

    packs: list[ActivationPack] = []
    raw_packs = value.get("packs")
    if not isinstance(raw_packs, list) or not raw_packs:
        raise ActivationError("activation must record at least one pack")
    for index, item in enumerate(raw_packs):
        field = f"packs[{index}]"
        if not isinstance(item, dict):
            raise ActivationError(f"{field} must be an object")
        raw_components = item.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise ActivationError(f"{field}.components must be a non-empty array")
        pack_components: list[str] = []
        for index, entry in enumerate(raw_components):
            component = safe_identifier(entry, field=f"{field}.components[{index}]")
            if component not in component_ids:
                raise ActivationError(
                    f"{field} references uninstalled component {component!r}"
                )
            pack_components.append(component)
        languages = item.get("languages")
        if not isinstance(languages, list):
            raise ActivationError(f"{field}.languages must be an array")
        packs.append(
            ActivationPack(
                id=safe_identifier(item.get("id"), field=f"{field}.id"),
                capability=_string(item.get("capability"), f"{field}.capability"),
                languages=tuple(
                    normalize_language(language, field=f"{field}.languages")
                    for language in languages
                ),
                components=tuple(pack_components),
            )
        )

    requested = value.get("requested")
    if not isinstance(requested, dict):
        raise ActivationError("activation must record what was requested")
    requested_languages = requested.get("languages")
    if requested_languages is not None and not isinstance(requested_languages, list):
        raise ActivationError("requested.languages must be an array or null")
    capabilities = requested.get("capabilities")
    if not isinstance(capabilities, list):
        raise ActivationError("requested.capabilities must be an array")

    effective_value = value.get("effective", {})
    if not isinstance(effective_value, dict):
        raise ActivationError("effective must be an object")

    unavailable_value = value.get("unavailable", [])
    if not isinstance(unavailable_value, list):
        raise ActivationError("unavailable must be an array")

    activation = Activation(
        schema_version=ACTIVATION_SCHEMA_VERSION,
        activation_id=safe_identifier(value.get("activation_id"), field="activation_id"),
        dataset_version=_string(value.get("dataset_version"), "dataset_version"),
        manifest_sha256=(
            str(value.get("manifest_sha256"))
            if is_sha256(value.get("manifest_sha256"))
            else ""
        ),
        created_at=_string(value.get("created_at"), "created_at"),
        requested_languages=(
            None if requested_languages is None else tuple(requested_languages)
        ),
        requested_capabilities=tuple(str(item) for item in capabilities),
        effective={
            str(capability): tuple(str(language) for language in languages)
            for capability, languages in effective_value.items()
        },
        unavailable=tuple(
            {str(key): str(item[key]) for key in item}
            for item in unavailable_value
            if isinstance(item, dict)
        ),
        components=tuple(components),
        packs=tuple(packs),
    )
    if not activation.manifest_sha256:
        raise ActivationError("manifest_sha256 must be a lowercase SHA-256 digest")
    return activation
