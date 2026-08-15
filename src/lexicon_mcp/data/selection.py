"""Resolve a requested language and capability set to release components.

The user asks for languages and capabilities; that is intent.  The installable
identity is the exact component set this module resolves it to, which is what
gets recorded in the activation record and what verification later checks.

Selection is deliberately total: a request that cannot be fully satisfied still
produces a selection plus an explicit list of what could not be served and why.
Silently installing less than was asked for, or reporting "no results" later for
data that was never installed, is the failure mode this exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .manifest import DatasetManifest, ManifestError, Pack, normalize_language

# Capabilities a caller may request.  "core" is always installed and is not
# requestable, so it is absent here.
REQUESTABLE_CAPABILITIES = ("lexical", "semantic", "pronunciation", "wordplay")

# Reasons a requested (capability, language) pair could not be served.  These
# are contract strings: the CLI and the runtime both surface them verbatim, so
# an operator and a model see the same distinction.
LANGUAGE_NOT_IN_DATASET = "language_not_in_dataset"
CAPABILITY_NOT_AVAILABLE_FOR_LANGUAGE = "capability_not_available_for_language"


class SelectionError(ValueError):
    """A selection request could not be resolved against the manifest."""


@dataclass(frozen=True, slots=True)
class Unavailable:
    capability: str
    language: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "capability": self.capability,
            "language": self.language,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Selection:
    """A fully resolved install request."""

    components: tuple[str, ...]
    packs: tuple[str, ...]
    effective: Mapping[str, tuple[str, ...]]
    unavailable: tuple[Unavailable, ...]
    compressed_size: int
    installed_size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "components": list(self.components),
            "packs": list(self.packs),
            "effective": {
                capability: list(languages)
                for capability, languages in sorted(self.effective.items())
            },
            "unavailable": [item.as_dict() for item in self.unavailable],
            "compressed_size": self.compressed_size,
            "installed_size": self.installed_size,
        }


def _normalized_request(languages: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for language in languages:
        try:
            normalized = normalize_language(language, field="requested language")
        except ManifestError as exc:
            raise SelectionError(str(exc)) from exc
        if normalized not in seen:
            seen.append(normalized)
    return tuple(seen)


def resolve(
    manifest: DatasetManifest,
    *,
    languages: Sequence[str] | None,
    capabilities: Sequence[str] = ("lexical",),
    strict: bool = True,
) -> Selection:
    """Resolve *languages* and *capabilities* to a component set.

    ``languages=None`` selects every language the dataset offers for each
    requested capability, which is how a full install is expressed.

    With ``strict`` (the default), a requested language that the dataset does not
    carry at all is an error rather than a silent omission -- a typo must not
    quietly install nothing.  A language the dataset carries but which lacks the
    requested capability, such as a language with no upstream semantic vectors,
    is never an error: it is reported through ``unavailable`` so the caller can
    say so precisely.
    """

    if manifest.schema_version < 2:
        raise SelectionError(
            "component selection requires a schema 2 manifest; "
            f"this release is schema {manifest.schema_version}"
        )
    for capability in capabilities:
        if capability not in REQUESTABLE_CAPABILITIES:
            raise SelectionError(
                f"unknown capability {capability!r}; "
                f"expected one of {list(REQUESTABLE_CAPABILITIES)}"
            )

    known_languages = set(manifest.languages)
    requested = None if languages is None else _normalized_request(languages)
    if requested is not None and strict:
        missing = [language for language in requested if language not in known_languages]
        if missing:
            raise SelectionError(
                "dataset does not contain these languages: " + ", ".join(sorted(missing))
            )

    selected: list[Pack] = list(manifest.required_packs())
    effective: dict[str, tuple[str, ...]] = {}
    unavailable: list[Unavailable] = []

    for capability in capabilities:
        wanted = (
            manifest.languages_for(capability) if requested is None else requested
        )
        covered: list[str] = []
        for language in wanted:
            pack = manifest.pack_for(capability, language)
            if pack is None:
                unavailable.append(
                    Unavailable(
                        capability=capability,
                        language=language,
                        reason=(
                            LANGUAGE_NOT_IN_DATASET
                            if language not in known_languages
                            else CAPABILITY_NOT_AVAILABLE_FOR_LANGUAGE
                        ),
                    )
                )
                continue
            covered.append(language)
            if pack not in selected:
                selected.append(pack)
        effective[capability] = tuple(sorted(covered))

    # A pack may serve several requested languages, and several packs may share a
    # component; both collapse here so sizes are never double counted.
    component_ids: list[str] = []
    for pack in selected:
        if pack.component not in component_ids:
            component_ids.append(pack.component)

    compressed = 0
    installed = 0
    for component_id in component_ids:
        try:
            component = manifest.component(component_id)
        except KeyError as exc:  # guarded by manifest parsing
            raise SelectionError(
                f"manifest pack references unknown component {component_id!r}"
            ) from exc
        compressed += component.compressed_size
        installed += component.final_size

    return Selection(
        components=tuple(sorted(component_ids)),
        packs=tuple(sorted(pack.id for pack in selected)),
        effective=effective,
        unavailable=tuple(unavailable),
        compressed_size=compressed,
        installed_size=installed,
    )
