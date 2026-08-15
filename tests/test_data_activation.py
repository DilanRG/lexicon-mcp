"""Activation records: the resolved identity of an install."""

from __future__ import annotations

import json

import pytest
from test_data_selection import manifest_bytes

from lexicon_mcp.data.activation import (
    ActivationError,
    build_activation,
    compute_activation_id,
    parse_activation,
)
from lexicon_mcp.data.manifest import parse_manifest
from lexicon_mcp.data.selection import resolve

CREATED = "2026-08-16T00:00:00Z"


def activation_for(languages: list[str] | None, capabilities: list[str]):
    manifest = parse_manifest(manifest_bytes())
    selection = resolve(manifest, languages=languages, capabilities=capabilities)
    return build_activation(
        manifest,
        selection,
        requested_languages=languages,
        requested_capabilities=capabilities,
        created_at=CREATED,
    )


def test_activation_records_what_was_resolved_not_just_what_was_asked() -> None:
    activation = activation_for(["en", "cy"], ["lexical", "semantic"])

    assert activation.requested_languages == ("en", "cy")
    assert activation.effective["lexical"] == ("cy", "en")
    assert activation.effective["semantic"] == ("en",)
    # Welsh has no upstream semantic vectors; that is recorded, not silently lost.
    assert activation.unavailable[0]["language"] == "cy"
    assert activation.unavailable[0]["capability"] == "semantic"


def test_activation_routes_a_language_to_its_component() -> None:
    activation = activation_for(["en", "cy"], ["lexical", "semantic"])

    assert activation.component_for("lexical", "en").id == "artifact-lexical-en"
    assert activation.component_for("lexical", "cy").id == "artifact-lexical-bundle"
    assert activation.component_for("semantic", "en").id == "artifact-semantic-en"


def test_uninstalled_language_routes_to_nothing_rather_than_failing() -> None:
    """None is the answer that lets the runtime say 'not installed'."""

    activation = activation_for(["en"], ["lexical"])

    assert activation.component_for("lexical", "gv") is None
    assert activation.component_for("semantic", "en") is None


def test_routing_normalizes_the_requested_tag() -> None:
    activation = activation_for(["en"], ["lexical"])

    assert activation.component_for("lexical", "EN").id == "artifact-lexical-en"


def test_the_same_selection_always_yields_the_same_activation_id() -> None:
    first = activation_for(["en"], ["lexical"])
    second = activation_for(["en"], ["lexical"])

    assert first.activation_id == second.activation_id

    wider = activation_for(["en", "cy"], ["lexical"])
    assert wider.activation_id != first.activation_id


def test_activation_id_depends_only_on_content() -> None:
    assert compute_activation_id("v1", "a" * 64, ["b" * 64, "c" * 64]) == (
        compute_activation_id("v1", "a" * 64, ["c" * 64, "b" * 64, "c" * 64])
    )


def test_installed_languages_and_size_come_from_the_resolved_packs() -> None:
    activation = activation_for(["en", "cy", "gv"], ["lexical"])

    assert activation.installed_languages("lexical") == ("cy", "en", "gv")
    assert activation.installed_size() == 2310


def test_a_full_install_records_that_every_language_was_requested() -> None:
    activation = activation_for(None, ["lexical"])

    assert activation.requested_languages is None
    assert activation.effective["lexical"] == ("cy", "en", "gv")


def test_activation_round_trips_through_json() -> None:
    activation = activation_for(["en"], ["lexical", "semantic"])

    restored = parse_activation(json.dumps(activation.to_dict()))

    assert restored == activation


def test_parsing_rejects_a_record_that_routes_to_a_missing_component() -> None:
    activation = activation_for(["en"], ["lexical"])
    payload = activation.to_dict()
    payload["packs"][-1]["components"] = ["artifact-not-installed"]

    with pytest.raises(ActivationError, match="uninstalled component"):
        parse_activation(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(schema_version=1), "unsupported activation schema"),
        (lambda p: p.update(components=[]), "at least one component"),
        (lambda p: p.update(packs=[]), "at least one pack"),
        (lambda p: p.update(manifest_sha256="nope"), "SHA-256 digest"),
        (lambda p: p["components"][0].update(sha256="nope"), "SHA-256 digest"),
        (lambda p: p["components"][0].update(size=-1), "non-negative integer"),
        (lambda p: p.pop("requested"), "what was requested"),
    ],
)
def test_malformed_activation_records_are_rejected(mutate, message: str) -> None:
    payload = activation_for(["en"], ["lexical"]).to_dict()
    mutate(payload)

    with pytest.raises(ActivationError, match=message):
        parse_activation(json.dumps(payload))
