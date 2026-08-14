from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexicon_mcp.pipeline.source_rows import measure_source, verify_source_row_fields

FIXTURES = Path(__file__).parent / "fixtures" / "build_inputs"


@pytest.mark.parametrize(
    ("source_id", "filename", "row_count", "row_digest"),
    (
        (
            "oewn",
            "oewn.xml",
            30,
            "b39a8794b6089879050e260df9c79560c474ef1a6d83b0d125cd51deae996ac1",
        ),
        (
            "wiktextract",
            "kaikki.jsonl",
            11,
            "d85f91ffb2fc380f4cde6bdeea7cca5b5dfce5b7d7c5b2081392f2f0ad5732ec",
        ),
        (
            "conceptnet",
            "conceptnet.tsv",
            4,
            "a2efeb4cf7dc2ba0a06bb00e0bd21ffb55cb195eb4304125f8ef90daab9800b0",
        ),
        (
            "numberbatch",
            "numberbatch.txt",
            10,
            "aeb937768cea175728d1e89117511de1ebeb94c24271cd6b808e11f7581e8456",
        ),
        (
            "cmudict",
            "cmudict.dict",
            7,
            "98ba151cc47c2c56a2f8f79624c21758ea7f6c30f5acabf6c7ecc1fafcf1d859",
        ),
    ),
)
def test_fixture_logical_row_contracts_are_stable(
    source_id: str,
    filename: str,
    row_count: int,
    row_digest: str,
) -> None:
    measured = measure_source(source_id, FIXTURES / filename)
    assert measured.row_count == row_count
    assert measured.row_digest == row_digest


def test_wiktextract_digest_is_independent_of_json_layout(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"word":"café","lang_code":"fr","senses":[]}\n', encoding="utf-8")
    second.write_text(
        json.dumps(
            {"senses": [], "lang_code": "fr", "word": "café"},
            ensure_ascii=True,
            indent=2,
        ).replace("\n", " ")
        + "\n",
        encoding="utf-8",
    )
    assert measure_source("wiktextract", first) == measure_source("wiktextract", second)


def test_source_lock_row_fields_are_exactly_verified() -> None:
    path = FIXTURES / "cmudict.dict"
    measured = measure_source("cmudict", path)
    record = measured.lock_fields()
    assert verify_source_row_fields("cmudict", record, path) == measured
    record["row_count"] = measured.row_count + 1
    with pytest.raises(ValueError, match="row_count mismatch"):
        verify_source_row_fields("cmudict", record, path)
