"""The schema-2 command surface: language selection, amendment, pruning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from release_fixture import PAYLOADS, VERSION

from lexicon_mcp.data_cli import build_parser, main, run


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_languages_and_all_languages_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse(
            ["install", "--version", VERSION, "--languages", "en", "--all-languages"]
        )


def test_language_and_capability_lists_are_validated() -> None:
    args = parse(
        [
            "install",
            "--version",
            VERSION,
            "--languages",
            "en, fr",
            "--capabilities",
            "lexical,semantic",
        ]
    )

    assert args.languages == ["en", "fr"]
    assert args.capabilities == ["lexical", "semantic"]

    with pytest.raises(SystemExit):
        parse(["install", "--version", VERSION, "--capabilities", "etymology"])

    with pytest.raises(SystemExit):
        parse(["install", "--version", VERSION, "--languages", "en,,fr"])


def test_capabilities_default_to_lexical() -> None:
    assert parse(["install", "--version", VERSION, "--languages", "en"]).capabilities == [
        "lexical"
    ]


def test_installing_a_schema_two_release_selects_languages(
    tmp_path: Path, release: Path
) -> None:
    data = tmp_path / "data"

    code = main(
        [
            "--data-dir",
            str(data),
            "install",
            "--version",
            VERSION,
            "--from",
            str(release),
            "--languages",
            "en",
        ]
    )

    assert code == 0
    pointer = json.loads((data / "current.json").read_text(encoding="utf-8"))
    assert pointer["schema_version"] == 2
    assert main(["--data-dir", str(data), "verify"]) == 0


def test_adding_and_removing_languages_through_the_cli(
    tmp_path: Path, release: Path
) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data)]
    main([*common, "install", "--version", VERSION, "--from", str(release), "--languages", "en"])

    assert (
        main(
            [
                *common,
                "add-language",
                "--version",
                VERSION,
                "--from",
                str(release),
                "--languages",
                "fr",
            ]
        )
        == 0
    )
    result, _code = run(
        build_parser().parse_args([*common, "status"])
    )
    assert result["current"]["effective"]["lexical"] == ["en", "fr"]

    assert (
        main(
            [
                *common,
                "remove-language",
                "--version",
                VERSION,
                "--from",
                str(release),
                "--languages",
                "fr",
            ]
        )
        == 0
    )
    result, _code = run(build_parser().parse_args([*common, "status"]))
    assert result["current"]["effective"]["lexical"] == ["en"]


def test_prune_reclaims_space_after_forgetting_an_activation(
    tmp_path: Path, release: Path
) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data)]
    main(
        [
            *common,
            "install",
            "--version",
            VERSION,
            "--from",
            str(release),
            "--languages",
            "en,fr",
        ]
    )
    wide, _ = run(build_parser().parse_args([*common, "status"]))
    wide_id = wide["current"]["activation_id"]
    main(
        [
            *common,
            "remove-language",
            "--version",
            VERSION,
            "--from",
            str(release),
            "--languages",
            "fr",
        ]
    )

    assert main([*common, "forget", "--activation", wide_id]) == 0
    result, _code = run(build_parser().parse_args([*common, "prune"]))

    assert len(result["removed"]) == 1
    assert main([*common, "verify"]) == 0


def test_activate_switches_back_to_a_retained_selection(
    tmp_path: Path, release: Path
) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data)]
    main([*common, "install", "--version", VERSION, "--from", str(release), "--languages", "en"])
    first, _ = run(build_parser().parse_args([*common, "status"]))
    first_id = first["current"]["activation_id"]
    main(
        [
            *common,
            "add-language",
            "--version",
            VERSION,
            "--from",
            str(release),
            "--languages",
            "fr",
        ]
    )

    assert main([*common, "activate", "--activation", first_id]) == 0

    result, _code = run(build_parser().parse_args([*common, "status"]))
    assert result["current"]["activation_id"] == first_id
    assert result["current"]["effective"]["lexical"] == ["en"]


def test_profile_is_refused_for_a_schema_two_release(tmp_path: Path, release: Path) -> None:
    code = main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "install",
            "--version",
            VERSION,
            "--from",
            str(release),
            "--profile",
            "full",
        ]
    )

    assert code == 2


def test_a_schema_two_release_requires_an_explicit_selection(
    tmp_path: Path, release: Path
) -> None:
    """Installing 23 GB by accident should not be the default."""

    code = main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "install",
            "--version",
            VERSION,
            "--from",
            str(release),
        ]
    )

    assert code == 2


def test_all_languages_installs_everything(tmp_path: Path, release: Path) -> None:
    data = tmp_path / "data"

    code = main(
        [
            "--data-dir",
            str(data),
            "install",
            "--version",
            VERSION,
            "--from",
            str(release),
            "--all-languages",
            "--capabilities",
            "lexical,semantic",
        ]
    )

    assert code == 0
    result, _code = run(build_parser().parse_args(["--data-dir", str(data), "status"]))
    assert result["current"]["effective"]["lexical"] == ["cy", "en", "fr", "gv"]
    assert result["current"]["effective"]["semantic"] == ["en"]
    assert result["current"]["components"] == len(PAYLOADS)


def test_languages_reports_what_this_install_actually_serves(
    tmp_path: Path, release: Path
) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data)]
    main([*common, "install", "--version", VERSION, "--from", str(release), "--languages", "en"])

    report, code = run(build_parser().parse_args([*common, "languages"]))

    assert code == 0
    assert report["dataset_languages"] == 5
    assert report["installed"]["lexical"] == ["en"]
    # Only the installed language is listed by default.
    assert [row["language"] for row in report["languages"]] == ["en"]
    english = report["languages"][0]
    assert english["term_count"] == 2_000_000
    assert english["capabilities"]["lexical"] == "installed"
    # English has vectors upstream, but the semantic pack was not selected.
    assert english["capabilities"]["semantic"] == "capability_not_installed"


def test_languages_distinguishes_every_reason_a_language_is_unavailable(
    tmp_path: Path, release: Path
) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data)]
    main([*common, "install", "--version", VERSION, "--from", str(release), "--languages", "en"])

    report, _code = run(
        build_parser().parse_args([*common, "languages", "--language", "de,cy,zz"])
    )

    rows = {row["language"]: row for row in report["languages"]}
    # In the corpus, not installed here: installing more would fix it.
    assert rows["de"]["capabilities"]["lexical"] == "language_not_installed"
    assert rows["de"]["in_dataset"] is True
    # Welsh exists but the corpus has no vectors for it at all.
    assert rows["cy"]["capabilities"]["semantic"] == "not_available_upstream"
    # Not a language this dataset carries.
    assert rows["zz"]["in_dataset"] is False
    assert rows["zz"]["capabilities"]["lexical"] == "unknown_language"


def test_languages_can_list_the_whole_dataset(tmp_path: Path, release: Path) -> None:
    data = tmp_path / "data"
    common = ["--data-dir", str(data)]
    main([*common, "install", "--version", VERSION, "--from", str(release), "--languages", "en"])

    report, _code = run(build_parser().parse_args([*common, "languages", "--all"]))

    assert [row["language"] for row in report["languages"]] == ["cy", "de", "en", "fr", "gv"]


def test_languages_parser_accepts_explicit_tags() -> None:
    args = build_parser().parse_args(["languages", "--language", "en,fr", "--all"])

    assert args.language == ["en", "fr"]
    assert args.all is True
