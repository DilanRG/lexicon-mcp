from __future__ import annotations

from lexicon_mcp.data_cli import _manifest_source, build_parser


def test_install_cli_accepts_english_profile() -> None:
    args = build_parser().parse_args(
        ["install", "--profile", "english", "--version", "data-en-v1.0.0"]
    )

    assert args.profile == "english"
    assert args.version == "data-en-v1.0.0"


def test_manifest_template_can_select_profile_specific_asset() -> None:
    source = _manifest_source(
        "https://example.invalid/{version}/{profile}/manifest.json",
        version="data-en-v1.0.0",
        profile="english",
    )

    assert source == "https://example.invalid/data-en-v1.0.0/english/manifest.json"
