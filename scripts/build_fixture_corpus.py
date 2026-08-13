"""Build the repository's tiny but structurally real acceptance corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from lexicon_mcp.pipeline import BuildInputs, build_full_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("tests/fixtures/build_inputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-state", type=Path, required=True)
    args = parser.parse_args()
    inputs = args.inputs
    build_full_corpus(
        BuildInputs(
            oewn=inputs / "oewn.xml",
            wiktextract=(inputs / "kaikki.jsonl",),
            conceptnet=inputs / "conceptnet.tsv",
            numberbatch=inputs / "numberbatch.txt",
            cmudict=inputs / "cmudict.dict",
            notices_dir=inputs / "notices",
        ),
        args.output,
        args.build_state,
        dataset_version="fixture-v1",
        retrieved_at="2026-01-01T00:00:00Z",
    )


if __name__ == "__main__":
    main()
