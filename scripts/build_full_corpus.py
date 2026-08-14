"""Build the pinned full Lexicon MCP corpus from already-downloaded sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lexicon_mcp.pipeline import BuildInputs, build_full_corpus


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--oewn", type=Path, required=True, help="OEWN 2025 WN-LMF XML[.gz]")
    value.add_argument(
        "--wiktextract",
        type=Path,
        action="append",
        required=True,
        help="Kaikki Wiktextract JSONL[.gz/.zst]; repeat for multiple shards",
    )
    value.add_argument("--conceptnet", type=Path, required=True, help="ConceptNet 5.7 TSV[.gz]")
    value.add_argument(
        "--numberbatch", type=Path, required=True, help="Numberbatch 19.08 text[.gz]"
    )
    value.add_argument("--cmudict", type=Path, required=True, help="Pinned cmudict.dict")
    value.add_argument(
        "--source-lock",
        type=Path,
        required=True,
        help="Pinned schema-v1 sources.lock.json; every source is hash-verified before build",
    )
    value.add_argument(
        "--notices-dir",
        type=Path,
        required=True,
        help="Directory containing DATA_LICENSES.md and exact licenses/ texts",
    )
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--build-state", type=Path, required=True)
    value.add_argument("--dataset-version", default="data-v1.0.0")
    value.add_argument("--retrieved-at", help="RFC 3339 timestamp recorded for every source")
    return value


def main() -> None:
    args = parser().parse_args()
    result = build_full_corpus(
        BuildInputs(
            oewn=args.oewn,
            wiktextract=tuple(args.wiktextract),
            conceptnet=args.conceptnet,
            numberbatch=args.numberbatch,
            cmudict=args.cmudict,
            source_lock=args.source_lock,
            notices_dir=args.notices_dir,
        ),
        args.output,
        args.build_state,
        dataset_version=args.dataset_version,
        retrieved_at=args.retrieved_at,
        enforce_corpus_floors=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
