"""Apply or recover a provenance-checked notice-only dataset transaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lexicon_mcp.pipeline import amend_promoted_dataset_notices


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Promoted dataset; an interrupted notice transaction is recovered first",
    )
    value.add_argument("--dataset-version", required=True)
    value.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Clean lexicon-mcp repository supplying DATA_LICENSES.md and licenses/",
    )
    value.add_argument(
        "--amendment-commit",
        required=True,
        help="Current clean 40-hex Git HEAD containing the notice amendment",
    )
    value.add_argument("--reason", required=True, help="Auditable reason for the amendment")
    value.add_argument("--expected-original-build-commit", required=True)
    value.add_argument("--expected-recovery-commit", required=True)
    value.add_argument("--expected-lexicon-sha256", required=True)
    value.add_argument("--expected-global-index-sha256", required=True)
    value.add_argument("--expected-global-vectors-sha256", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    result = amend_promoted_dataset_notices(
        args.dataset_root,
        args.repository,
        dataset_version=args.dataset_version,
        amendment_commit=args.amendment_commit,
        reason=args.reason,
        expected_original_build_commit=args.expected_original_build_commit,
        expected_recovery_commit=args.expected_recovery_commit,
        expected_lexicon_sha256=args.expected_lexicon_sha256,
        expected_global_index_sha256=args.expected_global_index_sha256,
        expected_global_vectors_sha256=args.expected_global_vectors_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
