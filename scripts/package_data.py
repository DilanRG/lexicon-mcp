"""Split a verified dataset tree into GitHub Release parts and manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lexicon_mcp.pipeline.packaging import package_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", default="data-v1.0.0")
    parser.add_argument("--repository", default="DilanRG/lexicon-mcp")
    parser.add_argument("--tag", default="data-v1.0.0")
    parser.add_argument("--transformation-commit", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--max-part-size", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    result = package_dataset(
        args.dataset,
        args.output,
        dataset_version=args.dataset_version,
        repository=args.repository,
        tag=args.tag,
        transformation_commit=args.transformation_commit,
        base_url=args.base_url,
        max_part_size=args.max_part_size,
        created_at=args.created_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
