"""Calibrate i8 USearch candidate recall against exact float cosine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from usearch.index import Index

from lexicon_mcp.pipeline.ann_calibration import candidate_recall


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--vectors", type=Path, required=True, help="row-major float16 vectors")
    value.add_argument("--index", type=Path, required=True, help="USearch i8/cos index")
    value.add_argument("--dimensions", type=int, default=300)
    value.add_argument("--queries", type=int, default=100)
    value.add_argument("--k", type=int, default=20)
    value.add_argument("--fetch", type=int, default=80)
    value.add_argument("--expansion-search", type=int, default=512)
    value.add_argument("--seed", type=int, default=1908)
    return value


def main() -> None:
    args = parser().parse_args()
    size = args.vectors.stat().st_size
    row_bytes = args.dimensions * np.dtype("<f2").itemsize
    if size == 0 or size % row_bytes:
        raise ValueError("vector file size is not divisible by the configured row size")
    count = size // row_bytes
    vectors: np.ndarray = np.memmap(
        args.vectors,
        dtype="<f2",
        mode="r",
        shape=(count, args.dimensions),
    )
    index = Index.restore(args.index, view=True)
    index.expansion_search = args.expansion_search
    recall = candidate_recall(
        vectors,
        index,
        queries=args.queries,
        k=args.k,
        fetch=args.fetch,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "vectors": count,
                "dimensions": args.dimensions,
                "queries": len(recall),
                "k": args.k,
                "fetch": args.fetch,
                "expansion_search": args.expansion_search,
                "mean_recall": float(np.mean(recall)),
                "p10_recall": float(np.percentile(recall, 10)),
                "min_recall": min(recall),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
