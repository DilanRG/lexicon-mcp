"""Build a schema-2 release by repartitioning an installed schema-1 corpus.

This is a transform, not a rebuild: lexical rows are copied verbatim out of a
verified corpus, so a pack's contents are bit-identical to the monolith's and
the two can be compared directly by the differential release gate.

Expensive corpus-wide inputs -- the language census and the full-corpus term
counts -- are computed once and cached, because every pack reuses them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from lexicon_mcp.pipeline.packs import (
    DEFAULT_BUNDLE_TARGET,
    DEFAULT_INDIVIDUAL_THRESHOLD,
    LanguageSize,
    PlannedPack,
    plan_lexical_packs,
)
from lexicon_mcp.pipeline.release import load_sources, package_packs
from lexicon_mcp.pipeline.transform import (
    PackResult,
    build_core_pack,
    build_lexical_pack,
    build_term_counts,
    language_sizes,
)


def semantic_languages(dataset_root: Path) -> list[str]:
    """Languages carrying Numberbatch vectors, read from the semantic mapping."""

    mapping = dataset_root / "semantic" / "mapping.sqlite3"
    if not mapping.is_file():
        return []
    connection = sqlite3.connect(f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT language FROM semantic_languages ORDER BY language"
            )
        ]
    finally:
        connection.close()


def cached_language_sizes(source: Path, work: Path) -> tuple[LanguageSize, ...]:
    cache = work / "language-sizes.json"
    if cache.is_file():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        return tuple(LanguageSize(row["language"], row["terms"]) for row in rows)
    sizes = language_sizes(source)
    cache.write_text(
        json.dumps([{"language": item.language, "terms": item.terms} for item in sizes], indent=2),
        encoding="utf-8",
    )
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="installed schema-1 dataset root"
    )
    parser.add_argument(
        "--work", type=Path, required=True, help="scratch directory for caches and packs"
    )
    parser.add_argument("--output", type=Path, required=True, help="release directory to write")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--repository", default="DilanRG/lexicon-mcp")
    parser.add_argument("--transformation-commit", required=True)
    parser.add_argument(
        "--only",
        nargs="*",
        help="build only these pack ids (default: every planned pack)",
    )
    parser.add_argument("--individual-threshold", type=int, default=DEFAULT_INDIVIDUAL_THRESHOLD)
    parser.add_argument("--bundle-target", type=int, default=DEFAULT_BUNDLE_TARGET)
    parser.add_argument(
        "--plan-only", action="store_true", help="print the pack plan and stop"
    )
    parser.add_argument(
        "--skip-package", action="store_true", help="build packs but do not compress a release"
    )
    args = parser.parse_args()

    source = args.dataset / "lexicon.sqlite3"
    if not source.is_file():
        raise SystemExit(f"no lexical corpus at {source}")
    args.work.mkdir(parents=True, exist_ok=True)

    sizes = cached_language_sizes(source, args.work)
    plan = plan_lexical_packs(
        sizes,
        individual_threshold=args.individual_threshold,
        bundle_target=args.bundle_target,
    )
    individual = [pack for pack in plan if not pack.bundled]
    bundles = [pack for pack in plan if pack.bundled]
    print(
        f"planned {len(plan)} lexical packs from {len(sizes):,} languages: "
        f"{len(individual)} individual, {len(bundles)} bundled",
        flush=True,
    )
    if args.plan_only:
        for pack in plan:
            print(
                f"  {pack.id:<24} {len(pack.languages):>5} languages"
                f"  ~{pack.estimated_compressed / 1024**2:>8,.1f} MiB"
            )
        return 0

    counts = args.work / "term-counts.sqlite3"
    if not counts.is_file():
        started = time.monotonic()
        print("materializing full-corpus term counts ...", flush=True)
        build_term_counts(source, counts)
        print(f"  done [{time.monotonic() - started:.1f}s]", flush=True)

    wanted = set(args.only) if args.only else None
    packs_dir = args.work / "packs"
    built: list[PackResult] = []

    core_plan = PlannedPack("core", "core", (), 0)
    if wanted is None or "core" in wanted:
        started = time.monotonic()
        print("building core catalogue ...", flush=True)
        core_path = build_core_pack(
            source,
            packs_dir / "core.sqlite3",
            dataset_version=args.dataset_version,
            semantic_languages=semantic_languages(args.dataset),
            pronunciation_languages=["en"],
            wordplay_languages=["en"],
        )
        built.append(
            PackResult(core_plan, core_path, core_path.stat().st_size, 0, 0, 0, 0, 0, 0)
        )
        print(
            f"  {core_path.stat().st_size / 1024**2:,.1f} MiB"
            f"  [{time.monotonic() - started:.1f}s]",
            flush=True,
        )

    for pack in plan:
        if wanted is not None and pack.id not in wanted:
            continue
        started = time.monotonic()
        print(f"building {pack.id} ({len(pack.languages)} languages) ...", flush=True)
        result = build_lexical_pack(
            source,
            counts,
            packs_dir / f"{pack.id}.sqlite3",
            pack,
            dataset_version=args.dataset_version,
        )
        built.append(result)
        print(
            f"  {result.raw_bytes / 1024**2:>10,.1f} MiB"
            f"  terms {result.terms:>9,}  stubs {result.stubs:>9,}"
            f"  relations {result.relations:>10,}"
            f"  [{time.monotonic() - started:.1f}s]",
            flush=True,
        )

    summary = args.work / "build-report.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "pack": item.pack.id,
                    "capability": item.pack.capability,
                    "languages": len(item.pack.languages),
                    "raw_bytes": item.raw_bytes,
                    "terms": item.terms,
                    "stubs": item.stubs,
                    "relations": item.relations,
                    "translations": item.translations,
                }
                for item in built
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary}", flush=True)

    if args.skip_package:
        return 0
    package_packs(
        built,
        args.output,
        dataset_version=args.dataset_version,
        repository=args.repository,
        tag=args.dataset_version,
        transformation_commit=args.transformation_commit,
        sources=load_sources(args.dataset),
        source_dataset={
            "dataset_version": json.loads(
                (args.dataset / "build-manifest.json").read_text(encoding="utf-8")
            )["dataset_version"],
            "manifest_sha256": hashlib.sha256(
                (args.dataset / "manifest.json").read_bytes()
            ).hexdigest(),
        },
    )
    print(f"packaged release into {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
