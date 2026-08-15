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
    SemanticPackResult,
    build_core_pack,
    build_lexical_pack,
    build_semantic_pack,
    build_term_counts,
    build_wordplay_pack,
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


def package_existing(args, plan, packs_dir: Path) -> int:
    """Package packs already on disk, so a single rebuilt pack does not cost a
    full rebuild of the other 149."""

    built: list[PackResult] = []
    semantic_built: list[SemanticPackResult] = []

    core_path = packs_dir / "core.sqlite3"
    if not core_path.is_file():
        raise SystemExit(f"no core pack at {core_path}; run a build first")
    built.append(
        PackResult(PlannedPack("core", "core", (), 0), core_path,
                   core_path.stat().st_size, 0, 0, 0, 0, 0, 0)
    )
    wordplay_path = packs_dir / "wordplay-en.sqlite3"
    if wordplay_path.is_file():
        built.append(
            PackResult(PlannedPack("wordplay-en", "wordplay", ("en",), 0), wordplay_path,
                       wordplay_path.stat().st_size, 0, 0, 0, 0, 0, 0)
        )
    for pack in plan:
        path = packs_dir / f"{pack.id}.sqlite3"
        if not path.is_file():
            raise SystemExit(f"planned pack is missing: {path}")
        built.append(PackResult(pack, path, path.stat().st_size, 0, 0, 0, 0, 0, 0))

    for language in semantic_languages(args.dataset):
        directory = packs_dir / f"semantic-{language}"
        mapping = directory / "mapping.sqlite3"
        vectors = directory / "vectors.f16"
        index = directory / f"{language.replace('-', '_')}.usearch"
        if not (mapping.is_file() and vectors.is_file() and index.is_file()):
            raise SystemExit(f"semantic pack is incomplete: {directory}")
        connection = sqlite3.connect(f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            terms = int(
                connection.execute("SELECT COUNT(*) FROM semantic_terms").fetchone()[0]
            )
            dimensions = int(
                connection.execute(
                    "SELECT value FROM metadata WHERE key='dimensions'"
                ).fetchone()[0]
            )
        finally:
            connection.close()
        semantic_built.append(
            SemanticPackResult(language, mapping, vectors, index, terms, dimensions)
        )

    print(f"packaging {len(built)} packs and {len(semantic_built)} semantic packs", flush=True)
    package_packs(
        built,
        args.output,
        dataset_version=args.dataset_version,
        repository=args.repository,
        tag=args.dataset_version,
        transformation_commit=args.transformation_commit,
        semantic=semantic_built,
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
    parser.add_argument(
        "--skip-semantic", action="store_true", help="build lexical packs only"
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="package the packs already in --work without rebuilding them",
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
    if args.package_only:
        return package_existing(args, plan, packs_dir)
    built: list[PackResult] = []

    core_plan = PlannedPack("core", "core", (), 0)
    if wanted is None or "core" in wanted:
        started = time.monotonic()
        print("building core catalogue ...", flush=True)
        core_path = build_core_pack(
            source,
            counts,
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

    if wanted is None or "wordplay-en" in wanted:
        started = time.monotonic()
        print("building wordplay-en ...", flush=True)
        wordplay = build_wordplay_pack(
            source,
            packs_dir / "wordplay-en.sqlite3",
            dataset_version=args.dataset_version,
        )
        built.append(wordplay)
        print(
            f"  {wordplay.raw_bytes / 1024**2:>10,.1f} MiB  terms {wordplay.terms:>9,}"
            f"  [{time.monotonic() - started:.1f}s]",
            flush=True,
        )

    semantic_built: list[SemanticPackResult] = []
    if not args.skip_semantic:
        for language in semantic_languages(args.dataset):
            pack_id = f"semantic-{language}"
            if wanted is not None and pack_id not in wanted:
                continue
            started = time.monotonic()
            print(f"building {pack_id} ...", flush=True)
            result = build_semantic_pack(
                args.dataset / "semantic",
                packs_dir / pack_id,
                language,
                dataset_version=args.dataset_version,
            )
            semantic_built.append(result)
            total = (
                result.mapping.stat().st_size
                + result.vectors.stat().st_size
                + result.index.stat().st_size
            )
            print(
                f"  {total / 1024**2:>10,.1f} MiB  terms {result.terms:>9,}"
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
            ]
            + [
                {
                    "pack": f"semantic-{item.language}",
                    "capability": "semantic",
                    "languages": 1,
                    "raw_bytes": (
                        item.mapping.stat().st_size
                        + item.vectors.stat().st_size
                        + item.index.stat().st_size
                    ),
                    "terms": item.terms,
                    "stubs": 0,
                    "relations": 0,
                    "translations": 0,
                }
                for item in semantic_built
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
        semantic=semantic_built,
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
