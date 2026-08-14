"""Conservative installed-size projection for the pinned full corpus.

The rates are deliberately transparent and are calibrated from real source
samples and USearch 2.26 artifacts.  They are a release preflight, not a
replacement for reporting the final artifact sizes in the release manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3
INSTALLED_LIMIT = 30 * GIB
PEAK_BUILD_LIMIT = 80 * GIB


@dataclass(frozen=True)
class FullCorpusShape:
    """Known physical/source-feature counts for the pinned v1 corpus."""

    wiktextract_entries: int = 10_806_865
    wiktextract_senses: int = 12_999_262
    wiktextract_examples: int = 1_535_987
    wiktextract_translations: int = 3_564_851
    # Raw source-list items, retained only as a conservative sizing input. The
    # importer stores 3,688,516 rows after source-sense-scoped duplicate,
    # self-link, and invalid-term filtering; never use this raw count as a
    # physical corpus floor.
    wiktextract_synonyms: int = 7_481_661
    wiktextract_antonyms: int = 164_003
    # Sound objects include audio/rhyme metadata; the builder stores only the
    # measured 6,870,610 rows carrying non-empty IPA text.
    wiktextract_pronunciations: int = 6_870_610
    conceptnet_assertions: int = 18_501_416
    numberbatch_terms: int = 9_161_912
    numberbatch_dimensions: int = 300
    oewn_rows: int = 1_120_304
    cmudict_entries: int = 135_166


@dataclass(frozen=True)
class SizeRates:
    """Conservative bytes/row rates from compact-v2 sampling.

    SQLite rates include production query indexes and a 2.5x safety factor
    over the initial compact-v2 Wiktextract feature model. ConceptNet uses its
    measured 148 B/accepted-row sample plus margin. Semantic mapping and HNSW
    rates conservatively exceed real 100k/200k Numberbatch samples; both global
    and per-language HNSW totals contain every accepted vector once.
    """

    wiktextract_entry: float = 250.0
    wiktextract_sense: float = 250.0
    wiktextract_example: float = 350.0
    wiktextract_link: float = 162.5
    wiktextract_pronunciation: float = 175.0
    conceptnet_assertion: float = 180.0
    oewn_row: float = 90.0
    cmudict_entry: float = 250.0
    semantic_mapping: float = 210.0
    hnsw_i8: float = 470.0


def estimate_installed_bytes(
    shape: FullCorpusShape | None = None,
    rates: SizeRates | None = None,
) -> dict[str, int]:
    """Return a deterministic component projection in bytes."""

    shape = shape or FullCorpusShape()
    rates = rates or SizeRates()

    wiktextract = (
        shape.wiktextract_entries * rates.wiktextract_entry
        + shape.wiktextract_senses * rates.wiktextract_sense
        + shape.wiktextract_examples * rates.wiktextract_example
        + (
            shape.wiktextract_translations
            + shape.wiktextract_synonyms
            + shape.wiktextract_antonyms
        )
        * rates.wiktextract_link
        + shape.wiktextract_pronunciations * rates.wiktextract_pronunciation
    )
    lexical = (
        wiktextract
        + shape.conceptnet_assertions * rates.conceptnet_assertion
        + shape.oewn_rows * rates.oewn_row
        + shape.cmudict_entries * rates.cmudict_entry
    )
    vector_bytes = shape.numberbatch_terms * shape.numberbatch_dimensions * 2
    mapping = shape.numberbatch_terms * rates.semantic_mapping
    global_hnsw = shape.numberbatch_terms * rates.hnsw_i8
    language_hnsw = global_hnsw
    semantic = vector_bytes + mapping + global_hnsw + language_hnsw
    total = lexical + semantic
    return {
        "lexicon.sqlite3": round(lexical),
        "semantic/mapping.sqlite3": round(mapping),
        "semantic/vectors/global.f16": vector_bytes,
        "semantic/indexes/global.usearch": round(global_hnsw),
        "semantic/indexes/languages": round(language_hnsw),
        "total": round(total),
    }


def estimate_peak_build_bytes(
    installed_bytes: int,
    *,
    compressed_sources: int = 6_530_144_980,
    packaging_workspace: int | None = None,
    sqlite_wal_and_temp: int = 12 * GIB,
) -> int:
    """Conservative peak budget retaining sources and all streamed release parts."""

    if packaging_workspace is None:
        # The public package retains all zstd parts. In the worst useful case
        # they are approximately the installed bytes plus framing overhead;
        # one GiB is deliberately more margin than the per-frame overhead.
        packaging_workspace = installed_bytes + GIB
    return installed_bytes + compressed_sources + packaging_workspace + sqlite_wal_and_temp


def assert_size_targets(
    shape: FullCorpusShape | None = None,
    rates: SizeRates | None = None,
) -> dict[str, int]:
    """Raise before a release build when calibrated projections exceed gates."""

    estimate = estimate_installed_bytes(shape, rates)
    peak = estimate_peak_build_bytes(estimate["total"])
    if estimate["total"] > INSTALLED_LIMIT:
        raise RuntimeError(
            f"projected installed size {estimate['total']} exceeds {INSTALLED_LIMIT} bytes"
        )
    if peak > PEAK_BUILD_LIMIT:
        raise RuntimeError(f"projected peak build size {peak} exceeds {PEAK_BUILD_LIMIT} bytes")
    return {**estimate, "peak_build": peak}
