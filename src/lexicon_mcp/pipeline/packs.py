"""Schema-2 pack planning and the physical schema of a lexical pack.

Packing is decided here and executed in :mod:`lexicon_mcp.pipeline.transform`.
Keeping the plan pure means tier assignment is testable without touching a
corpus, and reproducible: the same language sizes always yield the same packs
with the same identifiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MIB = 1024 * 1024

# A language gets its own pack once it is worth downloading alone.  Measured on
# data-v1.1.0, 5 MiB compressed lands near rank 50, which holds 5.8 MiB.
DEFAULT_INDIVIDUAL_THRESHOLD = 5 * MIB

# Bundles below the threshold accumulate to roughly this size.  Large enough
# that the pack count stays small, small enough that pulling one obscure
# language never costs much.
DEFAULT_BUNDLE_TARGET = 40 * MIB

# Compressed bytes per lexical term, calibrated from real packs built out of
# data-v1.1.0 at zstd level 10:
#
#   en   620.0 MiB / 1,985,802 terms = 327 B    (outlier: 5.1M target stubs)
#   fr   153.5 MiB / 1,548,392 terms = 104 B
#   r25   20.1 MiB /   129,098 terms = 163 B
#   r50    5.8 MiB /    51,906 terms = 117 B
#   b51+  69.3 MiB /   455,298 terms = 160 B
#
# This is only ever used to choose a tier.  A language misjudged near the
# threshold lands in a bundle instead of its own pack, which costs a slightly
# larger download and nothing else, so a single coefficient is enough.
DEFAULT_BYTES_PER_TERM = 160


@dataclass(frozen=True, slots=True)
class LanguageSize:
    """One language's weight, as counted from the source corpus."""

    language: str
    terms: int


@dataclass(frozen=True, slots=True)
class PlannedPack:
    """A pack the transform should emit."""

    id: str
    capability: str
    languages: tuple[str, ...]
    estimated_compressed: int

    @property
    def bundled(self) -> bool:
        return len(self.languages) > 1


def estimate_compressed(terms: int, *, bytes_per_term: int = DEFAULT_BYTES_PER_TERM) -> int:
    return terms * bytes_per_term


def plan_lexical_packs(
    sizes: Sequence[LanguageSize],
    *,
    individual_threshold: int = DEFAULT_INDIVIDUAL_THRESHOLD,
    bundle_target: int = DEFAULT_BUNDLE_TARGET,
    bytes_per_term: int = DEFAULT_BYTES_PER_TERM,
) -> tuple[PlannedPack, ...]:
    """Assign every language to exactly one lexical pack.

    Languages heavy enough to be worth fetching alone get their own pack; the
    rest accumulate into size-balanced bundles in descending size order, so
    bundles stay contiguous and legible.  Whatever is left over at the end forms
    the final bundle -- the long tail needs no special case, because on
    data-v1.1.0 all 4,508 languages below rank 1000 come to 6.7 MiB together.

    Ordering is by size then language tag, so the plan does not depend on the
    order rows came out of the corpus.
    """

    if individual_threshold < 1 or bundle_target < 1 or bytes_per_term < 1:
        raise ValueError("pack planning thresholds must be positive")
    ordered = sorted(sizes, key=lambda item: (-item.terms, item.language))
    seen: set[str] = set()
    for item in ordered:
        if item.terms < 0:
            raise ValueError(f"language {item.language!r} has a negative term count")
        if item.language in seen:
            raise ValueError(f"duplicate language in pack plan: {item.language}")
        seen.add(item.language)

    packs: list[PlannedPack] = []
    bundle: list[LanguageSize] = []
    bundle_bytes = 0

    def flush() -> None:
        nonlocal bundle, bundle_bytes
        if not bundle:
            return
        packs.append(
            PlannedPack(
                id=f"lexical-bundle-{sum(1 for pack in packs if pack.bundled) + 1:03d}",
                capability="lexical",
                languages=tuple(item.language for item in bundle),
                estimated_compressed=bundle_bytes,
            )
        )
        bundle = []
        bundle_bytes = 0

    for item in ordered:
        estimated = estimate_compressed(item.terms, bytes_per_term=bytes_per_term)
        if estimated >= individual_threshold:
            packs.append(
                PlannedPack(
                    id=f"lexical-{item.language}",
                    capability="lexical",
                    languages=(item.language,),
                    estimated_compressed=estimated,
                )
            )
            continue
        bundle.append(item)
        bundle_bytes += estimated
        if bundle_bytes >= bundle_target:
            flush()
    flush()
    return tuple(packs)


def plan_capability_packs(
    capability: str, languages: Sequence[str]
) -> tuple[PlannedPack, ...]:
    """Plan one pack per language for a capability that is not size-tiered.

    Semantic, pronunciation and wordplay coverage is narrow enough -- 78
    languages for semantic, English alone for the other two -- that per-language
    packs need no bundling.
    """

    return tuple(
        PlannedPack(
            id=f"{capability}-{language}",
            capability=capability,
            languages=(language,),
            estimated_compressed=0,
        )
        for language in sorted(set(languages))
    )


# The physical schema of a schema-2 lexical pack.
#
# It differs from the schema-1 monolith in one way: foreign terms referenced by
# this pack's edges live in `target_catalogue` rather than as rows in
# `lexical_terms`.  Keeping them out of `lexical_terms` matters twice over --
# that table carries UNIQUE (language, normalized_term, term), whose index
# measured 1.93x the row cost on stub data, and several queries read
# `lexical_terms` bare, where stubs would otherwise read as real headwords.
#
# `entry_count` and `sense_count` are computed from the canonical full corpus at
# package time, so relation ranking is identical no matter which languages a
# user installed.
LEXICAL_PACK_SCHEMA = """
CREATE TABLE provenance (
    provenance_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_url TEXT NOT NULL,
    UNIQUE (source, source_license, source_url)
);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE lexical_terms (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    UNIQUE (language, normalized_term, term)
);
CREATE TABLE target_catalogue (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL,
    entry_count INTEGER NOT NULL,
    sense_count INTEGER NOT NULL
);
CREATE TABLE lexical_entries (
    entry_id TEXT PRIMARY KEY,
    term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
    part_of_speech TEXT,
    etymology TEXT,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
);
CREATE TABLE senses (
    sense_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES lexical_entries(entry_id) ON DELETE CASCADE,
    gloss TEXT
);
CREATE TABLE examples (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    example TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, position)
) WITHOUT ROWID;
CREATE TABLE pronunciations (
    entry_id TEXT NOT NULL REFERENCES lexical_entries(entry_id) ON DELETE CASCADE,
    ipa TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL,
    PRIMARY KEY (entry_id, position)
) WITHOUT ROWID;
CREATE TABLE translations (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    target_term_id INTEGER NOT NULL,
    part_of_speech TEXT,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, target_term_id, position)
) WITHOUT ROWID;
CREATE TABLE synonyms (
    sense_id TEXT NOT NULL REFERENCES senses(sense_id) ON DELETE CASCADE,
    target_term_id INTEGER NOT NULL,
    part_of_speech TEXT,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id),
    position INTEGER NOT NULL,
    PRIMARY KEY (sense_id, target_term_id, position)
) WITHOUT ROWID;
CREATE TABLE relations (
    source_term_id INTEGER NOT NULL,
    source_sense_id TEXT,
    relation_code INTEGER NOT NULL CHECK (relation_code BETWEEN 1 AND 12),
    target_term_id INTEGER NOT NULL,
    target_sense_id TEXT,
    direction_code INTEGER NOT NULL CHECK (direction_code BETWEEN 1 AND 3),
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
);
"""

LEXICAL_PACK_INDEXES = """
CREATE INDEX lexical_entries_lookup ON lexical_entries(term_id, part_of_speech);
CREATE INDEX senses_entry_lookup ON senses(entry_id);
CREATE INDEX relations_source_lookup ON relations(source_term_id, relation_code);
CREATE INDEX relations_target_lookup ON relations(target_term_id, relation_code);
"""

# The always-installed catalogue. Deliberately tiny: it holds routing and
# capability metadata only, never lexical payload, so a Japanese-only install
# never pays for English resources.
CORE_PACK_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE language_catalogue (
    language TEXT PRIMARY KEY,
    term_count INTEGER NOT NULL,
    entry_count INTEGER NOT NULL,
    sense_count INTEGER NOT NULL,
    translation_count INTEGER NOT NULL,
    relation_count INTEGER NOT NULL,
    has_semantic INTEGER NOT NULL CHECK (has_semantic IN (0, 1)),
    has_pronunciation INTEGER NOT NULL CHECK (has_pronunciation IN (0, 1)),
    has_wordplay INTEGER NOT NULL CHECK (has_wordplay IN (0, 1))
);
"""
