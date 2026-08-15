"""Compare a sharded installation against the monolith it was built from.

The release gate: a schema-2 install and the schema-1 corpus it was
repartitioned from must return the same ordered logical results across the
serving-path query matrix. Anything else means the transform lost or reordered
something, which no amount of unit testing on fixtures would catch.

Ordering is compared, not just membership. Relation ranking depends on
target-side corpus statistics, and the whole point of materializing those at
package time is that they do not shift with the installed set -- so a
divergence in order is a real failure, not a cosmetic one.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexicon_mcp.data.component_lifecycle import ComponentLifecycle
from lexicon_mcp.runtime.pack_queries import relation_rows, translation_rows
from lexicon_mcp.runtime.router import PackRouter

# Mirrors runtime/service.py's forward relation query, including every
# tie-break. Divergence from it is the thing being tested.
MONOLITH_RELATIONS = """
WITH candidates AS (
    SELECT target.term AS target_term,
           target.language AS target_language,
           target.normalized_term AS target_normalized,
           (SELECT COUNT(*) FROM lexical_entries AS e
             WHERE e.term_id = target.term_id) AS target_entry_count,
           (SELECT COUNT(*) FROM senses AS s
              JOIN lexical_entries AS e2 ON e2.entry_id = s.entry_id
             WHERE e2.term_id = target.term_id) AS target_sense_count,
           relation.target_sense_id, provenance.provenance_id,
           relation.direction_code,
           ROW_NUMBER() OVER (
               PARTITION BY target.term_id
               ORDER BY CASE WHEN relation.source_sense_id IS NULL
                              AND relation.target_sense_id IS NULL THEN 1 ELSE 0 END,
                        relation.source_sense_id, relation.target_sense_id,
                        provenance.provenance_id, relation.direction_code
           ) AS target_variant_rank
    FROM relations AS relation
    JOIN lexical_terms AS source ON source.term_id = relation.source_term_id
    JOIN lexical_terms AS target ON target.term_id = relation.target_term_id
    JOIN provenance ON provenance.provenance_id = relation.provenance_id
    WHERE source.normalized_term = ? AND source.language = ?
      AND relation.relation_code = ?
)
SELECT * FROM candidates
ORDER BY target_variant_rank, target_sense_count DESC, target_entry_count DESC,
         (LENGTH(target_normalized) - LENGTH(REPLACE(target_normalized, ' ', ''))),
         LENGTH(target_normalized), target_language, target_normalized,
         target_term, target_sense_id, provenance_id, direction_code
LIMIT ?
"""

MONOLITH_SENSES = """
SELECT sense.sense_id, entry.part_of_speech, sense.gloss
FROM senses AS sense
JOIN lexical_entries AS entry ON entry.entry_id = sense.entry_id
JOIN lexical_terms AS term ON term.term_id = entry.term_id
WHERE term.normalized_term = ? AND term.language = ?
ORDER BY CASE WHEN sense.gloss IS NULL THEN 1 ELSE 0 END, sense.sense_id
LIMIT ?
"""

MONOLITH_TRANSLATIONS = """
SELECT term.term, term.normalized_term, term.language, translation.position
FROM translations AS translation
JOIN lexical_terms AS term ON term.term_id = translation.target_term_id
WHERE translation.sense_id = ?
ORDER BY translation.position, term.language, term.normalized_term, term.term
LIMIT ?
"""


@dataclass
class Report:
    checked: int = 0
    divergent: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        kind: str,
        language: str,
        word: str,
        expected: Sequence[Any],
        actual: Sequence[Any],
    ) -> None:
        self.checked += 1
        if list(expected) == list(actual):
            return
        self.divergent += 1
        if len(self.examples) < 20:
            self.examples.append(
                {
                    "kind": kind,
                    "language": language,
                    "word": word,
                    "expected": list(expected)[:6],
                    "actual": list(actual)[:6],
                }
            )


def relation_key(rows: Sequence[sqlite3.Row]) -> list[tuple[Any, ...]]:
    return [
        (
            row["target_term"],
            row["target_language"],
            int(row["target_entry_count"]),
            int(row["target_sense_count"]),
            int(row["target_variant_rank"]),
        )
        for row in rows
    ]


def sample_words(pack: sqlite3.Connection, language: str, limit: int, seed: int) -> list[str]:
    """Words with the most senses, plus a random spread across the language."""

    pack.row_factory = sqlite3.Row
    richest = [
        str(row["normalized_term"])
        for row in pack.execute(
            "SELECT t.normalized_term, COUNT(*) AS n FROM lexical_terms t"
            " JOIN lexical_entries e ON e.term_id = t.term_id"
            " JOIN senses s ON s.entry_id = e.entry_id"
            " WHERE t.language = ? GROUP BY t.term_id ORDER BY n DESC LIMIT ?",
            (language, max(1, limit // 2)),
        )
    ]
    everything = [
        str(row["normalized_term"])
        for row in pack.execute(
            "SELECT normalized_term FROM lexical_terms WHERE language = ?", (language,)
        )
    ]
    rng = random.Random(seed)
    spread = rng.sample(everything, min(len(everything), limit - len(richest)))
    return list(dict.fromkeys(richest + spread))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monolith", type=Path, required=True)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--languages", nargs="*", help="default: every installed language")
    parser.add_argument("--words", type=int, default=200, help="words sampled per language")
    parser.add_argument("--limit", type=int, default=25, help="rows per query")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    mono = sqlite3.connect(
        f"file:{(args.monolith / 'lexicon.sqlite3').as_posix()}?mode=ro&immutable=1", uri=True
    )
    mono.row_factory = sqlite3.Row

    manager = ComponentLifecycle(args.install)
    activation = manager.active_activation()
    if activation is None:
        raise SystemExit("no active installation to compare")

    started = time.monotonic()
    report = Report()
    with PackRouter(activation, manager.store, max_open_packs=4) as router:
        languages = args.languages or list(router.installed_languages("lexical"))
        print(f"comparing {len(languages)} languages", flush=True)
        for language in languages:
            pack = router.connection_for("lexical", language)
            if pack is None:
                print(f"  {language}: not installed, skipped", flush=True)
                continue
            words = sample_words(pack, language, args.words, args.seed)
            for word in words:
                for code in range(1, 13):
                    expected = relation_key(
                        mono.execute(
                            MONOLITH_RELATIONS, (word, language, code, args.limit)
                        ).fetchall()
                    )
                    actual = relation_key(
                        relation_rows(
                            pack,
                            word=word,
                            language=language,
                            relation_code=code,
                            limit=args.limit,
                        )
                    )
                    report.record("relations", language, word, expected, actual)

                expected_senses = [
                    tuple(row) for row in mono.execute(
                        MONOLITH_SENSES, (word, language, args.limit)
                    ).fetchall()
                ]
                pack.row_factory = sqlite3.Row
                actual_senses = [
                    tuple(row) for row in pack.execute(
                        MONOLITH_SENSES, (word, language, args.limit)
                    ).fetchall()
                ]
                report.record("senses", language, word, expected_senses, actual_senses)

                for sense_row in actual_senses[:3]:
                    sense_id = sense_row[0]
                    expected_translations = [
                        tuple(row) for row in mono.execute(
                            MONOLITH_TRANSLATIONS, (sense_id, args.limit)
                        ).fetchall()
                    ]
                    actual_translations = [
                        (
                            row["term"],
                            row["normalized_term"],
                            row["target_language"],
                            row["position"],
                        )
                        for row in translation_rows(
                            pack, sense_id=sense_id, limit=args.limit
                        )
                    ]
                    report.record(
                        "translations", language, word, expected_translations, actual_translations
                    )
            print(
                f"  {language}: {len(words)} words, {report.divergent} divergent so far",
                flush=True,
            )

    mono.close()
    elapsed = time.monotonic() - started
    result = {
        "ok": report.divergent == 0,
        "comparisons": report.checked,
        "divergent": report.divergent,
        "seconds": round(elapsed, 1),
        "examples": report.examples,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report.divergent == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
