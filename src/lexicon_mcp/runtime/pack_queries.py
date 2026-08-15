"""Pack-native forms of the queries that used to join a monolithic corpus.

A pack owns the full payload for its own languages and names every foreign term
its edges reach through `target_catalogue`.  These queries therefore resolve a
relation's target without opening another pack, and rank results by counts
computed from the whole corpus at package time -- so ordering does not shift
when an unrelated language is installed or removed.

Two facts make the ranking identical to the monolith's:

* a pack holds *every* entry and sense for the languages it owns, so counting
  them locally gives the same number the full corpus would; and
* for foreign terms the counts were materialized from the full corpus, so they
  are right even though the payload is absent.
"""

from __future__ import annotations

import sqlite3
from typing import Any

# Every term a pack can name: its own headwords, plus the foreign stubs its
# edges reach. `payload_local` tells a caller whether the dictionary entry
# behind a term is actually installed here.
TERM_UNION = """
    term_union AS (
        SELECT t.term_id, t.term, t.normalized_term, t.language,
               (SELECT COUNT(*) FROM lexical_entries AS e
                 WHERE e.term_id = t.term_id) AS entry_count,
               (SELECT COUNT(*) FROM senses AS s
                  JOIN lexical_entries AS e2 ON e2.entry_id = s.entry_id
                 WHERE e2.term_id = t.term_id) AS sense_count,
               1 AS payload_local
        FROM lexical_terms AS t
        UNION ALL
        SELECT c.term_id, c.term, c.normalized_term, c.language,
               c.entry_count, c.sense_count, 0
        FROM target_catalogue AS c
    )
"""

# Ordering is copied from the monolith deliberately, including its tie-breaks.
# The differential gate compares ordered results, so any divergence here is a
# release failure rather than a cosmetic difference.
_RELATION_ORDER = """
    ORDER BY target_variant_rank,
             target_sense_count DESC, target_entry_count DESC,
             (LENGTH(target_normalized) -
              LENGTH(REPLACE(target_normalized, ' ', ''))),
             LENGTH(target_normalized), target_language,
             target_normalized, target_term, target_sense_id,
             provenance_id, direction_code
    LIMIT ?
"""

_FORWARD = f"""
WITH {TERM_UNION},
candidates AS (
    SELECT source.term_id AS source_term_id,
           source.term AS source_term,
           source.normalized_term AS source_normalized,
           source.language AS source_language,
           relation.source_sense_id,
           target.term_id AS target_term_id,
           target.term AS target_term,
           target.normalized_term AS target_normalized,
           target.language AS target_language,
           target.payload_local AS target_payload_local,
           relation.target_sense_id,
           relation.direction_code,
           provenance.provenance_id,
           provenance.source, provenance.source_license, provenance.source_url,
           target.entry_count AS target_entry_count,
           target.sense_count AS target_sense_count,
           ROW_NUMBER() OVER (
               PARTITION BY target.term_id
               ORDER BY CASE
                            WHEN relation.source_sense_id IS NULL
                             AND relation.target_sense_id IS NULL THEN 1
                            ELSE 0
                        END,
                        relation.source_sense_id,
                        relation.target_sense_id,
                        provenance.provenance_id,
                        relation.direction_code
           ) AS target_variant_rank
    FROM relations AS relation
    JOIN lexical_terms AS source ON source.term_id = relation.source_term_id
    JOIN term_union AS target ON target.term_id = relation.target_term_id
    JOIN provenance ON provenance.provenance_id = relation.provenance_id
    WHERE source.normalized_term = ?
      AND source.language = ?
      AND relation.relation_code = ?
)
SELECT * FROM candidates
{_RELATION_ORDER}
"""

_REVERSE = f"""
WITH {TERM_UNION},
candidates AS (
    SELECT target.term_id AS source_term_id,
           target.term AS source_term,
           target.normalized_term AS source_normalized,
           target.language AS source_language,
           relation.target_sense_id AS source_sense_id,
           source.term_id AS target_term_id,
           source.term AS target_term,
           source.normalized_term AS target_normalized,
           source.language AS target_language,
           source.payload_local AS target_payload_local,
           relation.source_sense_id AS target_sense_id,
           relation.direction_code,
           provenance.provenance_id,
           provenance.source, provenance.source_license, provenance.source_url,
           source.entry_count AS target_entry_count,
           source.sense_count AS target_sense_count,
           ROW_NUMBER() OVER (
               PARTITION BY source.term_id
               ORDER BY CASE
                            WHEN relation.source_sense_id IS NULL
                             AND relation.target_sense_id IS NULL THEN 1
                            ELSE 0
                        END,
                        relation.target_sense_id,
                        relation.source_sense_id,
                        provenance.provenance_id,
                        relation.direction_code
           ) AS target_variant_rank
    FROM relations AS relation
    JOIN term_union AS source ON source.term_id = relation.source_term_id
    JOIN lexical_terms AS target ON target.term_id = relation.target_term_id
    JOIN provenance ON provenance.provenance_id = relation.provenance_id
    WHERE target.normalized_term = ?
      AND target.language = ?
      AND relation.relation_code = ?
)
SELECT * FROM candidates
{_RELATION_ORDER}
"""

_TRANSLATIONS = f"""
WITH {TERM_UNION}
SELECT term.term, term.normalized_term,
       term.language AS target_language,
       term.payload_local AS target_payload_local,
       translation.part_of_speech, translation.position,
       provenance.source, provenance.source_license, provenance.source_url
FROM translations AS translation
JOIN term_union AS term ON term.term_id = translation.target_term_id
JOIN provenance ON provenance.provenance_id = translation.provenance_id
WHERE translation.sense_id = ?
ORDER BY translation.position, term.language,
         term.normalized_term, term.term
LIMIT ?
"""


def relation_rows(
    connection: sqlite3.Connection,
    *,
    word: str,
    language: str,
    relation_code: int,
    limit: int,
    reverse: bool = False,
) -> list[sqlite3.Row]:
    """Direct relations for a word, resolved entirely within one pack."""

    connection.row_factory = sqlite3.Row
    return connection.execute(
        _REVERSE if reverse else _FORWARD, (word, language, relation_code, limit)
    ).fetchall()


def translation_rows(
    connection: sqlite3.Connection, *, sense_id: str, limit: int = 100
) -> list[sqlite3.Row]:
    """Translations of a sense, including targets whose language is absent.

    A directional translation assertion stays useful without the target
    language installed -- an English-only install can still answer what the
    French for a word is -- so these rows are returned with
    ``target_payload_local`` false rather than filtered away.
    """

    connection.row_factory = sqlite3.Row
    return connection.execute(_TRANSLATIONS, (sense_id, limit)).fetchall()


def expandable(row: sqlite3.Row) -> bool:
    """Whether a result's target can be followed to its own dictionary entry."""

    keys = row.keys()
    return "target_payload_local" in keys and bool(row["target_payload_local"])


def as_result(row: sqlite3.Row) -> dict[str, Any]:
    """Shape a row for a tool response, marking non-expandable targets."""

    # .keys() is required: sqlite3.Row iterates its *values*, so `for key in row`
    # would yield cell contents and `"x" in row` is a value membership test.
    value = {key: row[key] for key in row.keys()}  # noqa: SIM118
    if "target_payload_local" in value:
        local = bool(value.pop("target_payload_local"))
        value["target_language_installed"] = local
        value["target_details_available"] = local
    return value
