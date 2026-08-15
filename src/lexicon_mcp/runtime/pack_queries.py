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

# A pack names a term either as its own headword or as a catalogue stub. Both
# are reached by primary key, so the target is resolved with two LEFT JOINs and
# a COALESCE rather than a union of the two tables.
#
# This matters enormously. A CTE unioning lexical_terms and target_catalogue is
# materialized in full before it is filtered -- for English that is 1.99M terms
# with correlated count subqueries plus 5.1M stubs, on every query, measured at
# ~39s against 0.14s for the equivalent monolith query. Joining by key instead
# touches only the rows an edge actually references.
#
# Counts come from the catalogue for stubs and are computed locally for
# headwords, which is exact either way: a pack holds every entry and sense for
# the languages it owns.
_TARGET_COLUMNS = """
           COALESCE(local.term_id, stub.term_id) AS target_term_id,
           COALESCE(local.term, stub.term) AS target_term,
           COALESCE(local.normalized_term, stub.normalized_term) AS target_normalized,
           COALESCE(local.language, stub.language) AS target_language,
           CASE WHEN local.term_id IS NULL THEN 0 ELSE 1 END AS target_payload_local,
           COALESCE(
               stub.entry_count,
               (SELECT COUNT(*) FROM lexical_entries AS le
                 WHERE le.term_id = local.term_id)
           ) AS target_entry_count,
           COALESCE(
               stub.sense_count,
               (SELECT COUNT(*) FROM senses AS ls
                  JOIN lexical_entries AS le2 ON le2.entry_id = ls.entry_id
                 WHERE le2.term_id = local.term_id)
           ) AS target_sense_count,
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

def _relation_sql(
    *,
    reverse: bool,
    target_language: bool,
    sense_filter: bool,
) -> str:
    """Build the relation query for one orientation and filter combination.

    The service filters relations by target language and by source sense, so the
    pack-native form must too. The clauses are assembled rather than templated
    at import time because the filtered and unfiltered plans differ.
    """

    if reverse:
        anchor = "target"
        rank_partition = "relation.source_term_id"
        sense_column = "relation.target_sense_id"
        resolved_key = "relation.source_term_id"
        head = """
           target.term_id AS source_term_id,
           target.term AS source_term,
           target.normalized_term AS source_normalized,
           target.language AS source_language,
           relation.target_sense_id AS source_sense_id,"""
        tail_sense = "relation.source_sense_id AS target_sense_id"
        rank_order = "relation.target_sense_id, relation.source_sense_id"
    else:
        anchor = "source"
        rank_partition = "relation.target_term_id"
        sense_column = "relation.source_sense_id"
        resolved_key = "relation.target_term_id"
        head = """
           source.term_id AS source_term_id,
           source.term AS source_term,
           source.normalized_term AS source_normalized,
           source.language AS source_language,
           relation.source_sense_id,"""
        tail_sense = "relation.target_sense_id"
        rank_order = "relation.source_sense_id, relation.target_sense_id"

    clauses = [
        f"{anchor}.normalized_term = ?",
        f"{anchor}.language = ?",
        "relation.relation_code = ?",
    ]
    if target_language:
        clauses.append("COALESCE(local.language, stub.language) = ?")
    if sense_filter:
        clauses.append(f"{sense_column} = ?")

    return f"""
WITH candidates AS (
    SELECT{head}
{_TARGET_COLUMNS}
           {tail_sense},
           relation.direction_code,
           provenance.provenance_id,
           provenance.source, provenance.source_license, provenance.source_url,
           {2 if reverse else 1} AS query_orientation,
           ROW_NUMBER() OVER (
               PARTITION BY {rank_partition}
               ORDER BY CASE
                            WHEN relation.source_sense_id IS NULL
                             AND relation.target_sense_id IS NULL THEN 1
                            ELSE 0
                        END,
                        {rank_order},
                        provenance.provenance_id,
                        relation.direction_code
           ) AS target_variant_rank
    FROM relations AS relation
    JOIN lexical_terms AS {anchor} ON {anchor}.term_id = relation.{anchor}_term_id
    LEFT JOIN lexical_terms AS local ON local.term_id = {resolved_key}
    LEFT JOIN target_catalogue AS stub ON stub.term_id = {resolved_key}
    JOIN provenance ON provenance.provenance_id = relation.provenance_id
    WHERE {" AND ".join(clauses)}
)
SELECT * FROM candidates
{_RELATION_ORDER}
"""


def relation_rows(
    connection: sqlite3.Connection,
    *,
    word: str,
    language: str,
    relation_code: int,
    limit: int,
    reverse: bool = False,
    target_language: str | None = None,
    sense_id: str | None = None,
) -> list[sqlite3.Row]:
    """Direct relations for a word, resolved entirely within one pack."""

    sql = _relation_sql(
        reverse=reverse,
        target_language=target_language is not None,
        sense_filter=sense_id is not None,
    )
    parameters: list[Any] = [word, language, relation_code]
    if target_language is not None:
        parameters.append(target_language)
    if sense_id is not None:
        parameters.append(sense_id)
    parameters.append(limit)
    connection.row_factory = sqlite3.Row
    return connection.execute(sql, parameters).fetchall()


_TRANSLATIONS = """
SELECT COALESCE(local.term, stub.term) AS term,
       COALESCE(local.normalized_term, stub.normalized_term) AS normalized_term,
       COALESCE(local.language, stub.language) AS target_language,
       CASE WHEN local.term_id IS NULL THEN 0 ELSE 1 END AS target_payload_local,
       translation.part_of_speech, translation.position,
       provenance.source, provenance.source_license, provenance.source_url
FROM translations AS translation
LEFT JOIN lexical_terms AS local ON local.term_id = translation.target_term_id
LEFT JOIN target_catalogue AS stub ON stub.term_id = translation.target_term_id
JOIN provenance ON provenance.provenance_id = translation.provenance_id
WHERE translation.sense_id = ?{language_filter}
ORDER BY translation.position, target_language, normalized_term, term
LIMIT ?
"""

# Distinct target languages and row counts for a bounded sense set, resolved
# through the catalogue like everything else that names a foreign term.
_TRANSLATION_COVERAGE = """
SELECT translation.sense_id,
       COUNT(DISTINCT COALESCE(local.language, stub.language)) AS language_count,
       COUNT(*) AS translation_count
FROM translations AS translation
LEFT JOIN lexical_terms AS local ON local.term_id = translation.target_term_id
LEFT JOIN target_catalogue AS stub ON stub.term_id = translation.target_term_id
WHERE translation.sense_id IN ({placeholders})
GROUP BY translation.sense_id
"""


def translation_coverage(
    connection: sqlite3.Connection, sense_ids: list[str]
) -> dict[str, tuple[int, int]]:
    """Language and row counts per sense, for truncation reporting."""

    if not sense_ids:
        return {}
    placeholders = ", ".join("?" for _ in sense_ids)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        _TRANSLATION_COVERAGE.format(placeholders=placeholders), sense_ids
    ).fetchall()
    return {
        str(row["sense_id"]): (int(row["language_count"]), int(row["translation_count"]))
        for row in rows
    }


def translation_rows(
    connection: sqlite3.Connection,
    *,
    sense_id: str,
    limit: int = 100,
    target_language: str | None = None,
) -> list[sqlite3.Row]:
    """Translations of a sense, including targets whose language is absent.

    A directional translation assertion stays useful without the target
    language installed -- an English-only install can still answer what the
    French for a word is -- so these rows are returned with
    ``target_payload_local`` false rather than filtered away.
    """

    connection.row_factory = sqlite3.Row
    if target_language is None:
        sql = _TRANSLATIONS.format(language_filter="")
        parameters: list[Any] = [sense_id, limit]
    else:
        sql = _TRANSLATIONS.format(
            language_filter=" AND COALESCE(local.language, stub.language) = ?"
        )
        parameters = [sense_id, target_language, limit]
    return connection.execute(sql, parameters).fetchall()


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
