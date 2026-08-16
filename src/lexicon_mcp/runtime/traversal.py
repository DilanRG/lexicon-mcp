"""Bounded multi-hop relation traversal across packs.

Direct relations resolve inside one pack, because a pack names every term its
edges reach.  Following a second hop does not: expanding a French target means
reading French's *own* outbound edges, which live in the French pack.

So traversal is a routed operation, not a fan-out.  The frontier is grouped by
the pack that owns each node, each installed pack is queried once, and results
merge under one deterministic order.  Frontier nodes whose language is not
installed are reported rather than dropped -- a traversal that could not be
completed is a different answer from one that found nothing, and saying so is
the difference between "there is no path" and "you do not have French".
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .pack_queries import relation_rows
from .router import PackRouter


@dataclass(frozen=True, slots=True)
class FrontierNode:
    """One node to expand: a word in a language."""

    language: str
    word: str


@dataclass
class TraversalResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    unexpanded: list[dict[str, str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unexpanded

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "complete": self.complete,
            "unexpanded": self.unexpanded,
        }


def _order_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """The monolith's relation ordering, applied across merged packs.

    Ranking inputs are full-corpus counts baked in at package time, so ordering
    the union of several packs' results reproduces the order a single corpus
    would have produced.
    """

    normalized = str(row["target_normalized"])
    return (
        int(row["target_variant_rank"]),
        -int(row["target_sense_count"]),
        -int(row["target_entry_count"]),
        len(normalized) - len(normalized.replace(" ", "")),
        len(normalized),
        str(row["target_language"]),
        normalized,
        str(row["target_term"]),
        str(row["target_sense_id"] or ""),
        int(row["provenance_id"]),
        int(row["direction_code"]),
    )


def expand_frontier(
    router: PackRouter,
    frontier: Sequence[FrontierNode],
    *,
    relation_code: int,
    limit: int,
    reverse: bool = False,
) -> TraversalResult:
    """Expand every frontier node, one query per pack rather than per node."""

    result = TraversalResult()
    by_language: dict[str, list[FrontierNode]] = {}
    for node in frontier:
        by_language.setdefault(node.language, []).append(node)

    seen_unexpandable: set[str] = set()
    # Group languages by the pack serving them, so a bundle is opened once even
    # when the frontier reaches several of its languages.
    packs: dict[int, tuple[sqlite3.Connection, list[FrontierNode]]] = {}
    for language, nodes in sorted(by_language.items()):
        connection = router.connection_for("lexical", language)
        if connection is None:
            if language not in seen_unexpandable:
                seen_unexpandable.add(language)
                availability = router.availability("lexical", language)
                result.unexpanded.append(
                    {"language": language, "reason": availability.reason}
                )
            continue
        entry = packs.setdefault(id(connection), (connection, []))
        entry[1].extend(nodes)

    merged: list[dict[str, Any]] = []
    for connection, nodes in packs.values():
        for node in nodes:
            for row in relation_rows(
                connection,
                word=node.word,
                language=node.language,
                relation_code=relation_code,
                limit=limit,
                reverse=reverse,
            ):
                value = {key: row[key] for key in row.keys()}  # noqa: SIM118
                value["via_language"] = node.language
                value["via_word"] = node.word
                merged.append(value)

    merged.sort(key=_order_key)
    result.rows = merged[:limit]
    return result


def frontier_from_rows(rows: Sequence[dict[str, Any]]) -> tuple[FrontierNode, ...]:
    """Turn first-hop results into the next frontier.

    Every distinct target becomes a node, including targets whose payload is not
    installed.  Filtering them here would silently lose branches; instead
    :func:`expand_frontier` reports them as unexpanded, so the caller learns the
    path exists and what it would take to follow it.
    """

    nodes: list[FrontierNode] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        identity = (str(row["target_language"]), str(row["target_normalized"]))
        if identity in seen:
            continue
        seen.add(identity)
        nodes.append(FrontierNode(language=identity[0], word=identity[1]))
    return tuple(nodes)
