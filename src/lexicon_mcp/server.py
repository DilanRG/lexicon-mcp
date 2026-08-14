"""FastMCP adapter exposing the six public lexicon tools."""

from __future__ import annotations

import threading
from typing import Annotated, Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .runtime import LexiconService
from .runtime.offline import install_network_guard

Relation = Literal[
    "antonym",
    "hypernym",
    "hyponym",
    "meronym",
    "holonym",
    "derived_from",
    "etymologically_related",
    "used_for",
    "capable_of",
    "at_location",
    "related",
]
WordplayMode = Literal["rhyme", "near_rhyme", "sounds_like", "spelled_like", "prefix"]
QueryText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        description="Unicode query text; normalized with NFKC and casefold for lookup.",
    ),
]
LanguageTag = Annotated[
    str,
    Field(
        min_length=2,
        max_length=256,
        description="ISO/BCP-47 language tag.",
    ),
]
OptionalQueryText = Annotated[
    str,
    Field(min_length=1, max_length=256),
]
ResultLimit = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        le=100,
        description="Maximum total result budget for this tool.",
    ),
]
DetailBudget = Annotated[
    int,
    Field(
        strict=True,
        ge=0,
        le=100,
        description="Fixed total response budget for this detail class; 0 disables it.",
    ),
]
MaxSenses = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        le=100,
        description="Maximum source-native lexical senses to inspect.",
    ),
]
CandidateAllocation = Annotated[
    int,
    Field(
        strict=True,
        ge=0,
        le=100,
        description="Maximum candidates from this result class within limit.",
    ),
]
RelationDepth = Literal[1, 2]
Similarity = Annotated[
    float,
    Field(
        strict=True,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        description="Minimum finite cosine similarity.",
    ),
]


class _LazyService:
    def __init__(self, service: LexiconService | None = None) -> None:
        self._service = service
        self._lock = threading.Lock()

    def get(self) -> LexiconService:
        with self._lock:
            if self._service is None:
                self._service = LexiconService.from_locator()
            return self._service


def create_mcp(service: LexiconService | None = None) -> FastMCP:
    """Create a server; service injection keeps protocol tests deterministic."""

    provider = _LazyService(service)
    server = FastMCP(
        "lexicon-mcp",
        instructions=(
            "Offline multilingual lexical data. Choose a returned sense before requesting "
            "sense-specific synonyms or translations. Unsensed fallbacks are labelled."
        ),
    )

    @server.tool()
    def dictionary_lookup(
        word: QueryText,
        language: LanguageTag = "en",
        part_of_speech: OptionalQueryText | None = None,
        limit: ResultLimit = 8,
        examples_limit: DetailBudget = 8,
        pronunciations_limit: DetailBudget = 8,
        translations_limit: DetailBudget = 20,
    ) -> dict[str, Any]:
        """Look up distinct dictionary senses for a word.

        Returns stable sense IDs, parts of speech, glosses, examples, IPA,
        etymology, translations, and source provenance. Use an ISO/BCP-47
        language tag such as en, de, es, ja, or zh-Hant. limit is the number
        of senses. Each detail limit is an independent total response budget,
        fairly shared across returned senses; 0 disables that detail class.
        Every sense lists any detail fields truncated by these fixed budgets.
        """

        return provider.get().dictionary_lookup(
            word,
            language,
            part_of_speech,
            limit,
            examples_limit,
            pronunciations_limit,
            translations_limit,
        )

    @server.tool()
    def dictionary_synonyms(
        word: QueryText,
        language: LanguageTag = "en",
        sense_id: OptionalQueryText | None = None,
        part_of_speech: OptionalQueryText | None = None,
        limit: ResultLimit = 20,
        max_senses: MaxSenses = 20,
        unsensed_limit: CandidateAllocation = 5,
    ) -> dict[str, Any]:
        """Find synonyms grouped by sense and part of speech.

        Pass a sense_id from dictionary_lookup when context matters. Candidates
        without a source sense are returned only as explicitly unsensed groups.
        max_senses controls how many lexical senses are inspected; unsensed_limit
        allocates part of the total candidate limit to all unsensed results. Set
        0 for strictly sense-scoped candidates. Allocations must not exceed limit.
        count is the number of sense groups; candidate_count is the total number
        of nested synonym candidates governed by limit.
        """

        return provider.get().dictionary_synonyms(
            word,
            language,
            sense_id,
            part_of_speech,
            limit,
            max_senses,
            unsensed_limit,
        )

    @server.tool()
    def dictionary_translate(
        word: QueryText,
        source_language: LanguageTag,
        target_language: LanguageTag,
        sense_id: OptionalQueryText | None = None,
        part_of_speech: OptionalQueryText | None = None,
        limit: ResultLimit = 20,
        max_senses: MaxSenses = 100,
    ) -> dict[str, Any]:
        """Translate a word while preserving its source-sense association.

        Use dictionary_lookup first for ambiguous words, then pass its sense_id.
        Results are grouped by source sense and never silently cross senses.
        max_senses bounds the source-native senses inspected independently of
        limit. limit is the total translation-candidate budget, fairly shared
        across matching sense groups. count reports groups and candidate_count
        reports nested translations.
        """

        return provider.get().dictionary_translate(
            word,
            source_language,
            target_language,
            sense_id,
            part_of_speech,
            limit,
            max_senses,
        )

    @server.tool()
    def dictionary_relations(
        word: QueryText,
        relation: Relation,
        language: LanguageTag = "en",
        target_language: LanguageTag | None = None,
        sense_id: OptionalQueryText | None = None,
        limit: ResultLimit = 20,
        max_depth: RelationDepth = 2,
        transitive_limit: CandidateAllocation = 5,
    ) -> dict[str, Any]:
        """Find one directed lexical or commonsense relation.

        Supported relations include antonym, hypernym, hyponym, meronym,
        holonym, derivation, etymology, use, capability, location, and related.
        Each result states its direction and provenance. Direct results have
        relation_scope="direct" and distance=1. Hypernym and hyponym queries
        may also return distance-2 results explicitly labelled
        relation_scope="transitive", with both sourced edges in path. max_depth
        is the relation graph hop limit (v1 supports one or two), while
        transitive_limit allocates part of the total candidate limit. Set
        max_depth=1 or transitive_limit=0 for direct results only. Allocations
        must not exceed limit.
        """

        return provider.get().dictionary_relations(
            word,
            relation,
            language,
            target_language,
            sense_id,
            limit,
            max_depth,
            transitive_limit,
        )

    @server.tool()
    async def dictionary_semantic_neighbors(
        word: QueryText,
        source_language: LanguageTag = "en",
        target_language: LanguageTag | None = None,
        limit: ResultLimit = 20,
        min_similarity: Similarity | None = None,
    ) -> dict[str, Any]:
        """Find distributional semantic neighbours with an optional language filter.

        This uses the installed memory-mapped Numberbatch ANN index. It returns
        an empty unavailable response when semantic artifacts are not installed.
        Omit target_language for global multilingual results, set it equal to
        source_language for monolingual results, or use another tag for
        cross-lingual results. Similarity is finite cosine similarity in the
        range -1 through 1.
        """

        # Keep the blocking worker protocol off FastMCP's stdio event loop so
        # other MCP requests remain responsive while ANN search is in flight.
        return await anyio.to_thread.run_sync(
            lambda: provider.get().dictionary_semantic_neighbors(
                word, source_language, target_language, limit, min_similarity
            )
        )

    @server.tool()
    def dictionary_wordplay(
        mode: WordplayMode, text: QueryText, limit: ResultLimit = 20
    ) -> dict[str, Any]:
        """Find English rhymes, near rhymes, homophones, spelling patterns, or prefixes.

        Results are English CMUdict-backed headwords. near_rhyme means exactly
        one ARPAbet-token insertion, deletion, or substitution; sounds_like
        requires an exact full-phoneme match. This fixed v1 behavior has no
        automatic or configurable edit distance.
        In spelled_like mode, ? matches exactly one character and * matches any
        sequence. The query itself is excluded from results.
        """

        return provider.get().dictionary_wordplay(mode, text, limit)

    return server


mcp = create_mcp()


async def _run_stdio_offline() -> None:
    """Serve stdio only after AnyIO has created its local event-loop plumbing.

    Windows' ProactorEventLoop creates an internal loopback socketpair during
    loop construction.  Installing the process-wide socket guard before that
    would mistake the runtime's local plumbing for an external connection and
    prevent the server from starting.  ``anyio.run`` constructs the loop before
    this coroutine is entered; from here onward every application-initiated
    network connection remains denied.
    """

    install_network_guard()
    await mcp.run_stdio_async()


def main() -> None:
    anyio.run(_run_stdio_offline)


if __name__ == "__main__":
    main()
