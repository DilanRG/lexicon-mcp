"""FastMCP adapter exposing the six public lexicon tools."""

from __future__ import annotations

import threading
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .runtime import LexiconService

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
        word: str,
        language: str = "en",
        part_of_speech: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Look up distinct dictionary senses for a word.

        Returns stable sense IDs, parts of speech, glosses, examples, IPA,
        etymology, translations, and source provenance. Use an ISO/BCP-47
        language tag such as en, de, es, ja, or zh-Hant.
        """

        return provider.get().dictionary_lookup(word, language, part_of_speech, limit)

    @server.tool()
    def dictionary_synonyms(
        word: str,
        language: str = "en",
        sense_id: str | None = None,
        part_of_speech: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Find synonyms grouped by sense and part of speech.

        Pass a sense_id from dictionary_lookup when context matters. Candidates
        without a source sense are returned only as explicitly unsensed groups.
        """

        return provider.get().dictionary_synonyms(
            word, language, sense_id, part_of_speech, limit
        )

    @server.tool()
    def dictionary_translate(
        word: str,
        source_language: str,
        target_language: str,
        sense_id: str | None = None,
        part_of_speech: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Translate a word while preserving its source-sense association.

        Use dictionary_lookup first for ambiguous words, then pass its sense_id.
        Results are grouped by source sense and never silently cross senses.
        """

        return provider.get().dictionary_translate(
            word,
            source_language,
            target_language,
            sense_id,
            part_of_speech,
            limit,
        )

    @server.tool()
    def dictionary_relations(
        word: str,
        relation: Relation,
        language: str = "en",
        target_language: str | None = None,
        sense_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Find one directed lexical or commonsense relation.

        Supported relations include antonym, hypernym, hyponym, meronym,
        holonym, derivation, etymology, use, capability, location, and related.
        Each result states its direction and provenance.
        """

        return provider.get().dictionary_relations(
            word, relation, language, target_language, sense_id, limit
        )

    @server.tool()
    def dictionary_semantic_neighbors(
        word: str,
        source_language: str = "en",
        target_language: str | None = None,
        limit: int = 20,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """Find distributional semantic neighbours, optionally in another language.

        This uses the installed memory-mapped Numberbatch ANN index. It returns
        an empty unavailable response when semantic artifacts are not installed.
        Similarity is finite cosine similarity in the range -1 through 1.
        """

        return provider.get().dictionary_semantic_neighbors(
            word, source_language, target_language, limit, min_similarity
        )

    @server.tool()
    def dictionary_wordplay(
        mode: WordplayMode, text: str, limit: int = 20
    ) -> dict[str, Any]:
        """Find English rhymes, near rhymes, homophones, spelling patterns, or prefixes.

        In spelled_like mode, ? matches exactly one character and * matches any
        sequence. The query itself is excluded from results.
        """

        return provider.get().dictionary_wordplay(mode, text, limit)

    return server


mcp = create_mcp()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
