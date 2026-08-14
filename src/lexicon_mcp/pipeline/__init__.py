"""Reproducible builders for the offline Lexicon MCP corpus.

The build package is deliberately independent of the MCP adapter.  Every
builder accepts local, pinned source files and writes deterministic, read-only
runtime artifacts.  Downloading source corpora is an orchestration concern and
never happens here.
"""

from .notice_amendment import amend_promoted_dataset_notices
from .orchestrator import (
    BuildInputs,
    build_full_corpus,
    recover_full_corpus_from_semantic_partial,
)

__all__ = [
    "BuildInputs",
    "amend_promoted_dataset_notices",
    "build_full_corpus",
    "recover_full_corpus_from_semantic_partial",
]
