"""Reproducible builders for the offline Lexicon MCP corpus.

The build package is deliberately independent of the MCP adapter.  Every
builder accepts local, pinned source files and writes deterministic, read-only
runtime artifacts.  Downloading source corpora is an orchestration concern and
never happens here.
"""

from .orchestrator import BuildInputs, build_full_corpus

__all__ = ["BuildInputs", "build_full_corpus"]

