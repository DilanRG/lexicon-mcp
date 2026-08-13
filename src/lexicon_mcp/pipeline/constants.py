"""Corpus source and schema constants used by build scripts."""

from __future__ import annotations

SCHEMA_VERSION = "1"

SOURCE_EXPECTATIONS = {
    "oewn": {
        "url": "https://en-word.net/static/english-wordnet-2025.xml.gz",
        "snapshot": "2025",
        "license": "CC-BY-4.0",
    },
    "wiktextract": {
        "url": "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz",
        "snapshot": "pinned-by-sources-lock",
        "license": "CC-BY-SA-4.0 and GFDL-1.3-or-later",
    },
    "conceptnet": {
        "url": "https://conceptnet.s3.amazonaws.com/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
        "snapshot": "5.7.0",
        "license": "CC-BY-SA-4.0",
    },
    "numberbatch": {
        "url": "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz",
        "snapshot": "19.08",
        "license": "CC-BY-SA-4.0",
    },
    "cmudict": {
        "url": "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict",
        "snapshot": "pinned-by-sources-lock",
        "license": "BSD-2-Clause-style",
    },
}

SUPPORTED_RELATIONS = frozenset(
    {
        "synonym",
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
    }
)
