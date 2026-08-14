"""Validation and Unicode normalization shared by every query path."""

from __future__ import annotations

import re
import unicodedata

MAX_TEXT_LENGTH = 256
MAX_LIMIT = 100

_LANGUAGE_SUBTAG = re.compile(r"^[A-Za-z]{2,8}$")
_OTHER_SUBTAG = re.compile(r"^[A-Za-z0-9]{1,8}$")


def normalize_key(value: str, *, field: str = "word", allow_wildcards: bool = False) -> str:
    """Return the canonical lookup key while retaining no display-side mutation."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    value = value.strip()
    if not value:
        raise ValueError(f"{field} cannot be empty")
    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field} must be at most {MAX_TEXT_LENGTH} characters")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError(f"{field} cannot contain control characters")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not allow_wildcards and ("*" in normalized or "?" in normalized):
        # Wildcards are ordinary headword characters outside the one pattern API.
        return normalized
    return normalized


def normalize_language(value: str, *, field: str = "language") -> str:
    """Validate and normalize a practical BCP-47 tag to on-disk lowercase."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    raw = value.strip().replace("_", "-")
    if not raw:
        raise ValueError(f"{field} cannot be empty")
    parts = raw.split("-")
    if not _LANGUAGE_SUBTAG.fullmatch(parts[0]) or any(
        not _OTHER_SUBTAG.fullmatch(part) for part in parts[1:]
    ):
        raise ValueError(f"{field} must be a valid BCP-47 language tag")

    # Builders and Numberbatch shard filenames use casefolded tags. Keep that
    # representation at the query boundary so valid script/region tags such as
    # zh-Hant and pt-BR address the rows and shards stored as zh-hant/pt-br.
    return "-".join(part.casefold() for part in parts)


def validate_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("limit must be an integer")
    if not 1 <= value <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return value


def normalize_optional_text(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return normalize_key(value, field=field)


def sense_scope(sense_id: str | None) -> str:
    if not sense_id or ":unsensed:" in sense_id:
        return "unsensed"
    return "sense"
