"""Pure letter and ARPAbet derivations backing the v3 wordplay indexes.

Normalization contract (first release is English/CMUdict-backed):

* Input is normalized with NFKC followed by casefold, matching the query
  boundary in ``runtime.normalization``.
* ``normalized_letters`` retains only ASCII ``a-z`` produced by that
  normalization; every other code point (spaces, hyphens, apostrophes,
* digits, diacritics that do not fold to ASCII) is dropped.
* A term is wordplay-eligible only when its NFKC+casefold form already
  consists solely of ASCII ``a-z``.  Terms containing any other character
  are stored for completeness but excluded from the anagram index, so the
  tool never claims phrase or punctuated anagrams.
"""

from __future__ import annotations

import re
import unicodedata

# ARPAbet vowels as used by CMUdict; stress suffixes 0/1/2 are stripped
# before comparison.  Everything else in the ARPAbet inventory is a consonant.
_ARPABET_VOWELS = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AX",
        "AXR",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IX",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
        "UX",
    }
)

_STRESS_SUFFIX = re.compile(r"([012])$")

_ELIGIBLE = re.compile(r"^[a-z]+$")


def _nfkc_casefold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def normalized_letters(value: str) -> str:
    """Return the NFKC+casefold form of ``value`` reduced to ASCII a-z."""

    if not isinstance(value, str):
        raise ValueError("term must be text")
    return "".join(char for char in _nfkc_casefold(value) if "a" <= char <= "z")


def is_wordplay_eligible(value: str) -> bool:
    """True when the normalized form is a non-empty ASCII a-z headword."""

    return bool(_ELIGIBLE.fullmatch(_nfkc_casefold(value.strip())))


def letter_signature(letters: str) -> str:
    """Return the deterministic anagram key: sorted normalized letters."""

    return "".join(sorted(letters))


def reverse_letters(letters: str) -> str:
    """Return the normalized letters in reverse reading order."""

    return letters[::-1]


def is_palindrome(letters: str) -> bool:
    """True when letters of length >= 2 read identically in reverse.

    One-letter inputs are excluded so single code points never surface as
    palindrome candidates.
    """

    return len(letters) >= 2 and letters == letters[::-1]


def split_arpabet_onset(phonemes: str) -> tuple[str, str]:
    """Split a CMUdict phoneme string into ``(onset, remainder)``.

    The onset is every ARPAbet token before the first vowel; vowel-initial
    words have an empty onset.  Stress digits are ignored for classification
    but preserved verbatim in the returned tokens.
    """

    if not isinstance(phonemes, str):
        raise ValueError("phonemes must be text")
    tokens = phonemes.split()
    for index, token in enumerate(tokens):
        bare = _STRESS_SUFFIX.sub("", token)
        if bare in _ARPABET_VOWELS:
            return " ".join(tokens[:index]), " ".join(tokens[index:])
    # Consonant-only pronunciations keep everything in the onset.
    return " ".join(tokens), ""
