# v1.1 Actual Wordplay Handoff

## Decision and boundary

Do **not** expose a `wordplay` MCP tool against the released `data-v1.0.0`
artifact.  The current read-only schema has `lexical_terms`, source-native
senses, and CMU pronunciations, but none of the bounded reverse indexes needed
for anagrams or phoneme-onset swaps.  A table scan of all terms per call is not
an acceptable hidden latency/memory contract.  `pun` also cannot truthfully be
returned as a deterministic fact from CMUdict: it needs two distinguishable
senses (and, ideally, a supplied context), while CMUdict pronunciations are
unsensed.

`rhymes(text, mode="exact"|"near", limit=20)` remains exactly as released in
code commit `57f5caf`; do not restore the misleading `dictionary_wordplay`
public name.  The existing private `SQLiteWordplaySearch` module can be split
or renamed internally later without API impact.

This is a **new data release**, not a code-only change.  Preserve immutable
v1.0.0 and ship code `v1.1.0` with a matching `data-v1.1.0` manifest/artifact.

## Proposed public MCP schema

Add one seventh tool (do not overload `rhymes`):

```python
WordplayKind = Literal["anagram", "palindrome", "spoonerism", "pun"]

wordplay(
    text: QueryText,
    kind: WordplayKind,
    context: Annotated[str, Field(min_length=1, max_length=512)] | None = None,
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 20,
) -> dict[str, Any]
```

There is no automatic mode and no `-1`: Pydantic and runtime validation must
reject it with the same `1..100` policy as `rhymes`.  `context` is permitted
only for `pun`; reject it for the other kinds rather than silently ignoring it.
For `pun`, context is optional, but absent context must be reported as
`context_scope: "uncontextualized"`, never claimed to be a finished joke.

Common response envelope:

```json
{
  "type": "wordplay",
  "dataset_version": "1.1.0",
  "query": {"text":"...", "normalized_text":"...", "kind":"...", "context":"...|null", "limit":20},
  "count": 0,
  "results": []
}
```

Candidate shapes, all deterministically ordered and all with explicit
provenance:

* `anagram`: `{term, normalized_term, signature, language:"en", explanation:
  "same normalized letters", provenance:[...]}`.  Exclude the query term;
  only alphabetic normalized English headwords, no spaces/hyphens, to avoid
  claiming phrase anagrams.
* `palindrome`: `{term, normalized_term, palindrome_key, language:"en",
  explanation:"normalized letters read identically in reverse", provenance:[...]}`.
  This is a corpus lookup, not a generated assertion.  Exclude the query and
  require at least two code points to avoid one-letter noise.
* `spoonerism`: `{left:{term, phonemes}, right:{term, phonemes}, swapped_left,
  swapped_right, onset_left, onset_right, language:"en", explanation:
  "initial consonant clusters exchanged", provenance:[CMU...]}`.  Returned
  swapped outputs are pronunciation-derived candidate phrases, labelled
  `lexicality_scope: "generated_candidate"`; they are not dictionary words
  unless both generated keys resolve in `lexical_terms` (then state that fact).
  Require exactly two whitespace-separated English headwords and reject any
  other form.
* `pun`: `{term, phonemes, query_sense_ids:[...], candidate_sense_ids:[...],
  sound_relation:"homophone"|"near_homophone", context_scope, explanation,
  provenance:[CMU..., OEWN/Wiktextract...]}`.  Label it `candidate`, not a
  joke.  Only emit if the sound-alike term has at least one source-native sense
  that differs from a source-native sense on the query term; do not infer
  meanings from vector similarity.

`provenance` is an array because spelling/senses and phonetics have different
sources.  Each entry retains `source`, `license`, and `url` as existing output
does.  Never fabricate an explanation beyond the stored/derived relation.

## Schema v3 / data indexes

Increment `SCHEMA_VERSION` from `2` to `3`; update runtime schema validation.
Retain all v2 tables and add these post-import tables/indexes:

```sql
CREATE TABLE wordplay_terms (
  term_id INTEGER PRIMARY KEY REFERENCES lexical_terms(term_id),
  normalized_letters TEXT NOT NULL,
  letter_signature TEXT NOT NULL,
  reverse_letters TEXT NOT NULL,
  is_palindrome INTEGER NOT NULL CHECK (is_palindrome IN (0,1)),
  wordplay_eligible INTEGER NOT NULL CHECK (wordplay_eligible IN (0,1))
) WITHOUT ROWID;
CREATE INDEX wordplay_terms_anagram
  ON wordplay_terms(letter_signature, normalized_letters, term_id)
  WHERE wordplay_eligible = 1;
CREATE INDEX wordplay_terms_palindrome
  ON wordplay_terms(normalized_letters, term_id)
  WHERE is_palindrome = 1;

CREATE TABLE pronunciation_onsets (
  term_id INTEGER NOT NULL REFERENCES lexical_terms(term_id),
  phonemes TEXT NOT NULL,
  onset TEXT NOT NULL,
  remainder TEXT NOT NULL,
  PRIMARY KEY (term_id, phonemes)
) WITHOUT ROWID;
CREATE INDEX pronunciation_onsets_lookup
  ON pronunciation_onsets(onset, remainder, term_id);
CREATE INDEX pronunciation_onsets_reverse
  ON pronunciation_onsets(remainder, onset, term_id);
```

Do not create a broad all-pairs spoonerism table: its cardinality can explode.
Two indexed source-word lookups plus bounded pronunciation alternatives are
enough.  Define onset as all ARPAbet consonant tokens before the first vowel;
vowel-initial words have empty onset.  Exclude empty-to-empty swaps and return
at most one deterministic pronunciation pairing per output phrase.

No separate `pun` table is initially required: query exact phoneme matches via
the existing `pronunciations_words_phonemes` index, join `lexical_entries` /
`senses`, and apply a bounded SQL `LIMIT`.  Near homophones, if implemented,
must use a separate, specified phoneme edit distance with precomputed index;
do not reuse broad `near` rhyme keys as homophones.

## Build changes

1. Add pure functions in `pipeline/wordplay.py`:
   `normalized_letters`, `letter_signature`, `is_palindrome`, and
   `split_arpabet_onset`.  Normalization must be documented: NFKC + casefold;
   eligibility is ASCII `a-z` only after normalization (the first release is
   explicitly English/CMUdict-backed).
2. Extend `pipeline/schema.py` with v3 DDL and `create_wordplay_indexes()`;
   invoke it in `orchestrator.py` after all term and CMU imports, alongside
   `create_lexical_query_indexes()`.
3. Populate `wordplay_terms` by streaming `lexical_terms WHERE language='en'`;
   do not hold the term corpus in Python.  Populate `pronunciation_onsets` by
   streaming `pronunciations_words` in batches.  Record their row counts in
   the build report and validate foreign-key/integrity checks.
4. Add dataset metadata: `wordplay_index_version=1`, eligible-term count,
   palindrome count, onset-row count, and exact source lock hashes.  Include
   the new behavior in README/data licenses; CMUdict continues to cover
   phonetic derivations, while OEWN/Wiktextract provenance covers senses.
5. Re-run the normal full build, deterministic rebuild comparison, package,
   manifest/SHA256 generation, and immutable GitHub data release flow.  Do not
   mutate `data-v1.0.0`.

### Sidecar/migration alternative

If a full lexical corpus rebuild is too expensive, build a read-only sidecar
from an already verified `lexicon.sqlite3` and package it atomically as part of
`data-v1.1.0`:

```
wordplay-v1.sqlite3
  wordplay_terms
  pronunciation_onsets
  metadata(parent_lexical_sha256, schema_version, index_version)
```

The runtime must verify `parent_lexical_sha256` against the installed
`lexicon.sqlite3` before attaching/opening it; otherwise return a clear
`wordplay indexes unavailable or incompatible` error, never partial results.
The sidecar still requires a new signed/hashed data release, but avoids
reimporting upstream sources.  Prefer a single v3 database if size calibration
shows it remains within the established artifact budget.

## Runtime plan

* Add `SQLiteActualWordplaySearch` (or extend the current class without mixing
  rhyme modes) with read-only immutable SQLite and schema/index validation.
* `anagram` performs `letter_signature = ?`, excludes same normalized key,
  groups deterministic duplicate spellings, and uses SQL limit.
* `palindrome` requires that the input itself passes normalized-letter rules;
  it returns corpus palindromic alternatives only, with a primary direct
  `input_is_palindrome` boolean in `query` if desired.  Do not return the input
  itself as a fake candidate.
* `spoonerism` resolves both terms and their CMU alternatives, limits source
  alternatives before Cartesian work (e.g. 8 each), swaps onsets, and uses
  exact indexed phoneme/key lookup for validation.  Bound every loop by
  `limit` and a fixed internal pairing cap documented in code.
* `pun` uses only exact homophones in the first release.  Fetch no more than
  `limit * 4` candidate term/sense rows, then filter and return `limit`; use
  sense IDs and provenance rather than summaries generated at runtime.
* Add `LexiconService.wordplay`, FastMCP `wordplay`, and update tool-list/live
  acceptance expectations from six to seven.  Preserve `rhymes` unchanged.

## Required tests and benchmarks

Unit/fixture tests:

* unicode normalization and eligibility; `listen/silent/enlist` anagram group;
  self exclusion; deterministic ordering; `-1`, `0`, `101`, context misuse.
* corpus palindrome candidates such as `level`, but fixture rows must include
  only verified dictionary terms; reject punctuation/one-character examples.
* ARPAbet onset extraction (`K AE1 T`, `S T R IY1 T`, vowel initial) and a
  fixture `light rain -> right lane`-style swap only if its phonemes and terms
  are explicitly present; assert generated-candidate labelling.
* homophone pun candidate with two fixture sense IDs; ensure same-sense or
  missing-sense rows are suppressed; assert no claim that output is a joke.
* SQLite schema mismatch/missing index and sidecar parent-hash mismatch fail
  closed; all provenance fields are present.
* protocol test confirms exactly seven public tools and `rhymes` remains named
  `rhymes`.

Performance acceptance on the full packaged artifact (cold then warm, median
and p95): record DB size delta, RSS delta, and each kind at `limit=1,20,100`.
Set conservative gates before release (suggest p95 <=100 ms warm and <=500 ms
cold per lookup on the target machine; fail rather than silently relaxing).
Add adversarial high-fanout anagram and high-pronunciation pair cases.  Verify
read-only mode makes no database files and inspect `EXPLAIN QUERY PLAN` to
confirm the indexes above are used.

## Rollout and compatibility commands

Run from `E:\AI\lexicon-mcp` after implementation (adapt established release
flags rather than inventing a second release process):

```powershell
uv run pytest tests/test_runtime_wordplay_compact.py tests/test_runtime_mcp.py tests/test_pipeline_builders.py -q
uv run mypy src
uv run python scripts/build_full_corpus.py --help
uv run python scripts/package_data.py --help
uv run python scripts/publish_data_release.py --help
uv run python scripts/run_live_acceptance.py --help
```

Then build/package/publish `data-v1.1.0` using the repository's existing pinned
source-lock workflow, install it into an isolated fresh data directory, and
run the normal live acceptance runner against the actual MCPO/Open WebUI
connection.  Manually ask the live UI separately for: an anagram of `listen`,
a palindrome candidate, a spoonerism of two known fixture-supported words, and
a context-labelled pun candidate.  Capture the actual MCP tool calls/results;
do not accept model prose as proof.  Finally restore/unload any local model as
the current stack hygiene requires.

## Suggested implementation order

1. Implement pure derivation helpers and fixtures/tests.
2. Implement v3 schema/index population and test a tiny fixture build.
3. Implement anagram + palindrome runtime and benchmarks.
4. Implement bounded spoonerism.
5. Implement exact-homophone sense-backed pun candidates.
6. Run full build/release gates and live stack acceptance; tag code `v1.1.0`
   only after the matching immutable data release is available.

