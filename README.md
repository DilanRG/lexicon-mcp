# Lexicon MCP

Lexicon MCP is a local-first multilingual dictionary, thesaurus, translation,
lexical-relations, semantic-neighbour, and English rhyme-search server for the Model
Context Protocol (MCP). It serves a versioned corpus from disk and performs no
network access or data mutation while running.

> [!IMPORTANT]
> The software and corpus are licensed separately. The Python source is Apache-2.0.
> Corpus components retain their upstream licenses; see [DATA_LICENSES.md](DATA_LICENSES.md).

## Model-visible tools

The server exposes exactly seven tools:

1. `dictionary_lookup`
2. `dictionary_synonyms`
3. `dictionary_translate`
4. `dictionary_relations`
5. `dictionary_semantic_neighbors`
6. `rhymes`
7. `wordplay`

Relation results label direct edges as `relation_scope="direct"`, `distance=1`.
Hypernym and hyponym queries can additionally return a bounded, homogeneous
two-edge expansion labelled `relation_scope="transitive"`, `distance=2`; its
`path` contains both directed edges with their exact sense scope and provenance.

`dictionary_lookup` uses `limit` for returned senses and has independent total
response budgets for examples, pronunciations, and translations. Their defaults
are `8`, `8`, and `20`; each is shared round-robin across the returned senses and
`0` disables that detail class. Every sense reports `truncated_fields`, so a fixed
budget never silently presents a partial detail list as complete.

`dictionary_translate` inspects up to `max_senses=100` source-native senses by
default, independently of its total translation-candidate `limit`. Translation
candidates are distributed round-robin across matching source-sense groups.
Grouped synonym and translation responses retain `count` for the number of groups
and also report `candidate_count` for their nested candidates.

`dictionary_synonyms` accepts `max_senses` (the number of source-native lexical
senses inspected) and `unsensed_limit`; `dictionary_relations` accepts `max_depth`
(one or two relation-graph hops) and `transitive_limit`. In both tools, `limit` is
the total returned-candidate cap. An allocation of `0` disables the broader class,
a positive value requests an explicit bounded allocation, and the default is `5`.
Allocations cannot exceed `limit`; automatic allocation is reserved for a future
release. The sentinel `-1` is reserved for future benchmark-tuned automatic
allocation and is not accepted in v1; `0` disables the allocated class. Unused
broader allocation is returned to sense-scoped or direct candidates.

For semantic neighbours, omitting `target_language` searches the global
multilingual index; setting it to the source tag produces monolingual results,
and another tag requests cross-lingual results. English `rhymes(mode="near")` is
fixed to exactly one ARPAbet-token insertion, deletion, or substitution in v1;
there is no automatic or caller-configurable edit distance.

`wordplay(text, kind, context=None, limit=20)` answers exactly one requested
kind per call, `kind` being `anagram`, `palindrome`, `spoonerism`, or `pun`.
There is no automatic mode and no `-1`; `limit` follows the same `1..100`
policy as every other tool, and `context` (1..512 characters) is rejected for
every kind except `pun`.

* **anagram** returns English headwords whose normalized letters (NFKC +
  casefold, ASCII `a-z` only) are a reordering of the query's letters. Only
  strictly alphabetic single-token headwords are eligible, so phrase or
  punctuated anagrams are never claimed, and the query itself is excluded.
* **palindrome** reports `input_is_palindrome` for the query and returns
  stored corpus palindromes other than the query. Candidates are enumerated
  deterministically in `(normalized_letters, term_id)` index order starting
  at the query's letters and wrapping once; every candidate's normalized
  letters read identically in reverse. One-code-point inputs yield no
  candidates.
* **spoonerism** requires exactly two whitespace-separated English headwords.
  It exchanges their initial ARPAbet consonant clusters (the onset is every
  consonant token before the first vowel; vowel-initial words have an empty
  onset) using at most eight CMU pronunciation alternatives per word. Swapped
  outputs are pronunciation-derived phrases labelled
  `lexicality_scope="generated_candidate"` unless both swapped pronunciations
  resolve to corpus headwords (then `"lexical_term"` and every resolved
  headword is reported, since one pronunciation can spell several words).
  Empty-to-empty and identical onsets are never swapped.
* **pun** returns exact CMUdict homophones of the query that carry at least
  one source-native sense distinct from the query term's senses, labelled
  `result_class="candidate"` with `sound_relation="homophone"`; it is never
  claimed to be a finished joke. Without `context` the response is labelled
  `context_scope="uncontextualized"`. Meanings are never inferred from vector
  similarity.

Every wordplay result carries a `provenance` array: spelling and senses come
from Open English WordNet or Wiktionary via Wiktextract entries where present,
and phonetic derivations come from the CMU Pronouncing Dictionary. The first
wordplay-capable dataset is `data-v1.1.0` (lexical schema 3); `data-v1.0.0`
artifacts do not contain the wordplay indexes and are rejected by this server
version.

Dataset installation, verification, repair, and rollback are deliberately CLI-only.

## Local development

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Run the MCP server after installing a verified dataset:

```powershell
$env:LEXICON_DATA_DIR = 'E:\AI\data\lexicon-mcp'
uv run --frozen lexicon-mcp
```

The server never downloads data. If no verified corpus is active, it exits with a
diagnostic containing the exact `lexicon-data install` command.

## Dataset lifecycle

Install only the languages and capabilities you want:

```powershell
lexicon-data install --version data-v2.0.0 --languages en --capabilities lexical,semantic,wordplay
lexicon-data install --version data-v2.0.0 --languages en,fr,de --capabilities lexical
lexicon-data install --version data-v2.0.0 --all-languages

lexicon-data languages           # coverage, and what this install serves
lexicon-data add-language --version data-v2.0.0 --languages fr
lexicon-data remove-language --version data-v2.0.0 --languages fr
lexicon-data status
lexicon-data verify
lexicon-data activate --activation <id>   # switch back to a retained selection
lexicon-data prune                        # reclaim unreferenced components
```

The corpus carries **5,508 lexical languages**, of which 78 also have semantic
vectors, and English additionally has pronunciation and wordplay indexes. Those
are selected independently, so "every language, English vectors only" is a valid
install. A selection is never silently narrowed: a language the release does not
carry fails outright, while one that simply lacks the capability you asked for is
reported alongside the install.

Typical footprints:

| Selection | Download | Installed |
| --- | --- | --- |
| English lexical | 0.60 GiB | 2.01 GiB |
| English + semantic + wordplay | 1.11 GiB | 2.88 GiB |
| All 5,508 lexical languages | 2.67 GiB | 8.24 GiB |

`LEXICON_DATA_DIR` selects the installation root. Releases are immutable.
Components are stored by content hash, so a component two selections share is
held once, `add-language` fetches only what is genuinely new, and switching back
to an earlier selection is a pointer swap rather than a re-download. Every
mutation ends in an atomic swap, so an interrupted operation leaves the previous
install exactly as it was. Downloads resume into `.partial` files and are
hash-checked before anything is activated.

Verification is scoped to what you installed rather than to the whole release,
so a deliberately partial install is not reported as damaged.

### Air-gapped and mirrored installation

Runtime operation is fully offline, but installation normally downloads its
release. To install without ever putting the target machine on a network, mirror
the release somewhere connected and carry it across:

```powershell
# On a connected machine
lexicon-data fetch --profile full --version data-v1.0.0 --dest E:\transfer\data-v1.0.0

# On the isolated machine, after copying the directory across
lexicon-data install --profile full --version data-v1.0.0 --from E:\transfer\data-v1.0.0
```

`fetch` writes the exact published release layout — `manifest.json` beside one
file per part — and verifies every part against its manifest SHA-256 before
giving it its final name. It is resumable and idempotent: rerun it after an
interruption and it continues, skipping whatever is already valid. It never
writes to the dataset root, never activates anything, and the manifest is written
last, so an interrupted mirror fails loudly on install rather than looking
complete.

**`install --from` performs no network access at all.** Every part is resolved on
disk; a release that cannot be satisfied locally is an error rather than a silent
fall back to the network. Because the transferred assets carry their manifest
hashes, the transfer itself is integrity-checked end to end.

`--from` accepts a mirror directory, a `manifest.json` path, an HTTP(S) URL, or a
template containing `{version}` and `{profile}`. `repair` accepts it too, so a
damaged air-gapped install can be repaired from the same media. To install from a
self-hosted mirror instead, publish the release with `base_url` set and point
`--from` at its `manifest.json`; `LEXICON_MANIFEST_URL` sets a default template.

`--manifest-url` remains as a deprecated alias for `--from` and is removed in
2.0.0.

### Asking for something you did not install

A language you did not install is never confused with a word that does not
exist. Every tool distinguishes these, and says which:

| Reason | Meaning |
| --- | --- |
| `language_not_installed` | the corpus has it; install it and the query works |
| `unknown_language` | the corpus never had it |
| `capability_not_installed` | the language is installed, that capability was not selected |
| `not_available_upstream` | the corpus has no such data for that language at all |

Semantic search over a subset reports which languages it actually searched and
whether the result was restricted, because an unrestricted search on a full
corpus covers all 78 vector languages and on a subset covers what is installed.

Relation results also mark whether a target's own entry is installed, so a
translation into a language you do not have is still returned and still usable
-- it simply cannot be expanded further.

## Corpus

The full build combines independently attributed snapshots of:

- Open English WordNet 2025 for English synsets and lexical relations;
- the English Wiktionary Wiktextract/Kaikki raw dump for multilingual entries,
  senses, examples, pronunciation, etymology, and translations;
- ConceptNet 5.7 for multilingual lexical and commonsense relations;
- ConceptNet Numberbatch 19.08 for multilingual semantic neighbours; and
- CMUdict for English pronunciation and rhyme search.

Every public result identifies its source, dataset version, language, sense scope,
and license. ConceptNet-only results are explicitly unsensed.

Lexical data uses a compact, interned SQLite schema with deferred read-path indexes.
English rhyme search and internal prefix completion use a contentless FTS5 index; exact senses,
translations, relation direction, and provenance remain in ordinary relational tables.
Numberbatch vectors are searched through memory-mapped cosine/i8 USearch HNSW indexes
and exact-reranked from float16 vectors without loading or scanning the full matrix.

## Data builds and releases

`sources.lock.json` freezes exact source revisions and hashes. The build pipeline is
streaming and checkpointed; it does not build the unused Google n-gram database from
the earlier community project. Data artifacts are packaged into independently hashed
parts below 1 GiB and published under a separate immutable `data-v*` release.

The production build accepts only already-downloaded inputs matching the pinned byte
hashes, logical row counts, and logical row digests. Recheck them before a release build:

```powershell
uv run --frozen python scripts/build_source_lock.py verify `
  --lock sources.lock.json `
  --source 'oewn=E:\AI\state\lexicon-mcp-build\sources\oewn-2025.xml.gz' `
  --source 'wiktextract=E:\AI\state\lexicon-mcp-build\sources\wiktextract-en-2026-08-12.jsonl.gz' `
  --source 'conceptnet=E:\AI\state\lexicon-mcp-build\sources\conceptnet-assertions-5.7.0.csv.gz' `
  --source 'numberbatch=E:\AI\state\lexicon-mcp-build\sources\numberbatch-19.08.txt.gz' `
  --source 'cmudict=E:\AI\state\lexicon-mcp-build\sources\cmudict.dict'
```

Build into staging, not into the live installation root. On the production Windows
host, the wrapper below runs the same frozen command and records free-disk, private
memory, and working-set telemetry throughout the build:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_full_build.ps1
```

For the compact English profile, use distinct output/version paths:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_full_build.ps1 `
  -Profile english `
  -DatasetVersion data-en-v1.0.0 `
  -Output E:\AI\state\lexicon-mcp-build\built\data-en-v1.0.0
```

The equivalent direct command is:

```powershell
uv run --frozen python scripts/build_full_corpus.py `
  --oewn E:\AI\state\lexicon-mcp-build\sources\oewn-2025.xml.gz `
  --wiktextract E:\AI\state\lexicon-mcp-build\sources\wiktextract-en-2026-08-12.jsonl.gz `
  --conceptnet E:\AI\state\lexicon-mcp-build\sources\conceptnet-assertions-5.7.0.csv.gz `
  --numberbatch E:\AI\state\lexicon-mcp-build\sources\numberbatch-19.08.txt.gz `
  --cmudict E:\AI\state\lexicon-mcp-build\sources\cmudict.dict `
  --source-lock E:\AI\lexicon-mcp\sources.lock.json `
  --notices-dir E:\AI\lexicon-mcp `
  --output E:\AI\state\lexicon-mcp-build\built\data-v1.0.0 `
  --build-state E:\AI\state\lexicon-mcp-build `
  --dataset-version data-v1.0.0
```

The direct English build uses the same pinned inputs and adds
`--profile english`, with `data-en-v1.0.0` supplied for both `--output` and
`--dataset-version`.

Package with the exact clean transformation commit, then validate a clean offline
install from the same release bundle before uploading it:

```powershell
uv run --frozen python scripts/package_data.py `
  --dataset E:\AI\state\lexicon-mcp-build\built\data-v1.0.0 `
  --output E:\AI\state\lexicon-mcp-build\release\data-v1.0.0 `
  --dataset-version data-v1.0.0 `
  --repository DilanRG/lexicon-mcp `
  --tag data-v1.0.0 `
  --transformation-commit <40-character-commit>

uv run --frozen lexicon-data --data-dir E:\AI\data\lexicon-mcp install `
  --profile full --version data-v1.0.0 `
  --from E:\AI\state\lexicon-mcp-build\release\data-v1.0.0
```

Release acceptance is separate from ordinary CI because it reads the full activated
corpus with networking denied:

```powershell
uv run --frozen pytest -m full_corpus -ra
uv run --frozen pytest -m ann -ra
uv run --frozen pytest -m performance -ra
```

The Windows live-stack gate has a separate, explicit runner. It refuses to touch
services unless `--execute-live` is present, always performs exactly ten cycles,
and leaves append-only JSONL evidence plus a final JSON report:

```powershell
uv run --frozen --project E:\AI\lexicon-mcp python `
  E:\AI\lexicon-mcp\scripts\run_live_acceptance.py `
  --execute-live `
  --base-report E:\AI\state\lexicon-mcp-build\acceptance\corpus-gates.json
```

Each cycle proves the old MCPO root and children exited, obtains exclusive handles
to every active dataset artifact while stopped, starts through the normal
`E:\AI\scripts` entrypoints, checks router/Open WebUI/MCPO health, validates the
exact six-operation Lexicon OpenAPI surface, invokes a lightweight Lexicon lookup
and Calculator in every cycle, and finishes with `active_models=[]`. In the first
post-restart cycle it also invokes and validates all six Lexicon tools through MCPO:
`bank` lookup selects unique Wiktionary river and financial senses by source and
gloss, then passes each exact sense ID into a separate German translation call. The
river call must return `Ufer` and the financial call must return `Bank`; acceptance
does not depend on either term appearing in lookup's bounded embedded-translation
page. The lookup itself uses a total `translations_limit=3`, proves the aggregate
budget is respected, verifies every returned translation remains attached to its
source sense, and requires per-sense `truncated_fields` evidence. Synonyms, directed
relations, language-filtered finite-cosine semantic neighbours, and query-excluding
rhymes are checked against pinned corpus anchors. The full cross-tool flow,
request/result hashes, both selected sense IDs, and per-tool assertions are written
to both JSONL events and the final report.

Recursive directory notifications plus baseline, per-cycle, and final inventories
reject dataset or project-venv rewrites, including short-lived transient files.
Full content fingerprints are calculated before and after the run. The runner checks
Open WebUI health only: it does **not** claim that an ordinary chat prompt selected
or invoked a tool. That UI-level prompt evidence must be captured separately during
live acceptance. A fixture replay is available for safe runner validation, but its
report deliberately records `live_stack_ok=false`, labels all six-tool calls as
`fixture-replay`, and cannot satisfy publication:

```powershell
uv run --frozen --project E:\AI\lexicon-mcp python `
  E:\AI\lexicon-mcp\scripts\run_live_acceptance.py `
  --dry-run-fixture E:\AI\lexicon-mcp\tests\fixtures\live_acceptance\happy.json
```

`scripts/publish_data_release.py --stage` creates or resumes a draft and verifies every
remote asset. `--publish` additionally requires an acceptance report tied to the exact
manifest hash and refuses publication unless clean-install, offline, corpus, ANN,
performance, live-stack, ten-restart, and final-model-unload gates are all recorded.

Code and data releases are intentionally separate:

- code: `v1.0.0`
- data: `data-v1.0.0`

The code release pins an exact compatible data release and never resolves `latest`.

## Prior art

This is an independent clean-room implementation. The project idea was informed by
[`Eyalm321/multilingual-dictionary-mcp`](https://github.com/Eyalm321/multilingual-dictionary-mcp),
which is acknowledged as prior art. No source code or Git history was copied.

## Security and privacy

- Runtime databases are opened read-only and query-only.
- SQL uses bound parameters; query text is never interpreted as SQL.
- Tool inputs and result counts are bounded.
- Runtime operation is fully offline.
- Dataset administration is never exposed to the model.
