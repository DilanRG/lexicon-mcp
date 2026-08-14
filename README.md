# Lexicon MCP

Lexicon MCP is a local-first multilingual dictionary, thesaurus, translation,
lexical-relations, semantic-neighbour, and English wordplay server for the Model
Context Protocol (MCP). It serves a versioned corpus from disk and performs no
network access or data mutation while running.

> [!IMPORTANT]
> The software and corpus are licensed separately. The Python source is Apache-2.0.
> Corpus components retain their upstream licenses; see [DATA_LICENSES.md](DATA_LICENSES.md).

## Model-visible tools

The server exposes exactly six tools:

1. `dictionary_lookup`
2. `dictionary_synonyms`
3. `dictionary_translate`
4. `dictionary_relations`
5. `dictionary_semantic_neighbors`
6. `dictionary_wordplay`

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
and another tag requests cross-lingual results. English `near_rhyme` wordplay is
fixed to exactly one ARPAbet-token insertion, deletion, or substitution in v1;
there is no automatic or caller-configurable edit distance.

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

```powershell
lexicon-data install --profile full --version data-v1.0.0
lexicon-data status
lexicon-data verify
lexicon-data repair
lexicon-data rollback
```

`LEXICON_DATA_DIR` selects the installation root. Releases are immutable and the
active dataset is selected through an atomically replaced `current.json` pointer.
Downloads resume into `.partial` files, are hash-checked before extraction, and are
activated only after database/index integrity checks.

## Corpus

The full build combines independently attributed snapshots of:

- Open English WordNet 2025 for English synsets and lexical relations;
- the English Wiktionary Wiktextract/Kaikki raw dump for multilingual entries,
  senses, examples, pronunciation, etymology, and translations;
- ConceptNet 5.7 for multilingual lexical and commonsense relations;
- ConceptNet Numberbatch 19.08 for multilingual semantic neighbours; and
- CMUdict for English pronunciation and wordplay.

Every public result identifies its source, dataset version, language, sense scope,
and license. ConceptNet-only results are explicitly unsensed.

Lexical data uses a compact, interned SQLite schema with deferred read-path indexes.
English wordplay and prefix completion use a contentless FTS5 index; exact senses,
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
  --manifest-url E:\AI\state\lexicon-mcp-build\release\data-v1.0.0\manifest.json
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
wordplay are checked against pinned corpus anchors. The full cross-tool flow,
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
