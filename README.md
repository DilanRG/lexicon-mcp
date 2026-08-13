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

## Data builds and releases

`sources.lock.json` freezes exact source revisions and hashes. The build pipeline is
streaming and checkpointed; it does not build the unused Google n-gram database from
the earlier community project. Data artifacts are packaged into independently hashed
parts below 1 GiB and published under a separate immutable `data-v*` release.

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

