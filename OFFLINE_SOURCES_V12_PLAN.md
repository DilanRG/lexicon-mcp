# Roadmap — offline install sources (v1.2.0) and language selection (v2.0.0)

Durable handoff for the two-release plan. v1.2.0 is fully specified and ready to
implement. v2.0.0 is specified down to its open decisions, which are marked as
such so a future session knows what still needs choosing rather than guessing.

Status: **v1.2.0 implemented, awaiting a release tag. v2.0.0 design, not
started — blocked on the open decisions in D6b, D7 and the open questions list.**

---

## 0. Context and principles

Lexicon MCP's offline guarantee is a **runtime** guarantee. The serving process
denies sockets outright (`src/lexicon_mcp/runtime/offline.py:21`) and the README
scopes the claim precisely: "Runtime operation is fully offline." Installation
has always required the network.

v2's install-time language selection turns installation from a single network
event into a recurring one, because changing your language set means fetching new
components. For users who chose an offline MCP deliberately, that is a real
change in exposure even though it violates nothing we have promised. The answer
is to make local and custom install sources a documented, provably network-free
path — and to ship that **before** the sharding work, so air-gapped users are
served on the current corpus and the install-source plumbing v2 depends on is
exercised against real releases first.

Principles carried through both releases:

- **Hashes are the trust boundary.** Anything installed is verified against a
  manifest digest. Any deviation from that model is called out explicitly.
- **Atomic activation is non-negotiable.** A failed or partial operation never
  changes what the runtime serves.
- **Fail loudly, never silently degrade.** Especially: never reach for the
  network when the user asked for local, and never answer "no results" when the
  truth is "not installed".

## Versioning decision

| Stream | Version | Reason |
| --- | --- | --- |
| Package | `1.2.0` | Purely additive: new subcommand, new flag, deprecated alias kept working. No schema change, no behavior change for existing installs. |
| Dataset | unchanged | No manifest schema change in the first slice. |
| Package | `2.0.0` | `profile` stops being the identity of an install, and `--profile english` returns materially different artifacts under the same flag. |
| Dataset | `data-v2.0.0` | Manifest `schema_version` 1 → 2, sharded components. |

Recorded here so it does not get relitigated: the size of the v2 change is not
what makes it major — the changed meaning of an existing flag is.

---

# Part 1 — v1.2.0: local and custom install sources

## What already works (do not rebuild)

Local installation is **implemented but undocumented**:

- `DatasetLifecycle._local_asset_root` (`data/lifecycle.py:504`) resolves a local
  manifest path to its containing directory.
- `DatasetLifecycle._download_part` (`data/lifecycle.py:548`) reads parts
  directly off disk from that directory, with symlink rejection, size bounding,
  SHA-256 verification, and resume support.
- Custom mirrors work through `release.base_url` (`data/lifecycle.py:497`).
- Three tests already cover the local path
  (`tests/test_data_lifecycle.py:356,380,409`), including a `NoNetworkTransport`
  that asserts zero network calls.
- `README.md:229` uses the local path, but only in the maintainer release
  validation section.

The gaps are discoverability, a way to *produce* a local asset set, a misleading
flag name, and one correctness hole (D2).

## Decisions

### D1 — one `--from` flag unifies every source · DECIDED

`install --from VALUE` accepts, in this detection order:

1. a directory containing `manifest.json`
2. a path to a `manifest.json` file
3. an HTTP(S) URL
4. a template containing `{version}` and/or `{profile}`

`--manifest-url` remains a deprecated alias for the same destination, emits a
notice on stderr, and is removed in 2.0.0. Supplying both is an error.

Implementation note: argparse `dest` must be `source`. `from` is a Python
keyword, so `args.from` will not parse.

### D2 — `install --from` must be a hard offline guarantee · DECIDED

Current behavior has a hole. The local branch in `_download_part` is gated on
`part.url is None` (`data/lifecycle.py:548`), so a manifest whose parts carry
explicit `url` fields falls through to HTTP *even when the user pointed at a
local directory*. Published manifests emit part `name`s and no `url`, so this
never fires today — but it is exactly the silent network access the feature
exists to prevent.

Change: when a local asset root is set, always resolve parts by `part.name`; if a
part has no `name`, fail with a clear error instead of falling back to the
network.

### D3 — the mirror layout is the release layout · DECIDED

`fetch --dest DIR` writes `manifest.json` plus one file per part named by its
`part.name` — byte-identical to what `pipeline/manifest.py:package_dataset` emits
and to a GitHub release's flat asset list. One layout, three producers (release,
packaging, `fetch`), one consumer (`install --from`).

### D4 — `fetch` verifies before it names · DECIDED

Each part downloads to `<name>.partial`, is checked against its manifest size and
SHA-256, then atomically renamed to `<name>`. So: resumable, idempotent, and a
file at its final name is always a verified file. `fetch` never touches
`versions/`, never writes `current.json`, and never takes the installation lock.
Free-space preflight runs against `--dest`.

### D5 — reuse the download loop, do not fork it · DECIDED

Extract the HTTP retry/resume/Range logic from `_download_part` into a private
helper used by both `_download_part` (cache paths) and `fetch` (destination
paths). Behavior preserved exactly; the existing resume, retry-backoff, and
`Content-Range` tests are the guard.

## Work items

All implemented. `ruff check .` clean, mypy clean on the touched modules, 244
tests passing (14 new).

### 1. `lifecycle.py`
- [x] Extract `_http_download` from `_download_part`; no behavior change.
- [x] `_local_asset_root` resolves a directory to its `manifest.json` parent,
      via the new module-level `local_manifest_path`.
- [x] Strict local part resolution per D2, in `_copy_local_asset`.
- [x] New `DatasetLifecycle.fetch(...)` per D3/D4, with `_mirror_targets` and
      `_fetch_preflight`.

### 2. `data_cli.py`
- [x] `install --from` with `dest="source"`; `--manifest-url` deprecated alias
      via `_add_source_arguments` / `_resolved_source`.
- [x] `fetch` subcommand: `--profile`, `--version`, `--dest`, `--from`.
- [x] `--from` added to `repair` too, so air-gapped repair works from the same
      media.
- [x] `_manifest_source` no longer runs `.format()` on sources without
      placeholders, so Windows paths cannot be reinterpreted as format fields.

### 3. Tests — `tests/test_data_local_sources.py`
- [x] `install --from <directory>` resolves `manifest.json` inside it.
- [x] Local install with explicit part URLs stays offline (D2 regression).
- [x] Local install fails loudly when a part has no local asset name.
- [x] `fetch` output installs cleanly with zero network calls (round trip).
- [x] `fetch` resumes from a truncated `.partial` (asserts the byte offset).
- [x] `fetch` is idempotent and skips already-valid parts.
- [x] `fetch` rejects a corrupt part, leaving neither it nor a manifest behind.
- [x] `fetch` refuses to mirror a directory onto itself.
- [x] `fetch` touches no part of the dataset root.
- [x] `--manifest-url` still works and reports deprecation.
- [x] `--from` plus `--manifest-url` together is an error.
- [x] Full CLI round trip through `main()`: fetch → install → verify, offline.

### 4. Docs
- [x] README: air-gapped install — `fetch` on a connected machine, transfer,
      `install --from` on the isolated one.
- [x] README: custom mirror via `release.base_url` and `LEXICON_MANIFEST_URL`.
- [x] README: states plainly that `install --from` performs no network access,
      and that the runtime offline guarantee is separate and unchanged.
- [x] README: maintainer clean-install example switched to `--from`.
- [x] `pyproject.toml` version bumped to `1.2.0` so a `v1.2.0` tag does not
      publish a 1.1.0 artifact through `code-release.yml`.

---

# Part 2 — v2.0.0: install-time language selection

## The architecture shift

Today a profile is a **build-time** artifact: `build_dataset(profile="english")`
filters at ingest (`pipeline/wiktextract.py:181`, `pipeline/conceptnet.py:82`),
produces its own `lexicon.sqlite3` and semantic tree, and is packaged as its own
release reached through the `{profile}` slot in the URL template
(`data_cli.py:29`). Supporting arbitrary language combinations on that model
means one pipeline run and one release upload per combination — 2^N over roughly
4,000 language codes. Dead end.

v2 moves selection from build time to install time: **one full build**, packaged
into language-scoped components, with the installer downloading only the subset
requested. Ingest filters do not change at all; only packaging becomes
language-aware. Build cost stays at exactly one run.

## Decisions

### D6 — tier the shards; per-language sharding is impossible · DECIDED

Measured against the installed `data-v1.1.0` corpus (22.62 GiB, 16,513,979
lexical terms):

| Scope | Count |
| --- | --- |
| Languages in `lexical_terms` | **5,508** |
| Wiktextract language codes alone | 5,029 |
| Languages with semantic (Numberbatch) vectors | **78** |

The distribution is severely long-tailed:

| Rank | Share of lexical terms |
| --- | --- |
| top 10 | 52.70% |
| top 25 | 72.22% |
| top 50 | 84.51% |
| top 100 | 92.46% |
| top 200 | 96.49% |
| top 1000 | 99.68% |

- 4,476 languages hold fewer than 100 terms each.
- 5,141 hold fewer than 1,000.
- The 5,408 languages ranked 101 and lower hold 1,244,736 terms — 7.54% total.

One shard per language is therefore not an option. It would produce 5,508 SQLite
files of which 4,476 carry under 100 rows, where fixed per-file overhead (page
size, schema, indexes) dwarfs the content — on the order of hundreds of megabytes
of pure overhead for half a percent of the corpus — and 5,508 components would
push the manifest against the 16 MiB bound in `load_manifest`
(`data/lifecycle.py:168`) that every install must parse.

Decided tiering:

- **Tier 1 — top ~50 languages, individually shardable.** 84.5% of terms, and
  the only languages anyone realistically selects by name.
- **Tier 2 — ranks ~51–200,** grouped into a handful of bundles. Takes coverage
  to 96.5%.
- **Tier 3 — the remaining ~5,300 languages** in one tail bundle: 7.5% of terms,
  small in absolute size.

Result: roughly **55 lexical shards plus one shared core**, not 5,508. For
perspective, the release already ships 79 USearch index files (78 per-language
plus the global), so this is not a step change in release asset count.

**To resolve:** the exact tier-1 cutoff and tier-2 grouping, from measured shard
*bytes* rather than the term counts above.

### D6b — a full install must stay first-class · OPEN

Installing every language must remain supported and must not degrade. Two ways:

**Option 1 — full install uses the shards** like any other selection, just with
all of them. One artifact set to build and verify. Costs whatever sharding
duplication adds (open question 3), paid by the full-install user.

**Option 2 — keep publishing the monolithic `lexicon.sqlite3` alongside the
shards** as the component a full install selects. Today's full install then stays
byte-identical with an unchanged runtime path, which de-risks v2 substantially:
the existing, validated full-corpus path is untouched and only subset installs
exercise new code. Costs roughly double release storage (~22 GiB monolithic plus
the sharded set).

**Leaning: Option 2 for the 2.0.0 release, revisited once sharding overhead is
measured.** Not being able to break the full corpus while shipping this is worth
a lot; release storage is the cheaper currency. Confirm GitHub's practical total
release size first (open question 4).

### D7 — the lexical shard boundary · OPEN — decide before writing any v2 code

This is the decision the rest of v2 hangs on. `lexicon.sqlite3` is one file;
everything else follows from how it is split.

**Hard constraint, measured:** the bundled SQLite (3.49.1) caps attached
databases at **10** — confirmed by `getlimit(SQLITE_LIMIT_ATTACHED)` and by
attaching until failure. Any design that `ATTACH`es one database per installed
language cannot support more than 10 languages. That rules out the most obvious
approach outright.

This cannot be worked around. `SQLITE_MAX_ATTACHED` is a compile-time ceiling
(default 10, hard maximum 125) driven by SQLite's internal `yDbMask` bitmask
representation, per-attachment pager/cache/file-handle cost, and the two-phase
commit needed across multiple attached databases. `sqlite3_limit()` can only
*lower* it, never raise it, so reaching 125 requires a custom SQLite build — and
we run against whatever SQLite the user's Python was linked against. Design
around it.

Two viable options:

**Option A — per-language shard files, multi-connection routing.**
Ship `lexical/{lang}.sqlite3` plus a shared core (provenance, metadata, OEWN,
CMUdict — the latter two English-scoped anyway). Runtime opens one connection per
installed shard and routes queries by language.

- Term IDs are already globally interned, so shards compose without rewriting.
- Most queries are already language-scoped (`runtime/semantic.py:160` filters
  `term.language = ?`), so routing is usually a single-shard lookup, not a
  fan-out. Fan-out is needed only for cross-language operations.
- Every installed file stays covered by a manifest hash. Trust model unchanged.
- **The 10-database cap does not apply here.** It bounds `ATTACH` within a single
  connection; independent connections are limited only by file handles and
  per-connection page cache (~2 MiB default each). With D6 tiering a full install
  is ~55 shards, so open shard connections **lazily on first use with an LRU
  cap** (8 or so) rather than opening every installed shard at startup. Typical
  sessions touch one to three languages.
- Cost: `runtime/service.py` (2,051 lines) grows a routing layer, and
  cross-language queries need explicit merge logic.

**Option B — download shards, merge into one `lexicon.sqlite3` at install time.**
Runtime is untouched; it keeps opening one database.

- Cost: the merged file's hash is not in the manifest. Verification would check
  shard hashes *before* the merge, then record the merged file's local digest in
  the activation record for later tamper detection. That is a real weakening of
  the "hashes are the trust boundary" principle, and it must be a deliberate,
  documented choice if taken.
- Also costs install time and transient disk for the merge, and `add-language`
  becomes a re-merge rather than a file addition.

**Leaning: Option A.** It preserves the trust model, and the language-scoped
nature of existing queries makes the routing layer far smaller than it first
appears. Confirm by auditing how many queries in `runtime/service.py` are
genuinely cross-language before committing.

### D8 — cross-language data rules · RECOMMENDED

Three cases, three different rules:

- **Translations** are directional string pairs, not foreign keys. The English
  build currently drops them when the target language is filtered out
  (`pipeline/wiktextract.py:490-499`). Under install-time selection, keep them in
  the source-language shard: an `en`-only install can then still answer "what is
  the French for *dog*", which today's English profile cannot. The subset becomes
  strictly better than the profile it replaces.
- **ConceptNet edges** are today filtered to both-endpoints-in-set
  (`pipeline/conceptnet.py:82`). Pair-sharding is O(N²) components — do not.
  Write each cross-lingual edge into both endpoint shards and filter at query
  time against the installed set. Costs some duplication, keeps components O(N).
- **Semantic vectors** live in one cross-lingual Numberbatch space, so merging
  results across installed languages' indexes is mathematically sound. But
  `semantic/vectors.bin` and `semantic/mapping.sqlite3` are monolithic and must
  be sharded too, or index selection buys nothing. `global.usearch` becomes an
  optional component worth downloading only for a full install.

### D9 — manifest schema v2 · DECIDED

- `schema_version: 2`, with per-component `languages` (absent = shared core,
  always installed).
- Language set becomes the identity of an install; `profile` is retained as a
  derived compatibility field.
- v1.1/v1.2 installers reject unknown schema versions cleanly
  (`data/manifest.py:428`), so the forward-compat story is free.
- **New code must keep parsing schema v1** so existing installs keep working for
  `status`/`verify`/`repair`/`rollback` after upgrade.

### D10 — verification targets the selected set, not the manifest · DECIDED

`_verify_path` walks every manifest component and reports absent ones as missing
(`data/lifecycle.py:791`). With subsetting, absent is legal. The selected
language set and resolved component ID set must be recorded in the activation
record and become the verification target — otherwise every partial install
reports damaged forever. `repair` follows the same set.

This is the single most likely source of subtle breakage in v2. Test it first.

### D11 — `add-language` without re-downloading · RECOMMENDED

Falls out of sharding almost free, and is what users will actually want. It
collides with atomic activation, which swaps a whole version directory. Preserve
the invariant by building the new version directory from **hardlinks** to the
retained components plus the newly downloaded shards, then swapping. Same-volume
NTFS handles this; verify behavior when the data root spans volumes.

For air-gapped users this composes with v1.2.0: `fetch --languages de` on a
connected machine carries only the new shards across.

### D12 — runtime honesty contract · DECIDED

A `de` query against an `en,fr` install must return "language not installed", not
an empty result that reads as "word does not exist". The installed set must be
visible through the MCP tool responses, not just `status`. This is a correctness
requirement, not a nicety — a silently empty answer is a wrong answer.

### D13 — build gates become per-language · DECIDED

`FULL_CORPUS_FLOORS` / `ENGLISH_CORPUS_FLOORS`
(`pipeline/orchestrator.py:47,73`) and the 30 GiB `INSTALLED_LIMIT`
(`pipeline/size_estimator.py:13`) are per-profile constants. They must become
per-language recorded counts in the manifest, with the installed-size gate
checked against the selected set rather than the whole corpus.

### D14 — retiring the English profile · DECIDED

`--profile english` becomes an alias for `--languages en`, and the dedicated
English *build path* is retired rather than maintained alongside sharding. The
artifacts genuinely differ (the English build collapses the global index onto the
en index and drops outbound translations), so this is a replacement, not a
re-label — which is precisely why the package goes to 2.0.0. Existing English
installs keep working: releases are immutable and schema v1 parsing is retained.

## Open questions to resolve before implementation

1. **D7 shard boundary** — Option A or B. Audit cross-language query surface in
   `runtime/service.py` first.
2. **D6 tier-1 cutoff** — derive from measured per-language sizes.
3. **Sharding overhead** — measure how much total release size grows from
   duplicated cross-lingual edges and interned strings. Expect 10–20%; confirm.
4. **Release asset count** — confirm GitHub's practical limits at the component
   count D6 produces, including part splitting.
5. **`add-language` across volumes** — hardlink fallback when the data root and
   staging directory differ.

## Work phases

- **P0** — resolve D7 and D6; measure per-language sizes. No code.
- **P1** — packaging emits sharded components; manifest schema v2; build once,
  shard at package time.
- **P2** — installer selection, D10 verification against the selected set,
  `--languages` on `install` and `fetch`.
- **P3** — runtime routing (D7) and the D12 honesty contract.
- **P4** — `add-language` / `remove-language` (D11).
- **P5** — retire the English build path (D14); per-language gates (D13).

## Deferred to v2, not v1.2.0

**`lexicon-data languages` discovery.** Without sharded components there is no
per-language size to report, and today's honest answer splits awkwardly:
`semantic_languages` in `semantic/mapping.sqlite3` carries precomputed
per-language term counts but covers only languages with Numberbatch vectors,
while the lexical corpus spans roughly 4,000 Wiktextract language codes.
Reporting either number alone would mislead. The command becomes genuinely useful
in P1, when components are language-scoped and sizes are exact.

**Recording the language set in the activation record.** Until components are
subsettable this copies `manifest.languages` into `current.json` and changes
nothing. It becomes load-bearing in P2 (see D10).
