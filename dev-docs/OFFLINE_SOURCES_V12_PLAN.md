# Roadmap — offline install sources (v1.2.0) and language selection (v2.0.0)

Durable handoff for the two-release plan. v1.2.0 is fully specified and ready to
implement. v2.0.0 is specified down to its open decisions, which are marked as
such so a future session knows what still needs choosing rather than guessing.

Status: **v1.2.0 released (tag `v1.2.0`). v2.0.0 architecture settled by
`LEXICON_V2_ARCHITECTURE_RECOMMENDATION.md`, which supersedes the D6b/D7/D10/D11
decisions below — see "Accepted architecture" at the end of this document for the
delta and the measured refinement to its §2.**

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

### D7 — the lexical shard boundary · AUDITED, awaiting sign-off on Option C

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

## P0 audit result — cross-language query surface · COMPLETE

Every SQL statement on the serving path was classified. 43 queries across
`service.py` (21), `actual_wordplay.py` (12), `wordplay.py` (6) and
`semantic.py` (4). `ann_validation.py` and `acceptance.py` are tooling, not
serving, and are excluded.

**Already single-shard — no routing needed at all:**

- **All 18 wordplay queries** are hardcoded to `t.language = 'en'`
  (`wordplay.py:227,256,283,324`; `actual_wordplay.py:337,383,468,492`). Rhyme,
  anagram, palindrome, spoonerism and homophone search never leave English.
- **Semantic search is already self-contained.** `semantic/mapping.sqlite3`
  carries its *own* `lexical_terms` copy, so it never reads `lexicon.sqlite3`
  and is completely unaffected by lexical sharding. It shards on its own
  78-language axis, which `semantic_languages` and the per-language USearch
  indexes already express.
- `_sense_rows` (`service.py:317`) and the semantic seed lookup filter
  `term.language = ?` directly. `_dependent_rows` (examples, pronunciations) is
  keyed by `sense_id`/`entry_id` belonging to the shard already being read.

**Genuinely cross-language — six queries, and they are the complex ones:**

- `_translation_rows` (`service.py:386`) and `_synonym_rows` (`service.py:478`)
  join `lexical_terms` on `target_term_id`, whose row lives in another language.
- `_translation_coverage` (`service.py:411`) aggregates
  `COUNT(DISTINCT term.language)` across *every* target language at once.
- The relation traversals — forward (`service.py:894`), reverse
  (`service.py:952`) and the bounded prefetch (`service.py:1156`) — join
  `lexical_terms` **twice**, as both source and target.

**The blocker is not the joins. It is the ranking.**

The relation queries rank results by target-side data computed in correlated
subqueries — `target_entry_count` and `target_sense_count` over the *target*
term's entries and senses (`service.py:871-880, 929-938, 1173-1183`) — and then
apply `ROW_NUMBER() OVER (PARTITION BY target.term_id ...)` before the final
`ORDER BY`. Under fan-out those counts are only as complete as the installed
shard set, so **the same query would return differently ordered results
depending on which languages the user installed**. This codebase enforces
deterministic ordering everywhere; install-dependent ranking is not acceptable.
Fan-out would also mean reimplementing a SQL window function in Python.

Multi-hop makes it worse. Transitive traversal keys the frontier on
`(target_normalized, target_language)` (`service.py:1266-1272`) and feeds it
back through `_relation_rows_many` / `_batched_relation_rows`, so hop N+1 queries
whatever languages hop N happened to land in. Seven relation methods share this
shape.

### Option C — denormalize the target payload into the edge tables · SUPERSEDED

Superseded by `LEXICON_V2_ARCHITECTURE_RECOMMENDATION.md` §2 and the measurements
below. Kept for the reasoning; see "Accepted architecture" at the end of this
document for what is actually being built.

Neither A nor B. The audit points at a third option the codebase has already
used once.

Each shard stores, on its own outbound edge rows in `translations`, `synonyms`
and `relations`, the target's **term text, language, and the two ranking counts**
as columns, instead of joining `lexical_terms` for them. Then:

- Every serving query becomes single-shard. Routing collapses to "open the shard
  for the query language" — no fan-out, no cross-shard merge, no window function
  reimplementation, no LRU connection pool complexity beyond the obvious.
- Ranking is computed at build time from the **full** corpus, so ordering is
  identical no matter which languages are installed. Determinism preserved.
- The trust model is untouched: every shard is still a hash-verified artifact.
- It is the exact pattern `semantic/mapping.sqlite3` already uses by carrying its
  own `lexical_terms` copy. Precedent exists and has shipped.

**Cost:** `relations` holds 18,957,715 rows. Adding target term text, language
and two counts is roughly 30 bytes per row, about 570 MB, or ~6% on the 9.0 GB
lexical database. `translations` (3,552,289) and `synonyms` (3,974,062) add
proportionally less. This is the price of collapsing the entire routing problem,
and it is the same trade the semantic artifact already made.

**Open sub-question for sign-off:** what to do with an edge whose target language
is not installed. Either return it with the target marked unavailable — honest,
more useful, and consistent with the D8 rule that translations are directional
strings — or filter it out. Recommend the former.

**Bonus finding for D12.** The honesty contract already exists in miniature:
`_supports_languages` and `_unsupported_language_response` (`service.py:278-295`)
already return `available: false` with
`unavailable_reason: "english_profile_supports_only_en"`. D12 generalizes that
mechanism to arbitrary language sets rather than inventing one.

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

1. ~~**D7 shard boundary** — audit cross-language query surface.~~ **Done.** See
   the P0 audit result: Option C recommended, needs sign-off, plus the sub-question
   on edges pointing at uninstalled languages.
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

---

# Accepted architecture (supersedes D6b/D7/D10/D11 above)

`LEXICON_V2_ARCHITECTURE_RECOMMENDATION.md` is the governing design. It is
adopted in full except for §2, refined below on measured evidence. Summary of
what it changes relative to the earlier decisions in this document:

| Earlier decision here | Superseded by |
| --- | --- |
| D7 Option C — copy target payload onto every edge | §2 target catalogue, refined below |
| D6 — one long-tail bundle | §7 — 8–32 stable tail packs |
| D6b — monolith possibly on a separate path | §9 — monolith on the *same* v2 schema |
| D10 — language set as install identity | §8 — resolved *component* set as identity |
| D11 — hardlink a new version directory | §12 — content-addressed component store |
| D12 — one honesty contract | §6/§13 — per-capability, per-failure-mode reporting |
| "shared core incl. OEWN + CMUdict" | §11 — English resources leave the mandatory core |

Two corrections to my own P0 write-up, both from the review:

- I claimed Option C makes **every** serving query single-shard. Wrong. Multi-hop
  relation traversal genuinely needs the intermediate language's shard, because a
  target catalogue supplies the target's *display and ranking* data but not its
  *outbound edges* (§4). Bounded routed traversal is required.
- My ~570 MB estimate for per-edge denormalization was too high. Measured: 243.5
  MiB (see below).

## Measured refinement to §2 — right idea, wrong table

Measured on the installed `data-v1.1.0` corpus, treating each source language as
its own shard (the upper bound on stub duplication):

| Table | Edges | Cross-language | Distinct (shard, target) stubs | Reuse |
| --- | --- | --- | --- | --- |
| `relations` | 18,957,715 | 9,495,190 (50.1%) | 5,578,033 | 1.7 edges/stub |
| `translations` | 3,552,289 | 3,552,287 (100%) | 2,560,035 | 1.4 edges/stub |
| `synonyms` | 3,974,062 | **0 (0.0%)** | — | — |

**§2's disk argument does not hold.** Targets are barely reused within a shard,
so the catalogue saves far less than "substantially less disk":

| Design | `relations` | `translations` |
| --- | --- | --- |
| Per-edge denormalization | 243.5 MiB | 110.9 MiB |
| Target catalogue | 207.5 MiB | 108.4 MiB |
| Ratio | 1.2x cheaper | 1.0x — no saving |

**And stubs inside `lexical_terms` would cost more, not less.** That table carries
`UNIQUE (language, normalized_term, term)`, and every stub row pays for an index
entry holding all three text values. Measured on 500,000 synthetic stub rows:

| Layout | Size |
| --- | --- |
| Stub rows, no unique index | 14.9 MiB |
| Stub rows, with `UNIQUE (language, normalized_term, term)` | 28.8 MiB (**1.93x**) |

Applied to the 5.58M relation stubs, `payload_local` rows inside `lexical_terms`
land near ~400 MiB — *worse* than the 243.5 MiB per-edge design §2 set out to
beat.

**Refinement: keep the catalogue, put it in its own table.**

```text
target_catalogue
    term_id        INTEGER PRIMARY KEY   -- globally stable, no text unique index
    term           TEXT
    normalized_term TEXT
    language       TEXT
    entry_count    INTEGER               -- from the canonical full corpus
    sense_count    INTEGER               -- from the canonical full corpus
```

- ~207 MiB rather than ~400 MiB: cheapest of the three designs.
- Edge tables keep referencing `target_term_id`, so §2's relational shape and
  §3's package-time ranking both survive intact.
- It avoids a real footgun. `lexical_terms` is queried *directly* in places that
  would silently pick up stubs as though they were real headwords — the profile
  bounds probe (`service.py:268`) and the English wordplay term lookup
  (`actual_wordplay.py:383`) both do bare `SELECT ... FROM lexical_terms`. A
  `payload_local` flag only helps if every such query remembers to filter on it;
  a separate table makes the mistake impossible.

Whichever way this lands, note the stakes: 207 vs 243 vs 400 MiB on a 9.0 GB
lexical database is 2–4%. **Decide it on maintainability, not bytes.**

## Two further findings

**Synonyms are single-shard by data.** All 3,974,062 synonym edges are
same-language; zero cross the boundary. My P0 audit flagged `_synonym_rows` as
cross-language because the *schema* permits a foreign `target_term_id`, but the
corpus never does it. Add a build-time invariant asserting this so the runtime
can rely on it, rather than carrying routing machinery for a case that does not
occur.

**Cross-language edges are English-centric.** 57.1% of cross-language relation
edges point at English. Non-English shards will be dominated by English target
stubs, and — relevant to §7 — bundling medium languages together saves little on
stubs, because bundled languages mostly point at English rather than at each
other. Size-balance the bundles on payload, and treat stub overhead as roughly
additive per bundle.

## Open items carried forward

1. Sign-off on `target_catalogue` as a separate table (above).
2. §7 tier cutoffs from measured post-`VACUUM` bytes — still the gating
   prototype measurement.
3. §12 content-addressed store vs `runtime/locator.py`: the runtime resolves
   artifacts by *relative path* from the version directory, and manifest
   integrity fields such as `semantic_mapping` are relative paths
   (`data/manifest.py:292`). Either the store materializes a compatibility
   directory view, or path resolution moves into the activation record. §12
   allows the former but does not decide it.
4. Bundle membership stability across dataset versions: if a language moves
   between tail packs in a new release, a user's resolved component set churns on
   upgrade. Needs a stability rule.

---

# v2.0.0 scope decisions — signed off

| Question | Decision |
| --- | --- |
| Scope of 2.0.0 | **Everything in `LEXICON_V2_ARCHITECTURE_RECOMMENDATION.md`**, including the §12 content-addressed store, capability splitting, and add/remove-language. No deferral to a 2.1. |
| Data source | **Transform the installed `data-v1.1.0` corpus.** No pipeline re-run, no source re-download. |
| v1 dataset compatibility | **Hard cut.** 2.0.0 requires `data-v2.0.0`. Manifest v1 parsing is retained only so `status` reports an installed v1 dataset as incompatible instead of failing; the runtime never serves it. Users needing the old corpus pin `v1.2.0`, which keeps working because releases are immutable. |

## The corpus is now irreplaceable — treat it as such

`state/lexicon-mcp-build` holds 892 KB of acceptance records. The build tree and
every raw source are gone. The only surviving copy of the corpus is the 23 GB
installed dataset at `data/lexicon-mcp/versions/data-v1.1.0`.

`sources.lock.json` pins `enwiktionary-2026-08-05`, and Wikimedia rotates dumps
out of the archive, so a from-sources rebuild may not be reproducible at all —
the pinned digest would no longer be fetchable. Back this dataset up before the
transform work starts. It is not just convenient to reuse; it may be the only
copy that will ever exist at these exact digests.

## Locator resolution under the content-addressed store · DECIDED

Choosing the full §12 store forces open item 3. Decision: **component-relative
paths resolved through the activation record. No materialized directory view.**

Rejected: hardlinking a compatibility tree per activation. §12 permits it, but on
Windows hardlinks need the same NTFS volume and the copy fallback would double a
22 GB full install on disk — exactly the cost the store exists to avoid.

This is less invasive than it first appears, because both consumers are already
being rewritten in v2:

- The lexical **shard pool** is new code. It asks the activation record for
  "the component owning language X" and opens the store path directly.
- The **semantic packs** are repackaged anyway (§10), so we control their path
  convention. Today `semantic_languages.index_file` and the `vector_file` /
  `global_index` metadata rows encode paths relative to the *dataset root*.
  In v2 they become relative to their *component*, joined at load time to
  wherever the store placed it.
- `data/manifest.py`'s `semantic_mapping` integrity field
  (`data/manifest.py:292`) becomes component-relative for the same reason.
- `verify_component(directory, component, ...)` already takes the directory as a
  parameter, so verification works unchanged against a component's store
  location.

The invariant to preserve: nothing outside the activation record may assume a
dataset-root-relative layout.

---

# P1 prototype measurements — real packs built from the v1.1.0 corpus

Built with a disposable prototype that partitions the installed corpus into
v2-shaped lexical packs, VACUUMs, and compresses at zstd level 10. Each pack
carries every edge where it owns *either* endpoint, because `_relation_rows`
queries forward and reverse — so cross-language edges are stored in two packs.

| Pack | Langs | Raw | zstd | Terms | Stubs | Relations |
| --- | --- | --- | --- | --- | --- | --- |
| `en` | 1 | 2,056.1 MiB | 620.0 MiB | 1,985,802 | 5,102,203 | 10,575,211 |
| `fr` | 1 | 512.1 MiB | 153.5 MiB | 1,548,392 | 838,395 | 4,245,700 |
| rank 25 | 1 | 54.5 MiB | 20.1 MiB | 129,098 | 75,883 | 229,079 |
| rank 50 | 1 | 18.6 MiB | 5.8 MiB | 51,906 | 38,874 | 68,030 |
| ranks 51–60 | 10 | 196.7 MiB | 69.3 MiB | 455,298 | 90,027 | 359,178 |
| ranks 1000–1500 | 500 | 11.6 MiB | 3.6 MiB | 31,755 | 0 | 133 |
| **ranks 1000+** | **4,508** | **18.8 MiB** | **6.7 MiB** | 53,457 | 163 | 188 |

## Finding 1 — drop §7's tail-pack splitting · DECIDED

§7 asks for 8–32 stable tail packs so that "one obscure language does not cause
the entire 5,000-language tail pack to download." Measured, that download is
**6.7 MiB**: every one of the 4,508 languages ranked 1000 and below, together,
with 188 relations between them. The whole tail is smaller than a single
rank-50 language.

Ship **one tail pack**. The 8–32 pack scheme is deleted, and with it the bundle
membership stability rule flagged as open item 4.

Stability is a non-issue for a second reason: users select *languages*, and the
installer resolves language → component **per dataset version**. Membership
changing between dataset versions costs nothing, because a new dataset version
is a fresh download regardless.

## Finding 2 — do not split English on relation class · REJECTED

English carries 10,575,211 relations, 56% of the corpus, and 5.1M stubs at 2.6x
its own term count, because it is the target of 57% of cross-language edges and
must hold both orientations. Hypothesis: split the pack so an English-only
install skips edges into languages it does not have.

Measured decomposition of the 2,056.1 MiB pack:

| Component | Raw | zstd | Relations |
| --- | --- | --- | --- |
| Dictionary only (no relations) | 1,372.8 MiB | 433.5 MiB | 0 |
| \+ intra-English relations | 1,597.2 MiB | 489.6 MiB | 3,516,359 |
| \+ cross-language relations | 2,056.1 MiB | 620.0 MiB | 10,575,211 |

**The hypothesis is wrong.** The dictionary alone is 67% of the pack raw and 70%
compressed. Cross-language relations cost 458.9 MiB raw but only **63.4 MiB
compressed — 10% of the download**, because integer edge columns compress ~7x.

English is large because English's dictionary is large: 1.99M terms, 3.55M
outbound translations, and the senses and examples behind them. Note the
dictionary-only variant still needs 2,558,415 stubs — translations are 100%
cross-language, so naming their targets is inherent, not overhead that a
different pack boundary could avoid.

Splitting would buy 10% of one pack's download in exchange for a new component
axis and reverse-relation queries that silently return fewer results. Rejected.

## Proposed tiering rule · needs sign-off

Express cutoffs as measured byte thresholds, not fixed ranks, so the assignment
is re-derivable for every dataset version:

- **Individual pack** when a language's compressed pack is **≥ 5 MiB**. On this
  corpus that lands near rank 50 (rank 50 measures 5.8 MiB).
- **Bundled** otherwise, into size-balanced groups targeting ~25–50 MiB
  compressed, assembled in rank order so bundles stay contiguous and legible.
- **One tail pack** for everything the bundling rule leaves under threshold.

The language → component map is materialized into the manifest at package time,
so the installer never re-derives it and the runtime never guesses.

---

# P1 progress — schema-2 manifest contract

Landed (uncommitted, working tree):

- `data/manifest.py` — `parse_manifest` dispatches on `schema_version`. Schema 1
  parses exactly as before so an installed v1 dataset stays reportable after the
  v2 upgrade. Schema 2 adds `Pack`, `SourceDataset`, `normalize_language`, and
  refuses `profile` / root `languages`, which capability packs replace.
- `data/selection.py` — `resolve(manifest, languages=, capabilities=)` returns
  the component set, per-capability effective coverage, and an explicit
  `unavailable` list. `languages=None` expresses a full install.
- `tests/test_data_selection.py` — 15 tests. Full suite 259 passing, ruff and
  mypy clean.

Contract decisions made while implementing:

- **Exactly one pack may serve a (capability, language) pair.** Enforced at parse
  time, so selection is unambiguous and needs no tie-break rule. A language may
  of course appear under several *capabilities*.
- **Tiering is invisible to callers.** A request names a language; whether that
  language has its own pack or shares a bundle is resolved through the pack
  table. This is what makes the Finding 1 decision (bundle membership may change
  between dataset versions) harmless.
- **Selection is total.** An unsatisfiable request yields a selection plus typed
  reasons — `language_not_in_dataset` or
  `capability_not_available_for_language` — rather than an exception or a silent
  omission. Strict mode still makes an unknown language fatal so a typo cannot
  quietly install nothing.
- **`profile` becomes `"components"`** for schema 2. `active_version()` still
  validates the v1 profile set (`data/lifecycle.py:123`) and must accept this in
  P2.

Next in P1: the packaging transform that emits these components and manifest
from the v1.1.0 corpus. That step is resource-heavy — it needs the full corpus
scan, VACUUM and zstd — so it waits for a free machine.

## P1/P2 landed on branch `v2-component-architecture`

Four commits, 336 tests passing, ruff and mypy clean. All resource-light; the
corpus was never touched.

| Module | Purpose |
| --- | --- |
| `data/manifest.py` | schema 1 + 2 dispatch, `Pack`, `SourceDataset`, `normalize_language`, `is_sha256` |
| `data/selection.py` | `resolve()` -> component set, effective coverage, typed unavailability |
| `pipeline/packs.py` | pure tier planning, lexical pack schema, core pack schema |
| `pipeline/transform.py` | repartition a schema-1 corpus into packs, read-only at source |
| `data/store.py` | content-addressed component store with prune |
| `data/activation.py` | immutable activation records; the runtime's routing table |
| `data/component_lifecycle.py` | install / add / remove / activate / prune / verify |
| `data_cli.py` | schema-aware dispatch, `--languages`, `--capabilities`, amendment commands |

Decisions taken while implementing:

- **A schema-2 install refuses to run without an explicit selection.** Defaulting
  to everything means a 23 GB download for a forgotten flag.
- **`relation_count` in the core catalogue counts rows once**, not once per
  endpoint, via inclusion-exclusion. A packing rule that retains an edge when
  either endpoint matches would otherwise be mispredicted by the catalogue for
  every intra-language edge.
- **Packs are checked closed before indexing**: every edge target resolves to a
  local headword or a catalogue stub, and the two never overlap. This is the
  invariant that lets the runtime render a result without opening another shard.
- **`materialize()` is the single fetch seam.** Both schemas download through
  the same resume, retry and integrity path rather than forking it.
- **The CLI passes the install *source*, never a preloaded manifest**, so a local
  source still resolves its sibling assets instead of silently reaching for the
  network.

Still to do, all light: the runtime pack router with lazy LRU connections, the
capability/honesty layer, and the bounded multi-hop traversal. Then the
differential harness, whose *execution* is heavy.

---

# Heavy run — real corpus results

## The measured pack plan · 71 components

Planned from the real language census (5,508 languages, 16,513,979 terms, 15s):

- **62 individual language packs**, English (~303 MiB estimated) down to
  Azerbaijani (~5.2 MiB).
- **8 bundles**, seven targeting ~40 MiB and holding 10, 13, 18, 27, 46, 88 and
  291 languages respectively as the tail thins.
- **1 tail bundle** holding the remaining **4,953 languages in ~22.5 MiB**.
- **1 core catalogue.**

Confirms Finding 1 decisively: the tail needs no special handling, and the
generic accumulate-to-target rule produces it without a special case.

Note the estimator underestimates English roughly 2x (303 MiB estimated against
620 MiB measured in the prototype) because English carries 5.1M target stubs and
56% of all relations. That does not affect tiering -- English is far above any
plausible threshold -- but the bundle targets are calibrated on small languages
where the coefficient holds.

## Two problems the real corpus exposed

**The core catalogue query did not finish.** Grouping `COUNT(DISTINCT)` over
`lexical_terms` joined to entries joined to senses builds an enormous temporary
b-tree; it had produced nothing after twenty minutes on the real corpus. Since
per-term counts are already materialized for relation ranking, the catalogue now
sums them in one indexed pass. This is the class of thing only a real run finds:
the same query returns instantly on a four-row fixture.

**A pack cannot always be one component.** A semantic pack needs its mapping
rows, its vectors and its USearch index as separate artifacts -- the index is
loaded from a file, and this codebase deliberately avoids restoring one from
memory (`tests/test_data_lifecycle.py` forbids `Index.restore`). `Pack.component`
became `Pack.components`, with the manifest accepting either form.

## Precomputed inputs, cached

| Input | Cost | Reused by |
| --- | --- | --- |
| Language census | 15s | pack planning |
| Full-corpus term counts (174.7 MiB) | 231s | every pack's catalogue, and the core catalogue |

## The differential gate exists at unit scale

`test_pack_relations_match_the_monolith_exactly` runs the monolith's relation
query and the pack-native one over the same fixture corpus and compares *ordered*
results, including a cross-language edge whose target is only a stub. That is the
release gate's shape; scaling it to the full corpus is the remaining heavy step.

---

# Full corpus build — results

150 packs, 18.89 GiB raw, built from the installed `data-v1.1.0` corpus.

| Capability | Packs | Raw |
| --- | --- | --- |
| lexical | 70 | 8.24 GiB |
| semantic | 78 | 10.36 GiB |
| wordplay | 1 | 0.29 GiB |
| core | 1 | 0.2 MiB |

Largest lexical packs:

| Pack | Raw | Terms | Stubs | Relations |
| --- | --- | --- | --- | --- |
| `lexical-en` | 2,056.1 MiB | 1,985,802 | 5,102,203 | 10,575,211 |
| `lexical-fr` | 512.1 MiB | 1,548,392 | 838,395 | 4,245,700 |
| `lexical-la` | 426.0 MiB | 900,179 | 110,990 | 418,663 |
| `lexical-es` | 401.3 MiB | 862,893 | 165,697 | 489,254 |

**Sharding overhead is ~2%, not the 10-20% predicted.** Lexical packs total
8.24 GiB against an 8.35 GiB monolith that *also* contained the wordplay tables
now split out (0.29 GiB). Per-pack `VACUUM` and index locality roughly cancel
the duplicated cross-language edges. Semantic grew more (~20%), because each
pack's mapping carries its own `lexical_terms` copy across 78 packs.

## The wordplay pack — a gap the build exposed

`wordplay_terms`, `pronunciation_onsets`, `pronunciations_words` and
`wordplay_fts` were in no pack schema. The first full build would have completed
cleanly and silently dropped an entire MCP tool. The pack is deliberately
self-contained -- it carries its own English `lexical_terms` copy -- so rhyme and
anagram search costs 301 MiB rather than requiring the 2 GB English dictionary.
That is the capability/language split paying off concretely.

Its reverse indexes are copied verbatim so they match the corpus. The FTS index
cannot be copied (contentless, rows unreadable) and is rebuilt from the exact
statement the corpus build used.

# Remaining work

**Critical path: `runtime/service.py`.** The seven MCP tools still assume one
lexical database. The routing seam is in (`_db(language)`, language threaded
through the sense-keyed helpers, `LanguageNotInstalled`), and monolith behaviour
is unchanged, but the remaining integration is:

1. Relation, translation and synonym SQL must switch to the `term_union` forms
   in `runtime/pack_queries.py` when a router is present -- a pack's
   `lexical_terms` holds only local terms, so the existing joins would silently
   drop foreign targets.
2. `SQLiteWordplaySearch` and `SQLiteActualWordplaySearch` bind to
   `self.database_path`; under schema 2 they bind to the wordplay pack.
3. Semantic search binds to a directory; under schema 2 it binds to the
   per-language semantic packs.
4. Tool responses must surface `PackRouter.availability` so an uninstalled
   language is reported rather than returned empty.
5. `from_components` construction, and `server.py` choosing it by layout.

Then: install the full release, run `scripts/differential_gate.py` against it,
run release acceptance, and retire the v1 install path, `--profile`, and the
English build.

# Release validated

`data-v2.0.0` packaged: **307 assets, 6.5 GiB compressed** from 18.89 GiB raw
(2.9x), manifest 408 KiB, 306 components across 150 packs.

## What selection actually costs

| Selection | Download | Installed |
| --- | --- | --- |
| English lexical | 0.60 GiB | 2.01 GiB |
| English + semantic + wordplay | 1.11 GiB | 2.88 GiB |
| All 5,508 lexical languages | 2.67 GiB | 8.24 GiB |

Against 22.62 GiB for a v1 full install. English with every capability is an
**eightfold reduction installed**, and the whole multilingual lexical corpus
still fits in 2.67 GiB of download.

## Differential gate

Run against the real release, comparing ordered results between the schema-2
install and the schema-1 corpus it came from:

| Languages | Comparisons | Divergent |
| --- | --- | --- |
| cy, gv | 3,556 | **0** |

Covers relations across all twelve relation codes in both orientations, sense
lookup, and translations, for 120 sampled words per language -- the richest
headwords, where ranking ties are most likely, plus a seeded random spread.

## Tiering, observed

Installing `en,fr,cy,gv` yields **30 installed lexical languages**: Manx shares
a bundle with 27 others, so they arrive at no extra cost. The `languages`
command reports this correctly, and distinguishes `installed`,
`language_not_installed` (German, in the corpus but not selected) and
`not_available_upstream` (wordplay for anything but English).

Verification of the 18 installed components passes.

## Differential gate — full result

| Scope | Comparisons | Divergent | Time |
| --- | --- | --- | --- |
| cy, gv | 3,556 | 0 | 142s |
| en, fr | 4,457 | 0 | 80s |
| **all 30 installed languages** | **44,548** | **0** | **66s** |

Relations across all twelve codes in both orientations, sense lookup, and
translations, comparing *ordered* results against the schema-1 corpus.

### The gate caught a 200x performance bug

The first en/fr attempt burned 8,316s of CPU without finishing. The cause was
not the monolith: it was `pack_queries`. A CTE unioning `lexical_terms` with
`target_catalogue` is materialized in full before filtering -- for English,
1.99M rows with correlated count subqueries plus 5.1M stubs, rebuilt on every
query.

| Query | Before | After |
| --- | --- | --- |
| `dog` relations, monolith | 0.14s | 0.14s |
| `dog` relations, pack | **39.10s** | **0.00s** |
| `run` relations, pack | — | 0.01s (monolith 0.35s) |

Resolving targets through two primary-key LEFT JOINs with COALESCE fixed it. The
pack is now 25-35x *faster* than the monolith, because stub ranking counts are
read rather than recomputed.

The fixture corpus could never have caught this: four rows materialize
instantly. It only exists at corpus scale, which is precisely what the gate is
for. The lesson recorded for next time -- measure both sides of a comparison
before concluding which one is slow.

---

# v2.0.0 — service integration complete

## Tool parity against the monolith

Every tool, run against a real schema-2 install and the schema-1 corpus it was
built from, comparing ordered results:

| Tool | Comparisons | Divergent |
| --- | --- | --- |
| lookup | 21 | 0 |
| relations | 105 (462 in the wider sweep) | 0 |
| synonyms | 21 | 0 |
| translate | 21 | 0 |
| semantic | 63 | 0 |
| rhymes | 13 | 0 |
| wordplay | 28 | 0 |
| **total** | **272** | **0** |

Plus the base-query differential gate: 44,548 comparisons across 30 languages,
0 divergent.

## Bugs the live service found

Each passed every structural check and failed only on a real query:

1. **Pack connections had no `row_factory`** -- rows came back as tuples.
2. **The primary connection was whichever language sorted first** (Welsh), so
   any call site that did not thread its language silently queried the wrong
   pack. English translations returned zero. `_db(None)` now raises in pack
   mode, so this class of bug cannot be silent again.
3. **Double ranking** -- delegating batched relations to the single-source path
   ranked and truncated per source; the monolith ranks the merged set once.
4. **Wrong bound** -- the monolith bounds each *orientation* at
   `min(128, max(32, limit*4))` and appends forward-first.
5. **Wrong truncation order** -- the pack query selected its top N by the
   direct-query ranking and only then re-sorted into prefetch order, so a
   different set of rows survived. Ordering had to move into the SQL, before
   the LIMIT.
6. **The wordplay pack was not self-contained** -- it lacked the entries, senses
   and provenance its results are annotated with (301 MiB -> 998 MiB to fix).
7. **The wordplay pack dropped `wordplay_index_version`**, which the runtime
   pins; it built, packaged and installed cleanly, then failed to open.

## Semantic search across packs

A schema-1 dataset finds neighbours everywhere from one global index. Packs
cannot: the pack holding the seed does not hold the index that would find
neighbours elsewhere. Because Numberbatch shares one vector space, the seed is
taken from its own pack and searched against every installed pack's index, then
reranked exactly against that pack's vectors and merged.

Targeted searches match the monolith exactly, in both directions across packs.
An *unrestricted* search cannot match and does not pretend to -- the monolith
covers 78 vector languages, an install covers what it installed -- so the
response reports which languages were searched, how many exist upstream, and
whether the result was restricted.

## Hard cut honored

The server serves component datasets only. A schema-1 root is reported with its
remedy (reinstall, or pin 1.2.x) rather than served, because maintaining both
query surfaces would mean testing two of everything indefinitely.

## Known state

- `test_full_corpus_latency_and_private_memory_gates` fails on this host at
  812 MB against a 512 MB idle-memory gate. **Pre-existing**: it fails
  identically at the shipped `v1.2.0` tag (811.8 MB), so it is environmental,
  not a regression.
- `mypy` with no arguments cannot run: `packages = ["lexicon_mcp"]` resolves to
  the installed copy, which has no `py.typed` marker. Explicit file paths work.
- v1 code paths remain in the tree, now unused by the server. Removing them,
  along with `--profile` and the English build, is a separate cleanup.
