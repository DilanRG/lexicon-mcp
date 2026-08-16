# Lexicon MCP v2 — Architecture Recommendation

## Decision

Use **Option A's lazy multi-connection shard routing**, combined with a refined version of Option C:

> **Option C2: shard-local materialized target dimensions**

Option A and Option C solve different problems:

- **Option A** answers: *How does the runtime locate and open the right lexical files?*
- **Option C2** answers: *What data must each shard contain so queries remain deterministic, self-contained, and useful even when target-language shards are not installed?*

The broader v2 direction is sound:

- one canonical full-corpus build;
- package-time sharding;
- install-time language/capability selection;
- manifest-verified immutable components;
- atomic activation;
- explicit "not installed" responses rather than misleading empty results;
- a first-class full-install path.

---

## 1. Keep lazy multi-connection routing

Do **not** use SQLite `ATTACH` as the primary sharding mechanism.

The SQLite attachment limit applies to databases attached to one connection. It does not prevent Lexicon from opening independent read-only connections to different shards.

Use:

- one lexical router / shard pool;
- lazy open on first access;
- an LRU of roughly **4–8 open shard connections** initially;
- idle expiry if useful;
- tune the limit from measurements rather than treating 8 as fixed.

Typical sessions will probably touch only one to three languages, so a full installation does **not** need dozens of simultaneously open SQLite connections.

### Why not install-time merge?

Install-time merging would preserve today's single-database runtime, but it has worse properties:

- creates a locally derived artifact not directly covered by the published manifest hash;
- requires temporary disk space;
- costs additional install time;
- makes `add-language` a merge operation rather than a component addition;
- complicates rollback and repair;
- weakens the clean "published hashes are the trust boundary" model.

So the recommended runtime architecture is **lazy multi-connection routing**.

---

## 2. Do not copy full target strings onto every edge

The proposed denormalization direction is correct, but literal per-edge duplication is unnecessarily expensive.

Instead of storing target text, normalized text, target language, entry count, and sense count on every translation/relation/synonym row, give each shard a compact local target catalogue.

Conceptually:

```text
lexical_terms
    term_id
    term
    normalized_term
    language
    entry_count
    sense_count
    payload_local
```

Where:

- `payload_local = true` means this shard owns the term's actual dictionary payload;
- `payload_local = false` means it is a lightweight target stub used by edges;
- `term_id` remains globally stable;
- `entry_count` and `sense_count` are computed from the canonical full corpus at package time.

Then edge tables can continue referencing `target_term_id`.

### Benefits

This preserves Option C's main advantages:

- target display information is available without opening another language shard;
- ranking is based on canonical full-corpus counts;
- results do not change depending on which unrelated languages happen to be installed;
- no cross-database join is required for direct queries;
- target strings are stored once per referenced target **per shard**, rather than once per edge;
- the current relational design remains recognizable.

This should consume substantially less disk than repeating target text across millions of relation rows.

---

## 3. Bake deterministic ranking data at package time

Any ranking value that currently depends on target-side corpus statistics must be independent of the installed subset.

That includes values such as:

- target entry count;
- target sense count;
- any target variant rank used by window functions or equivalent ordering logic.

These should be computed from the **canonical full corpus** during package generation and materialized into shard-local target metadata.

The invariant should be:

> Installing or removing an unrelated language must never change the ordering of results for a language that remains installed.

This should become a differential-test requirement.

---

## 4. Most queries can be single-shard, but not all

Direct operations can usually remain single-shard:

- dictionary lookup;
- translations;
- synonyms;
- direct relations;
- most language-scoped lexical queries.

However, **multi-hop relation traversal can cross a shard boundary**.

Example:

```text
English source
    -> first-hop French target
        -> second-hop relation from French
```

The second hop logically belongs to the French shard.

So multi-hop traversal should work as a bounded routed operation:

1. query the source shard;
2. collect the first-hop frontier;
3. group frontier nodes by resolved shard;
4. query installed frontier shards;
5. merge results;
6. apply deterministic global ordering;
7. report any intermediate languages that could not be expanded because they are not installed.

This is still far simpler than generic fan-out across every installed language.

---

## 5. Return edges into uninstalled languages

If English contains a translation:

```text
dog -> fr -> chien
```

an English-only lexical install should still be able to return `chien`.

The French dictionary payload does not need to be installed for the directional English translation assertion to remain useful.

A result can expose something like:

```json
{
  "term": "chien",
  "language": "fr",
  "target_language_installed": false,
  "target_details_available": false
}
```

Do **not**:

- suppress the translation;
- return "no results";
- require the French shard just to display the target term.

But if the user then asks for a full French dictionary lookup, Lexicon should report:

```text
lexical_language_not_installed
```

Likewise, if French semantic vectors are absent:

```text
semantic_vectors_not_installed
```

And if a relation traversal needs French as an intermediate language, the result should say that traversal was incomplete rather than pretending no second-hop edges exist.

---

## 6. Split capabilities, not just languages

The v2 activation model should not have only one `languages` field.

Lexicon has fundamentally different coverage domains:

- **5,508 lexical language codes** in the installed v1 corpus;
- **78 semantic-vector languages** in Numberbatch;
- English-specific pronunciation and wordplay capabilities.

Represent these separately.

For example:

```text
requested_languages
resolved_components

lexical_languages
semantic_languages
pronunciation_languages
wordplay_languages
```

A language can therefore be:

- lexically installed but have no upstream semantic vectors;
- lexically installed with semantic vectors available but not installed;
- available only as a foreign target stub from another language;
- unsupported by pronunciation or wordplay functionality.

### Runtime requirements by operation

| Operation | Required local capability |
| --- | --- |
| Dictionary lookup | Source lexical language |
| Translation | Source lexical language; target shard not required |
| Direct relation | Source lexical language; target shard not required |
| Synonym | Source lexical language; target shard not required |
| Two-hop relation | Source plus installed intermediate lexical languages |
| Semantic source search | Source semantic component |
| Semantic target search | Target semantic component |
| Rhyme / English wordplay | English wordplay component |

This is much more expressive than a single "supported languages" list.

---

## 7. Use tiered physical packs, but do not create one giant tail bundle

The language distribution strongly supports tiering.

The installed corpus measurements show:

- 5,508 lexical language codes;
- top 10 = 52.70% of lexical terms;
- top 25 = 72.22%;
- top 50 = 84.51%;
- top 100 = 92.46%;
- top 200 = 96.49%;
- ranks 101+ together = only 7.54%.

So one database per language is clearly wasteful.

Recommended physical layout:

### Tier 1

Large languages get individual components.

The cutoff should be chosen from **actual post-build SQLite bytes**, not an arbitrary "top 50" rule.

### Tier 2

Medium languages are packed into a handful of deterministic, size-balanced bundles.

### Tier 3

The long tail should be split into **multiple stable tail packs**, not one enormous catch-all file.

Something like **8–32 tail packs** is a reasonable prototype range.

Why?

If one obscure language causes the entire 5,000-language tail pack to download, subset installation becomes less useful.

The user should still select languages individually. Physical bundles are an implementation detail resolved by the installer.

---

## 8. Treat components, not languages, as the real installation identity

The user's requested language set is only intent.

The actual immutable activation identity should contain the exact component set.

For example:

```text
requested_languages
requested_capabilities

resolved_component_ids
resolved_component_hashes

effective_lexical_languages
effective_semantic_languages

dataset_version
manifest_hash
```

This matters because two installs could both request English and French while differing substantially:

- English + French lexical only;
- English + French lexical + English semantic;
- English + French lexical + both semantic;
- the above plus English wordplay.

Component identity also prepares Lexicon for future non-language packs such as:

- forensic terminology;
- medical vocabulary;
- legal terminology;
- specialist glossaries;
- domain pronunciation packs.

---

## 9. Keep a first-class full monolith for v2.0

For v2.0, publish a monolithic lexical component alongside the subsettable shard set.

But do **not** maintain two fundamentally different query schemas.

Instead:

- package a **v2 monolith** using the same logical v2 schema;
- treat it as one lexical component covering all languages;
- full installations select the monolith;
- subset installations select shards/bundles;
- never activate both representations simultaneously.

Benefits of the monolith for full installs:

- one lexical SQLite connection;
- no lexical routing overhead;
- minimal file-handle and page-cache overhead;
- simpler known-good full-corpus path;
- easy differential oracle against the sharded layout.

Once the sharded full installation has accumulated enough testing and performance evidence, publishing both can be reconsidered.

For v2.0, release storage is cheaper than runtime uncertainty.

---

## 10. Apply the same strategy to semantic data

Semantic data should be independently selectable from lexical data.

For subset installs, a semantic-language component should contain the necessary language-local pieces:

- mapping rows;
- vector data;
- USearch index.

Because Numberbatch languages share one vector space, a source vector from one installed semantic language can be searched against another installed language's index.

When `target_language` is unspecified:

1. search installed semantic-language indexes;
2. exact-rerank as required;
3. merge deterministically;
4. keep simultaneously mapped indexes bounded.

For full semantic installs, retain a global semantic component / global index path to avoid mandatory fan-out across all 78 language indexes.

This gives users useful combinations such as:

```text
all lexical languages
+ English semantic only
```

which could preserve enormous lexical versatility while avoiding semantic resources they do not need.

---

## 11. Make the shared core genuinely small

Do not put English-specific resources into a mandatory shared core.

A better component split is:

### Shared control/catalogue

Tiny:

- component metadata;
- language metadata;
- aliases;
- capability metadata;
- activation/runtime routing information.

### Lexical packs

- owned lexical payload;
- shard-local foreign target stubs;
- required local provenance.

### English wordplay/pronunciation pack

- OEWN/CMUdict-derived English-specific resources;
- rhyme;
- anagram;
- spoonerism;
- homophone;
- related indexes.

### Semantic packs

- one semantic-language component per supported semantic language or logical semantic bundle.

### Full monoliths

Optional components for broad/full installations.

This prevents a Japanese-only or German-only install from paying for English-specific resources.

---

## 12. Prefer a content-addressed component store

Instead of making activation depend primarily on recreating a directory tree of hardlinks, store verified immutable components by digest.

Example:

```text
data-root/
  components/
    sha256/
      <digest>/
  activations/
    <activation-id>.json
  current.json
```

Installation becomes:

1. resolve requested capabilities;
2. download missing components;
3. verify each against the manifest;
4. place each verified component into the content-addressed store;
5. create a new immutable activation record;
6. atomically switch `current.json`.

Advantages:

- downloaded components are stored only once;
- add/remove-language primarily changes activation metadata;
- rollback becomes a pointer swap;
- garbage collection can safely remove unreferenced components later;
- published component hashes remain the trust boundary;
- no full local merge is required.

Hardlinks can still materialize a compatibility directory layout if needed, but should not be the conceptual storage model.

---

## 13. Improve language discovery and honesty

"5,508 languages" means language codes occur in the lexical corpus. It does not imply equal dictionary quality.

`lexicon-data languages` should expose useful capability/coverage information such as:

- native headword count;
- entry count;
- sense count;
- outbound translation count;
- inbound translation count;
- relation count;
- semantic-vector availability;
- semantic component installed;
- pronunciation support;
- wordplay support;
- resolved physical component;
- component download size.

Failures should distinguish at least:

```text
language_not_installed
semantic_not_available_upstream
semantic_component_not_installed
feature_not_supported_for_language
no_results
```

That distinction is critical for an MCP used by an LLM. "No results" and "you do not have the necessary corpus installed" are not equivalent statements.

---

# Recommended v2 architecture

```text
                     Canonical full build
                            |
                    package-time partition
                            |
       +--------------------+--------------------+
       |                    |                    |
  lexical packs        semantic packs       full monoliths
       |                    |                    |
       +--------------------+--------------------+
                            |
                    signed/hashed manifest
                            |
                       installer resolver
                            |
              requested languages/capabilities
                            |
                    cheapest component set
                            |
                 content-addressed local store
                            |
                    immutable activation
                            |
                    runtime capability map
                      /              \
             lexical shard pool     semantic loader
             lazy connections       lazy mmap/index
```

---

# Exact sign-off

| Decision | Recommendation |
| --- | --- |
| Install-time language selection | **Yes** |
| 5,508 individual databases | **No** |
| Tiered/bundled lexical components | **Yes** |
| Fixed top-50 cutoff | **No — derive from measured bytes** |
| One enormous long-tail bundle | **No — split into stable tail packs** |
| SQLite `ATTACH` routing | **No** |
| Install-time database merge | **No** |
| Lazy independent SQLite connections | **Yes** |
| Small LRU connection pool | **Yes** |
| Full target payload copied onto every edge | **No** |
| Shard-local materialized target catalogue | **Yes** |
| Full-corpus ranking counts baked at package time | **Yes** |
| Return direct edges into uninstalled target languages | **Yes** |
| Mark those targets as non-expandable | **Yes** |
| Claim every lexical query is single-shard | **No** |
| Bounded multi-shard traversal for multi-hop relations | **Yes** |
| Separate lexical and semantic installation sets | **Yes** |
| Capability-aware activation records | **Yes** |
| Content-addressed component store | **Yes** |
| Publish a full lexical monolith in v2.0 | **Yes** |
| Use the same logical v2 schema for monolith and shards | **Yes** |
| Global semantic component for full installs | **Yes** |
| Per-language semantic components for subsets | **Yes** |
| Explicit unsupported/not-installed/no-result distinction | **Yes** |

---

# What to prototype before P1

Do not commit the final shard boundaries until a disposable packaging prototype measures them.

Use at least:

- English;
- another very large language;
- one medium-language bundle;
- one tail pack;
- selected semantic-language components;
- a full v2 monolith for comparison.

Measure:

- post-`VACUUM` SQLite bytes;
- compressed release bytes;
- target-stub duplication;
- index overhead;
- cold query latency;
- warm query latency;
- process RSS;
- SQLite page-cache impact;
- number of open file handles;
- semantic mmap usage;
- download/install time.

Correctness tests should include:

- dictionary lookup parity;
- translation parity;
- synonym parity;
- direct relation parity;
- deterministic relation ranking;
- translation into an uninstalled target language;
- two-hop traversal crossing a language boundary;
- two-hop traversal with the intermediate language absent;
- semantic source/target language combinations;
- semantic search with no target language specified.

The strongest release gate should be differential testing:

> A full v2 monolith and a complete v2 sharded installation must return the same ordered logical results across the serving-path query matrix, except for intentionally added installation/capability metadata.

---

## Bottom line

The best v2 design is a **hybrid component architecture**:

> **Tiered lexical packs + shard-local target catalogues + lazy independent SQLite routing + separately selectable semantic packs + a content-addressed component store + a first-class full monolith.**

That preserves the things Lexicon already does well:

- offline runtime;
- immutable verified data;
- deterministic results;
- full-corpus versatility;
- low idle resource usage.

And it adds:

- tiny selective installs;
- independent lexical/semantic footprint control;
- efficient add/remove-language operations;
- honest capability reporting;
- clean future extensibility to specialist knowledge packs.

In short: **spend complexity at package/install time so the runtime stays lean, deterministic, and adaptable.**
