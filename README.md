# Lexicon MCP — development notes

Design records, release acceptance evidence, and corpus measurements. This is an
orphan branch: it shares no history with `main` and is never merged into it. The
shipping code and user documentation live on `main`.

## design/

- `OFFLINE_SOURCES_V12_PLAN.md` — the v1.2.0 and v2.0.0 roadmap, with the
  measurements behind each decision and the ones that were reversed by them.
- `LEXICON_V2_ARCHITECTURE_RECOMMENDATION.md` — the architecture review that
  settled the v2 component design.
- `WORDPLAY_V11_HANDOFF.md` — the v1.1.0 wordplay handoff.
- `v2_prepare.py` — precomputes the two corpus-wide inputs the pack transform
  reuses: the language census and full-corpus term counts.

## acceptance/data-v2.0.0/

The release gate evidence for `data-v2.0.0`: the non-live gates, and the
ten-cycle live-stack run against MCPO and Open WebUI. `publish_data_release.py`
refuses to publish without a report showing all seven gates true, ten restart
cycles and no active models.

## measurements/

- `build-report.json` — every pack built for `data-v2.0.0`: sizes, term counts,
  target-catalogue stubs and relation counts.
- `language-sizes.json` — lexical term counts for all 5,508 languages, which is
  what the pack tiering is derived from.

## Known follow-up

Unrestricted semantic search on a full install fans out across all 78 semantic
packs at ~1,534 ms; a targeted search is ~11 ms. The fix is a global index plus
a semantic-id-to-language map, so the global index is searched once and only the
packs holding top candidates are reranked. Deferred to 2.0.1.
