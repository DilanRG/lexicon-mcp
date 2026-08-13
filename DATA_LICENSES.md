# Data licensing and provenance

The Apache-2.0 software license does **not** apply to the downloadable corpus bundle.
Each component remains under its upstream terms, and generated databases/indexes are
distributed with the attribution and share-alike notices applicable to their inputs.

| Component | Upstream | Data terms |
|---|---|---|
| Open English WordNet 2025 | https://github.com/globalwordnet/english-wordnet | CC BY 4.0 plus the underlying Princeton WordNet attribution/license |
| English Wiktionary content extracted by Wiktextract/Kaikki | https://kaikki.org/dictionary/rawdata.html | Wiktionary content under CC BY-SA 4.0 and GFDL; Wiktextract extraction code is MIT and is not the data license |
| ConceptNet 5.7 assertions | https://conceptnet.io | CC BY-SA 4.0 |
| ConceptNet Numberbatch 19.08 | https://github.com/commonsense/conceptnet-numberbatch | CC BY-SA 4.0 |
| CMU Pronouncing Dictionary | https://github.com/cmusphinx/cmudict | CMUdict's BSD-style license and acknowledgement |

Every published bundle must include:

- exact upstream URL, version or dump date, retrieval time, and SHA-256;
- the transformation commit and a description of modifications;
- compressed and extracted artifact hashes and byte counts;
- the applicable license text and attribution alongside each separable component;
- Wiktionary source/history links sufficient for attribution; and
- a clear statement that the transformed data is modified and is not endorsed by its
  upstream projects.

The release manifest is authoritative for the exact snapshots in a given data tag.

