# World Cup 2026 v2 supporting evidence

Use the [SFT release](../world_cup_2026_tournament_lt20_v2/) for training and the
[preparation guide](../../docs/tournament_sft_v2.md) for the task definition.
This directory preserves the evidence behind its activity filter, historical
news and initial contract semantics.

| Files | Scope |
| --- | --- |
| `trade_count_ledger/` | All 2,259,806 observed wallet/contract count pairs, including excluded wallets; 12,250,443 observations from 255,921 wallets |
| `manifest.json` | Trade-evidence file checksums, full ledger totals and gzip readback results |
| `contract_exhaustion_ledger.json` | Exact 312-contract universe, with 293 inherited exhausted captures and 19 freshly exhausted replacements |
| `registry.json` | Exact registry bound to the original checkpoint; final metadata is retrospective |
| `original_*.json`, `replacement_*.json` | Original and replacement collection, normalization and raw-response provenance reports |
| `completion_report.json` | Replacement-containment checks and the recomputed tournament-wide 1–19 observation cohort |
| `source_page_ledger.jsonl.gz` | Source-page references, including superseded original pages; do not sum these row counts as the current corpus total |
| `gdelt/` | Exact retained historical GKG records, candidate headlines, visited batches, source archive hashes and collection policy |
| `news_catalog/` | Joined news input, direct-fixture links, assistant-reviewed annotations, background-only links and candidate audit |
| `contracts/` | Initial on-chain questions, fixture titles, Yes/No token mappings, supporting evidence and validation reports |
| `package_manifest.json` | SHA-256 and size of every file in all packages, excluding this aggregate manifest itself |

The trade `manifest.json` covers only its explicitly listed trade-evidence files;
the aggregate `package_manifest.json` also covers the other subdirectories.
The count ledger independently reconstructs 208,296 retained wallets and 979,000
retained observations. Original raw API response bodies are not included in this
Git package. Fresh raw bodies were replayed locally before release; the inherited
293 contracts retain their previous provenance checks and hashes.

`news_catalog/fixture_coverage.json` is the news-selection planning report measured
against the previous cohort. Use the [final completion report](../../reports/tournament_sft_v2_completion.json)
for coverage measured against the rebuilt and exported v2 cohort. Background-only
links are audit material, not direct-match news or prompt features.

API exhaustion is scoped to the recorded traversal and minimum-size filter.
GDELT discovery samples recorded archive batches and does not enumerate every
article. Historical Polygon evidence relies on the responding RPC provider.
These sources establish public context and captured observations, not actor
exposure, private beliefs, human identity or canonical fill completeness.
