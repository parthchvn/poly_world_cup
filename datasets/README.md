# Prepared World Cup SFT data

[Full preparation and loading guide](../docs/tournament_sft.md) · [Readable examples](../reports/sft_preview.json) · [Validation report](../reports/sft_validation.json)

## world_cup_2026_tournament_lt20_v1

104 fixtures, 312 primary binary match-result contracts. Retain wallets with **1–19 saved observations across the whole tournament**. Original datasets are unchanged.

- **206,474 wallets; 969,024 attributed observations.**
- **954,951 targets per profile** in compressed conversational JSONL.
- **14,073 rows** retained only in the audit because fixture/time partitions disagree.
- Two alternative profiles: `verified_news_only` and `execution_history_proxy`. Do not concatenate them as independent observations.

| Split | Fixtures | Targets per profile |
| --- | ---: | ---: |
| train | 72 | 533,858 |
| validation | 24 | 253,877 |
| test | 8 | 167,216 |

[Open the release directory](world_cup_2026_tournament_lt20_v1/) · [Manifest and checksums](world_cup_2026_tournament_lt20_v1/manifest.json)

Every target has verified prior broad tournament news. **1,906** also have verified prior direct fixture news, covering only **7 of 104 fixtures**. See [per-fixture coverage](../reports/tournament_sft_fixture_coverage.json). **750,032** have prior executions under the explicitly unverified availability proxy. All eligible news IDs are preserved in the context catalog; prompts select the latest eight eligible items per scope.

The prepared artifacts total about **1.19 GB compressed**. Each JSONL shard has at most 20,000 rows. The full audit includes each retained action and pointers to its news, history, and exported profile rows.

**Scope:** format-ready for experimental retrospective conditional-observation SFT. Source collection is incomplete (19 contract histories unfinished), quantities are provider-reported, and the activity filter is a retrospective heuristic. The corpus does not establish human identity, market-maker status, actor exposure, private beliefs, order decisions, complete holdings, or prospective readiness. See the guide for exact feature and evaluation restrictions.
