# Historical contract context

The SFT input can identify the contract's initial question, fixture title and
Yes/No token IDs without treating a current API title as historical evidence.

`scripts/collect_historical_contracts.py` reads candidate question IDs from the
original SHA-256 checked Gamma capture. It then retrieves the Polygon
`NegRiskAdapter` events `MarketPrepared` and `QuestionPrepared`. Their byte
payloads contain the original fixture and question metadata. Each event is
checked against its block header. Three `eth_call` reads at the question's
preparation block reproduce the condition ID and its Yes/No position IDs; all
three must match the audited registry. The original metadata's numeric market
and event IDs and the fixture's team pair must also agree.

The adapter ABI and meaning of `getPositionId(questionId, true/false)` are pinned
to [Polymarket's contract source at commit
f78b35b](https://github.com/Polymarket/neg-risk-ctf-adapter/blob/f78b35b0863b4308a431ca307d06f49b2ea65e78/src/NegRiskAdapter.sol).
The deployment is `0xd91e80cf2e7be2e162c6513ced06f1dd0da35296` on Polygon
(chain ID 137). This adapter emits initial metadata once when it prepares a
market/question. Initial text may differ from a later API presentation or a
later clarification; the collector reports current/initial title differences.

`contract_evidence.jsonl` contains each decoded record and the full underlying
events, headers and historical call requests/results. `validate_contract_record`
re-decodes this evidence and rejects disagreement with any derived feature.
`verified_contract_context(record, query_us)` returns the minimal feature only
when both initialization events strictly precede the query. It assumes the
record was validated once when the catalog was loaded, avoiding repeated
decoding for every training observation.

The prompt contains `question`, `fixture_title`, `yes_token_id`, `no_token_id`,
`initialized_at_utc`, `semantics_scope` and `evidence_id`. It also includes
`initial_resolution_clause` when an exact captured sentence unambiguously
identifies the first 90 minutes of regular play plus stoppage time. The small
extractor accepts only the audited sentence template and rejects additional or
conflicting duration, extra-time, overtime or penalty statements. It never
supplies a fallback from general football knowledge. All 312 captured contracts
contain the accepted clause, which distinguishes regulation results from the
team that eventually advances.

The complete initial resolution rules remain in the evidence catalog. This
capture does not establish a complete timeline of later clarifications.
The evidence establishes public contract content, not that a particular actor
read it. RPC results are trusted for canonical chain events, headers and state;
this process is not a cryptographic consensus proof or a bytecode compilation
audit.

The separate `fixture_known_at.jsonl` gives each fixture's historically verified
pairing time. A news linker can use that time to gate team-background relevance
after validating the associated contract record. Its evidence kind is
`polygon_market_prepared`, not a news archive capture. Pairing evidence is not
itself match news and must not be counted toward direct fixture-news coverage.

Reproduce the collection in a new output directory:

```bash
python scripts/collect_historical_contracts.py \
  --registry data/completion_20260923/registry.json \
  --gamma-cache-bodies /path/to/original/data/cache/bodies \
  --output data/completion_20260923/contracts
```

The default public RPC is `https://polygon.gateway.tenderly.co`; `--rpc-url`
accepts another Polygon archival provider. SHA-256 checked RPC checkpoints
support resumption. The first captured block number freezes the event-query
snapshot for that output directory. A new snapshot requires a new directory.
