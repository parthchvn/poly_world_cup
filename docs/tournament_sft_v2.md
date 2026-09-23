# World Cup 2026 SFT dataset, version 2

The prepared dataset is
[`datasets/world_cup_2026_tournament_lt20_v2`](../datasets/world_cup_2026_tournament_lt20_v2/).
Its target universe comprises **all 104 World Cup matches, 312 primary binary
match-result contracts, and 624 Yes/No outcome tokens**. Every fixture must have
at least one exported target whose prompt contains verified, strictly earlier,
direct match news. The [completion report](../reports/tournament_sft_v2_completion.json)
records this requirement and the actual coverage of every fixture.

Version 2 adds completed API traversals, historical contract questions and token
mappings, and additional historically captured news. The original datasets and
the [version 1 release](../datasets/world_cup_2026_tournament_lt20_v1/) are unchanged.
The companion [evidence package](../datasets/world_cup_2026_tournament_lt20_v2_evidence/)
preserves the records needed to inspect the new historical evidence and collection
scope.

## Release totals

The full row validator and all-104 completion gate passed. These figures come
from the release manifest and completion report.

| Measure | Final value |
| --- | ---: |
| Fixtures / contracts / outcome tokens | 104 / 312 / 624 |
| Exhausted finite-snapshot API traversals | 312 |
| Retained wallets, with 1–19 tournament observations each | 208,296 |
| Retained observations, including quarantined targets | 979,000 |
| Exported targets in each profile | 961,023 |
| Training / validation / test targets in each profile | 535,758 / 260,624 / 164,641 |
| Quarantined observations retained in the audit | 17,977 |
| Exported targets with selected prior direct fixture news | 773,253 (80.46%) |
| Exported targets with historical contract-question context | 961,023 |
| Fixtures with at least one exported direct-news target | 104 / 104 |

The two profiles contain the same targets and partitions. Use them as alternative
experiments; concatenating them would duplicate labels.

## Prediction task and profiles

Given a wallet identifier, a binary contract, an execution-time query, and the
selected earlier context, predict the provider-reported execution observation:

```json
{"side":"BUY","outcome":"Yes","shares":"12.5","price":"0.42"}
```

The example above illustrates the schema. Quantities and prices remain decimal
strings in the actual labels. This task is **conditional on an execution being
observed**. It does not label whether or when someone decides to trade, intended
orders, private beliefs, rationales, inventory, or profit.

| Profile | Inputs besides wallet, contract and query time |
| --- | --- |
| `verified_news_only` | Earlier verified fixture and tournament headlines, plus the contract's verified initial question, fixture title and Yes/No token mapping when available before the query |
| `execution_history_proxy` | The same context, plus the wallet's strictly earlier captured tournament executions and their own historically eligible contract context |

Prior executions use block timestamps as an exploratory availability proxy. Their
availability to an actor at order-decision time is unverified. Publicly relevant
news does not establish that a wallet read it or reacted to it. Wallet addresses
are actor identifiers for this dataset, not verified identities of people.

`format_ready_for_sft` describes usable conversational JSONL. The completion gate
`all104_fixture_context_ready` describes the defined retrospective capture and
per-fixture context requirements. `prospective_training_ready` and the broader
`training_ready` remain false: whole-period cohort selection, unverified decision
timing, and provider-reported execution economics remain part of the experiment.

## Entire-tournament activity filter and collection completion

A wallet qualifies when its total saved observation count across **all 312
contracts together is 1–19**. Both outcomes, BUY and SELL observations, and all
captured times contribute. The threshold is exclusive: a wallet with 20
observations is excluded. Counts are never reset per match. This is an activity
heuristic; it does not prove that retained wallets are humans or that excluded
wallets are market makers.

The count ledger includes every wallet in the captured tournament, including
excluded wallets. Filtering does not estimate activity from the already filtered
rows. Every retained wallet's full captured history is checked against both its
tournament total and its wallet/contract counts. Whole-period counts can include
activity later than a target, so membership is retrospective. Counts remain
audit metadata and are excluded from model prompts.

The completed source combines **293 previously exhausted contract histories**
with **19 freshly collected, exhausted replacements**. Replacement counts replace
the corresponding earlier partial counts; they are never added to them. The
merger verifies that replacement wallet/contract counts do not shrink and that
the old surviving observations are contained in the new histories with their
original multiplicities. It then recomputes the global cohort.

The 293 inherited histories retain their earlier provenance checks and page
hashes. The 19 replacement histories undergo fresh raw-response replay and
normalized-page verification. The per-contract exhaustion ledger records those
different provenance scopes. The page-reference ledger keeps superseded old pages
for audit, so summing its row counts would double-count replacement histories.
The complete wallet/contract count ledger defines the current observation total.
Exhaustion refers to the specified API traversal
with requested minimum size `0.000001` tokens. It is not a claim of canonical
on-chain fill completeness, activity below the provider filter, or activity in
other markets. API observations are not necessarily unique orders or economic
fills.

## Historical news and fixture attribution

Existing archive evidence is supplemented by historical GDELT GKG records. The
collector preserves the captured `PAGE_TITLE`, source URL, exact retained record,
record location, source ZIP digest, and batch identity. It checks that the record
date and record-ID prefix agree with the archive batch.

The availability upper bound is **the batch timestamp plus 15 minutes**, using
a conservative convention for GKG's fifteen-minute seen/processed resolution.
GDELT does not document the rounding direction, so this is an explicit timestamp
resolution convention, not an exact page-fetch time or the publisher's claimed
publication time. For example, evidence in the 11:00 batch becomes
eligible only for a target strictly after 11:15. Publication dates and present-day
article pages cannot backdate a headline.

Direct fixture relevance requires an already verified game-specific archive link
or an explicit, historically captured matchup under the 2026 World Cup matching
policy. Opposing participants must occur as an explicit pair; other editions,
competitions, hypothetical matchups, simulations, and ambiguous pairs are
rejected. Current registry participants nominate candidates but cannot by
themselves establish that a future knockout pairing was public. Team background
and broad tournament stories do not count as direct fixture news.

Seven explicit upcoming-match headlines use nonadjacent participant names, such
as a player's injury ahead of a named opponent. Their exact titles, URLs, content
hashes, timestamps, fixtures, and rationales are recorded in
[`configs/reviewed_news_links_v2.json`](../configs/reviewed_news_links_v2.json).
An independent assistant review checked each annotation against its captured
record; this is not a human annotation claim. These links also require validated
on-chain evidence that the pairing was already public. Any source mismatch fails
closed. Explicit semicolon-separated matchup lists are checked pair by pair.

Content availability and link availability must both be strictly earlier than
the target. Among eligible versions, the exporter keeps the highest verified
version per news item, sorts by effective availability, and selects **up to eight
distinct headlines per scope**. Headline deduplication uses case folding and
whitespace normalization; the most recent eligible representative is selected.
The fixture and tournament scopes remain separate. The context catalog preserves
all eligible item IDs and their timestamps, including items omitted from the
bounded prompt.

Collection searches a recorded archive batch schedule and adds denser searches
around remaining fixture gaps. The evidence package records visited batches and
errors. Covering every fixture is not a census of all articles or every 15-minute
archive batch. Per-fixture coverage percentages use **exported targets with
selected prior direct news**, not merely articles mentioning one of the teams.
An empty news list before the first verified capture is valid and remains empty.
Reaching 104 fixtures does not imply that every trade has direct news context.

## Historical contract semantics

The contract catalog covers 312 initial questions. Polygon preparation events,
their block headers, and historical adapter calls establish initial question
text, fixture title, and the condition-to-Yes/No-token mapping. Current Gamma
metadata discovers candidate IDs; the event evidence and historical mappings
are checked independently. Export and validation re-decode the stored evidence
and reject disagreement with the trade's contract, fixture, token or outcome.

Contract context is present only when initialization is strictly earlier than
the target timestamp. A prior-execution entry is checked against **that prior
execution's own time**, so a later target cannot supply historical meaning to an
earlier record. Missing or not-yet-available context remains null.

This is initial contract metadata. It does not certify later clarifications,
subsequent rule edits, or actor exposure. The catalog retains the supporting
evidence; the prompt carries the minimal initial question and token semantics.
It also retains the exact verified initial resolution clause specifying regular
play plus stoppage time, so a match-result contract is not confused with a bet on
advancing after extra time or penalties.
The RPC provider is trusted for canonical events, headers and historical call
results; the packaged evidence is not a consensus proof. See also
[historical contract context](historical_contract_context.md).

## Frozen splits and target exclusions

Version 2 retains [`configs/tournament_sft_v1.json`](../configs/tournament_sft_v1.json).

| Split | Disjoint target fixtures | Observation timestamp, UTC |
| --- | --- | --- |
| Train | 72 group-stage matches | Before 2026-06-28 12:00 |
| Validation | 24 round-of-32 and round-of-16 matches | From 2026-06-28 12:00 until 2026-07-09 00:00 |
| Test | 8 quarterfinal-and-later matches | From 2026-07-09 00:00 |

Both fixture assignment and time must agree. Incompatible rows, invalid targets,
duplicate observation identities, and transactions spanning target splits remain
in the audit with explicit exclusion reasons. Every observation in the target
transaction, and all equal-time or future observations, are excluded from its
history prompt.

Wallets can appear in multiple splits. Strictly earlier captured executions from
another target partition can appear in a proxy history, including observations
purged only for fixture/time mismatch. The split separates targets; it does not
claim that all held-out contract identifiers are absent from training histories.
The manifest reports seen and unseen evaluation wallets. A base model's existing
knowledge of tournament results is a separate leakage risk: record its training
cutoff and assess memorization before interpreting performance.

## Inspect and load

Open the [readable examples](../reports/sft_v2_preview.json) to inspect one actual
exported record from each partition before downloading all shards.

After cloning the repository, run this from its root with Python's standard
library. It reads three actual training examples without unpacking every shard.

```python
from pathlib import Path
import gzip
import itertools
import json

root = Path("datasets/world_cup_2026_tournament_lt20_v2")
profile = "execution_history_proxy"  # or "verified_news_only"
paths = sorted((root / profile / "train").glob("part-*.jsonl.gz"))
assert paths, "Clone/download the prepared dataset before loading it"
with gzip.open(paths[0], "rt", encoding="utf-8") as stream:
    for line in itertools.islice(stream, 3):
        example = json.loads(line)
        prompt = json.loads(example["messages"][1]["content"])
        target = json.loads(example["messages"][2]["content"])
        print(json.dumps({"context": prompt, "target": target}, indent=2))
```

For Hugging Face, install the optional `datasets` package in your training
environment. The [official JSON loading documentation](https://huggingface.co/docs/datasets/en/loading#json)
describes this loader. No model-specific training dependencies are needed merely
to inspect the standard-library JSONL files.

```bash
python -m pip install datasets
```

```python
from pathlib import Path
from datasets import load_dataset

root = Path("datasets/world_cup_2026_tournament_lt20_v2")
profile = "execution_history_proxy"
files = {
    split: [str(path) for path in sorted((root / profile / split).glob("part-*.jsonl.gz"))]
    for split in ("train", "validation", "test")
}
assert all(files.values()), "One or more split directories are missing"
dataset = load_dataset("json", data_files=files)
print(dataset)
print(dataset["train"][0]["messages"])
```

Rows have one `messages` field containing system, user and assistant turns. The
user and assistant contents are JSON strings. Preserve them when applying the
chosen model's chat template. A training stack may additionally require
`transformers`, `trl`, `accelerate`, `torch`, and optionally `peft`; choose and pin
compatible versions for the selected model and hardware. Measure token lengths
and verify assistant-loss masks in that environment. Dataset serialization does
not constitute a model-specific training smoke test. Do not silently truncate
away the final action label.

## Artifacts and reproduction

| Release artifact | Purpose |
| --- | --- |
| `{profile}/{split}/part-*.jsonl.gz` | Training-ready conversational records |
| `audit/part-*.jsonl.gz` | Every retained observation, original label, history references, context IDs, exclusions and profile locations |
| `contexts.jsonl.gz` | Eligible news states and the distinct headlines selected for prompts |
| `source_news.jsonl.gz` and `source_news_links.jsonl.gz` | News-version and fixture-link evidence |
| `source_contracts.jsonl.gz` | Historical initial contract records and raw supporting evidence |
| `split_policy.json` | Frozen partition assignments and boundaries |
| `manifest.json` | Counts, capture scope, limitations and artifact SHA-256 checksums |

The [completion report](../reports/tournament_sft_v2_completion.json) checks the
104/312/624 universe, exhaustion ledger, full count ledger, exact retained
histories, source-to-audit correspondence, target eligibility and actual selected
prior direct news for every fixture. The
[full SFT validation report](../reports/sft_v2_validation.json) checks every audit
and profile row, independently reconstructed context, labels, history exclusions,
split boundaries and artifact hashes. The exporter also decompresses finalized
gzip streams through their CRC/footer and checks row counts before publication.

To verify the prepared release after downloading it:

```bash
python scripts/validate_sft.py \
  --dataset datasets/world_cup_2026_tournament_lt20_v2 \
  --report data/local_checks/sft_v2_validation.json
```

Rebuilding the source cohort requires the original per-contract subset **with its
full count ledger**, original collection/provenance checkpoints, and the fresh
replacement capture manifests and raw-response cache. Run
`scripts/build_trade_completion.py` on those inputs to produce a new completed
cohort. `scripts/collect_gdelt_news.py` supports a recorded `--stamps` schedule;
`scripts/collect_historical_contracts.py` reconstructs the initial contract
catalog. `scripts/refresh_news_context.py` writes a separate cohort database with
the verified joined news records and contract evidence. The original source
tables are not edited. The inherited 293 raw capture bodies are not freshly
replayed by this release, so independent recollection is a different snapshot.

From an already reconstructed v2 attribution database and its exact registry,
export to a fresh directory and run both checks:

```bash
python scripts/prepare_sft.py \
  --database /path/to/v2/attribution.sqlite \
  --output data/rebuilt_sft_v2 \
  --split-policy configs/tournament_sft_v1.json \
  --shard-rows 20000 --news-limit 8 --deduplicate-headlines

python scripts/validate_sft.py \
  --dataset data/rebuilt_sft_v2 \
  --report data/local_checks/rebuilt_sft_v2_validation.json

python scripts/check_tournament_completion.py \
  --database /path/to/v2/attribution.sqlite \
  --registry /path/to/v2/registry.json \
  --release data/rebuilt_sft_v2 \
  --report data/local_checks/rebuilt_sft_v2_completion.json
```

No `--allow-partial` override is used for the v2 release. Export, completion
checking and validation refuse existing output paths; use new paths for another
run. Reproducing preparation from the same retained evidence is distinct from
redownloading a changing provider's complete history.

Train and tune using the training and validation partitions, then reserve the
test partition for final evaluation. Compare the two profiles and simple
frequency/history baselines; report JSON validity, side/outcome accuracy, price
error and share error, including results for unseen wallets and per-fixture
uncertainty. Mechanistic interpretation should follow useful held-out prediction
and controlled interventions. Model explanations are not measurements of a
trader's private beliefs.
