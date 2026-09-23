# Tournament-wide filtered SFT release

> This guide describes the archived version 1 release. Use the [version 2 guide](tournament_sft_v2.md) for completed API traversals and direct-news coverage across all 104 fixtures.

The prepared files are in
[`datasets/world_cup_2026_tournament_lt20_v1`](../datasets/world_cup_2026_tournament_lt20_v1/).
They cover the captured corpus for all **104 World Cup matches**, represented by
312 primary binary match-result contracts. This release contains actual compressed
chat JSONL, attribution records, a news catalog, checksums, and frozen splits.

## What the model learns

Given a public wallet identifier, a binary contract identifier, an execution-time
query, and selected prior context, predict the provider-reported `side`, `outcome`,
`shares`, and `price` of an observation **conditional on that observation occurring**.
This is a retrospective, trade-conditional experiment. It does not train whether
or when a wallet trades, order placement, private beliefs, or inventory balances.
Targets preserve decimal strings and distinguish BUY/SELL and Yes/No.

Two profiles have identical targets and partitions:

| Profile | Prompt context | Intended interpretation |
| --- | --- | --- |
| `verified_news_only` | Historically verified prior headlines, with separate fixture and tournament streams | Public news context; no claim the wallet saw it |
| `execution_history_proxy` | The same news plus strictly earlier saved executions by that wallet | Exploratory execution-time proxy; historical availability of executions is unverified |

Contract and wallet identifiers are opaque. Final team assignments, the current
market registry, results, and kickoff metadata are excluded from prompts because
their historical versions are not established. Consequently, this release cannot
teach semantic team-to-contract reasoning for unseen contracts. Historically
versioned market descriptions are needed for that extension. There are no invented
belief summaries or assistant reasoning labels.
Prior-execution prompts retain raw token IDs; past Yes/No labels are available in
the audit but are not added as historical registry features. This further limits
interpretation of prior positions in unseen contracts.

## Filtering and coverage

A wallet qualifies only if its **total saved observation count across all 312
contracts is between 1 and 19**. The count includes both sides, all outcomes,
pregame and in-play activity, and all saved periods. It is not reset per match.
This is a low-observed-activity heuristic, not a verified market-maker classifier.

The cohort contains **206,474 wallets and 969,024 observations**, retaining all
104 fixtures and 312 contracts. The original captured corpus contains 252,612
wallets and 11,858,584 observations. All original datasets remain unchanged.

Each profile has **954,951 examples**: 533,858 training, 253,877 validation,
and 167,216 test. The remaining 14,073 observations stay in the audit because
fixture membership and time boundaries disagree. All exported targets have prior
verified broad tournament news, but only **1,906 (0.20%)** have verified prior
direct fixture news. There are 750,032 exported targets with prior executions in
the proxy profile. Treat the two profiles as alternative comparisons, not as
independent observations to concatenate.
Direct-news coverage reaches only **7 of 104 fixtures**; the other 97 have no
verified prior direct match news in this snapshot. The
[per-fixture coverage report](../reports/tournament_sft_fixture_coverage.json)
lists every match, its retained and exported counts, and both news coverage levels.

The source is the previously saved per-contract subset, which also preserves the
original counts for every wallet/contract pair. Recovery is exact **within the
captured snapshot**: any wallet with fewer than 20 observations in total necessarily
had fewer than 20 in every contract, so all of its saved rows survived the earlier
filter. The new script verifies pair counts, total counts, and retained histories.

**Collection remains partial:** 19 of 312 API histories were unfinished. Missing
activity can cause a genuinely active wallet to pass the filter. Selection uses
whole-period counts, including future activity relative to some targets, so cohort
membership is retrospective. Counts appear only in the audit, never in prompts.
Source observations are not chain-reconciled canonical fills or verified human
decisions. Prior quantity discrepancies and ambiguous receipt mappings remain.

## News attribution

Eligibility requires both the captured headline version and its tournament or
fixture relationship to have verified availability **strictly before** the target
timestamp. A publication date alone is insufficient. Select the highest eligible
historically ordered version of each item, then the latest eight items per scope.
The context catalog retains every eligible item ID even when the prompt is bounded.

Direct match links and broad tournament relevance are separate. A missing direct
news list means no verified prior direct link was recovered; no link is fabricated.
The catalog contains bibliographic metadata, short headlines, and inherited
archive-evidence descriptors, not full articles or HTML. Export rechecks those
descriptors; it does not freshly download or reverify the original archive bodies.
Attribution establishes public relevance, not actor exposure or causation.

## Splits and leakage controls

[`configs/tournament_sft_v1.json`](../configs/tournament_sft_v1.json) freezes:

| Split | Disjoint fixtures | Observation time constraint (UTC) |
| --- | --- | --- |
| Train | 72 group fixtures | Before 2026-06-28 12:00 |
| Validation | 24 round-of-32 / round-of-16 fixtures | From 2026-06-28 12:00, before 2026-07-09 00:00 |
| Test | 8 quarterfinal-and-later fixtures | From 2026-07-09 00:00 |

Both fixture membership and time must agree. Earlier trading on held-out fixtures
and late trading on training fixtures are retained in the audit and excluded as
targets. The final schedule is used only for offline partition assignment.
Duplicate observation IDs, invalid targets, and transactions crossing splits are
quarantined. All observations in the target transaction and observations at equal
or later timestamps are excluded from its execution-history prompt.

Wallets may recur across splits. The manifest reports seen and unseen evaluation
wallets. Fixture separation applies to **targets**: proxy prompts in any split
can contain strictly earlier captured executions from contracts assigned to
another split. Purged prefixes can therefore supply observed history without
becoming training targets. This is a sequential-context evaluation, not a claim
that held-out contract identifiers never appeared in training prompts. No complete
global wallet history is claimed.

## Files and reproducibility

| Path inside the release | Contents |
| --- | --- |
| `{profile}/{split}/part-*.jsonl.gz` | `{"messages": [...]}` rows with system, user, and assistant turns |
| `audit/part-*.jsonl.gz` | Every retained observation, reported action, context IDs, history row IDs, split, exclusions, and training-row locations |
| `contexts.jsonl.gz` | Deduplicated prior-news states and selected prompt item IDs |
| `source_news.jsonl.gz` | News versions and availability evidence descriptors |
| `source_news_links.jsonl.gz` | Fixture links and their eligibility evidence |
| `split_policy.json` | Exact frozen partition policy |
| `manifest.json` | Counts, scientific limitations, and SHA-256 hashes of generated artifacts |

Build from the saved source without changing it:

```bash
python scripts/filter_tournament_wallets.py \
  --database /path/to/world_cup_lt20.sqlite \
  --output data/tournament_lt20/attribution.sqlite \
  --threshold 20 --require-tournament-coverage \
  --report data/tournament_lt20/filter_report.json

python scripts/prepare_sft.py \
  --database data/tournament_lt20/attribution.sqlite \
  --output datasets/world_cup_2026_tournament_lt20_v1 \
  --split-policy configs/tournament_sft_v1.json \
  --shard-rows 20000 --news-limit 8 --allow-partial

python scripts/validate_sft.py \
  --dataset datasets/world_cup_2026_tournament_lt20_v1 \
  --report data/tournament_lt20/validation.json
```

These commands refuse to overwrite existing output. `--allow-partial` explicitly
permits the captured corpus; it does not certify source completeness or remove
scientific blockers. `format_ready_for_sft` means the serialized training format
is usable. `prospective_training_ready` and the original broader `training_ready`
remain false.
When rebuilding after cloning this release, select a fresh output directory such
as `data/rebuilt_sft` and point the validator there; the committed dataset already
occupies the example release path.

The validator streams every audit and training row. It verifies file checksums,
exact target equality, complete prior-news version selection, prompt-field
allowlists, execution-history exclusions, and fixture/time separation. Its
published result is [the full SFT validation report](../reports/sft_validation.json).
The exporter also reads every completed gzip stream through its footer, verifies
CRC and row counts, and refuses publication if compression is incomplete.

## Inspect and load

The repository contains the complete prepared subset, not only a preview. Clone it
and open a few rows using only Python's standard library:

```python
from pathlib import Path
import gzip, itertools, json

root = Path("datasets/world_cup_2026_tournament_lt20_v1")
first = sorted((root / "execution_history_proxy/train").glob("*.jsonl.gz"))[0]
with gzip.open(first, "rt", encoding="utf-8") as stream:
    for line in itertools.islice(stream, 3):
        print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
```

For Hugging Face Datasets, after installing your chosen training environment:

```python
from datasets import load_dataset
from pathlib import Path

root = Path("datasets/world_cup_2026_tournament_lt20_v1")
profile = "execution_history_proxy"  # use verified_news_only for the comparison
dataset = load_dataset("json", data_files={
    split: [str(p) for p in sorted((root / profile / split).glob("*.jsonl.gz"))]
    for split in ("train", "validation", "test")
})
```

The conversational format is supported by
[TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) and the
[Datasets JSON loader](https://huggingface.co/docs/datasets/en/loading).
Choose and pin a model, tokenizer, chat template, and training-library versions
before training. For assistant-only loss, TRL requires a compatible template with
assistant generation masks. Verify those masks and measure actual token lengths;
do not silently truncate away the final action. No model-specific tokenizer or
trainer smoke test has been run in this standard-library preparation environment.
Record the base model's pretraining cutoff and audit tournament-result memorization,
as described in [the study plan](study_plan.md); prompt filtering alone cannot
remove knowledge already present in model weights.

First compare majority/prior-frequency baselines and the two profiles on the
frozen validation partition. Report JSON validity, side/outcome classification,
price error, and share error (including a log-scale measure), with wallet/fixture
grouped uncertainty and separate seen/unseen-wallet results. Keep the test set
for the final evaluation. An observed improvement establishes retrospective
predictive value only. Mechanistic interpretation requires successful held-out
prediction and controlled interventions; generated rationales are not evidence
of traders' beliefs.
