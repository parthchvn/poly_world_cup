# Actor conversations for World Cup trading

Version 3 adds chronological actor conversations while preserving earlier
releases. Each compressed JSONL line is one complete, bounded conversation for
one actor, one prediction task and one split. The pipeline does not train a model.

## Release status: exported, pending full validation

The complete export covers **104 fixtures and 312 binary contracts**. These are
scope counts, not proof that every historical execution was captured. The
[release status](../reports/actor_sequences_release_status.json) distinguishes
completed integrity checks from the pending independent validation. The export
can be inspected now. The final report at `reports/actor_sequences_validation.json`
is not yet published and must say `passed`, bind the exact
[release manifest](../datasets/world_cup_2026_actor_sequences_v3/manifest.json)
by SHA-256, and confirm the full token recount before the training loader will
accept it. An export manifest alone is not a passing validation result.

| Export measurement | Value, pending independent validation |
| --- | ---: |
| Selected actors | 255,438 |
| Selected API observations | 3,974,200 |
| Observations excluded from targets | 61,943 |
| Conditional conversations / target observations | 750,420 / 3,912,257 |
| Scheduled conversations / forecast windows | 924,775 / 20,827,084 |
| Retained `NO_TRADE` windows: train / validation / test | 1,367,751 / 12,142,690 / 6,524,504 |
| Reference tokens / largest conversation | 8,283,490,246 / 10,301 |
| Conversations exceeding the 8,192-token target | 872 |
| Fixtures with retained observations in each profile | 104 |

Counts describe the exported files and remain subject to independent full
reconciliation. Even a passing report establishes consistency within the
recorded capture. Independent completeness of all canonical on-chain executions
remains unproven. No model has been trained, and no predictive-accuracy result is implied.

## How to read a row

A `.jsonl.gz` file is gzip-compressed newline-delimited JSON. Decompress it and
parse each line as a JSON object. Its `messages` array starts with a system
instruction, then alternates user context and assistant targets in time order.
User and assistant `content` values are themselves JSON strings; parse those
strings separately when inspecting their fields. The system content is text.

| Field | Meaning | Send to the model? |
| --- | --- | --- |
| `messages` | The complete ordered conversation | Yes, through the trainer's chat template |
| `actor_id`, `profile`, `split` | Actor, prediction task and dataset partition | Outer metadata only; relevant context is already inside messages |
| `sequence_id`, `chunk_index` | Identity and position of this actor's conversation chunk | No |
| `token_count`, `requires_long_context` | Reference-template length and whether it exceeds the configured target | No; use to choose a compatible training context |
| `turn_audit` | Query/window times, target/history IDs, summary checks and news attribution | No; includes target information for validation |

**Attention connects earlier turns within the same row. It does not carry
memory between separate training rows**, even if those rows have the same
actor and are adjacent. Keep every conversation's turns in order; shuffle
whole conversations if needed. During standard SFT, later targets see the true
earlier assistant answers through causal attention. They cannot see later turns.
At inference, supply the corresponding earlier observed context and actions;
feeding earlier model predictions instead is a separate rollout evaluation.

A row count is a **conversation count**, not a trade count. An assistant target
can contain several captured execution observations; a `NO_TRADE` target
contains none. Conditional targets can batch observations at the same timestamp,
while scheduled targets represent forecast windows. The two profiles can use
the same observations, so their target totals must not be added as unique trades.
These are provider observations, not independently reconciled fills or orders.

## Two prediction tasks

| Profile | What is known at the query | What the assistant predicts |
| --- | --- | --- |
| `conditional_trades` | Execution-time proxy and queried contracts; an execution is observed | Recorded BUY/SELL, outcome, shares and price for the observations at that time |
| `scheduled_windows` | Fixed query time and contracts already monitored from past activity | Captured observations over the next 15 minutes, or `NO_TRADE` |

`conditional_trades` preserves every eligible observed execution as a target.
Observations with the same timestamp form one joint target, with no claimed
internal causal order. This profile does not predict whether trading occurs.

`scheduled_windows` uses UTC clock-grid intervals `[query, query+900s)`, not
windows chosen from future inter-trade gaps. A contract enters an actor's
monitored set only after an earlier valid observation there and remains eligible
for 24 hours after the latest observation. The whole forecast interval must lie
within the declared captured-data bounds. A first observation in a new contract,
or one after a long inactive period, remains in the conditional profile but is
not automatically a scheduled forecast target.

A headline appearing inside a forecast interval becomes available only at a
later query. The default retains all positive windows, a stable-hash sample of
5% of training negatives, and all validation/test negatives. The policy and
eligible class counts record that sampling; these alternative tasks should be
trained and evaluated separately.

`NO_TRADE` means **no captured observation in the monitored contracts during
that interval**. It is not a label for deliberate inactivity, no activity in
other contracts, or complete on-chain inactivity. The API capture requests a
minimum size of 0.000001 tokens and includes maker and taker observations as
exposed by the provider. Exhausting API cursors does not independently audit
canonical completeness.

## Filtering, context and splits

The v3 default retains an **actor–binary-contract pair with at most 20 captured
observations**, including exactly 20. BUY/SELL and both Yes/No outcomes count
together for that contract. This differs from v2's global rule of **fewer than
20 observations for the actor across the captured tournament**. An actor can
therefore have many more than 20 retained observations across multiple contracts
in v3. This is not the first 20 observations from an otherwise excluded pair.

The filter uses the full captured period, not just a training split. It is a
retrospective cohort rule, not proof that retained actors are human or excluded
actors are market makers. `--filter-scope actor` instead drops an actor if any
tournament contract exceeds 20. The release's `policy.json` records the setting.

The first turn of each chunk provides verified initial contract semantics,
the latest eight distinct verified headlines per relevant scope, a numerical
summary of strictly earlier same-split observations, and up to 16 recent exact
observations. Summary fields include observation counts, market counts and mean
BUY/SELL notional (`shares * price`). Holdings, holding durations and private
beliefs cannot be identified from these observations and are not fabricated.

Later turns add new headlines and newly observed history. Earlier messages
remain available through attention, so their full content need not be repeated.
A newly introduced fixture receives its bounded initial context. News advances
for every fixture already introduced in the conversation and must be verified
available strictly before the query. Public relevance does not establish that
an actor read or reacted to a headline. `context_catalog.json` preserves source
URLs, identities, timestamps and compact alias mappings; short IDs reduce prompt
length. Omitted delta fields do not erase already introduced context.

The frozen fixture/time split assigns 72 group fixtures to training,
24 round-of-32/round-of-16 fixtures to validation, and eight later fixtures to
testing, with global time boundaries in
[the split configuration](../configs/tournament_sft_v1.json). Incompatible
fixture/time observations remain in the observation archive and quarantine
ledger but are excluded from targets. Source-valid earlier observations within
the same fixture partition can supply inference history, including observations
before a held-out target period. Histories and summaries cannot cross fixture
partitions. The base model's release date and training cutoff require a separate
check; the dataset partition alone does not establish that it never saw a match.

## Files, inspection and training input

- `{conditional_trades,scheduled_windows}/{train,validation,test}/part-*.jsonl.gz`
  contains conversations with audit metadata.
- `actor_index/` maps actors and chunks to exact shard/line locations.
- `observations/` retains every observation selected by the activity filter;
  `quarantine/` records exclusions from targets.
- `source_evidence/pair_counts/` contains the full activity-count ledger,
  including excluded pairs; `source_evidence/pages/` contains available provenance.
- `policy.json`, `split_policy.json`, `source_coverage.json`,
  `context_catalog.json` and `fixture_coverage.json` describe scope and attribution.
- `manifest.json` binds artifact hashes and counts to the validation report.

Print an actual complete row without shortening it:

```bash
python scripts/inspect_actor_sequence.py datasets/world_cup_2026_actor_sequences_v3 \
  --profile scheduled_windows --split train
```

Add `--actor 0x...` to select a known actor or `--messages-only` to inspect just
the conversation. Inspection is not a substitute for validation.

After the full release passes validation, use the
[messages-only loader/exporter](../scripts/export_actor_messages.py) to stream
one profile/split or create a trainer input file outside the frozen dataset:

```bash
python scripts/export_actor_messages.py datasets/world_cup_2026_actor_sequences_v3 \
  --validation reports/actor_sequences_validation.json \
  --profile scheduled_windows --split train \
  --output data/training/scheduled_windows_train.jsonl.gz
```

The loader requires a matching passed validation report and full token recount.
It preserves every message and turn, excludes outer audit metadata, verifies
selected shard checksums, and refuses to overwrite an existing output. It does
not silently remove oversized conversations. Follow the
[training guide](actor_sequence_training.md) for direct streaming, assistant-only
loss masking, context limits and evaluation.

## Build and validate

Reassemble and extract the existing `world_cup_lt20.part01` and `.part02` ZIP
archive. Keep its `MANIFEST.json` beside `world_cup_lt20.sqlite`, then run:

```bash
python scripts/recover_sequence_trades.py \
  --archive /path/to/world_cup_lt20.sqlite --collect
```

Recovery resumes missing contract histories and the additional exactly-20
pairs, then produces a closed SQLite source. Eligible pair counts must reconcile
to the full published tournament count ledger. A selected-only source explicitly
declares that scope. Its earliest/latest selected observations provide
conservative scheduled-window bounds, not full-contract first/last timestamps.
Recovery reports distinguish archived normalized rows from replayed raw captures.

```bash
python scripts/prepare_actor_sequences.py \
  --source data/sequence_v3_recovery/source.sqlite \
  --output datasets/world_cup_2026_actor_sequences_v3 --workers 8
python scripts/validate_actor_sequences.py \
  --dataset datasets/world_cup_2026_actor_sequences_v3 \
  --source data/sequence_v3_recovery/source.sqlite \
  --evidence datasets/world_cup_2026_tournament_lt20_v2_evidence \
  --tokenizer data/tokenizer_reference/qwen3_06b \
  --report reports/actor_sequences_validation.json
```

Reference lengths use the pinned Qwen3 tokenizer and supplied chat template,
with an 8,192-token target. Chunk boundaries depend only on earlier turns,
current user context and a fixed response reservation; an unseen answer cannot
change its prompt. An unexpectedly long answer or initial context is preserved
and marked `requires_long_context`. Such rows need a larger compatible context
window or a separately documented transformation. The independent validator
recounts every conversation using the complete chat template without truncation.
