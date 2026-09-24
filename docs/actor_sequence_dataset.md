# Actor conversations for World Cup trading

This pipeline adds a separate version 3 release. Earlier datasets stay unchanged.
The full release is not ready until its manifest has a passing independent
validation report. The pipeline itself does not train a model.

Each compressed JSONL row contains a complete bounded conversation for one
actor, one prediction task and one split. Later turns can attend to earlier
turns in that same row. Separate training rows do not share attention, even
when sorted by actor. At each chunk boundary, a numerical activity summary and
up to 16 earlier observations supply bounded memory. Training should shuffle
complete conversations, not rearrange their turns.

## Two tasks

`conditional_trades` preserves every eligible observed execution as an action
target. The query time and queried markets are known. Executions with the same
timestamp form one joint target, with no claimed internal causal order. This
profile predicts recorded attributes conditional on an execution occurring.

`scheduled_windows` forecasts a fixed 15-minute interval `[query, query+900s)`.
Queries lie on a UTC clock grid. A contract enters an actor's monitored set only
after that actor has an earlier valid observation there. It remains eligible
for 24 hours after its latest observation. The whole forecast interval must
lie within the declared captured-data bounds. The label is the list of all
captured observations in the monitored contracts, or `NO_TRADE` if that list is
empty. A first trade in a new contract, or one after a long inactive period,
is retained in the conditional profile but is not automatically a scheduled
forecast target. These are alternative tasks, not interchangeable examples.

Windows are not constructed from future inter-trade gaps. A headline appearing
inside a forecast interval becomes available only at a later query. The
default retains all positive windows, 5% of training negatives using a stable
hash, and all validation/test negatives. Sampling probabilities and eligible
class counts are recorded so evaluation can use the natural class balance.

`NO_TRADE` means no captured observation in that monitored set and time window.
It does not establish deliberate inactivity, absence of trades elsewhere, or
complete on-chain history. The API capture has a requested minimum size of
0.000001 tokens and includes both maker and taker observations as exposed by
the provider. A complete cursor traversal is not an independent coverage audit.

## Filtering, context and splits

The default retains an actor–binary-contract pair when its complete captured
count is **at most 20**, counting BUY/SELL and both outcomes together. This is
a retrospective activity filter, not proof that an actor is human or that
excluded actors are market makers. An actor may have many retained markets.
`--filter-scope actor` instead drops an actor if any tournament contract exceeds
20. Counts always come from the full tournament ledger, never a training split.

The first turn supplies historically verified initial contract semantics,
the latest eight distinct verified headlines per relevant scope, a summary of
strictly earlier same-split observations, and the bounded exact history.
Summary fields are observation counts, market counts, and mean BUY/SELL
notional (`shares * price`). Holdings and holding durations are not identifiable
from these API observations and are not invented. No private prior belief is
fabricated. Initial public background is evidence, not an actor belief label.

Later turns contain new headlines and newly observed history only. News
advances for every fixture already introduced in the conversation. A newly
introduced fixture also receives its bounded initial context. All news must
be verified available strictly before the query. Headlines are attributed by
public relevance, not by evidence that the actor read or reacted to them.
Source URLs, identities, timestamps and compact alias mappings remain in
`context_catalog.json`; the prompt uses short IDs to reduce tokens.

The existing frozen fixture/time split is retained: 72 group fixtures in
training, 24 round-of-32/round-of-16 fixtures in validation, and eight later
fixtures in testing, with global time boundaries as specified in
`configs/tournament_sft_v1.json`. Incompatible fixture/time observations remain
in the canonical observation files and quarantine ledger but are not targets.
Source-valid earlier observations for the same fixture partition can supply
inference history, including observations before the held-out target period.
Histories and numerical summaries cannot cross fixture partitions.
This is an offline evaluation partition; choosing a released base model still
requires checking its own training cutoff separately.

## Files and inspection

- `{conditional_trades,scheduled_windows}/{train,validation,test}/part-*.jsonl.gz`
  contains conversations and turn-level audits.
- `actor_index/` maps each actor/chunk to its exact shard and line.
- `observations/` retains every observation selected by the activity filter.
- `quarantine/` records exclusions from training, without deleting observations.
- `source_evidence/pair_counts/` contains the full activity-count ledger,
  including excluded pairs; `source_evidence/pages/` contains available provenance.
- `policy.json`, `split_policy.json`, `source_coverage.json`,
  `context_catalog.json`, and `fixture_coverage.json` record release assumptions.
- `manifest.json` binds the artifacts to SHA-256 hashes and actual counts.

Print an actual complete row:

```bash
python scripts/inspect_actor_sequence.py datasets/world_cup_2026_actor_sequences_v3 \
  --profile scheduled_windows --split train
```

Add `--actor 0x...` to select a known actor and `--messages-only` to show the chat
portion. Outer fields such as `turn_audit`, token counts and source identities
are for validation, not model inputs. Feed only the `messages` field to SFT.

## Build and validate

Reassemble and extract the existing `world_cup_lt20.part01` and `.part02` ZIP
archive. Keep its `MANIFEST.json` beside `world_cup_lt20.sqlite`, then run:

```bash
python scripts/recover_sequence_trades.py \
  --archive /path/to/world_cup_lt20.sqlite --collect
```

This resumes the missing contract histories and the additional exactly-20
pairs, then recovers the selected observations into a closed SQLite database, with all
eligible pair counts exactly reconciled to the full published tournament count
ledger. A recovered selected-only database must explicitly declare that scope.
Conservative interval bounds based on the earliest/latest recovered selected
observations narrow scheduled opportunities; they must not be described as
full-condition first/last timestamps. Source recovery reports distinguish
archived normalized rows from newly replayed raw captures.

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

Reference token lengths use the pinned Qwen3 tokenizer and supplied assistant
loss template, with an 8,192-token target. Every completed conversation is
counted exactly. Chunk boundaries use only earlier turns, current user context
and a fixed response reservation, so an unseen answer cannot change its prompt.
An unexpectedly long answer or initial context is preserved and marked
`requires_long_context`; it is never silently truncated. Such rows need a
larger compatible context window or a separately documented transformation.
The independent validator recounts tokens with the complete chat template.
See [training setup](actor_sequence_training.md) for assistant-only masking and
the distinction between teacher-forced evaluation and rollout prediction.
