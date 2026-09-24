# Training from actor conversations

## Current status

The complete v3 export is available for all 104 fixtures and 312 binary
contracts. Full independent validation remains pending. Export counts and
inspection instructions are in the [dataset guide](actor_sequence_dataset.md),
and the [release status](../reports/actor_sequences_release_status.json) records
the checks completed so far.

Before training, require a `passed` report at
`reports/actor_sequences_validation.json` that binds the exact
[release manifest](../datasets/world_cup_2026_actor_sequences_v3/manifest.json)
and includes the all-row token recount. That final report is not yet published,
and the messages-only loader deliberately refuses training input export without
it. An export manifest or [bounded smoke test](../reports/actor_sequence_smoke.json)
does not establish full-release readiness. No model has been trained by this pipeline.

## What one row means

An actor conversation is one JSONL record containing a `messages` list. Its
successive user messages introduce context and assistant messages contain the
observed targets. Earlier messages in that same record are available through
causal attention. Later messages cannot condition an earlier prediction.

Outer `actor_id`, `sequence_id`, `profile`, `split`, source references, token
counts and audit fields are metadata. Feed only `messages` to the model. Do
not serialize the entire dataset record as a prompt: outer fields may include
future-dependent selection counts or target provenance.

Sorting separate training rows by actor does not create memory. Neither does
putting two chunks for the same actor next to one another in a training file.
A chunk's initial user message must supply the retained historical context.
Its stated scope matters: missing history is not evidence of no activity.

## Prediction tasks and context updates

The release path is `datasets/world_cup_2026_actor_sequences_v3`.
It has two task views:

- `conditional_trades` predicts captured executions given that they occur at
  the stated execution-time proxy. It preserves initial executions as targets
  without claiming to predict their occurrence. Equal-time observations are
  jointly predicted, with no asserted internal causal order.
- `scheduled_windows` predicts all captured executions during a specified
  future window using only context available before the window starts. An
  empty answer is `NO_TRADE`. The default horizon is 15 minutes. The monitored
  set contains eligible contracts this actor entered in the preceding 24
  hours. Initial or re-entry executions needed to establish monitoring are
  not automatically prospective positive targets.

Keep these task instructions explicit if combining the views. A conditional
answer and a scheduled window containing that execution are different tasks,
not two independent economic decisions. Report their metrics separately.

The initial scheduled user message gives the full monitored `markets` list
and `horizon_seconds`. Later turns inherit omitted values. `markets_add` and
`markets_remove` update the monitored list before the current query. An
omitted field does not mean an empty scope. The audit preserves each turn's
full monitored contract list. Conditional turns explicitly identify their
queried contracts independently of any older contracts still in context.

News is introduced when it becomes eligible before a query. A headline arriving
inside a prediction window cannot enter that window's initial prompt. Repeated
unchanged headlines need not be inserted again in the same conversation.
Updates must cover time since the preceding retained query, including skipped
negative windows. An omitted news field does not erase earlier context.

Private initial beliefs, actor exposure to headlines, intended orders and true
holdings are unobserved. Do not fill those fields with invented explanations.
`NO_TRADE` means zero captured execution observations in the monitored scope,
not no orders, no activity elsewhere, or a deliberate decision against trading.
Source observation bounds do not certify continuously open trading. Incomplete
or invalid capture must not be relabeled as inactivity.

The default cohort retains actor–binary-contract pairs with at most 20 captured
observations, inclusive, counting BUY/SELL and both Yes/No outcomes together.
It removes an ineligible pair completely rather than keeping its first 20
observations. Final whole-period counts make cohort membership retrospective.
The threshold is an activity heuristic, not a human/bot label. An actor can
have many more than 20 observations across multiple contracts. This differs
from v2's fewer-than-20 rule across the actor's entire captured tournament.
The actual manifest records the selected filter scope and takes precedence
over these defaults.

## Chunking and sampling

Default chunks target at most 8,192 reference-template tokens and 128 target
turns. Their initial context retains up to 16 exact earlier executions plus
small numerical activity summaries. The summary is computed from strictly
earlier observations in its declared scope. Older exact history can therefore
be lost across chunk boundaries. A chunk boundary uses only past turns, the
current user context and a fixed 1,024-token response reservation. It cannot depend on
the unseen answer's length. Check separately flagged oversized conversations:
an initial context or unexpectedly large joint answer may exceed the budget,
including in a conversation with several turns. Never assume every row fits
merely because a limit is configured.

The completed export manifest currently records a largest conversation of
10,301 reference tokens and 872 conversations above the 8,192-token target;
these counts still await the independent full recount. Plan for a context
window of at least 16,384 tokens with this reference tokenizer/template to
preserve the exported conversations. Remeasure with the actual training
model's tokenizer and template: a nominal 16k context for a different model
is not automatically equivalent. Do not let a trainer silently truncate rows
to 8,192 tokens.

The default training policy keeps all eligible positive windows and samples
negative windows with probability 0.05. Validation and test retain all eligible
negative windows by default. Read the release policy for actual overrides and
sampling counts. Preserve natural evaluation prevalence or account for the
sampling probabilities when reporting population metrics.

Shuffle complete conversations if desired. Do not shuffle their turns or
silently truncate them. Rebuild oversized rows with explicit context limits,
or select a supported larger window, while preserving every intended target
exactly once within its task view.

## Load only the conversation messages

Choose one task profile and one split. After the full release passes validation,
create a trainer input file:

```bash
python scripts/export_actor_messages.py \
  datasets/world_cup_2026_actor_sequences_v3 \
  --validation reports/actor_sequences_validation.json \
  --profile scheduled_windows --split train \
  --output data/sft_inputs/scheduled_windows_train.jsonl.gz
```

Run separately with `--split validation` and a different output filename for
evaluation. Choose `conditional_trades` instead to predict execution details
conditional on an execution being observed.

Every output row has exactly one outer key, `messages`, containing the complete
original conversation. The script excludes audit metadata, streams one row at a
time, preserves all turns, and never truncates or silently filters oversized
conversations. Its report includes the maximum reference length and the count
requiring a longer context window. The real training tokenizer, template and
context limit must still be checked separately.

The exporter requires a completed, non-sample manifest and a matching passed
full validation report, including the actual all-row token recount. It verifies
the selected shards' sizes, checksums, rows and totals before publishing the
new output filename. It writes outside the frozen release and refuses to
overwrite an existing output.

Hugging Face Datasets can stream the exported JSONL gzip file:

```python
from datasets import load_dataset

train = load_dataset(
    "json",
    data_files={"train": "data/sft_inputs/scheduled_windows_train.jsonl.gz"},
    split="train",
    streaming=True,
)
example = next(iter(train))
assert set(example) == {"messages"}
```

Pass complete conversations to the trainer's chat-data interface. For a custom
loader, this repository iterator avoids creating a second corpus copy:

```python
from scripts.export_actor_messages import iter_messages

rows = iter_messages(
    dataset="datasets/world_cup_2026_actor_sequences_v3",
    validation="reports/actor_sequences_validation.json",
    profile="scheduled_windows",
    split="train",
)
for example in rows:
    messages = example["messages"]
    # Hand this whole conversation to the trainer's dataset adapter.
```

Each selected shard is verified before its first row is yielded. Consume this
iterator to exhaustion for final aggregate and file-stability checks. The CLI
additionally withholds its output filename until all checks finish. These
helpers prepare inputs; they do not choose a model or run training.

## Measure tokens without model weights

The reference tokenizer is `Qwen/Qwen3-0.6B` at immutable revision
`c1899de289a04d12100db370d81485cdf75e47ca`. This is a token-length reference,
not a choice of training model. No weights are required.

```bash
python -m pip install 'transformers==4.57.6' 'tokenizers==0.22.2' 'jinja2==3.1.6'
```

Download tokenizer files only:

```python
from pathlib import Path
from shutil import copyfile

from huggingface_hub import snapshot_download

tokenizer_dir = Path("data/tokenizer_reference/qwen3_06b")
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"],
    local_dir=tokenizer_dir,
)
copyfile(
    "configs/actor_sequence_tokenizer_reference.json",
    tokenizer_dir / "reference_manifest.json",
)
```

The committed [reference receipt](../configs/actor_sequence_tokenizer_reference.json)
records the exact upstream file sizes and SHA-256 hashes. Copy it unchanged:
the release also fingerprints these receipt bytes as `reference_manifest.json`.
Generating equivalent metadata with different formatting would change that
fingerprint. No model weights are downloaded.

After the profiles have been exported:

```bash
python scripts/measure_sequence_tokens.py \
  datasets/world_cup_2026_actor_sequences_v3/conditional_trades \
  datasets/world_cup_2026_actor_sequences_v3/scheduled_windows \
  --tokenizer data/tokenizer_reference/qwen3_06b \
  --chat-template configs/actor_sequence_chat_template.jinja \
  --max-tokens 8192 \
  --limit-per-group 1000 \
  --prefix-baseline-limit-per-group 100 \
  --report reports/actor_sequence_tokens_sample.json
```

Pass profile directories, excluding audit directories. A positive
`--limit-per-group` measures the first N rows of each profile/split, not a random
or full-dataset sample. Use zero to measure every row. Reports include file,
tokenizer and template hashes, library versions, total tokens, nearest-rank
median/95th percentile, maximum length and oversized-row counts. The script
does not truncate and exits with status 2 if a measured row exceeds its limit.

Counts use the actual chat rendering, including control tokens. The repository
template uses plain Qwen/ChatML role delimiters with no thinking block. It
replaces the upstream Qwen3 template's thinking behavior. Use the same template
for reference chunking and measurement. For inference, a generation prompt
appends the assistant header before answer generation. Remeasure using the
actual training tokenizer and template before training another model.

The prefix comparison repeatedly serializes each conversation prefix through
each assistant answer, then compares that total with the same complete
conversation. It is not a comparison against v2 and does not estimate GPU
speedup. Attention length, padding, packing and the trainer determine runtime.

`poly_world_cup.sequence_tokens.TokenBudget` accelerates preparation by caching
exact per-message counts only for the fingerprinted reference backend and
template. Its non-stripping special-token boundaries make message token IDs
additive; startup checks and tests verify this, including Unicode and literal
control-token strings. Unknown configurations fall back to full chat rendering.
The cache is bounded by 4,096 entries and 2 MiB including a per-entry allowance.
Tokenizer components must remain immutable while this counter is in use.

## Loss masking and evaluation

Use a causal model and score eligible assistant answers. User context, system
instructions, historical prefixes and padding should not receive label loss.
The upstream Qwen template has no generation annotations. The repository
template adds them around assistant contents and their `<|im_end|>` token.
Role headers and trailing newlines remain outside supervised spans.

Real-tokenizer tests check answer/end-token masks and confirm that words such
as "assistant" or "NO_TRADE" inside a user headline do not become labels.
The measurement script also probes one row per profile/split. These checks do
not validate an actual trainer. Before training, inspect its produced labels:

1. Decode tokens whose labels are not `-100`. Check every intended answer and
   the selected end-of-answer convention.
2. Check that all intended assistant turns receive loss and user/system spans
   remain masked. Content-only token counts are not loss-mask counts.
3. Confirm the chosen template is applied consistently at inference, and
   remeasure token lengths after any template change.
4. Verify causal attention within conversations and attention separation
   between independently packed conversations.

Earlier true answers can condition later predictions during training. This is
teacher forcing. Evaluation with earlier observed executions tests a different
setting from rolling forward on the model's own predictions. State which one
is used, and never update model weights from held-out answers.

Hold out whole matches and their linked contracts. For the stricter unseen-match
experiment, remove their executions from training inputs and summaries as well
as labels. Keep windows within split time boundaries. Record the exact frozen
base checkpoint and adapters. A checkpoint released before the evaluated
observations helps rule out those observations being memorized in pretraining.

For scheduled predictions, report trade precision/recall and probability
calibration or a proper probability score alongside the natural positive rate.
Accuracy alone can reward predicting inactivity almost everywhere. Report
side/outcome, price and size errors separately, including missed positive
windows. Report by match and by actors seen/unseen during training.

Serialization and token checks do not establish predictive performance or
faithful recovery of an actor's reasoning. No training run is implied here.

## Primary references

- [Hugging Face chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)
- [Hugging Face dataset loading](https://huggingface.co/docs/datasets/en/loading)
- [TRL supervised fine-tuning](https://huggingface.co/docs/trl/en/sft_trainer)
- [Pinned Qwen tokenizer configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
