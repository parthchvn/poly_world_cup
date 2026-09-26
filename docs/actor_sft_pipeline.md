# Collect, convert, and train World Cup actor decisions

Use the current repository after the history cleanup. Existing collaborators
should use fresh clones rather than merging old history into the rewritten main.

```bash
git clone https://github.com/parthchvn/poly_world_cup.git
cd poly_world_cup
```

If already inside the fresh repository, update it with:

```bash
git pull --ff-only origin main
```

## 1. Collect several matches

Run the standalone builder from the repository root. One invocation collects one
binary market. For example, these bundled registry IDs refer to three different
World Cup matches:

```bash
python3 scripts/build_actor_dataset.py 1897035
python3 scripts/build_actor_dataset.py 1897038
python3 scripts/build_actor_dataset.py 1897059
```

| Market ID | Market |
|---|---|
| 1897035 | Mexico–South Africa draw |
| 1897038 | South Korea–Czechia draw |
| 1897059 | Germany–Curaçao draw |

Germany–Curaçao has a confirmed local collection run. Collection for the other
IDs still depends on accessible trade/ESPN sources with usable timestamps. If a
dataset already exists, use it; the builder deliberately refuses overwrites.

All completed exports live under `data/market_<id>/` by default. Keep these local.
For a substantial experiment, collect more matches. Three matches are only the
minimum for a train/validation/test smoke pipeline, not a robust benchmark.

## 2. Convert actor files to conversations

From the repository root on RunPod, using the Python environment and model
directory from the successful training run:

```bash
python scripts/prepare_actor_sft.py \
  --input-root data \
  --out datasets/world_cup_sft \
  --tokenizer /workspace/models/Qwen3.6-27B
```

The converter discovers completed `actor_market_intervals_v1` exports directly
inside `data/`. It ignores other directories such as the collection cache. It
rejects two exports of the same market instead of double-counting observations.
If there are several versions of a market in `data/`, select inputs explicitly:

```bash
python scripts/prepare_actor_sft.py \
  data/market_1897035 data/market_1897038 data/market_1897059 \
  --out datasets/world_cup_sft \
  --tokenizer /workspace/models/Qwen3.6-27B
```

On a Mac without the model tokenizer, omit `--tokenizer`. This needs only Python
3.11+ and its standard library. The resulting files have the same schema, but
exact length checks are deferred until the trainer tokenizes them. To check
before training on RunPod, rerun conversion there with `--tokenizer` and a new
output directory.

The converter never truncates, silently samples, or discards targets to fit a
token budget. `--max-length` defaults to 8192 when a tokenizer is supplied.
Overlong conversations fail with the source actor filename. Increasing the
limit requires compatible model/GPU capacity and the same training limit.
Automatic conversation chunking is not implemented.

Output:

| File | Contents |
|---|---|
| `train.jsonl` | Training conversations |
| `validation.jsonl` | Validation conversations |
| `test.jsonl` | Held-out test conversations; not opened by the trainer |
| `manifest.json` | Counts, source hashes, split identities and limitations |
| `split_plan.json` | Reusable assignment of fixture IDs to splits |
| `source_audit.jsonl` | Per-conversation actor-file hashes and original target row indices |

One JSONL line is one entire actor/binary-market conversation, with multiple
assistant decision targets. Counts of conversations and targets therefore differ.

## 3. Training smoke test, then full run

Use the existing working training environment. The model weights and training
dependencies must already be present; these scripts do not install or download
them. Use GPUs in the same machine/Pod.

```bash
python scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir datasets/world_cup_sft \
  --smoke
```

After that succeeds:

```bash
python scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir datasets/world_cup_sft
```

Use `--gpus 1 --gpu-ids 0` for one GPU, or `--gpus 4 --gpu-ids 0,1,2,3` for four.
Always specify the new `--dataset-dir`; the trainer's historical default points
to the older pilot export. A full run starts separately from its smoke run.

## Conversation and supervision contract

Each conversation starts with a system instruction and a user message containing
the actor ID, market question/outcomes, query time, first interval news, and an
empty `past_observed_trades` list. Later user messages contain the next query
time and new interval news. Earlier assistant messages provide the actor's
previous trade history within this market.

Each assistant answer is JSON:

```json
{"action":"TRADE","trades":[{"side":"BUY","outcome":"Yes","shares":"15.18","price":"0.31"}]}
```

This is an illustrative label, not a retrieved execution. Simultaneous executions
remain separate entries in the same `trades` array. Values retain their source
decimal strings. The target's execution values are not inserted into its user
message. Causal training attention prevents later answers from being used to
predict earlier ones.

The converter validates interval rows, but creates no `NO_TRADE` targets. Those
rows are retrospective gaps ending at the next trade; they are not prospective
samples of a decision to abstain. Each gap's news is taken once from the trade
row, avoiding duplication from its adjacent interval row. No additional news is
retrieved or invented during conversion.

This predicts execution attributes conditional on a trade being observed. It
does not predict whether/when a trade occurs, P&L, match resolution, or verified
human reasoning. History is within this actor/market conversation, not across
the actor's other markets. Source rows before an export's start boundary cannot
be reconstructed by the converter.

## Splits, validation, and limitations

By default, the converter groups all binary markets by ESPN fixture ID, orders
fixtures by kickoff time, then allocates earlier fixtures to training, the next
fixtures to validation, and the latest fixtures to test. Validation/test each
receive 10% of fixtures rounded down, with at least one each. At least one fixture
must remain in training. The allocation is by fixtures, not by target counts.

There is no fixture overlap between splits. Wallets may appear in different
splits in different matches. Sorting by kickoff does not enforce a strict
global execution-time cutoff because market trading periods may overlap. The
converter does not establish that held-out matches were unseen in base-model
pretraining. These require separate evaluation design.

Adding new fixtures can change the automatic split. To preserve an experiment,
use its saved `split_plan.json` with exactly the same fixture set:

```bash
python scripts/prepare_actor_sft.py --input-root data \
  --split-file datasets/world_cup_sft/split_plan.json \
  --out datasets/world_cup_sft_rebuilt \
  --tokenizer /workspace/models/Qwen3.6-27B
```

Conversion checks actor/market identities, interval continuity, grouped execution
times, decimal validity, strict prior news timestamps, manifest counts, duplicate
market inputs, and reserved chat markers. A failed build does not publish a
partial output directory. Existing outputs are never overwritten.

Market questions and mappings are retrospective metadata, explicitly marked as
not time-verified in prompts and the manifest. ESPN timestamps remain occurrence
proxies rather than verified publication times. This conversion preserves those
limitations; it does not certify historical information availability or causal
news attribution. The new prompts and splits need fresh evaluation rather than
assuming the earlier pilot's 72.5% validation category accuracy carries over.

Developer verification uses synthetic exports produced with the current
builder's row generator and the real trainer's input/masking functions. Actual
Qwen tokenizer length validation and H100 execution must run in the target
training environment.
