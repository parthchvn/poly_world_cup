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

## Collect and prepare in one command

The former preparation script is now included in `build_actor_dataset.py`.
`scripts/` contains only this builder and `train_world_cup_multigpu.py`.

To rebuild existing RunPod exports with market-price context, use a new actor
output root and the **same existing cache**:

```bash
cd /poly_world_cup
git pull --ff-only origin main

python3 scripts/build_actor_dataset.py sft 1897035 1897038 1897059 \
  --data-root /workspace/world_cup_actor_data/data_with_market_context \
  --cache /workspace/world_cup_actor_data/data/market_actor_cache \
  --reuse-existing \
  --http-transport curl \
  --out /workspace/datasets/world_cup_sft_with_market_context \
  --tokenizer /workspace/models/Qwen3.6-27B
```

This collects the selected markets, then validates their exports and writes
fixture-disjoint SFT splits. `--out` names the new SFT dataset; actor exports go
under `--data-root/market_<ID>`. The cache defaults to
`--data-root/market_actor_cache`, or use `--cache` explicitly. The requested split
is checked before trade collection: normally at least three distinct matches,
or two with `--train-validation-only`.

`--reuse-existing` explicitly reuses completed matching exports that contain
market context (`market_context_version: 1`). It rejects older exports that lack
these features. Collection filters and source options apply only to new exports.
Without this flag, existing actor exports cause an error. If collection fails
partway through, rerun with the same cache and `--reuse-existing`: finished
exports are retained, and unfinished trade captures resume. An exhausted capture
is reused without scraping its trade pages again. Building the price timeline
itself makes no additional API requests. An existing SFT output is never
overwritten; choose another `--out` when preparing a new version.

The separate `prepare` mode still accepts a set of entirely legacy exports, with
a warning that their conversations lack market context. Mixing legacy and
enriched exports is rejected. Preparation does not add prices to legacy actor
files; rebuild them from the saved captures using the command above. Already
trained adapters are unchanged. Train a new run with
`--dataset-dir /workspace/datasets/world_cup_sft_with_market_context` to learn
from the added inputs.

Match-specific files or overrides, such as `--espn-file`, cannot be shared across
multiple market IDs. Collect those markets individually, then run `prepare`.
The single-market command remains supported unchanged. The following sections
show separate collection and preparation when you need those controls.

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

If Python API requests fail with a connection reset but terminal `curl` can
retrieve the same URL, select curl for all live ESPN/Polymarket requests:

```bash
python3 scripts/build_actor_dataset.py 1897059 --http-transport curl
```

This requires the `curl` executable, with no additional Python packages.
Keep the same `--cache` and `--out` paths when resuming a failed collection.
Both transports share cached responses and resumable trade pages. HTTP errors,
transfer failures, and invalid JSON cannot become successful cached responses.
Changing transports does not guarantee access from every network.

Live collection pauses one second after each HTTP attempt and retries transient
network failures, HTTP 429, and server errors up to eight times per request.
Retry waits start at two seconds and double up to sixty seconds; `Retry-After`
can extend the wait. Permanent HTTP errors such as 403 are not retried.
The builder prints retry waits, saved page counts, and the resume position.
Tune these with `--http-min-interval`, `--http-retries`, `--http-retry-delay`,
and `--http-timeout`. A persistent outage still stops the run safely; rerunning
with the same cache continues from the last committed page. Do not delete the
cache to retry. These controls do not change the requested trade filters.

All completed exports live under `data/market_<id>/` by default. Keep these local.
For a substantial experiment, collect more matches. Three matches are only the
minimum for a train/validation/test smoke pipeline, not a robust benchmark.

## Market prices, execution evidence, and payoff fields

The actor-by-actor layout is unchanged. The builder first creates one shared
price timeline per market from **all captured wallets**, including wallets that
will be excluded by the actor filter. It then looks up that timeline when
writing each actor's rows. Other actors' individual histories are not copied
into the training conversation. A completed trade capture can supply this
context without an additional price-history API request.

Every new actor row carries `market_context`. YES and NO are identified by their
outcome token IDs and tracked separately. For a row at query time `t`, each
outcome uses only observations whose timestamps are **strictly less than `t`**.
Trades at `t`, including the target trade, cannot affect that row's context.
There is no plus-or-minus time tolerance for model inputs.

If several executions of the same outcome share an earlier timestamp, their
price is the share-volume-weighted average for that timestamp. The builder
aggregates exactly, then rounds the published price to 12 decimal places with
round-half-even. It does not invent an ordering within a timestamp. The latest
earlier price for each outcome comes with its observation timestamp and age in
seconds. If no prior observation exists for an outcome, its value is `null`.
`market_price_history.jsonl` records the timestamp/outcome observations for audit.
The export manifest records `market_context_version: 1` and the captured source
scope; an exhausted API traversal does not prove that every historical trade
was captured.

Under the binary contract's $1 winning and $0 losing payout convention, a price
of $0.40 corresponds to a price-implied probability of 0.40 (40%). This is not a
measured true winning probability. The YES and NO observations may come from
different times and need not sum to one. Neither outcome is filled in from the
complement of the other, and the two values are not normalized. These are
historical execution-price estimates, not historical best bids/asks, order-book
depth, or a verified record of the probability displayed to the actor. Retain the
observation ages when inspecting or using the data.

Each row also carries `execution_info` and `payoff_analysis` for analysis.
On a `TRADE` row, payoff entries follow the order of the executions in `trades`.
On a `NO_TRADE` row, the observed execution time is `null` and payoff entries
are empty; the row describes only the observed open interval.

| Feature | What the export can establish |
|---|---|
| Observed execution | A captured trade execution and its recorded time; `observed_execution_time` remains a block-time proxy, not a verified order-entry clock. |
| Order submission time and type | Unknown from public executions alone. In particular, an execution does not identify whether the order was a resting limit order. |
| Submission-to-fill window | `--fill-window-seconds` records the requested nonnegative window (default 5). `filled_within_seconds_of_submission` remains `null` without submission evidence. |
| Entire order filled, or filled later | `order_fully_filled` and `full_fill_time` remain `null`. An execution can be a partial fill of an unseen larger order. |
| Canceled or unfilled orders | Not reconstructible from the captured executions. A `NO_TRADE` interval is not an unfilled order. |
| Expected or actual realized profit | `expected_profit` and `realized_profit` remain `null`; the export does not infer an independent winning probability or complete position cost basis. |

The payoff calculation is per observed execution, separate from model inputs.
For a BUY of `shares` at execution `price`, if those shares are held to normal
binary resolution:

- `winning_payout = shares`.
- `potential_profit_if_win_before_fees = shares * (1 - price)`.
- `pnl_if_lose_before_fees = -shares * price`.
- `cash_flow_before_fees = -shares * price`.

These are conditional payout calculations, not an expected-profit forecast or
proof the actor held until resolution. For a SELL, only positive sale proceeds
(`cash_flow_before_fees = shares * price`) are established. Profit fields remain
`null` because proceeds alone do not reveal acquisition cost or position
history. Fees remain unknown (`null`), rather than silently assumed to be zero.
Winning payout and profit-if-win values are explicitly **before fees** where
applicable.

The current trade's execution price, quantity, fulfillment metadata, and payoff
calculations are not inserted into the SFT user message. The target remains the
observed trade attributes. A compact form of the earlier `market_context` is
added to every user turn, with news and that actor's preceding conversation.
Each `yes` or `no` entry is either `null` or an object with only `price` and
`age_seconds`. The system instruction defines each price as the implied
probability under the $1/$0 payout convention, and the user message's
`query_time` supplies the reference time. This avoids repeating equivalent
probabilities, timestamps, and source descriptions in every turn, saving tokens.
The full raw context and audit timeline retain those fields for inspection.
Thus missing order lifecycle data does not become a fabricated immediate-fill
label, and future resolution does not leak into the decision input.

## 2. Convert actor files to conversations

From the repository root on RunPod, using the Python environment and model
directory from the successful training run:

```bash
python scripts/build_actor_dataset.py prepare \
  --input-root data \
  --out datasets/world_cup_sft \
  --tokenizer /workspace/models/Qwen3.6-27B
```

The converter discovers completed `actor_market_intervals_v1` exports directly
inside `data/`. It ignores other directories such as the collection cache. It
rejects two exports of the same market instead of double-counting observations.
If there are several versions of a market in `data/`, select inputs explicitly:

```bash
python scripts/build_actor_dataset.py prepare \
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

## 3. Check training and continue with one model load

If only two distinct matches have completed collection and you need a training
smoke test now, pass their completed export directories explicitly and opt in to
train/validation-only preparation:

```bash
python scripts/build_actor_dataset.py prepare \
  /workspace/world_cup_actor_data/data/market_1897035 \
  /workspace/world_cup_actor_data/data/market_1897059 \
  --train-validation-only \
  --out /workspace/datasets/world_cup_sft_two_matches \
  --tokenizer /workspace/models/Qwen3.6-27B
```

This places the earlier match in training and the later one in validation. It
uses only completed exports, preserves all their targets, and makes no network
requests. There is no held-out test set in this mode: `test.jsonl` is empty and
the manifest records `held_out_test_available: false`. Use this new output as
the trainer's `--dataset-dir`. The default three-way split still requires at
least three matches. Collect a separate test match before reporting test results.

Use the existing working training environment. The model weights and training
dependencies must already be present; these scripts do not install or download
them. Use GPUs in the same machine/Pod.

```bash
python scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir datasets/world_cup_sft \
  --smoke-then-full
```

This loads one model replica per GPU once. After ten optimizer steps (or the
last step of a shorter run), it evaluates up to 32 validation conversations.
A missing or nonfinite validation loss stops training. On success, training
continues with the same model, LoRA adapter, optimizer, and scheduler. The first
ten steps count toward the full run, whose learning-rate schedule is used from
the start. The normal evaluation cadence and final evaluation use the full
validation split; the early subset metrics are named `smoke_eval_*` and are
recorded separately in `training_metadata.json` as `smoke_check`.

This is an execution and finite-loss check, not an accuracy threshold or a
complete numerical validation. On resume, retain `--smoke-then-full`; if the
checkpoint is already beyond step ten, the subset check runs after the first
resumed optimizer step. Model loading is necessary again after a process exits.
Offline regression tests cover routing, data selection, state preservation,
failure handling, and CLI modes. Actual multi-GPU integration for this new mode
must be checked in the target training environment.

For a full run without the early subset check:

```bash
python scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir datasets/world_cup_sft
```

Use `--gpus 1 --gpu-ids 0` for one GPU, or `--gpus 4 --gpu-ids 0,1,2,3` for four.
Always specify the new `--dataset-dir`; the trainer's historical default points
to the older pilot export. The original `--smoke` remains a standalone ten-step
run that exits. Starting a separate full command afterward reloads the weights.

## Conversation and supervision contract

Each conversation starts with a system instruction and a user message containing
the actor ID, market question/outcomes, query time, first interval news, an empty
`past_observed_trades` list, and compact `market_context` for enriched exports.
Later user messages contain the next query time, new interval news, and that
time's earlier YES/NO prices and ages. Earlier assistant messages provide the
actor's previous trade history within this market. Legacy-only conversion omits
market context and records that limitation.

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
python scripts/build_actor_dataset.py prepare --input-root data \
  --split-file datasets/world_cup_sft/split_plan.json \
  --out datasets/world_cup_sft_rebuilt \
  --tokenizer /workspace/models/Qwen3.6-27B
```

Conversion checks actor/market identities, interval continuity, grouped execution
times, decimal validity, strict prior news timestamps, market-context timestamps
and structure, manifest counts, duplicate market inputs, and reserved chat
markers. A failed build does not publish a partial output directory. Existing
outputs are never overwritten.

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
