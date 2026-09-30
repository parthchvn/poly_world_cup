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
The core pipeline uses this builder and `train_world_cup_multigpu.py`.
The optional `derive_actor_metrics.py` and `derive_global_actor_metrics.py`
postprocessors add earlier actor history within the current market or across
markets. For a controlled comparison of all three datasets, follow
[the three-model workflow](actor_dataset_variants.md).

To rebuild existing RunPod exports with market-price context and actor snapshots,
use a new actor output root and the **same existing cache**:

```bash
cd /poly_world_cup
git pull --ff-only origin main

python3 scripts/build_actor_dataset.py sft 1897035 1897038 1897059 \
  --data-root /workspace/world_cup_actor_data/data_with_actor_snapshots \
  --cache /workspace/world_cup_actor_data/data/market_actor_cache \
  --reuse-existing \
  --http-transport curl \
  --out /workspace/datasets/world_cup_sft_with_actor_snapshots \
  --tokenizer /workspace/models/Qwen3.6-27B
```

This collects the selected markets, then validates their exports and writes
fixture-disjoint SFT splits. `--out` names the new SFT dataset; actor exports go
under `--data-root/market_<ID>`. The cache defaults to
`--data-root/market_actor_cache`, or use `--cache` explicitly. The requested split
is checked before trade collection: normally at least three distinct matches,
or two with `--train-validation-only`.

`--reuse-existing` explicitly reuses completed matching exports with official
price history (`market_context_version: 2`) and actor snapshots
(`actor_snapshots.version: 1`). It rejects older exports missing either feature.
Collection filters and source options apply only to new exports. Without this
flag, existing actor exports cause an error. If collection fails partway through,
rerun with the same cache and `--reuse-existing`: finished exports are retained,
and unfinished trade captures resume. An exhausted capture is reused without
scraping its trade pages again. Official token price history is a separate
cached source; an old trade capture alone does not contain it. Actor snapshots
also require their own API requests if not cached. An existing SFT output is never
overwritten; choose another `--out` when preparing a new version.

The separate `prepare` mode remains compatible with version-0 exports (no price
context) and version-1 exports (previous-execution price estimates), with a
warning describing the older context. All selected exports must use the same
version. It also accepts legacy exports without actor snapshots. Preparation
does not fetch prices or snapshots, or upgrade old actor files; rebuild them
from saved trade captures using the command above. Already trained adapters are
unchanged. Train a new run with
`--dataset-dir /workspace/datasets/world_cup_sft_with_actor_snapshots` to learn from
added historical price inputs. Collection-time actor snapshots are for analysis
and do not become model inputs.

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
cache to retry. These controls do not change the requested trade filters. Actor
snapshot collection uses up to eight workers, each pausing at least one second between
live requests; see the snapshot section below.

All completed exports live under `data/market_<id>/` by default. Keep these local.
For a substantial experiment, collect more matches. Three matches are only the
minimum for a train/validation/test smoke pipeline, not a robust benchmark.

## Market prices, execution evidence, and payoff fields

The actor-by-actor layout is unchanged. The builder fetches Polymarket's official
CLOB `/prices-history` series separately for the YES and NO token IDs, requesting
one-minute fidelity. HTTP responses are cached. The shared timeline is loaded
once per market and reused for every actor; the builder does not request prices
separately for each wallet or trade. Other actors' individual histories are not
copied into the training conversation.

Every new actor row carries `market_context`. For a row at query time `t`, each
outcome uses only history points whose timestamps are **strictly less than `t`**.
The same-time point is excluded because ordering relative to the execution is
not established. There is no nearest-point matching, interpolation, or
plus-or-minus time tolerance. The builder does not fall back to execution VWAP
or derive one outcome from the complement of the other.

A price is eligible only when its age is at most
`--market-price-max-age-seconds` (default 300 seconds). Missing and stale prices
remain `null`; `missing_reasons` distinguishes `no_earlier_observation` from
`stale`. Exported coverage records make these gaps visible. The age
limit is a freshness policy, not a claim that a five-minute-old price was still
available when the actor traded. A one-minute fidelity request does not
establish second-by-second historical coverage or guarantee a point in every
minute. API/network failures are errors, not successful empty histories. An
outcome with no returned history stops the export. The export also stops if
either outcome has zero usable prices across the retained trade rows, even if
some history was returned. Partial coverage prints a warning and preserves
missing values rather than silently presenting complete market context.

`market_price_history.jsonl` records the shared observations for audit, and
`market_price_sources.json` records request/cache provenance. Each available
raw snapshot has `price`, equal `implied_probability`, `observed_at`,
`age_seconds`, `source: "polymarket_clob_prices_history"`, `token_id`, and
`requested_fidelity_minutes: 1`. The context also records `max_age_seconds`
and per-outcome `missing_reasons`.

The export manifest records `market_context_version: 2`,
`market_price_max_age_seconds`, and `market_price_coverage`. Coverage uses flat
counts: `trade_rows`, `both_outcomes_available`, `yes_available`, `no_available`,
`yes_missing`, `no_missing`, and per-outcome reasons such as `yes_stale` and
`yes_no_earlier_observation` (with the corresponding `no_` fields). The manifest's
`price_context_complete_for_exported_rows` is true only when both outcomes have
usable context in every retained trade row. These counts refer to grouped
execution-time rows, not every individual fill or duplicated interval row.
The requested fidelity and actual observation times must be retained when
interpreting the data. An exhausted trade traversal does not prove complete
on-chain history, and a returned price series does not prove complete historical
quote coverage.

Under the binary contract's $1 winning and $0 losing payout convention, a price
of $0.40 corresponds to a price-implied probability of 0.40 (40%). This is not a
measured true winning probability. YES and NO observations may come from
different times and need not sum to one; the two values are not normalized.
These historical prices are not archived best bids/asks, order-book depth, or a
verified record of the exact probability displayed to the actor. Current wallet
positions, present-day order books and eventual resolution are not inserted as
historical pre-trade features.

The strict boundary is relative to the captured **execution block-time proxy**.
Order submission and matching may have happened earlier. A series point before
the block timestamp can therefore still be after the actual decision, or reflect
an execution before it was mined. This collector cannot certify a causally prior
order-entry quote for arbitrary historical wallets. It records the available
price series and timing limits rather than claiming exact information exposure.

For offline collection, supply `--price-history-file prices.json` with both
outcome tokens in this format (replace the placeholders with the market's real
condition and token IDs):

```json
{
  "format": "polymarket_clob_price_history_v1",
  "condition_id": "0xCONDITION_ID",
  "fidelity_minutes": 1,
  "histories": [
    {"token_id": "YES_TOKEN_ID", "history": [{"t": 1781456764, "p": "0.043"}]},
    {"token_id": "NO_TOKEN_ID", "history": [{"t": 1781456765, "p": "0.957"}]}
  ]
}
```

`t` is integer Unix seconds and `p` is a decimal price in [0, 1]. Both histories
must belong to the selected market. A supplied file is an explicit caller
snapshot, not proof of original API provenance or complete coverage. Supply it
when collecting a single market; match-specific source files cannot be shared
across a multi-market `sft` invocation.

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
The SFT prompts also omit wallet/market identifiers, ESPN identifiers, and source
URLs. Those remain in outer records or audit files for joins and verification.
News text and times, market meaning, prior actions, and price freshness remain
readable model inputs.
Thus missing order lifecycle data does not become a fabricated immediate-fill
label, and later payoff analysis is not inserted into the decision input. The
block-time limitation above still applies to interpreting the price context.

## Actor value and position snapshots

New exports collect these three public API results for every retained actor,
filtered by the actor's wallet and this market's condition ID:

| Field | API result |
|---|---|
| `actor_market_value` | `/v2/value`: the API's current marked-to-market holdings value for this actor and condition. This is fetched directly, not derived by summing positions or treated as profit. |
| `actor_positions_open` | `/v2/positions` with `status=OPEN` and `include_archived=true`: all returned position pages, including their current size/value, cost, fees, and P&L fields. OPEN can include resolved but unredeemed winning positions. |
| `actor_positions_closed` | `/v2/positions` with `status=CLOSED`: all returned closed-position pages, preserving the API's cost, fee, and P&L fields. |

Each actor's results are stored once in `actor_snapshots/<wallet>.json`. Every
raw actor row and its `actor_index.jsonl` entry carries `actor_snapshot_ref`, a
path relative to the export root. The snapshot has `version: 1`, actor/market/
condition identifiers, `temporal_scope: "collection_time_not_trade_time"`, and
`historical_model_input: false`. Each of the three result objects contains
`status: "ok"`, the API's `data` (an object for value, a list for positions), and
`pages` with request URLs, retrieval times, body hashes, and cache-use flags.
Successful `value: 0` and empty position lists are valid results. An HTTP,
validation, or pagination error aborts the export; it does not become zero or
an empty successful result. Saved HTTP responses remain available for retry.

These are **collection-time snapshots, not historical state at each trade**.
They can contain later sales, redemptions, and P&L. Position fields reflect the
API's returned scope; they do not establish every order or a complete transaction
history. Requests and pages may have different retrieval times, so the three
results are not an atomic account snapshot. They do not supply a historical
`as_of` view, and position event-time filters would not create one. Position P&L
is not attributed to an individual historical fill, and it does not replace the
unknown realized-profit or fee fields in that fill's `payoff_analysis`.
Requests record a `0.000001`-token filter floor. The API still excludes inactive
markets, even with `include_archived=true`; CLOSED requests omit that flag.

Snapshots require at least three requests per retained actor, plus pagination
(for example, at least 13,731 requests for 4,577 actors). Collection uses eight
bounded workers by default. `--actor-snapshot-workers` accepts 1 through 8; each
worker uses at least a one-second live-request pause, or the larger
`--http-min-interval`. Cached responses avoid repeat requests and retain their
original retrieval timestamps. Reusing a cache therefore does not refresh
holdings or P&L; use a new cache and output directory for a fresh capture.
Interrupting collection can wait for an in-flight request or retry delay to finish;
queued work is canceled and completed HTTP captures remain cached.

For offline single-market collection, supply `--actor-snapshots-dir PATH`,
containing one `<wallet>.json` file per retained actor in the same validated
snapshot format. This is supplied evidence, not independent proof of the
original API responses. Collect markets separately before preparation when
using this option.

The preparer validates and copies these files into
`audit/actor_snapshots/<condition_id>/<wallet>.json` in the SFT output. It records
references and hashes in `source_audit.jsonl`. Snapshot data is never inserted
into user/assistant messages or training targets, avoiding later financial state
leaking into earlier decisions. The trainer continues to use the conversation
files without needing changes.

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
| `source_audit.jsonl` | Per-conversation actor-file hashes, original target row indices, and actor-snapshot references/hashes when present |
| `audit/actor_snapshots/<condition_id>/<wallet>.json` | Collection-time actor value and positions copied from enriched exports; excluded from training messages |

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

### Live loss logging and plotting

The trainer writes `metrics.jsonl` (all logged metrics) and `losses.csv` (loss
records) inside `--out`. The default is one training-loss record per optimizer
step, after gradient accumulation; use `--logging-steps 10` to average/log over
longer windows. The first and last steps are logged, and `--smoke-then-full`
also logs each step until its initial validation check. These are Trainer's
reported, distributed training losses, not individual microbatch losses.
Validation loss is recorded at evaluations, not at every training step.

Rows include UTC time, optimizer step, epoch, and available loss, learning rate,
and gradient norm. Only the main GPU process writes, with each record flushed
and synced to disk immediately, independently of checkpoint saves. Blank CSV
cells mean that metric was not measured for that row. Nonfinite logged losses
are retained instead of being replaced by a previous finite average.

```bash
# Use the output directory printed by the trainer, or your explicit --out.
python3 scripts/plot_training_losses.py --run-dir /workspace/runs/YOUR_RUN

# Refresh the PNG during training; Ctrl-C stops only the plot watcher.
python3 scripts/plot_training_losses.py --run-dir /workspace/runs/YOUR_RUN --watch 10
```

The default output is `YOUR_RUN/loss_curve.png`. Use `--output figure.pdf` or
`--output figure.svg` for other formats. If needed, install `matplotlib` in the
environment where you plot with `python3 -m pip install matplotlib`; the trainer
does not need it. Plotting runs on CPU without loading model weights or CUDA.
The chart distinguishes training loss, full-validation loss, and the early
smoke subset. The final cumulative `train_loss` summary is not a curve point.

**Existing models do not need retraining.** Earlier trainer versions already
wrote `metrics.jsonl` in the output directory (usually step 1 and then every 10
steps for full runs). The plotting command reads those files directly, even if
training is still running. It cannot reconstruct unlogged intermediate losses.
Copy the log to another machine if you prefer to plot there. If only
`losses.csv` is present, the plotter uses it instead.

For runs made with this version, a `train_begin` event records the restored
step on resume. Raw completed records are retained, while the plot discards the
abandoned tail after the restored checkpoint and uses the latest value for each
metric/step. A partially written last line is ignored during plotting and
removed before resumed logging. Earlier logs lack these explicit resume
markers; repeated metric/step pairs use their latest value. Logs and plots are
local run outputs: keep them on your persistent RunPod volume if you need them
after deleting a pod. Use the original training checkout/settings to resume an
older run: the existing trainer and launchers enforce code/data signatures;
plotting older logs does not require resuming or changing them.

## Conversation and supervision contract

Each conversation starts with a system instruction and a user message containing
the fixture name, market question/outcomes, query time, first interval news, and
compact `market_context` for enriched exports. Opaque actor/market/ESPN IDs and
empty history placeholders are omitted from prompts; identifiers stay in the
outer conversation record and audit files.
Later user messages contain the next query time, new interval news, and that
time's earlier YES/NO prices and ages. Earlier assistant messages provide the
actor's previous trade history within this market. Legacy-only conversion omits
market context and records that limitation. The shared system instruction also
defines absent derived metrics as unavailable, not zero, and risk ratios as
not annualized. Both derived variants use that same instruction so only their
feature inputs differ. Unavailable derived metric values and detailed audit
counts are not repeated in model messages.

Each assistant answer is JSON:

```json
{"action":"TRADE","trades":[{"side":"BUY","outcome":"Yes","shares":"15.18","price":"0.31"}]}
```

This is an illustrative label, not a retrieved execution. Simultaneous executions
remain separate entries in the same `trades` array. Values retain their source
decimal strings. The target's execution values are not inserted into its user
message. Causal training attention prevents later answers from being used to
predict earlier ones.

The default converter preserves every interval answer as `{"action":"NO_TRADE"}`,
followed by its endpoint execution answer. The interval prompt carries its open
boundaries, news and earlier market prices; the following timestamp prompt reuses
that context without repeating the news. Equal query times are allowed only for
this validated interval/execution pair. Counts for both label classes are printed
and stored. `--trade-only` explicitly reproduces the old converter that omitted
interval answers. See [migration and limitations](no_trade_supervision.md).

These labels reconstruct event-bounded historical gaps. Their alternating order
and query scope reveal the action class, so this is not a prospective trade-timing
benchmark. It does not establish conscious abstention, P&L, match resolution, or
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
not time-verified in prompts and the manifest. Full rules text stays in
`market.json`; it is not inserted into the prompt because this collector does
not establish which version of those rules was visible at the decision time. ESPN timestamps remain occurrence
proxies rather than verified publication times. This conversion preserves those
limitations; it does not certify historical information availability or causal
news attribution. The new prompts and splits need fresh evaluation rather than
assuming the earlier pilot's 72.5% validation category accuracy carries over.

Developer verification uses synthetic exports produced with the current
builder's row generator and the real trainer's input/masking functions. Actual
Qwen tokenizer length validation and H100 execution must run in the target
training environment.
