# Strictly prior actor metrics: definitions and input schemas

`scripts/derive_actor_metrics.py` adds 18 actor behavior and performance metrics
to the existing World Cup dataset. It is a separate Python 3.11+ script using
only the standard library, with no imports from other repository scripts or
modules. It reads local files, makes no API requests, and writes a new output
directory. It does not change the collector or source data.

The scope is one actor in one binary market, identified by `actor_id` and
`condition_id`. It does not pool actor history across markets, combine training
and validation performance, use collection-time actor snapshots, or infer missing
capital and cost basis. Time since the previous trade and historical markout are
deliberately excluded. `scripts/derive_global_actor_metrics.py` uses the same
formulas across a wallet's captured markets. See [the three-dataset workflow](actor_dataset_variants.md)
for the scope comparison, shared cohort, chronological split requirements, and
three independent training runs.

## Run against an existing export

From the repository root on macOS or Linux:

```bash
python3 scripts/derive_actor_metrics.py data/market_1897059 \
  --out data/market_1897059_metrics
```

Supply multiple export paths as positional arguments, or discover exports under
a directory with `--input-root`:

```bash
python3 scripts/derive_actor_metrics.py \
  --input-root data \
  --out data/actor_metrics_v1
```

Use a fresh `--out`. Each market's actor metrics are written beneath
`markets/<condition_id>/actors/<actor_id>.jsonl`, with one record per distinct
trade timestamp. Each row has an `actor_metrics` object containing `values`,
`unavailable_reasons`, `sample_counts`, `window`, and metric-scope metadata.
Numeric metric values are decimal strings, except the integer loss streak;
unavailable raw values are JSON `null`. Counts and reasons distinguish an unavailable
value from zero. The output manifest records source hashes and configuration.

With no extra financial input files, the four execution-based metrics are
available when sufficient earlier fills exist. The other metrics are `null`.
That is intentional: prices and sizes of public executions do not establish
complete realized P&L or portfolio returns.

## The 18 metrics

Each formula uses only eligible earlier observations in the selected window.
An execution means a captured fill; multiple fills can belong to the same order.
A completed position means one fully closed accounting unit supplied in the
completed-position ledger, not an individual partial fill.

| Metric | Definition | Required history |
|---|---|---|
| `average_execution_notional` | Mean of `shares × price` over earlier fills. | Captured executions |
| `execution_notional_cv` | Population standard deviation of fill notional divided by its mean. | Captured executions |
| `executions_per_day` | Earlier captured fill count divided by the recorded window length in days. | Captured executions |
| `buy_notional_share` | Earlier BUY notional divided by total BUY and SELL notional. | Captured executions |
| `net_realized_pnl` | Sum of net P&L for eligible completed positions. | Completed-position ledger |
| `mean_realized_roi` | Unweighted mean of each completed position's `net_pnl / entry_cost`. | Completed-position ledger |
| `win_rate` | Positive-P&L position count divided by all completed positions, including breakevens. | Completed-position ledger |
| `average_winning_profit` | Mean net P&L among positive-P&L completed positions. | Completed-position ledger |
| `average_losing_loss` | Mean absolute net P&L among negative-P&L completed positions. | Completed-position ledger |
| `payoff_ratio` | Average winning profit divided by average losing loss. | Completed-position ledger |
| `profit_factor` | Sum of positive net P&L divided by absolute sum of negative net P&L. | Completed-position ledger |
| `historical_expectancy` | Mean net P&L per completed position; this is a historical statistic, not a forecast. | Completed-position ledger |
| `consecutive_loss_streak` | Number of consecutive most recent losing completed positions. | Completed-position ledger |
| `average_holding_seconds` | Mean of `closed_at − opened_at` over completed positions. | Completed-position ledger |
| `sharpe_ratio` | Mean of `period_return − benchmark_return`, divided by its sample standard deviation. | Regular equity-return series |
| `sortino_ratio` | Mean of `period_return − target_return`, divided by the square root of the mean squared negative deviation from the target. | Regular equity-return series |
| `maximum_drawdown` | Largest peak-to-trough fractional decline in compounded equity growth, starting at 1. | Regular equity-return series |
| `return_volatility` | Sample standard deviation of `period_return`. | Regular equity-return series |

All financial amounts use one consistent currency supplied by the data producer.
Ratios are fractions, not percentages. Sharpe and Sortino are **not annualized**.
Sample standard deviation uses `ddof=1`; Sortino's downside mean includes every
eligible period, using zero for periods at or above the target. Risk ratios and
return volatility require at least 30 periods by default. Change this with
`--min-return-periods N`, where `N` must be at least 2. A smaller threshold does
not make a short history a reliable performance estimate. Maximum drawdown can
be calculated from one eligible return period; notional variability requires
at least two eligible executions.

No-loss denominators and zero return variability yield `null` ratios with a
reason, not infinity. The absence of a supplied ledger or return series also
yields `null`, rather than a claim of zero profit or zero risk. If the latest
closed positions have the same timestamp and mixed loss/non-loss outcomes, their
within-timestamp order is unknown; the loss streak is `null` when that ambiguity
prevents a unique result.

`net_realized_pnl` covers the supplied completed positions only. It does not
include realized gains from partial exits of positions still open. Mean ROI is
an equal-position average, not the actor's return on total invested capital.

Without `--lookback-seconds`, execution frequency measures the interval from the
earliest eligible captured fill to the query time. With that option it uses the
specified fixed window length. It is a captured activity rate, not proof that
every fill during that period was observed. The other execution summaries have
the same captured-history limitation.

## Completed-position input

Pass `--closed-positions PATH` to supply JSONL or gzip-compressed JSONL. Each
record represents one completed position under a consistent accounting policy.
This deliberately requires accounting input instead of assuming that the first
captured BUY started from an empty inventory. Public capture may miss opening
holdings, transfers, splits, merges, redemptions, fees, and earlier fills.

Required fields:

| Field | Meaning |
|---|---|
| `actor_id` | Wallet matching the raw export. |
| `condition_id` | Binary-market condition matching the raw export. |
| `position_id` | Unique identifier for the completed accounting position. |
| `opened_at` | Opening timestamp under the chosen accounting policy. |
| `closed_at` | Time the position was fully closed. |
| `known_at` | Earliest time all information used in this record was available. |
| `entry_cost` | Positive entry cost allocated to this completed position. |
| `net_pnl` | Realized profit/loss after all allocated fees. |

This hypothetical record illustrates the schema; it is not a real actor's P&L:

```json
{"actor_id":"0x0000cccf1d05a843fefa1913eccf62a57040348e","condition_id":"0x18f73aca12019d3fc2a03e7af28f6ebcec12634413819605d7cfa3db20073f26","position_id":"example-position-1","opened_at":"2026-06-14T16:00:00Z","closed_at":"2026-06-14T16:30:00Z","known_at":"2026-06-14T16:30:03Z","entry_cost":"10.00","net_pnl":"1.25"}
```

Allocate partial fills consistently into completed positions before supplying
the file. Treating every sale fill as a separate winning or losing position
would change the win rate, payoff ratio, expectancy, and holding-duration
definitions. The script validates schema and timing; it cannot independently
verify your economic accounting or establish when information was available.

## Equity-return input

Pass `--returns-file PATH` for JSONL or gzip-compressed JSONL containing one
regular return interval per record. The required capital base is equity,
including cash allocated to the market and the marked value of holdings, with
external cash flows adjusted out. Today's `actor_market_value` alone is not
this equity history. Nor is a sequence of execution prices a return series for
the actor's portfolio.

| Field | Meaning |
|---|---|
| `actor_id`, `condition_id` | The actor and binary market to which returns belong. |
| `period_start`, `period_end` | Start and end of a positive-length, regular interval. |
| `known_at` | Earliest time all information used for the return was available. |
| `period_return` | Capital-flow-adjusted fractional return, at least `-1`. |
| `benchmark_return` | Same-period benchmark return; defaults to zero. |
| `target_return` | Same-period target used by Sortino; defaults to zero. |
| `capital_flow_adjusted` | Must be the JSON boolean `true`. |

This is one hypothetical daily observation. Supply sufficient contiguous regular
observations for the selected minimum sample size:

```json
{"actor_id":"0x0000cccf1d05a843fefa1913eccf62a57040348e","condition_id":"0x18f73aca12019d3fc2a03e7af28f6ebcec12634413819605d7cfa3db20073f26","period_start":"2026-06-12T00:00:00Z","period_end":"2026-06-13T00:00:00Z","known_at":"2026-06-13T00:00:05Z","period_return":"0.012","benchmark_return":"0","target_return":"0","capital_flow_adjusted":true}
```

Do not annualize daily values before supplying them. Keep return intervals,
capital allocation, fee accounting, and benchmark conventions consistent within
each actor-market series. Use a common return-period duration across actors and
compared variants to keep risk statistics on the same timescale. Overlapping periods are rejected. Unequal durations or
gaps in the eligible prior window make risk metrics `null` with an explicit
reason; they are not silently treated as a regular series. A series cannot
continue after a `-1` return, since its capital has reached zero.

## Time boundaries and leakage prevention

For a query at timestamp `t`:

- A fill contributes only if its timestamp is strictly less than `t`.
- A completed position contributes only if both `closed_at < t` and
  `known_at < t`.
- A return contributes only if both `period_end < t` and `known_at < t`.
- Equality is excluded. Every fill at `t` receives the same pre-execution state;
  no fill at that second can update the features for another fill at that second.
- Missing history stays unavailable. There is no backward filling from later
  holdings, final outcomes, P&L, or returns.

All timestamps require an explicit UTC offset, such as `Z` or `+00:00`.
`known_at` is the information-availability time, not necessarily the time a
local preprocessing script ran. For retrospective accounting, use the time the
underlying observations actually became available; do not backdate later
settlement information to an earlier trade. This provenance is the input
producer's responsibility.

By default the script uses all available eligible prior history. With
`--lookback-seconds N`, it additionally requires fill timestamps, position close
times, and return **start** times to be at least `t − N`. Requiring a return's
start to be inside the window avoids including a partly overlapping interval.
Feature values are computed separately at each query: a Sharpe ratio updated
after a later period cannot change an earlier row's ratio.

The script's causal boundary is the recorded execution timestamp. Public trade
block times can follow actual order submission or matching. Strictly prior to
that timestamp does not prove prior to the trader's actual decision. Likewise,
this postprocessor does not fix source-cohort selection, incomplete capture,
unverified news publication times, or other limitations in the raw export. The
collector's default maximum of 20 captured executions is a retrospective actor
filter, not a prospective evaluation protocol.

## Add metrics to model inputs

First prepare SFT with the existing builder. Then supply the corresponding raw
exports and prepared SFT directory to the metrics script:

```bash
python3 scripts/derive_actor_metrics.py \
  --input-root /workspace/world_cup_actor_data/data \
  --closed-positions /workspace/actor_history/closed_positions.jsonl.gz \
  --returns-file /workspace/actor_history/returns.jsonl.gz \
  --sft-dir /workspace/datasets/world_cup_sft \
  --out /workspace/datasets/world_cup_actor_metrics

python3 scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir /workspace/datasets/world_cup_actor_metrics/sft \
  --smoke-then-full
```

Omit the two financial-input options if you do not have those histories yet;
their raw metrics will be `null`, and unavailable values will be omitted from
model inputs. On a Mac, use local paths instead of `/workspace`.
Without `--sft-dir`, the script produces metrics files for inspection only.

The SFT copy adds a compact `actor_metrics` object to each matching user context:
a `scope` label (`current_market` or `global_wallet`), the available selected
`values`, and supporting `sample_counts`. Unavailable values are omitted from
model messages; the shared system instruction defines absence as unavailable,
not zero. An initial row can have `values: {}` while retaining scope and counts.

Only `captured_executions`, `completed_positions`, and `eligible_return_periods`
can appear as model sample counts, and only for metric families selected by
`--features`. Profitable, losing, and breakeven position counts stay in raw
audits so a feature subset cannot accidentally restore excluded win-rate
information through its count fields. Raw outputs always retain all 18 metric
values, including nulls and reasons, and all diagnostic counts.

When an eligible regular return series supports a selected risk metric, `return_period_seconds` identifies its time basis.
By default all 18 are selected; `--features` accepts a comma-separated subset of
metric names. Derived numbers in model inputs use 10 significant digits by default, adjustable with
`--metric-significant-digits` from 4 to 16. Raw metric values keep full precision.
Original execution labels and market prices are unchanged. Full window boundaries,
unavailable reasons, and scope metadata remain in the separate actor-metrics
JSONL files; the manifest records the shared configuration, scope, and minimum return-period threshold.
This avoids repeating the full audit metadata in every model turn. The copy
preserves original split assignments, target trades, news, and conversation order.
It checks that a target corresponds to the raw export; it does not silently
join an unrelated dataset. Collection-time actor snapshots remain outside
model messages. The trainer tokenizes the enriched conversations and checks
their new lengths; overlong conversations fail instead of being silently
truncated. Use a new training run to train with these additional inputs.

The task remains prediction of observed trade attributes, conditional on an
execution. These metrics do not supply labels for whether an actor should trade,
and realized historical performance is not a guaranteed prediction of future
performance.

## Global wallet inputs

The global variant uses `scripts/derive_global_actor_metrics.py` with the same
raw actor exports and `--sft-dir` as the in-market variant. The exported actors
and query times identify who to summarize and when. Their history is gathered
across markets, including the queried market. Raw actor filtering for the current
market does not remove other markets from a retained actor's global history.

Without `--wallet-trades`, the script requests wallet executions from the public
`/v2/trades` API without a condition filter. It follows pagination using
resumable captures beneath `--cache` (default `data/market_actor_cache`). Saved
captures record their time bounds and request provenance. Reuse the same cache
after an interrupted request. A partial traversal, including one stopped by
`--wallet-max-pages`, fails instead of becoming a completed global history.
Use a new cache for a fresh capture; there is no implicit refresh of exhausted
captures. This fetch can be much larger than the current-market export.

For unattended collection, use the curl transport and read
[collection reliability and estimates](collection_reliability.md). The Global
collector uses four wallet workers by default, a shared request limiter, gzip
for new capture pages and cached per-wallet metric results. It preserves old
plain JSON capture pages. The default two-hour collection budget stops and
retains progress; it does not claim the dataset is complete. Inspect
`wallet_cache/global_progress/` and run the offline estimator before increasing
time or storage allowances. These controls do not shorten historical windows.

Wallet executions accept both 31-byte and 32-byte hexadecimal condition IDs,
matching the [official Polymarket SDK's condition ID schema](https://github.com/Polymarket/ts-sdk/blob/main/packages/bindings/src/shared.ts).
This retains executions such as combination-market trades in wallet-wide
execution metrics. IDs are lowercased without padding; transaction hashes still
require 32 bytes. This does not reconstruct combo settlements or whole-wallet
PnL. Existing captures remain reusable after this parser update.

Alternatively, pass `--wallet-trades PATH` with normalized JSONL or gzip JSONL:

| Field | Meaning |
|---|---|
| `actor_id` | Wallet being summarized. |
| `condition_id` | Market or combination condition where this execution occurred; a 31-byte or 32-byte hexadecimal ID. |
| `execution_id` | Unique identifier for this captured execution within the actor's supplied history. |
| `timestamp` | Explicitly zoned execution time. |
| `known_at` | Explicitly zoned time the observation became available, at or after execution. |
| `side` | `BUY` or `SELL`. |
| `shares`, `price` | Positive fill size and execution price in `[0, 1]`. |
| `outcome` | Optional outcome label. |

A hypothetical normalized record:

```json
{"actor_id":"0x0000cccf1d05a843fefa1913eccf62a57040348e","condition_id":"0x18f73aca12019d3fc2a03e7af28f6ebcec12634413819605d7cfa3db20073f26","execution_id":"example-fill-1","timestamp":"2026-06-14T16:00:00Z","known_at":"2026-06-14T16:00:03Z","side":"BUY","shares":"10","price":"0.15","outcome":"Yes"}
```

Both execution time and `known_at` must precede the query. In automatic API
capture, recorded execution time is an information-availability proxy, not
verified feed-publication time. The script records that limitation. All-market
execution capture cannot itself certify complete wallet accounting.

Global `--closed-positions` uses the completed-position schema above, retaining
condition IDs to distinguish accounting positions from different markets. The
script pools eligible completed positions by actor, regardless of condition.
Its realized metrics still cover only supplied completed positions, not realized
partial exits from positions still open.

Global `--returns-file` requires a **whole-wallet** equity series. Use the return
schema above with `actor_id` and `scope: "wallet"`, and omit `condition_id` and
`market_id`. All other timing, capital-flow, duration, and minimum-observation
requirements still apply. Market-scoped returns are rejected rather than summed
or averaged into a wallet Sharpe ratio.

```json
{"actor_id":"0x0000cccf1d05a843fefa1913eccf62a57040348e","scope":"wallet","period_start":"2026-06-12T00:00:00Z","period_end":"2026-06-13T00:00:00Z","known_at":"2026-06-13T00:00:05Z","period_return":"0.012","benchmark_return":"0","target_return":"0","capital_flow_adjusted":true}
```

A whole-wallet capital base includes allocated cash and marked holdings across
markets, with external deposits and withdrawals adjusted out. Its return is not
the sum of separate markets' percentage returns. Supply this financial history
only when it can be reconstructed under consistent accounting and availability
rules; otherwise the risk metrics remain unavailable.
