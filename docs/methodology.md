# Methodology and validation gates

## Research target

The corpus should cover every official fixture in the 2026 World Cup, with
explicit coverage statuses for unavailable or ambiguous Polymarket contracts.
The prediction unit is a public wallet and an observed executed action. A
wallet does not establish a person's identity. A fill does not establish order
placement time, attention to a news item, or a private belief.

Two distinct tasks are valid:

1. **Conditional execution details:** Given that an execution occurs, predict
   its token, direction, shares, and optionally realized execution price using
   information available before the execution.
2. **Execution occurrence:** At a predeclared wallet/fixture/time checkpoint,
   predict whether an eligible execution occurs in a fixed future horizon. If
   positive, predict the first execution bundle and its details.

Trade-only examples identify the first task. They cannot estimate whether or
when a wallet acts. Empty observed windows in the second task mean
`NO_OBSERVED_TRADE`, with no claim of deliberate abstention.

## Present implementation boundary

`poly_world_cup.temporal` implements conservative, independently testable
guards. It does not reconstruct historical news, prove fill-source coverage,
parse exchange economics, infer private beliefs, or export training examples.
Its dictionaries are a normalized interface that ingestion adapters must
populate from evidence. A passed guard is conditional on those source facts
being correct.

Before SFT export, establish fixture mappings, raw-fill reconciliation,
wallet-side roles, execution grouping, position completeness, historical
content versions, eligible cohorts, and observation coverage. Preserve a
separate untouched raw archive and source manifests for regeneration.

## Time and observability

Every timestamp must have an explicit timezone. The code normalizes it to UTC.
Event occurrence, public availability, source capture, and ingestion are
different timestamps. A score at match minute 68 does not establish when a
trader could receive the corresponding report. A block timestamp does not by
itself establish the earlier off-chain matching time or public observation
time.

An eligible feature record has:

| Field | Meaning |
| --- | --- |
| `event_time_utc` | When an event occurred, if applicable |
| `availability_verified` | Explicit assertion backed by source evidence |
| `availability_lower_utc` | Optional lower bound on public availability |
| `availability_upper_utc` | Required finite upper bound on public availability |
| `atomic_id` | Canonical transaction/bundle identifier when applicable |

For a query time q, include only records with verified availability and upper
bound strictly before q. Unknown upper bounds and equal-cutoff records fail
eligibility. Preserve original precision and evidence in the source table.
Do not derive availability from ingestion time alone or silently equate it
with occurrence time.

If the target execution's true time is itself uncertain, use a conservative
query boundary before its earliest plausible occurrence. Do not put
same-block activity into its immediate history based only on log order.
Pass the entire target transaction/bundle in `excluded_atomic_ids`, so another
fill leg cannot reveal the target through history or last-price features.

## Historical versions and public background

Version news, market rules, fixture identities, and schedule metadata. The
complete final registry is useful for auditing coverage but must not become
unversioned historical model input. Knockout participants may have been unknown
when a contract opened. The actual kickoff time, final closure state, final
volume, later rule revisions, and eventual resolved outcome cannot become
earlier features.

`select_versions` requires `item_id` and an established integer `version_rank`.
It chooses the greatest eligible rank, not the last file retrieved. A current
article bearing an old publication date does not establish that its present
text existed then. Conflicting eligible versions with the same identity/rank
raise an error.

Start with source-linked factual background. Generated summaries must retain
claim-level evidence and source versions. An LLM can insert memorized results
even when given only earlier sources, so grounding and source validation are
prerequisites. A wallet-specific belief summary is an optional inference from
its prior history, evaluated against use of the underlying history itself.

## Wallet universe and no-trade windows

Choose the checkpoint grid and eligibility rule before examining future
participation. An example rule is at least one observed execution during the
preceding 30 days. The source universe must be either:

- `global_prefix`: wallet activity from the global historical ledger.
- `world_cup_prefix`: wallets already observed trading World Cup markets.

The latter excludes first participation until the wallet qualifies. Report
its coverage and state that this is a different population. Fetching histories
only for eventual tournament traders and then filtering their past activity
still conditions on future participation.

`build_historical_universe` requires an explicit `source_scope`, checks both
execution and availability times, and binds membership to a query timestamp.
`occurrence_label` rejects a wallet outside that universe or a universe from
another query. The scope is an upstream assertion: software cannot detect
future selection hidden inside a falsely declared source ledger. Save the
source manifest and population-selection procedure for audit.

Labels use the half-open interval `[q, q + horizon)`. An execution at q is a
label, while one at the right boundary is outside the window. Features remain
strictly earlier than q.

`CoverageInterval` records verified complete observation for a wallet,
fixture, and target role. `*` explicitly means every wallet or fixture. Its
`complete` default is false and must be explicitly set true with evidence.
Intervals can be joined only when there is no gap. A missing API page, absent
chain range, truncated history, or unknown role cannot justify a negative.

The helper labels incomplete horizons `CENSORED`, including horizons with a
known positive whose earlier executions might be missing. This preserves the
meaning of *first* execution. An unknown-role execution censors a role-specific
target rather than silently becoming a taker or maker negative.

If negatives are sampled, retain inclusion probabilities and use sampling
weights or an appropriate probability correction. Validate/calibrate on a
representative population or with correct weights. Fit selection rules and
bucket boundaries only on the designated training data.

## Execution identity and positions

Retain raw event identity separately from wallet-side action identity. One
raw event can legitimately produce two wallet-side records. Deduplicating all
wallet actions by raw log ID alone would lose one side. Normalize token IDs,
BUY/SELL direction, decimal amounts, and maker/taker roles from actual source
semantics before these guards are used.

Group verified atomic execution bundles, preserving every component and raw
reference. Do not group all later partial fills of an order into an earlier
target. Bundles need not correspond to a single human decision. Earliest
bundles sharing a timestamp remain explicitly ambiguous unless additional
execution-order evidence exists. The current helper returns all tied earliest
action IDs and does not invent an order.

Positions require a known starting balance and all relevant position-changing
events, including transfers and token operations. Unknown starting balances
remain unknown. Current balances, future P&L, future lifetime counts, and
subsequent market responses cannot be historical features.

Activity thresholds may define optional cohorts. They do not verify that a
wallet is human. Recompute any cohort statistics from the prefix and report
all-wallet results plus sensitivity to filters.

## Train, validation, and test separation

Whole-fixture chronological splitting alone is insufficient because markets
for several fixtures trade concurrently. Require both disjoint fixtures and
global time boundaries:

| Split | Query and label constraints |
| --- | --- |
| Train | Query before T1 and exclusive label end at or before T1 |
| Validation | Query at or after T1, before T2, and label end at or before T2 |
| Test | Query at or after T2 |

Purge horizons crossing boundaries. `validate_splits` checks these constraints,
fixture overlap, duplicate example IDs, and positive label-interval length.
Point targets require explicit timing precision before this interval-based
check applies. The function validates supplied assignments; it does not
choose cutoffs or silently move data between splits.

Preserve prior wallet activity as context during evaluation when it was
already observed at the query, without fitting model weights or calibration
on evaluation labels. Report seen-wallet and unseen-wallet subsets. Audit
base-model knowledge of completed matches and run a future-event evaluation
before making a prospective forecasting claim.

## SFT and evaluation after data gates pass

Use actual observed labels. For occurrence, supervise a constrained
`TRADE`/`NO_OBSERVED_TRADE` outcome with conditional action fields. An observed
binary label does not provide a ground-truth numerical trade probability.
Derive probabilities from constrained model scores or a classifier head, then
calibrate using validation data. Omit action-detail losses on negative rows.

Compare recent-activity, market-only, wallet-history, and structured models
before SFT. Report occurrence log loss and calibration, conditional action
accuracy/log loss, size error, and missingness. Run no-news and no-history
ablations, including comparison of inferred belief summaries to raw history.
Report performance variation across fixtures and repeated wallets; millions
of correlated rows do not supply millions of independent sporting contexts.

Interpretability interventions can test what the trained model uses. They
cannot establish the historical trader's private reasoning or that a
temporally preceding news item caused the trade.

## Executable checks

Run `python -m unittest discover -s tests -p test_temporal.py -v`.
Synthetic invariant tests cover late publication, unknown availability,
strict timestamp equality, atomic target exclusion, future-extension
invariance, source version order, knockout placeholders, historical universe
membership, coverage gaps, role uncertainty, half-open horizons, duplicate
actions, same-time ambiguity, and simultaneous time/fixture split constraints.
These tests establish behavior of the helpers, not real-world source coverage.
