# Study plan: from context to observed execution

## Verdict

The proposed research is feasible as **observational behavioral prediction**.
The initial tuple is useful when written `(B_before_query, A_after_query)`.
Neither successful prediction nor mechanistic interpretation of the model
identifies the trader's private beliefs or the causal effect of news.

There is currently no empirical evidence that SFT will outperform simpler
models on this corpus. The first implementation validates collection mechanics
and temporal invariants; it does not validate model performance.

## Freeze these definitions before training

| Choice | Proposed first experiment |
| --- | --- |
| Corpus | All 104 fixtures; supported primary match-result contracts; explicit missingness |
| Actor | Public wallet identifier, with unknown links between wallets/people |
| Observation | Canonically reconciled wallet-side execution bundle; components retained |
| Time target | Publicly recorded execution, not an unobserved mental decision |
| Occurrence grid | Every 5 minutes, using only market-open and participant information known then |
| Horizon | `[query, query + 5 minutes)`; purge incomplete, out-of-scope, or boundary-crossing horizons |
| Initial actor population | Prior 30-day global activity; if unavailable, explicitly use already-observed World Cup wallets and report first-participation exclusions |
| Context | Verified prior wallet history, available inventory, market state, sourced public facts; missing values explicit |
| Belief representation | Start with source-grounded facts; test inferred summaries as an ablation |
| Initial cohorts | All eligible wallets; optional activity cohorts computed from historical prefixes |
| Output | Observed occurrence label, plus conditional token/direction/size fields |

The grid and horizon are proposed defaults for a first experiment, not proven
optimal constants. A separate trade-conditional experiment predicts action
details given that an execution occurs. Report these tasks separately.

With only block timestamps, a future settled fill may reflect an order placed
before the checkpoint. The claim is prediction of subsequent observed
execution flow. A decision-time claim requires verified order/matching data
and a defensible observability/latency model. Do not invent a fixed latency.

## First executable milestones

1. **Discovery — implemented.** Run `discover`; require all fixture rows,
   reviewed mappings, and explicit rules. Keep known schedule discrepancies.
2. **Observed-fill collection — implemented, smoke validated.** Run full API
   traversal separately. Compare it with canonical execution data and any
   existing `trades.parquet`; verify schema and provenance before reuse.
   A transformed quantity file must not be mistaken for original raw trades.
3. **Canonical executions and history — next.** Identify exchange versions,
   reconstruct maker/taker wallet sides, group only verified atomic bundles,
   and reconcile balances and lifecycle events. Record unreconciled intervals.
4. **Context archive — next.** Capture/source historical news versions,
   announcement times, schedules, contract rules, and market-state snapshots.
   Use a verified pre-opening public background where available. A separate
   wallet history may condition its inferred prior; uncertainty stays explicit.
5. **Example builder — gated.** Bind each checkpoint to its historical wallet
   universe, context evidence, and target coverage. Apply `temporal.py` guards
   and test future-extension invariance end to end. Export `B -> A`, preserving
   censoring and retaining every source reference outside the prompt too.

If required historical data cannot be recovered, narrow and disclose the
estimand (for example, executions above a verified threshold) or collect
prospectively. Unknown context is not recovered by asking an LLM to guess it.

## Split and model protocol

1. Choose global cutoffs T1/T2 and disjoint fixture partitions before tuning.
   All training labels must finish by T1; validation queries start at T1 and
   labels finish by T2; test queries start at T2. Drop incompatible portions of
   markets whose trading periods overlap these boundaries. Report exclusions
   per fixture. Curating all matches does not mean putting all of them in train.
2. Fit size buckets, feature transforms, cohort thresholds, and any selection
   policy on training data only. Keep inclusion probabilities when downsampling
   negatives. Use a representative or correctly weighted validation/test set.
3. Establish global-rate, recent-wallet-activity, market-only, history-only,
   and structured-model baselines. Random train/test row splitting is invalid.
4. Run an open-weight LLM baseline with fixed prompts and then SFT the same
   backbone. Pick its size/adapter method after measuring corpus size, context
   lengths, hardware, and baseline learning curves. Save exact model/tokenizer
   revisions, seeds, transforms, training manifest, and hyperparameters.
5. Use constrained structured outputs. Train on observed discrete occurrence
   labels, not invented probability strings; obtain scores from constrained
   logits/a classifier head and calibrate on validation. Apply detail losses
   only on positives. Predict size with train-fitted buckets or a separate
   numeric head; evaluate decoded values too. Treat price as an optional
   realized-outcome target, not an intended limit price.
6. Compare market-only, +history, +facts/news, and +inferred-summary variants
   on identical eligible rows. Also evaluate seen and unseen wallets and
   pregame versus live periods. Publish missingness and selection rates.
7. Report weighted occurrence log loss, Brier score, calibration, and PR-AUC;
   token/direction accuracy and log loss conditional on positive action;
   size error and optional price error. Use fixture-clustered uncertainty and
   assess repeated-wallet dependence. Only 104 fixture contexts are available.

For 2026 retrospective evaluation, a pretrained LLM may already know match
outcomes. Historical retrieval does not erase that knowledge. Prefer a
documented earlier training cutoff where feasible, audit factual memorization,
and require a prospective future-event holdout for strong forecasting claims.

## Mechanistic interpretation after predictive validation

Use a model whose weights and activations can be inspected. First establish
simple input ablations and sensitivity tests. Then compare matched held-out
examples where only a supported context feature changes, examine candidate
representations, and test activation interventions on their predictions.

Keep interventions plausible, use controls, and measure effects on held-out
outputs. A probe or a plausible generated explanation is not enough to infer a
mechanism. Even a validated internal mechanism describes the **model's** use
of information; it does not establish the wallet owner's cognitive process or
causal response to a news item.
