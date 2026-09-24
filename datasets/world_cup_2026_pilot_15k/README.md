# World Cup 2026 conditional-execution pilot

This pilot contains 15,000 supervised decision targets from 20 matches and all
60 associated binary contracts. No model has been trained on this export.

| Split | Assistant target turns | Conversation JSONL lines | Captured executions |
|---|---:|---:|---:|
| train | 12,000 | 6,648 | 12,726 |
| validation | 1,500 | 819 | 1,578 |
| test | 1,500 | 781 | 1,597 |

A decision target is one actor-market execution timestamp. Executions sharing
that timestamp form one assistant answer. Each JSONL line contains one complete
actor-market conversation, so its earlier turns are available through attention.
Only the `messages` field is model input. The outer fields are audit metadata.
Training files are `train.jsonl.gz`, `validation.jsonl.gz`, and `test.jsonl.gz`.
They are gzip-compressed JSONL, not archives requiring a special dataset format.

The first 12 fixtures chronologically are training matches, the next four are
validation matches, and the next four are test matches. All three binary
contracts from each match remain in the same partition. Targets occur between
scheduled kickoff and ESPN's end-of-regular-time event, with a three-hour cap.
Actual target time ranges are strictly separated between splits, as recorded
in `manifest.json`. This is a held-out-match evaluation. Some wallets occur in
multiple splits, so it is not an unseen-actor evaluation.

Selection keeps complete actor-market target sequences. A deterministic seeded
round-robin samples across the three contracts, with 1,000 target turns per
training match and 375 per validation/test match. Sequences that exceed the
remaining budget are skipped, never truncated. The total cap is 15,000 assistant
target turns, not 15,000 conversations.

The full captured wallet-contract count ledger provides a <=20-observation
filter. This is a retrospective low-activity cohort: full-capture future activity
affects eligibility and sequence-length budgeting. Sampling does not use side,
outcome, shares, or price. The filter does not establish that actors are humans
or exclude all market makers.

This is a TRADE-only task conditional on a captured execution occurring. The
model predicts BUY/SELL, Yes/No, shares, and price. It does not learn whether or
when to trade. No NO_TRADE interval rows are used. The initial prompt includes
the evidenced initial market question and this actor's earlier captured trades
in that same contract. Later turns append only newly available match-event facts
and the current query time. Prior answers serve as history; evaluation using
true previous answers measures prediction conditional on observed history,
not autonomous rollouts.

The `news` lists contain `time`, `type`, and factual `text`, rendered from public
ESPN structured event data. Original narrative commentary and article bodies
are not redistributed. The 20 shared `sources/espn_*.json` files retain event
identities, source URL, capture timestamp, response hash, and timing provenance.
Only events with provider wallclock strictly before the target are included.
Event time is not verified historical publication or actor exposure. These
retrospective captures may contain later corrections. Pre-match reporting is
not included. Current ESPN final scores/status never enter the model prompts.

`audit.jsonl.gz` connects every selected target to source observation IDs and trade
row IDs, lists earlier history rows, and records which events entered each turn.
The SQLite source is read-only and original datasets remain unchanged. This
pilot build does not resume the stopped whole-tournament validation. API
observations are not certified canonical fills or complete on-chain history.

Before training, measure lengths with the exact base-model tokenizer. Do not
silently truncate targets. Apply loss only to assistant answers, and preserve
causal attention within each conversation. The model release/training cutoff
must be checked separately; chronological dataset splitting alone does not
prove that the base model has never seen the matches.
