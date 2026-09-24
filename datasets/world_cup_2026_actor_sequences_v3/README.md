# World Cup 2026 actor conversations, version 3

This export contains **3,974,200 selected observations** from **255,438 actors** across the configured 104 matches. Check `fixture_coverage.json` for actual retained target coverage.

There are **1,675,195 complete multi-turn conversations**. Each compressed JSONL line is one conversation; only its `messages` field is a model input. Turns within the line share attention. Separate lines do not.

The `conditional_trades` profile predicts observed execution attributes. The alternative `scheduled_windows` profile predicts observations or `NO_TRADE` over fixed 15-minute windows in explicitly monitored contracts. The two profiles have different prediction tasks.

The activity filter is at most 20 captured observations per actor and binary contract. News appears initially and as incremental updates. Initial summaries are numerical activity summaries, not inferred private beliefs.

**61,943 observations** are retained in the observation archive but excluded from targets by temporal or source-quality checks. Source-valid earlier observations can still supply history within their own fixture partition.

Reference tokens: maximum **10,301**. Check `requires_long_context` before selecting a trainer context limit. No target is silently truncated. Counts use the pinned Qwen3 tokenizer and supplied template.

Export status alone is not a validation result. The authoritative [independent report](../../reports/actor_sequences_validation.json) must say passed and match this manifest's SHA-256 before using this as the completed release. No model has been trained.

[Format and inspection](../../docs/actor_sequence_dataset.md) · [Training guide](../../docs/actor_sequence_training.md)
