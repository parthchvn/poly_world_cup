# Supporting and historical workflow tools

The current SFT workflow lives in `scripts/`:

1. `scripts/build_actor_dataset.py` collects actor rows and prepares validated SFT conversations and splits.
2. `scripts/train_world_cup_multigpu.py` trains the adapter.

See the [current workflow guide](../docs/actor_sft_pipeline.md) for commands.

This directory preserves the other 26 utilities for tournament collection,
news and contract context, older dataset formats, recovery, validation,
and reporting. They are not required for the current two-script workflow.
Their move does not make the older data formats interchangeable with the
current actor exports.

Run these tools from the repository root. For example:

```bash
python3 tools/collect_gdelt_news.py --help
python3 tools/validate_actor_sequences.py --help
```

Older commands of the form `python3 scripts/<utility>.py` now use
`python3 tools/<utility>.py`. Existing options and repository-relative data
locations are unchanged. Related workflow documentation remains in `docs/`.
