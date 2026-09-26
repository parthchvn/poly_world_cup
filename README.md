# World Cup collection and training scripts

Generate World Cup actor datasets locally and use local conversation exports for
supervised fine-tuning. This repository keeps scripts, their supporting Python
modules, configuration, documentation, and tests. Generated datasets, captures,
download archives, model weights, and run outputs belong outside Git.

## Generate an actor dataset

Python 3.11+ on Linux/macOS. The standalone collector uses the standard library.

```bash
git clone --depth 1 https://github.com/parthchvn/poly_world_cup.git
cd poly_world_cup
python3 scripts/build_actor_dataset.py 1897059
```

Replace `1897059` with a World Cup binary market ID. This creates:

```text
data/market_1897059/actors/<wallet>.jsonl
```

Run from the repository root to keep outputs under its ignored `data/` directory.
The script resolves paths from the working directory, not its own file location.
Existing output directories are not overwritten; use `--out` for a new export.

Each actor's chronological file contains an open-interval `NO_TRADE` row followed
by a `TRADE` row for each distinct execution time. Simultaneous executions share
one trade row. Both rows carry news strictly inside the preceding interval.
Earlier actor trades remain in preceding rows rather than being copied into
every later row. The default actor filter is **at most 20 captured executions**
in the selected binary market.

Inspect the first two records:

```bash
python3 - <<'PY'
import json
from itertools import islice
from pathlib import Path

path = next(Path('data/market_1897059/actors').glob('*.jsonl'), None)
if path is None:
    raise SystemExit('No actor files found; check the output path and manifest.')
print(path)
with path.open() as stream:
    for line in islice(stream, 2):
        print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
PY
```

The same output directory contains `manifest.json`, `market.json`, an actor index,
and ESPN event/source files. These are generated locally, not committed.

## Source availability and cache behavior

Germany–Curaçao (`1897059`) has a confirmed local end-to-end run. The standalone
script includes its small metadata/event snapshot. For other World Cup markets,
it uses pinned historical registry/snapshot URLs when available and public APIs
as fallbacks. It downloads data, never executable source. This cleanup preserves
those historical commits, so their pinned URLs continue to work. The script is
not a promise that every match can be reconstructed from live endpoints alone.

ESPN/API access can fail or lack usable absolute timestamps. Inspect those errors
and the coverage metadata rather than treating missing information as complete.
An exhausted trade API traversal does not certify complete on-chain history.
`NO_TRADE` describes an observed gap, not proof of a conscious decision to abstain.
The next trade determines the interval end retrospectively. ESPN event wallclock
is an occurrence proxy, not verified historical text-publication time.

Trade collection is cached and resumable. Reusing the cache reuses the capture;
choose a new `--cache` and `--out` for an independent fresh collection.

## Train on local conversation files

The intended workflow is:

1. Collect with `scripts/build_actor_dataset.py`.
2. Convert actor rows into chronological `messages` conversations and split by
   match into local training, validation, and test files.
3. Train with `scripts/train_world_cup_multigpu.py`.

**Step 2 is not yet implemented for this standalone builder's interval format.**
The trainer does not read `actors/` directly. Existing legacy exporters target
other validated schemas; their presence is not a converter for these rows.

Once compatible conversation splits exist locally, run the GPU smoke test in
your established training environment, supplying the model and dataset paths:

```bash
python scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir /workspace/poly_world_cup/datasets/world_cup_sft \
  --smoke
```

The example dataset directory must first contain `train.jsonl` or `train.jsonl.gz`
and the corresponding validation file in the trainer's conversation schema.
Remove `--smoke` after a successful test to start a full run. The launcher does
not install its training dependencies or download weights/data. Use an explicit
`--dataset-dir`; its historical default refers to the earlier local pilot export.

The trainer performs single-machine data-parallel QLoRA, with one model replica
per GPU and one resulting shared adapter. Actual H100/NCCL execution still needs
the smoke test. Generated runs, checkpoints, and adapters must stay outside Git.

## Repository layout and local development

- `scripts/`: collection, conversion/validation utilities for their documented
  schemas, and trainers.
- `poly_world_cup/`: supporting modules imported by the repository's scripts.
- `configs/`: small configuration files and reviewed mappings used by the code.
- `tests/` and `.github/workflows/`: offline regression tests and CI.
- `docs/`: methodology and older workflow references.

```bash
python -m unittest discover -s tests -v
```

Older documentation may refer to prepared releases and generated reports. Those
files are no longer present in the current tree. The
[pre-cleanup tree](https://github.com/parthchvn/poly_world_cup/tree/ecd4b04e083fad9e6188f12129a7278b4b6b0e93)
preserves that historical record, including the previous release README.

`.gitignore` excludes generated data directories, archives, databases, and model
artifacts. Stage specific source files when committing; do not force-add outputs.
The cleanup removes about 10 GB of artifacts from the current tree, without
rewriting Git history. Use `--depth 1` for a fresh checkout without that history.
