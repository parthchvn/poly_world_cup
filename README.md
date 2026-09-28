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
data/market_1897059/actor_snapshots/<wallet>.json
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

Each row also carries `market_context`: the latest eligible earlier YES and NO
prices from Polymarket's official CLOB `/prices-history` series, their
price-implied probabilities, and their timestamps/ages. The builder fetches and
caches each outcome token's history once for the market, requesting one-minute
fidelity, then shares it across actors. It does not estimate these prices from
captured trades, interpolate, or use future observations. Prices older than
300 seconds are unavailable by default; change this with
`--market-price-max-age-seconds`. Missing or stale row context remains `null`;
an outcome with no usable prices across the retained trade rows stops the
export instead of producing an apparently enriched dataset. Partial coverage
prints a warning and preserves the missing values.
Training still runs actor by actor. Historical series points are not executable
quotes or verified screenshots of what the actor saw. The timestamp cutoff uses
the recorded execution block time, which can follow order submission and
matching; it does not certify that a point predates the actual decision.

`execution_info` records observed fills and explicitly leaves unavailable order
submission times, limit-order status, and submission-to-fill delays unknown.
`payoff_analysis` records a BUY's potential profit if held to a winning
resolution, before fees. It does not claim an expected profit or reconstruct
unobserved orders. See the guide for the exact limitations.

The builder also fetches three **collection-time actor snapshots**, scoped to
this wallet and market: `actor_market_value`, `actor_positions_open`, and
`actor_positions_closed`. They preserve the API's position, cost, fee, and P&L
fields. Each actor's snapshot is stored once in `actor_snapshots/<wallet>.json`;
every raw row and actor-index entry points to it through `actor_snapshot_ref`.
These are the wallet's state when queried, not its state at the historical trade.
They are retained for analysis and excluded from historical SFT inputs and targets.

Inspect the first two records and their shared actor snapshot:

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
with path.open() as stream:
    row = json.loads(next(stream))
snapshot = path.parent.parent / row['actor_snapshot_ref']
print(f'Actor snapshot: {snapshot}')
print(json.dumps(json.loads(snapshot.read_text()), indent=2, ensure_ascii=False))
PY
```

The same output directory contains `manifest.json`, `market.json`, an actor index,
ESPN event/source files, `market_price_history.jsonl`, and
`market_price_sources.json` for auditing the shared prices and request sources.
The manifest records `market_context_version: 2` and `actor_snapshots.version: 1`.
These outputs are generated locally, not committed.

## Source availability and cache behavior

Germany–Curaçao (`1897059`) has a confirmed local end-to-end run. The builder
bundles the 104-fixture/312-contract World Cup registry and the 93-event
Germany–Curaçao snapshot. Other matches use cached or supplied ESPN data, or live
ESPN requests. No historical GitHub data URL is needed for current collection.

ESPN/API access can fail or lack usable absolute timestamps. Inspect those errors
and the coverage metadata rather than treating missing information as complete.
An exhausted trade API traversal does not certify complete on-chain history.
`NO_TRADE` describes an observed gap, not proof of a conscious decision to abstain.
The next trade determines the interval end retrospectively. ESPN event wallclock
is an occurrence proxy, not verified historical text-publication time.

Trade collection and actor snapshot requests are cached and resumable. Reusing
the cache retains each snapshot's original retrieval time; it does not refresh
holdings or P&L. Choose a new `--cache` and `--out` for a fresh collection.
Snapshot collection needs at least three requests per retained actor, plus
position pagination. It uses eight bounded workers by default; reduce this with
`--actor-snapshot-workers`. Failed requests stop the export rather than inventing
zero balances or empty positions. Successful zero values and empty lists are valid.

## Train on local conversation files

The core workflow has two steps: `build_actor_dataset.py` collects actor exports
and prepares SFT conversations; `train_world_cup_multigpu.py` trains them.
The optional `derive_actor_metrics.py` adds strictly prior actor metrics between
preparation and training.

Collect three distinct matches and prepare the dataset in one command. On RunPod,
use a new data directory to rebuild earlier exports with market prices and actor
snapshots while reusing the existing trade cache:

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

python3 scripts/train_world_cup_multigpu.py \
  --gpus 2 --gpu-ids 0,1 \
  --model /workspace/models/Qwen3.6-27B \
  --dataset-dir /workspace/datasets/world_cup_sft_with_actor_snapshots \
  --smoke-then-full
```

`--reuse-existing` requires completed matching exports with both version-2 market
context and version-1 actor snapshots. Rebuild older exports into a new directory;
collection options apply only to newly built exports. An exhausted trade capture
is reused without another trade scrape; incomplete captures resume. Price history
and actor snapshots are separately cached sources and may need new API requests.
Choose a new `--out` if the SFT dataset already exists. Preparation copies the actor
snapshots into the SFT dataset's `audit/actor_snapshots/` directory, with references
and hashes in `source_audit.jsonl`; it never inserts them into model messages.
Earlier trained adapters do not gain historical market-price inputs automatically;
prepare an enriched dataset and train a new run to use those inputs.

To prepare existing exports without collection, use:

```bash
python3 scripts/build_actor_dataset.py prepare \
  --input-root /workspace/world_cup_actor_data/data_with_actor_snapshots \
  --out /workspace/datasets/world_cup_sft_new \
  --tokenizer /workspace/models/Qwen3.6-27B
```

`--smoke-then-full` checks up to 32 validation conversations after the first ten
optimizer steps, then continues the full run with the same loaded model and
optimizer. Those steps count toward training and use the full-run learning-rate
schedule. Use `--smoke` only for a standalone ten-step test that exits afterward.
These commands require your local model files and working training dependencies. On a Mac,
conversion can run without `--tokenizer`, deferring exact length checks to the
trainer. Conversion does not truncate conversations or discard targets.

The converter predicts observed **trade attributes only**. Each user turn gets
that time's earlier YES/NO prices and their ages alongside news and actor history.
SFT uses a compact `market_context` containing only `price` and `age_seconds`
for each available outcome; the full timestamps, probabilities, and provenance
remain in the actor files and audit timeline.
Collection-time actor value and positions, present-day order books, and later
resolution outcomes are not substituted for historical context. Actor snapshots,
execution metadata, and payoff metadata are retained for analysis, not inserted
into SFT inputs or targets. The converter checks but does not train on
retrospective `NO_TRADE` intervals. Earlier trades stay in preceding conversation turns;
interval news appears once per turn. All markets from one match stay in one split.
Three matches are a smoke pipeline, not a sufficient performance benchmark.

See [the complete collection-to-training guide](docs/actor_sft_pipeline.md) for
input selection, split reuse, output inspection, token limits, and limitations.
Always specify `--dataset-dir`; the trainer's historical default refers to the
older local pilot export.

The trainer performs single-machine data-parallel QLoRA, with one model replica
per GPU and one resulting shared adapter. Actual H100/NCCL execution still needs
the smoke test. Generated runs, checkpoints, and adapters must stay outside Git.

## Add actor performance and behavior metrics

`scripts/derive_actor_metrics.py` is a separate, standard-library-only postprocessor.
It calculates 18 metrics for each actor and market using information strictly
before each execution timestamp. It excludes time since the previous trade and
historical markout. It does not fetch APIs, use collection-time actor snapshots,
or change the source exports.

```bash
python3 scripts/derive_actor_metrics.py data/market_1897059 \
  --out data/market_1897059_metrics
```

Execution notional, notional variability, frequency, and buy share can use the
captured fills directly. Realized performance and holding duration need a supplied
completed-position ledger; Sharpe, Sortino, drawdown, and return volatility need
a supplied, capital-flow-adjusted equity-return series. Missing inputs produce
`null` metrics with reasons and sample counts. The script does not invent a
portfolio Sharpe from execution prices or assume public trade capture is complete.

To add the metrics to an existing prepared SFT dataset, pass
`--sft-dir datasets/world_cup_sft`. The enriched dataset is written under the new
output's `sft/` directory; point the trainer's `--dataset-dir` there. Each user
turn gets only its strictly prior metrics, with original splits and targets
preserved. See [actor metrics](docs/actor_metrics.md) for the input schemas,
formulas, time cutoffs, and complete training example.

## Current entry points

The `scripts/` directory contains only the current SFT workflow:

| Script | Purpose |
|---|---|
| `scripts/build_actor_dataset.py` | Collect actor rows, prepare existing exports, or collect multiple markets and prepare SFT in one command. |
| `scripts/derive_actor_metrics.py` | Add strictly prior actor behavior/performance metrics; optionally enrich prepared SFT conversations. |
| `scripts/train_world_cup_multigpu.py` | Run QLoRA training on one or multiple GPUs. |

Additional collection, recovery, reporting, and historical workflow utilities
are in [`tools/`](tools/README.md). Run them from the repository root with
`python3 tools/<name>.py`; their existing command-line options are retained.

The old
`build_market_actor_dataset.py`, `train_market_qlora.py`, news-compaction helper,
and GitHub dataset uploader have been retired. Their source remains in the
external pre-cleanup Git backup.

## Repository layout and local development

- `scripts/`: the current dataset builder, optional actor-metrics postprocessor, and trainer.
- `tools/`: additional and historical collection, recovery, validation, and reporting utilities.
- `poly_world_cup/`: supporting modules imported by the repository's scripts.
- `configs/`: small configuration files and reviewed mappings used by the code.
- `tests/` and `.github/workflows/`: offline regression tests and CI.
- `docs/`: methodology and older workflow references.

```bash
python -m unittest discover -s tests -v
```

Older methodology documents describe historical experiments and may include
commit links that no longer resolve after the history rewrite. Archived datasets
and reports are not part of this repository. Use local exports for training.

Generated data and model outputs are ignored by Git. A normal clone no longer
includes the removed datasets from earlier commits; --depth 1 is optional.
