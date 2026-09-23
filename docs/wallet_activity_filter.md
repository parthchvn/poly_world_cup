# Wallet activity filter

`scripts/filter_wallet_activity.py` creates a separate filtered dataset. It opens
the original database read-only and refuses to overwrite any existing output.
It does not replace previously generated or uploaded datasets.

The rule is:

```sql
GROUP BY wallet, condition_id
HAVING COUNT(*) < 20
```

A market is one Polymarket binary contract, identified by `condition_id`.
Its Yes/No outcomes and BUY/SELL observations all count together. Every saved
observation for a qualifying wallet-market pair is retained. Pairs with 20 or
more are excluded entirely. A wallet can qualify in one market and fail in
another. This does not keep the first 19 observations from an active wallet.

This is an activity-based proxy for excluding market makers, not verified
classification of market makers or people. It counts the API observations in
the source snapshot, not unique transactions or reconciled fills. Unfinished
source histories can undercount activity. The count uses the full captured
period and is a retrospective selection rule, not a pre-trade input feature.

## Generate a separate dataset

Run from the repository root after obtaining the original attribution database:

```bash
python scripts/filter_wallet_activity.py \
  --database data/full/partial_attribution.sqlite \
  --output data/filtered/world_cup_lt20.sqlite \
  --report data/filtered/filter_report.json
```

The default threshold is strictly less than 20. `--threshold 10`, for example,
retains pairs with 1–9 observations. Outputs are never overwritten; choose new
paths when rerunning. The original input must be closed to writers and have no
uncheckpointed SQLite WAL.

The generated database contains:

| Table | Contents |
| --- | --- |
| `selected_trades` | All qualifying target observations, with original row IDs, exact decimal strings, and context references |
| `wallet_market_counts` | Observation totals for every source wallet-market pair, including excluded pairs |
| `news`, `fixture_news`, `global_news`, `context_states` | Original news records, links, availability cutoffs, and state IDs |
| `source_pages` | Original source-page references, hashes, and full-source page counts |
| `source_metadata` | Original dataset metadata, unchanged |
| `metadata` | Filter report, source identity, counts, and limitations |

Original source-page counts describe the original pages, not the selected
subset. The source metadata hashes and filesystem identity in the report are
not a hash of the entire database file.

## View the filtered rows

The prepared download is split into two files to meet the per-file download
limit. Download both into the same folder, then run:

```bash
cat world_cup_lt20.part01 world_cup_lt20.part02 > world_cup_lt20.zip
unzip world_cup_lt20.zip
```

The ZIP includes the SQLite database, count report, instructions, and a small
preview. [Download checksums](../reports/wallet_activity_downloads.json) identify
the exact files. You can also view [20 selected rows](../reports/wallet_activity_preview.jsonl)
directly in the repository. After extracting, use `world_cup_lt20.sqlite` in the
example below; a locally generated copy uses the shown `data/filtered/` path.

```python
import sqlite3

db = sqlite3.connect("data/filtered/world_cup_lt20.sqlite")
for row in db.execute("""
    SELECT t.trade_row_id, t.fixture_id, t.wallet, t.condition_id,
           t.side, t.shares, t.price, c.observation_count
    FROM selected_trades t
    JOIN wallet_market_counts c USING (wallet, condition_id)
    ORDER BY t.trade_row_id
    LIMIT 20
"""):
    print(row)
```

## Retrieve the original context

Use a retained `trade_row_id` with the **original** database:

```bash
python -m poly_world_cup inspect-context \
  --database data/full/partial_attribution.sqlite \
  --row RETAINED_TRADE_ROW_ID
```

The filtered database deliberately names its target table `selected_trades`.
This prevents the existing context reader from silently treating selected
targets as the actor's full history. Earlier observations from an excluded
market can still be relevant context for a retained decision in another market.
They remain available through the unchanged original database. No historical
availability or model-input eligibility flags are upgraded by this filter.

The current results are recorded in [the filter report](../reports/wallet_activity_filter.json).
The captured checkpoint retained **3,810,250 observations** from **252,141
wallets** in **2,127,228 qualifying wallet-market pairs**. It excluded
**8,048,334 observations** in **54,391 pairs**. All 104 fixtures and 312 markets
still have selected observations. A retained wallet can also have excluded
observations in a different market.
