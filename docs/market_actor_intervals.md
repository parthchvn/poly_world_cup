# Per-actor market records with ESPN commentary

Run this from the repository, or from the extracted script bundle. It requires
Python 3.11 or later and uses only the standard library.

```bash
python scripts/build_market_actor_dataset.py 1897059 --out data/market_1897059
```

`1897059` is the Germany–Curaçao draw contract. Replace it with a Polymarket
numeric market ID, condition ID, or binary-market slug. An entire Polymarket
event containing several binary markets is not a single market ID.

The script resolves the fixture, fetches ESPN match commentary once, collects
the market's captured trades, and writes one chronologically ordered JSONL
file per actor. The default keeps actors with **at most 20 captured trades in
that market**, including exactly 20. To include every actor:

```bash
python scripts/build_market_actor_dataset.py 1897059 \
  --max-trades-per-actor 0 --out data/market_1897059_all_actors
```

## The requested row order

For distinct trade times `t1, t2, …, tn`, each actor file contains:

1. Interval `(origin, t1)`: all selected ESPN text with timestamps strictly
   inside that interval, labeled `NO_TRADE`.
2. Timestamp `t1`: the same interval's news, labeled with the trade at `t1`.
3. Interval `(t1, t2)`: new ESPN text in that interval, labeled `NO_TRADE`.
4. Timestamp `t2`: the same interval's news and the trade at `t2`.
5. Continue through `tn`. There is no added interval after the last trade.

`origin` is the reported market opening, or `--start` if supplied. When no
reliable opening is available, `start` is `null`, meaning all supplied ESPN
history before the first trade. The script does not invent an opening time.
Equal-time executions form one `TRADE` record with a `trades` array. News
exactly at a trade timestamp is excluded because within-timestamp ordering is
unknown. News text is deliberately repeated in each interval/trade pair,
matching this requested format.

Each row includes `actor_id`, `market_id`, `condition_id`, `record_type`,
`row_index`, `news`, and `label`. Interval rows have `interval`. Trade rows have
`timestamp` and `context_interval`. Shares and prices remain decimal strings.

## Actual ESPN content

The `news` array contains readable text, not article links. For example, an
item from ESPN event `401915443` is represented as:

```json
{
  "time": "2026-09-10T19:16:34Z",
  "match_clock": "14'",
  "type": "start-delay",
  "text": "Delay in match because of an injury Luis Díaz (Bayern Munich).",
  "time_basis": "provider_event_time_proxy"
}
```

The default includes all timestamped commentary/key events returned for that
match, plus explicitly associated headline metadata where available. It does
not crawl ESPN's full historical article archive. The same parser retains
reported goals, yellow/red cards, substitutions, shots, fouls, and injuries.
`--key-events-only` restricts the output to goals, cards, substitutions,
penalties, VAR, injuries, and match-phase events. `--include-core-plays` also
fetches ESPN's paginated granular plays, which can include routine passes.

ESPN's event `wallclock` is used as an event-time proxy. It does not prove when
the actor saw the update or when ESPN first published it. Events without a
usable absolute timestamp are saved in `espn_unplaced_events.jsonl`, outside
the actor rows. The script reports their count. If no selected items can be
placed, it stops with instructions instead of producing empty news features.

## Output and inspection

The output directory contains:

- `actors/<wallet>.jsonl`: the requested interval and trade rows.
- `actor_index.jsonl`: actor file paths and row/trade counts.
- `espn_events.jsonl`: shared timestamped event text and source IDs.
- `espn_unplaced_events.jsonl`: text that could not be placed in a UTC interval.
- `market.json`, `espn_sources.json`, `manifest.json`: mappings and collection details.

Use `--gzip` to write compressed actor files. Without that flag, inspect one
complete row with:

```bash
head -n 1 data/market_1897059/actors/0x00aec21f5151f554dc1452724dd2b4f5c7ce5de1.jsonl | python -m json.tool
```

That wallet appears in the previously captured market data. New captures may
differ; `actor_index.jsonl` lists the actors present in your export.

## Existing trades and other leagues

If ESPN returns HTTP 403 for Germany–Curaçao, the exporter automatically uses
the included historical event file for ESPN match `760422`. To skip the live
ESPN request altogether, run:

```bash
python3 scripts/build_market_actor_dataset.py 1897059 \
  --espn-file data_sources/espn/fifa.world_760422.json.gz \
  --out data/germany_curacao_draw
```

This file preserves the captured event types, teams, participants, match clocks,
and provider UTC event times. Its short text is rendered from those structured
facts, rather than reproducing article bodies or narrative commentary. Capture
details are included in `source_provenance` inside the file. It is a historical
snapshot, not a live feed. For other matches without a bundled snapshot, the
error identifies the failing URL and explains how to supply `--espn-file`.

Reuse an existing CSV, JSON array, or JSONL file (optionally gzip):

```bash
python scripts/build_market_actor_dataset.py 1897059 \
  --trades-file my_market_trades.csv --out data/market_from_csv
```

Supported fields include `actor_id`/`wallet`/`proxyWallet`,
`timestamp`/`block_timestamp`/`query_us`, `side`, `outcome`, `shares`/`size`,
and `price`. Token IDs can supply the outcome through the market mapping.
An existing repository database can be used read-only with `--sqlite PATH`.
Previously filtered inputs cannot restore excluded actors or executions.

For a different competition, set its ESPN league and, if necessary, match ID:

```bash
python scripts/build_market_actor_dataset.py YOUR_MARKET_ID \
  --league uefa.champions --espn-event-id 401915443 \
  --out data/your_market
```

That ESPN ID is the Bayern–Bodø/Glimt match on September 10, 2026. Use it only
with the corresponding Polymarket market. Otherwise supply the matching
ESPN ID, or let the script find a unique fixture from the teams and date.

For offline operation, combine `--market-metadata market.json`,
`--trades-file trades.jsonl`, and one or more `--espn-file summary.json`
arguments. The metadata input must be a saved Gamma/registry market object,
not this exporter's normalized `market.json`. ESPN files can be saved summary
JSON, core-play pages, or normalized JSONL with `text` and `time_utc`.

`--time-map times.json` accepts explicit event times, for example the schema
`{"events": {"PLAY_ID": "ISO_UTC_TIMESTAMP"}}`. Optional `period_anchors`
entries map a period number to `time_utc` and `clock_seconds`.
`--allow-clock-estimates` permits estimates only with an anchor for the same
period and marks them as estimates. The script never silently treats halftime
or extra time as a continuous countdown from scheduled kickoff.

## Efficiency and interpretation

ESPN is fetched once per match, not once per actor. HTTP responses and the
cursor-based trade capture are cached for reuse. Trade sorting uses temporary
SQLite storage, and news windows use binary search over the shared event
timeline. Memory usage does not scale with the full market's trade count.
Actors are filtered only after all supplied market observations are counted.
Each output must be a new directory, so prior datasets are preserved.

The cache is a saved capture. Use a new `--cache` directory to collect a fresh
snapshot of an ongoing market. API exhaustion describes the returned feed,
not proof of every historical on-chain execution. `NO_TRADE` means no captured
execution in this actor–market interval.

These are **retrospective interval descriptions**: the future trade determines
where each gap ends, and interval/trade row types reveal the action class.
They should not be presented as forward trade-occurrence prediction examples.
For sequence training, put the actor's records within the same model input;
ordering separate training rows does not create attention between them.

Source endpoints used by the script:

- Polymarket: `https://data-api.polymarket.com/v2/trades` and the Gamma market API.
- ESPN: `https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary?event={id}`.
- Optional ESPN core plays: `https://sports.core.api.espn.com/v2/sports/soccer/leagues/{league}/events/{id}/competitions/{id}/plays`.
