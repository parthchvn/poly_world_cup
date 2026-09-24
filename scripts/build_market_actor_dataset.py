#!/usr/bin/env python3
"""Export one market as per-actor interval / execution rows with ESPN text.

Usage: python scripts/build_market_actor_dataset.py 1897059 --out data/market_1897059

The two records for each distinct execution time are:
  (previous execution, current execution): news in the open interval, NO_TRADE
  current execution: the same news, and all captured executions at that time.

These are retrospective descriptions of observed gaps, not prospective
trade-occurrence targets. They never assert absence outside the supplied feed.
Only Python 3.11+ and the standard-library modules in this repository are needed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import gzip
from itertools import groupby
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poly_world_cup.espn_events import collect_espn_context
from poly_world_cup.http import HttpClient
from poly_world_cup.market_trade_source import load_market_trades, resolve_market, timestamp_us, utc_time


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_events(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(compact(row) + "\n")


def key_event(event):
    """Optional text/type filter. The default retains every timed commentary."""
    value = (str(event.get("kind", "")) + " " + event["text"]).casefold()
    return bool(re.search(r"\b(goal|yellow|red card|second yellow|substitution|substitute|"
                          r"penalty|penalties|var|injur\w*|half[- ]?time|full[- ]?time|"
                          r"kick[- ]?off|disallow\w*|lineup|line-up)\b", value))


def news_feature(event):
    # The feature contains the actual text. IDs/URLs are kept in the shared
    # source files, never substituted for the news supplied to the model.
    # Match clocks and per-event timing provenance also remain in those files.
    return {"time": event["time_utc"], "type": event.get("kind"), "text": event["text"]}


def stage_trades(db, iterator):
    """Disk-backed sort: do not keep the full market or its wallets in RAM."""
    db.execute("CREATE TABLE trades (ordinal INTEGER PRIMARY KEY, actor TEXT, instant INTEGER, payload TEXT)")
    batch, count = [], 0
    for row in iterator:
        batch.append((row["actor_id"], row["time_us"], compact(row)))
        count += 1
        if len(batch) >= 10000:
            db.executemany("INSERT INTO trades(actor,instant,payload) VALUES (?,?,?)", batch)
            batch.clear()
    if batch:
        db.executemany("INSERT INTO trades(actor,instant,payload) VALUES (?,?,?)", batch)
    db.execute("CREATE INDEX actor_time ON trades(actor,instant,ordinal)")
    db.execute("CREATE TABLE actor_counts AS SELECT actor,COUNT(*) n FROM trades GROUP BY actor")
    db.execute("CREATE UNIQUE INDEX actor_count_id ON actor_counts(actor)")
    db.commit()
    return count


def actor_records(actor, trades, market, events, event_times, origin):
    """Yield exactly two records per distinct execution timestamp.

    News uses strict lower < news time < current execution time. Equal-time
    executions are grouped rather than assigned an invented internal order.
    A gap's news is deliberately repeated on the following trade row, matching
    the requested schema. No history is expanded again on subsequent rows.
    """
    previous = origin
    for index, (instant, executions) in enumerate(groupby(trades, lambda row: row["time_us"]), 1):
        lo = 0 if previous is None else bisect_right(event_times, previous)
        hi = bisect_left(event_times, instant)
        news = [news_feature(event) for event in events[lo:hi]]
        interval = {"start": utc_time(previous) if previous is not None else None,
                    "end": utc_time(instant), "start_inclusive": False, "end_inclusive": False}
        base = {"actor_id": actor, "market_id": market["market_id"], "condition_id": market["condition_id"]}
        yield {**base, "record_type": "interval", "row_index": 2 * index - 2,
               "interval": interval, "news": news, "label": {"action": "NO_TRADE"}}
        values = list(executions)
        yield {**base, "record_type": "trade", "row_index": 2 * index - 1,
               "timestamp": utc_time(instant), "context_interval": interval,
               "news": news, "label": {"action": "TRADE", "trades": [
                   {"time": value["time"], **value["trade"]} for value in values]}}
        previous = instant


def export(args):
    if args.max_trades_per_actor < 0:
        raise ValueError("--max-trades-per-actor must be nonnegative; 0 includes all actors")
    if args.trade_capture and (args.trades_file or args.sqlite):
        raise ValueError("Choose --trade-capture, --trades-file, or --sqlite, not several sources")
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", args.market_id)
    output = (args.out or ROOT / "data" / "actor_market_intervals" / key).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new --out directory.")
    # This exporter always creates a separate dataset. Never write inside a
    # previous prepared release, even if a new subdirectory was requested.
    for parent in (output, *output.parents):
        if (parent / "manifest.json").exists():
            raise ValueError("Choose an output directory outside existing dataset releases")
    cache = args.cache.resolve()
    if cache == output or cache.is_relative_to(output):
        raise ValueError("Keep --cache outside the new --out directory")
    client = HttpClient(cache / "http", compress=True)
    print(f"Resolving market {args.market_id}", flush=True)
    market = resolve_market(args.market_id, client=client, metadata_file=args.market_metadata)
    event_id = args.espn_event_id or market.get("espn_event_id")
    fixture_date = args.date or market.get("fixture_date")
    teams = args.teams or market.get("team_names")
    print("Fetching ESPN commentary once for this match", flush=True)
    context_options = dict(event_id=event_id, league=args.league,
        fixture_date=fixture_date, teams=teams, espn_files=args.espn_file, time_map=args.time_map,
        time_policy="provider", allow_clock_estimates=args.allow_clock_estimates,
        include_core_plays=args.include_core_plays)
    try:
        context = collect_espn_context(client, **context_options)
    except HTTPError as error:
        # Public ESPN availability can differ between networks. A previously
        # captured factual event file needs no live ESPN request or credentials.
        safe_id = str(event_id) if str(event_id).isdigit() else "unknown"
        snapshot = ROOT / "data_sources" / "espn" / f"{args.league}_{safe_id}.json.gz"
        if error.code == 403 and not args.espn_file and not args.include_core_plays and snapshot.is_file():
            print(f"ESPN returned HTTP 403; using saved match events: {snapshot}", flush=True)
            context_options.update(espn_files=[snapshot], include_core_plays=False)
            context = collect_espn_context(client, **context_options)
        else:
            raise ValueError(f"ESPN returned HTTP {error.code} for {error.url}. "
                "Use --espn-file PATH with a saved ESPN summary or event JSON/JSONL file. "
                "No trade collection or actor export has started.") from error
    events = [event for event in context["timed_events"] if not args.key_events_only or key_event(event)]
    events.sort(key=lambda event: (event["timestamp_us"], event["news_id"]))
    event_times = [event["timestamp_us"] for event in events]
    # Do not silently return an apparently news-enriched dataset whose entire
    # commentary could not be assigned to any real-world time interval.
    if not events:
        diagnostic = cache / "espn_unplaced" / (str(context["event_id"]) + ".json")
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        write_json(diagnostic, context)
        raise ValueError(f"No timestamped ESPN text is available for this selection. Saved source details: {diagnostic}. "
                         "Supply --espn-file or --time-map, or use --allow-clock-estimates with period anchors.")
    print(f"Loaded {len(events):,} timed ESPN items; {len(context['untimed_events']):,} items have no assignable UTC time", flush=True)
    iterator, source_report = load_market_trades(market, cache_dir=cache, client=client,
        trades_file=args.trades_file, sqlite_path=args.sqlite, capture_dir=args.trade_capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling directory, then publish it with one rename. An error
    # leaves the existing datasets untouched, while the HTTP/trade cache stays.
    work = Path(tempfile.mkdtemp(prefix="market-actor-build-", dir=output.parent))
    try:
        db_path = work / "sort.sqlite"
        db = sqlite3.connect(db_path)
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            print("Reading captured trades and sorting actors on disk", flush=True)
            observations = stage_trades(db, iterator)
            if not observations:
                raise ValueError("No captured trades found for this market")
            minimum = db.execute("SELECT MIN(instant) FROM trades").fetchone()[0]
            origin = timestamp_us(args.start) if args.start else (
                timestamp_us(market["market_open_utc"]) if market.get("market_open_utc") else None)
            origin_basis = "user_supplied" if args.start else (market.get("market_open_basis") or "unbounded_source_history")
            if not args.start and origin is not None and origin > minimum:
                # A current opening field may refer to a reopening. Keep all
                # historical executions instead of silently cutting them away.
                origin, origin_basis = None, "opening_later_than_first_observation_unbounded"
            threshold = args.max_trades_per_actor
            query = """SELECT t.actor,t.payload FROM trades t JOIN actor_counts c ON c.actor=t.actor
                WHERE (?=0 OR c.n<=?) AND (? IS NULL OR t.instant>=?)
                ORDER BY t.actor,t.instant,t.ordinal"""
            rows = db.execute(query, (threshold, threshold, origin, origin))
            actors_dir = work / "actors"
            actors_dir.mkdir()
            counts = Counter()
            with (work / "actor_index.jsonl").open("w", encoding="utf-8") as index_stream:
                for actor, actor_rows in groupby(rows, lambda item: item[0]):
                    filename = actor + (".jsonl.gz" if args.gzip else ".jsonl")
                    path = actors_dir / filename
                    opener = gzip.open if args.gzip else open
                    actor_count = Counter()
                    trades = (json.loads(item[1]) for item in actor_rows)
                    with opener(path, "wt", encoding="utf-8") as stream:
                        for row in actor_records(actor, trades, market, events, event_times, origin):
                            stream.write(compact(row) + "\n")
                            actor_count["rows"] += 1
                            actor_count["news_entries"] += len(row["news"])
                            if row["record_type"] == "trade":
                                actor_count["distinct_trade_times"] += 1
                                actor_count["trade_observations"] += len(row["label"]["trades"])
                    index_stream.write(compact({"actor_id": actor, "path": "actors/" + filename, **dict(actor_count)}) + "\n")
                    counts.update(actor_count)
                    counts["actors"] += 1
                    if counts["actors"] % 1000 == 0:
                        print(f"Wrote {counts['actors']:,} actors / {counts['rows']:,} rows", flush=True)
            all_actors = db.execute("SELECT COUNT(*) FROM actor_counts").fetchone()[0]
            excluded = db.execute("SELECT COUNT(*) FROM actor_counts WHERE ?!=0 AND n>?", (threshold, threshold)).fetchone()[0]
        finally:
            db.close()
        db_path.unlink()
        write_json(work / "market.json", market)
        write_events(work / "espn_events.jsonl", context["timed_events"])
        write_events(work / "espn_unplaced_events.jsonl", context["untimed_events"])
        write_json(work / "espn_sources.json", {key: value for key, value in context.items()
                   if key not in {"timed_events", "untimed_events"}})
        manifest = {"format": "actor_market_intervals_v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "market_id": market["market_id"], "condition_id": market["condition_id"],
            "espn_event_id": context["event_id"], "counts": dict(counts),
            "source_trade_observations": observations, "source_actors": all_actors,
            "actors_excluded_above_trade_limit": excluded,
            "max_trades_per_actor": threshold or None,
            "filter_scope": "all_captured_observations_for_actor_in_this_binary_market_before_applying_start_cutoff",
            "origin_utc": utc_time(origin) if origin is not None else None, "origin_basis": origin_basis,
            "espn_timed_items": len(events), "espn_unplaced_items": len(context["untimed_events"]),
            "key_events_only": args.key_events_only, "source": source_report,
            "record_order": ["open_interval_before_trade", "trade_timestamp"],
            "equal_time_trades": "one_trade_row_containing_all_same_time_observations",
            "news_boundary_rule": "start < news_timestamp < trade_timestamp; equal-time news excluded",
            "news_repeated_on_adjacent_trade_row": True,
            "trailing_interval_after_last_trade": False,
            "no_trade_semantics": "zero_captured_observations_in_this_actor_market_open_interval",
            "task_semantics": "retrospective_gap_description; future_execution_defines_interval_end",
            "prospective_trade_occurrence_training_ready": False,
            "news_feature_fields": ["time", "type", "text"],
            "news_source_metadata": "espn_events.jsonl",
            "timestamp_semantics": "ESPN provider wallclock is an occurrence proxy; per-event timing bases and explicit estimates are recorded in espn_events.jsonl",
            "complete_historical_espn_news_archive": False,
            "previous_datasets_modified": False}
        write_json(work / "manifest.json", manifest)
        if output.exists():
            raise ValueError("The output directory appeared during export; choose another --out")
        work.rename(output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), **dict(counts)}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("market_id", help="Polymarket numeric market ID, condition ID, or binary-market slug")
    parser.add_argument("--out", type=Path, help="New output directory, never an existing dataset")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "market_actor_cache")
    parser.add_argument("--espn-event-id", help="Override automatic match lookup")
    parser.add_argument("--league", default="fifa.world", help="ESPN league slug, e.g. fifa.world or uefa.champions")
    parser.add_argument("--date", help="Fixture date YYYY-MM-DD, if absent from market metadata")
    parser.add_argument("--teams", nargs=2, help="Two ESPN team names, if absent from market metadata")
    parser.add_argument("--max-trades-per-actor", type=int, default=20, help="Inclusive maximum in this market; 0 keeps every actor")
    parser.add_argument("--key-events-only", action="store_true", help="Keep goal/card/substitution/penalty/VAR/injury/match-phase text")
    parser.add_argument("--include-core-plays", action="store_true", help="Also fetch paginated granular ESPN plays, including routine play")
    parser.add_argument("--gzip", action="store_true", help="Compress each actor's JSONL file")
    parser.add_argument("--start", help="Origin of the first interval (zoned ISO time or epoch seconds)")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--trades-file", type=Path, help="Existing market trades CSV/JSON/JSONL, optionally gzip")
    inputs.add_argument("--sqlite", type=Path, help="Existing repository trades SQLite, read-only")
    parser.add_argument("--trade-capture", type=Path, help="Directory of existing resumable v2 condition captures")
    parser.add_argument("--market-metadata", type=Path, help="Saved Gamma market object for offline lookup")
    parser.add_argument("--espn-file", type=Path, action="append", help="Saved ESPN summary/core-play JSON or normalized JSONL; repeatable")
    parser.add_argument("--time-map", type=Path, help="Explicit event UTC timestamps and/or same-period clock anchors")
    parser.add_argument("--allow-clock-estimates", action="store_true", help="Explicitly permit approximate same-period anchor timings")
    args = parser.parse_args()
    try:
        export(args)
    except (ValueError, OSError, KeyError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
