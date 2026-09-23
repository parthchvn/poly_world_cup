#!/usr/bin/env python3
"""Retain wallets with fewer than N observations across the entire tournament.

Input is the earlier per-contract subset, whose wallet_market_counts table
records counts for ALL wallets in the original captured tournament. Every
wallet below the tournament threshold must have survived the per-contract
filter. This script checks that all of those wallets' captured rows are present
before publishing a separate database. Existing datasets are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_wallet_activity import (
    CONTEXT_TABLES, _canonical, _identity, _reject_live_wal, _row_digest, _table_sql,
)

REQUIRED_TABLES = {
    "selected_trades", "wallet_market_counts", "metadata", "source_metadata",
    *CONTEXT_TABLES,
}


def _report(rows: list, label: str) -> dict:
    for key, serialized in rows:
        if key == "report":
            report = json.loads(serialized)
            if isinstance(report, dict):
                return report
            raise ValueError(f"{label} report must be an object")
    raise ValueError(f"Missing {label} report")


def filter_tournament_database(source: Path, output: Path, *, threshold: int = 20,
                               require_tournament_coverage: bool = False) -> dict:
    """Count all original observations by wallet, then retain complete wallets.

    Threshold is exclusive, combines both outcomes and both trade sides, and
    applies across all source contracts together. It is a retrospective activity
    criterion, not evidence that a wallet belongs to a human or market maker.
    """
    if type(threshold) is not int or threshold < 1:
        raise ValueError("threshold must be a positive integer")
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if not source.is_file():
        raise ValueError("Source must be a regular database file")
    if os.path.lexists(output) or output.resolve() == source:
        raise FileExistsError(f"Refusing to overwrite source or existing output: {output}")
    _reject_live_wal(source)
    before = _identity(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    db = None
    try:
        db = sqlite3.connect(temporary, uri=True)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("ATTACH DATABASE ? AS source", (source.as_uri() + "?mode=ro",))
        tables = {row[0] for row in db.execute("SELECT name FROM source.sqlite_master WHERE type='table'")}
        if not REQUIRED_TABLES.issubset(tables):
            raise ValueError("Source is missing required tables: " + ", ".join(sorted(REQUIRED_TABLES - tables)))
        db.execute("BEGIN")
        immediate_metadata = list(db.execute("SELECT key,value_json FROM source.metadata ORDER BY key"))
        original_metadata = list(db.execute("SELECT key,value_json FROM source.source_metadata ORDER BY key"))
        immediate_report = _report(immediate_metadata, "per-contract input")
        original_report = _report(original_metadata, "original tournament")
        prior_threshold = immediate_report.get("threshold_exclusive")
        if (immediate_report.get("filter_grouping") != ["wallet", "condition_id"]
                or type(prior_threshold) is not int or prior_threshold < threshold):
            raise ValueError("Input must be a per-contract subset with a threshold at least as large as the tournament threshold")
        if db.execute("""SELECT 1 FROM source.wallet_market_counts
            WHERE typeof(observation_count)<>'integer' OR observation_count<=0
                OR wallet IS NULL OR condition_id IS NULL LIMIT 1""").fetchone():
            raise ValueError("Invalid original wallet/market observation count")

        db.execute("""CREATE TABLE wallet_counts(
            wallet TEXT PRIMARY KEY, observed_count INTEGER NOT NULL CHECK(observed_count>0)
        ) WITHOUT ROWID""")
        db.execute("""INSERT INTO wallet_counts
            SELECT wallet,SUM(observation_count) FROM source.wallet_market_counts GROUP BY wallet""")
        source_wallets, source_rows, retained_wallets, retained_rows = db.execute("""
            SELECT COUNT(*),COALESCE(SUM(observed_count),0),
                COALESCE(SUM(observed_count<?),0),
                COALESCE(SUM(CASE WHEN observed_count<? THEN observed_count ELSE 0 END),0)
            FROM wallet_counts""", (threshold, threshold)).fetchone()
        source_markets = db.execute("SELECT COUNT(DISTINCT condition_id) FROM source.wallet_market_counts").fetchone()[0]
        source_fixtures = db.execute("SELECT COUNT(DISTINCT fixture_id) FROM source.selected_trades").fetchone()[0]
        input_rows = db.execute("SELECT COUNT(*) FROM source.selected_trades").fetchone()[0]
        for actual, expected, label in (
            (source_rows, original_report.get("observation_count"), "original observation"),
            (source_wallets, original_report.get("distinct_wallet_count"), "original wallet"),
            (source_rows, immediate_report.get("source_observations"), "input original observation"),
            (input_rows, immediate_report.get("retained_observations"), "input retained observation"),
        ):
            if expected is None or actual != expected:
                raise ValueError(f"{label} count disagrees with source metadata")
        if require_tournament_coverage and (
            source_markets != 312 or source_fixtures != 104
            or original_report.get("registry_condition_count") != 312
            or original_report.get("fixtures_with_observations") != 104
        ):
            raise ValueError("Production input must represent all 104 fixtures and 312 source contracts")

        copied_table_rows = {}
        for table in CONTEXT_TABLES:
            db.execute(_table_sql(db, table))
            db.execute(f'INSERT INTO "{table}" SELECT * FROM source."{table}"')
            copied_table_rows[table] = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        db.execute(_table_sql(db, "source_metadata"))
        db.execute("INSERT INTO source_metadata SELECT * FROM source.source_metadata")
        db.execute(_table_sql(db, "metadata", "immediate_input_metadata"))
        db.execute("INSERT INTO immediate_input_metadata SELECT * FROM source.metadata")
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)")
        db.execute(_table_sql(db, "selected_trades", "trades"))
        db.execute("""INSERT INTO trades SELECT t.* FROM source.selected_trades t NOT INDEXED
            JOIN wallet_counts w ON w.wallet=t.wallet WHERE w.observed_count<?""", (threshold,))
        db.execute("CREATE INDEX trades_wallet_time ON trades(wallet,query_us)")
        if db.execute("""SELECT 1 FROM wallet_counts w
            LEFT JOIN (SELECT wallet,COUNT(*) n FROM trades GROUP BY wallet) t USING(wallet)
            WHERE w.observed_count<? AND COALESCE(t.n,0)<>w.observed_count LIMIT 1""", (threshold,)).fetchone():
            raise ValueError("A qualifying wallet is missing rows or disagrees with its full captured-tournament count")
        if db.execute("""SELECT 1 FROM (
                SELECT wallet,condition_id,COUNT(*) n FROM trades GROUP BY wallet,condition_id
            ) t LEFT JOIN source.wallet_market_counts c USING(wallet,condition_id)
            WHERE c.observation_count IS NULL OR c.observation_count<>t.n LIMIT 1""").fetchone():
            raise ValueError("A qualifying wallet's contract counts disagree with its captured rows")
        if db.execute("SELECT COUNT(*) FROM trades").fetchone()[0] != retained_rows:
            raise ValueError("Retained observations disagree with original wallet counts")
        for field in ("context_state_id", "global_context_state_id"):
            if db.execute(f"""SELECT 1 FROM trades t LEFT JOIN context_states c
                ON t.{field}=c.context_state_id
                WHERE t.{field} IS NOT NULL AND c.context_state_id IS NULL LIMIT 1""").fetchone():
                raise ValueError(f"Retained observations have missing {field} references")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Filtered database has broken foreign key references")
        for sql in (
            "CREATE INDEX trades_fixture_time ON trades(fixture_id,query_us)",
            "CREATE INDEX trades_observation ON trades(observation_id)",
            "CREATE INDEX fixture_news_time ON fixture_news(fixture_id,eligible_us)",
            "CREATE INDEX global_news_time ON global_news(eligible_us)",
        ):
            db.execute(sql)
        retained_markets, retained_fixtures = db.execute(
            "SELECT COUNT(DISTINCT condition_id),COUNT(DISTINCT fixture_id) FROM trades"
        ).fetchone()
        report = {
            "format_version": 1, "schema_version": original_report.get("schema_version", 2),
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "output_table": "trades", "threshold_exclusive": threshold,
            "threshold_scope": "tournament", "filter_grouping": ["wallet"],
            "filter_predicate": f"SUM(original_wallet_market_observation_count) < {threshold}",
            "market_semantics": "all captured World Cup primary result contracts together; both outcomes and sides combined",
            "count_semantics": "saved API observation rows, not unique canonical fills or transactions",
            "source_observations": source_rows, "immediate_input_observations": input_rows,
            "observation_count": retained_rows, "retained_observations": retained_rows,
            "excluded_observations": source_rows - retained_rows,
            "source_wallets": source_wallets, "retained_wallets": retained_wallets,
            "distinct_wallet_count": retained_wallets, "excluded_wallets": source_wallets - retained_wallets,
            "source_markets": source_markets, "retained_markets": retained_markets,
            "source_fixtures": source_fixtures, "retained_fixtures": retained_fixtures,
            "fixtures_with_observations": retained_fixtures,
            "tournament_coverage_required": require_tournament_coverage,
            "copied_table_rows": copied_table_rows,
            "source_database_identity": before,
            "source_metadata_sha256": _row_digest(original_metadata),
            "immediate_input_metadata_sha256": _row_digest(immediate_metadata),
            "source_pages_metadata_sha256": _row_digest(db.execute(
                "SELECT source_page_id,path,uncompressed_sha256,row_count FROM source_pages ORDER BY source_page_id")),
            "wallet_counts_sha256": _row_digest(db.execute("SELECT wallet,observed_count FROM wallet_counts ORDER BY wallet")),
            "fingerprint_semantics": "ordered metadata and aggregate count hashes plus filesystem identity; not a whole database hash",
            "source_report": original_report,
            "partial_api_collection": original_report.get("partial_api_collection", True),
            "api_traversals_exhausted": original_report.get("api_traversals_exhausted"),
            "registry_condition_count": original_report.get("registry_condition_count"),
            "source_completeness_certified": False, "training_ready": False,
            "market_maker_status_verified": False, "human_identity_verified": False,
            "selection_is_retrospective": True, "activity_count_feature_eligible": False,
            "incomplete_histories_can_undercount": True,
            "full_wallet_history_included": True,
            "wallet_history_scope": "captured_tournament_snapshot_only",
            "wallet_history_instruction": "For every retained wallet, trades contains every observation counted in the original captured tournament snapshot. Missing upstream observations and activity outside this tournament remain unknown.",
            "complete_retained_wallet_counts_verified": True,
            "news_context_preserved": True, "source_database_modified": False,
        }
        db.execute("INSERT INTO metadata VALUES('report',?)", (_canonical(report),))
        db.execute("INSERT INTO metadata VALUES('registry_feature_eligible','false')")
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Filtered database integrity check failed")
        db.close()
        db = None
        _reject_live_wal(source)
        if _identity(source) != before:
            raise ValueError("Source changed while filtering; output was not published")
        os.link(temporary, output)
        temporary.unlink()
        return report
    finally:
        if db is not None:
            db.close()
        for path in (temporary, Path(str(temporary) + "-journal"),
                     Path(str(temporary) + "-wal"), Path(str(temporary) + "-shm")):
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=20,
                        help="Retain wallets with fewer observations across ALL tournament markets (default: 20)")
    parser.add_argument("--require-tournament-coverage", action="store_true",
                        help="Require all 104 fixtures and 312 contracts in the source; report retained coverage separately")
    parser.add_argument("--report", type=Path, help="Write a separate JSON report; never overwrite existing files")
    args = parser.parse_args()
    if args.report is not None and (
        os.path.lexists(args.report) or args.report.resolve() in {args.database.resolve(), args.output.resolve()}
    ):
        parser.error("Refusing to overwrite an existing report, source, or database output")
    report = filter_tournament_database(args.database, args.output, threshold=args.threshold,
                                        require_tournament_coverage=args.require_tournament_coverage)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
