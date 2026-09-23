#!/usr/bin/env python3
"""Create a separate low-activity target dataset without editing its source.

Counts are retrospective API observation counts per (wallet, condition_id).
The original database remains the source of full prior wallet execution history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile


CONTEXT_TABLES = (
    "news", "fixture_news", "global_news", "context_states", "source_pages",
)
REQUIRED_TABLES = {"trades", "metadata", *CONTEXT_TABLES}


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev, "inode": stat.st_ino,
    }


def _reject_live_wal(path: Path) -> None:
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source has a nonempty WAL; stop writers and checkpoint it before filtering")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _row_digest(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((_canonical(list(row)) + "\n").encode())
    return digest.hexdigest()


def _table_sql(db: sqlite3.Connection, original: str, renamed: str | None = None) -> str:
    row = db.execute(
        "SELECT sql FROM source.sqlite_master WHERE type='table' AND name=?", (original,),
    ).fetchone()
    if not row or not row[0]:
        raise ValueError(f"Source is missing a regular table: {original}")
    sql = row[0]
    if renamed is not None:
        # Table names above are fixed program constants, never CLI SQL fragments.
        pattern = r'^\s*CREATE\s+TABLE\s+(?:"' + original + r'"|`' + original + r'`|\[' + original + r'\]|' + original + r')(?=\s|\()'
        sql, count = re.subn(pattern, 'CREATE TABLE "' + renamed + '"', sql, count=1, flags=re.I)
        if count != 1:
            raise ValueError(f"Unsupported table declaration: {original}")
    return sql


def filter_database(source: Path, output: Path, *, threshold: int = 20) -> dict:
    """Retain all observations for wallet/market pairs below ``threshold``.

    ``selected_trades`` deliberately differs from the original ``trades`` table:
    a reader must not accidentally substitute selected targets for full actor
    history. News and provenance tables are copied without changing their data.
    Source identity and metadata hashes bind this derived snapshot cheaply; they
    are explicitly not a SHA-256 hash of the entire source database.
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
            raise ValueError("Source is missing attribution tables: " + ", ".join(sorted(REQUIRED_TABLES - tables)))
        db.execute("BEGIN")
        source_metadata = list(db.execute("SELECT key,value_json FROM source.metadata ORDER BY key"))
        source_metadata_sha256 = _row_digest(source_metadata)
        source_pages_metadata_sha256 = _row_digest(db.execute(
            "SELECT source_page_id,path,uncompressed_sha256,row_count FROM source.source_pages ORDER BY source_page_id",
        ))
        source_report = {}
        for key, serialized in source_metadata:
            if key == "report":
                source_report = json.loads(serialized)
                if not isinstance(source_report, dict):
                    raise ValueError("Source metadata report must be an object")

        copied_table_rows = {}
        for table in CONTEXT_TABLES:
            db.execute(_table_sql(db, table))
            db.execute(f'INSERT INTO "{table}" SELECT * FROM source."{table}"')
            copied_table_rows[table] = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        db.execute(_table_sql(db, "metadata", "source_metadata"))
        db.execute("INSERT INTO source_metadata SELECT * FROM source.metadata")
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)")
        db.execute("""CREATE TABLE wallet_market_counts(
            wallet TEXT NOT NULL, condition_id TEXT NOT NULL,
            observation_count INTEGER NOT NULL CHECK(observation_count > 0),
            PRIMARY KEY(wallet,condition_id)
        ) WITHOUT ROWID""")
        db.execute("""INSERT INTO wallet_market_counts
            SELECT wallet,condition_id,COUNT(*) FROM source.trades NOT INDEXED
            GROUP BY wallet,condition_id""")
        source_pairs, source_rows, retained_pairs, retained_rows = db.execute("""
            SELECT COUNT(*),COALESCE(SUM(observation_count),0),
                COALESCE(SUM(observation_count < ?),0),
                COALESCE(SUM(CASE WHEN observation_count < ? THEN observation_count ELSE 0 END),0)
            FROM wallet_market_counts
        """, (threshold, threshold)).fetchone()

        db.execute(_table_sql(db, "trades", "selected_trades"))
        db.execute("""INSERT INTO selected_trades
            SELECT t.* FROM source.trades AS t NOT INDEXED
            JOIN wallet_market_counts AS c ON c.wallet=t.wallet AND c.condition_id=t.condition_id
            WHERE c.observation_count < ?""", (threshold,))
        if db.execute("SELECT COUNT(*) FROM selected_trades").fetchone()[0] != retained_rows:
            raise ValueError("Retained observations disagree with aggregated counts")
        if db.execute("""SELECT 1 FROM selected_trades t JOIN wallet_market_counts c
            ON c.wallet=t.wallet AND c.condition_id=t.condition_id
            WHERE c.observation_count >= ? LIMIT 1""", (threshold,)).fetchone():
            raise ValueError("A retained pair violates the exclusive threshold")
        for field in ("context_state_id", "global_context_state_id"):
            if db.execute(f"""SELECT 1 FROM selected_trades t LEFT JOIN context_states c
                ON t.{field}=c.context_state_id
                WHERE t.{field} IS NOT NULL AND c.context_state_id IS NULL LIMIT 1""").fetchone():
                raise ValueError(f"Retained observations have missing {field} references")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Filtered database has broken foreign key references")

        for sql in (
            "CREATE INDEX selected_wallet_time ON selected_trades(wallet,query_us)",
            "CREATE INDEX selected_fixture_time ON selected_trades(fixture_id,query_us)",
            "CREATE INDEX selected_observation ON selected_trades(observation_id)",
            "CREATE INDEX fixture_news_time ON fixture_news(fixture_id,eligible_us)",
            "CREATE INDEX global_news_time ON global_news(eligible_us)",
        ):
            db.execute(sql)
        source_wallets, retained_wallets, source_markets, retained_markets = db.execute("""
            SELECT COUNT(DISTINCT wallet),
                COUNT(DISTINCT CASE WHEN observation_count < ? THEN wallet END),
                COUNT(DISTINCT condition_id),
                COUNT(DISTINCT CASE WHEN observation_count < ? THEN condition_id END)
            FROM wallet_market_counts
        """, (threshold, threshold)).fetchone()
        report = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "output_table": "selected_trades",
            "threshold_exclusive": threshold,
            "filter_grouping": ["wallet", "condition_id"],
            "filter_predicate": f"COUNT(*) < {threshold}",
            "market_semantics": "Polymarket binary contract condition_id; all outcomes and sides combined",
            "count_semantics": "saved API observation rows, not unique canonical fills or transactions",
            "source_observations": source_rows,
            "retained_observations": retained_rows,
            "excluded_observations": source_rows - retained_rows,
            "source_wallet_market_pairs": source_pairs,
            "retained_wallet_market_pairs": retained_pairs,
            "excluded_wallet_market_pairs": source_pairs - retained_pairs,
            "source_wallets": source_wallets,
            "retained_wallets": retained_wallets,
            "excluded_wallets": source_wallets - retained_wallets,
            "source_markets": source_markets,
            "retained_markets": retained_markets,
            "excluded_markets": source_markets - retained_markets,
            "excluded_wallet_market_count_semantics": "wallets or markets with no retained target observations",
            "retained_fixtures": db.execute("SELECT COUNT(DISTINCT fixture_id) FROM selected_trades").fetchone()[0],
            "copied_table_rows": copied_table_rows,
            "source_database_identity": before,
            "source_metadata_sha256": source_metadata_sha256,
            "source_pages_metadata_sha256": source_pages_metadata_sha256,
            "fingerprint_semantics": "ordered metadata-row hashes and filesystem identity; not a full database file hash",
            "source_report": source_report,
            "partial_api_collection": source_report.get("partial_api_collection", True),
            "source_completeness_certified": False,
            "training_ready": False,
            "market_maker_status_verified": False,
            "human_identity_verified": False,
            "selection_is_retrospective": True,
            "activity_count_feature_eligible": False,
            "incomplete_histories_can_undercount": True,
            "full_wallet_history_included": False,
            "wallet_history_source": str(source),
            "wallet_history_scope": source_report.get("wallet_history_scope", "tournament_only"),
            "wallet_history_instruction": "Resolve target trade_row_id against the unchanged original database for full prior wallet execution history; do not use selected_trades as history.",
            "news_context_preserved": True,
            "source_database_modified": False,
        }
        db.execute("INSERT INTO metadata VALUES('report',?)", (_canonical(report),))
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Filtered database integrity check failed")
        db.close()
        db = None
        _reject_live_wal(source)
        if _identity(source) != before:
            raise ValueError("Source database changed while filtering; output was not published")
        # Hard linking atomically publishes a complete file and, unlike replace,
        # cannot overwrite an output another process created in the meantime.
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
                        help="Keep wallet/market pairs with fewer observations than this number (default: 20)")
    parser.add_argument("--report", type=Path, help="Also create a JSON report; existing files are never overwritten")
    args = parser.parse_args()
    if args.report is not None:
        if os.path.lexists(args.report) or args.report.resolve() in {args.database.resolve(), args.output.resolve()}:
            parser.error("Refusing to overwrite an existing report, source, or database output")
    report = filter_database(args.database, args.output, threshold=args.threshold)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
