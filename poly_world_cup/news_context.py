"""Replace context in a new database while preserving every trade observation."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .attribution import build_attribution_index, _GLOBAL_SCOPE
from .sft import _identity, _json, _reject_wal, _sha


def replace_news_context(source: Path, output: Path, registry: dict, records: list,
                         *, contract_records=None, progress=None) -> dict:
    """Copy an immutable cohort and rebuild all context pointers from evidence.

    Target fields, wallet counts, and source-page provenance are copied exactly.
    Context prefix IDs follow the same construction as build_attribution_index.
    The operation refuses overwrite and publishes only after foreign-key checks.
    """
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if os.path.lexists(output) or output.resolve() == source:
        raise FileExistsError("Refusing to overwrite a source or existing context database")
    _reject_wal(source)
    before = _identity(source)
    source_sha = _sha(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    announce = progress or (lambda message: None)
    with tempfile.TemporaryDirectory(prefix=".news-context-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        news_db = temporary / "news.sqlite"
        context_report = build_attribution_index(registry=registry, trade_pages=[],
            news_records=records, output_path=news_db)
        announce(f"Validated {len(records):,} news versions; copying source cohort")
        stage = temporary / "cohort.sqlite"
        original = sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)
        db = sqlite3.connect(stage, uri=True)
        try:
            original.backup(db)
            original.close()
            original = None
            db.execute("PRAGMA cache_size=-65536")
            db.execute("ATTACH DATABASE ? AS refreshed", (news_db.as_uri() + "?mode=ro",))
            count_before = db.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            with db:
                if contract_records is not None:
                    from .historical_contracts import validate_contract_record
                    db.execute("CREATE TABLE IF NOT EXISTS contract_evidence(condition_id TEXT PRIMARY KEY,record_json TEXT NOT NULL)")
                    db.execute("DELETE FROM contract_evidence")
                    allowed = {row["condition_id"]: row for row in registry["contracts"]}
                    for record in contract_records:
                        validate_contract_record(record)
                        if record["condition_id"] not in allowed:
                            raise ValueError("Contract evidence does not belong to the registry")
                        contract = allowed[record["condition_id"]]
                        tokens = {token["outcome"]: token["token_id"] for token in contract["tokens"]}
                        if (record["fixture_id"] != contract["fixture_id"]
                                or record["yes_token_id"] != tokens.get("Yes")
                                or record["no_token_id"] != tokens.get("No")):
                            raise ValueError("Historical contract evidence disagrees with fixture/token mapping")
                        db.execute("INSERT INTO contract_evidence VALUES(?,?)", (record["condition_id"], _json(record)))
                for table in ("fixture_news", "global_news", "context_states", "news"):
                    db.execute(f"DELETE FROM {table}")
                for table in ("news", "fixture_news", "global_news", "context_states"):
                    db.execute(f"INSERT INTO {table} SELECT * FROM refreshed.{table}")
                times, states, candidates = defaultdict(list), defaultdict(list), defaultdict(list)
                for state, fixture, count, newest in db.execute("""SELECT context_state_id,
                        fixture_id,event_count,newest_event_us FROM context_states
                        ORDER BY fixture_id,event_count"""):
                    states[fixture].append(state)
                    if count:
                        times[fixture].append(newest)
                for fixture, published in db.execute("""SELECT f.fixture_id,n.published_us
                        FROM fixture_news f JOIN news n USING(news_id)
                        WHERE f.eligible_us IS NULL AND n.published_us IS NOT NULL
                        ORDER BY f.fixture_id,n.published_us"""):
                    candidates[fixture].append(published)
                cursor = db.execute("SELECT trade_row_id,fixture_id,query_us FROM trades ORDER BY trade_row_id")
                updated = 0
                while rows := cursor.fetchmany(20_000):
                    changes = []
                    for identity, fixture, query in rows:
                        if fixture not in states:
                            raise ValueError("Trade fixture absent from refreshed context registry")
                        prefix = bisect_left(times[fixture], query)
                        global_prefix = bisect_left(times[_GLOBAL_SCOPE], query)
                        changes.append((states[fixture][prefix], prefix,
                            bisect_left(candidates[fixture], query), states[_GLOBAL_SCOPE][global_prefix],
                            global_prefix, identity))
                    db.executemany("""UPDATE trades SET context_state_id=?,eligible_context_event_count=?,
                        retrospective_candidate_count=?,global_context_state_id=?,
                        global_eligible_context_event_count=? WHERE trade_row_id=?""", changes)
                    updated += len(rows)
                    if updated % 100_000 == 0:
                        announce(f"Reattributed {updated:,}/{count_before:,} observations")
                if updated != count_before:
                    raise ValueError("Context refresh changed observation accounting")
                report = {
                    "source_database_sha256": source_sha, "source_database_modified": False,
                    "news_records_sha256": hashlib.sha256(_json(records).encode()).hexdigest(),
                    "observation_count": updated, "trade_labels_modified": False,
                    "news_catalog_report": context_report,
                    "verified_contract_records": None if contract_records is None else len(contract_records),
                }
                db.execute("INSERT OR REPLACE INTO metadata VALUES('news_refresh',?)", (_json(report),))
            if db.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("Refreshed database has broken foreign keys")
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Refreshed database failed SQLite integrity check")
            db.close()
            db = None
            _reject_wal(source)
            if _identity(source) != before:
                raise ValueError("Source changed during context refresh")
            # Same-filesystem hard-link publication cannot replace a destination
            # created after our initial existence check.
            os.link(stage, output)
            stage.unlink()
            announce(f"Published refreshed context for {updated:,} observations")
            return report
        finally:
            if original is not None:
                original.close()
            if db is not None:
                db.close()
