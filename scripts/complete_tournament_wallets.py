#!/usr/bin/env python3
"""Replace unfinished contract captures and rebuild the tournament-wide cohort.

The earlier per-contract subset retains full wallet/contract counts. All wallets
that can qualify for the global threshold have complete retained histories in
that subset. Fresh complete captures replace, rather than add to, unfinished
contracts. We require monotone per-wallet counts and multiset containment of
all surviving observations from those contracts before accepting replacement.
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
from scripts.filter_tournament_wallets import _report

SEMANTIC_FIELDS = ("wallet", "condition_id", "token_id", "side", "shares", "price",
                   "query_us", "transaction_hash")


def complete_tournament_database(source: Path, fresh: Path, output: Path, *,
        original_checkpoint: dict, original_provenance: dict, fresh_collection: dict, fresh_provenance: dict,
        threshold: int = 20, require_tournament_coverage: bool = False) -> dict:
    if type(threshold) is not int or threshold < 1:
        raise ValueError("threshold must be a positive integer")
    source, fresh = Path(source).resolve(strict=True), Path(fresh).resolve(strict=True)
    output = Path(output).absolute()
    if os.path.lexists(output) or output.resolve() in {source, fresh}:
        raise FileExistsError("Refusing to replace existing input or output")
    for path in (source, fresh):
        if not path.is_file():
            raise ValueError("Inputs must be regular database files")
        _reject_live_wal(path)
    before = {str(path): _identity(path) for path in (source, fresh)}
    replacement = original_checkpoint.get("unfinished_condition_ids")
    if (not isinstance(replacement, list) or not replacement
            or len(set(replacement)) != len(replacement)
            or any(not isinstance(x, str) for x in replacement)):
        raise ValueError("Original checkpoint must identify unfinished conditions exactly")
    conditions = fresh_collection.get("conditions", [])
    if ({row.get("condition_id") for row in conditions} != set(replacement)
            or len(conditions) != len(replacement)
            or any(row.get("status") != "exhausted" or row.get("integrity_validated") is not True
                   for row in conditions)
            or fresh_collection.get("status") != "api_exhausted"
            or fresh_collection.get("requested_minimum_size_tokens") != "0.000001"
            or fresh_collection.get("taker_only") is not False
            or original_checkpoint.get("trades",{}).get("requested_minimum_size_tokens") != "0.000001"
            or original_checkpoint.get("trades",{}).get("taker_only") is not False):
        raise ValueError("Fresh collection must exhaust exactly the unfinished condition set at threshold 0.000001")
    if (fresh_provenance.get("raw_provenance_verified") is not True
            or fresh_provenance.get("verified_condition_count") != len(replacement)
            or fresh_provenance.get("api_exhausted_conditions") != len(replacement)
            or fresh_provenance.get("condition_selection") != "explicit"
            or fresh_provenance.get("requested_condition_count") != len(replacement)):
        raise ValueError("Fresh normalized pages require complete raw provenance replay")
    old_manifest_hashes = original_provenance.get("condition_manifest_sha256", {})
    new_manifest_hashes = fresh_provenance.get("condition_manifest_sha256", {})
    if (original_provenance.get("raw_provenance_verified") is not True
            or not isinstance(old_manifest_hashes, dict)
            or not set(replacement).issubset(old_manifest_hashes)
            or set(new_manifest_hashes) != set(replacement)
            or new_manifest_hashes != fresh_collection.get("condition_manifest_sha256")
            or any(not isinstance(value,str) or len(value)!=64
                   for value in [*old_manifest_hashes.values(),*new_manifest_hashes.values()])):
        raise ValueError("Original and replacement contract manifest identities require verified provenance")
    if original_checkpoint.get("integrity", {}).get("all_saved_pages_replayed_from_raw_captures") is not True:
        raise ValueError("Original capture lacks inherited provenance verification")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    db = None
    try:
        db = sqlite3.connect(temporary, uri=True)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("ATTACH DATABASE ? AS source", (source.as_uri()+"?mode=ro",))
        db.execute("ATTACH DATABASE ? AS fresh", (fresh.as_uri()+"?mode=ro",))
        db.execute("BEGIN")
        original_metadata = list(db.execute("SELECT key,value_json FROM source.source_metadata ORDER BY key"))
        input_metadata = list(db.execute("SELECT key,value_json FROM source.metadata ORDER BY key"))
        fresh_metadata = list(db.execute("SELECT key,value_json FROM fresh.metadata ORDER BY key"))
        original_report = _report(original_metadata, "original")
        input_report = _report(input_metadata, "per-contract subset")
        fresh_report = _report(fresh_metadata, "fresh attribution")
        if (input_report.get("filter_grouping") != ["wallet", "condition_id"]
                or type(input_report.get("threshold_exclusive")) is not int
                or input_report["threshold_exclusive"] < threshold):
            raise ValueError("Input subset cannot guarantee complete histories at this threshold")
        if (fresh_report.get("expected_page_hashes_verified") is not True
                or fresh_report.get("expected_page_row_counts_verified") is not True):
            raise ValueError("Fresh attribution must be bound to verified page hashes and counts")
        db.execute("CREATE TEMP TABLE replacement(condition_id TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO replacement VALUES(?)", ((x,) for x in replacement))
        actual_fresh = {row[0] for row in db.execute("SELECT DISTINCT condition_id FROM fresh.trades")}
        if actual_fresh != set(replacement):
            raise ValueError("Fresh attribution condition set differs from replacement set")
        if db.execute("SELECT 1 FROM source.wallet_market_counts WHERE typeof(observation_count)<>'integer' OR observation_count<=0 OR wallet IS NULL OR condition_id IS NULL LIMIT 1").fetchone():
            raise ValueError("Original wallet/contract count ledger has invalid counts or identities")
        old_rows, old_wallets = db.execute("SELECT SUM(observation_count),COUNT(DISTINCT wallet) FROM source.wallet_market_counts").fetchone()
        old_condition_ids = {row[0] for row in db.execute("SELECT DISTINCT condition_id FROM source.wallet_market_counts")}
        old_conditions = len(old_condition_ids)
        if set(old_manifest_hashes) != old_condition_ids:
            raise ValueError("Original manifest identity ledger disagrees with captured contract universe")
        old_fixture_count = db.execute("SELECT COUNT(DISTINCT fixture_id) FROM source.selected_trades").fetchone()[0]
        if (old_rows != original_report.get("observation_count")
                or old_wallets != original_report.get("distinct_wallet_count")
                or old_rows != original_checkpoint.get("trades", {}).get("saved_observations")
                or old_conditions != original_checkpoint.get("scope", {}).get("contracts")):
            raise ValueError("Original count ledger disagrees with recorded capture")
        prior_exhausted = original_checkpoint.get("trades", {}).get("contract_status_counts", {}).get("exhausted")
        if (type(prior_exhausted) is not int or prior_exhausted + len(replacement) != old_conditions
                or original_provenance.get("api_exhausted_conditions") != prior_exhausted
                or original_provenance.get("verified_condition_count") != old_conditions
                or original_provenance.get("verified_observation_count") != old_rows):
            raise ValueError("Replacement set does not finish the original contract universe")
        if require_tournament_coverage and (old_conditions != 312 or old_fixture_count != 104):
            raise ValueError("Production build requires all 104 fixtures and 312 contracts")
        actual_fresh_rows = db.execute("SELECT COUNT(*) FROM fresh.trades").fetchone()[0]
        actual_fresh_pages = db.execute("SELECT COUNT(*) FROM fresh.source_pages").fetchone()[0]
        verified_pages = fresh_provenance.get("verified_pages", [])
        if (fresh_provenance.get("verified_condition_ids") != sorted(replacement)
                or not isinstance(verified_pages,list) or len(verified_pages)!=actual_fresh_pages):
            raise ValueError("Fresh provenance must identify all verified conditions and pages")
        db.execute("CREATE TEMP TABLE verified_pages(path TEXT PRIMARY KEY,uncompressed_sha256 TEXT NOT NULL,row_count INTEGER NOT NULL,condition_id TEXT NOT NULL)")
        try:
            db.executemany("INSERT INTO verified_pages VALUES(?,?,?,?)", [(row["path"],row["uncompressed_sha256"],row["row_count"],row["condition_id"]) for row in verified_pages])
        except (KeyError,TypeError,sqlite3.IntegrityError) as error:
            raise ValueError("Invalid fresh provenance page ledger") from error
        if (db.execute("SELECT path,uncompressed_sha256,row_count FROM fresh.source_pages EXCEPT SELECT path,uncompressed_sha256,row_count FROM verified_pages LIMIT 1").fetchone()
                or db.execute("SELECT path,uncompressed_sha256,row_count FROM verified_pages EXCEPT SELECT path,uncompressed_sha256,row_count FROM fresh.source_pages LIMIT 1").fetchone()
                or db.execute("SELECT 1 FROM fresh.trades t JOIN fresh.source_pages p USING(source_page_id) JOIN verified_pages v USING(path) WHERE t.condition_id<>v.condition_id LIMIT 1").fetchone()):
            raise ValueError("Fresh attributed pages differ from raw-replayed provenance ledger")
        if (actual_fresh_rows != fresh_collection.get("validated_observation_count")
                or actual_fresh_rows != fresh_provenance.get("verified_observation_count")
                or actual_fresh_rows != fresh_report.get("observation_count")
                or actual_fresh_pages != fresh_provenance.get("verified_raw_page_count")):
            raise ValueError("Fresh row/page counts disagree with collection and provenance")
        if db.execute("SELECT 1 FROM fresh.trades WHERE token_mapping_status<>'exact_current_registry_token' OR fixture_id IS NULL LIMIT 1").fetchone():
            raise ValueError("Fresh trades have unresolved registry mappings")
        if db.execute("SELECT 1 FROM fresh.trades GROUP BY observation_id HAVING COUNT(*)>1 LIMIT 1").fetchone():
            raise ValueError("Fresh capture contains duplicate observation identities")
        db.execute("""CREATE TEMP TABLE fresh_counts AS
            SELECT wallet,condition_id,COUNT(*) observation_count FROM fresh.trades GROUP BY wallet,condition_id""")
        db.execute("CREATE UNIQUE INDEX fresh_counts_key ON fresh_counts(wallet,condition_id)")
        mismatch = db.execute("""SELECT s.wallet,s.condition_id,s.observation_count,COALESCE(f.observation_count,0)
            FROM source.wallet_market_counts s JOIN replacement r USING(condition_id)
            LEFT JOIN fresh_counts f USING(wallet,condition_id)
            WHERE s.observation_count>COALESCE(f.observation_count,0) LIMIT 1""").fetchone()
        if mismatch:
            raise ValueError("Fresh capture loses observations from an old wallet/contract count: " + str(mismatch))
        keys = ",".join(SEMANTIC_FIELDS)
        join = " AND ".join("s."+key+"=f."+key for key in SEMANTIC_FIELDS)
        db.execute(f"CREATE TEMP TABLE old_multiset AS SELECT {keys},COUNT(*) n FROM source.selected_trades JOIN replacement USING(condition_id) GROUP BY {keys}")
        db.execute(f"CREATE TEMP TABLE fresh_multiset AS SELECT {keys},COUNT(*) n FROM fresh.trades GROUP BY {keys}")
        db.execute(f"CREATE UNIQUE INDEX fresh_multiset_key ON fresh_multiset({keys})")
        if db.execute(f"SELECT 1 FROM old_multiset s LEFT JOIN fresh_multiset f ON {join} WHERE s.n>COALESCE(f.n,0) LIMIT 1").fetchone():
            raise ValueError("Fresh capture does not contain the full multiset of surviving old observations")
        retained_comparison_rows = db.execute("SELECT COALESCE(SUM(n),0) FROM old_multiset").fetchone()[0]
        # Context construction is deterministic. Identical tables permit safe
        # reuse of context IDs; news upgrades are a separate subsequent stage.
        for table in CONTEXT_TABLES:
            if table == "source_pages":
                continue
            if (db.execute(f'SELECT * FROM source."{table}" EXCEPT SELECT * FROM fresh."{table}" LIMIT 1').fetchone()
                    or db.execute(f'SELECT * FROM fresh."{table}" EXCEPT SELECT * FROM source."{table}" LIMIT 1').fetchone()):
                raise ValueError("Fresh context tables differ from original: " + table)
        db.execute(_table_sql(db, "wallet_market_counts"))
        db.execute("""INSERT INTO wallet_market_counts SELECT * FROM source.wallet_market_counts
            WHERE condition_id NOT IN (SELECT condition_id FROM replacement)""")
        db.execute("INSERT INTO wallet_market_counts SELECT * FROM fresh_counts")
        db.execute("CREATE TABLE wallet_counts(wallet TEXT PRIMARY KEY,observed_count INTEGER NOT NULL CHECK(observed_count>0)) WITHOUT ROWID")
        db.execute("INSERT INTO wallet_counts SELECT wallet,SUM(observation_count) FROM wallet_market_counts GROUP BY wallet")
        for table in CONTEXT_TABLES:
            db.execute(_table_sql(db, table))
            db.execute(f'INSERT INTO "{table}" SELECT * FROM source."{table}"')
        page_offset = db.execute("SELECT COALESCE(MAX(source_page_id),0) FROM source_pages").fetchone()[0]
        db.execute("INSERT INTO source_pages SELECT source_page_id+?,path,uncompressed_sha256,row_count FROM fresh.source_pages", (page_offset,))
        db.execute(_table_sql(db, "source_metadata"))
        db.execute("INSERT INTO source_metadata SELECT * FROM source.source_metadata")
        db.execute(_table_sql(db, "metadata", "immediate_input_metadata"))
        db.execute("INSERT INTO immediate_input_metadata SELECT * FROM source.metadata")
        db.execute(_table_sql(db, "metadata", "fresh_input_metadata"))
        db.execute("INSERT INTO fresh_input_metadata SELECT * FROM fresh.metadata")
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)")
        db.execute(_table_sql(db, "selected_trades", "trades"))
        db.execute("""INSERT INTO trades SELECT t.* FROM source.selected_trades t JOIN wallet_counts w USING(wallet)
            WHERE w.observed_count<? AND t.condition_id NOT IN (SELECT condition_id FROM replacement)""", (threshold,))
        row_offset = db.execute("SELECT COALESCE(MAX(trade_row_id),0) FROM source.selected_trades").fetchone()[0]
        columns = [row[1] for row in db.execute("PRAGMA table_info(trades)")]
        expressions = ["t.trade_row_id+?" if col == "trade_row_id" else "t.source_page_id+?" if col == "source_page_id" else "t."+col for col in columns]
        db.execute("INSERT INTO trades SELECT " + ",".join(expressions) + " FROM fresh.trades t JOIN wallet_counts w USING(wallet) WHERE w.observed_count<?", (row_offset,page_offset,threshold))
        db.execute("CREATE INDEX trades_wallet_time ON trades(wallet,query_us)")
        if db.execute("""SELECT 1 FROM wallet_counts w LEFT JOIN
            (SELECT wallet,COUNT(*) n FROM trades GROUP BY wallet) t USING(wallet)
            WHERE w.observed_count<? AND COALESCE(t.n,0)<>w.observed_count LIMIT 1""", (threshold,)).fetchone():
            raise ValueError("A qualifying wallet is missing captured observations")
        if db.execute("""SELECT 1 FROM (SELECT wallet,condition_id,COUNT(*) n FROM trades GROUP BY wallet,condition_id) t
            LEFT JOIN wallet_market_counts c USING(wallet,condition_id)
            WHERE c.observation_count IS NULL OR c.observation_count<>t.n LIMIT 1""").fetchone():
            raise ValueError("Retained wallet/contract counts disagree with captured observations")
        if db.execute("SELECT 1 FROM trades GROUP BY observation_id HAVING COUNT(*)>1 LIMIT 1").fetchone():
            raise ValueError("Merged capture contains duplicate observation identities")
        for column in ("context_state_id", "global_context_state_id"):
            if db.execute(f"SELECT 1 FROM trades t LEFT JOIN context_states c ON t.{column}=c.context_state_id WHERE t.{column} IS NOT NULL AND c.context_state_id IS NULL LIMIT 1").fetchone():
                raise ValueError("Merged capture has missing context reference")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Merged capture has missing source/news references")
        for sql in ("CREATE INDEX trades_fixture_time ON trades(fixture_id,query_us)",
                    "CREATE INDEX trades_observation ON trades(observation_id)",
                    "CREATE INDEX fixture_news_time ON fixture_news(fixture_id,eligible_us)",
                    "CREATE INDEX global_news_time ON global_news(eligible_us)"):
            db.execute(sql)
        total_wallets,total_rows,kept_wallets,kept_rows = db.execute("""SELECT COUNT(*),SUM(observed_count),
            SUM(observed_count<?),SUM(CASE WHEN observed_count<? THEN observed_count ELSE 0 END) FROM wallet_counts""", (threshold,threshold)).fetchone()
        kept_fixtures,kept_conditions = db.execute("SELECT COUNT(DISTINCT fixture_id),COUNT(DISTINCT condition_id) FROM trades").fetchone()
        kept_first,kept_last = db.execute("SELECT MIN(block_timestamp),MAX(block_timestamp) FROM trades").fetchone()
        fresh_first,fresh_last = db.execute("SELECT MIN(block_timestamp),MAX(block_timestamp) FROM fresh.trades").fetchone()
        report = {
            "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "output_table": "trades", "threshold_exclusive": threshold, "threshold_scope": "tournament",
            "filter_grouping": ["wallet"], "source_observations": total_rows, "source_wallets": total_wallets,
            "observation_count": kept_rows or 0, "retained_observations": kept_rows or 0,
            "retained_wallets": kept_wallets or 0, "distinct_wallet_count": kept_wallets or 0,
            "excluded_observations": total_rows-(kept_rows or 0), "excluded_wallets": total_wallets-(kept_wallets or 0),
            "source_markets": old_conditions, "source_fixtures": old_fixture_count,
            "registry_condition_count": old_conditions, "retained_markets": kept_conditions,
            "retained_fixtures": kept_fixtures, "fixtures_with_observations": kept_fixtures,
            "partial_api_collection": False, "api_traversals_exhausted": old_conditions,
            "old_exhausted_contracts_inherited": prior_exhausted,
            "replacement_contract_count": len(replacement), "replacement_condition_ids": sorted(replacement),
            "fresh_capture_observations": actual_fresh_rows, "net_added_observations": total_rows-old_rows,
            "retained_execution_window": {"first": kept_first, "last": kept_last},
            "fresh_capture_execution_window": {"first": fresh_first, "last": fresh_last},
            "inherited_original_capture_window": {
                "first": original_checkpoint.get("trades",{}).get("earliest_saved_block_timestamp"),
                "last": original_checkpoint.get("trades",{}).get("latest_saved_block_timestamp")},
            "old_retained_observations_checked_for_multiset_containment": retained_comparison_rows,
            "old_wallet_contract_counts_monotone_verified": True,
            "old_retained_multiset_containment_verified": True,
            "complete_retained_wallet_counts_verified": True, "full_wallet_history_included": True,
            "wallet_history_scope": "captured_tournament_snapshot_only",
            "capture_completion_scope": "all_filtered_API_traversals_exhausted_not_canonical_chain_completeness",
            "query_time_semantics": "execution_block_timestamp_proxy_not_decision_time",
            "count_semantics": "wallet-side_API_observations_not_unique_canonical_fills_or_orders",
            "requested_minimum_size_tokens": "0.000001", "taker_only": False,
            "inherited_capture_provenance": "inherited_verified_capture_page_hashes_raw_files_not_available_locally",
            "fresh_replacement_provenance": "raw_response_replay_verified_this_build",
            "source_pages_include_superseded_capture_references": True,
            "source_metadata_sha256": _row_digest(original_metadata),
            "fresh_metadata_sha256": _row_digest(fresh_metadata),
            "wallet_counts_sha256": _row_digest(db.execute("SELECT wallet,observed_count FROM wallet_counts ORDER BY wallet")),
            "original_checkpoint_sha256": __import__("hashlib").sha256(_canonical(original_checkpoint).encode()).hexdigest(),
            "original_provenance_report_sha256": __import__("hashlib").sha256(_canonical(original_provenance).encode()).hexdigest(),
            "fresh_collection_report_sha256": __import__("hashlib").sha256(_canonical(fresh_collection).encode()).hexdigest(),
            "fresh_provenance_report_sha256": __import__("hashlib").sha256(_canonical(fresh_provenance).encode()).hexdigest(),
            "source_database_modified": False, "fresh_database_modified": False,
            "selection_is_retrospective": True, "activity_count_feature_eligible": False,
            "market_maker_status_verified": False, "human_identity_verified": False,
            "source_completeness_certified": False, "training_ready": False,
            "chain_reconciliation_verified": False, "holdings_reconstructed": False,
            "news_context_preserved": True, "news_completion_claimed": False,
            "inherited_source_report": original_report,
        }
        db.execute("INSERT INTO metadata VALUES('report',?)", (_canonical(report),))
        db.execute("INSERT INTO metadata VALUES('registry_feature_eligible','false')")
        ledger = [{"condition_id": condition, "status": "api_exhausted",
                   "provenance_scope": "fresh" if condition in new_manifest_hashes else "inherited",
                   "manifest_sha256": new_manifest_hashes.get(condition, old_manifest_hashes[condition]),
                   "checkpoint_digest_semantics": "SHA256_of_canonical_JSON_object",
                   "checkpoint_sha256": report["fresh_provenance_report_sha256"] if condition in new_manifest_hashes else report["original_provenance_report_sha256"]}
                  for condition in sorted(old_condition_ids)]
        db.execute("INSERT INTO metadata VALUES('contract_exhaustion_ledger',?)", (_canonical(ledger),))
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Completed cohort integrity check failed")
        db.close(); db = None
        for path in (source,fresh):
            _reject_live_wal(path)
            if _identity(path) != before[str(path)]:
                raise ValueError("An input changed while completing the cohort")
        os.link(temporary,output)
        temporary.unlink()
        return report
    finally:
        if db is not None:
            db.close()
        for suffix in ("","-journal","-wal","-shm"):
            Path(str(temporary)+suffix).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","fresh","output","original-checkpoint","original-provenance","fresh-collection","fresh-provenance","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threshold",type=int,default=20)
    parser.add_argument("--require-tournament-coverage",action="store_true")
    args=parser.parse_args()
    if os.path.lexists(args.report) or args.report.resolve() in {args.source.resolve(),args.fresh.resolve(),args.output.resolve()}:
        parser.error("Refusing to replace report or input/output database")
    read=lambda p:json.loads(p.read_text())
    report=complete_tournament_database(args.source,args.fresh,args.output,original_checkpoint=read(args.original_checkpoint),original_provenance=read(args.original_provenance),
        fresh_collection=read(args.fresh_collection),fresh_provenance=read(args.fresh_provenance),threshold=args.threshold,
        require_tournament_coverage=args.require_tournament_coverage)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as stream: stream.write(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__ == "__main__":
    main()
