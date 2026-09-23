import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index, read_trade_context
from scripts.filter_tournament_wallets import filter_tournament_database
from scripts.filter_wallet_activity import filter_database


C1 = "0x" + "1" * 64
C2 = "0x" + "2" * 64
REGISTRY = {
    "fixtures": [{"fixture_id": "fixture:1"}, {"fixture_id": "fixture:2"}],
    "contracts": [
        {"condition_id": condition, "fixture_id": fixture, "selection": "home",
         "tokens": [{"token_id": "123", "outcome": "Yes"},
                    {"token_id": "456", "outcome": "No"}]}
        for condition, fixture in [(C1, "fixture:1"), (C2, "fixture:2")]
    ],
}


class TournamentWalletFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = self.root / "original.sqlite"
        self.source = self.root / "per_contract.sqlite"
        self.output = self.root / "tournament.sqlite"
        rows = []
        for wallet, condition, count in [
            ("wallet:19", C1, 10), ("wallet:19", C2, 9),
            ("wallet:20", C1, 10), ("wallet:20", C2, 10),
            ("wallet:21", C1, 20), ("wallet:21", C2, 1),
            ("wallet:1", C2, 1),
        ]:
            for i in range(count):
                row_id = len(rows) + 1
                rows.append({
                    "observation_id": f"obs:{row_id}", "condition_id": condition,
                    "token_id": "123" if i % 2 else "456",
                    "side": "BUY" if i % 2 else "SELL", "size": "100.123456",
                    "price": "0.12345678", "proxy_wallet": wallet,
                    "block_timestamp": f"2026-06-10T12:{row_id:02d}:00Z" if row_id < 60 else "2026-06-11T12:00:00Z",
                    "transaction_hash": f"tx:{row_id}",
                })
        page = self.root / "trades.jsonl"
        page.write_text("".join(json.dumps(row) + "\n" for row in rows))
        news = [{"news_id": "news:1", "news_item_id": "item:1", "version_rank": 0,
                 "title": "World Cup team news", "fixture_ids": ["fixture:1"],
                 "published_at_utc": "2026-06-08T12:00:00Z"}]
        build_attribution_index(registry=REGISTRY, trade_pages=[page], news_records=news,
                                output_path=self.original)
        with sqlite3.connect(self.original) as db:
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            report.update(partial_api_collection=True, api_traversals_exhausted=1,
                          registry_condition_count=2)
            db.execute("UPDATE metadata SET value_json=? WHERE key='report'", (json.dumps(report),))
        filter_database(self.original, self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_19_across_contracts_kept_20_across_contracts_excluded(self):
        report = filter_tournament_database(self.source, self.output)
        with sqlite3.connect(self.output) as db:
            self.assertEqual(db.execute("SELECT wallet,COUNT(*) FROM trades GROUP BY wallet ORDER BY wallet").fetchall(),
                             [("wallet:1", 1), ("wallet:19", 19)])
            self.assertEqual(db.execute("SELECT observed_count FROM wallet_counts WHERE wallet='wallet:21'").fetchone()[0], 21)
            self.assertEqual(db.execute("SELECT SUM(observed_count) FROM wallet_counts").fetchone()[0], 61)
        self.assertEqual(report["source_observations"], 61)
        self.assertEqual(report["immediate_input_observations"], 41)
        self.assertEqual(report["retained_observations"], 20)
        self.assertEqual(report["threshold_scope"], "tournament")
        self.assertTrue(report["full_wallet_history_included"])
        self.assertTrue(report["complete_retained_wallet_counts_verified"])
        self.assertTrue(report["partial_api_collection"])
        self.assertFalse(report["training_ready"])
        self.assertFalse(report["source_completeness_certified"])
        self.assertFalse(report["market_maker_status_verified"])

    def test_source_bytes_exact_rows_and_context_unchanged(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        filter_tournament_database(self.source, self.output)
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).hexdigest())
        with sqlite3.connect(self.original) as original, sqlite3.connect(self.output) as filtered:
            self.assertEqual(filtered.execute("SELECT * FROM trades ORDER BY trade_row_id").fetchall(),
                             original.execute("SELECT * FROM trades WHERE wallet IN ('wallet:1','wallet:19') ORDER BY trade_row_id").fetchall())
            for table in ("news", "fixture_news", "global_news", "context_states", "source_pages"):
                self.assertEqual(filtered.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall(),
                                 original.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall())
            self.assertEqual(filtered.execute("SELECT * FROM source_metadata ORDER BY key").fetchall(),
                             original.execute("SELECT * FROM metadata ORDER BY key").fetchall())

    def test_qualifying_wallet_history_is_complete_in_new_database(self):
        filter_tournament_database(self.source, self.output)
        expected = read_trade_context(self.original, 19, wallet_history_limit=100)
        actual = read_trade_context(self.output, 19, wallet_history_limit=100)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual["retrospective_prior_tournament_executions"]), 18)
        self.assertEqual({row["condition_id"] for row in actual["retrospective_prior_tournament_executions"]}, {C1, C2})

    def test_missing_qualifying_row_rejected_even_if_input_row_metadata_adjusted(self):
        with sqlite3.connect(self.source) as db:
            db.execute("DELETE FROM selected_trades WHERE trade_row_id=1")
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            report["retained_observations"] -= 1
            db.execute("UPDATE metadata SET value_json=? WHERE key='report'", (json.dumps(report),))
        with self.assertRaisesRegex(ValueError, "qualifying wallet"):
            filter_tournament_database(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".tournament.sqlite.*")))

    def test_tampered_global_counts_rejected(self):
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE wallet_market_counts SET observation_count=9 WHERE wallet='wallet:20' AND condition_id=?", (C1,))
        with self.assertRaisesRegex(ValueError, "count disagrees"):
            filter_tournament_database(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_tampered_pair_counts_rejected_even_when_wallet_total_unchanged(self):
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE wallet_market_counts SET observation_count=9 WHERE wallet='wallet:19' AND condition_id=?", (C1,))
            db.execute("UPDATE wallet_market_counts SET observation_count=10 WHERE wallet='wallet:19' AND condition_id=?", (C2,))
        with self.assertRaisesRegex(ValueError, "contract counts disagree"):
            filter_tournament_database(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_threshold_one_yields_empty_database(self):
        report = filter_tournament_database(self.source, self.output, threshold=1)
        self.assertEqual(report["retained_observations"], 0)
        self.assertEqual(report["retained_wallets"], 0)

    def test_cannot_increase_threshold_beyond_prior_selection(self):
        with self.assertRaisesRegex(ValueError, "threshold at least"):
            filter_tournament_database(self.source, self.output, threshold=21)
        self.assertFalse(self.output.exists())

    def test_production_coverage_gate_rejects_tiny_source(self):
        with self.assertRaisesRegex(ValueError, "104 fixtures"):
            filter_tournament_database(self.source, self.output, require_tournament_coverage=True)
        self.assertFalse(self.output.exists())

    def test_invalid_thresholds_and_overwrites_rejected_without_mutation(self):
        before = self.source.read_bytes()
        for value in [0, -1, True, 20.5, "20"]:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                filter_tournament_database(self.source, self.output, threshold=value)
        with self.assertRaises(FileExistsError):
            filter_tournament_database(self.source, self.source)
        alias = self.root / "alias.sqlite"
        alias.hardlink_to(self.source)
        with self.assertRaises(FileExistsError):
            filter_tournament_database(self.source, alias)
        self.output.write_bytes(b"untouched")
        with self.assertRaises(FileExistsError):
            filter_tournament_database(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"untouched")
        self.assertEqual(self.source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
