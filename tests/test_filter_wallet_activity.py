import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index, read_trade_context
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


class WalletActivityFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.sqlite"
        self.output = self.root / "filtered.sqlite"
        rows = []
        for wallet, condition, count in [
            ("wallet:a", C1, 19), ("wallet:a", C2, 20),
            ("wallet:b", C1, 20), ("wallet:c", C1, 21), ("wallet:d", C1, 1),
        ]:
            for i in range(count):
                row_id = len(rows) + 1
                rows.append({
                    "observation_id": f"obs:{row_id}", "condition_id": condition,
                    "token_id": "123" if i % 2 else "456",
                    "side": "BUY" if i % 2 else "SELL", "size": "100.123456",
                    "price": "0.12345678", "proxy_wallet": wallet,
                    "block_timestamp": ("2026-06-09T12:00:00Z" if condition == C2
                                        else "2026-06-10T12:00:00Z"),
                    "transaction_hash": "one-transaction-many-observations" if wallet == "wallet:b" else f"tx:{row_id}",
                })
        page = self.root / "trades.jsonl"
        page.write_text("".join(json.dumps(row) + "\n" for row in rows))
        news = [{"news_id": "news:1", "news_item_id": "item:1", "version_rank": 0,
                 "title": "World Cup team news", "fixture_ids": ["fixture:1"],
                 "published_at_utc": "2026-06-08T12:00:00Z"}]
        build_attribution_index(registry=REGISTRY, trade_pages=[page], news_records=news,
                                output_path=self.source)
        with sqlite3.connect(self.source) as db:
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            report.update(partial_api_collection=True, api_traversals_exhausted=1,
                          registry_condition_count=2)
            db.execute("UPDATE metadata SET value_json=? WHERE key='report'", (json.dumps(report),))

    def tearDown(self):
        self.temp.cleanup()

    def test_strict_threshold_whole_pairs_and_per_market_scope(self):
        report = filter_database(self.source, self.output)
        with sqlite3.connect(self.output) as db:
            pairs = db.execute("SELECT wallet,condition_id,COUNT(*) FROM selected_trades GROUP BY wallet,condition_id ORDER BY wallet").fetchall()
            self.assertEqual(pairs, [("wallet:a", C1, 19), ("wallet:d", C1, 1)])
            self.assertEqual(db.execute("SELECT observation_count FROM wallet_market_counts WHERE wallet='wallet:b'").fetchone()[0], 20)
            self.assertEqual(db.execute("SELECT SUM(observation_count) FROM wallet_market_counts").fetchone()[0], 81)
        self.assertEqual(report["source_observations"], 81)
        self.assertEqual(report["retained_observations"], 20)
        self.assertEqual(report["excluded_observations"], 61)

    def test_source_bytes_and_rows_context_decimals_are_preserved(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        filter_database(self.source, self.output)
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).hexdigest())
        with sqlite3.connect(self.source) as original, sqlite3.connect(self.output) as filtered:
            self.assertEqual(filtered.execute("SELECT * FROM selected_trades ORDER BY trade_row_id").fetchall(),
                             original.execute("SELECT * FROM trades WHERE trade_row_id<=19 OR trade_row_id=81 ORDER BY trade_row_id").fetchall())
            for table in ("news", "fixture_news", "global_news", "context_states", "source_pages"):
                self.assertEqual(filtered.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall(),
                                 original.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall())
            self.assertEqual(filtered.execute("SELECT * FROM source_metadata ORDER BY key").fetchall(),
                             original.execute("SELECT * FROM metadata ORDER BY key").fetchall())

    def test_full_wallet_history_remains_in_original_source(self):
        filter_database(self.source, self.output)
        context = read_trade_context(self.source, 1, wallet_history_limit=100)
        past = context["retrospective_prior_tournament_executions"]
        self.assertEqual(len(past), 20)
        self.assertTrue(all(row["condition_id"] == C2 for row in past))
        with sqlite3.connect(self.output) as db:
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='trades'").fetchone())
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            self.assertFalse(report["training_ready"])

    def test_nondefault_threshold_is_strict(self):
        report = filter_database(self.source, self.output, threshold=21)
        self.assertEqual(report["retained_observations"], 60)
        with sqlite3.connect(self.output) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM selected_trades WHERE wallet='wallet:c'").fetchone()[0], 0)

    def test_threshold_one_can_produce_empty_selection(self):
        report = filter_database(self.source, self.output, threshold=1)
        self.assertEqual(report["retained_observations"], 0)
        with sqlite3.connect(self.output) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM selected_trades").fetchone()[0], 0)

    def test_rejects_invalid_thresholds_without_output(self):
        for value in [0, -1, True, 20.5, "20"]:
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    filter_database(self.source, self.output, threshold=value)
                self.assertFalse(self.output.exists())

    def test_refuses_original_path_and_existing_outputs(self):
        before = self.source.read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            filter_database(self.source, self.source)
        self.output.write_bytes(b"keep-existing-output")
        with self.assertRaises((ValueError, FileExistsError)):
            filter_database(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep-existing-output")
        self.assertEqual(self.source.read_bytes(), before)

    def test_refuses_source_alias(self):
        alias = self.root / "alias.sqlite"
        alias.hardlink_to(self.source)
        with self.assertRaises((ValueError, FileExistsError)):
            filter_database(self.source, alias)


if __name__ == "__main__":
    unittest.main()
