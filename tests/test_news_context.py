import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from poly_world_cup.news_context import replace_news_context


class NewsContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.sqlite"
        self.output = self.root / "new.sqlite"
        self.registry = {"fixtures": [{"fixture_id": "fixture:a"}], "contracts": [{
            "condition_id": "condition:a", "fixture_id": "fixture:a", "selection": "home",
            "tokens": [{"token_id": "123", "outcome": "Yes"}]}]}
        self.page = self.root / "trades.jsonl"
        self.page.write_text("".join(json.dumps({
            "observation_id": str(i), "condition_id": "condition:a", "token_id": "123",
            "proxy_wallet": "wallet:a", "side": "BUY", "size": "3", "price": "0.2",
            "block_timestamp": time, "transaction_hash": "tx:" + str(i),
        }) + "\n" for i, time in enumerate([
            "2026-06-10T11:00:00Z", "2026-06-10T12:00:00Z", "2026-06-10T13:00:00Z"])))
        build_attribution_index(registry=self.registry, trade_pages=[self.page],
            news_records=[], output_path=self.source)
        with sqlite3.connect(self.source) as db:
            db.execute("CREATE TABLE wallet_counts(wallet TEXT PRIMARY KEY, observed_count INTEGER)")
            db.execute("INSERT INTO wallet_counts VALUES('wallet:a',3)")

    def tearDown(self):
        self.tmp.cleanup()

    def news(self):
        time = "2026-06-10T12:00:00Z"
        evidence = [{"kind": "archive_snapshot", "captured_at_utc": time,
            "source_url": "https://example.org/archive", "content_sha256": "a" * 64}]
        return {"news_id": "news:a", "title": "World Cup match preview",
            "source_url": "https://example.org/story", "context_scope": "tournament",
            "historical_availability_verified": True, "availability_upper_utc": time,
            "historical_content_sha256": "a" * 64, "availability_evidence": evidence,
            "fixture_links": [{"fixture_id": "fixture:a", "historical_link_verified": True,
                "link_availability_upper_utc": time, "link_availability_evidence": evidence}]}

    def test_refresh_matches_complete_rebuild_and_preserves_original(self):
        source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        report = replace_news_context(self.source, self.output, self.registry, [self.news()])
        expected = self.root / "expected.sqlite"
        build_attribution_index(registry=self.registry, trade_pages=[self.page],
            news_records=[self.news()], output_path=expected)
        with sqlite3.connect(self.output) as actual, sqlite3.connect(expected) as rebuilt:
            for table in ("trades", "news", "fixture_news", "global_news", "context_states", "source_pages"):
                self.assertEqual(actual.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall(),
                    rebuilt.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall(), table)
            self.assertEqual(actual.execute("SELECT * FROM wallet_counts").fetchall(), [("wallet:a", 3)])
            self.assertEqual(actual.execute("SELECT eligible_context_event_count FROM trades ORDER BY trade_row_id").fetchall(), [(0,), (0,), (1,)])
        self.assertEqual(report["observation_count"], 3)
        self.assertEqual(source_hash, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_failure_does_not_publish_or_modify_source(self):
        source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        invalid = self.news()
        invalid["fixture_links"][0]["fixture_id"] = "fixture:unknown"
        with self.assertRaises(ValueError):
            replace_news_context(self.source, self.output, self.registry, [invalid])
        self.assertFalse(self.output.exists())
        self.assertEqual(source_hash, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_refuses_overwrite(self):
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError):
            replace_news_context(self.source, self.output, self.registry, [self.news()])
        self.assertEqual(self.output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
