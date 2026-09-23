import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from poly_world_cup.sft import export_sft


CONDITIONS = {f"fixture:{i}": "0x" + str(i) * 64 for i in range(1, 4)}
REGISTRY = {
    "fixtures": [{"fixture_id": fixture} for fixture in CONDITIONS],
    "contracts": [
        {"condition_id": condition, "fixture_id": fixture, "selection": "home",
         "tokens": [{"token_id": "123", "outcome": "Yes"},
                    {"token_id": "456", "outcome": "No"}]}
        for fixture, condition in CONDITIONS.items()
    ],
}
POLICY = {
    "fixture_splits": {"fixture:1": "train", "fixture:2": "validation", "fixture:3": "test"},
    "train_before_utc": "2026-06-11T00:00:00Z",
    "validation_before_utc": "2026-06-12T00:00:00Z",
}


def trade(identity, *, time="2026-06-10T12:00:00Z", fixture="fixture:1", wallet="wallet:a", tx=None, **changes):
    return {"observation_id": identity, "condition_id": CONDITIONS[fixture], "token_id": "123",
            "side": "BUY", "size": "100.123456", "price": "0.12345678", "proxy_wallet": wallet,
            "block_timestamp": time, "transaction_hash": tx or "tx:" + identity, **changes}


def news(identity, *, time="2026-06-10T11:00:00Z", verified=True, item=None, version=0):
    evidence = [{"kind": "archive_snapshot", "captured_at_utc": time,
                 "source_url": "https://example.org/archive", "content_sha256": "a" * 64}]
    return {
        "news_id": identity, "news_item_id": item or identity, "version_rank": version,
        "version_order_historically_verified": verified,
        "title": "World Cup 2026 " + identity, "source_url": "https://example.org/" + identity,
        "fixture_ids": ["fixture:1"], "published_at_utc": "2026-06-01T00:00:00Z",
        "captured_at_utc": "2026-09-23T00:00:00Z", "context_scope": "tournament",
        "historical_availability_verified": verified, "availability_upper_utc": time,
        "historical_content_sha256": "a" * 64, "availability_evidence": evidence,
        "fixture_links": [{"fixture_id": "fixture:1", "relationship": "direct_match",
                           "historical_link_verified": verified, "link_availability_upper_utc": time,
                           "link_availability_evidence": evidence}],
    }


class SFTExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "source.sqlite"
        self.output = self.root / "sft"

    def tearDown(self):
        self.temp.cleanup()

    def build(self, rows=None, records=None):
        page = self.root / "trades.jsonl"
        page.write_text("".join(json.dumps(row) + "\n" for row in (rows or [trade("target")])) )
        build_attribution_index(registry=REGISTRY, trade_pages=[page], news_records=records or [],
                                output_path=self.database)
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE wallet_counts(wallet TEXT PRIMARY KEY, observed_count INTEGER NOT NULL)")
            db.execute("INSERT INTO wallet_counts SELECT wallet,COUNT(*) FROM trades GROUP BY wallet")
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            report.update(partial_api_collection=True, threshold_scope="tournament", threshold_exclusive=20,
                          full_wallet_history_included=True)
            db.execute("UPDATE metadata SET value_json=? WHERE key='report'", (json.dumps(report),))

    def export(self, **kwargs):
        return export_sft(self.database, self.output, POLICY, shard_rows=2, allow_partial=True, **kwargs)

    def load(self, path):
        if path.is_dir():
            return [row for part in sorted(path.glob("*.jsonl.gz")) for row in self.load(part)]
        with gzip.open(path, "rt") as stream:
            return [json.loads(line) for line in stream]

    def audit(self):
        return {row["trade_row_id"]: row for row in self.load(self.output / "audit")}

    def example(self, row_id, profile="verified_news_only"):
        audit = self.audit()[row_id]
        location = audit["profile_locations"][profile]
        path = self.output / profile / audit["exported_split"] / f"part-{location['shard']:05d}.jsonl.gz"
        return self.load(path)[location["line"] - 1]

    def prompt(self, row_id, profile="verified_news_only"):
        return json.loads(self.example(row_id, profile)["messages"][1]["content"])

    def test_strict_prior_news_excludes_equal_future_and_retrospective_records(self):
        self.build(records=[news("earlier"), news("equal", time="2026-06-10T12:00:00Z"),
                            news("later", time="2026-06-10T13:00:00Z"), news("unverified", verified=False)])
        # A stale or forged eligibility column must not override evidence checks.
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE fixture_news SET eligible_us=1 WHERE news_id='unverified'")
        self.export()
        prompt = self.prompt(1)
        self.assertEqual([row["news_id"] for row in prompt["verified_fixture_news"]], ["earlier"])
        self.assertEqual([row["news_id"] for row in prompt["verified_tournament_news"]], ["earlier"])
        self.assertEqual(prompt["verified_fixture_news"][0]["verified_available_at_utc"], "2026-06-10T11:00:00Z")

    def test_news_versions_change_only_after_verified_revision_and_link_availability(self):
        original = news("original", item="article")
        revision = news("revision", item="article", version=1, time="2026-06-10T13:00:00Z")
        unknown_order = news("unknown-order", item="article", version=2)
        unknown_order["version_order_historically_verified"] = False
        late_link = news("late-link")
        late_link["fixture_links"][0]["link_availability_upper_utc"] = "2026-06-10T13:00:00Z"
        self.build(rows=[trade("before"), trade("after", time="2026-06-10T14:00:00Z")],
                   records=[revision, unknown_order, original, late_link])
        self.export()
        before, after = self.prompt(1), self.prompt(2)
        self.assertEqual([row["news_id"] for row in before["verified_fixture_news"]], ["original"])
        self.assertEqual({row["news_id"] for row in before["verified_tournament_news"]}, {"original", "late-link"})
        self.assertEqual({row["news_id"] for row in after["verified_fixture_news"]}, {"revision", "late-link"})
        self.assertNotIn("unknown-order", json.dumps(before) + json.dumps(after))

    def test_strict_profile_excludes_history_and_target_values_remain_exact(self):
        self.build(rows=[trade("past", time="2026-06-10T11:00:00Z"),
                         trade("target", token_id="456", side="SELL", size="0002.123456789", price="0.010000")])
        self.export()
        example = self.example(2)
        self.assertEqual([row["role"] for row in example["messages"]], ["system", "user", "assistant"])
        prompt = self.prompt(2)
        self.assertNotIn("prior_tournament_executions", prompt)
        self.assertNotIn("observed_tournament_count", prompt)
        self.assertNotIn("side", prompt)
        self.assertNotIn("price", prompt)
        self.assertEqual(json.loads(example["messages"][2]["content"]),
                         {"side": "SELL", "outcome": "No", "shares": "0002.123456789", "price": "0.010000"})

    def test_proxy_history_excludes_equal_future_and_all_target_transaction_rows(self):
        self.build(rows=[trade("target", tx="target-tx"),
                         trade("same-transaction", time="2026-06-10T11:00:00Z", tx="target-tx"),
                         trade("equal-time"), trade("past", time="2026-06-10T10:00:00Z"),
                         trade("future", time="2026-06-10T13:00:00Z")])
        self.export()
        prompt = self.prompt(1, "execution_history_proxy")
        self.assertEqual(self.audit()[1]["proxy_history_trade_row_ids"], [4])
        self.assertEqual(len(prompt["prior_tournament_executions"]), 1)
        self.assertEqual(prompt["prior_tournament_executions"][0]["execution_time_proxy_utc"], "2026-06-10T10:00:00Z")
        self.assertFalse(prompt["prior_execution_availability_verified"])

    def test_fixture_time_purge_and_cross_split_transactions(self):
        self.build(rows=[
            trade("train-ok", time="2026-06-10T08:00:00Z"),
            trade("train-late", time="2026-06-11T00:00:00Z"),
            trade("validation-early", fixture="fixture:2", time="2026-06-10T09:00:00Z"),
            trade("validation-ok", fixture="fixture:2", time="2026-06-11T00:00:00Z"),
            trade("test-early", fixture="fixture:3", time="2026-06-11T09:00:00Z"),
            trade("test-ok", fixture="fixture:3", time="2026-06-12T00:00:00Z"),
            trade("cross-train", time="2026-06-10T10:00:00Z", tx="cross-split"),
            trade("cross-validation", fixture="fixture:2", time="2026-06-11T10:00:00Z", tx="cross-split"),
        ])
        manifest = self.export()
        self.assertEqual(manifest["split_target_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(manifest["quarantined_target_observations"], 5)
        self.assertEqual(manifest["cross_split_transaction_count"], 1)
        audit = self.audit()
        for row_id in (2, 3, 5):
            self.assertIn("fixture_time_split_mismatch", audit[row_id]["excluded_reasons"])
        for row_id in (7, 8):
            self.assertIn("transaction_crosses_splits", audit[row_id]["excluded_reasons"])
        self.assertEqual(audit[6]["proxy_history_trade_row_ids"], [1, 3, 2, 4, 5])

    def test_invalid_targets_and_every_duplicate_id_row_are_quarantined(self):
        self.build(rows=[trade("duplicate", time="2026-06-10T09:00:00Z"),
                         trade("duplicate", time="2026-06-10T10:00:00Z"),
                         trade("bad-price", time="2026-06-10T10:30:00Z"),
                         trade("bad-side", time="2026-06-10T10:45:00Z"),
                         trade("bad-outcome", time="2026-06-10T11:00:00Z", token_id="999"),
                         trade("valid")])
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE trades SET price='NaN' WHERE trade_row_id=3")
            db.execute("UPDATE trades SET side='HOLD' WHERE trade_row_id=4")
        manifest = self.export()
        self.assertEqual(manifest["included_target_observations"], 1)
        self.assertEqual(manifest["quarantined_target_observations"], 5)
        self.assertEqual(manifest["exclusion_reason_counts"]["duplicate_observation_id"], 2)
        self.assertEqual(manifest["duplicate_observation_id_groups"], 1)
        self.assertEqual(self.audit()[6]["proxy_history_trade_row_ids"], [])
        self.assertEqual(self.prompt(6, "execution_history_proxy")["prior_tournament_executions"], [])

    def test_invalid_execution_timestamp_quarantined_without_fabricating_context(self):
        self.build(rows=[trade("invalid"), trade("valid", time="2026-06-10T13:00:00Z")], records=[news("earlier")])
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE trades SET query_us='not-an-integer' WHERE trade_row_id=1")
        manifest = self.export()
        self.assertEqual(manifest["included_target_observations"], 1)
        self.assertEqual(manifest["exclusion_reason_counts"]["invalid_execution_timestamp"], 1)
        self.assertIsNone(self.audit()[1]["execution_time_proxy_utc"])
        self.assertEqual(self.audit()[2]["proxy_history_trade_row_ids"], [])

    def test_partial_gate_and_invalid_split_leave_no_output(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "partial"):
            export_sft(self.database, self.output, POLICY)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".sft.*")))
        with self.assertRaisesRegex(ValueError, "cover every"):
            export_sft(self.database, self.output, {**POLICY, "fixture_splits": {}}, allow_partial=True)
        self.assertFalse(self.output.exists())

    def test_missing_qualifying_wallet_fails_instead_of_silent_history_loss(self):
        self.build()
        with sqlite3.connect(self.database) as db:
            db.execute("INSERT INTO wallet_counts VALUES('missing-wallet',1)")
        with self.assertRaisesRegex(ValueError, "every qualifying wallet"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_source_existing_output_and_artifact_hashes(self):
        self.build(rows=[trade("one"), trade("two", time="2026-06-10T13:00:00Z"),
                         trade("three", time="2026-06-10T14:00:00Z")])
        before = self.database.read_bytes()
        manifest = self.export()
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(manifest["source_database_sha256"], hashlib.sha256(before).hexdigest())
        self.assertTrue(manifest["format_ready_for_sft"])
        self.assertFalse(manifest["training_ready"])
        self.assertFalse(manifest["prospective_training_ready"])
        for artifact in manifest["files"]:
            data = (self.output / artifact["path"]).read_bytes()
            self.assertEqual(len(data), artifact["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), artifact["sha256"])
        shards = sorted((self.output / "verified_news_only" / "train").glob("*.jsonl.gz"))
        self.assertEqual([len(self.load(path)) for path in shards], [2, 1])
        saved = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual((self.output / "manifest.json").read_bytes(), saved)
        with self.assertRaises((ValueError, FileExistsError)):
            export_sft(self.database, self.database, POLICY, allow_partial=True)
        self.assertEqual(self.database.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
