import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from poly_world_cup.sft import export_sft
from scripts.filter_wallet_activity import filter_database
from scripts.filter_tournament_wallets import filter_tournament_database
from scripts.validate_sft import validate_sft
from tests.test_sft import REGISTRY, POLICY, trade, news


class SFTArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "tournament.sqlite"
        self.output = self.root / "sft"

    def tearDown(self):
        self.temp.cleanup()

    def build(self, *, shared_transaction=False, invalid_timestamp=False):
        rows = [trade("first", tx="shared" if shared_transaction else "first"),
                trade("second", time="2026-06-10T13:00:00Z"),
                trade("heldout-prefix", fixture="fixture:3", time="2026-06-10T14:00:00Z"),
                trade("validation", fixture="fixture:2", time="2026-06-11T12:00:00Z",
                      tx="shared" if shared_transaction else "validation"),
                trade("test", fixture="fixture:3", time="2026-06-12T12:00:00Z")]
        page = self.root / "page.jsonl"
        page.write_text("".join(json.dumps(row) + "\n" for row in rows))
        build_attribution_index(registry=REGISTRY, trade_pages=[page], news_records=[news("known")],
                                output_path=self.root / "original.sqlite")
        filter_database(self.root / "original.sqlite", self.root / "per_market.sqlite")
        filter_tournament_database(self.root / "per_market.sqlite", self.database)
        if invalid_timestamp:
            with sqlite3.connect(self.database) as db:
                db.execute("UPDATE trades SET query_us='invalid' WHERE trade_row_id=1")
        return export_sft(self.database, self.output, POLICY, shard_rows=2, allow_partial=True)

    def mutate(self, relative, change, *, rehash=True):
        path = self.output / relative
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        change(rows)
        data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        path.write_bytes(gzip.compress(data, mtime=0))
        if rehash:
            manifest_path = self.output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = next(row for row in manifest["files"] if row["path"] == relative)
            item.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(manifest))

    def test_entire_production_pipeline_profiles_and_quarantine_validate(self):
        self.build()
        report = validate_sft(self.output)
        self.assertEqual(report["audited_source_targets"], 5)
        self.assertEqual(report["profile_examples_verified"], 8)
        self.assertEqual(report["quarantined_targets"], 1)
        self.assertTrue(report["transaction_separation_verified"])
        self.assertFalse(report["prospective_training_ready"])

    def test_changed_compressed_bytes_fail_checksum(self):
        self.build()
        path = self.output / "verified_news_only/train/part-00001.jsonl.gz"
        payload = bytearray(path.read_bytes())
        payload[10] ^= 1
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_sft(self.output)

    def test_rehashed_wrong_label_is_rejected(self):
        self.build()
        def change(rows):
            target = json.loads(rows[0]["messages"][2]["content"])
            target["side"] = "SELL"
            rows[0]["messages"][2]["content"] = json.dumps(target)
        self.mutate("verified_news_only/train/part-00001.jsonl.gz", change)
        with self.assertRaisesRegex(ValueError, "Assistant label differs"):
            validate_sft(self.output)

    def test_rehashed_unsupported_headline_is_rejected(self):
        self.build()
        def change(rows):
            prompt = json.loads(rows[0]["messages"][1]["content"])
            prompt["verified_fixture_news"][0]["headline"] = "Invented later result"
            rows[0]["messages"][1]["content"] = json.dumps(prompt)
        self.mutate("verified_news_only/train/part-00001.jsonl.gz", change)
        with self.assertRaisesRegex(ValueError, "Prompt differs"):
            validate_sft(self.output)

    def test_rehashed_target_transaction_in_audited_history_is_rejected(self):
        self.build()
        def change(rows):
            rows[0]["proxy_history_trade_row_ids"] = [rows[0]["trade_row_id"]]
        self.mutate("audit/part-00001.jsonl.gz", change)
        with self.assertRaisesRegex(ValueError, "Audit history"):
            validate_sft(self.output)

    def test_rehashed_extra_prompt_target_field_is_rejected(self):
        self.build()
        def change(rows):
            prompt = json.loads(rows[0]["messages"][1]["content"])
            prompt["target_price"] = "0.12345678"
            rows[0]["messages"][1]["content"] = json.dumps(prompt)
        self.mutate("verified_news_only/train/part-00001.jsonl.gz", change)
        with self.assertRaisesRegex(ValueError, "prohibited target/audit fields"):
            validate_sft(self.output)

    def test_invalid_timestamp_quarantine_is_supported(self):
        self.build(invalid_timestamp=True)
        report = validate_sft(self.output)
        self.assertEqual(report["quarantined_targets"], 2)

    def test_cross_split_transactions_are_quarantined_and_excluded_from_history(self):
        self.build(shared_transaction=True)
        report = validate_sft(self.output)
        self.assertEqual(report["quarantined_targets"], 3)
        self.assertEqual(report["profile_examples_verified"], 4)


if __name__ == "__main__":
    unittest.main()
