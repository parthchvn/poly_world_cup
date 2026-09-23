import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index, _micros
from poly_world_cup.sft import export_sft, _utc
from scripts.validate_sft import validate_sft
from tests.test_historical_contracts import contract_record_fixture


class HistoricalContractSFTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.record = contract_record_fixture()
        self.database = self.root / "cohort.sqlite"
        self.output = self.root / "sft"
        condition, fixture = self.record["condition_id"], self.record["fixture_id"]
        self.registry = {"fixtures": [{"fixture_id": fixture}], "contracts": [{
            "condition_id": condition, "fixture_id": fixture, "selection": "mexico",
            "tokens": [{"token_id": self.record["yes_token_id"], "outcome": "Yes"},
                       {"token_id": self.record["no_token_id"], "outcome": "No"}]}]}
        initialized = _micros(self.record["initialized_at_utc"])
        page = self.root / "page.jsonl"
        page.write_text("".join(json.dumps({"observation_id": str(i), "condition_id": condition,
            "token_id": self.record["yes_token_id"], "proxy_wallet": "wallet:a", "side": "BUY",
            "size": "2", "price": "0.5", "transaction_hash": "tx:" + str(i),
            "block_timestamp": _utc(initialized + offset * 1_000_000),
        }) + "\n" for i, offset in enumerate([-1, 0, 1, 2])))
        build_attribution_index(registry=self.registry, trade_pages=[page], news_records=[], output_path=self.database)
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE wallet_counts(wallet TEXT PRIMARY KEY,observed_count INTEGER)")
            db.execute("INSERT INTO wallet_counts VALUES('wallet:a',4)")
            db.execute("CREATE TABLE contract_evidence(condition_id TEXT PRIMARY KEY,record_json TEXT)")
            db.execute("INSERT INTO contract_evidence VALUES(?,?)", (condition, json.dumps(self.record)))
        self.policy = {"fixture_splits": {fixture: "train"},
            "train_before_utc": "2026-06-28T12:00:00Z", "validation_before_utc": "2026-07-09T00:00:00Z"}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return export_sft(self.database, self.output, self.policy, allow_partial=True)

    def test_question_context_is_strictly_historical_for_targets_and_prior_executions(self):
        manifest = self.build()
        self.assertEqual(manifest["historical_contract_evidence_count"], 1)
        self.assertEqual(manifest["attribution_coverage_all_source_targets"]["exported_targets_with_verified_contract_context"], 2)
        with gzip.open(self.output / "execution_history_proxy/train/part-00001.jsonl.gz", "rt") as stream:
            prompts = [json.loads(json.loads(line)["messages"][1]["content"]) for line in stream]
        self.assertIsNone(prompts[0]["verified_contract_context"])
        self.assertIsNone(prompts[1]["verified_contract_context"])
        self.assertEqual(prompts[2]["verified_contract_context"]["question"], self.record["question"])
        history = prompts[3]["prior_tournament_executions"]
        self.assertEqual([row["verified_contract_context"] is not None for row in history], [False, False, True])
        self.assertEqual(validate_sft(self.output)["profile_examples_verified"], 8)

    def test_rehashed_current_question_injection_fails_evidence_replay(self):
        self.build()
        relative = "source_contracts.jsonl.gz"
        path = self.output / relative
        changed = {**self.record, "question": "Invented current market question"}
        path.write_bytes(gzip.compress((json.dumps(changed) + "\n").encode(), mtime=0))
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifact = next(row for row in manifest["files"] if row["path"] == relative)
        artifact.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "derived field: question"):
            validate_sft(self.output)

    def test_export_rejects_trade_token_outside_historical_contract_mapping(self):
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE trades SET token_id='99999' WHERE trade_row_id=4")
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_export_rejects_trade_outcome_disagreeing_with_historical_token(self):
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE trades SET token_outcome='No' WHERE trade_row_id=4")
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_export_rejects_trade_fixture_disagreeing_with_contract_evidence(self):
        # Include the wrong fixture in the split policy so a missing policy row
        # cannot accidentally make this test pass before the evidence check.
        self.policy["fixture_splits"]["espn:999999"] = "train"
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE trades SET fixture_id='espn:999999' WHERE trade_row_id=4")
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rehashed_audit_token_must_match_independently_decoded_contract(self):
        self.build()
        relative = "audit/part-00001.jsonl.gz"
        path = self.output / relative
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        # The final target cannot appear in any later prompt's history, so this
        # mutation specifically requires the evidence-to-audit mapping check.
        rows[-1]["token_id"] = "99999"
        path.write_bytes(gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0))
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifact = next(row for row in manifest["files"] if row["path"] == relative)
        artifact.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            validate_sft(self.output)


if __name__ == "__main__":
    unittest.main()
