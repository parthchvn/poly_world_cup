import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from poly_world_cup.historical_contracts import (
    ADAPTER, MARKET_TOPIC, QUESTION_TOPIC, SOURCE_URL, build_contract_record,
    decode_metadata_event, extract_initial_resolution_clause, mapping_call_specs, validate_contract_record,
    verified_contract_context,
)


def contract_record_fixture():
    """Small synthetic RPC evidence for integration tests, never production data."""
    market_id = "0x" + "ab" * 31 + "00"
    question_id = "0x" + "ab" * 31 + "01"

    def event(topic, text, timestamp, block, *, market=False):
        raw = text.encode()
        payload = ((0 if market else 1).to_bytes(32, "big") + (64).to_bytes(32, "big") +
                   len(raw).to_bytes(32, "big") + raw + bytes((-len(raw)) % 32))
        log = {"address": ADAPTER, "topics": [topic, market_id,
               "0x" + "00" * 12 + "ef" * 20 if market else question_id],
               "data": "0x" + payload.hex(), "removed": False,
               "blockNumber": hex(block), "blockHash": "0x" + f"{block:064x}",
               "transactionHash": "0x" + f"{block + 99:064x}", "logIndex": "0x1",
               "blockTimestamp": hex(timestamp)}
        header = {"hash": log["blockHash"], "number": log["blockNumber"], "timestamp": hex(timestamp)}
        return log, header

    qlog, qblock = event(QUESTION_TOPIC,
        "question: Will Mexico win on 2026-06-11?, description: Yes if Mexico wins; No otherwise., id: 100",
        1781000100, 85_000_100)
    mlog, mblock = event(MARKET_TOPIC,
        "title: Mexico vs. South Africa, description: FIFA World Cup 2026., id: 200",
        1781000000, 85_000_000, market=True)
    results = ["0x" + "cc" * 32, "0x" + f"{123:064x}", "0x" + f"{456:064x}"]
    specs = mapping_call_specs(question_id, qlog["blockNumber"])
    evidence = {"chain_id": 137, "adapter": ADAPTER, "rpc_url": "https://example.test/rpc",
                "source_code_url": SOURCE_URL, "snapshot_block_number": hex(90_000_000),
                "question_log": qlog, "question_block": qblock,
                "market_log": mlog, "market_block": mblock,
                "mapping_calls": [{**spec, "result": result} for spec, result in zip(specs, results)]}
    return build_contract_record(evidence, fixture_id="espn:760415")


class HistoricalContractTests(unittest.TestCase):
    def test_valid_evidence_reconstructs_initial_text_and_tokens(self):
        row = validate_contract_record(contract_record_fixture())
        self.assertEqual(row["question"], "Will Mexico win on 2026-06-11?")
        self.assertEqual(row["fixture_title"], "Mexico vs. South Africa")
        self.assertEqual((row["yes_token_id"], row["no_token_id"]), ("123", "456"))

    def test_strict_timestamp_and_minimal_feature_scope(self):
        row = contract_record_fixture()
        timestamp = int(datetime.fromisoformat(row["initialized_at_utc"].replace("Z", "+00:00")).timestamp()) * 1_000_000
        self.assertIsNone(verified_contract_context(row, None))
        self.assertIsNone(verified_contract_context(row, timestamp - 1))
        self.assertIsNone(verified_contract_context(row, timestamp))
        context = verified_contract_context(row, timestamp + 1)
        self.assertEqual(context["question"], row["question"])
        self.assertNotIn("initial_rules", context)
        self.assertNotIn("initial_fixture_description", context)
        self.assertNotIn("fixture_id", context)

    def test_current_question_cannot_replace_initial_question(self):
        row = contract_record_fixture()
        row["question"] = "Later edited question"
        with self.assertRaisesRegex(ValueError, "derived field: question"):
            validate_contract_record(row)

    def test_backdated_initialized_time_rejected(self):
        row = contract_record_fixture()
        row["initialized_at_utc"] = "2020-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "initialized_at_utc"):
            validate_contract_record(row)

    def test_future_or_latest_mapping_call_rejected(self):
        for timestamp in ("latest", hex(90_000_000)):
            row = contract_record_fixture()
            row["evidence"]["mapping_calls"][0]["params"][1] = timestamp
            with self.assertRaisesRegex(ValueError, "historical preparation block"):
                validate_contract_record(row)

    def test_token_order_tamper_rejected(self):
        row = contract_record_fixture()
        row["yes_token_id"], row["no_token_id"] = row["no_token_id"], row["yes_token_id"]
        with self.assertRaisesRegex(ValueError, "yes_token_id"):
            validate_contract_record(row)

    def test_wrong_block_header_or_timestamp_rejected(self):
        for key, value in (("hash", "0x" + "dd" * 32), ("timestamp", "0x1")):
            row = contract_record_fixture()
            row["evidence"]["question_block"][key] = value
            with self.assertRaises(ValueError):
                validate_contract_record(row)

    def test_removed_or_other_contract_event_rejected(self):
        for key, value in (("removed", True), ("address", "0x" + "ef" * 20)):
            row = contract_record_fixture()
            row["evidence"]["question_log"][key] = value
            with self.assertRaisesRegex(ValueError, "another contract"):
                validate_contract_record(row)

    def test_malformed_dynamic_bytes_rejected(self):
        row = contract_record_fixture()
        log, block = row["evidence"]["question_log"], row["evidence"]["question_block"]
        for data in (log["data"][:-2], log["data"] + "00" * 32,
                     log["data"][:66] + "00" * 32 + log["data"][130:]):
            with self.assertRaises(ValueError):
                decode_metadata_event({**log, "data": data}, block)

    def test_question_market_linkage_rejected(self):
        row = contract_record_fixture()
        row["evidence"]["question_log"]["topics"][1] = "0x" + "ef" * 31 + "00"
        with self.assertRaisesRegex(ValueError, "Question market ID mismatch"):
            validate_contract_record(row)

    def test_event_content_mutation_changes_evidence_id(self):
        row = contract_record_fixture()
        changed = copy.deepcopy(row["evidence"])
        changed["rpc_url"] = "https://different.example.test/rpc"
        other = build_contract_record(changed, fixture_id=row["fixture_id"])
        self.assertNotEqual(other["evidence_id"], row["evidence_id"])


class HistoricalRPCCacheTests(unittest.TestCase):
    def test_reordered_batch_is_cached_and_response_tamper_is_rejected(self):
        from scripts.collect_historical_contracts import RPC
        with tempfile.TemporaryDirectory() as directory:
            rpc = RPC("https://example.test/rpc", directory)
            specs = [{"method": "eth_chainId", "params": []},
                     {"method": "eth_blockNumber", "params": []}]
            response = [{"jsonrpc": "2.0", "id": 1, "result": "0x123"},
                        {"jsonrpc": "2.0", "id": 0, "result": "0x89"}]
            with patch("urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))) as fetch:
                self.assertEqual(rpc.batch(specs), ["0x89", "0x123"])
                self.assertEqual(rpc.batch(specs), ["0x89", "0x123"])
                self.assertEqual(fetch.call_count, 1)
            path = next(Path(directory).glob("*.json"))
            saved = json.loads(path.read_text())
            saved["result"] = "0xffff"
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "cache integrity"):
                rpc.batch(specs)

    def test_cache_cannot_change_provider(self):
        from scripts.collect_historical_contracts import RPC
        with tempfile.TemporaryDirectory() as directory:
            rpc = RPC("https://example.test/rpc", directory)
            spec = {"method": "eth_chainId", "params": []}
            response = [{"jsonrpc": "2.0", "id": 0, "result": "0x89"}]
            with patch("urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))):
                rpc.batch([spec])
            path = next(Path(directory).glob("*.json"))
            saved = json.loads(path.read_text())
            saved["rpc_url"] = "https://different.example.test/rpc"
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "cache integrity"):
                rpc.batch([spec])


class InitialResolutionClauseTests(unittest.TestCase):
    CLAUSE = "This market refers only to the outcome within the first 90 minutes of regular play plus stoppage time."

    def test_returns_exact_captured_sentence_without_synthesis(self):
        rules = "If Mexico wins, this market resolves Yes.\n" + self.CLAUSE + "\nThe official statistics are the source."
        self.assertEqual(extract_initial_resolution_clause(rules), self.CLAUSE)
        self.assertIn(extract_initial_resolution_clause(rules), rules)
        self.assertIsNone(extract_initial_resolution_clause("If Mexico wins, this market resolves Yes."))

    def test_conflicting_or_additional_duration_statements_are_not_inferred(self):
        for extra in ("Extra time and penalties are included.", "Extra time does not count.",
                      "The game is decided after overtime.", "Only regulation time counts."):
            self.assertIsNone(extract_initial_resolution_clause(self.CLAUSE + "\n" + extra))

    def test_negated_or_unsupported_template_has_no_fallback(self):
        for rules in (self.CLAUSE.replace("refers only", "does not refer only"),
                      "The team advancing after penalties wins.",
                      self.CLAUSE + " " + self.CLAUSE, None):
            self.assertIsNone(extract_initial_resolution_clause(rules))

    def test_verified_context_adds_exact_clause_only_after_initialization(self):
        row = contract_record_fixture()
        evidence = copy.deepcopy(row["evidence"])
        old_data = bytes.fromhex(evidence["question_log"]["data"][2:])
        old_text = old_data[96:96 + int.from_bytes(old_data[64:96], "big")].decode()
        new_text = old_text.replace("Yes if Mexico wins; No otherwise.", self.CLAUSE).encode()
        raw = old_data[:64] + len(new_text).to_bytes(32, "big") + new_text + bytes((-len(new_text)) % 32)
        evidence["question_log"]["data"] = "0x" + raw.hex()
        updated = validate_contract_record(build_contract_record(evidence, fixture_id=row["fixture_id"]))
        context = verified_contract_context(updated, 10**30)
        self.assertEqual(context["initial_resolution_clause"], self.CLAUSE)
        self.assertLessEqual(len(context["initial_resolution_clause"]), 400)
        self.assertIsNone(verified_contract_context(updated, 1781000100 * 1_000_000))


if __name__ == "__main__":
    unittest.main()
