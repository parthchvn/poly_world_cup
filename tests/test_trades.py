"""Behavioral regression tests for incomplete and ambiguous API trade history."""
import hashlib
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from poly_world_cup.trades import API_URL, TradeIngestionError, ingest_condition, validate_collection

CONDITION = "0x" + "a" * 64
WALLET = "0x" + "b" * 40
TX = "0x" + "c" * 64


def trade(**overrides):
    return {"proxy_wallet": WALLET, "condition_id": CONDITION, "token_id": "1234",
            "side": "BUY", "size": 1.25, "price": 0.4,
            "timestamp": 1780000000, "transaction_hash": TX,
            "name": "Private profile label", **overrides}


def page(rows, next_cursor=None):
    return {"data": rows, "pagination": {"has_more": next_cursor is not None,
                                         "next_cursor": next_cursor}}


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params)))
        if not self.responses:
            raise AssertionError("Unexpected HTTP call")
        data = self.responses.pop(0)
        body = json.dumps(data).encode()
        return SimpleNamespace(data=data, url=url + "?" + urlencode(params),
                               body_sha256=hashlib.sha256(body).hexdigest(),
                               retrieved_at="2026-09-22T00:00:00Z")


class TradesTest(unittest.TestCase):
    def test_requested_microfill_threshold_is_recorded_and_immutable_on_resume(self):
        client = Client([page([trade(size="0.000001")])])
        state = self.collect(client, minimum_size="0.000001")
        self.assertEqual(client.calls[0][1]["filter_amount"], "0.000001")
        self.assertEqual(state["minimum_size_filter"]["amount"], "0.000001")
        self.assertEqual(validate_collection(self.output, condition_id=CONDITION), state)
        with self.assertRaises(TradeIngestionError):
            self.collect(Client([]), minimum_size="0.01")

    def test_compressed_pages_resume_and_verify_without_losing_rows(self):
        first = self.collect(Client([page([trade(), trade()], "next")]), max_pages=1, compress=True)
        path = self.output / CONDITION / first["pages"][0]["file"]
        self.assertTrue(path.name.endswith(".jsonl.gz"))
        self.assertEqual(len(gzip.decompress(path.read_bytes()).splitlines()), 2)
        final = self.collect(Client([page([trade(timestamp=1779999900)])]), compress=False)
        self.assertEqual(final["row_count"], 3)
        self.assertEqual(validate_collection(self.output, condition_id=CONDITION), final)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def collect(self, client, **kwargs):
        return ingest_condition(client, condition_id=CONDITION,
                                output_dir=self.output, **kwargs)

    def saved_state(self):
        return json.loads((self.output / CONDITION / "manifest.json").read_text())

    def all_rows(self, manifest):
        return [json.loads(line) for p in manifest["pages"]
                for line in (self.output / CONDITION / p["file"]).read_text().splitlines()]

    def test_resume_preserves_filters_and_cannot_certify_coverage(self):
        first = Client([page([trade()], "opaque cursor")])
        state = self.collect(first, limit=2, max_pages=1)
        self.assertEqual(state["api_traversal_status"], "paused")
        self.assertEqual(state["row_count"], 1)
        second = Client([page([trade(timestamp=1779999900)])])
        state = self.collect(second, limit=2)
        filters = first.calls[0][1]
        self.assertEqual(second.calls[0][1], {**filters, "cursor": "opaque cursor"})
        self.assertEqual(filters["taker_only"], "false")
        self.assertEqual(filters["filter_amount"], "0.01")
        self.assertNotIn("start", filters)
        self.assertNotIn("end", filters)
        self.assertEqual(state["api_traversal_status"], "exhausted")
        self.assertFalse(state["training_coverage_certified"])
        self.assertEqual(state["row_count"], 2)
        exhausted = Client([])
        self.assertEqual(self.collect(exhausted, limit=2), state)
        self.assertEqual(exhausted.calls, [])

    def test_identical_rows_and_two_wallet_legs_are_retained(self):
        state = self.collect(Client([page([trade(), trade(), trade(
            proxy_wallet="0x" + "d" * 40, side="SELL")])]))
        rows = self.all_rows(state)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r["observation_id"] for r in rows}), 3)
        self.assertEqual(len({r["transaction_hash"] for r in rows}), 1)
        self.assertEqual(rows[0]["size"], "1.25")
        self.assertEqual(rows[0]["price"], "0.4")
        self.assertIsNone(rows[0]["maker_taker_role"])
        self.assertIsNone(rows[0]["publicly_available_at_upper_bound"])
        self.assertNotIn("name", rows[0])
        self.assertEqual(rows[0]["identity_quality"], "api_observation")

    def test_repeated_cursor_does_not_commit_bad_page(self):
        client = Client([page([trade()], "cursor1"), page([trade()], "cursor1")])
        with self.assertRaisesRegex(TradeIngestionError, "Repeated"):
            self.collect(client)
        self.assertEqual(self.saved_state()["row_count"], 1)
        self.assertEqual(self.saved_state()["api_traversal_status"], "paused")
        self.assertEqual(len(list((self.output / CONDITION / "pages").glob("*.jsonl"))), 1)

    def test_different_condition_is_rejected_atomically(self):
        client = Client([page([trade(), trade(condition_id="0x" + "f" * 64)])])
        with self.assertRaisesRegex(TradeIngestionError, "different condition"):
            self.collect(client)
        self.assertEqual(self.saved_state()["row_count"], 0)
        self.assertEqual(list((self.output / CONDITION / "pages").glob("*.jsonl")), [])

    def test_empty_page_does_not_prove_absence_of_trading(self):
        state = self.collect(Client([page([])]))
        self.assertEqual(state["api_traversal_status"], "exhausted")
        self.assertEqual(state["row_count"], 0)
        self.assertIsNone(state["earliest_block_timestamp"])
        self.assertFalse(state["training_coverage_certified"])
        self.assertTrue(state["coverage_limitations"])

    def test_invalid_values_never_enter_saved_observations(self):
        for field, value in (("timestamp", 1.5), ("timestamp", True),
                             ("size", 0), ("size", float("nan")),
                             ("price", 1.01), ("side", "buy")):
            with self.subTest(field=field, value=value):
                with self.assertRaises(TradeIngestionError):
                    self.collect(Client([page([trade(**{field: value})])]))
                self.assertEqual(self.saved_state()["row_count"], 0)

    def test_page_recovery_after_crash_does_not_fetch_or_append_twice(self):
        import poly_world_cup.trades as module
        real_write = module._atomic_json

        def interrupted(path, value):
            if path.name == "manifest.json" and value["page_count"] == 1:
                raise OSError("simulated crash before manifest commit")
            real_write(path, value)

        with patch.object(module, "_atomic_json", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self.collect(Client([page([trade(), trade()])]))
        self.assertEqual(self.saved_state()["row_count"], 0)
        resumed = Client([])
        state = self.collect(resumed)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(state["row_count"], 2)
        self.assertEqual(len(self.all_rows(state)), 2)

    def test_saved_page_corruption_blocks_resume(self):
        state = self.collect(Client([page([trade()], "cursor1")]), max_pages=1)
        (self.output / CONDITION / state["pages"][0]["file"]).write_text("corrupt\n")
        with self.assertRaisesRegex(TradeIngestionError, "checksum"):
            self.collect(Client([]))

    def test_cannot_resume_with_changed_query_parameters(self):
        self.collect(Client([page([trade()], "cursor1")]), limit=10, max_pages=1)
        with self.assertRaisesRegex(TradeIngestionError, "incompatible"):
            self.collect(Client([]), limit=11)

    def test_paused_manifest_cannot_restart_terminal_traversal(self):
        self.collect(Client([page([trade()])]))
        state = self.saved_state()
        state["api_traversal_status"] = "paused"
        (self.output / CONDITION / "manifest.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(TradeIngestionError, "status contradicts"):
            self.collect(Client([]))

    def test_saved_page_cursor_cannot_revisit_an_earlier_page(self):
        self.collect(Client([page([trade()], "cursor1"), page([trade()], "cursor2")]),
                     max_pages=2)
        state = self.saved_state()
        state["pages"][1]["next_cursor"] = "cursor1"
        state["next_cursor"] = "cursor1"
        (self.output / CONDITION / "manifest.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(TradeIngestionError, "cursor progression"):
            self.collect(Client([]))

    def test_recovery_metadata_must_match_normalized_content(self):
        import poly_world_cup.trades as module
        real_write = module._atomic_json

        def interrupted(path, value):
            if path.name == "manifest.json" and value["page_count"] == 1:
                raise OSError("simulated crash")
            real_write(path, value)

        with patch.object(module, "_atomic_json", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.collect(Client([page([trade()])]))
        metadata = next((self.output / CONDITION / "pages").glob("*.json"))
        original = json.loads(metadata.read_text())
        for field, value in (("row_count", 0),
                             ("earliest_block_timestamp", "2026-01-01T00:00:00Z"),
                             ("has_more", "false"),
                             ("next_cursor", "unexpected")):
            with self.subTest(field=field):
                metadata.write_text(json.dumps({**original, field: value}))
                with self.assertRaises(TradeIngestionError):
                    self.collect(Client([]))
                self.assertEqual(self.saved_state()["page_count"], 0)
        metadata.write_text(json.dumps(original))
        self.assertEqual(self.collect(Client([]))["row_count"], 1)

    def test_invalid_pagination_cannot_be_treated_as_exhaustion(self):
        client = Client([{"data": [], "pagination": {"next_cursor": None}}])
        with self.assertRaisesRegex(TradeIngestionError, "has_more"):
            self.collect(client)
        self.assertEqual(self.saved_state()["api_traversal_status"], "paused")


if __name__ == "__main__":
    unittest.main()
