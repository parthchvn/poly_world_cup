import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from poly_world_cup.batch import RequestPacer, run_batch


CONDITIONS = ["0x" + digit * 64 for digit in "abc"]


class Client:
    def __init__(self, rows):
        self.responses = list(rows)
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append(dict(params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        data = {"data": [{"proxy_wallet": "0x" + "d" * 40,
                          "condition_id": params["condition"], "token_id": "123",
                          "side": "BUY", "price": "0.4", "size": "12.5",
                          "timestamp": 1780000000, "transaction_hash": "0x" + "e" * 64}],
                "pagination": {"has_more": response is not None, "next_cursor": response}}
        body = json.dumps(data).encode()
        return SimpleNamespace(data=data, url=url + "?" + urlencode(params),
                               body_sha256=hashlib.sha256(body).hexdigest(),
                               retrieved_at="2026-09-23T00:00:00Z")


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_collection(self, conditions, clients, **kwargs):
        caches = []

        def factory(path):
            caches.append(path)
            return clients[path.name]

        kwargs.setdefault("requests_per_second", 100000)
        result = run_batch(conditions, output_dir=self.root / "trades",
                           cache_dir=self.root / "cache", client_factory=factory,
                           compress=False, **kwargs)
        return result, caches

    def test_concurrent_conditions_use_isolated_caches_and_keep_multiplicity(self):
        clients = {c: Client([None]) for c in CONDITIONS}
        result, caches = self.run_collection([*CONDITIONS, CONDITIONS[0].upper().replace("0X", "0x")], clients)
        self.assertEqual(result["condition_count"], 3)
        self.assertEqual(set(caches), {self.root / "cache" / c for c in CONDITIONS})
        self.assertEqual(result["validated_observation_count"], 3)
        self.assertEqual(result["status"], "api_exhausted")
        self.assertFalse(result["training_coverage_certified"])
        self.assertFalse(result["canonical_fill_identity_available"])
        self.assertEqual(result, json.loads((self.root / "trades" / "batch_progress.json").read_text()))

    def test_failure_is_isolated_and_retry_resumes_only_unfinished_condition(self):
        clients = {CONDITIONS[0]: Client(["next", OSError("offline")]),
                   CONDITIONS[1]: Client([None])}
        result, _ = self.run_collection(CONDITIONS[:2], clients)
        self.assertEqual(result["status"], "failed_or_partial")
        self.assertEqual(result["status_counts"]["failed"], 1)
        self.assertEqual(result["committed_observation_count"], 2)
        self.assertEqual(result["validated_observation_count"], 1)
        resumed = {CONDITIONS[0]: Client([None]), CONDITIONS[1]: Client([])}
        result, _ = self.run_collection(CONDITIONS[:2], resumed)
        self.assertEqual(result["validated_observation_count"], 3)
        self.assertEqual(resumed[CONDITIONS[0]].calls[0]["cursor"], "next")
        self.assertEqual(resumed[CONDITIONS[1]].calls, [])
        self.assertEqual(result["status"], "api_exhausted")

    def test_bounded_jobs_report_paused_and_do_not_claim_full_traversal(self):
        result, _ = self.run_collection([CONDITIONS[0]], {CONDITIONS[0]: Client(["next"])},
                                        max_pages_per_condition=1)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["status_counts"]["paused"], 1)
        self.assertFalse(result["training_coverage_certified"])

    def test_requested_minimum_size_is_sent_and_reported(self):
        client = Client([None])
        result, _ = self.run_collection([CONDITIONS[0]], {CONDITIONS[0]: client},
                                        minimum_size="0.000001")
        self.assertEqual(client.calls[0]["filter_amount"], "0.000001")
        self.assertEqual(result["requested_minimum_size_tokens"], "0.000001")

    def test_saved_report_cannot_hide_corrupt_previously_exhausted_collection(self):
        self.run_collection([CONDITIONS[0]], {CONDITIONS[0]: Client([None])})
        page = next((self.root / "trades" / CONDITIONS[0] / "pages").glob("*.jsonl"))
        page.write_text("corrupt\n")
        result, _ = self.run_collection([CONDITIONS[0]], {CONDITIONS[0]: Client([])})
        self.assertEqual(result["status_counts"]["failed"], 1)
        self.assertEqual(result["validated_observation_count"], 0)

    def test_progress_is_emitted_before_final_results(self):
        updates = []
        self.run_collection([CONDITIONS[0]], {CONDITIONS[0]: Client([None])},
                            on_progress=updates.append)
        self.assertEqual(updates[0]["status"], "running")
        self.assertEqual(updates[0]["status_counts"]["queued"], 1)
        self.assertEqual(updates[-1]["status"], "api_exhausted")
        self.assertIsNotNone(updates[-1]["finished_at"])

    def test_invalid_arguments_fail_before_creating_output(self):
        for conditions, options in (([], {}), (["../bad"], {}),
                                     (CONDITIONS, {"workers": 0}),
                                     (CONDITIONS, {"workers": 65}),
                                     (CONDITIONS, {"requests_per_second": float("nan")}),
                                     (CONDITIONS, {"max_pages_per_condition": 0})):
            with self.subTest(conditions=conditions, options=options):
                with self.assertRaises(ValueError):
                    self.run_collection(conditions, {}, **options)
                self.assertFalse((self.root / "trades").exists())

    def test_sixty_four_workers_remain_valid_with_global_request_pacing(self):
        result, _ = self.run_collection([CONDITIONS[0]], {CONDITIONS[0]: Client([None])}, workers=64)
        self.assertEqual(result["workers"], 64)
        self.assertEqual(result["status"], "api_exhausted")

    def test_pacing_reserves_distinct_start_times_across_callers(self):
        pacer = RequestPacer(2)
        with patch("poly_world_cup.batch.time.monotonic", return_value=100), \
                patch("poly_world_cup.batch.time.sleep") as sleep:
            pacer.acquire()
            pacer.acquire()
            pacer.acquire()
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
