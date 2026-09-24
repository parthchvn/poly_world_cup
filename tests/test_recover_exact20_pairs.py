"""Focused integrity checks for the original archive's exactly-20 gap."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from poly_world_cup.http import FetchResult, request_url
from poly_world_cup.trades import API_URL, TradeIngestionError

_SPEC = importlib.util.spec_from_file_location(
    "recover_exact20_pairs", Path(__file__).parents[1] / "scripts" / "recover_exact20_pairs.py")
recovery = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recovery)

WALLET = "0x" + "1" * 40
CONDITION = "0x" + "2" * 64
OTHER_CONDITION = "0x" + "3" * 64
CAPTURED = "2026-09-24T00:00:00Z"


def envelope():
    # Repeated economic records are distinct provider observations. Retaining
    # row multiplicity is required to reconcile the full count ledger.
    row = {"condition_id": CONDITION, "proxy_wallet": WALLET,
           "transaction_hash": "0x" + "4" * 64, "token_id": "123",
           "side": "BUY", "size": "90.909089", "price": "0.4399999648",
           "timestamp": 1781542800}
    return {"data": [deepcopy(row) for _ in range(20)],
            "pagination": {"has_more": False, "next_cursor": None}}


def response(payload=None, url=None):
    payload = envelope() if payload is None else payload
    body = json.dumps(payload).encode()
    return FetchResult(payload, url or request_url(API_URL, recovery.parameters(WALLET, CONDITION)),
                       CAPTURED, hashlib.sha256(body).hexdigest(), True)


class Exact20RecoveryTests(unittest.TestCase):
    def test_repeated_economic_rows_and_decimal_strings_are_preserved(self):
        rows = recovery.validate_response(response(), WALLET, CONDITION, {"123", "456"})
        self.assertEqual(len({r["observation_id"] for r in rows}), 20)
        self.assertEqual({r["price"] for r in rows}, {"0.4399999648"})
        self.assertEqual({r["size"] for r in rows}, {"90.909089"})

    def test_inexact_count_and_nonterminal_capture_fail_closed(self):
        for size in (19, 21):
            payload = envelope()
            payload["data"] = [payload["data"][0]] * size
            with self.subTest(size=size), self.assertRaises(TradeIngestionError):
                recovery.validate_response(response(payload), WALLET, CONDITION, {"123"})
        payload = envelope()
        payload["pagination"] = {"has_more": True, "next_cursor": "more"}
        with self.assertRaises(TradeIngestionError):
            recovery.validate_response(response(payload), WALLET, CONDITION, {"123"})

    def test_query_wallet_condition_token_and_size_are_bound(self):
        for key, value in (("proxy_wallet", "0x" + "8" * 40),
                           ("condition_id", OTHER_CONDITION),
                           ("token_id", "987"), ("size", "0.0000001")):
            payload = envelope()
            payload["data"][0][key] = value
            with self.subTest(key=key), self.assertRaises(TradeIngestionError):
                recovery.validate_response(response(payload), WALLET, CONDITION, {"123"})
        with self.assertRaisesRegex(TradeIngestionError, "URL"):
            recovery.validate_response(response(url=API_URL), WALLET, CONDITION, {"123"})

    def _cache(self, cache_root):
        result = response()
        pair_cache = cache_root / CONDITION / WALLET
        (pair_cache / "requests").mkdir(parents=True)
        (pair_cache / "bodies").mkdir()
        body = json.dumps(result.data).encode()
        (pair_cache / "bodies" / f"{result.body_sha256}.json.gz").write_bytes(gzip.compress(body))
        request_hash = hashlib.sha256(result.url.encode()).hexdigest()
        path = pair_cache / "requests" / f"{request_hash}.json"
        path.write_text(json.dumps({"url": result.url, "retrieved_at": CAPTURED,
                                   "body_sha256": result.body_sha256, "body_compression": "gzip"}))
        return pair_cache, result, path

    def test_committed_pair_replays_raw_cache_and_detects_normalized_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._cache(root / "cache")
            kwargs = dict(output_dir=root / "output", cache_dir=root / "cache", token_ids={"123"})
            result = recovery.recover_pair(WALLET, CONDITION, **kwargs)
            repeated = recovery.recover_pair(WALLET, CONDITION, **kwargs)
            self.assertEqual(result, repeated)
            manifest_path = root / "output" / result["manifest"]
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["raw_replay_verified"])
            self.assertFalse(manifest["training_coverage_certified"])
            (manifest_path.parent / "observations.jsonl.gz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(TradeIngestionError, "differs"):
                recovery.recover_pair(WALLET, CONDITION, **kwargs)

    def test_missing_or_corrupted_original_cache_cannot_be_refetched(self):
        for failure in ("missing_request", "corrupted_body"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pair_cache, result, request_path = self._cache(root / "cache")
                kwargs = dict(output_dir=root / "output", cache_dir=root / "cache", token_ids={"123"})
                recovery.recover_pair(WALLET, CONDITION, **kwargs)
                if failure == "missing_request":
                    request_path.unlink()
                else:
                    (pair_cache / "bodies" / f"{result.body_sha256}.json.gz").write_bytes(gzip.compress(b"{}"))
                with self.assertRaises(ValueError):
                    recovery.recover_pair(WALLET, CONDITION, **kwargs)

    def test_read_only_import_adapter_replays_without_network_and_rejects_manifest_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._cache(root / "cache")
            recovery.recover_pair(WALLET, CONDITION, output_dir=root / "output",
                                  cache_dir=root / "cache", token_ids={"123"})
            with patch("poly_world_cup.http.urlopen", side_effect=AssertionError("Network forbidden")):
                rows, proof = recovery.verify_saved_pair(root / "output", root / "cache",
                                                         WALLET, CONDITION, {"123"})
            self.assertEqual(len(rows), 20)
            self.assertEqual(proof["page"]["retrieved_at"], CAPTURED)
            self.assertTrue(Path(proof["raw_path"]).is_file())
            manifest_path = Path(proof["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["parameters"]["taker_only"] = True
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TradeIngestionError, "manifest differs"):
                recovery.verify_saved_pair(root / "output", root / "cache", WALLET, CONDITION, {"123"})

    def test_pair_selection_excludes_replacements_and_non_twenty_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trade_count_ledger").mkdir()
            (root / "replacement_collection.json").write_text(json.dumps({
                "conditions": [{"condition_id": OTHER_CONDITION}]}))
            rows = [{"wallet": WALLET, "condition_id": CONDITION, "observation_count": 20},
                    {"wallet": WALLET, "condition_id": OTHER_CONDITION, "observation_count": 20},
                    {"wallet": "0x" + "9" * 40, "condition_id": CONDITION, "observation_count": 21}]
            ledger = root / "trade_count_ledger" / "part-00000.jsonl.gz"
            ledger.write_bytes(gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode()))
            self.assertEqual(recovery.selected_pairs(root), [(WALLET, CONDITION)])


if __name__ == "__main__":
    unittest.main()
