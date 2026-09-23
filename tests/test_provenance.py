"""Release evidence must survive refreshes and detect detached source data."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from poly_world_cup.http import HttpClient
from poly_world_cup.provenance import verify_raw_provenance
from poly_world_cup.trades import API_URL, ingest_condition


CONDITION = "0x" + "a" * 64
OTHER = "0x" + "d" * 64


class Response:
    headers = {"Date": "Wed, 23 Sep 2026 00:00:00 GMT"}

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def body(condition=CONDITION, *, cursor=None, price="0.12345678901234567890123456789", duplicate=False):
    row = (f'{{"proxy_wallet":"0x{"b" * 40}","condition_id":"{condition}",'
           f'"transaction_hash":"0x{"c" * 64}","token_id":"1234",'
           f'"side":"BUY","size":0.000001,"price":{price},"timestamp":1780000000}}')
    rows = ",".join([row, row] if duplicate else [row])
    pagination = json.dumps({"has_more": cursor is not None, "next_cursor": cursor})
    return f'{{"data":[{rows}],"pagination":{pagination}}}'.encode()


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trades = self.root / "trades"
        self.cache = self.root / "cache"

    def collect(self, bodies=None, *, condition=CONDITION, compress=False,
                layout="shared", max_pages=None):
        bodies = [body(condition)] if bodies is None else bodies
        cache = self.cache / condition if layout == "per_condition" else self.cache
        with patch("poly_world_cup.http.urlopen", side_effect=[Response(raw) for raw in bodies]):
            state = ingest_condition(HttpClient(cache, compress=compress), condition_id=condition,
                                     output_dir=self.trades, compress=compress, max_pages=max_pages,
                                     minimum_size="0.000001")
        return state, cache

    def verify(self, *, conditions=(CONDITION,), layout="shared"):
        return verify_raw_provenance(self.trades, self.cache, condition_ids=conditions,
                                     cache_layout=layout)

    def save_manifest(self, state, condition=CONDITION):
        (self.trades / condition / "manifest.json").write_text(json.dumps(state))

    def raw_path(self, state, cache, *, compressed=False):
        suffix = ".json.gz" if compressed else ".json"
        return cache / "bodies" / f"{state['pages'][0]['body_sha256']}{suffix}"

    def test_plain_and_compressed_replay_preserve_precise_values_duplicates_and_cursors(self):
        for compressed, condition in ((False, CONDITION), (True, OTHER)):
            with self.subTest(compressed=compressed):
                state, _ = self.collect([body(condition, cursor="opaque +/=", duplicate=True),
                                         body(condition)], condition=condition,
                                        compress=compressed, layout="per_condition")
                report = self.verify(conditions=[condition], layout="per_condition")
                self.assertTrue(report["raw_provenance_verified"], report["errors"])
                self.assertEqual(report["verified_raw_page_count"], 2)
                self.assertEqual(report["verified_observation_count"], 3)
                self.assertEqual(report["api_exhausted_conditions"], 1)
                self.assertFalse(report["source_completeness_verified"])
                self.assertFalse(report["training_coverage_certified"])
                self.assertEqual(state["row_count"], 3)

    def test_missing_or_corrupt_raw_body_fails_without_network_or_source_writes(self):
        state, cache = self.collect()
        path = self.raw_path(state, cache)
        for mode in ("corrupt", "missing"):
            with self.subTest(mode=mode), patch("poly_world_cup.http.urlopen") as network:
                if mode == "corrupt":
                    path.write_bytes(b'{}')
                else:
                    path.unlink()
                before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                report = self.verify()
                after = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                self.assertFalse(report["raw_provenance_verified"])
                self.assertEqual(report["verified_observation_count"], 0)
                self.assertEqual(before, after)
                network.assert_not_called()

    def test_truncated_gzip_is_a_reported_failure(self):
        state, cache = self.collect(compress=True)
        path = self.raw_path(state, cache, compressed=True)
        path.write_bytes(path.read_bytes()[:15])
        report = self.verify()
        self.assertFalse(report["raw_provenance_verified"])
        self.assertEqual(len(report["errors"]), 1)

    def test_invalid_compressed_payload_is_a_reported_failure(self):
        state, cache = self.collect(compress=True)
        path = self.raw_path(state, cache, compressed=True)
        payload = bytearray(path.read_bytes())
        payload[10] = 255  # Invalid DEFLATE block after the fixed gzip header.
        path.write_bytes(payload)
        report = self.verify()
        self.assertFalse(report["raw_provenance_verified"])
        self.assertEqual(len(report["errors"]), 1)

    def test_mutable_request_index_is_insufficient_without_immutable_capture(self):
        self.collect()
        for path in (self.cache / "captures").glob("*.json"):
            path.unlink()
        report = self.verify()
        self.assertFalse(report["raw_provenance_verified"])
        self.assertIn("immutable capture", report["errors"][0]["error"])

    def test_refresh_or_missing_request_index_retains_old_capture_validity(self):
        state, cache = self.collect()
        with patch("poly_world_cup.http.urlopen", return_value=Response(body(price="0.7"))):
            HttpClient(cache, refresh=True).get_json(API_URL, state["parameters"])
        self.assertTrue(self.verify()["raw_provenance_verified"])
        for path in (cache / "requests").glob("*.json"):
            path.unlink()
        self.assertTrue(self.verify()["raw_provenance_verified"])

    def test_modified_capture_metadata_without_matching_filename_fails(self):
        self.collect()
        path = next((self.cache / "captures").glob("*.json"))
        metadata = json.loads(path.read_text())
        metadata["response_headers"]["Date"] = "changed"
        path.write_text(json.dumps(metadata))
        self.assertFalse(self.verify()["raw_provenance_verified"])

    def test_rehashed_normalized_tampering_is_detected_by_source_replay(self):
        state, _ = self.collect()
        page = state["pages"][0]
        path = self.trades / CONDITION / page["file"]
        row = json.loads(path.read_text())
        row["price"] = "0.9"
        content = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path.write_bytes(content)
        page["normalized_sha256"] = hashlib.sha256(content).hexdigest()
        self.save_manifest(state)
        report = self.verify()
        self.assertEqual(report["normalized_integrity_verified_conditions"], 1)
        self.assertFalse(report["raw_provenance_verified"])
        self.assertIn("normalization replay", report["errors"][0]["error"])

    def test_request_url_must_match_actual_query_and_cursor(self):
        state, _ = self.collect()
        state["pages"][0]["request_url"] += "&cursor=unrequested"
        self.save_manifest(state)
        report = self.verify()
        self.assertFalse(report["raw_provenance_verified"])
        self.assertIn("query and cursor", report["errors"][0]["error"])

    def test_raw_cursor_must_match_self_consistent_manifest(self):
        state, _ = self.collect([body(cursor="actual")], max_pages=1)
        state["pages"][0]["next_cursor"] = state["next_cursor"] = "invented"
        self.save_manifest(state)
        report = self.verify()
        self.assertEqual(report["normalized_integrity_verified_conditions"], 1)
        self.assertFalse(report["raw_provenance_verified"])
        self.assertIn("pagination", report["errors"][0]["error"])

    def test_unchanged_normalized_rows_cannot_be_bound_to_a_different_capture(self):
        state, cache = self.collect()
        with patch("poly_world_cup.http.urlopen", return_value=Response(body(price="0.7"))):
            result = HttpClient(cache, refresh=True).get_json(API_URL, state["parameters"])
        page = state["pages"][0]
        page["body_sha256"], page["retrieved_at"] = result.body_sha256, result.retrieved_at
        self.save_manifest(state)
        report = self.verify()
        self.assertFalse(report["raw_provenance_verified"])
        self.assertIn("normalization replay", report["errors"][0]["error"])

    def test_paths_cannot_escape_cache_or_trade_directories(self):
        state, cache = self.collect()
        original = self.raw_path(state, cache)
        outside = self.root / "outside.json"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
        report = self.verify()
        self.assertFalse(report["raw_provenance_verified"])
        self.assertIn("escapes", report["errors"][0]["error"])
        with self.assertRaises(ValueError):
            self.verify(conditions=["../other"])

    def test_one_bad_condition_does_not_discard_another_conditions_evidence(self):
        state, cache = self.collect(layout="per_condition")
        self.collect(condition=OTHER, layout="per_condition")
        self.raw_path(state, cache).unlink()
        report = self.verify(conditions=[CONDITION, OTHER], layout="per_condition")
        self.assertFalse(report["raw_provenance_verified"])
        self.assertEqual(report["verified_condition_count"], 1)
        self.assertEqual(report["verified_observation_count"], 1)
        self.assertEqual(report["api_exhausted_conditions"], 2)

    def test_paused_traversal_can_have_valid_provenance_without_completeness(self):
        self.collect([body(cursor="next")], max_pages=1)
        report = self.verify()
        self.assertTrue(report["raw_provenance_verified"])
        self.assertEqual(report["api_exhausted_conditions"], 0)
        self.assertFalse(report["source_completeness_verified"])


if __name__ == "__main__":
    unittest.main()
