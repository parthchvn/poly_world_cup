import hashlib
import gzip
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from poly_world_cup.http import HttpClient, request_url


class Response:
    headers = {"Date": "Mon, 01 Jan 2024 00:00:00 GMT"}

    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class HttpTests(unittest.TestCase):
    def test_compressed_capture_replays_exact_bytes_and_decimals(self):
        body = b'{"size": 0.12345678901234567890123456789}'
        with tempfile.TemporaryDirectory() as folder, patch("poly_world_cup.http.urlopen", return_value=Response(body)) as fetch:
            first = HttpClient(Path(folder), compress=True).get_json("https://example.org/feed")
            second = HttpClient(Path(folder)).get_json("https://example.org/feed")
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(first.data, second.data)
            self.assertEqual(first.retrieved_at, second.retrieved_at)
            self.assertEqual(gzip.decompress((Path(folder) / "bodies" / f"{first.body_sha256}.json.gz").read_bytes()), body)

    def test_replay_retains_capture_time_and_exact_body(self):
        body = b'{"price": 0.12345678901234567890123456789, "rows": []}'
        with tempfile.TemporaryDirectory() as folder, patch("poly_world_cup.http.urlopen", return_value=Response(body)) as fetch:
            client = HttpClient(Path(folder))
            first = client.get_json("https://example.org/feed", {"offset": 0, "taker_only": False})
            second = client.get_json("https://example.org/feed", {"taker_only": False, "offset": 0})
            self.assertEqual(fetch.call_count, 1)
            self.assertTrue(second.from_cache)
            self.assertEqual(first.retrieved_at, second.retrieved_at)
            self.assertEqual(first.data["price"], Decimal("0.12345678901234567890123456789"))
            self.assertEqual(second.data["price"], first.data["price"])
            self.assertEqual(first.body_sha256, hashlib.sha256(body).hexdigest())
            self.assertEqual((Path(folder) / "bodies" / f"{first.body_sha256}.json").read_bytes(), body)

    def test_corrupt_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder, patch("poly_world_cup.http.urlopen", return_value=Response(b'{}')):
            client = HttpClient(Path(folder))
            first = client.get_json("https://example.org/feed")
            (Path(folder) / "bodies" / f"{first.body_sha256}.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "integrity"):
                client.get_json("https://example.org/feed")

    def test_refresh_preserves_prior_capture(self):
        with tempfile.TemporaryDirectory() as folder, patch("poly_world_cup.http.urlopen", side_effect=[Response(b'{"version":1}'), Response(b'{"version":2}')]):
            a = HttpClient(Path(folder)).get_json("https://example.org/feed")
            b = HttpClient(Path(folder), refresh=True).get_json("https://example.org/feed")
            self.assertNotEqual(a.body_sha256, b.body_sha256)
            self.assertEqual(len(list((Path(folder) / "captures").glob("*.json"))), 2)
            self.assertTrue((Path(folder) / "bodies" / f"{a.body_sha256}.json").exists())

    def test_url_filters_are_deterministic_and_boolean_lowercase(self):
        self.assertEqual(request_url("https://example.org/feed", {"taker_only": False, "cursor": "a+/="}), "https://example.org/feed?cursor=a%2B%2F%3D&taker_only=false")
        with self.assertRaises(ValueError):
            request_url("https://user:password@example.org/")


if __name__ == "__main__":
    unittest.main()
