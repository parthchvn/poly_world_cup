import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


spec = importlib.util.spec_from_file_location(
    "actor_builder_http", Path(__file__).resolve().parents[1] / "scripts/build_actor_dataset.py")
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


class CurlHTTPTests(unittest.TestCase):
    def run_response(self, status=200, body=b'{"size":0.123456789012345678901}'):
        def run(command, **kwargs):
            self.assertEqual(command[1], "--disable")
            self.assertNotIn("shell", kwargs)
            Path(command[command.index("--output") + 1]).write_bytes(body)
            Path(command[command.index("--dump-header") + 1]).write_bytes(
                b'HTTP/1.1 200 Connection established\r\nDate: proxy-date\r\n\r\n'
                + f'HTTP/2 {status}\r\ndate: origin-date\r\netag: test-etag\r\n\r\n'.encode())
            return subprocess.CompletedProcess(command, 0, str(status).encode(), b'')
        return run

    def test_curl_cache_can_be_replayed_by_urllib_without_network(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.shutil, 'which', return_value='/usr/bin/curl'), \
                patch.object(builder.subprocess, 'run', side_effect=self.run_response()) as run, \
                patch.object(builder, 'urlopen', side_effect=AssertionError('unexpected urllib')):
            root = Path(directory)
            first = builder.HttpClient(root, transport='curl', compress=True).get_json('https://example.org/trades')
            second = builder.HttpClient(root).get_json('https://example.org/trades')
            self.assertEqual(first.data['size'], Decimal('0.123456789012345678901'))
            self.assertEqual(first.data, second.data)
            self.assertEqual(first.body_sha256, second.body_sha256)
            self.assertEqual(first.retrieved_at, second.retrieved_at)
            self.assertTrue(second.from_cache)
            self.assertEqual(run.call_count, 1)
            metadata = json.loads(next((root / 'requests').glob('*.json')).read_text())
            self.assertEqual(metadata['response_headers']['Date'], 'origin-date')

    def test_curl_403_is_not_retried_or_cached(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.shutil, 'which', return_value='/usr/bin/curl'), \
                patch.object(builder.subprocess, 'run', side_effect=self.run_response(403, b'Forbidden')) as run:
            with self.assertRaises(HTTPError) as error:
                builder.HttpClient(Path(directory), transport='curl').get_json('https://example.org/trades')
            self.assertEqual(error.exception.code, 403)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((Path(directory) / 'requests').exists())

    def test_failed_transfer_retries_then_saves_only_complete_body(self):
        success = self.run_response()
        attempts = []
        def run(command, **kwargs):
            attempts.append(command)
            if len(attempts) == 1:
                Path(command[command.index('--output') + 1]).write_bytes(b'{"partial":')
                return subprocess.CompletedProcess(command, 56, b'200', b'Connection reset by peer')
            return success(command, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.shutil, 'which', return_value='/usr/bin/curl'), \
                patch.object(builder.subprocess, 'run', side_effect=run), \
                patch.object(builder.time, 'sleep'):
            response = builder.HttpClient(Path(directory), transport='curl').get_json('https://example.org/trades')
            self.assertEqual(len(attempts), 2)
            self.assertIn('size', response.data)
            self.assertEqual(len(list((Path(directory) / 'bodies').iterdir())), 1)

    def test_invalid_json_never_enters_cache(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.shutil, 'which', return_value='/usr/bin/curl'), \
                patch.object(builder.subprocess, 'run', side_effect=self.run_response(200, b'<html>error</html>')):
            with self.assertRaises(json.JSONDecodeError):
                builder.HttpClient(Path(directory), transport='curl').get_json('https://example.org/trades')
            self.assertFalse((Path(directory) / 'requests').exists())

    def test_missing_curl_is_actionable(self):
        with patch.object(builder.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'requires the curl executable'):
                builder.HttpClient(Path('unused'), transport='curl')


if __name__ == '__main__':
    unittest.main()
