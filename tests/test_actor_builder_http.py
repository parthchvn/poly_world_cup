import importlib.util
import io
import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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

    def test_persistent_reset_has_elapsed_budget_and_no_cache_entry(self):
        clock = [0.0]
        def pause(seconds):
            clock[0] += seconds
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', side_effect=URLError('Connection reset')) as fetch, \
                patch.object(builder.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(builder.time, 'sleep', side_effect=pause) as sleep, \
                patch('sys.stderr', new_callable=io.StringIO) as log:
            client = builder.HttpClient(Path(directory), retries=8, retry_delay=2, retry_budget=30, log_retries=True)
            with self.assertRaisesRegex(URLError, '30s request budget'):
                client.get_json('https://example.org/trades')
            self.assertEqual(fetch.call_count, 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8])
            self.assertIn('connection reset from example.org', log.getvalue())
            self.assertIn('request budget left', log.getvalue())
            self.assertFalse((Path(directory) / 'requests').exists())

    def test_attempt_timeouts_shrink_to_remaining_budget(self):
        clock, timeouts = [0.0], []
        with tempfile.TemporaryDirectory() as directory:
            client = builder.HttpClient(Path(directory), timeout=45, retry_delay=2, retry_budget=10)
            def fetch(request):
                timeouts.append(client._local.attempt_timeout)
                clock[0] += 4
                raise TimeoutError('timed out')
            with patch.object(builder.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(builder.time, 'sleep', side_effect=lambda s: clock.__setitem__(0, clock[0] + s)), \
                    patch.object(client, '_fetch', side_effect=fetch):
                with self.assertRaisesRegex(URLError, 'request budget'):
                    client.get_json('https://example.org/trades')
            self.assertEqual(timeouts, [10, 4])
            self.assertEqual(clock[0], 10)

    def test_shared_host_stops_after_repeated_transport_failures(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', side_effect=URLError('Connection reset')) as fetch, \
                patch.object(builder.time, 'sleep'):
            client = builder.HttpClient(Path(directory), retries=8, host_failure_limit=3)
            with self.assertRaisesRegex(URLError, 'stopped after 3 attempts'):
                client.get_json('https://example.org/first')
            with self.assertRaisesRegex(URLError, 'Stopped requests to example.org'):
                client.get_json('https://example.org/second')
            self.assertEqual(fetch.call_count, 3)

    def test_retry_after_is_respected(self):
        error = HTTPError('https://example.org/trades', 429, 'rate limit', {'retry-after': '15'}, None)
        clock = [0.0]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', side_effect=[error, (b'{}', {})]), \
                patch.object(builder.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(builder.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)) as sleep:
            builder.HttpClient(Path(directory), retry_delay=2).get_json('https://example.org/trades')
            sleep.assert_called_once_with(15)

    def test_long_retry_after_stops_instead_of_waiting_or_retrying_early(self):
        error = HTTPError('https://example.org/trades', 429, 'rate limit', {'Retry-After': '300'}, None)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', side_effect=error) as fetch, \
                patch.object(builder.time, 'sleep') as sleep:
            with self.assertRaisesRegex(URLError, 'cannot accommodate the next 300s wait'):
                builder.HttpClient(Path(directory), retry_budget=90).get_json('https://example.org/trades')
            self.assertEqual(fetch.call_count, 1)
            sleep.assert_not_called()

    def test_certificate_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.shutil, 'which', return_value='/usr/bin/curl'), \
                patch.object(builder.subprocess, 'run', return_value=subprocess.CompletedProcess([], 60, b'000', b'certificate verification failed')) as run:
            with self.assertRaises(builder.HttpTransportError) as error:
                builder.HttpClient(Path(directory), transport='curl').get_json('https://example.org/trades')
            self.assertEqual(run.call_count, 1)
            self.assertEqual(error.exception.category, 'TLS certificate verification')
            self.assertFalse(error.exception.retryable)
            self.assertFalse((Path(directory) / 'requests').exists())

    def test_dns_and_tls_reset_have_distinct_diagnostics(self):
        for curl_code, category in ((6, 'DNS resolution'), (35, 'TLS connection'), (56, 'receive/reset failure')):
            with self.subTest(curl_code=curl_code), tempfile.TemporaryDirectory() as directory, \
                    patch.object(builder.shutil, 'which', return_value='/usr/bin/curl'), \
                    patch.object(builder.subprocess, 'run', return_value=subprocess.CompletedProcess([], curl_code, b'000', b'failed')) as run, \
                    patch.object(builder.time, 'sleep'), patch('sys.stderr', new_callable=io.StringIO) as log:
                with self.assertRaisesRegex(URLError, category):
                    builder.HttpClient(Path(directory), transport='curl', retries=1, log_retries=True).get_json('https://example.org/trades')
                self.assertEqual(run.call_count, 2)
                self.assertIn(f'{category} (curl {curl_code})', log.getvalue())

    def test_concurrent_identical_requests_share_one_capture(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', return_value=(b'{"n":1}', {})) as fetch:
            client = builder.HttpClient(Path(directory), compress=True)
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: client.get_json('https://example.org/shared'), range(8)))
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(sum(row.from_cache for row in results), 7)
            self.assertEqual(len({row.retrieved_at for row in results}), 1)
            self.assertEqual(client.stats()['live_attempts'], 1)
            self.assertEqual(client.stats()['cache_hits'], 7)
            self.assertEqual(client.stats()['response_bytes'], len(b'{"n":1}'))

    def test_distinct_requests_can_transfer_concurrently(self):
        barrier = threading.Barrier(2)
        def fetch(request):
            barrier.wait(timeout=2)
            return b'{}', {}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', side_effect=fetch):
            client = builder.HttpClient(Path(directory))
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(client.get_json, ['https://example.org/one', 'https://example.org/two']))
            self.assertEqual(len(results), 2)
            self.assertEqual(client.stats()['responses'], 2)

    def test_cancellation_stops_before_network_request(self):
        canceled = threading.Event()
        canceled.set()
        with tempfile.TemporaryDirectory() as directory, patch.object(builder.HttpClient, '_fetch') as fetch:
            with self.assertRaisesRegex(URLError, 'collection canceled'):
                builder.HttpClient(Path(directory), cancel_event=canceled).get_json('https://example.org/trades')
            fetch.assert_not_called()

    def test_only_live_attempts_are_paced(self):
        clock = [100.0]
        def fetch(request):
            clock[0] += 0.25
            return b'{}', {}
        def sleep(seconds):
            clock[0] += seconds
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.HttpClient, '_fetch', side_effect=fetch) as network, \
                patch.object(builder.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(builder.time, 'sleep', side_effect=sleep) as pause:
            client = builder.HttpClient(Path(directory), min_interval=1)
            client.get_json('https://example.org/first')
            client.get_json('https://example.org/first')
            client.get_json('https://example.org/second')
            self.assertEqual(network.call_count, 2)
            pause.assert_called_once_with(1)

    def test_collection_resumes_after_mid_pagination_network_failure(self):
        from tests.test_trades import Client, CONDITION, page, trade
        class FailingClient(Client):
            def get_json(self, url, params=None):
                if not self.responses:
                    raise URLError('Connection reset')
                return super().get_json(url, params)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = FailingClient([page([trade()], 'saved-cursor')])
            with self.assertRaises(URLError):
                builder.ingest_condition(first, condition_id=CONDITION, output_dir=root)
            manifest = json.loads((root / CONDITION / 'manifest.json').read_text())
            original = (root / CONDITION / manifest['pages'][0]['file']).read_bytes()
            resumed = Client([page([trade(timestamp=1779999900)])])
            result = builder.ingest_condition(resumed, condition_id=CONDITION, output_dir=root)
            self.assertEqual(resumed.calls[0][1]['cursor'], 'saved-cursor')
            self.assertEqual(result['row_count'], 2)
            self.assertEqual(result['api_traversal_status'], 'exhausted')
            self.assertEqual(original, (root / CONDITION / result['pages'][0]['file']).read_bytes())


if __name__ == '__main__':
    unittest.main()
