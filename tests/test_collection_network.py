import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('collection_network_tested',
    Path(__file__).resolve().parents[1] / 'tools/check_collection_network.py')
network = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(network)


class NetworkTests(unittest.TestCase):
    def test_distinguishes_dns_tls_and_http_failure_without_retries(self):
        for code, status, failure in ((6, '000', 'dns'), (35, '000', 'tls_connect'),
                                      (0, '429', 'http_429'), (60, '000', 'tls_certificate')):
            with self.subTest(code=code, status=status):
                response = SimpleNamespace(returncode=code, stdout=status, stderr='diagnostic')
                with patch.object(network.subprocess, 'run', return_value=response) as run:
                    result = network.probe(('price_history', 'https://example.org', 'history'))
                self.assertFalse(result['ok'])
                self.assertEqual(result['failure'], failure)
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertIn('--max-time', command)
                self.assertNotIn('--insecure', command)

    def test_http_200_must_contain_expected_json_shape(self):
        for payload, success in (({'history': []}, True), ({'error': 'blocked'}, False)):
            def respond(command, **kwargs):
                Path(command[command.index('--output') + 1]).write_text(json.dumps(payload))
                return SimpleNamespace(returncode=0, stdout='200\t1.2.3.4\t.01\t.02\t.1', stderr='')
            with self.subTest(payload=payload), patch.object(network.subprocess, 'run', side_effect=respond):
                result = network.probe(('price_history', 'https://example.org', 'history'))
            self.assertEqual(result['ok'], success)

    def test_process_timeout_returns_diagnostic(self):
        with patch.object(network.subprocess, 'run', side_effect=subprocess.TimeoutExpired('curl', 17)):
            result = network.probe(('price_history', 'https://example.org', 'history'))
        self.assertEqual(result['failure'], 'process_timeout')

    def test_failed_first_round_stops_and_does_not_retry_other_rounds(self):
        with patch.object(network.shutil, 'which', return_value='/usr/bin/curl'), \
             patch.object(network, 'endpoints', return_value=[('a', 'url', 'data')]), \
             patch.object(network, 'probe', return_value={'ok': False, 'failure': 'dns'}) as probe:
            result = network.check_network(repeats=3)
        self.assertFalse(result['ok'])
        probe.assert_called_once()

    def test_registered_endpoints_include_real_market_and_bounded_wallet_window(self):
        endpoints = network.endpoints(wallet='0x' + 'a' * 40)
        self.assertEqual(len(endpoints), 4)
        wallet_url = endpoints[-1][1]
        self.assertIn('start=1&end=', wallet_url)
        self.assertIn('limit=1', wallet_url)


if __name__ == '__main__':
    unittest.main()
