import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import urlencode
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / 'tools' / 'wallet_history.py'
SPEC = importlib.util.spec_from_file_location('wallet_history_test_module', MODULE)
wallet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wallet)
ACTOR = '0x' + '1' * 40
OTHER = '0x' + '2' * 40
CONDITION = '0x' + 'a' * 64
CONDITION2 = '0x' + 'b' * 64
COMBO_CONDITION = '0x03c536d68f3e625e61e694424ca23fa0740000000000000000000000000000'


def trade(when=100, **changes):
    return {'proxy_wallet': ACTOR, 'condition_id': CONDITION,
            'transaction_hash': '0x' + 'c' * 64, 'token_id': '1234',
            'timestamp': when, 'side': 'BUY', 'size': '2.5000', 'price': '0.3',
            'outcome': 'Yes', **changes}


def page(rows, cursor=None):
    return {'data': rows, 'pagination': {'has_more': cursor is not None, 'next_cursor': cursor}}


class Client:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []

    def get_json(self, url, params):
        self.requests.append((url, params.copy()))
        if not self.pages:
            raise AssertionError('Unexpected request')
        response = self.pages.pop(0)
        if isinstance(response, Exception):
            raise response
        body = json.dumps(response).encode()
        return SimpleNamespace(data=response, url=url + '?' + urlencode(sorted(params.items())),
                               body_sha256=hashlib.sha256(body).hexdigest(),
                               retrieved_at='2026-09-28T00:00:00Z')


class WalletHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def ingest(self, client, **kwargs):
        return wallet.ingest_wallet(client, actor_id=ACTOR, output_dir=self.directory,
                                    end_seconds=1000, **kwargs)

    def test_all_markets_bound_filters_and_opaque_pagination(self):
        client = Client([page([trade(100), trade(101, condition_id=CONDITION2)], 'next'),
                         page([trade(99)])])
        result = self.ingest(client)
        rows = list(wallet.iter_wallet_observations(result))
        self.assertEqual(len(rows), 3)
        self.assertEqual({r['condition_id'] for r in rows}, {CONDITION, CONDITION2})
        first, second = [params for _, params in client.requests]
        self.assertEqual(second, {**first, 'cursor': 'next'})
        self.assertEqual(first['start'], 1)
        self.assertEqual(first['end'], 1000)
        self.assertFalse(first['taker_only'])
        self.assertNotIn('condition', first)
        self.assertFalse(result['historical_completeness_verified'])
        self.assertEqual(result['api_traversal_status'], 'exhausted')

    def test_equal_rows_preserve_multiplicity_with_distinct_observation_ids(self):
        result = self.ingest(Client([page([trade(), trade()], 'next'), page([trade()])]))
        rows = list(wallet.iter_wallet_observations(result))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r['execution_id'] for r in rows}), 3)
        self.assertTrue(all(r['shares'] == '2.5' for r in rows))

    def test_reported_combo_execution_is_retained_without_padding_or_outcome(self):
        row = trade(100, condition_id=COMBO_CONDITION,
                    token_id='1705385896327775440347894695719834322871254396457394571926279144725081489408',
                    size='8.932559', price='0.2238998925', outcome='',
                    transaction_hash='0xfbeec34f01e8a78e14910f3dbb29d4d55e3f15f21358e241c28cdcfff1275602')
        result = self.ingest(Client([page([trade(99), row])]))
        rows = list(wallet.iter_wallet_observations(result))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['condition_id'], COMBO_CONDITION)
        self.assertEqual(len(bytes.fromhex(rows[1]['condition_id'][2:])), 31)
        self.assertEqual(rows[1]['shares'], '8.932559')
        self.assertEqual(rows[1]['price'], '0.2238998925')
        self.assertEqual(rows[1]['token_id'], row['token_id'])
        self.assertNotIn('outcome', rows[1])

    def test_short_condition_resumes_after_existing_binary_page_and_reuses_cache(self):
        first = self.ingest(Client([page([trade()], 'second')]), max_pages=1)
        first_page = Path(first['capture_dir']) / first['pages'][0]['file']
        original = first_page.read_bytes()
        client = Client([page([trade(99, condition_id=COMBO_CONDITION)])])
        result = self.ingest(client)
        self.assertEqual(client.requests[0][1]['cursor'], 'second')
        self.assertEqual(first_page.read_bytes(), original)
        self.assertEqual([r['condition_id'] for r in wallet.iter_wallet_observations(result)],
                         [CONDITION, COMBO_CONDITION])
        self.assertEqual(self.ingest(Client([])), result)

    def test_malformed_condition_ids_and_short_transaction_hashes_still_fail(self):
        for condition in (None, '', '0x' + 'a' * 60, '0x' + 'a' * 63,
                          '0x' + 'a' * 66, '0x' + 'g' * 62):
            with self.subTest(condition=condition), self.assertRaisesRegex(ValueError, 'condition_id'):
                self.ingest(Client([page([trade(condition_id=condition)])]))
        with self.assertRaisesRegex(ValueError, 'transaction_hash'):
            self.ingest(Client([page([trade(transaction_hash=COMBO_CONDITION)])]))

    def test_no_fake_publication_time_or_accounting(self):
        result = self.ingest(Client([page([trade()])]))
        row = next(wallet.iter_wallet_observations(result))
        self.assertEqual(row['timestamp'], '1970-01-01T00:01:40Z')
        self.assertEqual(row['retrieved_at'], '2026-09-28T00:00:00Z')
        self.assertIsNone(row['publicly_available_at_upper_bound'])
        self.assertEqual(row['availability_semantics'], wallet.AVAILABILITY)
        self.assertNotIn('known_at', row)
        self.assertNotIn('realized_pnl', row)

    def test_source_execution_id_and_provenance_preserved(self):
        result = self.ingest(Client([page([trade(execution_id='fill123', log_index=42)])]))
        row = next(wallet.iter_wallet_observations(result))
        self.assertEqual(row['execution_id'], 'api_execution_id:fill123')
        self.assertEqual(row['source']['source_ids']['log_index'], '42')
        self.assertEqual(row['source']['identity_quality'], 'source_execution_id')

    def test_duplicate_source_execution_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.ingest(Client([page([trade(fill_id='same'), trade(fill_id='same')])]))

    def test_resume_does_not_replay_committed_page(self):
        first = self.ingest(Client([page([trade()], 'second')]), max_pages=1)
        self.assertEqual(first['api_traversal_status'], 'paused')
        client = Client([page([trade(99)])])
        result = self.ingest(client)
        self.assertEqual(client.requests[0][1]['cursor'], 'second')
        self.assertEqual(len(list(wallet.iter_wallet_observations(result))), 2)
        again = self.ingest(Client([]))
        self.assertEqual(again, result)

    def test_network_failure_does_not_commit_partial_page(self):
        with self.assertRaises(OSError):
            self.ingest(Client([page([trade()], 'second'), OSError('reset')]))
        result = self.ingest(Client([page([trade(99)])]))
        self.assertEqual(result['row_count'], 2)

    def test_orphan_page_recovers_without_http(self):
        real_write = wallet._write
        counter = [0]
        def fail_manifest(path, value):
            if path.name == 'manifest.json':
                counter[0] += 1
                if counter[0] == 2:
                    raise OSError('interrupted after durable page')
            return real_write(path, value)
        with patch.object(wallet, '_write', side_effect=fail_manifest):
            with self.assertRaises(OSError):
                self.ingest(Client([page([trade()])]))
        result = self.ingest(Client([]))
        self.assertEqual(result['row_count'], 1)

    def test_paused_capture_cannot_feed_metrics(self):
        result = self.ingest(Client([page([trade()], 'second')]), max_pages=1)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            list(wallet.iter_wallet_observations(result))

    def test_empty_exhaustion_is_audited_but_required_actor_fails(self):
        with self.assertRaisesRegex(ValueError, 'no executions'):
            self.ingest(Client([page([])]))
        result = self.ingest(Client([]), require_nonempty=False)
        self.assertEqual(result['row_count'], 0)
        self.assertEqual(result['api_traversal_status'], 'exhausted')
        with self.assertRaisesRegex(ValueError, 'no executions'):
            list(wallet.iter_wallet_observations(result))

    def test_cursor_self_loop_fails_without_committing_second_page(self):
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.ingest(Client([page([trade()], 'second'), page([trade(99)], 'second')]))
        result = self.ingest(Client([page([trade(99)])]))
        self.assertEqual(result['page_count'], 2)

    def test_longer_cursor_loop_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.ingest(Client([page([trade()], 'a'), page([trade(99)], 'b'),
                                page([trade(98)], 'a')]))

    def test_nonterminal_empty_page_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no observations'):
            self.ingest(Client([page([], 'next')]))

    def test_contradictory_pagination_rejected(self):
        for response in ({'data': [trade()]},
                         {'data': [trade()], 'pagination': {'has_more': False, 'next_cursor': 'next'}},
                         {'data': [trade()], 'pagination': {'has_more': 'true', 'next_cursor': 'next'}}):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    self.ingest(Client([response]))

    def test_wrong_actor_or_out_of_bounds_rejected(self):
        for row in (trade(proxy_wallet=OTHER), trade(1001), trade(0)):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    self.ingest(Client([page([row])]))

    def test_bounds_are_inclusive(self):
        result = self.ingest(Client([page([trade(1), trade(1000)])]))
        self.assertEqual(result['row_count'], 2)

    def test_invalid_execution_economics_rejected(self):
        for field, value in [('size', '0'), ('size', '-1'), ('price', '1.1'),
                             ('price', 'NaN'), ('price', True), ('side', 'REDEEM'),
                             ('condition_id', '../bad'), ('token_id', '0')]:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    self.ingest(Client([page([trade(**{field: value})])]))

    def test_checksum_tampering_rejected(self):
        result = self.ingest(Client([page([trade()])]))
        path = Path(result['capture_dir']) / result['pages'][0]['file']
        content = json.loads(path.read_text())
        content['rows'][0]['price'] = '0.9'
        path.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            list(wallet.iter_wallet_observations(result))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.ingest(Client([]))

    def test_changed_window_creates_independent_immutable_capture(self):
        before = self.ingest(Client([page([trade()])]))
        later = wallet.ingest_wallet(Client([page([trade()])]), actor_id=ACTOR,
                                     output_dir=self.directory, end_seconds=2000)
        self.assertNotEqual(before['capture_dir'], later['capture_dir'])
        self.assertEqual(len(list(wallet.iter_wallet_observations(before))), 1)

    def test_zero_start_cannot_accidentally_use_three_year_default(self):
        with self.assertRaisesRegex(ValueError, 'positive epoch'):
            self.ingest(Client([]), start_seconds=0)

    def test_manifest_path_is_accepted_by_reader(self):
        result = self.ingest(Client([page([trade()])]))
        path = Path(result['capture_dir']) / 'manifest.json'
        self.assertEqual(len(list(wallet.iter_wallet_observations(path))), 1)


if __name__ == '__main__':
    unittest.main()
