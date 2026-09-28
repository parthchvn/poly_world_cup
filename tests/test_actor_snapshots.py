"""Current actor positions stay complete, scoped, and outside historical inputs."""
import copy
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

from tests import actor_snapshot_fixtures as fixtures
from tests import test_prepare_actor_sft as sft_fixtures

builder = sft_fixtures.builder


class ActorSnapshotCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.actor = '0x' + 'a' * 40
        self.other = '0x' + 'b' * 40
        self.market = {
            'market_id': '1', 'condition_id': '0x' + '1' * 64,
            'tokens': [{'outcome': 'Yes', 'token_id': '101'},
                       {'outcome': 'No', 'token_id': '102'}],
        }
        self.args = SimpleNamespace(
            actor_snapshots_dir=None, actor_snapshot_workers=1,
            cache=self.root / 'cache', http_transport='urllib', http_timeout=2,
            http_retries=0, http_retry_delay=0, http_min_interval=0,
            refresh=False, refresh_cache=False,
        )
        self.output = self.root / 'output'
        self.requests = []

    def collect(self, actors=None, output=None):
        with patch('sys.stdout', new_callable=io.StringIO), patch.object(builder.time, 'sleep'):
            return builder.collect_actor_snapshots(actors or [self.actor], self.market,
                                                   self.args, output or self.output)

    def read(self, output=None):
        return json.loads(((output or self.output) / 'actor_snapshots' /
                           (self.actor + '.json')).read_text())

    @staticmethod
    def page(rows, cursor=None):
        return {'data': rows, 'pagination': {'limit': 500, 'offset': 0,
                                            'has_more': cursor is not None,
                                            'next_cursor': cursor}}

    def fetcher(self, *, opens=None, closed=None, value=Decimal('0'), change=None):
        def fetch(request):
            url = request.full_url
            query = {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}
            self.requests.append((urlsplit(url).path, query))
            if urlsplit(url).path == '/v2/value':
                data = {'data': {'proxy_wallet': query['user'], 'value': value,
                                 'upstream_extra': 'keep value metadata'}}
            else:
                source = opens if query['status'] == 'OPEN' else closed
                if isinstance(source, dict):
                    data = source.get(query.get('cursor'), self.page([]))
                else:
                    data = self.page(source or [])
            if change:
                data = change(urlsplit(url).path, query, copy.deepcopy(data))
            # Numbers must arrive through the real Decimal-aware HTTP parser.
            encoded = json.dumps(data, default=lambda item: float(item) if isinstance(item, Decimal) else item)
            return encoded.encode(), {}
        return fetch

    def test_zero_value_and_empty_positions_are_successful_data(self):
        with patch.object(builder.HttpClient, '_fetch', side_effect=self.fetcher()):
            report = self.collect()
        self.assertEqual(report['version'], 1)
        self.assertEqual(report['actors'], 1)
        self.assertFalse(report['historical_model_input'])
        snapshot = self.read()
        self.assertEqual(Decimal(snapshot['actor_market_value']['data']['value']), 0)
        self.assertEqual(snapshot['actor_market_value']['data']['upstream_extra'], 'keep value metadata')
        for feature in ('actor_positions_open', 'actor_positions_closed'):
            self.assertEqual(snapshot[feature]['status'], 'ok')
            self.assertEqual(snapshot[feature]['data'], [])
        self.assertEqual(snapshot['temporal_scope'], 'collection_time_not_trade_time')
        for _, params in self.requests:
            self.assertEqual(params['condition'], self.market['condition_id'])
            self.assertEqual(params['user'], self.actor)
        positions = [params for path, params in self.requests if path == '/v2/positions']
        self.assertEqual({params['status'] for params in positions}, {'OPEN', 'CLOSED'})
        for params in positions:
            self.assertEqual(params['filter_type'], 'TOKENS')
            self.assertEqual(Decimal(params['filter_amount']), Decimal('0.000001'))
            if params['status'] == 'OPEN':
                self.assertEqual(params['include_archived'], 'true')
            else:
                self.assertNotIn('include_archived', params)

    def test_all_pages_and_original_position_fields_are_retained(self):
        yes = fixtures.position_fixture(self.actor, self.market, unrealized_pnl='1.234567891234',
                                        upstream_extra={'retained': True})
        no = fixtures.position_fixture(self.actor, self.market, token_id='102')
        closed = fixtures.position_fixture(self.actor, self.market, status='CLOSED')
        pages = {None: self.page([yes], 'next-page'), 'next-page': self.page([no])}
        with patch.object(builder.HttpClient, '_fetch', side_effect=self.fetcher(opens=pages, closed=[closed])):
            self.collect()
        snapshot = self.read()
        self.assertEqual(snapshot['actor_positions_open']['data'], [yes, no])
        self.assertEqual(snapshot['actor_positions_closed']['data'], [closed])
        self.assertEqual(len(snapshot['actor_positions_open']['pages']), 2)
        followup = [q for _, q in self.requests if q.get('cursor') == 'next-page']
        self.assertEqual(len(followup), 1)
        self.assertEqual(followup[0]['condition'], self.market['condition_id'])
        self.assertEqual(followup[0]['user'], self.actor)
        self.assertEqual(followup[0]['status'], 'OPEN')

    def test_cache_resume_retains_fetch_time_and_makes_no_new_requests(self):
        with patch.object(builder.HttpClient, '_fetch', side_effect=self.fetcher()):
            self.collect()
        first = self.read()
        other_output = self.root / 'resumed'
        with patch.object(builder.HttpClient, '_fetch', side_effect=AssertionError('Must reuse cache')):
            self.collect(output=other_output)
        second = self.read(other_output)
        for feature in fixtures.FEATURES:
            self.assertEqual(first[feature]['data'], second[feature]['data'])
            self.assertEqual(first[feature]['pages'][0]['retrieved_at'],
                             second[feature]['pages'][0]['retrieved_at'])
            self.assertEqual(first[feature]['pages'][0]['body_sha256'],
                             second[feature]['pages'][0]['body_sha256'])
            self.assertTrue(second[feature]['pages'][0]['from_cache'])

    def test_numeric_api_precision_and_resolved_open_holdings_are_preserved(self):
        exact = '0.12345678901234567890123456789'
        winner = fixtures.position_fixture(self.actor, self.market, status='REDEEMABLE',
                                           redeemable=True, current_value='3', current_price='1')
        loser = fixtures.position_fixture(self.actor, self.market, status='REDEEMABLE_LOST',
                                          token_id='102', current_value='0', current_price='0',
                                          archived=True)
        basic_fetcher = self.fetcher(opens=[winner, loser])

        def numeric_fetch(request):
            if urlsplit(request.full_url).path == '/v2/value':
                return ('{"data":{"proxy_wallet":"' + self.actor + '","value":' + exact + '}}').encode(), {}
            return basic_fetcher(request)

        with patch.object(builder.HttpClient, '_fetch', side_effect=numeric_fetch):
            self.collect()
        snapshot = self.read()
        self.assertEqual(snapshot['actor_market_value']['data']['value'], exact)
        self.assertEqual(snapshot['actor_positions_open']['data'], [winner, loser])

    def test_request_failure_is_not_replaced_with_empty_positions_or_zero_value(self):
        with patch.object(builder.HttpClient, '_fetch', side_effect=URLError('offline')):
            with self.assertRaises((URLError, ValueError, RuntimeError)):
                self.collect()
        self.assertFalse((self.output / 'actor_snapshots' / (self.actor + '.json')).exists())

    def test_wrong_wallet_condition_token_and_status_abort_collection(self):
        for field, value in [('proxy_wallet', self.other), ('condition_id', '0x' + '2' * 64),
                             ('token_id', 'unknown-token'), ('status', 'CLOSED')]:
            with self.subTest(field=field):
                self.args.cache = self.root / ('cache_bad_' + field)
                row = fixtures.position_fixture(self.actor, self.market, **{field: value})
                with patch.object(builder.HttpClient, '_fetch', side_effect=self.fetcher(opens=[row])):
                    with self.assertRaises(ValueError):
                        self.collect(output=self.root / ('out_bad_' + field))

    def test_wrong_value_wallet_aborts_even_with_other_valid_data(self):
        def change(path, query, data):
            if path == '/v2/value':
                data['data']['proxy_wallet'] = self.other
            return data
        with patch.object(builder.HttpClient, '_fetch', side_effect=self.fetcher(change=change)):
            with self.assertRaises(ValueError):
                self.collect()

    def test_repeated_cursor_and_inconsistent_pagination_abort(self):
        row = fixtures.position_fixture(self.actor, self.market)
        examples = [
            {None: self.page([row], 'loop'), 'loop': self.page([], 'loop')},
            {None: {'data': [row], 'pagination': {'has_more': True, 'next_cursor': None}}},
            {None: {'data': [row], 'pagination': {'has_more': False, 'next_cursor': 'unexpected'}}},
        ]
        for index, pages in enumerate(examples):
            with self.subTest(case=index):
                self.args.cache = self.root / f'pagination_cache_{index}'
                with patch.object(builder.HttpClient, '_fetch', side_effect=self.fetcher(opens=pages)):
                    with self.assertRaises(ValueError):
                        self.collect(output=self.root / f'pagination_out_{index}')

    def test_offline_import_requires_all_three_valid_scoped_snapshots(self):
        directory = self.root / 'offline'
        path = fixtures.write_snapshot(directory, self.actor, self.market)
        self.args.actor_snapshots_dir = directory
        with patch.object(builder.HttpClient, '_fetch', side_effect=AssertionError('Offline import')):
            self.collect()
        self.assertEqual(self.read(), json.loads(path.read_text()))
        value = json.loads(path.read_text())
        del value['actor_positions_closed']
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.collect(output=self.root / 'missing_feature')

    def test_offline_provenance_cannot_claim_historical_state_or_wrong_request_scope(self):
        original = fixtures.snapshot_fixture(self.actor, self.market)
        mutations = [
            lambda s: s.update(historical_model_input=True),
            lambda s: s.update(temporal_scope='historical_as_of_trade'),
            lambda s: s['actor_market_value']['pages'][0].update(retrieved_at='2026-09-28T04:00:00'),
            lambda s: s['actor_market_value']['pages'][0].update(url='https://data-api.polymarket.com/v2/value?user=' + self.other),
            lambda s: s['actor_positions_open']['pages'][0].update(body_sha256='unverified'),
            lambda s: s['actor_market_value']['data'].update(value='NaN'),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                snapshot = copy.deepcopy(original)
                mutate(snapshot)
                with self.assertRaises(ValueError):
                    builder.validate_actor_snapshot(snapshot, self.actor, self.market)


class ActorSnapshotSFTTests(unittest.TestCase):
    def setUp(self):
        self.fixture = sft_fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root

    def source(self, market_id=1):
        return self.fixture.source(market_id, market_id, with_market_context=True,
                                   context_version=2, with_actor_snapshots=True)

    def test_snapshot_finances_are_portable_audit_data_never_training_messages(self):
        sources = []
        for market_id in range(1, 4):
            path, actor_file, _ = self.source(market_id)
            market = json.loads((path / 'market.json').read_text())
            actor = actor_file.stem
            position = fixtures.position_fixture(actor, market, realized_pnl='987654.321',
                                                 upstream_extra='FUTURE_FINANCIAL_SENTINEL')
            fixtures.write_snapshot(path / 'actor_snapshots', actor, market,
                                    value='1234567.89', open_positions=[position])
            sources.append(path)
        with patch('sys.stdout', new_callable=io.StringIO):
            manifest = builder.sft_export(self.fixture.args())
        output = self.root / 'sft'
        self.assertFalse(manifest['actor_snapshots_used_as_model_input'])
        for split in builder.SFT_SPLITS:
            data = (output / (split + '.jsonl')).read_text()
            for private in ('987654.321', '1234567.89', 'FUTURE_FINANCIAL_SENTINEL',
                            'actor_market_value', 'actor_positions_open', 'actor_positions_closed',
                            'actor_snapshot_ref'):
                self.assertNotIn(private, data)
        audit_rows = [json.loads(line) for line in (output / 'source_audit.jsonl').read_text().splitlines()]
        self.assertEqual(len(audit_rows), 3)
        for audit in audit_rows:
            ref = audit['actor_snapshot_ref']
            self.assertTrue(ref.startswith('audit/actor_snapshots/'))
            self.assertFalse(Path(ref).is_absolute())
            copied = output / ref
            self.assertTrue(copied.is_file())
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), audit['actor_snapshot_sha256'])
            self.assertIn('FUTURE_FINANCIAL_SENTINEL', copied.read_text())

    def test_missing_or_cross_actor_sidecar_cannot_pass_prepare(self):
        path, actor_file, _ = self.source()
        snapshot_file = next((path / 'actor_snapshots').glob('*.json'))
        original = snapshot_file.read_text()
        snapshot_file.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.fixture.record(path)
        snapshot = json.loads(original)
        snapshot['actor_id'] = '0x' + 'b' * 40
        snapshot_file.write_text(json.dumps(snapshot))
        with self.assertRaises(ValueError):
            self.fixture.record(path)

    def test_inconsistent_or_traversing_row_reference_is_rejected(self):
        path, actor_file, rows = self.source()
        for ref in ('../outside.json', 'actor_snapshots/' + '0x' + 'b' * 40 + '.json'):
            with self.subTest(ref=ref):
                changed = copy.deepcopy(rows)
                changed[1]['actor_snapshot_ref'] = ref
                actor_file.write_text(''.join(json.dumps(row) + '\n' for row in changed))
                with self.assertRaises(ValueError):
                    self.fixture.record(path)

    def test_raw_exports_without_snapshot_feature_remain_preparable(self):
        path, _, _ = self.fixture.source(with_market_context=True, context_version=2)
        record, audit, _ = self.fixture.record(path)
        self.assertEqual(record['target_count'], 2)
        self.assertNotIn('actor_snapshot_ref', audit)


if __name__ == '__main__':
    unittest.main()
