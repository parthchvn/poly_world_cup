"""Official historical prices are independent, causal, bounded, and verifiable."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from tests import test_prepare_actor_sft as fixtures

builder = fixtures.builder


class ClobHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = builder.timestamp_us('2026-06-01T16:00:00Z')
        self.market = {
            'market_id': '1', 'condition_id': '0x' + '1' * 64,
            'tokens': [{'outcome': 'Yes', 'token_id': '101'},
                       {'outcome': 'No', 'token_id': '102'}],
        }

    def instant(self, seconds):
        return self.origin + int(Decimal(str(seconds)) * 1_000_000)

    def point(self, seconds, price):
        return {'t': self.instant(seconds) // 1_000_000, 'p': price}

    def staged(self, seconds=(10, 20)):
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        actor = '0x' + 'a' * 40
        builder.stage_trades(db, iter({
            'actor_id': actor, 'time_us': self.instant(t),
            'time': builder.utc_time(self.instant(t)),
            'trade': {'side': 'BUY', 'outcome': 'Yes', 'shares': '10', 'price': '0.99'},
        } for t in seconds))
        return db

    def payload(self, yes=None, no=None):
        return {
            'format': 'polymarket_clob_price_history_v1',
            'condition_id': self.market['condition_id'], 'fidelity_minutes': 1,
            'histories': [
                {'token_id': '101', 'history': yes if yes is not None else [self.point(0, '0.4')]},
                {'token_id': '102', 'history': no if no is not None else [self.point(1, '0.62')]},
            ],
        }

    def build(self, payload=None, seconds=(10, 20)):
        db = self.staged(seconds)
        source = self.root / 'input.json'
        source.write_text(json.dumps(payload if payload is not None else self.payload()))
        output = self.root / 'market_price_history.jsonl'
        result = builder.build_clob_price_timeline(db, output, self.market, None, history_file=source)
        return db, output, result

    def context(self, db, seconds, max_age=300):
        return builder.market_context_at(db, self.instant(seconds), version=2, max_age_seconds=max_age)

    def test_official_series_excludes_same_time_and_future_and_never_uses_target_price(self):
        payload = self.payload(
            yes=[self.point(0, '0.04'), self.point(10, '0.08'), self.point(11, '0.1')],
            no=[self.point(1, '0.97'), self.point(10, '0.92')])
        db, output, _ = self.build(payload)
        result = self.context(db, 10)
        self.assertEqual(result['version'], 2)
        self.assertEqual(result['price_semantics'], 'clob_historical_price_not_quote')
        self.assertEqual(result['yes']['price'], '0.04')
        self.assertEqual(result['no']['price'], '0.97')
        self.assertEqual(result['yes']['age_seconds'], '10')
        self.assertEqual(result['no']['age_seconds'], '9')
        self.assertEqual(result['missing_reasons'], {'yes': None, 'no': None})
        self.assertEqual(result['max_age_seconds'], 300)
        for outcome, token in [('yes', '101'), ('no', '102')]:
            point = result[outcome]
            self.assertEqual(point['source'], 'polymarket_clob_prices_history')
            self.assertEqual(point['token_id'], token)
            self.assertEqual(point['requested_fidelity_minutes'], 1)
            self.assertEqual(point['price'], point['implied_probability'])
            self.assertNotIn('observation_count', point)
        self.assertNotEqual(Decimal(result['yes']['price']) + Decimal(result['no']['price']), 1)
        self.assertTrue(output.is_file())

    def test_no_earlier_price_and_stale_prices_remain_null_without_trade_fallback(self):
        db, _, _ = self.build()
        before = self.context(db, 0)
        self.assertIsNone(before['yes'])
        self.assertIsNone(before['no'])
        self.assertEqual(before['missing_reasons'],
                         {'yes': 'no_earlier_observation', 'no': 'no_earlier_observation'})
        boundary = self.context(db, 300)
        self.assertEqual(boundary['yes']['age_seconds'], '300')
        after = self.context(db, '300.000001')
        self.assertIsNone(after['yes'])
        self.assertEqual(after['missing_reasons']['yes'], 'stale')
        self.assertIsNotNone(after['no'])
        self.assertEqual(self.context(db, 20, max_age=5)['missing_reasons'],
                         {'yes': 'stale', 'no': 'stale'})

    def test_zero_and_one_are_valid_and_decimal_precision_is_preserved(self):
        db, _, _ = self.build(self.payload(yes=[self.point(0, '0'), self.point(5, '0.123456789123')],
                                          no=[self.point(0, '1')]))
        self.assertEqual(self.context(db, 1)['yes']['price'], '0')
        self.assertEqual(self.context(db, 1)['no']['price'], '1')
        self.assertEqual(self.context(db, 10)['yes']['price'], '0.123456789123')

    def test_missing_or_empty_token_history_fails_instead_of_complementing_other_side(self):
        missing = self.payload()
        missing['histories'].pop()
        empty = self.payload(no=[])
        wrong = self.payload()
        wrong['histories'][1]['token_id'] = '999'
        for payload in (missing, empty, wrong):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.build(payload)

    def test_offline_source_identity_and_fidelity_must_match(self):
        for field, value in [('format', 'arbitrary_prices'),
                             ('condition_id', '0x' + '2' * 64),
                             ('fidelity_minutes', 60)]:
            payload = self.payload()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(payload)

    def test_malformed_samples_and_conflicting_same_time_values_fail(self):
        for bad in [self.point(0, 'NaN'), self.point(0, 'Infinity'), self.point(0, '-0.1'),
                    self.point(0, '1.1'), {'t': True, 'p': '0.5'},
                    {'t': 1780329600.5, 'p': '0.5'}, {'p': '0.5'},
                    {'t': self.instant(0) // 1_000_000}]:
            with self.subTest(sample=bad), self.assertRaises(ValueError):
                self.build(self.payload(yes=[bad]))
        with self.assertRaises(ValueError):
            self.build(self.payload(yes=[self.point(0, '0.4'), self.point(0, '0.5')]))

    def test_duplicate_identical_samples_are_deduplicated_deterministically(self):
        db, output, _ = self.build(self.payload(yes=[self.point(5, '0.5'), self.point(0, '0.4'),
                                                   self.point(0, '0.40')]))
        self.assertEqual(db.execute('SELECT COUNT(*) FROM market_prices').fetchone()[0], 3)
        first = output.read_bytes()
        payload = self.payload(yes=[self.point(0, '0.40'), self.point(5, '0.5'), self.point(0, '0.4')])
        _, output, _ = self.build(payload)
        self.assertEqual(first, output.read_bytes())

    def test_api_uses_both_token_ids_minute_fidelity_and_bounded_windows(self):
        days = 16 * 86400
        db = self.staged((10, days))
        calls = []

        def fetch(url, params=None):
            calls.append((url, params.copy()))
            point = {'t': params['startTs'], 'p': '0.4' if params['market'] == '101' else '0.6'}
            return builder.FetchResult({'history': [point]}, url, '2026-09-28T00:00:00Z', 'a' * 64, False)

        client = Mock()
        client.get_json.side_effect = fetch
        builder.build_clob_price_timeline(db, self.root / 'network.jsonl', self.market, client)
        self.assertGreaterEqual(len(calls), 6)
        for token in ('101', '102'):
            windows = [params for _, params in calls if params['market'] == token]
            self.assertEqual(min(p['startTs'] for p in windows), self.instant(10) // 1_000_000 - 300)
            self.assertEqual(max(p['endTs'] for p in windows), self.instant(days) // 1_000_000 + 1)
            ordered = sorted(windows, key=lambda p: p['startTs'])
            for left, right in zip(ordered, ordered[1:]):
                self.assertLessEqual(right['startTs'], left['endTs'])
            for params in windows:
                self.assertEqual(params['fidelity'], 1)
                self.assertGreater(params['endTs'], params['startTs'])
                self.assertLessEqual(params['endTs'] - params['startTs'], 7 * 86400)
        self.assertTrue(all(url == 'https://clob.polymarket.com/prices-history' for url, _ in calls))


class ClobHistorySFTTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def source(self):
        return self.fixture.source(with_market_context=True, context_version=2)

    @staticmethod
    def write(file, rows):
        file.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_official_prices_enter_context_and_never_assistant_target(self):
        path, _, _ = self.source()
        record, _, _ = self.fixture.record(path)
        user = json.loads(record['messages'][1]['content'])
        self.assertEqual(user['market_context'], {
            'yes': {'price': '0.3', 'age_seconds': '60'},
            'no': {'price': '0.7', 'age_seconds': '60'}})
        target = json.loads(record['messages'][2]['content'])
        self.assertEqual(target['trades'][0]['price'], '0.31')
        self.assertNotIn('execution_info', user)
        self.assertNotIn('payoff_analysis', user)

    def test_v2_rejects_forged_provenance_token_id_staleness_and_future_prices(self):
        path, file, original = self.source()
        for field, value in [('source', 'captured_execution_timestamp_vwap'), ('token_id', '102'),
                             ('requested_fidelity_minutes', 60),
                             ('observed_at', original[1]['timestamp']), ('age_seconds', '0')]:
            rows = copy.deepcopy(original)
            for row in rows[:2]:
                row['market_context']['yes'][field] = value
            self.write(file, rows)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.fixture.record(path)
        rows = copy.deepcopy(original)
        for row in rows[:2]:
            row['market_context']['max_age_seconds'] = 30
        self.write(file, rows)
        with self.assertRaises(ValueError):
            self.fixture.record(path)

    def test_v1_v2_mixture_cannot_silently_train_with_different_price_semantics(self):
        self.source()
        self.fixture.source(2, 2, with_market_context=True, context_version=1)
        with self.assertRaisesRegex(ValueError, 'Mixed market-context versions'):
            builder.sft_discover([], self.fixture.root)


if __name__ == '__main__':
    unittest.main()
