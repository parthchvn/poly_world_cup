"""Offline tests for market context, execution evidence, and SFT causality."""
import copy
from decimal import Decimal, localcontext
import io
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures
from tests import actor_snapshot_fixtures as snapshots

builder = fixtures.builder


class ActorMarketContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = builder.timestamp_us('2026-06-01T16:00:00Z')
        self.actor = '0x' + 'a' * 40
        self.other = '0x' + 'b' * 40
        self.market = {
            'market_id': '1', 'condition_id': '0x' + '1' * 64,
            'fixture_id': 'espn:1', 'espn_event_id': '1',
            'kickoff_utc': '2026-06-01T17:00:00Z',
            'question': 'Will the match end in a draw?', 'fixture_title': 'Example match',
            'tokens': [{'outcome': 'Yes', 'token_id': '101'},
                       {'outcome': 'No', 'token_id': '102'}],
            'market_open_utc': builder.utc_time(self.origin),
            'market_open_basis': 'test_fixture',
        }

    def instant(self, seconds):
        return self.origin + int(Decimal(str(seconds)) * 1_000_000)

    def trade(self, seconds, price, *, outcome='Yes', shares='1', actor=None, side='BUY'):
        when = self.instant(seconds)
        return {
            'actor_id': actor or self.actor, 'time_us': when, 'time': builder.utc_time(when),
            'trade': {'side': side, 'outcome': outcome, 'shares': shares, 'price': price},
        }

    def timeline(self, rows, name='timeline'):
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        builder.stage_trades(db, iter(rows))
        path = self.root / (name + '.jsonl')
        metadata = builder.build_price_timeline(db, path)
        self.assertTrue(path.is_file())
        return db, path, metadata

    def snapshot(self, db, seconds):
        return builder.market_context_at(db, self.instant(seconds))

    def test_strict_prior_prices_exclude_target_same_timestamp_and_future(self):
        rows = [self.trade(0, '0.20'), self.trade(1, '0.65', outcome='No'),
                self.trade(10, '0.90'), self.trade(10, '0.99', actor=self.other),
                self.trade(11, '0.95', outcome='No')]
        db, _, _ = self.timeline(rows)
        context = self.snapshot(db, 10)
        self.assertEqual(context['as_of'], builder.utc_time(self.instant(10)))
        self.assertEqual(context['price_semantics'], 'prior_execution_vwap_not_quote')
        self.assertEqual(Decimal(context['yes']['price']), Decimal('0.20'))
        self.assertEqual(Decimal(context['no']['price']), Decimal('0.65'))
        self.assertEqual(Decimal(context['yes']['age_seconds']), Decimal('10'))
        self.assertEqual(Decimal(context['no']['age_seconds']), Decimal('9'))
        for outcome in ('yes', 'no'):
            observed = context[outcome]
            self.assertLess(builder.timestamp_us(observed['observed_at']), self.instant(10))
            self.assertEqual(observed['price'], observed['implied_probability'])
            self.assertEqual(observed['source'], 'captured_execution_timestamp_vwap')

    def test_same_timestamp_vwap_is_deterministic_and_weighted(self):
        rows = [self.trade(5, '0.20', shares='1'),
                self.trade(5, '0.60', shares='3', actor=self.other, side='SELL'),
                self.trade(5, '0.61', shares='2', outcome='No'),
                self.trade(6, '0.91')]
        expected = None
        expected_bytes = None
        for seed in range(4):
            shuffled = list(rows)
            random.Random(seed).shuffle(shuffled)
            db, path, _ = self.timeline(shuffled, 'shuffle_' + str(seed))
            current = self.snapshot(db, 6)
            self.assertEqual(Decimal(current['yes']['price']), Decimal('0.50'))
            self.assertEqual(current['yes']['observation_count'], 2)
            self.assertEqual(Decimal(current['no']['price']), Decimal('0.61'))
            if expected is None:
                expected, expected_bytes = current, path.read_bytes()
            else:
                self.assertEqual(current, expected)
                self.assertEqual(path.read_bytes(), expected_bytes)

    def test_vwap_rounds_to_documented_twelve_places(self):
        db, _, metadata = self.timeline([self.trade(0, '0', shares='1'),
                                         self.trade(0, '1', shares='2')])
        self.assertEqual(self.snapshot(db, 1)['yes']['price'], '0.666666666667')
        self.assertEqual(metadata['decimal_places'], 12)
        self.assertEqual(metadata['rounding'], 'ROUND_HALF_EVEN')

    def test_missing_outcomes_and_empty_history_stay_unknown(self):
        db, _, _ = self.timeline([self.trade(2, '0.40')])
        for seconds in (0, 2):
            context = self.snapshot(db, seconds)
            self.assertIsNone(context['yes'])
            self.assertIsNone(context['no'])
        context = self.snapshot(db, 100)
        self.assertEqual(Decimal(context['yes']['price']), Decimal('0.40'))
        self.assertEqual(Decimal(context['yes']['age_seconds']), Decimal('98'))
        self.assertIsNone(context['no'])
        self.assertEqual(context['winning_payout_per_share'], '1')
        self.assertEqual(context['losing_payout_per_share'], '0')

    def test_yes_no_are_independent_and_microsecond_age_is_preserved(self):
        db, _, _ = self.timeline([self.trade('0.000001', '0.4'),
                                 self.trade('0.000002', '0.61', outcome='No')])
        context = self.snapshot(db, '0.000003')
        self.assertEqual(Decimal(context['yes']['price']) + Decimal(context['no']['price']),
                         Decimal('1.01'))
        self.assertEqual(Decimal(context['yes']['age_seconds']), Decimal('0.000002'))
        self.assertEqual(Decimal(context['no']['age_seconds']), Decimal('0.000001'))

    def test_endpoint_prices_zero_and_one_remain_valid(self):
        db, _, _ = self.timeline([self.trade(0, '0'), self.trade(1, '1', outcome='No')])
        context = self.snapshot(db, 2)
        self.assertEqual(Decimal(context['yes']['implied_probability']), 0)
        self.assertEqual(Decimal(context['no']['implied_probability']), 1)

    def test_every_actor_row_uses_context_before_its_execution_time(self):
        earlier = self.trade(0, '0.40', actor=self.other)
        target = self.trade(10, '0.90')
        db, _, _ = self.timeline([earlier, target])
        rows = list(builder.actor_records(
            self.actor, iter([target]), self.market, [], [], self.origin,
            market_context_lookup=lambda instant: builder.market_context_at(db, instant)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['market_context'], rows[1]['market_context'])
        self.assertEqual(Decimal(rows[1]['market_context']['yes']['price']), Decimal('0.40'))
        self.assertEqual(rows[1]['label']['trades'][0]['price'], '0.90')
        self.assertEqual(rows[0]['execution_info']['status'], 'no_observed_execution_in_open_interval')
        self.assertEqual(rows[1]['execution_info']['status'], 'observed_execution')
        self.assertEqual(rows[0]['payoff_analysis'], [])
        self.assertEqual(len(rows[1]['payoff_analysis']), 1)

    def test_observed_execution_does_not_establish_filled_order_or_submission_delay(self):
        evidence = builder.execution_evidence(self.instant(10), True, 5)
        self.assertEqual(evidence['status'], 'observed_execution')
        self.assertEqual(evidence['observed_execution_time'], builder.utc_time(self.instant(10)))
        self.assertEqual(evidence['fill_window_seconds'], 5)
        for key in ('order_submitted_at', 'order_type', 'order_fully_filled',
                    'filled_within_seconds_of_submission', 'full_fill_time',
                    'submission_to_execution_seconds'):
            self.assertIsNone(evidence[key])
        self.assertEqual(evidence['unknown_reason'], 'public_trade_feed_has_no_order_lifecycle')

    def test_gap_is_not_evidence_of_an_unfilled_order(self):
        evidence = builder.execution_evidence(self.instant(10), False, 2)
        self.assertEqual(evidence['status'], 'no_observed_execution_in_open_interval')
        self.assertIsNone(evidence['observed_execution_time'])
        self.assertIsNone(evidence['order_fully_filled'])
        self.assertIsNone(evidence['filled_within_seconds_of_submission'])
        self.assertIsNone(evidence['order_submitted_at'])
        self.assertEqual(evidence['fill_window_seconds'], 2)

    def test_buy_payoff_is_conditional_profit_not_expected_or_realized_profit(self):
        trade = self.trade(0, '0.31', shares='13.123456789')['trade']
        payoff = builder.execution_payoff(trade)
        self.assertEqual(payoff['basis'], 'hypothetical_buy_held_to_binary_resolution')
        self.assertEqual(Decimal(payoff['cash_flow_before_fees']), -Decimal('13.123456789') * Decimal('0.31'))
        self.assertEqual(Decimal(payoff['winning_payout']), Decimal('13.123456789'))
        self.assertEqual(Decimal(payoff['potential_profit_if_win_before_fees']),
                         Decimal('13.123456789') * Decimal('0.69'))
        self.assertEqual(payoff['pnl_if_lose_before_fees'], payoff['cash_flow_before_fees'])
        for key in ('expected_profit', 'fees', 'realized_profit'):
            self.assertIsNone(payoff[key])
        self.assertEqual(payoff['unknown_reason'], 'winning_probability_and_fees_unknown')

    def test_selling_reports_proceeds_without_fabricating_cost_basis_or_profit(self):
        payoff = builder.execution_payoff(self.trade(0, '0.61', shares='100', side='SELL')['trade'])
        self.assertEqual(payoff['basis'], 'sale_proceeds_only_cost_basis_unknown')
        self.assertEqual(Decimal(payoff['cash_flow_before_fees']), Decimal('61'))
        for key in ('winning_payout', 'potential_profit_if_win_before_fees',
                    'pnl_if_lose_before_fees', 'expected_profit', 'fees', 'realized_profit'):
            self.assertIsNone(payoff[key])
        self.assertEqual(payoff['unknown_reason'], 'sale_cost_basis_and_fees_unknown')

    def test_payoff_at_endpoint_prices_never_divides_by_price(self):
        free = builder.execution_payoff(self.trade(0, '0', shares='10')['trade'])
        certain = builder.execution_payoff(self.trade(0, '1', shares='10')['trade'])
        self.assertEqual(Decimal(free['cash_flow_before_fees']), 0)
        self.assertEqual(Decimal(free['potential_profit_if_win_before_fees']), 10)
        self.assertEqual(Decimal(certain['cash_flow_before_fees']), -10)
        self.assertEqual(Decimal(certain['potential_profit_if_win_before_fees']), 0)

    def test_payoff_preserves_more_than_twenty_eight_significant_digits(self):
        shares = '123456789012345678901234567890123456789.987654321'
        price = '0.12345678901234567890123456789'
        payoff = builder.execution_payoff(self.trade(0, price, shares=shares)['trade'])
        with localcontext() as ctx:
            ctx.prec = 200
            cost = Decimal(shares) * Decimal(price)
            gain = Decimal(shares) * (1 - Decimal(price))
            self.assertEqual(Decimal(payoff['cash_flow_before_fees']), -cost)
            self.assertEqual(Decimal(payoff['pnl_if_lose_before_fees']), -cost)
            self.assertEqual(Decimal(payoff['potential_profit_if_win_before_fees']), gain)
            self.assertEqual(Decimal(payoff['winning_payout']), Decimal(shares))

    def official_history_file(self, condition_id=None):
        path = self.root / 'official_history.json'
        path.write_text(json.dumps({
            'format': 'polymarket_clob_price_history_v1',
            'condition_id': condition_id or self.market['condition_id'],
            'fidelity_minutes': 1,
            'histories': [
                {'token_id': '101', 'history': [
                    {'t': self.instant(0) // 1_000_000, 'p': '0.21'},
                    {'t': self.instant(15) // 1_000_000, 'p': '0.46'}]},
                {'token_id': '102', 'history': [
                    {'t': self.instant(5) // 1_000_000, 'p': '0.71'}]},
            ]}))
        return path

    def export_fixture(self, history_file=None):
        rows = [self.trade(0, '0.20', actor=self.other),
                self.trade(5, '0.70', actor=self.other, outcome='No'),
                self.trade(10, '0.90'), self.trade(15, '0.45', actor=self.other),
                self.trade(20, '0.80', side='SELL')]
        when = self.instant(3)
        event = {'timestamp_us': when, 'time_utc': builder.utc_time(when), 'news_id': 'event:1',
                 'text': 'A timestamped match update', 'kind': 'commentary'}
        context = {'event_id': '1', 'timed_events': [event], 'untimed_events': []}
        output = self.root / 'market_1'
        snapshots.write_snapshot(self.root / 'snapshot_inputs', self.actor, self.market)
        with patch.object(builder, 'resolve_market', return_value=self.market), \
                patch.object(builder, 'automatic_espn_file', return_value=None), \
                patch.object(builder, 'collect_espn_context', return_value=context), \
                patch.object(builder, 'load_market_trades', return_value=(iter(rows), {'source_type': 'fixture'})), \
                patch.object(builder.HttpClient, 'get_json', side_effect=AssertionError('No network')), \
                patch('sys.stdout', new_callable=io.StringIO):
            builder.main(['1', '--out', str(output), '--cache', str(self.root / 'cache'),
                          '--max-trades-per-actor', '2',
                          '--actor-snapshots-dir', str(self.root / 'snapshot_inputs'),
                          '--price-history-file', str(history_file or self.official_history_file())])
        return output

    def test_unusable_equal_time_history_never_publishes_or_deletes_cached_data(self):
        cache_file = self.root / 'cache/trade_capture/saved_page.json.gz'
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(b'previously saved trade capture')
        previous = self.root / 'existing_dataset/manifest.json'
        previous.parent.mkdir()
        previous.write_text('{"previous_export": true}\n')
        original_cache, original_export = cache_file.read_bytes(), previous.read_bytes()
        # Even when one token has valid earlier prices, the other token being
        # equal to the last execution makes it unusable for every exported row.
        for yes_time in (20, 0):
            source = self.official_history_file()
            payload = json.loads(source.read_text())
            for item, seconds in zip(payload['histories'], (yes_time, 20)):
                item['history'] = [{'t': self.instant(seconds) // 1_000_000, 'p': '0.5'}]
            source.write_text(json.dumps(payload))
            with self.subTest(yes_time=yes_time), \
                    patch('sys.stderr', new_callable=io.StringIO) as error, \
                    self.assertRaises(SystemExit):
                self.export_fixture(history_file=source)
            self.assertIn('no usable earlier prices', error.getvalue())
            self.assertIn('No actor export published', error.getvalue())
            self.assertFalse((self.root / 'market_1').exists())
            self.assertEqual(list(self.root.glob('market-actor-build-*')), [])
            self.assertEqual(cache_file.read_bytes(), original_cache)
            self.assertEqual(previous.read_bytes(), original_export)

    def test_export_uses_official_prices_and_sft_keeps_only_prior_context(self):
        output = self.export_fixture()
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['market_context_version'], 2)
        self.assertTrue(manifest['price_context_complete_for_exported_rows'])
        self.assertEqual(manifest['actors_excluded_above_trade_limit'], 1)
        self.assertFalse((output / 'actors' / (self.other + '.jsonl')).exists())
        actor_file = output / 'actors' / (self.actor + '.jsonl')
        rows = [json.loads(line) for line in actor_file.read_text().splitlines()]
        self.assertTrue(all('market_context' in row for row in rows))
        reference = 'actor_snapshots/' + self.actor + '.json'
        self.assertTrue(all(row['actor_snapshot_ref'] == reference for row in rows))
        index = [json.loads(line) for line in (output / 'actor_index.jsonl').read_text().splitlines()]
        self.assertEqual(index[0]['actor_snapshot_ref'], reference)
        self.assertEqual(manifest['actor_snapshots']['actors'], 1)
        self.assertEqual(len(list((output / 'actor_snapshots').glob('*.json'))), 1)
        self.assertEqual(json.loads((output / reference).read_text())['actor_market_value']['data']['value'], '0')
        self.assertEqual(Decimal(rows[1]['market_context']['yes']['price']), Decimal('0.21'))
        self.assertEqual(Decimal(rows[3]['market_context']['yes']['price']), Decimal('0.46'))
        self.assertEqual(Decimal(rows[1]['market_context']['no']['price']), Decimal('0.71'))
        source = builder.sft_discover([output], None)[0]
        record, _, _ = builder.sft_convert_actor(actor_file, source, include_no_trade=False)
        users = [json.loads(message['content']) for message in record['messages'] if message['role'] == 'user']
        assistants = [json.loads(message['content']) for message in record['messages'] if message['role'] == 'assistant']
        self.assertEqual(len(users), 2)
        self.assertEqual(users[0]['market_context'], {
            'yes': {'price': '0.21', 'age_seconds': '10'},
            'no': {'price': '0.71', 'age_seconds': '5'},
        })
        self.assertEqual(users[1]['market_context'], {
            'yes': {'price': '0.46', 'age_seconds': '5'},
            'no': {'price': '0.71', 'age_seconds': '15'},
        })
        self.assertEqual(assistants[0]['trades'][0]['price'], '0.90')
        for context in users:
            self.assertNotIn('execution_info', context)
            self.assertNotIn('payoff_analysis', context)
            self.assertNotIn('label', context)

    def test_sft_rejects_equal_time_price_observation(self):
        output = self.export_fixture()
        actor_file = output / 'actors' / (self.actor + '.jsonl')
        rows = [json.loads(line) for line in actor_file.read_text().splitlines()]
        bad = copy.deepcopy(rows[1]['market_context'])
        bad['yes']['observed_at'] = rows[1]['timestamp']
        bad['yes']['age_seconds'] = '0'
        rows[0]['market_context'] = rows[1]['market_context'] = bad
        actor_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        source = builder.sft_discover([output], None)[0]
        with self.assertRaises(ValueError):
            builder.sft_convert_actor(actor_file, source, include_no_trade=False)

    def test_full_offline_file_collection_then_prepare_cli(self):
        """Exercise real file readers, CLI options, exports, and published SFT files."""
        data_root = self.root / 'actor_data'
        raw = [self.trade(0, '0.20', actor=self.other),
               self.trade(5, '0.70', actor=self.other, outcome='No'),
               self.trade(10, '0.90'), self.trade(15, '0.45', actor=self.other),
               self.trade(20, '0.80', side='SELL')]
        rows = [{'actor_id': row['actor_id'], 'time': row['time'], **row['trade']} for row in raw]
        trades_file = self.root / 'trades.jsonl'
        trades_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        espn_file = self.root / 'espn.jsonl'
        espn_file.write_text(json.dumps({'id': '1', 'time_utc': builder.utc_time(self.instant(3)),
                                         'text': 'A known match event', 'type': 'commentary'}) + '\n')
        with patch.object(builder.HttpClient, 'get_json', side_effect=AssertionError('No network')), \
                patch('sys.stdout', new_callable=io.StringIO):
            for market_id in range(1, 4):
                market = dict(self.market, market_id=str(market_id), condition_id='0x' + f'{market_id:064x}',
                              fixture_id=f'espn:{market_id}', espn_event_id=str(market_id),
                              game_start_time=f'2026-06-{market_id:02d}T17:00:00Z',
                              accepting_orders_at=builder.utc_time(self.origin))
                metadata_file = self.root / f'metadata_{market_id}.json'
                metadata_file.write_text(json.dumps(market))
                output = data_root / f'market_{market_id}'
                snapshot_dir = self.root / f'snapshot_inputs_{market_id}'
                snapshots.write_snapshot(snapshot_dir, self.actor, market)
                builder.main([str(market_id), '--market-metadata', str(metadata_file),
                              '--espn-file', str(espn_file), '--trades-file', str(trades_file),
                              '--out', str(output), '--cache', str(self.root / 'cache'),
                              '--actor-snapshots-dir', str(snapshot_dir),
                              '--max-trades-per-actor', '2', '--fill-window-seconds', '7',
                              '--price-history-file', str(self.official_history_file(market['condition_id']))])
                exported = [json.loads(line) for line in
                            (output / 'actors' / (self.actor + '.jsonl')).read_text().splitlines()]
                self.assertEqual(exported[1]['execution_info']['fill_window_seconds'], 7)
                self.assertIsNone(exported[1]['execution_info']['filled_within_seconds_of_submission'])
                self.assertEqual(exported[1]['execution_info']['observed_execution_time'], raw[2]['time'])
                self.assertEqual(Decimal(exported[1]['payoff_analysis'][0]['potential_profit_if_win_before_fees']),
                                 Decimal('0.1'))
            sft_out = self.root / 'prepared'
            builder.main(['prepare', '--input-root', str(data_root), '--out', str(sft_out)])
        manifest = json.loads((sft_out / 'manifest.json').read_text())
        self.assertEqual(manifest['fixture_to_split'], {'espn:1': 'train', 'espn:2': 'validation', 'espn:3': 'test'})
        for split in ('train', 'validation', 'test'):
            conversations = [json.loads(line) for line in (sft_out / (split + '.jsonl')).read_text().splitlines()]
            self.assertEqual(len(conversations), 1)
            record = conversations[0]
            self.assertEqual(record['target_count'], 4)
            self.assertEqual(record['no_trade_target_count'], 2)
            first = json.loads(record['messages'][1]['content'])
            self.assertEqual(Decimal(first['market_context']['yes']['price']), Decimal('0.21'))
            self.assertNotIn('payoff_analysis', first)
            self.assertNotIn('execution_info', first)


if __name__ == '__main__':
    unittest.main()
