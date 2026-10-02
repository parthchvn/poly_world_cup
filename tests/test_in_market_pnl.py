"""Unrealized inventory, causal cutoffs, and every interval's model feature."""
from decimal import Decimal
from fractions import Fraction
import io
import json
import random
import sys
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures

sys.path.insert(0, str(fixtures.ROOT / "scripts"))
sys.path.insert(0, str(fixtures.ROOT / "tools"))
builder = fixtures.builder


def trade(side='BUY', shares='10', price='0.4', outcome='Yes'):
    return dict(side=side, shares=shares, price=price, outcome=outcome)


def prices(yes='0.6', no='0.4'):
    return {o: {'price': p, 'age_seconds': '1'} if p is not None else None
            for o, p in [('yes', yes), ('no', no)]}


class InventoryTests(unittest.TestCase):
    def value(self, state, context=None):
        return state.snapshot(prices() if context is None else context, '2026-06-01T17:00:00Z')

    def test_first_query_is_zero_even_without_prices(self):
        self.assertEqual(self.value(builder.InMarketPnL(), {})['unrealized_in_market_pnl'], '0')

    def test_partial_sale_excludes_realized_profit_and_full_exit_is_zero(self):
        state = builder.InMarketPnL()
        state.apply([trade()])
        self.assertEqual(self.value(state)['unrealized_in_market_pnl'], '2')
        state.apply([trade('SELL', '4', '0.9')])
        result = self.value(state)
        self.assertEqual(result['unrealized_in_market_pnl'], '1.2')
        self.assertEqual(result['in_market_pnl_context']['holdings']['yes']['remaining_cost_basis'], '2.4')
        state.apply([trade('SELL', '6', '0.7')])
        self.assertEqual(self.value(state, {})['unrealized_in_market_pnl'], '0')
        state.apply([trade('BUY', '3', '0.2')])
        self.assertEqual(self.value(state)['unrealized_in_market_pnl'], '1.2')

    def test_multiple_entries_and_independent_outcome_marks(self):
        state = builder.InMarketPnL()
        state.apply([trade(), trade(shares='10', price='0.2'), trade(shares='5', price='0.7', outcome='No')])
        self.assertEqual(self.value(state)['unrealized_in_market_pnl'], '4.5')
        state.apply([trade('SELL', '5', '0.8')])
        self.assertEqual(self.value(state)['unrealized_in_market_pnl'], '3')

    def test_missing_price_and_unmatched_sales_are_not_zero(self):
        state = builder.InMarketPnL()
        state.apply([trade()])
        result = self.value(state, prices(None))
        self.assertIsNone(result['unrealized_in_market_pnl'])
        self.assertEqual(result['in_market_pnl_context']['missing_reasons'], {'yes': 'missing_prior_price'})
        self.assertEqual(self.value(state)['unrealized_in_market_pnl'], '2')
        state.apply([trade('SELL', '11')])
        self.assertIsNone(self.value(state)['unrealized_in_market_pnl'])
        self.assertEqual(self.value(state)['in_market_pnl_context']['missing_reasons']['yes'], 'unmatched_sell')

    def test_same_time_group_does_not_invent_a_fill_order(self):
        values = [trade(), trade('SELL', '4', '0.6')]
        left, right = builder.InMarketPnL(), builder.InMarketPnL()
        left.apply(values)
        right.apply(list(reversed(values)))
        self.assertEqual(self.value(left), self.value(right))
        self.assertIsNone(self.value(left)['unrealized_in_market_pnl'])
        left.apply([trade('SELL', '6')])
        self.assertEqual(self.value(left)['unrealized_in_market_pnl'], '0')

    def test_current_and_future_mark_rejected(self):
        for at in ('2026-06-01T17:00:00Z', '2026-06-01T17:01:00Z'):
            context = prices()
            context['yes']['observed_at'] = at
            with self.assertRaisesRegex(ValueError, 'strictly precede'):
                self.value(builder.InMarketPnL(), context)

    def test_running_cost_matches_independent_remaining_lot_reference(self):
        rng, state, lots = random.Random(8), builder.InMarketPnL(), []
        for _ in range(100):
            held = sum((q for q, p in lots), Fraction(0))
            if not held or rng.random() < 0.6:
                q, p = Fraction(rng.randint(1, 20)), Fraction(rng.randint(1, 99), 100)
                state.apply([trade(shares=str(q.numerator), price=str(float(p)))])
                lots.append((q, p))
            else:
                q = held / 2
                # Proportional lot reduction is an independent average-cost oracle.
                state.apply([trade('SELL', str(float(q)), '0.5')])
                lots = [(quantity / 2, price) for quantity, price in lots]
            expected = sum((q * (Fraction(3, 5) - p) for q, p in lots), Fraction(0))
            actual = Decimal(self.value(state)['unrealized_in_market_pnl'])
            self.assertAlmostEqual(float(actual), float(expected), places=10)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def source(self, market_id=1, fixture_id=1, start_offset=0):
        path, actor_file, _ = self.fixture.source(market_id, fixture_id)
        market = json.loads((path / 'market.json').read_text())
        manifest = json.loads((path / 'manifest.json').read_text())
        origin = builder.timestamp_us(manifest['origin_utc'])
        executions = [trade(), trade('SELL', '4', '0.9'), trade('BUY', '1', '0.1')]
        values = [dict(time_us=origin + (i+1)*60_000_000, time=builder.utc_time(origin + (i+1)*60_000_000),
                       trade=t) for i, t in enumerate(executions)]
        def lookup(instant):
            marks = prices()
            for mark in marks.values():
                mark.update(observed_at=builder.utc_time(instant-1_000_000),
                            implied_probability=mark['price'], source='captured_execution_timestamp_vwap',
                            observation_count=1)
            return dict(version=1, as_of=builder.utc_time(instant), price_semantics='prior_execution_vwap_not_quote',
                        winning_payout_per_share='1', losing_payout_per_share='0', **marks)
        start = origin + start_offset*60_000_000
        rows = list(builder.actor_records(actor_file.stem, values, market, [], [], start, lookup))
        actor_file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest.update(in_market_pnl_version=1, market_context_version=1, fill_window_seconds=5,
                        origin_utc=builder.utc_time(start), origin_basis='user_supplied' if start_offset else 'market_open',
                        counts=dict(actors=1, rows=len(rows), distinct_trade_times=len(rows)//2,
                                    trade_observations=len(rows)//2, news_entries=0))
        (path/'manifest.json').write_text(json.dumps(manifest))
        return path, actor_file, rows

    def test_raw_sft_metrics_and_mask_keep_both_labels_and_pnl(self):
        import derive_actor_metrics as metrics
        path, file, rows = self.source()
        self.assertEqual([r['unrealized_in_market_pnl'] for r in rows], ['0','0','2','2','1.2','1.2'])
        source = builder.sft_discover([path], None)[0]
        record, _, _ = builder.sft_convert_actor(file, source)
        users = [json.loads(m['content']) for m in record['messages'] if m['role']=='user']
        self.assertEqual([u['unrealized_in_market_pnl'] for u in users], ['0','0','2','2','1.2','1.2'])
        labels = [json.loads(m['content'])['action'] for m in record['messages'] if m['role']=='assistant']
        self.assertEqual(labels, ['NO_TRADE','TRADE']*3)
        _, groups, _ = metrics.actor_trade_groups(file, metrics.discover_exports([path], None)[0])
        self.assertEqual([g['pnl_features']['unrealized_in_market_pnl'] for g in groups], ['0','2','1.2'])
        tokenizer = fixtures.OffsetTokenizer()
        encoded, targets = fixtures.trainer.encode_conversation(record, tokenizer, 30000, 'pnl')
        loss_text = ''.join(tokenizer.lookup[t] for t in encoded['labels'] if t != -100)
        self.assertEqual(targets, 6)
        self.assertEqual(loss_text.count('NO_TRADE'), 3)
        self.assertNotIn('unrealized_in_market_pnl', loss_text)

    def test_current_execution_changes_do_not_change_its_input_feature(self):
        path, file, rows = self.source()
        source = builder.sft_discover([path], None)[0]
        original = builder.sft_convert_actor(file, source)[0]
        # Legacy raw input is recomputed; labels at t cannot affect the feature at t.
        source['manifest'].pop('in_market_pnl_version')
        rows[-1]['label']['trades'][0]['price'] = '0.99'
        rows[-1]['payoff_analysis'] = [builder.execution_payoff(rows[-1]['label']['trades'][0])]
        file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        changed = builder.sft_convert_actor(file, source)[0]
        self.assertEqual(original['messages'][-2], changed['messages'][-2])

    def test_cutoff_replays_earlier_inventory_without_adding_earlier_labels(self):
        path, file, rows = self.source(start_offset=2)
        record, _, _ = builder.sft_convert_actor(file, builder.sft_discover([path], None)[0])
        self.assertEqual(rows[0]['unrealized_in_market_pnl'], '2')
        self.assertEqual(record['target_count'], 4)
        self.assertEqual(json.loads(record['messages'][1]['content'])['unrealized_in_market_pnl'], '2')

    def test_new_raw_feature_is_validated_and_legacy_raw_is_upgraded(self):
        path, file, rows = self.source()
        source = builder.sft_discover([path], None)[0]
        rows[2]['unrealized_in_market_pnl'] = '999'
        file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'unrealized_in_market_pnl'):
            builder.sft_convert_actor(file, source)
        source['manifest'].pop('in_market_pnl_version')
        record, _, _ = builder.sft_convert_actor(file, source)
        self.assertEqual(json.loads(record['messages'][5]['content'])['unrealized_in_market_pnl'], '2')

    def test_local_global_and_offline_variants_preserve_feature_and_intervals(self):
        import derive_actor_metrics as metrics
        import run_world_cup_experiments as experiments
        import compare_actor_variants as variants
        import world_cup_eval_common as common
        paths = [self.source(i, i)[0] for i in range(1, 4)]
        with patch('sys.stdout', new_callable=io.StringIO):
            builder.prepare_main(['--input-root', str(self.root), '--out', str(self.root/'basic')])
            experiments.derive_inmarket(self.root/'basic', self.root/'offline')
        basic = variants.scan_dataset(self.root/'basic')
        index = {}
        for source in metrics.discover_exports(paths, None):
            for file in (source['path']/'actors').iterdir():
                actor, groups, _ = metrics.actor_trade_groups(file, source)
                history = []
                for g in groups:
                    index[(actor, source['market_id'], g['time_us'])] = {
                        'actor_metrics': metrics.compute_metrics(history, [], [], g['time_us']),
                        'trades': g['expected'], 'pnl_features': g['pnl_features']}
                    history.extend(g['trades'])
        for variant, scope in [('local','actor_and_binary_market'), ('global','actor_across_all_markets')]:
            metrics.enrich_sft(self.root/'basic', self.root/variant, index,
                {'history_scope': scope, 'metric_significant_digits': 10})
        for variant in ('local','global','offline'):
            for split in ('train','validation','test'):
                record = next(variants.lines(variants.split_file(self.root/variant, split)))[1]
                expected = next(variants.lines(basic['paths'][split]))[1]
                for left, right in zip(expected['messages'], record['messages']):
                    if left['role'] == 'user':
                        self.assertEqual(json.loads(left['content'])['unrealized_in_market_pnl'],
                                         json.loads(right['content'])['unrealized_in_market_pnl'])
                    else:
                        self.assertEqual(left, right)
                self.assertEqual(len(list(common.targets(record))), 6)

    def test_prospective_activity_uses_all_prior_holdings_and_its_own_mark(self):
        import actor_activity_common as activity
        import derive_actor_metrics as metrics
        path, file, _ = self.source()
        source = metrics.discover_exports([path], None)[0]
        source['market'] = json.loads((path/'market.json').read_text())
        source['market']['kickoff_utc'] = '2026-06-01T17:00:00Z'
        source['manifest']['market_price_max_age_seconds'] = 300
        _, groups, _ = metrics.actor_trade_groups(file, source)
        query = groups[1]['time_us'] + 30_000_000
        timeline = {'prices': {'yes': [(query-1_000_000, '0.7')], 'no': [(query-1_000_000, '0.3')]},
                    'price_times': {'yes': [query-1_000_000], 'no': [query-1_000_000]},
                    'news': [], 'news_times': []}
        basic, enriched, _ = activity.context_at(source, timeline, groups, query, 60_000_000,
            {'history_scope': 'actor_and_binary_market', 'metric_significant_digits': 10}, 1, 1200)
        self.assertEqual(basic['unrealized_in_market_pnl'], '1.8')
        self.assertEqual(enriched['unrealized_in_market_pnl'], '1.8')
        self.assertEqual(basic['earlier_execution_groups_omitted'], 1)

    def test_trainer_rejects_missing_interval_counts_and_protocol(self):
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            fixtures.trainer.validate_target_counts({'action_counts': {'TRADE': 3, 'NO_TRADE': 2}})
        path, file, _ = self.source()
        record, _, _ = builder.sft_convert_actor(file, builder.sft_discover([path], None)[0])
        record.pop('target_protocol')
        split = self.root/'broken.jsonl'
        split.write_text(json.dumps(record)+'\n')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            fixtures.trainer.read_split(split, fixtures.OffsetTokenizer(), 30000)


if __name__ == '__main__':
    unittest.main()
