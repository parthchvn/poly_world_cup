"""Derived actor features use only information available before each query."""
import copy
from decimal import Decimal
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'derive_actor_metrics.py'
spec = importlib.util.spec_from_file_location('derive_actor_metrics_tested', SCRIPT)
metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)

D = Decimal
DAY = 86_400_000_000
NAMES = {
    'net_realized_pnl', 'mean_realized_roi', 'win_rate',
    'average_winning_profit', 'average_losing_loss', 'payoff_ratio',
    'profit_factor', 'historical_expectancy', 'sharpe_ratio', 'sortino_ratio',
    'maximum_drawdown', 'return_volatility', 'consecutive_loss_streak',
    'average_holding_seconds', 'average_execution_notional',
    'execution_notional_cv', 'executions_per_day', 'buy_notional_share',
}


def trade(day, side='BUY', shares='10', price='0.5'):
    return {'time_us': int(day * DAY), 'side': side,
            'shares': D(shares), 'price': D(price)}


def closed(opened, end, pnl, cost='10', known=None):
    return {'opened_us': int(opened * DAY), 'closed_us': int(end * DAY),
            'known_us': int((end if known is None else known) * DAY),
            'entry_cost': D(cost), 'net_pnl': D(pnl)}


def ret(start, end, value, known=None, benchmark='0', target='0'):
    return {'start_us': int(start * DAY), 'end_us': int(end * DAY),
            'known_us': int((end if known is None else known) * DAY),
            'return': D(value), 'benchmark_return': D(benchmark),
            'target_return': D(target)}


class CausalMetricTests(unittest.TestCase):
    def compute(self, trades=None, closes=None, returns=None, query=10, **kwargs):
        return metrics.compute_metrics(trades or [], closes or [], returns or [],
                                       int(query * DAY), **kwargs)

    def assert_decimal(self, actual, expected, places=10):
        self.assertIsInstance(actual, str)
        self.assertAlmostEqual(float(D(actual)), float(expected), places=places)

    def test_exactly_eighteen_requested_metrics(self):
        result = self.compute()
        self.assertEqual(set(result['values']), NAMES)
        self.assertNotIn('time_since_previous_trade', result['values'])
        self.assertNotIn('historical_markout', result['values'])
        json.dumps(result, allow_nan=False)

    def test_current_and_future_records_do_not_change_any_feature(self):
        trades = [trade(1), trade(2, 'SELL', '20')]
        closes = [closed(1, 3, '2'), closed(2, 4, '-1')]
        returns = [ret(0, 1, '.1'), ret(1, 2, '-.05')]
        baseline = self.compute(trades, closes, returns, query=5, min_return_periods=2)
        actual = self.compute(
            trades + [trade(5, 'SELL', '999999'), trade(6, shares='999999')],
            closes + [closed(3, 5, '999999'), closed(3, 6, '-999999')],
            returns + [ret(4, 5, '.999'), ret(5, 6, '-.999')],
            query=5, min_return_periods=2)
        self.assertEqual(actual, baseline)

    def test_late_known_financial_records_are_unavailable_before_knowledge_time(self):
        earlier = self.compute([trade(1)], [], [], query=5, min_return_periods=2)
        actual = self.compute([trade(1)], [closed(1, 2, '999', known=5)],
                              [ret(0, 1, '.8', known=5)], query=5,
                              min_return_periods=2)
        self.assertEqual(actual, earlier)
        later = self.compute([trade(1)], [closed(1, 2, '999', known=5)],
                             [], query=6)
        self.assert_decimal(later['values']['net_realized_pnl'], 999)

    def test_inputs_are_not_mutated(self):
        trades = [trade(3), trade(1)]
        closes = [closed(0, 2, '-1')]
        returns = [ret(0, 1, '.1')]
        before = copy.deepcopy((trades, closes, returns))
        self.compute(trades, closes, returns)
        self.assertEqual((trades, closes, returns), before)

    def test_realized_metrics_are_based_on_completed_positions(self):
        positions = [closed(0, 1, '4'), closed(0, 2, '-2'), closed(0, 3, '0')]
        values = self.compute([], positions, [])['values']
        expected = {
            'net_realized_pnl': 2, 'mean_realized_roi': D('2') / 30,
            'win_rate': D(1) / 3, 'average_winning_profit': 4,
            'average_losing_loss': 2, 'payoff_ratio': 2, 'profit_factor': 2,
            'historical_expectancy': D(2) / 3,
            'average_holding_seconds': 2 * 86400,
        }
        for name, amount in expected.items():
            with self.subTest(name=name):
                self.assert_decimal(values[name], amount)
        self.assertEqual(values['consecutive_loss_streak'], 0)

    def test_loss_streak_uses_close_order_and_breakeven_resets(self):
        positions = [closed(0, 4, '-1'), closed(0, 1, '-3'),
                     closed(0, 3, '-2'), closed(0, 2, '0')]
        values = self.compute([], positions, [])['values']
        self.assertEqual(values['consecutive_loss_streak'], 2)

    def test_basic_trade_metrics_use_notional_and_prior_fills_only(self):
        trades = [trade(1, 'BUY', '10', '.2'), trade(2, 'SELL', '10', '.8')]
        values = self.compute(trades)['values']
        self.assert_decimal(values['average_execution_notional'], 5)
        self.assert_decimal(values['buy_notional_share'], D('0.2'))
        self.assertIsNotNone(values['execution_notional_cv'])
        self.assertGreater(D(values['executions_per_day']), 0)

    def test_no_fake_zero_or_infinite_metrics_without_evidence(self):
        result = self.compute()
        for name in NAMES:
            self.assertIsNone(result['values'][name], name)
            self.assertTrue(result['unavailable_reasons'].get(name), name)
        json.dumps(result, allow_nan=False)

    def test_zero_loss_denominators_are_null_not_infinite(self):
        values = self.compute([], [closed(0, 1, '2'), closed(0, 2, '3')], [])['values']
        self.assertIsNone(values['profit_factor'])
        self.assertIsNone(values['payoff_ratio'])
        self.assertIsNone(values['average_losing_loss'])

    def test_sharpe_uses_period_returns_not_dollar_pnl(self):
        returns = [ret(0, 1, '.1'), ret(1, 2, '-.05'), ret(2, 3, '.02')]
        values = self.compute([], [closed(0, 1, '1000000')], returns,
                              min_return_periods=3)['values']
        observations = [.1, -.05, .02]
        self.assert_decimal(values['sharpe_ratio'], statistics.mean(observations) /
                            statistics.stdev(observations))
        self.assert_decimal(values['return_volatility'], statistics.stdev(observations))
        self.assert_decimal(values['maximum_drawdown'], .05)

    def test_risk_minimum_samples_and_zero_variance(self):
        returns = [ret(0, 1, '.1'), ret(1, 2, '-.05')]
        too_short = self.compute([], [], returns, min_return_periods=3)
        self.assertIsNone(too_short['values']['sharpe_ratio'])
        constant = self.compute([], [], [ret(0, 1, '.1'), ret(1, 2, '.1')],
                                min_return_periods=2)
        self.assertIsNone(constant['values']['sharpe_ratio'])
        self.assertIsNone(constant['values']['sortino_ratio'])
        json.dumps(constant, allow_nan=False)

    def test_gapped_periods_cannot_be_silently_compounded_for_drawdown(self):
        result = self.compute([], [], [ret(0, 1, '.1'), ret(2, 3, '-.05')],
                              min_return_periods=2)
        self.assertIsNone(result['values']['maximum_drawdown'])
        self.assertTrue(result['unavailable_reasons'].get('maximum_drawdown'))

    def test_lookback_discards_old_observations(self):
        result = self.compute([trade(1, shares='9999'), trade(9, shares='2')],
                              [closed(0, 2, '9999'), closed(8, 9, '-2')], [],
                              query=10, lookback_seconds=2 * 86400)
        self.assert_decimal(result['values']['average_execution_notional'], 1)
        self.assert_decimal(result['values']['net_realized_pnl'], -2)


    def test_return_benchmarks_and_targets_are_applied_per_period(self):
        returns = [ret(0, 1, '.1', benchmark='.01', target='.03'),
                   ret(1, 2, '-.05', benchmark='.02', target='.03'),
                   ret(2, 3, '.02', benchmark='.01', target='.03')]
        values = self.compute([], [], returns, min_return_periods=3)['values']
        excess = [.09, -.07, .01]
        targeted = [.07, -.08, -.01]
        self.assert_decimal(values['sharpe_ratio'], statistics.mean(excess) /
                            statistics.stdev(excess))
        downside = math.sqrt(sum(min(r, 0) ** 2 for r in targeted) / 3)
        self.assert_decimal(values['sortino_ratio'], statistics.mean(targeted) / downside)

    def test_rolling_risk_window_does_not_include_partial_return_period(self):
        result = self.compute([], [], [ret(7, 9, '.9'), ret(9, 11, '-.1')],
                              query=12, lookback_seconds=4 * 86400,
                              min_return_periods=2)
        self.assertEqual(result['sample_counts']['eligible_return_periods'], 1)
        self.assert_decimal(result['values']['maximum_drawdown'], '.1')

    def test_tied_mixed_closures_do_not_invent_loss_streak_order(self):
        result = self.compute([], [closed(0, 2, '-1'), closed(1, 2, '1')], [])
        self.assertIsNone(result['values']['consecutive_loss_streak'])
        self.assertEqual(result['unavailable_reasons']['consecutive_loss_streak'],
                         'ambiguous_close_order')


class MetricsCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, compressed=False):
        from tests.test_prepare_actor_sft import ActorSFTTests
        fixture = ActorSFTTests()
        fixture.root = self.root
        source, actor_file, rows = fixture.source(compressed=compressed)
        return source, actor_file, rows

    def run_cli(self, source, out, *flags):
        return subprocess.run([sys.executable, str(SCRIPT), '--input-root', str(source),
                               '--out', str(out), *map(str, flags)],
                              text=True, capture_output=True, timeout=30)

    def output_rows(self, output, row):
        actor_file = output / 'markets' / row['condition_id'] / 'actors' / (row['actor_id'] + '.jsonl')
        return [json.loads(line) for line in actor_file.read_text().splitlines()]

    def test_gzip_raw_actor_rows_are_processed_without_modifying_source(self):
        source, actor_file, rows = self.source(compressed=True)
        before = actor_file.read_bytes()
        output = self.root / 'metrics'
        result = self.run_cli(source, output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        derived = self.output_rows(output, rows[1])
        self.assertEqual(len(derived), 2)
        self.assertEqual(derived[0]['source_trade_row_index'], 1)
        self.assertEqual(derived[1]['source_trade_row_index'], 3)
        self.assertEqual(derived[0]['actor_metrics']['sample_counts']['captured_executions'], 0)
        self.assertEqual(derived[1]['actor_metrics']['sample_counts']['captured_executions'], 2)
        self.assertIsNone(derived[0]['actor_metrics']['values']['average_execution_notional'])
        self.assertEqual(actor_file.read_bytes(), before)

    def test_target_trade_mutation_cannot_change_features_at_its_timestamp(self):
        source, actor_file, rows = self.source()
        first = self.root / 'first'
        result = self.run_cli(source, first)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline = self.output_rows(first, rows[1])
        rows[3]['label']['trades'][0]['shares'] = '9999999'
        rows[3]['label']['trades'][0]['price'] = '.999'
        actor_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        second = self.root / 'second'
        result = self.run_cli(source, second)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.output_rows(second, rows[1]), baseline)

    def test_collection_time_snapshots_do_not_enter_metrics(self):
        source, actor_file, rows = self.source()
        snapshot_dir = source / 'actor_snapshots'
        snapshot_dir.mkdir()
        snapshot = snapshot_dir / (rows[1]['actor_id'] + '.json')
        snapshot.write_text(json.dumps({'actor_market_value': 999999999,
                                        'actor_positions_closed': [{'realized_pnl': 999999999}]}))
        for row in rows:
            row['actor_snapshot_ref'] = 'actor_snapshots/' + snapshot.name
        actor_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        output = self.root / 'derived'
        result = self.run_cli(source, output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for row in self.output_rows(output, rows[1]):
            self.assertIsNone(row['actor_metrics']['values']['net_realized_pnl'])
            self.assertIsNone(row['actor_metrics']['values']['sharpe_ratio'])

    def test_closed_position_is_available_only_after_its_known_at_time(self):
        source, _, rows = self.source()
        supplemental = self.root / 'closed.jsonl'
        supplemental.write_text(json.dumps({
            'actor_id': rows[1]['actor_id'], 'condition_id': rows[1]['condition_id'],
            'position_id': 'settled-before-but-known-later',
            'opened_at': '2026-06-01T15:00:00Z',
            'closed_at': '2026-06-01T16:00:30Z',
            'known_at': '2026-06-01T16:02:00Z',
            'entry_cost': '10', 'net_pnl': '2',
        }) + '\n')
        output = self.root / 'with_financials'
        result = self.run_cli(source, output, '--closed-positions', supplemental)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        derived = self.output_rows(output, rows[1])
        self.assertIsNone(derived[0]['actor_metrics']['values']['net_realized_pnl'])
        self.assertEqual(D(derived[1]['actor_metrics']['values']['net_realized_pnl']), 2)

    def test_supplemental_history_is_isolated_by_actor_and_condition(self):
        source, _, rows = self.source()
        own = {
            'actor_id': rows[1]['actor_id'], 'condition_id': rows[1]['condition_id'],
            'position_id': 'closed-1', 'opened_at': '2026-06-01T15:00:00Z',
            'closed_at': '2026-06-01T15:59:00Z', 'known_at': '2026-06-01T16:00:00Z',
            'entry_cost': '10', 'net_pnl': '2',
        }
        records = [own, dict(own, actor_id='0x' + 'b' * 40, net_pnl='99999'),
                   dict(own, condition_id='0x' + '2' * 64, net_pnl='88888')]
        supplemental = self.root / 'closed-multiple-actors.jsonl'
        supplemental.write_text(''.join(json.dumps(row) + '\n' for row in records))
        output = self.root / 'scoped'
        result = self.run_cli(source, output, '--closed-positions', supplemental)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for row in self.output_rows(output, rows[1]):
            self.assertEqual(D(row['actor_metrics']['values']['net_realized_pnl']), 2)
            self.assertEqual(row['actor_metrics']['sample_counts']['completed_positions'], 1)
        metadata = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(metadata['supplemental_sources']['closed_positions']['ignored_out_of_scope_rows'], 2)

    def test_financial_availability_before_the_event_is_rejected(self):
        source, _, rows = self.source()
        supplemental = self.root / 'inconsistent-closed.jsonl'
        supplemental.write_text(json.dumps({
            'actor_id': rows[1]['actor_id'], 'condition_id': rows[1]['condition_id'],
            'position_id': 'impossible-knowledge', 'opened_at': '2026-06-01T15:00:00Z',
            'closed_at': '2026-06-01T16:02:00Z', 'known_at': '2026-06-01T16:00:00Z',
            'entry_cost': '10', 'net_pnl': '2',
        }) + '\n')
        output = self.root / 'bad-availability'
        result = self.run_cli(source, output, '--closed-positions', supplemental)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_return_period_known_at_query_is_excluded_until_next_query(self):
        source, _, rows = self.source()
        supplemental = self.root / 'returns.jsonl'
        supplemental.write_text(json.dumps({
            'actor_id': rows[1]['actor_id'], 'condition_id': rows[1]['condition_id'],
            'period_start': '2026-05-31T16:00:00Z',
            'period_end': '2026-06-01T16:00:00Z',
            'known_at': '2026-06-01T16:01:00Z',
            'period_return': '-.2', 'capital_flow_adjusted': True,
        }) + '\n')
        output = self.root / 'with_returns'
        result = self.run_cli(source, output, '--returns-file', supplemental)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        derived = self.output_rows(output, rows[1])
        self.assertIsNone(derived[0]['actor_metrics']['values']['maximum_drawdown'])
        self.assertEqual(D(derived[1]['actor_metrics']['values']['maximum_drawdown']), D('.2'))

    def test_no_overwrite_of_existing_output(self):
        source, _, _ = self.source()
        output = self.root / 'existing'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_text('keep me')
        result = self.run_cli(source, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), 'keep me')

    def test_period_returns_without_capital_adjustment_attestation_are_rejected(self):
        source, _, rows = self.source()
        returns_file = self.root / 'returns.jsonl'
        returns_file.write_text(json.dumps({
            'actor_id': rows[1]['actor_id'], 'condition_id': rows[1]['condition_id'],
            'period_start': '2026-05-01T00:00:00Z', 'period_end': '2026-05-02T00:00:00Z',
            'known_at': '2026-05-02T00:00:01Z', 'period_return': '.1',
        }) + '\n')
        output = self.root / 'invalid'
        result = self.run_cli(source, output, '--returns-file', returns_file)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
