"""Wallet-wide metrics preserve time causality without inventing equity returns."""
import copy
from decimal import Decimal
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'derive_global_actor_metrics.py'
spec = importlib.util.spec_from_file_location('global_actor_metrics_tested', SCRIPT)
global_metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = global_metrics
spec.loader.exec_module(global_metrics)
base = global_metrics.base

ACTOR = '0x' + 'a' * 40
OTHER_ACTOR = '0x' + 'b' * 40
MARKET = '0x' + '1' * 64
OTHER_MARKET = '0x' + '2' * 64
DAY = 86_400_000_000
D = Decimal


def execution(identifier='fill-1', timestamp='2026-06-01T15:59:00Z', known=None,
              actor=ACTOR, condition=MARKET, side='BUY', shares='10', price='.2'):
    return {'actor_id': actor, 'condition_id': condition, 'execution_id': identifier,
            'timestamp': timestamp, 'known_at': known or timestamp,
            'side': side, 'shares': shares, 'price': price}


def normalized(day, known=None, condition=MARKET, side='BUY', shares='10', price='.2'):
    return {'time_us': day * DAY, 'known_us': (day if known is None else known) * DAY,
            'condition_id': condition, 'side': side, 'shares': D(shares), 'price': D(price)}


def wallet_return(day=1, value='.1', known=None, **changes):
    row = {'actor_id': ACTOR, 'scope': 'wallet',
           'period_start': f'2026-05-{day:02d}T00:00:00Z',
           'period_end': f'2026-05-{day+1:02d}T00:00:00Z',
           'known_at': known or f'2026-05-{day+1:02d}T00:00:00Z',
           'period_return': value, 'capital_flow_adjusted': True}
    return dict(row, **changes)


class GlobalCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, rows, name='history.jsonl'):
        path = self.root / name
        value = ''.join(json.dumps(row) + '\n' for row in rows)
        if name.endswith('.gz'):
            path.write_bytes(gzip.compress(value.encode()))
        else:
            path.write_text(value)
        return path

    def compute(self, trades=(), closes=(), returns=(), query=10, **flags):
        return global_metrics.compute_global_metrics(list(trades), list(closes), list(returns),
                                                       query * DAY, **flags)

    def test_cross_market_history_is_pooled_but_metrics_have_same_names(self):
        result = self.compute([normalized(1), normalized(2, condition=OTHER_MARKET, side='SELL', price='.8')])
        self.assertEqual(set(result['values']), set(base.ACTOR_METRIC_NAMES))
        self.assertEqual(D(result['values']['average_execution_notional']), 5)
        self.assertEqual(D(result['values']['buy_notional_share']), D('.2'))
        self.assertEqual(result['sample_counts']['captured_executions'], 2)
        self.assertIn('across_markets', result['metric_scope']['execution_metrics'])

    def test_all_markets_current_timestamp_and_future_are_excluded(self):
        baseline = self.compute([normalized(1)], query=3)
        actual = self.compute([normalized(1), normalized(3),
                               normalized(3, condition=OTHER_MARKET, shares='999'),
                               normalized(4, condition=OTHER_MARKET, shares='999999')], query=3)
        self.assertEqual(actual, baseline)

    def test_delayed_execution_availability_and_equality_boundary(self):
        rows = [normalized(1, known=3), normalized(2, known=4, condition=OTHER_MARKET)]
        self.assertEqual(self.compute(rows, query=3)['sample_counts']['captured_executions'], 0)
        self.assertEqual(self.compute(rows, query=4)['sample_counts']['captured_executions'], 1)
        self.assertEqual(self.compute(rows, query=5)['sample_counts']['captured_executions'], 2)

    def test_future_records_never_change_counts_or_rolling_metrics(self):
        before = [normalized(2), normalized(6)]
        expected = self.compute(before, query=8, lookback_seconds=4 * 86400)
        actual = self.compute(before + [normalized(9, price='.999')], query=8,
                              lookback_seconds=4 * 86400)
        self.assertEqual(expected, actual)
        self.assertEqual(actual['sample_counts']['captured_executions'], 1)
        self.assertEqual(D(actual['values']['executions_per_day']), D('.25'))

    def test_wallet_loader_isolates_actors_and_preserves_decimal_precision(self):
        path = self.file([execution(shares='20.408162', price='0.048999954'),
                          execution(actor=OTHER_ACTOR, shares='99999')], 'history.jsonl.gz')
        groups = global_metrics.load_wallet_trades(path)
        self.assertEqual(len(groups[ACTOR]), 1)
        self.assertEqual(groups[ACTOR][0]['shares'] * groups[ACTOR][0]['price'], D('.999998999224548'))

    def test_offline_requires_known_at_and_rejects_earlier_knowledge(self):
        for row in (dict(execution(), known_at='2026-06-01T15:58:59Z'),
                    {key: value for key, value in execution().items() if key != 'known_at'}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                global_metrics.normalize_wallet_row(row)

    def test_api_proxy_must_be_explicit_and_cannot_have_manufactured_known_at(self):
        row = execution()
        del row['known_at']
        with self.assertRaises(ValueError):
            global_metrics.normalize_wallet_row(row, api_proxy=True)
        row['availability_semantics'] = 'execution_timestamp_proxy_not_verified_publication_time'
        _, parsed = global_metrics.normalize_wallet_row(row, api_proxy=True)
        self.assertEqual(parsed['time_us'], parsed['known_us'])
        row['known_at'] = row['timestamp']
        with self.assertRaises(ValueError):
            global_metrics.normalize_wallet_row(row, api_proxy=True)

    def test_duplicate_execution_ids_are_rejected_even_across_conditions(self):
        path = self.file([execution(), execution(condition=OTHER_MARKET)])
        with self.assertRaisesRegex(ValueError, 'Duplicate wallet execution'):
            global_metrics.load_wallet_trades(path)

    def test_equal_fills_with_distinct_ids_are_not_deduplicated(self):
        path = self.file([execution('one'), execution('two')])
        self.assertEqual(len(global_metrics.load_wallet_trades(path)[ACTOR]), 2)

    def test_invalid_wallet_trade_fields_are_rejected(self):
        for update in ({'shares': '0'}, {'price': '1.1'}, {'side': 'MINT'},
                       {'price': 'NaN'}, {'execution_id': ''}, {'actor_id': 'anonymous'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                global_metrics.normalize_wallet_row(dict(execution(), **update))

    def test_wallet_returns_reject_market_returns_and_unadjusted_returns(self):
        for change in ({'condition_id': MARKET}, {'market_id': '1'}, {'scope': 'market'},
                       {'capital_flow_adjusted': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                global_metrics.load_wallet_returns(self.file([wallet_return(**change)]))

    def test_whole_wallet_returns_compute_sharpe_without_using_fill_prices(self):
        ledger = self.file([wallet_return(value='.2'), wallet_return(day=2, value='-.1')])
        returns = global_metrics.load_wallet_returns(ledger)[ACTOR]
        query = base.timestamp_us('2026-06-01T16:00:00Z')
        result = global_metrics.compute_global_metrics([], [], returns, query, min_return_periods=2)
        self.assertEqual(D(result['values']['maximum_drawdown']), D('.1'))
        self.assertAlmostEqual(float(result['values']['sharpe_ratio']), .23570226039551584)
        self.assertIsNone(result['values']['net_realized_pnl'])

    def test_late_wallet_returns_are_excluded_at_known_at_then_added(self):
        ledger = self.file([wallet_return(value='-.2', known='2026-06-01T16:00:00Z')])
        returns = global_metrics.load_wallet_returns(ledger)[ACTOR]
        query = base.timestamp_us('2026-06-01T16:00:00Z')
        earlier = global_metrics.compute_global_metrics([], [], returns, query)
        later = global_metrics.compute_global_metrics([], [], returns, query + 1)
        self.assertIsNone(earlier['values']['maximum_drawdown'])
        self.assertEqual(D(later['values']['maximum_drawdown']), D('.2'))

    def test_overlapping_duplicate_or_post_bankruptcy_wallet_returns_rejected(self):
        examples = [
            [wallet_return(), wallet_return()],
            [wallet_return(period_end='2026-05-03T00:00:00Z', known_at='2026-05-03T00:00:00Z'),
             wallet_return(day=2)],
            [wallet_return(value='-1'), wallet_return(day=2)],
        ]
        for rows in examples:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                global_metrics.load_wallet_returns(self.file(rows))

    def test_completed_positions_pool_markets_but_not_actors(self):
        row = {'actor_id': ACTOR, 'condition_id': MARKET, 'position_id': 'one',
               'opened_at': '2026-05-01T00:00:00Z', 'closed_at': '2026-05-02T00:00:00Z',
               'known_at': '2026-05-02T00:00:01Z', 'entry_cost': '10', 'net_pnl': '2'}
        path = self.file([row, dict(row, condition_id=OTHER_MARKET, net_pnl='-1'),
                          dict(row, actor_id=OTHER_ACTOR, net_pnl='999')])
        grouped = global_metrics.load_global_closed_positions(path)
        query = base.timestamp_us('2026-06-01T16:00:00Z')
        result = global_metrics.compute_global_metrics([], grouped[ACTOR], [], query)
        self.assertEqual(D(result['values']['net_realized_pnl']), 1)
        self.assertEqual(result['sample_counts']['completed_positions'], 2)


class GlobalCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        from tests.test_prepare_actor_sft import ActorSFTTests
        fixture = ActorSFTTests()
        fixture.root = self.root
        self.source, self.actor_file, self.rows = fixture.source()

    def wallet_file(self, rows=None, compressed=False):
        path = self.root / ('wallet.jsonl.gz' if compressed else 'wallet.jsonl')
        rows = rows if rows is not None else [
            execution('other-prior', condition=OTHER_MARKET),
            execution('target', timestamp='2026-06-01T16:01:00Z', price='.5'),
            execution('other-tied', timestamp='2026-06-01T16:01:00Z',
                      condition=OTHER_MARKET, shares='20', price='.5'),
            execution('future', timestamp='2026-06-01T16:04:00Z', price='.99'),
        ]
        text = ''.join(json.dumps(row) + '\n' for row in rows)
        if compressed:
            path.write_bytes(gzip.compress(text.encode()))
        else:
            path.write_text(text)
        return path

    def run_cli(self, output, history, *flags):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.source), '--out', str(output),
                               '--wallet-trades', str(history), *map(str, flags)],
                              text=True, capture_output=True, timeout=30)

    def records(self, output):
        path = output / 'markets' / self.rows[1]['condition_id'] / 'actors' / (ACTOR + '.jsonl')
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_offline_cli_includes_other_markets_and_excludes_current_target(self):
        history = self.wallet_file(compressed=True)
        before = self.actor_file.read_bytes(), history.read_bytes()
        output = self.root / 'global'
        run = self.run_cli(output, history)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        first, second = self.records(output)
        self.assertEqual(D(first['actor_metrics']['values']['average_execution_notional']), 2)
        self.assertEqual(first['actor_metrics']['sample_counts']['captured_executions'], 1)
        self.assertEqual(second['actor_metrics']['sample_counts']['captured_executions'], 3)
        self.assertAlmostEqual(float(second['actor_metrics']['values']['average_execution_notional']), 17/3)
        self.assertEqual((self.actor_file.read_bytes(), history.read_bytes()), before)
        self.assertFalse((output / 'wallet_history.sqlite').exists())
        metadata = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(metadata['config']['history_scope'], 'actor_across_all_markets')
        self.assertFalse(metadata['source_capture_complete'])

    def test_missing_wallet_actor_cannot_silently_become_zero_history(self):
        output = self.root / 'missing'
        run = self.run_cli(output, self.wallet_file([execution(actor=OTHER_ACTOR)]))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('no executions for target actors', run.stderr)
        self.assertFalse(output.exists())

    def test_duplicate_wallet_observation_fails_without_partial_output(self):
        output = self.root / 'duplicates'
        run = self.run_cli(output, self.wallet_file([execution(), execution()]))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('Duplicate wallet execution', run.stderr)
        self.assertFalse(output.exists())

    def test_no_finance_metrics_inferred_from_current_snapshots_or_trade_prices(self):
        (self.source / 'actor_snapshots').mkdir()
        (self.source / 'actor_snapshots' / (ACTOR + '.json')).write_text(
            json.dumps({'actor_positions_closed': [{'pnl': 999999}], 'actor_market_value': 999999}))
        output = self.root / 'no_finance'
        run = self.run_cli(output, self.wallet_file())
        self.assertEqual(run.returncode, 0, run.stderr)
        for row in self.records(output):
            self.assertIsNone(row['actor_metrics']['values']['sharpe_ratio'])
            self.assertIsNone(row['actor_metrics']['values']['net_realized_pnl'])

    def test_execution_known_only_at_second_query_remains_excluded_there(self):
        output = self.root / 'delayed'
        run = self.run_cli(output, self.wallet_file([execution(known='2026-06-01T16:03:00Z')]))
        self.assertEqual(run.returncode, 0, run.stderr)
        for row in self.records(output):
            self.assertEqual(row['actor_metrics']['sample_counts']['captured_executions'], 0)
            self.assertIsNone(row['actor_metrics']['values']['average_execution_notional'])

    def test_source_target_label_mutation_does_not_change_features(self):
        history = self.wallet_file()
        first, second = self.root / 'original', self.root / 'mutated'
        run = self.run_cli(first, history)
        self.assertEqual(run.returncode, 0, run.stderr)
        changed = copy.deepcopy(self.rows)
        changed[3]['label']['trades'][0]['shares'] = '99999999'
        self.actor_file.write_text(''.join(json.dumps(row) + '\n' for row in changed))
        run = self.run_cli(second, history)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.records(first), self.records(second))

    def test_cli_feature_selection_and_significant_digits_are_validated(self):
        history = self.wallet_file()
        for flags in (('--features', 'historical_markout'), ('--features', 'win_rate,win_rate'),
                      ('--metric-significant-digits', '2')):
            with self.subTest(flags=flags):
                output = self.root / 'invalid'
                run = self.run_cli(output, history, *flags)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(output.exists())

    def global_args(self, **changes):
        args = dict(exports=[self.source], input_root=None, out=self.root / 'global_auto',
                    wallet_trades=None, closed_positions=None, returns_file=None,
                    lookback_seconds=None, min_return_periods=30, sft_dir=None,
                    cache=self.root / 'cache', wallet_max_pages=None, http_transport='urllib',
                    http_timeout=10, http_retries=0, http_retry_delay=1, http_min_interval=0,
                    features=None, metric_significant_digits=10)
        return SimpleNamespace(**dict(args, **changes))

    def test_automatic_capture_actual_helper_feeds_global_metrics_and_audits_proxy(self):
        import build_actor_dataset
        from tests.test_wallet_history import Client, page, trade
        prior = base.timestamp_us('2026-06-01T15:59:00Z') // 1_000_000
        current = base.timestamp_us('2026-06-01T16:01:00Z') // 1_000_000
        client = Client([
            page([trade(current, proxy_wallet=ACTOR, condition_id=self.rows[1]['condition_id'],
                        size='10', price='.5', fill_id='target'),
                  trade(prior, proxy_wallet=ACTOR, condition_id=OTHER_MARKET,
                        size='10', price='.2', fill_id='prior')], 'next'),
            page([trade(prior - 60, proxy_wallet=ACTOR, condition_id=OTHER_MARKET,
                        size='10', price='.4', fill_id='older')]),
        ])
        args = self.global_args()
        with patch.object(build_actor_dataset, 'HttpClient', return_value=client):
            metadata = global_metrics.derive(args)
        first, second = self.records(args.out)
        self.assertEqual(D(first['actor_metrics']['values']['average_execution_notional']), 3)
        self.assertEqual(first['actor_metrics']['sample_counts']['captured_executions'], 2)
        self.assertEqual(second['actor_metrics']['sample_counts']['captured_executions'], 3)
        self.assertEqual(len(client.requests), 2)
        for _, params in client.requests:
            self.assertNotIn('condition', params)
            self.assertEqual(params['user'], ACTOR)
            self.assertEqual(params['start'], 1)
            self.assertEqual(params['end'], base.timestamp_us(self.rows[3]['timestamp']) // 1_000_000)
        self.assertEqual(metadata['config']['execution_availability_semantics'],
                         'execution_timestamp_proxy_not_verified_publication_time')
        self.assertEqual(metadata['wallet_captures'][0]['captured_executions'], 3)

    def prepare_three_fixtures(self, reverse=False):
        from tests.test_prepare_actor_sft import ActorSFTTests, builder
        fixture = ActorSFTTests()
        fixture.root = self.root
        paths = [self.source]
        for number in (2, 3):
            paths.append(fixture.source(market_id=number, fixture=number)[0])
        split = None
        if reverse:
            split = self.root / 'reversed_splits.json'
            split.write_text(json.dumps({'espn:1': 'test', 'espn:2': 'validation', 'espn:3': 'train'}))
        args = fixture.args(exports=paths, input_root=None, split_file=split)
        builder.sft_export(args)
        return paths, args.out

    def test_global_sft_rejects_bad_chronology_before_any_network_capture(self):
        import build_actor_dataset
        paths, source_sft = self.prepare_three_fixtures(reverse=True)
        args = self.global_args(exports=paths, sft_dir=source_sft)
        with patch.object(build_actor_dataset, 'HttpClient', side_effect=AssertionError('network before preflight')) as factory:
            with self.assertRaisesRegex(ValueError, 'overlap or touch'):
                global_metrics.derive(args)
        factory.assert_not_called()
        self.assertFalse(args.out.exists())
        self.assertFalse(args.cache.exists())

    def test_three_split_enrichment_preserves_targets_and_selects_global_prompt_features(self):
        paths, source_sft = self.prepare_three_fixtures()
        history = self.wallet_file()
        args = self.global_args(exports=paths, wallet_trades=history, sft_dir=source_sft,
                                features='average_execution_notional,buy_notional_share')
        metadata = global_metrics.derive(args)
        result_dir = args.out / 'sft'
        output_manifest = json.loads((result_dir / 'manifest.json').read_text())
        self.assertEqual(output_manifest['feature_variant'], 'global')
        self.assertEqual(output_manifest['actor_metrics']['history_scope'], 'actor_across_all_markets')
        self.assertIsNotNone(metadata['split_temporal_validation'])
        for split in ('train', 'validation', 'test'):
            original = next(base.iter_jsonl(source_sft / (split + '.jsonl')))
            enriched = next(base.iter_jsonl(result_dir / (split + '.jsonl')))
            self.assertEqual(enriched['target_count'], original['target_count'])
            self.assertEqual(enriched['execution_count'], original['execution_count'])
            for old, new in zip(original['messages'], enriched['messages']):
                if old['role'] != 'user':
                    self.assertEqual(old, new)
                    continue
                original_context, context = base.loads(old['content']), base.loads(new['content'])
                metrics = context.pop('actor_metrics')
                self.assertEqual(context, original_context)
                self.assertEqual(metrics['scope'], 'global_wallet')
                self.assertEqual(set(metrics['values']), {'average_execution_notional', 'buy_notional_share'})
                self.assertNotIn('unavailable_reasons', metrics)
                self.assertNotIn('window', metrics)


if __name__ == '__main__':
    unittest.main()
