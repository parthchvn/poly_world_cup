import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
import interval_decision_data as data
import build_actor_dataset as builder


class IntervalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'raw'
        self.inputs.mkdir()
        self.sources = []
        fixtures = sorted(builder.BUNDLED_REGISTRY['fixtures'], key=lambda r: r['kickoff_utc'])[:3]
        for index, fixture in enumerate(fixtures, 1):
            self.sources.append(self.make_source(index, fixture))

    def tearDown(self):
        self.temp.cleanup()

    def make_source(self, index, fixture):
        path = self.inputs / f'market_{index}'
        (path / 'actors').mkdir(parents=True)
        kickoff = data.activity.timestamp_us(fixture['kickoff_utc'])
        actor = '0x' + f'{index:040x}'
        market = {'market_id': str(index), 'condition_id': '0x' + f'{index:064x}',
            'fixture_id': fixture['fixture_id'], 'espn_event_id': fixture['espn_event_id'],
            'kickoff_utc': fixture['kickoff_utc'], 'question': 'Will the match be a draw?',
            'tokens': [{'outcome': 'Yes', 'token_id': '1'}, {'outcome': 'No', 'token_id': '2'}]}
        trades = [{'time_us': kickoff + offset * 1_000_000,
                   'time': data.activity.utc(kickoff + offset * 1_000_000),
                   'trade': {'side': 'BUY', 'outcome': 'Yes', 'shares': '10', 'price': '0.4'}}
                  for offset in (-30, 60, 240)]
        rows = list(builder.actor_records(actor, trades, market, [], [], kickoff - 3600_000_000))
        for row in rows:
            row['news'] = [{'time': '2099-01-01T00:00:00Z', 'text': 'FUTURE RAW DO NOT COPY'}]
        self.jsonl(path / 'actors' / (actor + '.jsonl'), rows)
        prices = [{'outcome': outcome, 'price': price, 'observed_at': data.activity.utc(kickoff + seconds * 1_000_000),
                   'source': 'polymarket_clob_prices_history'}
                  for seconds in (-1200, -900, -600, -300, -10, 0, 30, 60, 120, 240, 360)
                  for outcome, price in [('yes', '0.6'), ('no', '0.4')]]
        self.jsonl(path / 'market_price_history.jsonl', prices)
        news = [{'time_utc': data.activity.utc(kickoff + seconds * 1_000_000), 'kind': kind,
                 'text': f'{kind} at {seconds}'}
                for seconds, kind in [(-10, 'goal'), (0, 'foul'), (60, 'goal---volley'), (120, 'yellow-card'), (360, 'goal')]]
        self.jsonl(path / 'espn_events.jsonl', news)
        manifest = {'format': 'actor_market_intervals_v1', 'market_id': str(index),
            'condition_id': market['condition_id'], 'espn_event_id': fixture['espn_event_id'],
            'created_at': data.activity.utc(kickoff + 86400_000_000), 'origin_utc': data.activity.utc(kickoff - 3600_000_000),
            'origin_basis': 'market_open', 'max_trades_per_actor': 20, 'market_context_version': 2,
            'market_price_max_age_seconds': 300, 'source': {'api_traversal_status': 'exhausted'},
            'market_price_history': {'sha256': data.activity.sha(path / 'market_price_history.jsonl')},
            'counts': {'actors': 1, 'rows': 6, 'distinct_trade_times': 3, 'trade_observations': 3}}
        data.write_json(path / 'manifest.json', manifest)
        data.write_json(path / 'market.json', market)
        return path

    @staticmethod
    def jsonl(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    @staticmethod
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def prepare(self, out='prepared', **kwargs):
        defaults = dict(window_seconds=60, max_rows=200000, match_minutes=6)
        defaults.update(kwargs)
        with patch('sys.stdout', new_callable=io.StringIO), patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            manifest = data.prepare_interval_dataset(self.inputs, self.root / out, **defaults)
        return manifest, self.root / out

    def raw(self, index=0):
        source = data.metrics.discover_exports([self.sources[index]], None)[0]
        source['market'] = json.loads((source['path'] / 'market.json').read_text())
        source['kickoff'] = data.activity.timestamp_us(source['market']['kickoff_utc'])
        _, groups, _ = data.metrics.actor_trade_groups(next((source['path'] / 'actors').glob('*')), source)
        return source, groups, data.activity.load_timeline(source)

    def test_end_to_end_keeps_natural_prevalence_and_trailing_negatives(self):
        manifest, out = self.prepare()
        self.assertEqual(manifest['targets'], 18)
        for split in data.SPLITS:
            rows = self.rows(out / (split + '.features.jsonl'))
            self.assertEqual([row['label'] for row in rows], [0, 1, 0, 0, 1, 0])
            self.assertEqual(manifest['splits'][split]['action_counts'], {'NO_TRADE': 4, 'TRADE': 2})
            self.assertEqual(manifest['files'][split+'.features.jsonl'], data.activity.sha(out / (split+'.features.jsonl')))
            for feature_row, sft in zip(rows, self.rows(out / (split+'.jsonl'))):
                context = json.loads(sft['messages'][1]['content'])
                self.assertNotIn('FUTURE RAW', json.dumps(context))
                self.assertEqual(feature_row['features'], context['derived_features'])
                self.assertEqual(json.loads(sft['messages'][-1]['content'])['action'], feature_row['action'])
                self.assertTrue(context['actor_id'].startswith('actor_'))
                self.assertNotIn(sft['actor_id'], sft['messages'][1]['content'])
        self.assertEqual(manifest['test_actors_seen_in_train'], 0)
        self.assertTrue(manifest['retrospectively_filtered_actor_cohort'])

    def test_goal_kicks_are_not_scoring_events(self):
        self.assertFalse(data._kind_matches('goal-kick', 'goal'))
        self.assertTrue(data._kind_matches('goal---volley', 'goal'))
        self.assertTrue(data._kind_matches('goal', 'goal'))

    def test_causal_features_and_pnl_are_revalued_at_query(self):
        source, groups, timeline = self.raw()
        query = source['kickoff'] + 60_000_000
        context, features, _ = data.example_context(source, timeline, groups, query, 60_000_000)
        self.assertEqual(features['prior_execution_count'], 1)
        self.assertEqual(features['unrealized_in_market_pnl'], 2)
        self.assertEqual(context['market_context']['yes']['age_seconds'], '30.0')
        self.assertEqual([r['text'] for r in context['news']], ['goal at -10', 'foul at 0'])
        self.assertEqual(features['news_goal_count_900s'], 1)
        self.assertIsNone(features['sharpe_ratio'])
        self.assertEqual(data.activity.answer_at(groups, query, 60_000_000), 1)
        self.assertEqual(data.activity.answer_at(groups, source['kickoff'], 60_000_000), 0)
        before = copy.deepcopy((context, features))
        groups[1]['trades'][0]['shares'] *= 100
        groups[1]['expected'][0]['shares'] = '1000'
        timeline['prices']['yes'][-1] = (timeline['prices']['yes'][-1][0], '0.99')
        timeline['news'][-1][1]['text'] = 'changed future'
        after = data.example_context(source, timeline, groups, query, 60_000_000)
        self.assertEqual(before, after[:2])

    def test_news_and_history_truncation_keeps_full_prefix_features(self):
        source, groups, timeline = self.raw()
        context, features, _ = data.example_context(source, timeline, groups, source['kickoff'] + 300_000_000,
            60_000_000, history_groups=1, max_news_items=1, max_news_chars=4)
        self.assertEqual(features['prior_execution_count'], 3)
        self.assertEqual(len(context['prior_executions']), 1)
        self.assertEqual(context['earlier_execution_groups_omitted'], 2)
        self.assertEqual(len(context['news']), 1)
        self.assertEqual(len(context['news'][0]['text']), 4)
        self.assertEqual(features['news_goal_count_900s'], 2)

    def test_hash_sampling_stable_and_independent_of_later_trade_times(self):
        _, first = self.prepare('first', max_rows=12)
        for source in self.sources:
            path = next((source / 'actors').glob('*'))
            rows = self.rows(path)
            last_time = data.activity.timestamp_us(rows[-1]['timestamp'])
            value = data.activity.utc(last_time + 30_000_000)
            rows[-2]['interval']['end'] = value
            rows[-1]['context_interval']['end'] = value
            rows[-1]['timestamp'] = value
            for execution in rows[-1]['label']['trades']:
                execution['time'] = value
            self.jsonl(path, rows)
        _, second = self.prepare('second', max_rows=12)
        for split in data.SPLITS:
            left = self.rows(first / (split+'.features.jsonl'))
            right = self.rows(second / (split+'.features.jsonl'))
            self.assertEqual([r['row_id'] for r in left], [r['row_id'] for r in right])

    def test_enrollment_excludes_first_trade_and_exact_boundary_is_next_window(self):
        self.assertEqual(list(data.activity.eligible_queries(60, 0, 240, 60)), [120, 180])
        source, groups, _ = self.raw()
        q = source['kickoff']
        self.assertEqual(data.activity.answer_at(groups, q, 60_000_000), 0)
        self.assertEqual(data.activity.answer_at(groups, q+60_000_000, 60_000_000), 1)

    def test_partial_capture_future_origin_and_early_capture_fail_closed(self):
        path = self.sources[0] / 'manifest.json'
        original = json.loads(path.read_text())
        cases = [({'source': {'api_traversal_status': 'paused'}}, 'incomplete API'),
                 ({'created_at': self.raw()[0]['market']['kickoff_utc']}, 'predates'),
                 ({'origin_utc': data.activity.utc(self.raw()[0]['kickoff']+1)}, 'cutoff follows')]
        for change, message in cases:
            data.write_json(path, {**original, **change})
            with self.assertRaisesRegex(ValueError, message):
                self.prepare()
            self.assertFalse((self.root / 'prepared').exists())
        data.write_json(path, original)

    def test_missing_actor_file_and_price_checksum_fail(self):
        path = self.sources[0] / 'market_price_history.jsonl'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.prepare()
        path.write_bytes(original)
        next((self.sources[0] / 'actors').glob('*')).unlink()
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.prepare()

    def test_export_cap_cannot_be_disabled_after_source_filtering(self):
        with self.assertRaisesRegex(ValueError, 'cannot recover'):
            self.prepare(max_trades_per_actor=0)

    def test_outside_world_cup_is_rejected(self):
        path = self.sources[0] / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['espn_event_id'] = '401915443'
        data.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'World Cup registry'):
            self.prepare()

    def test_risk_metrics_only_use_prior_declared_returns(self):
        source, groups, timeline = self.raw()
        q = source['kickoff']
        returns = [{'start_us': q-(i+2)*60_000_000, 'end_us': q-(i+1)*60_000_000,
                    'known_us': q-(i+1)*60_000_000, 'return': data.metrics.Decimal(value),
                    'benchmark_return': data.metrics.Decimal('0'), 'target_return': data.metrics.Decimal('0')}
                   for i, value in enumerate(('0.01', '-0.02', '0.03'))]
        _, features, _ = data.example_context(source, timeline, groups, q, 60_000_000,
                                             returns=returns, min_return_periods=3)
        self.assertIsNotNone(features['sharpe_ratio'])
        self.assertIsNotNone(features['sortino_ratio'])
        self.assertEqual(features['eligible_return_periods'], 3)
        returns[0]['known_us'] = q
        _, features, _ = data.example_context(source, timeline, groups, q, 60_000_000,
                                             returns=returns, min_return_periods=3)
        self.assertIsNone(features['sharpe_ratio'])

    def test_variant_preserves_targets_ids_and_common_context(self):
        _, source = self.prepare()
        out = self.root / 'selected'
        result = data.export_sft_variant(source, out, ['unrealized_in_market_pnl', 'prior_execution_count'])
        self.assertEqual(result['selected_sft_features'], ['unrealized_in_market_pnl', 'prior_execution_count'])
        for split in data.SPLITS:
            for original, selected in zip(self.rows(source / (split+'.jsonl')), self.rows(out / (split+'.jsonl'))):
                self.assertEqual(original['row_id'], selected['row_id'])
                self.assertEqual(original['messages'][-1], selected['messages'][-1])
                old, new = [json.loads(r['messages'][1]['content']) for r in (original, selected)]
                self.assertEqual(set(new.pop('derived_features')), {'unrealized_in_market_pnl', 'prior_execution_count'})
                old.pop('derived_features')
                self.assertEqual(old, new)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            data.export_sft_variant(source, self.root/'invalid', ['actor_id'])

    def test_coverage_certificate_required_for_file_source(self):
        source = self.sources[0]
        path = source / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['source'] = {'source_type': 'trade_file'}
        data.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'coverage uncertified'):
            self.prepare()
        kickoff = self.raw()[0]['kickoff']
        certificate = self.root / 'coverage.json'
        data.write_json(certificate, {'markets': {'1': {
            'start': data.activity.utc(kickoff-3600_000_000),
            'end': data.activity.utc(kickoff+360_000_000),
            'complete': True, 'reason': 'Fixture capture includes this complete interval'}}})
        result, _ = self.prepare(coverage_file=certificate)
        self.assertEqual(result['optional_ledger_sha256']['coverage'], data.activity.sha(certificate))

    def test_strict_chronology_purges_windows_at_presampling_boundaries(self):
        import shutil
        registry = copy.deepcopy(builder.BUNDLED_REGISTRY)
        fixtures = sorted(registry['fixtures'], key=lambda r: r['kickoff_utc'])[:3]
        initial = data.activity.timestamp_us(fixtures[0]['kickoff_utc'])
        shutil.rmtree(self.inputs)
        self.inputs.mkdir()
        self.sources = []
        for index, fixture in enumerate(fixtures):
            fixture['kickoff_utc'] = data.activity.utc(initial + index*240_000_000)
            self.sources.append(self.make_source(index+1, fixture))
        with patch.object(builder, 'BUNDLED_REGISTRY', registry):
            manifest, out = self.prepare()
        self.assertEqual(manifest['sampling']['purged_boundary_windows'], {'train': 2, 'validation': 2})
        for left, right in zip(data.SPLITS, data.SPLITS[1:]):
            older = self.rows(out / (left+'.features.jsonl'))
            later = self.rows(out / (right+'.features.jsonl'))
            self.assertLessEqual(max(r['interval_end'] for r in older), min(r['interval_start'] for r in later))

    def test_actor_identifier_can_be_omitted_from_prompts(self):
        _, out = self.prepare(include_actor_id=False)
        row = self.rows(out / 'train.jsonl')[0]
        self.assertIn('actor_id', row)
        self.assertNotIn('actor_id', json.loads(row['messages'][1]['content']))

    def test_duplicate_worldcup_match_markets_never_cross_splits(self):
        market = json.loads((self.sources[0] / 'market.json').read_text())
        fixture = next(f for f in builder.BUNDLED_REGISTRY['fixtures'] if f['fixture_id'] == market['fixture_id'])
        self.make_source(9, fixture)
        manifest, _ = self.prepare()
        train_sources = [source for source in manifest['sources'] if source['fixture_id'] == market['fixture_id']]
        self.assertEqual({s['market_id'] for s in train_sources}, {'1', '9'})
        self.assertEqual(len(manifest['fixture_to_split']), 3)


if __name__ == '__main__':
    unittest.main()
