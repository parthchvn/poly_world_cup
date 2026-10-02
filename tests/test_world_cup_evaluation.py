import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
import world_cup_eval_common as common
import prepare_world_cup_evaluation as prepare
import evaluate_world_cup as evaluate
import compare_world_cup_evaluations as compare
from tests import test_actor_variant_comparison as fixture_module


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ActorVariantComparisonTests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def args(self):
        return prepare.parse_args(['--basic-sft', str(self.fixture.datasets['basic']),
            '--inmarket-sft', str(self.fixture.datasets['inmarket']), '--out', str(self.root / 'bundle')])

    def test_bundle_preserves_pairs_without_network(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
            meta = prepare.prepare(self.args())
        self.assertEqual(meta['targets'], 2)
        self.assertEqual(meta['fixtures'], ['fixture2'])
        loaded, rows = common.read_bundle(self.root / 'bundle')
        self.assertEqual(loaded, meta)
        self.assertEqual(rows['basic'][0], self.fixture.records['test'][0])
        self.assertEqual(set(p.name for p in (self.root / 'bundle').iterdir()),
                         {'manifest.json', 'basic.jsonl.gz', 'inmarket.jsonl.gz'})

    def test_target_prefix_excludes_current_and_future(self):
        record = copy.deepcopy(self.fixture.records['test'][0])
        record['messages'][2]['content'] = '{"action":"TRADE","trades":[{"marker":"CURRENT_SECRET"}]}'
        record['messages'][4]['content'] = '{"action":"TRADE","trades":[{"marker":"FUTURE_SECRET"}]}'
        ts = list(common.targets(record))
        self.assertNotIn('CURRENT_SECRET', common.dump(ts[0]['messages']))
        self.assertNotIn('FUTURE_SECRET', common.dump(ts[0]['messages']))
        self.assertIn('CURRENT_SECRET', common.dump(ts[1]['messages']))
        self.assertNotIn('FUTURE_SECRET', common.dump(ts[1]['messages']))
        self.assertEqual(ts[0]['messages'][-1]['role'], 'user')

    def test_pair_detects_target_and_context_changes(self):
        b = self.fixture.records['test'][0]
        m = next(common.lines(self.fixture.datasets['inmarket'] / 'test.jsonl'))[1]
        common.check_pair(b, m)
        changed = copy.deepcopy(m)
        changed['messages'][2]['content'] = changed['messages'][2]['content'].replace('BUY', 'SELL')
        with self.assertRaisesRegex(ValueError, 'target differs'):
            common.check_pair(b, changed)
        changed = copy.deepcopy(m)
        context = json.loads(changed['messages'][1]['content'])
        context['extra_future_pnl'] = 10
        changed['messages'][1]['content'] = json.dumps(context)
        with self.assertRaisesRegex(ValueError, 'contexts differ'):
            common.check_pair(b, changed)

    def test_rejects_same_match_different_market(self):
        self.fixture.records['test'][0]['fixture_id'] = 'fixture0'
        self.fixture.write_all()
        with self.assertRaisesRegex(ValueError, 'Fixture appears'):
            prepare.prepare(self.args())

    def test_rejects_overlapping_timestamps(self):
        for record in self.fixture.records['test']:
            for message in record['messages']:
                if message['role'] == 'user':
                    message['content'] = message['content'].replace('2026-06-09', '2026-06-01').replace('2026-06-10', '2026-06-02')
        self.fixture.write_all()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            prepare.prepare(self.args())

    def test_rejects_file_tampering(self):
        prepare.prepare(self.args())
        path = self.root / 'bundle/basic.jsonl.gz'
        path.write_bytes(path.read_bytes() + b'extra')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            common.read_bundle(self.root / 'bundle')

    def test_metrics_are_strictly_prior_and_same_timestamp_excluded(self):
        record = copy.deepcopy(self.fixture.records['test'][0])
        groups = []
        for i in (1, 3):
            when = prepare.timestamp_us(json.loads(record['messages'][i]['content'])['query_time'])
            expected = json.loads(record['messages'][i + 1]['content'])['trades']
            groups.append({'time_us': when, 'expected': expected,
                'trades': [{'time_us': when, 'shares': prepare.metrics.number('2', 'shares'),
                            'price': prepare.metrics.number('0.4', 'price'), 'side': 'BUY'}]})
        config = {'history_scope': 'actor_and_binary_market',
                  'selected_features': ['average_execution_notional', 'buy_notional_share']}
        enriched = prepare.enrich_record(record, groups, config)
        first = json.loads(enriched['messages'][1]['content'])['actor_metrics']
        second = json.loads(enriched['messages'][3]['content'])['actor_metrics']
        self.assertEqual(first['sample_counts']['captured_executions'], 0)
        self.assertEqual(first['values'], {})
        self.assertEqual(second['sample_counts']['captured_executions'], 1)
        self.assertEqual(second['values']['average_execution_notional'], '0.8')
        groups[-1]['trades'][0]['shares'] = prepare.metrics.number('999', 'shares')
        self.assertEqual(enriched, prepare.enrich_record(record, groups, config))

    def test_journal_recovers_only_partial_last_line_and_checks_identity(self):
        target = list(common.targets(self.fixture.records['test'][0]))[0]
        row = {**target, 'prompt_sha256': hashlib.sha256(common.dump(target['messages']).encode()).hexdigest()}
        path = self.root / 'predictions.jsonl'
        path.write_text(json.dumps(row) + '\n{"id":')
        recovered = evaluate.recover_predictions(path, [target], True)
        self.assertEqual(recovered, [row])
        self.assertTrue(path.read_bytes().endswith(b'\n'))
        with self.assertRaisesRegex(ValueError, 'use --resume'):
            evaluate.recover_predictions(path, [target], False)
        changed = {**target, 'answer': 'changed'}
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            evaluate.recover_predictions(path, [changed], True)

    def fake_training(self, meta):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        run = self.root / 'run'
        (run / 'adapter').mkdir(parents=True)
        for f in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'):
            (run / 'adapter' / f).write_text('{}')
        ref = meta['reference']['basic']
        training = {'status': 'completed', 'mode': 'full', 'test_used': False,
            'signature': {'data': {'model_config_sha256': common.sha(model / 'config.json'),
                'sources': {s: {'sha256': ref['source_sha256'][s]} for s in ('train', 'validation')}}},
            'data': {s: {'sha256': ref['split_sha256'][s], 'fixtures': ref['fixtures'][s]}
                     for s in ('train', 'validation')}}
        common.write_json(run / 'training_metadata.json', training)
        return model, run, training

    def test_fresh_export_pipeline_builds_same_causal_features(self):
        from tests.test_prepare_actor_experiment import ActorExperimentTests
        import prepare_actor_experiment as experiment
        exports = ActorExperimentTests()
        exports.setUp()
        try:
            export = exports.source(20, actors=1, compressed=True)
            args = self.args()
            args.market_ids = ['20']
            args.capture_root = export.parent
            manifest_path = self.fixture.datasets['inmarket'] / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['actor_metrics']['config'].update(selected_features=[
                'average_execution_notional', 'execution_notional_cv', 'executions_per_day', 'buy_notional_share'])
            common.write_json(manifest_path, manifest)
            registry = {'fixtures': [{'fixture_id': 'espn:20', 'kickoff_utc': '2026-06-20T17:00:00Z'}],
                        'contracts': [{'market_id': '20', 'fixture_id': 'espn:20'}]}
            with patch.object(experiment.builder, 'BUNDLED_REGISTRY', registry), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('No network')):
                meta = prepare.prepare(args)
            self.assertEqual(meta['source'], 'fresh_markets')
            self.assertEqual(meta['targets'], 4)
            _, records = common.read_bundle(args.out)
            first, second = [json.loads(records['inmarket'][0]['messages'][i]['content'])
                             for i in (1, 5)]
            self.assertEqual(first['actor_metrics']['sample_counts']['captured_executions'], 0)
            self.assertEqual(second['actor_metrics']['sample_counts']['captured_executions'], 2)
        finally:
            exports.tearDown()

    def test_run_identity_checks_training_hash_and_holdout(self):
        meta = prepare.prepare(self.args())
        model, run, training = self.fake_training(meta)
        evaluate.validate_run(run, model, 'basic', meta)
        training['data']['train']['fixtures'].append('fixture2')
        common.write_json(run / 'training_metadata.json', training)
        with self.assertRaisesRegex(ValueError, 'seen during'):
            evaluate.validate_run(run, model, 'basic', meta)
        training['data']['train']['fixtures'].pop()
        training['signature']['data']['sources']['train']['sha256'] = 'wrong'
        common.write_json(run / 'training_metadata.json', training)
        with self.assertRaisesRegex(ValueError, 'differs'):
            evaluate.validate_run(run, model, 'basic', meta)

    def test_run_rejects_incomplete_and_smoke(self):
        meta = prepare.prepare(self.args())
        model, run, training = self.fake_training(meta)
        for update in ({'status': 'running'}, {'status': 'completed', 'mode': 'smoke'},
                       {'status': 'running', 'mode': 'smoke_then_full'},
                       {'status': 'failed', 'mode': 'smoke_then_full'},
                       {'status': 'completed', 'mode': 'benchmark'}):
            training.update(update)
            common.write_json(run / 'training_metadata.json', training)
            with self.assertRaisesRegex(ValueError, 'COMPLETED full'):
                evaluate.validate_run(run, model, 'basic', meta)

    def test_completed_smoke_then_full_is_a_valid_full_run(self):
        meta = prepare.prepare(self.args())
        model, run, training = self.fake_training(meta)
        training.update(mode='smoke_then_full', completed_steps=2712,
                        smoke_check={'status': 'passed', 'step': 10})
        training['signature']['smoke_then_full'] = True
        common.write_json(run / 'training_metadata.json', training)
        result, adapter = evaluate.validate_run(run, model, 'basic', meta)
        self.assertEqual(result['completed_steps'], 2712)
        self.assertEqual(adapter, run / 'adapter')
        training['signature']['data']['sources']['train']['sha256'] = 'wrong'
        common.write_json(run / 'training_metadata.json', training)
        with self.assertRaisesRegex(ValueError, 'differs'):
            evaluate.validate_run(run, model, 'basic', meta)


class ScoringTests(unittest.TestCase):
    def label(self, trades=None):
        return json.dumps({'action': 'TRADE', 'trades': trades or [
            {'side': 'BUY', 'outcome': 'Yes', 'shares': '2', 'price': '0.4'}]})

    def test_numeric_equivalence_and_order_invariance(self):
        a = {'side': 'BUY', 'outcome': 'Yes', 'shares': '2', 'price': '.4'}
        b = {'side': 'SELL', 'outcome': 'No', 'shares': '3', 'price': '.6'}
        self.assertEqual(common.score(self.label([a, b]), self.label([b, a]))['exact_trade_multiset'], 1)
        c = {**a, 'shares': 2, 'price': .40}
        self.assertEqual(common.score(self.label([a]), self.label([c]))['exact_trade_multiset'], 1)

    def test_malformed_and_invalid_numeric_predictions_count_as_failures(self):
        for pred in ('garbage', '```json\n' + self.label() + '\n```',
                     self.label().replace('"2"', '"NaN"'),
                     self.label().replace('"0.4"', 'true'),
                     self.label().replace('"0.4"', '"1.1"'),
                     '{"action":"TRADE","action":"TRADE","trades":[]}'):
            self.assertEqual(common.score(self.label(), pred)['valid_json'], 0)

    def test_duplicates_and_count_errors_penalized(self):
        t = json.loads(self.label())['trades'][0]
        result = common.score(self.label([t, t]), self.label([t]))
        self.assertEqual(result['trade_count_correct'], 0)
        self.assertEqual(result['side_outcome_multiset_correct'], 0)
        self.assertEqual(result['numeric_matched_trades'], 0)

    def test_numeric_denominator_only_category_matched(self):
        rows = [{'answer': self.label(), 'prediction': self.label().replace('"0.4"', '"0.5"')},
                {'answer': self.label(), 'prediction': self.label().replace('BUY', 'SELL')}]
        summary = common.summarize(rows)
        self.assertEqual(summary['numeric_target_coverage'], .5)
        self.assertAlmostEqual(summary['price_mae_conditional'], .1)
        self.assertEqual(summary['exact_trade_multiset'], 0)

    def fake_result(self, root, variant, ids=None, fixtures=None):
        ids = ids or ['a', 'b']
        rows = [{'id': k, 'sequence_id': k, 'actor_id': k, 'fixture_id': (fixtures or ['f', 'f'])[i],
                 'market_id': '1', 'query_time': str(i), 'answer': self.label(), 'prediction': self.label()}
                for i, k in enumerate(ids)]
        root.mkdir()
        identity = {'variant': variant, 'selected_targets': len(rows),
            'selected_ids_sha256': hashlib.sha256(common.dump(ids).encode()).hexdigest(),
            'training_signature': {'max_length': 8192 if variant == 'basic' else 16384},
            'bundle_sha256': 'same', 'target_sha256': 'same', 'decoding': {}, 'limit': 0,
            'history_protocol': 'same', 'tokenizer_sha256': 'same', 'versions': {},
            'evaluator_sha256': 'same', 'common_sha256': 'same'}
        common.write_json(root / 'identity.json', identity)
        (root / 'predictions.jsonl').write_text(''.join(common.dump(r) + '\n' for r in rows))
        common.write_json(root / 'summary.json', {'status': 'completed', 'variant': variant,
             'identity_sha256': common.sha(root / 'identity.json'),
             'predictions_sha256': common.sha(root / 'predictions.jsonl')})

    def test_comparison_single_match_has_no_confidence_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for v in ('basic', 'inmarket'):
                self.fake_result(root / v, v)
            result = compare.compare(root / 'basic', root / 'inmarket')
            self.assertIsNone(result['match_cluster_bootstrap_95ci'])
            self.assertEqual(result['delta_inmarket_minus_basic']['exact_trade_multiset'], 0)

    def test_comparison_bootstrap_multiple_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for v in ('basic', 'inmarket'):
                self.fake_result(root / v, v, fixtures=['f1', 'f2'])
            result = compare.compare(root / 'basic', root / 'inmarket')
            self.assertEqual(result['match_cluster_bootstrap_95ci']['exact_trade_multiset'], [0, 0])

    def test_comparison_rejects_different_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fake_result(root / 'basic', 'basic')
            self.fake_result(root / 'inmarket', 'inmarket', ids=['a', 'c'])
            with self.assertRaisesRegex(ValueError, 'selected_ids'):
                compare.compare(root / 'basic', root / 'inmarket')


if __name__ == '__main__':
    unittest.main()
