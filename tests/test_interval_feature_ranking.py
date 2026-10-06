"""Safety invariants and a tiny real CPU XGBoost integration when installed."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rank_interval_features as ranking


class IntervalRankingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.out = self.root / 'ranking'

    def tearDown(self):
        self.temp.cleanup()

    def rows(self, count, fixture, seed=42):
        rng = random.Random(seed)
        rows = []
        for index in range(count):
            signal = rng.random()
            rows.append({'row_id': f'{fixture}-{index}', 'fixture_id': fixture,
                         'actor_id': f'actor-{index % 10}', 'label': int(signal > 0.65),
                         'features': {'signal': signal, 'noise': rng.random(),
                                      'constant': 1.0, 'all_missing': None}})
        return rows

    def write(self, split, rows):
        path = self.dataset / f'{split}.features.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def args(self, *extra):
        return ranking.parse_args(['train', '--dataset-dir', str(self.dataset), '--out', str(self.out),
                                   '--max-rounds', '40', '--early-stopping-rounds', '8',
                                   '--permutation-repeats', '2', '--top-k', '1', '--threads', '1',
                                   '--min-child-weight', '1', *extra])

    def test_metrics_handle_ties_and_one_class(self):
        metrics = ranking.binary_metrics([0, 1, 0, 1], [0.2, 0.2, 0.8, 0.8])
        self.assertAlmostEqual(metrics['roc_auc'], 0.5)
        self.assertAlmostEqual(metrics['average_precision'], 0.5)
        self.assertEqual(metrics['confusion'], {'tp': 1, 'tn': 1, 'fp': 1, 'fn': 1})
        self.assertEqual(sum(bin_['count'] for bin_ in metrics['calibration']), 4)
        perfect = ranking.binary_metrics([0, 1], [0, 1])
        self.assertEqual(perfect['roc_auc'], 1)
        self.assertEqual(perfect['average_precision'], 1)
        self.assertLess(perfect['log_loss'], 1e-12)
        one_class = ranking.binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(one_class['roc_auc'])
        self.assertIsNone(one_class['average_precision'])
        self.assertIsNone(one_class['recall'])
        self.assertEqual(ranking.choose_threshold([0, 0], [0.1, 0.2]), 0.5)

    def test_threshold_optimization_keeps_equal_scores_together(self):
        labels, scores = [1, 0, 1, 0], [0.8, 0.8, 0.4, 0.2]
        threshold = ranking.choose_threshold(labels, scores)
        self.assertEqual(threshold, 0.4)
        self.assertAlmostEqual(ranking.binary_metrics(labels, scores, threshold)['f1'], 0.8)
        with self.assertRaisesRegex(ValueError, 'Probabilities'):
            ranking.binary_metrics([1], [float('nan')])

    def test_reading_schema_and_split_guards(self):
        rows = self.rows(5, 'first')
        loaded = ranking.read_rows(self.write('train', rows))
        names, dropped = ranking.training_schema(loaded)
        self.assertEqual(names, ['noise', 'signal'])
        self.assertEqual(dropped, {'all_missing': 'all_missing_in_train', 'constant': 'constant_in_train'})
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            ranking.assert_disjoint(loaded, loaded)
        rows[1]['row_id'] = rows[0]['row_id']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ranking.read_rows(self.write('train', rows))
        rows = self.rows(1, 'second')
        rows[0]['features']['actor_id'] = 123
        with self.assertRaisesRegex(ValueError, 'leaking'):
            ranking.read_rows(self.write('train', rows))
        rows = self.rows(1, 'third')
        rows[0]['features']['signal'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'finite numeric'):
            ranking.read_rows(self.write('train', rows))

    def test_partial_missing_constant_can_convey_information(self):
        rows = self.rows(5, 'first')
        rows[0]['features']['constant'] = None
        names, dropped = ranking.training_schema(rows)
        self.assertIn('constant', names)
        self.assertNotIn('constant', dropped)

    def test_match_id_alias_and_string_labels(self):
        rows = self.rows(2, 'first')
        for index, row in enumerate(rows):
            row['match_id'] = row.pop('fixture_id')
            row['label'] = 'TRADE' if index else 'NO_TRADE'
        loaded = ranking.read_rows(self.write('train', rows))
        self.assertEqual([r['label'] for r in loaded], [0, 1])
        self.assertTrue(all(r['fixture_id'] == 'first' for r in loaded))

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'optional xgboost dependency not installed')
    def test_real_training_ranks_signal_never_opens_test_and_evaluates_frozen(self):
        self.write('train', self.rows(600, 'train', 1))
        self.write('validation', self.rows(200, 'validation', 2))
        test_file = self.dataset / 'test.features.jsonl'
        test_file.write_text('deliberately invalid: training must never read test')
        report = ranking.train(self.args())
        self.assertFalse(report['test_used'])
        self.assertGreater(report['validation']['roc_auc'], 0.97)
        self.assertEqual(json.loads((self.out / 'selected_features.json').read_text())['features'], ['signal'])
        ranking_rows = json.loads((self.out / 'feature_importance.json').read_text())
        self.assertEqual(ranking_rows[0]['feature'], 'signal')
        self.assertGreater(ranking_rows[0]['mean_log_loss_increase'], 0.05)
        with patch.object(ranking, 'fit_model', side_effect=AssertionError('must resume')):
            ranking.train(self.args())
        self.write('test', self.rows(200, 'test', 3))
        args = ranking.parse_args(['evaluate', '--dataset-dir', str(self.dataset),
                                   '--model-dir', str(self.out), '--threads', '1'])
        with patch.object(ranking, 'fit_model', side_effect=AssertionError('evaluation must never refit')):
            result = ranking.evaluate(args)
        self.assertGreater(result['metrics']['roc_auc'], 0.97)
        self.assertEqual(result['threshold_frozen_from_validation'], report['selected_validation']['threshold'])
        self.assertEqual(result['actor_cohorts']['seen_actors']['n'], 200)
        self.assertIsNone(result['actor_cohorts']['unseen_actors'])
        (self.out / 'selected_features.json').write_text('{"features": ["noise"]}')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            ranking.train(self.args())
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            ranking.evaluate(args)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'optional xgboost dependency not installed')
    def test_no_importance_selects_empty_list_and_constant_baseline(self):
        self.write('train', self.rows(100, 'train', 10))
        self.write('validation', self.rows(40, 'validation', 20))
        self.write('test', self.rows(40, 'test', 30))
        report = ranking.train(self.args('--min-child-weight', '1000'))
        self.assertEqual(json.loads((self.out / 'selected_features.json').read_text())['features'], [])
        self.assertIsNone(report['selected_best_iteration'])
        result = ranking.evaluate(ranking.parse_args(['evaluate', '--dataset-dir', str(self.dataset),
                                  '--model-dir', str(self.out), '--threads', '1']))
        self.assertAlmostEqual(result['metrics']['roc_auc'], 0.5)
        self.assertEqual(result['metrics']['log_loss'], result['baselines']['training_prevalence']['log_loss'])
        self.write('train', self.rows(100, 'train', 11))
        with self.assertRaisesRegex(ValueError, 'different data/settings'):
            ranking.train(self.args('--min-child-weight', '1000'))


if __name__ == '__main__':
    unittest.main()
