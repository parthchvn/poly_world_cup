"""Frozen binary probe leakage/alignment guards and tiny real CPU integration."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fit_frozen_activity_probe as probe
import rank_interval_features as ranking

AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'sklearn', 'matplotlib'))


@unittest.skipUnless(AVAILABLE, 'optional CPU fit dependencies absent')
class FrozenProbeFitTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_metrics_null_precision_and_threshold_diagnostic(self):
        metrics = probe.metrics([0, 0, 1], [0.01, 0.02, 0.2])
        self.assertIsNone(metrics['precision'])
        self.assertEqual(metrics['recall'], 0)
        self.assertEqual(metrics['average_precision'], 1)
        self.assertEqual(probe.metrics([0, 1], [0.4, 0.4])['average_precision'], 0.5)
        self.assertIsNone(probe.metrics([0, 0], [0.1, 0.2])['roc_auc'])
        threshold = ranking.choose_threshold([0, 0, 1], [0.01, 0.02, 0.2])
        self.assertEqual(probe.metrics([0, 0, 1], [0.01, 0.02, 0.2], threshold)['f1'], 1)

    def test_scaler_and_unweighted_portable_logistic(self):
        np = self.np
        train = np.asarray([[0., 2.], [1., 4.], [2., 6.], [3., 8.], [4., 10.], [5., 12.]], dtype=np.float32)
        validation = np.asarray([[100., 202.], [200., 402.]], dtype=np.float32)
        expected_mean = train.mean(axis=0)
        original_validation = validation.copy()
        fit = probe.fit_logistic_grid(train, [0, 0, 0, 0, 1, 1], validation, [0, 1], 'test', 500, np)
        np.testing.assert_allclose(fit['mean'], expected_mean)
        logits = ((original_validation - fit['mean']) / fit['scale']) @ fit['coef'].T + fit['intercept']
        np.testing.assert_allclose(1 / (1 + np.exp(-logits[:, 0])), fit['probabilities'], atol=1e-6)
        self.assertEqual([row['C'] for row in fit['candidates']], list(probe.C_VALUES))

    def test_recent_missing_imputation_uses_training_only(self):
        np = self.np
        def row(value):
            return {'features': {name: value for name in probe.RECENT_FEATURES}}
        x, v, preprocessing = probe.recent_arrays([row(0), row(8), row(None)], [row(None), row(10000)], np)
        np.testing.assert_allclose(preprocessing['medians'], np.log(9) / 2, atol=1e-6)
        np.testing.assert_allclose(v[0, :6], preprocessing['medians'])
        np.testing.assert_allclose(v[0, 6:], 1)

    def test_feature_alignment_rejects_leakage_and_wrong_gold(self):
        meta = [{'row_id': 'a', 'label': 0, 'actor_id': 'actor', 'fixture_id': 'fixture'}]
        feature = [{**meta[0], 'features': {'x': 1}}]
        self.assertEqual(probe.align_feature_rows(meta, feature), feature)
        with self.assertRaisesRegex(ValueError, 'label mismatch'):
            probe.align_feature_rows(meta, [{**feature[0], 'label': 1}])
        with self.assertRaisesRegex(ValueError, 'row IDs differ'):
            probe.align_feature_rows(meta, [])
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            ranking.assert_disjoint(meta, meta)

    def write_shards(self, meta, embeddings, directory, identity):
        directory.mkdir(exist_ok=True)
        ranking.atomic_json(directory / 'identity.json', identity)
        identity_hash = ranking.digest(directory / 'identity.json')
        for shard in range(2):
            target = directory / f'shard{shard}'
            target.mkdir(exist_ok=True)
            hashes = {}
            for split in probe.SPLITS:
                rows = meta[split][shard::2]
                self.np.savez(target / f'{split}.npz', X=embeddings[split][shard::2],
                    row_ids=self.np.asarray([r['row_id'] for r in rows]),
                    labels=self.np.asarray([r['label'] for r in rows]),
                    fixture_ids=self.np.asarray([r['fixture_id'] for r in rows]),
                    actor_ids=self.np.asarray([r['actor_id'] for r in rows]))
                hashes[split] = ranking.digest(target / f'{split}.npz')
            ranking.atomic_json(target / 'complete.json', {'identity_sha256': identity_hash,
                'hidden_size': embeddings['train'].shape[1], 'sha256': hashes, 'test_used': False})

    def fixture(self):
        np = self.np
        dataset, features, embeds, xgbdir = (self.root / name for name in ('sft', 'features', 'embeddings', 'xgboost'))
        dataset.mkdir(); features.mkdir()
        meta, vectors, sft_hashes, feature_hashes = {}, {}, {}, {}
        rng = np.random.default_rng(23)
        for split, count in (('train', 80), ('validation', 24)):
            x = rng.normal(size=(count, 4)).astype(np.float32)
            meta[split] = [{'row_id': f'{split}{i}', 'label': int(x[i, 0] > 0.7),
                'actor_id': f'actor{i % 12}', 'fixture_id': split + 'fixture'} for i in range(count)]
            vectors[split] = x
            sf, ff = [], []
            for i, row in enumerate(meta[split]):
                sf.append({**row, 'messages': [{'role': 'system', 'content': 'predict'},
                    {'role': 'user', 'content': '{}'}, {'role': 'assistant',
                     'content': json.dumps({'action': 'TRADE' if row['label'] else 'NO_TRADE'})}]})
                ff.append({**row, 'features': {name: float(abs(x[i, j % 4]))
                    for j, name in enumerate(probe.RECENT_FEATURES)}})
            sft_path = dataset / f'{split}.jsonl'
            feat_path = features / f'{split}.features.jsonl'
            sft_path.write_text(''.join(json.dumps(r) + '\n' for r in sf))
            feat_path.write_text(''.join(json.dumps(r) + '\n' for r in ff))
            sft_hashes[split] = ranking.digest(sft_path)
            feature_hashes[split] = ranking.digest(feat_path)
        ranking.atomic_json(dataset / 'manifest.json', {'files': {f'{s}.jsonl': d for s, d in sft_hashes.items()}})
        ranking.atomic_json(features / 'manifest.json', {'files': {f'{s}.features.jsonl': d for s, d in feature_hashes.items()}})
        identity = {'test_used': False, 'input_sha256': sft_hashes,
                    'splits': {s: {'rows': len(meta[s])} for s in probe.SPLITS}}
        self.write_shards(meta, vectors, embeds, identity)
        ranking.train(ranking.parse_args(['train', '--dataset-dir', str(features), '--out', str(xgbdir),
            '--max-rounds', '5', '--early-stopping-rounds', '2', '--permutation-repeats', '1', '--threads', '1']))
        args = probe.parse_args(['--dataset-dir', str(dataset), '--features-dir', str(features),
            '--embeddings-dir', str(embeds), '--xgb-dir', str(xgbdir), '--out', str(self.root / 'out'), '--threads', '1'])
        return args, meta, vectors, identity

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'optional XGBoost absent')
    def test_exact_shard_alignment_and_hash_guards(self):
        args, meta, vectors, identity = self.fixture()
        x, _ = probe.load_embeddings(args.embeddings_dir, 'train', meta['train'], identity, self.np)
        self.np.testing.assert_array_equal(x, vectors['train'])
        changed = [dict(row) for row in meta['train']]
        changed[0]['label'] = 1 - changed[0]['label']
        with self.assertRaisesRegex(ValueError, 'gold/identity mismatch'):
            probe.load_embeddings(args.embeddings_dir, 'train', changed, identity, self.np)
        with (args.embeddings_dir / 'shard0/train.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            probe.load_embeddings(args.embeddings_dir, 'train', meta['train'], identity, self.np)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'optional XGBoost absent')
    def test_end_to_end_never_opens_test_and_preserves_natural_labels(self):
        args, meta, vectors, identity = self.fixture()
        original = Path.open
        def protected(path, *a, **kw):
            if path.name.startswith('test.'):
                raise AssertionError(f'Test data was opened: {path}')
            return original(path, *a, **kw)
        with patch.object(Path, 'open', protected):
            report = probe.fit(args)
        self.assertFalse(report['test_used'])
        self.assertEqual(report['training_rows'], 80)
        self.assertEqual(report['validation_rows'], 24)
        self.assertAlmostEqual(report['training_prevalence'], sum(r['label'] for r in meta['train']) / 80)
        self.assertEqual(set(report['models']), {'training_prevalence', 'recent_activity_logistic', 'xgboost_full', 'frozen_9b_probe'})
        self.assertTrue((args.out / 'validation_pr_calibration.png').is_file())
        self.assertEqual(len((args.out / 'val_predictions.csv').read_text().splitlines()), 25)
        saved = self.np.load(args.out / 'frozen_probe.npz', allow_pickle=False)
        self.np.testing.assert_allclose(saved['mean'], vectors['train'].mean(axis=0), atol=1e-6)
        state = json.loads((args.out / 'run_state.json').read_text())
        self.assertEqual(state['status'], 'completed')
        for name, digest in state['output_sha256'].items():
            self.assertEqual(ranking.digest(args.out / name), digest)

    def test_partial_and_completed_resume_verify_signatures_and_hashes(self):
        out = self.root / 'resume'
        signature = {'input': 'hash', 'C_values': list(probe.C_VALUES)}
        self.assertIsNone(probe.prepare_output(out, signature))
        (out / 'partial.txt').write_text('interrupted work')
        self.assertIsNone(probe.prepare_output(out, signature))
        with self.assertRaisesRegex(ValueError, 'different inputs/settings'):
            probe.prepare_output(out, {**signature, 'input': 'changed'})
        names = ['report.json', 'selected_params.json', 'val_predictions.csv', 'frozen_probe.npz',
                 'recent_activity_probe.npz', 'validation_pr_calibration.png']
        for name in names:
            (out / name).write_text('{}')
        ranking.atomic_json(out / 'run_state.json', {'status': 'completed', 'test_used': False,
            'signature': signature, 'output_sha256': {n: ranking.digest(out / n) for n in names}})
        self.assertEqual(probe.prepare_output(out, signature), {})
        (out / 'report.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            probe.prepare_output(out, signature)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'optional XGBoost absent')
    def test_xgboost_artifact_tampering_is_rejected(self):
        args, meta, vectors, identity = self.fixture()
        model_path = args.xgb_dir / 'model.json'
        model_path.write_text(model_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Changed XGBoost artifact'):
            probe.fit(args)


if __name__ == '__main__':
    unittest.main()
