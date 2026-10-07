import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_interval_experiment as runner
import rank_interval_features as ranking
import train_world_cup_multigpu as trainer


class IntervalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'raw'
        self.input.mkdir()
        (self.input / 'source.json').write_text('{}')
        self.args = runner.parse_args(['--input-root', str(self.input), '--out', str(self.root / 'job')])
        (self.args.out / 'stages').mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_child_commands_parse_with_real_interfaces(self):
        xgb = runner.xgb_command(self.args, self.root / 'dataset', self.root / 'xgb')
        child = ranking.parse_args([str(v) for v in xgb[3:]])
        self.assertEqual(child.top_k, 12)
        self.assertEqual(child.n_jobs, self.args.xgb_threads)
        train = runner.train_command(self.args, self.root / 'dataset', self.root / 'run')
        child = trainer.parse_args([str(v) for v in train[3:]])
        self.assertTrue(child.smoke_then_full)
        self.assertFalse(child.allow_trade_only)
        self.assertEqual(child.dataset_dir, self.root / 'dataset')
        self.assertNotIn('--test', train)

    def test_resume_verifies_stages_and_rejects_changed_artifacts(self):
        destination = self.args.out / 'dataset'
        calls = []
        def build():
            calls.append(1)
            destination.mkdir()
            (destination / 'train.jsonl').write_text('original\n')
        runner.stage(self.args, 'prepare', destination, build)
        self.args.resume = True
        runner.stage(self.args, 'prepare', destination, build)
        self.assertEqual(calls, [1])
        (destination / 'train.jsonl').write_text('edited\n')
        with self.assertRaisesRegex(ValueError, 'outputs changed'):
            runner.stage(self.args, 'prepare', destination, build)

    def test_incomplete_stages_are_preserved_not_overwritten(self):
        destination = self.args.out / 'dataset'
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'Unfinished'):
            runner.stage(self.args, 'prepare', destination, lambda: self.fail('must not run'))

    def test_interrupted_ranking_uses_its_own_resume_guard(self):
        destination = self.args.out / 'xgboost'
        destination.mkdir()
        (destination / 'run_state.json').write_text('{"status":"running"}')
        self.args.resume = True
        calls = []
        runner.stage(self.args, 'xgboost', destination, lambda: calls.append('safe-ranker'))
        self.assertEqual(calls, ['safe-ranker'])
        self.assertTrue((self.args.out / 'stages/xgboost.json').exists())

    def test_output_cannot_enter_source_inventory(self):
        with self.assertRaisesRegex(ValueError, 'outside --input-root'):
            runner.parse_args(['--input-root', str(self.input), '--out', str(self.input / 'job')])

    def test_stop_modes_do_not_change_resume_configuration(self):
        initial = runner.configuration(self.args)
        self.args.prepare_only, self.args.resume = True, True
        self.assertEqual(initial, runner.configuration(self.args))
        self.args.window_seconds = 600
        self.assertNotEqual(initial, runner.configuration(self.args))

    def test_invalid_gpu_and_batch_config_rejected_before_backgrounding(self):
        for extra in (['--gpus', '2', '--gpu-ids', '0,0'], ['--global-batch', '3']):
            with self.assertRaises(ValueError):
                runner.parse_args(['--input-root', str(self.input), '--out', str(self.root / 'other'), *extra])

    def test_details_requires_explicit_tolerances_and_keeps_size_features(self):
        base = ['--input-root', str(self.input), '--out', str(self.root / 'details')]
        with self.assertRaisesRegex(ValueError, 'requires explicit'):
            runner.parse_args([*base, '--target-mode', 'trade-details'])
        args = runner.parse_args([*base, '--target-mode', 'trade-details',
            '--price-delta', '.02', '--shares-relative-delta', '.20'])
        self.assertEqual(args.sft_features, 'all')
        self.assertEqual(args.trade_tolerances, {'price_delta': '0.02',
            'shares_relative_delta': '0.2', 'shares_absolute_delta': '0'})
        self.assertEqual(self.args.sft_features, 'selected')
        with self.assertRaisesRegex(ValueError, 'Numeric tolerances require'):
            runner.parse_args([*base, '--price-delta', '.02'])


if __name__ == '__main__':
    unittest.main()
