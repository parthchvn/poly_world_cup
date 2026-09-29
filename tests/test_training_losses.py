"""Offline loss persistence and plotting tests; no model or CUDA dependencies."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load_script('train_world_cup_multigpu')
plotting = load_script('plot_training_losses')


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        self.state = SimpleNamespace(global_step=0, epoch=0.0, is_world_process_zero=True)
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(TrainerCallback=object)}):
            self.callback = training.make_loss_callback(self.run)

    def begin(self, step=0):
        self.state.global_step = step
        self.callback.on_train_begin(None, self.state, None)

    def log(self, step, **metrics):
        self.state.global_step = step
        self.callback.on_log(None, self.state, None, logs=metrics)

    def test_records_are_readable_before_training_finishes(self):
        self.begin()
        metrics = {'loss': 2.3, 'learning_rate': 0.0001, 'grad_norm': 0.5}
        self.log(1, **metrics)
        self.log(10, smoke_eval_loss=2.0)
        self.log(20, eval_loss=1.5)
        rows = plotting.read_rows(self.run / 'metrics.jsonl')
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]['loss'], 2.3)
        self.assertIn('timestamp_utc', rows[1])
        with (self.run / 'losses.csv').open() as source:
            exported = list(csv.DictReader(source))
        self.assertEqual(len(exported), 4)
        self.assertEqual(exported[1]['step'], '1')
        self.assertEqual(exported[1]['learning_rate'], '0.0001')
        self.assertEqual(exported[2]['loss'], '')
        self.assertEqual(exported[3]['eval_loss'], '1.5')
        self.assertEqual(plotting.loss_series(rows), plotting.loss_series(exported))

    def test_other_ranks_do_not_touch_files(self):
        self.state.is_world_process_zero = False
        self.begin()
        self.log(1, loss=2.0)
        self.assertFalse(self.run.exists())

    def test_resume_keeps_history_but_plot_removes_abandoned_tail(self):
        self.begin()
        self.log(1, loss=3.0)
        self.log(10, loss=2.0, eval_loss=2.5)
        self.log(20, loss=1.0, eval_loss=1.5)
        self.begin(step=10)
        self.log(11, loss=1.9)
        for filename in ('metrics.jsonl', 'losses.csv'):
            rows = plotting.read_rows(self.run / filename)
            self.assertEqual(len(rows), 6)
            self.assertEqual(plotting.loss_series(rows)['loss'], [(1, 3.0), (10, 2.0), (11, 1.9)])
            self.assertEqual(plotting.loss_series(rows)['eval_loss'], [(10, 2.5)])
        self.assertEqual((self.run / 'losses.csv').read_text().count('timestamp_utc,event,step'), 1)

    def test_truncated_live_record_is_ignored_and_repaired_before_resume(self):
        self.begin()
        self.log(1, loss=2.0)
        for filename in ('metrics.jsonl', 'losses.csv'):
            with (self.run / filename).open('a') as target:
                target.write('{"step":2,"loss":')
            self.assertEqual(len(plotting.read_rows(self.run / filename)), 2)
        self.begin(step=1)
        self.log(2, loss=1.5)
        for filename in ('metrics.jsonl', 'losses.csv'):
            rows = plotting.read_rows(self.run / filename)
            self.assertEqual(len(rows), 4)
            self.assertEqual(plotting.loss_series(rows)['loss'], [(1, 2.0), (2, 1.5)])

    def test_old_version_logs_and_summary_are_handled(self):
        self.run.mkdir()
        old_rows = [{'step': 1, 'loss': 2.0}, {'step': 10, 'loss': 1.0},
                    {'step': 10, 'eval_loss': 1.2}, {'step': 10, 'smoke_eval_loss': 1.3},
                    {'step': 10, 'train_loss': 1.5, 'train_runtime': 100}]
        (self.run / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in old_rows))
        series = plotting.loss_series(plotting.read_rows(self.run / 'metrics.jsonl'))
        self.assertEqual(series, {'loss': [(1, 2.0), (10, 1.0)], 'eval_loss': [(10, 1.2)],
                                  'smoke_eval_loss': [(10, 1.3)]})

    def test_nonfinite_loss_is_not_silently_shown_as_healthy(self):
        self.begin()
        self.log(1, loss=float('nan'))
        self.log(2, loss=1.0)
        with self.assertWarnsRegex(UserWarning, 'nonfinite'):
            points = plotting.loss_series(plotting.read_rows(self.run / 'metrics.jsonl'))['loss']
        self.assertTrue(plotting.math.isnan(points[0][1]))
        self.assertEqual(points[1], (2, 1.0))

    def test_logging_cadence_defaults_to_every_optimizer_step(self):
        self.assertEqual(training.parse_args(['--gpus', '2']).logging_steps, 1)
        self.assertEqual(training.parse_args(['--gpus', '2', '--logging-steps', '10']).logging_steps, 10)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            training.parse_args(['--gpus', '2', '--logging-steps', '0'])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is optional')
    def test_plot_cli_creates_png_from_old_jsonl_and_csv_fallback(self):
        self.begin()
        self.log(1, loss=2.0)
        self.log(10, loss=1.0, eval_loss=1.2)
        with contextlib.redirect_stdout(io.StringIO()):
            plotting.main(['--run-dir', str(self.run)])
        output = self.run / 'loss_curve.png'
        self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        (self.run / 'metrics.jsonl').unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            plotting.main(['--run-dir', str(self.run), '--output', str(self.run / 'from_csv.png')])
        self.assertGreater((self.run / 'from_csv.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
