"""Offline tests for the single-run smoke evaluation contract.

The Trainer stub tests our public evaluate override without GPU dependencies.
Real Qwen/Transformers/DDP execution remains an on-Pod integration check.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'single_load_trainer', Path(__file__).resolve().parents[1] / 'scripts/train_world_cup_multigpu.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Dataset:
    def __init__(self, rows):
        self.rows = tuple(rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return Dataset(self.rows[i] for i in indices)


class FakeTrainer:
    def __init__(self, model=None, train_dataset=None, eval_dataset=None, **kwargs):
        self.model, self.train_dataset, self.eval_dataset = model, train_dataset, eval_dataset
        self.optimizer, self.lr_scheduler = object(), object()
        self.state = SimpleNamespace(global_step=0)
        self.is_in_train = True
        self.evaluations = []
        self.next_loss = 0.5

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval', **kwargs):
        data = self.eval_dataset if eval_dataset is None else eval_dataset
        self.evaluations.append((self.state.global_step, data.rows, metric_key_prefix))
        return {metric_key_prefix + '_loss': self.next_loss}

    def is_world_process_zero(self):
        return False


class SmokeThenFullTests(unittest.TestCase):
    def trainer(self, size=100, check_step=10):
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(Trainer=FakeTrainer)}):
            cls = module.make_trainer_class()
        self.reports = []
        return cls(model=object(), train_dataset=Dataset(range(200)), eval_dataset=Dataset(range(size)),
                   smoke_check_step=check_step, smoke_reporter=self.reports.append)

    def test_check_preserves_model_optimizer_scheduler_and_full_datasets(self):
        trainer = self.trainer()
        objects = (trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.train_dataset, trainer.eval_dataset)
        trainer.state.global_step = 9
        self.assertIn('eval_loss', trainer.evaluate())
        trainer.state.global_step = 10
        self.assertIn('smoke_eval_loss', trainer.evaluate())
        trainer.state.global_step = 11
        self.assertIn('eval_loss', trainer.evaluate())
        self.assertEqual([len(rows) for _, rows, _ in trainer.evaluations], [100, 32, 100])
        self.assertEqual(len(self.reports), 1)
        self.assertEqual(trainer.smoke_check_result['step'], 10)
        self.assertEqual(objects, (trainer.model, trainer.optimizer, trainer.lr_scheduler,
                                   trainer.train_dataset, trainer.eval_dataset))
        self.assertEqual(trainer.state.global_step, 11)

    def test_nonfinite_check_fails_without_marking_passed(self):
        for loss in (float('nan'), float('inf'), None):
            with self.subTest(loss=loss):
                trainer = self.trainer()
                trainer.state.global_step = 10
                trainer.next_loss = loss
                with self.assertRaisesRegex(RuntimeError, 'smoke check failed'):
                    trainer.evaluate()
                self.assertIsNone(trainer.smoke_check_result)
                self.assertEqual(self.reports, [])

    def test_short_validation_and_short_training_run(self):
        trainer = self.trainer(size=3, check_step=2)
        trainer.state.global_step = 2
        trainer.evaluate()
        self.assertEqual(trainer.smoke_check_result['validation_conversations'], 3)
        self.assertEqual(trainer.smoke_check_result['step'], 2)

    def test_final_evaluation_uses_full_validation(self):
        trainer = self.trainer()
        trainer.state.global_step = 10
        trainer.is_in_train = False
        self.assertIn('eval_loss', trainer.evaluate())
        self.assertEqual(len(trainer.evaluations[0][1]), 100)
        self.assertIsNone(trainer.smoke_check_result)

    def test_normal_mode_does_not_intercept_evaluation(self):
        trainer = self.trainer(check_step=None)
        trainer.state.global_step = 10
        self.assertIn('eval_loss', trainer.evaluate())
        self.assertEqual(len(trainer.evaluations[0][1]), 100)

    def test_resumed_run_checks_once_after_resume(self):
        trainer = self.trainer()
        trainer.state.global_step = 101
        trainer.evaluate()
        trainer.state.global_step = 102
        trainer.evaluate()
        self.assertEqual([prefix for _, _, prefix in trainer.evaluations], ['smoke_eval', 'eval'])

    def test_new_mode_retains_full_training_length_and_signature(self):
        base = module.parse_args(['--gpus', '2'])
        combined = module.parse_args(['--gpus', '2', '--smoke-then-full'])
        smoke = module.parse_args(['--gpus', '2', '--smoke'])
        self.assertEqual(combined.max_steps, base.max_steps)
        self.assertEqual(combined.epochs, base.epochs)
        self.assertEqual(combined.learning_rate, base.learning_rate)
        self.assertEqual(smoke.max_steps, 10)
        normal_signature = module.run_signature(base, {'identity': {}})
        combined_signature = module.run_signature(combined, {'identity': {}})
        self.assertNotIn('smoke_then_full', normal_signature)
        self.assertTrue(combined_signature.pop('smoke_then_full'))
        self.assertEqual(normal_signature, combined_signature)

    def test_conflicting_modes_are_rejected(self):
        for flag in ('--smoke', '--benchmark'):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(['--gpus', '2', '--smoke-then-full', flag])

    def test_combined_mode_can_resume_its_own_checkpoints(self):
        args = module.parse_args(['--gpus', '2', '--smoke-then-full', '--resume', '/tmp/run/checkpoint-100'])
        self.assertEqual(args.out, Path('/tmp/run'))
        self.assertEqual(args.max_steps, -1)


if __name__ == '__main__':
    unittest.main()
