"""CPU integration tests: real data validation/reporting, simulated GPU subprocesses."""
import copy
from contextlib import ExitStack
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
import run_world_cup_experiments as pipeline
import world_cup_eval_common as common
from tests.test_actor_variant_comparison import ActorVariantComparisonTests


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ActorVariantComparisonTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'FAKE TEST WEIGHTS')
        self.calls = []
        self.barrier = None
        self.fail_train_once = False
        self.fail_eval_once = False
        self.fail_prepare = False
        self.versions = {key: 'fixture-version' for key in pipeline.PACKAGES}
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(pipeline, 'package_versions', return_value=self.versions))
        self.stack.enter_context(patch.object(pipeline, 'gpu_info', side_effect=lambda group:
            [(f'TEST-{self.root.name}-{gpu}', 80 * 1024) for gpu in group.split(',')]))
        self.stack.enter_context(patch.object(pipeline.Controller, 'command',
            lambda controller, cmd, name, group=None: self.command(controller.args, cmd, name, group)))

    def tearDown(self):
        self.stack.close()
        self.fixture.tearDown()

    def args(self, *extra):
        return pipeline.parse_args(['--data', str(self.fixture.datasets['basic']),
            '--data', str(self.fixture.datasets['inmarket']), '--model', str(self.model),
            '--out', str(self.root / 'experiment'), *extra])

    def checkpoint(self, run, step, gpus=2):
        checkpoint = run / f'checkpoint-{step}'
        checkpoint.mkdir(parents=True, exist_ok=True)
        names = ['optimizer.pt', 'scheduler.pt', 'adapter_config.json', 'adapter_model.safetensors']
        names += ['rng_state.pth'] if gpus == 1 else [f'rng_state_{i}.pth' for i in range(gpus)]
        for name in names:
            (checkpoint / name).write_text('fixture')
        pipeline.atomic_json(checkpoint / 'complete.json', {'step': step, 'gpus': gpus})
        pipeline.atomic_json(checkpoint / 'trainer_state.json', {'global_step': step})
        return checkpoint

    def trained(self, args, variant, run):
        meta, _ = common.read_bundle(args.out / 'bundle')
        ref = meta['reference'][variant]
        (run / 'adapter').mkdir(parents=True, exist_ok=True)
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'):
            (run / 'adapter' / name).write_text('{}')
        signature = {'max_length': args.max_length, 'epochs': args.epochs,
            'data': {'model_config_sha256': common.sha(args.model / 'config.json'),
                     'sources': {s: {'sha256': ref['source_sha256'][s]} for s in ('train', 'validation')}}}
        pipeline.atomic_json(run / 'training_metadata.json', {
            'status': 'completed', 'mode': 'smoke_then_full', 'test_used': False,
            'signature': signature,
            'data': {s: {'sha256': ref['split_sha256'][s], 'fixtures': ref['fixtures'][s]}
                     for s in ('train', 'validation')}})
        (run / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in (
            {'step': 0, 'event': 'train_begin'}, {'step': 1, 'loss': 0.8},
            {'step': 10, 'loss': 0.6, 'smoke_eval_loss': 0.7},
            {'step': 100, 'loss': 0.4, 'eval_loss': 0.5},
            {'step': 200, 'loss': 0.2, 'eval_loss': 0.35},
            {'step': 200, 'train_loss': 0.5})))

    def evaluated(self, args, variant, run):
        meta, records = common.read_bundle(args.out / 'bundle')
        selected = [t for r in records[variant] for t in common.targets(r)]
        out = args.out / 'evaluation' / variant
        out.mkdir(parents=True, exist_ok=True)
        identity = {'variant': variant, 'bundle_sha256': common.sha(args.out / 'bundle/manifest.json'),
            'adapter_sha256': common.sha(run / 'adapter/adapter_model.safetensors'),
            'training_metadata_sha256': common.sha(run / 'training_metadata.json'),
            'training_signature': pipeline.read_json(run / 'training_metadata.json')['signature'],
            'limit': 0, 'selected_targets': len(selected), 'target_sha256': meta['target_sha256'],
            'history_protocol': meta['history_protocol'],
            'selected_ids_sha256': hashlib.sha256(common.dump([t['id'] for t in selected]).encode()).hexdigest(),
            'decoding': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': args.max_new_tokens,
                         'max_context': args.max_context, 'seed': 42, 'attention': 'sdpa'},
            'versions': self.versions, 'tokenizer_sha256': 'fixture-tokenizer',
            'evaluator_sha256': common.sha(ROOT / 'scripts/evaluate_world_cup.py'),
            'common_sha256': common.sha(ROOT / 'tools/world_cup_eval_common.py')}
        common.write_json(out / 'identity.json', identity)
        rows = [{**{k: v for k, v in t.items() if k != 'messages'},
                 'prediction': t['answer'], 'hit_generation_limit': False} for t in selected]
        (out / 'predictions.jsonl').write_text(''.join(common.dump(r) + '\n' for r in rows))
        common.write_json(out / 'summary.json', {'status': 'completed', 'variant': variant,
            'pilot': False, 'predictions_sha256': common.sha(out / 'predictions.jsonl'),
            'identity_sha256': common.sha(out / 'identity.json'), 'metrics': common.summarize(rows)})

    def command(self, args, cmd, name, group=None):
        self.calls.append((name, cmd, group))
        if name == 'prepare_inmarket' and self.fail_prepare:
            raise RuntimeError('8493 tokens exceeds limit')
        if name.startswith('train_'):
            variant = name.removeprefix('train_')
            run = Path(cmd[cmd.index('--out') + 1])
            if self.barrier:
                self.barrier.wait(timeout=10)
            if variant == 'basic' and self.fail_train_once:
                self.fail_train_once = False
                self.checkpoint(run, 100)
                # Deliberately newer but incomplete save must never be selected.
                (run / 'checkpoint-200').mkdir()
                pipeline.atomic_json(run / 'training_metadata.json', {'status': 'interrupted'})
                raise RuntimeError('Simulated pod interruption')
            self.trained(args, variant, run)
        if name.startswith('evaluate_'):
            variant = name.removeprefix('evaluate_')
            if variant == 'basic' and self.fail_eval_once:
                self.fail_eval_once = False
                raise RuntimeError('Simulated evaluation interruption')
            run = Path(cmd[cmd.index('--run-dir') + 1])
            self.evaluated(args, variant, run)

    def test_one_group_trains_then_evaluates_without_overlap(self):
        args = self.args()
        self.assertEqual(pipeline.run(args), 0)
        stages = [n for n, _, _ in self.calls if n.startswith(('train_', 'evaluate_'))]
        self.assertEqual(stages, ['train_basic', 'evaluate_basic', 'train_inmarket', 'evaluate_inmarket'])
        train = next(c for n, c, _ in self.calls if n == 'train_basic')
        self.assertIn('--smoke-then-full', train)
        self.assertEqual(train[train.index('--max-length') + 1], '16384')
        evaluations = [(c, g) for n, c, g in self.calls if n.startswith('evaluate_')]
        self.assertTrue(all(g == '0' and '--resume' in c for c, g in evaluations))
        self.assertTrue((args.out / 'REPORT.md').is_file())
        self.assertTrue((args.out / 'plots/loss_comparison.png').is_file())
        self.assertEqual(pipeline.read_json(args.out / 'completed.json')['status'], 'completed')

    def test_completed_run_reuses_predictions_and_does_not_retrain(self):
        args = self.args()
        pipeline.run(args)
        self.calls.clear()
        self.assertEqual(pipeline.run(args), 0)
        self.assertFalse(any(n.startswith(('train_', 'evaluate_')) for n, _, _ in self.calls))

    def test_resume_uses_latest_complete_checkpoint_and_keeps_other_experiment(self):
        args = self.args()
        self.fail_train_once = True
        self.assertEqual(pipeline.run(args), 1)
        self.assertFalse((args.out / 'completed.json').exists())
        self.calls.clear()
        self.assertEqual(pipeline.run(args), 0)
        command = next(c for n, c, _ in self.calls if n == 'train_basic')
        self.assertTrue(command[command.index('--resume') + 1].endswith('checkpoint-100'))
        self.assertFalse(any(n == 'train_inmarket' for n, _, _ in self.calls))

    def test_interrupted_evaluation_resumes_without_loading_training(self):
        args = self.args()
        self.fail_eval_once = True
        self.assertEqual(pipeline.run(args), 1)
        self.calls.clear()
        self.assertEqual(pipeline.run(args), 0)
        self.assertFalse(any(n.startswith('train_') for n, _, _ in self.calls))
        command = next(c for n, c, _ in self.calls if n == 'evaluate_basic')
        self.assertIn('--resume', command)

    def test_disjoint_groups_run_training_concurrently(self):
        args = self.args('--gpu-group', '0,1', '--gpu-group', '2,3')
        self.barrier = threading.Barrier(2)
        self.assertEqual(pipeline.run(args), 0)
        groups = {g for n, _, g in self.calls if n.startswith('train_')}
        self.assertEqual(groups, {'0,1', '2,3'})
        eval_groups = {g for n, _, g in self.calls if n.startswith('evaluate_')}
        self.assertEqual(eval_groups, {'0', '2'})

    def test_split_length_failure_stops_before_any_training(self):
        self.fail_prepare = True
        args = self.args()
        with self.assertRaisesRegex(RuntimeError, 'exceeds'):
            pipeline.run(args)
        self.assertFalse(any(n.startswith(('train_', 'evaluate_')) for n, _, _ in self.calls))
        self.assertFalse((args.out / 'preparation_complete.json').exists())

    def test_changed_configuration_is_rejected_before_gpu_work(self):
        args = self.args('--check-only')
        pipeline.run(args)
        self.calls.clear()
        args.learning_rate *= 2
        with self.assertRaisesRegex(ValueError, 'changed'):
            pipeline.run(args)
        self.assertEqual(self.calls, [])

    def test_changed_staged_data_is_rejected(self):
        args = self.args('--check-only')
        pipeline.run(args)
        (args.out / 'data/basic/train.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            pipeline.run(args)

    def test_data_is_staged_persistently_and_can_resume_without_upload(self):
        args = self.args('--check-only')
        pipeline.run(args)
        args.data = []
        self.assertEqual(pipeline.run(args), 0)

    def test_basic_only_derives_causal_inmarket_features_and_preserves_targets(self):
        args = self.args('--check-only')
        args.data = [self.fixture.datasets['basic']]
        self.assertEqual(pipeline.run(args), 0)
        _, records = common.read_bundle(args.out / 'bundle')
        basic, enriched = records['basic'][0], records['inmarket'][0]
        common.check_pair(basic, enriched)
        first = json.loads(enriched['messages'][1]['content'])['actor_metrics']
        second = json.loads(enriched['messages'][3]['content'])['actor_metrics']
        self.assertEqual(first['sample_counts']['captured_executions'], 0)
        self.assertEqual(first['values'], {})
        self.assertEqual(second['sample_counts']['captured_executions'], 1)
        self.assertEqual(float(second['values']['average_execution_notional']), 0.8)
        self.assertEqual(float(second['values']['buy_notional_share']), 1)

    def test_automatic_features_exclude_all_current_timestamp_fills(self):
        self.fixture.records['test'][0]['messages'][2]['content'] = json.dumps({'action':'TRADE','trades':[
            {'side':'BUY','outcome':'Yes','shares':'2','price':'0.4'},
            {'side':'SELL','outcome':'No','shares':'10','price':'0.6'}]})
        self.fixture.records['test'][0]['execution_count'] = 3
        self.fixture.write_all()
        args = self.args('--check-only')
        args.data = [self.fixture.datasets['basic']]
        pipeline.run(args)
        _, records = common.read_bundle(args.out / 'bundle')
        enriched = records['inmarket'][0]
        first = json.loads(enriched['messages'][1]['content'])['actor_metrics']
        second = json.loads(enriched['messages'][3]['content'])['actor_metrics']
        self.assertEqual(first['sample_counts']['captured_executions'], 0)
        self.assertEqual(second['sample_counts']['captured_executions'], 2)
        self.assertAlmostEqual(float(second['values']['average_execution_notional']), 3.4)

    def test_basic_only_requires_untruncated_history(self):
        path = self.fixture.datasets['basic'] / 'manifest.json'
        manifest = pipeline.read_json(path)
        manifest['targets_truncated_or_dropped'] = 1
        pipeline.atomic_json(path, manifest)
        args = self.args('--check-only')
        args.data = [self.fixture.datasets['basic']]
        with self.assertRaisesRegex(ValueError, 'complete Basic conversations'):
            pipeline.run(args)

    def test_completed_results_need_no_gpu(self):
        args = self.args()
        pipeline.run(args)
        with patch.object(pipeline, 'gpu_info', side_effect=AssertionError('GPU should not be used')):
            self.assertEqual(pipeline.run(args), 0)

    def test_prepare_rejects_changed_target_between_variants(self):
        source = self.fixture.datasets['inmarket'] / 'test.jsonl'
        source.write_text(source.read_text().replace('BUY', 'SELL'))
        manifest_path = source.parent / 'manifest.json'
        manifest = pipeline.read_json(manifest_path)
        manifest['split_sha256']['test'] = common.sha(source)
        pipeline.atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'target differs'):
            pipeline.run(self.args('--check-only'))
        self.assertFalse(any(n.startswith('train_') for n, _, _ in self.calls))

    def test_reused_completed_models_skip_training(self):
        args = self.args('--check-only')
        pipeline.run(args)
        external = {v: self.root / ('old-' + v) for v in pipeline.VARIANTS}
        for variant, run in external.items():
            self.trained(args, variant, run)
        args = self.args('--out', str(self.root / 'reuse'), '--basic-run', str(external['basic']),
                         '--inmarket-run', str(external['inmarket']))
        self.calls.clear()
        self.assertEqual(pipeline.run(args), 0)
        self.assertFalse(any(n.startswith(('train_', 'prepare_')) for n, _, _ in self.calls))

    def test_no_checkpoint_preserves_unfinished_attempt_before_restart(self):
        args = self.args('--check-only')
        pipeline.run(args)
        run = args.out / 'runs/basic'
        run.mkdir(parents=True)
        (run / 'interrupted.log').write_text('preserve me')
        args.check_only = False
        pipeline.run(args)
        backups = list((args.out / 'unfinished_attempts').glob('basic-*'))
        self.assertEqual((backups[0] / 'interrupted.log').read_text(), 'preserve me')

    def test_second_pod_finishes_shared_report(self):
        args = self.args('--variant', 'basic')
        self.assertEqual(pipeline.run(args), 0)
        self.assertFalse((args.out / 'comparison.json').exists())
        args.variant = 'inmarket'
        self.assertEqual(pipeline.run(args), 0)
        self.assertTrue((args.out / 'comparison.json').exists())

    def test_gpu_gate_fails_without_killing_processes(self):
        with patch.object(pipeline, 'gpu_info', return_value=[('gpu', 1000)]), \
             patch.object(pipeline.os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, 'insufficient free VRAM'):
                pipeline.wait_for_memory('0', 60, 0, threading.Event())
            kill.assert_not_called()

    def test_checkpoint_requires_rng_files_and_consistent_step(self):
        run = self.root / 'checkpoints'
        old = self.checkpoint(run, 100)
        new = self.checkpoint(run, 200)
        (new / 'rng_state_1.pth').unlink()
        self.assertEqual(pipeline.complete_checkpoint(run, 2), old)
        pipeline.atomic_json(old / 'trainer_state.json', {'global_step': 99})
        self.assertIsNone(pipeline.complete_checkpoint(run, 2))

    def test_gpu_group_overlap_and_batch_divisibility_are_rejected(self):
        for flags in (('--gpu-group', '0,1', '--gpu-group', '1,2'),
                      ('--gpu-group', '0,00'), ('--gpu-group', '0,1,2')):
            with self.assertRaises(ValueError):
                self.args(*flags)

    def test_lock_is_exclusive(self):
        lock = self.root / 'exclusive.lock'
        with pipeline.file_lock(lock):
            with self.assertRaisesRegex(ValueError, 'Another runner'):
                with pipeline.file_lock(lock, blocking=False):
                    self.fail('Duplicate lock acquired')

    def test_busy_variant_does_not_overwrite_active_status(self):
        args = self.args('--check-only')
        data, config = pipeline.initialize(args, pipeline.Controller(args))
        pipeline.status(args, 'basic', 'training')
        with pipeline.file_lock(args.out / 'locks/basic.lock'):
            self.assertFalse(pipeline.run_variant(args, pipeline.Controller(args), 'basic', '0,1', data, config))
        self.assertEqual(pipeline.read_json(args.out / 'status/basic.json')['stage'], 'training')

    def test_upload_archive_extracts_prepared_data(self):
        archive = self.root / 'upload.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            for variant in pipeline.VARIANTS:
                output.add(self.fixture.datasets[variant], arcname=variant)
        args = self.args('--check-only')
        args.data = [archive]
        self.assertEqual(pipeline.run(args), 0)
        self.assertTrue((args.out / 'data/inmarket/train.jsonl').is_file())

    def archive(self, name, kind=None):
        archive = self.root / 'bad.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            item = tarfile.TarInfo(name)
            if kind:
                item.type, item.linkname = kind, '/etc/passwd'
                output.addfile(item)
            else:
                item.size = 1
                output.addfile(item, io.BytesIO(b'x'))
        return archive

    def test_archive_rejects_traversal_and_links(self):
        for name, kind in (('../escape', None), ('/absolute', None), ('link', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE)):
            with self.assertRaisesRegex(ValueError, 'Unsafe|links'):
                pipeline.unpack(self.archive(name, kind), self.root / 'extract', 1024**2)
        self.assertFalse((self.root / 'escape').exists())

    def test_archive_rejects_truncation_crc_and_size_limit(self):
        archive = self.archive('file')
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            pipeline.unpack(archive, self.root / 'extract', 1)
        archive.write_bytes(archive.read_bytes()[:-4])
        with self.assertRaises((EOFError, OSError)):
            pipeline.unpack(archive, self.root / 'extract', 1024**2)
        self.assertFalse(list((self.root / 'extract').glob('*/.archive.json')))

    def test_existing_unowned_run_directory_not_adopted(self):
        args = self.args('--check-only')
        (args.out / 'runs').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'without an experiment identity'):
            pipeline.run(args)


class ControllerTests(unittest.TestCase):
    def test_subprocess_failure_reports_log_and_exit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            args = pipeline.parse_args(['--out', temp])
            controller = pipeline.Controller(args)
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                controller.command([sys.executable, '-c', 'print("error fixture"); raise SystemExit(7)'], 'failure')
            self.assertIn('error fixture', (args.out / 'logs/failure.log').read_text())
            self.assertFalse(controller.active)

    def test_cancellation_stops_child_process_and_keeps_log(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            args = pipeline.parse_args(['--out', temp])
            controller = pipeline.Controller(args)
            errors = []
            def child():
                try:
                    controller.command([sys.executable, '-c', 'import time; time.sleep(60)'], 'sleep')
                except (InterruptedError, RuntimeError) as error:
                    errors.append(error)
            thread = threading.Thread(target=child)
            thread.start()
            deadline = time.monotonic() + 5
            while not controller.active and time.monotonic() < deadline:
                time.sleep(0.02)
            controller.cancel()
            thread.join(timeout=20)
            self.assertFalse(thread.is_alive())
            self.assertFalse(controller.active)
            self.assertTrue(errors)


if __name__ == '__main__':
    unittest.main()
