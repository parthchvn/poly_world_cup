"""RunPod orchestration checks without GPUs, networking, or model weights."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('runpod_actor_launcher_tested', ROOT / 'tools/runpod_actor_experiment.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class RunPodActorLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.model = self.base / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"model_type":"qwen3_5"}\n')
        self.args = launcher.parse_args([
            '--variant', 'basic', '--root', str(self.base / 'experiment'),
            '--model', str(self.model), '--cache', str(self.base / 'capture_cache'),
            '--targets', '40', '--validation-targets', '20', '--test-targets', '20', '--collect'])
        self.args.root.mkdir()
        # Keep this suite independent of the preparation helper's construction.
        self.code_patch = mock.patch.object(launcher, 'CODE_FILES', ('tools/runpod_actor_experiment.py',))
        self.code_patch.start()
        self.addCleanup(self.code_patch.stop)

    def ready_fixture(self):
        basic = self.args.root / 'common' / 'basic'
        exports = self.args.root / 'common' / 'exports'
        basic.mkdir(parents=True)
        exports.mkdir(parents=True)
        manifest = {'split_sha256': {'train': 'train_hash', 'validation': 'validation_hash', 'test': 'test_hash'},
                    'stats': {'train': {'targets': 40}, 'validation': {'targets': 20}, 'test': {'targets': 20}}}
        launcher.atomic_json(basic / 'manifest.json', manifest)
        data = {'manifest': manifest, 'splits': copy.deepcopy(manifest['stats'])}
        ready = {'status': 'ready', 'config': launcher.experiment_config(self.args),
                 'basic_path': str(basic), 'exports_path': str(exports),
                 'manifest_sha256': launcher.metrics.sha256(basic / 'manifest.json'),
                 'split_sha256': manifest['split_sha256'], 'actor_inventory_sha256': 'actor_hash',
                 'counts': manifest['stats']}
        return ready, data

    def ready_patches(self, data):
        return (mock.patch.object(launcher, 'scan_dataset', return_value=data),
                mock.patch.object(launcher, 'source_inventory', return_value='actor_hash'),
                mock.patch.object(launcher, 'validate_chronological_splits', return_value={}))

    def test_paths_share_only_common_cohort_and_isolate_training_writes(self):
        variants = []
        for name in ('basic', 'inmarket', 'global'):
            args = copy.copy(self.args)
            args.variant = name
            variants.append(launcher.paths(args))
        for key in ('common', 'ready', 'failed'):
            self.assertEqual(len({paths[key] for paths in variants}), 1)
        for key in ('features', 'run', 'token_cache', 'receipt'):
            self.assertEqual(len({paths[key] for paths in variants}), 3)

    def test_default_launcher_requires_dataset_before_loading_gpu_or_fetching(self):
        self.args.collect = False
        with mock.patch.object(launcher, 'preflight') as gpu, mock.patch.object(launcher, 'wait_ready') as wait:
            with self.assertRaisesRegex(ValueError, 'No dataset supplied'):
                launcher.run(self.args)
        gpu.assert_not_called()
        wait.assert_not_called()

    def test_prepared_training_does_not_collect_or_wait(self):
        ready, data = self.ready_fixture()
        self.args.dataset_dir = Path(ready['basic_path'])
        self.args.collect = False
        with mock.patch.object(launcher, 'scan_dataset', return_value=data), \
             mock.patch.object(launcher, 'validate_chronological_splits', return_value={}), \
             mock.patch.object(launcher, 'preflight'), \
             mock.patch.object(launcher, 'wait_ready', side_effect=AssertionError('must not wait')), \
             mock.patch.object(launcher, 'enrich', side_effect=AssertionError('must not collect')), \
             mock.patch.object(launcher.subprocess, 'run') as run:
            launcher.run(self.args)
        command = run.call_args.args[0]
        self.assertIn('train_world_cup_multigpu.py', command[1])
        self.assertEqual(command[command.index('--dataset-dir') + 1], ready['basic_path'])

    def test_prepared_wrong_variant_fails_before_gpu_load(self):
        ready, data = self.ready_fixture()
        self.args.dataset_dir = Path(ready['basic_path'])
        self.args.collect = False
        self.args.variant = 'global'
        with mock.patch.object(launcher, 'scan_dataset', return_value=data), \
             mock.patch.object(launcher, 'validate_chronological_splits', return_value={}), \
             mock.patch.object(launcher, 'preflight') as gpu:
            with self.assertRaisesRegex(ValueError, 'Dataset variant'):
                launcher.run(self.args)
        gpu.assert_not_called()

    def test_configuration_is_variant_independent_but_tracks_seed_model_and_code(self):
        initial = launcher.experiment_config(self.args)
        self.args.variant = 'global'
        self.assertEqual(initial, launcher.experiment_config(self.args))
        self.args.seed += 1
        self.assertNotEqual(initial, launcher.experiment_config(self.args))
        self.args.seed -= 1
        (self.model / 'config.json').write_text('{"model_type":"changed"}\n')
        self.assertNotEqual(initial, launcher.experiment_config(self.args))

    def test_atomic_ready_has_no_partial_file(self):
        path = launcher.paths(self.args)['ready']
        launcher.atomic_json(path, {'status': 'ready'})
        self.assertEqual(json.loads(path.read_text()), {'status': 'ready'})
        self.assertEqual(list(path.parent.glob(path.name + '.*.tmp')), [])

    def test_ready_validates_config_source_and_chronology(self):
        ready, data = self.ready_fixture()
        a, b, c = self.ready_patches(data)
        with a as scan, b as inventory, c as chronology:
            self.assertEqual(launcher.validate_ready(self.args, ready), ready)
        scan.assert_called_once_with(Path(ready['basic_path']))
        inventory.assert_called_once_with(Path(ready['exports_path']))
        chronology.assert_called_once()

    def test_ready_rejects_other_seed_before_reading_dataset(self):
        ready, _ = self.ready_fixture()
        ready['config']['seed'] += 1
        with mock.patch.object(launcher, 'scan_dataset') as scan:
            with self.assertRaisesRegex(ValueError, 'configuration/code/model/packages differ'):
                launcher.validate_ready(self.args, ready)
        scan.assert_not_called()

    def test_ready_rejects_changed_basic_manifest(self):
        ready, data = self.ready_fixture()
        (Path(ready['basic_path']) / 'manifest.json').write_text('{}\n')
        a, b, c = self.ready_patches(data)
        with a, b, c:
            with self.assertRaisesRegex(ValueError, 'dataset changed'):
                launcher.validate_ready(self.args, ready)

    def test_ready_rejects_changed_selected_actor_rows(self):
        ready, data = self.ready_fixture()
        a, _, c = self.ready_patches(data)
        with a, c, mock.patch.object(launcher, 'source_inventory', return_value='changed_actor_hash'):
            with self.assertRaisesRegex(ValueError, 'actor exports changed'):
                launcher.validate_ready(self.args, ready)

    def test_ready_rejects_paths_outside_shared_experiment(self):
        ready, _ = self.ready_fixture()
        ready['basic_path'] = str(self.base / 'other_experiment')
        with self.assertRaisesRegex(ValueError, 'escape the experiment root'):
            launcher.validate_ready(self.args, ready)

    def test_wait_propagates_base_failure_without_sleeping(self):
        launcher.atomic_json(launcher.paths(self.args)['failed'], {'error': 'API unavailable'})
        with mock.patch.object(launcher.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'Base preparation failed: API unavailable'):
                launcher.wait_ready(self.args)
        sleep.assert_not_called()

    def test_wait_timeout_does_not_loop_forever(self):
        with mock.patch.object(launcher.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'Timed out waiting'):
                launcher.wait_ready(self.args, timeout=0)
        sleep.assert_not_called()

    def test_wait_validates_existing_ready_marker(self):
        ready, data = self.ready_fixture()
        launcher.atomic_json(launcher.paths(self.args)['ready'], ready)
        a, b, c = self.ready_patches(data)
        with a, b, c, mock.patch.object(launcher.time, 'sleep') as sleep:
            self.assertEqual(launcher.wait_ready(self.args), ready)
        sleep.assert_not_called()

    def test_publish_enforces_target_counts_and_writes_ready_last(self):
        ready, data = self.ready_fixture()
        a, b, c = self.ready_patches(data)
        with a, b, c:
            report = launcher.publish_ready(self.args, ready)
        self.assertEqual(report['status'], 'ready')
        self.assertEqual(json.loads(launcher.paths(self.args)['ready'].read_text()), report)
        launcher.paths(self.args)['ready'].unlink()
        data['splits']['train']['targets'] = 39
        a, b, c = self.ready_patches(data)
        with a, b, c:
            with self.assertRaisesRegex(ValueError, 'complete-conversation targets'):
                launcher.publish_ready(self.args, ready)
        self.assertFalse(launcher.paths(self.args)['ready'].exists())

    def test_three_training_commands_hold_hyperparameters_fixed(self):
        commands = []
        for name in ('basic', 'inmarket', 'global'):
            args = copy.copy(self.args)
            args.variant = name
            command = launcher.training_command(args, self.args.root / name / 'sft')
            self.assertIn('--smoke-then-full', command)
            self.assertIn('--no-group-by-length', command)
            self.assertNotIn('--init-adapter', command)
            self.assertEqual(command[command.index('--gpus') + 1], '2')
            self.assertEqual(command[command.index('--global-batch') + 1], '8')
            for path_flag in ('--dataset-dir', '--out', '--cache-dir'):
                command[command.index(path_flag) + 1] = '<variant_path>'
            commands.append(command)
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(commands[1], commands[2])

    def checkpoint(self, run, step, *, gpus=2, omit=None):
        checkpoint = run / f'checkpoint-{step}'
        checkpoint.mkdir(parents=True)
        for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json',
                     'adapter_model.safetensors', 'rng_state_0.pth', 'rng_state_1.pth'):
            if name != omit:
                (checkpoint / name).write_text('{}')
        launcher.atomic_json(checkpoint / 'complete.json', {'gpus': gpus, 'step': step})
        return checkpoint

    def test_resume_selects_latest_complete_two_gpu_checkpoint(self):
        run = launcher.paths(self.args)['run']
        self.checkpoint(run, 9)
        expected = self.checkpoint(run, 100)
        self.checkpoint(run, 200, omit='rng_state_1.pth')
        self.checkpoint(run, 300, gpus=1)
        self.assertEqual(launcher.choose_resume(run), expected)
        self.args.resume = True
        command = launcher.training_command(self.args, self.args.root / 'sft')
        self.assertEqual(command[command.index('--resume') + 1], str(expected))
        self.assertIn('--smoke-then-full', command)

    def test_resume_rejects_adapter_only_directory(self):
        run = launcher.paths(self.args)['run']
        (run / 'adapter').mkdir(parents=True)
        (run / 'adapter' / 'adapter_model.safetensors').write_text('weights')
        with self.assertRaisesRegex(ValueError, 'No complete two-GPU checkpoint'):
            launcher.choose_resume(run)

    def test_variant_lock_prevents_duplicate_writers_and_allows_other_variants(self):
        other = copy.copy(self.args)
        other.variant = 'global'
        with launcher.variant_lock(self.args):
            with self.assertRaisesRegex(ValueError, 'Another basic launcher'):
                with launcher.variant_lock(self.args):
                    self.fail('Second basic writer acquired the lock')
            with launcher.variant_lock(other):
                pass

    def test_preflight_failure_signals_waiting_pods_before_any_collection(self):
        with mock.patch.object(launcher, 'preflight', side_effect=ValueError('missing training packages')):
            with self.assertRaisesRegex(ValueError, 'missing training packages'):
                launcher.run(self.args)
        failure = json.loads(launcher.paths(self.args)['failed'].read_text())
        self.assertEqual(failure['error'], 'missing training packages')
        self.assertFalse(launcher.paths(self.args)['ready'].exists())

    def test_basic_enrichment_does_not_launch_a_subprocess(self):
        ready, _ = self.ready_fixture()
        with mock.patch.object(launcher.subprocess, 'run') as run:
            self.assertEqual(launcher.enrich(self.args, ready), Path(ready['basic_path']))
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
