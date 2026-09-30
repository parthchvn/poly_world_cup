import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures

builder = fixtures.builder

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_actor_experiment_tested', ROOT / 'tools/prepare_actor_experiment.py')
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class ActorExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.root = self.root / 'experiment' / 'exports'
        self.fixture.root.mkdir(parents=True)
        self.plan = {split: [{'market_id': str(i), 'fixture_id': f'espn:{i}',
                             'kickoff_utc': f'2026-06-{i:02d}T17:00:00Z'}]
                     for i, split in enumerate(experiment.SPLITS, 1)}

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **changes):
        args = experiment.parse_args(['--out', str(self.root / 'experiment'), '--targets', '5',
                                      '--validation-targets', '5', '--test-targets', '5'])
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def source(self, i, actors=3, compressed=False):
        source, actor_file, rows = self.fixture.source(i, i, compressed=compressed,
                                                      with_market_context=True, context_version=2)
        if compressed:
            import gzip
            serialize = lambda value: gzip.compress(''.join(json.dumps(r) + '\n' for r in value).encode())
        else:
            serialize = lambda value: ''.join(json.dumps(r) + '\n' for r in value).encode()
        for number in range(1, actors):
            actor = '0x' + f'{number:040x}'
            cloned = copy.deepcopy(rows)
            for row in cloned:
                row['actor_id'] = actor
            (source / 'actors' / (actor + ('.jsonl.gz' if compressed else '.jsonl'))).write_bytes(serialize(cloned))
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest['counts'] = {name: value * actors for name, value in manifest['counts'].items()}
        manifest['actor_snapshots'] = None
        manifest['actor_snapshots_skipped'] = True
        (source / 'manifest.json').write_text(json.dumps(manifest))
        return source

    def run_prepare(self, args=None):
        with patch('sys.stdout', new_callable=io.StringIO):
            return experiment.prepare(args or self.args(), plan=self.plan)

    def test_registered_plan_has_distinct_real_draw_markets(self):
        plan = experiment.candidate_plan()
        self.assertEqual([len(plan[s]) for s in experiment.SPLITS], [24, 12, 12])
        ids = [v['market_id'] for s in experiment.SPLITS for v in plan[s]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(plan['train'][0]['market_id'], '1897035')
        self.assertLess(plan['train'][-1]['kickoff_utc'], plan['validation'][0]['kickoff_utc'])
        self.assertLess(plan['validation'][-1]['kickoff_utc'], plan['test'][0]['kickoff_utc'])

    def test_full_pipeline_counts_targets_keeps_conversations_and_raw_bytes(self):
        sources = [self.source(i, compressed=i == 2) for i in range(1, 4)]
        before = {str(p): p.read_bytes() for s in sources for p in (s / 'actors').iterdir()}
        result = self.run_prepare()
        self.assertEqual(result['counts']['train']['targets'], 8)
        self.assertEqual(result['counts']['train']['conversations'], 2)
        selected = Path(result['exports_path'])
        for i in range(1, 4):
            source = selected / f'market_{i}'
            manifest = json.loads((source / 'manifest.json').read_text())
            self.assertEqual(manifest['counts']['actors'], 2)
            self.assertEqual(manifest['counts']['rows'], 8)
            self.assertEqual(manifest['experiment_selection']['original_full_export_counts']['actors'], 3)
            self.assertEqual(manifest['max_trades_per_actor'], 20)
            for p in (source / 'actors').iterdir():
                original = sources[i - 1] / 'actors' / p.name
                self.assertEqual(p.read_bytes(), before[str(original)])
        for path, content in before.items():
            self.assertEqual(Path(path).read_bytes(), content)
        basic = Path(result['basic_path'])
        manifest = json.loads((basic / 'manifest.json').read_text())
        for source in manifest['sources']:
            self.assertTrue(Path(source['path']).is_dir())
            self.assertNotIn('.prepared-', source['path'])
        for line in (basic / 'source_audit.jsonl').read_text().splitlines():
            self.assertTrue(Path(json.loads(line)['source_export']).is_dir())
        self.assertFalse(manifest['token_lengths_checked'])
        self.assertEqual(self.run_prepare(), result)

    def test_chronology_rejects_touching_or_overlapping_whole_conversation(self):
        items = [{'sequence_id': str(i), 'target_count': 2, 'first_query_time': time}
                 for i, time in enumerate(['2026-06-01T00:00:00Z', '2026-06-01T01:00:00Z', '2026-06-01T01:00:01Z'])]
        chosen, rejected = experiment.choose_conversations(items, remaining=99, seed=42,
            after=builder.sft_instant('2026-06-01T01:00:00Z'))
        self.assertEqual([i['sequence_id'] for i in chosen], ['2'])
        self.assertEqual(rejected, 2)

    def test_selection_depends_on_identity_seed_and_time_not_profit_or_metadata(self):
        items = [{'sequence_id': str(i), 'target_count': 2, 'first_query_time': '2026-06-02T00:00:00Z',
                  'future_pnl': i, 'metadata': 'initial'} for i in range(10)]
        chosen, _ = experiment.choose_conversations(items, remaining=3, seed=42)
        changed = list(reversed(copy.deepcopy(items)))
        for item in changed:
            item['future_pnl'] = -9999
            item['metadata'] = 'future updates'
        revised, _ = experiment.choose_conversations(changed, remaining=3, seed=42)
        self.assertEqual([i['sequence_id'] for i in chosen], [i['sequence_id'] for i in revised])
        self.assertEqual(sum(i['target_count'] for i in chosen), 4)

    def test_insufficient_candidates_keep_captures_without_publishing(self):
        source = self.source(1, actors=1)
        with self.assertRaisesRegex(ValueError, 'only 4 eligible targets'):
            self.run_prepare()
        self.assertTrue(source.is_dir())
        self.assertFalse((self.root / 'experiment/prepared').exists())

    def test_corrupt_capture_counts_fail_before_any_bundle_is_published(self):
        source = self.source(1)
        path = source / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['counts']['actors'] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'counts disagree'):
            self.run_prepare()
        self.assertFalse((self.root / 'experiment/prepared').exists())

    def test_actual_overlapping_holdout_is_rejected_without_publishing(self):
        source = self.source(1)
        self.source(2)
        # Move all training observations and price/news context one day later.
        # Different fixture IDs alone must not be accepted as time separation.
        for actor in (source / 'actors').iterdir():
            actor.write_text(actor.read_text().replace('2026-06-01T', '2026-06-02T'))
        manifest_path = source / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['origin_utc'] = manifest['origin_utc'].replace('2026-06-01T', '2026-06-02T')
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'validation: only 0 eligible targets'):
            self.run_prepare()
        self.assertFalse((self.root / 'experiment/prepared').exists())

    def test_reusing_bundle_with_different_budget_fails(self):
        for i in range(1, 4):
            self.source(i)
        self.run_prepare()
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            self.run_prepare(self.args(targets=9))

    def test_frozen_actor_tampering_is_detected(self):
        for i in range(1, 4):
            self.source(i)
        result = self.run_prepare()
        actor = next((Path(result['exports_path']) / 'market_1/actors').iterdir())
        actor.write_bytes(actor.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Frozen selected export changed'):
            self.run_prepare()

    def test_complete_cached_exports_need_no_api_even_when_unselected_files_exist(self):
        for i in range(1, 4):
            self.source(i)
        self.source(4)
        result = self.run_prepare()
        self.assertEqual(sorted(p.name for p in Path(result['exports_path']).iterdir()),
                         ['market_1', 'market_2', 'market_3'])

    def test_snapshots_in_cached_source_rejected_without_mutating_rows(self):
        source, actor, _ = self.fixture.source(1, 1, with_market_context=True, context_version=2,
                                             with_actor_snapshots=True)
        original = actor.read_bytes()
        with self.assertRaisesRegex(ValueError, 'contains collection-time snapshots'):
            self.run_prepare()
        self.assertEqual(actor.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
