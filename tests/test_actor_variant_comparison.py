import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('actor_comparison_tested', ROOT / 'tools/compare_actor_variants.py')
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class ActorVariantComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.datasets = {name: self.root / name for name in ('basic', 'inmarket', 'global')}
        self.records = {
            split: [self.record(index, index * 4 + 1, index * 4 + 2)]
            for index, split in enumerate(compare.SPLITS)
        }
        self.manifest = {
            'format': 'actor_market_trade_messages_v1', 'feature_variant': 'basic',
            'enabled_splits': list(compare.SPLITS),
            'fixture_to_split': {f'fixture{i}': split for i, split in enumerate(compare.SPLITS)},
            'token_lengths_checked': True, 'max_length_checked': 8192,
            'targets_truncated_or_dropped': 0,
            'stats': {}, 'split_sha256': {}, 'sources': [{'path': 'original-export'}],
        }
        self.write_all()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def record(number, first, last, fixture=None):
        actor = '0x' + f'{number:040x}'
        market = str(number)
        contexts = [
            {'actor_id': actor, 'market': {'market_id': market}, 'news': [],
             'query_time': f'2026-06-{first:02d}T10:00:00Z'},
            {'news': [], 'query_time': f'2026-06-{last:02d}T10:00:00Z'},
        ]
        messages = [{'role': 'system', 'content': 'Predict the observed execution.'}]
        for context in contexts:
            messages.extend([
                {'role': 'user', 'content': json.dumps(context)},
                {'role': 'assistant', 'content': json.dumps({'action': 'TRADE', 'trades': [
                    {'side': 'BUY', 'outcome': 'Yes', 'price': '0.4', 'shares': '2'}]})},
            ])
        return {'sequence_id': f'sequence{number}', 'fixture_id': fixture or f'fixture{number}',
                'actor_id': actor, 'market_id': market, 'target_count': 2, 'execution_count': 2,
                'messages': messages}

    def write(self, variant, records=None, manifest=None):
        path = self.datasets[variant]
        path.mkdir(exist_ok=True)
        rows = copy.deepcopy(self.records if records is None else records)
        metadata = copy.deepcopy(self.manifest if manifest is None else manifest)
        metadata['feature_variant'] = variant
        if variant != 'basic':
            metadata['actor_metrics'] = {
                'config': {'history_scope': compare.SCOPES[variant]},
                'history_scope': compare.SCOPES[variant], 'strict_prior': True,
                'target_messages_unchanged': True,
            }
            for records in rows.values():
                for record in records:
                    for message in record['messages']:
                        if message['role'] == 'user':
                            context = json.loads(message['content'])
                            context['actor_metrics'] = {'values': {'sharpe_ratio': None},
                                                        'sample_counts': {'captured_executions': 0},
                                                        'scope': ('current_market' if variant == 'inmarket'
                                                                  else 'global_wallet')}
                            message['content'] = json.dumps(context)
        for split, records in rows.items():
            content = ''.join(json.dumps(record) + '\n' for record in records).encode()
            (path / f'{split}.jsonl').write_bytes(content)
            metadata['split_sha256'][split] = hashlib.sha256(content).hexdigest()
            metadata['stats'][split] = {
                'conversations': len(records), 'targets': sum(row['target_count'] for row in records),
                'executions': sum(row['execution_count'] for row in records), 'tokens': 100,
            } if records else {}
        (path / 'manifest.json').write_text(json.dumps(metadata))
        (path / 'split_plan.json').write_text(json.dumps({'fixture_to_split': metadata['fixture_to_split']}))
        (path / 'source_audit.jsonl').write_text(''.join(
            json.dumps({'sequence_id': row['sequence_id'], 'split': split}) + '\n'
            for split, records in rows.items() for row in records))
        (path / 'audit').mkdir(exist_ok=True)
        (path / 'audit/collection_snapshot.json').write_text('{"current_pnl":"10000"}')

    def write_all(self):
        for variant in self.datasets:
            self.write(variant)

    def check(self):
        return compare.compare_variants(self.datasets['basic'], self.datasets['inmarket'], self.datasets['global'])

    def edit_line(self, variant, split, edit):
        path = self.datasets[variant] / f'{split}.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        edit(records)
        content = ''.join(json.dumps(row) + '\n' for row in records).encode()
        path.write_bytes(content)
        manifest_path = path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['split_sha256'][split] = hashlib.sha256(content).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def test_variants_match_targets_and_baseline_with_character_report(self):
        report = self.check()
        self.assertEqual(report['status'], 'comparable')
        for stats in report['splits'].values():
            self.assertEqual(stats['targets'], 2)
            self.assertEqual(len(stats['target_sha256']), 64)
            self.assertGreater(stats['prompt_characters']['global']['mean'],
                               stats['prompt_characters']['basic']['mean'])

    def test_gzip_hash_uses_uncompressed_bytes(self):
        path = self.datasets['global'] / 'test.jsonl'
        path.with_name(path.name + '.gz').write_bytes(gzip.compress(path.read_bytes()))
        path.unlink()
        self.assertEqual(self.check()['status'], 'comparable')

    def test_tampered_hash_or_manifest_count_rejected(self):
        path = self.datasets['inmarket'] / 'manifest.json'
        original = json.loads(path.read_text())
        for fault in ('hash', 'count'):
            manifest = copy.deepcopy(original)
            if fault == 'hash':
                manifest['split_sha256']['train'] = '0' * 64
            else:
                manifest['stats']['train']['targets'] = 99
            path.write_text(json.dumps(manifest))
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'differs from manifest'):
                self.check()

    def test_changed_assistant_and_system_rejected_even_with_valid_hashes(self):
        for offset in (0, 2):
            self.write('global')
            def edit(rows):
                if offset == 0:
                    rows[0]['messages'][offset]['content'] += ' Global only.'
                else:
                    label = json.loads(rows[0]['messages'][offset]['content'])
                    label['trades'][0]['price'] = '0.8'
                    rows[0]['messages'][offset]['content'] = json.dumps(label)
            self.edit_line('global', 'train', edit)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'system or assistant'):
                self.check()

    def test_changed_basic_news_rejected(self):
        def edit(rows):
            context = json.loads(rows[0]['messages'][1]['content'])
            context['news'].append({'text': 'Hidden future result'})
            rows[0]['messages'][1]['content'] = json.dumps(context)
        self.edit_line('inmarket', 'validation', edit)
        with self.assertRaisesRegex(ValueError, 'basic user context differs'):
            self.check()

    def test_changed_sequence_identity_rejected(self):
        self.edit_line('global', 'train', lambda rows: rows[0].update(sequence_id='different'))
        with self.assertRaisesRegex(ValueError, 'identity, order or metadata'):
            self.check()

    def test_compact_prompt_may_omit_ids_while_preserving_outer_identity(self):
        for rows in self.records.values():
            context = json.loads(rows[0]['messages'][1]['content'])
            context.pop('actor_id')
            context['market'] = {'question': 'Will a team win?'}
            rows[0]['messages'][1]['content'] = json.dumps(context)
        self.write_all()
        self.assertEqual(self.check()['status'], 'comparable')

    def test_scope_and_optional_period_metadata(self):
        def add_period(rows):
            context = json.loads(rows[0]['messages'][1]['content'])
            context['actor_metrics']['return_period_seconds'] = '86400'
            rows[0]['messages'][1]['content'] = json.dumps(context)
        self.edit_line('global', 'train', add_period)
        self.assertEqual(self.check()['status'], 'comparable')
        def wrong_scope(rows):
            context = json.loads(rows[0]['messages'][1]['content'])
            context['actor_metrics']['scope'] = 'current_market'
            rows[0]['messages'][1]['content'] = json.dumps(context)
        self.edit_line('global', 'train', wrong_scope)
        with self.assertRaisesRegex(ValueError, 'prompt scope'):
            self.check()

    def test_mismatched_metric_settings_rejected(self):
        path = self.datasets['global'] / 'manifest.json'
        original = json.loads(path.read_text())
        for key, value in [('selected_features', ['sharpe_ratio']), ('min_return_periods', 90),
                           ('lookback_seconds', 86400), ('metric_significant_digits', 8)]:
            manifest = copy.deepcopy(original)
            manifest['actor_metrics']['config'][key] = value
            path.write_text(json.dumps(manifest))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f'{key} differs'):
                self.check()

    def test_wrong_scope_or_missing_variant_rejected(self):
        path = self.datasets['global'] / 'manifest.json'
        original = json.loads(path.read_text())
        for fault in ('scope', 'variant'):
            value = copy.deepcopy(original)
            if fault == 'scope':
                value['actor_metrics']['config']['history_scope'] = compare.SCOPES['inmarket']
            else:
                value.pop('feature_variant')
            path.write_text(json.dumps(value))
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'scope|feature_variant'):
                self.check()

    def test_missing_features_or_redundant_context_metadata_rejected(self):
        for fault in ('missing', 'audit'):
            self.write('global')
            def edit(rows):
                context = json.loads(rows[0]['messages'][1]['content'])
                if fault == 'missing':
                    del context['actor_metrics']
                else:
                    context['actor_metrics']['audit_link'] = 'not-needed'
                rows[0]['messages'][1]['content'] = json.dumps(context)
            self.edit_line('global', 'train', edit)
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'actor_metrics'):
                self.check()

    def test_query_overlap_and_equality_are_rejected(self):
        for first in (1, 2):
            self.records['validation'] = [self.record(1, first, 6)]
            self.write_all()
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, 'overlap or touch'):
                self.check()

    def test_duplicate_split_file_rejected(self):
        path = self.datasets['basic'] / 'train.jsonl'
        path.with_name('train.jsonl.gz').write_bytes(gzip.compress(path.read_bytes()))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.check()

    def test_empty_disabled_test_is_allowed(self):
        self.records['test'] = []
        self.manifest['enabled_splits'] = ['train', 'validation']
        self.manifest['fixture_to_split'].pop('fixture2')
        self.write_all()
        result = self.check()
        self.assertEqual(result['splits']['test']['targets'], 0)
        self.assertIsNone(result['splits']['test']['prompt_characters']['basic']['mean'])

    def test_purge_preserves_train_and_retained_conversations_exactly(self):
        self.records['validation'].insert(0, self.record(3, 2, 3, fixture='fixture1'))
        self.records['test'].insert(0, self.record(4, 6, 7, fixture='fixture2'))
        self.write('basic')
        source = self.datasets['basic']
        before = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
        output = self.root / 'purged'
        report = compare.purge_overlapping_splits(source, output)
        self.assertEqual(report['conversations_dropped'], 2)
        self.assertEqual(report['targets_dropped'], 4)
        self.assertEqual((output / 'train.jsonl').read_bytes(), before[Path('train.jsonl')])
        self.assertEqual((output / 'validation.jsonl').read_bytes(),
                         before[Path('validation.jsonl')].splitlines(keepends=True)[1])
        self.assertEqual((output / 'test.jsonl').read_bytes(),
                         before[Path('test.jsonl')].splitlines(keepends=True)[1])
        self.assertEqual(len((output / 'source_audit.jsonl').read_text().splitlines()), 3)
        self.assertEqual(len((output / 'audit/purged_conversations.jsonl').read_text().splitlines()), 2)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertFalse(manifest['token_lengths_checked'])
        self.assertNotIn('tokens', manifest['stats']['train'])
        self.assertTrue(manifest['global_query_time_separation_enforced'])
        for path, content in before.items():
            self.assertEqual((source / path).read_bytes(), content)
        compare.validate_chronological_splits(output)

    def test_purge_uses_latest_kept_validation_query_for_test_boundary(self):
        # Dropped validation ends after all test rows. It must not set the test boundary.
        self.records['validation'].insert(0, self.record(3, 1, 25, fixture='fixture1'))
        self.write('basic')
        result = compare.purge_overlapping_splits(self.datasets['basic'], self.root / 'purged')
        self.assertEqual(result['splits']['test']['conversations'], 1)

    def test_purge_empty_holdout_fails_without_output(self):
        self.records['validation'] = [self.record(1, 2, 6)]
        self.write('basic')
        output = self.root / 'purged'
        with self.assertRaisesRegex(ValueError, 'validation empty.*Collect later markets'):
            compare.purge_overlapping_splits(self.datasets['basic'], output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.purged.*')), [])

    def test_purge_refuses_already_enriched_or_existing_outputs(self):
        with self.assertRaisesRegex(ValueError, 'basic dataset'):
            compare.purge_overlapping_splits(self.datasets['global'], self.root / 'purged')
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            compare.purge_overlapping_splits(self.datasets['basic'], self.datasets['global'])

    def test_purge_rejects_symlink_audit(self):
        source = self.datasets['basic']
        (source / 'audit/link').symlink_to(source / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            compare.purge_overlapping_splits(source, self.root / 'purged')
        self.assertFalse((self.root / 'purged').exists())

    def test_cli_comparison_and_purge(self):
        command = [sys.executable, str(ROOT / 'tools/compare_actor_variants.py')]
        result = subprocess.run(command + ['--basic', str(self.datasets['basic']),
                                '--inmarket', str(self.datasets['inmarket']),
                                '--global', str(self.datasets['global'])], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'comparable')
        result = subprocess.run(command + ['purge', '--input', str(self.datasets['basic']),
                                '--out', str(self.root / 'purged')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['conversations_dropped'], 0)


if __name__ == '__main__':
    unittest.main()
