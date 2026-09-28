import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('actor_metrics_sft_tested', ROOT / 'scripts/derive_actor_metrics.py')
metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)


class EnrichSFTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'result/sft'
        self.actor = '0x' + 'a' * 40
        self.times = ['2026-06-01T10:00:00Z', '2026-06-01T11:00:00Z']
        self.trade = {'side': 'BUY', 'outcome': 'Yes', 'shares': '2.25', 'price': '0.40'}
        self.index = {}
        self.records = {}
        self.manifest = {
            'format': 'actor_market_trade_messages_v1',
            'enabled_splits': ['train', 'validation'],
            'stats': {}, 'split_sha256': {},
            'token_lengths_checked': True, 'max_length_checked': 8192,
            'actor_snapshots_used_as_model_input': False,
            'fixture_to_split': {'fixture1': 'train', 'fixture2': 'validation'},
            'sources': [{'source': 'preserved'}],
        }
        for split, market in [('train', '1'), ('validation', '2')]:
            messages = [{'role': 'system', 'content': 'Predict the observed execution attributes.'}]
            for position, when in enumerate(self.times):
                context = {'query_time': when, 'news': []}
                if position == 0:
                    context.update(actor_id=self.actor, market={'market_id': market}, past_observed_trades=[])
                messages.extend([
                    {'role': 'user', 'content': json.dumps(context)},
                    {'role': 'assistant', 'content': json.dumps({'action': 'TRADE', 'trades': [self.trade]})},
                ])
                instant = metrics.timestamp_us(when)
                self.index[(self.actor, market, instant)] = {
                    'actor_metrics': {
                        'values': {'mean_execution_notional': None if position == 0 else '0.90'},
                        'sample_counts': {'executions': position},
                        'unavailable_reasons': {},
                        'window': {'end_us': instant, 'end_inclusive': False, 'start_us': None},
                    },
                    'trades': [copy.deepcopy(self.trade)],
                }
            self.records[split] = [{
                'actor_id': self.actor, 'market_id': market, 'sequence_id': f'sequence{market}',
                'fixture_id': f'fixture{market}', 'target_count': 2, 'execution_count': 2,
                'messages': messages,
            }]
            self.manifest['stats'][split] = {
                'conversations': 1, 'targets': 2, 'executions': 2, 'tokens': 100, 'max_tokens': 100,
            }
        self.records['test'] = []
        self.manifest['stats']['test'] = {}
        (self.source / 'source_audit.jsonl').write_text('{"original":"audit"}\n')
        (self.source / 'split_plan.json').write_text(json.dumps({'fixture_to_split': self.manifest['fixture_to_split']}))
        (self.source / 'audit').mkdir()
        (self.source / 'audit/snapshots.json').write_text('{"total_pnl":"99","temporal_scope":"collection_time"}\n')
        self.sync_source()

    def tearDown(self):
        self.tmp.cleanup()

    def sync_source(self):
        for split, rows in self.records.items():
            data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            (self.source / f'{split}.jsonl').write_bytes(data)
            self.manifest['split_sha256'][split] = hashlib.sha256(data).hexdigest()
        (self.source / 'manifest.json').write_text(json.dumps(self.manifest))

    def run_join(self):
        return metrics.enrich_sft(self.source, self.output, self.index, {'window_days': 30})

    def first_result(self):
        return json.loads((self.output / 'train.jsonl').read_text().splitlines()[0])

    def test_exact_join_changes_only_user_context_and_preserves_audit(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        report = self.run_join()
        result = self.first_result()
        original = self.records['train'][0]
        self.assertEqual(result['messages'][0], original['messages'][0])
        for offset in (2, 4):
            self.assertEqual(result['messages'][offset], original['messages'][offset])
        for offset, position in [(1, 0), (3, 1)]:
            context = json.loads(result['messages'][offset]['content'])
            self.assertEqual(context['actor_metrics']['sample_counts']['executions'], position)
            self.assertEqual(set(context['actor_metrics']), {'values', 'sample_counts'})
            del context['actor_metrics']
            self.assertEqual(context, json.loads(original['messages'][offset]['content']))
        self.assertEqual(report['splits']['train']['targets'], 2)
        self.assertEqual(report['output'], 'sft')
        self.assertNotIn('total_pnl', (self.output / 'train.jsonl').read_text())
        self.assertNotIn('unavailable_reasons', (self.output / 'train.jsonl').read_text())
        self.assertEqual((self.output / 'audit/snapshots.json').read_bytes(), before[Path('audit/snapshots.json')])
        self.assertEqual((self.output / 'audit/metrics_source_manifest.json').read_bytes(), before[Path('manifest.json')])
        for path, content in before.items():
            self.assertEqual((self.source / path).read_bytes(), content)
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertFalse(manifest['token_lengths_checked'])
        self.assertIsNone(manifest['max_length_checked'])
        self.assertNotIn('tokens', manifest['stats']['train'])
        self.assertNotIn('max_tokens', manifest['stats']['train'])
        self.assertEqual(manifest['fixture_to_split'], self.manifest['fixture_to_split'])
        self.assertTrue(manifest['actor_metrics']['strict_prior'])
        self.assertEqual(manifest['actor_metrics']['prompt_fields'], ['values', 'sample_counts'])
        self.assertEqual(manifest['split_sha256']['train'], hashlib.sha256((self.output / 'train.jsonl').read_bytes()).hexdigest())

    def test_compressed_input_hash_is_uncompressed_bytes(self):
        path = self.source / 'train.jsonl'
        path.with_name('train.jsonl.gz').write_bytes(gzip.compress(path.read_bytes()))
        path.unlink()
        self.run_join()
        self.assertEqual(self.first_result()['target_count'], 2)

    def test_existing_context_number_types_and_text_are_preserved(self):
        msg = self.records['train'][0]['messages'][1]
        context = json.loads(msg['content'])
        context['temperature'] = 0.1
        msg['content'] = json.dumps(context, indent=2) + '  \n'
        self.sync_source()
        self.run_join()
        enriched = self.first_result()['messages'][1]['content']
        self.assertTrue(enriched.startswith(msg['content'].rstrip()[:-1]))
        self.assertEqual(json.loads(enriched)['temperature'], 0.1)
        self.assertIsInstance(json.loads(enriched)['temperature'], float)

    def test_missing_exact_query_fails_instead_of_nearest_join(self):
        first = (self.actor, '1', metrics.timestamp_us(self.times[0]))
        entry = self.index.pop(first)
        self.index[(first[0], first[1], first[2] + 1)] = entry
        with self.assertRaisesRegex(ValueError, 'No exact raw-history'):
            self.run_join()
        self.assertFalse(self.output.exists())

    def test_changed_assistant_trade_fails(self):
        label = json.loads(self.records['train'][0]['messages'][2]['content'])
        label['trades'][0]['price'] = '0.41'
        self.records['train'][0]['messages'][2]['content'] = json.dumps(label)
        self.sync_source()
        with self.assertRaisesRegex(ValueError, 'assistant trades differ'):
            self.run_join()
        self.assertFalse(self.output.exists())

    def test_non_strict_or_wrong_metrics_cutoff_fails(self):
        key = (self.actor, '1', metrics.timestamp_us(self.times[0]))
        window = self.index[key]['actor_metrics']['window']
        for change in ({'end_inclusive': True}, {'end_us': key[2] + 1}):
            with self.subTest(change=change):
                old = copy.deepcopy(window)
                window.update(change)
                with self.assertRaisesRegex(ValueError, 'strict prior cutoff'):
                    self.run_join()
                window.clear()
                window.update(old)
                self.assertFalse(self.output.exists())

    def test_existing_context_metrics_fail(self):
        msg = self.records['train'][0]['messages'][1]
        context = json.loads(msg['content'])
        context['actor_metrics'] = {'future': 1}
        msg['content'] = json.dumps(context)
        self.sync_source()
        with self.assertRaisesRegex(ValueError, 'already contains actor_metrics'):
            self.run_join()

    def test_wrong_actor_or_market_context_fails(self):
        msg = self.records['train'][0]['messages'][1]
        original = msg['content']
        for field in ('actor', 'market'):
            context = json.loads(original)
            if field == 'actor':
                context['actor_id'] = '0x' + 'b' * 40
            else:
                context['market']['market_id'] = '2'
            msg['content'] = json.dumps(context)
            self.sync_source()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'differs from conversation'):
                self.run_join()

    def test_duplicate_actor_market_across_splits_fails(self):
        record = copy.deepcopy(self.records['train'][0])
        record['sequence_id'] = 'different_id_same_actor_market'
        self.records['validation'] = [record]
        self.sync_source()
        with self.assertRaisesRegex(ValueError, 'Duplicate actor/market'):
            self.run_join()

    def test_repeated_query_or_wrong_role_fails(self):
        original = copy.deepcopy(self.records['train'][0]['messages'])
        for fault in ('timestamp', 'role'):
            messages = copy.deepcopy(original)
            if fault == 'timestamp':
                context = json.loads(messages[3]['content'])
                context['query_time'] = self.times[0]
                messages[3]['content'] = json.dumps(context)
            else:
                messages[3]['role'] = 'assistant'
            self.records['train'][0]['messages'] = messages
            self.sync_source()
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'strictly'):
                self.run_join()

    def test_declared_hash_and_counts_must_match(self):
        original = copy.deepcopy(self.manifest)
        for fault in ('hash', 'counts'):
            self.manifest = copy.deepcopy(original)
            if fault == 'hash':
                self.manifest['split_sha256']['train'] = '0' * 64
            else:
                self.manifest['stats']['train']['targets'] = 99
            (self.source / 'manifest.json').write_text(json.dumps(self.manifest))
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'differs from its manifest'):
                self.run_join()

    def test_symlink_in_audit_is_rejected(self):
        (self.source / 'audit/link').symlink_to(self.source / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_join()
        self.assertFalse(self.output.exists())

    def test_ambiguous_compressed_and_plain_split_is_rejected(self):
        (self.source / 'train.jsonl.gz').write_bytes(gzip.compress((self.source / 'train.jsonl').read_bytes()))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.run_join()

    def test_duplicate_context_keys_and_nonfinite_values_are_rejected(self):
        msg = self.records['train'][0]['messages'][1]
        original = msg['content']
        for suffix in (',"query_time":"2026-06-01T10:00:00Z"}', ',"unexpected":NaN}'):
            msg['content'] = original[:-1] + suffix
            self.sync_source()
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'Duplicate JSON key|Nonfinite JSON number'):
                self.run_join()
            self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
