"""Regression: raw interval labels must survive conversion, features and loss masking."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
import run_world_cup_experiments as pipeline
import prepare_world_cup_evaluation as evaluation
import world_cup_eval_common as common
import compare_actor_variants as variants
import derive_actor_metrics as metrics


class NoTradeSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def record(self):
        path, actor, rows = self.fixture.source(with_market_context=True, context_version=2)
        source = fixtures.builder.sft_discover([path], None)[0]
        return fixtures.builder.sft_convert_actor(actor, source)[0], rows

    def dataset(self):
        for i in range(1, 4):
            self.fixture.source(i, i, with_market_context=True, context_version=2)
        with patch('sys.stdout', new_callable=io.StringIO):
            # CLI default, deliberately no include/legacy option.
            fixtures.builder.prepare_main(['--input-root', str(self.root), '--out', str(self.root / 'basic')])
        return self.root / 'basic'

    def test_default_keeps_every_raw_label_and_news_once(self):
        record, rows = self.record()
        labels = [json.loads(m['content']) for m in record['messages'] if m['role'] == 'assistant']
        self.assertEqual([r['action'] for r in labels], ['NO_TRADE', 'TRADE', 'NO_TRADE', 'TRADE'])
        for raw, label in zip(rows, labels):
            if raw['label']['action'] == 'NO_TRADE':
                self.assertEqual(label, raw['label'])
            else:
                self.assertEqual(label['trades'], [{k: t[k] for k in ('side','outcome','shares','price')}
                                                  for t in raw['label']['trades']])
        self.assertEqual(record['target_count'], len(rows))
        self.assertEqual(record['execution_count'], 3)
        self.assertEqual(record['no_trade_target_count'], 2)
        self.assertEqual(sum(m['content'].count('Unique news 0.5') for m in record['messages']), 1)
        self.assertEqual(sum(m['content'].count('Unique news 2') for m in record['messages']), 1)
        variants.conversation(record)

    def test_real_trainer_mask_includes_no_trade_answers(self):
        record, _ = self.record()
        tokenizer = fixtures.OffsetTokenizer()
        encoded, count = fixtures.trainer.encode_conversation(record, tokenizer, 20000, 'mixed')
        supervised = ''.join(tokenizer.lookup[x] for x in encoded['labels'] if x != -100)
        self.assertEqual(count, 4)
        self.assertEqual(supervised.count('NO_TRADE'), 2)
        self.assertEqual(supervised, ''.join(m['content'] + '<|im_end|>' for m in record['messages']
                                            if m['role'] == 'assistant'))
        self.assertNotIn('Unique news', supervised)

    def test_manifest_and_tokenizer_report_both_classes(self):
        basic = self.dataset()
        meta = json.loads((basic / 'manifest.json').read_text())
        self.assertTrue(meta['no_trade_targets'])
        self.assertFalse(meta['prospective_trade_timing_benchmark'])
        variants.scan_dataset(basic)
        for split in ('train', 'validation', 'test'):
            self.assertEqual(meta['stats'][split]['targets'], 4)
            self.assertEqual(meta['stats'][split]['no_trade_targets'], 2)
            _, stats = fixtures.trainer.read_split(basic / f'{split}.jsonl', fixtures.OffsetTokenizer(), 20000)
            self.assertEqual(stats['action_counts'], {'NO_TRADE': 2, 'TRADE': 2})

    def test_old_trade_only_mode_requires_explicit_option(self):
        path, actor, _ = self.fixture.source()
        source = fixtures.builder.sft_discover([path], None)[0]
        old, _, _ = fixtures.builder.sft_convert_actor(actor, source, include_no_trade=False)
        self.assertEqual(old['target_count'], 2)
        self.assertNotIn('NO_TRADE', ''.join(m['content'] for m in old['messages']))

    def test_training_rejects_trade_only_data_without_explicit_override(self):
        with self.assertRaisesRegex(ValueError, 'no NO_TRADE'):
            fixtures.trainer.validate_target_counts({'action_counts': {'TRADE': 10}})
        fixtures.trainer.validate_target_counts({'action_counts': {'TRADE': 10}}, True)
        fixtures.trainer.validate_target_counts({'action_counts': {'TRADE': 10, 'NO_TRADE': 10}})

    def test_deleted_interval_or_changed_interval_label_rejected(self):
        record, _ = self.record()
        changed = copy.deepcopy(record)
        del changed['messages'][1:3]
        changed['target_count'] -= 1
        with self.assertRaises(ValueError):
            variants.conversation(changed)
        changed = copy.deepcopy(record)
        changed['messages'][2]['content'] = record['messages'][4]['content']
        with self.assertRaisesRegex(ValueError, 'NO_TRADE'):
            variants.conversation(changed)

    def test_wrong_endpoint_or_continuity_rejected(self):
        record, _ = self.record()
        for key, value in [('start_inclusive', True), ('end', '2026-06-01T16:02:00Z')]:
            changed = copy.deepcopy(record)
            context = json.loads(changed['messages'][1]['content'])
            context['interval'][key] = value
            changed['messages'][1]['content'] = json.dumps(context)
            with self.assertRaises(ValueError):
                variants.conversation(changed)

    def test_evaluation_ids_unique_and_current_future_labels_hidden(self):
        record, _ = self.record()
        targets = list(common.targets(record))
        self.assertEqual(len({t['id'] for t in targets}), 4)
        self.assertEqual(targets[0]['query_time'], targets[1]['query_time'])
        self.assertEqual(len(targets[0]['messages']), 2)
        self.assertEqual(len(targets[1]['messages']), 4)
        self.assertNotIn('13.123456789', json.dumps(targets[0]['messages']))
        self.assertNotIn('13.123456789', json.dumps(targets[1]['messages']))
        self.assertIn('13.123456789', json.dumps(targets[2]['messages']))

    def test_offline_inmarket_keeps_labels_and_excludes_current_execution(self):
        basic = self.dataset()
        pipeline.derive_inmarket(basic, self.root / 'inmarket')
        source = variants.scan_dataset(self.root / 'inmarket')
        for split in ('train', 'validation', 'test'):
            record = next(variants.lines(source['paths'][split]))[1]
            original = next(variants.lines(basic / f'{split}.jsonl'))[1]
            common.check_pair(original, record)
            counts = [json.loads(m['content'])['actor_metrics']['sample_counts']['captured_executions']
                      for m in record['messages'] if m['role'] == 'user']
            self.assertEqual(counts, [0, 0, 2, 2])

    def test_raw_metric_enrichment_supports_local_and_global_scope(self):
        basic = self.dataset()
        for variant, scope in [('inmarket', 'actor_and_binary_market'), ('global', 'actor_across_all_markets')]:
            index = {}
            for split in ('train', 'validation', 'test'):
                record = next(variants.lines(basic / f'{split}.jsonl'))[1]
                for offset in (3, 7):
                    context = json.loads(record['messages'][offset]['content'])
                    when = metrics.timestamp_us(context['query_time'])
                    index[(record['actor_id'], record['market_id'], when)] = {
                        'trades': json.loads(record['messages'][offset+1]['content'])['trades'],
                        'actor_metrics': metrics.compute_metrics([], [], [], when, None, 30)}
            config = {'feature_variant': variant, 'history_scope': scope,
                      'selected_features': ['average_execution_notional'], 'strict_prior': True}
            metrics.enrich_sft(basic, self.root / variant, index, config)
            enriched = variants.scan_dataset(self.root / variant)
            record = next(variants.lines(enriched['paths']['train']))[1]
            labels = [json.loads(m['content'])['action'] for m in record['messages'] if m['role']=='assistant']
            self.assertEqual(labels, ['NO_TRADE', 'TRADE', 'NO_TRADE', 'TRADE'])

    def test_bundle_keeps_all_targets_and_labels_task_honestly(self):
        basic = self.dataset()
        pipeline.derive_inmarket(basic, self.root / 'inmarket')
        with patch('sys.stdout', new_callable=io.StringIO):
            meta = evaluation.prepare(evaluation.parse_args(['--basic-sft', str(basic), '--inmarket-sft',
                str(self.root / 'inmarket'), '--out', str(self.root / 'bundle')]))
        self.assertEqual(meta['targets'], 4)
        self.assertEqual(meta['task'], 'observed_interval_and_execution_reconstruction')
        _, records = common.read_bundle(self.root / 'bundle')
        self.assertEqual(len(list(common.targets(records['basic'][0]))), 4)

    def test_no_trade_correctness_does_not_inflate_trade_detail_metrics(self):
        trade = '{"action":"TRADE","trades":[{"side":"BUY","outcome":"Yes","shares":"2","price":"0.4"}]}'
        no = '{"action":"NO_TRADE"}'
        result = common.summarize([{'answer': no, 'prediction': no}, {'answer': trade, 'prediction': no}])
        self.assertEqual(result['action_correct'], 0.5)
        self.assertEqual(result['no_trade_recall'], 1)
        self.assertEqual(result['trade_recall'], 0)
        self.assertEqual(result['exact_trade_multiset'], 0)
        self.assertEqual(result['numeric_target_coverage'], 0)
        self.assertEqual(common.summarize([{'answer': trade, 'prediction': trade}])['exact_trade_multiset'], 1)


if __name__ == '__main__':
    unittest.main()
