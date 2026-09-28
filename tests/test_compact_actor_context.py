"""Keep useful SFT context and exact targets while removing audit-only tokens."""
import copy
import hashlib
import io
import json
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures
from tests import test_derive_actor_metrics_sft as metric_fixtures

metrics = metric_fixtures.metrics


class CompactActorContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_opaque_identity_and_repeated_metadata_stay_outside_prompt(self):
        path, _, rows = self.fixture.source(with_market_context=True, context_version=2)
        record, _, _ = self.fixture.record(path)
        first = json.loads(record['messages'][1]['content'])
        self.assertEqual(record['actor_id'], rows[1]['actor_id'])
        self.assertEqual(record['market_id'], rows[1]['market_id'])
        self.assertEqual(first['market'], {
            'fixture': 'Match 1', 'question': 'Will the match end in a draw?',
            'outcomes': ['Yes', 'No'],
        })
        prompts = '\n'.join(message['content'] for message in record['messages'])
        for redundant in ('actor_id', 'market_id', 'condition_id', 'metadata_status',
                          'past_observed_trades', rows[1]['actor_id'], rows[1]['condition_id']):
            self.assertNotIn(redundant, prompts)
        self.assertIn('Contract metadata is retrospective', record['messages'][0]['content'])
        self.assertNotIn('market', json.loads(record['messages'][3]['content']))

    def test_news_and_execution_targets_are_preserved_without_source_links(self):
        path, actor_file, rows = self.fixture.source(with_market_context=True, context_version=2)
        for row in rows:
            for item in row['news']:
                item.update(url='https://espn.com/match/example', event_id='audit-only-event-id')
        actor_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        raw_hash = hashlib.sha256(actor_file.read_bytes()).hexdigest()
        record, audit, _ = self.fixture.record(path)
        self.assertEqual(hashlib.sha256(actor_file.read_bytes()).hexdigest(), raw_hash)
        self.assertEqual(audit['source_sha256'], raw_hash)
        for i in range(2):
            trade_row = rows[2 * i + 1]
            user = json.loads(record['messages'][2 * i + 1]['content'])
            target = json.loads(record['messages'][2 * i + 2]['content'])
            self.assertEqual(user['query_time'], trade_row['timestamp'])
            self.assertEqual(user['news'], [
                {key: item[key] for key in ('time', 'type', 'text')}
                for item in trade_row['news']
            ])
            self.assertEqual(target, {'action': 'TRADE', 'trades': [
                {key: trade[key] for key in ('side', 'outcome', 'shares', 'price')}
                for trade in trade_row['label']['trades']
            ]})
            for outcome in ('yes', 'no'):
                self.assertEqual(user['market_context'][outcome], {
                    key: trade_row['market_context'][outcome][key]
                    for key in ('price', 'age_seconds')
                })
        prompts = '\n'.join(message['content'] for message in record['messages'])
        self.assertNotIn('https://', prompts)
        self.assertNotIn('audit-only-event-id', prompts)

    def test_manifest_identifies_basic_schema_and_retains_split_and_target_contract(self):
        for index in (1, 2, 3):
            self.fixture.source(index, index, with_market_context=True, context_version=2)
        with patch('sys.stdout', new_callable=io.StringIO):
            metadata = fixtures.builder.sft_export(self.fixture.args())
        self.assertEqual(metadata['prompt_schema'], 'actor_market_prompt_v2')
        self.assertEqual(metadata['prompt_schema_version'], 2)
        self.assertEqual(metadata['feature_variant'], 'basic')
        self.assertEqual(metadata['fixture_to_split'], {
            'espn:1': 'train', 'espn:2': 'validation', 'espn:3': 'test',
        })
        self.assertEqual(metadata['targets_truncated_or_dropped'], 0)
        for split in ('train', 'validation', 'test'):
            self.assertEqual(metadata['stats'][split]['targets'], 2)
            self.assertEqual(metadata['stats'][split]['executions'], 3)

    def prepared_feature_inputs(self):
        index = {}
        for number in (1, 2, 3):
            _, _, rows = self.fixture.source(number, number, with_market_context=True, context_version=2)
            for trade_row in rows[1::2]:
                instant = metrics.timestamp_us(trade_row['timestamp'])
                index[(trade_row['actor_id'], trade_row['market_id'], instant)] = {
                    'actor_metrics': {
                        'values': {
                            'average_execution_notional': '12345.6789123456789',
                            'sharpe_ratio': '0.1234567891234567890',
                            'return_volatility': None,
                            'consecutive_loss_streak': 2,
                        },
                        'sample_counts': {'captured_executions': 3, 'eligible_return_periods': 30},
                        'window': {'end_us': instant, 'end_inclusive': False},
                    },
                    'trades': [{key: trade[key] for key in ('side', 'outcome', 'shares', 'price')}
                               for trade in trade_row['label']['trades']],
                }
        with patch('sys.stdout', new_callable=io.StringIO):
            fixtures.builder.sft_export(self.fixture.args())
        return self.fixture.root / 'sft', index

    def test_metrics_accept_real_id_free_prompt_preserving_labels_news_and_number_types(self):
        source, index = self.prepared_feature_inputs()
        path = source / 'train.jsonl'
        original = json.loads(path.read_text())
        context = json.loads(original['messages'][1]['content'])
        self.assertNotIn('actor_id', context)
        self.assertNotIn('market_id', context['market'])
        # Non-derived numeric context must retain its type and full JSON text.
        context['numeric_context'] = {'number': 0.123456789012345, 'count': 7, 'flag': True}
        original['messages'][1]['content'] = json.dumps(context, indent=2)
        path.write_text(json.dumps(original) + '\n')
        manifest_path = source / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['split_sha256']['train'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        destination = self.fixture.root / 'enriched'
        metrics.enrich_sft(source, destination, index, {'feature_variant': 'inmarket'})
        result = json.loads((destination / 'train.jsonl').read_text())
        for offset in (0, 2, 4):
            self.assertEqual(result['messages'][offset], original['messages'][offset])
        for offset in (1, 3):
            enriched_context = json.loads(result['messages'][offset]['content'])
            del enriched_context['actor_metrics']
            self.assertEqual(enriched_context, json.loads(original['messages'][offset]['content']))
        enriched_text = result['messages'][1]['content']
        self.assertTrue(enriched_text.startswith(original['messages'][1]['content'].rstrip()[:-1]))
        numbers = json.loads(enriched_text)['numeric_context']
        self.assertIs(type(numbers['number']), float)
        self.assertIs(type(numbers['count']), int)
        self.assertIs(type(numbers['flag']), bool)
        self.assertEqual(numbers['number'], 0.123456789012345)

    def test_derived_prompt_rounding_and_feature_subset_leave_full_precision_audit_unchanged(self):
        source, index = self.prepared_feature_inputs()
        raw_before = copy.deepcopy(index)
        raw_audit = self.fixture.root / 'raw_metrics.jsonl'
        raw_audit.write_text(''.join(json.dumps(value['actor_metrics']) + '\n' for value in index.values()))
        raw_bytes = raw_audit.read_bytes()
        destination = self.fixture.root / 'enriched'
        selected = metrics.select_features('sharpe_ratio,average_execution_notional,consecutive_loss_streak')
        metrics.enrich_sft(source, destination, index, {'selected_features': selected})
        result = json.loads((destination / 'train.jsonl').read_text())
        prompt_metrics = json.loads(result['messages'][1]['content'])['actor_metrics']
        self.assertEqual(prompt_metrics['values'], {
            'average_execution_notional': '12345.67891',
            'sharpe_ratio': '0.1234567891',
            'consecutive_loss_streak': 2,
        })
        self.assertEqual(prompt_metrics['sample_counts'], {'captured_executions': 3, 'eligible_return_periods': 30})
        self.assertNotIn('return_volatility', prompt_metrics['values'])
        self.assertEqual(index, raw_before)
        self.assertEqual(raw_audit.read_bytes(), raw_bytes)
        self.assertEqual(json.loads(result['messages'][2]['content'])['trades'][0]['shares'], '13.123456789')

    def test_feature_selection_rejects_unknown_duplicate_and_empty_names(self):
        self.assertEqual(metrics.select_features(None), list(metrics.ACTOR_METRIC_NAMES))
        for invalid in ('future_profit', 'sharpe_ratio,sharpe_ratio',
                        'sharpe_ratio, sharpe_ratio', '', 'sharpe_ratio,'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                metrics.select_features(invalid)
        core = {'values': {'sharpe_ratio': '0.12345678901234567890'}, 'sample_counts': {}}
        for names in (['future_profit'], ['sharpe_ratio', 'sharpe_ratio'], []):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'selected metric names'):
                metrics.model_metric_fields(core, {'selected_features': names})


if __name__ == '__main__':
    unittest.main()
