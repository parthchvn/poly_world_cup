"""Offline prospective-protocol and completion-scoring regression coverage."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_world_cup_multigpu as training
import evaluate_interval_decisions as evaluate
from tests.test_prepare_actor_sft import OffsetTokenizer


def record(action='TRADE', row_id='row1'):
    start, end = '2026-06-01T17:00:00Z', '2026-06-01T17:05:00Z'
    query = {'query_time': start, 'prediction_window': {'start': start, 'end': end},
        'market': {'question': 'Draw?', 'outcomes': ['Yes', 'No']},
        'news': [{'time': '2026-06-01T16:59:00Z', 'type': 'goal', 'text': 'Earlier goal'}],
        'prior_executions': [{'time': '2026-06-01T16:58:00Z',
            'trades': [{'side': 'BUY', 'outcome': 'Yes', 'shares': '3', 'price': '.4'}]}],
        'unrealized_in_market_pnl': 0, 'derived_features': {'prior_count': 1}}
    return {'target_protocol': training.INTERVAL_PROTOCOL, 'row_id': row_id, 'actor_id': 'actor',
        'market_id': 'market', 'fixture_id': 'fixture', 'target_count': 1,
        'query_time': start, 'interval_start': start, 'interval_end': end,
        'messages': [{'role': 'system', 'content': 'Predict activity in the given interval.'},
                     {'role': 'user', 'content': json.dumps(query)},
                     {'role': 'assistant', 'content': evaluate.canonical_answer(action)}]}


class IntervalTrainingTests(unittest.TestCase):
    def test_unbalanced_interval_counts_pass_without_relaxing_legacy(self):
        counts = {'action_counts': {'TRADE': 2, 'NO_TRADE': 19},
                  'protocol_counts': {training.INTERVAL_PROTOCOL: 21}}
        training.validate_target_counts(counts)
        with self.assertRaisesRegex(ValueError, 'target counts differ'):
            training.validate_target_counts({'action_counts': counts['action_counts']})
        counts['protocol_counts']['observed_interval_and_execution_v1'] = 1
        with self.assertRaisesRegex(ValueError, 'Do not mix'):
            training.validate_target_counts(counts)

    def test_one_class_interval_validation_is_allowed(self):
        training.validate_target_counts({'action_counts': {'NO_TRADE': 10},
            'protocol_counts': {training.INTERVAL_PROTOCOL: 10}})
        with self.assertRaisesRegex(ValueError, 'no NO_TRADE'):
            training.validate_target_counts({'action_counts': {'TRADE': 10}})

    def test_cutoff_and_half_open_interval_metadata_agree(self):
        training.validate_interval_record(record())
        changed = record()
        changed['query_time'] = changed['interval_end']
        with self.assertRaisesRegex(ValueError, 'cutoffs'):
            training.validate_interval_record(changed)
        changed = record()
        changed['interval_end'] = changed['interval_start']
        with self.assertRaisesRegex(ValueError, 'positive duration'):
            training.validate_interval_record(changed)

    def test_observations_at_cutoff_and_later_are_rejected(self):
        for field in ('news', 'prior_executions'):
            for timestamp in ('2026-06-01T17:00:00Z', '2026-06-01T17:01:00Z'):
                changed = record()
                query = json.loads(changed['messages'][-2]['content'])
                query[field][0]['time'] = timestamp
                changed['messages'][-2]['content'] = json.dumps(query)
                with self.assertRaisesRegex(ValueError, 'at/after query cutoff'):
                    training.validate_interval_record(changed)

    def test_binary_only_target_and_one_query_are_required(self):
        changed = record()
        changed['messages'][-1]['content'] = '{"action":"TRADE","trades":[]}'
        with self.assertRaisesRegex(ValueError, 'only binary action'):
            training.validate_interval_record(changed)
        changed = record()
        changed['messages'].extend(copy.deepcopy(changed['messages'][-2:]))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            training.validate_interval_record(changed)

    def test_supplied_price_age_must_be_positive(self):
        for age in (0, -1, float('inf')):
            changed = record()
            query = json.loads(changed['messages'][-2]['content'])
            query['market_context'] = {'yes': {'price': '.4', 'age_seconds': age}}
            changed['messages'][-2]['content'] = json.dumps(query)
            with self.assertRaisesRegex(ValueError, 'price ages must be positive'):
                training.validate_interval_record(changed)

    def test_actual_builder_outputs_match_trainer_and_evaluation_protocol(self):
        from tests.test_interval_decision_data import IntervalDatasetTests
        fixture = IntervalDatasetTests()
        fixture.setUp()
        try:
            manifest, directory = fixture.prepare()
            self.assertEqual(manifest['target_protocol'], training.INTERVAL_PROTOCOL)
            for split in ('train', 'validation', 'test'):
                _, stats = training.read_split(directory / (split + '.jsonl'), OffsetTokenizer(), 20000)
                training.validate_target_counts(stats)
                for row in fixture.rows(directory / (split + '.jsonl')):
                    evaluate.encode_candidates(row, OffsetTokenizer(), 20000)
        finally:
            fixture.tearDown()

    def test_read_split_records_protocol_counts_and_masks_both_actions(self):
        tokenizer = OffsetTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            records = [record('NO_TRADE', str(i)) for i in range(3)] + [record('TRADE', 'last')]
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            encoded, stats = training.read_split(path, tokenizer, 3000)
        training.validate_target_counts(stats)
        self.assertEqual(stats['protocol_counts'], {training.INTERVAL_PROTOCOL: 4})
        self.assertEqual(stats['action_counts'], {'NO_TRADE': 3, 'TRADE': 1})
        for value, row in zip(encoded, records):
            supervised = ''.join(tokenizer.lookup[token] for token in value['labels'] if token != -100)
            self.assertEqual(supervised, row['messages'][-1]['content'] + '<|im_end|>')


class IntervalEvaluationTests(unittest.TestCase):
    def test_frozen_manifest_rejects_test_replacement_before_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, run, dataset = root / 'model', root / 'run', root / 'dataset'
            model.mkdir()
            dataset.mkdir()
            (run / 'adapter').mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'):
                (run / 'adapter' / name).write_text('{}')
            for split in ('train', 'validation', 'test'):
                (dataset / (split + '.jsonl')).write_text(json.dumps(record()) + '\n')
            manifest = {'target_protocol': training.INTERVAL_PROTOCOL,
                'files': {p.name: training.sha256_file(p) for p in dataset.iterdir()}}
            training.atomic_json(dataset / 'manifest.json', manifest)
            metadata = {'status': 'completed', 'mode': 'full', 'test_used': False,
                'data': {s: {'protocol_counts': {training.INTERVAL_PROTOCOL: 1}} for s in ('train', 'validation')},
                'signature': {'data': {'manifest_sha256': training.sha256_file(dataset / 'manifest.json'),
                    'model_config_sha256': training.sha256_file(model / 'config.json'),
                    'sources': {s: {'sha256': manifest['files'][s+'.jsonl']} for s in ('train', 'validation')}}}}
            training.atomic_json(run / 'training_metadata.json', metadata)
            evaluate.validate_run(run, dataset, model)
            (dataset / 'test.jsonl').write_text(json.dumps(record('NO_TRADE')) + '\n')
            with self.assertRaisesRegex(ValueError, 'test file differs'):
                evaluate.validate_run(run, dataset, model)
            manifest['files']['test.jsonl'] = training.sha256_file(dataset / 'test.jsonl')
            training.atomic_json(dataset / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                evaluate.validate_run(run, dataset, model)

    def test_probabilities_use_joint_likelihood_not_length_normalization(self):
        self.assertAlmostEqual(evaluate.normalized_trade_probability(math.log(.2), math.log(.6)), .75)
        self.assertEqual(evaluate.normalized_trade_probability(-10000, 0), 1.)
        self.assertEqual(evaluate.normalized_trade_probability(0, -10000), 0.)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            evaluate.normalized_trade_probability(float('nan'), 0.)

    def test_candidates_share_prompt_and_supervise_action_plus_end_marker(self):
        tokenizer = OffsetTokenizer()
        original = record('TRADE')
        candidates = evaluate.encode_candidates(original, tokenizer, 3000)
        self.assertEqual(json.loads(original['messages'][-1]['content'])['action'], 'TRADE')
        for action, encoded in zip(evaluate.ACTIONS, candidates):
            supervised = ''.join(tokenizer.lookup[token] for token in encoded['labels'] if token != -100)
            self.assertEqual(supervised, evaluate.canonical_answer(action) + '<|im_end|>')
            self.assertNotIn('Earlier goal', supervised)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            evaluate.encode_candidates(original, tokenizer, 10)

    def test_noncanonical_training_target_is_rejected(self):
        changed = record()
        changed['messages'][-1]['content'] = json.dumps({'action': 'TRADE'})
        with self.assertRaisesRegex(ValueError, 'canonical'):
            evaluate.encode_candidates(changed, OffsetTokenizer(), 3000)

    def test_journal_recovers_incomplete_last_line_and_checks_prompt(self):
        source = record()
        prediction = {'row_id': source['row_id'], 'label': 1,
            'prompt_sha256': hashlib.sha256(evaluate.dump(source['messages'][:-1]).encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            path.write_text(json.dumps(prediction) + '\n{"row_id":')
            self.assertEqual(evaluate.recover_predictions(path, [source], True), [prediction])
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            changed = copy.deepcopy(source)
            changed['messages'][0]['content'] += ' Changed'
            with self.assertRaisesRegex(ValueError, 'prompts'):
                evaluate.recover_predictions(path, [changed], True)

    def test_metrics_include_baselines_and_group_equal_weight_loss(self):
        rows = [dict(row_id=str(i), label=label, probability_trade=probability, actor_id=actor,
            fixture_id='match' + str(i % 2), actor_seen_in_training=i < 2)
            for i, (label, probability, actor) in enumerate([(0, .1, 'a'), (1, .7, 'a'), (1, .8, 'b')])]
        summary = evaluate.summarize(rows, .2, .5)
        self.assertEqual(summary['model']['confusion'], {'tp': 2, 'tn': 1, 'fp': 0, 'fn': 0})
        self.assertEqual(summary['baselines']['always_no_trade']['recall'], 0.)
        self.assertEqual(summary['baselines']['train_frequency']['prevalence'], 2/3)
        actor_losses = [m['log_loss'] for m in summary['by_actor_id'].values()]
        self.assertAlmostEqual(summary['macro_actor_id']['log_loss'], sum(actor_losses)/2)


if __name__ == '__main__':
    unittest.main()
