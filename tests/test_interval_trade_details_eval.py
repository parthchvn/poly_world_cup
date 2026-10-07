"""Exact aggregate training targets and frozen-tolerance generation evaluation."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
import train_world_cup_multigpu as training
import evaluate_interval_decisions as binary
import evaluate_interval_trade_details as details
from tests.test_interval_training_eval import record as activity_record
from tests.test_prepare_actor_sft import OffsetTokenizer

TOLERANCES = {'price_delta': '0.02', 'shares_relative_delta': '0.2', 'shares_absolute_delta': '0'}


def record(action='TRADE'):
    result = activity_record(action)
    result['target_protocol'] = training.INTERVAL_DETAILS_PROTOCOL
    answer = {'action': action}
    if action == 'TRADE':
        answer['trades'] = [{'side': 'BUY', 'outcome': 'Yes', 'price': '0.41', 'shares': '100'}]
    result['messages'][-1]['content'] = json.dumps(answer, separators=(',', ':'), sort_keys=True)
    return result


class GenerationTokenizer(OffsetTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, **kwargs)
        return rendered + ('<|im_start|>assistant\n' if kwargs.get('add_generation_prompt') else '')


class TradeDetailTrainingTests(unittest.TestCase):
    def test_exact_numeric_target_gets_full_assistant_supervision(self):
        source = record()
        training.validate_interval_record(source)
        tokenizer = GenerationTokenizer()
        encoded, count = training.encode_conversation(source, tokenizer, 4000, 'details')
        self.assertEqual(count, 1)
        actual = ''.join(tokenizer.lookup[token] for token in encoded['labels'] if token != -100)
        self.assertEqual(actual, source['messages'][-1]['content'] + '<|im_end|>')
        training.validate_interval_record(record('NO_TRADE'))

    def test_numeric_labels_remain_exact_and_valid(self):
        for key, value in (('price', 'NaN'), ('price', '1.1'), ('shares', '0'), ('shares', '-2'), ('shares', 100)):
            source = record()
            answer = json.loads(source['messages'][-1]['content'])
            answer['trades'][0][key] = value
            source['messages'][-1]['content'] = json.dumps(answer)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                training.validate_interval_record(source)
        source = record('NO_TRADE')
        source['messages'][-1]['content'] = '{"action":"NO_TRADE","trades":[]}'
        with self.assertRaises(ValueError):
            training.validate_interval_record(source)

    def test_binary_scoring_rejects_details_and_mixed_training_protocols(self):
        with self.assertRaisesRegex(ValueError, 'Binary likelihood scoring requires activity'):
            binary.encode_candidates(record(), GenerationTokenizer(), 4000)
        training.validate_target_counts({'protocol_counts': {training.INTERVAL_DETAILS_PROTOCOL: 5},
            'action_counts': {'TRADE': 1, 'NO_TRADE': 4}})
        with self.assertRaisesRegex(ValueError, 'Do not mix'):
            training.validate_target_counts({'protocol_counts': {training.INTERVAL_PROTOCOL: 2,
                training.INTERVAL_DETAILS_PROTOCOL: 3}, 'action_counts': {'TRADE': 2, 'NO_TRADE': 3}})

    def test_details_builder_records_reach_generation_without_future_fills(self):
        from tests.test_interval_decision_data import IntervalDatasetTests
        fixture = IntervalDatasetTests()
        fixture.setUp()
        try:
            _, directory = fixture.prepare(target_mode='trade-details', trade_tolerances=TOLERANCES)
            for split in ('train', 'validation', 'test'):
                _, stats = training.read_split(directory / (split+'.jsonl'), GenerationTokenizer(), 30000)
                training.validate_target_counts(stats)
                for source in fixture.rows(directory / (split+'.jsonl')):
                    details.encode_generation_prompt(source, GenerationTokenizer(), 30000, 2048)
        finally:
            fixture.tearDown()


class TradeDetailEvaluationTests(unittest.TestCase):
    def test_scorer_rules_are_frozen_with_tolerances(self):
        manifest = {'trade_detail_semantics': details.DETAIL_SEMANTICS,
            'trade_tolerances': TOLERANCES,
            'implementation_sha256': {'tools/interval_trade_tolerances.py':
                training.sha256_file(ROOT / 'tools/interval_trade_tolerances.py')}}
        self.assertEqual(details.frozen_tolerances(manifest), TOLERANCES)
        manifest['implementation_sha256']['tools/interval_trade_tolerances.py'] = 'different version'
        with self.assertRaisesRegex(ValueError, 'scoring implementation differs'):
            details.frozen_tolerances(manifest)

    def test_generation_uses_only_training_conditioning_prefix(self):
        source, tokenizer = record(), GenerationTokenizer()
        answer = json.loads(source['messages'][-1]['content'])
        answer['trades'][0]['shares'] = '987654321'
        source['messages'][-1]['content'] = json.dumps(answer, separators=(',', ':'))
        ids = details.encode_generation_prompt(source, tokenizer, 4000, 512)
        prompt = ''.join(tokenizer.lookup[token] for token in ids)
        self.assertNotIn('987654321', prompt)
        self.assertIn('Earlier goal', prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))
        with self.assertRaisesRegex(ValueError, 'budget exceeds'):
            details.encode_generation_prompt(source, tokenizer, 512, 400)

    def test_malformed_or_unterminated_output_is_never_correct_no_trade(self):
        gold = {'action': 'NO_TRADE'}
        malformed = details.score_prediction(gold, 'not json', TOLERANCES, True)
        self.assertFalse(malformed['valid_prediction'])
        self.assertFalse(malformed['joint_correct'])
        truncated = details.score_prediction(gold, '{"action":"NO_TRADE"}', TOLERANCES, False)
        self.assertFalse(truncated['valid_prediction'])
        self.assertFalse(truncated['joint_correct'])
        self.assertIn('terminate', truncated['parse_error'])

    def test_fixed_tolerances_score_aggregate_predictions_and_baseline(self):
        gold = json.loads(record()['messages'][-1]['content'])
        prediction = copy.deepcopy(gold)
        prediction['trades'][0].update(price='0.43', shares='120')
        scored = details.score_prediction(gold, prediction, TOLERANCES, True)
        self.assertTrue(scored['joint_correct'])
        rows = [{'gold': gold, 'score': scored, 'actor_id': 'a', 'fixture_id': 'f',
            'actor_seen_in_training': True, 'terminated_with_eos': True, 'hit_generation_limit': False}]
        summary = details.summarize(rows, TOLERANCES)
        self.assertEqual(summary['model']['trade_window_exact_match_rate'], 1.)
        self.assertEqual(summary['baselines']['always_no_trade']['trade_window_exact_match_rate'], 0.)
        self.assertEqual(summary['by_actor_id']['a']['trade_f1'], 1.)

    def test_resume_checks_gold_prompt_and_frozen_score(self):
        source = record()
        gold = json.loads(source['messages'][-1]['content'])
        prediction = json.dumps(gold)
        row = {'row_id': source['row_id'], 'gold': gold, 'prediction': prediction,
            'score': details.score_prediction(gold, prediction, TOLERANCES, True),
            'terminated_with_eos': True,
            'prompt_sha256': hashlib.sha256(details.dump(source['messages'][:-1]).encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            path.write_text(details.dump(row) + '\n{"row_id":')
            self.assertEqual(details.recover_predictions(path, [source], True, TOLERANCES), [row])
            row['score']['joint_correct'] = False
            path.write_text(details.dump(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'frozen scorer'):
                details.recover_predictions(path, [source], True, TOLERANCES)

    def test_cli_has_no_test_time_tolerance_override(self):
        required = ['--dataset-dir', '/data', '--run-dir', '/run', '--model', '/model', '--out', '/output']
        self.assertEqual(details.parse_args(required).split, 'test')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            details.parse_args(required + ['--price-delta', '0.4'])


if __name__ == '__main__':
    unittest.main()
