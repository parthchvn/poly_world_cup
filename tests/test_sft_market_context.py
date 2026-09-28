"""Market context stays prior to targets; derived execution audits never enter SFT."""
import copy
import io
import json
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures

builder = fixtures.builder


class SFTMarketContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def source(self, market_id=1, fixture=1):
        path, actor_file, rows = self.fixture.source(market_id, fixture)
        manifest_path = path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(market_context_version=1, fill_window_seconds=5)
        manifest_path.write_text(json.dumps(manifest))
        for index in range(0, len(rows), 2):
            when = rows[index + 1]['timestamp']
            instant = builder.timestamp_us(when)
            context = {
                'version': 1, 'as_of': when, 'price_semantics': 'prior_execution_vwap_not_quote',
                'yes': {'price': '0.4', 'implied_probability': '0.4',
                        'observed_at': builder.utc_time(instant - 5_000_000), 'age_seconds': '5',
                        'source': 'captured_execution_timestamp_vwap', 'observation_count': 3},
                'no': {'price': '0.62', 'implied_probability': '0.62',
                       'observed_at': builder.utc_time(instant - 8_000_000), 'age_seconds': '8',
                       'source': 'captured_execution_timestamp_vwap', 'observation_count': 1},
                'winning_payout_per_share': '1', 'losing_payout_per_share': '0'}
            for offset in (0, 1):
                row = rows[index + offset]
                row['market_context'] = copy.deepcopy(context)
                row['execution_info'] = builder.execution_evidence(instant, bool(offset), 5)
                row['payoff_analysis'] = [builder.execution_payoff(x) for x in row['label']['trades']] if offset else []
        self.write(actor_file, rows)
        return path, actor_file, rows

    @staticmethod
    def write(actor_file, rows):
        actor_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_only_prior_market_context_enters_each_user_turn(self):
        path, _, rows = self.source()
        record, _, counts = self.fixture.record(path)
        for i, user in enumerate(record['messages'][1::2]):
            payload = json.loads(user['content'])
            raw_context = rows[2 * i + 1]['market_context']
            expected_context = {
                outcome: {key: raw_context[outcome][key] for key in ('price', 'age_seconds')}
                for outcome in ('yes', 'no')}
            self.assertEqual(payload['market_context'], expected_context)
            self.assertEqual(payload['query_time'], raw_context['as_of'])
            for outcome in ('yes', 'no'):
                self.assertEqual(set(payload['market_context'][outcome]), {'price', 'age_seconds'})
                self.assertNotIn('source', payload['market_context'][outcome])
                self.assertNotIn('observed_at', payload['market_context'][outcome])
                self.assertNotIn('observation_count', payload['market_context'][outcome])
                self.assertEqual(raw_context[outcome]['source'], 'captured_execution_timestamp_vwap')
                self.assertIn('observed_at', raw_context[outcome])
            self.assertNotIn('execution_info', payload)
            self.assertNotIn('payoff_analysis', payload)
            self.assertNotIn('label', payload)
            self.assertEqual(payload['market_context']['yes']['price'], '0.4')
            self.assertEqual(payload['market_context']['no']['price'], '0.62')
        for assistant in record['messages'][2::2]:
            self.assertEqual(set(json.loads(assistant['content'])), {'action', 'trades'})
        self.assertEqual(counts['trade_observations'], 3)

    def test_unknown_prices_are_preserved_as_null(self):
        path, file, rows = self.source()
        for row in rows:
            row['market_context']['yes'] = None
            row['market_context']['no'] = None
        self.write(file, rows)
        record, _, _ = self.fixture.record(path)
        for user in record['messages'][1::2]:
            context = json.loads(user['content'])['market_context']
            self.assertIsNone(context['yes'])
            self.assertIsNone(context['no'])

    def test_current_future_and_unverifiable_snapshot_values_rejected(self):
        path, file, original = self.source()
        cases = [
            ('observed_at', original[1]['timestamp']),
            ('observed_at', builder.utc_time(builder.timestamp_us(original[1]['timestamp']) + 1_000_000)),
            ('age_seconds', '6'), ('age_seconds', 5), ('price', 'NaN'), ('price', '1.1'),
            ('implied_probability', '0.6'), ('observation_count', True), ('observation_count', 0),
            ('source', 'future_quote'), ('extra_future_information', 'secret')]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                rows = copy.deepcopy(original)
                for row in rows[:2]:
                    row['market_context']['yes'][name] = value
                self.write(file, rows)
                with self.assertRaises(ValueError):
                    self.fixture.record(path)

    def test_interval_and_trade_context_must_agree(self):
        path, file, rows = self.source()
        rows[0]['market_context']['yes']['price'] = '0.5'
        self.write(file, rows)
        with self.assertRaisesRegex(ValueError, 'adjacent market_context'):
            self.fixture.record(path)

    def test_submission_fill_and_profit_claims_cannot_be_invented(self):
        path, file, original = self.source()
        for key, value in [('order_submitted_at', original[1]['timestamp']),
                           ('order_type', 'market'), ('order_fully_filled', True),
                           ('filled_within_seconds_of_submission', True), ('fill_window_seconds', 5.0)]:
            with self.subTest(key=key):
                rows = copy.deepcopy(original)
                rows[1]['execution_info'][key] = value
                self.write(file, rows)
                with self.assertRaisesRegex(ValueError, 'execution_info'):
                    self.fixture.record(path)
        rows = copy.deepcopy(original)
        rows[1]['payoff_analysis'][0]['invented_expected_profit'] = '100'
        self.write(file, rows)
        with self.assertRaisesRegex(ValueError, 'payoff_analysis'):
            self.fixture.record(path)
        rows = copy.deepcopy(original)
        rows[0]['payoff_analysis'] = rows[1]['payoff_analysis']
        self.write(file, rows)
        with self.assertRaisesRegex(ValueError, 'payoff_analysis'):
            self.fixture.record(path)

    def test_new_manifest_requires_all_row_features(self):
        path, file, original = self.source()
        for key in ('market_context', 'execution_info', 'payoff_analysis'):
            rows = copy.deepcopy(original)
            del rows[1][key]
            self.write(file, rows)
            with self.assertRaisesRegex(ValueError, 'missing'):
                self.fixture.record(path)

    def test_legacy_exports_work_but_cannot_hide_new_fields(self):
        path, file, rows = self.fixture.source()
        record, _, _ = self.fixture.record(path)
        self.assertNotIn('market_context', json.loads(record['messages'][1]['content']))
        rows[1]['market_context'] = {}
        self.write(file, rows)
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            self.fixture.record(path)

    def test_mixed_and_unsupported_versions_are_rejected(self):
        self.source(1, 1)
        path, _, _ = self.fixture.source(2, 2)
        with self.assertRaisesRegex(ValueError, 'Mixed market-context versions'):
            builder.sft_discover([], self.root)
        manifest_path = path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for version in (None, 0, 3, True, '1'):
            manifest['market_context_version'] = version
            manifest_path.write_text(json.dumps(manifest))
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'Unsupported market_context_version'):
                builder.sft_discover([path], None)

    def test_full_export_reports_feature_version_and_keeps_assistant_only_loss(self):
        for index in (1, 2, 3):
            self.source(index, index)
        with patch('sys.stdout', new_callable=io.StringIO):
            metadata = builder.sft_export(self.fixture.args())
        self.assertEqual(metadata['market_context_version'], 1)
        tokenizer = fixtures.OffsetTokenizer()
        record = json.loads((self.root / 'sft/train.jsonl').read_text().splitlines()[0])
        encoded, _ = fixtures.trainer.encode_conversation(record, tokenizer, 20000, 'context-test')
        supervised = ''.join(tokenizer.lookup[x] for x in encoded['labels'] if x != -100)
        self.assertNotIn('market_context', supervised)
        self.assertNotIn('payoff_analysis', supervised)
        self.assertNotIn('execution_info', supervised)
        expected = ''.join(x['content'] + '<|im_end|>' for x in record['messages'] if x['role'] == 'assistant')
        self.assertEqual(supervised, expected)

    def test_all_legacy_export_warns_and_records_version_zero(self):
        for index in (1, 2, 3):
            self.fixture.source(index, index)
        with patch('sys.stdout', new_callable=io.StringIO), patch('sys.stderr', new_callable=io.StringIO) as warning:
            metadata = builder.sft_export(self.fixture.args())
        self.assertEqual(metadata['market_context_version'], 0)
        self.assertIn('legacy', warning.getvalue())


if __name__ == '__main__':
    unittest.main()
