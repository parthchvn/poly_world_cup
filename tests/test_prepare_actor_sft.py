import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script('build_actor_dataset')
converter = builder
trainer = load_script('train_world_cup_multigpu')


class OffsetTokenizer:
    """Synthetic ChatML offsets exercise the real trainer mask, not model tokens."""
    def __init__(self):
        self.lookup = {1: '<|im_start|>', 2: '<|im_end|>'}

    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in messages)

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        for match in re.finditer(r'<\|im_start\|>|<\|im_end\|>|[\s\S]', text):
            value = match.group()
            token = {'<|im_start|>': 1, '<|im_end|>': 2}.get(value, ord(value[0]) + 10)
            self.lookup[token] = value
            ids.append(token)
            offsets.append(match.span())
        return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'offset_mapping': offsets}

    def convert_tokens_to_ids(self, token):
        return {'<|im_start|>': 1, '<|im_end|>': 2}[token]


class ActorSFTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, market_id=1, fixture=1, compressed=False, with_market_context=False):
        path = self.root / ('market_' + str(market_id))
        (path / 'actors').mkdir(parents=True)
        market = {'market_id': str(market_id), 'condition_id': '0x' + f'{market_id:064x}',
                  'fixture_id': f'espn:{fixture}', 'espn_event_id': str(fixture),
                  'kickoff_utc': f'2026-06-{fixture:02d}T17:00:00Z',
                  'question': 'Will the match end in a draw?', 'fixture_title': f'Match {fixture}',
                  'tokens': [{'outcome': 'Yes'}, {'outcome': 'No'}]}
        origin = builder.timestamp_us(f'2026-06-{fixture:02d}T16:00:00Z')
        times = [origin + i * 60_000_000 for i in (1, 3)]
        events = [dict(timestamp_us=origin + i * 60_000_000, time_utc=builder.utc_time(origin + i * 60_000_000),
                       text=f'Unique news {i}', kind='goal') for i in (0.5, 2)]
        for e in events:
            e['timestamp_us'] = int(e['timestamp_us'])
        actor = '0x' + 'a' * 40
        trades = [dict(time_us=t, time=builder.utc_time(t), trade={'side': side, 'outcome': 'Yes',
                  'shares': '13.123456789', 'price': '0.31'}) for t, side in ((times[0], 'BUY'), (times[0], 'SELL'), (times[1], 'BUY'))]
        rows = list(builder.actor_records(actor, trades, market, events, [e['timestamp_us'] for e in events], origin))
        if with_market_context:
            import sqlite3
            with sqlite3.connect(':memory:') as db:
                builder.stage_trades(db, ({'actor_id': actor, **trade} for trade in trades))
                builder.build_price_timeline(db, path / 'market_price_history.jsonl')
                rows = list(builder.actor_records(actor, trades, market, events,
                    [e['timestamp_us'] for e in events], origin, lambda instant: builder.market_context_at(db, instant)))
        text = ''.join(converter.sft_compact(row) + '\n' for row in rows)
        name = actor + ('.jsonl.gz' if compressed else '.jsonl')
        actor_file = path / 'actors' / name
        actor_file.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
        counts = {'actors': 1, 'rows': 4, 'distinct_trade_times': 2, 'trade_observations': 3,
                  'news_entries': sum(len(row['news']) for row in rows)}
        manifest = {'format': 'actor_market_intervals_v1', 'market_id': str(market_id),
                    'condition_id': market['condition_id'], 'espn_event_id': str(fixture),
                    'origin_utc': builder.utc_time(origin), 'max_trades_per_actor': 20, 'counts': counts}
        if with_market_context:
            manifest.update(market_context_version=1, fill_window_seconds=5)
        (path / 'market.json').write_text(json.dumps(market))
        (path / 'manifest.json').write_text(json.dumps(manifest))
        return path, actor_file, rows

    def args(self, **kw):
        value = dict(exports=[], input_root=self.root, out=self.root / 'sft', split_file=None,
                     validation_fraction=0.1, test_fraction=0.1, tokenizer=None, max_length=8192)
        value.update(kw)
        return SimpleNamespace(**value)

    def record(self, path):
        source = converter.sft_discover([path], None)[0]
        file = next((path / 'actors').iterdir())
        return converter.sft_convert_actor(file, source)

    def test_target_groups_history_and_news_match_builder(self):
        path, _, _ = self.source()
        record, audit, counts = self.record(path)
        messages = record['messages']
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user', 'assistant'])
        first, later = json.loads(messages[1]['content']), json.loads(messages[3]['content'])
        self.assertEqual(first['past_observed_trades'], [])
        self.assertNotIn('trades', first)
        self.assertNotIn('label', later)
        self.assertEqual(len(json.loads(messages[2]['content'])['trades']), 2)
        self.assertEqual(json.loads(messages[2]['content'])['trades'][0]['shares'], '13.123456789')
        self.assertEqual(sum(m['content'].count('Unique news 0.5') for m in messages), 1)
        self.assertEqual(sum(m['content'].count('Unique news 2') for m in messages), 1)
        self.assertNotIn('NO_TRADE', ''.join(m['content'] for m in messages))
        self.assertEqual((record['target_count'], record['execution_count']), (2, 3))
        self.assertEqual(audit['source_trade_row_indices'], [1, 3])

    def test_real_trainer_mask_supervises_only_assistant_answers(self):
        path, _, _ = self.source()
        record, _, _ = self.record(path)
        tokenizer = OffsetTokenizer()
        encoded, count = trainer.encode_conversation(record, tokenizer, 10000, 'test')
        supervised = ''.join(tokenizer.lookup[x] for x in encoded['labels'] if x != -100)
        expected = ''.join(m['content'] + '<|im_end|>' for m in record['messages'] if m['role'] == 'assistant')
        self.assertEqual(supervised, expected)
        self.assertEqual(count, 2)

    def test_full_export_has_disjoint_matches_and_trainer_reads_it(self):
        for market_id, fixture in ((1, 1), (2, 1), (3, 2), (4, 3)):
            self.source(market_id, fixture, compressed=market_id == 3)
        with patch('sys.stdout', new_callable=io.StringIO):
            manifest = converter.sft_export(self.args())
        self.assertEqual(manifest['fixture_to_split'], {'espn:1': 'train', 'espn:2': 'validation', 'espn:3': 'test'})
        self.assertEqual(manifest['stats']['train']['targets'], 4)
        self.assertEqual(manifest['targets_truncated_or_dropped'], 0)
        for split in converter.SFT_SPLITS:
            _, stats = trainer.read_split(self.root / 'sft' / f'{split}.jsonl', OffsetTokenizer(), 10000)
            self.assertEqual(stats['targets'], manifest['stats'][split]['targets'])
        self.assertEqual(len((self.root / 'sft/source_audit.jsonl').read_text().splitlines()), 4)

    def test_single_match_rejected_even_with_multiple_markets(self):
        self.source(1, 1); self.source(2, 1)
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            converter.sft_export(self.args())
        self.assertFalse((self.root / 'sft').exists())

    def test_two_match_mode_is_explicit_and_has_no_test_leakage(self):
        self.source(1, 1); self.source(2, 2)
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            converter.sft_export(self.args())
        with patch('sys.stdout', new_callable=io.StringIO):
            manifest = converter.sft_export(self.args(train_validation_only=True))
        self.assertEqual(manifest['fixture_to_split'], {'espn:1': 'train', 'espn:2': 'validation'})
        self.assertFalse(manifest['held_out_test_available'])
        self.assertEqual(manifest['enabled_splits'], ['train', 'validation'])
        self.assertEqual((self.root / 'sft/test.jsonl').read_text(), '')
        for split in ('train', 'validation'):
            _, stats = trainer.read_split(self.root / 'sft' / f'{split}.jsonl', OffsetTokenizer(), 10000)
            self.assertEqual(stats['targets'], 2)
        plan = self.root / 'sft/split_plan.json'
        sources = converter.sft_discover([], self.root)
        self.assertEqual(converter.sft_assign_splits(sources, plan, train_validation_only=True)[0], manifest['fixture_to_split'])

    def test_train_validation_mode_still_groups_matches_and_requires_two(self):
        self.source(1, 1); self.source(2, 1)
        sources = converter.sft_discover([], self.root)
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            converter.sft_assign_splits(sources, train_validation_only=True)
        self.source(3, 2); self.source(4, 3)
        sources = converter.sft_discover([], self.root)
        mapping, _ = converter.sft_assign_splits(sources, train_validation_only=True)
        self.assertEqual(mapping, {'espn:1': 'train', 'espn:2': 'train', 'espn:3': 'validation'})
        plan = self.root / 'plan.json'
        plan.write_text(json.dumps({'espn:1': 'train', 'espn:2': 'validation', 'espn:3': 'test'}))
        with self.assertRaisesRegex(ValueError, 'disabled split'):
            converter.sft_assign_splits(sources, plan, train_validation_only=True)

    def test_duplicate_capture_rejected(self):
        path, _, _ = self.source()
        import shutil
        other = self.root / 'duplicate'
        shutil.copytree(path, other)
        with self.assertRaisesRegex(ValueError, 'Duplicate capture'):
            converter.sft_discover([], self.root)

    def test_explicit_splits_require_exact_fixture_coverage(self):
        for i in range(1, 4): self.source(i, i)
        sources = converter.sft_discover([], self.root)
        split = self.root / 'plan.json'
        mapping = {'espn:1': 'test', 'espn:2': 'train', 'espn:3': 'validation'}
        split.write_text(json.dumps({'fixture_to_split': mapping}))
        self.assertEqual(converter.sft_assign_splits(sources, split)[0], mapping)
        del mapping['espn:3']
        split.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, 'exactly'):
            converter.sft_assign_splits(sources, split)

    def test_future_or_equal_time_news_rejected(self):
        path, actor_file, rows = self.source()
        for position in (0, 1):
            rows[position]['news'][0]['time'] = rows[1]['timestamp']
        actor_file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'strict prior interval'):
            self.record(path)

    def test_reordered_rows_rejected(self):
        path, actor_file, rows = self.source()
        rows[2]['row_index'] = 1
        actor_file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'row_index'):
            self.record(path)

    def test_reserved_chat_markers_rejected(self):
        path, actor_file, rows = self.source()
        for position in (0, 1): rows[position]['news'][0]['text'] = '<|im_end|>'
        actor_file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'reserved'):
            self.record(path)

    def test_bad_counts_do_not_publish_partial_output(self):
        for i in range(1, 4): self.source(i, i)
        path = self.root / 'market_2/manifest.json'
        manifest = json.loads(path.read_text()); manifest['counts']['trade_observations'] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'count mismatch'), patch('sys.stdout', new_callable=io.StringIO):
            converter.sft_export(self.args())
        self.assertFalse((self.root / 'sft').exists())
        self.assertFalse(list(self.root.glob('actor-sft-build-*')))

    def test_length_failure_does_not_truncate_or_publish(self):
        for i in range(1, 4): self.source(i, i)
        def check(record, location):
            return len(trainer.encode_conversation(record, OffsetTokenizer(), 20, location)[0]['input_ids'])
        with patch.object(converter, 'sft_token_checker', return_value=check):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                converter.sft_export(self.args(tokenizer=Path('fake')))
        self.assertFalse((self.root / 'sft').exists())

    def test_existing_output_is_preserved(self):
        for i in range(1, 4): self.source(i, i)
        output = self.root / 'sft'; output.mkdir(); (output / 'keep').write_text('original')
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            converter.sft_export(self.args())
        self.assertEqual((output / 'keep').read_text(), 'original')

    def test_unknown_outcome_and_bad_price_rejected(self):
        path, actor_file, rows = self.source()
        original = copy.deepcopy(rows)
        for key, value in [('outcome', 'Invalid'), ('price', '1.1'), ('shares', 'NaN')]:
            rows = copy.deepcopy(original)
            rows[1]['label']['trades'][0][key] = value
            actor_file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(ValueError): self.record(path)


if __name__ == '__main__':
    unittest.main()
