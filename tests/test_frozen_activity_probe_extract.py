"""Extraction safety checks without loading a model or touching a GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_frozen_activity_probe as probe
from tests.test_interval_trade_details_eval import record, GenerationTokenizer


class ExtractionTests(unittest.TestCase):
    def test_targets_cannot_change_context_vectors_input(self):
        positive, negative = record('TRADE'), record('NO_TRADE')
        negative['messages'][:-1] = copy.deepcopy(positive['messages'][:-1])
        tokenizer = GenerationTokenizer()
        self.assertEqual(probe.encode_generation_prompt(positive, tokenizer, 8000, 2048),
                         probe.encode_generation_prompt(negative, tokenizer, 8000, 2048))
        self.assertEqual(probe.identifiers([positive, negative])['labels'].tolist(), [1, 0])

    def test_shards_cover_every_row_once_including_uneven_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.jsonl'
            path.write_text(''.join(json.dumps({'row_id': i}) + '\n' for i in range(11)))
            shards = [probe.shard_records(path, i, 4) for i in range(4)]
            ids = [r['row_id'] for rows in shards for r in rows]
            self.assertEqual(sorted(ids), list(range(11)))
            self.assertEqual(len(ids), len(set(ids)))

    def test_resume_rejects_wrong_rows_labels_and_nonfinite_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chunk.npz'
            rows = [record('TRADE'), record('NO_TRADE')]
            rows[1]['row_id'] = 'another-row'
            arrays = dict(X=np.arange(8, dtype=np.float16).reshape(2, 4), **probe.identifiers(rows))
            probe.save_npz(path, **arrays)
            self.assertTrue(np.array_equal(probe.validate_chunk(path, rows, 4), arrays['X']))
            self.assertFalse(path.with_suffix('.npz.partial').exists())
            with self.assertRaisesRegex(ValueError, 'row_ids mismatch'):
                probe.validate_chunk(path, rows[::-1], 4)
            for key, value in [('labels', np.array([0, 0])), ('X', np.full((2, 4), np.nan))]:
                probe.save_npz(path, **dict(arrays, **{key: value}))
                with self.assertRaises(ValueError):
                    probe.validate_chunk(path, rows, 4)

    def test_busy_selected_gpu_stops_launcher_but_gpu_zero_is_ignored(self):
        status = SimpleNamespace(stdout='0, 7000\n1, 0\n2, 18000\n3, 0\n4, 0\n')
        with patch.object(probe.subprocess, 'run', return_value=status):
            probe.ensure_gpus_free(['1', '3', '4'])
            with self.assertRaisesRegex(ValueError, 'memory allocated'):
                probe.ensure_gpus_free(['1', '2', '3', '4'])

    def test_supervisor_reserves_gpu_zero_before_any_mutation(self):
        with self.assertRaisesRegex(ValueError, 'GPU 0 is reserved'):
            probe.supervise(SimpleNamespace(gpu_ids=['0', '1']))


if __name__ == '__main__':
    unittest.main()
