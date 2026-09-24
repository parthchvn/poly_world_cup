"""Recovery witnesses and refusal gates, with a tiny isolated source fixture."""
from argparse import Namespace
import contextlib
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from poly_world_cup.actor_sequences import SequencePolicy
from poly_world_cup.sft import _JSONLines, _json, _sha
from scripts.reconstruct_actor_sequence_shard import canonical_sha, read_prefix, reconstruct


class SequenceShardRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dataset = self.root / "dataset"
        self.relative = "conditional_trades/train/part-00002.jsonl.gz"
        self.rows = [{"sequence_id": f"sequence-{number}", "actor_id": "wallet-a",
                      "profile": "conditional_trades", "split": "train", "chunk_index": number,
                      "payload": hashlib.sha256(str(number).encode()).hexdigest()}
                     for number in range(6)]
        for number in range(1, 4):
            self.save_rows(self.dataset / "conditional_trades/train" / f"part-{number:05d}.jsonl.gz",
                           self.rows[(number-1)*2:number*2])
        self.target = self.dataset / self.relative
        self.original_good = self.target.read_bytes()
        self.target.write_bytes(self.original_good[:-8])  # Missing footer, all content survived.
        self.entries = [{**{key: row[key] for key in ("sequence_id", "actor_id", "profile", "split", "chunk_index")},
            "path": self.relative, "line": number, "sha256": canonical_sha(row)}
            for number, row in enumerate(self.rows[2:4], 1)]
        self.index = self.dataset / "actor_index/part-00001.jsonl.gz"
        self.save_rows(self.index, self.entries)
        (self.dataset / "policy.json").write_text(_json(asdict(SequencePolicy())))
        (self.dataset / "split_policy.json").write_text("{}")
        self.source = self.root / "source.sqlite"
        db = sqlite3.connect(self.source)
        db.executescript("""CREATE TABLE condition_coverage(condition_id TEXT);
            INSERT INTO condition_coverage VALUES('condition-a');
            CREATE TABLE trades(wallet TEXT,condition_id TEXT,query_us INTEGER,
                observation_id TEXT,trade_row_id INTEGER);
            INSERT INTO trades VALUES('wallet-a','condition-a',1,'observation-a',1);
            CREATE TABLE wallet_market_counts(wallet TEXT,condition_id TEXT,observation_count INTEGER);
            INSERT INTO wallet_market_counts VALUES('wallet-a','condition-a',1);""")
        db.commit(); db.close()
        self.recovery_report = self.root / "recovery.json"
        self.recovery_report.write_text(json.dumps({"source_database_sha256": _sha(self.source)}))
        self.args = Namespace(dataset=self.dataset, source=self.source, output_root=self.root / "private",
            relative_shard=self.relative, expected_rows=2, closed_index_through=1,
            recovery_report=self.recovery_report, evidence=self.root, tokenizer=self.root,
            chat_template=self.root)

    def save_rows(self, path, rows):
        if path.exists():
            path.unlink()
        writer = _JSONLines(path)
        for row in rows:
            writer.write(row)
        writer.close()

    def run_reconstruction(self, rebuilt=None):
        # These tests isolate recovery evidence, not the separately tested model
        # conversation builder; the real source and gzip I/O remain in use.
        with patch("scripts.reconstruct_actor_sequence_shard.ContextCatalog", return_value=SimpleNamespace()), \
             patch("scripts.reconstruct_actor_sequence_shard.load_reference_tokenizer", return_value=object()), \
             patch("scripts.reconstruct_actor_sequence_shard.TokenBudget", return_value=object()), \
             patch("scripts.reconstruct_actor_sequence_shard.actor_rows", return_value=rebuilt or self.rows), \
             contextlib.redirect_stdout(io.StringIO()):
            return reconstruct(self.args)

    def test_private_replacement_matches_all_witnesses_and_original_compressed_prefix(self):
        damaged = self.target.read_bytes()
        report = self.run_reconstruction()
        prepared = Path(report["prepared"])
        self.assertEqual(prepared.read_bytes(), self.original_good)
        self.assertTrue(report["original_compressed_prefix_matches"])
        self.assertEqual(report["original_index_witness_count"], 2)
        self.assertEqual(report["rows_with_direct_original_witness"], 2)
        self.assertEqual(report["rows_without_direct_original_witness"], [])
        self.assertEqual(report["boundary_rows_compared"], 2)
        self.assertEqual(self.target.read_bytes(), damaged)

    def test_source_hash_mismatch_refuses_reconstruction(self):
        self.recovery_report.write_text(json.dumps({"source_database_sha256": "0"*64}))
        with self.assertRaisesRegex(ValueError, "source SHA differs"):
            self.run_reconstruction()
        self.assertFalse(list(self.args.output_root.rglob("replacement/*.gz")))

    def test_changed_neighbor_refuses_gap_reconstruction(self):
        changed = copy.deepcopy(self.rows)
        changed[1]["payload"] = "different neighboring source result"
        with self.assertRaisesRegex(ValueError, "neighboring boundary row changed"):
            self.run_reconstruction(changed)
        self.assertFalse(list(self.args.output_root.rglob("replacement/*.gz")))

    def test_changed_original_index_hash_refuses_reconstruction(self):
        self.entries[0]["sha256"] = "0"*64
        self.save_rows(self.index, self.entries)
        with self.assertRaisesRegex(ValueError, "data/index witnesses disagree"):
            self.run_reconstruction()

    def test_missing_index_is_reported_without_fabricating_original_witnesses(self):
        self.save_rows(self.index, [])
        report = self.run_reconstruction()
        self.assertEqual(report["original_index_witness_count"], 0)
        self.assertEqual(report["surviving_original_rows_compared"], 2)
        self.assertEqual(report["rows_with_direct_original_witness"], 2)

    def test_truncated_index_footer_keeps_every_complete_original_hash(self):
        self.index.write_bytes(self.index.read_bytes()[:-8])
        report = self.run_reconstruction()
        self.assertEqual(report["original_index_witness_count"], 2)
        self.assertEqual(report["actor_index_scans"][0]["complete_rows"], 2)
        self.assertIn("EOFError", report["actor_index_scans"][0]["error"])

    def test_nonmatching_original_compressed_prefix_never_gets_a_success_report(self):
        altered = bytearray(self.target.read_bytes())
        altered[4] = 1  # Valid gzip MTIME byte, inconsistent with the original deterministic writer.
        self.target.write_bytes(altered)
        with self.assertRaisesRegex(ValueError, "differs from the original damaged compressed prefix"):
            self.run_reconstruction()
        self.assertFalse(list(self.args.output_root.rglob("report.json")))

    def test_incorrect_gap_size_refuses_reconstruction(self):
        changed = self.rows[:3] + self.rows[4:]
        with self.assertRaisesRegex(ValueError, "Boundary interval has 1 rows, expected 2"):
            self.run_reconstruction(changed)

    def test_partial_decompressed_final_row_is_preserved_in_prefix_evidence(self):
        path = self.root / "partial.jsonl.gz"
        rows = [{"number": i, "text": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(100)]
        self.save_rows(path, rows)
        compressed = path.read_bytes()
        path.write_bytes(compressed[:len(compressed)//2])
        decoder = zlib.decompressobj(31)
        body = decoder.decompress(path.read_bytes())
        retained, info = read_prefix(path, allow_truncated=True)
        self.assertEqual(retained, rows[:len(retained)])
        self.assertGreater(len(retained), 0)
        self.assertLess(len(retained), len(rows))
        self.assertEqual(info["decompressed_prefix_bytes"], len(body))
        self.assertEqual(info["decompressed_prefix_sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(info["partial_final_row_bytes"], len(body.rsplit(b"\n", 1)[-1]))
        with self.assertRaises(EOFError):
            read_prefix(path)


if __name__ == "__main__":
    unittest.main()
