"""Messages-only export must preserve conversation attention and validation gates."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_actor_messages import export_messages

PROFILES = ("conditional_trades", "scheduled_windows")
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportActorMessagesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "release"
        self.dataset.mkdir()
        self.validation_path = self.root / "validation.json"
        self.output = self.root / "training.jsonl"
        self.rows = []
        self.shards = []
        counts = {profile: {split: {"conversations": 0, "tokens": 0,
                  "target_turns": 0, "long_context_sequences": 0}
                  for split in SPLITS} for profile in PROFILES}
        for profile in PROFILES:
            for split in SPLITS:
                (self.dataset / profile / split).mkdir(parents=True)
        for index, token_count in enumerate((321, 9001), start=1):
            messages = [{"role": "system", "content": "Predict captured observations.\nFollow chronological context."}]
            for turn in range(3):
                messages.extend([
                    {"role": "user", "content": json.dumps({"t": turn, "news": [f"Café 日本 {index}:{turn}"],
                                                               "literal": "assistant: NO_TRADE\n\n"}, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps({"side": "BUY", "outcome": "Yes",
                                                                    "shares": str(turn + 1), "price": "0.25"})},
                ])
            row = {"profile": "conditional_trades", "split": "train", "sequence_id": f"seq-{index}",
                   "actor_id": f"actor-{index}", "messages": messages, "token_count": token_count,
                   "target_turns": 3, "requires_long_context": token_count > 8192,
                   "audit": {"target_answer": "DO_NOT_TRAIN_ON_OUTER_AUDIT_TARGET", "future_news": "FUTURE_LEAK"},
                   "target_observation_ids": ["SECRET_TARGET_ID"],
                   "source_provenance": {"raw_target": "DO_NOT_PROMPT_WITH_SOURCE_METADATA"}}
            self.rows.append(row)
            shard = self.dataset / f"conditional_trades/train/part-{index:05d}.jsonl.gz"
            with gzip.open(shard, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.shards.append(shard)
        counts["conditional_trades"]["train"] = {"conversations": 2, "tokens": 9322,
                                                      "target_turns": 6, "long_context_sequences": 1}
        audit = self.dataset / "audit.json"
        audit.write_text(json.dumps({"target_answer": "OUTER_AUDIT_IS_NOT_MODEL_INPUT"}) + "\n")
        self.manifest = {"schema_version": 3, "status": "exported", "sample": False,
                         "profiles": counts, "source": {"sha256": "a" * 64},
                         "tokenizer": {"files": {}, "chat_template_sha256": "b" * 64},
                         "artifacts": {p.relative_to(self.dataset).as_posix():
                             {"bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in [*self.shards, audit]}}
        token_files = [{"path": p.relative_to(self.dataset).as_posix(), "bytes": p.stat().st_size,
                        "sha256": sha256(p), "profile": "conditional_trades", "split": "train",
                        "conversations": 1, "tokens": row["token_count"]}
                       for p, row in zip(self.shards, self.rows)]
        self.validation = {"schema_version": 1, "status": "passed", "source_sha256": "a" * 64,
                           "artifact_count": len(self.manifest["artifacts"]), "fixtures": 104, "contracts": 312,
                           "profiles": copy.deepcopy(counts), "actual_token_counts_recomputed": True,
                           "token_recount": {"schema_version": 1, "status": "passed", "all_rows_recomputed": True,
                               "method": "full_apply_chat_template_no_truncation", "profiles": copy.deepcopy(counts),
                               "conversations": 2, "tokens": 9322, "files": token_files,
                               "tokenizer_files": {}, "chat_template_sha256": "b" * 64}}
        self.write_documents()

    def write_documents(self):
        manifest_path = self.dataset / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, sort_keys=True) + "\n")
        digest = sha256(manifest_path)
        self.validation["manifest_sha256"] = digest
        self.validation["token_recount"]["manifest_sha256"] = digest
        self.write_validation()

    def write_validation(self):
        self.validation_path.write_text(json.dumps(self.validation, sort_keys=True) + "\n")

    def export(self, **overrides):
        arguments = {"dataset": self.dataset, "validation": self.validation_path,
                     "profile": "conditional_trades", "split": "train", "output": self.output}
        arguments.update(overrides)
        return export_messages(**arguments)

    def assert_refused(self, **overrides):
        with self.assertRaises((ValueError, FileNotFoundError, FileExistsError)):
            self.export(**overrides)
        self.assertFalse(self.output.exists(), "Failed validation must not publish a training file")

    def test_exact_messages_only_preserves_every_turn_and_oversized_conversation(self):
        report = self.export()
        self.assertIsInstance(report, dict)
        written = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(written, [{"messages": row["messages"]} for row in self.rows])
        self.assertEqual([sum(m["role"] == "assistant" for m in row["messages"]) for row in written], [3, 3])
        self.assertTrue(self.rows[1]["requires_long_context"])
        self.assertNotIn("DO_NOT_TRAIN_ON_OUTER_AUDIT_TARGET", self.output.read_text())
        self.assertNotIn("SECRET_TARGET_ID", self.output.read_text())

    def test_missing_validation_is_refused(self):
        self.validation_path.unlink()
        self.assert_refused()

    def test_stale_validation_manifest_digest_is_refused(self):
        self.validation["manifest_sha256"] = "0" * 64
        self.write_validation()
        self.assert_refused()

    def test_stale_token_recount_manifest_digest_is_refused(self):
        self.validation["token_recount"]["manifest_sha256"] = "0" * 64
        self.write_validation()
        self.assert_refused()

    def test_structural_only_report_is_refused(self):
        self.validation["actual_token_counts_recomputed"] = False
        self.validation.pop("token_recount")
        self.write_validation()
        self.assert_refused()

    def test_incomplete_all_row_token_recount_is_refused(self):
        self.validation["token_recount"]["all_rows_recomputed"] = False
        self.write_validation()
        self.assert_refused()

    def test_sample_export_is_refused_even_with_matching_report_digest(self):
        self.manifest["sample"] = True
        self.write_documents()
        self.assert_refused()

    def test_failed_validation_status_is_refused(self):
        self.validation["status"] = "failed"
        self.write_validation()
        self.assert_refused()

    def test_source_digest_mismatch_is_refused(self):
        self.validation["source_sha256"] = "c" * 64
        self.write_validation()
        self.assert_refused()

    def test_selected_shard_checksum_mismatch_is_refused_without_partial_output(self):
        with gzip.open(self.shards[-1], "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({**self.rows[-1], "audit": {"modified": True}}) + "\n")
        self.assert_refused()

    def test_report_profile_counts_mismatch_is_refused(self):
        self.validation["profiles"]["conditional_trades"]["train"]["conversations"] += 1
        self.write_validation()
        self.assert_refused()

    def test_output_inside_release_is_refused(self):
        forbidden = self.dataset / "training.jsonl"
        self.assert_refused(output=forbidden)
        self.assertFalse(forbidden.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.write_text("preserve existing user file\n")
        with self.assertRaises((ValueError, FileExistsError)):
            self.export()
        self.assertEqual(self.output.read_text(), "preserve existing user file\n")

    def test_unknown_profile_and_split_are_refused(self):
        for overrides in ({"profile": "mixed"}, {"split": "everything"}):
            with self.subTest(overrides=overrides):
                self.assert_refused(**overrides)


if __name__ == "__main__":
    unittest.main()
