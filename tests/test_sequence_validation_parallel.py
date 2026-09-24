"""Equivalent source checks with ordered, bounded spawn-worker batches."""
from collections import Counter
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import sqlite3
import unittest
from unittest.mock import patch

from poly_world_cup import sequence_validation as validation
from poly_world_cup.actor_sequences import build_actor_records
from poly_world_cup.sft import _Shards
import test_sequence_validation as fixtures


class ParallelSequenceValidationTests(unittest.TestCase):
    # Reuse actual evidence/source fixtures without inheriting their test methods.
    setUp = fixtures.SequenceValidatorTests.setUp
    trade = fixtures.SequenceValidatorTests.trade
    budget = fixtures.SequenceValidatorTests.budget
    build = fixtures.SequenceValidatorTests.build
    release_fixture = fixtures.SequenceValidatorTests.release_fixture

    def multi_actor_release(self, actors=9):
        """A hash-bound SQLite release that spans more than two actor batches."""
        dataset, source, manifest, rehash = self.release_fixture()
        db = sqlite3.connect(source)
        db.row_factory = sqlite3.Row
        template = [dict(r) for r in db.execute("SELECT * FROM trades ORDER BY query_us")]
        columns = list(template[0])
        db.execute("DELETE FROM trades")
        db.execute("DELETE FROM wallet_market_counts")
        results, pairs = [], []
        for number in range(actors):
            actor = "0x" + format(number + 1, "040x")
            raw = copy.deepcopy(template)
            for index, row in enumerate(raw):
                row.update(wallet=actor, observation_id=f"actor-{number}-{index}",
                           transaction_hash=f"transaction-{number}-{index}",
                           trade_row_id=number * len(raw) + index + 1)
            db.executemany("INSERT INTO trades VALUES(" + ",".join("?" for _ in columns) + ")",
                           ([r[c] for c in columns] for r in raw))
            pairs.append({"wallet": actor, "condition_id": self.first["condition_id"],
                          "observation_count": len(raw)})
            results.append(build_actor_records(actor, raw, self.catalog, self.coverage,
                                               self.split_policy, self.policy, self.budget()))
        pairs.append({"wallet": "z-excluded", "condition_id": self.first["condition_id"],
                      "observation_count": 21})
        db.executemany("INSERT INTO wallet_market_counts VALUES(?,?,?)",
                       (tuple(r.values()) for r in pairs))
        selected = len(template) * actors
        db.execute("UPDATE condition_coverage SET observation_count=?,selected_observation_count=?",
                   (selected + 21, selected))
        coverage = {r["condition_id"]: dict(r) for r in db.execute("SELECT * FROM condition_coverage")}
        db.commit()
        db.close()

        for path in dataset.rglob("*.jsonl.gz"):
            path.unlink()
        for relative, rows in (("observations", (r for result in results for r in result["observations"])),
                               ("source_evidence/pair_counts", iter(pairs))):
            writer = _Shards(dataset / relative, 100)
            for row in rows:
                writer.write(row)
            writer.close()

        profiles = {p: {s: {} for s in validation.SPLITS} for p in validation.PROFILES}
        index_writer = _Shards(dataset / "actor_index", 100)
        lengths = []
        for profile in validation.PROFILES:
            writer = _Shards(dataset / profile / "train", 100)
            counts = Counter()
            for result in results:
                for row in (r for r in result["conversations"] if r["profile"] == profile):
                    location = writer.write(row)
                    index_writer.write({k: row[k] for k in
                                        ("sequence_id", "actor_id", "profile", "split", "chunk_index")} |
                                       {"path": profile + "/train/part-00001.jsonl.gz", "line": location["line"],
                                        "sha256": hashlib.sha256(validation.canonical(row).encode()).hexdigest()})
                    lengths.append(row["token_count"])
                    counts.update(conversations=1, target_turns=len(row["turn_audit"]),
                                  tokens=row["token_count"],
                                  long_context_sequences=int(row["requires_long_context"]))
            writer.close()
            profiles[profile]["train"] = dict(counts)
        index_writer.close()
        fixture_counts = Counter()
        all_counts = Counter(actors=actors)
        for result in results:
            all_counts.update(result["counts"])
            fixture_counts.update(result["fixture_counts"][self.first["fixture_id"]])
        (dataset / "fixture_coverage.json").write_text(
            validation.canonical({self.first["fixture_id"]: dict(fixture_counts)}) + "\n")
        (dataset / "source_coverage.json").write_text(validation.canonical(coverage) + "\n")
        lengths.sort()
        manifest.update(source={"sha256": validation.sha(source), "bytes": source.stat().st_size},
                        selected_observations=selected, counts=dict(all_counts), profiles=profiles,
                        tokens={"total": sum(lengths), "p50": lengths[math.ceil(len(lengths) * .5) - 1],
                                "p95": lengths[math.ceil(len(lengths) * .95) - 1], "max": max(lengths)})
        rehash()
        return dataset, source, manifest, rehash

    def test_real_spawn_batches_equal_serial_report(self):
        dataset, source, _, _ = self.multi_actor_release()
        serial = validation.validate_sequences(dataset, source, self.root, workers=1, progress=lambda _: None)
        parallel = validation.validate_sequences(dataset, source, self.root, workers=2, progress=lambda _: None)
        self.assertEqual(parallel, serial)
        self.assertEqual(parallel["status"], "passed")
        self.assertEqual(parallel["source_reconciliation"]["selected_observations"], 27)
        self.assertGreater(parallel["profiles"]["scheduled_windows"]["train"]["target_turns"], 0)

    def test_parallel_worker_rejects_rehashed_incorrect_label(self):
        dataset, source, _, rehash = self.multi_actor_release()
        path = dataset / "conditional_trades/train/part-00001.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        answer = json.loads(rows[-1]["messages"][2]["content"])
        answer["trades"][0]["shares"] = "11"
        rows[-1]["messages"][2]["content"] = validation.canonical(answer)
        with gzip.open(path, "wt") as stream:
            for row in rows:
                stream.write(validation.canonical(row) + "\n")
        index_path = dataset / "actor_index/part-00001.jsonl.gz"
        with gzip.open(index_path, "rt") as stream:
            index = [json.loads(line) for line in stream]
        for row in index:
            if row["sequence_id"] == rows[-1]["sequence_id"]:
                row["sha256"] = hashlib.sha256(validation.canonical(rows[-1]).encode()).hexdigest()
        with gzip.open(index_path, "wt") as stream:
            for row in index:
                stream.write(validation.canonical(row) + "\n")
        rehash()
        with self.assertRaisesRegex(ValueError, "Answer differs from full scoped source observations"):
            validation.validate_sequences(dataset, source, self.root, workers=2, progress=lambda _: None)

    def test_default_serial_path_does_not_create_process_pool(self):
        dataset, source, _, _ = self.release_fixture()
        with patch.object(validation, "ProcessPoolExecutor", side_effect=AssertionError("Unexpected pool")):
            report = validation.validate_sequences(dataset, source, self.root, progress=lambda _: None)
        self.assertEqual(report["status"], "passed")

    def test_worker_initializer_reuses_validated_catalog_snapshot(self):
        raw = [self.trade("first", 100), self.trade("second", 1000)]
        result = self.build(raw)
        chunks = [r for r in result["conversations"]
                  if r["profile"] == "conditional_trades" and r["split"] == "train"]
        with patch.object(validation, "_WORKER_VALIDATION", None), \
                patch.object(validation, "ContextCatalog", side_effect=AssertionError("Evidence reread")):
            validation._validation_worker_init(self.catalog, asdict(self.policy), self.split_policy,
                                               self.coverage, "conditional_trades", "train")
            self.assertIs(validation._WORKER_VALIDATION[0], self.catalog)
            self.assertEqual(validation._WORKER_VALIDATION[1], self.policy)
            _, counts, _, _ = validation._validate_actor_batch([(self.actor, raw, chunks)])
        self.assertEqual(counts["target_observations"], 2)

    def test_invalid_worker_values_are_rejected(self):
        dataset, source, _, _ = self.release_fixture()
        for workers in (0, -1, 33, True, False, 1.5, "2", None):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "workers|Workers"):
                    validation.validate_sequences(dataset, source, self.root,
                                                  workers=workers, progress=lambda _: None)


class TrackingExecutor:
    """Track submitted but unconsumed futures without threads or timing races."""
    def __init__(self):
        self.outstanding = 0
        self.maximum = 0
        self.submitted = []
        self.consumed = []

    def submit(self, function, value):
        self.outstanding += 1
        self.maximum = max(self.maximum, self.outstanding)
        self.submitted.append(value)
        owner = self

        class Future:
            def result(self):
                owner.outstanding -= 1
                owner.consumed.append(value)
                return function(value)

            def cancel(self):
                return True

        return Future()


class BoundedOrderedResultsTests(unittest.TestCase):
    def test_order_and_outstanding_work_are_bounded(self):
        executor = TrackingExecutor()
        result = validation._bounded_ordered_results(executor, lambda n: 20 - n, iter(range(12)), 3)
        self.assertEqual(next(result), 20)
        self.assertLess(len(executor.submitted), 12)
        self.assertEqual([20] + list(result), list(range(20, 8, -1)))
        self.assertEqual(executor.submitted, list(range(12)))
        self.assertEqual(executor.consumed, list(range(12)))
        self.assertLessEqual(executor.maximum, 3)
        self.assertEqual(executor.outstanding, 0)

    def test_empty_input_submits_nothing(self):
        executor = TrackingExecutor()
        self.assertEqual(list(validation._bounded_ordered_results(executor, lambda n: n, iter(()), 2)), [])
        self.assertEqual(executor.submitted, [])

    def test_worker_error_propagates_without_consuming_entire_source(self):
        executor = TrackingExecutor()

        def fail(value):
            if value == 1:
                raise ValueError("synthetic source label mismatch")
            return value

        result = validation._bounded_ordered_results(executor, fail, iter(range(100)), 2)
        self.assertEqual(next(result), 0)
        with self.assertRaisesRegex(ValueError, "synthetic source label mismatch"):
            next(result)
        self.assertLess(len(executor.submitted), 100)
        self.assertLessEqual(executor.maximum, 2)


if __name__ == "__main__":
    unittest.main()
