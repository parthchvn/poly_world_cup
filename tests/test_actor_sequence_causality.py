"""Adversarial tests for outcome-independent prompts and historical enrollment."""
import copy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from poly_world_cup.actor_sequences import (
    ConversationBuilder, SequencePolicy, build_actor_records, normalize_observations,
)
from poly_world_cup.historical_contracts import build_contract_record
from poly_world_cup.sequence_context import ContextCatalog, _utc
from poly_world_cup.sequence_tokens import TokenBudget
from test_historical_contracts import contract_record_fixture

SECOND = 1_000_000


class CharacterTokenizer:
    """Deterministic test budget, independent of tokenizer downloads."""
    chat_template = "synthetic character accounting"

    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, messages, **kwargs):
        return list("".join(m["role"] + "\n" + m["content"] + "\n" for m in messages))


class ActorSequenceCausalityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "contracts").mkdir()
        (self.root / "news_catalog").mkdir()
        self.first = contract_record_fixture()
        evidence = copy.deepcopy(self.first["evidence"])
        evidence["mapping_calls"][0]["result"] = "0x" + "dd" * 32
        self.second = build_contract_record(evidence, fixture_id="espn:760416")
        with gzip.open(self.root / "contracts/contract_evidence.jsonl.gz", "wt") as stream:
            for contract in (self.first, self.second):
                stream.write(json.dumps(contract) + "\n")
        (self.root / "news_catalog/news_for_sft.jsonl").write_text("")
        self.catalog = ContextCatalog(self.root)
        initialized = self.catalog.contracts[self.first["condition_id"]]["initialized_us"]
        self.base = (initialized // (900 * SECOND) + 1) * 900 * SECOND
        self.policy = SequencePolicy(max_tokens=5000, response_reserve_tokens=256,
            activity_seconds=1800, train_negative_probability=1, evaluation_negative_probability=1)
        self.split_policy = {
            "fixture_splits": {self.first["fixture_id"]: "train", self.second["fixture_id"]: "train"},
            "train_before_utc": _utc(self.base + 10000 * SECOND),
            "validation_before_utc": _utc(self.base + 20000 * SECOND),
        }
        self.coverage = {c["condition_id"]: {"earliest_query_us": self.base,
            "latest_query_us": self.base + 9000 * SECOND} for c in (self.first, self.second)}
        self.actor = "0x" + "12" * 20

    def trade(self, name, seconds, *, contract=None, transaction=None, shares="10"):
        contract = contract or self.first
        return {"trade_row_id": len(name), "observation_id": name, "wallet": self.actor,
                "condition_id": contract["condition_id"], "token_id": contract["yes_token_id"],
                "query_us": self.base + seconds * SECOND,
                "transaction_hash": transaction or "transaction-" + name,
                "side": "BUY", "shares": shares, "price": "0.4"}

    def budget(self, policy=None):
        return TokenBudget(CharacterTokenizer(), (policy or self.policy).max_tokens)

    def query_prefix(self, records, query):
        for row in records:
            for index, audit in enumerate(row["turn_audit"]):
                if audit["query_us"] == query:
                    return row["chunk_index"], row["messages"][:2 + 2 * index]
        self.fail("Missing query in generated conversations")

    def build_two_queries(self, targets, policy):
        history = normalize_observations([self.trade("earlier", 100)], self.catalog, self.split_policy)
        builder = ConversationBuilder(self.actor, "scheduled_windows", "train", history,
                                      self.catalog, policy, self.budget(policy))
        condition = self.first["condition_id"]
        for seconds, answers in ((900, []), (1800, targets)):
            query = self.base + seconds * SECOND
            builder.add(query, query + 900 * SECOND, {condition}, answers)
        builder.flush()
        return builder.records

    def test_future_large_answer_cannot_change_current_prefix_or_create_boundary(self):
        target = normalize_observations([self.trade("future", 1801, shares="1" + "0" * 15000)],
                                        self.catalog, self.split_policy)[0]
        empty = self.build_two_queries([], self.policy)
        huge = self.build_two_queries([target], self.policy)
        query = self.base + 1800 * SECOND
        self.assertEqual(self.query_prefix(empty, query), self.query_prefix(huge, query))
        self.assertEqual(self.query_prefix(huge, query)[0], 0)
        self.assertTrue(huge[0]["requires_long_context"])
        self.assertEqual(len(huge[0]["turn_audit"]), 2)
        self.assertFalse(empty[0]["requires_long_context"])

    def test_fixed_turn_boundary_is_invariant_to_unseen_answer(self):
        policy = replace(self.policy, max_turns=1)
        target = normalize_observations([self.trade("future", 1801, shares="1" + "0" * 15000)],
                                        self.catalog, self.split_policy)[0]
        empty, huge = self.build_two_queries([], policy), self.build_two_queries([target], policy)
        query = self.base + 1800 * SECOND
        self.assertEqual(self.query_prefix(empty, query), self.query_prefix(huge, query))
        chunk, prefix = self.query_prefix(empty, query)
        self.assertEqual(chunk, 1)
        self.assertEqual(json.loads(prefix[-1]["content"])["actor_summary"]["observations"], 1)

    def test_context_budget_boundary_is_invariant_to_unseen_answer(self):
        policy = replace(self.policy, response_reserve_tokens=self.policy.max_tokens)
        target = normalize_observations([self.trade("future", 1801, shares="1" + "0" * 15000)],
                                        self.catalog, self.split_policy)[0]
        empty, huge = self.build_two_queries([], policy), self.build_two_queries([target], policy)
        query = self.base + 1800 * SECOND
        self.assertEqual(self.query_prefix(empty, query), self.query_prefix(huge, query))
        self.assertEqual(self.query_prefix(empty, query)[0], 1)

    def test_contradictory_transaction_invalidates_every_fragment_and_pair_negatives(self):
        raw = [self.trade("fragment-a", 100, transaction="conflicting"),
               self.trade("fragment-b", 100, transaction="conflicting"),
               self.trade("fragment-c", 200, contract=self.second, transaction="conflicting"),
               self.trade("valid-first", 300), self.trade("valid-second", 400, contract=self.second)]
        normalized = normalize_observations(raw, self.catalog, self.split_policy)
        for row in normalized:
            self.assertEqual("invalid_source_transaction_timestamp" in row["errors"],
                             row["transaction_hash"] == "conflicting")
        result = build_actor_records(self.actor, raw, self.catalog, self.coverage, self.split_policy,
                                     self.policy, self.budget())
        self.assertEqual(result["counts"]["conditional_target_observations"], 2)
        self.assertFalse(any(r["profile"] == "scheduled_windows" for r in result["conversations"]))
        self.assertFalse(result["counts"].get("train_eligible_negative_windows", 0))

    def test_consistent_simultaneous_transaction_is_retained_as_joint_target(self):
        raw = [self.trade("fragment-a", 100, transaction="consistent"),
               self.trade("fragment-b", 100, contract=self.second, transaction="consistent")]
        result = build_actor_records(self.actor, raw, self.catalog, self.coverage, self.split_policy,
                                     self.policy, self.budget())
        self.assertTrue(all(not r["errors"] for r in result["observations"]))
        conditional = [r for r in result["conversations"] if r["profile"] == "conditional_trades"]
        self.assertEqual(len(conditional), 1)
        self.assertEqual(conditional[0]["turn_audit"][0]["target_observation_ids"], ["fragment-a", "fragment-b"])
        self.assertEqual(conditional[0]["turn_audit"][0]["history_observation_ids"], [])

    def test_precutoff_heldout_history_enrolls_first_forecast_without_entering_train(self):
        split_policy = {"fixture_splits": {self.first["fixture_id"]: "train", self.second["fixture_id"]: "validation"},
            "train_before_utc": _utc(self.base + 900 * SECOND),
            "validation_before_utc": _utc(self.base + 3600 * SECOND)}
        raw = [self.trade("train-history", 100), self.trade("heldout-history", 800, contract=self.second),
               self.trade("heldout-target", 1000, contract=self.second)]
        result = build_actor_records(self.actor, raw, self.catalog, self.coverage, split_policy,
                                     self.policy, self.budget())
        source = {r["observation_id"]: r for r in result["observations"]}
        self.assertEqual(source["heldout-history"]["errors"], ["fixture_time_split_mismatch"])
        self.assertEqual(source["heldout-history"]["history_split"], "validation")
        scheduled = [r for r in result["conversations"]
                     if r["profile"] == "scheduled_windows" and r["split"] == "validation"]
        first = scheduled[0]["turn_audit"][0]
        self.assertEqual(first["query_us"], self.base + 900 * SECOND)
        self.assertEqual(first["history_observation_ids"], ["heldout-history"])
        self.assertEqual(first["target_observation_ids"], ["heldout-target"])
        self.assertEqual(first["summary_observation_count"], 1)
        conditional = [r for r in result["conversations"]
                       if r["profile"] == "conditional_trades" and r["split"] == "validation"]
        self.assertEqual(conditional[0]["turn_audit"][0]["history_observation_ids"], ["heldout-history"])
        for row in result["conversations"]:
            if row["split"] == "train":
                for audit in row["turn_audit"]:
                    self.assertNotIn("heldout-history", audit["history_observation_ids"] + audit["target_observation_ids"])
                    self.assertNotIn("heldout-target", audit["history_observation_ids"] + audit["target_observation_ids"])

    def test_retained_query_receives_unshown_elapsed_trade_without_repeating_previous_target(self):
        raw = [self.trade("initial", 100), self.trade("previous-target", 1000), self.trade("elapsed-unshown", 2600)]
        history = normalize_observations(raw, self.catalog, self.split_policy)
        builder = ConversationBuilder(self.actor, "scheduled_windows", "train", history,
                                      self.catalog, self.policy, self.budget())
        condition = self.first["condition_id"]
        first = self.base + 900 * SECOND
        builder.add(first, first + 900 * SECOND, {condition}, [history[1]])
        # A later retained query can skip intervening sampled-out windows.
        later = self.base + 2700 * SECOND
        builder.add(later, later + 900 * SECOND, {condition}, [])
        builder.flush()
        audit = next(a for row in builder.records for a in row["turn_audit"] if a["query_us"] == later)
        self.assertEqual(audit["history_observation_ids"], ["elapsed-unshown"])
        self.assertNotIn("previous-target", audit["history_observation_ids"])


if __name__ == "__main__":
    unittest.main()
