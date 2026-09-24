"""Source reconciliation and mutation attacks against the independent auditor."""
from collections import Counter
import copy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import unittest

from poly_world_cup.actor_sequences import build_actor_records
from poly_world_cup.sequence_context import ContextCatalog, _utc
from poly_world_cup.sequence_validation import (
    normalize_source, clock_windows, expected_turns, validate_actor_chunks,
    validate_sequences, source_actors, sha, canonical, PROFILES, SPLITS,
)
from poly_world_cup.sft import _Shards
import test_actor_sequence_causality as causality
SECOND = causality.SECOND


class SequenceValidatorTests(unittest.TestCase):
    setUp = causality.ActorSequenceCausalityTests.setUp
    trade = causality.ActorSequenceCausalityTests.trade
    budget = causality.ActorSequenceCausalityTests.budget

    def build(self, raw, policy=None):
        policy = policy or self.policy
        return build_actor_records(self.actor, copy.deepcopy(raw), self.catalog, self.coverage,
            self.split_policy, policy, self.budget(policy))

    def audit(self, result, raw, policy=None):
        policy = policy or self.policy
        normalized = normalize_source(raw, self.catalog, self.split_policy)
        self.assertEqual(result["observations"], normalized)
        invalid = {r["condition_id"] for r in normalized if set(r["errors"])-{"fixture_time_split_mismatch"}}
        fixture_counts, all_counts = {c["fixture_id"]:Counter() for c in self.catalog.contracts.values()},Counter()
        for profile in PROFILES:
            for split in SPLITS:
                prior = [r for r in normalized if r["history_split"] == split and not(set(r["errors"])-{"fixture_time_split_mismatch"})]
                expected = expected_turns(self.actor,prior,invalid,profile,split,self.split_policy,self.coverage,policy,all_counts)
                chunks = [r for r in result["conversations"] if r["profile"] == profile and r["split"] == split]
                counts = Counter()
                validate_actor_chunks(chunks,self.actor,prior,expected,profile,split,self.catalog,policy,
                    counts,fixture_counts,Counter())
        return fixture_counts

    def test_independent_clock_grid_initial_enrollment_and_trailing_windows(self):
        rows = normalize_source([self.trade("a",100),self.trade("b",1800)],self.catalog,self.split_policy)
        events = list(clock_windows(rows,-10**30,10**30,self.coverage,900*SECOND,1800*SECOND))
        self.assertEqual([(q-self.base)//SECOND for q,_,_ in events],[900,1800,2700,3600])
        self.assertEqual([[r["observation_id"] for r in targets] for _,_,targets in events],[[],["b"],[],[]])

    def test_positive_negative_and_same_time_joint_labels_match_source(self):
        raw = [self.trade("a",100),self.trade("b",1000),self.trade("c",1000)]
        result = self.build(raw)
        self.audit(result,raw)
        conditional = [r for r in result["conversations"] if r["profile"] == "conditional_trades"]
        audits = [a for r in conditional for a in r["turn_audit"]]
        self.assertEqual(audits[-1]["target_observation_ids"],["b","c"])

    def test_forged_no_trade_is_rejected_even_when_audit_claims_empty(self):
        raw = [self.trade("a",100),self.trade("b",1000)]
        result = self.build(raw)
        row = next(r for r in result["conversations"] if r["profile"] == "scheduled_windows")
        row["messages"][2]["content"] = '{"action":"NO_TRADE"}'
        row["turn_audit"][0]["target_observation_ids"] = []
        with self.assertRaisesRegex(ValueError,"source targets"):
            self.audit(result,raw)

    def test_missing_trailing_no_trade_window_fails(self):
        raw = [self.trade("a",100),self.trade("b",1000)]
        result = self.build(raw)
        row = [r for r in result["conversations"] if r["profile"] == "scheduled_windows"][-1]
        row["messages"] = row["messages"][:-2]
        row["turn_audit"] = row["turn_audit"][:-1]
        with self.assertRaisesRegex(ValueError,"omitted"):
            self.audit(result,raw)

    def test_future_summary_and_missing_query_markets_fail(self):
        raw = [self.trade("a",100),self.trade("b",1000)]
        result = self.build(raw)
        row = next(r for r in result["conversations"] if r["profile"] == "conditional_trades")
        user = json.loads(row["messages"][1]["content"])
        user["actor_summary"]["observations"] += 1
        row["messages"][1]["content"] = canonical(user)
        with self.assertRaisesRegex(ValueError,"actor summary"):
            self.audit(result,raw)
        result = self.build(raw)
        row = next(r for r in result["conversations"] if r["profile"] == "conditional_trades")
        user = json.loads(row["messages"][1]["content"])
        del user["query_markets"]
        row["messages"][1]["content"] = canonical(user)
        with self.assertRaisesRegex(ValueError,"conditional query markets"):
            self.audit(result,raw)

    def test_contradictory_transaction_times_quarantine_all_fragments(self):
        raw = [self.trade("a",100,transaction="same"),self.trade("b",1000,transaction="same"),self.trade("c",2000)]
        result = self.build(raw)
        self.assertEqual(sum("invalid_source_transaction_timestamp" in r["errors"] for r in result["observations"]),2)
        self.assertFalse(any(r["profile"] == "scheduled_windows" for r in result["conversations"]))
        self.audit(result,raw)

    def test_time_mismatched_history_remains_only_in_its_fixture_partition(self):
        self.split_policy["fixture_splits"][self.first["fixture_id"]] = "validation"
        self.split_policy["train_before_utc"] = _utc(self.base+900*SECOND)
        raw = [self.trade("a",100),self.trade("other",200,contract=self.second),self.trade("b",1000)]
        result = self.build(raw)
        self.audit(result,raw)
        row = next(r for r in result["conversations"] if r["profile"] == "conditional_trades" and r["split"] == "validation")
        self.assertEqual(row["turn_audit"][0]["history_observation_ids"],["a"])
        self.assertEqual(row["turn_audit"][0]["target_observation_ids"],["b"])

    def test_incomplete_end_window_is_not_exported(self):
        self.coverage[self.first["condition_id"]]["latest_query_us"] = self.base+2200*SECOND
        raw = [self.trade("a",100),self.trade("b",1000)]
        result = self.build(raw)
        self.audit(result,raw)
        windows = [a for r in result["conversations"] if r["profile"] == "scheduled_windows" for a in r["turn_audit"]]
        self.assertEqual([a["query_us"] for a in windows],[self.base+900*SECOND])

    def test_chunk_carry_and_summary_hash_are_checked(self):
        policy = replace(self.policy,max_turns=1,carry_trades=1)
        raw = [self.trade("a",100),self.trade("b",1000),self.trade("c",2000)]
        result = self.build(raw,policy)
        self.audit(result,raw,policy)
        chunks = [r for r in result["conversations"] if r["profile"] == "conditional_trades"]
        self.assertEqual(chunks[-1]["turn_audit"][0]["history_observation_ids"],["b"])
        self.assertEqual(chunks[-1]["turn_audit"][0]["summary_observation_count"],2)
        chunks[-1]["turn_audit"][0]["summary_observation_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"Summary source count/hash"):
            self.audit(result,raw,policy)

    def test_equal_or_future_news_in_prompt_is_rejected(self):
        raw = [self.trade("a",100),self.trade("b",1000)]
        result = self.build(raw)
        row = next(r for r in result["conversations"] if r["profile"] == "conditional_trades")
        user = json.loads(row["messages"][1]["content"])
        user["news"] = [{"news_id":"n9999","headline":"A future World Cup headline",
            "available_at":_utc(self.base+1000*SECOND)}]
        row["messages"][1]["content"] = canonical(user)
        with self.assertRaisesRegex(ValueError,"time-invalid news"):
            self.audit(result,raw)

    def test_duplicate_prompt_key_cannot_hide_unvalidated_future_text(self):
        raw = [self.trade("a",100),self.trade("b",1000)]
        result = self.build(raw)
        row = next(r for r in result["conversations"] if r["profile"] == "conditional_trades")
        original = row["messages"][1]["content"]
        # JSON parsers keep the last duplicate key, but an LLM sees both.
        row["messages"][1]["content"] = '{"query_markets":["future secret action"],'+original[1:]
        with self.assertRaisesRegex(ValueError,"Noncanonical prompt"):
            self.audit(result,raw)

    def test_filter_includes_exactly_twenty_but_excludes_twenty_one(self):
        db = sqlite3.connect(":memory:");self.addCleanup(db.close);db.row_factory=sqlite3.Row
        db.executescript("""CREATE TABLE trades(wallet TEXT,condition_id TEXT,query_us INTEGER,observation_id TEXT,trade_row_id INTEGER);
            CREATE TABLE wallet_market_counts(wallet TEXT,condition_id TEXT,observation_count INTEGER);""")
        db.executemany("INSERT INTO wallet_market_counts VALUES(?,?,?)",[("a","m1",20),("a","m2",21),("b","m1",1)])
        db.executemany("INSERT INTO trades VALUES(?,?,?,?,?)",[("a","m1",1,"a1",1),("a","m2",2,"a2",2),("b","m1",3,"b1",3)])
        selected = [(a,[r["condition_id"] for r in rows]) for a,rows in source_actors(db,self.policy)]
        self.assertEqual(selected,[("a",["m1"]),("b",["m1"])])
        self.assertEqual([a for a,_ in source_actors(db,replace(self.policy,filter_scope="actor"))],["b"])

    def release_fixture(self):
        # Three recovered eligible rows plus an excluded full-ledger pair.
        # A final trade supplies a conservative observation bound but is not
        # required to label trailing windows in the earlier active period.
        with gzip.open(self.root/"contracts/contract_evidence.jsonl.gz","wt") as stream:
            stream.write(json.dumps(self.first)+"\n")
        self.catalog = ContextCatalog(self.root)
        self.split_policy["fixture_splits"] = {self.first["fixture_id"]:"train"}
        condition = self.first["condition_id"]
        self.coverage = {condition:{"earliest_query_us":self.base+100*SECOND,"latest_query_us":self.base+9000*SECOND}}
        raw = [self.trade("a",100),self.trade("b",1000),self.trade("c",9000)]
        result = self.build(raw)
        source = self.root/"source.sqlite"
        db = sqlite3.connect(source)
        db.executescript("""CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT);
            CREATE TABLE trades(trade_row_id INTEGER,observation_id TEXT,wallet TEXT,condition_id TEXT,
                token_id TEXT,query_us INTEGER,transaction_hash TEXT,side TEXT,shares TEXT,price TEXT);
            CREATE TABLE wallet_market_counts(wallet TEXT,condition_id TEXT,observation_count INTEGER);
            CREATE TABLE source_pages(source_page_id INTEGER,path TEXT,provenance_scope TEXT);
            CREATE TABLE condition_coverage(condition_id TEXT,fixture_id TEXT,status TEXT,
                observation_count INTEGER,selected_observation_count INTEGER,earliest_query_us INTEGER,
                latest_query_us INTEGER,coverage_bound_scope TEXT);""")
        for key,value in (("source_scope","selected_cohort_with_full_count_ledger"),("coverage_bound_scope","selected_cohort_observed_interval")):
            db.execute("INSERT INTO metadata VALUES(?,?)",(key,json.dumps(value)))
        columns = [r[1] for r in db.execute("PRAGMA table_info(trades)")]
        db.executemany("INSERT INTO trades VALUES("+",".join("?" for _ in columns)+")",([r[k] for k in columns] for r in raw))
        pair_rows = [{"wallet":self.actor,"condition_id":condition,"observation_count":3},
                     {"wallet":"z-excluded","condition_id":condition,"observation_count":21}]
        db.executemany("INSERT INTO wallet_market_counts VALUES(?,?,?)",(tuple(r.values()) for r in pair_rows))
        scope = {"condition_id":condition,"fixture_id":self.first["fixture_id"],"status":"inherited_api_exhausted_selected_cohort_verified",
            "observation_count":24,"selected_observation_count":3,**self.coverage[condition],"coverage_bound_scope":"selected_cohort_observed_interval"}
        db.execute("INSERT INTO condition_coverage VALUES(?,?,?,?,?,?,?,?)",tuple(scope.values()))
        db.commit(); db.close()
        dataset = self.root/"release"; dataset.mkdir()
        def save(name,value):
            (dataset/name).write_text(canonical(value)+"\n")
        save("policy.json",asdict(self.policy)); save("split_policy.json",self.split_policy)
        save("source_metadata.json",{"source_scope":"selected_cohort_with_full_count_ledger","coverage_bound_scope":"selected_cohort_observed_interval"})
        save("context_catalog.json",self.catalog.audit_catalog()); save("source_coverage.json",{condition:scope})
        save("fixture_coverage.json",result["fixture_counts"])
        writer = _Shards(dataset/"source_evidence/pair_counts",100)
        for row in pair_rows: writer.write(row)
        writer.close()
        writer = _Shards(dataset/"observations",100)
        for row in result["observations"]: writer.write(row)
        writer.close()
        profiles = {p:{s:{} for s in SPLITS} for p in PROFILES}
        index = _Shards(dataset/"actor_index",100)
        lengths = []
        for profile in PROFILES:
            writer = _Shards(dataset/profile/"train",100)
            for row in (r for r in result["conversations"] if r["profile"] == profile):
                location = writer.write(row)
                index.write({k:row[k] for k in ("sequence_id","actor_id","profile","split","chunk_index")}|
                    {"path":profile+"/train/part-00001.jsonl.gz","line":location["line"],"sha256":hashlib.sha256(canonical(row).encode()).hexdigest()})
                lengths.append(row["token_count"])
                counter = profiles[profile]["train"]
                for key,number in (("conversations",1),("target_turns",len(row["turn_audit"])),("tokens",row["token_count"]),("long_context_sequences",int(row["requires_long_context"]))):
                    counter[key] = counter.get(key,0)+number
            writer.close()
        index.close()
        lengths.sort()
        manifest = {"schema_version":3,"sample":False,"status":"exported","source":{"sha256":sha(source),"bytes":source.stat().st_size},
            "selected_observations":3,"quarantined_observations":0,"counts":{"actors":1,**result["counts"]},"profiles":profiles,
            "fixture_target_coverage":{key:{"present":1,"missing":[]} for key in ("selected_observations","conditional_targets","scheduled_target_observations")},
            "tokens":{"total":sum(lengths),"p50":lengths[math.ceil(len(lengths)*.5)-1],"p95":lengths[math.ceil(len(lengths)*.95)-1],"max":max(lengths)}}
        def rehash():
            manifest["artifacts"] = {p.relative_to(dataset).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size}
                for p in dataset.rglob("*") if p.is_file() and p.name != "manifest.json"}
            save("manifest.json",manifest)
        rehash()
        return dataset,source,manifest,rehash

    def test_full_release_source_reconciliation_and_rehashed_index_attack(self):
        dataset,source,manifest,rehash = self.release_fixture()
        report = validate_sequences(dataset,source,self.root,progress=lambda _:None)
        self.assertEqual(report["status"],"passed")
        self.assertEqual(report["source_reconciliation"]["full_source_observations"],24)
        self.assertEqual(report["source_reconciliation"]["selected_observations"],3)
        path = dataset/"actor_index/part-00001.jsonl.gz"
        with gzip.open(path,"rt") as stream: rows = [json.loads(line) for line in stream]
        rows[0]["line"] = 999
        with gzip.open(path,"wt") as stream:
            for row in rows: stream.write(json.dumps(row)+"\n")
        rehash()
        with self.assertRaisesRegex(ValueError,"Actor index"):
            validate_sequences(dataset,source,self.root,progress=lambda _:None)

    def test_full_ledger_excluded_pairs_cannot_be_dropped(self):
        dataset,source,manifest,rehash = self.release_fixture()
        path = dataset/"source_evidence/pair_counts/part-00001.jsonl.gz"
        with gzip.open(path,"rt") as stream: first = next(stream)
        with gzip.open(path,"wt") as stream: stream.write(first)
        rehash()
        with self.assertRaisesRegex(ValueError,"Full pair count ledger"):
            validate_sequences(dataset,source,self.root,progress=lambda _:None)

    def test_missing_source_row_fails_even_with_updated_source_hash(self):
        dataset,source,manifest,rehash = self.release_fixture()
        db = sqlite3.connect(source); db.execute("DELETE FROM trades WHERE observation_id='b'");db.commit();db.close()
        manifest["source"] = {"sha256":sha(source),"bytes":source.stat().st_size}; rehash()
        with self.assertRaisesRegex(ValueError,"Eligible pair missing or incomplete"):
            validate_sequences(dataset,source,self.root,progress=lambda _:None)


if __name__ == "__main__": unittest.main()
