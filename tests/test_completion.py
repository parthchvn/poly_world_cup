import copy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from poly_world_cup.completion import assess, coverage_for_records, eligible_target_times
from poly_world_cup.sft import export_sft
from scripts.validate_sft import validate_sft


POLICY = {"fixture_splits": {"f1": "train", "f2": "validation", "f3": "test"},
          "train_before_utc": "2026-06-11T00:00:00Z", "validation_before_utc": "2026-06-12T00:00:00Z"}
REGISTRY = {"fixtures": [{"fixture_id": f"f{i}"} for i in (1, 2, 3)], "contracts": [
    {"condition_id": f"c{i}", "fixture_id": f"f{i}", "selection": "home", "tokens": [
        {"token_id": f"{i}yes", "outcome": "Yes"}, {"token_id": f"{i}no", "outcome": "No"}]}
    for i in (1, 2, 3)]}


def trade(i, **changes):
    return {"observation_id": f"trade{i}", "condition_id": f"c{i}", "token_id": f"{i}yes",
            "side": "BUY", "size": "2", "price": "0.4", "proxy_wallet": f"wallet{i}",
            "block_timestamp": f"2026-06-{9+i}T12:00:00Z", "transaction_hash": f"tx{i}", **changes}


def news(i, **changes):
    time = f"2026-06-{9+i}T11:00:00Z"
    evidence = [{"kind": "archive_snapshot", "captured_at_utc": time,
                 "source_url": "https://archive.example/evidence", "content_sha256": "a" * 64}]
    return {"news_id": f"news{i}", "title": f"World Cup 2026 match {i}",
            "source_url": f"https://news.example/{i}", "historical_availability_verified": True,
            "historical_content_sha256": "a" * 64, "availability_upper_utc": time,
            "availability_evidence": evidence, "context_scope": "tournament",
            "fixture_links": [{"fixture_id": f"f{i}", "historical_link_verified": True,
                               "relationship": "direct_match", "link_availability_upper_utc": time,
                               "link_availability_evidence": evidence}], **changes}


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "source.sqlite"
        self.release = self.root / "release"

    def tearDown(self):
        self.temp.cleanup()

    def build(self, rows=None, records=None):
        page = self.root / "trades.jsonl"
        page.write_text("".join(json.dumps(row) + "\n" for row in (rows or [trade(i) for i in (1, 2, 3)])))
        build_attribution_index(registry=REGISTRY, trade_pages=[page],
                                news_records=records if records is not None else [news(i) for i in (1, 2, 3)],
                                output_path=self.database)
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE wallet_counts(wallet TEXT PRIMARY KEY, observed_count INTEGER NOT NULL)")
            db.execute("INSERT INTO wallet_counts SELECT wallet,COUNT(*) FROM trades GROUP BY wallet")
            db.execute("INSERT INTO wallet_counts VALUES('excluded',22)")
            db.execute("CREATE TABLE wallet_market_counts(wallet TEXT,condition_id TEXT,observation_count INTEGER,PRIMARY KEY(wallet,condition_id))")
            db.execute("INSERT INTO wallet_market_counts SELECT wallet,condition_id,COUNT(*) FROM trades GROUP BY wallet,condition_id")
            db.execute("INSERT INTO wallet_market_counts VALUES('excluded','c1',22)")
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            report.update(partial_api_collection=False, api_traversals_exhausted=3, registry_condition_count=3,
                          threshold_scope="tournament", threshold_exclusive=20, full_wallet_history_included=True,
                          source_wallets=db.execute("SELECT COUNT(*) FROM wallet_counts").fetchone()[0],
                          source_observations=db.execute("SELECT SUM(observed_count) FROM wallet_counts").fetchone()[0])
            db.execute("UPDATE metadata SET value_json=? WHERE key='report'", (json.dumps(report),))
            ledger = [{"condition_id": f"c{i}", "status": "api_exhausted", "provenance_scope": "fresh",
                       "manifest_sha256": "a" * 64, "checkpoint_sha256": "b" * 64} for i in (1, 2, 3)]
            db.execute("INSERT INTO metadata VALUES('contract_exhaustion_ledger',?)", (json.dumps(ledger),))

    def export(self, **kwargs):
        export_sft(self.database, self.release, POLICY, shard_rows=2, **kwargs)

    def assess(self, *, release=True, registry=None):
        return assess(self.database, registry or REGISTRY, self.release if release else None,
                      expected_fixtures=3, expected_contracts=3, expected_tokens=6, split_policy=POLICY)

    def test_all_fixtures_ready_with_direct_verified_news_and_count_ledger(self):
        self.build()
        self.export()
        report = self.assess()
        self.assertTrue(report["all104_fixture_context_ready"], report["failed_checks"])
        self.assertEqual(report["fixtures_with_exported_prior_direct_news"], 3)
        self.assertFalse(report["prospective_fully_ready"])
        self.assertFalse(report["canonical_chain_completeness_certified"])

    def test_preflight_is_not_final_release_and_uses_export_eligibility(self):
        self.build(rows=[trade(1), trade(2), trade(3),
                         trade(1, observation_id="late", block_timestamp="2026-06-13T12:00:00Z")])
        target_times = eligible_target_times(self.database, POLICY)
        self.assertEqual([len(target_times[f"f{i}"]) for i in (1, 2, 3)], [1, 1, 1])
        report = self.assess(release=False)
        self.assertEqual(report["failed_checks"], ["release_present"])
        self.assertFalse(report["all104_fixture_context_ready"])

    def test_partial_capture_and_missing_ledger_fail_closed(self):
        self.build()
        with sqlite3.connect(self.database) as db:
            report = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()[0])
            report["partial_api_collection"] = True
            db.execute("UPDATE metadata SET value_json=? WHERE key='report'", (json.dumps(report),))
            db.execute("DELETE FROM metadata WHERE key='contract_exhaustion_ledger'")
        self.export(allow_partial=True)
        report = self.assess()
        self.assertIn("all_contract_api_traversals_exhausted", report["failed_checks"])
        self.assertIn("contract_capture_ledger_valid", report["failed_checks"])

    def test_team_background_and_tournament_news_do_not_count_as_direct(self):
        values = [news(i) for i in (1, 2, 3)]
        values[0]["fixture_links"][0]["relationship"] = "team_category_background"
        values[1]["fixture_links"] = []
        self.build(records=values)
        self.export()
        report = self.assess()
        self.assertEqual(report["fixtures_with_exported_prior_direct_news"], 1)
        self.assertEqual(report["fixture_ids_missing_eligible_prior_direct_news"], ["f1", "f2"])
        self.assertFalse(report["all104_fixture_context_ready"])

    def test_equal_timestamp_news_is_excluded_but_early_empty_context_is_valid(self):
        self.build(rows=[trade(1), trade(2), trade(3), trade(1, observation_id="early", block_timestamp="2026-06-10T11:00:00Z")])
        self.export()
        report = self.assess()
        row = report["fixtures"][0]
        self.assertEqual(row["exported_targets"], 2)
        self.assertEqual(row["exported_targets_with_selected_prior_direct_fixture_news"], 1)
        self.assertTrue(report["all104_fixture_context_ready"])
        self.assertFalse(report["every_target_has_direct_fixture_news"])

    def test_late_link_availability_cannot_backdate_content(self):
        values = [news(i) for i in (1, 2, 3)]
        values[0]["fixture_links"][0]["link_availability_upper_utc"] = "2026-06-10T12:00:00Z"
        self.build(records=values)
        self.export()
        self.assertEqual(self.assess()["fixtures_with_exported_prior_direct_news"], 2)

    def test_prompt_limit_can_displace_older_direct_news(self):
        self.build()
        times = eligible_target_times(self.database, POLICY)
        old = news(1)
        newer = news(1, news_id="newer", availability_upper_utc="2026-06-10T11:30:00Z")
        newer["fixture_links"][0]["relationship"] = "team_category_background"
        result = coverage_for_records(times, [old, newer], news_limit=1)
        self.assertEqual(result[0]["verified_direct_fixture_news_versions"], 1)
        self.assertEqual(result[0]["eligible_targets_with_selected_prior_direct_fixture_news"], 0)

    def test_headline_deduplication_matches_v2_export_policy(self):
        values = [news(i) for i in (1, 2, 3)]
        for index, minute in enumerate((30, 45)):
            record = news(1, news_id=f"background{index}", title="World Cup 2026 same background",
                          availability_upper_utc=f"2026-06-10T11:{minute}:00Z")
            record["fixture_links"][0]["relationship"] = "team_category_background"
            values.append(record)
        self.build(records=values)
        times = eligible_target_times(self.database, POLICY)
        plain = coverage_for_records(times, values, news_limit=2)
        deduplicated = coverage_for_records(times, values, news_limit=2, deduplicate_headlines=True)
        self.assertEqual(plain[0]["eligible_targets_with_selected_prior_direct_fixture_news"], 0)
        self.assertEqual(deduplicated[0]["eligible_targets_with_selected_prior_direct_fixture_news"], 1)
        self.export(news_limit=2, deduplicate_headlines=True)
        report = self.assess()
        self.assertTrue(report["all104_fixture_context_ready"], report["failed_checks"])

    def test_legacy_archived_game_url_link_counts_without_new_relationship(self):
        values = [news(i) for i in (1, 2, 3)]
        link = values[0]["fixture_links"][0]
        del link["relationship"]
        link["relevance_basis"] = "archived_direct_game_url"
        self.build(records=values)
        self.export()
        self.assertTrue(self.assess()["all104_fixture_context_ready"])

    def test_whole_tournament_ledger_cannot_be_recomputed_from_subset(self):
        self.build()
        self.export()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE wallet_market_counts SET observation_count=21 WHERE wallet='wallet1'")
        report = self.assess()
        self.assertIn("all_wallet_contract_count_ledger_valid", report["failed_checks"])
        self.assertIn("release_source_database_hash_matches", report["failed_checks"])

    def test_global_token_reuse_is_invalid_even_with_expected_total(self):
        self.build()
        self.export()
        registry = copy.deepcopy(REGISTRY)
        registry["contracts"][1]["tokens"][0]["token_id"] = "1yes"
        report = self.assess(registry=registry)
        self.assertIn("registry_universe_valid", report["failed_checks"])
        self.assertIn("selected_token_mappings_valid", report["failed_checks"])

    def test_exhausted_count_does_not_replace_per_contract_identity_proof(self):
        self.build()
        with sqlite3.connect(self.database) as db:
            ledger = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='contract_exhaustion_ledger'").fetchone()[0])
            ledger[0]["condition_id"] = "wrong-contract"
            db.execute("UPDATE metadata SET value_json=? WHERE key='contract_exhaustion_ledger'", (json.dumps(ledger),))
        self.export()
        report = self.assess()
        self.assertTrue(report["checks"]["all_contract_api_traversals_exhausted"])
        self.assertIn("contract_capture_ledger_valid", report["failed_checks"])

    def test_missing_selected_history_fails_even_if_wallet_is_under20(self):
        self.build()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE wallet_counts SET observed_count=2 WHERE wallet='wallet1'")
            db.execute("UPDATE wallet_market_counts SET observation_count=2 WHERE wallet='wallet1'")
        report = self.assess(release=False)
        self.assertIn("retained_wallet_histories_match_full_counts", report["failed_checks"])

    def test_release_audit_cannot_claim_a_different_fixture(self):
        self.build()
        self.export()
        path = sorted((self.release / "audit").glob("*.jsonl.gz"))[0]
        with gzip.open(path, "rt") as stream:
            audits = [json.loads(line) for line in stream]
        audits[0]["fixture_id"] = "f2"
        with gzip.open(path, "wt") as stream:
            for row in audits:
                stream.write(json.dumps(row) + "\n")
        report = self.assess()
        self.assertIn("release_audit_exactly_covers_source", report["failed_checks"])
        self.assertIn("release_selected_news_strictly_prior_and_verified", report["failed_checks"])

    def _repack_and_rehash(self, path, rows):
        path.write_bytes(gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0))
        manifest_path = self.release / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        relative = path.relative_to(self.release).as_posix()
        artifact = next(item for item in manifest["files"] if item["path"] == relative)
        artifact.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest))

    def test_consistently_rehashed_audit_and_profile_labels_must_match_source(self):
        self.build()
        self.export()
        audit_path = self.release / "audit/part-00001.jsonl.gz"
        with gzip.open(audit_path, "rt") as stream:
            original_audits = [json.loads(line) for line in stream]
        profiles = {}
        audit = original_audits[0]
        for profile, position in audit["profile_locations"].items():
            path = self.release / profile / audit["exported_split"] / f"part-{position['shard']:05d}.jsonl.gz"
            with gzip.open(path, "rt") as stream:
                profiles[path] = ([json.loads(line) for line in stream], position["line"] - 1)
        for field, changed in (("side", "SELL"), ("outcome", "No"), ("shares", "999.123"), ("price", "0.6")):
            with self.subTest(field=field):
                audits = copy.deepcopy(original_audits)
                audits[0]["reported_action"][field] = changed
                self._repack_and_rehash(audit_path, audits)
                for path, (original_examples, index) in profiles.items():
                    examples = copy.deepcopy(original_examples)
                    examples[index]["messages"][2]["content"] = json.dumps(audits[0]["reported_action"])
                    self._repack_and_rehash(path, examples)
                # Artifact validation establishes internal consistency. The
                # completion gate must additionally bind labels to the source.
                self.assertEqual(validate_sft(self.release)["profile_examples_verified"], 6)
                report = self.assess()
                self.assertIn("release_audit_exactly_covers_source", report["failed_checks"])
                self.assertEqual(report["details"]["release"]["source_mismatched_audits"], 1)

    def test_rehashed_audit_token_transaction_and_page_identity_must_match_source(self):
        self.build()
        self.export()
        path = self.release / "audit/part-00001.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            original = [json.loads(line) for line in stream]
        for field, value in (("token_id", "wrong-token"), ("transaction_hash", "wrong-transaction"),
                             ("source_page_id", 999), ("source_line", 999)):
            with self.subTest(field=field):
                audits = copy.deepcopy(original)
                audits[0][field] = value
                self._repack_and_rehash(path, audits)
                # This wallet has one observation, so transaction changes cannot
                # be rejected incidentally through another row's history.
                self.assertEqual(validate_sft(self.release)["profile_examples_verified"], 6)
                report = self.assess()
                self.assertIn("release_audit_exactly_covers_source", report["failed_checks"])
                self.assertEqual(report["details"]["release"]["source_mismatched_audits"], 1)

    def test_split_purged_history_timestamp_is_bound_to_source(self):
        self.build(rows=[trade(i) for i in (1, 2, 3)] + [
            trade(3, observation_id="prefix", transaction_hash="prefix-tx",
                  block_timestamp="2026-06-10T10:00:00Z")])
        self.export()
        path = self.release / "audit/part-00002.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            audits = [json.loads(line) for line in stream]
        self.assertIsNone(audits[0]["exported_split"])
        self.assertEqual(audits[0]["excluded_reasons"], ["fixture_time_split_mismatch"])
        audits[0]["execution_time_proxy_utc"] = "2026-06-10T09:00:00Z"
        self._repack_and_rehash(path, audits)
        path = self.release / "execution_history_proxy/test/part-00001.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            examples = [json.loads(line) for line in stream]
        prompt = json.loads(examples[0]["messages"][1]["content"])
        prompt["prior_tournament_executions"][0]["execution_time_proxy_utc"] = "2026-06-10T09:00:00Z"
        examples[0]["messages"][1]["content"] = json.dumps(prompt)
        self._repack_and_rehash(path, examples)
        self.assertEqual(validate_sft(self.release)["profile_examples_verified"], 6)
        report = self.assess()
        self.assertIn("release_audit_exactly_covers_source", report["failed_checks"])
        self.assertEqual(report["details"]["release"]["source_mismatched_audits"], 1)

    def test_invalid_timestamp_preserved_raw_fields_are_bound_to_source(self):
        self.build()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE trades SET query_us='invalid-original' WHERE trade_row_id=1")
        self.export()
        self.assertTrue(self.assess()["checks"]["release_audit_exactly_covers_source"])
        path = self.release / "audit/part-00001.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            original = [json.loads(line) for line in stream]
        self.assertIsNone(original[0]["execution_time_proxy_utc"])
        for field, value in (("source_query_us", "different-invalid-time"),
                             ("source_block_timestamp", "2026-06-10T09:00:00Z")):
            with self.subTest(field=field):
                audits = copy.deepcopy(original)
                audits[0][field] = value
                self._repack_and_rehash(path, audits)
                report = self.assess()
                self.assertIn("release_audit_exactly_covers_source", report["failed_checks"])
                self.assertEqual(report["details"]["release"]["source_mismatched_audits"], 1)


if __name__ == "__main__":
    unittest.main()
