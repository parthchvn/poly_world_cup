import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index, read_trade_context, is_2026_world_cup_headline


CONDITION = "0x" + "1" * 64
REGISTRY = {
    "fixtures": [{"fixture_id": "fixture:1"}, {"fixture_id": "fixture:2"}],
    "contracts": [{"condition_id": CONDITION, "fixture_id": "fixture:1", "selection": "france",
                   "tokens": [{"token_id": "123", "outcome": "Yes"},
                              {"token_id": "456", "outcome": "No"}]}],
}


def observation(identity="obs:1", *, timestamp="2026-06-10T12:00:00Z", token="123", tx="target", **kw):
    return {"observation_id": identity, "condition_id": CONDITION, "token_id": token,
            "side": "BUY", "size": "100.123456", "price": "0.12345678",
            "proxy_wallet": "wallet:a", "block_timestamp": timestamp, "transaction_hash": tx, **kw}


def news(identity="news:1", *, time="2026-06-10T11:00:00Z", verified=False, version=0, item=None):
    evidence = [{"kind": "archive_snapshot", "captured_at_utc": time,
                 "source_url": "https://example.org/archive", "content_sha256": "a" * 64}]
    return {"news_id": identity, "news_item_id": item or identity, "version_rank": version, "version_order_historically_verified": verified,
            "fixture_ids": ["fixture:1"], "published_at_utc": "2026-06-01T00:00:00Z",
            "captured_at_utc": "2026-09-23T00:00:00Z", "title": identity,
            "historical_availability_verified": verified, "availability_upper_utc": time,
            "historical_content_sha256": "a" * 64, "availability_evidence": evidence,
            "fixture_links": [{"fixture_id": "fixture:1", "relationship": "direct_match",
                               "historical_link_verified": verified,
                               "link_availability_upper_utc": time,
                               "link_availability_evidence": evidence}]}


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "attribution.sqlite"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, trades=None, records=None, *, compressed=False):
        path = self.directory / ("trades.jsonl.gz" if compressed else "trades.jsonl")
        data = "".join(json.dumps(row) + "\n" for row in (trades or [observation()])).encode()
        path.write_bytes(gzip.compress(data) if compressed else data)
        return build_attribution_index(registry=REGISTRY, trade_pages=[path],
                                       news_records=records or [], output_path=self.output)

    def test_current_capture_with_old_date_is_only_retrospective(self):
        report = self.build(records=[news()])
        context = read_trade_context(self.output, 1)
        self.assertEqual(report["historically_eligible_fixture_news_links"], 0)
        self.assertEqual(len(context["retrospective_news_candidates"]), 1)
        self.assertEqual(context["verified_public_context_at_execution_proxy"], [])
        self.assertFalse(context["actor_exposure_verified"])
        self.assertFalse(context["training_ready"])
        self.assertIsNone(context["holdings"])
        self.assertEqual(context["trade_amounts_semantics"], "provider_reported_not_chain_reconciled")
        self.assertFalse(context["chain_reconciliation_verified"])

    def test_strict_cutoff_excludes_equal_and_future_availability(self):
        report = self.build(records=[news("earlier", verified=True),
            news("equal", time="2026-06-10T12:00:00Z", verified=True),
            news("later", time="2026-06-10T13:00:00Z", verified=True)])
        context = read_trade_context(self.output, 1)
        self.assertEqual(report["historically_eligible_fixture_news_links"], 3)
        self.assertEqual(context["trade"]["eligible_context_event_count"], 1)
        self.assertEqual([r["news_id"] for r in context["verified_public_context_at_execution_proxy"]], ["earlier"])

    def test_future_extension_does_not_change_earlier_context_state(self):
        self.build(records=[news("past", verified=True)])
        before = read_trade_context(self.output, 1)["trade"]["context_state_id"]
        self.build(records=[news("future", time="2026-06-10T13:00:00Z", verified=True),
                            news("past", verified=True)])
        self.assertEqual(read_trade_context(self.output, 1)["trade"]["context_state_id"], before)

    def test_historical_assertion_requires_version_evidence(self):
        record = news(verified=True)
        record["availability_evidence"] = []
        report = self.build(records=[record])
        self.assertEqual(report["rejected_historical_availability_claims"], 1)
        self.assertEqual(report["historically_eligible_fixture_news_links"], 0)

    def test_wrong_version_hash_cannot_certify_article(self):
        record = news(verified=True)
        record["historical_content_sha256"] = "b" * 64
        report = self.build(records=[record])
        self.assertEqual(report["rejected_historical_availability_claims"], 1)

    def test_future_capture_cannot_prove_earlier_availability(self):
        record = news(verified=True)
        record["availability_evidence"][0]["captured_at_utc"] = "2026-09-23T00:00:00Z"
        report = self.build(records=[record])
        self.assertEqual(report["historically_eligible_fixture_news_links"], 0)

    def test_final_knockout_link_is_not_assumed_knowable_in_past(self):
        record = news(verified=True)
        record["fixture_links"][0]["historical_link_verified"] = False
        report = self.build(records=[record])
        self.assertEqual(report["historically_eligible_fixture_news_links"], 0)
        self.assertEqual(len(read_trade_context(self.output, 1)["retrospective_news_candidates"]), 1)

    def test_link_availability_also_respects_cutoff(self):
        record = news(verified=True)
        record["fixture_links"][0]["link_availability_upper_utc"] = "2026-06-10T13:00:00Z"
        self.build(records=[record])
        self.assertEqual(read_trade_context(self.output, 1)["verified_public_context_at_execution_proxy"], [])

    def test_version_selection_does_not_use_future_revision(self):
        original = news("original", verified=True, item="article")
        revision = news("revision", verified=True, item="article", version=1,
                        time="2026-06-10T13:00:00Z")
        self.build(records=[revision, original])
        result = read_trade_context(self.output, 1)["verified_public_context_at_execution_proxy"]
        self.assertEqual([item["news_id"] for item in result], ["original"])

    def test_unverified_version_order_cannot_supersede_historical_context(self):
        revision = news("revision", verified=True, item="article", version=1)
        revision["version_order_historically_verified"] = False
        report = self.build(records=[revision])
        self.assertEqual(report["historically_eligible_fixture_news_links"], 0)
        self.assertEqual(report["rejected_historical_availability_claims"], 1)

    def test_later_eligible_revision_supersedes_original(self):
        self.build(records=[news("original", verified=True, item="article"),
            news("revision", verified=True, item="article", version=1,
                 time="2026-06-10T11:30:00Z")])
        result = read_trade_context(self.output, 1)["verified_public_context_at_execution_proxy"]
        self.assertEqual([item["news_id"] for item in result], ["revision"])

    def test_equal_observations_and_both_economic_legs_are_preserved(self):
        report = self.build(trades=[observation(), observation(), observation("obs:2", side="SELL")])
        self.assertEqual(report["observation_count"], 3)
        self.assertEqual(report["duplicate_observation_id_groups"], 1)
        self.assertEqual(read_trade_context(self.output, 3)["trade"]["side"], "SELL")

    def test_buy_no_is_preserved_and_decimals_remain_exact(self):
        self.build(trades=[observation(token="456")])
        trade = read_trade_context(self.output, 1)["trade"]
        self.assertEqual((trade["side"], trade["token_outcome"]), ("BUY", "No"))
        self.assertEqual(trade["shares"], "100.123456")
        self.assertEqual(trade["price"], "0.12345678")

    def test_unresolved_historical_token_not_guessed(self):
        report = self.build(trades=[observation(token="999")])
        trade = read_trade_context(self.output, 1)["trade"]
        self.assertEqual(trade["fixture_id"], "fixture:1")
        self.assertEqual(trade["token_mapping_status"], "unresolved_token")
        self.assertIsNone(trade["token_outcome"])
        self.assertEqual(report["token_mapping_counts"], {"unresolved_token": 1})

    def test_unknown_condition_retained_without_fabricated_fixture(self):
        self.build(trades=[observation(condition_id="0x" + "2" * 64)])
        trade = read_trade_context(self.output, 1)["trade"]
        self.assertIsNone(trade["fixture_id"])
        self.assertIsNone(trade["context_state_id"])
        self.assertEqual(trade["token_mapping_status"], "unmapped_condition")

    def test_wallet_history_excludes_equal_times_and_entire_target_transaction(self):
        self.build(trades=[observation(), observation("same-time", tx="other"),
            observation("same-tx", timestamp="2026-06-10T11:59:59Z"),
            observation("past", timestamp="2026-06-10T11:59:59Z", tx="past-tx")])
        context = read_trade_context(self.output, 1)
        self.assertEqual([r["observation_id"] for r in context["retrospective_prior_tournament_executions"]], ["past"])
        self.assertFalse(context["wallet_history_feature_eligible"])
        self.assertEqual(context["wallet_history_scope"], "tournament_only")

    def test_gzip_and_plain_pages_produce_same_context(self):
        report = self.build(records=[news()], compressed=True)
        compressed = read_trade_context(self.output, 1)
        other = self.build(records=[news()])
        self.assertEqual(report, other)
        self.assertEqual(compressed, read_trade_context(self.output, 1))

    def test_duplicate_input_path_fails_instead_of_silent_reingestion(self):
        self.build()
        original = self.output.read_bytes()
        page = self.directory / "trades.jsonl"
        with self.assertRaises(sqlite3.IntegrityError):
            build_attribution_index(registry=REGISTRY, trade_pages=[page, page],
                                    news_records=[], output_path=self.output)
        self.assertEqual(self.output.read_bytes(), original)

    def test_failed_rebuild_keeps_previous_snapshot(self):
        self.build()
        original = self.output.read_bytes()
        with self.assertRaises(ValueError):
            self.build(trades=[observation(side="HOLD")])
        self.assertEqual(self.output.read_bytes(), original)

    def test_unknown_fixture_and_conflicting_news_versions_fail(self):
        bad = news()
        bad["fixture_ids"] = ["missing"]
        bad["fixture_links"] = []
        with self.assertRaises(ValueError):
            self.build(records=[bad])
        with self.assertRaises(sqlite3.IntegrityError):
            self.build(records=[news("one", item="shared"), news("two", item="shared")])

    def test_verified_global_news_does_not_need_retrospective_fixture_link(self):
        record = news(verified=True)
        record.update(title="World Cup team news", context_scope="tournament")
        record["fixture_ids"] = []
        record["fixture_links"] = []
        report = self.build(records=[record])
        context = read_trade_context(self.output, 1)
        self.assertEqual(report["historically_eligible_fixture_news_links"], 0)
        self.assertEqual(report["historically_eligible_global_news_versions"], 1)
        self.assertEqual(context["trade"]["global_eligible_context_event_count"], 1)
        self.assertEqual(context["verified_global_public_context"][0]["news_id"], record["news_id"])
        self.assertFalse(context["global_context_direct_fixture_relevance_verified"])
        with sqlite3.connect(self.output) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM fixture_news").fetchone()[0], 0)

    def test_old_current_global_headline_is_not_historical_evidence(self):
        record = news()
        record.update(title="World Cup team news", context_scope="tournament")
        self.build(records=[record])
        self.assertEqual(read_trade_context(self.output, 1)["verified_global_public_context"], [])

    def test_global_scope_needs_archived_title_or_scope_evidence(self):
        record = news(verified=True)
        record.update(title="France injury news", context_scope="tournament")
        self.build(records=[record])
        self.assertEqual(read_trade_context(self.output, 1)["verified_global_public_context"], [])
        record.update(historical_tournament_scope_verified=True,
            tournament_scope_availability_upper_utc="2026-06-10T11:30:00Z",
            tournament_scope_availability_evidence=record["availability_evidence"])
        self.build(records=[record])
        self.assertEqual(len(read_trade_context(self.output, 1)["verified_global_public_context"]), 1)

    def test_global_future_extension_and_equal_time_exclusion(self):
        past = news("past", verified=True)
        past.update(title="World Cup news", context_scope="tournament")
        self.build(records=[past])
        state = read_trade_context(self.output, 1)["trade"]["global_context_state_id"]
        equal = news("equal", verified=True, time="2026-06-10T12:00:00Z")
        equal.update(title="World Cup later news", context_scope="tournament")
        self.build(records=[equal, past])
        context = read_trade_context(self.output, 1)
        self.assertEqual(context["trade"]["global_context_state_id"], state)
        self.assertEqual(context["verified_global_public_context_item_count"], 1)

    def test_global_scope_rejects_other_editions_even_with_legacy_verified_flag(self):
        titles = ["Spain will host 2030 World Cup final, federation says",
                  "Club World Cup final preview", "Women's World Cup favorites",
                  "U-20 World Cup final", "Under 17 World Cup stars",
                  "2022 World Cup lessons for 2026", "Rugby World Cup squads"]
        for title in titles:
            with self.subTest(title=title):
                record = news(verified=True)
                record.update(title=title, context_scope="tournament",
                    historical_tournament_scope_verified=True,
                    tournament_scope_availability_upper_utc=record["availability_upper_utc"],
                    tournament_scope_availability_evidence=record["availability_evidence"])
                self.build(records=[record])
                self.assertEqual(read_trade_context(self.output, 1)["verified_global_public_context"], [])
                self.assertFalse(is_2026_world_cup_headline(title))
        self.assertTrue(is_2026_world_cup_headline("2026 World Cup squad news"))
        self.assertTrue(is_2026_world_cup_headline("World Cup: Gilmour ready for Scotland"))
        self.assertFalse(is_2026_world_cup_headline("Scotland squad news"))

    def test_global_display_limits_are_explicit(self):
        records = [news(str(i), verified=True) for i in range(3)]
        for record in records:
            record.update(title="World Cup news", context_scope="tournament")
        self.build(records=records)
        context = read_trade_context(self.output, 1, global_context_limit=1)
        self.assertEqual(context["verified_global_public_context_item_count"], 3)
        self.assertEqual(len(context["verified_global_public_context"]), 1)
        self.assertTrue(context["global_public_context_display_truncated"])

    def test_unmapped_condition_does_not_inherit_tournament_global_context(self):
        record = news(verified=True)
        record.update(title="World Cup news", context_scope="tournament")
        self.build(trades=[observation(condition_id="0x" + "2" * 64)], records=[record])
        context = read_trade_context(self.output, 1)
        self.assertEqual(context["trade"]["global_eligible_context_event_count"], 0)
        self.assertEqual(context["verified_global_public_context"], [])

    def test_read_unknown_trade_does_not_create_database(self):
        nonexistent = self.directory / "missing.sqlite"
        with self.assertRaises(sqlite3.OperationalError):
            read_trade_context(nonexistent, 1)
        self.assertFalse(nonexistent.exists())


if __name__ == "__main__":
    unittest.main()
