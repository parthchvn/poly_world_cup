"""Historical cutoff and source-scope tests for compact sequence context."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from poly_world_cup.attribution import _micros
from poly_world_cup.historical_contracts import build_contract_record
from poly_world_cup.sequence_context import ContextCatalog, _utc
from test_historical_contracts import contract_record_fixture


class SequenceContextTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "contracts").mkdir()
        (self.root / "news_catalog").mkdir()
        self.contract = contract_record_fixture()
        self.fixture = self.contract["fixture_id"]
        self.condition = self.contract["condition_id"]
        self.base = _micros(self.contract["initialized_at_utc"])
        other_evidence = copy.deepcopy(self.contract["evidence"])
        other_evidence["mapping_calls"][0]["result"] = "0x" + "dd" * 32
        self.other_contract = build_contract_record(other_evidence, fixture_id="espn:760416")
        self.other_fixture = self.other_contract["fixture_id"]
        self.save_contracts([self.contract, self.other_contract])

    def save_contracts(self, records):
        with gzip.open(self.root / "contracts/contract_evidence.jsonl.gz", "wt") as stream:
            for row in records:
                stream.write(json.dumps(row) + "\n")

    def evidence(self, offset):
        return [{"kind": "archive_snapshot", "captured_at_utc": _utc(self.base + offset),
                 "content_sha256": "a" * 64, "source_url": "https://example.test/archive"}]

    def news(self, name, offset, *, title=None, fixture=None, link_offset=None,
             global_scope=True, direct=True, rank=0, item=None):
        row = {"news_id": name, "news_item_id": item or name, "version_rank": rank,
               "version_order_historically_verified": True,
               "title": title or "World Cup " + name,
               "source_url": "https://example.test/" + name,
               "historical_availability_verified": True,
               "availability_upper_utc": _utc(self.base + offset),
               "historical_content_sha256": "a" * 64,
               "availability_evidence": self.evidence(offset),
               "context_scope": "tournament" if global_scope else "fixture", "fixture_links": []}
        if fixture is not None:
            link_offset = offset if link_offset is None else link_offset
            row["fixture_links"].append({
                "fixture_id": fixture, "historical_link_verified": True,
                "direct_fixture_relevance_verified": direct,
                "relevance_basis": "direct_fixture" if direct else "retrospective_team_candidate",
                "link_availability_upper_utc": _utc(self.base + link_offset),
                "link_availability_evidence": self.evidence(link_offset)})
        return row

    def catalog(self, records=(), **kwargs):
        (self.root / "news_catalog/news_for_sft.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
        return ContextCatalog(self.root, **kwargs)

    def ids(self, rows):
        return [r["news_id"] for r in rows]

    def test_contract_strict_cutoff_and_minimal_initial_semantics(self):
        catalog = self.catalog()
        self.assertIsNone(catalog.contract(self.condition, self.base))
        self.assertIsNone(catalog.contract(self.condition, self.base - 1))
        value = catalog.contract(self.condition, self.base + 1)
        self.assertEqual(value["question"], self.contract["question"])
        self.assertEqual(value["outcomes"], ["Yes", "No"])
        self.assertEqual(set(value), {"market_id", "fixture", "question", "outcomes"})
        self.assertIsNone(catalog.contract("missing", self.base + 1))

    def test_contract_evidence_mutation_is_rejected(self):
        altered = copy.deepcopy(self.contract)
        altered["question"] = "A later edited question"
        self.save_contracts([altered])
        with self.assertRaisesRegex(ValueError, "derived field: question"):
            self.catalog()

    def test_initial_per_scope_limit_and_chronological_output(self):
        rows = [self.news("global-old", 10), self.news("global-new", 30),
                self.news("fixture-old", 20, fixture=self.fixture, global_scope=False),
                self.news("fixture-new", 40, fixture=self.fixture, global_scope=False)]
        catalog = self.catalog(rows, max_initial_news=1)
        self.assertEqual(self.ids(catalog.news_before([self.fixture], self.base + 50)),
                         ["global-new", "fixture-new"])

    def test_delta_lower_boundary_inclusive_upper_boundary_exclusive(self):
        catalog = self.catalog([self.news("old", 10), self.news("lower", 20),
                                self.news("between", 25), self.news("equal", 30)])
        result = catalog.news_before([], self.base + 30, since_us=self.base + 20, initial_limit=0)
        self.assertEqual(self.ids(result), ["lower", "between"])
        self.assertTrue(all(self.base + 20 <= n["available_us"] < self.base + 30 for n in result))
        self.assertEqual(catalog.news_before([], self.base + 30, initial_limit=0), [])

    def test_unverified_news_is_never_prompt_context(self):
        flags = self.news("flag-only", 10)
        flags["availability_evidence"] = []
        unverified = self.news("unverified", 10)
        unverified["historical_availability_verified"] = False
        catalog = self.catalog([flags, unverified])
        self.assertEqual(catalog.news_before([], self.base + 100), [])
        self.assertEqual(len(catalog.audit_catalog()["news"]), 2)

    def test_direct_fixture_link_is_gated_independently(self):
        catalog = self.catalog([self.news("direct", 10, fixture=self.fixture,
            link_offset=30, global_scope=False)])
        self.assertEqual(catalog.news_before([self.fixture], self.base + 30), [])
        result = catalog.news_before([self.fixture], self.base + 31)
        self.assertEqual(self.ids(result), ["direct"])
        self.assertEqual(result[0]["available_us"], self.base + 30)
        self.assertEqual(catalog.news_before([self.other_fixture], self.base + 31), [])

    def test_global_relevance_does_not_require_future_fixture_link(self):
        catalog = self.catalog([self.news("both", 10, fixture=self.fixture, link_offset=30)])
        before = catalog.news_before([self.fixture], self.base + 20)
        self.assertEqual(before[0]["scope_available_us"], {"world_cup": self.base + 10})
        after = catalog.news_before([self.fixture], self.base + 31, since_us=self.base + 20)
        self.assertEqual(after[0]["scope_available_us"], {self.fixture: self.base + 30})
        self.assertEqual(before[0]["headline_key"], after[0]["headline_key"])

    def test_fixture_identity_is_not_known_before_initial_market_event(self):
        known = _micros(self.contract["fixture_initialized_at_utc"])
        early = known - self.base - 100
        catalog = self.catalog([self.news("early", early, fixture=self.fixture, global_scope=False)])
        self.assertEqual(catalog.news_before([self.fixture], known), [])
        self.assertEqual(catalog.news_before([self.fixture], known + 1)[0]["available_us"], known)

    def test_retrospective_team_links_are_not_promoted_to_direct_context(self):
        row = self.news("team-only", 10, fixture=self.fixture, global_scope=False, direct=False)
        catalog = self.catalog([row])
        self.assertEqual(catalog.news_before([self.fixture], self.base + 20), [])
        row["fixture_links"][0]["relevance_basis"] = "archived_direct_game_url"
        catalog = self.catalog([row])
        self.assertEqual(self.ids(catalog.news_before([self.fixture], self.base + 20)), ["team-only"])

    def test_version_and_headline_dedup_use_only_eligible_history(self):
        catalog = self.catalog([
            self.news("v0", 10, item="item", title="World Cup original"),
            self.news("v1", 30, item="item", rank=1, title="World Cup updated"),
            self.news("repeat", 40, title="  world   cup UPDATED ")])
        self.assertEqual(self.ids(catalog.news_before([], self.base + 30)), ["v0"])
        self.assertEqual(self.ids(catalog.news_before([], self.base + 31)), ["v1"])
        self.assertEqual(self.ids(catalog.news_before([], self.base + 41)), ["repeat"])

    def test_delayed_older_version_cannot_revert_an_already_eligible_version(self):
        catalog = self.catalog([
            self.news("v1", 10, item="item", rank=1, title="World Cup new"),
            self.news("v0", 30, item="item", title="World Cup old")])
        self.assertEqual(catalog.news_before([], self.base + 40, since_us=self.base + 20), [])

    def test_delayed_fixture_link_cannot_revert_newer_global_version(self):
        catalog = self.catalog([
            self.news("v1", 10, item="item", rank=1, title="World Cup new"),
            self.news("v0", 5, item="item", title="Old fixture news", fixture=self.fixture,
                      link_offset=30, global_scope=False)])
        self.assertEqual(catalog.news_before([self.fixture], self.base + 40, since_us=self.base + 20), [])

    def test_duplicate_id_version_fixture_link_and_unknown_fixture_are_rejected(self):
        for records, message in (
            ([self.news("same", 10), self.news("same", 20)], "Duplicate news_id"),
            ([self.news("one", 10, item="item"), self.news("two", 20, item="item")], "Ambiguous news item/version"),
            ([self.news("unknown", 10, fixture="espn:999999")], "Unknown fixture")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.catalog(records)
        row = self.news("duplicate-link", 10, fixture=self.fixture)
        row["fixture_links"].append(copy.deepcopy(row["fixture_links"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate fixture/news link"):
            self.catalog([row])

    def test_audit_and_compact_context_keep_provenance_out_of_prompt(self):
        catalog = self.catalog([self.news("one", 10)])
        full = catalog.news_before([], self.base + 20)[0]
        compact = catalog.compact_news(full)
        self.assertEqual(set(compact), {"news_id", "headline", "available_at"})
        self.assertEqual(compact["news_id"], full["short_id"])
        audit = catalog.audit_catalog()
        self.assertIsNone(audit["private_beliefs"])
        self.assertFalse(audit["actor_news_exposure_verified"])
        self.assertEqual(audit["news"][0]["source_url"], "https://example.test/one")
        self.assertEqual(audit["contracts"][0]["outcomes"], {"123": "Yes", "456": "No"})

    def test_invalid_query_arguments_and_alternative_compression(self):
        catalog = self.catalog()
        for args in ((["unknown"], self.base), ([], True)):
            with self.assertRaises(ValueError):
                catalog.news_before(*args)
        with self.assertRaises(ValueError):
            catalog.news_before([], self.base, since_us=self.base + 1)
        with self.assertRaises(ValueError):
            catalog.news_before([], self.base, initial_limit=-1)
        with self.assertRaises(ValueError):
            ContextCatalog(self.root, max_initial_news=True)
        source = self.root / "news_catalog/news_for_sft.jsonl"
        with gzip.open(str(source) + ".gz", "wt") as stream:
            stream.write(source.read_text())
        source.unlink()
        self.assertEqual(ContextCatalog(self.root).audit_catalog()["news"], [])


if __name__ == "__main__":
    unittest.main()
