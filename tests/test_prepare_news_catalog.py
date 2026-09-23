import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from scripts.prepare_news_catalog import prepare, jsonl, sha, write_jsonl
from test_sft import REGISTRY, POLICY, trade, news


class PrepareNewsCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "cohort.sqlite"
        self.output = self.root / "catalog"
        self.cache = self.root / "target_times.json.gz"
        self.gdelt = self.root / "gdelt.jsonl"
        self.registry = copy.deepcopy(REGISTRY)
        for fixture, pair in zip(self.registry["fixtures"], [("Mexico", "South Africa"), ("Spain", "Argentina"), ("France", "England")]):
            fixture.update(home_team=pair[0], away_team=pair[1])
        page = self.root / "trades.jsonl"
        write_jsonl(page, [trade("target")])
        build_attribution_index(registry=self.registry, trade_pages=[page],
            news_records=[news("old-unverified", verified=False)], output_path=self.database)
        direct, broad = news("gdelt:direct"), news("gdelt:broad")
        for row in (direct, broad):
            row["fixture_ids"], row["fixture_links"] = [], []
        direct["title"] = "Mexico vs South Africa World Cup preview"
        broad["title"] = "World Cup preparations continue"
        write_jsonl(self.gdelt, [direct, broad])

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self):
        return prepare(self.database, self.registry, self.gdelt, self.output,
            split_policy=POLICY, target_times_cache=self.cache)

    def test_only_direct_additions_train_old_candidates_and_all_evidence_survive(self):
        original_sha = sha(self.database)
        result = self.run_prepare()
        rows = jsonl(self.output / "news_for_sft.jsonl")
        self.assertEqual({row["news_id"] for row in rows}, {"old-unverified", "gdelt:direct"})
        self.assertEqual(next(row for row in rows if row["news_id"] == "gdelt:direct")["context_scope"], "fixture")
        self.assertEqual(len(jsonl(self.output / "all_news_candidates.jsonl.gz")), 3)
        self.assertEqual(result["eligible_target_count"], 1)
        self.assertEqual(result["eligible_targets_with_selected_prior_direct_fixture_news"], 1)
        self.assertTrue(result["deduplicate_headlines"])
        self.assertEqual(sha(self.database), original_sha)
        self.assertTrue(self.cache.exists())
        with self.assertRaises(FileExistsError):
            self.run_prepare()

    def test_cached_targets_bound_to_source_and_policy(self):
        self.run_prepare()
        with gzip.open(self.cache, "rt") as stream:
            cache = json.load(stream)
        cache["source_database_sha256"] = "bad"
        self.cache.unlink()
        write_jsonl(self.cache, [cache])
        self.output = self.root / "second"
        with self.assertRaisesRegex(ValueError, "different cohort"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_item_version_rejected_before_output(self):
        rows = jsonl(self.gdelt)
        rows[1]["news_item_id"] = rows[0]["news_item_id"]
        self.gdelt.unlink()
        write_jsonl(self.gdelt, rows)
        with self.assertRaisesRegex(ValueError, "item/version"):
            self.run_prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
