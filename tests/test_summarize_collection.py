import hashlib
import unittest

from scripts.summarize_collection import compose_summary, render_readme


def finished_inputs():
    fixtures, contracts, conditions, coverage = [], [], [], []
    for index in range(104):
        fid = f"fixture:{index:03}"
        fixtures.append({"fixture_id": fid, "mapping_status": "matched", "home_team": {"name": f"Home {index}"},
                         "away_team": {"name": f"Away {index}"}, "kickoff_utc": "2026-06-11T19:00:00Z", "stage": "test"})
        coverage.append({"fixture_id": fid, "candidate_news_versions": 0, "claimed_pregame_versions": 0,
                         "source_status": "no_event_article"})
        for selection in range(3):
            identifier = index * 3 + selection + 1
            condition = "0x" + f"{identifier:064x}"
            contracts.append({"fixture_id": fid, "condition_id": condition,
                              "tokens": [{"token_id": str(identifier * 2)}, {"token_id": str(identifier * 2 + 1)}]})
            conditions.append({"condition_id": condition, "status": "exhausted", "api_traversal_status": "exhausted",
                               "integrity_validated": True, "row_count": 2, "page_count": 1,
                               "earliest_block_timestamp": "2026-04-10T01:00:00Z", "latest_block_timestamp": "2026-06-11T22:00:00Z"})
    condition_hash = hashlib.sha256("\n".join(sorted(x["condition_id"] for x in contracts)).encode()).hexdigest()
    return {"registry": {"fixtures": fixtures, "contracts": contracts},
        "batch": {"conditions": conditions, "condition_count": 312, "status": "api_exhausted",
                  "finished_at": "2026-09-23T12:00:00Z", "started_at": "2026-09-23T01:00:00Z",
                  "status_counts": {"exhausted": 312, "failed": 0}, "committed_observation_count": 624,
                  "validated_observation_count": 624, "committed_page_count": 312,
                  "condition_set_sha256": condition_hash, "requested_minimum_size_tokens": "0.000001"},
        "provenance": {"raw_provenance_verified": True, "errors": [], "condition_selection": "explicit",
                       "requested_condition_count": 312, "verified_condition_count": 312,
                       "normalized_integrity_verified_conditions": 312, "api_exhausted_conditions": 312,
                       "verified_observation_count": 624, "committed_page_count": 312, "verified_raw_page_count": 312},
        "attribution": {"registry_condition_count": 312, "api_traversals_exhausted": 312, "partial_collection_allowed": False,
                        "observation_count": 624, "source_page_count": 312, "distinct_wallet_count": 10,
                        "token_mapping_counts": {"exact_current_registry_token": 624}, "news_count": 0,
                        "rejected_historical_availability_claims": 0, "fixtures_with_observations": 104,
                        "historically_eligible_fixture_news_links": 0, "historically_eligible_global_news_versions": 0,
                        "observations_with_verified_prior_news": 0, "observations_with_verified_prior_global_news": 0},
        "news_coverage": {"metadata_versions": 0, "unique_articles": 0, "in_window_metadata_versions": 0,
                          "fixture_coverage": coverage, "window_start_utc": "2026-01-01T00:00:00Z",
                          "window_end_utc": "2026-07-20T00:00:00Z", "fixture_source_errors": 0},
        "archive_report": {"completed_articles": 0, "requested_articles": 0, "statuses": {},
                           "verified_headline_versions": 0, "fixture_links_verified": 0},
        "current_news": [], "archived_news": [],
        "receipt_sample": {"sample_collection_status": "completed", "results": [
            {"observation_id": "a", "status": "ambiguous_multiple_log_candidates"}],
            "status_counts": {"ambiguous_multiple_log_candidates": 1}},
        "earlier_probe": {"results": [{"observation_id": "b", "status": "amount_or_price_mismatch"}]}}


class CollectionSummaryTests(unittest.TestCase):
    def test_finished_counts_keep_completeness_and_training_caveats(self):
        result = compose_summary(**finished_inputs(), code_revision="a" * 40)
        self.assertEqual(result["trades"]["observation_count"], 624)
        self.assertEqual(len(result["fixture_coverage"]), 104)
        self.assertEqual(sum(x["observation_count"] for x in result["fixture_coverage"]), 624)
        self.assertEqual(result["trades"]["requested_minimum_size_tokens"], "0.000001")
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["trades"]["source_completeness_certified"])
        self.assertFalse(result["news"]["complete_news_coverage"])
        self.assertEqual(result["reconciliation"]["earlier_diagnostic_probe"]["status_counts"], {"amount_or_price_mismatch": 1})
        readme = render_readme(result)
        self.assertIn("**This is not yet an SFT-ready dataset.**", readme)
        self.assertIn("624", readme)
        self.assertIn("amount_or_price_mismatch", readme)

    def test_each_independent_count_mismatch_refuses_completion(self):
        mutations = [("batch", "committed_observation_count", 623),
                     ("batch", "validated_observation_count", 623),
                     ("batch", "committed_page_count", 311),
                     ("provenance", "verified_observation_count", 623),
                     ("provenance", "verified_raw_page_count", 311),
                     ("provenance", "verified_condition_count", 311),
                     ("attribution", "observation_count", 623),
                     ("attribution", "source_page_count", 311),
                     ("attribution", "registry_condition_count", 311),
                     ("attribution", "news_count", 1)]
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                inputs = finished_inputs()
                inputs[group][key] = value
                with self.assertRaises(ValueError):
                    compose_summary(**inputs)

    def test_running_condition_cannot_hide_in_successful_aggregate_counts(self):
        inputs = finished_inputs()
        inputs["batch"]["conditions"][0]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "condition lacks validated"):
            compose_summary(**inputs)

    def test_wrong_condition_universe_rejected_even_when_count_is_312(self):
        inputs = finished_inputs()
        inputs["batch"]["conditions"][0]["condition_id"] = "0x" + "f" * 64
        with self.assertRaisesRegex(ValueError, "do not match"):
            compose_summary(**inputs)

    def test_discovered_only_provenance_is_not_full_registry_proof(self):
        inputs = finished_inputs()
        inputs["provenance"]["condition_selection"] = "discovered_manifests"
        with self.assertRaisesRegex(ValueError, "explicit registry"):
            compose_summary(**inputs)

    def test_incomplete_archive_lookup_rejected(self):
        inputs = finished_inputs()
        inputs["archive_report"]["requested_articles"] = 10
        with self.assertRaisesRegex(ValueError, "lookup has not finished"):
            compose_summary(**inputs)

    def test_perfixture_news_mismatch_rejected(self):
        inputs = finished_inputs()
        inputs["news_coverage"]["fixture_coverage"][0]["candidate_news_versions"] = 1
        with self.assertRaisesRegex(ValueError, "candidate news count disagrees"):
            compose_summary(**inputs)

    def test_bounds_are_true_observed_extrema(self):
        inputs = finished_inputs()
        inputs["batch"]["conditions"][0]["earliest_block_timestamp"] = "2026-04-07T00:00:00Z"
        inputs["batch"]["conditions"][-1]["latest_block_timestamp"] = "2026-07-21T00:00:00Z"
        result = compose_summary(**inputs)
        self.assertEqual(result["trades"]["earliest_block_timestamp"], "2026-04-07T00:00:00Z")
        self.assertEqual(result["trades"]["latest_block_timestamp"], "2026-07-21T00:00:00Z")
        self.assertFalse(result["news"]["window_covers_observed_trade_interval"])

    def test_receipt_status_counts_cannot_omit_a_mismatch(self):
        inputs = finished_inputs()
        inputs["receipt_sample"]["results"].append({"observation_id": "c", "status": "amount_or_price_mismatch"})
        with self.assertRaisesRegex(ValueError, "Receipt status counts disagree"):
            compose_summary(**inputs)

    def test_code_revision_requires_full_sha(self):
        with self.assertRaisesRegex(ValueError, "full Git SHA"):
            compose_summary(**finished_inputs(), code_revision="main")


if __name__ == "__main__":
    unittest.main()
