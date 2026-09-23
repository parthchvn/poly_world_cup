import copy
import unittest

from poly_world_cup.attribution import _link_time, _verified_news_time
from poly_world_cup.news_relevance import build_news_relevance


FIXTURE = {"fixture_id": "espn:1", "home_team": "Mexico", "away_team": "South Africa"}


def news(title, nid="n1", time="2026-06-10T12:00:00Z", **changes):
    row = {"news_id": nid, "news_item_id": nid, "version_rank": 0, "title": title,
           "source_url": "https://example.test/news/article", "historical_availability_verified": True,
           "availability_upper_utc": time, "historical_content_sha256": "a" * 64,
           "availability_evidence": [{"kind": "archive_snapshot", "captured_at_utc": time,
                                      "source_url": "https://archive.test/capture", "content_sha256": "a" * 64}],
           "fixture_ids": [], "fixture_links": []}
    row.update(changes)
    return row


class NewsRelevanceTests(unittest.TestCase):
    def classify(self, title, **changes):
        return build_news_relevance([news(title, **changes)], [FIXTURE])

    def test_explicit_preview_is_historical_at_capture_not_publication(self):
        result = self.classify("Mexico vs. South Africa: 2026 World Cup team news", published_at_utc="2026-06-01T00:00:00Z")
        link = result["news_records"][0]["fixture_links"][0]
        self.assertEqual(link["link_availability_upper_utc"], "2026-06-10T12:00:00Z")
        self.assertTrue(link["direct_fixture_relevance_verified"])
        self.assertEqual(_link_time(link, _verified_news_time(result["news_records"][0])), result["direct_links"][0]["eligible_us"])

    def test_reverse_pair_and_captured_url_scope(self):
        result = self.classify("South Africa versus Mexico team news", source_url="https://example.test/world-cup-2026/preview")
        self.assertEqual(len(result["direct_links"]), 1)

    def test_all_versus_delimiters(self):
        for delimiter in ["v", "vs", "vs.", "versus", "against", "meets", "faces", "takes on"]:
            with self.subTest(delimiter=delimiter):
                self.assertEqual(len(self.classify(f"Mexico {delimiter} South Africa World Cup preview")["direct_links"]), 1)

    def test_explicit_dash_pair_is_distinct_from_comma_list(self):
        for delimiter in ["-", " - ", "\u2013", " \u2014 "]:
            with self.subTest(delimiter=delimiter):
                self.assertEqual(len(self.classify(f"Mexico{delimiter}South Africa World Cup preview")["direct_links"]), 1)
        self.assertEqual(self.classify("Mexico, South Africa World Cup squads")["direct_links"], [])
        self.assertEqual(self.classify("Mexico South Africa World Cup squads")["direct_links"], [])
        self.assertEqual(self.classify("Mexico-South Africa World Cup diplomatic relations")["direct_links"], [])

    def test_tactical_preview_requires_the_explicit_preview_cue(self):
        self.assertEqual(len(self.classify("World Cup tactical preview: How South Africa can beat Mexico")["direct_links"]), 1)
        self.assertEqual(self.classify("How South Africa can beat Mexico in a potential World Cup final")["direct_links"], [])
        self.assertEqual(self.classify("How South Africa can beat Mexico at the World Cup")["direct_links"], [])

    def test_team_mentions_and_opponent_lists_are_not_matchups(self):
        for title in ["World Cup: Mexico and South Africa train", "World Cup: Mexico, Canada vs. Switzerland, South Africa",
                      "World Cup: Mexico star versus South Africa goalkeeper", "New Mexico vs South Africa World Cup preview"]:
            with self.subTest(title=title):
                self.assertEqual(self.classify(title)["direct_links"], [])

    def test_other_competition_edition_and_hypothetical_rejected(self):
        for extra in ["2022 World Cup", "Women's World Cup", "World Cup qualifier", "World Cup warmup friendly",
                      "World Cup 2030", "rugby World Cup", "possible World Cup final", "World Cup head-to-head",
                      "World Cup classic clashes", "World Cup rivalry", "World Cup U17", "World Cup Finalissima",
                      "would be a thrilling World Cup final", "baseball World Cup", "World Cup simulation",
                      "the World Cup final everyone wants", "World Cup dream final", "World Cup EA FC simulation"]:
            with self.subTest(extra=extra):
                self.assertEqual(self.classify("Mexico vs South Africa " + extra)["direct_links"], [])

    def test_old_archive_not_given_current_edition(self):
        self.assertEqual(self.classify("Mexico vs South Africa World Cup", time="2022-06-10T12:00:00Z")["direct_links"], [])

    def test_current_capture_not_promoted(self):
        self.assertEqual(self.classify("Mexico vs South Africa World Cup", historical_availability_verified=False)["direct_links"], [])

    def test_no_evidence_and_unordered_revision_rejected(self):
        for changes in [{"availability_evidence": []}, {"version_rank": 1, "version_order_historically_verified": False}]:
            self.assertEqual(self.classify("Mexico vs South Africa World Cup", **changes)["direct_links"], [])

    def test_ambiguous_repeated_fixture_pair_rejected(self):
        result = build_news_relevance([news("Mexico vs South Africa World Cup preview")], [FIXTURE, {**FIXTURE, "fixture_id": "espn:2"}])
        self.assertEqual(result["direct_links"], [])

    def test_multiple_actual_matchups_not_linked_as_one(self):
        result = build_news_relevance([news("Mexico vs South Africa and Spain vs Argentina World Cup previews")],
                                      [FIXTURE, {"fixture_id": "espn:2", "home_team": "Spain", "away_team": "Argentina"}])
        self.assertEqual(result["direct_links"], [])

    def test_explicit_semicolon_matchups_get_individual_links(self):
        result = build_news_relevance([news("World Cup round of 32: Mexico vs South Africa; Spain vs Argentina")],
            [FIXTURE, {"fixture_id": "espn:2", "home_team": "Spain", "away_team": "Argentina"}])
        self.assertEqual({link["fixture_id"] for link in result["direct_links"]}, {"espn:1", "espn:2"})
        self.assertTrue(all(link["link"]["explicit_semicolon_matchup_list"] for link in result["direct_links"]))

    def test_background_waits_for_pairing_and_stays_separate(self):
        earlier = news("Mexico World Cup injury news", "early", "2026-06-01T00:00:00Z")
        pairing = news("Mexico vs South Africa World Cup preview", "pair")
        result = build_news_relevance([earlier, pairing], [FIXTURE])
        self.assertEqual(result["news_records"][0]["fixture_links"], [])
        bg = result["team_background_links"][0]
        self.assertFalse(bg["direct_fixture_relevance_verified"])
        self.assertEqual(bg["pairing_evidence_news_id"], "pair")
        self.assertEqual(bg["link_availability_upper_utc"], pairing["availability_upper_utc"])
        self.assertEqual(bg["eligible_us"], _link_time(bg, _verified_news_time(earlier)))

    def test_final_registry_pair_does_not_activate_background(self):
        result = build_news_relevance([news("Mexico World Cup squad")], [FIXTURE])
        self.assertEqual(result["team_background_links"], [])
        self.assertEqual(result["fixture_pairing_evidence"], {})

    def test_postgame_report_keeps_late_capture(self):
        result = self.classify("Mexico vs South Africa World Cup player ratings after opening win", time="2026-06-12T06:00:00Z")
        self.assertEqual(result["direct_links"][0]["link"]["link_availability_upper_utc"], "2026-06-12T06:00:00Z")

    def test_input_unchanged_and_candidate_links_preserved(self):
        row = news("Mexico vs South Africa World Cup preview", fixture_ids=["espn:2"],
                   fixture_links=[{"fixture_id": "espn:2", "historical_link_verified": False}])
        original = copy.deepcopy(row)
        result = build_news_relevance([row], [FIXTURE])
        self.assertEqual(row, original)
        self.assertEqual(result["news_records"][0]["fixture_ids"], ["espn:1", "espn:2"])

    def test_verified_background_is_upgraded_only_with_direct_evidence(self):
        row = news("Mexico vs South Africa World Cup preview")
        row["fixture_links"] = [{"fixture_id": "espn:1", "historical_link_verified": True,
            "relationship": "team_background", "link_type": "team_background",
            "link_availability_upper_utc": row["availability_upper_utc"],
            "link_availability_evidence": row["availability_evidence"]}]
        row["fixture_ids"] = ["espn:1"]
        result = build_news_relevance([row], [FIXTURE])
        self.assertEqual(result["news_records"][0]["fixture_links"][0]["relationship"], "direct_match")

    def test_existing_verified_direct_game_link_is_preserved(self):
        row = news("Mexico vs South Africa World Cup preview")
        link = {"fixture_id": "espn:1", "historical_link_verified": True,
            "relevance_basis": "archived_direct_game_url",
            "link_availability_upper_utc": row["availability_upper_utc"],
            "link_availability_evidence": row["availability_evidence"]}
        row["fixture_links"] = [link]
        row["fixture_ids"] = ["espn:1"]
        result = build_news_relevance([row], [FIXTURE])
        self.assertEqual(result["news_records"][0]["fixture_links"][0], link)

    def test_aliases_accents_and_registry_objects(self):
        fixtures = [{"fixture_id": "espn:3", "home_team": {"name": "Türkiye"}, "away_team": {"name": "United States"}}]
        result = build_news_relevance([news("World Cup: USMNT vs. Turkey lineup")], {"fixtures": fixtures})
        self.assertEqual(result["direct_links"][0]["fixture_id"], "espn:3")
        result = build_news_relevance([news("Lineup and Predictions for Tunisia vs the Netherlands' World Cup Game")],
            [{"fixture_id": "espn:4", "home_team": "Tunisia", "away_team": "Netherlands"}])
        self.assertEqual(result["direct_links"][0]["fixture_id"], "espn:4")
        result = build_news_relevance([news("IR Iran vs New Zealand World Cup preview")],
            [{"fixture_id": "espn:5", "home_team": "Iran", "away_team": "New Zealand"}])
        self.assertEqual(result["direct_links"][0]["fixture_id"], "espn:5")

    def test_duplicate_news_identity_refused(self):
        with self.assertRaisesRegex(ValueError, "News IDs"):
            build_news_relevance([news("a"), news("b")], [FIXTURE])

    def test_chain_pairing_gates_background_without_creating_direct_news(self):
        from test_historical_contracts import contract_record_fixture
        contract = contract_record_fixture()
        fixture = {**FIXTURE, "fixture_id": contract["fixture_id"]}
        result = build_news_relevance([news("Mexico World Cup injury news", time="2026-06-01T00:00:00Z")],
            [fixture], contract_records=[contract])
        self.assertEqual(result["direct_links"], [])
        bg = result["team_background_links"][0]
        self.assertEqual(bg["link_availability_upper_utc"], contract["fixture_initialized_at_utc"])
        self.assertEqual(bg["pairing_evidence_id"], contract["evidence_id"])
        self.assertEqual(bg["pairing_evidence_basis"], "polygon_market_prepared")
        self.assertFalse(bg["direct_fixture_relevance_verified"])

    def test_tampered_chain_pairing_time_and_wrong_fixture_rejected(self):
        from test_historical_contracts import contract_record_fixture
        contract = contract_record_fixture()
        contract["fixture_initialized_at_utc"] = "2026-01-01T00:00:00Z"
        fixture = {**FIXTURE, "fixture_id": contract["fixture_id"]}
        with self.assertRaisesRegex(ValueError, "derived field"):
            build_news_relevance([], [fixture], contract_records=[contract])
        contract = contract_record_fixture()
        with self.assertRaisesRegex(ValueError, "unique fixture pair"):
            build_news_relevance([], [{**fixture, "away_team": "Canada"}], contract_records=[contract])

    def test_chain_condition_must_belong_to_fixture_in_registry(self):
        from test_historical_contracts import contract_record_fixture
        contract = contract_record_fixture()
        registry = {"fixtures": [{**FIXTURE, "fixture_id": contract["fixture_id"]}],
                    "contracts": [{"condition_id": contract["condition_id"], "fixture_id": "wrong"}]}
        with self.assertRaisesRegex(ValueError, "registry"):
            build_news_relevance([], registry, contract_records=[contract])

    def reviewed(self):
        from test_historical_contracts import contract_record_fixture
        contract = contract_record_fixture()
        fixture = {**FIXTURE, "fixture_id": contract["fixture_id"]}
        row = news("Mexico goalkeeper cleared for World Cup opener against South Africa")
        annotation = {key: row[key] for key in ("news_id", "title", "source_url", "historical_content_sha256", "availability_upper_utc")}
        annotation.update(fixture_id=fixture["fixture_id"], rationale="The headline identifies the opponent for Mexico's World Cup opener.", requires_validated_polygon_pairing=True)
        return row, fixture, contract, {"schema_version": 1, "review_status": "reviewed_by_independent_assistant", "links": [annotation]}

    def test_reviewed_nonadjacent_matchup_has_auditable_source_and_pairing(self):
        row, fixture, contract, manifest = self.reviewed()
        result = build_news_relevance([row], [fixture], contract_records=[contract], reviewed_links=manifest)
        link = result["news_records"][0]["fixture_links"][0]
        self.assertEqual(link["relationship"], "direct_match")
        self.assertEqual(link["relevance_basis"], "independently_reviewed_captured_matchup")
        self.assertEqual(link["pairing_evidence_id"], contract["evidence_id"])
        self.assertIn("reviewed_link_manifest_sha256", link)

    def test_reviewed_link_requires_exact_version_completed_review_and_chain_proof(self):
        row, fixture, contract, manifest = self.reviewed()
        with self.assertRaisesRegex(ValueError, "Polygon pairing"):
            build_news_relevance([row], [fixture], reviewed_links=manifest)
        pending = copy.deepcopy(manifest); pending["review_status"] = "pending"
        with self.assertRaisesRegex(ValueError, "completed independent review"):
            build_news_relevance([row], [fixture], contract_records=[contract], reviewed_links=pending)
        for field in ("title", "source_url", "historical_content_sha256", "availability_upper_utc"):
            tampered = copy.deepcopy(manifest); tampered["links"][0][field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact captured news"):
                build_news_relevance([row], [fixture], contract_records=[contract], reviewed_links=tampered)

    def test_review_does_not_override_future_pairing_time(self):
        row, fixture, contract, manifest = self.reviewed()
        row = news(row["title"], time="2026-06-01T00:00:00Z")
        manifest["links"][0]["availability_upper_utc"] = row["availability_upper_utc"]
        result = build_news_relevance([row], [fixture], contract_records=[contract], reviewed_links=manifest)
        self.assertEqual(result["news_records"][0]["fixture_links"][0]["link_availability_upper_utc"], contract["fixture_initialized_at_utc"])


if __name__ == "__main__":
    unittest.main()
