import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from urllib.parse import urlencode

from poly_world_cup.registry import ESPN_SCOREBOARD, canonical_team, discover_registry


def fixture(identifier="1", home="France", away="Senegal", kickoff="2026-06-17T01:00Z"):
    return {
        "id": identifier, "date": kickoff, "season": {"slug": "group-stage"},
        "competitions": [{"competitors": [
            {"homeAway": "home", "score": "4", "winner": True, "form": "WWWWW",
             "team": {"id": "10", "displayName": home}},
            {"homeAway": "away", "score": "0", "winner": False,
             "team": {"id": "11", "displayName": away}},
        ]}],
    }


def event(identifier="1", home="France", away="Senegal", slug="fifwc-fra-sen-2026-06-16", kickoff="2026-06-17T01:00:00Z"):
    number = int(identifier)
    markets = []
    for i, title in enumerate([home, f"Draw ({home} vs. {away})", away]):
        key = number * 10 + i
        markets.append({
            "id": str(key), "groupItemTitle": title, "sportsMarketType": "moneyline",
            "question": f"Will {title} win?", "conditionId": "0x" + f"{key:064x}",
            "outcomes": '["No", "Yes"]', "clobTokenIds": json.dumps([str(2 * key), str(2 * key + 1)]),
            "description": "This refers to the first 90 minutes of regular play plus stoppage time.",
            "createdAt": "2026-04-01T12:00:00Z", "startDate": "2026-04-01T12:01:00Z",
            "acceptingOrdersTimestamp": "2026-04-01T12:02:00Z", "gameStartTime": kickoff,
            "outcomePrices": '["1", "0"]', "volume": 200,
        })
    return {"id": identifier, "slug": slug, "startTime": kickoff, "markets": markets,
            "score": "4-0", "description": "France won after scoring four goals."}


class FakeClient:
    def __init__(self, fixtures, pages):
        self.fixtures = fixtures
        self.pages = pages
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        payload = {"events": self.fixtures} if url == ESPN_SCOREBOARD else self.pages[params["offset"]]
        return SimpleNamespace(data=copy.deepcopy(payload), url=url + "?" + urlencode(params),
                               retrieved_at="2026-09-22T00:00:00Z",
                               body_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest())


class RegistryTests(unittest.TestCase):
    def test_team_aliases_match_both_sources(self):
        for a, b in [("Cabo Verde", "Cape Verde"), ("IR Iran", "Iran"),
                     ("Côte d'Ivoire", "Ivory Coast"), ("Korea Republic", "South Korea"),
                     ("DR Congo", "Congo DR"), ("Bosnia and Herzegovina", "Bosnia-Herzegovina")]:
            with self.subTest(a=a):
                self.assertEqual(canonical_team(a), canonical_team(b))

    def test_multiple_short_pages_local_date_and_no_result_leakage(self):
        base = event()
        prop = event("2", slug="fifwc-fra-sen-2026-06-16-exact-score")
        prop["markets"][0]["sportsMarketType"] = "soccer_exact_score"
        client = FakeClient([fixture()], {0: [base], 1: [prop], 2: []})
        result = discover_registry(client)
        self.assertEqual([p["offset"] for url, p in client.calls if url != ESPN_SCOREBOARD], [0, 1, 2])
        self.assertEqual(result["fixtures"][0]["mapping_status"], "matched")
        self.assertEqual(result["fixtures"][0]["kickoff_difference_seconds"], 0)
        self.assertEqual(result["report"]["excluded_other_event_count"], 1)
        self.assertEqual(len(result["contracts"]), 3)
        self.assertEqual(result["contracts"][0]["tokens"], [
            {"outcome_index": 0, "outcome": "No", "token_id": "20"},
            {"outcome_index": 1, "outcome": "Yes", "token_id": "21"},
        ])
        for record in result["fixtures"] + result["contracts"]:
            self.assertFalse(record["feature_eligible"])
            self.assertNotIn("score", record)
            self.assertNotIn("outcomePrices", record)
            self.assertIn("body_sha256", record["source"])
        text = json.dumps(result["fixtures"] + result["contracts"])
        self.assertNotIn("WWWWW", text)
        self.assertNotIn("France won", text)
        self.assertFalse(result["report"]["coverage_complete"])
        self.assertIn("fixture_count_mismatch", [f["code"] for f in result["report"]["failures"]])

    def test_all_fixtures_retained_with_missing_and_ambiguous_events(self):
        fixtures = [fixture(), fixture("2", "Brazil", "Japan"), fixture("3", "England", "Spain")]
        events = [event(), event("2"), event("3", "Brazil", "Japan", "fifwc-bra-jpn-2026-06-16")]
        result = discover_registry(FakeClient(fixtures, {0: events, 3: []}))
        statuses = {f["fixture_id"]: f["mapping_status"] for f in result["fixtures"]}
        self.assertEqual(statuses, {"espn:1": "ambiguous", "espn:2": "matched", "espn:3": "unmatched"})
        self.assertEqual({c["fixture_id"] for c in result["contracts"]}, {"espn:2"})

    def test_one_hour_discrepancy_is_retained_and_flagged(self):
        result = discover_registry(FakeClient([fixture(kickoff="2026-06-17T02:00Z")], {0: [event()], 1: []}))
        self.assertEqual(result["fixtures"][0]["mapping_status"], "matched")
        self.assertEqual(result["fixtures"][0]["kickoff_difference_seconds"], -3600)
        self.assertIn("kickoff_disagreement", [w["code"] for w in result["report"]["warnings"]])

    def test_far_apart_kickoffs_do_not_match(self):
        result = discover_registry(FakeClient([fixture(kickoff="2026-06-18T01:00Z")], {0: [event()], 1: []}))
        self.assertEqual(result["fixtures"][0]["mapping_status"], "unmatched")

    def test_malformed_token_mapping_fails_coverage(self):
        raw = event()
        raw["markets"][0]["clobTokenIds"] = '["20", "20"]'
        result = discover_registry(FakeClient([fixture()], {0: [raw], 1: []}))
        self.assertEqual(result["fixtures"][0]["mapping_status"], "incomplete_contracts")
        self.assertEqual(len(result["contracts"]), 2)
        self.assertFalse(result["report"]["coverage_complete"])

    def test_repeated_page_is_an_error_not_false_completion(self):
        with self.assertRaisesRegex(ValueError, "pagination repeated"):
            discover_registry(FakeClient([fixture()], {0: [event()], 1: [event()]}))

    def test_complete_fixture_universe_with_no_markets_remains_104_rows(self):
        raw = [fixture(str(i)) for i in range(104)]
        result = discover_registry(FakeClient(raw, {0: []}))
        self.assertEqual(len(result["fixtures"]), 104)
        self.assertEqual(result["report"]["mapping_status_counts"], {"unmatched": 104})
        self.assertNotIn("fixture_count_mismatch", [f["code"] for f in result["report"]["failures"]])


if __name__ == "__main__":
    unittest.main()
