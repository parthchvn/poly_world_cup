import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from poly_world_cup.news import (ESPN_ARCHIVE, ESPN_SUMMARY, collect_news,
                                link_articles, merge_articles, normalize_article)

SOURCE = {"url": "https://example.test/api", "retrieved_at": "2026-09-23T00:00:00Z",
          "body_sha256": "a" * 64, "historical_availability_verified": False}
FIXTURE = {"fixture_id": "espn:123", "espn_event_id": "123",
           "earliest_accepting_orders_at": "2026-04-06T00:00:00Z",
           "kickoff_utc": "2026-06-11T19:00:00Z",
           "home_team": {"espn_team_id": "203", "name": "Mexico", "canonical_name": "mexico"},
           "away_team": {"espn_team_id": "467", "name": "South Africa", "canonical_name": "southafrica"}}
REGISTRY = {"fixtures": [FIXTURE]}


def article(identifier=1, **overrides):
    item = {"id": identifier, "headline": "Mexico announces its squad", "published": "2026-05-01T12:00:00Z",
            "lastModified": "2026-05-01T13:00:00Z", "categories": [{"type": "team", "teamId": 203}],
            "links": {"web": {"href": "https://example.test/" + str(identifier)}}}
    item.update(overrides)
    return item


def normalized(**overrides):
    return normalize_article(article(**overrides), SOURCE, origin="league_archive")


def link(rows):
    return link_articles(rows, REGISTRY, window_start="2026-04-06T00:00:00Z",
                         window_end="2026-07-20T00:00:00Z")


class FakeClient:
    def __init__(self, pages, summaries=None):
        self.pages, self.summaries, self.calls = pages, summaries or {}, []

    def get_json(self, url, params):
        self.calls.append((url, params))
        if url == ESPN_SUMMARY:
            data = self.summaries.get(params["event"], {})
        else:
            data = self.pages[params["offset"]]
        if isinstance(data, Exception):
            raise data
        return SimpleNamespace(data=data, url=url + "?" + str(params), retrieved_at=SOURCE["retrieved_at"],
                               body_sha256="a" * 64)


def page(offset, rows):
    return {"status": "success", "resultsOffset": offset, "headlines": rows}


class NewsTests(unittest.TestCase):
    def test_current_capture_never_verifies_old_publication(self):
        row = normalized(story="A full copyrighted body.", description="A publisher snippet.")
        self.assertFalse(row["historical_availability_verified"])
        self.assertFalse(row["feature_eligible"])
        self.assertIsNone(row["availability_upper_utc"])
        self.assertNotIn("story", row)
        self.assertNotIn("description", row)
        self.assertEqual(row["published_at_utc"], "2026-05-01T12:00:00Z")
        self.assertEqual(row["captured_at_utc"], "2026-09-23T00:00:00Z")

    def test_unknown_timestamp_stays_unknown_and_unlinked(self):
        row = link([normalized(published="2026-05-01")])[0]
        self.assertIsNone(row["published_at_utc"])
        self.assertEqual(row["publication_precision"], "unknown")
        self.assertFalse(row["within_study_window"])
        self.assertEqual(row["fixture_ids"], [])

    def test_team_category_is_relevance_not_actor_exposure(self):
        row = link([normalized()])[0]
        self.assertEqual(row["fixture_ids"], ["espn:123"])
        self.assertEqual(row["fixture_links"][0]["relationship"], "team_category_context")
        self.assertFalse(row["fixture_links"][0]["historical_link_verified"])
        self.assertFalse(row["actor_exposure_verified"])
        self.assertFalse(row["causal_attribution"])

    def test_historical_fixture_pairing_is_not_assumed_known(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["stage"] = "final"
        rows = link_articles([normalized()], {"fixtures": [fixture]}, window_start="2026-04-06T00:00:00Z", window_end="2026-07-20T00:00:00Z")
        self.assertFalse(rows[0]["fixture_links"][0]["historical_link_verified"])
        self.assertIsNone(rows[0]["fixture_links"][0]["link_availability_upper_utc"])

    def test_fixture_context_window_is_half_open(self):
        before = normalized(published="2026-04-05T23:59:59Z")
        opening = normalized(published="2026-04-06T00:00:00Z")
        close = normalized(published="2026-06-12T01:00:00Z")
        self.assertEqual([len(row["fixture_ids"]) for row in link([before, opening, close])], [0, 1, 0])

    def test_premarket_background_has_a_separate_relationship(self):
        rows = link_articles([normalized(published="2026-01-10T00:00:00Z"),
                              normalized(published="2026-01-01T00:00:00Z")],
                             REGISTRY, window_start="2026-01-01T00:00:00Z",
                             window_end="2026-07-20T00:00:00Z")
        self.assertEqual(rows[0]["fixture_links"][0]["relationship"], "team_category_background")
        self.assertFalse(rows[0]["feature_eligible"])
        self.assertEqual(rows[1]["fixture_ids"], [])

    def test_league_widget_is_not_an_event_article(self):
        client = FakeClient({0: page(0, [])}, {"123": {"news": {"articles": [article(gameId="123")]}}})
        with tempfile.TemporaryDirectory() as tmp:
            report = collect_news(client, REGISTRY, Path(tmp))
            self.assertEqual(report["metadata_versions"], 0)
            self.assertEqual(report["fixture_coverage"][0]["source_status"], "no_event_article")
            self.assertEqual(Path(tmp, "news.jsonl").read_text(), "")

    def test_wrong_game_id_does_not_gain_direct_association(self):
        row = normalize_article(article(gameId="999"), SOURCE, origin="fixture_summary_article", direct_fixture=FIXTURE)
        self.assertEqual(row["direct_fixture_ids"], [])
        self.assertNotEqual(link([row])[0]["fixture_links"][0]["relationship"], "explicit_game_id")

    def test_versions_preserve_changes_and_merge_duplicate_captures(self):
        first = normalized()
        duplicate = normalize_article(article(), {**SOURCE, "retrieved_at": "2026-09-24T00:00:00Z"}, origin="fixture_summary_article")
        changed = normalized(headline="Mexico changes its squad", lastModified="2026-05-02T12:00:00Z")
        rows = merge_articles([changed, duplicate, first])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["version_rank"] for r in rows], [0, 1])
        self.assertEqual(len(rows[0]["source_provenance"]), 2)
        self.assertEqual(rows[0]["captured_at_utc"], SOURCE["retrieved_at"])
        self.assertEqual(rows[0]["title"], first["title"])
        self.assertFalse(rows[1]["version_order_historically_verified"])

    def test_broad_tournament_context_is_not_expanded_to_all_fixtures(self):
        row = normalized(headline="World Cup expands global broadcast coverage", categories=[{"type": "league", "leagueId": 4}])
        row = link([row])[0]
        self.assertEqual(row["fixture_ids"], [])
        self.assertEqual(row["context_scope"], "tournament")

    def test_headline_matching_has_word_boundaries(self):
        row = normalized(headline="New Mexico university announces event", categories=[])
        # Mexico appears as a distinct word, illustrating why this is only a
        # low-confidence candidate, never an authoritative fixture assignment.
        row = link([row])[0]
        self.assertEqual(row["fixture_links"][0]["confidence"], "low")
        self.assertFalse(row["fixture_links"][0]["historical_link_verified"])
        self.assertEqual(link([normalized(headline="MexicoXYZ", categories=[])])[0]["fixture_ids"], [])

    def test_ignored_offset_is_error_not_complete_coverage(self):
        client = FakeClient({0: page(0, [article()]), 1: page(0, [article()])})
        with tempfile.TemporaryDirectory() as tmp:
            report = collect_news(client, REGISTRY, Path(tmp))
            self.assertEqual(report["archive_status"], "source_error")
            self.assertIn("ignored requested offset", report["errors"][0]["error"])
            self.assertFalse(report["complete_news_coverage"])

    def test_repeated_page_is_detected_even_when_offset_changes(self):
        client = FakeClient({0: page(0, [article()]), 1: page(1, [article()])})
        with tempfile.TemporaryDirectory() as tmp:
            report = collect_news(client, REGISTRY, Path(tmp))
            self.assertEqual(report["archive_status"], "source_error")
            self.assertIn("repeated a page", report["errors"][0]["error"])

    def test_source_failure_keeps_fixture_coverage_row(self):
        client = FakeClient({0: page(0, [])}, {"123": RuntimeError("Unavailable")})
        with tempfile.TemporaryDirectory() as tmp:
            report = collect_news(client, REGISTRY, Path(tmp))
            self.assertEqual(report["fixture_sources_attempted"], 1)
            self.assertEqual(report["fixture_source_errors"], 1)
            self.assertEqual(report["fixture_coverage"][0]["source_status"], "source_error")
            self.assertEqual(report["fixture_coverage"][0]["coverage_status"], "no_candidates")

    def test_two_old_pages_are_a_boundary_not_completeness_certificate(self):
        client = FakeClient({0: page(0, [article(1, published="2026-04-01T00:00:00Z")]),
                             1: page(1, [article(2, published="2026-03-01T00:00:00Z")])})
        with tempfile.TemporaryDirectory() as tmp:
            report = collect_news(client, REGISTRY, Path(tmp), window_start="2026-04-06T00:00:00Z")
            self.assertEqual(report["archive_status"], "crossed_window_start_unverified")
            self.assertEqual(report["in_window_metadata_versions"], 0)
            self.assertEqual(report["metadata_versions"], 2)
            self.assertFalse(report["complete_news_coverage"])
            self.assertEqual(report["historically_verified_versions"], 0)


if __name__ == "__main__":
    unittest.main()
