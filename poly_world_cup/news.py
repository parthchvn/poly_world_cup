"""Historical news metadata collection with explicit retrospective attribution.

A publisher's old publication timestamp does not verify that the headline we
captured today existed then. This module never promotes such metadata to a
point-in-time training feature, and never infers a trader read an article.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Callable

from .io import write_json, write_jsonl

ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
ESPN_ARCHIVE = "https://content.core.api.espn.com/v1/sports/soccer/fifa.world/news"


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result.astimezone(timezone.utc) if result.tzinfo else None


def _utc(value: Any) -> str | None:
    dt = _time(value)
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _source(result: Any) -> dict:
    return {"url": result.url, "retrieved_at": result.retrieved_at,
            "body_sha256": result.body_sha256, "historical_availability_verified": False}


def _text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


# Phrase aliases are explicit and word bounded; short ambiguous abbreviations
# such as US, IR, and SA are deliberately excluded.
ALIASES = {"unitedstates": ["United States", "USMNT", "USA"],
           "southkorea": ["South Korea", "Korea Republic"],
           "ivorycoast": ["Ivory Coast", "Cote d'Ivoire"],
           "capeverde": ["Cape Verde", "Cabo Verde"],
           "turkiye": ["Turkiye", "Turkey"],
           "drcongo": ["DR Congo", "Congo DR", "Democratic Republic of Congo"],
           "bosniaherzegovina": ["Bosnia", "Bosnia and Herzegovina"],
           "czechia": ["Czechia", "Czech Republic"]}


def normalize_article(article: dict, source: dict, *, origin: str,
                      direct_fixture: dict | None = None) -> dict:
    """Select bibliographic metadata only; do not export full text or snippets."""
    article_id = str(article.get("id", ""))
    title = article.get("headline") or article.get("title")
    if not article_id or not isinstance(title, str) or not title.strip():
        raise ValueError("News article needs an ID and a headline")
    links = article.get("links", {})
    source_url = links.get("web", {}).get("href") or links.get("api", {}).get("self", {}).get("href")
    if not isinstance(source_url, str) or not source_url.startswith(("https://", "http://")):
        raise ValueError("News article needs a public source URL")
    published = _utc(article.get("published"))
    modified = _utc(article.get("lastModified"))
    original = _utc(article.get("originallyPosted"))
    # A version identifier describes the observed metadata, never its claimed
    # historical contents. The complete raw response has a separate SHA256.
    version = {"source_article_id": article_id, "title": title.strip(),
               "source_url": source_url, "published_at_utc": published,
               "modified_at_utc": modified}
    version_hash = _hash(version)
    categories = article.get("categories", [])
    team_ids = sorted({str(c.get("teamId", c.get("team", {}).get("id")))
                       for c in categories if c.get("type") == "team"
                       and c.get("teamId", c.get("team", {}).get("id")) is not None})
    league = any(c.get("type") == "league" and
                 (str(c.get("leagueId")) == "4" or str(c.get("league", {}).get("id")) == "606")
                 for c in categories)
    game_id = str(article.get("gameId", "")) or None
    direct = []
    # summary.news is a current league widget. Only summary.article with an
    # agreeing explicit gameId is eligible for direct fixture association.
    if direct_fixture and game_id == str(direct_fixture["espn_event_id"]):
        direct.append(direct_fixture["fixture_id"])
    return {**version, "news_id": f"espn:{article_id}:{version_hash[:16]}",
            "news_item_id": f"espn:{article_id}", "metadata_sha256": version_hash,
            "source_provider": "ESPN", "content_type": article.get("type"),
            "published_at_raw": article.get("published"),
            "publication_precision": "second" if published else "unknown",
            "originally_posted_at_utc": original,
            "captured_at_utc": source["retrieved_at"], "source_provenance": [source],
            "source_team_ids": team_ids, "source_game_id": game_id,
            "tournament_category": league, "collection_origins": [origin],
            "direct_fixture_ids": direct,
            "availability_upper_utc": None, "historical_availability_verified": False,
            "historical_content_sha256": None, "availability_evidence": [],
            "feature_eligible": False, "historical_text_version_status": "unverified_current_capture",
            "fixture_ids": [], "fixture_links": [],
            "actor_exposure_verified": False, "causal_attribution": False,
            "full_article_text_collected_in_export": False}


def merge_articles(rows: list[dict]) -> list[dict]:
    """Deduplicate identical metadata versions while preserving all provenance."""
    indexed: dict[str, dict] = {}
    for row in rows:
        key = row["news_id"]
        if key not in indexed:
            indexed[key] = dict(row)
            continue
        current = indexed[key]
        if current["metadata_sha256"] != row["metadata_sha256"]:
            raise ValueError("Conflicting news version ID")
        for field in ("source_team_ids", "collection_origins", "direct_fixture_ids"):
            current[field] = sorted(set(current[field]) | set(row[field]))
        provenance = {_hash(p): p for p in current["source_provenance"] + row["source_provenance"]}
        current["source_provenance"] = sorted(provenance.values(), key=lambda p: (p["retrieved_at"], p["url"]))
        current["captured_at_utc"] = min(current["captured_at_utc"], row["captured_at_utc"])
        current["tournament_category"] |= row["tournament_category"]
    ordered = sorted(indexed.values(), key=lambda n: (n["news_item_id"], n["modified_at_utc"] or "", n["captured_at_utc"], n["news_id"]))
    ranks: Counter = Counter()
    for row in ordered:
        row["version_rank"] = ranks[row["news_item_id"]]
        row["version_order_historically_verified"] = False
        ranks[row["news_item_id"]] += 1
    return sorted(ordered, key=lambda n: (n["published_at_utc"] or "", n["news_id"]))


def link_articles(rows: list[dict], registry: dict, *, window_start: str,
                  window_end: str) -> list[dict]:
    """Deterministic relevance candidates; every match link remains retrospective.

    Exact source game IDs establish subject matter, not historical availability.
    Team categories link news only to fixtures within their market-open to
    kickoff+6h context window, with a 90-day pre-opening background lookback.
    Team mention fallback is explicitly weaker.
    """
    start, end = _time(window_start), _time(window_end)
    if start is None or end is None or start >= end:
        raise ValueError("News window requires ordered timezone-aware bounds")
    for row in rows:
        published = _time(row["published_at_utc"])
        row["within_study_window"] = bool(published is not None and start <= published < end)
        links = []
        title = " " + _text(row["title"]) + " "
        for fixture in registry["fixtures"]:
            fid = fixture["fixture_id"]
            direct = fid in row["direct_fixture_ids"] or row["source_game_id"] == str(fixture["espn_event_id"])
            opens = _time(fixture.get("earliest_accepting_orders_at"))
            kick = _time(fixture.get("kickoff_utc"))
            in_window = published is not None and opens is not None and kick is not None and opens - timedelta(days=90) <= published < kick + timedelta(hours=6)
            is_background = published is not None and opens is not None and published < opens
            evidence, relationship, confidence = [], None, None
            if direct:
                relationship, confidence = "explicit_game_id", "high"
                evidence = [f"ESPN article.gameId={fixture['espn_event_id']}"]
            elif row["within_study_window"] and in_window:
                team_ids = {str(fixture[side]["espn_team_id"]) for side in ("home_team", "away_team")}
                matches = sorted(team_ids & set(row["source_team_ids"]))
                if matches:
                    relationship, confidence = ("team_category_background" if is_background else "team_category_context"), "medium"
                    evidence = ["ESPN category.teamId=" + value for value in matches]
                else:
                    names = []
                    for side in ("home_team", "away_team"):
                        team = fixture[side]
                        aliases = [team["name"]] + ALIASES.get(team["canonical_name"], [])
                        if any(" " + _text(alias) + " " in title for alias in aliases):
                            names.append(team["name"])
                    if names:
                        relationship, confidence = ("headline_team_background" if is_background else "headline_team_mention"), "low"
                        evidence = ["headline phrase=" + name for name in names]
            if relationship:
                links.append({"fixture_id": fid, "relationship": relationship, "confidence": confidence,
                              "evidence": evidence, "historical_link_verified": False,
                              "link_availability_upper_utc": None, "retrospective_link": True})
        row["fixture_links"] = sorted(links, key=lambda item: item["fixture_id"])
        row["fixture_ids"] = [item["fixture_id"] for item in row["fixture_links"]]
        row["context_scope"] = "fixture_or_team" if links else "tournament" if row["tournament_category"] else "unmatched"
    return rows


def collect_news(client: Any, registry: dict, output_dir: Path, *, workers: int = 4,
                 max_archive_pages: int = 100, window_start: str | None = None,
                 window_end: str | None = None,
                 progress: Callable[[dict], None] | None = None) -> dict:
    """Collect all fixture recap metadata and a paged historical league archive.

    Archive pagination stops after two complete pages older than the requested
    window or a finite page limit. That is a bounded provider query, never an
    assertion of complete coverage of all news or historical source versions.
    Re-running reuses immutable HTTP captures unless the client refreshes them.
    """
    if workers < 1 or max_archive_pages < 1:
        raise ValueError("workers and max_archive_pages must be positive")
    fixtures = registry["fixtures"]
    if not fixtures:
        raise ValueError("Cannot collect news for an empty registry")
    starts = [f["earliest_accepting_orders_at"] for f in fixtures if f.get("earliest_accepting_orders_at")]
    window_start = window_start or ((_time(min(starts)) - timedelta(days=90)).isoformat().replace("+00:00", "Z") if starts else None)
    window_end = window_end or (_time(max(f["kickoff_utc"] for f in fixtures)) + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    start, end = _time(window_start), _time(window_end)
    if not start or not end or start >= end:
        raise ValueError("A valid news collection window is required")
    rows, errors, fixture_results, page_records = [], [], [], []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def report(event: dict) -> None:
        if progress:
            progress(event)

    def collect_fixture(fixture: dict) -> tuple[dict, list[dict]]:
        result = client.get_json(ESPN_SUMMARY, {"event": fixture["espn_event_id"]})
        data = result.data
        if not isinstance(data, dict):
            raise ValueError("Malformed ESPN summary")
        article = data.get("article")
        extracted = []
        status = "no_event_article"
        if isinstance(article, dict) and article:
            item = normalize_article(article, _source(result), origin="fixture_summary_article", direct_fixture=fixture)
            extracted.append(item)
            status = "explicit_game_id_article" if item["direct_fixture_ids"] else "unverified_article_association"
        return {"fixture_id": fixture["fixture_id"], "status": status, "source": _source(result),
                "current_news_widget_ignored": True}, extracted

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect_fixture, f): f for f in fixtures}
        for future in as_completed(futures):
            fixture = futures[future]
            try:
                result, extracted = future.result()
                fixture_results.append(result)
                rows.extend(extracted)
            except Exception as exc:
                error = {"stage": "fixture_summary", "fixture_id": fixture["fixture_id"], "error": str(exc)}
                errors.append(error)
                fixture_results.append({"fixture_id": fixture["fixture_id"], "status": "source_error", "error": str(exc)})
            report({"stage": "fixture_summary", "completed": len(fixture_results), "total": len(fixtures)})

    seen_pages, archive_status, older_pages, offset = set(), "page_limit", 0, 0
    for page_number in range(max_archive_pages):
        try:
            result = client.get_json(ESPN_ARCHIVE, {"limit": 50, "offset": offset})
            data = result.data
            if not isinstance(data, dict) or not isinstance(data.get("headlines"), list) or data.get("status") != "success":
                raise ValueError("Malformed ESPN archive response")
            if int(data.get("resultsOffset", -1)) != offset:
                raise ValueError("Archive ignored requested offset")
            articles = data["headlines"]
            identifiers = tuple(str(article.get("id")) for article in articles)
            if articles and identifiers in seen_pages:
                raise ValueError("Archive repeated a page; paging is not reliable")
            seen_pages.add(identifiers)
            extracted = [normalize_article(article, _source(result), origin="league_archive") for article in articles]
            rows.extend(extracted)
            dates = [_time(item["published_at_utc"]) for item in extracted]
            all_old = bool(dates) and all(date is not None and date < start for date in dates)
            older_pages = older_pages + 1 if all_old else 0
            page_records.append({"offset": offset, "count": len(articles), "source": _source(result),
                                 "earliest_claimed_publication": min((x["published_at_utc"] for x in extracted if x["published_at_utc"]), default=None),
                                 "latest_claimed_publication": max((x["published_at_utc"] for x in extracted if x["published_at_utc"]), default=None)})
            report({"stage": "league_archive", "page": page_number + 1, "offset": offset, "articles": len(rows)})
            if not articles:
                archive_status = "endpoint_exhausted_unverified"
                break
            offset += len(articles)
            if older_pages >= 2:
                archive_status = "crossed_window_start_unverified"
                break
        except Exception as exc:
            errors.append({"stage": "league_archive", "offset": offset, "error": str(exc)})
            archive_status = "source_error"
            break
        # Keep a lightweight checkpoint even during a long archive sweep.
        write_json(output_dir / "collection_progress.json", {"archive_next_offset": offset, "archive_pages": page_records,
                   "fixture_sources": sorted(fixture_results, key=lambda x: x["fixture_id"]), "errors": errors})

    merged = link_articles(merge_articles(rows), registry, window_start=window_start, window_end=window_end)
    within = [row for row in merged if row["within_study_window"]]
    links = Counter(fid for row in within for fid in row["fixture_ids"])
    fixture_sources = {row["fixture_id"]: row for row in fixture_results}
    coverage = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        related = [row for row in within if fid in row["fixture_ids"]]
        kickoff = _time(fixture["kickoff_utc"])
        coverage.append({"fixture_id": fid, "source_status": fixture_sources[fid]["status"],
                         "candidate_news_versions": links[fid],
                         "claimed_pregame_versions": sum(_time(row["published_at_utc"]) < kickoff for row in related),
                         "historically_verified_versions": 0,
                         "coverage_status": "retrospective_candidates_only" if related else "no_candidates",
                         "complete_news_coverage": False})
    report_data = {"schema_version": 1, "window_start_utc": window_start, "window_end_utc": window_end,
                  "providers": ["ESPN"], "archive_status": archive_status,
                  "premarket_background_lookback_days": 90,
                  "archive_pages": len(page_records), "fixture_sources_attempted": len(fixtures),
                  "fixture_source_errors": sum(x["status"] == "source_error" for x in fixture_results),
                  "unique_articles": len({x["news_item_id"] for x in merged}),
                  "metadata_versions": len(merged), "in_window_metadata_versions": len(within),
                  "historically_verified_versions": 0, "feature_eligible_versions": 0,
                  "complete_news_coverage": False, "fixture_coverage": coverage, "errors": errors,
                  "limitations": ["One publisher archive is not the universe of news.",
                                  "Publication claims do not prove historical availability of captured headlines.",
                                  "Current knockout participants and fixture windows only support retrospective relevance.",
                                  "An article's relevance does not show wallet exposure or causal influence.",
                                  "Raw source responses may contain article text; exports retain bibliographic metadata only.",
                                  "Archive offsets can move during collection; source completeness is not certified."]}
    write_jsonl(output_dir / "news.jsonl", merged)
    write_json(output_dir / "coverage.json", report_data)
    write_json(output_dir / "sources.json", {"fixture_sources": sorted(fixture_results, key=lambda x: x["fixture_id"]),
                                            "archive_pages": page_records, "errors": errors})
    write_json(output_dir / "collection_progress.json", {"status": "finished", "archive_status": archive_status,
                                                        "archive_next_offset": offset, "errors": errors})
    return report_data
