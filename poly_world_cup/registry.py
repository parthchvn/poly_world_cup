"""Discover every fixture and conservatively map its regulation-result contracts.

The source APIs are fetched retrospectively. Registry records are identifiers and
audit metadata, not evidence that their current content existed before a trade.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
WORLD_CUP_SERIES_ID = "11433"
MAX_KICKOFF_DIFFERENCE_SECONDS = 3 * 60 * 60
MAX_EVENT_PAGES = 100

# Both sources use these spellings for the 2026 national teams. No fuzzy match is
# allowed: a previously unseen alias must be reviewed explicitly.
TEAM_ALIASES = {
    "caboverde": "capeverde",
    "capeverdeislands": "capeverde",
    "congodr": "drcongo",
    "cotedivoire": "ivorycoast",
    "iriran": "iran",
    "korearepublic": "southkorea",
    "bosniaandherzegovina": "bosniaherzegovina",
    "czechrepublic": "czechia",
    "turkey": "turkiye",
}


def canonical_team(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.casefold())
    key = "".join(character for character in text if character.isalnum())
    return TEAM_ALIASES.get(key, key)


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("A source timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _utc(value: str) -> str:
    return _datetime(value).isoformat().replace("+00:00", "Z")


def _source(result: Any) -> dict[str, Any]:
    return {
        "url": result.url,
        "retrieved_at": result.retrieved_at,
        "body_sha256": result.body_sha256,
        "retrospective_metadata": True,
        "feature_eligible": False,
    }


def _array(value: Any) -> list[Any]:
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError("Expected a JSON array")
    return result


def _event_pair(event: dict[str, Any]) -> tuple[str, str] | None:
    markets = [m for m in event.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
    if len(markets) != 3:
        return None
    names = []
    draws = 0
    for market in markets:
        title = str(market.get("groupItemTitle", "")).strip()
        if title.casefold() == "draw" or title.casefold().startswith("draw ("):
            draws += 1
        elif title:
            names.append(canonical_team(title))
    if draws != 1 or len(names) != 2 or len(set(names)) != 2:
        return None
    return tuple(sorted(names))


def _event_kickoff(event: dict[str, Any]) -> datetime:
    if event.get("startTime"):
        return _datetime(event["startTime"])
    times = {
        _datetime(m["gameStartTime"])
        for m in event.get("markets", [])
        if m.get("sportsMarketType") == "moneyline" and m.get("gameStartTime")
    }
    if len(times) != 1:
        raise ValueError("No unambiguous event kickoff timestamp")
    return times.pop()


def _contract(market: dict[str, Any], event: dict[str, Any], fixture_id: str,
              source: dict[str, Any]) -> dict[str, Any]:
    condition_id = str(market.get("conditionId", "")).lower()
    if not re.fullmatch(r"0x[0-9a-f]{64}", condition_id):
        raise ValueError("Missing or malformed condition ID")
    outcomes = _array(market.get("outcomes"))
    token_ids = _array(market.get("clobTokenIds"))
    if len(outcomes) != 2 or len(token_ids) != 2:
        raise ValueError("A binary result contract must have two outcomes and tokens")
    normalized = [str(outcome).strip().casefold() for outcome in outcomes]
    if set(normalized) != {"yes", "no"}:
        raise ValueError("Result contract outcomes are not Yes/No")
    if any(not re.fullmatch(r"[0-9]+", str(token)) for token in token_ids):
        raise ValueError("Malformed token ID")
    if len(set(map(str, token_ids))) != 2:
        raise ValueError("Duplicate token IDs within a contract")
    title = str(market.get("groupItemTitle", ""))
    selection = "draw" if title.casefold() == "draw" or title.casefold().startswith("draw (") else canonical_team(title)
    rules = str(market.get("description", ""))
    # The exact source text remains the authority. This flag only detects the
    # explicit phrase used by the observed contracts, never infers a rule.
    regulation_explicit = bool(re.search(r"first\s+90\s+minutes.*stoppage time", rules, re.I | re.S))
    return {
        "fixture_id": fixture_id,
        "event_id": str(event["id"]),
        "event_slug": event["slug"],
        "market_id": str(market["id"]),
        "condition_id": condition_id,
        "market_slug": market.get("slug"),
        "question": market.get("question"),
        "selection": selection,
        "sports_market_type": "moneyline",
        "tokens": [
            {"outcome_index": index, "outcome": outcome.title(), "token_id": str(token)}
            for index, (outcome, token) in enumerate(zip(normalized, token_ids))
        ],
        "rules_text": rules,
        "regulation_time_explicit_in_rules": regulation_explicit,
        "created_at": _utc(market["createdAt"]) if market.get("createdAt") else None,
        "start_date": _utc(market["startDate"]) if market.get("startDate") else None,
        "accepting_orders_at": _utc(market["acceptingOrdersTimestamp"]) if market.get("acceptingOrdersTimestamp") else None,
        "game_start_time": _utc(market["gameStartTime"]) if market.get("gameStartTime") else None,
        "retrospective_metadata": True,
        "feature_eligible": False,
        "source": source,
    }


def discover_registry(client: Any, *, year: int = 2026) -> dict[str, Any]:
    """Return ``fixtures``, ``contracts``, ``sources`` and a coverage ``report``.

    ``client.get_json(url, params)`` must return an object with ``data``, ``url``,
    ``retrieved_at`` and ``body_sha256``. Fetch/parse failures raise, so a failed
    retrieval is never silently treated as an empty universe. Coverage failures
    instead remain in the returned report and set ``coverage_complete=False``.

    Match pairs must agree after the explicit alias map. Kickoffs may differ by
    at most three hours (two live 2026 records differ by one hour); every nonzero
    difference is reported. Multiple matching events are always ambiguous.
    """
    if year != 2026:
        raise ValueError("Only the audited 2026 World Cup universe is supported")
    scoreboard = client.get_json(ESPN_SCOREBOARD, {"dates": str(year), "limit": 1000})
    if not isinstance(scoreboard.data, dict) or not isinstance(scoreboard.data.get("events"), list):
        raise ValueError("Unexpected ESPN scoreboard response")
    sources = [_source(scoreboard)]
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    seen_fixture_ids: set[str] = set()
    for raw in scoreboard.data["events"]:
        espn_id = str(raw["id"])
        if espn_id in seen_fixture_ids:
            raise ValueError(f"Duplicate ESPN fixture ID: {espn_id}")
        seen_fixture_ids.add(espn_id)
        competitions = raw.get("competitions", [])
        if len(competitions) != 1:
            raise ValueError(f"Fixture {espn_id} does not have one competition")
        competition = competitions[0]
        competitors = competition.get("competitors", [])
        if len(competitors) != 2 or {c.get("homeAway") for c in competitors} != {"home", "away"}:
            raise ValueError(f"Fixture {espn_id} does not have two identified teams")
        teams = {}
        for competitor in competitors:
            team = competitor["team"]
            teams[competitor["homeAway"]] = {
                "espn_team_id": str(team["id"]), "name": team["displayName"],
                "canonical_name": canonical_team(team["displayName"]),
            }
        fixtures.append({
            "fixture_id": f"espn:{espn_id}", "espn_event_id": espn_id, "year": year,
            "kickoff_utc": _utc(raw["date"]), "stage": raw.get("season", {}).get("slug"),
            "home_team": teams["home"], "away_team": teams["away"],
            "mapping_status": "unmatched", "candidate_event_ids": [],
            "retrospective_metadata": True, "feature_eligible": False, "source": sources[0],
        })
    fixtures.sort(key=lambda f: (_datetime(f["kickoff_utc"]), f["fixture_id"]))
    if len(fixtures) != 104:
        failures.append({"code": "fixture_count_mismatch", "expected": 104, "observed": len(fixtures)})

    events: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_event_ids: set[str] = set()
    offset = 0
    page_count = 0
    for _ in range(MAX_EVENT_PAGES):
        response = client.get_json(GAMMA_EVENTS, {
            "series_id": WORLD_CUP_SERIES_ID, "limit": 100, "offset": offset,
            "order": "id", "ascending": "true",
        })
        source = _source(response)
        sources.append(source)
        page_count += 1
        page = response.data
        if not isinstance(page, list):
            raise ValueError("Unexpected Gamma event-list response")
        if not page:
            break
        for event in page:
            event_id = str(event["id"])
            if event_id in seen_event_ids:
                raise ValueError(f"Gamma pagination repeated event {event_id}; retry discovery")
            seen_event_ids.add(event_id)
            events.append((event, source))
        # Gamma silently caps oversized requests at 100. Advance by actual rows
        # and continue to an empty page, including after a short response.
        offset += len(page)
    else:
        raise ValueError("Gamma pagination exceeded the finite page guard")

    base_pattern = re.compile(rf"fifwc-[a-z0-9]+-[a-z0-9]+-{year}-\d{{2}}-\d{{2}}")
    base_events = [(e, s) for e, s in events if base_pattern.fullmatch(e.get("slug", ""))]
    candidates: list[tuple[dict[str, Any], dict[str, Any], tuple[str, str], datetime]] = []
    for event, source in base_events:
        pair = _event_pair(event)
        if pair is None:
            failures.append({"code": "invalid_result_market_set", "event_id": str(event["id"])})
            continue
        try:
            kickoff = _event_kickoff(event)
        except (ValueError, TypeError) as exc:
            failures.append({"code": "invalid_event_kickoff", "event_id": str(event["id"]), "reason": str(exc)})
            continue
        candidates.append((event, source, pair, kickoff))

    contracts: list[dict[str, Any]] = []
    used_event_ids: set[str] = set()
    for fixture in fixtures:
        pair = tuple(sorted([fixture["home_team"]["canonical_name"], fixture["away_team"]["canonical_name"]]))
        kickoff = _datetime(fixture["kickoff_utc"])
        matches = [(e, s, t) for e, s, p, t in candidates if p == pair and abs((t - kickoff).total_seconds()) <= MAX_KICKOFF_DIFFERENCE_SECONDS]
        fixture["candidate_event_ids"] = [str(e["id"]) for e, _, _ in matches]
        if not matches:
            failures.append({"code": "unmatched_fixture", "fixture_id": fixture["fixture_id"]})
            continue
        if len(matches) != 1:
            fixture["mapping_status"] = "ambiguous"
            failures.append({"code": "ambiguous_fixture", "fixture_id": fixture["fixture_id"], "event_ids": fixture["candidate_event_ids"]})
            continue
        event, source, event_time = matches[0]
        event_id = str(event["id"])
        if event_id in used_event_ids:
            # Do not allow a later duplicate fixture to borrow a previously used
            # match. Both mappings are invalidated in the final pass below.
            failures.append({"code": "event_reused", "event_id": event_id})
        used_event_ids.add(event_id)
        difference = int((event_time - kickoff).total_seconds())
        fixture.update({
            "mapping_status": "matched", "polymarket_event_id": event_id,
            "polymarket_event_slug": event["slug"],
            "gamma_kickoff_utc": event_time.isoformat().replace("+00:00", "Z"),
            "kickoff_difference_seconds": difference,
        })
        if difference:
            warnings.append({"code": "kickoff_disagreement", "fixture_id": fixture["fixture_id"], "difference_seconds": difference})
        fixture_contracts = []
        for market in event.get("markets", []):
            if market.get("sportsMarketType") != "moneyline":
                continue
            try:
                contract = _contract(market, event, fixture["fixture_id"], source)
                fixture_contracts.append(contract)
                if not contract["regulation_time_explicit_in_rules"]:
                    warnings.append({"code": "regulation_rule_requires_review", "market_id": contract["market_id"]})
                if not contract["accepting_orders_at"]:
                    warnings.append({"code": "opening_time_unknown", "market_id": contract["market_id"]})
            except (ValueError, TypeError, KeyError) as exc:
                failures.append({"code": "invalid_contract", "market_id": str(market.get("id")), "reason": str(exc)})
        if len(fixture_contracts) != 3:
            fixture["mapping_status"] = "incomplete_contracts"
        contracts.extend(fixture_contracts)
        openings = [c["accepting_orders_at"] for c in fixture_contracts if c["accepting_orders_at"]]
        fixture["earliest_accepting_orders_at"] = min(openings, key=_datetime) if len(openings) == 3 else None

    assigned = Counter(f.get("polymarket_event_id") for f in fixtures if f.get("polymarket_event_id"))
    reused = {event_id for event_id, count in assigned.items() if count > 1}
    if reused:
        for fixture in fixtures:
            if fixture.get("polymarket_event_id") in reused:
                fixture["mapping_status"] = "ambiguous"
        contracts = [c for c in contracts if c["event_id"] not in reused]
    for key in ("market_id", "condition_id"):
        duplicates = [value for value, count in Counter(c[key] for c in contracts).items() if count > 1]
        if duplicates:
            failures.append({"code": f"duplicate_{key}", "values": duplicates})
    token_duplicates = [value for value, count in Counter(t["token_id"] for c in contracts for t in c["tokens"]).items() if count > 1]
    if token_duplicates:
        failures.append({"code": "duplicate_token_id", "values": token_duplicates})
    unmapped = sorted(str(e["id"]) for e, _ in base_events if str(e["id"]) not in used_event_ids)
    if unmapped:
        failures.append({"code": "unmapped_base_events", "event_ids": unmapped})
    statuses = dict(Counter(f["mapping_status"] for f in fixtures))
    return {
        "schema_version": "1.0", "year": year,
        "retrospective_metadata": True, "feature_eligible": False,
        "fixtures": fixtures, "contracts": contracts, "sources": sources,
        "report": {
            "expected_fixtures": 104, "fixture_count": len(fixtures),
            "total_fixtures": len(fixtures), "mapped_fixtures": statuses.get("matched", 0),
            "series_id": WORLD_CUP_SERIES_ID, "gamma_pages": page_count,
            "gamma_event_count": len(events), "base_match_event_count": len(base_events),
            "excluded_other_event_count": len(events) - len(base_events),
            "contract_count": len(contracts), "token_count": sum(len(c["tokens"]) for c in contracts),
            "mapping_status_counts": statuses,
            "coverage_complete": not failures and len(fixtures) == 104 and len(contracts) == 312,
            "retrospective_metadata": True, "feature_eligible": False,
            "kickoff_tolerance_seconds": MAX_KICKOFF_DIFFERENCE_SECONDS,
            "failures": failures, "warnings": warnings,
            "limits": [
                "This parser requires the three binary moneyline contracts observed for 2026 (each team and draw); changed market structures are flagged for review, not inferred.",
                "Current registry metadata cannot establish historical availability or rule versions.",
                "A matched event establishes identifiers, not trade-history or context completeness.",
                "The two sources' kickoff timestamps are retained separately; disagreements require review before time-sensitive features.",
                "Opening timestamps are retrospective API claims, not independently captured proof of the first executable order.",
                "Source scores, winners, prices, volume, descriptions of final results, and current team form are excluded from normalized records.",
            ],
        },
    }
