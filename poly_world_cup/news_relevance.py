"""Recover fixture relevance from historically captured, explicit matchups.

This module classifies subject matter, not a trader's exposure or motivation.
Current registry participants only nominate pairs to look for. A captured title
must itself name the pair before its association becomes usable. Team background
is returned separately and cannot activate before the pairing is established.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Iterable, Mapping
import unicodedata
from urllib.parse import unquote, urlsplit

from .attribution import _link_time, _micros, _verified_news_time

POLICY_VERSION = "captured-explicit-matchup-v1"
_ALIASES = {
    "unitedstates": ["United States", "USA", "USMNT", "U.S."],
    "southkorea": ["South Korea", "Korea Republic"],
    "ivorycoast": ["Ivory Coast", "Cote d'Ivoire"],
    "capeverde": ["Cape Verde", "Cabo Verde"],
    "turkiye": ["Türkiye", "Turkiye", "Turkey"],
    "congodr": ["Congo DR", "DR Congo", "Democratic Republic of Congo"],
    "drcongo": ["Congo DR", "DR Congo", "Democratic Republic of Congo"],
    "bosniaherzegovina": ["Bosnia-Herzegovina", "Bosnia and Herzegovina"],
    "bosniaandherzegovina": ["Bosnia-Herzegovina", "Bosnia and Herzegovina"],
    "czechia": ["Czechia", "Czech Republic"],
    "australia": ["Australia", "Socceroos"],
    "netherlands": ["Netherlands", "the Netherlands"],
    "iran": ["Iran", "IR Iran"],
}
_OTHER_COMPETITION = re.compile(
    r"\b(?:club|women|womens|girls|youth|under\s*\d+|u\s*\d+|rugby|cricket|"
    r"hockey|basketball|baseball|softball|netball|handball|volleyball|lacrosse|ski|skiing|"
    r"futsal|beach|esports|qualifiers?|qualifying|qualification|"
    r"friendlies|friendly|finalissima|afcon|africa cup of nations|euros?)\b"
)
_HISTORICAL = re.compile(
    r"\b(?:head to head|h2h|classic clashes|oral history|previous meetings|"
    r"past meetings|last met|rivalr(?:y|ies)|throwback|remember|flashback)\b"
    r"|\branking\b.*\b(?:moments|matches|clashes)\b"
)
_HYPOTHETICAL = re.compile(
    r"\b(?:if|possible|potential|hypothetical|could|would|might|may face|predicted final|"
    r"projected|simulation|simulated|simulator|fictional|dream final|dream matchup|"
    r"everyone wants|we want|video game|videogame|ea sports|ea fc|eafc)\b"
)
_NON_NATIONAL_NAMES = re.compile(r"\b(?:new mexico|mexico city|south africa a|england b)\b")
_MATCH_CUE = re.compile(r"\b(?:match|clash|preview|live|kick\s*off|line\s*ups?|final|semifinals?|quarterfinals?|round|opener|game|prediction|odds|score|team news)\b")


def _text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _dash_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub("[\u2010-\u2015\u2212]", "-", value)
    return " ".join(re.sub(r"[^a-z0-9-]+", " ", value).split())


def _team(value: str | Mapping) -> tuple[str, tuple[str, ...]]:
    name = value.get("name") if isinstance(value, Mapping) else value
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Fixture participants require nonempty names")
    key = _text(name).replace(" ", "")
    aliases = tuple(sorted({_text(name), *(_text(a) for a in _ALIASES.get(key, []))}))
    # Normalize equivalent spellings to the same pair identity.
    canonical = {"bosniaandherzegovina": "bosniaherzegovina", "drcongo": "congodr",
                 "turkey": "turkiye", "czechrepublic": "czechia"}.get(key, key)
    return canonical, aliases


def _utc(micros: int) -> str:
    return datetime.fromtimestamp(micros / 1_000_000, timezone.utc).isoformat().replace("+00:00", "Z")


def _fixture_index(registry: Mapping | Iterable[Mapping]) -> list[dict]:
    fixtures = registry["fixtures"] if isinstance(registry, Mapping) else registry
    result, identities = [], set()
    for fixture in fixtures:
        fid = fixture.get("fixture_id")
        if not isinstance(fid, str) or not fid or fid in identities:
            raise ValueError("Fixture IDs must be distinct and nonempty")
        identities.add(fid)
        home, home_aliases = _team(fixture["home_team"])
        away, away_aliases = _team(fixture["away_team"])
        if home == away:
            raise ValueError("Fixture participants must differ")
        a = "(?:" + "|".join(map(re.escape, sorted(home_aliases, key=len, reverse=True))) + ")"
        b = "(?:" + "|".join(map(re.escape, sorted(away_aliases, key=len, reverse=True))) + ")"
        versus = r"\s+(?:v|vs|versus|against|meet|meets|face|faces|play|plays|take on|takes on)\s+"
        dash_a = "(?:" + "|".join(r"[ -]+".join(map(re.escape, alias.split())) for alias in home_aliases) + ")"
        dash_b = "(?:" + "|".join(r"[ -]+".join(map(re.escape, alias.split())) for alias in away_aliases) + ")"
        result.append({"fixture_id": fid, "pair": tuple(sorted((home, away))),
                       "aliases": home_aliases + away_aliases,
                       "participant_aliases": (home_aliases, away_aliases),
                       "pair_pattern": re.compile(r"\b(?:" + a + versus + b + "|" + b + versus + a + r")\b"),
                       "tactical_preview_pattern": re.compile(r"\bhow\s+(?:" + a + r"\s+can\s+beat\s+" + b + "|" + b + r"\s+can\s+beat\s+" + a + r")\b"),
                       "dash_pattern": re.compile(r"\b(?:" + dash_a + r"\s*-\s*" + dash_b + "|" + dash_b + r"\s*-\s*" + dash_a + r")\b")})
    frequencies = Counter(row["pair"] for row in result)
    for row in result:
        row["pair_ambiguous"] = frequencies[row["pair"]] != 1
    return result


def _competition(record: Mapping) -> tuple[bool, str]:
    """Use only the archived title and its captured URL, never current tags."""
    title = _text(str(record.get("title", "")))
    path = _text(unquote(urlsplit(str(record.get("source_url", ""))).path))
    combined = title + " " + path
    if _OTHER_COMPETITION.search(combined):
        return False, "other_competition"
    if any(year != "2026" for year in re.findall(r"\b(?:19|20)\d{2}\b", combined)):
        return False, "other_edition"
    if not re.search(r"\bworld\s+cup\b", combined):
        return False, "no_world_cup_scope"
    # Implicit edition is limited to a 2026 archived capture, never an old
    # archive of an undated World Cup headline.
    if not str(record.get("availability_upper_utc", "")).startswith("2026-"):
        return False, "capture_outside_edition"
    return True, "captured_world_cup_title_or_url"


def _pair_order(proof: Mapping) -> tuple[int, str]:
    return proof["eligible_us"], proof.get("news_id") or proof.get("evidence_id") or ""


def _contract_pairings(contract_records: Iterable[Mapping] | None, fixtures: list[dict], registry) -> dict:
    if contract_records is None:
        return {}
    from .historical_contracts import validate_contract_record
    by_fixture = {fixture["fixture_id"]: fixture for fixture in fixtures}
    conditions = {row["condition_id"]: row["fixture_id"] for row in registry.get("contracts", [])} if isinstance(registry, Mapping) else {}
    result = {}
    for supplied in contract_records:
        record = validate_contract_record(dict(supplied))
        fid = record["fixture_id"]
        if fid not in by_fixture or (conditions and conditions.get(record["condition_id"]) != fid):
            raise ValueError("Contract pairing evidence does not match the registry")
        fixture = by_fixture[fid]
        title = _text(record["fixture_title"])
        if fixture["pair_ambiguous"] or not fixture["pair_pattern"].fullmatch(title):
            raise ValueError("Initial market title does not identify a unique fixture pair")
        if not _competition({"title": record["initial_fixture_description"], "source_url": "",
                             "availability_upper_utc": record["fixture_initialized_at_utc"]})[0]:
            raise ValueError("Initial market description lacks the tournament identity")
        evidence = record["evidence"]
        digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        proof = {"fixture_id": fid, "news_id": None, "evidence_id": record["evidence_id"],
                 "eligible_us": _micros(record["fixture_initialized_at_utc"]),
                 "available_at_utc": record["fixture_initialized_at_utc"], "basis": "polygon_market_prepared",
                 "evidence": [{"kind": "polygon_market_prepared", "event_at_utc": record["fixture_initialized_at_utc"],
                     "source_url": evidence["rpc_url"], "content_sha256": digest,
                     "evidence_id": record["evidence_id"], "transaction_hash": evidence["market_log"]["transactionHash"],
                     "block_hash": evidence["market_log"]["blockHash"]}]}
        if fid not in result or _pair_order(proof) < _pair_order(result[fid]):
            result[fid] = proof
    return result


def build_news_relevance(records: Iterable[Mapping], registry: Mapping | Iterable[Mapping],
                         *, contract_records: Iterable[Mapping] | None = None,
                         reviewed_links: Mapping | None = None) -> dict:
    """Return copied records with strict direct links and separate backgrounds.

    Downstream joins must use ``eligible_us < target_time``, not ``<=``. The
    publication claim, current team tags and current candidate fixture links do
    not establish a historical association. Existing verified direct game-ID
    links are preserved but do not, by themselves, establish participant names.
    Optional initial-contract records are redecoded from their chain evidence,
    checked against the registry and exact participant title, then may establish
    pairing time for team background. They never create direct news coverage.
    """
    fixtures = _fixture_index(registry)
    output = [deepcopy(dict(record)) for record in records]
    seen = set()
    direct_links, blocked = [], Counter()
    pairing = _contract_pairings(contract_records, fixtures, registry)
    contract_pairing = deepcopy(pairing)
    verified = []
    for record in output:
        nid = record.get("news_id")
        if not isinstance(nid, str) or not nid or nid in seen:
            raise ValueError("News IDs must be distinct and nonempty")
        seen.add(nid)
        try:
            available = _verified_news_time(record)
        except (ValueError, TypeError, KeyError):
            available = None
        if available is None:
            blocked["unverified_content"] += 1
            continue
        verified.append((record, available))
        links = {link["fixture_id"]: deepcopy(link) for link in record.get("fixture_links", [])}
        in_competition, reason = _competition(record)
        title = _text(str(record.get("title", "")))
        if not in_competition:
            blocked[reason] += 1
            continue
        if _HISTORICAL.search(title):
            blocked["historical_comparison"] += 1
            continue
        if _HYPOTHETICAL.search(title):
            blocked["hypothetical_pairing"] += 1
            continue
        if _NON_NATIONAL_NAMES.search(title):
            blocked["ambiguous_national_team_name"] += 1
            continue
        dash_title = _dash_text(record["title"])
        padded_title = " " + title + " "
        candidate_fixtures = [fixture for fixture in fixtures if all(
            any(" " + alias + " " in padded_title for alias in aliases)
            for aliases in fixture["participant_aliases"])]
        has_match_cue, has_preview = bool(_MATCH_CUE.search(title)), bool(re.search(r"\bpreview\b", title))
        matches = [(fixture, fixture["pair_pattern"].search(title) or
                    (has_match_cue and fixture["dash_pattern"].search(dash_title)) or
                    (has_preview and fixture["tactical_preview_pattern"].search(title)))
                   for fixture in candidate_fixtures]
        matches = [(fixture, match) for fixture, match in matches if match]
        if len(matches) > 1:
            # An explicit semicolon-separated schedule contains independently
            # named games, unlike an unstructured list of possible opponents.
            segments = record["title"].split(";")
            segment_matches = [[fixture for fixture in fixtures
                                if fixture["pair_pattern"].search(_text(segment))]
                               for segment in segments]
            explicit_list = (len(segments) == len(matches)
                and all(len(found) == 1 for found in segment_matches)
                and {found[0]["fixture_id"] for found in segment_matches} == {f["fixture_id"] for f, _ in matches}
                and len({team for f, _ in matches for team in f["pair"]}) == 2 * len(matches))
        else:
            explicit_list = False
        if not matches or any(f["pair_ambiguous"] for f, _ in matches) or (len(matches) > 1 and not explicit_list):
            blocked["ambiguous_or_multiple_matchups" if matches else "no_explicit_adjacent_pair"] += 1
            continue
        for fixture, match in matches:
            fid = fixture["fixture_id"]
            evidence = deepcopy(record["availability_evidence"])
            link = {"fixture_id": fid, "historical_link_verified": True,
                "direct_fixture_relevance_verified": True, "link_type": "direct_fixture",
                "relationship": "direct_match", "relevance_basis": "captured_adjacent_matchup_title",
                "relevance_policy_version": POLICY_VERSION,
                "link_availability_upper_utc": _utc(available), "link_availability_evidence": evidence,
                "matched_title_phrase": match.group(0),
                "explicit_semicolon_matchup_list": explicit_list,
                "actor_exposure_verified": False, "causal_attribution": False}
            previous = links.get(fid)
            try:
                previous_time = _link_time(previous, available) if previous else None
            except (ValueError, TypeError, KeyError):
                previous_time = None
            # Preserve stronger or earlier existing direct evidence.
            previous_is_direct = bool(previous and (
                previous.get("relevance_basis") == "archived_direct_game_url"
                or (previous.get("link_type") == "direct_fixture"
                    and previous.get("direct_fixture_relevance_verified") is True)
                or previous.get("relationship") == "direct_match"))
            if not previous_is_direct or previous_time is None or previous_time > available:
                links[fid] = link
            record["fixture_links"] = [links[key] for key in sorted(links)]
            record["fixture_ids"] = sorted(set(record.get("fixture_ids", [])) | set(links))
            direct_links.append({"fixture_id": fid, "news_id": nid, "eligible_us": available, "link": link})
            candidate = {"fixture_id": fid, "news_id": nid, "eligible_us": available,
                     "available_at_utc": _utc(available), "basis": "captured_adjacent_matchup_title",
                     "evidence": evidence}
            if fid not in pairing or _pair_order(candidate) < _pair_order(pairing[fid]):
                pairing[fid] = candidate
    reviewed_count = 0
    if reviewed_links is not None:
        if (reviewed_links.get("schema_version") != 1
                or reviewed_links.get("review_status") != "reviewed_by_independent_assistant"
                or not isinstance(reviewed_links.get("links"), list)):
            raise ValueError("Reviewed links require a completed independent review manifest")
        review_digest = hashlib.sha256(json.dumps(reviewed_links, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        by_news = {record["news_id"]: record for record in output}
        by_fixture = {fixture["fixture_id"]: fixture for fixture in fixtures}
        review_keys = set()
        direct_keys = {(row["fixture_id"], row["news_id"]) for row in direct_links}
        for annotation in reviewed_links["links"]:
            nid, fid = annotation["news_id"], annotation["fixture_id"]
            key = (fid, nid)
            if key in review_keys or nid not in by_news or fid not in contract_pairing:
                raise ValueError("Reviewed link is duplicate, unknown, or lacks validated Polygon pairing")
            review_keys.add(key)
            record = by_news[nid]
            if any(annotation.get(field) != record.get(field) for field in
                   ("title", "source_url", "historical_content_sha256", "availability_upper_utc")):
                raise ValueError("Reviewed link does not match the exact captured news version")
            available = _verified_news_time(record)
            title = _text(record["title"])
            participants = by_fixture[fid]["participant_aliases"]
            if (available is None or not _competition(record)[0] or _HISTORICAL.search(title)
                    or _HYPOTHETICAL.search(title) or _NON_NATIONAL_NAMES.search(title)
                    or not all(any(" " + alias + " " in " " + title + " " for alias in aliases) for aliases in participants)
                    or not isinstance(annotation.get("rationale"), str) or not annotation["rationale"].strip()
                    or annotation.get("requires_validated_polygon_pairing") is not True):
                raise ValueError("Reviewed link fails content, competition, or participant checks")
            if key in direct_keys:
                continue
            proof = contract_pairing[fid]
            eligible = max(available, proof["eligible_us"])
            link = {"fixture_id": fid, "historical_link_verified": True,
                    "direct_fixture_relevance_verified": True, "link_type": "direct_fixture",
                    "relationship": "direct_match", "relevance_basis": "independently_reviewed_captured_matchup",
                    "relevance_policy_version": POLICY_VERSION, "reviewed_link_manifest_sha256": review_digest,
                    "review_method": "independent_assistant_headline_review", "review_rationale": annotation["rationale"],
                    "pairing_evidence_id": proof["evidence_id"],
                    "link_availability_upper_utc": _utc(eligible),
                    "link_availability_evidence": deepcopy(record["availability_evidence"]) + deepcopy(proof["evidence"]),
                    "actor_exposure_verified": False, "causal_attribution": False}
            links = {row["fixture_id"]: row for row in record.get("fixture_links", [])}
            links[fid] = link
            record["fixture_links"] = [links[key] for key in sorted(links)]
            record["fixture_ids"] = sorted(set(record.get("fixture_ids", [])) | set(links))
            direct_links.append({"fixture_id": fid, "news_id": nid, "eligible_us": eligible, "link": link})
            direct_keys.add(key)
            reviewed_count += 1
    backgrounds = []
    direct_keys = {(link["fixture_id"], link["news_id"]) for link in direct_links}
    for record, available in verified:
        if not _competition(record)[0]:
            continue
        title = " " + _text(record["title"]) + " "
        for fixture in fixtures:
            fid = fixture["fixture_id"]
            if fid not in pairing or (fid, record["news_id"]) in direct_keys:
                continue
            mentions = sorted({alias for alias in fixture["aliases"] if " " + alias + " " in title})
            if not mentions:
                continue
            proof = pairing[fid]
            eligibility = max(available, proof["eligible_us"])
            backgrounds.append({"fixture_id": fid, "news_id": record["news_id"],
                "eligible_us": eligibility, "link_type": "team_background",
                "direct_fixture_relevance_verified": False, "historical_link_verified": True,
                "relationship": "captured_participant_background_after_pairing_known",
                "relevance_basis": "participant_mention_and_prior_captured_pairing",
                "relevance_policy_version": POLICY_VERSION, "matched_participant_aliases": mentions,
                "pairing_evidence_news_id": proof["news_id"],
                "pairing_evidence_id": proof.get("evidence_id", proof["news_id"]),
                "pairing_evidence_basis": proof["basis"],
                "link_availability_upper_utc": _utc(eligibility),
                "link_availability_evidence": deepcopy(record["availability_evidence"]) + deepcopy(proof["evidence"]),
                "actor_exposure_verified": False, "causal_attribution": False})
    per_fixture = defaultdict(lambda: {"recovered_direct_links": 0, "team_background_links": 0})
    for link in direct_links:
        per_fixture[link["fixture_id"]]["recovered_direct_links"] += 1
    for link in backgrounds:
        per_fixture[link["fixture_id"]]["team_background_links"] += 1
    report = {"policy_version": POLICY_VERSION, "input_news_records": len(output),
              "verified_content_records": len(verified), "recovered_direct_links": len(direct_links),
              "independently_reviewed_links_added": reviewed_count,
              "fixtures_with_recovered_direct_links": len({x["fixture_id"] for x in direct_links}),
              "fixtures_with_pairing_evidence": len(pairing), "team_background_links": len(backgrounds),
              "pairing_evidence_basis_counts": dict(sorted(Counter(proof["basis"] for proof in pairing.values()).items())),
              "blocked_record_counts": dict(sorted(blocked.items())),
              "team_background_is_direct_match_news": False, "public_relevance_is_actor_exposure": False,
              "historical_contract_semantics_verified": False,
              "fixtures": [{"fixture_id": f["fixture_id"], **per_fixture[f["fixture_id"]],
                            "pairing_available_at_utc": pairing.get(f["fixture_id"], {}).get("available_at_utc")}
                           for f in sorted(fixtures, key=lambda f: f["fixture_id"])]}
    return {"news_records": output, "direct_links": direct_links,
            "team_background_links": backgrounds, "fixture_pairing_evidence": pairing, "report": report}
