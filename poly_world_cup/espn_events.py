"""Fetch ESPN soccer commentary once and retain its actual textual context.

An ESPN wallclock is an event-time proxy, not proof of the time the feed was
published. A match clock alone is never silently converted to a UTC timestamp.
No model, article-body scraper, or paid API is required.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

from .registry import canonical_team

SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _datetime(value: Any) -> datetime | None:
    """Read timezone-aware ISO or Unix seconds/milliseconds/microseconds."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float, Decimal)) or (
            isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value)
        ):
            seconds = Decimal(str(value))
            if seconds >= Decimal("1e14"):
                seconds /= 1_000_000
            elif seconds >= Decimal("1e11"):
                seconds /= 1_000
            return EPOCH + timedelta(microseconds=int(seconds * 1_000_000))
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return None


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _us(value: datetime) -> int:
    delta = value - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _source(result: Any) -> dict:
    return {"url": result.url, "retrieved_at": result.retrieved_at,
            "body_sha256": result.body_sha256, "from_cache": result.from_cache}


def _read(path: Path) -> Any:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if str(path).removesuffix(".gz").endswith(".jsonl"):
            return [json.loads(line, parse_float=Decimal) for line in stream if line.strip()]
        return json.load(stream, parse_float=Decimal)


def discover_espn_event(client: Any, *, fixture_date: str, teams: list[str] | tuple[str, str],
                        league: str = "fifa.world") -> tuple[str, dict, list[dict]]:
    """Require an exact two-team match on the requested UTC date (+/- one day)."""
    if len(teams) != 2:
        raise ValueError("ESPN discovery requires exactly two team names")
    date = datetime.fromisoformat(str(fixture_date)[:10]).date()
    dates = (date - timedelta(days=1)).strftime("%Y%m%d") + "-" + (date + timedelta(days=1)).strftime("%Y%m%d")
    response = client.get_json(f"{SITE}/{league}/scoreboard", {"dates": dates, "limit": 1000})
    wanted = {canonical_team(name) for name in teams}
    candidates = []
    for event in response.data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        names = {canonical_team(str(c.get("team", {}).get("displayName", "")))
                 for c in competitions[0].get("competitors", [])}
        if names == wanted:
            candidates.append(event)
    if len(candidates) != 1:
        ids = [str(event.get("id")) for event in candidates]
        raise ValueError(f"ESPN match is ambiguous or missing ({ids}); provide --espn-event-id and --league")
    return str(candidates[0]["id"]), candidates[0], [_source(response)]


def _play_rows(payload: Any) -> list[dict]:
    """Unify summary commentary/keyEvents and the core plays endpoint."""
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = list(payload.get("commentary", [])) + list(payload.get("keyEvents", []))
        raw += list(payload.get("plays", [])) + list(payload.get("items", []))
        if not raw and isinstance(payload.get("events"), list):
            # Offline normalized context documents, not a scoreboard.
            raw = [item for item in payload["events"] if "text" in item or "headline" in item]
        if not raw and ("text" in payload or "headline" in payload):
            raw = [payload]
    else:
        raise ValueError("ESPN input must be an object or a list")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        row = dict(item.get("play") or item)
        if item.get("text"):
            row["text"] = item["text"]
        if "clock" not in row and isinstance(item.get("time"), dict):
            row["clock"] = item["time"]
        for field in ("timestamp_us", "time_utc", "available_at_utc", "observed_at_utc", "published", "wallclock", "sequence"):
            if field in item and field not in row:
                row[field] = item[field]
        rows.append(row)
    return rows


def _article_rows(payload: Any, event_id: str, team_ids: set[str]) -> list[dict]:
    """Keep the match report and explicit team/match-related headline metadata.

    summary.news is a league widget: do not attribute every article to this match.
    Article bodies are deliberately not requested or copied.
    """
    if not isinstance(payload, dict):
        return []
    candidates = [payload.get("article", {})]
    widget = payload.get("news", {})
    if isinstance(widget, dict):
        candidates += widget.get("articles", [])
    candidates += payload.get("articles", [])
    selected = []
    for article in candidates:
        if not isinstance(article, dict) or not article.get("headline"):
            continue
        explicit_game = str(article.get("gameId", ""))
        category_teams = {str(c.get("teamId", c.get("team", {}).get("id", "")))
                          for c in article.get("categories", []) if isinstance(c, dict)}
        if explicit_game == event_id or (team_ids & category_teams):
            selected.append({**article, "_article": True})
    return selected


def _kind(row: dict) -> str:
    if row.get("_article") or row.get("headline"):
        return "headline"
    if row.get("kind"):
        return str(row["kind"])
    play_type = row.get("type", {})
    if isinstance(play_type, dict):
        return str(play_type.get("type") or play_type.get("text") or "commentary")
    return str(row.get("kind") or play_type or "commentary")


def _play_text(row: dict) -> str:
    text = _text(row.get("headline") or row.get("text") or row.get("shortText"))
    if text:
        return text
    names = [_text(p.get("athlete", {}).get("displayName"))
             for p in row.get("participants", []) if isinstance(p, dict)]
    names = [name for name in names if name]
    team = _text(row.get("team", {}).get("displayName"))
    kind = _kind(row)
    # Structured facts only, never an inferred rationale or invented incident.
    parts = [kind.replace("-", " ")]
    if names:
        parts.append(", ".join(names))
    if team:
        parts.append(team)
    return ": ".join(parts) if names or team else ""


def _raw_id(row: dict) -> str:
    if row.get("news_id") or row.get("id"):
        return str(row.get("news_id") or row["id"])
    identity = {key: row.get(key) for key in ("text", "headline", "period", "clock", "sequence")}
    return "text-" + hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _clock(row: dict) -> tuple[int | None, Decimal | None, str | None]:
    period = row.get("period", {})
    period = period.get("number") if isinstance(period, dict) else period
    clock = row.get("clock", {})
    if not isinstance(clock, dict):
        clock = {}
    try:
        period = int(period) if period is not None else None
        seconds = Decimal(str(clock["value"])) if clock.get("value") is not None else None
    except (ValueError, TypeError, InvalidOperation):
        return None, None, None
    return period, seconds, clock.get("displayValue")


def collect_espn_context(client: Any, *, event_id: str | None = None,
                         league: str = "fifa.world", fixture_date: str | None = None,
                         teams: list[str] | tuple[str, str] | None = None,
                         espn_files: list[str | Path] | None = None,
                         time_map: dict | str | Path | None = None,
                         time_policy: str = "provider", allow_clock_estimates: bool = False,
                         include_core_plays: bool = False) -> dict:
    """Return timed_events and untimed_events containing real ESPN text.

    ``time_map`` accepts ``{"PLAY_ID": "ISO_UTC"}`` or an object containing
    ``events`` with that mapping and optional ``period_anchors``. A period anchor
    is ``{"1": {"time_utc": "...Z", "clock_seconds": 0}}``. Clock estimates
    require the explicit opt-in and an anchor for that same period. Their timing
    remains approximate because match-clock stoppages are not reconstructed.

    Offline inputs can be saved summary JSON, core play-page JSON, JSONL, gzip,
    or normalized text events with ``time_utc``/``timestamp_us``. Their declared
    timestamps are carried through with an explicit provenance label.
    """
    if time_policy not in {"provider", "observed"}:
        raise ValueError("time_policy must be provider or observed")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", league):
        raise ValueError("Invalid ESPN league slug")
    if isinstance(time_map, (str, Path)):
        time_map = _read(Path(time_map))
    time_map = time_map or {}
    if not isinstance(time_map, dict):
        raise ValueError("The time map must be a JSON object")
    mapped_times = time_map.get("events", time_map)
    if not isinstance(mapped_times, dict):
        raise ValueError("time_map.events must be an event-ID to timestamp object")
    sources, payloads, warnings = [], [], []
    header: dict = {}
    if espn_files:
        for filename in espn_files:
            path = Path(filename)
            payloads.append(_read(path))
            sources.append({"path": str(path), "body_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source_kind": "user_supplied_espn_file"})
    else:
        if not event_id:
            if not fixture_date or not teams:
                raise ValueError("Supply an ESPN event ID or a fixture date and two teams")
            event_id, _, discovery_sources = discover_espn_event(client, fixture_date=fixture_date, teams=teams, league=league)
            sources.extend(discovery_sources)
        event_id = str(event_id)
        if not event_id.isdigit():
            raise ValueError("ESPN event ID must be numeric")
        summary = client.get_json(f"{SITE}/{league}/summary", {"event": event_id})
        payloads.append(summary.data)
        sources.append(_source(summary))
        if include_core_plays:
            page = 1
            while True:
                result = client.get_json(f"{CORE}/{league}/events/{event_id}/competitions/{event_id}/plays",
                                         {"limit": 1000, "page": page})
                payloads.append(result.data)
                sources.append(_source(result))
                pages = int(result.data.get("pageCount", 1))
                if page >= pages:
                    break
                page += 1
                if page > 100:
                    raise ValueError("ESPN plays endpoint exceeded 100 pages")
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("header"):
            header = payload["header"]
            break
    event_id = str(event_id or header.get("id") or "offline")
    competitions = header.get("competitions", [])
    competition = competitions[0] if competitions else {}
    kickoff = _datetime(competition.get("date") or header.get("date"))
    team_ids = {str(c.get("team", {}).get("id", "")) for c in competition.get("competitors", [])}
    rows: dict[str, dict] = {}
    for payload in payloads:
        for row in _play_rows(payload) + _article_rows(payload, event_id, team_ids):
            if row.get("valid") is False:
                continue
            identity = ("article:" if row.get("_article") else "play:") + _raw_id(row)
            previous = rows.get(identity, {})
            # Commentary has more readable text; supplement its missing fields
            # from the core feed without replacing it with low-level play text.
            merged = {**row, **previous}
            for key, value in row.items():
                if merged.get(key) in (None, "", {}):
                    merged[key] = value
            rows[identity] = merged
    anchors = dict(time_map.get("period_anchors", {}))
    for row in rows.values():
        period, seconds, _ = _clock(row)
        wallclock = _datetime(row.get("wallclock"))
        if period and seconds is not None and wallclock and _kind(row).lower() in {"kickoff", "start-period"}:
            if kickoff is None or kickoff - timedelta(hours=6) <= wallclock <= kickoff + timedelta(hours=12):
                anchors.setdefault(str(period), {"time_utc": _utc(wallclock), "clock_seconds": str(seconds)})
    observed = min((_datetime(source.get("retrieved_at")) for source in sources if source.get("retrieved_at")), default=None)
    timed, untimed = [], []
    for identity, row in rows.items():
        text = _play_text(row)
        if not text:
            continue
        raw_id = _raw_id(row)
        news_id = f"espn:{event_id}:{identity}"
        period, seconds, display = _clock(row)
        when, basis = None, None
        explicit = mapped_times.get(news_id, mapped_times.get(raw_id))
        if isinstance(explicit, dict):
            explicit = explicit.get("time_utc") or explicit.get("available_at_utc") or explicit.get("timestamp_us")
        if explicit is not None:
            when, basis = _datetime(explicit), "user_supplied_event_time"
            if when is None:
                raise ValueError(f"Invalid explicit timestamp for ESPN event {raw_id}")
        if when is None and time_policy == "observed":
            when, basis = _datetime(row.get("observed_at_utc")) or observed, "captured_at"
        if when is None and time_policy == "provider":
            for key in ("available_at_utc", "observed_at_utc", "time_utc", "timestamp_us"):
                if row.get(key) is not None:
                    when = _datetime(row[key])
                    if when:
                        basis = "input_" + key
                        break
            if when is None and _kind(row) == "headline":
                published = _datetime(row.get("published") or row.get("originallyPosted"))
                modified = _datetime(row.get("lastModified"))
                when = max(published, modified) if published and modified else published
                basis = "publisher_revision_time_proxy" if when else None
            if when is None:
                when = _datetime(row.get("wallclock"))
                basis = "provider_event_time_proxy" if when else None
                if when and kickoff and not kickoff - timedelta(hours=6) <= when <= kickoff + timedelta(hours=12):
                    warnings.append(f"Ignored implausible wallclock for {raw_id}: {_utc(when)}")
                    when, basis = None, None
        if when is None and allow_clock_estimates and period and seconds is not None:
            anchor = anchors.get(str(period), {})
            anchor_time = _datetime(anchor.get("time_utc"))
            if anchor_time is not None and anchor.get("clock_seconds") is not None:
                offset = seconds - Decimal(str(anchor["clock_seconds"]))
                when = anchor_time + timedelta(microseconds=int(offset * 1_000_000))
                basis = "estimated_from_period_anchor"
        event = {"news_id": news_id, "source": "ESPN", "event_id": event_id,
                 "source_event_id": raw_id, "kind": _kind(row), "text": text,
                 "match_clock": display, "period": period,
                 "time_utc": _utc(when) if when else None,
                 "timestamp_us": _us(when) if when else None,
                 "available_at_utc": _utc(when) if when else None,
                 "time_basis": basis or "no_absolute_time",
                 "historical_availability_verified": False}
        if when:
            timed.append(event)
        else:
            untimed.append(event)
    timed.sort(key=lambda event: (event["timestamp_us"], event["news_id"]))
    untimed.sort(key=lambda event: event["news_id"])
    return {"event_id": event_id, "league": league, "fixture_name": header.get("name"),
            "scheduled_kickoff_utc": _utc(kickoff) if kickoff else None,
            "timed_events": timed, "untimed_events": untimed, "sources": sources,
            "timing_counts": dict(Counter(event["time_basis"] for event in timed + untimed)),
            "warnings": warnings,
            "news_scope": "match_commentary_and_explicit_match_or_team_headlines_in_saved_responses",
            "complete_historical_espn_news_archive": False,
            "historical_availability_verified": False}
