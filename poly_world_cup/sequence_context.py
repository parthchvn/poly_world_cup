"""Time-valid, compact public context for actor-sequence exports.

The source is the separately audited v2 evidence package. Short identifiers are
serialization aliases. Public context never establishes private beliefs or
whether a wallet read an article. Callers must request every fixture already
introduced in a conversation when advancing a shared news watermark.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import gzip
import json
from pathlib import Path
from typing import Iterable, Mapping

from .attribution import _global_time, _link_time, _micros, _verified_news_time
from .historical_contracts import validate_contract_record, verified_contract_context

GLOBAL_SCOPE = "world_cup"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _utc(microseconds: int) -> str:
    return (_EPOCH + timedelta(microseconds=microseconds)).isoformat().replace("+00:00", "Z")


def _jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            yield value


def _source(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_file():
        return path
    alternate = Path(str(path)[:-3]) if path.suffix == ".gz" else Path(str(path) + ".gz")
    if alternate.is_file():
        return alternate
    raise FileNotFoundError(path)


def _headline_key(text: str) -> str:
    return " ".join(text.casefold().split())


class ContextCatalog:
    """Validate source evidence once and enforce strict historical cutoffs.

    ``contracts`` contains audit mappings, including future contracts. Always
    use ``contract`` for prompt fields. News deltas cover [since_us, query_us).
    Initial context selects the latest distinct items/headlines per requested
    scope, then deduplicates across scopes. Deltas are not capped. Only verified
    direct-fixture links and verified broad tournament relevance are eligible.
    """

    def __init__(self, evidence_root: str | Path, max_initial_news: int = 8):
        if type(max_initial_news) is not int or max_initial_news < 0:
            raise ValueError("max_initial_news must be a nonnegative integer")
        self.evidence_root = Path(evidence_root)
        self.max_initial_news = max_initial_news
        self.contracts: dict[str, dict] = {}
        self._contract_records: dict[str, dict] = {}
        self.fixture_known_us: dict[str, int] = {}
        for row in _jsonl(_source(self.evidence_root, "contracts/contract_evidence.jsonl.gz")):
            validate_contract_record(row)
            condition = row["condition_id"].lower()
            if condition in self._contract_records:
                raise ValueError(f"Duplicate contract: {condition}")
            self._contract_records[condition] = row
            fixture = row["fixture_id"]
            known = _micros(row["fixture_initialized_at_utc"])
            self.fixture_known_us[fixture] = min(known, self.fixture_known_us.get(fixture, known))
        for index, condition in enumerate(sorted(self._contract_records), 1):
            row = self._contract_records[condition]
            self.contracts[condition] = {
                "short_id": f"m{index:03d}",
                "fixture_id": row["fixture_id"],
                "initialized_us": _micros(row["initialized_at_utc"]),
                "fixture_title": row["fixture_title"],
                "question": row["question"],
                "outcomes": {row["yes_token_id"]: "Yes", row["no_token_id"]: "No"},
            }

        self._news: dict[str, dict] = {}
        self._news_times: dict[str, int | None] = {}
        self._scope_times: dict[str, dict[str, int]] = defaultdict(dict)
        identities: set[tuple[str, int]] = set()
        for row in _jsonl(_source(self.evidence_root, "news_catalog/news_for_sft.jsonl")):
            news_id = row.get("news_id")
            item_id = row.get("news_item_id", news_id)
            rank = row.get("version_rank", 0)
            if (not isinstance(news_id, str) or not news_id or
                    not isinstance(item_id, str) or not item_id or
                    type(rank) is not int or rank < 0):
                raise ValueError("Invalid news identity/version")
            if news_id in self._news:
                raise ValueError(f"Duplicate news_id: {news_id}")
            if (item_id, rank) in identities:
                raise ValueError(f"Ambiguous news item/version: {(item_id, rank)}")
            identities.add((item_id, rank))
            if not isinstance(row.get("title"), str) or not row["title"].strip():
                raise ValueError(f"Missing headline: {news_id}")
            self._news[news_id] = row
            verified_time = _verified_news_time(row)
            self._news_times[news_id] = verified_time
            global_time = _global_time(row, verified_time)
            if global_time is not None:
                self._scope_times[news_id][GLOBAL_SCOPE] = global_time
            links = row.get("fixture_links", [])
            if not isinstance(links, list) or any(not isinstance(link, dict) for link in links):
                raise ValueError("fixture_links must contain objects")
            linked: set[str] = set()
            for link in links:
                fixture = link.get("fixture_id")
                if fixture not in self.fixture_known_us:
                    raise ValueError(f"Unknown fixture in news link: {fixture}")
                if fixture in linked:
                    raise ValueError(f"Duplicate fixture/news link: {fixture}/{news_id}")
                linked.add(fixture)
                direct = (link.get("direct_fixture_relevance_verified") is True
                          or link.get("relevance_basis") == "archived_direct_game_url")
                link_time = _link_time(link, verified_time) if direct else None
                if link_time is not None:
                    self._scope_times[news_id][fixture] = max(link_time, self.fixture_known_us[fixture])
        self._short_news = {news_id: f"n{index:04d}" for index, news_id in enumerate(sorted(self._news), 1)}
        events: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for news_id, scopes in self._scope_times.items():
            for scope, instant in scopes.items():
                events[scope].append((instant, news_id))
        self._events = {scope: tuple(sorted(values)) for scope, values in events.items()}
        self._times = {scope: tuple(instant for instant, _ in values) for scope, values in self._events.items()}
        self._item_events: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        for scope, values in self._events.items():
            for instant, news_id in values:
                item = self._news[news_id].get("news_item_id", news_id)
                self._item_events[scope, item].append((instant, news_id))

    def contract(self, condition_id: str, query_us: int) -> dict | None:
        """Return initial public semantics only after validated initialization."""
        condition = condition_id.lower()
        row = self._contract_records.get(condition)
        context = verified_contract_context(row, query_us) if row else None
        if context is None:
            return None
        result = {"market_id": self.contracts[condition]["short_id"],
                  "fixture": context["fixture_title"], "question": context["question"],
                  "outcomes": ["Yes", "No"]}
        if "initial_resolution_clause" in context:
            result["initial_resolution_clause"] = context["initial_resolution_clause"]
        return result

    def _distinct(self, events: Iterable[tuple[int, str]]) -> list[tuple[int, str]]:
        """Choose latest eligible item versions, then deduplicate headline text."""
        items: dict[str, tuple[int, str]] = {}
        for event in events:
            instant, news_id = event
            row = self._news[news_id]
            item = row.get("news_item_id", news_id)
            previous = items.get(item)
            if previous is None or (row.get("version_rank", 0), instant, news_id) > (
                    self._news[previous[1]].get("version_rank", 0), *previous):
                items[item] = event
        headlines: dict[str, tuple[int, str]] = {}
        for event in sorted(items.values(), reverse=True):
            headlines.setdefault(_headline_key(self._news[event[1]]["title"]), event)
        return sorted(headlines.values(), reverse=True)

    @lru_cache(maxsize=16384)
    def _initial(self, scope: str, end: int, limit: int) -> tuple[tuple[int, str], ...]:
        return tuple(self._distinct(self._events.get(scope, ())[:end])[:limit])

    def news_before(self, fixture_ids: Iterable[str], query_us: int,
                    since_us: int | None = None, initial_limit: int | None = None) -> list[dict]:
        """Return historical headlines, with each relevance scope gated separately.

        In a delta, available_us is the earliest newly eligible requested scope.
        A later direct-fixture link can therefore re-emit a globally known item;
        callers should retain headline_key across turns to suppress repetition.
        Delayed older versions cannot replace a newer eligible version in any
        requested scope. No output uses a future link or participant mapping.
        """
        if type(query_us) is not int or (since_us is not None and type(since_us) is not int):
            raise ValueError("Query timestamps must be integer microseconds")
        if since_us is not None and since_us > query_us:
            raise ValueError("since_us cannot follow query_us")
        limit = self.max_initial_news if initial_limit is None else initial_limit
        if type(limit) is not int or limit < 0:
            raise ValueError("initial_limit must be a nonnegative integer")
        fixtures = set(fixture_ids)
        unknown = fixtures - self.fixture_known_us.keys()
        if unknown:
            raise ValueError(f"Unknown fixtures: {sorted(unknown)}")
        scopes = [GLOBAL_SCOPE] + sorted(fixtures)
        chosen: dict[str, dict[str, int]] = defaultdict(dict)
        for scope in scopes:
            times = self._times.get(scope, ())
            end = bisect_left(times, query_us)
            if since_us is None:
                selected = self._initial(scope, end, limit)
            else:
                start = bisect_left(times, since_us)
                selected = self._distinct(self._events.get(scope, ())[start:end])
            for instant, news_id in selected:
                chosen[news_id][scope] = instant
        # Check all historically eligible requested scopes, not only events in
        # this delta. Otherwise an old version's delayed fixture link could
        # supersede a newer version seen earlier as broad tournament context.
        for news_id in list(chosen):
            row = self._news[news_id]
            item = row.get("news_item_id", news_id)
            if any(instant < query_us and self._news[other].get("version_rank", 0) > row.get("version_rank", 0)
                   for scope in scopes for instant, other in self._item_events.get((scope, item), ())):
                del chosen[news_id]
        selected = self._distinct((min(eligible.values()), news_id) for news_id, eligible in chosen.items())
        result = []
        for instant, news_id in reversed(selected):
            row = self._news[news_id]
            result.append({"news_id": news_id, "short_id": self._short_news[news_id],
                           "item_id": row.get("news_item_id", news_id), "version_rank": row.get("version_rank", 0),
                           "headline": row["title"], "headline_key": _headline_key(row["title"]),
                           "available_us": instant, "available_at": _utc(instant),
                           "scopes": sorted(chosen[news_id]),
                           "scope_available_us": dict(sorted(chosen[news_id].items()))})
        return result

    @staticmethod
    def compact_news(record: Mapping) -> dict:
        return {"news_id": record["short_id"], "headline": record["headline"], "available_at": record["available_at"]}

    def audit_catalog(self) -> dict:
        """Aliases and provenance. This object must not become prompt context."""
        contracts = []
        for condition, mapping in self.contracts.items():
            row = self._contract_records[condition]
            contracts.append({"condition_id": condition, **mapping,
                              "initialized_at": row["initialized_at_utc"],
                              "fixture_known_at": _utc(self.fixture_known_us[row["fixture_id"]]),
                              "evidence_id": row["evidence_id"]})
        news = []
        for news_id in sorted(self._news):
            row = self._news[news_id]
            news.append({"news_id": news_id, "short_id": self._short_news[news_id],
                         "item_id": row.get("news_item_id", news_id), "version_rank": row.get("version_rank", 0),
                         "headline": row["title"], "source_url": row.get("source_url"),
                         "historical_content_sha256": row.get("historical_content_sha256"),
                         "verified_available_us": self._news_times[news_id],
                         "scope_available_us": dict(sorted(self._scope_times.get(news_id, {}).items()))})
        return {"schema_version": 1, "private_beliefs": None,
                "contract_semantics_scope": "initial_question_and_token_mapping",
                "actor_news_exposure_verified": False, "contracts": contracts, "news": news}
