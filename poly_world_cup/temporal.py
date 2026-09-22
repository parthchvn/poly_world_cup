"""Conservative temporal guards. These helpers do not certify source truth.

All timestamps must include a timezone. Context availability is strictly before
the query. Labels use half-open intervals [query, query + horizon).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Collection, Iterable, Mapping, Sequence

Time = str | datetime
Record = Mapping[str, Any]


def parse_utc(value: Time) -> datetime:
    """Parse an aware ISO 8601 timestamp and normalize to UTC; reject naive time."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return value.astimezone(timezone.utc)


def available_before(
    record: Record, cutoff: Time, *, excluded_atomic_ids: Collection[str] = ()
) -> bool:
    """Check verified public availability, independent of event occurrence time.

    Required fields are availability_verified=True and availability_upper_utc.
    An optional lower bound must not exceed the upper bound. atomic_id should
    include chain/contract scope as needed to identify an entire target bundle.
    Unknown bounds fail closed. Invalid supplied timestamps raise ValueError.
    """
    query = parse_utc(cutoff)
    if record.get("atomic_id") in excluded_atomic_ids:
        return False
    if record.get("availability_verified") is not True:
        return False
    if record.get("availability_upper_utc") is None:
        return False
    upper = parse_utc(record["availability_upper_utc"])
    lower = record.get("availability_lower_utc")
    if lower is not None and parse_utc(lower) > upper:
        raise ValueError("Availability lower bound exceeds upper bound")
    return upper < query


def select_versions(
    records: Iterable[Record], cutoff: Time, *, excluded_atomic_ids: Collection[str] = ()
) -> list[dict[str, Any]]:
    """Choose the highest eligible version_rank per item_id deterministically.

    version_rank must reflect established source version order, not retrieval
    order. Conflicting eligible records with the same identity/rank are invalid.
    Future or unverifiable versions are excluded before selecting a version.
    """
    selected: dict[str, dict[str, Any]] = {}
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if not available_before(record, cutoff, excluded_atomic_ids=excluded_atomic_ids):
            continue
        item_id, rank = record.get("item_id"), record.get("version_rank")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Eligible version must have a nonempty item_id")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("version_rank must be a nonnegative integer")
        value = dict(record)
        identity = (item_id, rank)
        if identity in seen and seen[identity] != value:
            raise ValueError(f"Conflicting eligible version: {identity!r}")
        seen[identity] = value
        if item_id not in selected or rank > selected[item_id]["version_rank"]:
            selected[item_id] = value
    return [selected[item_id] for item_id in sorted(selected)]


@dataclass(frozen=True)
class HistoricalUniverse:
    query_time: datetime
    lookback_start: datetime
    wallets: frozenset[str]
    source_scope: str


def build_historical_universe(
    activity: Iterable[Record], cutoff: Time, *, lookback: timedelta = timedelta(days=30),
    source_scope: str,
) -> HistoricalUniverse:
    """Build membership from verified prior activity, never eventual participation.

    source_scope is an explicit upstream assertion: global_prefix means the
    global ledger; world_cup_prefix means already observed World Cup activity.
    This function cannot detect a falsely declared, future-filtered source.
    """
    if source_scope not in {"global_prefix", "world_cup_prefix"}:
        raise ValueError("Universe source must be a historical prefix, not eventual participants")
    if not isinstance(lookback, timedelta) or lookback <= timedelta(0):
        raise ValueError("lookback must be a positive timedelta")
    query = parse_utc(cutoff)
    start = query - lookback
    wallets: set[str] = set()
    for record in activity:
        if not available_before(record, query):
            continue
        event_time = parse_utc(record["event_time_utc"])
        if event_time > parse_utc(record["availability_upper_utc"]):
            raise ValueError("Execution availability cannot precede its event time")
        wallet = record.get("wallet")
        if start <= event_time < query:
            if not isinstance(wallet, str) or not wallet:
                raise ValueError("Historical activity must identify a wallet")
            wallets.add(wallet)
    return HistoricalUniverse(query, start, frozenset(wallets), source_scope)


@dataclass(frozen=True)
class CoverageInterval:
    """Verified complete target-source coverage for [start, end).

    '*' is an explicit all-wallet or all-fixture scope assertion. target_kind
    describes completeness for taker, maker, or all executed fills. Coverage
    must be established by ingestion, not inferred from the absence of rows.
    """

    start: Time
    end: Time
    wallet: str
    fixture_id: str
    target_kind: str = "taker"
    complete: bool = False


def _covered(
    coverage: Sequence[CoverageInterval], start: datetime, end: datetime,
    wallet: str, fixture_id: str, target_kind: str,
) -> bool:
    intervals = []
    for interval in coverage:
        left, right = parse_utc(interval.start), parse_utc(interval.end)
        if right <= left:
            raise ValueError("Coverage interval must have positive length")
        if interval.target_kind not in {"taker", "maker", "all"}:
            raise ValueError("Unknown coverage target_kind")
        if (interval.complete is True and interval.wallet in {"*", wallet}
                and interval.fixture_id in {"*", fixture_id}
                and interval.target_kind in {"all", target_kind}):
            intervals.append((left, right))
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def occurrence_label(
    actions: Iterable[Record], *, wallet: str, fixture_id: str, query_time: Time,
    horizon: timedelta, coverage: Sequence[CoverageInterval],
    universe: HistoricalUniverse, target_kind: str = "taker",
) -> dict[str, Any]:
    """Label an observed execution horizon, or mark inadequate coverage censored.

    Actions must already be normalized atomic execution bundles with action_id,
    wallet, fixture_id, event_time_utc and role ('taker', 'maker', or 'unknown').
    Equal-time earliest bundles remain explicitly ambiguous, not arbitrarily
    ordered. Target availability may follow execution; it is not a feature.
    """
    query = parse_utc(query_time)
    if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
        raise ValueError("horizon must be a positive timedelta")
    if target_kind not in {"taker", "maker", "all"}:
        raise ValueError("Unknown target_kind")
    if universe.query_time != query or universe.source_scope not in {"global_prefix", "world_cup_prefix"}:
        raise ValueError("Universe must be established at this exact query time")
    if wallet not in universe.wallets:
        raise ValueError("Wallet was not eligible before the query")
    end = query + horizon
    base = {"wallet": wallet, "fixture_id": fixture_id, "query_time_utc": query.isoformat(),
            "label_end_utc": end.isoformat(), "target_kind": target_kind}
    if not _covered(coverage, query, end, wallet, fixture_id, target_kind):
        return {**base, "status": "CENSORED", "label_coverage_complete": False,
                "censor_reason": "incomplete_target_source_coverage"}
    eligible: dict[str, dict[str, Any]] = {}
    for action in actions:
        if action.get("wallet") != wallet or action.get("fixture_id") != fixture_id:
            continue
        event_time = parse_utc(action["event_time_utc"])
        if not query <= event_time < end:
            continue
        role = action.get("role")
        if role not in {"taker", "maker", "unknown"}:
            raise ValueError("Normalized action must contain its observed role")
        if role == "unknown" and target_kind != "all":
            return {**base, "status": "CENSORED", "label_coverage_complete": False,
                    "censor_reason": "unclassified_execution_role"}
        if target_kind != "all" and role != target_kind:
            continue
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("Normalized action must have a nonempty action_id")
        value = dict(action)
        if action_id in eligible and eligible[action_id] != value:
            raise ValueError("Conflicting duplicate normalized action")
        eligible[action_id] = value
    if not eligible:
        return {**base, "status": "NO_OBSERVED_TRADE", "label_coverage_complete": True,
                "action_count": 0, "first_action_ids": [], "ordering_ambiguous": False}
    earliest = min(parse_utc(row["event_time_utc"]) for row in eligible.values())
    first = sorted(key for key, row in eligible.items() if parse_utc(row["event_time_utc"]) == earliest)
    return {**base, "status": "TRADE", "label_coverage_complete": True,
            "action_count": len(eligible), "first_action_ids": first,
            "ordering_ambiguous": len(first) > 1, "first_event_time_utc": earliest.isoformat()}


def validate_splits(
    examples: Iterable[Record], *, train_cutoff: Time, validation_cutoff: Time
) -> list[str]:
    """Report violations of disjoint fixtures AND global temporal boundaries.

    Each record requires example_id, fixture_id, split, query_time_utc and an
    exclusive label_end_utc strictly after its query. Point targets need an
    explicit precision interval before this check can be applied.
    """
    train_end, validation_end = parse_utc(train_cutoff), parse_utc(validation_cutoff)
    if train_end >= validation_end:
        raise ValueError("train_cutoff must precede validation_cutoff")
    errors: list[str] = []
    identities: set[str] = set()
    fixtures: dict[str, str] = {}
    for row in examples:
        identity = row.get("example_id")
        prefix = str(identity or "<missing example_id>")
        if not isinstance(identity, str) or not identity:
            errors.append(f"{prefix}: missing example_id")
        elif identity in identities:
            errors.append(f"{prefix}: duplicate example_id")
        else:
            identities.add(identity)
        split, fixture = row.get("split"), row.get("fixture_id")
        if split not in {"train", "validation", "test"}:
            errors.append(f"{prefix}: invalid split")
            continue
        if not isinstance(fixture, str) or not fixture:
            errors.append(f"{prefix}: missing fixture_id")
        elif fixture in fixtures and fixtures[fixture] != split:
            errors.append(f"{prefix}: fixture appears in multiple splits")
        else:
            fixtures[fixture] = split
        try:
            query, end = parse_utc(row["query_time_utc"]), parse_utc(row["label_end_utc"])
        except (KeyError, ValueError) as exc:
            errors.append(f"{prefix}: invalid temporal boundary: {exc}")
            continue
        if end <= query:
            errors.append(f"{prefix}: exclusive label end must follow query")
        if split == "train" and (query >= train_end or end > train_end):
            errors.append(f"{prefix}: training overlaps evaluation time")
        if split == "validation" and (query < train_end or query >= validation_end or end > validation_end):
            errors.append(f"{prefix}: validation falls outside its time window")
        if split == "test" and query < validation_end:
            errors.append(f"{prefix}: test precedes final evaluation cutoff")
    return errors
