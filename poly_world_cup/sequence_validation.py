"""Independent source and causal-context checks for actor sequence releases."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
from itertools import groupby
import json
from pathlib import Path, PurePosixPath
import sqlite3
import math
import multiprocessing

from .actor_sequences import SequencePolicy, SYSTEM, PROFILE_SYSTEM

from .attribution import _micros
from .sequence_context import ContextCatalog, _utc

SECOND = 1_000_000
PROFILES = ("conditional_trades", "scheduled_windows")
SPLITS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(root, relative, on_record=None):
    for index, path in enumerate(sorted((Path(root)/relative).glob("part-*.jsonl.gz")), 1):
        require(path.name == f"part-{index:05d}.jsonl.gz", "Noncontiguous shard names")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                require(line.strip(), f"Blank JSONL row: {path}:{line_number}")
                value = json.loads(line)
                require(isinstance(value, dict), "JSONL row must be an object")
                if on_record:
                    on_record(value, path.relative_to(root).as_posix(), line_number)
                yield value


def clock_windows(rows, lower, upper, coverage, horizon, activity):
    """Enumerate the union of intervals activated by strictly earlier trades.

    This construction is independent of the exporter's cursor traversal and
    includes eligible trailing windows even without another future trade.
    """
    by_market, ranges = defaultdict(list), []
    for row in rows:
        condition, instant = row["condition_id"], row["query_us"]
        by_market[condition].append(instant)
        begin = max((instant//horizon+1)*horizon,
                    ((max(lower, coverage[condition]["earliest_query_us"])+horizon-1)//horizon)*horizon)
        finish = min(((instant+activity)//horizon)*horizon,
                     ((coverage[condition]["latest_query_us"]-horizon)//horizon)*horizon,
                     ((upper-horizon)//horizon)*horizon)
        if begin <= finish:
            ranges.append((begin, finish))
    merged = []
    for begin, finish in sorted(ranges):
        if merged and begin <= merged[-1][1]+horizon:
            merged[-1] = (merged[-1][0], max(finish, merged[-1][1]))
        else:
            merged.append((begin, finish))
    times = [r["query_us"] for r in rows]
    for begin, finish in merged:
        for query in range(begin, finish+1, horizon):
            active = []
            for condition, instants in by_market.items():
                previous = bisect_left(instants, query)-1
                if (previous >= 0 and query-instants[previous] <= activity
                    and coverage[condition]["earliest_query_us"] <= query
                    and query+horizon <= coverage[condition]["latest_query_us"]):
                    active.append(condition)
            if active:
                first, last = bisect_left(times, query), bisect_left(times, query+horizon)
                targets = [r for r in rows[first:last] if r["condition_id"] in active]
                yield query, sorted(active), targets


def retain_negative(actor, split, query, probability, seed):
    digest = hashlib.sha256(f"{seed}|{actor}|{split}|{query}".encode()).hexdigest()[:16]
    return int(digest, 16) < probability*(1 << 64)


def normalize_source(raw, catalog, split_policy):
    """Validate source terms independently, retaining every raw observation."""
    first, second = (_micros(split_policy[k]) for k in ("train_before_utc", "validation_before_utc"))
    require(first < second, "Invalid split cutoffs")
    result = []
    transaction_times = defaultdict(set)
    for row in raw:
        transaction_times[row["transaction_hash"]].add(row["query_us"])
    for raw_row in raw:
        row, errors = dict(raw_row), []
        require(type(row["query_us"]) is int, "Invalid source timestamp")
        require(all(isinstance(row[k], str) and row[k] for k in ("observation_id", "wallet", "transaction_hash")), "Missing source identity")
        mapping = catalog.contracts.get(row["condition_id"])
        fixture = mapping["fixture_id"] if mapping else None
        desired = split_policy["fixture_splits"].get(fixture)
        actual = "train" if row["query_us"] < first else "validation" if row["query_us"] < second else "test"
        row["fixture_id"], row["split"] = fixture, desired if desired == actual else None
        row["history_split"] = desired
        row["outcome"] = mapping["outcomes"].get(row["token_id"]) if mapping else None
        if row["split"] is None:
            errors.append("fixture_time_split_mismatch")
        if row["outcome"] not in ("Yes", "No"):
            errors.append("unverified_token_mapping")
        if mapping is None or mapping["initialized_us"] >= row["query_us"]:
            errors.append("contract_not_known_before_observation")
        if row["side"] not in ("BUY", "SELL"):
            errors.append("invalid_side")
        if len(transaction_times[row["transaction_hash"]]) != 1:
            errors.append("invalid_source_transaction_timestamp")
        try:
            require(isinstance(row["shares"], str) and isinstance(row["price"], str), "Source decimals must remain strings")
            size, price = Decimal(row["shares"]), Decimal(row["price"])
            require(size.is_finite() and price.is_finite() and size > 0 and 0 <= price <= 1, "Invalid source decimal")
        except (ValueError, TypeError, InvalidOperation):
            errors.append("invalid_decimal_terms")
        row["errors"] = errors
        result.append(row)
    return sorted(result, key=lambda r: (r["query_us"], r["observation_id"], r["trade_row_id"]))


def source_actors(db, policy):
    extra = "" if policy.filter_scope == "actor_market" else (
        " AND NOT EXISTS (SELECT 1 FROM wallet_market_counts z WHERE z.wallet=t.wallet AND z.observation_count>?)")
    parameters = [policy.max_trades_per_market]*(1 if not extra else 2)
    query = """SELECT t.* FROM trades t JOIN wallet_market_counts w
        ON w.wallet=t.wallet AND w.condition_id=t.condition_id
        WHERE w.observation_count<=?""" + extra + " ORDER BY t.wallet,t.query_us,t.observation_id,t.trade_row_id"
    for actor, rows in groupby(db.execute(query, parameters), lambda row: row["wallet"]):
        yield actor, [dict(row) for row in rows]


def source_action(row, catalog, include_time):
    result = {"market": catalog.contracts[row["condition_id"]]["short_id"], "side": row["side"],
              "outcome": row["outcome"], "shares": row["shares"], "price": row["price"]}
    if include_time:
        result["time"] = _utc(row["query_us"])
    return result


def summary(rows):
    result = {"observations": len(rows), "markets": len({r["condition_id"] for r in rows})}
    for side in ("BUY", "SELL"):
        values = [Decimal(r["shares"])*Decimal(r["price"]) for r in rows if r["side"] == side]
        result[side.lower()+"_observations"] = len(values)
        result["mean_"+side.lower()+"_notional"] = format(sum(values, Decimal(0))/len(values), ".6f") if values else None
    return result


def expected_turns(actor, rows, invalid, profile, split, split_policy, coverage, policy, counts):
    if profile == "conditional_trades":
        for query, values in groupby((r for r in rows if r["split"] == split), lambda row: row["query_us"]):
            targets = list(values)
            yield query, query, sorted({r["condition_id"] for r in targets}), targets
        return
    first, second = (_micros(split_policy[k]) for k in ("train_before_utc", "validation_before_utc"))
    lower, upper = {"train": (-10**30, first), "validation": (first, second), "test": (second, 10**30)}[split]
    # Censored/malformed pairs cannot generate a confident empty-window label.
    eligible_rows = [r for r in rows if r["condition_id"] not in invalid]
    probability = policy.train_negative_probability if split == "train" else policy.evaluation_negative_probability
    for query, monitored, targets in clock_windows(eligible_rows, lower, upper, coverage,
            policy.horizon_seconds*SECOND, policy.activity_seconds*SECOND):
        label = "positive" if targets else "negative"
        counts[split+"_eligible_windows"] += 1
        counts[split+"_eligible_"+label+"_windows"] += 1
        if targets or retain_negative(actor, split, query, probability, policy.sampling_seed):
            counts[split+"_retained_windows"] += 1
            counts[split+"_retained_"+label+"_windows"] += 1
            yield query, query+policy.horizon_seconds*SECOND, monitored, targets


def validate_actor_chunks(chunks, actor, prior, expected, profile, split, catalog, policy,
                          counts, fixture_counts, token_histogram):
    """Check source-exact targets and all context visible through attention."""
    expected = iter(expected)
    times = [r["query_us"] for r in prior]
    require(len({r["observation_id"] for r in prior}) == len(prior), "Ambiguous duplicate source IDs")
    last_actor_query = previous_end = None
    for index, row in enumerate(chunks):
        require(set(row) == {"sequence_id", "actor_id", "profile", "split", "chunk_index", "messages", "token_count", "requires_long_context", "turn_audit"}, "Unexpected conversation schema")
        require((row["actor_id"], row["profile"], row["split"], row["chunk_index"]) == (actor, profile, split, index), "Conversation grouping/order mismatch")
        require(row["sequence_id"] == hashlib.sha256(f"{actor}|{profile}|{split}|{index}".encode()).hexdigest(), "Sequence identity mismatch")
        messages, audits = row["messages"], row["turn_audit"]
        require(0 < len(audits) <= policy.max_turns and len(messages) == 1+2*len(audits), "Message/audit turn count mismatch")
        require(messages[0] == {"role": "system", "content": SYSTEM+PROFILE_SYSTEM[profile]}, "System prompt mismatch")
        require(type(row["token_count"]) is int and row["token_count"] > 0, "Invalid token count")
        require(row["requires_long_context"] is (row["token_count"] > policy.max_tokens), "Long-context flag mismatch")
        known, delivered, visible_transactions, headlines = set(), set(), set(), set()
        previous_query, watched = None, set()
        for turn, audit in enumerate(audits):
            want = next(expected, None)
            require(want is not None, "Unexpected exported prediction")
            query, end, conditions, targets = want
            require(audit["query_us"] == query and audit["window_end_us"] == end, "Missing or misordered query/window")
            require(last_actor_query is None or query > last_actor_query, "Actor times do not strictly increase")
            require(previous_end is None or previous_end <= query, "Previous assistant reveals future target-window data")
            require(audit["condition_ids"] == conditions, "Wrong monitored/query condition set")
            require(audit["target_observation_ids"] == [r["observation_id"] for r in targets], "Missing, duplicated or invented source targets")
            target_transactions = {r["transaction_hash"] for r in targets}
            require(not target_transactions & visible_transactions, "Target transaction leaked through previous context or summary")
            user_message, answer_message = messages[1+turn*2:3+turn*2]
            require(set(user_message) == set(answer_message) == {"role", "content"}
                    and user_message["role"] == "user" and answer_message["role"] == "assistant", "Invalid message roles")
            user, answer = json.loads(user_message["content"]), json.loads(answer_message["content"])
            require(user_message["content"] == canonical(user) and answer_message["content"] == canonical(answer),
                    "Noncanonical prompt/answer JSON may contain hidden duplicate keys or unvalidated text")
            require(_micros(user["query_time"]) == query, "Prompt query time differs from audit")
            expected_answer = {"action": "TRADE", "trades": [source_action(r, catalog, profile == "scheduled_windows") for r in targets]} if targets else {"action": "NO_TRADE"}
            require(answer == expected_answer, "Answer differs from full scoped source observations")
            safe_prior = prior[:bisect_left(times, query)]
            require(not target_transactions & {r["transaction_hash"] for r in safe_prior}, "Target transaction intersects prior history")
            summarized = safe_prior if turn == 0 else []
            history = ((safe_prior[-policy.carry_trades:] if policy.carry_trades else []) if turn == 0
                       else [r for r in safe_prior if r["observation_id"] not in delivered])
            require(audit["history_observation_ids"] == [r["observation_id"] for r in history], "Wrong exact history/carry")
            require(user.get("history", []) == [source_action(r, catalog, True) for r in history], "History values differ from source")
            require(audit["summary_observation_count"] == len(summarized)
                    and audit["summary_observation_sha256"] == hashlib.sha256(canonical([r["observation_id"] for r in summarized]).encode()).hexdigest(), "Summary source count/hash mismatch")
            if turn == 0:
                require(user.get("actor_id") == actor and user.get("actor_summary") == summary(summarized), "Initial actor summary contains wrong or future information")
            else:
                require("actor_id" not in user and "actor_summary" not in user, "Unexpected repeated actor summary")
            required = set(conditions) | {r["condition_id"] for r in history}
            introduced = required-known
            require(user.get("contracts", []) == [catalog.contract(c, query) for c in sorted(introduced)], "Wrong/missing contract definitions")
            require(all(catalog.contract(c, query) is not None for c in required), "Contract unavailable at prediction time")
            fixtures = {catalog.contracts[c]["fixture_id"] for c in known|required}
            candidate_news = catalog.news_before(fixtures, query, since_us=previous_query)
            if turn and introduced:
                candidate_news += catalog.news_before({catalog.contracts[c]["fixture_id"] for c in introduced}, query)
            unique = {}
            for item in sorted(candidate_news, key=lambda n: (n["available_us"], n["news_id"])):
                require(item["available_us"] < query, "Equal-time or future headline in context")
                if item["headline_key"] not in headlines:
                    unique[item["headline_key"]] = item
            news = list(unique.values())
            require(user.get("news", []) == [catalog.compact_news(n) for n in news], "Omitted, duplicated or time-invalid news")
            require(audit["news_ids"] == [n["news_id"] for n in news], "News provenance mismatch")
            aliases = lambda cs: [catalog.contracts[c]["short_id"] for c in sorted(cs)]
            allowed = {"query_time", "actor_id", "actor_summary", "history", "contracts", "news"}
            if profile == "conditional_trades":
                require(user.get("query_markets") == aliases(conditions), "Missing or wrong conditional query markets")
                allowed.add("query_markets")
            else:
                allowed |= {"markets", "markets_add", "markets_remove", "horizon_seconds"}
                if turn == 0:
                    require(user.get("markets") == aliases(conditions) and user.get("horizon_seconds") == policy.horizon_seconds, "Initial forecast watchset/horizon mismatch")
                    require("markets_add" not in user and "markets_remove" not in user, "Unexpected initial watchset delta")
                else:
                    require("markets" not in user and "horizon_seconds" not in user, "Unexpected full watchset repetition")
                    require(user.get("markets_add", []) == aliases(set(conditions)-watched)
                            and user.get("markets_remove", []) == aliases(watched-set(conditions)), "Forecast watchset delta mismatch")
                for fixture in {catalog.contracts[c]["fixture_id"] for c in conditions}:
                    fixture_counts[fixture]["scheduled_retained_windows"] += 1
                for target in targets:
                    fixture_counts[target["fixture_id"]]["scheduled_target_observations"] += 1
            require(set(user) <= allowed, "Unexpected prompt feature")
            known.update(required)
            delivered.update(r["observation_id"] for r in summarized+history+targets)
            visible_transactions.update(r["transaction_hash"] for r in summarized+history+targets)
            headlines.update(n["headline_key"] for n in news)
            previous_query = last_actor_query = query
            previous_end, watched = end, set(conditions)
            counts["target_turns"] += 1
            counts["target_observations"] += len(targets)
        counts["conversations"] += 1
        counts["tokens"] += row["token_count"]
        counts["long_context_sequences"] += int(row["requires_long_context"])
        token_histogram[row["token_count"]] += 1
    require(next(expected, None) is None, "Actor prediction windows or trade targets omitted")


def _bounded_ordered_results(executor, function, iterable, max_pending):
    """Keep at most ``max_pending`` submitted batches, preserving source order.

    ProcessPoolExecutor.map eagerly consumes an iterable on supported Python
    versions. An explicit queue keeps both source rows and conversation payloads
    bounded instead of materializing a tournament-wide list of actor jobs.
    """
    require(type(max_pending) is int and max_pending > 0, "Invalid pending batch limit")
    source, pending = iter(iterable), deque()
    try:
        for _ in range(max_pending):
            try:
                item = next(source)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))
        while pending:
            yield pending.popleft().result()
            try:
                item = next(source)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()


_WORKER_VALIDATION = None


def _validation_worker_init(catalog, policy_dict, splits, coverage, profile, split):
    """Receive the parent's validated evidence snapshot once per spawned worker.

    Passing the catalog through spawn preserves exactly the context already
    compared with the published audit catalog. Workers never reopen evidence
    files whose contents could change after the parent's validation.
    """
    global _WORKER_VALIDATION
    policy = SequencePolicy(**policy_dict)
    _WORKER_VALIDATION = (catalog, policy, splits, coverage, profile, split)


def _validate_actor_batch(batch):
    """Independently reconcile bounded actor jobs; return additive audit totals."""
    require(_WORKER_VALIDATION is not None, "Validation worker is not initialized")
    catalog, policy, splits, coverage, profile, split = _WORKER_VALIDATION
    expected_counts, counts, token_histogram = Counter(), Counter(), Counter()
    fixture_counts = defaultdict(Counter)
    for actor, raw, chunks in batch:
        normalized = normalize_source(raw, catalog, splits)
        invalid = {r["condition_id"] for r in normalized if set(r["errors"])-{"fixture_time_split_mismatch"}}
        prior = [r for r in normalized if r["history_split"] == split and not(set(r["errors"])-{"fixture_time_split_mismatch"})]
        expected = expected_turns(actor, prior, invalid, profile, split, splits, coverage, policy, expected_counts)
        validate_actor_chunks(chunks, actor, prior, expected, profile, split, catalog, policy,
                              counts, fixture_counts, token_histogram)
    return expected_counts, counts, dict(fixture_counts), token_histogram


def _actor_validation_batches(db, policy, groups, batch_size=4):
    """Materialize only a few actors while the parent verifies row ordering.

    Reading ``groups`` invokes the parent's index-entry callback. The index
    digest therefore retains exactly the serial file order, independent of
    worker completion order. Actors with no exported chunks are still submitted
    so workers detect missing targets and missing trailing prediction windows.
    """
    group, batch = next(groups, None), []
    for actor, raw in source_actors(db, policy):
        require(group is None or group[0] >= actor, "Unknown or unordered exported actor")
        chunks = list(group[1]) if group is not None and group[0] == actor else []
        if group is not None and group[0] == actor:
            group = next(groups, None)
        batch.append((actor, raw, chunks))
        if len(batch) == batch_size:
            yield batch
            batch = []
    require(group is None, "Extra output actors")
    if batch:
        yield batch


def validate_sequences(dataset_root, source_path, evidence_root, *, progress=print, workers=1):
    """Validate every release row against source-selected observations.

    A selected-cohort source is acceptable only when every eligible actor/market
    count matches the independently retained full activity ledger. Time bounds
    are recomputed from that narrower observed cohort, never asserted to be
    full-source bounds. Exact tokenizer recount is a separate same-manifest pass.
    """
    require(type(workers) is int and 1 <= workers <= 32, "Source validation workers must be an integer from 1 to 32")
    root, source = Path(dataset_root).resolve(), Path(source_path).resolve()
    before = source.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    wal = Path(str(source)+"-wal")
    require(not wal.exists() or wal.stat().st_size == 0, "Source has a live WAL")
    manifest = json.loads((root/"manifest.json").read_text())
    require(manifest.get("schema_version") == 3 and manifest.get("sample") is False, "Full validation requires a complete v3 export, not a sample")
    policy = SequencePolicy(**json.loads((root/"policy.json").read_text()))
    splits = json.loads((root/"split_policy.json").read_text())
    catalog = ContextCatalog(evidence_root, policy.initial_news_per_scope)
    fixtures = {c["fixture_id"] for c in catalog.contracts.values()}
    require(set(splits["fixture_splits"]) == fixtures and set(splits["fixture_splits"].values()) <= set(SPLITS), "Fixture split coverage mismatch")
    require(json.loads((root/"context_catalog.json").read_text()) == catalog.audit_catalog(), "Context catalog differs from verified evidence")
    require(manifest["source"] == {"sha256": sha(source), "bytes": before.st_size}, "Source hash/size mismatch")
    listed = set()
    for name, metadata in manifest["artifacts"].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and name not in listed, "Unsafe or repeated artifact path")
        listed.add(name)
        file = root/path
        require(file.is_file() and file.stat().st_size == metadata["bytes"] and sha(file) == metadata["sha256"], "Artifact checksum mismatch: "+name)
    require(listed == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}, "Unlisted or missing artifact")
    manifest_sha = sha(root/"manifest.json")
    db = sqlite3.connect(source.as_uri()+"?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    totals, expected_counts, token_histogram = Counter(), Counter(), Counter()
    fixture_counts = {f: Counter() for f in fixtures}
    profiles = {p: {s: {} for s in SPLITS} for p in PROFILES}
    expected_index = {(p,s): hashlib.sha256() for p in PROFILES for s in SPLITS}
    actual_index = {(p,s): hashlib.sha256() for p in PROFILES for s in SPLITS}

    def index_entry(row, path, line):
        value = {k: row[k] for k in ("sequence_id", "actor_id", "profile", "split", "chunk_index")}
        value.update(path=path, line=line, sha256=hashlib.sha256(canonical(row).encode()).hexdigest())
        expected_index[row["profile"],row["split"]].update((canonical(value)+"\n").encode())

    try:
        metadata = {r["key"]: json.loads(r["value_json"]) for r in db.execute("SELECT key,value_json FROM metadata")}
        require(json.loads((root/"source_metadata.json").read_text()) == metadata, "Published source scope/provenance metadata mismatch")
        require(metadata.get("source_scope") == "selected_cohort_with_full_count_ledger", "Unknown source observation scope")
        require(metadata.get("coverage_bound_scope") == "selected_cohort_observed_interval", "Unknown source coverage-bound scope")
        coverage = {r["condition_id"]: dict(r) for r in db.execute("SELECT * FROM condition_coverage")}
        require(set(coverage) == set(catalog.contracts), "Source contract coverage incomplete")
        require(json.loads((root/"source_coverage.json").read_text()) == coverage, "Published source coverage mismatch")
        allowed_statuses = {"api_exhausted","api_exhausted_raw_replay_verified","api_exhausted_selected_cohort_verified",
            "inherited_api_exhausted_selected_cohort_verified","recaptured_api_exhausted_selected_cohort_verified"}
        require(all(r["status"] in allowed_statuses and r["coverage_bound_scope"] == "selected_cohort_observed_interval"
                    and type(r["earliest_query_us"]) is int and type(r["latest_query_us"]) is int
                    and r["earliest_query_us"] <= r["latest_query_us"] for r in coverage.values()), "Invalid capture exhaustion/bound scope")
        pairs = iter(records(root, "source_evidence/pair_counts"))
        for row in db.execute("SELECT * FROM wallet_market_counts ORDER BY wallet,condition_id"):
            require(row["observation_count"] > 0, "Nonpositive full source pair count")
            require(next(pairs, None) == dict(row), "Full pair count ledger omitted or changed source rows")
            totals["full_source_pairs"] += 1
            totals["full_source_observations"] += row["observation_count"]
        require(next(pairs, None) is None, "Extra full pair count ledger rows")
        pages = iter(records(root, "source_evidence/pages"))
        for row in db.execute("SELECT * FROM source_pages ORDER BY source_page_id"):
            require(next(pages, None) == dict(row), "Source page provenance ledger mismatch")
            totals["source_page_references"] += 1
        require(next(pages, None) is None, "Extra source page provenance rows")
        # This query starts from the FULL denominator, so an entirely missing
        # eligible pair is detected as well as a partially recovered pair.
        missing = db.execute("""WITH actual AS (
            SELECT wallet,condition_id,COUNT(*) n FROM trades GROUP BY wallet,condition_id)
            SELECT w.wallet,w.condition_id,w.observation_count,a.n FROM wallet_market_counts w
            LEFT JOIN actual a ON a.wallet=w.wallet AND a.condition_id=w.condition_id
            WHERE w.observation_count<=20 AND COALESCE(a.n,0)<>w.observation_count LIMIT 1""").fetchone()
        require(missing is None, "Eligible pair missing or incomplete against full count ledger: "+str(tuple(missing) if missing else ""))
        unexpected = db.execute("""SELECT 1 FROM trades t LEFT JOIN wallet_market_counts w
            ON w.wallet=t.wallet AND w.condition_id=t.condition_id
            WHERE w.wallet IS NULL OR w.observation_count>20 LIMIT 1""").fetchone()
        require(unexpected is None, "Selected-cohort source contains undeclared or excluded pair rows")
        require(policy.max_trades_per_market <= 20, "Requested target threshold exceeds recovered cohort")
        full_conditions = {r["condition_id"]: r["n"] for r in db.execute("SELECT condition_id,SUM(observation_count) n FROM wallet_market_counts GROUP BY condition_id")}
        require(set(full_conditions) == set(coverage), "Full count ledger contract set mismatch")
        actual_conditions = list(db.execute("SELECT condition_id,COUNT(*) n,MIN(query_us) first,MAX(query_us) last FROM trades GROUP BY condition_id"))
        require({r["condition_id"] for r in actual_conditions} == set(coverage), "Selected source misses a configured contract")
        for row in actual_conditions:
            c = coverage[row["condition_id"]]
            require((row["n"],row["first"],row["last"]) == (c["selected_observation_count"],c["earliest_query_us"],c["latest_query_us"]), "Selected observation interval/count mismatch")
            require(c["observation_count"] == full_conditions[row["condition_id"]], "Full-condition denominator mismatch")
            require(c["fixture_id"] == catalog.contracts[row["condition_id"]]["fixture_id"], "Coverage fixture mapping mismatch")
        progress("Hashes and every recovered eligible pair verified against full ledger")
        observations, quarantine = iter(records(root,"observations")), Counter()
        for actor, raw in source_actors(db, policy):
            expected_counts["actors"] += 1
            for row in normalize_source(raw,catalog,splits):
                require(next(observations,None) == row, "Canonical selected observation omitted or changed")
                totals["selected_observations"] += 1
                fixture_counts[row["fixture_id"]]["selected_observations"] += 1
                if row["errors"]:
                    quarantine[row["observation_id"],tuple(row["errors"])] += 1
                    totals["quarantined_observations"] += 1
                if not row["errors"]:
                    fixture_counts[row["fixture_id"]]["conditional_targets"] += 1
        require(next(observations,None) is None, "Extra selected observations")
        require(Counter((r["observation_id"],tuple(r["errors"])) for r in records(root,"quarantine")) == quarantine, "Quarantine fails source reconciliation")
        for profile in PROFILES:
            for split in SPLITS:
                groups = iter(groupby(records(root,profile+"/"+split,index_entry),lambda r:r["actor_id"]))
                counts = Counter()
                if workers == 1:
                    group = next(groups,None)
                    for actor, raw in source_actors(db, policy):
                        normalized = normalize_source(raw,catalog,splits)
                        invalid = {r["condition_id"] for r in normalized if set(r["errors"])-{"fixture_time_split_mismatch"}}
                        prior = [r for r in normalized if r["history_split"] == split and not(set(r["errors"])-{"fixture_time_split_mismatch"})]
                        require(group is None or group[0] >= actor, "Unknown or unordered exported actor")
                        chunks = group[1] if group is not None and group[0] == actor else []
                        expected = expected_turns(actor,prior,invalid,profile,split,splits,coverage,policy,expected_counts)
                        validate_actor_chunks(chunks,actor,prior,expected,profile,split,catalog,policy,counts,fixture_counts,token_histogram)
                        if group is not None and group[0] == actor:
                            group = next(groups,None)
                    require(group is None, "Extra output actors")
                else:
                    batches = _actor_validation_batches(db, policy, groups)
                    completed_batches = 0
                    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                            initializer=_validation_worker_init,
                            initargs=(catalog, asdict(policy), splits, coverage, profile, split)) as executor:
                        for expected_delta, counts_delta, fixture_delta, histogram_delta in _bounded_ordered_results(
                                executor, _validate_actor_batch, batches, workers*2):
                            expected_counts.update(expected_delta)
                            counts.update(counts_delta)
                            token_histogram.update(histogram_delta)
                            for fixture, delta in fixture_delta.items():
                                fixture_counts[fixture].update(delta)
                            completed_batches += 1
                            if completed_batches % 1250 == 0:
                                progress(f"Source validation {profile}/{split}: "
                                         f"{completed_batches*4:,} actors / {counts['target_turns']:,} turns")
                profiles[profile][split] = dict(counts)
                for key in ("conversations","target_turns","tokens","long_context_sequences"):
                    require(counts[key] == manifest["profiles"][profile][split].get(key,0), "Manifest profile totals mismatch")
                target_key = "conditional_target_observations" if profile == "conditional_trades" else "scheduled_target_observations"
                expected_counts[target_key] += counts["target_observations"]
                progress(f"Verified {profile}/{split}: {counts['conversations']:,} conversations / {counts['target_turns']:,} turns")
        require(Counter(manifest["counts"]) == expected_counts, "Manifest actor/window/target counts mismatch")
        for key in ("selected_observations","quarantined_observations"):
            require(manifest[key] == totals[key], "Manifest selected/quarantine count mismatch")
        require(json.loads((root/"fixture_coverage.json").read_text()) == {f:dict(c) for f,c in fixture_counts.items() if c}, "Fixture coverage counts mismatch")
        coverage_report = {key:{"present":sum(c[key]>0 for c in fixture_counts.values()),
            "missing":sorted(f for f,c in fixture_counts.items() if c[key]<=0)}
            for key in ("selected_observations","conditional_targets","scheduled_target_observations")}
        require(manifest["fixture_target_coverage"] == coverage_report, "Fixture coverage summary mismatch")
        require(not any(r["missing"] for r in coverage_report.values()), "Some configured matches lack positive targets in one profile")
        index_count = 0
        for row in records(root,"actor_index"):
            require(set(row) == {"sequence_id","actor_id","profile","split","chunk_index","path","line","sha256"}, "Actor index schema mismatch")
            key = row["profile"],row["split"]
            require(key in actual_index,"Unknown actor index partition")
            actual_index[key].update((canonical(row)+"\n").encode())
            index_count += 1
        require(all(actual_index[k].digest()==expected_index[k].digest() for k in actual_index), "Actor index omitted, duplicated, changed or mislocated a row")
        require(index_count == sum(token_histogram.values()), "Actor index count mismatch")
        def quantile(fraction):
            boundary, accumulated = max(1,math.ceil(index_count*fraction)),0
            for value,number in sorted(token_histogram.items()):
                accumulated += number
                if accumulated >= boundary:
                    return value
        require(manifest["tokens"] == {"total":sum(k*v for k,v in token_histogram.items()),
            "p50":quantile(.5),"p95":quantile(.95),"max":max(token_histogram,default=0)}, "Token distribution manifest mismatch")
    finally:
        db.close()
    after = source.stat()
    require(identity == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            and (not wal.exists() or wal.stat().st_size == 0), "Source changed during validation")
    require(sha(root/"manifest.json") == manifest_sha, "Manifest changed during validation")
    return {"schema_version":1,"status":"passed","manifest_sha256":manifest_sha,
        "source_sha256":manifest["source"]["sha256"],"artifact_count":len(listed),
        "contracts":len(catalog.contracts),"fixtures":len(fixtures),
        "source_reconciliation":dict(totals),"profiles":profiles,"fixture_target_coverage":coverage_report,
        "actual_token_counts_recomputed":False,
        "validation_scope":"All selected observations, complete retained prediction windows, all labels/context, full cohort denominator, indexes, hashes and gzip integrity",
        "limitations":["Absence is within recovered captured observations and conservative selected-cohort time bounds",
                       "Inherited page references are provenance, not freshly replayed complete raw captures",
                       "API block timestamps are availability proxies; no predictive performance has been measured"]}
