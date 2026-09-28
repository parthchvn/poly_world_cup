"""Resumable capture of the public API's wallet execution feed.

The v2 user feed accepts inclusive start/end epoch seconds and keyset cursors.
Every request repeats the same filters, including start=1 (omitting it defaults
to three years). Exhausting this API does not prove complete wallet accounting.
Execution timestamps are historical proxies, not verified publication times.

The caller supplies a client with get_json(url, params=...) returning data, url,
retrieved_at and body_sha256 attributes (build_actor_dataset.HttpClient works).
No positions, PnL snapshots or current wallet totals enter this capture.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

API_URL = 'https://data-api.polymarket.com/v2/trades'
SCHEMA = 'wallet_execution_capture_v1'
AVAILABILITY = 'execution_timestamp_proxy_not_verified_publication_time'
ADDRESS = re.compile(r'0x[0-9a-fA-F]{40}')
CONDITION = re.compile(r'0x[0-9a-fA-F]{64}')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _write(path, value):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    _require(path.is_file() and not path.is_symlink(), f'Missing or unsafe capture file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


@contextmanager
def _lock(directory):
    import fcntl
    with (directory / '.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another collector is writing this wallet capture') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _decimal(value, name, *, positive=False, maximum=None):
    _require(not isinstance(value, bool) and isinstance(value, (int, float, str, Decimal)),
             f'Invalid {name}')
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f'Invalid {name}') from error
    _require(number.is_finite() and number >= 0 and (not positive or number > 0)
             and (maximum is None or number <= maximum), f'Invalid {name}')
    text = format(number, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _iso(seconds):
    _require(type(seconds) is int and seconds >= 1, 'Timestamp must be positive epoch seconds')
    try:
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace('+00:00', 'Z')
    except (ValueError, OverflowError, OSError) as error:
        raise ValueError('Timestamp is outside the supported range') from error


def _parameters(actor_id, start_seconds, end_seconds, limit, minimum_size):
    _require(isinstance(actor_id, str) and ADDRESS.fullmatch(actor_id), 'Invalid actor wallet')
    _iso(start_seconds)
    _iso(end_seconds)
    _require(start_seconds <= end_seconds, 'Wallet capture start is after end')
    _require(type(limit) is int and 1 <= limit <= 1000, 'limit must be between 1 and 1000')
    return {'user': actor_id.lower(), 'start': start_seconds, 'end': end_seconds,
            'limit': limit, 'taker_only': False, 'filter_type': 'TOKENS',
            'filter_amount': _decimal(minimum_size, 'minimum_size', positive=True)}


def _normalize(row, parameters, source, row_index):
    _require(isinstance(row, dict), 'Wallet execution must be an object')
    actor = row.get('proxy_wallet')
    _require(isinstance(actor, str) and actor.lower() == parameters['user'],
             'Wallet API returned a different actor')
    condition = row.get('condition_id')
    _require(isinstance(condition, str) and CONDITION.fullmatch(condition), 'Invalid condition_id')
    seconds = row.get('timestamp')
    timestamp = _iso(seconds)
    _require(parameters['start'] <= seconds <= parameters['end'],
             'Wallet API returned an execution outside the requested time bounds')
    _require(row.get('side') in ('BUY', 'SELL'), 'Invalid execution side')
    token = row.get('token_id')
    _require(isinstance(token, str) and token.isascii() and token.isdigit() and int(token) > 0,
             'Invalid token_id')
    transaction = row.get('transaction_hash')
    _require(isinstance(transaction, str) and CONDITION.fullmatch(transaction), 'Invalid transaction_hash')
    source_ids = {name: str(row[name]) for name in
                  ('execution_id', 'trade_id', 'fill_id', 'id', 'sequence', 'log_index')
                  if row.get(name) is not None}
    unique = next((name for name in ('execution_id', 'trade_id', 'fill_id')
                   if source_ids.get(name)), None)
    # Content-based deduplication would destroy legitimate equal-price fills.
    # An ordinal identifies a served observation, not a canonical on-chain fill.
    identity = ('api_' + unique + ':' + source_ids[unique] if unique else
                'api_observation:' + _hash(_bytes([source['request_url'], row_index])))
    normalized = {
        'actor_id': parameters['user'], 'condition_id': condition.lower(),
        'execution_id': identity, 'timestamp': timestamp, 'side': row['side'],
        'shares': _decimal(row.get('size'), 'size', positive=True),
        'price': _decimal(row.get('price'), 'price', maximum=Decimal(1)),
        'token_id': token, 'transaction_hash': transaction.lower(),
        'availability_semantics': AVAILABILITY,
        'publicly_available_at_upper_bound': None,
        'retrieved_at': source['retrieved_at'],
        'source': {**source, 'row_index': row_index, 'source_ids': source_ids,
                   'identity_quality': 'source_execution_id' if unique else 'api_observation'},
    }
    if isinstance(row.get('outcome'), str) and row['outcome']:
        normalized['outcome'] = row['outcome']
    return normalized


def _page(result, parameters, cursor, seen_cursors, index):
    envelope = result.data
    _require(isinstance(envelope, dict) and isinstance(envelope.get('data'), list),
             'Expected v2 wallet {data, pagination} envelope')
    pagination = envelope.get('pagination')
    _require(isinstance(pagination, dict) and type(pagination.get('has_more')) is bool,
             'Missing pagination.has_more')
    has_more, next_cursor = pagination['has_more'], pagination.get('next_cursor')
    if has_more:
        _require(isinstance(next_cursor, str) and next_cursor and next_cursor != cursor
                 and next_cursor not in seen_cursors, 'Repeated or invalid wallet pagination cursor')
        _require(bool(envelope['data']), 'Nonterminal wallet page contains no observations')
    else:
        _require(next_cursor in (None, ''), 'Terminal wallet page has a continuation cursor')
        next_cursor = None
    source = {'provider': 'polymarket_data_api_v2', 'request_url': result.url,
              'body_sha256': result.body_sha256, 'retrieved_at': result.retrieved_at}
    rows = [_normalize(row, parameters, source, number)
            for number, row in enumerate(envelope['data'])]
    _require(len(rows) <= parameters['limit'], 'Wallet API exceeded the requested page size')
    return {'schema': SCHEMA, 'page_index': index, 'parameters': parameters,
            'requested_cursor': cursor, 'next_cursor': next_cursor, 'has_more': has_more,
            'source': source, 'rows': rows}


def _check_page(page, parameters, index, cursor, seen_cursors):
    _require(page.get('schema') == SCHEMA and page.get('parameters') == parameters
             and page.get('page_index') == index and page.get('requested_cursor') == cursor,
             'Wallet page identity or cursor chain differs from its manifest')
    more, next_cursor = page.get('has_more'), page.get('next_cursor')
    _require(type(more) is bool, 'Invalid saved wallet pagination')
    _require((isinstance(next_cursor, str) and bool(next_cursor)
              and next_cursor != cursor and next_cursor not in seen_cursors) if more else next_cursor is None,
             'Invalid saved wallet continuation cursor')
    rows = page.get('rows')
    _require(isinstance(rows, list) and len(rows) <= parameters['limit'] and (rows or not more),
             'Invalid saved wallet rows')
    for row in rows:
        _require(isinstance(row, dict) and row.get('actor_id') == parameters['user']
                 and row.get('availability_semantics') == AVAILABILITY
                 and 'known_at' not in row, 'Invalid saved wallet identity or availability')
    return rows


def _commit(state, page, directory):
    index = len(state['pages'])
    filename = f'pages/{index:08d}.json'
    state['pages'].append({'file': filename, 'sha256': _hash((directory / filename).read_bytes()),
                           'row_count': len(page['rows'])})
    state['row_count'] += len(page['rows'])
    state['page_count'] = len(state['pages'])
    state['next_cursor'] = page['next_cursor']
    state['api_traversal_status'] = 'paused' if page['has_more'] else 'exhausted'
    times = [row['timestamp'] for row in page['rows']]
    if times:
        state['earliest_execution'] = min([state['earliest_execution']] + times
                                           if state['earliest_execution'] else times)
        state['latest_execution'] = max([state['latest_execution']] + times
                                         if state['latest_execution'] else times)
    _write(directory / 'manifest.json', state)


def _validated_state(directory, expected=None):
    state = _read(directory / 'manifest.json')
    _require(state.get('schema') == SCHEMA, 'Unsupported wallet capture format')
    parameters = state.get('parameters', {})
    valid_parameters = _parameters(parameters.get('user'), parameters.get('start'), parameters.get('end'),
                                   parameters.get('limit'), parameters.get('filter_amount'))
    _require(parameters == valid_parameters and (expected is None or parameters == expected),
             'Wallet capture parameters differ from the requested capture')
    pages = state.get('pages')
    _require(isinstance(pages, list) and len(pages) == state.get('page_count'), 'Invalid wallet page count')
    cursor, count, seen, ids, times = None, 0, set(), set(), []
    for index, item in enumerate(pages):
        filename = f'pages/{index:08d}.json'
        _require(item.get('file') == filename, 'Invalid wallet page path')
        path = directory / filename
        page = _read(path)
        _require(_hash(path.read_bytes()) == item.get('sha256'), 'Wallet page checksum mismatch')
        rows = _check_page(page, parameters, index, cursor, seen)
        _require(len(rows) == item.get('row_count'), 'Wallet page row count mismatch')
        _require(page['has_more'] or index == len(pages) - 1, 'Pages occur after wallet feed exhaustion')
        for row in rows:
            identity = row.get('execution_id')
            _require(isinstance(identity, str) and identity and identity not in ids,
                     'Duplicate wallet execution identity across captured pages')
            ids.add(identity)
            times.append(row['timestamp'])
        seen.add(cursor)
        cursor = page['next_cursor']
        count += len(rows)
    status = 'exhausted' if pages and not page['has_more'] else 'paused'
    _require(state.get('next_cursor') == cursor and state.get('row_count') == count
             and state.get('api_traversal_status') == status
             and state.get('earliest_execution') == (min(times) if times else None)
             and state.get('latest_execution') == (max(times) if times else None),
             'Wallet manifest summary differs from its pages')
    _require(state.get('historical_completeness_verified') is False
             and state.get('availability_semantics') == AVAILABILITY, 'Invalid wallet coverage claims')
    return state


def ingest_wallet(client, *, actor_id, output_dir, end_seconds, start_seconds=1,
                  limit=1000, max_pages=None, minimum_size='0.000001',
                  progress=False, require_nonempty=True):
    """Return a resumable capture manifest plus ``capture_dir`` for the reader.

    Each capture is immutable for a given parameter set. ``max_pages`` bounds
    pages added by this invocation; paused captures cannot enter metrics. An
    exhausted empty response is persisted as audit but fails by default.
    """
    parameters = _parameters(actor_id, start_seconds, end_seconds, limit, minimum_size)
    _require(max_pages is None or type(max_pages) is int and max_pages > 0,
             'max_pages must be a positive integer')
    directory = Path(output_dir) / actor_id.lower() / _hash(_bytes(parameters))[:24]
    (directory / 'pages').mkdir(parents=True, exist_ok=True)
    with _lock(directory):
        if (directory / 'manifest.json').exists():
            state = _validated_state(directory, parameters)
        else:
            state = {'schema': SCHEMA, 'actor_id': actor_id.lower(), 'parameters': parameters,
                     'api_url': API_URL, 'api_traversal_status': 'paused', 'next_cursor': None,
                     'page_count': 0, 'row_count': 0, 'pages': [],
                     'earliest_execution': None, 'latest_execution': None,
                     'coverage': 'api_served_wallet_executions_in_requested_window',
                     'historical_completeness_verified': False,
                     'availability_semantics': AVAILABILITY,
                     'missing_accounting': ['deposits', 'withdrawals', 'transfers', 'redemptions',
                                            'splits', 'merges', 'fees', 'combo_activity']}
            _write(directory / 'manifest.json', state)
        added, seen, ids = 0, set(), set()
        for item in state['pages']:
            saved = _read(directory / item['file'])
            seen.add(saved['requested_cursor'])
            ids.update(row['execution_id'] for row in saved['rows'])
        while state['api_traversal_status'] != 'exhausted' and (max_pages is None or added < max_pages):
            index, cursor = state['page_count'], state['next_cursor']
            path = directory / 'pages' / f'{index:08d}.json'
            if path.exists():
                page = _read(path)  # recover a page committed before an interrupted manifest update
                _check_page(page, parameters, index, cursor, seen)
            else:
                params = dict(parameters)
                if cursor is not None:
                    params['cursor'] = cursor
                result = client.get_json(API_URL, params=params)
                page = _page(result, parameters, cursor, seen, index)
            page_ids = [row['execution_id'] for row in page['rows']]
            _require(len(page_ids) == len(set(page_ids)) and not ids.intersection(page_ids),
                     'Duplicate source execution identity in wallet traversal')
            if not path.exists():
                _write(path, page)
            _commit(state, page, directory)
            seen.add(cursor)
            ids.update(page_ids)
            added += 1
            if progress:
                print(f"Wallet {actor_id}: {state['page_count']:,} pages / {state['row_count']:,} executions; "
                      f"status={state['api_traversal_status']}", flush=True)
        if require_nonempty and state['api_traversal_status'] == 'exhausted':
            _require(state['row_count'] > 0, f'Wallet API returned no executions for required actor {actor_id}')
        return {**state, 'capture_dir': str(directory.resolve())}


def iter_wallet_observations(capture, *, require_exhausted=True, require_nonempty=True):
    """Verify the capture and yield observations, preserving source multiplicity."""
    directory = Path(capture['capture_dir'] if isinstance(capture, dict) else capture)
    if directory.name == 'manifest.json':
        directory = directory.parent
    state = _validated_state(directory)
    if require_exhausted:
        _require(state['api_traversal_status'] == 'exhausted', 'Wallet capture is incomplete; resume collection')
    if require_nonempty:
        _require(state['row_count'] > 0, 'Wallet capture has no executions')
    for item in state['pages']:
        yield from _read(directory / item['file'])['rows']
