"""Small, explicitly current-state API snapshots for offline tests."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode


FEATURES = ('actor_market_value', 'actor_positions_open', 'actor_positions_closed')


def snapshot_fixture(actor, market, *, value='0', open_positions=None, closed_positions=None):
    snapshot = {
        'version': 1, 'actor_id': actor, 'market_id': market['market_id'],
        'condition_id': market['condition_id'],
        'temporal_scope': 'collection_time_not_trade_time', 'historical_model_input': False,
    }
    for feature in FEATURES:
        params = {'user': actor, 'condition': market['condition_id']}
        if feature == 'actor_market_value':
            endpoint = 'value'
            data = {'proxy_wallet': actor, 'value': value}
        else:
            endpoint = 'positions'
            status = 'OPEN' if feature.endswith('_open') else 'CLOSED'
            params.update(status=status, limit=500, filter_type='TOKENS', filter_amount='0.000001')
            if status == 'OPEN':
                params['include_archived'] = 'true'
            data = (open_positions if status == 'OPEN' else closed_positions) or []
        snapshot[feature] = {
            'status': 'ok', 'data': data,
            'pages': [{
                'url': 'https://data-api.polymarket.com/v2/' + endpoint + '?' + urlencode(params),
                'retrieved_at': '2026-09-28T04:00:00Z',
                'body_sha256': hashlib.sha256(json.dumps(data).encode()).hexdigest(),
                'from_cache': False,
            }],
        }
    return snapshot


def position_fixture(actor, market, *, status='OPEN', token_id='101', **updates):
    row = {
        'proxy_wallet': actor, 'condition_id': market['condition_id'], 'token_id': token_id,
        'current_size': '3' if status == 'OPEN' else '0', 'avg_price': '0.2',
        'entry_cost_usdc': '0.6', 'entry_fees_usdc': '0.01', 'total_cost_usdc': '0.61',
        'current_price': '0.8', 'current_value': '2.4' if status == 'OPEN' else '0',
        'total_size': '3', 'realized_pnl': '0.2', 'unrealized_pnl': '1.8',
        'total_pnl': '2', 'status': status,
        'outcome': 'Yes' if token_id == '101' else 'No',
        'redeemable': False, 'mergeable': False,
        'last_event_at': '2026-06-01T17:10:00Z',
    }
    row.update(updates)
    return row


def write_snapshot(directory, actor, market, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (actor + '.json')
    path.write_text(json.dumps(snapshot_fixture(actor, market, **kwargs)) + '\n')
    return path


def report(actors=1):
    return {'version': 1, 'directory': 'actor_snapshots', 'actors': actors,
            'temporal_scope': 'collection_time_not_trade_time', 'historical_model_input': False}
