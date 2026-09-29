"""CPU-only contracts shared by held-out preparation, prediction and reporting."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from compare_actor_variants import conversation, lines, _metric_context, require, loads

FORMAT = 'world_cup_paired_evaluation_v1'
METRICS = ('valid_json', 'trade_count_correct', 'side_multiset_correct',
           'outcome_multiset_correct', 'side_outcome_multiset_correct', 'exact_trade_multiset')


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def check_pair(basic, inmarket):
    conversation(basic)
    conversation(inmarket)
    require({k: v for k, v in basic.items() if k != 'messages'} ==
            {k: v for k, v in inmarket.items() if k != 'messages'}, 'Paired conversation metadata differs')
    require(len(basic['messages']) == len(inmarket['messages']), 'Paired message count differs')
    for left, right in zip(basic['messages'], inmarket['messages']):
        require(left['role'] == right['role'], 'Paired roles differ')
        if left['role'] == 'user':
            require(_metric_context(loads(left['content']), 'basic') ==
                    _metric_context(loads(right['content']), 'inmarket'), 'Base contexts differ')
        else:
            require(left == right, 'System prompt or target differs')


def targets(record):
    """Observed-history, one-step prediction. Never include the current answer."""
    conversation(record)
    for offset in range(2, len(record['messages']), 2):
        context = loads(record['messages'][offset - 1]['content'])
        key = [record['sequence_id'], context['query_time']]
        yield {'id': hashlib.sha256(dump(key).encode()).hexdigest(),
               'sequence_id': record['sequence_id'], 'actor_id': record['actor_id'],
               'fixture_id': record['fixture_id'], 'market_id': record['market_id'],
               'query_time': context['query_time'], 'turn': offset // 2,
               'messages': record['messages'][:offset],
               'answer': record['messages'][offset]['content']}


def read_bundle(root):
    root = Path(root)
    meta = json.loads((root / 'manifest.json').read_text())
    require(meta.get('format') == FORMAT, 'Unsupported evaluation bundle')
    records = {}
    for variant in ('basic', 'inmarket'):
        path = root / f'{variant}.jsonl.gz'
        require(sha(path) == meta['files'][variant], f'{variant} evaluation file checksum failed')
        records[variant] = [r for _, r in lines(path)]
    require(len(records['basic']) == len(records['inmarket']) > 0, 'Empty or unpaired evaluation')
    seen, identities = set(), []
    for b, m in zip(records['basic'], records['inmarket']):
        check_pair(b, m)
        for target in targets(b):
            parsed_trades(target['answer'])
            require(target['id'] not in seen, 'Duplicate evaluation target')
            seen.add(target['id'])
            identities.append([target['id'], target['answer']])
    require(len(identities) == meta['targets'], 'Evaluation count mismatch')
    require(hashlib.sha256(dump(identities).encode()).hexdigest() == meta['target_sha256'],
            'Evaluation labels changed')
    return meta, records


def parsed_trades(text):
    value = loads(text.strip())
    require(isinstance(value, dict) and set(value) == {'action', 'trades'} and
            value['action'] == 'TRADE', 'Expected TRADE JSON with only action/trades')
    trades = value['trades']
    require(isinstance(trades, list) and 0 < len(trades) <= 20, 'Expected 1..20 trades')
    result = []
    for t in trades:
        require(isinstance(t, dict) and set(t) == {'side', 'outcome', 'shares', 'price'},
                'Unexpected trade schema')
        require(t['side'] in ('BUY', 'SELL') and t['outcome'] in ('Yes', 'No'), 'Invalid side/outcome')
        for k in ('shares', 'price'):
            require(isinstance(t[k], (str, int, float, Decimal)) and not isinstance(t[k], bool),
                    'Invalid numeric field')
        shares, price = Decimal(str(t['shares'])), Decimal(str(t['price']))
        require(shares.is_finite() and 0 < shares <= Decimal('1e100') and price.is_finite() and 0 <= price <= 1,
                'Invalid shares/price range')
        result.append((t['side'], t['outcome'], shares, price))
    return result


def score(answer, prediction):
    truth = parsed_trades(answer)
    result = dict.fromkeys(METRICS, 0)
    result.update(numeric_matched_trades=0, price_abs_error_sum=0.0,
                  shares_abs_error_sum=0.0, notional_abs_error_sum=0.0)
    try:
        pred = parsed_trades(prediction)
    except (ValueError, TypeError, ArithmeticError):
        return result
    result['valid_json'] = 1  # Includes strict schema and finite numeric ranges.
    result['trade_count_correct'] = int(len(truth) == len(pred))
    for indices, key in (((0,), 'side_multiset_correct'), ((1,), 'outcome_multiset_correct'),
                         ((0, 1), 'side_outcome_multiset_correct'), ((0, 1, 2, 3), 'exact_trade_multiset')):
        result[key] = int(Counter(tuple(t[i] for i in indices) for t in truth) ==
                          Counter(tuple(t[i] for i in indices) for t in pred))
    # Conditional numeric errors: exact category/count match required. Within each
    # side/outcome category sort by shares, then price; do not use oracle assignment.
    if result['side_outcome_multiset_correct']:
        for a, b in zip(sorted(truth), sorted(pred)):
            result['numeric_matched_trades'] += 1
            result['price_abs_error_sum'] += float(abs(a[3] - b[3]))
            result['shares_abs_error_sum'] += float(abs(a[2] - b[2]))
            result['notional_abs_error_sum'] += float(abs(a[2] * a[3] - b[2] * b[3]))
    return result


def summarize(rows):
    require(bool(rows), 'No predictions to score')
    scores = [score(r['answer'], r['prediction']) for r in rows]
    total = len(scores)
    matched = sum(s['numeric_matched_trades'] for s in scores)
    result = {'targets': total, **{k: sum(s[k] for s in scores) / total for k in METRICS},
              'numeric_matched_trades': matched,
              'numeric_target_coverage': sum(s['side_outcome_multiset_correct'] for s in scores) / total,
              'generation_limit_hits': sum(bool(r.get('hit_generation_limit')) for r in rows)}
    for k in ('price', 'shares', 'notional'):
        result[k + '_mae_conditional'] = (sum(s[k + '_abs_error_sum'] for s in scores) / matched
                                         if matched else None)
    return result
