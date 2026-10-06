"""Fixed evaluation tolerances for interval trade point predictions.

Each (side, outcome) is one prediction unit: total executed shares and their
share-weighted mean price in the interval. This removes exchange fill splitting
from the score. Tolerances are fixed before training, not inferred confidence
intervals and not price changes relative to the current quote.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
import json

SEMANTICS = 'side_outcome_window_aggregates_v1'
TOLERANCE_KEYS = {'price_delta', 'shares_relative_delta', 'shares_absolute_delta'}


def decimal_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f'{name} must be a finite decimal number')
    text = str(value)
    if len(text) > 128:
        raise ValueError(f'{name} has excessive numeric precision')
    try:
        number = Decimal(text)
    except InvalidOperation:
        raise ValueError(f'{name} must be a finite decimal number') from None
    if not number.is_finite() or (number and abs(number.adjusted()) > 100):
        raise ValueError(f'{name} must be finite and within the supported numerical range')
    return number


def decimal_text(value):
    with localcontext() as context:
        context.prec = 40
        result = format(+value, 'f')
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    return result or '0'


def validate_tolerances(value):
    if not isinstance(value, dict) or set(value) != TOLERANCE_KEYS:
        raise ValueError('Provide price_delta, shares_relative_delta, and shares_absolute_delta explicitly')
    parsed = {name: decimal_number(value[name], name) for name in TOLERANCE_KEYS}
    if not 0 <= parsed['price_delta'] <= 1:
        raise ValueError('price_delta must be in [0,1], in price units (0.02 means two cents)')
    if not 0 <= parsed['shares_relative_delta'] < 1:
        raise ValueError('shares_relative_delta must be in [0,1), e.g. 0.20 for twenty percent')
    if parsed['shares_absolute_delta'] < 0:
        raise ValueError('shares_absolute_delta must be nonnegative')
    return {name: decimal_text(parsed[name]) for name in sorted(parsed)}


def normalize_trade(trade):
    if not isinstance(trade, dict) or set(trade) != {'side', 'outcome', 'price', 'shares'}:
        raise ValueError('Each trade must contain exactly side, outcome, price and shares')
    if not isinstance(trade['side'], str) or trade['side'].upper() not in ('BUY', 'SELL'):
        raise ValueError('side must be BUY or SELL')
    if not isinstance(trade['outcome'], str) or trade['outcome'].lower() not in ('yes', 'no'):
        raise ValueError('outcome must be Yes or No')
    price = decimal_number(trade['price'], 'price')
    shares = decimal_number(trade['shares'], 'shares')
    if not 0 <= price <= 1 or shares <= 0:
        raise ValueError('price must be in [0,1] and shares must be positive')
    return {'side': trade['side'].upper(), 'outcome': trade['outcome'].title(),
            'price': price, 'shares': shares}


def aggregate_trades(trades):
    """Canonical total shares / VWAP per category; never net BUY against SELL."""
    if not isinstance(trades, list):
        trades = list(trades)
    groups = defaultdict(lambda: [Decimal(0), Decimal(0)])
    # Source decimal strings retain precision before the final serialisation.
    with localcontext() as context:
        context.prec = 256
        for raw in trades:
            trade = normalize_trade(raw)
            values = groups[(trade['side'], trade['outcome'])]
            values[0] += trade['shares']
            values[1] += trade['shares'] * trade['price']
        return [{'side': side, 'outcome': outcome, 'shares': decimal_text(total),
                 'price': decimal_text(notional / total)}
                for (side, outcome), (total, notional) in sorted(groups.items())]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_decision(value):
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_float=Decimal, object_pairs_hook=_unique_object)
        except (ValueError, TypeError) as error:
            raise ValueError('Prediction must be one valid JSON object') from error
    if not isinstance(value, dict) or value.get('action') not in ('TRADE', 'NO_TRADE'):
        raise ValueError('Expected action TRADE or NO_TRADE')
    if value['action'] == 'NO_TRADE':
        if set(value) != {'action'}:
            raise ValueError('NO_TRADE must not contain trade details or extra fields')
        return {'action': 'NO_TRADE', 'trades': []}
    if set(value) != {'action', 'trades'} or not isinstance(value.get('trades'), list) or not value['trades']:
        raise ValueError('TRADE needs a nonempty trades list')
    if len(value['trades']) > 256:
        raise ValueError('Prediction contains more than 256 trade entries')
    aggregates = aggregate_trades(value['trades'])
    # Individually finite entries can add up to an out-of-range quantity. Reject
    # that inside parsing so a malformed prediction cannot abort evaluation.
    for trade in aggregates:
        normalize_trade(trade)
    return {'action': 'TRADE', 'trades': aggregates}


def score_trade_details(gold, prediction, tolerances):
    """Score one interval with inclusive absolute-price and relative-size bounds.

    A category matches iff side/outcome agree and BOTH numeric errors pass.
    Relative share tolerance is measured against the observed total shares:
    |q_pred-q_true| <= max(absolute_delta, relative_delta*q_true).
    """
    tolerance = {key: Decimal(value) for key, value in validate_tolerances(tolerances).items()}
    truth = parse_decision(gold)  # Bad source labels must never be counted as a model error.
    result = {'gold_action': truth['action'], 'predicted_action': None,
              'valid_prediction': False, 'action_correct': False,
              'true_count': len(truth['trades']), 'predicted_count': 0, 'matched_count': 0,
              'price_matched_count': 0, 'shares_matched_count': 0,
              'all_trades_matched': False, 'joint_correct': False,
              'parse_error': None, 'matched_pairs': []}
    try:
        predicted = parse_decision(prediction)
    except (ValueError, TypeError) as error:
        result['parse_error'] = str(error)
        return result
    result.update(valid_prediction=True, predicted_action=predicted['action'],
                  action_correct=truth['action'] == predicted['action'], predicted_count=len(predicted['trades']))
    expected = {(t['side'], t['outcome']): (index, normalize_trade(t))
                for index, t in enumerate(truth['trades'])}
    with localcontext() as context:
        context.prec = 256
        for index, raw in enumerate(predicted['trades']):
            category = (raw['side'], raw['outcome'])
            if category not in expected:
                continue
            gold_index, actual = expected[category]
            guessed = normalize_trade(raw)
            price_error = abs(guessed['price'] - actual['price'])
            shares_error = abs(guessed['shares'] - actual['shares'])
            shares_limit = max(tolerance['shares_absolute_delta'],
                               tolerance['shares_relative_delta'] * actual['shares'])
            price_ok = price_error <= tolerance['price_delta']
            shares_ok = shares_error <= shares_limit
            result['price_matched_count'] += int(price_ok)
            result['shares_matched_count'] += int(shares_ok)
            if price_ok and shares_ok:
                result['matched_count'] += 1
                result['matched_pairs'].append({'gold_index': gold_index, 'prediction_index': index,
                    'side': category[0], 'outcome': category[1],
                    'absolute_price_error': decimal_text(price_error),
                    'absolute_shares_error': decimal_text(shares_error),
                    'allowed_shares_error': decimal_text(shares_limit)})
    result['all_trades_matched'] = result['matched_count'] == result['true_count'] == result['predicted_count']
    result['joint_correct'] = result['action_correct'] and result['all_trades_matched']
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _f1(precision, recall):
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize_trade_details(scores):
    """Point estimates; dependent windows are not treated as independent trials."""
    scores = list(scores)
    n = len(scores)
    true_count = sum(row['true_count'] for row in scores)
    pred_count = sum(row['predicted_count'] for row in scores)
    matched = sum(row['matched_count'] for row in scores)
    precision, recall = _ratio(matched, pred_count), _ratio(matched, true_count)
    positive = [row for row in scores if row['gold_action'] == 'TRADE']
    negative = [row for row in scores if row['gold_action'] == 'NO_TRADE']
    tp = sum(row['gold_action'] == row['predicted_action'] == 'TRADE' for row in scores)
    fp = sum(row['gold_action'] == 'NO_TRADE' and row['predicted_action'] == 'TRADE' for row in scores)
    fn = sum(row['gold_action'] == 'TRADE' and row['predicted_action'] != 'TRADE' for row in scores)
    tn = sum(row['gold_action'] == row['predicted_action'] == 'NO_TRADE' for row in scores)
    action_precision, action_recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {'rows': n, 'json_valid_rate': _ratio(sum(row['valid_prediction'] for row in scores), n),
        'invalid_prediction_count': sum(not row['valid_prediction'] for row in scores),
        'action_accuracy': _ratio(sum(row['action_correct'] for row in scores), n),
        'trade_action_precision': action_precision, 'trade_action_recall': action_recall,
        'trade_action_f1': _ratio(2 * tp, 2 * tp + fp + fn),
        'action_counts': {'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
                          'invalid_on_no_trade': sum(not row['valid_prediction'] for row in negative)},
        'true_trade_count': true_count, 'predicted_trade_count': pred_count, 'matched_trade_count': matched,
        'trade_precision': precision, 'trade_recall': recall, 'trade_f1': _ratio(2 * matched, true_count + pred_count),
        'exact_interval_match_rate': _ratio(sum(row['joint_correct'] for row in scores), n),
        'trade_window_exact_match_rate': _ratio(sum(row['joint_correct'] for row in positive), len(positive)),
        'no_trade_window_match_rate': _ratio(sum(row['joint_correct'] for row in negative), len(negative)),
        'price_category_recall': _ratio(sum(row['price_matched_count'] for row in scores), true_count),
        'shares_category_recall': _ratio(sum(row['shares_matched_count'] for row in scores), true_count),
        'trade_unit': SEMANTICS,
        'precision_denominator': 'category predictions in valid decoded answers; invalid answers always fail interval success',
        'numeric_criteria': 'same side/outcome; inclusive absolute price error; shares error <= max(absolute,relative*observed_shares)'}
