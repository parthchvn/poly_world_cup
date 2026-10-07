import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import interval_trade_tolerances as scoring


def trade(price='0.40', shares='100', side='BUY', outcome='Yes'):
    return dict(price=price, shares=shares, side=side, outcome=outcome)


def answer(*trades):
    return {'action': 'TRADE', 'trades': list(trades)} if trades else {'action': 'NO_TRADE'}


class TradeToleranceTests(unittest.TestCase):
    tolerances = {'price_delta': '0.02', 'shares_relative_delta': '0.20', 'shares_absolute_delta': '0'}

    def score(self, truth, predicted, tolerance=None):
        return scoring.score_trade_details(truth, predicted, tolerance or self.tolerances)

    def test_inclusive_decimal_boundaries_and_both_numeric_requirements(self):
        truth = answer(trade())
        for price, shares, expected in [('0.42', '120', True), ('0.38', '80', True),
                                        ('0.420000001', '100', False), ('0.4', '120.000001', False)]:
            self.assertEqual(self.score(truth, answer(trade(price, shares)))['joint_correct'], expected)

    def test_share_relative_error_uses_observed_quantity(self):
        # 100 prediction versus 80 observed is 25% error, not 20%.
        self.assertFalse(self.score(answer(trade(shares='80')), answer(trade(shares='100')))['joint_correct'])
        self.assertTrue(self.score(answer(trade(shares='100')), answer(trade(shares='80')))['joint_correct'])

    def test_absolute_share_floor_applies_for_small_trades(self):
        tolerance = {**self.tolerances, 'shares_absolute_delta': '1'}
        self.assertTrue(self.score(answer(trade(shares='1')), answer(trade(shares='2')), tolerance)['joint_correct'])
        self.assertFalse(self.score(answer(trade(shares='1')), answer(trade(shares='2.01')), tolerance)['joint_correct'])

    def test_fill_splitting_and_order_do_not_change_window_prediction(self):
        truth = answer(trade('0.3', '50'), trade('0.5', '50'), trade('0.7', '30', 'SELL', 'No'))
        predicted = answer(trade('0.4', '100'), trade('0.7', '30', 'SELL', 'No'))
        for perm in itertools.permutations(truth['trades']):
            result = self.score(answer(*perm), predicted)
            self.assertTrue(result['joint_correct'])
            self.assertEqual(result['true_count'], 2)

    def test_no_buy_sell_netting_or_outcome_substitution(self):
        truth = answer(trade(), trade(side='SELL'))
        self.assertEqual(self.score(truth, answer())['matched_count'], 0)
        self.assertFalse(self.score(answer(trade()), answer(trade(outcome='No')))['joint_correct'])

    def test_duplicate_predictions_increase_quantity_instead_of_match_count(self):
        result = self.score(answer(trade()), answer(trade(), trade()))
        self.assertEqual(result['predicted_count'], 1)
        self.assertEqual(result['matched_count'], 0)
        self.assertFalse(result['joint_correct'])

    def test_numeric_json_and_decimal_strings_are_equivalent(self):
        result = self.score(answer(trade()), '{"action":"TRADE","trades":[{"side":"buy","outcome":"YES","price":0.4,"shares":100}]}')
        self.assertTrue(result['joint_correct'])

    def test_invalid_outputs_never_turn_into_correct_no_trade(self):
        for prediction in ['garbage', '{"action":"NO_TRADE","action":"TRADE"}',
                           answer(trade(shares='NaN')), answer(trade(shares='1e999999999')),
                           answer(trade(shares=True)), answer(trade(price='Infinity')),
                           {'action': 'TRADE', 'trades': []}, {'action': 'NO_TRADE', 'trades': []}]:
            result = self.score(answer(), prediction)
            self.assertFalse(result['valid_prediction'])
            self.assertFalse(result['joint_correct'])
        with self.assertRaises(ValueError):
            self.score({'action': 'TRADE', 'trades': []}, answer())
        self.assertFalse(self.score(answer(), answer(*[trade(shares='1e100')] * 10))['valid_prediction'])

    def test_summary_reports_positive_window_success_separately(self):
        scores = [self.score(answer(), answer())] * 9 + [self.score(answer(trade()), answer())]
        summary = scoring.summarize_trade_details(scores)
        self.assertEqual(summary['exact_interval_match_rate'], .9)
        self.assertEqual(summary['trade_window_exact_match_rate'], 0)
        self.assertEqual(summary['trade_recall'], 0)
        self.assertEqual(summary['trade_f1'], 0)
        self.assertEqual(summary['trade_action_f1'], 0)
        invalid = scoring.summarize_trade_details([self.score(answer(), 'invalid')])
        self.assertEqual(invalid['exact_interval_match_rate'], 0)
        self.assertEqual(invalid['json_valid_rate'], 0)

    def test_tolerance_ranges_and_missing_configuration_fail(self):
        for tolerance in [{}, {**self.tolerances, 'price_delta': '-.1'},
                          {**self.tolerances, 'shares_relative_delta': '1'},
                          {**self.tolerances, 'shares_absolute_delta': 'Infinity'}]:
            with self.assertRaises(ValueError):
                scoring.validate_tolerances(tolerance)


if __name__ == '__main__':
    unittest.main()
