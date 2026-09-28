import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import derive_actor_metrics as metrics

class FeatureAblationReviewTests(unittest.TestCase):
    def test_execution_only_is_invariant_to_financial_values_and_counts(self):
        core = {'values': dict.fromkeys(metrics.ACTOR_METRIC_NAMES),
                'sample_counts': {'captured_executions': 7, 'completed_positions': 5,
                                  'profitable_completed_positions': 4, 'losing_completed_positions': 1,
                                  'breakeven_completed_positions': 0, 'eligible_return_periods': 40},
                'return_period_seconds': '86400'}
        core['values'].update(average_execution_notional='12.345678912345', win_rate='0.8', sharpe_ratio='1.4')
        config = {'selected_features': ['average_execution_notional'],
                  'history_scope': 'actor_and_binary_market'}
        original = copy.deepcopy(core)
        expected = metrics.model_metric_fields(core, config)
        self.assertEqual(expected['sample_counts'], {'captured_executions': 7})
        self.assertEqual(set(expected['values']), {'average_execution_notional'})
        self.assertNotIn('return_period_seconds', expected)
        for key in set(core['values']) - {'average_execution_notional'}:
            core['values'][key] = '987654'
        for key in set(core['sample_counts']) - {'captured_executions'}:
            core['sample_counts'][key] = 9999
        core['return_period_seconds'] = '3600'
        self.assertEqual(metrics.model_metric_fields(core, config), expected)
        self.assertIsNone(original['values']['maximum_drawdown'])

    def test_risk_unknown_omitted_but_support_is_visible_without_outcome_counts(self):
        core = {'values': dict.fromkeys(metrics.ACTOR_METRIC_NAMES),
                'sample_counts': {'captured_executions': 7, 'completed_positions': 5,
                                  'profitable_completed_positions': 4, 'losing_completed_positions': 1,
                                  'breakeven_completed_positions': 0, 'eligible_return_periods': 2},
                'return_period_seconds': '86400'}
        original = copy.deepcopy(core)
        result = metrics.model_metric_fields(core, {
            'selected_features': ['sharpe_ratio'], 'history_scope': 'actor_across_all_markets'})
        self.assertEqual(result, {'values': {}, 'sample_counts': {'eligible_return_periods': 2},
                                 'scope': 'global_wallet', 'return_period_seconds': '86400'})
        self.assertEqual(core, original)

    def test_completed_metric_keeps_only_position_support_not_win_loss_counts(self):
        core = {'values': dict.fromkeys(metrics.ACTOR_METRIC_NAMES),
                'sample_counts': {'captured_executions': 7, 'completed_positions': 5,
                                  'profitable_completed_positions': 4, 'losing_completed_positions': 1,
                                  'breakeven_completed_positions': 0, 'eligible_return_periods': 40},
                'return_period_seconds': '86400'}
        core['values']['average_holding_seconds'] = '300'
        result = metrics.model_metric_fields(core, {'selected_features': ['average_holding_seconds']})
        self.assertEqual(result, {'values': {'average_holding_seconds': '3E+2'},
                                 'sample_counts': {'completed_positions': 5}})

if __name__ == '__main__':
    unittest.main()
