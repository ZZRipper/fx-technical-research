"""Synthetic two-state entry rejection checks; no research outcomes are read."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from fx_baselines import fixed_execute
from fx_entry_interaction_filters import run_entry_interaction_filter


class EntryInteractionTests(unittest.TestCase):
    @staticmethod
    def sample(target, returns=None):
        n = len(target)
        returns = np.zeros(n) if returns is None else np.asarray(returns)
        prices = 100. * np.cumprod(1. + returns)
        dates = pd.bdate_range('2010-01-04', periods=n)
        features = pd.DataFrame({
            'date': dates, 'realized_vol_20_pct': 10., 'spot_return_20_pct': 0.,
            'distance_ma_60_pct': 0., 'vol_ratio_20_63': .7, 'efficiency_63': .05,
            'K_spread_bp': np.nan, 'K_spread_d20_bp': np.nan, 'K_dxy_d20_pct': np.nan,
            'K_copper_d20_pct': np.nan, 'K_brent_d20_pct': np.nan,
        })
        return prices, dates, fixed_execute(prices, target), features

    @staticmethod
    def spec(first=('vol_ratio_20_63', 0), second=('efficiency_63', 0)):
        return {'id': 'SYNTHETIC_AND', 'conditions': [{'feature': first[0], 'state': first[1]},
                                                    {'feature': second[0], 'state': second[1]}]}

    def execute(self, inputs, spec=None):
        return run_entry_interaction_filter(*inputs, spec or self.spec())

    def test_and_truth_table_does_not_become_or_and_conditions_are_sorted(self):
        for first, second in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(first=first, second=second):
                inputs = self.sample(np.ones(8))
                inputs[3].loc[0, 'vol_ratio_20_63'] = .7 if first else 1.4
                inputs[3].loc[0, 'efficiency_63'] = .05 if second else .5
                run, rows = self.execute(inputs)
                self.assertEqual(rows.iloc[0].rejected, first and second)
                self.assertEqual(rows.iloc[0].condition_1_feature, 'efficiency_63')
                self.assertEqual(rows.iloc[0].condition_2_feature, 'vol_ratio_20_63')
                np.testing.assert_array_equal(run['target'], 0 if first and second else 1)

    def test_both_conditions_use_the_unfiltered_base_return_history(self):
        # First entry has unknown 63-day BASE PnL, so AND retains it. Its 6.3%
        # earned BASE history must remain visible at the next entry signal.
        inputs = self.sample(np.r_[np.ones(70), -np.ones(10)], np.r_[0., np.full(79, .001)])
        spec = self.spec(('efficiency_63', 0), ('base_pnl_63_pct', 3))
        run, rows = self.execute(inputs, spec)
        self.assertEqual(rows.iloc[0].status, 'UNKNOWN')
        self.assertTrue(rows.iloc[0].accepted)
        self.assertAlmostEqual(rows.iloc[1].condition_1_value, 6.3)
        self.assertEqual(rows.iloc[1].condition_1_state, 3)
        self.assertTrue(rows.iloc[1].rejected)
        np.testing.assert_array_equal(run['target'][:70], 1)
        np.testing.assert_array_equal(run['target'][70:], 0)

    def test_nonfinite_unknown_retains_and_no_mid_episode_reentry(self):
        for value in (np.nan, np.inf, -np.inf):
            inputs = self.sample(np.ones(8))
            inputs[3].loc[0, 'efficiency_63'] = value
            run, rows = self.execute(inputs)
            self.assertEqual(rows.iloc[0].status, 'UNKNOWN')
            self.assertEqual(rows.iloc[0].condition_1_state, -1)
            np.testing.assert_array_equal(run['target'], 1)
        inputs = self.sample(np.ones(8))
        inputs[3].loc[1:, 'efficiency_63'] = .9
        run, rows = self.execute(inputs)
        self.assertTrue(rows.iloc[0].rejected)
        np.testing.assert_array_equal(run['target'], 0)

    def test_next_close_reversal_natural_exit_and_pending_entry(self):
        target = np.r_[np.ones(5), -np.ones(5), 0, 0, 1]
        inputs = self.sample(target, np.r_[0., np.full(12, .01)])
        inputs[3].loc[5, 'efficiency_63'] = .9
        inputs[3].loc[12, 'efficiency_63'] = .9
        run, rows = self.execute(inputs)
        self.assertTrue(rows.iloc[0].rejected)
        self.assertEqual(rows.iloc[1].actual_entry_idx, 6)
        self.assertEqual(run['gross_return'][6], 0.)
        self.assertAlmostEqual(run['gross_return'][7], -.01)
        self.assertEqual(run['executed'][11], 0)
        self.assertAlmostEqual(run['gross_return'][11], -.01)
        self.assertEqual(run['target'][12], 1)
        self.assertTrue(rows.iloc[2].entry_pending)
        self.assertTrue(pd.isna(rows.iloc[2].actual_entry_idx))
        rejected_last = list(inputs)
        rejected_last[3] = inputs[3].copy()
        rejected_last[3].loc[12, 'efficiency_63'] = .05
        other, other_rows = self.execute(rejected_last)
        self.assertTrue(other_rows.iloc[2].rejected)
        self.assertEqual(other['target'][12], 0)
        for field in ('executed', 'held', 'gross_return', 'equity'):
            np.testing.assert_array_equal(run[field], other[field])

    def test_registered_mechanism_pairs_have_prefix_causality_and_preserve_inputs(self):
        rng = np.random.default_rng(537)
        target = np.r_[np.ones(70), -np.ones(65), np.zeros(5), np.ones(70), -np.ones(60)]
        inputs = self.sample(target, np.r_[0., rng.normal(0., .01, 269)])
        features = inputs[3]
        features['efficiency_63'] = rng.uniform(0, .6, 270)
        features['vol_ratio_20_63'] = rng.uniform(.5, 1.5, 270)
        features['spot_return_20_pct'] = rng.normal(0., 4., 270)
        features['K_dxy_d20_pct'] = rng.normal(0., 2., 270)
        features['K_spread_bp'] = rng.normal(0., 130., 270)
        features['K_dxy_obs_date'] = inputs[1] - pd.Timedelta(days=1)
        features['K_rate_obs_date'] = inputs[1] - pd.Timedelta(days=1)
        before = features.copy(deep=True)
        base_before = inputs[2]['target'].copy()
        pairs = [('efficiency_63', 'vol_ratio_20_63'), ('efficiency_63', 'direction_return_20_vol'),
                 ('direction_dxy_d20_pct', 'direction_spread_bp')]
        for first, second in pairs:
            for states in ((0, 0), (1, 3), (3, 1)):
                spec = self.spec((first, states[0]), (second, states[1]))
                original_spec = deepcopy(spec)
                full, _ = self.execute(inputs, spec)
                for stop in (1, 2, 63, 70, 71, 136, 141, 211, 269, 270):
                    with self.subTest(first=first, states=states, stop=stop):
                        prices, dates, base, frame = inputs
                        partial = (prices[:stop], dates[:stop], fixed_execute(prices[:stop], target[:stop]), frame.iloc[:stop].copy())
                        result, _ = self.execute(partial, spec)
                        for field in ('target', 'executed', 'held', 'gross_return', 'equity'):
                            np.testing.assert_array_equal(result[field], full[field][:stop])
                self.assertEqual(spec, original_spec)
        pd.testing.assert_frame_equal(features, before)
        np.testing.assert_array_equal(inputs[2]['target'], base_before)

    def test_empty_account_spec_validation_and_cost_guard(self):
        inputs = self.sample(np.zeros(8))
        run, rows = self.execute(inputs)
        self.assertTrue(rows.empty)
        np.testing.assert_array_equal(run['target'], 0)
        good = self.spec()
        bad = [dict(good, threshold=0.), {'id': 'X', 'conditions': good['conditions'][:1]},
               {'id': 'X', 'conditions': good['conditions'] * 2},
               self.spec(('efficiency_63', 0), ('efficiency_63', 1)),
               self.spec(('not_registered', 0)), self.spec(([], 0)), self.spec(('efficiency_63', -1)),
               self.spec(('efficiency_63', True)), self.spec(('efficiency_63', 1.0)),
               {'id': 'X', 'conditions': [dict(good['conditions'][0], bins=[0,1,2]), good['conditions'][1]]}]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.execute(inputs, spec)
        costly = list(inputs)
        costly[2] = fixed_execute(inputs[0], inputs[2]['target'], cost_bp=1.)
        with self.assertRaises(ValueError):
            self.execute(costly)


if __name__ == '__main__':
    unittest.main()
