"""Synthetic event-clock and causal-prefix checks; no research data is read."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from fx_baselines import fixed_execute
from fx_holding_filters import CHECKPOINT_AGES, CONDITIONS, run_holding_filter


class HoldingFilterTests(unittest.TestCase):
    @staticmethod
    def sample(returns, targets, direction20=None):
        prices = 100. * np.cumprod(1. + np.asarray(returns, float))
        dates = pd.bdate_range('2010-01-04', periods=len(prices))
        feature = (pd.Series(prices).pct_change(20) * 100. if direction20 is None else direction20)
        features = pd.DataFrame({'date': dates, 'spot_return_20_pct': feature})
        return prices, dates, fixed_execute(prices, np.asarray(targets)), features

    def run_filter(self, inputs, condition='pnl_negative', age=5):
        return run_holding_filter(*inputs, {'id': 'SYNTHETIC', 'checkpoint_age': age, 'condition': condition})

    def test_exit_signal_earns_next_close_and_locks_until_new_episode(self):
        returns = np.r_[0., 0., np.full(12, -.01)]
        targets = np.r_[np.ones(10), -np.ones(4)]
        inputs = self.sample(returns, targets)
        run, decisions = self.run_filter(inputs)
        first, second = decisions.iloc[0], decisions.iloc[1]
        self.assertEqual((first.entry_idx, first.checkpoint_idx, first.risk_signal_idx, first.actual_exit_idx), (1, 6, 6, 7))
        self.assertAlmostEqual(first.known_pnl_pct, -5.)
        self.assertEqual(first.original_exit_idx, 11)
        self.assertTrue(first.early_exit)
        np.testing.assert_array_equal(run['target'][6:10], 0)
        self.assertAlmostEqual(run['gross_return'][7], -.01)
        self.assertEqual(run['gross_return'][8], 0.)
        self.assertEqual(run['target'][10], -1)
        self.assertEqual(second.entry_idx, 11)
        self.assertFalse(second.early_exit)

    def test_known_technical_exit_or_reversal_preempts_checkpoint(self):
        for next_target in (0, -1):
            with self.subTest(next_target=next_target):
                inputs = self.sample(np.r_[0., 0., np.full(8, -.01)], np.r_[np.ones(6), np.full(4, next_target)])
                run, decisions = self.run_filter(inputs)
                first = decisions.iloc[0]
                self.assertEqual(first.checkpoint_idx, 6)
                self.assertFalse(first.evaluated)
                self.assertFalse(first.early_exit)
                self.assertTrue(pd.isna(first.risk_signal_idx))
                self.assertEqual(first.actual_exit_idx, 7)
                np.testing.assert_array_equal(run['target'], inputs[2]['target'])

    def test_false_or_unknown_checkpoint_is_never_retried(self):
        inputs = self.sample(np.r_[0., 0., np.full(5, .01), np.full(7, -.04)], np.ones(14))
        run, decisions = self.run_filter(inputs)
        self.assertEqual(decisions.iloc[0].condition_state, 'FALSE')
        self.assertFalse(decisions.iloc[0].early_exit)
        np.testing.assert_array_equal(run['target'], inputs[2]['target'])
        opposite = np.full(14, -10.)
        opposite[6] = np.nan
        unknown = self.sample(np.r_[0., 0., np.full(12, -.01)], np.ones(14), opposite)
        run, decisions = self.run_filter(unknown, ['pnl_negative', 'opposite20'])
        self.assertEqual(decisions.iloc[0].condition_state, 'UNKNOWN')
        np.testing.assert_array_equal(run['target'], unknown[2]['target'])

    def test_giveback_uses_signed_simple_path_and_new_episode_resets_peak(self):
        # For a short, signed simple returns differ from a short price endpoint.
        returns = np.array([0., 0., -.04, -.01, .03, 0., 0., .02, 0., 0., 0., 0., 0., 0., 0.])
        targets = np.r_[-np.ones(8), np.ones(7)]
        inputs = self.sample(returns, targets)
        run, decisions = self.run_filter(inputs, 'giveback_half')
        first, second = decisions.iloc[0], decisions.iloc[1]
        self.assertAlmostEqual(first.known_pnl_pct, 2.)
        self.assertAlmostEqual(first.running_peak_pct, 5.)
        self.assertAlmostEqual(first.giveback_fraction, .6)
        self.assertTrue(first.early_exit)
        endpoint = -(inputs[0][6] / inputs[0][1] - 1.) * 100.
        self.assertNotAlmostEqual(first.known_pnl_pct, endpoint, places=5)
        self.assertAlmostEqual(run['gross_return'][7], -.02)
        self.assertEqual(second.checkpoint_idx, 14)
        self.assertAlmostEqual(second.running_peak_pct, 0.)
        self.assertFalse(second.early_exit)

    def test_last_checkpoint_is_pending_without_fill_or_liquidation(self):
        inputs = self.sample(np.r_[0., 0., np.full(5, -.01)], np.ones(7))
        run, decisions = self.run_filter(inputs)
        row = decisions.iloc[0]
        self.assertTrue(row.early_exit and row.risk_pending and row.exit_pending)
        self.assertTrue(pd.isna(row.actual_exit_idx))
        self.assertEqual(run['target'][-1], 0)
        for field in ('executed', 'held', 'gross_return', 'equity'):
            np.testing.assert_array_equal(run[field], inputs[2][field])
        one = self.sample([0.], [1])
        pending, rows = self.run_filter(one)
        self.assertTrue(rows.iloc[0].entry_pending)
        self.assertFalse(rows.iloc[0].early_exit)
        self.assertTrue(pd.isna(rows.iloc[0].entry_idx))
        np.testing.assert_array_equal(pending['target'], [1])
        np.testing.assert_array_equal(pending['executed'], [0])

    def test_all_ages_conditions_and_random_prefixes_are_causal(self):
        rng = np.random.default_rng(762)
        returns = np.r_[0., rng.normal(.0002, .012, 239)]
        targets = np.r_[np.ones(80), np.zeros(2), -np.ones(74), np.ones(1), np.zeros(3), -np.ones(80)]
        inputs = self.sample(returns, targets)
        features_before = inputs[3].copy(deep=True)
        target_before = inputs[2]['target'].copy()
        conditions = [*CONDITIONS, ['pnl_negative', 'opposite20'], ['pnl_positive', 'giveback_half']]
        random_stops = rng.choice(np.arange(2, 240), 8, replace=False).tolist()
        for age in CHECKPOINT_AGES:
            stops = sorted(set([1, age + 1, age + 2, age + 3, 80, 81, 83, 84, 156, 157, 239, 240, *random_stops]))
            for condition in conditions:
                full, full_rows = self.run_filter(inputs, condition, age)
                for stop in stops:
                    with self.subTest(age=age, condition=condition, stop=stop):
                        prices, dates, base, features = inputs
                        short = (prices[:stop], dates[:stop], fixed_execute(prices[:stop], base['target'][:stop]), features.iloc[:stop].copy())
                        prefix, prefix_rows = self.run_filter(short, condition, age)
                        for field in ('target', 'executed', 'held', 'gross_return', 'equity'):
                            np.testing.assert_array_equal(prefix[field], full[field][:stop])
                        checked = prefix_rows.loc[prefix_rows.evaluated]
                        for _, row in checked.iterrows():
                            reference = full_rows.loc[full_rows.signal_idx.eq(row.signal_idx)].iloc[0]
                            self.assertEqual(row.condition_state, reference.condition_state)
                            self.assertEqual(row.early_exit, reference.early_exit)
                            self.assertAlmostEqual(row.known_pnl_pct, reference.known_pnl_pct)
        pd.testing.assert_frame_equal(inputs[3], features_before)
        np.testing.assert_array_equal(inputs[2]['target'], target_before)

    def test_empty_account_fixed_spec_and_zero_cost_contract(self):
        inputs = self.sample(np.zeros(8), np.zeros(8))
        run, rows = self.run_filter(inputs)
        self.assertTrue(rows.empty)
        np.testing.assert_array_equal(run['target'], 0)
        base = {'id': 'CHECK', 'checkpoint_age': 5, 'condition': 'pnl_negative'}
        invalid = [dict(base, checkpoint_age=True), dict(base, checkpoint_age=6),
                   dict(base, threshold=0.), dict(base, condition='pnl_negative&opposite20'),
                   dict(base, condition=['pnl_negative', 'pnl_positive']),
                   dict(base, condition=['opposite20', 'opposite20'])]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                run_holding_filter(*inputs, spec)
        costly = list(inputs)
        costly[2] = fixed_execute(inputs[0], inputs[2]['target'], cost_bp=1.)
        with self.assertRaises(ValueError):
            run_holding_filter(*costly, base)
        dates_bad = list(inputs)
        dates_bad[3] = inputs[3].assign(date=inputs[1] + pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            run_holding_filter(*dates_bad, base)


if __name__ == '__main__':
    unittest.main()
