"""Synthetic ALL-checkpoint tests; never read an IS or OOS result artifact."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fx_baselines import fixed_execute
from fx_filter_evaluation import simple_ledger
from fx_holding_filters import run_holding_filter
from fx_timecap_filters import CHECKPOINT_AGES, run_timecap_filter
from build_holding_observations import build_checkpoints


class TimecapFilterTests(unittest.TestCase):
    @staticmethod
    def sample(returns, targets):
        prices = 100. * np.cumprod(1. + np.asarray(returns, float))
        dates = pd.bdate_range('2010-01-04', periods=len(prices))
        features = pd.DataFrame({
            'date': dates, 'realized_vol_20_pct': 10., 'vol_ratio_20_63': 1.,
            'spot_return_20_pct': pd.Series(prices).pct_change(20) * 100.,
            'spot_return_63_pct': pd.Series(prices).pct_change(63) * 100.,
        })
        return prices, dates, fixed_execute(prices, targets), features

    @staticmethod
    def execute(inputs, age=5):
        return run_timecap_filter(*inputs, {'id': 'TIMECAP_SYNTHETIC', 'checkpoint_age': age})

    def test_all_exits_negative_zero_positive_pnl_and_earns_execution_day(self):
        for step in (-.01, 0., .01):
            with self.subTest(step=step):
                inputs = self.sample(np.r_[0., 0., np.full(10, step)], np.ones(12))
                run, rows = self.execute(inputs)
                row = rows.iloc[0]
                self.assertEqual((row.entry_idx, row.checkpoint_idx, row.risk_signal_idx, row.actual_exit_idx), (1, 6, 6, 7))
                self.assertEqual(row.condition, 'ALL')
                self.assertEqual(row.condition_state, 'TRUE')
                self.assertTrue(row.evaluated and row.early_exit)
                self.assertAlmostEqual(row.known_pnl_pct, step * 500.)
                self.assertAlmostEqual(run['gross_return'][7], step)
                np.testing.assert_array_equal(run['gross_return'][8:], 0.)
                np.testing.assert_array_equal(run['target'][6:], 0)

    def test_known_technical_exit_and_reversal_preempt_all(self):
        for position in (0, -1):
            with self.subTest(position=position):
                inputs = self.sample(np.r_[0., np.full(9, .01)], np.r_[np.ones(6), np.full(4, position)])
                run, rows = self.execute(inputs)
                self.assertFalse(rows.iloc[0].evaluated)
                self.assertFalse(rows.iloc[0].early_exit)
                self.assertTrue(pd.isna(rows.iloc[0].risk_signal_idx))
                np.testing.assert_array_equal(run['target'], inputs[2]['target'])

    def test_lock_resets_only_at_next_original_episode(self):
        inputs = self.sample(np.zeros(20), np.r_[np.ones(10), -np.ones(10)])
        run, rows = self.execute(inputs)
        np.testing.assert_array_equal(run['target'][6:10], 0)
        np.testing.assert_array_equal(run['target'][10:16], -1)
        np.testing.assert_array_equal(run['target'][16:], 0)
        self.assertEqual(list(rows.risk_signal_idx), [6, 16])
        self.assertEqual(list(rows.actual_exit_idx), [7, 17])

    def test_last_signal_pending_and_last_new_entry_do_not_fabricate_fills(self):
        inputs = self.sample(np.zeros(7), np.ones(7))
        run, rows = self.execute(inputs)
        row = rows.iloc[0]
        self.assertTrue(row.early_exit and row.risk_pending and row.exit_pending)
        self.assertTrue(pd.isna(row.actual_exit_idx))
        self.assertEqual(run['target'][-1], 0)
        for field in ('executed', 'held', 'gross_return', 'equity'):
            np.testing.assert_array_equal(run[field], inputs[2][field])
        last = self.sample(np.zeros(7), np.r_[np.ones(6), -1])
        run, rows = self.execute(last)
        self.assertFalse(rows.iloc[0].early_exit)
        self.assertTrue(rows.iloc[1].entry_pending)
        self.assertEqual(run['target'][-1], -1)
        np.testing.assert_array_equal(run['executed'], last[2]['executed'])

    def test_multiple_age_prefixes_and_observation_all_actionable_alignment(self):
        rng = np.random.default_rng(214)
        returns = np.r_[0., rng.normal(0., .01, 244)]
        targets = np.r_[np.ones(80), np.zeros(2), -np.ones(74), np.ones(1), np.zeros(3), np.ones(85)]
        inputs = self.sample(returns, targets)
        prices, dates, base, features = inputs
        before = features.copy(deep=True)
        base_before = base['target'].copy()
        observations, _ = build_checkpoints(base, dates, features, simple_ledger(base, dates), 'JPY', 'BB_MR_20')
        for age in CHECKPOINT_AGES:
            full, rows = self.execute(inputs, age)
            ref = observations.loc[observations.checkpoint_age.eq(age) &
                                   (observations.actionable_checkpoint | observations.risk_exit_pending)]
            self.assertEqual(set(rows.loc[rows.early_exit, 'entry_idx']), set(ref.entry_idx))
            for row in rows.loc[rows.evaluated].itertuples(index=False):
                observed = ref.loc[ref.entry_idx.eq(row.entry_idx)].iloc[0]
                self.assertEqual(row.risk_signal_idx, observed.checkpoint_idx)
                self.assertAlmostEqual(row.known_pnl_pct, observed.known_cumulative_simple_pnl_pct)
            stops = sorted(set([1, 2, age + 1, age + 2, age + 3, 80, 81, 83, 84, 156, 157, 161, 162, 244, 245]))
            for stop in stops:
                with self.subTest(age=age, stop=stop):
                    partial = (prices[:stop], dates[:stop], fixed_execute(prices[:stop], targets[:stop]), features.iloc[:stop].copy())
                    run, _ = self.execute(partial, age)
                    for field in ('target', 'executed', 'held', 'gross_return', 'equity'):
                        np.testing.assert_array_equal(run[field], full[field][:stop])
        pd.testing.assert_frame_equal(features, before)
        np.testing.assert_array_equal(base['target'], base_before)

    def test_schema_empty_account_fixed_spec_and_zero_cost(self):
        inputs = self.sample(np.zeros(10), np.zeros(10))
        _, rows = self.execute(inputs)
        _, holding_rows = run_holding_filter(*inputs, {'id': 'H', 'checkpoint_age': 5, 'condition': 'pnl_positive'})
        self.assertTrue(rows.empty)
        self.assertEqual(list(rows), list(holding_rows))
        invalid = [
            {'id': 'X', 'checkpoint_age': 5, 'condition': 'ALL'},
            {'id': 'X', 'checkpoint_age': 6}, {'id': 'X', 'checkpoint_age': True},
            {'id': '', 'checkpoint_age': 5}, {'checkpoint_age': 5},
        ]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                run_timecap_filter(*inputs, spec)
        costly = list(inputs)
        costly[2] = fixed_execute(inputs[0], inputs[2]['target'], cost_bp=1.)
        with self.assertRaises(ValueError):
            self.execute(costly)


if __name__ == '__main__':
    unittest.main()
