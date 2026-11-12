"""Synthetic direction rejection and execution-clock checks only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from fx_baselines import fixed_execute
from fx_direction_filter import run_direction_filter


class DirectionFilterTests(unittest.TestCase):
    @staticmethod
    def sample(target, returns=None, capital=123456.):
        n = len(target)
        returns = np.zeros(n) if returns is None else np.asarray(returns)
        prices = 100. * np.cumprod(1. + returns)
        dates = pd.bdate_range('2010-01-04', periods=n)
        return prices, dates, fixed_execute(prices, target, capital=capital), pd.DataFrame({'date': dates, 'irrelevant_feature': np.nan})

    @staticmethod
    def execute(inputs, rejected=1):
        return run_direction_filter(*inputs, {'id': 'DIRECTION_SYNTHETIC', 'reject_position': rejected})

    def test_next_close_reversal_and_natural_exit_keep_execution_day_return(self):
        inputs = self.sample(np.r_[np.ones(4), -np.ones(4), 0, 0, 1, 1], np.r_[0., np.full(11, .01)])
        run, rows = self.execute(inputs, 1)
        self.assertEqual(list(rows.position), [1, -1, 1])
        self.assertEqual(list(rows.accepted), [False, True, False])
        self.assertTrue(pd.isna(rows.iloc[0].actual_entry_idx))
        self.assertEqual(rows.iloc[1].entry_idx, 5)
        self.assertEqual(rows.iloc[1].actual_entry_idx, 5)
        self.assertEqual(rows.iloc[1].original_end_signal_idx, 8)
        self.assertEqual(rows.iloc[1].actual_exit_idx, 9)
        self.assertEqual(run['gross_return'][5], 0.)
        self.assertAlmostEqual(run['gross_return'][6], -.01)
        self.assertAlmostEqual(run['gross_return'][9], -.01)
        self.assertEqual(run['executed'][9], 0)
        self.assertEqual(run['gross_return'][10], 0.)
        other, _ = self.execute(inputs, -1)
        self.assertAlmostEqual(other['gross_return'][5], .01)
        self.assertEqual(other['executed'][5], 0)

    def test_rejected_segment_never_reenters_and_features_have_no_effect(self):
        inputs = self.sample(np.r_[np.ones(8), 0, -np.ones(7)])
        run, rows = self.execute(inputs, 1)
        np.testing.assert_array_equal(run['target'][:9], 0)
        np.testing.assert_array_equal(run['target'][9:], -1)
        modified = list(inputs)
        modified[3] = inputs[3].assign(irrelevant_feature=np.arange(16) * 100.)
        other, _ = self.execute(modified, 1)
        for field in ('target', 'executed', 'held', 'gross_return', 'equity'):
            np.testing.assert_array_equal(run[field], other[field])
        self.assertEqual(rows.iloc[0].status, 'REJECT')

    def test_open_tail_pending_entry_and_pending_natural_exit_are_not_fills(self):
        open_inputs = self.sample([-1, -1, -1, -1])
        run, rows = self.execute(open_inputs, 1)
        self.assertTrue(pd.isna(rows.iloc[0].actual_exit_idx))
        self.assertFalse(rows.iloc[0].exit_pending)
        self.assertEqual(run['executed'][-1], -1)
        for rejected in (-1, 1):
            with self.subTest(rejected=rejected):
                pending = self.sample([0, 0, 0, 1])
                filtered, rows = self.execute(pending, rejected)
                self.assertTrue(rows.iloc[0].entry_pending)
                self.assertTrue(pd.isna(rows.iloc[0].entry_idx))
                self.assertTrue(pd.isna(rows.iloc[0].actual_entry_idx))
                self.assertEqual(filtered['target'][-1], 0 if rejected == 1 else 1)
                np.testing.assert_array_equal(filtered['executed'], 0)
        natural = self.sample([-1, -1, -1, 0])
        filtered, rows = self.execute(natural, 1)
        self.assertTrue(rows.iloc[0].original_exit_pending and rows.iloc[0].exit_pending)
        self.assertTrue(pd.isna(rows.iloc[0].actual_exit_idx))
        self.assertEqual(filtered['target'][-1], 0)
        self.assertEqual(filtered['executed'][-1], -1)

    def test_random_prefixes_both_directions_and_input_preservation(self):
        rng = np.random.default_rng(166)
        target = rng.choice([-1, 0, 1], 90)
        target[20:35], target[50:70], target[-1] = 1, -1, 1
        inputs = self.sample(target, np.r_[0., rng.normal(0., .01, 89)])
        original = deepcopy(inputs)
        for rejected in (-1, 1):
            full, _ = self.execute(inputs, rejected)
            for stop in range(1, 91):
                with self.subTest(rejected=rejected, stop=stop):
                    prices, dates, base, features = inputs
                    partial = (prices[:stop], dates[:stop], fixed_execute(prices[:stop], target[:stop], capital=123456.), features.iloc[:stop].copy())
                    run, rows = self.execute(partial, rejected)
                    for field in ('target', 'executed', 'held', 'gross_return', 'equity'):
                        np.testing.assert_array_equal(run[field], full[field][:stop])
                    self.assertTrue(rows.accepted.eq(rows.position.ne(rejected)).all())
                    self.assertEqual(run['initial_capital_usd'], 123456.)
        np.testing.assert_array_equal(inputs[0], original[0])
        pd.testing.assert_frame_equal(inputs[3], original[3])
        for field, value in inputs[2].items():
            np.testing.assert_array_equal(value, original[2][field])

    def test_strict_spec_zero_cost_capital_and_base_clock_validation(self):
        inputs = self.sample(np.ones(8))
        invalid = [{'id': 'X', 'reject_position': value} for value in (True, False, 0, 2, 1., '1', None)]
        invalid += [{'id': '', 'reject_position': 1}, {'id': 'X'}, {'id': 'X', 'reject_position': 1, 'threshold': 0}]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                run_direction_filter(*inputs, spec)
        for field in ('executed', 'held', 'gross_return', 'net_return', 'spot_return', 'equity'):
            corrupted = deepcopy(inputs)
            corrupted[2][field][-1] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.execute(corrupted)
        for capital in (0., -1., np.inf, np.nan):
            corrupted = deepcopy(inputs)
            corrupted[2]['initial_capital_usd'] = capital
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                self.execute(corrupted)
        wrong_capital = deepcopy(inputs)
        wrong_capital[2]['initial_capital_usd'] = 100000.
        with self.assertRaises(ValueError):
            self.execute(wrong_capital)
        costly = list(inputs)
        costly[2] = fixed_execute(inputs[0], inputs[2]['target'], cost_bp=1.)
        with self.assertRaises(ValueError):
            self.execute(costly)

    def test_price_calendar_empty_account_and_optional_feature_contract(self):
        inputs = self.sample(np.zeros(6))
        run, rows = self.execute(inputs)
        self.assertTrue(rows.empty)
        np.testing.assert_array_equal(run['target'], 0)
        no_features = list(inputs)
        no_features[3] = None
        other, _ = self.execute(no_features)
        np.testing.assert_array_equal(other['target'], run['target'])
        for value in (0., -1., np.nan, np.inf):
            corrupt = deepcopy(inputs)
            corrupt[0][-1] = value
            with self.subTest(price=value), self.assertRaises(ValueError):
                self.execute(corrupt)
        mismatch = list(inputs)
        mismatch[3] = inputs[3].assign(date=inputs[1] + pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            self.execute(mismatch)
        for dates in (inputs[1][::-1], pd.DatetimeIndex([inputs[1][0]] * 6), inputs[1].tz_localize('UTC')):
            corrupt = list(inputs)
            corrupt[1] = dates
            with self.subTest(dates=str(dates)), self.assertRaises(ValueError):
                self.execute(corrupt)


if __name__ == '__main__':
    unittest.main()
