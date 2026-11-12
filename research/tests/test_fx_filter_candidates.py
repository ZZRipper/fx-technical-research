"""Hand-calculated episode-filter tests using synthetic data only."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fx_baselines import fixed_execute
from fx_filter_candidates import run_candidate, specs


class FilterCandidateTests(unittest.TestCase):
    def run_filter(self, prices, targets, identifier, features=None, strategy="RSI_MR_14"):
        dates = pd.date_range("2010-01-01", periods=len(prices))
        features = pd.DataFrame(index=dates) if features is None else features.copy()
        if not isinstance(features.index, pd.DatetimeIndex):
            features.index = dates
        base = fixed_execute(prices, targets)
        spec = next(item for item in specs() if item["id"] == identifier)
        run, decisions = run_candidate(prices, dates, base, features, spec, pair="JPY", strategy=strategy)
        return base, run, decisions

    def test_baseline_is_exact_and_registry_is_not_mutable(self):
        original = specs()
        changed = specs()
        changed[1]["parameters"]["entry"]["maximum_ratio"] = 99
        self.assertEqual(specs(), original)
        base, run, _ = self.run_filter([100, 101, 99, 103, 104, 102], [1, 1, 0, -1, -1, 1], "BASE")
        for field in ("target", "executed", "held", "gross_return", "equity"):
            np.testing.assert_array_equal(run[field], base[field])

    def test_entry_reads_signal_date_and_rejection_locks_whole_episode(self):
        features = pd.DataFrame({"vol_ratio_20_63": [1.5, .5, .5, .5, .5, .5, .5, .5]})
        _, run, decisions = self.run_filter(np.full(8, 100.), [1, 1, 1, 0, -1, -1, 0, 0], "VOL_NOT_EXPANDING", features)
        self.assertEqual(decisions.state.tolist(), ["REJECT", "KEEP"])
        np.testing.assert_array_equal(run["executed"], [0, 0, 0, 0, 0, -1, -1, 0])
        self.assertTrue(pd.isna(decisions.actual_entry_idx.iloc[0]))

    def test_macro_unknown_is_retained_and_future_macro_cannot_change_entry(self):
        missing = pd.DataFrame({"K_vix": [np.nan, 80., 80., 80., 80.], "K_vix_d5": [np.nan, 30., 30., 30., 30.]})
        base, run, decisions = self.run_filter([100, 100, 102, 101, 100], [1, 1, 1, 0, 0], "VIX_RISING", missing)
        np.testing.assert_array_equal(run["executed"], base["executed"])
        self.assertEqual(decisions.state.iloc[0], "UNKNOWN")
        missing.loc[0, ["K_vix", "K_vix_d5"]] = [20., 3.]
        _, rejected, decisions = self.run_filter([100, 100, 102, 101, 100], [1, 1, 1, 0, 0], "VIX_RISING", missing)
        self.assertFalse(rejected["executed"].any())
        self.assertEqual(decisions.state.iloc[0], "REJECT")

    def test_time_exit_earns_next_close_return_and_does_not_reenter(self):
        prices = np.array([100, 100, 99, 98, 97, 96, 95, 90, 110, 120, 130, 140.])
        _, run, decisions = self.run_filter(prices, np.ones(len(prices)), "MOM_NO_PROGRESS_5", strategy="TSMOM_21")
        self.assertEqual(decisions.risk_signal_idx.iloc[0], 6)
        self.assertEqual(decisions.actual_exit_idx.iloc[0], 7)
        self.assertEqual(decisions.risk_signal_age.iloc[0], 5)
        self.assertTrue(decisions.early_exit.iloc[0])
        self.assertAlmostEqual(run["gross_return"][7], 90 / 95 - 1)
        np.testing.assert_array_equal(run["executed"], [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(run["gross_return"][8], 0)

    def test_last_day_risk_signal_is_pending_without_fabricated_close(self):
        prices = np.array([100, 100, 99, 98, 97, 96, 95.])
        base, run, decisions = self.run_filter(prices, np.ones(len(prices)), "MOM_NO_PROGRESS_5", strategy="TSMOM_21")
        np.testing.assert_array_equal(run["executed"], base["executed"])
        self.assertTrue(decisions.risk_exit_pending.iloc[0])
        self.assertTrue(decisions.actual_open.iloc[0])
        self.assertFalse(decisions.early_exit.iloc[0])
        self.assertTrue(pd.isna(decisions.actual_exit_idx.iloc[0]))
        self.assertEqual(run["target"][-1], 0)

    def test_adverse_exit_uses_entry_volatility_and_signed_simple_path(self):
        prices = [100, 100, 99, 97, 98, 110, 120]
        features = pd.DataFrame({"realized_vol_20_pct": [np.sqrt(252), 500., 500., 500., 500., 500., 500.]})
        _, run, decisions = self.run_filter(prices, np.ones(7), "MR_ADVERSE_2VOL", features)
        self.assertAlmostEqual(decisions.entry_daily_volatility.iloc[0], .01)
        self.assertEqual(decisions.risk_signal_idx.iloc[0], 3)
        self.assertEqual(decisions.actual_exit_idx.iloc[0], 4)
        self.assertAlmostEqual(decisions.simple_return_at_risk_signal.iloc[0], (99 / 100 - 1) + (97 / 99 - 1))
        self.assertAlmostEqual(run["gross_return"][4], 98 / 97 - 1)
        self.assertEqual(run["gross_return"][5], 0)

    def test_trailing_exit_requires_prior_activation(self):
        prices = [100, 100, 103, 100, 101, 120]
        features = pd.DataFrame({"realized_vol_20_pct": np.full(6, np.sqrt(252))})
        _, run, decisions = self.run_filter(prices, np.ones(6), "TRAIL_2VOL", features)
        self.assertEqual(decisions.risk_signal_idx.iloc[0], 3)
        self.assertEqual(decisions.actual_exit_idx.iloc[0], 4)
        self.assertAlmostEqual(decisions.running_peak_at_risk_signal.iloc[0], .03)
        self.assertAlmostEqual(run["gross_return"][4], .01)
        self.assertEqual(run["gross_return"][5], 0)
        _, _, no_activation = self.run_filter([100, 100, 101, 99, 98, 97], np.ones(6), "TRAIL_2VOL", features)
        self.assertFalse(no_activation.early_exit.any())

    def test_family_inapplicable_and_input_objects_remain_unchanged(self):
        prices = np.array([100, 100, 99, 98, 97, 96, 95.])
        dates = pd.date_range("2010-01-01", periods=7)
        base = fixed_execute(prices, np.ones(7))
        features = pd.DataFrame({"efficiency_63": .8}, index=dates)
        before_base, before_features = deepcopy(base), features.copy(deep=True)
        spec = next(item for item in specs() if item["id"] == "MOM_NO_PROGRESS_5")
        run, decisions = run_candidate(prices, dates, base, features, spec, strategy="RSI_MR_14")
        np.testing.assert_array_equal(run["target"], base["target"])
        self.assertEqual(decisions.state.iloc[0], "NOT_APPLICABLE")
        for key, value in before_base.items():
            np.testing.assert_array_equal(base[key], value)
        pd.testing.assert_frame_equal(features, before_features)

    def test_rejects_changed_registry_parameter_or_misaligned_features(self):
        prices = np.full(4, 100.)
        dates = pd.date_range("2010-01-01", periods=4)
        base = fixed_execute(prices, np.ones(4))
        spec = next(item for item in specs() if item["id"] == "VOL_NOT_EXPANDING")
        features = pd.DataFrame({"vol_ratio_20_63": .8}, index=dates)
        spec["parameters"]["entry"]["maximum_ratio"] = 1.01
        with self.assertRaisesRegex(ValueError, "fixed registry"):
            run_candidate(prices, dates, base, features, spec)
        valid = next(item for item in specs() if item["id"] == "BASE")
        features.index = dates + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "calendar"):
            run_candidate(prices, dates, base, features, valid)

    def test_last_known_base_signal_has_prefix_consistent_filtered_target(self):
        for targets, terminal_ratio, expected in (
            ([1, 1, 0, 0, 0], .8, 0),
            ([0, 0, 1, 1, 0], .8, 1),
            ([0, 0, 1, 1, 0], 1.5, 0),
            ([1, 1, -1, -1, 0], .8, -1),
            ([1, 1, -1, -1, 0], 1.5, 0),
        ):
            with self.subTest(targets=targets, terminal_ratio=terminal_ratio):
                prices = np.full(5, 100.)
                dates = pd.date_range("2010-01-01", periods=5)
                features = pd.DataFrame({"vol_ratio_20_63": [.8, .8, terminal_ratio, .8, .8]}, index=dates)
                spec = next(item for item in specs() if item["id"] == "VOL_NOT_EXPANDING")
                full, _ = run_candidate(prices, dates, fixed_execute(prices, targets), features, spec)
                prefix, _ = run_candidate(prices[:3], dates[:3], fixed_execute(prices[:3], targets[:3]), features.iloc[:3], spec)
                self.assertEqual(prefix["target"][-1], expected)
                np.testing.assert_array_equal(prefix["target"], full["target"][:3])
                np.testing.assert_array_equal(prefix["executed"], full["executed"][:3])

    def test_pending_time_exit_cannot_suppress_known_terminal_base_reversal(self):
        prices = np.array([100, 100, 99, 98, 97, 96, 95, 94, 93.])
        targets = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1])
        dates = pd.date_range("2010-01-01", periods=len(prices))
        features = pd.DataFrame(index=dates)
        spec = next(item for item in specs() if item["id"] == "MOM_NO_PROGRESS_5")
        full, _ = run_candidate(prices, dates, fixed_execute(prices, targets), features, spec, strategy="TSMOM_21")
        prefix, _ = run_candidate(prices[:7], dates[:7], fixed_execute(prices[:7], targets[:7]), features.iloc[:7], spec, strategy="TSMOM_21")
        self.assertEqual(prefix["target"][-1], -1)
        np.testing.assert_array_equal(prefix["target"], full["target"][:7])


if __name__ == "__main__":
    unittest.main()
