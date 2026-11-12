"""Synthetic causal-clock and fixed-state contract checks; no research data."""

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fx_baselines import fixed_execute
from fx_is_states import STATE_BINS
from fx_state_filters import run_state_filter


class StateFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 270
        rng = np.random.default_rng(46)
        cls.dates = pd.bdate_range("2010-01-04", periods=n)
        cls.prices = 100. * np.cumprod(1. + rng.normal(0., .006, n))
        cls.targets = np.repeat([1, 0, -1, 1, 0, -1, 1, 0, -1], 30)
        cls.targets[-1] = 1
        cls.base = fixed_execute(cls.prices, cls.targets)
        cls.features = pd.DataFrame({
            "date": cls.dates,
            "realized_vol_20_pct": rng.choice([4., 8., 12., 18.], n),
            "spot_return_20_pct": rng.normal(0., 5., n),
            "distance_ma_60_pct": rng.normal(0., 5., n),
            "K_spread_bp": rng.normal(0., 150., n),
            "K_spread_d20_bp": rng.normal(0., 15., n),
            "K_dxy_d20_pct": rng.normal(0., 2., n),
            "K_copper_d20_pct": rng.normal(0., 3., n),
            "K_brent_d20_pct": rng.normal(0., 3., n),
            "vol_ratio_20_63": rng.choice([.7, .9, 1.1, 1.4], n),
            "efficiency_63": rng.choice([.05, .15, .3, .6], n),
            "K_vix": rng.choice([10., 17., 24., 35.], n),
            "K_vix_d5": rng.choice([-4., -1., 1., 4.], n),
        })
        for field in ("K_vix_obs_date", "K_rate_obs_date", "K_dxy_obs_date", "K_copper_obs_date", "K_brent_obs_date"):
            cls.features[field] = cls.dates - pd.Timedelta(days=1)
        cls.features.loc[0, "vol_ratio_20_63"] = 1.1
        cls.features.loc[1:29, "vol_ratio_20_63"] = .7
        cls.features.loc[60, "vol_ratio_20_63"] = .7
        cls.features.loc[n - 1, "vol_ratio_20_63"] = np.nan

    def run_filter(self, spec=None, features=None, base=None):
        return run_state_filter(self.prices, self.dates, self.base if base is None else base,
                                self.features if features is None else features,
                                spec or {"feature": "vol_ratio_20_63", "state": 2}, "AUD", "MA_20_60")

    def test_every_feature_bin_is_prefix_causal(self):
        # 13 features x 4 bins x 12 stops = 624 independently truncated replays.
        stops = [1, 2, 7, 8, 63, 64, 100, 251, 252, 253, 269, 270]
        for feature in STATE_BINS:
            for state in range(4):
                spec = {"feature": feature, "state": state, "id": f"{feature}/{state}"}
                full, _ = self.run_filter(spec)
                for stop in stops:
                    with self.subTest(feature=feature, state=state, stop=stop):
                        part, _ = run_state_filter(
                            self.prices[:stop], self.dates[:stop],
                            fixed_execute(self.prices[:stop], self.targets[:stop]),
                            self.features.iloc[:stop], spec, "AUD", "MA_20_60")
                        for key in ("target", "executed", "held", "gross_return", "equity"):
                            np.testing.assert_array_equal(part[key], full[key][:stop], err_msg=key)

    def test_rejected_episode_stays_locked_and_new_episode_can_enter(self):
        run, decisions = self.run_filter()
        self.assertTrue(np.all(run["target"][:30] == 0))
        self.assertFalse(decisions.iloc[0].accepted)
        self.assertEqual(decisions.iloc[0].state, 2)
        self.assertEqual(run["target"][60], -1)
        self.assertTrue(decisions.loc[decisions.signal_idx.eq(60)].iloc[0].accepted)

    def test_pending_final_signal_has_no_fill_and_unknown_is_retained(self):
        kept, decisions = self.run_filter()
        last = decisions.iloc[-1]
        self.assertTrue(last.entry_pending)
        self.assertTrue(last.accepted)
        self.assertEqual(last.state, -1)
        self.assertEqual(last.status, "UNKNOWN")
        self.assertTrue(pd.isna(last.entry_idx))
        self.assertTrue(pd.isna(last.actual_entry_idx))
        self.assertTrue(pd.isna(last.entry_date))
        self.assertEqual(kept["target"][-1], 1)
        changed = self.features.copy(deep=True)
        changed.loc[len(changed) - 1, "vol_ratio_20_63"] = 1.1
        rejected, rejected_decisions = self.run_filter(features=changed)
        self.assertFalse(rejected_decisions.iloc[-1].accepted)
        self.assertEqual(rejected["target"][-1], 0)
        for key in ("executed", "held", "gross_return", "equity"):
            np.testing.assert_array_equal(kept[key], rejected[key])

    def test_empty_account_and_input_objects_are_preserved(self):
        saved = self.features.copy(deep=True)
        saved_target = self.base["target"].copy()
        self.run_filter()
        pd.testing.assert_frame_equal(self.features, saved)
        np.testing.assert_array_equal(self.base["target"], saved_target)
        zero = fixed_execute(self.prices, np.zeros(len(self.prices), dtype=int))
        run, decisions = self.run_filter(base=zero)
        self.assertTrue(decisions.empty)
        self.assertFalse(run["held"].any())

    def test_unknown_state_and_custom_thresholds_cannot_be_rejected(self):
        invalid = [
            {"feature": "vol_ratio_20_63", "state": -1},
            {"feature": "vol_ratio_20_63", "state": 4},
            {"feature": "vol_ratio_20_63", "state": True},
            {"feature": "vol_ratio_20_63", "state": 2, "bins": [.8, 1., 2.]},
            {"feature": "vol_ratio_20_63", "state": 2, "threshold": 1.1},
            {"feature": "future_trade_outcome", "state": 2},
        ]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.run_filter(spec)

    def test_feature_calendar_and_macro_observation_clock_are_checked(self):
        shifted = self.features.copy(deep=True)
        shifted["date"] = shifted.date + pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            self.run_filter(features=shifted)
        contemporaneous = self.features.copy(deep=True)
        contemporaneous.loc[0, "K_vix_obs_date"] = self.dates[0]
        with self.assertRaises(AssertionError):
            self.run_filter(features=contemporaneous)


if __name__ == "__main__":
    unittest.main()
