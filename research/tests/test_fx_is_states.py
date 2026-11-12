"""Synthetic tests of fixed IS state labels and signal-time information use."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fx_is_states import MACRO_PROVENANCE, STATE_BINS, entry_features, state_codes


class InSampleStateTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2010-01-01", periods=300)
        step = np.arange(300, dtype=float)
        self.features = pd.DataFrame({
            "date": self.dates,
            "realized_vol_20_pct": np.sqrt(252.),
            "spot_return_20_pct": 2. + step / 100,
            "distance_ma_60_pct": 1. + step / 200,
            "K_spread_bp": 100. + step,
            "K_spread_d20_bp": 10. + step / 10,
            "K_dxy_d20_pct": 1. + step / 100,
            "K_copper_d20_pct": 2. + step / 100,
            "K_brent_d20_pct": -3. - step / 100,
            "K_vix": 20. + step / 100,
            "K_vix_d5": 3. + step / 100,
        })
        for name in ("K_rate_obs_date", "K_dxy_obs_date", "K_copper_obs_date", "K_brent_obs_date", "K_vix_obs_date"):
            self.features[name] = self.dates - pd.Timedelta(days=1)
        self.run = {"gross_return": step / 10000.}
        indices = [62, 251, 260]
        self.ledger = pd.DataFrame({
            "signal_idx": indices, "signal_date": self.dates[indices],
            "entry_date": self.dates[np.array(indices) + 1],
            "position": [1, -1, 1], "gross_simple_return_pct": [999., -999., 0.],
        })

    def test_fixed_registry_and_left_inclusive_bin_boundaries(self):
        self.assertEqual(len(STATE_BINS), 13)
        for feature, edges in STATE_BINS.items():
            with self.subTest(feature=feature):
                self.assertEqual(edges, sorted(set(edges)))
                values = [np.nextafter(edges[0], -np.inf), edges[0],
                          np.nextafter(edges[1], -np.inf), edges[1],
                          np.nextafter(edges[2], -np.inf), edges[2],
                          np.nan, np.inf, -np.inf]
                np.testing.assert_array_equal(state_codes(values, edges), [0, 1, 1, 2, 2, 3, -1, -1, -1])

    def test_exact_signal_row_direction_and_volatility_units(self):
        result = entry_features(self.features, self.run, self.ledger, "AUD")
        for row, signal, direction in zip(result.itertuples(index=False), self.ledger.signal_idx, self.ledger.position):
            self.assertEqual(row.date, self.dates[signal])
            self.assertAlmostEqual(row.direction_return_20_vol, direction * self.features.spot_return_20_pct.iloc[signal] / np.sqrt(20.))
            self.assertAlmostEqual(row.direction_distance_ma_60_vol, direction * self.features.distance_ma_60_pct.iloc[signal] / np.sqrt(20.))
            self.assertEqual(row.direction_spread_bp, direction * self.features.K_spread_bp.iloc[signal])
            self.assertEqual(row.direction_spread_d20_bp, direction * self.features.K_spread_d20_bp.iloc[signal])
            # Positive direction-adjusted DXY retains the established conflict convention.
            self.assertEqual(row.direction_dxy_d20_pct, direction * self.features.K_dxy_d20_pct.iloc[signal])

    def test_commodity_mapping_and_structural_unknown(self):
        for pair, feature in (("AUD", "K_copper_d20_pct"), ("CAD", "K_brent_d20_pct"), ("NOK", "K_brent_d20_pct")):
            with self.subTest(pair=pair):
                result = entry_features(self.features, self.run, self.ledger, pair)
                expected = self.features[feature].iloc[self.ledger.signal_idx].to_numpy() * self.ledger.position.to_numpy()
                np.testing.assert_array_equal(result.direction_commodity_d20_pct, expected)
        result = entry_features(self.features, self.run, self.ledger, "EUR")
        self.assertTrue(result.direction_commodity_d20_pct.isna().all())
        np.testing.assert_array_equal(state_codes(result.direction_commodity_d20_pct, STATE_BINS["direction_commodity_d20_pct"]), [-1, -1, -1])

    def test_trailing_base_pnl_includes_only_returns_earned_by_signal_close(self):
        result = entry_features(self.features, self.run, self.ledger, "JPY")
        for row, signal in zip(result.itertuples(index=False), self.ledger.signal_idx):
            self.assertAlmostEqual(row.base_pnl_63_pct, self.run["gross_return"][signal - 62:signal + 1].sum() * 100.)
            if signal >= 251:
                self.assertAlmostEqual(row.base_pnl_252_pct, self.run["gross_return"][signal - 251:signal + 1].sum() * 100.)
            else:
                self.assertTrue(np.isnan(row.base_pnl_252_pct))

    def test_future_return_macro_and_outcome_changes_do_not_change_entry(self):
        ledger = self.ledger.iloc[[1]].copy()
        original = entry_features(self.features, self.run, ledger, "AUD")
        changed_features = self.features.copy(deep=True)
        future = changed_features.date.gt(ledger.signal_date.iloc[0])
        numeric = changed_features.select_dtypes(include=[np.number]).columns
        changed_features.loc[future, numeric] = 999999.
        for column in changed_features:
            if column.endswith("_obs_date"):
                changed_features.loc[future, column] = pd.Timestamp("2099-01-01")
        changed_run = {"gross_return": self.run["gross_return"].copy()}
        changed_run["gross_return"][252:] = -99.
        ledger["gross_simple_return_pct"] = 123456.
        changed = entry_features(changed_features, changed_run, ledger, "AUD")
        pd.testing.assert_frame_equal(original, changed)

    def test_macro_observations_are_strictly_prior_and_at_most_seven_days_old(self):
        ledger = self.ledger.iloc[[0]]
        for lag in (0, -1, 8):
            with self.subTest(lag=lag):
                features = self.features.copy()
                features.loc[62, "K_vix_obs_date"] = self.dates[62] - pd.Timedelta(days=lag)
                with self.assertRaises((AssertionError, ValueError)):
                    entry_features(features, self.run, ledger, "JPY")
        features = self.features.copy()
        features.loc[62, "K_vix_obs_date"] = self.dates[62] - pd.Timedelta(days=7)
        entry_features(features, self.run, ledger, "JPY")

    def test_missing_macro_values_remain_unknown(self):
        features = self.features.copy()
        features.loc[62, ["K_vix", "K_vix_d5", "K_vix_obs_date"]] = [np.nan, np.nan, pd.NaT]
        result = entry_features(features, self.run, self.ledger.iloc[[0]], "JPY")
        self.assertEqual(state_codes(result.K_vix, STATE_BINS["K_vix"])[0], -1)
        self.assertEqual(state_codes(result.K_vix_d5, STATE_BINS["K_vix_d5"])[0], -1)

    def test_finite_macro_requires_its_observation_date(self):
        ledger = self.ledger.iloc[[0]]
        for observation, sources in MACRO_PROVENANCE.items():
            for source in sources:
                for mode in ("missing_cell", "missing_column"):
                    with self.subTest(source=source, mode=mode):
                        features = self.features.copy()
                        features.loc[62, list(sources)] = np.nan
                        features.loc[62, source] = 0.
                        if mode == "missing_cell":
                            features.loc[62, observation] = pd.NaT
                        else:
                            features = features.drop(columns=observation)
                        with self.assertRaisesRegex(AssertionError, "observation date"):
                            entry_features(features, self.run, ledger, "AUD")

    def test_missing_macro_provenance_outside_selected_signals_has_no_effect(self):
        ledger = self.ledger.iloc[[0]]
        expected = entry_features(self.features, self.run, ledger, "AUD")
        features = self.features.copy()
        unselected = features.index != 62
        for observation in MACRO_PROVENANCE:
            features.loc[unselected, observation] = pd.NaT
        pd.testing.assert_frame_equal(expected, entry_features(features, self.run, ledger, "AUD"))

    def test_signal_indices_must_be_integral_finite_and_in_range(self):
        for invalid in (-1, 300, 62.5, np.nan, np.inf, -np.inf):
            with self.subTest(index=invalid):
                ledger = self.ledger.iloc[[0]].copy()
                ledger["signal_idx"] = invalid
                with self.assertRaisesRegex(AssertionError, "Signal indices"):
                    entry_features(self.features, self.run, ledger, "AUD")

    def test_run_must_have_exact_feature_calendar_length_and_one_dimension(self):
        for returns in (np.zeros(299), np.zeros(301), np.zeros((300, 1)), np.array(0.)):
            with self.subTest(shape=returns.shape):
                with self.assertRaisesRegex(AssertionError, "BASE return run"):
                    entry_features(self.features, {"gross_return": returns}, self.ledger, "AUD")

    def test_empty_ledger_preserves_empty_feature_schema(self):
        result = entry_features(self.features, self.run, self.ledger.iloc[:0], "AUD")
        self.assertEqual(len(result), 0)
        self.assertIn("base_pnl_252_pct", result)
        self.assertIn("direction_dxy_d20_pct", result)

    def test_signal_date_mismatch_is_rejected(self):
        ledger = self.ledger.copy()
        ledger.loc[0, "signal_date"] += pd.Timedelta(days=1)
        with self.assertRaises((AssertionError, ValueError)):
            entry_features(self.features, self.run, ledger, "JPY")

    def test_zero_volatility_cannot_create_a_numeric_extension_state(self):
        features = self.features.copy()
        features.loc[62, "realized_vol_20_pct"] = 0.
        result = entry_features(features, self.run, self.ledger.iloc[[0]], "JPY")
        self.assertTrue(result.direction_return_20_vol.isna().all())
        self.assertTrue(result.direction_distance_ma_60_vol.isna().all())

    def test_inputs_are_not_mutated(self):
        features, ledger, returns = self.features.copy(deep=True), self.ledger.copy(deep=True), self.run["gross_return"].copy()
        entry_features(self.features, self.run, self.ledger, "AUD")
        pd.testing.assert_frame_equal(features, self.features)
        pd.testing.assert_frame_equal(ledger, self.ledger)
        np.testing.assert_array_equal(returns, self.run["gross_return"])


if __name__ == "__main__":
    unittest.main()
