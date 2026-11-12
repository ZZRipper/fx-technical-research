"""Independent timing, accounting-cohort and descriptive-feature checks."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fx_observation_features import (  # noqa: E402
    FEATURE_COLUMNS, MACRO_OBSERVATION_FIELDS, build_entry_features,
)


class EntryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2014-09-01", periods=220)
        self.price = 100 + np.arange(220) * 0.1 + np.sin(np.arange(220) / 4)
        self.prices = pd.DataFrame({
            "pair": "AUD", "date": self.dates,
            "price_USD_per_foreign": self.price,
        })
        self.known = pd.DataFrame({"pair": "AUD", "date": self.dates})
        for observation, values in MACRO_OBSERVATION_FIELDS.items():
            self.known[observation] = self.dates - pd.Timedelta(days=1)
            for value in values:
                self.known[value] = np.arange(220, dtype=float) / 10
        self.trades = pd.DataFrame([
            {
                "pair": "AUD", "strategy": "TEST", "trade_id": number,
                "signal_date": self.dates[signal],
                "entry_date": self.dates[signal + 1],
                "exit_date": self.dates[exit_] if exit_ is not None else pd.NaT,
                "position": position, "closed": exit_ is not None,
                "eligible_closed": exit_ is not None, "simple_return_pct": return_,
            }
            for number, (signal, exit_, position, return_) in enumerate([
                (80, 85, 1, 1), (90, 95, -1, -2),
                (100, 105, 1, 2), (110, 115, -1, -1),
                (119, 125, 1, 0), (130, 135, 1, 0), (140, None, 1, 1),
            ])
        ])

    def build(self, **replacements):
        inputs = {name: getattr(self, name) for name in ("prices", "trades", "known")}
        inputs.update(replacements)
        return build_entry_features(**inputs)

    def test_price_formulas_and_direction(self):
        enriched, _, _ = self.build()
        price_window = self.price[60:81]
        self.assertAlmostEqual(enriched.loc[0, "spot_return_20_pct"],
                               (price_window[-1] / price_window[0] - 1) * 100)
        self.assertAlmostEqual(enriched.loc[0, "realized_vol_20_pct"],
                               pd.Series(price_window).pct_change().dropna().std(ddof=1) * np.sqrt(252) * 100)
        self.assertAlmostEqual(enriched.loc[0, "efficiency_20"],
                               abs(price_window[-1] - price_window[0]) / np.abs(np.diff(price_window)).sum())
        self.assertAlmostEqual(enriched.loc[0, "distance_ma_20_pct"],
                               (self.price[80] / self.price[61:81].mean() - 1) * 100)
        self.assertEqual(enriched.loc[1, "direction_dxy_d20_pct"], -self.known.loc[90, "K_dxy_d20_pct"])
        self.assertEqual(enriched.loc[1, "direction_commodity_d20_pct"], -self.known.loc[90, "K_copper_d20_pct"])

    def test_future_changes_do_not_change_entry_features(self):
        original = self.build(trades=self.trades.iloc[:1])[0]
        changed = self.prices.copy()
        changed.loc[changed.date.gt(self.dates[80]), "price_USD_per_foreign"] *= 4
        future_trade = self.trades.iloc[:1].copy()
        future_trade["simple_return_pct"] = -999
        future_trade["exit_date"] = self.dates[150]
        recomputed = self.build(prices=changed, trades=future_trade)[0]
        pd.testing.assert_frame_equal(original[list(FEATURE_COLUMNS)], recomputed[list(FEATURE_COLUMNS)])

    def test_cohorts_require_exit_and_keep_cross_period_counts(self):
        _, contrasts, coverage = self.build()
        coverage = coverage.set_index("cohort")
        old = coverage.loc["2005_2014"]
        self.assertEqual(old.n_entered, 5)
        self.assertEqual(old.n_cross_period, 1)
        self.assertEqual(old.n_eligible_closed, 4)
        self.assertEqual((old.n_win, old.n_loss, old.n_flat), (2, 2, 0))
        self.assertEqual(coverage.loc["FULL", "n_eligible_closed"], 6)
        self.assertEqual(coverage.loc["FULL", "n_open"], 1)
        self.assertEqual(coverage.loc["FULL", "n_flat"], 2)
        self.assertEqual(len(contrasts), 4 * len(FEATURE_COLUMNS))

    def test_macro_timing_and_missing_provenance_fail(self):
        for lag in (0, 8):
            with self.subTest(lag=lag):
                invalid = self.known.copy()
                invalid.loc[80, "K_vix_obs_date"] = self.dates[80] - pd.Timedelta(days=lag)
                with self.assertRaisesRegex(ValueError, "timing"):
                    self.build(known=invalid)
        invalid = self.known.copy()
        invalid.loc[80, "K_rate_obs_date"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "no supporting"):
            self.build(known=invalid)
        valid = self.known.copy()
        valid.loc[80, "K_vix_obs_date"] = self.dates[80] - pd.Timedelta(days=7)
        self.build(known=valid)

    def test_coverage_distinguishes_missing_from_inapplicable(self):
        known = self.known.copy()
        known.loc[80, "K_vix"] = np.nan
        _, contrasts, _ = self.build(known=known)
        row = contrasts.loc[contrasts.cohort.eq("FULL") & contrasts.feature.eq("K_vix")].iloc[0]
        self.assertEqual(row.n_applicable, 6)
        self.assertEqual(row.feature_n, 5)
        self.assertAlmostEqual(row.coverage, 5 / 6)
        inputs = {name: getattr(self, name).assign(pair="EUR") for name in ("prices", "trades", "known")}
        enriched, contrasts, _ = build_entry_features(**inputs)
        self.assertTrue(enriched.commodity_d20_pct.isna().all())
        commodity = contrasts.loc[contrasts.feature.eq("commodity_d20_pct")]
        self.assertTrue(commodity.n_applicable.eq(0).all())
        self.assertTrue(commodity.coverage.isna().all())

    def test_accounts_without_eligible_outcomes_remain_visible(self):
        trades = self.trades.iloc[[-1]].copy()
        enriched, contrasts, coverage = self.build(trades=trades)
        self.assertEqual(enriched.outcome.iloc[0], "OPEN")
        self.assertEqual(len(coverage), 4)
        self.assertEqual(len(contrasts), 4 * len(FEATURE_COLUMNS))
        self.assertTrue(coverage.n_eligible_closed.eq(0).all())
        self.assertTrue(contrasts.standardized_mean_difference.isna().all())
        self.assertTrue(coverage.low_sample_flag.all())

    def test_standardized_difference_uses_sample_pooled_variance(self):
        _, contrasts, _ = self.build()
        row = contrasts.loc[contrasts.cohort.eq("2005_2014") & contrasts.feature.eq("K_vix")].iloc[0]
        wins, losses = np.array([8., 10.]), np.array([9., 11.])
        pooled_sd = np.sqrt((wins.var(ddof=1) + losses.var(ddof=1)) / 2)
        self.assertAlmostEqual(row.standardized_mean_difference, (wins.mean() - losses.mean()) / pooled_sd)
        self.assertEqual(row.n_feature_win, 2)
        self.assertTrue(row.low_sample_flag)

    def test_flat_zero_volatility_and_insufficient_warmup(self):
        prices = self.prices.assign(price_USD_per_foreign=100.)
        enriched, contrasts, _ = self.build(prices=prices)
        self.assertTrue(enriched.efficiency_20.eq(0).all())
        self.assertTrue(enriched.realized_vol_20_pct.eq(0).all())
        self.assertTrue(enriched.vol_ratio_20_63.isna().all())
        self.assertTrue(contrasts.loc[contrasts.feature.eq("efficiency_20"), "standardized_mean_difference"].isna().all())
        warmup = self.trades.iloc[:1].copy()
        warmup["signal_date"] = self.dates[10]
        warmup["entry_date"] = self.dates[11]
        early = self.build(trades=warmup)[0]
        self.assertTrue(early.spot_return_20_pct.isna().all())

    def test_inputs_are_not_mutated(self):
        originals = {name: getattr(self, name).copy(deep=True) for name in ("prices", "trades", "known")}
        self.build()
        for name, original in originals.items():
            pd.testing.assert_frame_equal(getattr(self, name), original)


if __name__ == "__main__":
    unittest.main()
