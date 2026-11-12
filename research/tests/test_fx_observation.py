"""Small, hand-calculated boundary cases for observation-only diagnostics."""

import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fx_baselines import fixed_execute, trade_ledger
from fx_observation import (
    calendar_periods,
    drawdown_episodes,
    select_rolling_windows,
    trade_excursions,
    window_attribution,
)


def empty_trades():
    return pd.DataFrame({
        "entry_date": pd.Series(dtype="datetime64[ns]"),
        "exit_date": pd.Series(dtype="datetime64[ns]"),
        "eligible_closed": pd.Series(dtype=bool),
        "closed": pd.Series(dtype=bool),
        "simple_return_pct": pd.Series(dtype=float),
    })


def reversal_case():
    dates = pd.to_datetime([
        "2020-12-29", "2020-12-30", "2020-12-31",
        "2021-01-04", "2021-01-05", "2021-01-06",
    ])
    # The long earns +10%, -10%; the short then earns -10%, +10%.
    run = fixed_execute([100., 100., 110., 99., 108.9, 98.01], [1, 1, -1, -1, -1, -1])
    ledger = trade_ledger(run, dates, "EUR", "HAND_CALCULATED")
    return dates, run, ledger


class TradeExcursionTests(unittest.TestCase):
    def test_reversal_day_belongs_to_old_trade_and_tail_stays_open(self):
        dates, run, ledger = reversal_case()
        result = trade_excursions(run, dates, ledger)

        self.assertEqual(result.entry_idx.tolist(), [1, 3])
        self.assertEqual(result.exit_idx.tolist(), [3, 5])
        self.assertEqual(result.closed.tolist(), [True, False])
        self.assertEqual(result.holding_observations.tolist(), [2, 2])
        np.testing.assert_allclose(result.simple_return_pct, [0., 0.], atol=1e-12)
        np.testing.assert_allclose(result.compound_return_pct, [-1., -1.], atol=1e-12)
        np.testing.assert_allclose(result.mfe_simple_pct, [10., 0.], atol=1e-12)
        np.testing.assert_allclose(result.mae_simple_pct, [0., -10.], atol=1e-12)
        self.assertEqual(result.iloc[0].mfe_date, dates[2])
        self.assertEqual(result.iloc[1].mae_date, dates[4])
        self.assertAlmostEqual(result.simple_pnl_usd.sum(), 100000. * run["daily_ret"].sum())

        # A short with daily 1x exposure does not earn minus the endpoint spot move.
        naive_short_pct = -(98.01 / 99. - 1.) * 100.
        self.assertAlmostEqual(naive_short_pct, 1.)
        self.assertNotAlmostEqual(result.iloc[1].compound_return_pct, naive_short_pct)

    def test_last_close_entry_has_zero_observations_and_is_not_liquidated(self):
        dates = pd.bdate_range("2021-01-04", periods=4)
        run = fixed_execute([100., 100., 110., 99.], [0, 0, 1, 1])
        ledger = trade_ledger(run, dates)
        result = trade_excursions(run, dates, ledger)
        row = result.iloc[0]
        self.assertFalse(row.closed)
        self.assertFalse(row.eligible_closed)
        self.assertEqual(row.holding_observations, 0)
        self.assertEqual(row.simple_return_pct, 0.)
        self.assertEqual(row.compound_return_pct, 0.)
        self.assertEqual(row.mfe_simple_pct, 0.)
        self.assertEqual(row.mae_simple_pct, 0.)
        self.assertTrue(pd.isna(row.giveback_ratio))
        self.assertEqual(row.mfe_date, dates[-1])

    def test_giveback_can_exceed_one_and_no_prior_gain_has_no_ratio(self):
        dates = pd.bdate_range("2021-01-04", periods=4)
        run = fixed_execute([100., 100., 110., 88.], [1, 1, 1, 1])
        row = trade_excursions(run, dates, trade_ledger(run, dates)).iloc[0]
        self.assertAlmostEqual(row.simple_return_pct, -10.)
        self.assertAlmostEqual(row.compound_return_pct, -12.)
        self.assertAlmostEqual(row.mfe_simple_pct, 10.)
        self.assertAlmostEqual(row.giveback_simple_pct, 20.)
        self.assertAlmostEqual(row.giveback_ratio, 2.)
        self.assertTrue(row.loser_with_prior_gain)

        losing_run = fixed_execute([100., 100., 90., 81.], [1, 1, 1, 1])
        losing = trade_excursions(losing_run, dates, trade_ledger(losing_run, dates)).iloc[0]
        self.assertEqual(losing.mfe_simple_pct, 0.)
        self.assertTrue(pd.isna(losing.giveback_ratio))
        self.assertFalse(losing.loser_with_prior_gain)

    def test_no_trade_account_retains_empty_typed_outcomes(self):
        dates = pd.bdate_range("2021-01-04", periods=4)
        run = fixed_execute([100., 101., 99., 102.], [0, 0, 0, 0])
        result = trade_excursions(run, dates, trade_ledger(run, dates))
        self.assertTrue(result.empty)
        self.assertTrue({"simple_return_pct", "compound_return_pct", "mfe_simple_pct", "mae_simple_pct"}.issubset(result.columns))
        self.assertEqual(result.simple_return_pct.sum(), 0.)


class DrawdownTests(unittest.TestCase):
    def test_last_plateau_peak_cross_year_recovery_and_unrecovered_tail(self):
        dates = pd.to_datetime([
            "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04",
            "2021-01-05", "2021-01-06", "2021-01-07",
        ])
        returns = np.array([.10, 0., -.04, -.02, .06, 0., -.03])
        episodes = drawdown_episodes(returns, dates, "2020-12-28")
        self.assertEqual(len(episodes), 2)
        first, tail = episodes.iloc[0], episodes.iloc[1]
        self.assertEqual(first.peak_date, dates[1])
        self.assertEqual(first.decline_start_date, dates[2])
        self.assertEqual(first.trough_date, dates[3])
        self.assertEqual(first.recovery_date, dates[4])
        self.assertTrue(first.recovered)
        self.assertAlmostEqual(first.depth_pp, 6.)
        self.assertAlmostEqual(first.peak_equity_drawdown_pct, 6. / 110. * 100.)
        self.assertEqual(first.duration_observations, 3)
        self.assertEqual(first.duration_calendar_days, 6)
        self.assertEqual(first.decline_observations, 2)
        self.assertEqual(first.recovery_observations, 1)
        self.assertEqual(tail.peak_date, dates[5])
        self.assertEqual(tail.trough_date, dates[6])
        self.assertEqual(tail.observed_end_date, dates[-1])
        self.assertFalse(tail.recovered)
        self.assertTrue(pd.isna(tail.recovery_date))
        self.assertTrue(pd.isna(tail.recovery_observations))
        self.assertAlmostEqual(tail.depth_pp, 3.)
        self.assertLess(first.end_idx, tail.peak_idx)

    def test_flat_path_has_no_drawdown_episode(self):
        dates = pd.bdate_range("2021-01-04", periods=3)
        result = drawdown_episodes(np.zeros(3), dates, "2020-12-31")
        self.assertTrue(result.empty)
        self.assertIn("recovered", result)


class CalendarTests(unittest.TestCase):
    def test_cross_year_daily_attribution_keeps_entry_cohorts_censored(self):
        dates, run, ledger = reversal_case()
        original_ledger = ledger.copy(deep=True)
        trades = trade_excursions(run, dates, ledger)
        periods = calendar_periods(run["daily_ret"], run["held"], dates, trades)
        years = periods.loc[periods.period_type.eq("year")].set_index("period")
        self.assertAlmostEqual(years.loc["2020", "simple_return_pct"], 10.)
        self.assertAlmostEqual(years.loc["2021", "simple_return_pct"], -10.)
        self.assertAlmostEqual(years.loc["2021", "compound_return_pct"], -10.9)
        self.assertEqual(years.loc["2020", "cohort_entries"], 1)
        self.assertEqual(years.loc["2020", "cohort_closed_trades"], 0)
        self.assertEqual(years.loc["2020", "cohort_censored_trades"], 1)
        self.assertEqual(years.loc["2021", "cohort_entries"], 1)
        self.assertEqual(years.loc["2021", "cohort_closed_trades"], 0)
        self.assertEqual(years.loc["2021", "cohort_censored_trades"], 1)
        for _, group in periods.groupby("period_type"):
            self.assertAlmostEqual(group.simple_return_pct.sum(), run["daily_ret"].sum() * 100.)
            self.assertAlmostEqual(np.prod(1. + group.compound_return_pct / 100.), run["equity"][-1])
        pd.testing.assert_frame_equal(ledger, original_ledger)
        self.assertTrue(ledger.iloc[0].closed)
        self.assertFalse(ledger.iloc[1].closed)

        windows = [{"start_idx": 3, "end_idx": 5, "window_id": "2021", "window_type": "year"}]
        attributed = window_attribution(run["daily_ret"], dates, trades, windows, offset=0)
        carry = attributed.loc[attributed.trade_id.eq(1)].iloc[0]
        tail = attributed.loc[attributed.trade_id.eq(2)].iloc[0]
        self.assertTrue(carry.carry_in)
        self.assertEqual(carry.overlap_observations, 1)
        self.assertAlmostEqual(carry.simple_return_contribution_pp, -10.)
        self.assertTrue(tail.continues_after_window)
        self.assertAlmostEqual(attributed.simple_return_contribution_pp.sum(), -10.)

    def test_period_missing_its_beginning_is_incomplete(self):
        dates = pd.bdate_range("2021-07-02", "2021-12-31")
        periods = calendar_periods(np.zeros(len(dates)), np.zeros(len(dates)), dates, empty_trades())
        self.assertEqual(set(periods.period), {"2021", "2021H2"})
        self.assertFalse(periods.complete_calendar_period.any())


class RollingTests(unittest.TestCase):
    def test_complete_quote_windows_include_cash_days_and_selected_cases_are_disjoint(self):
        dates = pd.bdate_range("2005-01-03", periods=504)
        returns = np.zeros(504)
        held = np.zeros(504, dtype=np.int8)
        returns[:126], held[:126] = .002, 1
        returns[252:378], held[252:378] = -.001, 1
        windows, rolling = select_rolling_windows(returns, held, dates, empty_trades(), per_sign=1)
        self.assertTrue(np.isnan(rolling[:251]).all())
        self.assertEqual(np.isfinite(rolling).sum(), 253)
        self.assertAlmostEqual(rolling[251], 25.2)
        self.assertAlmostEqual(rolling[503], -12.6)
        self.assertEqual(set(zip(windows.start_idx, windows.end_idx)), {(0, 251), (252, 503)})
        self.assertEqual(windows.observations.tolist(), [252, 252])
        self.assertEqual(windows.active_days.tolist(), [126, 126])
        coverage = np.zeros(504, dtype=int)
        for row in windows.itertuples(index=False):
            coverage[row.start_idx:row.end_idx + 1] += 1
        self.assertLessEqual(coverage.max(), 1)

    def test_short_sample_does_not_borrow_warmup_to_make_a_window(self):
        dates = pd.bdate_range("2005-01-03", periods=251)
        windows, rolling = select_rolling_windows(np.full(251, .001), np.ones(251), dates, empty_trades())
        self.assertTrue(windows.empty)
        self.assertTrue(np.isnan(rolling).all())


if __name__ == "__main__":
    unittest.main()
