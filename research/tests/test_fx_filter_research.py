"""Synthetic arithmetic and selection-isolation tests; no research files read."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fx_baselines import fixed_execute
from fx_filter_evaluation import (
    daily_account, delayed_schedule, evaluate_window, fresh_start,
    quality, screen_candidates, simple_ledger,
)


def candidate_metrics(identifier="candidate", pair="EUR", strategy="TSMOM_21"):
    rows = []
    for split, baseline_return, filtered_return in (
        ("IS", 20., 25.), ("IS_EARLY", 5., 6.),
        ("IS_MIDDLE", -2., -1.), ("IS_LATE", 2., 3.),
    ):
        for variant, total in (("BASE", baseline_return), ("FILTER", filtered_return)):
            rows.append({
                "candidate_id": identifier, "pair": pair, "strategy": strategy,
                "filter_id": "TEST_FIXED", "is_new": True, "split": split, "variant": variant,
                "gross_simple_return_pct": total, "net_simple_return_pct": total,
                "gross_annualized_simple_pct": 1.,
                "gross_max_drawdown_pp": -10. if variant == "BASE" else -12.,
                "win_rate": .5 if variant == "BASE" else .52,
                "payoff_ratio": 1. if variant == "BASE" else 1.1,
                "closed_n": 100 if variant == "BASE" else 50,
            })
    return pd.DataFrame(rows)


class WindowAccountingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2010-01-01", periods=5)
        self.run = fixed_execute([100, 100, 110, 99, 108.9], [1, 1, 0, 0, 0])

    def test_simple_accounting_is_additive_and_optional_cost_is_fixed_notional(self):
        result = evaluate_window(self.run, self.dates, self.dates[0], self.dates[-1], cost_bp=1.)
        self.assertAlmostEqual(result["gross_simple_return_pct"], 0.)
        self.assertAlmostEqual(result["gross_compound_return_pct"], -1.)
        self.assertAlmostEqual(result["net_simple_return_pct"], -.02)
        self.assertAlmostEqual(result["cost_drag_pp"], .02)
        self.assertAlmostEqual(result["gross_max_drawdown_pp"], -10.)
        self.assertAlmostEqual(result["net_max_drawdown_pp"], -10.01)
        self.assertEqual(result["turnover_units"], 2)
        self.assertAlmostEqual(result["calendar_years"], 4 / 365.25)
        ledger = simple_ledger(self.run, self.dates, cost_bp=1.)
        self.assertAlmostEqual(ledger.net_trade_return_pct.iloc[0], -.02)
        self.assertAlmostEqual(ledger.net_pnl_usd.iloc[0], -20.)

    def test_continuous_split_preserves_carry_in_without_synthetic_entry_cost(self):
        result = evaluate_window(self.run, self.dates, self.dates[2], self.dates[-1], cost_bp=1.)
        self.assertAlmostEqual(result["gross_simple_return_pct"], 0.)
        self.assertAlmostEqual(result["net_simple_return_pct"], -.01)
        self.assertEqual(result["carry_in_n"], 1)
        self.assertAlmostEqual(result["carry_in_net_return_pp"], -.01)
        self.assertEqual(result["entry_n"], 0)
        self.assertEqual(result["closed_n"], 0)
        self.assertEqual(result["anchor_date"], self.dates[1])
        self.assertAlmostEqual(result["calendar_years"], 3 / 365.25)

    def test_cross_boundary_trade_is_censored_for_quality_not_liquidated(self):
        result = evaluate_window(self.run, self.dates, self.dates[0], self.dates[2], cost_bp=0.)
        self.assertAlmostEqual(result["gross_simple_return_pct"], 10.)
        self.assertEqual(result["closed_n"], 0)
        self.assertEqual(result["censored_entry_n"], 1)
        self.assertEqual(result["entry_n"], 1)
        self.assertTrue(np.isnan(result["win_rate"]))

    def test_reversal_cost_has_two_units_and_open_trade_has_no_exit_charge(self):
        run = fixed_execute(np.full(5, 100.), [1, -1, -1, -1, -1])
        _, net, turnover = daily_account(run, cost_bp=1.)
        np.testing.assert_array_equal(turnover, [0, 1, 2, 0, 0])
        self.assertAlmostEqual(net.sum() * 100, -.03)
        ledger = simple_ledger(run, self.dates, cost_bp=1.)
        np.testing.assert_allclose(ledger.net_trade_return_pct, [-.02, -.01])
        self.assertEqual(ledger.closed.tolist(), [True, False])

    def test_quality_counts_flat_trades_in_win_rate_denominator(self):
        result = quality([1., -1., 1e-10, -1e-10, 0.])
        self.assertEqual(result["closed_n"], 5)
        self.assertEqual(result["flat_n"], 3)
        self.assertAlmostEqual(result["win_rate"], .2)
        self.assertAlmostEqual(result["payoff_ratio"], 1.)

    def test_fresh_start_does_not_restart_a_base_episode_already_underway(self):
        dates = pd.date_range("2010-01-01", periods=10)
        base = fixed_execute(np.arange(100., 110.), [1, 1, 1, 1, 1, 0, -1, -1, 0, 0])
        fresh = fresh_start(base, base, dates, dates[3])
        np.testing.assert_array_equal(fresh["executed"], [0, 0, 0, 0, 0, 0, 0, -1, -1, 0])
        self.assertEqual(fresh["gross_return"][7], 0.)
        self.assertAlmostEqual(fresh["gross_return"][8], -(108 / 107 - 1))

    def test_first_oos_close_entry_is_allowed_but_earns_only_next_quote(self):
        dates = pd.date_range("2010-01-01", periods=8)
        base = fixed_execute(np.arange(100., 108.), [1, 1, 1, -1, -1, 0, 1, 1])
        fresh = fresh_start(base, base, dates, dates[4])
        self.assertEqual(fresh["executed"][4], -1)
        self.assertEqual(fresh["held"][4], 0)
        self.assertEqual(fresh["gross_return"][4], 0)
        self.assertAlmostEqual(fresh["gross_return"][5], -(105 / 104 - 1))
        np.testing.assert_array_equal(fresh["target"][:-1], fresh["executed"][1:])

    def test_schedule_delay_is_one_quote_shift_without_return_recomputation(self):
        delayed = delayed_schedule(self.run)
        np.testing.assert_array_equal(delayed["target"], [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(delayed["executed"], [0, 0, 1, 1, 0])
        np.testing.assert_array_equal(delayed["held"], [0, 0, 0, 1, 1])
        np.testing.assert_allclose(delayed["gross_return"], [0, 0, 0, -.1, .1])
        np.testing.assert_array_equal(delayed["spot_return"], self.run["spot_return"])


class InSampleSelectionTests(unittest.TestCase):
    def test_inclusive_hard_thresholds_pass_without_requiring_quality_in_each_block(self):
        frame = candidate_metrics()
        frame.loc[frame.split.ne("IS"), ["win_rate", "payoff_ratio", "closed_n"]] = [np.nan, np.nan, 1]
        result = screen_candidates(frame).iloc[0]
        self.assertTrue(result["pass"])
        self.assertTrue(result.frozen)
        self.assertAlmostEqual(result.retention, .5)

    def test_each_hard_gate_is_necessary(self):
        for column, value, flag in (
            ("gross_annualized_simple_pct", .999, "positive_and_annualized_pass"),
            ("gross_simple_return_pct", 24.99, "return_improvement_pass"),
            ("win_rate", .5199, "win_rate_pass"),
            ("payoff_ratio", 1.099, "payoff_pass"),
            ("gross_max_drawdown_pp", -12.001, "drawdown_pass"),
            ("closed_n", 49, "sample_pass"),
        ):
            with self.subTest(column=column):
                frame = candidate_metrics()
                frame.loc[frame.split.eq("IS") & frame.variant.eq("FILTER"), column] = value
                result = screen_candidates(frame).iloc[0]
                self.assertFalse(result[flag])
                self.assertFalse(result["pass"])
                self.assertFalse(result.frozen)
        frame = candidate_metrics()
        frame.loc[frame.split.eq("IS") & frame.variant.eq("BASE"), "closed_n"] = 200
        self.assertFalse(screen_candidates(frame).iloc[0].sample_pass)

    def test_nan_or_infinite_payoff_cannot_pass(self):
        for variant, value in (("BASE", 0.), ("BASE", np.nan), ("BASE", np.inf),
                               ("FILTER", np.nan), ("FILTER", np.inf)):
            with self.subTest(variant=variant, value=value):
                frame = candidate_metrics()
                frame.loc[frame.split.eq("IS") & frame.variant.eq(variant), "payoff_ratio"] = value
                self.assertFalse(screen_candidates(frame).iloc[0].payoff_pass)

    def test_return_and_drawdown_limits_do_not_replace_ratio_improvement(self):
        frame = candidate_metrics()
        in_sample = frame.split.eq("IS")
        frame.loc[in_sample & frame.variant.eq("BASE"), "gross_max_drawdown_pp"] = -2.
        frame.loc[in_sample & frame.variant.eq("FILTER"), "gross_max_drawdown_pp"] = -4.
        result = screen_candidates(frame).iloc[0]
        self.assertTrue(result.return_improvement_pass)
        self.assertTrue(result.drawdown_pass)
        self.assertFalse(result.return_to_drawdown_pass)
        self.assertFalse(result["pass"])
        self.assertFalse(result.frozen)

    def test_blocks_require_two_positive_two_improved_and_last_both(self):
        for updates, expected_flag in (
            ({"IS_LATE": -1.}, "block_positive_pass"),
            ({"IS_EARLY": 0.}, "block_positive_pass"),
            ({"IS_EARLY": 4., "IS_MIDDLE": -3.}, "block_improvement_pass"),
            ({"IS_LATE": 1.}, "block_improvement_pass"),
        ):
            with self.subTest(updates=updates):
                frame = candidate_metrics()
                for split, value in updates.items():
                    frame.loc[frame.split.eq(split) & frame.variant.eq("FILTER"), "gross_simple_return_pct"] = value
                result = screen_candidates(frame).iloc[0]
                self.assertFalse(result[expected_flag])
                self.assertFalse(result["pass"])

    def test_oos_rows_are_rejected_before_selection(self):
        frame = candidate_metrics()
        extra = frame.iloc[:1].copy()
        extra["split"] = "OOS"
        extra["gross_simple_return_pct"] = 1e9
        with self.assertRaisesRegex(ValueError, "OOS"):
            screen_candidates(pd.concat([frame, extra], ignore_index=True))

    def test_freezes_one_new_candidate_per_account_with_deterministic_ties(self):
        frames = []
        for identifier, gross, payoff, win, is_new, pair in (
            ("old_high_return", 90., 2., .7, False, "EUR"),
            ("less_return", 25., 2., .7, True, "EUR"),
            ("less_payoff", 26., 1.1, .7, True, "EUR"),
            ("less_win", 26., 1.2, .52, True, "EUR"),
            ("z_tie", 26., 1.2, .53, True, "EUR"),
            ("a_tie", 26., 1.2, .53, True, "EUR"),
            ("other_account", 25., 1.1, .52, True, "JPY"),
        ):
            frame = candidate_metrics(identifier=identifier, pair=pair)
            mask = frame.split.eq("IS") & frame.variant.eq("FILTER")
            frame.loc[mask, ["gross_simple_return_pct", "payoff_ratio", "win_rate"]] = [gross, payoff, win]
            frame["is_new"] = is_new
            frames.append(frame)
        result = screen_candidates(pd.concat(frames, ignore_index=True))
        self.assertTrue(result["pass"].all())
        self.assertEqual(set(result.loc[result.frozen, "candidate_id"]), {"a_tie", "other_account"})
        self.assertEqual(len(result), 7)

    def test_zero_cost_research_does_not_use_cost_sensitivity_as_a_gate(self):
        frame = candidate_metrics()
        frame.loc[frame.variant.eq("FILTER"), "net_simple_return_pct"] = -999.
        result = screen_candidates(frame).iloc[0]
        self.assertNotIn("cost_sensitivity_pass", result.index)
        self.assertTrue(result["pass"])


if __name__ == "__main__":
    unittest.main()
