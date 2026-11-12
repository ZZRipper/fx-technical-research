"""Hand-computed attribution and predeclared chronological screening tests.

These tests use synthetic paths only. They never open research outcomes.
"""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_state_filter_research as research
from fx_baselines import fixed_execute


class EpisodeEffectTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2020-01-01", periods=16)
        spot = np.zeros(16)
        spot[[2, 3, 5, 6, 8, 9, 11, 12, 14, 15]] = [
            -.01, -.02, -.03, -.04, .05, .06, .02, .02, -.04, -.05,
        ]
        prices = 100. * np.cumprod(1. + spot)
        targets = np.array([1, 1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, 1, 1, 1, 1])
        self.base = fixed_execute(prices, targets)
        self.filtered = fixed_execute(prices, np.zeros(16, dtype=int))

    def test_top_three_uses_original_episode_effects_and_keeps_harmful_changes(self):
        result = research.affected(self.base, self.filtered, self.dates, self.dates[0], self.dates[-1])
        # Avoided episode losses: +3, +7, +11, -4, +9 percentage points.
        self.assertAlmostEqual(result["delta_pp"], 26.)
        self.assertAlmostEqual(result["delta_without_best3_pp"], -1.)
        self.assertEqual(result["changed_episode_n"], 5)
        self.assertEqual(result["changed_closed_n"], 4)

    def test_carry_in_and_open_tail_have_window_pnl_but_not_closed_support(self):
        result = research.affected(self.base, self.filtered, self.dates, self.dates[3], self.dates[14])
        # Actual window effects: carry-in +2, then +7, +11, -4, open tail +4.
        self.assertAlmostEqual(result["delta_pp"], 20.)
        self.assertAlmostEqual(result["delta_without_best3_pp"], -2.)
        self.assertEqual(result["changed_episode_n"], 5)
        self.assertEqual(result["changed_closed_n"], 3)

    def test_exit_day_return_belongs_to_old_episode_and_entry_close_earns_none(self):
        exit_day = research.affected(self.base, self.filtered, self.dates, self.dates[3], self.dates[3])
        self.assertAlmostEqual(exit_day["delta_pp"], 2.)
        self.assertAlmostEqual(exit_day["delta_without_best3_pp"], 0.)
        self.assertEqual(exit_day["changed_episode_n"], 1)
        self.assertEqual(exit_day["changed_closed_n"], 0)
        entry_day = research.affected(self.base, self.filtered, self.dates, self.dates[4], self.dates[4])
        self.assertAlmostEqual(entry_day["delta_pp"], 0.)
        self.assertEqual(entry_day["changed_episode_n"], 0)

    def test_reversal_close_is_attributed_once_to_the_departing_episode(self):
        dates = pd.date_range("2020-01-01", periods=6)
        prices = 100. * np.cumprod(1. + np.array([0., 0., -.01, -.02, .03, .04]))
        base = fixed_execute(prices, [1, 1, -1, -1, 0, 0])
        filtered = fixed_execute(prices, [0, 0, -1, -1, 0, 0])
        result = research.affected(base, filtered, dates, dates[0], dates[-1])
        self.assertAlmostEqual(result["delta_pp"], 3.)
        self.assertAlmostEqual(result["delta_without_best3_pp"], 0.)
        self.assertEqual(result["changed_episode_n"], 1)
        self.assertEqual(result["changed_closed_n"], 1)

    def test_unchanged_path_has_no_effect_support(self):
        result = research.affected(self.base, self.base, self.dates, self.dates[0], self.dates[-1])
        self.assertEqual(result, {"delta_pp": 0., "changed_closed_n": 0,
                                  "changed_episode_n": 0, "delta_without_best3_pp": 0.})


class ChronologicalGateTests(unittest.TestCase):
    def metrics(self, phase="DISCOVERY"):
        base = {"split": phase, "variant": "BASE", "gross_simple_return_pct": 20.,
                "gross_annualized_simple_pct": 2., "gross_max_drawdown_pp": -10.,
                "win_rate": .5, "payoff_ratio": 1., "closed_n": 60, "win_n": 30, "loss_n": 30}
        filtered = {**base, "variant": "FILTER", "gross_simple_return_pct": 25.,
                    "gross_annualized_simple_pct": 2.5, "gross_max_drawdown_pp": -12.,
                    "win_rate": .52, "payoff_ratio": 1.1, "closed_n": 30, "win_n": 16, "loss_n": 14}
        return pd.DataFrame([base, filtered])

    def test_discovery_exact_quality_retention_and_effect_thresholds(self):
        flags = research.phase_flags(self.metrics(), "DISCOVERY", {"changed_closed_n": 20}, True)
        self.assertTrue(all(flags.values()), flags)
        cases = [("gross_annualized_simple_pct", .999, "annualized"),
                 ("gross_simple_return_pct", 24.999, "return_improvement"),
                 ("win_rate", .519, "quality"), ("payoff_ratio", 1.099, "quality"),
                 ("closed_n", 29, "sample"), ("gross_max_drawdown_pp", -12.001, "drawdown")]
        for column, value, flag in cases:
            with self.subTest(column=column):
                metrics = self.metrics()
                metrics.loc[metrics.variant.eq("FILTER"), column] = value
                self.assertFalse(research.phase_flags(metrics, "DISCOVERY", {"changed_closed_n": 20}, True)[flag])
        self.assertFalse(research.phase_flags(self.metrics(), "DISCOVERY", {"changed_closed_n": 19}, True)["affected_sample"])

    def test_drawdown_deterioration_limit_does_not_replace_ratio_improvement(self):
        metrics = self.metrics()
        metrics.loc[metrics.variant.eq("BASE"), "gross_max_drawdown_pp"] = -2.
        metrics.loc[metrics.variant.eq("FILTER"), "gross_max_drawdown_pp"] = -4.
        flags = research.phase_flags(metrics, "DISCOVERY", {"changed_closed_n": 20}, True)
        self.assertTrue(flags["return_improvement"])
        self.assertTrue(flags["drawdown"])
        self.assertFalse(flags["return_drawdown"])

    def test_validation_and_confirmation_keep_quality_and_actual_effect_support(self):
        for phase in ("VALIDATION", "CONFIRMATION"):
            with self.subTest(phase=phase):
                metrics = self.metrics(phase)
                metrics.loc[0, "closed_n"] = 40
                metrics.loc[1, ["gross_simple_return_pct", "gross_annualized_simple_pct",
                                "gross_max_drawdown_pp", "win_rate", "payoff_ratio",
                                "closed_n", "win_n", "loss_n"]] = [21., 1., -10., .5, 1., 20, 10, 10]
                flags = research.phase_flags(metrics, phase, {"changed_closed_n": 5})
                self.assertTrue(all(flags.values()), flags)
                self.assertFalse(research.phase_flags(metrics, phase, {"changed_closed_n": 4})["affected_sample"])
                for column in ("win_n", "loss_n"):
                    small = metrics.copy()
                    small.loc[1, column] = 4
                    self.assertFalse(research.phase_flags(small, phase, {"changed_closed_n": 5})["outcome_sample"])
                worsened = metrics.copy()
                worsened.loc[1, "win_rate"] = .499
                self.assertFalse(research.phase_flags(worsened, phase, {"changed_closed_n": 5})["quality"])
                missing = metrics.copy()
                missing.loc[1, "payoff_ratio"] = np.nan
                self.assertFalse(research.phase_flags(missing, phase, {"changed_closed_n": 5})["quality"])

    def test_proposals_ignore_later_outcomes_and_require_both_discovery_blocks(self):
        fields = ["pair", "strategy", "feature", "state", "block", "closed_n",
                  "return_sum_pct", "return_without_worst_trade_pct"]
        template = dict(pair="JPY", strategy="RSI_MR_7", feature="vol_ratio_20_63", state=3,
                        closed_n=10, return_sum_pct=-3., return_without_worst_trade_pct=-1.)
        rows = [{**template, "block": block} for block in ("DISCOVERY_1", "DISCOVERY_2", "DEVELOPMENT")]
        # Later blocks are not even parsed as outcomes: deliberately invalid numeric text.
        rows.extend({**template, "block": block, "closed_n": "UNREAD", "return_sum_pct": "UNREAD"}
                    for block in ("VALIDATION", "CONFIRMATION", "IS"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_states.csv"
            def write():
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            self.assertEqual(len(research.proposal_rows(path)), 1)
            rows[0]["return_sum_pct"] = 0.
            write()
            self.assertEqual(research.proposal_rows(path), [])
            rows[0]["return_sum_pct"] = -3.
            rows[2]["return_without_worst_trade_pct"] = 0.
            write()
            self.assertEqual(research.proposal_rows(path), [])


if __name__ == "__main__":
    unittest.main()
