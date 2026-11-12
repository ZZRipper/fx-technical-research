"""New quality policy boundaries and immutable saved IS-reference regression.

Only saved v3/v4 IS metrics and the pre-existing discussion review are read.
No strategy, price data, or OOS result is loaded by this test module.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from fx_filter_evaluation import IS_BLOCKS, OOS_SCREEN, SCREEN, TOL, screen_candidates
from fx_tradeoff_evaluation import (
    ACTIVE_GATES, RETAINED_GATES, TRADEOFF_OOS_SCREEN, TRADEOFF_SCREEN,
    add_trade_quality, assess_tradeoff_oos, screen_tradeoff_candidates,
)


def trade_stats(wins=40, losses=60, mean_win=1.8, mean_loss=1., flat=0):
    count = wins + losses + flat
    profit = wins * mean_win if wins else 0.
    loss = losses * mean_loss if losses else 0.
    return {'closed_n': count, 'win_n': wins, 'loss_n': losses, 'flat_n': flat,
            'mean_win_pct': mean_win if wins else np.nan,
            'mean_loss_abs_pct': mean_loss if losses else np.nan,
            'expectancy_pct': (profit - loss) / count if count else np.nan,
            'win_rate': wins / count if count else np.nan,
            'payoff_ratio': mean_win / mean_loss if wins and losses else np.nan}


def is_metrics(candidate='candidate', pair='AUD', strategy='TSMOM_63', delta=15., source='holding_filter_v3',
               base_quality=None, filter_quality=None):
    base_quality = trade_stats() if base_quality is None else base_quality
    filter_quality = trade_stats(60, 40, 1.4, 1.) if filter_quality is None else filter_quality
    rows = []
    for split in ['IS', *[name for name, _, _ in IS_BLOCKS]]:
        for variant, stats in [('BASE', base_quality), ('FILTER', filter_quality)]:
            total = (20. if split == 'IS' else 2.) + (delta if split == 'IS' else 1.) * (variant == 'FILTER')
            rows.append({'candidate_id': candidate, 'pair': pair, 'strategy': strategy,
                         'filter_id': 'FIXED_' + candidate, 'filter_name': 'Fixed ' + candidate,
                         'is_new': True, 'source_campaign': source, 'split': split, 'variant': variant,
                         'gross_simple_return_pct': total, 'net_simple_return_pct': total,
                         'gross_annualized_simple_pct': 2., 'gross_max_drawdown_pp': -10., **stats})
    return pd.DataFrame(rows)


def oos_records(filter_closed=20):
    base = {'candidate_id': 'x', 'pair': 'EUR', 'strategy': 'BB_MR_20', 'filter_id': 'FIXED',
            'source_campaign': 'holding_filter_v3', 'is_new': True, 'split': 'OOS', 'variant': 'BASE',
            'net_simple_return_pct': 4., 'net_annualized_simple_pct': 1., 'net_max_drawdown_pp': -15.,
            **trade_stats(12, 8, 1., 1.)}
    filtered = {**base, 'variant': 'FILTER', 'freeze_rank': 1,
                **trade_stats(10, filter_closed - 10, 2., 1.)}
    years = pd.DataFrame({'candidate_id': ['x'] * 3, 'variant': ['FILTER'] * 3,
                          'split': ['OOS_YEAR_1', 'OOS_YEAR_2', 'OOS_YEAR_3'],
                          'net_simple_return_pct': [2., 3., -1.]})
    return base, filtered, years


class TradeoffQualityTests(unittest.TestCase):
    def test_flat_cohort_arithmetic_zero_sides_and_idempotence(self):
        original = pd.DataFrame([trade_stats(2, 1, 3., 2., flat=2),
                                 trade_stats(0, 3, flat=2), trade_stats(3, 0), trade_stats(0, 0, flat=3)],
                                index=[9, 4, 8, 1])
        before = original.copy(deep=True)
        result = add_trade_quality(original)
        self.assertAlmostEqual(result.loc[9, 'expectancy_pct'], .8)
        self.assertEqual(result.loc[9, 'gross_profit_factor'], 3.)
        self.assertEqual(result.loc[4, 'gross_profit_factor'], 0.)
        self.assertTrue(result.loc[4, 'trade_quality_finite'])
        self.assertTrue(np.isnan(result.loc[8, 'gross_profit_factor']))
        self.assertTrue(np.isnan(result.loc[1, 'gross_profit_factor']))
        pd.testing.assert_frame_equal(result, add_trade_quality(result))
        pd.testing.assert_frame_equal(original, before)

    def test_nonfinite_quality_and_overflow_never_become_infinite_pf(self):
        bad_mean = trade_stats()
        bad_mean['mean_win_pct'] = np.inf
        overflow = trade_stats(1, 1, 1e308, 1e-308)
        result = add_trade_quality(pd.DataFrame([bad_mean, overflow]))
        self.assertTrue(result.gross_profit_factor.isna().all())
        self.assertFalse(result.trade_quality_finite.any())
        for change in ({'closed_n': 101}, {'win_n': .5}, {'loss_n': -1}, {'win_n': True},
                       {'expectancy_pct': 3.}, {'mean_win_pct': 0.}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                add_trade_quality(pd.DataFrame([{**trade_stats(), **change}]))

    def test_strict_comparison_has_no_selection_epsilon(self):
        base = trade_stats(50, 50, 1., 1.)
        tiny_better = trade_stats(50, 50, 1. + 2e-12, 1.)
        result = screen_tradeoff_candidates(is_metrics(base_quality=base, filter_quality=tiny_better)).iloc[0]
        self.assertTrue(0 < result.expectancy_improvement_pp_per_trade < TOL)
        self.assertTrue(0 < result.profit_factor_improvement < TOL)
        self.assertTrue(result['pass'])
        equal = screen_tradeoff_candidates(is_metrics(base_quality=base, filter_quality=base)).iloc[0]
        self.assertFalse(equal.expectancy_pass)
        self.assertFalse(equal.profit_factor_pass)
        self.assertFalse(equal['pass'])

    def test_both_quality_measures_required(self):
        base = trade_stats(50, 50, 2., 1.)
        for filtered, expectancy_pass, factor_pass in [
                (trade_stats(50, 50, 4., 2.5), True, False),
                (trade_stats(50, 50, .6, .2), False, True)]:
            with self.subTest(filtered=filtered):
                result = screen_tradeoff_candidates(is_metrics(base_quality=base, filter_quality=filtered)).iloc[0]
                self.assertEqual(result.expectancy_pass, expectancy_pass)
                self.assertEqual(result.profit_factor_pass, factor_pass)
                self.assertFalse(result['pass'])

    def test_old_gates_constants_input_and_rank_are_preserved(self):
        original_screen, original_oos = deepcopy(SCREEN), deepcopy(OOS_SCREEN)
        frame = pd.concat([is_metrics('b', delta=15., source='holding_filter_v3'),
                           is_metrics('a', delta=15., source='timecap_filter_v4'),
                           is_metrics('z', pair='CAD', delta=16.)], ignore_index=True)
        before = frame.copy(deep=True)
        old = screen_candidates(frame)
        new = screen_tradeoff_candidates(frame)
        self.assertEqual(new.candidate_id.tolist(), old.candidate_id.tolist())
        self.assertEqual(new.loc[new.frozen, 'candidate_id'].tolist(), ['z', 'a'])
        self.assertEqual(new.loc[new.frozen, 'freeze_rank'].tolist(), [1., 2.])
        self.assertEqual(new.set_index('candidate_id').loc['a', 'source_campaign'], 'timecap_filter_v4')
        self.assertEqual(new.set_index('candidate_id').loc['a', 'filter_name'], 'Fixed a')
        self.assertFalse(new.original_standard_pass.any())
        self.assertFalse(new.original_payoff_pass.any())
        self.assertTrue(new['pass'].all())
        for gate in RETAINED_GATES:
            np.testing.assert_array_equal(new[gate], old[gate])
        altered = frame.copy()
        altered.loc[(altered.candidate_id == 'z') & (altered.variant == 'FILTER'), 'gross_annualized_simple_pct'] = .5
        failure = screen_tradeoff_candidates(altered).set_index('candidate_id').loc['z']
        self.assertFalse(failure['pass'])
        self.assertIn('positive_and_annualized_pass', failure.failed_criteria)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(screen_candidates(frame), old)
        self.assertEqual(SCREEN, original_screen)
        self.assertEqual(OOS_SCREEN, original_oos)
        self.assertEqual(TRADEOFF_SCREEN['rank'], SCREEN['rank'])
        self.assertIsNot(TRADEOFF_SCREEN['rank'], SCREEN['rank'])
        self.assertEqual(TRADEOFF_SCREEN['quality_comparison_tolerance'], 0.)
        self.assertEqual(TRADEOFF_OOS_SCREEN['min_quality_closed_trades'], 20)

    def test_is_rejects_oos_before_quality_parse_and_incomplete_identity(self):
        frame = is_metrics()
        invalid = frame.copy()
        invalid['expectancy_pct'] = invalid.expectancy_pct.astype(object)
        invalid.loc[0, ['split', 'expectancy_pct']] = ['OOS', 'must_not_parse']
        with self.assertRaisesRegex(ValueError, 'cannot consume OOS'):
            screen_tradeoff_candidates(invalid)
        variants = [frame.iloc[:-1], pd.concat([frame, frame.iloc[[0]]], ignore_index=True)]
        mismatch = frame.copy()
        mismatch.loc[0, 'source_campaign'] = 'different'
        variants.append(mismatch)
        for item in variants:
            with self.assertRaises(ValueError):
                screen_tradeoff_candidates(item)

    def test_oos_curve_original_quality_and_new_sample_boundary_are_separate(self):
        base, filtered, years = oos_records()
        old_base, old_filtered, old_years = deepcopy(base), deepcopy(filtered), years.copy(deep=True)
        result = assess_tradeoff_oos(base, filtered, years)
        self.assertTrue(result['oos_curve_pass'])
        self.assertTrue(result['oos_quality_confirmed'])
        self.assertFalse(result['original_oos_quality_confirmed'])  # Win rate fell, but E/PF improved.
        self.assertEqual(result['failed_criteria'], '')
        self.assertEqual(result['quality_failed_criteria'], '')
        self.assertEqual(result['source_campaign'], 'holding_filter_v3')
        below = assess_tradeoff_oos(*oos_records(19))
        self.assertTrue(below['oos_curve_pass'])  # Curve sample threshold remains 10.
        self.assertFalse(below['oos_quality_confirmed'])
        self.assertEqual(below['quality_failed_criteria'], 'quality_sample_pass')
        filtered['net_simple_return_pct'] = base['net_simple_return_pct'] - 2*TOL
        curve_failure = assess_tradeoff_oos(base, filtered, years)
        self.assertFalse(curve_failure['oos_curve_pass'])
        self.assertTrue(curve_failure['oos_quality_confirmed'])
        self.assertEqual(curve_failure['failed_criteria'], 'baseline_improvement_pass')
        self.assertEqual(base, old_base)
        pd.testing.assert_frame_equal(years, old_years)
        self.assertEqual(old_filtered['net_simple_return_pct'], 4.)

    def test_oos_nonfinite_quality_no_losses_and_wrong_periods(self):
        base, filtered, years = oos_records()
        filtered.update(trade_stats(20, 0))
        result = assess_tradeoff_oos(base, filtered, years)
        self.assertTrue(result['oos_curve_pass'])
        self.assertFalse(result['oos_quality_confirmed'])
        self.assertTrue(np.isnan(result['oos_gross_profit_factor']))
        self.assertIn('quality_finiteness_pass', result['quality_failed_criteria'])
        self.assertIn('profit_factor_pass', result['quality_failed_criteria'])
        for bad_base, bad_years in [({**base, 'split': 'IS'}, years),
                                  (base, years.iloc[:2]),
                                  (base, years.assign(split='OOS_YEAR_1')),
                                  (base, years.assign(variant='BASE'))]:
            with self.assertRaises(ValueError):
                assess_tradeoff_oos(bad_base, filtered, bad_years)

    def test_saved_492_is_candidates_exactly_match_60_and_21_review(self):
        frames = []
        for campaign in ['holding_filter_v3', 'timecap_filter_v4']:
            frame = pd.read_csv(ROOT / 'reproduced' / campaign / 'is_all_metrics.csv')
            frames.append(frame.assign(source_campaign=campaign))
        result = screen_tradeoff_candidates(pd.concat(frames, ignore_index=True))
        review = pd.read_csv(ROOT / 'reproduced/holding_quality_tradeoff_review/review.csv')
        expected_pass = set(review.loc[review.diagnostic_alternative_criteria_satisfied, 'candidate_id'])
        self.assertEqual(len(result), 492)
        self.assertEqual(int(result['pass'].sum()), 60)
        self.assertEqual(set(result.loc[result['pass'], 'candidate_id']), expected_pass)
        wanted = review.loc[review.diagnostic_one_per_account_suggested].sort_values('diagnostic_suggestion_order').candidate_id.tolist()
        self.assertEqual(len(wanted), 21)
        self.assertEqual(result.loc[result.frozen].sort_values('freeze_rank').candidate_id.tolist(), wanted)
        joined = result.set_index('candidate_id').join(review.set_index('candidate_id'), rsuffix='_review')
        for gate in RETAINED_GATES:
            np.testing.assert_array_equal(joined[gate], joined[gate + '_review'])
        for column in ['base_is_expectancy_pct', 'filter_is_expectancy_pct',
                       'base_is_gross_profit_factor', 'filter_is_gross_profit_factor']:
            np.testing.assert_allclose(joined[column], joined[column + '_review'], atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
