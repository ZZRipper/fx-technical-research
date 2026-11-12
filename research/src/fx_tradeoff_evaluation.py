"""Authorized expectancy/profit-factor quality tradeoff, separate from old gates.

All original curve, sample, block and ranking rules remain unchanged. New
quality comparisons are finite and strictly greater, with no selection epsilon.
TOL is used only for arithmetic validation and the unchanged original gates.
This module consumes supplied metrics; it never loads data or executes filters.
"""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fx_filter_evaluation import IS_BLOCKS, OOS_SCREEN, SCREEN, TOL, screen_candidates


RETAINED_GATES = (
    'positive_and_annualized_pass', 'return_improvement_pass', 'drawdown_pass',
    'return_to_drawdown_pass', 'sample_pass', 'block_positive_pass', 'block_improvement_pass',
)
ACTIVE_GATES = (
    'positive_and_annualized_pass', 'return_improvement_pass', 'expectancy_pass',
    'profit_factor_pass', 'drawdown_pass', 'return_to_drawdown_pass', 'sample_pass',
    'block_positive_pass', 'block_improvement_pass',
)
QUALITY_DEFINITION = (
    'Eligible closed-trade expectancy_pct and gross profit factor must each be '
    'finite for BASE and FILTER and strictly improve (FILTER > BASE, no epsilon). '
    'Gross PF = win_n*mean_win_pct/(loss_n*mean_loss_abs_pct); zero-count sides '
    'contribute zero, and zero gross loss or nonfinite arithmetic is undefined.'
)
TRADEOFF_SCREEN = deepcopy(SCREEN)
for _name in ('min_win_rate_improvement_pp', 'min_payoff_relative_improvement'):
    TRADEOFF_SCREEN.pop(_name)
TRADEOFF_SCREEN.update(
    quality_definition=QUALITY_DEFINITION,
    expectancy_strict_improvement=True,
    gross_profit_factor_strict_improvement=True,
    quality_finiteness_required=True,
    quality_comparison_tolerance=0.0,
    replaced_original_quality_gates=['win_rate_pass', 'payoff_pass'],
    original_quality_gates_diagnostic_only=True,
)
TRADEOFF_OOS_SCREEN = deepcopy(OOS_SCREEN)
TRADEOFF_OOS_SCREEN.update(
    quality_confirmation='At least 20 eligible closed FILTER trades, plus ' + QUALITY_DEFINITION,
    min_quality_closed_trades=20,
    expectancy_strict_improvement=True,
    gross_profit_factor_strict_improvement=True,
    quality_finiteness_required=True,
    quality_comparison_tolerance=0.0,
    original_win_rate_payoff_quality_diagnostic_only=True,
)
IDENTITY_COLUMNS = ('candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'source_campaign')
QUALITY_REQUIRED = ('closed_n', 'win_n', 'loss_n', 'flat_n', 'mean_win_pct', 'mean_loss_abs_pct', 'expectancy_pct')


def _quality_row(row):
    missing = set(QUALITY_REQUIRED).difference(row.keys())
    if missing:
        raise ValueError('Missing closed-trade quality fields: ' + ', '.join(sorted(missing)))
    counts = {}
    for name in ('closed_n', 'win_n', 'loss_n', 'flat_n'):
        value = row[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('Trade counts must be nonnegative integers, not bool.')
        number = float(value)
        if not np.isfinite(number) or number < 0 or number != np.floor(number):
            raise ValueError('Trade counts must be finite nonnegative integers.')
        counts[name] = int(number)
    if counts['win_n'] + counts['loss_n'] + counts['flat_n'] != counts['closed_n']:
        raise ValueError('Winning, losing and flat counts must partition closed_n.')

    def contribution(count, mean):
        if count == 0:
            return 0.0
        mean = float(mean)
        if not np.isfinite(mean):
            return np.nan
        if mean <= 0:
            raise ValueError('A nonempty winning/losing cohort needs a positive absolute mean.')
        with np.errstate(over='ignore', invalid='ignore'):
            value = float(np.float64(count) * mean)
        return value if np.isfinite(value) else np.nan

    profit = contribution(counts['win_n'], row['mean_win_pct'])
    loss = contribution(counts['loss_n'], row['mean_loss_abs_pct'])
    expectancy = float(row['expectancy_pct'])
    if counts['closed_n'] == 0 and np.isfinite(expectancy):
        raise ValueError('Empty closed-trade expectancy must be undefined.')
    if counts['closed_n'] and np.isfinite([profit, loss, expectancy]).all():
        # Flat trades can have individual returns inside the existing +/- TOL band.
        reconstructed = (profit - loss) / counts['closed_n']
        if not np.isclose(expectancy, reconstructed, atol=TOL, rtol=0):
            raise ValueError('Saved expectancy disagrees with the closed-trade arithmetic.')
    factor = np.nan
    if np.isfinite([profit, loss]).all() and loss > 0:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            value = float(np.float64(profit) / loss)
        if np.isfinite(value):
            factor = value
    return {'gross_profit_sum_pct': profit, 'gross_loss_abs_sum_pct': loss,
            'gross_profit_factor': factor,
            'trade_quality_finite': bool(np.isfinite([expectancy, factor]).all())}


def add_trade_quality(frame):
    """Return a copy with reproducible PF fields; retain saved expectancy unchanged.

    Flat trades remain in closed_n. Missing wins contribute zero; missing losses
    produce undefined PF. Nonfinite inputs never become an infinite accepted PF.
    Invalid counts or inconsistent finite expectancy arithmetic raise ValueError.
    Existing derived columns are recomputed, making repeated calls idempotent.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Trade quality requires a metric DataFrame.')
    missing = set(QUALITY_REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError('Missing closed-trade quality fields: ' + ', '.join(sorted(missing)))
    result = frame.copy(deep=True)
    rows = [_quality_row(row) for row in result.to_dict('records')]
    for name in ('gross_profit_sum_pct', 'gross_loss_abs_sum_pct', 'gross_profit_factor', 'trade_quality_finite'):
        result[name] = pd.Series([row[name] for row in rows], index=result.index,
                                 dtype=bool if name == 'trade_quality_finite' else float)
    return result


def _comparisons(base, filtered):
    be, fe = float(base['expectancy_pct']), float(filtered['expectancy_pct'])
    bp, fp = float(base['gross_profit_factor']), float(filtered['gross_profit_factor'])
    expectancy_finite = bool(np.isfinite([be, fe]).all())
    factor_finite = bool(np.isfinite([bp, fp]).all())
    return {
        'quality_metrics_finite': expectancy_finite and factor_finite,
        'expectancy_improvement_pp_per_trade': fe - be if expectancy_finite else np.nan,
        'profit_factor_improvement': fp - bp if factor_finite else np.nan,
        'expectancy_pass': bool(expectancy_finite and fe > be),
        'profit_factor_pass': bool(factor_finite and fp > bp),
    }


def screen_tradeoff_candidates(metrics):
    """Apply the new quality pair to IS only, retaining all other original gates.

    Ranking remains return delta, payoff delta, win-rate delta, candidate ID.
    Original quality flags/pass/freeze are explicit diagnostics. Metadata such
    as source_campaign and filter_name is preserved from each candidate.
    """
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError('A nonempty complete IS metric table is required.')
    required = {'candidate_id', 'pair', 'strategy', 'filter_id', 'split', 'variant'}
    if not required.issubset(metrics.columns):
        raise ValueError('Missing IS identity, period or variant columns.')
    periods = {'IS', *[name for name, _, _ in IS_BLOCKS]}
    if not metrics.split.isin(periods).all():
        raise ValueError('IS quality selection cannot consume OOS or other outcome rows.')
    if not metrics.variant.isin(['BASE', 'FILTER']).all() or metrics.duplicated(['candidate_id', 'split', 'variant']).any():
        raise ValueError('IS metrics need unique BASE/FILTER candidate-period rows.')
    expected = {(split, variant) for split in periods for variant in ('BASE', 'FILTER')}
    metadata = {}
    for candidate_id, group in metrics.groupby('candidate_id', sort=False, dropna=False):
        if pd.isna(candidate_id) or set(group[['split', 'variant']].itertuples(index=False, name=None)) != expected:
            raise ValueError('Each IS candidate requires all four periods and both variants.')
        values = {}
        for name in IDENTITY_COLUMNS:
            if name in group:
                if group[name].nunique(dropna=False) != 1 or group[name].isna().any():
                    raise ValueError('Candidate identity must be constant and nonmissing: ' + name)
                values[name] = group[name].iloc[0]
        metadata[candidate_id] = values
    enriched = add_trade_quality(metrics)
    result = screen_candidates(metrics).rename(columns={
        'win_rate_pass': 'original_win_rate_pass', 'payoff_pass': 'original_payoff_pass',
        'pass': 'original_standard_pass', 'failed_criteria': 'original_failed_criteria',
        'freeze_rank': 'original_freeze_rank', 'frozen': 'original_frozen',
    })
    quality_rows = []
    for candidate_id in result.candidate_id:
        group = enriched.loc[enriched.candidate_id.eq(candidate_id) & enriched.split.eq('IS')].set_index('variant')
        base, filtered = group.loc['BASE'], group.loc['FILTER']
        quality_rows.append({
            **metadata[candidate_id], **_comparisons(base, filtered),
            'base_is_expectancy_pct': float(base.expectancy_pct),
            'filter_is_expectancy_pct': float(filtered.expectancy_pct),
            'base_is_gross_profit_factor': float(base.gross_profit_factor),
            'filter_is_gross_profit_factor': float(filtered.gross_profit_factor),
        })
    additions = pd.DataFrame(quality_rows).set_index('candidate_id')
    for name in additions.columns:
        result[name] = result.candidate_id.map(additions[name])
    result['pass'] = result[list(ACTIVE_GATES)].all(axis=1)
    result['failed_criteria'] = ['|'.join(name for name in ACTIVE_GATES if not row[name])
                                 for row in result.to_dict('records')]
    # screen_candidates already returned the unchanged original total ordering.
    result['freeze_rank'] = np.nan
    winners = result.loc[result['pass'] & result.is_new].drop_duplicates(['pair', 'strategy'])
    result.loc[winners.index, 'freeze_rank'] = np.arange(1, len(winners) + 1)
    result['frozen'] = result.freeze_rank.notna()
    return result


def assess_tradeoff_oos(base, filtered, anniversary):
    """Assess one frozen candidate using supplied OOS and FILTER anniversary rows.

    Curve flags/failed_criteria exactly retain OOS_SCREEN. The separate quality
    result requires >=20 closed trades and strict finite expectancy/PF gains.
    Original win-rate/payoff quality remains a diagnostic only. No IS selection
    or strategy evaluation is called. Identity is copied from the FILTER row.
    """
    if not isinstance(base, (Mapping, pd.Series)) or not isinstance(filtered, (Mapping, pd.Series)):
        raise TypeError('BASE and FILTER must be single metric records.')
    for row, variant in ((base, 'BASE'), (filtered, 'FILTER')):
        if ('split' in row and row['split'] != 'OOS') or ('variant' in row and row['variant'] != variant):
            raise ValueError('OOS assessment only accepts its BASE/FILTER OOS records.')
    for name in IDENTITY_COLUMNS:
        if name in base and name in filtered and base[name] != filtered[name]:
            raise ValueError('BASE/FILTER candidate identities differ: ' + name)
    if not isinstance(anniversary, pd.DataFrame) or len(anniversary) != 3:
        raise ValueError('Exactly three FILTER anniversary metric rows are required.')
    if 'variant' in anniversary and not anniversary.variant.eq('FILTER').all():
        raise ValueError('Anniversary rows must contain only FILTER.')
    for name in IDENTITY_COLUMNS:
        if name in anniversary and name in filtered and not anniversary[name].eq(filtered[name]).all():
            raise ValueError('Anniversary candidate identity differs: ' + name)
    if 'split' in anniversary and set(anniversary.split) != {'OOS_YEAR_1', 'OOS_YEAR_2', 'OOS_YEAR_3'}:
        raise ValueError('Anniversary periods must be the three unique OOS years.')
    if 'net_simple_return_pct' not in anniversary:
        raise ValueError('Anniversary net returns are required.')
    years = anniversary.net_simple_return_pct.to_numpy(float)
    if not np.isfinite(years).all():
        raise ValueError('Anniversary returns must be finite.')
    enriched = add_trade_quality(pd.DataFrame([dict(base), dict(filtered)]))
    b, f = enriched.iloc[0], enriched.iloc[1]
    if not np.isfinite([b.net_simple_return_pct, f.net_simple_return_pct,
                        f.net_annualized_simple_pct, f.net_max_drawdown_pp]).all():
        raise ValueError('OOS curve inputs must be finite.')
    comparison = _comparisons(b, f)
    delta = float(f.net_simple_return_pct - b.net_simple_return_pct)
    flags = {
        'annualized_pass': bool(f.net_annualized_simple_pct >= OOS_SCREEN['min_net_annualized_simple_pct'] - TOL),
        'baseline_improvement_pass': bool(delta >= OOS_SCREEN['min_net_return_improvement_pp'] - TOL),
        'drawdown_pass': bool(f.net_max_drawdown_pp >= OOS_SCREEN['worst_net_drawdown_pp'] - TOL),
        'positive_years_pass': bool(np.count_nonzero(years > TOL) >= OOS_SCREEN['min_positive_anniversary_years']),
        'sample_pass': bool(f.closed_n >= OOS_SCREEN['min_closed_trades']),
    }
    quality_flags = {
        'quality_sample_pass': bool(f.closed_n >= TRADEOFF_OOS_SCREEN['min_quality_closed_trades']),
        'quality_finiteness_pass': comparison['quality_metrics_finite'],
        'expectancy_pass': comparison['expectancy_pass'],
        'profit_factor_pass': comparison['profit_factor_pass'],
    }
    original_quality = bool(f.closed_n >= 20 and np.isfinite([f.win_rate, b.win_rate, f.payoff_ratio, b.payoff_ratio]).all()
                            and f.win_rate >= b.win_rate - TOL and f.payoff_ratio >= b.payoff_ratio - TOL)
    return {
        **{name: filtered[name] for name in (*IDENTITY_COLUMNS, 'freeze_rank') if name in filtered},
        'oos_return_pct': float(f.net_simple_return_pct), 'oos_base_return_pct': float(b.net_simple_return_pct),
        'oos_improvement_pp': delta, 'oos_annualized_simple_pct': float(f.net_annualized_simple_pct),
        'oos_max_drawdown_pp': float(f.net_max_drawdown_pp), 'oos_win_rate': float(f.win_rate),
        'oos_base_win_rate': float(b.win_rate), 'oos_payoff_ratio': float(f.payoff_ratio),
        'oos_base_payoff_ratio': float(b.payoff_ratio), 'oos_closed_n': int(f.closed_n),
        'oos_positive_years': int(np.count_nonzero(years > TOL)),
        'oos_expectancy_pct': float(f.expectancy_pct), 'oos_base_expectancy_pct': float(b.expectancy_pct),
        'oos_gross_profit_factor': float(f.gross_profit_factor), 'oos_base_gross_profit_factor': float(b.gross_profit_factor),
        'oos_expectancy_improvement_pp_per_trade': comparison['expectancy_improvement_pp_per_trade'],
        'oos_profit_factor_improvement': comparison['profit_factor_improvement'],
        **flags, **quality_flags,
        'oos_curve_pass': bool(all(flags.values())), 'oos_quality_confirmed': bool(all(quality_flags.values())),
        'original_oos_quality_confirmed': original_quality,
        'failed_criteria': '|'.join(name for name, value in flags.items() if not value),
        'quality_failed_criteria': '|'.join(name for name, value in quality_flags.items() if not value),
    }
