"""Reconstructed research entry points; original frozen source was not supplied."""

from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from fx_baselines import (
    CURRENCIES, CORE_RULES, RULES, STRATEGIES,
    canonical_prices_from_raw, original_exploration, event_osc_reversion,
    fixed_execute, reconstruct_fixed_baselines, annual_table, period_table,
)
from fx_economics import (
    read_native_features_from_raw, rebuild_economic_known,
    run_economic_filters, write_daily_accounts,
)
from fx_evidence import (
    verify_baselines, verify_known_features, verify_filter_metrics_and_paths,
    saved_notebook_table, print_saved_output,
)
from fx_legacy import reconstruct_v3, reconstruct_h10, verify_legacy
from fx_risk import run_vix_variants, conditional_daily, verify_vix_variants
from fx_analysis import episode_effects, episode_concentration, historical_screen, case_evidence, SCREEN_SPECIFICATION


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'FX_Research_Datasets' / 'data'


def verify_raw_files():
    """Verify supplied raw bytes and disclose missing historical artifacts."""
    manifest = json.loads((DATA / 'catalog/data_manifest.json').read_text())
    records = []
    for item in manifest['raw_files']:
        relative_path = Path(item['file']).relative_to('data')
        actual_path = DATA / relative_path
        if not actual_path.exists() and actual_path.name == 'FX_Features_Book(8).xlsx':
            actual_path = actual_path.with_name('FX_Features_Book.xlsx')
        available = actual_path.is_file()
        matches = available and hashlib.sha256(actual_path.read_bytes()).hexdigest() == item['sha256']
        records.append({'manifest_path': item['file'], 'actual_path': str(actual_path),
                        'available': available, 'sha256_matches': matches})
        if available and not matches:
            raise ValueError(f'Raw input hash mismatch: {actual_path}')
    return pd.DataFrame(records)


def window_diagnostics(runs, price_dates):
    """Compare fixed windows using continuous positions and calendar blocks."""
    groups = {'RSI_MR': ['RSI_MR_7', 'RSI_MR_14', 'RSI_MR_20'],
              'RSI_MOM': ['RSI_MOM_7', 'RSI_MOM_14', 'RSI_MOM_20'],
              'MA': ['MA_10_30', 'MA_20_60'], 'BB_MR': ['BB_MR_10', 'BB_MR_20'],
              'BB_BO': ['BB_BO_10', 'BB_BO_20'],
              'DONCHIAN': ['DONCHIAN_BO_10', 'DONCHIAN_BO_20'],
              'TSMOM': ['TSMOM_21', 'TSMOM_63']}
    records, voting = [], []
    for pair in CURRENCIES:
        dates = price_dates[pair]
        evaluation = dates >= '2005-01-03'
        periods = [(dates >= start) & (dates <= end) for start, end in
                   [('2005-01-03', '2014-12-31'), ('2015-01-01', '2019-12-31'),
                    ('2020-01-01', '2026-05-29')]]
        for family, names in groups.items():
            for shorter, longer in combinations(names, 2):
                first, second = runs[(pair, shorter)], runs[(pair, longer)]
                differences = [np.prod(1 + first['daily_ret'][mask]) -
                               np.prod(1 + second['daily_ret'][mask]) for mask in periods]
                records.append({'pair': pair, 'family': family,
                                'shorter_window_rule': shorter, 'longer_window_rule': longer,
                                'shorter_full_return_pct': 100 * (first['equity'][-1] - 1),
                                'longer_full_return_pct': 100 * (second['equity'][-1] - 1),
                                'position_disagreement_pct': 100 * np.mean(
                                    first['held'][evaluation] != second['held'][evaluation]),
                                'window_order_changes_across_periods': min(differences) < 0 < max(differences)})
        equal, strength = runs[(pair, 'VOTE_EQUAL_4')], runs[(pair, 'VOTE_STRENGTH_4')]
        voting.append({'pair': pair, 'equal_full_return_pct': 100 * (equal['equity'][-1] - 1),
                       'strength_full_return_pct': 100 * (strength['equity'][-1] - 1),
                       'strength_minus_equal_percentage_points': 100 * (strength['equity'][-1] - equal['equity'][-1]),
                       'equal_positive_periods': sum(np.prod(1 + equal['daily_ret'][mask]) > 1 for mask in periods),
                       'strength_positive_periods': sum(np.prod(1 + strength['daily_ret'][mask]) > 1 for mask in periods)})
    return pd.DataFrame(records), pd.DataFrame(voting)


def verify_original_metrics(metrics):
    reference = pd.read_csv(DATA / 'clean/original_exploration_metrics_rebuilt.csv')
    keys = ['pair', 'strategy', 'mode']
    merged = metrics.merge(reference, on=keys, suffixes=('_new', '_reference'), validate='one_to_one')
    records = [{'metric': 'account_coverage', 'accounts': len(merged),
                'max_abs_error': 0.0, 'pass': len(metrics) == len(reference) == len(merged) == 792}]
    for column in reference.columns.difference(keys):
        actual, expected = merged[column + '_new'], merged[column + '_reference']
        if pd.api.types.is_numeric_dtype(expected) and not pd.api.types.is_bool_dtype(expected):
            error = float((actual - expected).abs().max())
            matches = np.allclose(actual, expected, atol=1e-7, rtol=1e-9, equal_nan=True)
        else:
            matches = bool((actual.eq(expected) | (actual.isna() & expected.isna())).all())
            error = 0.0 if matches else np.nan
        records.append({'metric': column, 'accounts': len(merged), 'max_abs_error': error, 'pass': bool(matches)})
    return pd.DataFrame(records)


def currency_attributes(known):
    frame = known.assign(year=pd.to_datetime(known['date']).dt.year)
    return frame.groupby(['pair', 'year'], as_index=False).agg(
        median_known_spread_bp=('K_spread_bp', 'median'),
        known_spread_observations=('K_spread_bp', 'count'))


def reproduce_v8(output_dir, prices=None, known=None, baseline=None):
    """Recompute candidates and compare to independent retained HTML evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prices is None:
        prices, _ = canonical_prices_from_raw()
    if baseline is None:
        baseline = reconstruct_fixed_baselines(prices)
    _, baseline_trades, baseline_runs, price_dates, _ = baseline
    if known is None:
        known = rebuild_economic_known(read_native_features_from_raw(), prices.date)
    result = run_economic_filters(prices, known, baseline_runs, price_dates, baseline_trades)
    checks = verify_filter_metrics_and_paths(result['metrics'], result['runs'])
    tables = output_dir / 'tables'
    tables.mkdir(exist_ok=True)
    result['metrics'].to_csv(tables / 'account_metrics.csv', index=False)
    result['trades'].to_csv(tables / 'trade_ledger.csv.gz', index=False)
    result['episode_decisions'].to_csv(tables / 'episode_decisions.csv.gz', index=False)
    result['phase'].to_csv(tables / 'phase_metrics.csv', index=False)
    result['calendar'].to_csv(tables / 'calendar_period_pnl.csv', index=False)
    write_daily_accounts(result, tables / 'daily_accounts.csv.gz')
    effects = episode_effects(result)
    episode_concentration(result, effects).to_csv(tables / 'episode_concentration.csv', index=False)
    historical_screen(result).to_csv(tables / 'account_acceptance.csv', index=False)
    case_evidence(result, known, effects).to_csv(tables / 'case_evidence.csv.gz', index=False)
    (tables / 'reconstructed_screen_specification.json').write_text(json.dumps(SCREEN_SPECIFICATION, indent=2))
    checks.to_csv(output_dir / 'reconciliation.csv', index=False)
    return result['metrics'], checks, tables


def breadth_summary(metrics):
    keys = ['pair', 'base_strategy']
    baseline = metrics[metrics.variant.eq('BASE')]
    comparison = metrics[~metrics.variant.eq('BASE')].merge(baseline, on=keys, suffixes=('', '_base'))
    comparison = comparison[~comparison.variant.eq('COMMODITY_USD') | comparison.pair.isin(['AUD', 'CAD', 'NOK'])]
    records = []
    for variant, group in comparison.groupby('variant', sort=False):
        wins = group.win_rate > group.win_rate_base + 1e-12
        payoff = group.payoff_ratio > group.payoff_ratio_base + 1e-12
        returns = group.total_return_pct > group.total_return_pct_base + 1e-10
        drawdown = group.max_drawdown_pct >= group.max_drawdown_pct_base - 1e-10
        records.append({'variant': variant, 'accounts': len(group), 'win_rate_up': int(wins.sum()),
                        'payoff_up': int(payoff.sum()), 'both_up': int((wins & payoff).sum()),
                        'both_and_pnl_up': int((wins & payoff & returns).sum()),
                        'pnl_up': int(returns.sum()), 'dd_not_worse': int(drawdown.sum()),
                        'universal_breadth_target_met': bool(len(group) == 36 and
                            (wins & payoff).sum() >= 18 and returns.sum() >= 24 and drawdown.sum() >= 24)})
    return pd.DataFrame(records)


def source_manifest():
    paths = list((ROOT / 'src').glob('*.py')) + [ROOT / 'run_research.py', ROOT / 'FX_Technical_Research_Complete.ipynb']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}
