"""Reconstruct the earlier V3 price gates and separate-source H.10 replay.

V3 predicates come from the delivered Explorer. Indicators warm on earlier
prices, and each D/V/T account starts from cash independently. V3 gates apply
to every day's warmed target and can re-enter within a technical episode.
H.10 uses its own valid-quote calendar and never joins workbook price levels.
Archived outputs are read only by the verification function.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fx_baselines import (
    CURRENCIES, DATA_ROOT, STRATEGIES, complete_signals,
    fixed_execute, trade_ledger, account_metrics,
)


V3_STAGES = {
    'D': ('2005-01-01', '2014-12-31'),
    'V': ('2015-01-01', '2019-12-31'),
    'T': ('2020-01-01', '2026-05-29'),
}
V3_BASE_RULES = {
    'BB10_RANGE_GATE': 'BB_MR_10',
    'BB20_RANGE_GATE': 'BB_MR_20',
    'MOM63_TREND_GATE': 'TSMOM_63',
    'MOM_MA_CONFIRM': 'TSMOM_63',
}


def _add_benchmarks(signals, length, include_cash=True):
    result = {name: signals[name] for name in STRATEGIES}
    result['ALWAYS_LONG_FX'] = np.ones(length, np.int8)
    result['ALWAYS_SHORT_FX'] = -np.ones(length, np.int8)
    if include_cash:
        result['CASH'] = np.zeros(length, np.int8)
    return result


def _cash_reset_run(values, targets, dates, start_date):
    first_execution = int(dates.searchsorted(pd.Timestamp(start_date)))
    if first_execution >= len(dates):
        raise ValueError(f'No observations on or after {start_date}.')
    permitted = np.asarray(targets).copy()
    permitted[:max(0, first_execution - 1)] = 0
    run = fixed_execute(values, permitted)
    run['dates'] = dates
    return run


def reconstruct_v3(prices, stages=None):
    """Recompute 234 independent accounts per D/V/T cash-reset period.

    Runs use keys (stage, pair, strategy); price_dates uses (stage, pair).
    Trade statistics are newly reconstructed, because V3 trade ledgers were
    not included in the Explorer reference.
    """
    stages = V3_STAGES if stages is None else stages
    metrics, ledgers, runs, price_dates = [], [], {}, {}
    for pair in CURRENCIES:
        panel = prices.loc[prices.pair.eq(pair)].sort_values('date')
        full_dates = pd.DatetimeIndex(panel.date)
        full_values = panel.price_USD_per_foreign.to_numpy(float)
        signals, features = complete_signals(full_values)
        signals = _add_benchmarks(signals, len(full_values), include_cash=False)
        signals.update({
            'BB10_RANGE_GATE': np.where(features['eff63'] < .2, signals['BB_MR_10'], 0),
            'BB20_RANGE_GATE': np.where(features['eff63'] < .2, signals['BB_MR_20'], 0),
            'MOM63_TREND_GATE': np.where(features['eff63'] > .4, signals['TSMOM_63'], 0),
            'MOM_MA_CONFIRM': np.where(signals['TSMOM_63'] == signals['MA_20_60'], signals['TSMOM_63'], 0),
        })
        for stage, (start, end) in stages.items():
            stop = int(full_dates.searchsorted(pd.Timestamp(end), side='right'))
            dates, values = full_dates[:stop], full_values[:stop]
            price_dates[(stage, pair)] = dates
            for strategy, target in signals.items():
                run = _cash_reset_run(values, target[:stop], dates, start)
                ledger = trade_ledger(run, dates, pair, strategy)
                ledger['stage'] = stage
                runs[(stage, pair, strategy)] = run
                if not ledger.empty:
                    ledgers.append(ledger)
                metrics.append({
                    'pair': pair, 'strategy': strategy, 'stage': stage,
                    'base_strategy': V3_BASE_RULES.get(strategy, strategy),
                    'variant': strategy if strategy in V3_BASE_RULES else 'BASE',
                    'trade_quality_source': 'reconstructed_not_in_archived_V3_ledger',
                    **account_metrics(run, dates, ledger, start_date=start),
                })
    metrics = pd.DataFrame(metrics)
    originals = metrics[metrics.variant.eq('BASE')]
    comparisons = metrics[~metrics.variant.eq('BASE')].merge(
        originals[['pair', 'strategy', 'stage', 'account_pnl_usd', 'max_drawdown_pct']],
        left_on=['pair', 'base_strategy', 'stage'], right_on=['pair', 'strategy', 'stage'],
        suffixes=('', '_base'), validate='many_to_one',
    )
    comparisons['pnl_change_usd'] = comparisons.account_pnl_usd - comparisons.account_pnl_usd_base
    comparisons['drawdown_change_pp'] = comparisons.max_drawdown_pct - comparisons.max_drawdown_pct_base
    return {'metrics': metrics, 'runs': runs, 'price_dates': price_dates,
            'comparisons': comparisons, 'trades': pd.concat(ledgers, ignore_index=True)}


def reconstruct_h10(data_root=None, warmup_start='2024-01-02', sample_start='2026-06-01'):
    """Replay 207 accounts from the retained official H.10 quote extract.

    Missing/ND observations are excluded, never forward-filled. Each currency
    retains its own dates. Extra earlier quotes check warmup sensitivity.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    source = pd.read_csv(root / 'clean/h10_official_daily.csv', parse_dates=['date'])
    if source.duplicated(['pair', 'date']).any():
        raise ValueError('Duplicate H.10 currency/date records.')
    metrics, coverage, ledgers, runs, price_dates = [], [], [], {}, {}
    for pair in CURRENCIES:
        raw_panel = source.loc[source.pair.eq(pair)].sort_values('date')
        finite = np.isfinite(raw_panel.usd_per_foreign) & raw_panel.usd_per_foreign.gt(0)
        valid_panel = raw_panel.loc[finite]
        panel = valid_panel.loc[valid_panel.date.ge(warmup_start)]
        dates = pd.DatetimeIndex(panel.date)
        values = panel.usd_per_foreign.to_numpy(float)
        price_dates[pair] = dates
        signals, features = complete_signals(values)
        earlier_signals, earlier_features = complete_signals(valid_panel.usd_per_foreign.to_numpy(float))
        earlier_dates = pd.DatetimeIndex(valid_panel.date)
        first_execution = int(dates.searchsorted(pd.Timestamp(sample_start)))
        decision_indices = np.arange(max(0, first_execution - 1), len(dates))
        earlier_indices = earlier_dates.get_indexer(dates[decision_indices])
        warmed_targets_identical = all(
            np.array_equal(signals[name][decision_indices], earlier_signals[name][earlier_indices])
            for name in STRATEGIES
        )
        sample_dates = dates[dates >= sample_start]
        weekdays = pd.bdate_range(sample_dates[0], sample_dates[-1])
        sample_raw = raw_panel.loc[raw_panel.date.between(sample_dates[0], sample_dates[-1])]
        coverage.append({
            'pair': pair, 'warmup_start': dates[0].strftime('%Y-%m-%d'),
            'pre_sample_valid_observations': first_execution,
            'sample_first': sample_dates[0].strftime('%Y-%m-%d'),
            'sample_last': sample_dates[-1].strftime('%Y-%m-%d'),
            'sample_valid_quotes': len(sample_dates), 'sample_calendar_weekdays': len(weekdays),
            'sample_ND_dates': int(sample_raw.is_nd.sum()),
            'earlier_warmup_all_targets_identical': warmed_targets_identical,
            'source_url': raw_panel.source_url.iloc[0],
            'calendar_missing_quote_dates': ';'.join(weekdays.difference(sample_dates).strftime('%Y-%m-%d')),
        })
        for strategy, target in _add_benchmarks(signals, len(values)).items():
            run = _cash_reset_run(values, target, dates, sample_start)
            ledger = trade_ledger(run, dates, pair, strategy)
            if not ledger.empty:
                ledgers.append(ledger)
            runs[(pair, strategy)] = run
            statistics = account_metrics(run, dates, ledger, start_date=sample_start)
            metrics.append({'pair': pair, 'strategy': strategy, **statistics,
                            'closed_trades': statistics['eligible_closed_n'],
                            'open_trades_at_end': statistics['open_n']})
    return {'metrics': pd.DataFrame(metrics), 'runs': runs, 'price_dates': price_dates,
            'coverage': pd.DataFrame(coverage), 'trades': pd.concat(ledgers, ignore_index=True)}


def verify_legacy(result, dataset):
    """Compare reconstructed paths, endpoints and available trade quality to HTML."""
    from fx_evidence import load_explorer

    if dataset not in ('technical', 'h10'):
        raise ValueError('Legacy dataset must be technical or h10.')
    archive = load_explorer()
    accounts = [account for account in archive['accounts'] if account['ds'] == dataset]
    expected = 234 if dataset == 'technical' else 207
    checks = [{'check': f'{dataset} reference account count', 'pass': len(accounts) == expected,
               'observations': len(accounts), 'max_abs_error': abs(len(accounts) - expected)}]
    for account in accounts:
        strategy = account['rule'] if account['variant'] == 'BASE' else account['variant']
        key = ('T', account['pair'], strategy) if dataset == 'technical' else (account['pair'], strategy)
        run = result['runs'][key]
        dates = run['dates']
        reference_dates = pd.to_datetime(archive['calendars'][account['cal']], unit='D')
        locations = dates.get_indexer(reference_dates)
        aligned = bool((locations >= 0).all())
        differences = np.abs(run['equity'][locations] * 100000 - np.asarray(account['equity']))
        max_error = float(differences.max()) if aligned else float('inf')
        checks.append({'check': account['id'] + ' daily equity',
                       'pass': aligned and max_error < 1e-6,
                       'observations': len(reference_dates), 'max_abs_error': max_error})
        selected = result['metrics'].loc[result['metrics'].pair.eq(account['pair']) & result['metrics'].strategy.eq(strategy)]
        if dataset == 'technical':
            selected = selected.loc[selected.stage.eq('T')]
        metric = selected.iloc[0]
        for column, reference in account['source_metrics'].items():
            error = abs(float(metric[column]) - reference)
            checks.append({'check': account['id'] + ' ' + column, 'pass': error < 1e-7,
                           'observations': 1, 'max_abs_error': error})
        if account['quality_full'] is not None:
            for column, quality_key in [('eligible_closed_n', 'closed'), ('win_rate', 'win'), ('payoff_ratio', 'payoff')]:
                reference = account['quality_full'][quality_key]
                actual = float(metric[column])
                passed = bool(np.isnan(actual)) if reference is None else bool(np.isclose(actual, reference, atol=1e-9, rtol=1e-9))
                checks.append({'check': account['id'] + ' ' + column, 'pass': passed,
                               'observations': 1, 'max_abs_error': abs(actual - reference) if reference is not None else np.nan})
    checks = pd.DataFrame(checks)
    if not checks['pass'].all():
        raise AssertionError(checks.loc[~checks['pass']].to_dict('records'))
    return checks
