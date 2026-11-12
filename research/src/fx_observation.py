"""Observation-first diagnostics of every fixed technical BASE account.

Outcome windows, drawdowns and trade excursions are retrospective labels, never
inputs to a trading rule. No filter, threshold search or winner selection is run.
Primary accounting is fixed initial capital; compound paths remain a reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_baselines import CURRENCIES, STRATEGIES, RULES, FLAT_TOLERANCE


CAPITAL = 100000.0
TOL_PCT = FLAT_TOLERANCE * 100


def quality(values):
    values = np.asarray(values, dtype=float)
    wins, losses = values[values > TOL_PCT], values[values < -TOL_PCT]
    return {
        'closed_trades': len(values),
        'win_rate': len(wins) / len(values) if len(values) else np.nan,
        'payoff_ratio': wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan,
        'average_win_pct': wins.mean() if len(wins) else np.nan,
        'average_loss_abs_pct': -losses.mean() if len(losses) else np.nan,
        'expectancy_pct': values.mean() if len(values) else np.nan,
    }


def trade_excursions(run, dates, ledger):
    """Close-observed extrema on actual strategy returns, including entry zero."""
    dates = pd.DatetimeIndex(dates)
    returns = np.asarray(run['daily_ret'], float)
    covered = np.zeros(len(returns), dtype=np.int8)
    result = ledger.copy()
    records = []
    for trade in ledger.itertuples(index=False):
        start, end = int(trade.entry_idx), int(trade.exit_idx)
        segment = returns[start + 1:end + 1]
        covered[start + 1:end + 1] += 1
        simple = np.r_[0., np.cumsum(segment) * 100.]
        compound = np.r_[0., (np.cumprod(1. + segment) - 1.) * 100.]
        mfe, mae = float(simple.max()), float(simple.min())
        records.append({
            'simple_return_pct': float(simple[-1]),
            'compound_return_pct': float(compound[-1]),
            'simple_pnl_usd': float(simple[-1] * CAPITAL / 100),
            'mfe_simple_pct': mfe, 'mae_simple_pct': mae,
            'mfe_compound_pct': float(compound.max()),
            'mae_compound_pct': float(compound.min()),
            'mfe_date': dates[start + int(simple.argmax())],
            'mae_date': dates[start + int(simple.argmin())],
            'giveback_simple_pct': mfe - simple[-1],
            'giveback_ratio': (mfe - simple[-1]) / mfe if mfe > TOL_PCT else np.nan,
            'loser_with_prior_gain': bool(simple[-1] < -TOL_PCT and mfe > TOL_PCT),
            'entry_year': dates[start].year, 'exit_year': dates[end].year,
            'crosses_calendar_year': dates[start].year != dates[end].year,
        })
    if np.any(covered > 1) or np.any((run['held'] != 0) & (covered != 1)):
        raise AssertionError('Trade intervals do not partition held return intervals.')
    excursion_columns = ['simple_return_pct', 'compound_return_pct', 'simple_pnl_usd',
                         'mfe_simple_pct', 'mae_simple_pct', 'mfe_compound_pct', 'mae_compound_pct',
                         'mfe_date', 'mae_date', 'giveback_simple_pct', 'giveback_ratio',
                         'loser_with_prior_gain', 'entry_year', 'exit_year', 'crosses_calendar_year']
    result = pd.concat([result.reset_index(drop=True), pd.DataFrame(records, columns=excursion_columns)], axis=1)
    if not np.isclose(result.simple_return_pct.sum(), returns.sum() * 100, atol=1e-9, rtol=0):
        raise AssertionError('Trade simple returns fail account reconciliation.')
    if not np.allclose(result.compound_return_pct.to_numpy(dtype=float), result.return_pct.to_numpy(dtype=float), atol=1e-9, rtol=0):
        raise AssertionError('Compound trade outcomes differ from baseline ledger.')
    return result


def drawdown_episodes(daily_returns, dates, anchor_date):
    """Nonoverlapping peak-to-recovery episodes; unrecovered tails stay open."""
    dates = pd.DatetimeIndex(dates)
    all_dates = dates.insert(0, pd.Timestamp(anchor_date))
    pnl = np.r_[0., np.cumsum(daily_returns) * 100.]
    peak = 0
    active = False
    trough = 0
    records = []

    def record(end, recovered):
        depth = float(pnl[peak] - pnl[trough])
        return {
            'peak_date': all_dates[peak], 'decline_start_date': all_dates[peak + 1],
            'trough_date': all_dates[trough],
            'recovery_date': all_dates[end] if recovered else pd.NaT,
            'observed_end_date': all_dates[end], 'recovered': bool(recovered),
            'depth_pp': depth, 'peak_equity_drawdown_pct':
                depth / (100 + pnl[peak]) * 100 if 100 + pnl[peak] > 0 else np.nan,
            'duration_observations': end - peak, 'decline_observations': trough - peak,
            'recovery_observations': end - trough if recovered else np.nan,
            'duration_calendar_days': (all_dates[end] - all_dates[peak]).days,
            'peak_idx': peak - 1, 'trough_idx': trough - 1, 'end_idx': end - 1,
        }

    for i in range(1, len(pnl)):
        if not active:
            if pnl[i] >= pnl[peak] - TOL_PCT:
                peak = i
            else:
                active, trough = True, i
        else:
            if pnl[i] < pnl[trough]:
                trough = i
            if pnl[i] >= pnl[peak] - TOL_PCT:
                records.append(record(i, True))
                active, peak = False, i
    if active:
        records.append(record(len(pnl) - 1, False))
    columns = ['peak_date', 'decline_start_date', 'trough_date', 'recovery_date',
               'observed_end_date', 'recovered', 'depth_pp', 'peak_equity_drawdown_pct',
               'duration_observations', 'decline_observations', 'recovery_observations',
               'duration_calendar_days', 'peak_idx', 'trough_idx', 'end_idx']
    frame = pd.DataFrame(records, columns=columns)
    expected = float((np.maximum.accumulate(pnl) - pnl).max())
    actual = float(frame.depth_pp.max()) if len(frame) else 0.
    if not np.isclose(actual, expected, atol=1e-8, rtol=0):
        raise AssertionError('Drawdown episodes do not recover account maximum drawdown.')
    return frame


def _cohort(trades, start, end):
    entries = trades.loc[trades.entry_date.between(start, end)]
    eligible = entries.loc[entries.eligible_closed & entries.exit_date.le(end)]
    result = {'cohort_' + key: value for key, value in quality(eligible.simple_return_pct).items()}
    result['cohort_entries'] = len(entries)
    result['cohort_censored_trades'] = int((~entries.closed | entries.exit_date.gt(end)).sum())
    result['cohort_ineligible_closed_trades'] = int((entries.closed & ~entries.eligible_closed & entries.exit_date.le(end)).sum())
    return result


def _window(returns, held, dates, start, end, trades):
    values = returns[start:end + 1]
    path = np.r_[0., np.cumsum(values) * 100.]
    return {
        'start_date': dates[start], 'end_date': dates[end],
        'start_idx': int(start), 'end_idx': int(end),
        'observations': len(values), 'active_days': int(np.count_nonzero(held[start:end + 1])),
        'simple_return_pct': float(values.sum() * 100),
        'compound_return_pct': float((np.prod(1 + values) - 1) * 100),
        'simple_max_drawdown_pp': float((path - np.maximum.accumulate(path)).min()),
        **_cohort(trades, dates[start], dates[end]),
    }


def calendar_periods(returns, held, dates, trades):
    dates = pd.DatetimeIndex(dates)
    rows = []
    for period_type in ('year', 'half_year'):
        labels = [str(d.year) if period_type == 'year' else f'{d.year}H{1 if d.month <= 6 else 2}' for d in dates]
        groups = pd.Series(np.arange(len(dates))).groupby(labels, sort=True)
        for label, indices in groups:
            a, b = int(indices.iloc[0]), int(indices.iloc[-1])
            year = int(label[:4])
            end = pd.Timestamp(year, 12, 31) if period_type == 'year' or label.endswith('H2') else pd.Timestamp(year, 6, 30)
            start = pd.Timestamp(year, 7 if period_type == 'half_year' and label.endswith('H2') else 1, 1)
            # Coverage of the full calendar interval, allowing weekend endpoints.
            complete = (dates[0] <= pd.offsets.BDay().rollforward(start)
                        and pd.offsets.BDay().rollback(end) <= dates[-1])
            rows.append({'period_type': period_type, 'period': label,
                         'complete_calendar_period': bool(complete),
                         **_window(returns, held, dates, a, b, trades)})
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby('period_type'):
        if not np.isclose(group.simple_return_pct.sum(), np.sum(returns) * 100, atol=1e-9, rtol=0):
            raise AssertionError('Calendar simple additivity failed.')
        if not np.isclose(np.prod(1 + group.compound_return_pct / 100), np.prod(1 + returns), atol=1e-10, rtol=0):
            raise AssertionError('Calendar compounding failed.')
    return frame


def select_rolling_windows(returns, held, dates, trades, window=252, per_sign=3):
    """Greedy largest-absolute outcome windows, disjoint per account, capped by sign.

    These are deliberately ex-post diagnostics, not independent validation cases
    or thresholds for a live filter. All full-window rolling returns are returned.
    """
    rolling = pd.Series(returns).rolling(window, min_periods=window).sum().to_numpy() * 100
    used = np.zeros(len(returns), dtype=bool)
    counts = {'positive': 0, 'negative': 0}
    rows = []
    for end in np.argsort(-np.nan_to_num(np.abs(rolling), nan=-1.), kind='stable'):
        if not np.isfinite(rolling[end]) or abs(rolling[end]) <= TOL_PCT:
            continue
        kind = 'positive' if rolling[end] > 0 else 'negative'
        start = int(end - window + 1)
        if counts[kind] >= per_sign or used[start:end + 1].any():
            continue
        used[start:end + 1] = True
        counts[kind] += 1
        rows.append({'kind': kind, 'rank': counts[kind],
                     **_window(returns, held, dates, start, int(end), trades)})
        if all(v == per_sign for v in counts.values()):
            break
    return pd.DataFrame(rows), rolling


def window_attribution(returns, dates, trades, windows, offset):
    """Clip each original trade to the window's earned-return dates, including carry-in."""
    rows = []
    prefix = np.r_[0., np.cumsum(returns)]
    for window in windows:
        start, end = int(window['start_idx']), int(window['end_idx'])
        subtotal = 0.
        for trade in trades.itertuples(index=False):
            a = max(start, int(trade.entry_idx) + 1 - offset)
            b = min(end, int(trade.exit_idx) - offset)
            if a > b:
                continue
            contribution = float((prefix[b + 1] - prefix[a]) * 100)
            subtotal += contribution
            rows.append({
                'window_id': window['window_id'], 'window_type': window['window_type'],
                'window_start_date': dates[start], 'window_end_date': dates[end],
                'trade_id': int(trade.trade_id), 'position': int(trade.position),
                'entry_date': trade.entry_date, 'exit_date': trade.exit_date,
                'closed': bool(trade.closed), 'overlap_start_date': dates[a],
                'overlap_end_date': dates[b], 'overlap_observations': b - a + 1,
                'carry_in': bool(trade.entry_date < dates[start]),
                'continues_after_window': bool(trade.exit_date > dates[end] or not trade.closed),
                'simple_return_contribution_pp': contribution,
                'simple_pnl_contribution_usd': contribution * CAPITAL / 100,
            })
        if not np.isclose(subtotal, returns[start:end + 1].sum() * 100, atol=1e-8, rtol=0):
            raise AssertionError('Clipped trade contributions fail window reconciliation.')
    return pd.DataFrame(rows)


def _concentration(values, positive, n):
    selected = values[values > TOL_PCT] if positive else -values[values < -TOL_PCT]
    return float(np.sort(selected)[-n:].sum() / selected.sum()) if len(selected) else np.nan


def analyze_baselines(baseline, progress=print):
    """Build the full 180-account observation census without modifying the baseline."""
    _, ledger, runs, price_dates, _ = baseline
    result = {key: [] for key in ('daily', 'accounts', 'periods', 'rolling', 'drawdowns', 'trades', 'attribution')}
    for pair in CURRENCIES:
        progress(f'Observing {pair}: all 20 fixed rules...', flush=True)
        full_dates = pd.DatetimeIndex(price_dates[pair])
        offset = int(full_dates.searchsorted(pd.Timestamp('2005-01-01')))
        dates = full_dates[offset:]
        for strategy in STRATEGIES:
            keys = {'pair': pair, 'strategy': strategy}
            run = runs[(pair, strategy)]
            returns = np.asarray(run['daily_ret'][offset:], float)
            held = np.asarray(run['held'][offset:])
            original = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)]
            trades = trade_excursions(run, full_dates, original)
            periods = calendar_periods(returns, held, dates, trades)
            windows, rolling = select_rolling_windows(returns, held, dates, trades)
            drawdowns = drawdown_episodes(returns, dates, full_dates[offset - 1])
            simple = np.cumsum(returns) * CAPITAL
            compound = (np.cumprod(1 + returns) - 1) * CAPITAL
            daily = pd.DataFrame({**keys, 'date': dates, 'daily_return': returns, 'held': held,
                                  'simple_pnl_usd': simple, 'compound_pnl_usd': compound,
                                  'rolling_252_simple_return_pct': rolling})
            closed = trades.loc[trades.eligible_closed]
            q = quality(closed.simple_return_pct)
            cq = quality(closed.compound_return_pct)
            losses = closed.loc[closed.simple_return_pct < -TOL_PCT]
            winners = closed.loc[closed.simple_return_pct > TOL_PCT]
            annual = periods.loc[periods.period_type.eq('year')]
            outcome = closed.simple_return_pct.to_numpy()
            year_values = annual.simple_return_pct.to_numpy()
            accounts = {
                **keys, 'hypothesis': RULES[strategy][1], 'sample_start': dates[0], 'sample_end': dates[-1],
                'accounting_method': 'fixed_initial_capital_simple', 'initial_capital_usd': CAPITAL,
                'simple_return_pct': float(simple[-1] / CAPITAL * 100),
                'compound_return_pct': float(compound[-1] / CAPITAL * 100), **q,
                'compound_win_rate': cq['win_rate'], 'compound_payoff_ratio': cq['payoff_ratio'],
                'simple_max_drawdown_pp': -float(drawdowns.depth_pp.max()) if len(drawdowns) else 0.,
                'active_days': int(np.count_nonzero(held)), 'observations': len(dates),
                'positive_years': int((year_values > TOL_PCT).sum()), 'observed_years': len(annual),
                'positive_half_years': int((periods.loc[periods.period_type.eq('half_year'), 'simple_return_pct'] > TOL_PCT).sum()),
                'open_trades': int((~trades.closed).sum()), 'low_sample_flag': len(closed) < 20,
                'largest_loss_share_of_gross_losses': _concentration(outcome, False, 1),
                'top3_loss_share_of_gross_losses': _concentration(outcome, False, 3),
                'largest_win_share_of_gross_wins': _concentration(outcome, True, 1),
                'top3_win_share_of_gross_wins': _concentration(outcome, True, 3),
                'top5_win_share_of_gross_wins': _concentration(outcome, True, 5),
                'largest_year_share_of_positive_year_returns': _concentration(year_values, True, 1),
                'simple_return_without_best_closed_trade_pct': float(simple[-1] / CAPITAL * 100 - max(0., outcome.max())) if len(outcome) else float(simple[-1] / CAPITAL * 100),
                'losers_with_prior_gain_fraction': float(losses.loser_with_prior_gain.mean()) if len(losses) else np.nan,
                'median_winner_holding_observations': winners.holding_observations.median(),
                'median_loser_holding_observations': losses.holding_observations.median(),
                'median_loser_mfe_pct': losses.mfe_simple_pct.median(),
                'median_loser_mae_pct': losses.mae_simple_pct.median(),
            }
            attribution_windows = [dict(row, window_id=f"rolling_{row['kind']}_{row['rank']}", window_type='rolling') for row in windows.to_dict('records')]
            for rank, row in enumerate(drawdowns.nlargest(3, 'depth_pp').to_dict('records'), 1):
                attribution_windows.append({'start_idx': int(row['peak_idx']) + 1, 'end_idx': int(row['trough_idx']),
                                            'window_id': f'drawdown_{rank}', 'window_type': 'drawdown_decline'})
            attribution = window_attribution(returns, dates, trades, attribution_windows, offset)
            for key, frame in [('periods', periods), ('rolling', windows), ('drawdowns', drawdowns), ('attribution', attribution)]:
                result[key].append(frame.assign(**keys))
            result['daily'].append(daily)
            result['accounts'].append(accounts)
            result['trades'].append(trades)
    result = {key: pd.DataFrame(frames) if key == 'accounts' else pd.concat(frames, ignore_index=True)
              for key, frames in result.items()}
    if len(result['accounts']) != 180 or result['accounts'][['pair', 'strategy']].duplicated().any():
        raise AssertionError('Expected exactly 180 unique technical accounts.')
    accounts = result['accounts']
    result['rule_summary'] = accounts.groupby('strategy', sort=False).agg(
        accounts=('pair', 'size'), positive_simple_accounts=('simple_return_pct', lambda v: int((v > TOL_PCT).sum())),
        median_simple_return_pct=('simple_return_pct', 'median'), median_win_rate=('win_rate', 'median'),
        median_payoff_ratio=('payoff_ratio', 'median'), median_max_drawdown_pp=('simple_max_drawdown_pp', 'median'),
        median_top3_loss_share=('top3_loss_share_of_gross_losses', 'median'),
        median_losers_with_prior_gain_fraction=('losers_with_prior_gain_fraction', 'median'),
        median_winner_holding=('median_winner_holding_observations', 'median'),
        median_loser_holding=('median_loser_holding_observations', 'median'),
        median_positive_years=('positive_years', 'median'),
    ).reset_index()
    return result
