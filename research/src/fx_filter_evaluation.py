"""Fixed-capital IS/OOS evaluation, explicit gates, and deployment boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_baselines import trade_ledger

CAPITAL = 100000.0
TOL = 1e-10
IS_BLOCKS = (
    ('IS_EARLY', '2005-01-03', '2014-12-31'),
    ('IS_MIDDLE', '2015-01-01', '2019-12-31'),
    ('IS_LATE', '2020-01-01', '2023-05-29'),
)
SCREEN = {
    'min_gross_annualized_simple_pct': 1.0,
    'min_return_improvement_pp': 5.0,
    'min_win_rate_improvement_pp': 2.0,
    'min_payoff_relative_improvement': 0.10,
    'max_drawdown_deterioration_pp': 2.0,
    'return_to_drawdown_must_improve': True,
    'min_closed_trades': 50,
    'min_closed_retention': 0.50,
    'min_positive_is_blocks': 2,
    'min_improved_is_blocks': 2,
    'last_is_block_positive_and_improved': True,
    'one_way_cost_bp': 0.0,
    'cost_sensitivity_evaluated': False,
    'cost_instruction': 'User explicitly requested no costs for this round.',
    'rank': ['gross_return_improvement_pp DESC', 'payoff_absolute_improvement DESC',
             'win_rate_improvement_pp DESC', 'candidate_id ASC'],
    'freeze': 'One highest IS-ranked passing NEW filter per pair/strategy. All passing versions retained in IS table.',
    'schedule_delay_is_diagnostic_only': True,
}
OOS_SCREEN = {
    'min_net_annualized_simple_pct': 1.0,
    'min_net_return_improvement_pp': 0.0,
    'worst_net_drawdown_pp': -15.0,
    'min_positive_anniversary_years': 2,
    'min_closed_trades': 10,
    'quality_confirmation': 'Separately report whether both gross trade win rate and payoff are no worse than BASE, with >=20 closed trades. Not a license to tune on OOS.',
}


def quality(values):
    values = np.asarray(values, float)
    wins, losses = values[values > TOL], values[values < -TOL]
    return {
        'closed_n': len(values), 'win_n': len(wins), 'loss_n': len(losses),
        'flat_n': len(values) - len(wins) - len(losses),
        'win_rate': float(len(wins) / len(values)) if len(values) else np.nan,
        'payoff_ratio': float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
        'mean_win_pct': float(wins.mean()) if len(wins) else np.nan,
        'mean_loss_abs_pct': float(-losses.mean()) if len(losses) else np.nan,
        'expectancy_pct': float(values.mean()) if len(values) else np.nan,
    }


def simple_ledger(run, dates, cost_bp=0.0):
    """Each executed episode receives one entry fee and, if closed, one exit fee."""
    ledger = trade_ledger(run, dates)
    returns = np.asarray(run['gross_return'], float)
    prefix = np.r_[0., np.cumsum(returns)]
    if ledger.empty:
        for name in ('gross_simple_return_pct', 'net_trade_return_pct', 'net_pnl_usd'):
            ledger[name] = pd.Series(dtype=float)
        return ledger
    starts, ends = ledger.entry_idx.to_numpy(int), ledger.exit_idx.to_numpy(int)
    ledger['gross_simple_return_pct'] = (prefix[ends + 1] - prefix[starts + 1]) * 100
    fee_pct = (1. + ledger.closed.to_numpy(float)) * cost_bp / 100.
    ledger['net_trade_return_pct'] = ledger.gross_simple_return_pct - fee_pct
    ledger['net_pnl_usd'] = ledger.net_trade_return_pct * CAPITAL / 100
    return ledger


def daily_account(run, cost_bp=0.0):
    gross = np.asarray(run['gross_return'], float)
    turnover = np.abs(np.asarray(run['executed'], int) - np.asarray(run['held'], int))
    net = gross - turnover * cost_bp / 10000.
    return gross, net, turnover


def evaluate_window(run, dates, start, end, cost_bp=0.0, ledger=None):
    dates = pd.DatetimeIndex(dates)
    indices = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    if not len(indices):
        raise ValueError('Evaluation interval has no quotes.')
    first, last = int(indices[0]), int(indices[-1])
    gross, net, turnover = daily_account(run, cost_bp)
    if ledger is None:
        ledger = simple_ledger(run, dates, cost_bp)
    entered = ledger.loc[ledger.entry_date.between(dates[first], dates[last])]
    closed = entered.loc[entered.eligible_closed & entered.exit_date.le(dates[last])]
    q, nq = quality(closed.gross_simple_return_pct), quality(closed.net_trade_return_pct)
    anchor = dates[max(first - 1, 0)]
    years = (dates[last] - anchor).days / 365.25
    g, n = gross[indices], net[indices]
    gp, npnl = np.r_[0., np.cumsum(g) * 100], np.r_[0., np.cumsum(n) * 100]
    std = n.std(ddof=1) if len(n) > 1 else np.nan
    carry = ledger.loc[ledger.entry_date.lt(dates[first]) & ledger.exit_date.ge(dates[first])]
    carry_net = 0.
    for row in carry.itertuples(index=False):
        a, b = max(first, int(row.entry_idx) + 1), min(last, int(row.exit_idx))
        carry_net += gross[a:b + 1].sum() * 100 if a <= b else 0.
        if row.closed and first <= row.exit_idx <= last:
            carry_net -= cost_bp / 100.
    return {
        'start_date': dates[first], 'end_date': dates[last], 'anchor_date': anchor,
        'calendar_years': years, 'observations': len(indices),
        'gross_simple_return_pct': float(g.sum() * 100),
        'net_simple_return_pct': float(n.sum() * 100),
        'gross_annualized_simple_pct': float(g.sum() * 100 / years) if years else np.nan,
        'net_annualized_simple_pct': float(n.sum() * 100 / years) if years else np.nan,
        'gross_max_drawdown_pp': float((gp - np.maximum.accumulate(gp)).min()),
        'net_max_drawdown_pp': float((npnl - np.maximum.accumulate(npnl)).min()),
        'gross_compound_return_pct': float((np.prod(1 + g) - 1) * 100),
        'net_sharpe_252': float(n.mean() / std * np.sqrt(252)) if std > 0 else np.nan,
        'turnover_units': int(turnover[indices].sum()),
        'cost_drag_pp': float(turnover[indices].sum() * cost_bp / 100),
        'active_fraction': float(np.mean(np.asarray(run['held'])[indices] != 0)),
        **q, 'net_win_rate': nq['win_rate'], 'net_payoff_ratio': nq['payoff_ratio'],
        'entry_n': len(entered), 'carry_in_n': len(carry),
        'carry_in_net_return_pp': float(carry_net),
        'censored_entry_n': int((~entered.closed | entered.exit_date.gt(dates[last])).sum()),
        'low_sample_flag': len(closed) < 20,
    }


def _execution_run(reference, executed, targets):
    """Reaccount a zero-cost run using unchanged spot returns and explicit positions."""
    executed, targets = np.asarray(executed, np.int8), np.asarray(targets, np.int8)
    held = np.r_[np.int8(0), executed[:-1]].astype(np.int8)
    returns = held * np.asarray(reference['spot_return'], float)
    equity = np.cumprod(1 + returns)
    before = CAPITAL * np.r_[1., equity[:-1]]
    price_pnl = before * returns
    run = dict(reference)
    run.update(target=targets, sig=targets, executed=executed, held=held,
               gross_return=returns, daily_ret=returns, net_return=returns,
               turnover=np.abs(executed.astype(int) - held.astype(int)).astype(float),
               equity=equity, equity_before_usd=before, price_pnl_usd=price_pnl,
               equity_after_price_usd=before + price_pnl, net_pnl_usd=price_pnl,
               transaction_cost_usd=np.zeros(len(returns)), cost_fraction_post_pnl=np.zeros(len(returns)),
               one_way_cost_bp=0., initial_capital_usd=CAPITAL)
    return run


def delayed_schedule(run):
    """One-quote lag of the fixed target schedule, without re-solving exit decisions."""
    targets = np.r_[np.int8(0), np.asarray(run['target'])[:-1]].astype(np.int8)
    executed = np.r_[np.int8(0), targets[:-1]].astype(np.int8)
    return _execution_run(run, executed, targets)


def fresh_start(base_run, filtered_run, dates, start):
    """Cash account admits only original BASE episodes whose execution entry >= start."""
    dates = pd.DatetimeIndex(dates)
    ledger = trade_ledger(base_run, dates)
    allowed = np.zeros(len(dates), dtype=bool)
    for row in ledger.itertuples(index=False):
        if row.entry_date >= pd.Timestamp(start):
            stop = int(row.exit_idx) if row.closed else len(dates)
            allowed[int(row.entry_idx):stop] = True
    executed = np.where(allowed, filtered_run['executed'], 0).astype(np.int8)
    targets = np.r_[executed[1:], filtered_run['target'][-1] if allowed[-1] else 0].astype(np.int8)
    return _execution_run(filtered_run, executed, targets)


def screen_candidates(metrics):
    """All criteria are fixed in SCREEN; no OOS rows are consumed here."""
    frame = metrics.copy()
    if not frame.split.isin(['IS', *[p[0] for p in IS_BLOCKS]]).all():
        raise ValueError('IS screen cannot receive OOS or other outcome rows.')
    if frame.duplicated(['candidate_id', 'split', 'variant']).any():
        raise ValueError('Duplicate candidate/period/variant metrics.')
    rows = []
    for candidate_id, group in frame.groupby('candidate_id', sort=False):
        index = group.set_index(['split', 'variant'])
        b, f = index.loc[('IS', 'BASE')], index.loc[('IS', 'FILTER')]
        blocks = [index.loc[(name, 'FILTER')] for name, _, _ in IS_BLOCKS]
        base_blocks = [index.loc[(name, 'BASE')] for name, _, _ in IS_BLOCKS]
        block_returns = np.array([x.gross_simple_return_pct for x in blocks])
        block_delta = block_returns - np.array([x.gross_simple_return_pct for x in base_blocks])
        delta = float(f.gross_simple_return_pct - b.gross_simple_return_pct)
        wr_delta = float((f.win_rate - b.win_rate) * 100)
        payoff_delta = float(f.payoff_ratio - b.payoff_ratio)
        relative = float(f.payoff_ratio / b.payoff_ratio - 1) if np.isfinite(b.payoff_ratio) and b.payoff_ratio > 0 else np.nan
        retention = float(f.closed_n / b.closed_n) if b.closed_n > 0 else 0.
        bdd, fdd = -float(b.gross_max_drawdown_pp), -float(f.gross_max_drawdown_pp)
        ratio_improved = (np.isfinite([bdd, fdd]).all() and bdd >= 0 and fdd >= 0
                          and f.gross_simple_return_pct * bdd > b.gross_simple_return_pct * fdd + TOL)
        flags = {
            'positive_and_annualized_pass': bool(f.gross_simple_return_pct > 0 and f.gross_annualized_simple_pct >= SCREEN['min_gross_annualized_simple_pct'] - TOL),
            'return_improvement_pass': bool(delta >= SCREEN['min_return_improvement_pp'] - TOL),
            'win_rate_pass': bool(np.isfinite(wr_delta) and wr_delta >= SCREEN['min_win_rate_improvement_pp'] - TOL),
            'payoff_pass': bool(np.isfinite(relative) and relative >= SCREEN['min_payoff_relative_improvement'] - TOL),
            'drawdown_pass': bool(f.gross_max_drawdown_pp >= b.gross_max_drawdown_pp - SCREEN['max_drawdown_deterioration_pp'] - TOL),
            'return_to_drawdown_pass': bool(ratio_improved),
            'sample_pass': bool(f.closed_n >= SCREEN['min_closed_trades'] and retention >= SCREEN['min_closed_retention'] - TOL),
            'block_positive_pass': bool(np.count_nonzero(block_returns > TOL) >= SCREEN['min_positive_is_blocks'] and block_returns[-1] > TOL),
            'block_improvement_pass': bool(np.count_nonzero(block_delta > TOL) >= SCREEN['min_improved_is_blocks'] and block_delta[-1] > TOL),
        }
        rows.append({
            'candidate_id': candidate_id, 'pair': f['pair'], 'strategy': f.strategy, 'filter_id': f.filter_id,
            'is_new': bool(f.get('is_new', True)),
            'gross_return_improvement_pp': delta, 'win_rate_improvement_pp': wr_delta,
            'payoff_absolute_improvement': payoff_delta, 'payoff_relative_improvement': relative,
            'retention': retention, 'is_gross_return_pct': float(f.gross_simple_return_pct),
            'is_net_return_pct': float(f.net_simple_return_pct),
            'is_annualized_simple_pct': float(f.gross_annualized_simple_pct),
            'is_win_rate': float(f.win_rate), 'is_payoff_ratio': float(f.payoff_ratio),
            'is_closed_n': int(f.closed_n), 'is_max_drawdown_pp': float(f.gross_max_drawdown_pp),
            'positive_blocks': int(np.count_nonzero(block_returns > TOL)),
            'improved_blocks': int(np.count_nonzero(block_delta > TOL)),
            **flags, 'pass': bool(all(flags.values())),
            'failed_criteria': '|'.join(key for key, value in flags.items() if not value),
        })
    result = pd.DataFrame(rows).sort_values(
        ['gross_return_improvement_pp', 'payoff_absolute_improvement', 'win_rate_improvement_pp', 'candidate_id'],
        ascending=[False, False, False, True], kind='stable').reset_index(drop=True)
    result['freeze_rank'] = np.nan
    winners = result.loc[result['pass'] & result.is_new].drop_duplicates(['pair', 'strategy'])
    result.loc[winners.index, 'freeze_rank'] = np.arange(1, len(winners) + 1)
    result['frozen'] = result.freeze_rank.notna()
    return result
