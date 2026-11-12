"""One causal early-exit check at a fixed earned-return age per BASE episode.

Original TARGET changes start/end episodes. Entry signal s fills at s+1 and
first earns at s+2. At age k, checkpoint t=s+1+k can order an exit at t+1;
that execution day's return is still earned. False/UNKNOWN is never retried.
This module does not load research data, select conditions, or tune thresholds.
"""
from collections import deque

import numpy as np
import pandas as pd

try:
    from .fx_baselines import fixed_execute
except ImportError:
    from fx_baselines import fixed_execute


CHECKPOINT_AGES = (5, 10, 20, 40, 63)
CONDITIONS = ('pnl_negative', 'pnl_positive', 'giveback_half', 'recent5_negative', 'opposite20')
TOL_PCT = 1e-10  # Matches the holding-observation zero convention, in percentage points.
_INDEX_COLUMNS = ('entry_idx', 'checkpoint_idx', 'risk_signal_idx', 'original_end_signal_idx',
                  'original_exit_idx', 'actual_exit_idx')
_COLUMNS = (
    'pair', 'strategy', 'candidate_id', 'original_episode', 'signal_idx', 'signal_date',
    'entry_idx', 'entry_date', 'entry_pending', 'position', 'accepted',
    'checkpoint_age', 'condition', 'checkpoint_idx', 'checkpoint_date', 'evaluated',
    'condition_state', 'known_pnl_pct', 'running_peak_pct', 'giveback_fraction',
    'signed_recent5_pct', 'direction_return_20_pct', 'early_exit',
    'risk_signal_idx', 'risk_signal_date', 'risk_pending', 'original_end_signal_idx',
    'original_end_signal_date', 'original_exit_idx', 'original_exit_date',
    'actual_exit_idx', 'actual_exit_date', 'exit_pending', 'status',
)


def validate_spec(spec):
    """Only fixed predicate names and exact checkpoint ages are supported."""
    if not isinstance(spec, dict) or set(spec) != {'id', 'checkpoint_age', 'condition'}:
        raise ValueError('Spec must contain exactly id, checkpoint_age and condition.')
    if not isinstance(spec['id'], str) or not spec['id'].strip():
        raise ValueError('A nonempty candidate id is required.')
    age = spec['checkpoint_age']
    if isinstance(age, (bool, np.bool_)) or not isinstance(age, (int, np.integer)) or age not in CHECKPOINT_AGES:
        raise ValueError('checkpoint_age must be one of 5, 10, 20, 40, 63 earned returns.')
    condition = spec['condition']
    names = [condition] if isinstance(condition, str) else condition
    if not isinstance(names, list) or not 1 <= len(names) <= 2 or any(not isinstance(x, str) or x not in CONDITIONS for x in names):
        raise ValueError('condition must be one fixed predicate or a list of at most two predicates.')
    if len(set(names)) != len(names) or {'pnl_negative', 'pnl_positive'}.issubset(names):
        raise ValueError('Duplicate or contradictory predicates are not supported.')
    return int(age), tuple(name for name in CONDITIONS if name in names)


def _inputs(prices, dates, base_run, features, pair, names):
    prices, dates = np.asarray(prices, float), pd.DatetimeIndex(dates)
    if prices.ndim != 1 or not len(prices) or len(prices) != len(dates) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Prices must be a positive finite vector aligned with dates.')
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or dates.tz is not None:
        raise ValueError('Dates must be unique, increasing and timezone-naive.')
    if not isinstance(features, pd.DataFrame) or len(features) != len(dates) or 'date' not in features:
        raise ValueError('Features require one dated row per price observation.')
    if not pd.DatetimeIndex(pd.to_datetime(features.date)).equals(dates):
        raise ValueError('Feature dates differ from the supplied price calendar.')
    if pair and 'pair' in features and not features.pair.eq(pair).all():
        raise ValueError('Feature rows contain a different or missing currency.')
    if 'opposite20' in names and 'spot_return_20_pct' not in features:
        raise ValueError('opposite20 requires the causal spot_return_20_pct feature.')
    original = np.asarray(base_run['target'])
    if original.shape != prices.shape or not np.isin(original, [-1, 0, 1]).all():
        raise ValueError('BASE targets must be aligned signed unit positions.')
    executed = np.r_[0, original[:-1]]
    held = np.r_[0, executed[:-1]]
    gross = held * np.r_[0., prices[1:] / prices[:-1] - 1.]
    for field, expected in (('executed', executed), ('held', held), ('gross_return', gross), ('net_return', gross)):
        actual = np.asarray(base_run[field])
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-12, rtol=0):
            raise ValueError('BASE ' + field + ' differs from the zero-cost next-close path.')
    if float(base_run.get('one_way_cost_bp', 0.)) != 0.:
        raise ValueError('Holding research requires zero costs.')
    return prices, dates, original.astype(np.int8, copy=False), gross


def _checkpoint_snapshot(episode, features, index):
    pnl, peak = episode['pnl'] * 100., episode['peak'] * 100.
    recent = sum(episode['recent']) * 100. if len(episode['recent']) == 5 else np.nan
    raw = features.iloc[index].get('spot_return_20_pct', np.nan)
    try:
        direction = episode['position'] * float(raw)
    except (TypeError, ValueError):
        direction = np.nan
    if not np.isfinite(direction):
        direction = np.nan
    return {
        'known_pnl_pct': pnl, 'running_peak_pct': peak,
        'giveback_fraction': (peak - pnl) / peak if peak > TOL_PCT else np.nan,
        'signed_recent5_pct': recent, 'direction_return_20_pct': direction,
    }


def _condition_state(values, names):
    """UNKNOWN retains the episode, including when combined with another term."""
    pnl, peak = values['known_pnl_pct'], values['running_peak_pct']
    predicates = {
        'pnl_negative': pnl < -TOL_PCT,
        'pnl_positive': pnl > TOL_PCT,
        'giveback_half': peak > TOL_PCT and pnl <= peak / 2. + TOL_PCT,
        'recent5_negative': None if not np.isfinite(values['signed_recent5_pct']) else values['signed_recent5_pct'] < -TOL_PCT,
        'opposite20': None if not np.isfinite(values['direction_return_20_pct']) else values['direction_return_20_pct'] < -TOL_PCT,
    }
    selected = [predicates[name] for name in names]
    return 'UNKNOWN' if any(value is None for value in selected) else 'TRUE' if all(selected) else 'FALSE'


def run_holding_filter(prices, dates, base_run, features, spec, pair='', strategy=''):
    """Return (fixed_execute run, one diagnostic row per original TARGET episode).

    At signal close t only quotes/features through t and previous state are used.
    Original technical exits/reversals already known at t take precedence over
    risk evaluation. A risk exit locks that original episode; a new original
    TARGET episode resets PnL/peak/checkpoint state and always retains its entry.
    Nullable indices mean execution/checkpoint has not occurred in this sample.
    """
    age, names = validate_spec(spec)
    prices, dates, original, gross = _inputs(prices, dates, base_run, features, pair, names)
    size = len(prices)
    target = original.copy()
    signal_episode = np.full(size, -1, dtype=int)
    episodes, active = [], None

    for t in range(size):
        # The return at t belongs to the original signal in force at t-2, even
        # on the execution day of a later exit/reversal. No endpoint shortcut.
        earned = signal_episode[t - 2] if t >= 2 else -1
        if earned >= 0:
            owner = episodes[earned]
            owner['pnl'] += float(gross[t])
            owner['peak'] = max(owner['peak'], owner['pnl'])
            owner['recent'].append(float(gross[t]))
            owner['earned_age'] += 1

        previous = original[t - 1] if t else 0
        if original[t] != previous:
            if active is not None:
                old = episodes[active]
                old['end'] = t
                if t == old['signal'] + 1 + age:
                    old['checkpoint'] = t
                    old['snapshot'] = _checkpoint_snapshot(old, features, t)
                # Closing a TARGET segment is observed now, never looked up
                # in advance to decide whether an earlier risk signal is valid.
            active = None
            if original[t] != 0:
                active = len(episodes)
                episodes.append({
                    'signal': t, 'position': int(original[t]), 'end': None,
                    'pnl': 0., 'peak': 0., 'recent': deque(maxlen=5), 'earned_age': 0,
                    'checkpoint': None, 'snapshot': {}, 'evaluated': False,
                    'condition_state': 'NOT_EVALUATED', 'risk': None,
                })
        if active is None:
            continue
        signal_episode[t] = active
        current = episodes[active]
        if current['risk'] is not None:
            target[t] = 0
        elif t == current['signal'] + 1 + age:
            if current['earned_age'] != age:
                raise AssertionError('Checkpoint age must count exactly earned daily returns.')
            current['checkpoint'], current['evaluated'] = t, True
            current['snapshot'] = _checkpoint_snapshot(current, features, t)
            current['condition_state'] = _condition_state(current['snapshot'], names)
            if current['condition_state'] == 'TRUE':
                current['risk'], target[t] = t, 0

    rows = []
    for number, episode in enumerate(episodes):
        start, end, risk, checkpoint = (episode[name] for name in ('signal', 'end', 'risk', 'checkpoint'))
        entry = start + 1 if start + 1 < size else None
        natural_exit = end + 1 if end is not None and end + 1 < size else None
        actual_exit = (risk + 1 if risk + 1 < size else None) if risk is not None else natural_exit
        risk_pending = risk is not None and actual_exit is None
        pending = risk_pending or risk is None and end is not None and natural_exit is None
        status = ('EARLY_EXIT_PENDING' if risk_pending else 'EARLY_EXIT') if risk is not None else (
            'UNKNOWN' if episode['condition_state'] == 'UNKNOWN' else
            'KEEP' if episode['evaluated'] else 'TECHNICAL_EXIT' if end is not None else
            'PENDING_ENTRY' if entry is None else 'NOT_REACHED')
        row = {
            'pair': pair, 'strategy': strategy, 'candidate_id': spec['id'], 'original_episode': number,
            'signal_idx': start, 'signal_date': dates[start], 'position': episode['position'],
            'accepted': True, 'entry_pending': entry is None, 'checkpoint_age': age,
            'condition': '&'.join(names), 'evaluated': episode['evaluated'],
            'condition_state': episode['condition_state'], 'early_exit': risk is not None,
            'risk_pending': risk_pending, 'exit_pending': pending, 'status': status,
            **{name: episode['snapshot'].get(name, np.nan) for name in
               ('known_pnl_pct', 'running_peak_pct', 'giveback_fraction', 'signed_recent5_pct', 'direction_return_20_pct')},
        }
        for name, value in (('entry', entry), ('checkpoint', checkpoint), ('risk_signal', risk),
                            ('original_end_signal', end), ('original_exit', natural_exit), ('actual_exit', actual_exit)):
            row[name + '_idx'] = value if value is not None else pd.NA
            row[name + '_date'] = dates[value] if value is not None else pd.NaT
        rows.append(row)
    decisions = pd.DataFrame(rows, columns=_COLUMNS)
    for name in _INDEX_COLUMNS:
        decisions[name] = decisions[name].astype('Int64')
    for name in ('accepted', 'entry_pending', 'evaluated', 'early_exit', 'risk_pending', 'exit_pending'):
        decisions[name] = decisions[name].astype(bool)
    run = fixed_execute(prices, target, cost_bp=0., capital=float(base_run.get('initial_capital_usd', 100000.)))
    run['dates'], run['candidate_id'] = dates, spec['id']
    if 'account_start_date' in base_run:
        run['account_start_date'] = base_run['account_start_date']
    return run, decisions
