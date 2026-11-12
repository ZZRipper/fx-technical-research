"""Remove one fixed direction from original technical TARGET episodes.

Prices are USD per foreign currency: +1 is long foreign / short USD, and -1
is short foreign / long USD. No state, threshold or outcome selects direction
inside this engine. Entry signal t fills at t+1 and first earns at t+2.
"""
import numpy as np
import pandas as pd

try:
    from .fx_baselines import fixed_execute
except ImportError:
    from fx_baselines import fixed_execute


_COLUMNS = (
    'pair', 'strategy', 'candidate_id', 'original_episode', 'signal_idx', 'signal_date',
    'entry_idx', 'entry_date', 'actual_entry_idx', 'actual_entry_date',
    'position', 'reject_position', 'condition', 'accepted', 'rejected', 'status',
    'entry_pending', 'original_end_signal_idx', 'original_end_signal_date',
    'original_exit_idx', 'original_exit_date', 'actual_exit_idx', 'actual_exit_date',
    'original_exit_pending', 'exit_pending',
)


def _validate(prices, dates, base_run, features, spec):
    if not isinstance(spec, dict) or set(spec) != {'id', 'reject_position'}:
        raise ValueError('Direction spec must contain exactly id and reject_position.')
    if not isinstance(spec['id'], str) or not spec['id'].strip():
        raise ValueError('A nonempty candidate id is required.')
    rejected = spec['reject_position']
    if isinstance(rejected, (bool, np.bool_)) or not isinstance(rejected, (int, np.integer)) or rejected not in (-1, 1):
        raise ValueError('reject_position must be integer -1 or +1, never bool.')
    prices, dates = np.asarray(prices, dtype=float), pd.DatetimeIndex(dates)
    if prices.ndim != 1 or not len(prices) or len(prices) != len(dates) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError('Prices must be a positive finite vector aligned with dates.')
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or dates.tz is not None:
        raise ValueError('Dates must be unique increasing timezone-naive observations.')
    # Feature values are deliberately irrelevant to this direction-only rule.
    # If the caller supplies a dated frame, reject a mismatched calendar.
    if isinstance(features, pd.DataFrame) and 'date' in features:
        if not pd.DatetimeIndex(pd.to_datetime(features.date)).equals(dates):
            raise ValueError('Feature dates differ from the supplied price calendar.')
    if not isinstance(base_run, dict):
        raise ValueError('BASE must be a fixed_execute run dictionary.')
    required = {'target', 'executed', 'held', 'gross_return', 'net_return', 'one_way_cost_bp', 'initial_capital_usd'}
    if not required.issubset(base_run):
        raise ValueError('BASE is missing execution, return, cost or capital fields.')
    original = np.asarray(base_run['target'])
    if original.shape != prices.shape or not np.isin(original, [-1, 0, 1]).all():
        raise ValueError('BASE targets must be aligned signed unit positions.')
    if float(base_run['one_way_cost_bp']) != 0.:
        raise ValueError('Direction research requires zero costs.')
    capital = float(base_run['initial_capital_usd'])
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError('Initial capital must be positive and finite.')
    expected = fixed_execute(prices, original, cost_bp=0., capital=capital)
    for field in ('executed', 'held', 'gross_return', 'net_return', 'spot_return', 'turnover', 'equity',
                  'cost_fraction_post_pnl', 'transaction_cost_usd', 'equity_before_usd',
                  'equity_after_price_usd', 'price_pnl_usd', 'net_pnl_usd'):
        if field not in base_run:
            continue  # Optional diagnostics may be absent in an otherwise valid run.
        actual = np.asarray(base_run[field])
        if actual.shape != prices.shape or not np.allclose(actual, expected[field], atol=1e-12, rtol=0):
            raise ValueError('BASE ' + field + ' differs from the zero-cost next-close path.')
    if 'dates' in base_run and not pd.DatetimeIndex(base_run['dates']).equals(dates):
        raise ValueError('BASE dates differ from the supplied calendar.')
    return prices, dates, original.astype(np.int8, copy=False), int(rejected), capital


def run_direction_filter(prices, dates, base_run, features, spec, pair='', strategy=''):
    """Return (zero-cost run, one audit row per original nonzero TARGET segment).

    Rejected directions stay flat for the complete original episode. Accepted
    directions keep their original entry and natural exit, including reversal
    timing. All nullable fill indices require an observed execution close; an
    open tail is never forcibly closed and a final signal is never a fake fill.
    """
    prices, dates, original, rejected, capital = _validate(prices, dates, base_run, features, spec)
    target = np.where(original == rejected, 0, original).astype(np.int8)
    # Future segment ends are metadata only; the target mask above is pointwise.
    changes = np.flatnonzero(original != np.r_[0, original[:-1]])
    rows = []
    for number, start in enumerate(changes):
        if original[start] == 0:
            continue
        start = int(start)
        end = int(changes[number + 1]) if number + 1 < len(changes) else None
        entry = start + 1 if start + 1 < len(dates) else None
        natural_exit = end + 1 if end is not None and end + 1 < len(dates) else None
        accepted = int(original[start]) != rejected
        original_exit_pending = end is not None and natural_exit is None
        row = {
            'pair': pair, 'strategy': strategy, 'candidate_id': spec['id'],
            'original_episode': len(rows), 'signal_idx': start, 'signal_date': dates[start],
            'position': int(original[start]), 'reject_position': rejected,
            'condition': f'position=={rejected:+d}', 'accepted': accepted, 'rejected': not accepted,
            'status': 'KEEP' if accepted else 'REJECT', 'entry_pending': entry is None,
            'original_exit_pending': original_exit_pending, 'exit_pending': accepted and original_exit_pending,
        }
        for name, value in (('entry', entry), ('actual_entry', entry if accepted else None),
                            ('original_end_signal', end), ('original_exit', natural_exit),
                            ('actual_exit', natural_exit if accepted else None)):
            row[name + '_idx'] = value if value is not None else pd.NA
            row[name + '_date'] = dates[value] if value is not None else pd.NaT
        rows.append(row)
    decisions = pd.DataFrame(rows, columns=_COLUMNS)
    for field in ('entry_idx', 'actual_entry_idx', 'original_end_signal_idx', 'original_exit_idx', 'actual_exit_idx'):
        decisions[field] = decisions[field].astype('Int64')
    for field in ('accepted', 'rejected', 'entry_pending', 'original_exit_pending', 'exit_pending'):
        decisions[field] = decisions[field].astype(bool)
    run = fixed_execute(prices, target, cost_bp=0., capital=capital)
    run['dates'], run['candidate_id'] = dates, spec['id']
    if 'account_start_date' in base_run:
        run['account_start_date'] = base_run['account_start_date']
    return run, decisions
