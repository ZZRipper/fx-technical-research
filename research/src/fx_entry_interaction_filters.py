"""Reject an original entry only when two fixed, known entry states both match.

Both component diagnoses are obtained from the SAME unfiltered BASE path. Their
filtered runs are discarded: applying filters sequentially would implement OR
rejection and contaminate the second condition's trailing BASE-PnL features.
The fixed state bins, timing and provenance guards remain in the frozen modules.
"""
import numpy as np
import pandas as pd

try:
    from .fx_baselines import fixed_execute
    from .fx_is_states import STATE_BINS
    from .fx_state_filters import run_state_filter
except ImportError:
    from fx_baselines import fixed_execute
    from fx_is_states import STATE_BINS
    from fx_state_filters import run_state_filter


def _conditions(spec):
    if not isinstance(spec, dict) or set(spec) != {'id', 'conditions'}:
        raise ValueError('Interaction spec must contain exactly id and conditions.')
    if not isinstance(spec['id'], str) or not spec['id'].strip():
        raise ValueError('A nonempty candidate id is required.')
    conditions = spec['conditions']
    if not isinstance(conditions, list) or len(conditions) != 2:
        raise ValueError('Exactly two entry-state conditions are required.')
    for item in conditions:
        if (not isinstance(item, dict) or set(item) != {'feature', 'state'} or
                not isinstance(item.get('feature'), str) or item['feature'] not in STATE_BINS):
            raise ValueError('Each condition requires exactly one registered feature and state.')
        state = item['state']
        if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)) or not 0 <= state <= 3:
            raise ValueError('State must be an integer 0..3; UNKNOWN cannot be rejected.')
    if conditions[0]['feature'] == conditions[1]['feature']:
        raise ValueError('Interaction features must be different.')
    return sorted(({'feature': item['feature'], 'state': int(item['state'])} for item in conditions),
                  key=lambda item: item['feature'])


def run_entry_interaction_filter(prices, dates, base_run, features, spec, pair='', strategy=''):
    """Return (zero-cost run, original-episode decisions with two state diagnoses).

    Each condition is tested at the original entry signal close. Unknown values
    retain the episode; rejection requires both finite state codes to match.
    Rejected episodes remain flat until their original TARGET changes. Original
    exits and final pending entry instructions retain their next-close timing.
    """
    conditions = _conditions(spec)
    component_rows = []
    for number, item in enumerate(conditions, 1):
        _, rows = run_state_filter(prices, dates, base_run, features,
                                   {'id': spec['id'] + f'__component_{number}', **item}, pair, strategy)
        component_rows.append(rows)
    left, right = component_rows
    episode_fields = ['original_episode', 'signal_idx', 'signal_date', 'position', 'entry_idx',
                      'entry_date', 'entry_pending', 'original_end_signal_idx', 'original_end_signal_date']
    if left.signal_idx.duplicated().any() or right.signal_idx.duplicated().any():
        raise AssertionError('Component diagnoses must be unique by original entry signal.')
    pd.testing.assert_frame_equal(left[episode_fields], right[episode_fields], check_exact=True)
    known = left.state.ge(0) & right.state.ge(0)
    rejected = known & left.state.eq(conditions[0]['state']) & right.state.eq(conditions[1]['state'])
    decisions = left.drop(columns=['feature', 'rejected_state', 'feature_value', 'state']).copy()
    decisions['candidate_id'] = spec['id']
    decisions['condition'] = '&'.join(f"{item['feature']}={item['state']}" for item in conditions)
    decisions['conditions_known'] = known.to_numpy(bool)
    decisions['rejected'] = rejected.to_numpy(bool)
    decisions['accepted'] = ~rejected.to_numpy(bool)
    decisions['status'] = np.where(~known, 'UNKNOWN', np.where(rejected, 'REJECT', 'KEEP'))
    decisions['actual_entry_idx'] = decisions.entry_idx.where(decisions.accepted & ~decisions.entry_pending).astype('Int64')
    for number, (item, rows) in enumerate(zip(conditions, component_rows), 1):
        prefix = f'condition_{number}_'
        decisions[prefix + 'feature'] = item['feature']
        decisions[prefix + 'rejected_state'] = item['state']
        decisions[prefix + 'value'] = rows.feature_value.to_numpy(float)
        decisions[prefix + 'state'] = rows.state.to_numpy(int)
        decisions[prefix + 'matches'] = rows.state.eq(item['state']).to_numpy(bool)

    original = np.asarray(base_run['target'], dtype=np.int8)
    target = original.copy()
    rejected_signals = set(decisions.loc[decisions.rejected, 'signal_idx'].astype(int))
    blocked = False
    for index, position in enumerate(original):
        if position != (original[index - 1] if index else 0):
            blocked = index in rejected_signals
        if blocked:
            target[index] = 0
    run = fixed_execute(prices, target, cost_bp=0., capital=float(base_run.get('initial_capital_usd', 100000.)))
    run['dates'], run['candidate_id'] = pd.DatetimeIndex(dates), spec['id']
    if 'account_start_date' in base_run:
        run['account_start_date'] = base_run['account_start_date']
    return run, decisions
