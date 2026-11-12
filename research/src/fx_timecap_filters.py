"""Unconditional time-cap exits at one fixed earned-return age per episode.

The holding engine supplies its validated causal checkpoint diagnostics. Its
temporary pnl_positive predicate and filtered execution are deliberately NOT
used as decisions here: every evaluated checkpoint exits, including zero or
negative PnL. An independent forward TARGET scan implements the ALL rule.
No observations, research results, candidate selection or thresholds are loaded.
"""
import numpy as np
import pandas as pd

try:
    from .fx_baselines import fixed_execute
    from .fx_holding_filters import CHECKPOINT_AGES, run_holding_filter
except ImportError:
    from fx_baselines import fixed_execute
    from fx_holding_filters import CHECKPOINT_AGES, run_holding_filter


def run_timecap_filter(prices, dates, base_run, features, spec, pair='', strategy=''):
    """Return (zero-cost run, holding-schema decisions with condition='ALL').

    Spec contains exactly id and checkpoint_age (5/10/20/40/63). Original signal
    s fills at s+1 and first earns at s+2. At t=s+1+age, exit is signaled only if
    the original TARGET episode is still active; it fills at t+1 and still earns
    that day's return. The episode stays locked until the original TARGET changes.
    Final pending signals are retained without fabricating an execution.
    """
    if not isinstance(spec, dict) or set(spec) != {'id', 'checkpoint_age'}:
        raise ValueError('Time-cap spec must contain exactly id and checkpoint_age.')
    # Reuse the frozen holding engine's input/calendar/cost/age validation and
    # exact-checkpoint episode diagnostics; ignore its conditional output run.
    diagnostic_spec = {**spec, 'condition': 'pnl_positive'}
    _, decisions = run_holding_filter(prices, dates, base_run, features, diagnostic_spec, pair, strategy)
    dates = pd.DatetimeIndex(dates)
    original = np.asarray(base_run['target'], dtype=np.int8)
    target = original.copy()
    trigger_indices = set(decisions.loc[decisions.evaluated, 'checkpoint_idx'].astype(int))

    # Only the current original TARGET and the past lock are consulted. Future
    # original_end_signal_idx diagnostics never determine an earlier instruction.
    locked = False
    for t, position in enumerate(original):
        if position != (original[t - 1] if t else 0):
            locked = False
        if t in trigger_indices:
            if position == 0:
                raise AssertionError('A checkpoint cannot trigger after a known technical exit.')
            locked = True
        if locked:
            target[t] = 0

    decisions['condition'] = 'ALL'
    for index in decisions.index[decisions.evaluated]:
        signal = int(decisions.at[index, 'checkpoint_idx'])
        execution = signal + 1
        pending = execution >= len(dates)
        decisions.at[index, 'condition_state'] = 'TRUE'
        decisions.at[index, 'early_exit'] = True
        decisions.at[index, 'risk_signal_idx'] = signal
        decisions.at[index, 'risk_signal_date'] = dates[signal]
        decisions.at[index, 'actual_exit_idx'] = pd.NA if pending else execution
        decisions.at[index, 'actual_exit_date'] = pd.NaT if pending else dates[execution]
        decisions.at[index, 'risk_pending'] = pending
        decisions.at[index, 'exit_pending'] = pending
        decisions.at[index, 'status'] = 'EARLY_EXIT_PENDING' if pending else 'EARLY_EXIT'
    if not decisions.early_exit.equals(decisions.evaluated):
        raise AssertionError('ALL must trigger at every and only evaluated checkpoint.')

    run = fixed_execute(prices, target, cost_bp=0., capital=float(base_run.get('initial_capital_usd', 100000.)))
    run['dates'], run['candidate_id'] = dates, spec['id']
    if 'account_start_date' in base_run:
        run['account_start_date'] = base_run['account_start_date']
    return run, decisions
