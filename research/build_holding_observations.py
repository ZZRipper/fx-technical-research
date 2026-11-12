"""Observe fixed holding checkpoints in all 180 unfiltered IS accounts.

This module never simulates a filtered strategy or ranks a fitted exit rule.
Known checkpoint information is separate from closed-episode forward labels.
Only D1/D2/D enter the initial repeated-negative-residual fact table; later
IS blocks are saved for the subsequently declared research protocol.
"""
from pathlib import Path
import argparse
from itertools import combinations
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_baselines import CURRENCIES, STRATEGIES, RULES, fixed_execute
from fx_filter_evaluation import simple_ledger

ROOT = Path(__file__).resolve().parent
AGES = (5, 10, 20, 40, 63)
BLOCKS = (
    ('D1', '2005-01-03', '2009-12-31'),
    ('D2', '2010-01-01', '2014-12-31'),
    ('D', '2005-01-03', '2014-12-31'),
    ('V', '2015-01-01', '2019-12-31'),
    ('C', '2020-01-01', '2023-05-29'),
)
TOL_PCT = 1e-10
CONDITION_ORDER = ('pnl_negative', 'pnl_positive', 'giveback_half', 'recent5_negative', 'opposite20')
CONDITIONS = [('ALL', ())] + [(name, (name,)) for name in CONDITION_ORDER] + [
    ('&'.join(items), items) for items in combinations(CONDITION_ORDER, 2)
    if items != ('pnl_negative', 'pnl_positive')
]
KNOWN_FIELDS = (
    'known_cumulative_simple_pnl_pct', 'known_running_peak_pct', 'known_drawdown_from_peak_pp',
    'known_giveback_fraction', 'known_recent5_signed_return_pct', 'entry_signal_daily_vol20_pct',
    'current_daily_vol20_pct', 'current_vol_ratio_20_63', 'current_direction_return20_pct',
    'current_direction_return63_pct',
)
CHECKPOINT_COLUMNS = [
    'pair', 'strategy', 'family', 'trade_id', 'position', 'entry_signal_idx', 'entry_idx', 'original_exit_idx',
    'entry_signal_date', 'entry_date', 'original_exit_date', 'original_closed', 'original_eligible_closed',
    'checkpoint_age', 'checkpoint_idx', 'checkpoint_signal_date', 'hypothetical_exit_idx', 'hypothetical_exit_date',
    'original_active_at_checkpoint', 'technical_exit_already_signaled', 'next_close_observed',
    'risk_exit_pending', 'actionable_checkpoint', 'residual_observed_closed',
    *KNOWN_FIELDS, *CONDITION_ORDER,
    'label_next_close_return_pct', 'label_remaining_observations', 'label_remaining_simple_return_pct',
    'label_original_trade_simple_return_pct', 'label_known_fill_remaining_reconciliation_error_pp',
]


def condition_values(pnl, peak, recent5, opposite20):
    return {
        'pnl_negative': bool(pnl < -TOL_PCT),
        'pnl_positive': bool(pnl > TOL_PCT),
        'giveback_half': bool(peak > TOL_PCT and pnl <= .5 * peak + TOL_PCT),
        'recent5_negative': bool(recent5 < -TOL_PCT) if np.isfinite(recent5) else None,
        'opposite20': bool(opposite20 < -TOL_PCT) if np.isfinite(opposite20) else None,
    }


def build_checkpoints(run, dates, features, ledger, pair, strategy):
    """Known fields stop at m=entry+age; forward residual starts at m+2.

    A close signal m fills at close m+1 and still earns return m+1. Closed
    residual labels end at the actual original exit. An open original episode
    never receives a fabricated terminal residual. Existing BASE exit/reversal
    signals have priority and are recorded as non-actionable observations.
    """
    dates = pd.DatetimeIndex(dates)
    returns = np.asarray(run['gross_return'], float)
    if returns.ndim != 1 or len(returns) != len(dates) or len(features) != len(dates):
        raise AssertionError('Returns, dates, and causal features must align.')
    if not dates.equals(pd.DatetimeIndex(features.date)) or not np.isfinite(returns).all():
        raise AssertionError('Invalid causal feature calendar or returns.')
    targets, executed = np.asarray(run['target']), np.asarray(run['executed'])
    prefix = np.r_[0., np.cumsum(returns)]
    vol = features.realized_vol_20_pct.to_numpy(float)
    ratio = features.vol_ratio_20_63.to_numpy(float)
    ret20, ret63 = features.spot_return_20_pct.to_numpy(float), features.spot_return_63_pct.to_numpy(float)
    records = []
    validations = {'known_direct_slice_checks': 0, 'remaining_direct_slice_checks': 0,
                   'full_trade_reconciliation_checks': 0, 'max_remaining_slice_error_pp': 0.,
                   'max_full_trade_reconciliation_error_pp': 0.}
    for trade in ledger.itertuples(index=False):
        entry, end = int(trade.entry_idx), int(trade.exit_idx)
        direction = int(trade.position)
        local_returns = returns[entry + 1:end + 1]
        local_cumulative = np.r_[0., np.cumsum(local_returns)] * 100
        local_peak = np.maximum.accumulate(local_cumulative)
        for age in AGES:
            m = entry + age
            if m > end:
                continue
            active = bool(not trade.closed or m < end)
            technical_exit = bool(active and targets[m] != direction)
            next_observed = m + 1 < len(dates)
            actionable = bool(active and not technical_exit and next_observed)
            pending = bool(active and not technical_exit and not next_observed)
            pnl, peak = float(local_cumulative[age]), float(local_peak[age])
            recent5 = float(returns[m - 4:m + 1].sum() * 100)
            direction20, direction63 = direction * ret20[m], direction * ret63[m]
            known_direct = float(returns[entry + 1:m + 1].sum() * 100)
            peak_direct = float(max(0., *(returns[entry + 1:j + 1].sum() * 100 for j in range(entry + 1, m + 1))))
            if not (np.isclose(pnl, known_direct, atol=1e-10, rtol=0) and np.isclose(peak, peak_direct, atol=1e-10, rtol=0)):
                raise AssertionError('Known checkpoint fields disagree with past-only direct slices.')
            if active and executed[m] != direction:
                raise AssertionError('Original episode is not active at the checkpoint close.')
            validations['known_direct_slice_checks'] += 1
            observed_closed = bool(trade.eligible_closed and actionable and m + 2 <= end)
            next_return, residual, count, reconcile = np.nan, np.nan, 0, np.nan
            if trade.eligible_closed and active and next_observed:
                next_return = float(returns[m + 1] * 100)
                residual = float((prefix[end + 1] - prefix[m + 2]) * 100)
                direct = float(returns[m + 2:end + 1].sum() * 100)
                count = max(0, end - m - 1)
                error = abs(residual - direct)
                validations['remaining_direct_slice_checks'] += 1
                validations['max_remaining_slice_error_pp'] = max(validations['max_remaining_slice_error_pp'], error)
                if error > 1e-8:
                    raise AssertionError('Remaining BASE return prefix sum differs from direct slice.')
                reconcile = pnl + next_return + residual - float(trade.gross_simple_return_pct)
                validations['full_trade_reconciliation_checks'] += 1
                validations['max_full_trade_reconciliation_error_pp'] = max(validations['max_full_trade_reconciliation_error_pp'], abs(reconcile))
                if abs(reconcile) > 1e-8:
                    raise AssertionError('Known PnL + exit-fill return + remaining PnL does not reconcile.')
            records.append({
                'pair': pair, 'strategy': strategy, 'family': RULES[strategy][1], 'trade_id': int(trade.trade_id),
                'position': direction, 'entry_signal_idx': int(trade.signal_idx), 'entry_idx': entry, 'original_exit_idx': end,
                'entry_signal_date': trade.signal_date, 'entry_date': trade.entry_date, 'original_exit_date': trade.exit_date,
                'original_closed': bool(trade.closed), 'original_eligible_closed': bool(trade.eligible_closed),
                'checkpoint_age': age, 'checkpoint_idx': m, 'checkpoint_signal_date': dates[m],
                'hypothetical_exit_idx': m + 1 if next_observed else pd.NA,
                'hypothetical_exit_date': dates[m + 1] if next_observed else pd.NaT,
                'original_active_at_checkpoint': active, 'technical_exit_already_signaled': technical_exit,
                'next_close_observed': next_observed, 'risk_exit_pending': pending,
                'actionable_checkpoint': actionable, 'residual_observed_closed': observed_closed,
                'known_cumulative_simple_pnl_pct': pnl, 'known_running_peak_pct': peak,
                'known_drawdown_from_peak_pp': peak - pnl, 'known_giveback_fraction': (peak - pnl) / peak if peak > TOL_PCT else np.nan,
                'known_recent5_signed_return_pct': recent5,
                'entry_signal_daily_vol20_pct': vol[int(trade.signal_idx)] / np.sqrt(252.),
                'current_daily_vol20_pct': vol[m] / np.sqrt(252.), 'current_vol_ratio_20_63': ratio[m],
                'current_direction_return20_pct': direction20, 'current_direction_return63_pct': direction63,
                **condition_values(pnl, peak, recent5, direction20),
                'label_next_close_return_pct': next_return, 'label_remaining_observations': count,
                'label_remaining_simple_return_pct': residual,
                'label_original_trade_simple_return_pct': float(trade.gross_simple_return_pct) if trade.eligible_closed else np.nan,
                'label_known_fill_remaining_reconciliation_error_pp': reconcile,
            })
    frame = pd.DataFrame(records, columns=CHECKPOINT_COLUMNS)
    for column in CONDITION_ORDER:
        frame[column] = frame[column].astype('boolean')
    frame['hypothetical_exit_idx'] = frame.hypothetical_exit_idx.astype('Int64')
    return frame, validations


def residual_statistics(values):
    values = np.asarray(values, float)
    negatives = np.sort(values[values < -TOL_PCT])
    total = float(values.sum())
    gross_loss = float(-negatives.sum())
    return {
        'closed_n': len(values), 'remaining_return_sum_pct': total,
        'remaining_return_mean_pct': float(values.mean()) if len(values) else np.nan,
        'remaining_return_median_pct': float(np.median(values)) if len(values) else np.nan,
        'remaining_negative_n': len(negatives), 'remaining_positive_n': int((values > TOL_PCT).sum()),
        'remaining_flat_n': int((np.abs(values) <= TOL_PCT).sum()),
        'remaining_negative_rate': len(negatives) / len(values) if len(values) else np.nan,
        'remaining_without_worst1_pct': total - float(negatives[:1].sum()),
        'remaining_without_worst3_pct': total - float(negatives[:3].sum()),
        'worst_remaining_return_pct': float(negatives[0]) if len(negatives) else np.nan,
        'worst_loss_share_of_gross_losses': float(-negatives[0] / gross_loss) if gross_loss else np.nan,
    }


def summarize_account(ledger, checkpoints, pair, strategy):
    summaries, coverage = [], []
    for age in AGES:
        age_rows = checkpoints.loc[checkpoints.checkpoint_age.eq(age)]
        for block, start, end in BLOCKS:
            entered = ledger.loc[ledger.entry_date.between(start, end)]
            observations = age_rows.loc[age_rows.entry_date.between(start, end)]
            known_in_block = observations.loc[observations.checkpoint_signal_date.le(end)]
            eligible = observations.loc[observations.residual_observed_closed & observations.original_exit_date.le(end)]
            coverage.append({'pair': pair, 'strategy': strategy, 'family': RULES[strategy][1], 'checkpoint_age': age, 'block': block,
                'entered_episode_n': len(entered), 'checkpoint_record_n': len(observations),
                'checkpoint_signal_in_block_n': len(known_in_block), 'actionable_in_block_n': int(known_in_block.actionable_checkpoint.sum()),
                'complete_actionable_residual_n': len(eligible), 'closed_before_checkpoint_n': int(entered.holding_observations.lt(age).sum()),
                'technical_exit_priority_n': int(known_in_block.technical_exit_already_signaled.sum()),
                'already_closed_at_checkpoint_n': int((~known_in_block.original_active_at_checkpoint).sum()),
                'next_close_unobserved_n': int((~known_in_block.next_close_observed).sum()),
                'risk_exit_pending_n': int(known_in_block.risk_exit_pending.sum()),
                'cross_period_episode_n': int(entered.exit_date.gt(end).sum()),
                'open_at_IS_end_episode_n': int((~entered.closed).sum()),
                'checkpoint_after_block_end_n': int(observations.checkpoint_signal_date.gt(end).sum())})
            for condition, members in CONDITIONS:
                known = np.ones(len(eligible), dtype=bool)
                matched = np.ones(len(eligible), dtype=bool)
                for member in members:
                    known &= eligible[member].notna().to_numpy()
                    matched &= eligible[member].fillna(False).to_numpy(dtype=bool)
                selected = eligible.loc[matched & known]
                summaries.append({'pair': pair, 'strategy': strategy, 'family': RULES[strategy][1],
                    'checkpoint_age': age, 'condition': condition, 'block': block,
                    **residual_statistics(selected.label_remaining_simple_return_pct),
                    'actionable_cohort_n': len(eligible), 'condition_unknown_n': int((~known).sum()),
                    'known_pnl_mean_pct': float(selected.known_cumulative_simple_pnl_pct.mean()) if len(selected) else np.nan,
                    'known_peak_mean_pct': float(selected.known_running_peak_pct.mean()) if len(selected) else np.nan,
                    'entry_daily_vol20_mean_pct': float(selected.entry_signal_daily_vol20_pct.mean()) if len(selected) else np.nan,
                    'current_vol_ratio_mean': float(selected.current_vol_ratio_20_63.mean()) if len(selected) else np.nan,
                })
    return summaries, coverage


def discovery_facts(summary):
    """Never inspect V/C outcomes to construct the initial fact table."""
    discovery = summary.loc[summary.block.isin(['D1', 'D2', 'D']) & summary.condition.ne('ALL')]
    rows = []
    for keys, sub in discovery.groupby(['pair', 'strategy', 'checkpoint_age', 'condition'], sort=True):
        sub = sub.set_index('block')
        a, b, d = sub.loc['D1'], sub.loc['D2'], sub.loc['D']
        repeated_negative = a.remaining_return_sum_pct < -TOL_PCT and b.remaining_return_sum_pct < -TOL_PCT
        support10 = repeated_negative and min(a.closed_n, b.closed_n) >= 10
        support20 = repeated_negative and min(a.closed_n, b.closed_n) >= 20
        if not support10:
            continue
        row = dict(zip(['pair', 'strategy', 'checkpoint_age', 'condition'], keys))
        row.update(repeated_negative_support10=True, repeated_negative_support20=bool(support20), concentration_is_diagnostic_only=True)
        for block, record in [('D1', a), ('D2', b), ('D', d)]:
            for column in ('closed_n', 'remaining_return_sum_pct', 'remaining_return_mean_pct', 'remaining_return_median_pct',
                           'remaining_negative_n', 'remaining_negative_rate', 'remaining_without_worst1_pct', 'remaining_without_worst3_pct',
                           'worst_loss_share_of_gross_losses'):
                row[block + '_' + column] = record[column]
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_selfcheck():
    dates = pd.date_range('2010-01-01', periods=10)
    changes = np.array([0., 0., .05, -.02, .01, -.03, -.02, .04, -.03, -.02])
    prices = 100 * np.cumprod(1 + changes)
    target = np.r_[np.ones(8, dtype=int), 0, 0]
    run = fixed_execute(prices, target)
    features = pd.DataFrame({'date': dates, 'realized_vol_20_pct': np.sqrt(252.), 'vol_ratio_20_63': 1.,
                             'spot_return_20_pct': -2., 'spot_return_63_pct': -1.})
    trades = simple_ledger(run, dates)
    observations, validation = build_checkpoints(run, dates, features, trades, 'JPY', 'BB_MR_20')
    row = observations.iloc[0]
    assert row.checkpoint_age == 5 and row.checkpoint_idx == 6 and row.hypothetical_exit_idx == 7
    assert np.isclose(row.known_cumulative_simple_pnl_pct, -1.)
    assert np.isclose(row.known_running_peak_pct, 5.) and np.isclose(row.known_drawdown_from_peak_pp, 6.)
    assert np.isclose(row.label_next_close_return_pct, 4.) and np.isclose(row.label_remaining_simple_return_pct, -5.)
    assert row.actionable_checkpoint and row.giveback_half and row.opposite20
    modified = dict(run)
    modified['gross_return'] = run['gross_return'].copy()
    modified['gross_return'][8:] = [-.9, -.7]
    future_features = features.copy()
    future_features.loc[7:, ['realized_vol_20_pct', 'vol_ratio_20_63', 'spot_return_20_pct', 'spot_return_63_pct']] = 999.
    future_trades = trades.copy()
    future_trades['gross_simple_return_pct'] = modified['gross_return'][2:10].sum() * 100
    changed, _ = build_checkpoints(modified, dates, future_features, future_trades, 'JPY', 'BB_MR_20')
    pd.testing.assert_frame_equal(observations[list(KNOWN_FIELDS) + list(CONDITION_ORDER)], changed[list(KNOWN_FIELDS) + list(CONDITION_ORDER)])
    assert not np.isclose(changed.label_remaining_simple_return_pct.iloc[0], row.label_remaining_simple_return_pct)
    # A known BASE close at the checkpoint must supersede any risk signal.
    exit_target = np.r_[np.ones(6, dtype=int), np.zeros(4, dtype=int)]
    exit_run = fixed_execute(prices, exit_target)
    priority, _ = build_checkpoints(exit_run, dates, features, simple_ledger(exit_run, dates), 'JPY', 'BB_MR_20')
    assert priority.iloc[0].technical_exit_already_signaled and not priority.iloc[0].actionable_checkpoint
    # An open terminal checkpoint has no fabricated residual label or fill.
    open_dates = dates[:7]
    open_run = fixed_execute(prices[:7], np.ones(7, dtype=int))
    pending, _ = build_checkpoints(open_run, open_dates, features.iloc[:7], simple_ledger(open_run, open_dates), 'JPY', 'BB_MR_20')
    assert pending.iloc[0].risk_exit_pending and pd.isna(pending.iloc[0].label_remaining_simple_return_pct)
    assert not pending.iloc[0].residual_observed_closed
    return {'status': 'passed', 'checks': ['hand_calculated_known_peak_fill_residual', 'future_mutation_known_feature_invariance',
                                          'technical_exit_priority', 'open_terminal_checkpoint_has_no_outcome']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/holding_observations_v3')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    selfcheck = synthetic_selfcheck()
    if args.self_test:
        print(selfcheck)
        return
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Observation output already exists; never replace a campaign.')
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'created_utc': core.now(), 'version': 'holding_observations_v3', 'compute_start': core.IS_START, 'compute_end': core.IS_END,
        'cost_bp': 0, 'accounting': 'fixed_initial_capital_simple', 'pairs': CURRENCIES, 'strategies': STRATEGIES, 'accounts': 180,
        'checkpoint_ages': AGES, 'blocks': BLOCKS, 'condition_order': CONDITION_ORDER,
        'conditions': [{'condition': label, 'members': members} for label, members in CONDITIONS], 'max_condition_count': 2,
        'tolerance_pct': TOL_PCT,
        'clock': 'Entry close e; exactly age returns earned at checkpoint close m=e+age. Exit signal m fills at m+1 close, still earning m+1. Remaining BASE label is sum of original returns m+2 through original exit inclusive.',
        'known_fields': KNOWN_FIELDS,
        'known_running_peak': 'max(0, cumulative simple PnL from first earned return through checkpoint); includes the zero initial value.',
        'known_giveback': '(running_peak - known_PnL)/running_peak only when running_peak > tolerance.',
        'entry_volatility': '20-quote sample daily-return standard deviation as known at original entry SIGNAL, annualized source divided by sqrt(252), percent units.',
        'current_vol_ratio': 'Current 20-quote annualized simple-return standard deviation divided by current 63-quote value.',
        'directional_returns': 'Original episode position multiplied by current price change over 20 or 63 quotes, in percent.',
        'condition_definitions': {
            'pnl_negative': 'known cumulative simple PnL < -tolerance_pct', 'pnl_positive': 'known cumulative simple PnL > tolerance_pct',
            'giveback_half': 'known peak > tolerance_pct AND known PnL <= 0.5 * known peak + tolerance_pct',
            'recent5_negative': 'sum of already-earned original signed returns over last 5 quotes < -tolerance_pct',
            'opposite20': 'original direction * current 20-quote spot return < -tolerance_pct'},
        'actionable': 'Original episode remains active at checkpoint close, BASE target[checkpoint] still equals its direction, and next-close fill exists. Natural exit/reversal signals take priority.',
        'residual_cohort': 'Only complete original eligible-closed episodes whose entry and true exit both lie within each block, with an actionable checkpoint and at least one return after the hypothetical exit fill. Open/crossing/non-actionable records are preserved and counted separately.',
        'sampling': 'One observation per original episode per exact age. Different ages and overlapping conditions are correlated views, never pooled as independent trades. D overlaps D1/D2.',
        'discovery_reporting': 'Only D1/D2/D used initially. Two fixed support levels: each D1/D2 has >=10 or >=20 complete episodes and both residual sums negative. Removing worst 1/3 residuals is diagnostic only, never a pass gate.',
        'later_blocks': 'V/C stored for subsequent declared research; not used to nominate or rank initial patterns.',
        'not_a_backtest': 'No filtered path, optimized threshold, p-value, OOS evaluation, fitted exit or research-success claim is produced.',
        'oos_computed': False, 'source_sha256': {**core.source_hashes(), 'build_holding_observations.py': core.sha(__file__)},
        'input_sha256': core.input_hashes(), 'synthetic_selfcheck': selfcheck,
    }
    core.save_json(output / 'holding_observation_protocol.json', protocol)
    prepared = core.prepare(core.IS_END)
    ledgers, checkpoints, summaries, coverages, validations = [], [], [], [], []
    for pair in CURRENCIES:
        values, dates, features, targets = prepared[pair]
        if dates.max() > pd.Timestamp(core.IS_END):
            raise AssertionError('IS cutoff was violated.')
        for strategy in STRATEGIES:
            run = fixed_execute(values, targets[strategy], cost_bp=0.)
            ledger = simple_ledger(run, dates, cost_bp=0.).assign(pair=pair, strategy=strategy, family=RULES[strategy][1])
            records, checks = build_checkpoints(run, dates, features, ledger, pair, strategy)
            summary, coverage = summarize_account(ledger, records, pair, strategy)
            ledgers.append(ledger)
            checkpoints.append(records)
            summaries.extend(summary)
            coverages.extend(coverage)
            validations.append({'pair': pair, 'strategy': strategy, **checks})
        print('Holding observations complete:', pair, flush=True)
    ledger, checkpoints = pd.concat(ledgers, ignore_index=True), pd.concat(checkpoints, ignore_index=True)
    summary, coverage = pd.DataFrame(summaries), pd.DataFrame(coverages)
    facts = discovery_facts(summary)
    expected_accounts = {(p, s) for p in CURRENCIES for s in STRATEGIES}
    assert set(zip(ledger.pair, ledger.strategy)) == expected_accounts
    assert set(zip(summary.pair, summary.strategy)) == expected_accounts
    assert not checkpoints.duplicated(['pair', 'strategy', 'trade_id', 'checkpoint_age']).any()
    assert checkpoints.checkpoint_signal_date.max() <= pd.Timestamp(core.IS_END)
    assert checkpoints.loc[~checkpoints.original_closed, 'label_remaining_simple_return_pct'].isna().all()
    assert len(summary) == 180 * len(AGES) * len(BLOCKS) * len(CONDITIONS)
    ledger.to_csv(output / 'is_base_trade_ledger.csv.gz', index=False)
    checkpoints.to_csv(output / 'is_holding_checkpoints.csv.gz', index=False)
    summary.to_csv(output / 'holding_state_summary.csv', index=False)
    coverage.to_csv(output / 'holding_coverage.csv', index=False)
    facts.to_csv(output / 'discovery_repeated_negative_patterns.csv', index=False)
    core.save_json(output / 'discovery_repeated_negative_patterns.json', {
        'blocks_consumed': ['D1', 'D2', 'D'], 'support10_patterns': len(facts),
        'support20_patterns': int(facts.repeated_negative_support20.sum()) if len(facts) else 0,
        'not_independent_tests': 'Repeated ages, overlapping conditions, currencies and strategies are not independent trials.',
        'concentration_is_diagnostic_only': True, 'rows': facts.to_dict('records')})
    core.assert_hashes(protocol['source_sha256'])
    core.assert_hashes(protocol['input_sha256'])
    outputs = {p.name: core.sha(p) for p in output.iterdir() if p.name != 'holding_observation_protocol.json'}
    core.save_json(output / 'holding_observation_validation.json', {
        'status': 'passed', 'finished_utc': core.now(), 'protocol_sha256': core.sha(output / 'holding_observation_protocol.json'),
        'accounts': 180, 'base_episode_rows': len(ledger), 'checkpoint_rows': len(checkpoints),
        'summary_rows': len(summary), 'coverage_rows': len(coverage), 'compute_cutoff': core.IS_END, 'oos_computed': False,
        'unique_episode_age': True, 'open_labels_missing': True, 'source_integrity_pass': True,
        'synthetic_selfcheck': selfcheck, 'account_arithmetic_checks': validations,
        'known_direct_slice_checks': sum(r['known_direct_slice_checks'] for r in validations),
        'remaining_direct_slice_checks': sum(r['remaining_direct_slice_checks'] for r in validations),
        'full_trade_reconciliation_checks': sum(r['full_trade_reconciliation_checks'] for r in validations),
        'max_remaining_slice_error_pp': max(r['max_remaining_slice_error_pp'] for r in validations),
        'max_full_trade_reconciliation_error_pp': max(r['max_full_trade_reconciliation_error_pp'] for r in validations),
        'output_sha256': outputs,
    })
    print('Complete:', len(ledger), 'BASE episodes;', len(checkpoints), 'checkpoints;', len(summary), 'summary rows;',
          len(facts), 'D-only support10 patterns;', int(facts.repeated_negative_support20.sum()) if len(facts) else 0,
          'D-only support20 patterns.', flush=True)


if __name__ == '__main__':
    main()
