"""Independent read-only pre/post audit for the fixed JPY supplement.

Pre-OOS mode hashes source workbooks as bytes but never parses future prices or
opens OOS artifacts. Only a separate audit JSON is written; frozen inputs and
the strict-v2 decision remain unchanged. Tests are not rerun by this auditor.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_state_filter_research import Audit, ROOT, sha, same_number, independent_quality
from fx_baselines import _workbook_rows
from fx_filter_evaluation import SCREEN, OOS_SCREEN, screen_candidates

IDENTIFIER = 'JPY|BB_MR_20|STATE_efficiency_63_1'
SPEC = {'id': 'STATE_efficiency_63_1', 'feature': 'efficiency_63', 'state': 1}
OOS_FILES = ('oos_status.json', 'oos_daily.csv.gz', 'oos_metrics.csv', 'oos_anniversary_metrics.csv',
             'oos_fresh_start_metrics.csv', 'candidate_assessment.csv', 'frozen_full_trade_ledger.csv.gz',
             'frozen_full_episode_decisions.csv.gz')


def ledger_from_executed(calendar, executed, returns):
    """Independent episode reconstruction, with no synthetic terminal close."""
    changes = np.flatnonzero(executed != np.r_[0, executed[:-1]])
    rows = []
    for i, entry in enumerate(changes):
        if executed[entry] == 0:
            continue
        closed = i + 1 < len(changes)
        end = int(changes[i + 1]) if closed else len(calendar) - 1
        values = returns[entry + 1:end + 1]
        rows.append({'signal_idx': int(entry - 1), 'entry_idx': int(entry), 'exit_idx': end,
                     'signal_date': calendar[entry - 1], 'entry_date': calendar[entry], 'exit_date': calendar[end],
                     'position': int(executed[entry]), 'closed': closed, 'eligible_closed': bool(closed and end > entry),
                     'holding_observations': int(end - entry), 'gross_simple_return_pct': float(values.sum() * 100),
                     'trade_return': float(np.prod(1 + values) - 1)})
    return pd.DataFrame(rows)


def window_from_path(calendar, executed, held, returns, ledger, start, end):
    """Independent fixed-principal calendar metrics and complete-trade quality."""
    indices = np.flatnonzero((calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end)))
    first, last = int(indices[0]), int(indices[-1])
    anchor = calendar[max(0, first - 1)]
    years = (calendar[last] - anchor).days / 365.25
    values = returns[indices]
    curve = np.r_[0., np.cumsum(values) * 100]
    entered = ledger.loc[ledger.entry_date.between(calendar[first], calendar[last])]
    closed = entered.loc[entered.eligible_closed & entered.exit_date.le(calendar[last])]
    carry = ledger.loc[ledger.entry_date.lt(calendar[first]) & ledger.exit_date.ge(calendar[first])]
    carry_return = 0.
    for trade in carry.itertuples(index=False):
        begin, stop = max(first, trade.entry_idx + 1), min(last, trade.exit_idx)
        if begin <= stop:
            carry_return += returns[begin:stop + 1].sum() * 100
    quality = independent_quality(closed.gross_simple_return_pct)
    standard_deviation = values.std(ddof=1)
    result = {'start_date': calendar[first], 'end_date': calendar[last], 'anchor_date': anchor,
              'calendar_years': years, 'observations': len(values), 'gross_simple_return_pct': float(values.sum() * 100),
              'gross_annualized_simple_pct': float(values.sum() * 100 / years),
              'gross_max_drawdown_pp': float((curve - np.maximum.accumulate(curve)).min()),
              'gross_compound_return_pct': float((np.prod(1 + values) - 1) * 100),
              'net_sharpe_252': float(values.mean() / standard_deviation * np.sqrt(252)) if standard_deviation > 0 else np.nan,
              'turnover_units': int(np.abs(executed[indices] - held[indices]).sum()), 'cost_drag_pp': 0.,
              'active_fraction': float(np.mean(held[indices] != 0)), 'entry_n': len(entered), 'carry_in_n': len(carry),
              'carry_in_net_return_pp': float(carry_return),
              'censored_entry_n': int((~entered.closed | entered.exit_date.gt(calendar[last])).sum()),
              'low_sample_flag': len(closed) < 20, **quality}
    for source, target in [('gross_simple_return_pct', 'net_simple_return_pct'),
                            ('gross_annualized_simple_pct', 'net_annualized_simple_pct'),
                            ('gross_max_drawdown_pp', 'net_max_drawdown_pp'), ('win_rate', 'net_win_rate'), ('payoff_ratio', 'net_payoff_ratio')]:
        result[target] = result[source]
    return result


class SupplementAudit(Audit):
    def json_path(self, path):
        return json.loads(self.register(path).read_text(encoding='utf-8'))

    def frame_path(self, path):
        return pd.read_csv(self.register(path), float_precision='round_trip')

    def same_frame(self, label, actual, expected, keys):
        actual, expected = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-10)
        except AssertionError as error:
            self.check(label, False, error=str(error))
        else:
            self.check(label, True, rows=len(actual))

    def common(self):
        protocol = self.read_json('protocol.json')
        freeze = self.read_json('frozen_selection.json')
        status = self.read_json('development_status.json')
        diagnostics = self.read_json('supplement_diagnostics.json')
        source = ROOT / protocol['source_campaign']
        old_protocol = self.json_path(source / 'protocol.json')
        old_lock = self.json_path(source / 'validation_locked.json')
        old_freeze = self.json_path(source / 'frozen_selection.json')
        old_status = self.json_path(source / 'development_status.json')
        old_audit = self.json_path(source / 'independent_validation.json')
        for manifest in ('source_sha256', 'input_sha256', 'source_campaign_evidence_sha256'):
            self.check('manifest_agreement:' + manifest, freeze[manifest] == protocol[manifest])
            self.hashes(manifest, protocol[manifest])
        self.hashes('observation', protocol['observation_source_sha256'])
        self.hashes('frozen_IS', freeze['is_output_sha256'], self.campaign)
        self.hashes('original_v2_IS', old_freeze['is_output_sha256'], source)
        self.hashes('original_v2_lock_outputs', old_lock['fit_output_sha256'], source)
        self.check('protocol_freeze_link', sha(self.campaign / 'protocol.json') == freeze['protocol_sha256'] == status['protocol_sha256'])
        self.check('freeze_status_link', sha(self.campaign / 'frozen_selection.json') == status['freeze_sha256'])
        self.check('diagnostics_hash', sha(self.campaign / 'supplement_diagnostics.json') == freeze['supplement_diagnostics_sha256'])
        self.check('v2_freeze_status_link', sha(source / 'frozen_selection.json') == old_status['freeze_sha256'])
        self.check('v2_protocol_lock_links', old_freeze['protocol_sha256'] == old_lock['protocol_sha256'] == sha(source / 'protocol.json')
                   and old_freeze['preconfirmation_lock_sha256'] == sha(source / 'validation_locked.json'))
        self.check('v2_audit_passed', old_audit['status'] == 'passed' and old_audit['checks_passed'] == old_audit['checks_total']
                   and len(old_audit['checks']) == old_audit['checks_total'] and all(r['passed'] for r in old_audit['checks']))
        self.check('v2_audit_source_matches', sha(ROOT / 'verify_state_filter_research.py') == old_audit['audit_script_sha256'])
        self.hashes('audited_test_sources', old_audit['tests']['test_source_sha256'])
        self.hashes('audited_v2_evidence', old_audit['verified_files_sha256'])
        self.check('v2_tests_passed', old_audit['tests']['executed'] and old_audit['tests']['exit_code'] == 0 and old_audit['tests']['tests_run'] == 76)
        self.check('exact_original_SCREEN', protocol['screen'] == old_protocol['screen'] == SCREEN)
        self.check('exact_original_OOS_SCREEN', protocol['oos_screen'] == old_protocol['oos_screen'] == OOS_SCREEN)
        self.check('cost_zero_simple', protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
                   and protocol['accounting'] == 'fixed_initial_capital_simple')
        self.check('dates_unchanged', all(protocol[k] == old_protocol[k] for k in ('is_start', 'is_end', 'oos_start', 'oos_end', 'oos_anniversary_years', 'is_blocks')))
        self.check('original_state_bins', protocol['state_bins'] == old_protocol['state_bins'])
        self.check('sole_designated_scope', protocol['scope'] == {'pairs': ['JPY'], 'strategies': ['BB_MR_20'], 'baseline_accounts': 1}
                   and protocol['designated_candidate_id'] == IDENTIFIER and protocol['candidate_specs'] == [SPEC])
        locked = [r for r in old_lock['candidates'] if r['candidate_id'] == IDENTIFIER]
        self.check('previously_locked_same_spec', len(locked) == 1 and locked[0]['spec'] == SPEC)
        self.check('one_frozen_candidate', len(freeze['candidates']) == freeze['frozen_candidates'] == status['frozen_candidates'] == 1)
        candidate = freeze['candidates'][0]
        self.check('candidate_spec_and_rank_unchanged', candidate['candidate_id'] == IDENTIFIER and candidate['spec'] == SPEC
                   and candidate['preconfirmation_rank'] == locked[0]['preconfirmation_rank'] and candidate['freeze_rank'] == 1)
        self.check('strict_v2_failure_preserved', old_freeze['candidates'] == [] and old_freeze['frozen_candidates'] == old_status['frozen_candidates'] == 0
                   and protocol['strict_v2_pass'] is False and freeze['strict_v2_pass'] is False and candidate['strict_v2_pass'] is False
                   and status['strict_v2_pass'] is False and diagnostics['strict_v2_pass'] is False and diagnostics['strict_v2_failed'] is True)
        self.check('v2_outcomes_not_relabelled', diagnostics['strict_v2_confirmation_failed'] == candidate['strict_v2_confirmation_failed'] == 'outcome_sample'
                   and diagnostics['strict_v2_robustness_failed'] == candidate['strict_v2_robustness_failed'] == 'recent_top3_robust')
        self.check('historical_exposure_disclosed', protocol['prior_holdout_exposure'] == old_protocol['prior_holdout_exposure']
                   and 'not independent untouched OOS' in protocol['prior_holdout_exposure'])
        self.check('supplement_status_passed', status['status'] == 'passed' and status['is_screen_completed'] and status['compute_cutoff'] == '2023-05-29')
        self.check('freeze_precedes_this_audit', pd.Timestamp(old_lock['created_utc']) < pd.Timestamp(protocol['created_utc'])
                   < pd.Timestamp(freeze['created_utc']) < pd.Timestamp.now(tz='UTC'))
        self.check('freeze_records_no_OOS_computation', freeze['oos_evaluated_at_freeze'] is False
                   and status['holdout_returns_computed'] is False and status['oos_evaluated'] is False)
        selection, metrics, daily = self.read_csv('is_selection.csv'), self.read_csv('is_all_metrics.csv'), self.read_csv('is_frozen_daily.csv.gz')
        self.check('sole_candidate_in_IS_outputs', all(set(frame.candidate_id) == {IDENTIFIER} for frame in (selection, metrics, daily)))
        old_metrics = self.frame_path(source / 'confirmation_full_metrics.csv')
        old_daily = self.frame_path(source / 'locked_is_daily.csv.gz')
        old_selection = self.frame_path(source / 'confirmation_selection.csv')
        original = old_selection.loc[old_selection.candidate_id.eq(IDENTIFIER)].iloc[0]
        self.check('v2_saved_original_pass_strict_fail', bool(original.original_is_pass) and not bool(original['pass']) and not bool(original.frozen))
        self.same_frame('copied_IS_metrics_exact', metrics, old_metrics.loc[old_metrics.candidate_id.eq(IDENTIFIER)], ['split', 'variant'])
        self.same_frame('copied_IS_daily_exact', daily, old_daily.loc[old_daily.candidate_id.eq(IDENTIFIER)], ['variant', 'date'])
        self.same_frame('frozen_IS_metrics_subset', self.read_csv('is_frozen_metrics.csv'), metrics.loc[metrics.split.eq('IS')], ['variant'])
        rebuilt = screen_candidates(metrics)
        numeric = [c for c in rebuilt if pd.api.types.is_numeric_dtype(rebuilt[c]) and not pd.api.types.is_bool_dtype(rebuilt[c])]
        self.values('original_SCREEN_rescreen', selection, rebuilt, [c for c in rebuilt if c != 'candidate_id'], numeric=numeric)
        self.check('original_SCREEN_pass_and_freeze', len(rebuilt) == 1 and bool(rebuilt.iloc[0]['pass']) and bool(rebuilt.iloc[0].frozen)
                   and candidate['pass'] is True and candidate['original_is_pass'] is True)
        self.check('IS_only_dates', pd.to_datetime(daily.date).max() == pd.Timestamp('2023-05-29')
                   and pd.to_datetime(metrics.end_date).le('2023-05-29').all())
        self.check('IS_zero_cost_daily', np.array_equal(daily.gross_daily_return, daily.net_daily_return))
        for row in metrics.itertuples(index=False):
            path = daily.loc[daily.variant.eq(row.variant)].sort_values('date')
            part = path.loc[pd.to_datetime(path.date).between(row.start_date, row.end_date)]
            returns = part.gross_daily_return.to_numpy() * 100
            curve = np.r_[0., returns.cumsum()]
            self.check('IS_simple_sum_DD:' + row.split + ':' + row.variant,
                       same_number(returns.sum(), row.gross_simple_return_pct) and same_number((curve - np.maximum.accumulate(curve)).min(), row.gross_max_drawdown_pp)
                       and len(returns) == row.observations)
        self.details.update(candidate_id=IDENTIFIER, spec=SPEC, original_SCREEN_pass=True, strict_v2_pass=False,
                            inherited_tests={'tests_run': old_audit['tests']['tests_run'], 'exit_code': old_audit['tests']['exit_code'],
                                             'rerun': False, 'reason': 'No decision source changed; all audited test-source hashes checked.'},
                            protocol_sha256=sha(self.campaign / 'protocol.json'), freeze_sha256=sha(self.campaign / 'frozen_selection.json'))
        return protocol, freeze, status

    def pre(self):
        self.common()
        self.check('OOS_artifacts_absent_before_opening', all(not (self.campaign / name).exists() for name in OOS_FILES))
        self.details.update(OOS_results_opened=False, raw_workbooks_parsed=False)

    def post(self):
        protocol, freeze, _ = self.common()
        pre = self.read_json('pre_oos_independent_audit.json')
        snapshot = self.register(self.campaign / 'audit_source_snapshots/verify_state_filter_supplement_pre.py')
        self.check('pre_audit_source_snapshot_exact', sha(snapshot) == pre['audit_script_sha256'])
        self.check('pre_audit_passed', pre['status'] == 'passed' and pre['checks_passed'] == pre['checks_total']
                   and all(row['passed'] for row in pre['checks']))
        self.hashes('preflight_evidence_unchanged', pre['verified_files_sha256'])
        oos_status = self.read_json('oos_status.json')
        self.check('OOS_status_passed', oos_status['status'] == 'passed' and oos_status['oos_evaluated'] is True
                   and oos_status['candidate_count'] == 1 and oos_status['retuning_permitted'] is False)
        self.check('OOS_freeze_link', oos_status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
                   and oos_status['protocol_sha256'] == sha(self.campaign / 'protocol.json'))
        self.check('freeze_preflight_OOS_chronology', pd.Timestamp(freeze['created_utc']) < pd.Timestamp(pre['started_utc'])
                   <= pd.Timestamp(pre['finished_utc']) < pd.Timestamp(oos_status['opened_utc']) < pd.Timestamp(oos_status['finished_utc']))
        self.hashes('OOS_output', oos_status['output_sha256'], self.campaign)
        self.check('all_OOS_outputs_registered', set(OOS_FILES[1:]) <= set(oos_status['output_sha256']))
        daily = self.read_csv('oos_daily.csv.gz')
        is_daily = self.read_csv('is_frozen_daily.csv.gz')
        metrics = self.read_csv('oos_metrics.csv')
        years = self.read_csv('oos_anniversary_metrics.csv')
        fresh = self.read_csv('oos_fresh_start_metrics.csv')
        assessment = self.read_csv('candidate_assessment.csv')
        ledger = self.read_csv('frozen_full_trade_ledger.csv.gz')
        decisions = self.read_csv('frozen_full_episode_decisions.csv.gz')
        for name, frame in [('daily', daily), ('metrics', metrics), ('years', years), ('fresh', fresh),
                            ('assessment', assessment), ('ledger', ledger), ('decisions', decisions)]:
            self.check('sole_frozen_candidate_retained:' + name, set(frame.candidate_id) == {IDENTIFIER}
                       and frame.pair.eq('JPY').all() and frame.strategy.eq('BB_MR_20').all())
        self.check('assessment_single_frozen_rank', len(assessment) == 1 and assessment.freeze_rank.iloc[0] == 1)
        for frame in (daily, is_daily):
            frame['date'] = pd.to_datetime(frame.date)
            self.check('unique_daily_rows:' + ('OOS' if frame is daily else 'IS'),
                       not frame.duplicated(['variant', 'date']).any() and np.array_equal(frame.gross_daily_return, frame.net_daily_return))
        for column in ('signal_date', 'entry_date', 'exit_date'):
            ledger[column] = pd.to_datetime(ledger[column])
        for column in ('signal_date', 'entry_date', 'original_end_signal_date'):
            decisions[column] = pd.to_datetime(decisions[column])

        # Only the frozen JPY instrument is decoded. This is arithmetic verification,
        # not a rerun or search over strategy rules or alternative filters.
        workbook = ROOT / next(name for name in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in name)
        observations = []
        with zipfile.ZipFile(workbook) as archive:
            rows = _workbook_rows(archive, 'xl/worksheets/sheet6.xml', {'B', 'D', 'E'})
            next(rows)
            for row in rows:
                if row.get('D') != 'JPY' or row.get('E') is None:
                    continue
                date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(row['B']))
                if date <= pd.Timestamp(protocol['oos_end']):
                    observations.append((date, float(row['E'])))
        observations.sort()
        calendar = pd.DatetimeIndex([row[0] for row in observations])
        prices = np.array([row[1] for row in observations])
        spot = np.r_[0., prices[1:] / prices[:-1] - 1]
        self.check('JPY_quote_calendar_valid', calendar.is_unique and calendar.is_monotonic_increasing
                   and np.isfinite(prices).all() and (prices > 0).all())
        oos_mask = (calendar >= protocol['oos_start']) & (calendar <= protocol['oos_end'])
        is_mask = (calendar >= protocol['is_start']) & (calendar <= protocol['is_end'])
        paths, independently_rebuilt = {}, {}
        for variant in ('BASE', 'FILTER'):
            saved = ledger.loc[ledger.variant.eq(variant)].sort_values('entry_idx').reset_index(drop=True)
            executed = np.zeros(len(calendar), dtype=int)
            occupied = np.zeros(len(calendar), dtype=int)
            for trade in saved.itertuples(index=False):
                stop = int(trade.exit_idx) if trade.closed else int(trade.exit_idx) + 1
                executed[int(trade.entry_idx):stop] = int(trade.position)
                occupied[int(trade.entry_idx):stop] += 1
            self.check('ledger_no_overlapping_exposure:' + variant, (occupied <= 1).all())
            held = np.r_[0, executed[:-1]]
            returns = held * spot
            rebuilt = ledger_from_executed(calendar, executed, returns)
            self.same_frame('full_ledger_independent_reconstruction:' + variant, saved[list(rebuilt.columns)], rebuilt, ['entry_idx'])
            self.check('full_ledger_zero_cost:' + variant, saved.transaction_cost_usd.eq(0).all()
                       and np.array_equal(saved.gross_simple_return_pct, saved.net_trade_return_pct))
            for label, frame, mask in [('IS_prefix', is_daily, is_mask), ('OOS', daily, oos_mask)]:
                path = frame.loc[frame.variant.eq(variant)].sort_values('date')
                self.check('independent_daily_positions_and_PnL:' + label + ':' + variant,
                           pd.DatetimeIndex(path.date).equals(calendar[mask]) and np.array_equal(path.executed, executed[mask])
                           and np.array_equal(path.held, held[mask])
                           and np.allclose(path.gross_daily_return, returns[mask], atol=1e-14, rtol=0))
            paths[variant] = (executed, held, returns)
            independently_rebuilt[variant] = rebuilt

        # Verify every recorded entry decision uses the original, fixed state bin.
        signal = decisions.signal_idx.to_numpy(int)
        self.check('decision_signal_indices_and_dates', (signal >= 0).all() and (signal < len(calendar)).all()
                   and pd.DatetimeIndex(decisions.signal_date).equals(calendar[signal]))
        price = pd.Series(prices)
        path_length = price.diff().abs().rolling(63, min_periods=63).sum()
        efficiency = price.diff(63).abs() / path_length.where(path_length.ne(0))
        efficiency = efficiency.mask(path_length.eq(0), 0.).to_numpy()[signal]
        codes = np.searchsorted([.1, .2, .4], efficiency, side='right')
        codes[~np.isfinite(efficiency)] = -1
        self.check('frozen_state_rule_recomputed_from_signal_prices', decisions.feature.eq('efficiency_63').all()
                   and decisions.rejected_state.eq(1).all() and np.allclose(decisions.feature_value, efficiency, atol=1e-12, rtol=1e-12, equal_nan=True)
                   and np.array_equal(decisions.state, codes) and np.array_equal(decisions.accepted, codes != 1))
        executed_decisions = decisions.loc[~decisions.entry_pending].set_index('entry_idx')
        base_ledger = independently_rebuilt['BASE']
        self.check('one_decision_per_original_BASE_episode', len(executed_decisions) == len(base_ledger)
                   and set(executed_decisions.index) == set(base_ledger.entry_idx))
        expected_filtered = np.zeros(len(calendar), dtype=int)
        for trade in base_ledger.itertuples(index=False):
            if bool(executed_decisions.loc[trade.entry_idx, 'accepted']):
                stop = trade.exit_idx if trade.closed else trade.exit_idx + 1
                expected_filtered[trade.entry_idx:stop] = trade.position
        self.check('whole_episode_rejection_no_reentry', np.array_equal(expected_filtered, paths['FILTER'][0]))
        old_decisions = self.frame_path(ROOT / protocol['source_campaign'] / 'locked_is_decisions.csv.gz')
        old_decisions = old_decisions.loc[old_decisions.candidate_id.eq(IDENTIFIER)].copy()
        stable_fields = ['signal_idx', 'feature', 'rejected_state', 'position', 'feature_value', 'state', 'accepted', 'status']
        prefix_decisions = decisions.loc[decisions.signal_date.le(protocol['is_end'])]
        self.same_frame('IS_entry_decisions_unchanged', prefix_decisions[stable_fields], old_decisions[stable_fields], ['signal_idx'])

        expected_periods = [('OOS', protocol['oos_start'], protocol['oos_end']), *protocol['oos_anniversary_years']]
        regular_metrics = pd.concat([metrics, years], ignore_index=True)
        self.check('OOS_metric_grid', not regular_metrics.duplicated(['split', 'variant']).any()
                   and set(zip(regular_metrics.split, regular_metrics.variant)) == {(p[0], v) for p in expected_periods for v in ('BASE', 'FILTER')})
        reconstructed_metrics = {}
        for phase, start, end in expected_periods:
            for variant in ('BASE', 'FILTER'):
                executed, held, returns = paths[variant]
                calculated = window_from_path(calendar, executed, held, returns, independently_rebuilt[variant], start, end)
                reconstructed_metrics[phase, variant] = calculated
                row = regular_metrics.loc[regular_metrics.split.eq(phase) & regular_metrics.variant.eq(variant)].iloc[0]
                mismatches = [key for key, value in calculated.items() if
                              (pd.Timestamp(row[key]) != value if key.endswith('_date') else not same_number(row[key], value))]
                self.check('independent_OOS_window:' + phase + ':' + variant, not mismatches, failed_fields=mismatches)
        self.check('anniversary_partition_exact', sum(reconstructed_metrics[p[0], 'BASE']['observations'] for p in protocol['oos_anniversary_years'])
                   == reconstructed_metrics['OOS', 'BASE']['observations'])
        for variant in ('BASE', 'FILTER'):
            self.check('anniversary_sum_to_OOS:' + variant,
                       same_number(sum(reconstructed_metrics[p[0], variant]['gross_simple_return_pct'] for p in protocol['oos_anniversary_years']),
                                   reconstructed_metrics['OOS', variant]['gross_simple_return_pct']))

        # Fresh start is derived from the saved ORIGINAL BASE episode boundaries.
        # No production fresh_start or strategy engine is called.
        allowed = np.zeros(len(calendar), dtype=bool)
        for trade in base_ledger.itertuples(index=False):
            if trade.entry_date >= pd.Timestamp(protocol['oos_start']):
                stop = trade.exit_idx if trade.closed else trade.exit_idx + 1
                allowed[trade.entry_idx:stop] = True
        self.check('fresh_start_metric_grid', len(fresh) == 2 and set(fresh.variant) == {'BASE', 'FILTER'}
                   and fresh.split.eq('OOS_FRESH_START').all())
        for variant in ('BASE', 'FILTER'):
            executed = np.where(allowed, paths[variant][0], 0)
            held = np.r_[0, executed[:-1]]
            returns = held * spot
            fresh_ledger = ledger_from_executed(calendar, executed, returns)
            calculated = window_from_path(calendar, executed, held, returns, fresh_ledger, protocol['oos_start'], protocol['oos_end'])
            reconstructed_metrics['OOS_FRESH_START', variant] = calculated
            row = fresh.loc[fresh.variant.eq(variant)].iloc[0]
            mismatches = [key for key, value in calculated.items() if
                          (pd.Timestamp(row[key]) != value if key.endswith('_date') else not same_number(row[key], value))]
            self.check('independent_fresh_start:' + variant, not mismatches and calculated['carry_in_n'] == 0, failed_fields=mismatches)
            self.check('fresh_start_removes_whole_carry_episode:' + variant,
                       not (executed[calendar < protocol['oos_start']] != 0).any()
                       and same_number(calculated['gross_simple_return_pct'], reconstructed_metrics['OOS', variant]['gross_simple_return_pct']
                                       - reconstructed_metrics['OOS', variant]['carry_in_net_return_pp']))

        for frame in (regular_metrics, fresh):
            for row in frame.itertuples(index=False):
                b = reconstructed_metrics[row.split, 'BASE']
                self.check('retention:' + row.split + ':' + row.variant, row.base_closed_n == b['closed_n']
                           and same_number(row.retention, row.closed_n / b['closed_n'] if b['closed_n'] else np.nan))
        b, f = reconstructed_metrics['OOS', 'BASE'], reconstructed_metrics['OOS', 'FILTER']
        positive_years = sum(reconstructed_metrics[p[0], 'FILTER']['net_simple_return_pct'] > 1e-10 for p in protocol['oos_anniversary_years'])
        delta = f['net_simple_return_pct'] - b['net_simple_return_pct']
        rules = protocol['oos_screen']
        flags = {'annualized_pass': f['net_annualized_simple_pct'] >= rules['min_net_annualized_simple_pct'] - 1e-10,
                 'baseline_improvement_pass': delta >= rules['min_net_return_improvement_pp'] - 1e-10,
                 'drawdown_pass': f['net_max_drawdown_pp'] >= rules['worst_net_drawdown_pp'] - 1e-10,
                 'positive_years_pass': positive_years >= rules['min_positive_anniversary_years'],
                 'sample_pass': f['closed_n'] >= rules['min_closed_trades']}
        quality_flags = {'at_least_20_closed': f['closed_n'] >= 20,
                         'finite_quality': bool(np.isfinite([f['win_rate'], b['win_rate'], f['payoff_ratio'], b['payoff_ratio']]).all()),
                         'win_rate_not_worse': f['win_rate'] >= b['win_rate'] - 1e-10,
                         'payoff_not_worse': f['payoff_ratio'] >= b['payoff_ratio'] - 1e-10}
        curve_pass, quality_pass = all(flags.values()), all(quality_flags.values())
        row = assessment.iloc[0]
        expected = {'is_return_pct': freeze['candidates'][0]['is_gross_return_pct'],
                    'oos_return_pct': f['net_simple_return_pct'], 'oos_base_return_pct': b['net_simple_return_pct'],
                    'oos_improvement_pp': delta, 'oos_annualized_simple_pct': f['net_annualized_simple_pct'],
                    'oos_max_drawdown_pp': f['net_max_drawdown_pp'], 'oos_win_rate': f['win_rate'], 'oos_base_win_rate': b['win_rate'],
                    'oos_payoff_ratio': f['payoff_ratio'], 'oos_base_payoff_ratio': b['payoff_ratio'],
                    'oos_closed_n': f['closed_n'], 'oos_positive_years': positive_years}
        self.check('independent_assessment_numbers', all(same_number(row[key], value) for key, value in expected.items()))
        self.check('independent_OOS_SCREEN_flags', all(bool(row[key]) == value for key, value in flags.items())
                   and bool(row.oos_curve_pass) == curve_pass and bool(row.oos_quality_confirmed) == quality_pass
                   and row.failed_criteria == '|'.join(key for key, value in flags.items() if not value))
        self.check('status_and_metrics_keep_failure', oos_status['curve_passing_candidates'] == int(curve_pass)
                   and oos_status['quality_confirmed_candidates'] == int(quality_pass)
                   and oos_status['both_curve_and_quality_candidates'] == int(curve_pass and quality_pass)
                   and metrics.oos_curve_pass.eq(curve_pass).all() and metrics.oos_quality_confirmed.eq(quality_pass).all())
        self.details.update(OOS_results_opened=True, raw_workbooks_parsed='JPY quote dates and prices only; through frozen OOS_END.',
                            strategy_replays=0, candidate_searches=0, independently_recomputed_windows=len(reconstructed_metrics),
                            pre_audit_script_sha256=pre['audit_script_sha256'], pre_audit_source_snapshot=str(snapshot.relative_to(ROOT)).replace('\\', '/'),
                            audit_code_amendment='Only this independent audit gained post-OOS checks. Preflight source bytes and JSON preserved; every frozen decision-source hash remains unchanged.',
                            result=expected, OOS_SCREEN_flags={k: bool(v) for k, v in flags.items()},
                            quality_flags={k: bool(v) for k, v in quality_flags.items()}, oos_curve_pass=curve_pass,
                            oos_quality_confirmed=quality_pass,
                            fresh_start={v: {k: reconstructed_metrics['OOS_FRESH_START', v][k] for k in
                                            ('gross_simple_return_pct', 'gross_annualized_simple_pct', 'gross_max_drawdown_pp', 'closed_n', 'carry_in_n')}
                                         for v in ('BASE', 'FILTER')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pre', 'post'], required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'reproduced/state_filter_supplement_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else args.campaign.resolve() / ('pre_oos_independent_audit.json' if args.stage == 'pre' else 'independent_validation.json')
    if output.name in {'protocol.json', 'frozen_selection.json', 'development_status.json', 'supplement_diagnostics.json'}:
        raise ValueError('Audit must not overwrite decision evidence.')
    audit = SupplementAudit(args.campaign.resolve())
    started = datetime.now(timezone.utc).isoformat()
    try:
        (audit.pre if args.stage == 'pre' else audit.post)()
    except Exception:
        audit.check('audit_completed', False, error=traceback.format_exc())
    for path, digest in audit.files.items():
        audit.check('read_only_preserved:' + str(Path(path).relative_to(ROOT)), sha(path) == digest)
    passed = sum(row['passed'] for row in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'stage': args.stage,
              'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'audit_script_sha256': sha(__file__), 'checks_passed': passed, 'checks_total': len(audit.checks),
              'verified_file_count': len(audit.files), 'verified_files_sha256': {Path(p).relative_to(ROOT).as_posix(): h for p, h in audit.files.items()},
              'details': audit.details, 'checks': audit.checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': f'{passed}/{len(audit.checks)}', 'files': len(audit.files), 'output': str(output)}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
