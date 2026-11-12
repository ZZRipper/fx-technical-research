"""Independent post-evaluation audit of frozen direction-only filters.

Only a completed, explicitly authorized current-campaign OOS is read. Original
BASE episodes and raw prices reconstruct every path; no strategy is executed,
no selection changes, and old OOS results are never accessed.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_direction_filter_research import DirectionAudit, ROOT, TOL, SPECS, sha, same_number
from verify_entry_interaction_research import ledger_from_executed, window_from_path
from verify_entry_interaction_oos import read_frozen_quotes

BOOLEAN_COLUMNS = ('accepted', 'rejected', 'entry_pending', 'original_exit_pending', 'exit_pending')


class DirectionPostAudit(DirectionAudit):
    def run(self):
        status = self.read_json('oos_status.json')
        if status.get('status') != 'passed' or status.get('oos_evaluated') is not True:
            raise ValueError('Completed authorized OOS status is required before outcome-table access.')
        protocol, freeze = self.read_json('protocol.json'), self.read_json('frozen_selection.json')
        pre = self.read_json('independent_validation.json')
        self.check('passed_independent_IS_audit_unchanged', pre['status'] == 'passed'
                   and pre['checks_total'] == pre['checks_passed'] == len(pre['checks']) and all(c['passed'] for c in pre['checks'])
                   and pre['audit_script_sha256'] == sha(ROOT / 'verify_direction_filter_research.py')
                   and pre['details']['OOS_results_opened'] is False)
        self.hashes('preaudit_evidence', pre['verified_files_sha256'])
        self.check('freeze_audit_before_OOS', pd.Timestamp(freeze['created_utc']) < pd.Timestamp(pre['started_utc'])
                   <= pd.Timestamp(pre['finished_utc']) < pd.Timestamp(status['opened_utc']) <= pd.Timestamp(status['finished_utc'])
                   and status['freeze_created_before_oos'] is True and status['retuning_permitted'] is False)
        self.check('exact_protocol_and_freeze_chain', status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
                   and status['protocol_sha256'] == freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['candidate_count'] == len(freeze['candidates']) and status['source_integrity_pass']
                   and status['is_prefix_positions_and_returns_match'])
        self.check('fixed_direction_registry_cost_and_dates', protocol['version'] == 'direction_filter_v6'
                   and sorted(protocol['candidate_specs'], key=lambda s: s['id']) == sorted(SPECS, key=lambda s: s['id'])
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
                   and protocol['oos_start'] == '2023-05-30' and protocol['oos_end'] == '2026-05-29')
        for key in ('source_sha256', 'input_sha256'):
            self.check('manifest_agreement:' + key, freeze[key] == protocol[key])
            self.hashes(key, protocol[key])
        self.hashes('IS_output_hashes', freeze['is_output_sha256'], self.campaign)
        self.hashes('OOS_output_hashes', status['output_sha256'], self.campaign)
        items = freeze['candidates']; ids = [r['candidate_id'] for r in items]
        if not items:
            raise AssertionError('A completed OOS must contain frozen original-SCREEN winners.')
        daily, is_daily = self.read_csv('oos_daily.csv.gz'), self.read_csv('is_frozen_daily.csv.gz')
        daily.date, is_daily.date = pd.to_datetime(daily.date), pd.to_datetime(is_daily.date)
        ledger = self.read_csv('frozen_full_trade_ledger.csv.gz')
        for col in ('signal_date', 'entry_date', 'exit_date'):
            ledger[col] = pd.to_datetime(ledger[col])
        decisions, previous = self.read_csv('frozen_full_episode_decisions.csv.gz'), self.read_csv('is_frozen_decisions.csv.gz')
        metrics, annual, fresh = self.read_csv('oos_metrics.csv'), self.read_csv('oos_anniversary_metrics.csv'), self.read_csv('oos_fresh_start_metrics.csv')
        assessment = self.read_csv('candidate_assessment.csv')
        self.check('all_frozen_candidates_in_every_table', all(set(frame.candidate_id) == set(ids)
                   for frame in (daily, ledger, decisions, metrics, annual, fresh, assessment)))
        self.check('assessment_in_frozen_IS_rank_order', assessment.candidate_id.tolist() == ids
                   and assessment.freeze_rank.tolist() == [r['freeze_rank'] for r in items])
        self.check('result_unique_keys_and_date_scope', not daily.duplicated(['candidate_id', 'variant', 'date']).any()
                   and daily.date.between(protocol['oos_start'], protocol['oos_end']).all()
                   and not ledger.duplicated(['candidate_id', 'variant', 'entry_idx']).any()
                   and not decisions.duplicated(['candidate_id', 'signal_idx']).any())
        for label, frame in [('IS', previous), ('FULL', decisions)]:
            self.check('nonnullable_boolean_decisions:' + label, all(c in frame and frame[c].notna().all()
                       and frame[c].isin([True, False]).all() for c in BOOLEAN_COLUMNS))
        source = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in p and p.endswith('.xlsx'))
        quotes = read_frozen_quotes(self.register(source), pd.Timestamp(protocol['oos_end']), {r['pair'] for r in items})
        full_ledgers, rebuilt_daily, rebuilt_IS, metric_rows, annual_rows, fresh_rows = [], [], [], [], [], []
        decision_count = 0; terminal_limits = []
        for item in items:
            cid, pair, reject = item['candidate_id'], item['pair'], item['spec']['reject_position']
            self.check('frozen_spec_registered:' + cid, item['spec'] in SPECS and item['filter_id'] == item['spec']['id'])
            q = quotes[pair]; dates = pd.DatetimeIndex(q.date); prices = q.price.to_numpy(float)
            self.check('full_source_price_calendar:' + cid, dates.is_unique and dates.is_monotonic_increasing
                       and dates[-1] <= pd.Timestamp(protocol['oos_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1.]
            original = ledger.loc[ledger.candidate_id.eq(cid) & ledger.variant.eq('BASE')].sort_values('entry_idx')
            executed = np.zeros(len(dates), int); occupancy = executed.copy()
            for trade in original.itertuples(index=False):
                end = int(trade.exit_idx) if trade.closed else len(dates)
                executed[int(trade.entry_idx):end] = int(trade.position)
                occupancy[int(trade.entry_idx):end] += 1
            self.check('BASE_segments_nonoverlapping:' + cid, (occupancy <= 1).all())
            rows = decisions.loc[decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            prior = previous.loc[previous.candidate_id.eq(cid)].sort_values('signal_idx')
            prefix = rows.loc[pd.to_datetime(rows.signal_date).le(protocol['is_end'])]
            stable = ['candidate_id', 'original_episode', 'signal_idx', 'signal_date', 'position', 'reject_position',
                      'condition', 'accepted', 'rejected', 'status']
            self.compare_frames('all_IS_direction_decisions_preserved:' + cid, prefix[stable], prior[stable], ['candidate_id', 'signal_idx'])
            n_is = int((dates <= pd.Timestamp(protocol['is_end'])).sum())
            for row in prior.itertuples(index=False):
                if row.entry_pending or row.original_exit_pending:
                    terminal_limits.append({'candidate_id': cid, 'signal_idx': int(row.signal_idx),
                                            'pending_new_entry': bool(row.entry_pending),
                                            'pending_natural_exit_or_reversal': bool(row.original_exit_pending)})
            self.check('IS_pending_boundary_dates:' + cid,
                       prior.loc[prior.entry_pending, 'signal_idx'].eq(n_is - 1).all()
                       and prior.loc[prior.original_exit_pending, 'original_end_signal_idx'].eq(n_is - 1).all())
            filled = rows.loc[~rows.entry_pending]
            self.check('complete_original_episode_decisions:' + cid, len(filled) == len(original)
                       and np.array_equal(filled.signal_idx.to_numpy(), original.signal_idx.to_numpy()))
            original_index = original.set_index('signal_idx')
            target = np.zeros(len(dates), int); occupied = target.copy(); failed_signals = []
            for ordinal, row in enumerate(rows.itertuples(index=False)):
                start = int(row.signal_idx)
                end = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
                pending = start + 1 == len(dates); rejected = row.position == reject; closes = end < len(dates) - 1
                okay = (0 <= start < end <= len(dates) and row.original_episode == ordinal and row.position in (-1, 1)
                        and row.reject_position == reject and row.signal_date == str(dates[start].date())
                        and bool(row.entry_pending) == pending and bool(row.rejected) == rejected and bool(row.accepted) != rejected
                        and row.condition == f'position=={reject:+d}' and row.status == ('REJECT' if rejected else 'KEEP'))
                if not pending:
                    trade = original_index.loc[start]
                    okay &= row.entry_idx == start + 1 and pd.Timestamp(row.entry_date) == dates[start + 1] and row.position == trade.position
                    okay &= end == int(trade.exit_idx) - 1 if trade.closed else end >= len(dates) - 1
                else:
                    okay &= pd.isna(row.entry_idx) and pd.isna(row.entry_date)
                for name, exists, idx in [('actual_entry', not rejected and not pending, start + 1),
                                          ('original_end_signal', end < len(dates), end),
                                          ('original_exit', closes, end + 1), ('actual_exit', closes and not rejected, end + 1)]:
                    if exists:
                        okay &= getattr(row, name + '_idx') == idx and pd.Timestamp(getattr(row, name + '_date')) == dates[idx]
                    else:
                        okay &= pd.isna(getattr(row, name + '_idx')) and pd.isna(getattr(row, name + '_date'))
                okay &= bool(row.original_exit_pending) == (end == len(dates) - 1)
                okay &= bool(row.exit_pending) == (end == len(dates) - 1 and not rejected)
                target[start:end] = row.position; occupied[start:end] += 1
                if not okay:
                    failed_signals.append(start)
                decision_count += 1
            self.check('full_direction_decisions_execution_clocks:' + cid, not failed_signals and (occupied <= 1).all()
                       and np.array_equal(executed, np.r_[0, target[:-1]]), mismatched_signals=failed_signals[:10])
            filtered_exec = np.where(executed == reject, 0, executed)
            self.check('pointwise_original_direction_exclusion:' + cid,
                       np.array_equal(filtered_exec, np.r_[0, np.where(target == reject, 0, target)[:-1]]))
            paths = {}
            for variant, e in [('BASE', executed), ('FILTER', filtered_exec)]:
                held = np.r_[0, e[:-1]]; returns = held * spot
                rebuilt = ledger_from_executed(dates, e, returns)
                if rebuilt.empty:
                    rebuilt = original.iloc[:0].copy()
                paths[variant] = e, held, returns, rebuilt
                full_ledgers.append(rebuilt.assign(candidate_id=cid, variant=variant))
                for split, start, end, collector in [('IS', protocol['is_start'], protocol['is_end'], rebuilt_IS),
                                                     ('OOS', protocol['oos_start'], protocol['oos_end'], rebuilt_daily)]:
                    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                    collector.append(pd.DataFrame({'candidate_id': cid, 'variant': variant, 'split': split, 'date': dates[mask],
                        'executed': e[mask], 'held': held[mask], 'gross_daily_return': returns[mask], 'net_daily_return': returns[mask]}))
            for periods, collector in [([('OOS', protocol['oos_start'], protocol['oos_end'])], metric_rows),
                                       (protocol['oos_anniversary_years'], annual_rows)]:
                for split, start, end in periods:
                    results = {v: window_from_path(dates, *paths[v], start, end) for v in ('BASE', 'FILTER')}
                    for variant, values in results.items():
                        n = results['BASE']['closed_n']
                        collector.append({'candidate_id': cid, 'variant': variant, 'split': split,
                                          'base_closed_n': n, 'retention': values['closed_n'] / n if n else np.nan, **values})
            allowed = np.zeros(len(dates), bool)
            for trade in paths['BASE'][3].itertuples(index=False):
                if trade.entry_date >= pd.Timestamp(protocol['oos_start']):
                    end = int(trade.exit_idx) if trade.closed else len(dates)
                    allowed[int(trade.entry_idx):end] = True
            fresh_results = {}
            for variant, (e, _, _, _) in paths.items():
                cash = np.where(allowed, e, 0); held = np.r_[0, cash[:-1]]; returns = held * spot
                rebuilt = ledger_from_executed(dates, cash, returns)
                if rebuilt.empty:
                    rebuilt = original.iloc[:0].copy()
                fresh_results[variant] = window_from_path(dates, cash, held, returns, rebuilt, protocol['oos_start'], protocol['oos_end'])
            for variant, values in fresh_results.items():
                n = fresh_results['BASE']['closed_n']
                fresh_rows.append({'candidate_id': cid, 'variant': variant, 'split': 'OOS_FRESH_START',
                                   'base_closed_n': n, 'retention': values['closed_n'] / n if n else np.nan, **values})
        self.compare_frames('full_ledger_independent_prices_positions', ledger, pd.concat(full_ledgers, ignore_index=True), ['candidate_id', 'variant', 'entry_idx'])
        self.check('full_ledger_zero_cost_dollar_arithmetic', ledger.transaction_cost_usd.eq(0).all()
                   and np.allclose(ledger.gross_simple_return_pct, ledger.net_trade_return_pct, atol=1e-8, rtol=1e-10)
                   and np.allclose(ledger.net_pnl_usd, ledger.net_trade_return_pct * 1000, atol=1e-5, rtol=1e-10))
        self.compare_frames('frozen_IS_positions_returns_prefix', is_daily, pd.concat(rebuilt_IS, ignore_index=True), ['candidate_id', 'variant', 'date'])
        self.compare_frames('OOS_positions_returns', daily, pd.concat(rebuilt_daily, ignore_index=True), ['candidate_id', 'variant', 'date'])
        calculated, calculated_annual, calculated_fresh = map(pd.DataFrame, (metric_rows, annual_rows, fresh_rows))
        for label, saved, expected in [('continuous_OOS', metrics, calculated), ('all_anniversary_years', annual, calculated_annual),
                                       ('fresh_start_sensitivity', fresh, calculated_fresh)]:
            self.compare_frames(label, saved, expected, ['candidate_id', 'variant', 'split'])
        results = []; gate = protocol['oos_screen']
        for item in items:
            cid = item['candidate_id']; account = calculated.loc[calculated.candidate_id.eq(cid)].set_index('variant')
            b, f = account.loc['BASE'], account.loc['FILTER']
            positive = int((calculated_annual.loc[calculated_annual.candidate_id.eq(cid) & calculated_annual.variant.eq('FILTER'), 'net_simple_return_pct'] > TOL).sum())
            delta = float(f.net_simple_return_pct - b.net_simple_return_pct)
            flags = {'annualized_pass': bool(f.net_annualized_simple_pct >= gate['min_net_annualized_simple_pct'] - TOL),
                     'baseline_improvement_pass': bool(delta >= gate['min_net_return_improvement_pp'] - TOL),
                     'drawdown_pass': bool(f.net_max_drawdown_pp >= gate['worst_net_drawdown_pp'] - TOL),
                     'positive_years_pass': positive >= gate['min_positive_anniversary_years'],
                     'sample_pass': bool(f.closed_n >= gate['min_closed_trades'])}
            quality = bool(f.closed_n >= 20 and np.isfinite([f.win_rate, b.win_rate, f.payoff_ratio, b.payoff_ratio]).all()
                           and f.win_rate >= b.win_rate - TOL and f.payoff_ratio >= b.payoff_ratio - TOL)
            results.append({**{k: item[k] for k in ('candidate_id', 'pair', 'strategy', 'filter_id', 'freeze_rank')},
                'is_return_pct': item['is_gross_return_pct'], 'oos_return_pct': f.net_simple_return_pct,
                'oos_base_return_pct': b.net_simple_return_pct, 'oos_improvement_pp': delta,
                'oos_annualized_simple_pct': f.net_annualized_simple_pct, 'oos_max_drawdown_pp': f.net_max_drawdown_pp,
                'oos_win_rate': f.win_rate, 'oos_base_win_rate': b.win_rate, 'oos_payoff_ratio': f.payoff_ratio,
                'oos_base_payoff_ratio': b.payoff_ratio, 'oos_closed_n': int(f.closed_n), 'oos_positive_years': positive,
                **flags, 'oos_curve_pass': all(flags.values()), 'oos_quality_confirmed': quality,
                'failed_criteria': '|'.join(k for k, value in flags.items() if not value)})
        results = pd.DataFrame(results)
        self.compare_frames('prespecified_curve_and_quality_assessments', assessment, results, ['candidate_id'])
        self.compare_frames('metric_assessment_flags', metrics[['candidate_id', 'variant', 'oos_curve_pass', 'oos_quality_confirmed']],
                            calculated[['candidate_id', 'variant']].merge(results[['candidate_id', 'oos_curve_pass', 'oos_quality_confirmed']], on='candidate_id'),
                            ['candidate_id', 'variant'])
        self.check('status_counts_from_recomputed_assessments', status['curve_passing_candidates'] == int(results.oos_curve_pass.sum())
                   and status['quality_confirmed_candidates'] == int(results.oos_quality_confirmed.sum())
                   and status['both_curve_and_quality_candidates'] == int((results.oos_curve_pass & results.oos_quality_confirmed).sum()))
        self.check('all_evidence_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(candidate_ids_in_frozen_order=ids, candidates=len(ids), curve_passes=int(results.oos_curve_pass.sum()),
                            quality_confirmed=int(results.oos_quality_confirmed.sum()), full_episode_decisions_checked=decision_count,
                            independently_recomputed_metric_windows=len(calculated) + len(calculated_annual) + len(calculated_fresh),
                            all_frozen_IS_paths_match=True, current_campaign_OOS_only=True, strategy_engine_replays=0,
                            price_currencies=sorted({r['pair'] for r in items}), raw_price_cutoff=protocol['oos_end'],
                            pending_IS_terminal_signal_limit={'rows': terminal_limits,
                                'scope': 'An executed ledger cannot independently recover the final IS TARGET instruction, including pending entry, exit or reversal. Saved direction decisions and next-close metadata are checked; later full replay verifies realized positions when the next quote arrives. Original BASE strategy signal generation is not rerun.'},
                            evidence_status='Repeated historical verification; no claim of untouched independent OOS.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/direction_filter_v6')
    args = parser.parse_args(); audit = DirectionPostAudit(args.output.resolve())
    started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_direction_filter_research.py', ROOT / 'verify_entry_interaction_research.py',
                   ROOT / 'verify_entry_interaction_oos.py', ROOT / 'verify_state_filter_research.py', ROOT / 'verify_state_filter_supplement.py'):
        audit.register(source)
    try:
        audit.run()
    except Exception:
        audit.check('audit_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(row['passed'] for row in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
              'checks_total': len(audit.checks), 'checks_passed': passed, 'checks': audit.checks,
              'verified_files_sha256': audit.files, 'details': audit.details}
    destination = audit.campaign / 'post_oos_independent_validation.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(destination), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
