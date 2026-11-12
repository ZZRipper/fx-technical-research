"""Independent IS-only pre-OOS audit of direction rejection research.

Rebuild discovery nominations from the original BASE ledger, then reconstruct
all proposed direction-exclusion paths from saved original episodes and raw IS
prices. No strategy engine or OOS result is used. Only the audit JSON is written.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import gzip
import itertools
import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_entry_interaction_research import (
    InteractionAudit, ROOT, TOL, BLOCKS, SOURCE_LEDGER, truth, sha, same_number,
    independent_quality, read_IS_quotes, ledger_from_executed, window_from_path,
    CURRENCIES, STRATEGIES, SCREEN, screen_candidates,
)

OBS = ROOT / 'reproduced/direction_probe_v5'
SPECS = ({'id': 'REJECT_LONG', 'reject_position': 1}, {'id': 'REJECT_SHORT', 'reject_position': -1})


def direction_spec(position):
    if position not in (-1, 1):
        raise ValueError('Direction must be signed unit position.')
    return {'id': 'REJECT_LONG' if position == 1 else 'REJECT_SHORT', 'reject_position': int(position)}


def load_direction_D(path):
    """Keep D entry metadata, converting outcomes only for eligible D closes."""
    records = []; skipped_outcomes = decoded_outcomes = 0
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if not '2005-01-03' <= row['entry_date'] <= '2014-12-31':
                skipped_outcomes += 1
                continue
            record = {c: row[c] for c in ('pair', 'strategy', 'family', 'trade_id', 'signal_date', 'entry_date', 'exit_date')}
            record.update(position=int(row['position']), closed=row['closed'] == 'True',
                          eligible_closed=row['eligible_closed'] == 'True', gross_simple_return_pct=np.nan)
            if record['closed'] and record['eligible_closed'] and row['exit_date'] <= '2014-12-31':
                record['gross_simple_return_pct'] = float(row['gross_simple_return_pct'])
                decoded_outcomes += 1
            else:
                skipped_outcomes += 1
            records.append(record)
    return pd.DataFrame(records), decoded_outcomes, skipped_outcomes


def direction_quality(values):
    values = np.asarray(values, float)
    total = float(values.sum())
    return {**independent_quality(values), 'return_sum_pct': total,
            'return_without_worst_trade_pct': total - float(values.min()) if len(values) else np.nan}


class DirectionAudit(InteractionAudit):
    def discovery(self, probe):
        ledger, decoded, skipped = load_direction_D(self.register(SOURCE_LEDGER))
        self.check('D_only_metadata_and_numeric_cohorts', not ledger.duplicated(['pair', 'strategy', 'trade_id']).any()
                   and ledger.position.isin([-1, 1]).all()
                   and set(zip(ledger.pair, ledger.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES))
                   and np.isfinite(ledger.gross_simple_return_pct.dropna()).all())
        rows, comparisons, hypotheses = [], [], []
        for pair, strategy in itertools.product(CURRENCIES, STRATEGIES):
            account = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)]
            for position in (-1, 1):
                directional = account.loc[account.position.eq(position)]
                key = {'pair': pair, 'strategy': strategy, 'position': position,
                       'family': account.family.iloc[0],
                       'direction': 'LONG_FOREIGN_SHORT_USD' if position == 1 else 'SHORT_FOREIGN_LONG_USD'}
                values = {}
                for block, start, end in BLOCKS:
                    entered = directional.loc[directional.entry_date.between(start, end)]
                    closed = entered.loc[entered.eligible_closed & entered.closed & entered.exit_date.le(end)]
                    values[block] = {'entered_n': len(entered), 'censored_or_ineligible_entry_n': len(entered) - len(closed),
                                     **direction_quality(closed.gross_simple_return_pct)}
                    rows.append({**key, 'block': block, **values[block]})
                both_negative = all(values[b]['return_sum_pct'] < -TOL for b in ('D1', 'D2'))
                enough = all(values[b]['closed_n'] >= 10 for b in ('D1', 'D2'))
                supported = both_negative and enough
                d_negative = values['D']['return_sum_pct'] < -TOL
                comparison = {**key, **{f'{block.lower()}_{c}': value for block, fields in values.items() for c, value in fields.items()
                                       if c in ('entered_n', 'closed_n', 'censored_or_ineligible_entry_n', 'win_n', 'loss_n',
                                                'return_sum_pct', 'expectancy_pct', 'win_rate', 'payoff_ratio', 'return_without_worst_trade_pct')},
                              'negative_in_both_discovery_blocks': both_negative,
                              'each_discovery_block_at_least_10_closed': enough,
                              'repeated_negative_with_descriptive_sample_support': supported,
                              'combined_D_negative': d_negative,
                              'combined_D_negative_without_worst_trade': values['D']['return_without_worst_trade_pct'] < -TOL,
                              'cross_D1_D2_closed_n': values['D']['closed_n'] - values['D1']['closed_n'] - values['D2']['closed_n'],
                              'cross_D1_D2_return_sum_pct': values['D']['return_sum_pct'] - values['D1']['return_sum_pct'] - values['D2']['return_sum_pct']}
                comparisons.append(comparison)
                if supported and d_negative:
                    spec = direction_spec(position)
                    hypotheses.append({'candidate_id': '|'.join((pair, strategy, spec['id'])), 'pair': pair,
                                       'strategy': strategy, 'filter_id': spec['id'], 'spec': spec,
                                       'discovery_evidence': values})
        summary, comparison = pd.DataFrame(rows), pd.DataFrame(comparisons)
        self.compare_frames('all_D_direction_summaries', self.frame(OBS / 'direction_summary.csv'), summary,
                            ['pair', 'strategy', 'position', 'block'])
        self.compare_frames('all_D_direction_comparisons', self.frame(OBS / 'direction_comparison.csv'), comparison,
                            ['pair', 'strategy', 'position'])
        self.check('probe_parse_counts', decoded == probe['access_audit']['parsed_eligible_closed_trade_outcomes']
                   and skipped == probe['access_audit']['rows_skipped_before_outcome_conversion'] + len(ledger) - decoded
                   and len(ledger) == probe['access_audit']['D_entry_metadata_rows'])
        self.check('probe_support_counts', len(comparison) == probe['counts']['account_direction_groups']
                   and int(comparison.negative_in_both_discovery_blocks.sum()) == probe['counts']['negative_in_both_subperiods_groups']
                   and int(comparison.repeated_negative_with_descriptive_sample_support.sum()) == probe['counts']['repeated_negative_with_at_least_10_per_subperiod_groups']
                   and len(hypotheses) == probe['counts']['supported_groups_also_D_negative'])
        self.details.update(D_closed_outcomes_decoded=decoded, later_or_ineligible_outcomes_skipped_before_D_parse=skipped,
                            D_entry_metadata=len(ledger), direction_summary_rows=len(rows), direction_comparison_rows=len(comparisons),
                            discovery_supported_candidates=len(hypotheses),
                            discovery_overlap='D includes cross-2009/2010 trades excluded from both halves; long/short are not independent market samples.')
        return hypotheses

    def run(self):
        protocol, status = self.read_json('protocol.json'), self.read_json('development_status.json')
        freeze, hypotheses = self.read_json('frozen_selection.json'), self.read_json('is_hypotheses.json')
        probe = self.document(OBS / 'direction_probe.json')
        observation_validation = self.document(ROOT / 'reproduced/direction_observation_validation_v6/validation.json')
        self.check('D_observation_independently_validated', observation_validation['status'] == 'passed'
                   and observation_validation['checks_total'] == observation_validation['checks_passed'] == len(observation_validation['checks'])
                   and all(c['passed'] for c in observation_validation['checks'])
                   and observation_validation['accounts'] == 180 and observation_validation['compute_cutoff'] == '2014-12-31'
                   and observation_validation['oos_computed'] is False)
        self.hashes('D_observation_validation_sources', observation_validation['source_sha256'])
        self.hashes('D_observation_validation_inputs', observation_validation['input_sha256'])
        self.check('direction_protocol_original_SCREEN_zero_cost_scope', protocol['version'] == 'direction_filter_v6'
                   and protocol['is_start'] == '2005-01-03' and protocol['is_end'] == '2023-05-29'
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0 and protocol['screen'] == SCREEN
                   and protocol['initial_capital_usd'] == 100000 and protocol['accounting'] == 'fixed_initial_capital_simple'
                   and set(protocol['scope']['pairs']) == set(CURRENCIES) and set(protocol['scope']['strategies']) == set(STRATEGIES)
                   and protocol['scope']['baseline_accounts'] == 180
                   and sorted(protocol['candidate_specs'], key=lambda s: s['id']) == sorted(SPECS, key=lambda s: s['id']))
        self.check('completed_IS_without_OOS', status['status'] == 'passed' and status['is_screen_completed']
                   and status['source_integrity_pass'] and status['compute_cutoff'] == protocol['is_end']
                   and status['oos_evaluated'] is False and status['holdout_returns_computed'] is False)
        self.check('D_probe_scope', probe['cost_bp'] == 0 and probe['counts']['accounts'] == 180
                   and probe['blocks'] == [list(b) for b in BLOCKS]
                   and probe['access_audit']['allowed_outcome_end'] == '2014-12-31'
                   and probe['access_audit']['later_outcomes_read'] is False and probe['access_audit']['OOS_read'] is False)
        self.check('freeze_hash_chain', freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
                   and freeze['oos_evaluated_at_freeze'] is False)
        self.check('observation_proposals_replay_freeze_chronology', pd.Timestamp(probe['created_utc']) < pd.Timestamp(protocol['created_utc'])
                   and pd.Timestamp(observation_validation['finished_utc']) < pd.Timestamp(protocol['created_utc'])
                   and pd.Timestamp(protocol['created_utc']) < pd.Timestamp(hypotheses['created_utc']) <= pd.Timestamp(status['started_utc'])
                   < pd.Timestamp(freeze['created_utc']) <= pd.Timestamp(status['finished_utc']))
        for kind in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest:' + kind, freeze[kind] == protocol[kind])
            self.hashes(kind, protocol[kind])
        self.hashes('D_probe_original_ledger', probe['source_sha256'])
        self.hashes('D_probe_outputs', probe['output_sha256'], OBS)
        self.hashes('observation_manifest', protocol['observation_source_sha256'])
        self.hashes('frozen_IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.check('all_IS_artifacts_frozen', {p.name for p in self.campaign.glob('is_*') if p.is_file()} == set(freeze['is_output_sha256']))
        self.check('OOS_artifacts_absent', all(not (self.campaign / n).exists() for n in ('oos_status.json', 'oos_daily.csv.gz',
                   'oos_metrics.csv', 'oos_anniversary_metrics.csv', 'oos_fresh_start_metrics.csv', 'candidate_assessment.csv',
                   'frozen_full_trade_ledger.csv.gz', 'frozen_full_episode_decisions.csv.gz')))
        independent = self.discovery(probe)
        expected = {h['candidate_id']: h for h in independent}; items = hypotheses['candidates']
        self.check('exact_D_nominations_without_later_selection', len(items) == len(expected) and {h['candidate_id'] for h in items} == set(expected),
                   saved=len(items), computed=len(expected))
        for item in items:
            reference = expected[item['candidate_id']]
            evidence = item['discovery_evidence']
            self.check('nomination_spec_evidence:' + item['candidate_id'], item['spec'] == reference['spec']
                       and item['filter_id'] == reference['filter_id'] and item['pair'] == reference['pair']
                       and item['strategy'] == reference['strategy'] and item['is_new'] is True
                       and set(evidence) == {'D1', 'D2', 'D'}
                       and all({'closed_n', 'return_sum_pct'}.issubset(fields) and all(
                           name in reference['discovery_evidence'][block] and same_number(value, reference['discovery_evidence'][block][name])
                           for name, value in fields.items()) for block, fields in evidence.items()))
        metrics, selection, coverage = self.read_csv('is_all_metrics.csv'), self.read_csv('is_selection.csv'), self.read_csv('is_account_coverage.csv')
        counts = pd.Series([(h['pair'], h['strategy']) for h in items], dtype=object).value_counts()
        self.check('all_180_observed_accounts_and_proposal_counts', len(coverage) == 180
                   and set(zip(coverage.pair, coverage.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES))
                   and coverage.compute_end.eq(protocol['is_end']).all()
                   and all(row.candidate_count == counts.get((row.pair, row.strategy), 0) for row in coverage.itertuples())
                   and coverage.candidate_count.sum() == len(items))
        periods = [('IS', protocol['is_start'], protocol['is_end']), *protocol['is_blocks']]
        self.check('all_proposals_full_IS_eight_windows', set(selection.candidate_id) == set(expected)
                   and not metrics.duplicated(['candidate_id', 'variant', 'split']).any()
                   and set(zip(metrics.candidate_id, metrics.variant, metrics.split))
                   == set(itertools.product(expected, ('BASE', 'FILTER'), [p[0] for p in periods])))
        rescreened = screen_candidates(metrics) if items else selection.copy()
        self.compare_frames('original_SCREEN_rescreen', selection, rescreened, ['candidate_id'])
        self.check('original_ranking_order', selection.candidate_id.tolist() == rescreened.candidate_id.tolist())
        winners = rescreened.loc[rescreened.frozen.map(truth)].sort_values('freeze_rank')
        self.check('exact_ranked_freeze_and_counts', winners.candidate_id.tolist() == [h['candidate_id'] for h in freeze['candidates']]
                   and freeze['frozen_candidates'] == status['frozen_candidates'] == len(winners)
                   and freeze['all_passing_versions'] == status['all_passing_versions'] == int(rescreened['pass'].sum())
                   and freeze['all_candidate_accounts_tested'] == status['candidates_evaluated'] == len(items))
        for item in freeze['candidates']:
            row = winners.set_index('candidate_id').loc[item['candidate_id']]
            self.check('frozen_record_values:' + item['candidate_id'], item['spec'] == expected[item['candidate_id']]['spec'] and all(
                same_number(item[c], row[c]) if isinstance(row[c], (float, int, np.number, bool, np.bool_)) else
                (('' if item[c] is None else str(item[c])) == ('' if pd.isna(row[c]) else str(row[c]))) for c in rescreened if c != 'candidate_id'))
        self.audit_direction_paths(protocol, items, metrics, winners, periods)
        self.check('all_registered_evidence_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(campaign_version=protocol['version'], observed_accounts=180, candidate_accounts=int(coverage.candidate_count.gt(0).sum()),
                            candidates=len(items), original_SCREEN_passes=int(rescreened['pass'].sum()), frozen_candidates=len(winners),
                            OOS_results_opened=False, holdout_returns_computed=False, strategy_engine_replays=0,
                            raw_price_cutoff=protocol['is_end'], post_IS_price_values_decoded=0)

    def audit_direction_paths(self, protocol, items, metrics, winners, periods):
        ledger = self.frame(SOURCE_LEDGER)
        for c in ('signal_date', 'entry_date', 'exit_date'):
            ledger[c] = pd.to_datetime(ledger[c])
        self.check('original_IS_ledger_scope_unique', not ledger.duplicated(['pair', 'strategy', 'signal_idx']).any()
                   and ledger.exit_date.le(protocol['is_end']).all()
                   and set(zip(ledger.pair, ledger.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES)))
        workbook = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in p and p.endswith('.xlsx'))
        quotes = read_IS_quotes(self.register(workbook), pd.Timestamp(protocol['is_end']))
        paths, windows = {}, {}
        for pair, frame in quotes.items():
            dates = pd.DatetimeIndex(frame.date); prices = frame.price.to_numpy(float)
            self.check('IS_price_calendar:' + pair, dates.is_unique and dates.is_monotonic_increasing
                       and dates[-1] <= pd.Timestamp(protocol['is_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1.]
            for strategy in STRATEGIES:
                key = pair, strategy
                original = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)].sort_values('entry_idx')
                executed = np.zeros(len(dates), int); occupied = executed.copy()
                for row in original.itertuples(index=False):
                    end = int(row.exit_idx) if row.closed else len(dates)
                    executed[int(row.entry_idx):end] = int(row.position)
                    occupied[int(row.entry_idx):end] += 1
                held = np.r_[0, executed[:-1]]; gross = held * spot
                rebuilt = ledger_from_executed(dates, executed, gross)
                self.compare_frames('BASE_ledger:' + '|'.join(key), original, rebuilt, ['entry_idx'])
                self.check('BASE_no_overlap_zero_cost:' + '|'.join(key), (occupied <= 1).all() and original.transaction_cost_usd.eq(0).all())
                paths[key] = dates, spot, executed, held, gross, rebuilt, original
                windows[key] = {split: window_from_path(dates, executed, held, gross, rebuilt, start, end) for split, start, end in periods}
        decisions = self.read_csv('is_all_decisions.csv.gz') if items else pd.DataFrame(columns=['candidate_id', 'signal_idx'])
        self.check('all_proposal_decisions_complete_unique', set(decisions.candidate_id) == {h['candidate_id'] for h in items}
                   and not decisions.duplicated(['candidate_id', 'signal_idx']).any())
        metric_index = metrics.set_index(['candidate_id', 'variant', 'split'])
        recomputed, daily_rows, ledger_rows, pending_rows = [], [], [], []
        total_decisions = total_rejected = 0; frozen_ids = set(winners.candidate_id)
        for index, item in enumerate(items):
            cid, key = item['candidate_id'], (item['pair'], item['strategy'])
            dates, spot, base_exec, base_held, base_gross, base_ledger, original = paths[key]
            reject_position = item['spec']['reject_position']
            rows = decisions.loc[decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            filled = rows.loc[~rows.entry_pending.map(truth)]
            self.check('original_episode_coverage:' + cid, len(filled) == len(original)
                       and np.array_equal(filled.signal_idx.to_numpy(), original.signal_idx.to_numpy()))
            original_by_signal = original.set_index('signal_idx')
            target = np.zeros(len(dates), int); used = target.copy(); failures = []
            for ordinal, row in enumerate(rows.itertuples(index=False)):
                start = int(row.signal_idx)
                end = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
                pending = start + 1 == len(dates); rejected = row.position == reject_position
                closed_fill = end < len(dates) - 1
                okay = (0 <= start < end <= len(dates) and row.position in (-1, 1) and row.original_episode == ordinal
                        and dates[start] == pd.Timestamp(row.signal_date) and bool(row.entry_pending) == pending
                        and row.reject_position == reject_position and bool(row.rejected) == rejected and bool(row.accepted) != rejected
                        and row.status == ('REJECT' if rejected else 'KEEP') and row.condition == f'position=={reject_position:+d}')
                if pending:
                    okay &= pd.isna(row.entry_idx) and pd.isna(row.entry_date)
                    pending_rows.append({'candidate_id': cid, 'signal_idx': start})
                else:
                    trade = original_by_signal.loc[start]
                    okay &= row.position == trade.position and row.entry_idx == start + 1 and pd.Timestamp(row.entry_date) == dates[start + 1]
                    okay &= end == int(trade.exit_idx) - 1 if trade.closed else end >= len(dates) - 1
                for idx_col, date_col, exists, idx in [
                        ('actual_entry_idx', 'actual_entry_date', not rejected and not pending, start + 1),
                        ('original_exit_idx', 'original_exit_date', closed_fill, end + 1),
                        ('actual_exit_idx', 'actual_exit_date', not rejected and closed_fill, end + 1),
                        ('original_end_signal_idx', 'original_end_signal_date', end < len(dates), end)]:
                    if idx_col not in rows or date_col not in rows:
                        okay = False
                    elif exists:
                        okay &= getattr(row, idx_col) == idx and pd.Timestamp(getattr(row, date_col)) == dates[idx]
                    else:
                        okay &= pd.isna(getattr(row, idx_col)) and pd.isna(getattr(row, date_col))
                okay &= bool(row.original_exit_pending) == (end == len(dates) - 1)
                okay &= bool(row.exit_pending) == (end == len(dates) - 1 and not rejected)
                target[start:end] = row.position; used[start:end] += 1
                total_decisions += 1; total_rejected += int(rejected)
                if not okay:
                    failures.append(start)
            self.check('direction_decisions_clock_and_original_path:' + cid, not failures and (used <= 1).all()
                       and np.array_equal(np.r_[0, target[:-1]], base_exec), mismatched_signals=failures[:10])
            # Independently remove the original executed direction. This equals
            # next-close execution of the unaltered surviving TARGET episodes.
            executed = np.where(base_exec == reject_position, 0, base_exec)
            expected_target = np.where(target == reject_position, 0, target)
            self.check('whole_episode_rejection_without_reentry:' + cid, np.array_equal(executed, np.r_[0, expected_target[:-1]]))
            held = np.r_[0, executed[:-1]]; gross = held * spot
            filtered_ledger = ledger_from_executed(dates, executed, gross)
            if filtered_ledger.empty:
                filtered_ledger = base_ledger.iloc[:0].copy()
            for split, start, end in periods:
                for variant in ('BASE', 'FILTER'):
                    values = dict(windows[key][split]) if variant == 'BASE' else window_from_path(dates, executed, held, gross, filtered_ledger, start, end)
                    n = windows[key][split]['closed_n']; values.update(base_closed_n=n, retention=values['closed_n'] / n if n else np.nan)
                    saved = metric_index.loc[cid, variant, split]
                    failed = [name for name, value in values.items() if
                              (pd.Timestamp(saved[name]) != value if name.endswith('_date') else not same_number(saved[name], value))]
                    self.check('independent_window:' + cid + ':' + variant + ':' + split, not failed, failed_fields=failed)
                    recomputed.append({**{name: item[name] for name in ('candidate_id', 'pair', 'strategy', 'filter_id')},
                                       'is_new': True, 'variant': variant, 'split': split, **values})
            if cid in frozen_ids:
                mask = (dates >= pd.Timestamp(protocol['is_start'])) & (dates <= pd.Timestamp(protocol['is_end']))
                for variant, e, h, g, trades in [('BASE', base_exec, base_held, base_gross, base_ledger), ('FILTER', executed, held, gross, filtered_ledger)]:
                    daily_rows.append(pd.DataFrame({'candidate_id': cid, 'variant': variant, 'split': 'IS', 'date': dates[mask],
                        'executed': e[mask], 'held': h[mask], 'gross_daily_return': g[mask], 'net_daily_return': g[mask]}))
                    ledger_rows.append(trades.assign(candidate_id=cid, variant=variant))
            if (index + 1) % 25 == 0:
                print('Direction candidates independently audited:', index + 1, '/', len(items), flush=True)
        if items:
            recomputed = pd.DataFrame(recomputed)
            self.compare_frames('reconstructed_path_original_SCREEN', self.read_csv('is_selection.csv'), screen_candidates(recomputed), ['candidate_id'])
        daily, frozen_metrics = self.read_csv('is_frozen_daily.csv.gz'), self.read_csv('is_frozen_metrics.csv')
        if frozen_ids:
            self.compare_frames('frozen_daily_independent', daily, pd.concat(daily_rows, ignore_index=True), ['candidate_id', 'variant', 'date'])
            self.compare_frames('frozen_metric_independent', frozen_metrics, recomputed.loc[recomputed.candidate_id.isin(frozen_ids) & recomputed.split.eq('IS')], ['candidate_id', 'variant', 'split'])
            self.compare_frames('frozen_ledger_independent', self.read_csv('is_frozen_trades.csv.gz'), pd.concat(ledger_rows, ignore_index=True), ['candidate_id', 'variant', 'entry_idx'])
            self.compare_frames('frozen_decisions_exact_subset', self.read_csv('is_frozen_decisions.csv.gz'), decisions.loc[decisions.candidate_id.isin(frozen_ids)], ['candidate_id', 'signal_idx'])
        else:
            self.check('zero_freeze_no_fabricated_paths', daily.empty and frozen_metrics.empty
                       and not (self.campaign / 'is_frozen_trades.csv.gz').exists() and not (self.campaign / 'is_frozen_decisions.csv.gz').exists())
        self.details.update(BASE_accounts_independently_reconstructed=180, independent_metric_windows=len(items) * 8,
                            original_episode_decisions_checked=total_decisions, rejected_episode_decisions=total_rejected,
                            pending_terminal_original_signal_limit={'rows': pending_rows,
                                'scope': 'Final unexecuted BASE signals cannot be recovered from an executed ledger. Direction rejection and pending clocks are internally checked; these signals have no observed execution or realized PnL.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/direction_filter_v6')
    args = parser.parse_args(); audit = DirectionAudit(args.output.resolve())
    started = datetime.now(timezone.utc).isoformat()
    for path in (__file__, ROOT / 'verify_entry_interaction_research.py', ROOT / 'verify_state_filter_research.py', ROOT / 'verify_state_filter_supplement.py'):
        audit.register(path)
    try:
        audit.run()
    except Exception:
        audit.check('audit_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(row['passed'] for row in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
              'checks_total': len(audit.checks), 'checks_passed': passed, 'checks': audit.checks,
              'verified_files_sha256': audit.files, 'details': audit.details}
    destination = audit.campaign / 'independent_validation.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(destination), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
