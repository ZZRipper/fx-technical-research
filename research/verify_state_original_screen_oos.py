"""Independent post-OOS audit of the original fixed v2 state proposal campaign.

This campaign's completed OOS and the explicitly authorized prior JPY exact-rule
daily/metric comparison are opened. Raw source prices and independently
reconstructed prior-known native macro observations determine state decisions;
the original BASE execution ledger supplies episodes, never a strategy rerun.
All frozen members remain in original IS order, including previously exposed JPY.
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
from verify_state_original_screen import StateOriginalAudit, ROOT, BINS, TOL, PREVIOUSLY_REVIEWED, sha, same_number
from verify_entry_interaction_research import codes, ledger_from_executed, window_from_path
from verify_entry_interaction_oos import read_frozen_quotes


class StateOriginalPostAudit(StateOriginalAudit):
    def compare_authorized_prior_JPY(self, daily, metrics):
        """Compare only the explicitly authorized prior supplementary candidate."""
        source = ROOT / 'reproduced/state_filter_supplement_v1'
        status = self.document(source / 'oos_status.json')
        self.check('authorized_prior_JPY_completed_status', status['status'] == 'passed' and status['oos_evaluated'] is True)
        names = ('oos_daily.csv.gz', 'oos_metrics.csv')
        for name in names:
            self.check('authorized_prior_JPY_saved_hash:' + name, sha(self.register(source / name)) == status['output_sha256'][name])
        old_daily, old_metrics = self.frame(source / names[0]), self.frame(source / names[1])
        self.check('authorized_prior_tables_only_designated_JPY', set(old_daily.candidate_id) == {PREVIOUSLY_REVIEWED}
                   and set(old_metrics.candidate_id) == {PREVIOUSLY_REVIEWED})
        old_daily.date = pd.to_datetime(old_daily.date)
        local_daily = daily.loc[daily.candidate_id.eq(PREVIOUSLY_REVIEWED)]
        local_metrics = metrics.loc[metrics.candidate_id.eq(PREVIOUSLY_REVIEWED)]
        results = {}
        for label, previous, current, keys in [('daily', old_daily, local_daily, ['candidate_id', 'variant', 'date']),
                                               ('metrics', old_metrics, local_metrics, ['candidate_id', 'variant', 'split'])]:
            # Presentation names changed to display interval boundaries; all
            # other identifiers, paths, metrics and saved judgments must agree.
            columns = [c for c in previous if c != 'filter_name']
            self.compare_frames('authorized_prior_JPY_repeat:' + label, current, previous[columns], keys)
            a = current.sort_values(keys).reset_index(drop=True)
            b = previous.sort_values(keys).reset_index(drop=True)
            numeric = [c for c in columns if pd.api.types.is_numeric_dtype(b[c])]
            exact = len(a) == len(b) and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) for c in numeric)
            self.check('authorized_prior_JPY_bit_exact_numeric_repeat:' + label, exact)
            biggest = max((float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in numeric
                           if len(a) and np.isfinite(b[c].to_numpy(float)).any()), default=0.)
            results[label] = {'rows': len(b), 'numeric_columns': len(numeric), 'bit_exact_numeric_match': bool(exact),
                              'maximum_absolute_numeric_difference': biggest}
        return {'candidate_id': PREVIOUSLY_REVIEWED, 'source_campaign': 'state_filter_supplement_v1',
                'authorized_old_files': ['oos_status.json', *names], 'comparison': results,
                'interpretation': 'Deterministic reproduction of previously saved historical evidence, not a new independent OOS observation.'}

    def run(self):
        status = self.read_json('oos_status.json')
        if status.get('status') != 'passed' or status.get('oos_evaluated') is not True:
            raise ValueError('A completed authorized OOS is required before outcome-table access.')
        # This independent helper is not part of the frozen decision sources.
        # Import alone reads no data; actual raw reads follow the status guard.
        from fx_state_oos_audit_features import load_native_inputs, entry_feature_arrays
        protocol, freeze = self.read_json('protocol.json'), self.read_json('frozen_selection.json')
        pre = self.read_json('independent_validation.json')
        self.check('original_full_IS_preaudit_passed_unchanged', pre['status'] == 'passed'
                   and pre['checks_total'] == pre['checks_passed'] == len(pre['checks']) and all(c['passed'] for c in pre['checks'])
                   and pre['audit_script_sha256'] == sha(ROOT / 'verify_state_original_screen.py')
                   and pre['details']['OOS_results_opened'] is False and pre['details']['candidates'] == 1201)
        self.hashes('preaudit_evidence_unchanged', pre['verified_files_sha256'])
        self.check('freeze_and_passed_audit_before_OOS', pd.Timestamp(freeze['created_utc']) < pd.Timestamp(pre['started_utc'])
                   <= pd.Timestamp(pre['finished_utc']) < pd.Timestamp(status['opened_utc']) <= pd.Timestamp(status['finished_utc'])
                   and status['freeze_created_before_oos'] is True and status['retuning_permitted'] is False)
        self.check('OOS_exact_protocol_freeze_bindings', status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
                   and status['protocol_sha256'] == freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['source_integrity_pass'] and status['is_prefix_positions_and_returns_match'])
        self.check('fixed_original_state_universe_and_cost', protocol['version'] == 'state_original_screen_v7'
                   and protocol['state_bins'] == BINS and protocol['original_proposal_count'] == 1201
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
                   and protocol['oos_start'] == '2023-05-30' and protocol['oos_end'] == '2026-05-29')
        for field in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest:' + field, protocol[field] == freeze[field])
            self.hashes(field, protocol[field])
        self.hashes('IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.hashes('OOS_outputs', status['output_sha256'], self.campaign)
        items = freeze['candidates']; ids = [r['candidate_id'] for r in items]
        self.check('all_frozen_members_evaluated', len(items) == freeze['frozen_candidates'] == status['candidate_count']
                   and bool(items) and len(ids) == len(set(ids)))
        prior_known = {r['candidate_id'] for r in protocol['previously_evaluated_exact_candidates']}
        self.check('previously_exposed_JPY_preserved_and_disclosed', PREVIOUSLY_REVIEWED in prior_known
                   and all(r['spec'] == {'id': 'STATE_efficiency_63_1', 'feature': 'efficiency_63', 'state': 1}
                           for r in protocol['previously_evaluated_exact_candidates'] if r['candidate_id'] == PREVIOUSLY_REVIEWED))
        daily, prior_daily = self.read_csv('oos_daily.csv.gz'), self.read_csv('is_frozen_daily.csv.gz')
        daily.date, prior_daily.date = pd.to_datetime(daily.date), pd.to_datetime(prior_daily.date)
        ledger = self.read_csv('frozen_full_trade_ledger.csv.gz')
        for column in ('signal_date', 'entry_date', 'exit_date'):
            ledger[column] = pd.to_datetime(ledger[column])
        decisions, prior_decisions = self.read_csv('frozen_full_episode_decisions.csv.gz'), self.read_csv('is_frozen_decisions.csv.gz')
        metrics, annual, fresh = self.read_csv('oos_metrics.csv'), self.read_csv('oos_anniversary_metrics.csv'), self.read_csv('oos_fresh_start_metrics.csv')
        assessment = self.read_csv('candidate_assessment.csv')
        self.check('every_frozen_candidate_present_without_outcome_subselection', all(set(table.candidate_id) == set(ids)
                   for table in (daily, ledger, decisions, metrics, annual, fresh, assessment)))
        self.check('assessment_order_exactly_IS_freeze_rank', assessment.candidate_id.tolist() == ids
                   and assessment.freeze_rank.tolist() == [r['freeze_rank'] for r in items])
        self.check('unique_result_keys_and_dates', not daily.duplicated(['candidate_id', 'variant', 'date']).any()
                   and daily.date.between(protocol['oos_start'], protocol['oos_end']).all()
                   and not ledger.duplicated(['candidate_id', 'variant', 'entry_idx']).any()
                   and not decisions.duplicated(['candidate_id', 'signal_idx']).any())
        for label, frame in [('IS', prior_decisions), ('FULL', decisions)]:
            self.check('nonnullable_boolean_decisions:' + label, all(frame[c].notna().all() and frame[c].isin([True, False]).all()
                       for c in ('accepted', 'entry_pending')))
        price_book = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in p and p.endswith('.xlsx'))
        feature_book = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Features_Book' in p and p.endswith('.xlsx'))
        quotes = read_frozen_quotes(self.register(price_book), pd.Timestamp(protocol['oos_end']), {r['pair'] for r in items})
        native = load_native_inputs(self.register(feature_book), pd.Timestamp(protocol['oos_end']))
        rebuilt_ledgers, rebuilt_daily, rebuilt_IS = [], [], []
        metric_rows, annual_rows, fresh_rows, terminal_limits, feature_validations = [], [], [], [], []
        total_decisions = rejected_count = unknown_count = 0
        for index, item in enumerate(items):
            cid, pair, strategy = item['candidate_id'], item['pair'], item['strategy']
            feature, selected_state = item['spec']['feature'], item['spec']['state']
            self.check('frozen_spec_fixed_registry:' + cid, item['spec'] in protocol['candidate_specs']
                       and item['filter_id'] == item['spec']['id'] and type(selected_state) is int and selected_state in range(4))
            q = quotes[pair]; dates = pd.DatetimeIndex(q.date); prices = q.price.to_numpy(float)
            self.check('source_price_calendar:' + cid, dates.is_unique and dates.is_monotonic_increasing
                       and dates[-1] <= pd.Timestamp(protocol['oos_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1.]
            original = ledger.loc[ledger.candidate_id.eq(cid) & ledger.variant.eq('BASE')].sort_values('entry_idx')
            base_exec = np.zeros(len(dates), int); occupied = base_exec.copy()
            for trade in original.itertuples(index=False):
                end = int(trade.exit_idx) if trade.closed else len(dates)
                base_exec[int(trade.entry_idx):end] = int(trade.position)
                occupied[int(trade.entry_idx):end] += 1
            base_held = np.r_[0, base_exec[:-1]]; base_returns = base_held * spot
            self.check('BASE_episode_positions_disjoint:' + cid, (occupied <= 1).all())
            features = entry_feature_arrays(prices, dates, base_returns, pair, native)
            self.check('independent_feature_grid:' + cid, set(BINS).issubset(features)
                       and all(np.asarray(features[name]).shape == (len(dates),) for name in BINS))
            feature_validations.append({'candidate_id': cid, 'validation': features.get('_validation', {})})
            # Recheck prior-only macro dates at every finite native-derived row,
            # independently of the helper's own validation summary.
            macro_map = {'K_vix_obs_date': ('K_vix', 'K_vix_d5'), 'K_rate_obs_date': ('K_spread_bp', 'K_spread_d20_bp'),
                         'K_dxy_obs_date': ('K_dxy_d20_pct',), 'K_copper_obs_date': ('K_copper_d20_pct',),
                         'K_brent_obs_date': ('K_brent_d20_pct',)}
            for observation, sources in macro_map.items():
                known = np.zeros(len(dates), bool)
                for source in sources:
                    known |= np.isfinite(features['_known_inputs'][source])
                provenance = pd.DatetimeIndex(features['_provenance'][observation])
                age = dates - provenance
                self.check('independent_strict_prior_fresh_macro:' + cid + ':' + observation,
                           bool((provenance[known].notna() & (age[known] > pd.Timedelta(0)) & (age[known] <= pd.Timedelta(days=7))).all()))
            rows = decisions.loc[decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            previous = prior_decisions.loc[prior_decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            prefix = rows.loc[pd.to_datetime(rows.signal_date).le(protocol['is_end'])]
            stable = ['candidate_id', 'pair', 'strategy', 'original_episode', 'signal_idx', 'signal_date', 'position',
                      'feature', 'rejected_state', 'feature_value', 'state', 'accepted', 'status']
            self.compare_frames('all_frozen_IS_signal_states_unchanged:' + cid, prefix[stable], previous[stable], ['candidate_id', 'signal_idx'])
            filled = rows.loc[~rows.entry_pending]
            self.check('every_original_executed_episode_has_one_decision:' + cid, len(filled) == len(original)
                       and np.array_equal(filled.signal_idx.to_numpy(), original.signal_idx.to_numpy()))
            n_is = int((dates <= pd.Timestamp(protocol['is_end'])).sum())
            for row in previous.itertuples(index=False):
                pending_exit = pd.notna(row.original_end_signal_idx) and row.original_end_signal_idx == n_is - 1
                if row.entry_pending or pending_exit:
                    terminal_limits.append({'candidate_id': cid, 'signal_idx': int(row.signal_idx),
                                            'pending_entry': bool(row.entry_pending), 'pending_exit_or_reversal': bool(pending_exit)})
            original_by_signal = original.set_index('signal_idx')
            target = np.zeros(len(dates), int); filtered_target = target.copy(); used = target.copy()
            failed_signals, states_by_signal = [], {}
            for ordinal, row in enumerate(rows.itertuples(index=False)):
                start = int(row.signal_idx); end = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
                pending = start + 1 == len(dates)
                value = float(features[feature][start]) * (row.position if feature.startswith('direction_') else 1)
                state = int(codes([value], BINS[feature])[0]); accepted = state != selected_state
                states_by_signal[start] = state
                okay = (0 <= start < end <= len(dates) and row.position in (-1, 1) and row.original_episode == ordinal
                        and row.pair == pair and row.strategy == strategy and dates[start] == pd.Timestamp(row.signal_date)
                        and row.feature == feature and row.rejected_state == selected_state and row.state == state
                        and same_number(row.feature_value, value) and bool(row.accepted) == accepted
                        and row.status == ('UNKNOWN' if state == -1 else 'KEEP' if accepted else 'REJECT')
                        and bool(row.entry_pending) == pending)
                if not pending:
                    trade = original_by_signal.loc[start]
                    okay &= row.entry_idx == start + 1 and pd.Timestamp(row.entry_date) == dates[start + 1] and row.position == trade.position
                    okay &= end == int(trade.exit_idx) - 1 if trade.closed else end >= len(dates) - 1
                else:
                    okay &= pd.isna(row.entry_idx) and pd.isna(row.entry_date)
                if end < len(dates):
                    okay &= pd.Timestamp(row.original_end_signal_date) == dates[end]
                else:
                    okay &= pd.isna(row.original_end_signal_date)
                okay &= pd.isna(row.actual_entry_idx) if not accepted or pending else row.actual_entry_idx == start + 1
                target[start:end] = row.position; used[start:end] += 1
                if accepted:
                    filtered_target[start:end] = row.position
                if not okay:
                    failed_signals.append(start)
                total_decisions += 1; rejected_count += int(not accepted); unknown_count += int(state == -1)
            self.check('raw_price_native_macro_shadow_BASE_state_decisions:' + cid, not failed_signals and (used <= 1).all()
                       and np.array_equal(np.r_[0, target[:-1]], base_exec), mismatched_signals=failed_signals[:10])
            variant_paths = {}
            for variant, executed in [('BASE', base_exec), ('FILTER', np.r_[0, filtered_target[:-1]])]:
                held = np.r_[0, executed[:-1]]; returns = held * spot
                rebuilt = ledger_from_executed(dates, executed, returns)
                if rebuilt.empty:
                    rebuilt = original.iloc[:0].copy()
                variant_paths[variant] = executed, held, returns, rebuilt
                rebuilt_ledgers.append(rebuilt.assign(candidate_id=cid, variant=variant))
                if variant == 'FILTER':
                    survivor = original.loc[original.signal_idx.map(states_by_signal).ne(selected_state)]
                    self.compare_frames('exact_surviving_original_episodes:' + cid, survivor, rebuilt, ['entry_idx'])
                for split, start, end, collector in [('IS', protocol['is_start'], protocol['is_end'], rebuilt_IS),
                                                     ('OOS', protocol['oos_start'], protocol['oos_end'], rebuilt_daily)]:
                    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                    collector.append(pd.DataFrame({'candidate_id': cid, 'variant': variant, 'split': split, 'date': dates[mask],
                        'executed': executed[mask], 'held': held[mask], 'gross_daily_return': returns[mask], 'net_daily_return': returns[mask]}))
            for periods, collector in [([('OOS', protocol['oos_start'], protocol['oos_end'])], metric_rows),
                                       (protocol['oos_anniversary_years'], annual_rows)]:
                for split, start, end in periods:
                    values = {v: window_from_path(dates, *variant_paths[v], start, end) for v in ('BASE', 'FILTER')}
                    for variant, result in values.items():
                        n = values['BASE']['closed_n']
                        collector.append({'candidate_id': cid, 'variant': variant, 'split': split,
                                          'base_closed_n': n, 'retention': result['closed_n'] / n if n else np.nan, **result})
            allowed = np.zeros(len(dates), bool)
            for trade in variant_paths['BASE'][3].itertuples(index=False):
                if trade.entry_date >= pd.Timestamp(protocol['oos_start']):
                    end = int(trade.exit_idx) if trade.closed else len(dates)
                    allowed[int(trade.entry_idx):end] = True
            cash_values = {}
            for variant, (executed, _, _, _) in variant_paths.items():
                cash = np.where(allowed, executed, 0); held = np.r_[0, cash[:-1]]; returns = held * spot
                rebuilt = ledger_from_executed(dates, cash, returns)
                if rebuilt.empty:
                    rebuilt = original.iloc[:0].copy()
                cash_values[variant] = window_from_path(dates, cash, held, returns, rebuilt, protocol['oos_start'], protocol['oos_end'])
            for variant, result in cash_values.items():
                n = cash_values['BASE']['closed_n']
                fresh_rows.append({'candidate_id': cid, 'variant': variant, 'split': 'OOS_FRESH_START',
                                   'base_closed_n': n, 'retention': result['closed_n'] / n if n else np.nan, **result})
            print('Frozen state candidate independently audited:', index + 1, '/', len(items), cid, flush=True)
        self.compare_frames('full_ledger_from_raw_inputs', ledger, pd.concat(rebuilt_ledgers, ignore_index=True), ['candidate_id', 'variant', 'entry_idx'])
        self.check('full_ledger_zero_cost_dollars', ledger.transaction_cost_usd.eq(0).all()
                   and np.allclose(ledger.gross_simple_return_pct, ledger.net_trade_return_pct, atol=1e-8, rtol=1e-10)
                   and np.allclose(ledger.net_pnl_usd, ledger.net_trade_return_pct * 1000, atol=1e-5, rtol=1e-10))
        self.compare_frames('frozen_IS_daily_prefix', prior_daily, pd.concat(rebuilt_IS, ignore_index=True), ['candidate_id', 'variant', 'date'])
        self.compare_frames('all_continuous_OOS_daily_paths', daily, pd.concat(rebuilt_daily, ignore_index=True), ['candidate_id', 'variant', 'date'])
        computed, computed_annual, computed_fresh = map(pd.DataFrame, (metric_rows, annual_rows, fresh_rows))
        for label, saved, expected in [('OOS_continuous_all_metrics', metrics, computed), ('all_anniversary_metrics', annual, computed_annual),
                                       ('fresh_start_diagnostic_metrics', fresh, computed_fresh)]:
            self.compare_frames(label, saved, expected, ['candidate_id', 'variant', 'split'])
        expected_assessment = []; gate = protocol['oos_screen']
        for item in items:
            cid = item['candidate_id']; account = computed.loc[computed.candidate_id.eq(cid)].set_index('variant')
            b, f = account.loc['BASE'], account.loc['FILTER']
            positive = int((computed_annual.loc[computed_annual.candidate_id.eq(cid) & computed_annual.variant.eq('FILTER'), 'net_simple_return_pct'] > TOL).sum())
            delta = float(f.net_simple_return_pct - b.net_simple_return_pct)
            flags = {'annualized_pass': bool(f.net_annualized_simple_pct >= gate['min_net_annualized_simple_pct'] - TOL),
                     'baseline_improvement_pass': bool(delta >= gate['min_net_return_improvement_pp'] - TOL),
                     'drawdown_pass': bool(f.net_max_drawdown_pp >= gate['worst_net_drawdown_pp'] - TOL),
                     'positive_years_pass': positive >= gate['min_positive_anniversary_years'],
                     'sample_pass': bool(f.closed_n >= gate['min_closed_trades'])}
            quality = bool(f.closed_n >= 20 and np.isfinite([b.win_rate, f.win_rate, b.payoff_ratio, f.payoff_ratio]).all()
                           and f.win_rate >= b.win_rate - TOL and f.payoff_ratio >= b.payoff_ratio - TOL)
            expected_assessment.append({**{k: item[k] for k in ('candidate_id', 'pair', 'strategy', 'filter_id', 'freeze_rank')},
                'is_return_pct': item['is_gross_return_pct'], 'oos_return_pct': f.net_simple_return_pct,
                'oos_base_return_pct': b.net_simple_return_pct, 'oos_improvement_pp': delta,
                'oos_annualized_simple_pct': f.net_annualized_simple_pct, 'oos_max_drawdown_pp': f.net_max_drawdown_pp,
                'oos_win_rate': f.win_rate, 'oos_base_win_rate': b.win_rate, 'oos_payoff_ratio': f.payoff_ratio,
                'oos_base_payoff_ratio': b.payoff_ratio, 'oos_closed_n': int(f.closed_n), 'oos_positive_years': positive,
                **flags, 'oos_curve_pass': all(flags.values()), 'oos_quality_confirmed': quality,
                'failed_criteria': '|'.join(k for k, value in flags.items() if not value)})
        expected_assessment = pd.DataFrame(expected_assessment)
        self.compare_frames('all_prespecified_curve_quality_flags_and_failures', assessment, expected_assessment, ['candidate_id'])
        self.compare_frames('metric_assessment_flags', metrics[['candidate_id', 'variant', 'oos_curve_pass', 'oos_quality_confirmed']],
                            computed[['candidate_id', 'variant']].merge(expected_assessment[['candidate_id', 'oos_curve_pass', 'oos_quality_confirmed']], on='candidate_id'),
                            ['candidate_id', 'variant'])
        both = expected_assessment.oos_curve_pass & expected_assessment.oos_quality_confirmed
        self.check('status_counts_independently_reproduced', status['curve_passing_candidates'] == int(expected_assessment.oos_curve_pass.sum())
                   and status['quality_confirmed_candidates'] == int(expected_assessment.oos_quality_confirmed.sum())
                   and status['both_curve_and_quality_candidates'] == int(both.sum()))
        prior_repeat = self.compare_authorized_prior_JPY(daily, metrics)
        self.check('all_evidence_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(candidate_ids_in_frozen_order=ids, candidates=len(ids),
                            curve_passes=int(expected_assessment.oos_curve_pass.sum()), quality_confirmed=int(expected_assessment.oos_quality_confirmed.sum()),
                            both_curve_and_quality=int(both.sum()), independently_recomputed_metric_windows=len(computed) + len(computed_annual) + len(computed_fresh),
                            full_episode_decisions_checked=total_decisions, rejected_original_episode_decisions=rejected_count,
                            UNKNOWN_episode_decisions_retained=unknown_count, all_frozen_IS_paths_match=True,
                            strategy_engine_replays=0, current_campaign_outcome_audit_complete=True,
                            previously_evaluated_exact_candidates_preserved=sorted(set(ids) & prior_known),
                            prior_exact_replay_is_not_independent_evidence=True, old_OOS_results_opened=True,
                            authorized_prior_exact_replay=prior_repeat,
                            raw_price_currencies=sorted(quotes), raw_input_cutoff=protocol['oos_end'],
                            native_feature_input_validation=native.get('_validation', {}), feature_alignment_validations=feature_validations,
                            pending_IS_terminal_original_signal_limit={'rows': terminal_limits,
                                'scope': 'Original BASE signal generation is not rerun. Terminal pending entry/exit/reversal cannot be independently inferred from an executed ledger; full replay checks realized positions and all state values use independently rebuilt raw-input features.'},
                            historical_exposure='Every result is repeated historical verification; previously supplemented JPY is additionally an exact-rule repeat.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/state_original_screen_v7')
    args = parser.parse_args(); audit = StateOriginalPostAudit(args.output.resolve())
    started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_state_original_screen.py', ROOT / 'verify_entry_interaction_research.py',
                   ROOT / 'verify_entry_interaction_oos.py', ROOT / 'verify_state_filter_research.py',
                   ROOT / 'verify_state_filter_supplement.py', ROOT / 'src/fx_state_oos_audit_features.py'):
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
