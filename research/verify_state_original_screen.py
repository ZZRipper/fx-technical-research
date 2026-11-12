"""Independent IS-only audit of all original v2 discovery state proposals.

The historical 1201-member list is preserved exactly. Original D nomination
rules are reconstructed only to verify provenance, not to add a new IS gate.
All paths use cached signal-time BASE states and raw IS prices; no strategy or
candidate engine, previous OOS result, or later price is evaluated.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_entry_interaction_research import (
    InteractionAudit, ROOT, TOL, BINS, SOURCE_LEDGER, CURRENCIES, STRATEGIES,
    sha, same_number, truth, codes, independent_quality, load_D_ledger,
    read_IS_quotes, ledger_from_executed, window_from_path, SCREEN, screen_candidates,
)
import itertools

OLD = ROOT / 'reproduced/state_filter_v2'
CACHE = SOURCE_LEDGER.parent
DISCOVERY = (('DISCOVERY_1', '2005-01-03', '2009-12-31'),
             ('DISCOVERY_2', '2010-01-01', '2014-12-31'), ('DEVELOPMENT', '2005-01-03', '2014-12-31'))
PREVIOUSLY_REVIEWED = 'JPY|BB_MR_20|STATE_efficiency_63_1'
MACRO_SOURCES = {'K_vix_obs_date': ('K_vix', 'K_vix_d5'),
                 'K_rate_obs_date': ('K_spread_bp', 'K_spread_d20_bp'),
                 'K_dxy_obs_date': ('K_dxy_d20_pct',), 'K_copper_obs_date': ('K_copper_d20_pct',),
                 'K_brent_obs_date': ('K_brent_d20_pct',)}


def state_summary(values):
    values = np.asarray(values, float); total = float(values.sum())
    return {**independent_quality(values), 'return_sum_pct': total,
            'largest_win_pct': float(values.max()) if len(values) else np.nan,
            'largest_loss_pct': float(values.min()) if len(values) else np.nan,
            'return_without_best_trade_pct': total - max(0., float(values.max())) if len(values) else 0.,
            'return_without_worst_trade_pct': total - min(0., float(values.min())) if len(values) else 0.}


class StateOriginalAudit(InteractionAudit):
    def old_evidence_chain(self):
        protocol, proposals = self.document(OLD / 'protocol.json'), self.document(OLD / 'discovery_proposals.json')
        lock, status = self.document(OLD / 'validation_locked.json'), self.document(OLD / 'fit_status.json')
        self.check('old_proposals_bound_to_preconfirmation_lock', lock['protocol_sha256'] == sha(OLD / 'protocol.json')
                   and status['lock_sha256'] == sha(OLD / 'validation_locked.json')
                   and lock['fit_output_sha256']['discovery_proposals.json'] == sha(OLD / 'discovery_proposals.json')
                   and status['status'] == 'passed' and status['proposals'] == len(proposals['candidates']) == 1201
                   and lock['confirmation_computed'] is False and lock['oos_computed'] is False)
        self.check('old_creation_order', pd.Timestamp(protocol['created_utc']) < pd.Timestamp(proposals['created_utc'])
                   < pd.Timestamp(lock['created_utc']) <= pd.Timestamp(status['finished_utc']))
        self.hashes('old_decision_sources', protocol['source_sha256'])
        self.hashes('old_inputs', protocol['input_sha256'])
        self.hashes('old_fit_outputs', lock['fit_output_sha256'], OLD)
        original = self.document(CACHE / 'entry_state_protocol.json')
        validation = self.document(CACHE / 'entry_state_validation.json')
        amendment = self.document(CACHE / 'technical_validation_amendment.json')
        self.check('original_cache_scope_and_defensive_amendment', original['compute_cutoff'] == '2023-05-29'
                   and original['oos_computed'] is False and original['cost_bp'] == 0
                   and validation['status'] == amendment['status'] == 'passed' and validation['accounts'] == 180
                   and original['state_bins'] == protocol['state_bins'] == BINS
                   and amendment['original_protocol_sha256'] == sha(CACHE / 'entry_state_protocol.json')
                   and original['source_sha256']['src/fx_is_states.py'] == amendment['original_source_sha256']
                   and amendment['amended_source_sha256'] == sha(ROOT / 'src/fx_is_states.py')
                   and amendment['original_source_sha256'] == sha(ROOT / amendment['original_source_snapshot']))
        amended_sources = dict(original['source_sha256'])
        amended_sources['src/fx_is_states.py'] = amendment['amended_source_sha256']
        self.hashes('amended_cache_sources', amended_sources)
        self.hashes('cache_inputs', original['input_sha256'])
        self.hashes('cache_preserved_outputs', amendment['preserved_files_sha256'])
        self.hashes('cache_validation_outputs', validation['output_sha256'], CACHE)
        return proposals

    def discovery(self, old):
        frame, skipped = load_D_ledger(self.register(SOURCE_LEDGER))
        self.check('original_D_entry_close_clock', np.isfinite(frame.gross_simple_return_pct).all()
                   and frame.transaction_cost_usd.eq(0).all()
                   and frame.signal_date.lt(frame.entry_date).all() and frame.entry_date.lt(frame.exit_date).all()
                   and not frame.duplicated(['pair', 'strategy', 'trade_id']).any())
        for feature, edges in BINS.items():
            self.check('D_state_encoding:' + feature, np.array_equal(codes(frame[feature], edges), frame[feature + '_state']))
        rebuilt, nominations = [], []
        for pair, strategy in itertools.product(CURRENCIES, STRATEGIES):
            account = frame.loc[frame.pair.eq(pair) & frame.strategy.eq(strategy)]
            cohorts = {block: account.loc[account.entry_date.ge(start) & account.exit_date.le(end)] for block, start, end in DISCOVERY}
            for feature in BINS:
                for state in (-1, 0, 1, 2, 3):
                    evidence = {}
                    for block, cohort in cohorts.items():
                        values = cohort.loc[codes(cohort[feature], BINS[feature]) == state, 'gross_simple_return_pct']
                        stats = state_summary(values)
                        rebuilt.append({'pair': pair, 'strategy': strategy, 'feature': feature, 'state': state,
                                        'block': block, 'cohort_n': len(cohort), **stats})
                        evidence[block] = {n: stats[n] for n in ('closed_n', 'return_sum_pct', 'return_without_worst_trade_pct')}
                    # This is the exact OLD v2 nomination rule, with strict <0
                    # comparisons. It verifies the fixed list, not a new gate.
                    if state >= 0 and all(evidence[b]['closed_n'] >= 10 and evidence[b]['return_sum_pct'] < 0
                                          for b in ('DISCOVERY_1', 'DISCOVERY_2')) \
                            and evidence['DEVELOPMENT']['return_without_worst_trade_pct'] < 0:
                        identifier = f'STATE_{feature}_{state}'
                        nominations.append({'candidate_id': '|'.join((pair, strategy, identifier)), 'pair': pair,
                                            'strategy': strategy, 'filter_id': identifier,
                                            'spec': {'id': identifier, 'feature': feature, 'state': state},
                                            'discovery_evidence': evidence})
        columns = list(pd.DataFrame(rebuilt).columns); numeric = set(columns) - {'pair', 'strategy', 'feature', 'block'}
        saved = []
        with self.register(CACHE / 'is_entry_state_summary.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['block'] not in {p[0] for p in DISCOVERY}:
                    continue  # Later outcomes are not numerically parsed.
                saved.append({c: float(row[c]) if row[c] else np.nan for c in columns if c in numeric}
                             | {c: row[c] for c in columns if c not in numeric})
        self.compare_frames('all_original_D_state_summaries', pd.DataFrame(saved), pd.DataFrame(rebuilt),
                            ['pair', 'strategy', 'feature', 'state', 'block'])
        expected = {r['candidate_id']: r for r in nominations}; original = old['candidates']
        self.check('exact_original_1201_D_only_nominations', len(expected) == len(original) == 1201
                   and {r['candidate_id'] for r in original} == set(expected))
        for row in original:
            reference = expected[row['candidate_id']]
            self.check('old_D_nomination:' + row['candidate_id'], row['spec'] == reference['spec']
                       and all(row[n] == reference[n] for n in ('pair', 'strategy', 'filter_id')) and row['is_new'] is True
                       and set(row['discovery_evidence']) == set(reference['discovery_evidence'])
                       and all(same_number(row['discovery_evidence'][block][name], value)
                               for block, values in reference['discovery_evidence'].items() for name, value in values.items()))
        self.details.update(original_D_nomination_rows=len(original), D_state_summary_rows=len(rebuilt),
                            D_closed_outcomes_decoded=len(frame), later_or_nonclosed_outcomes_skipped_before_D_parse=skipped,
                            nomination_scope='Exactly the previously saved 1201 D-only candidates. The old worst-loss-removed D condition is provenance, not an added full-IS SCREEN criterion.')
        return expected

    def run(self):
        protocol, status = self.read_json('protocol.json'), self.read_json('development_status.json')
        freeze, hypotheses = self.read_json('frozen_selection.json'), self.read_json('is_hypotheses.json')
        old = self.old_evidence_chain()
        review = self.document(ROOT / 'reproduced/state_original_screen_scope_review/review.json')
        self.check('completed_scope_review_before_v7', review['status'] == 'passed'
                   and review['original_proposal_count'] == 1201 and review['no_new_filtered_returns_computed'] is True
                   and review['OOS_results_opened'] is False)
        self.hashes('scope_review_sources', review['source_sha256'])
        self.hashes('scope_review_inputs', review['input_sha256'])
        self.check('fixed_v7_original_SCREEN_zero_cost_scope', protocol['version'] == 'state_original_screen_v7'
                   and protocol['is_start'] == '2005-01-03' and protocol['is_end'] == '2023-05-29'
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0 and protocol['screen'] == SCREEN
                   and protocol['initial_capital_usd'] == 100000 and protocol['accounting'] == 'fixed_initial_capital_simple'
                   and protocol['state_bins'] == BINS and protocol['scope']['baseline_accounts'] == 180
                   and set(protocol['scope']['pairs']) == set(CURRENCIES) and set(protocol['scope']['strategies']) == set(STRATEGIES))
        self.check('IS_completed_before_OOS', status['status'] == 'passed' and status['compute_cutoff'] == protocol['is_end']
                   and status['is_screen_completed'] and status['source_integrity_pass']
                   and status['oos_evaluated'] is False and status['holdout_returns_computed'] is False)
        self.check('freeze_protocol_chain', freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json') and freeze['oos_evaluated_at_freeze'] is False)
        self.check('original_list_before_v7_and_freeze', pd.Timestamp(old['created_utc']) < pd.Timestamp(protocol['created_utc'])
                   < pd.Timestamp(hypotheses['created_utc']) <= pd.Timestamp(status['started_utc'])
                   < pd.Timestamp(freeze['created_utc']) <= pd.Timestamp(status['finished_utc']))
        for key in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest:' + key, protocol[key] == freeze[key])
            self.hashes(key, protocol[key])
        self.hashes('observation_source_chain', protocol['observation_source_sha256'])
        self.hashes('frozen_IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.check('all_IS_artifacts_in_freeze', {p.name for p in self.campaign.glob('is_*') if p.is_file()} == set(freeze['is_output_sha256']))
        self.check('OOS_outputs_absent', all(not (self.campaign / n).exists() for n in ('oos_status.json', 'oos_daily.csv.gz',
                   'oos_metrics.csv', 'oos_anniversary_metrics.csv', 'oos_fresh_start_metrics.csv', 'candidate_assessment.csv',
                   'frozen_full_trade_ledger.csv.gz', 'frozen_full_episode_decisions.csv.gz')))
        expected = self.discovery(old); items = hypotheses['candidates']
        original_by_id = {h['candidate_id']: h for h in old['candidates']}
        self.check('v7_preserves_exact_old_members_specs_evidence_order', len(items) == len(old['candidates'])
                   and [h['candidate_id'] for h in items] == [h['candidate_id'] for h in old['candidates']]
                   and all(all(h[n] == original_by_id[h['candidate_id']][n] for n in
                               ('pair', 'strategy', 'filter_id', 'spec', 'discovery_evidence')) for h in items))
        used_specs = {h['filter_id']: h['spec'] for h in items}
        registry = {s['id']: s for s in protocol['candidate_specs']}
        self.check('no_new_feature_state_parameters', len(registry) == len(protocol['candidate_specs'])
                   and all(registry.get(k) == v for k, v in used_specs.items())
                   and all(set(s) == {'id', 'feature', 'state'} and s['feature'] in BINS and type(s['state']) is int
                           and s['state'] in range(4) and s['id'] == f"STATE_{s['feature']}_{s['state']}" for s in registry.values()))
        metrics, selection, coverage = self.read_csv('is_all_metrics.csv'), self.read_csv('is_selection.csv'), self.read_csv('is_account_coverage.csv')
        counts = pd.Series([(h['pair'], h['strategy']) for h in items], dtype=object).value_counts()
        self.check('180_account_coverage_and_exact_proposal_counts', len(coverage) == 180
                   and set(zip(coverage.pair, coverage.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES))
                   and coverage.compute_end.eq(protocol['is_end']).all()
                   and all(r.candidate_count == counts.get((r.pair, r.strategy), 0) for r in coverage.itertuples())
                   and coverage.candidate_count.sum() == len(items))
        periods = [('IS', protocol['is_start'], protocol['is_end']), *protocol['is_blocks']]
        self.check('all_1201_candidates_all_eight_IS_windows', set(selection.candidate_id) == set(expected)
                   and not metrics.duplicated(['candidate_id', 'variant', 'split']).any()
                   and set(zip(metrics.candidate_id, metrics.variant, metrics.split))
                   == set(itertools.product(expected, ('BASE', 'FILTER'), [p[0] for p in periods])))
        rescreened = screen_candidates(metrics)
        self.compare_frames('original_SCREEN_only', selection, rescreened, ['candidate_id'])
        self.check('original_ranking_order', selection.candidate_id.tolist() == rescreened.candidate_id.tolist())
        winners = rescreened.loc[rescreened.frozen.map(truth)].sort_values('freeze_rank')
        self.check('exact_original_SCREEN_rank_freeze', winners.candidate_id.tolist() == [h['candidate_id'] for h in freeze['candidates']]
                   and freeze['frozen_candidates'] == status['frozen_candidates'] == len(winners)
                   and freeze['all_passing_versions'] == status['all_passing_versions'] == int(rescreened['pass'].sum())
                   and freeze['all_candidate_accounts_tested'] == status['candidates_evaluated'] == len(items))
        for item in freeze['candidates']:
            row = winners.set_index('candidate_id').loc[item['candidate_id']]
            self.check('frozen_values:' + item['candidate_id'], item['spec'] == expected[item['candidate_id']]['spec'] and all(
                same_number(item[c], row[c]) if isinstance(row[c], (float, int, np.number, bool, np.bool_)) else
                (('' if item[c] is None else str(item[c])) == ('' if pd.isna(row[c]) else str(row[c]))) for c in rescreened if c != 'candidate_id'))
        self.audit_state_paths(protocol, items, metrics, winners, periods)
        self.check('all_registered_inputs_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(campaign_version=protocol['version'], observed_accounts=180, candidates=len(items),
                            candidate_accounts=int(coverage.candidate_count.gt(0).sum()), original_SCREEN_passes=int(rescreened['pass'].sum()),
                            frozen_candidates=len(winners), OOS_results_opened=False, holdout_returns_computed=False,
                            strategy_engine_replays=0, raw_price_cutoff=protocol['is_end'], post_IS_price_values_decoded=0,
                            previously_supplemented_candidate={'candidate_id': PREVIOUSLY_REVIEWED,
                                'included_in_unmodified_original_list': PREVIOUSLY_REVIEWED in expected,
                                'frozen_again': PREVIOUSLY_REVIEWED in set(winners.candidate_id),
                                'disclosure': 'Previous supplementary historical OOS exposure is known from task provenance; no old OOS result was read. Re-freezing would not create new independent evidence.'})

    def audit_state_paths(self, protocol, items, metrics, winners, periods):
        ledger = self.frame(SOURCE_LEDGER)
        for c in ('signal_date', 'entry_date', 'exit_date'):
            ledger[c] = pd.to_datetime(ledger[c])
        self.check('original_IS_ledger_unique_scope', not ledger.duplicated(['pair', 'strategy', 'signal_idx']).any()
                   and ledger.exit_date.le(protocol['is_end']).all()
                   and set(zip(ledger.pair, ledger.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES)))
        for feature, edges in BINS.items():
            self.check('all_IS_cached_state_encoding:' + feature, np.array_equal(codes(ledger[feature], edges), ledger[feature + '_state']))
        for observation, sources in MACRO_SOURCES.items():
            known = np.isfinite(ledger[list(sources)]).any(axis=1)
            age = ledger.signal_date - pd.to_datetime(ledger[observation])
            self.check('macro_known_strictly_prior_and_fresh:' + observation,
                       (age[known].notna() & age[known].gt(pd.Timedelta(0)) & age[known].le(pd.Timedelta(days=7))).all())
        workbook = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in p and p.endswith('.xlsx'))
        quotes = read_IS_quotes(self.register(workbook), pd.Timestamp(protocol['is_end']))
        paths, windows = {}, {}
        for pair, frame in quotes.items():
            dates = pd.DatetimeIndex(frame.date); prices = frame.price.to_numpy(float)
            self.check('IS_raw_price_calendar:' + pair, dates.is_unique and dates.is_monotonic_increasing
                       and dates[-1] <= pd.Timestamp(protocol['is_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1.]
            price = pd.Series(prices); price_returns = price.pct_change()
            vol20 = price_returns.rolling(20).std() * np.sqrt(252) * 100
            vol63 = price_returns.rolling(63).std() * np.sqrt(252) * 100
            length = price.diff().abs().rolling(63).sum()
            scale = (vol20 * np.sqrt(20 / 252)).where(vol20.gt(0))
            technical = {'realized_vol_20_pct': vol20.to_numpy(),
                         'efficiency_63': (price.diff(63).abs() / length.where(length.ne(0))).mask(length.eq(0), 0.).to_numpy(),
                         'vol_ratio_20_63': (vol20 / vol63.where(vol63.gt(0))).to_numpy(),
                         'direction_return_20_vol': (price.pct_change(20) * 100 / scale).to_numpy(),
                         'direction_distance_ma_60_vol': ((price / price.rolling(60).mean() - 1) * 100 / scale).to_numpy()}
            for strategy in STRATEGIES:
                key = pair, strategy
                original = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)].sort_values('entry_idx')
                executed = np.zeros(len(dates), int); used = executed.copy()
                for trade in original.itertuples(index=False):
                    stop = int(trade.exit_idx) if trade.closed else len(dates)
                    executed[int(trade.entry_idx):stop] = int(trade.position)
                    used[int(trade.entry_idx):stop] += 1
                held = np.r_[0, executed[:-1]]; gross = held * spot
                rebuilt = ledger_from_executed(dates, executed, gross)
                self.compare_frames('BASE_ledger:' + '|'.join(key), original, rebuilt, ['entry_idx'])
                self.check('BASE_no_overlap_zero_cost:' + '|'.join(key), (used <= 1).all() and original.transaction_cost_usd.eq(0).all())
                idx = original.signal_idx.to_numpy(int); direction = original.position.to_numpy(float)
                values = {feature: v[idx] * (direction if feature.startswith('direction_') else 1) for feature, v in technical.items()}
                for feature, source in [('direction_spread_bp', 'K_spread_bp'), ('direction_spread_d20_bp', 'K_spread_d20_bp'),
                                        ('direction_dxy_d20_pct', 'K_dxy_d20_pct')]:
                    values[feature] = original[source].to_numpy(float) * direction
                commodity = original.K_copper_d20_pct if pair == 'AUD' else original.K_brent_d20_pct
                values['direction_commodity_d20_pct'] = commodity.to_numpy(float) * direction if pair in ('AUD', 'CAD', 'NOK') else np.full(len(original), np.nan)
                for window in (63, 252):
                    values[f'base_pnl_{window}_pct'] = pd.Series(gross).rolling(window).sum().to_numpy()[idx] * 100
                self.check('known_entry_feature_formulas:' + '|'.join(key), all(np.allclose(original[name], value, atol=1e-8, rtol=1e-10, equal_nan=True)
                           for name, value in values.items()))
                # Formula paths below also verify terminal technical/shadow-PnL
                # values when the last original entry has not filled yet.
                full_values = dict(technical)
                for window in (63, 252):
                    full_values[f'base_pnl_{window}_pct'] = pd.Series(gross).rolling(window).sum().to_numpy() * 100
                paths[key] = dates, spot, executed, held, gross, rebuilt, original, full_values
                windows[key] = {split: window_from_path(dates, executed, held, gross, rebuilt, start, end) for split, start, end in periods}
        decisions = self.read_csv('is_all_decisions.csv.gz')
        self.check('all_proposal_decisions_unique_complete', set(decisions.candidate_id) == {h['candidate_id'] for h in items}
                   and not decisions.duplicated(['candidate_id', 'signal_idx']).any()
                   and all(decisions[c].notna().all() and decisions[c].isin([True, False]).all() for c in ('accepted', 'entry_pending')))
        grouped = {cid: group.sort_values('signal_idx') for cid, group in decisions.groupby('candidate_id', sort=False)}
        metric_index = metrics.set_index(['candidate_id', 'variant', 'split'])
        recomputed, frozen_daily, frozen_ledgers, pending_rows = [], [], [], []
        total_decisions = unknown_count = reject_count = 0; frozen_ids = set(winners.candidate_id)
        for offset, item in enumerate(items):
            cid, key = item['candidate_id'], (item['pair'], item['strategy'])
            feature, selected_state = item['spec']['feature'], item['spec']['state']
            dates, spot, base_exec, base_held, base_gross, base_ledger, original, full_values = paths[key]
            rows = grouped[cid]; filled = rows.loc[~rows.entry_pending]
            self.check('original_episode_decision_coverage:' + cid, len(filled) == len(original)
                       and np.array_equal(filled.signal_idx.to_numpy(), original.signal_idx.to_numpy()))
            original_index = original.set_index('signal_idx')
            target = np.zeros(len(dates), int); filtered_target = target.copy(); used = target.copy(); failed = []
            for ordinal, row in enumerate(rows.itertuples(index=False)):
                start = int(row.signal_idx); stop = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
                pending = start + 1 == len(dates)
                state = int(codes([row.feature_value], BINS[feature])[0]); accepted = state != selected_state
                okay = (0 <= start < stop <= len(dates) and dates[start] == pd.Timestamp(row.signal_date)
                        and row.position in (-1, 1) and row.original_episode == ordinal and row.feature == feature
                        and row.rejected_state == selected_state and row.state == state and bool(row.accepted) == accepted
                        and row.status == ('UNKNOWN' if state == -1 else 'KEEP' if accepted else 'REJECT')
                        and bool(row.entry_pending) == pending)
                if pending:
                    okay &= pd.isna(row.entry_idx) and pd.isna(row.entry_date)
                    if feature in full_values:
                        value = full_values[feature][start] * (row.position if feature.startswith('direction_') else 1)
                        okay &= same_number(row.feature_value, value)
                else:
                    trade = original_index.loc[start]
                    okay &= row.entry_idx == start + 1 and pd.Timestamp(row.entry_date) == dates[start + 1] and row.position == trade.position
                    okay &= same_number(row.feature_value, trade[feature]) and row.state == trade[feature + '_state']
                    okay &= stop == int(trade.exit_idx) - 1 if trade.closed else stop >= len(dates) - 1
                if stop < len(dates):
                    okay &= pd.Timestamp(row.original_end_signal_date) == dates[stop]
                else:
                    okay &= pd.isna(row.original_end_signal_date)
                okay &= pd.isna(row.actual_entry_idx) if not accepted or pending else row.actual_entry_idx == start + 1
                if pending or stop == len(dates) - 1:
                    pending_rows.append({'candidate_id': cid, 'signal_idx': start, 'pending_entry': pending,
                                         'pending_exit_or_reversal': stop == len(dates) - 1,
                                         'terminal_entry_feature_formula_independently_recomputed': bool(pending and feature in full_values)})
                target[start:stop] = row.position; used[start:stop] += 1
                if accepted:
                    filtered_target[start:stop] = row.position
                unknown_count += int(state == -1); reject_count += int(not accepted); total_decisions += 1
                if not okay:
                    failed.append(start)
            self.check('known_state_rejection_and_execution_clock:' + cid, not failed and (used <= 1).all()
                       and np.array_equal(np.r_[0, target[:-1]], base_exec), mismatched_signals=failed[:10])
            executed = np.r_[0, filtered_target[:-1]]; held = np.r_[0, executed[:-1]]; gross = held * spot
            trades = ledger_from_executed(dates, executed, gross)
            if trades.empty:
                trades = base_ledger.iloc[:0].copy()
            # Removing complete original episodes cannot alter the returns or
            # entry/exit indices of any surviving true episode.
            surviving = original.loc[codes(original[feature], BINS[feature]) != selected_state]
            self.compare_frames('complete_original_episode_exclusion:' + cid, surviving, trades, ['entry_idx'])
            for split, start, end in periods:
                for variant in ('BASE', 'FILTER'):
                    values = dict(windows[key][split]) if variant == 'BASE' else window_from_path(dates, executed, held, gross, trades, start, end)
                    n = windows[key][split]['closed_n']; values.update(base_closed_n=n, retention=values['closed_n'] / n if n else np.nan)
                    row = metric_index.loc[cid, variant, split]
                    differences = [name for name, value in values.items() if
                                   (pd.Timestamp(row[name]) != value if name.endswith('_date') else not same_number(row[name], value))]
                    self.check('independent_window:' + cid + ':' + variant + ':' + split, not differences, failed_fields=differences)
                    recomputed.append({**{n: item[n] for n in ('candidate_id', 'pair', 'strategy', 'filter_id')},
                                       'is_new': True, 'variant': variant, 'split': split, **values})
            if cid in frozen_ids:
                mask = (dates >= pd.Timestamp(protocol['is_start'])) & (dates <= pd.Timestamp(protocol['is_end']))
                for variant, e, h, g, table in [('BASE', base_exec, base_held, base_gross, base_ledger), ('FILTER', executed, held, gross, trades)]:
                    frozen_daily.append(pd.DataFrame({'candidate_id': cid, 'variant': variant, 'split': 'IS', 'date': dates[mask],
                        'executed': e[mask], 'held': h[mask], 'gross_daily_return': g[mask], 'net_daily_return': g[mask]}))
                    frozen_ledgers.append(table.assign(candidate_id=cid, variant=variant))
            if (offset + 1) % 100 == 0:
                print('Original state proposals independently audited:', offset + 1, '/', len(items), flush=True)
        rebuilt_metrics = pd.DataFrame(recomputed)
        self.compare_frames('independent_path_original_SCREEN_rank_freeze', self.read_csv('is_selection.csv'), screen_candidates(rebuilt_metrics), ['candidate_id'])
        daily, frozen_metrics = self.read_csv('is_frozen_daily.csv.gz'), self.read_csv('is_frozen_metrics.csv')
        if frozen_ids:
            self.compare_frames('frozen_daily_independent_paths', daily, pd.concat(frozen_daily, ignore_index=True), ['candidate_id', 'variant', 'date'])
            self.compare_frames('frozen_metrics_independent_paths', frozen_metrics,
                                rebuilt_metrics.loc[rebuilt_metrics.candidate_id.isin(frozen_ids) & rebuilt_metrics.split.eq('IS')], ['candidate_id', 'variant', 'split'])
            self.compare_frames('frozen_ledger_independent_paths', self.read_csv('is_frozen_trades.csv.gz'), pd.concat(frozen_ledgers, ignore_index=True), ['candidate_id', 'variant', 'entry_idx'])
            self.compare_frames('frozen_decisions_exact_subset', self.read_csv('is_frozen_decisions.csv.gz'), decisions.loc[decisions.candidate_id.isin(frozen_ids)], ['candidate_id', 'signal_idx'])
        else:
            self.check('no_fabricated_frozen_paths', daily.empty and frozen_metrics.empty
                       and not (self.campaign / 'is_frozen_trades.csv.gz').exists() and not (self.campaign / 'is_frozen_decisions.csv.gz').exists())
        self.details.update(BASE_accounts_independently_reconstructed=180, independent_metric_windows=len(rebuilt_metrics),
                            original_episode_decisions_checked=total_decisions, rejected_original_episode_decisions=reject_count,
                            UNKNOWN_episode_decisions_retained=unknown_count,
                            pending_terminal_original_signal_limit={'rows': pending_rows,
                                'scope': 'Final unexecuted BASE entry/exit/reversal signals cannot be recovered from an executed ledger. Saved clocks/state truth are checked; terminal technical and shadow-BASE-PnL values are independently recomputed. Pending macro values cannot be independently compared to a filled-entry cache and have no observed entry PnL.'},
                            independent_feature_scope='Technical indicators, directional transforms and already-earned shadow BASE PnL recomputed; macro raw series/provenance come from the immutable audited IS cache. No fresh macro extraction or original BASE signal generation is claimed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/state_original_screen_v7')
    args = parser.parse_args(); audit = StateOriginalAudit(args.output.resolve())
    started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_entry_interaction_research.py', ROOT / 'verify_state_filter_research.py', ROOT / 'verify_state_filter_supplement.py'):
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
    destination = audit.campaign / 'independent_validation.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(destination), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
