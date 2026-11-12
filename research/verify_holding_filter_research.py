"""Independent IS-only audit of holding_filter_v3 or timecap_filter_v4.

Proposal reconstruction parses D1/D2/D outcome rows only. All saved decisions
are checked against independently computed checkpoint facts, then each path is
rebuilt by truncating original BASE episodes. No strategy or candidate engine
is called, and workbook price rows after 2023-05-29 are never converted to
prices or returns. Existing campaign and observation files remain unchanged.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_state_filter_research import Audit, ROOT, sha, same_number
from verify_state_filter_supplement import ledger_from_executed, window_from_path
from fx_baselines import CURRENCIES, STRATEGIES, _workbook_rows
from fx_filter_evaluation import SCREEN, screen_candidates

TOL = 1e-10
AGES = (5, 10, 20, 40, 63)
NAMES = ('pnl_negative', 'pnl_positive', 'giveback_half', 'recent5_negative', 'opposite20')
CONDITIONS = list(NAMES) + ['&'.join(p) for p in itertools.combinations(NAMES, 2)
                          if p != ('pnl_negative', 'pnl_positive')]
DISCOVERY = {'D1': ('2005-01-03', '2009-12-31'), 'D2': ('2010-01-01', '2014-12-31'),
             'D': ('2005-01-03', '2014-12-31')}


def missing_equal(a, b):
    return bool(pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b)


def predicates(pnl, peak, recent, direction):
    return {'pnl_negative': pnl < -TOL, 'pnl_positive': pnl > TOL,
            'giveback_half': peak > TOL and pnl <= peak / 2 + TOL,
            'recent5_negative': np.nan if not np.isfinite(recent) else recent < -TOL,
            'opposite20': np.nan if not np.isfinite(direction) else direction < -TOL}


def state(values, condition):
    if condition == 'ALL':
        return 'TRUE'
    selected = [values[n] for n in condition.split('&')]
    return 'UNKNOWN' if any(pd.isna(x) for x in selected) else 'TRUE' if all(selected) else 'FALSE'


class HoldingAudit(Audit):
    def json_path(self, path):
        return json.loads(self.register(path).read_text(encoding='utf-8'))

    def frame_path(self, path):
        return pd.read_csv(self.register(path), float_precision='round_trip')

    def run(self):
        protocol = self.read_json('protocol.json')
        if protocol['version'] not in ('holding_filter_v3', 'timecap_filter_v4'):
            raise ValueError('Unrecognized frozen campaign version.')
        timecap = protocol['version'] == 'timecap_filter_v4'
        status = self.read_json('development_status.json')
        freeze = self.read_json('frozen_selection.json')
        hypotheses = self.read_json('is_hypotheses.json')
        obs = ROOT / 'reproduced/holding_observations_v3'
        op = self.json_path(obs / 'holding_observation_protocol.json')
        ov = self.json_path(obs / 'holding_observation_validation.json')
        self.check('completed_IS_only_status', status['status'] == 'passed' and status['compute_cutoff'] == '2023-05-29'
                   and status['oos_evaluated'] is False and status['holdout_returns_computed'] is False)
        self.check('protocol_clock_scope_cost', protocol['is_start'] == '2005-01-03' and protocol['is_end'] == '2023-05-29'
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0 and protocol['screen'] == SCREEN
                   and set(protocol['scope']['pairs']) == set(CURRENCIES)
                   and set(protocol['scope']['strategies']) == set(STRATEGIES) and protocol['scope']['baseline_accounts'] == 180)
        self.check('freeze_protocol_status_hash_chain', freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json'))
        self.check('proposal_before_replay_before_freeze', pd.Timestamp(protocol['created_utc']) < pd.Timestamp(hypotheses['created_utc'])
                   <= pd.Timestamp(status['started_utc']) < pd.Timestamp(freeze['created_utc']) <= pd.Timestamp(status['finished_utc']))
        self.check('observation_before_research', pd.Timestamp(op['created_utc']) < pd.Timestamp(ov['finished_utc'])
                   < pd.Timestamp(protocol['created_utc']))
        for manifest in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest_agreement:' + manifest, freeze[manifest] == protocol[manifest])
            self.hashes(manifest, protocol[manifest])
            self.hashes('observation_' + manifest, op[manifest])
        self.hashes('observation_cache_chain', protocol['observation_source_sha256'])
        self.hashes('observation_outputs', ov['output_sha256'], obs)
        self.hashes('IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.check('all_IS_outputs_frozen', {p.name for p in self.campaign.glob('is_*') if p.is_file()}
                   == set(freeze['is_output_sha256']))
        self.check('observation_validated_scope', ov['status'] == 'passed' and ov['accounts'] == op['accounts'] == 180
                   and ov['compute_cutoff'] == op['compute_end'] == protocol['is_end']
                   and ov['oos_computed'] is op['oos_computed'] is False
                   and ov['protocol_sha256'] == sha(obs / 'holding_observation_protocol.json'))

        metrics = self.read_csv('is_all_metrics.csv')
        selection = self.read_csv('is_selection.csv')
        decisions = self.read_csv('is_all_decisions.csv.gz')
        coverage = self.read_csv('is_account_coverage.csv')
        ledger = self.frame_path(obs / 'is_base_trade_ledger.csv.gz')
        checkpoints = self.frame_path(obs / 'is_holding_checkpoints.csv.gz')
        for frame, columns in [(ledger, ('signal_date', 'entry_date', 'exit_date')),
                               (checkpoints, ('entry_date', 'original_exit_date', 'checkpoint_signal_date'))]:
            for name in columns:
                frame[name] = pd.to_datetime(frame[name])
        self.check('observation_unique_episodes_and_ages', not ledger.duplicated(['pair', 'strategy', 'trade_id']).any()
                   and not checkpoints.duplicated(['pair', 'strategy', 'trade_id', 'checkpoint_age']).any())
        expected_accounts = set(itertools.product(CURRENCIES, STRATEGIES))
        self.check('all_180_BASE_accounts_observed', set(zip(ledger.pair, ledger.strategy)) == expected_accounts)
        self.check('all_180_accounts_including_zero_proposals', len(coverage) == 180
                   and set(zip(coverage.pair, coverage.strategy)) == expected_accounts
                   and coverage.compute_end.eq(protocol['is_end']).all())
        self.check('IS_dates_never_exceed_cutoff', ledger.exit_date.le(protocol['is_end']).all()
                   and checkpoints.checkpoint_signal_date.le(protocol['is_end']).all()
                   and pd.to_datetime(metrics.end_date).le(protocol['is_end']).all())

        # Only D rows are numerically decoded, independently of runner.proposals.
        discovered, proposal_keys = {}, set()
        summary_path = self.register(obs / 'holding_state_summary.csv')
        with summary_path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['block'] not in DISCOVERY:
                    continue
                key = (row['pair'], row['strategy'], int(row['checkpoint_age']), row['condition'])
                evidence = {name: float(row[name]) for name in ('closed_n', 'remaining_return_sum_pct',
                            'remaining_without_worst1_pct', 'remaining_without_worst3_pct')}
                if row['block'] in discovered.setdefault(key, {}):
                    raise AssertionError('Duplicate discovery summary key.')
                discovered[key][row['block']] = evidence
        for key, blocks in discovered.items():
            if (key[3] == 'ALL') != timecap:
                continue
            if all(blocks[b]['closed_n'] >= 10 and blocks[b]['remaining_return_sum_pct'] < -TOL for b in ('D1', 'D2')) \
                    and blocks['D']['remaining_return_sum_pct'] < -TOL:
                proposal_keys.add(key)
        items = hypotheses['candidates']
        actual_keys = {(h['pair'], h['strategy'], h['spec']['checkpoint_age'],
                        '&'.join(h['spec']['condition']) if isinstance(h['spec'].get('condition'), list) else h['spec'].get('condition', 'ALL'))
                       for h in items}
        self.check('D_only_proposal_exact_set', actual_keys == proposal_keys and len(items) == len(proposal_keys),
                   expected=len(proposal_keys), actual=len(items))
        proposal_errors = []
        for h in items:
            parts = h['spec']['condition'] if isinstance(h['spec'].get('condition'), list) else [h['spec'].get('condition', 'ALL')]
            condition, age = '&'.join(parts), h['spec']['checkpoint_age']
            identifier = f'TIME_CAP_{age}' if timecap else f"CHECK_{age}_{condition.replace('&', '_AND_')}"
            key = (h['pair'], h['strategy'], age, condition)
            if not (condition in (['ALL'] if timecap else CONDITIONS) and age in AGES and h['filter_id'] == identifier
                    and set(h['spec']) == ({'id', 'checkpoint_age'} if timecap else {'id', 'checkpoint_age', 'condition'})
                    and h['candidate_id'] == '|'.join((h['pair'], h['strategy'], identifier))
                    and h['is_new'] is True and h['discovery_evidence'] == discovered[key]):
                proposal_errors.append(h['candidate_id'])
        self.check('proposal_specs_and_discovery_evidence_exact', not proposal_errors, mismatches=proposal_errors)
        counts = pd.Series([(h['pair'], h['strategy']) for h in items]).value_counts()
        self.check('all_account_candidate_counts', all(r.candidate_count == counts.get((r.pair, r.strategy), 0)
                   for r in coverage.itertuples(index=False)) and coverage.candidate_count.sum() == len(items))
        candidate_ids = {h['candidate_id'] for h in items}
        self.check('candidate_sets_complete', set(metrics.candidate_id) == set(decisions.candidate_id)
                   == set(selection.candidate_id) == candidate_ids)
        periods = [('IS', protocol['is_start'], protocol['is_end']), *protocol['is_blocks']]
        self.check('all_candidate_metric_grid', not metrics.duplicated(['candidate_id', 'variant', 'split']).any()
                   and set(zip(metrics.candidate_id, metrics.variant, metrics.split))
                   == set(itertools.product(candidate_ids, ('BASE', 'FILTER'), [p[0] for p in periods])))
        self.check('zero_cost_metrics', metrics.cost_drag_pp.eq(0).all()
                   and np.array_equal(metrics.gross_simple_return_pct, metrics.net_simple_return_pct)
                   and np.array_equal(metrics.gross_max_drawdown_pp, metrics.net_max_drawdown_pp))
        rescreened = screen_candidates(metrics)
        numeric = [c for c in rescreened if pd.api.types.is_numeric_dtype(rescreened[c])]
        self.values('original_SCREEN_rescreen', selection, rescreened, [c for c in rescreened if c != 'candidate_id'], numeric)
        self.check('zero_pass_zero_freeze_no_fallback', not selection['pass'].any() and not selection.frozen.any()
                   and not freeze['candidates'] and freeze['frozen_candidates'] == status['frozen_candidates'] == 0
                   and freeze['all_passing_versions'] == status['all_passing_versions'] == 0
                   and freeze['all_candidate_accounts_tested'] == status['candidates_evaluated'] == len(items))
        self.check('freeze_was_before_any_OOS', freeze['oos_evaluated_at_freeze'] is False)

        # Workbook row decoding is immediately clipped before converting prices.
        workbook = ROOT / next(n for n in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in n)
        quotes = {pair: [] for pair in CURRENCIES}
        with zipfile.ZipFile(workbook) as archive:
            rows = _workbook_rows(archive, 'xl/worksheets/sheet6.xml', {'B', 'D', 'E'})
            next(rows)
            for row in rows:
                if row.get('D') not in quotes or row.get('B') is None:
                    continue
                date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(row['B']))
                if date > pd.Timestamp(protocol['is_end']):
                    continue
                if row.get('E') is not None:
                    quotes[row['D']].append((date, float(row['E'])))
        paths, checkpoint_lookup, base_windows = {}, {}, {}
        checkpoint_count = residual_count = discovery_rows = 0
        terminal_target_unobservable = []
        for pair in CURRENCIES:
            q = sorted(quotes[pair])
            dates = pd.DatetimeIndex([r[0] for r in q]); prices = np.array([r[1] for r in q])
            self.check('IS_quote_calendar:' + pair, dates.is_unique and dates.is_monotonic_increasing
                       and dates[-1] <= pd.Timestamp(protocol['is_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1]
            direction20 = pd.Series(prices).pct_change(20).to_numpy() * 100
            for strategy in STRATEGIES:
                key = pair, strategy
                saved = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)].sort_values('entry_idx')
                executed = np.zeros(len(dates), dtype=int); occupancy = executed.copy()
                for trade in saved.itertuples(index=False):
                    stop = int(trade.exit_idx) if trade.closed else int(trade.exit_idx) + 1
                    executed[int(trade.entry_idx):stop] = int(trade.position)
                    occupancy[int(trade.entry_idx):stop] += 1
                held = np.r_[0, executed[:-1]]; gross = held * spot
                rebuilt = ledger_from_executed(dates, executed, gross)
                mismatches = []
                if len(saved) != len(rebuilt):
                    mismatches.append('episode_count')
                else:
                    for column in rebuilt:
                        a, b = saved[column].to_numpy(), rebuilt[column].to_numpy()
                        okay = np.array_equal(a, b) if column.endswith('_date') else np.allclose(a, b, atol=1e-8, rtol=1e-10, equal_nan=True)
                        if not okay:
                            mismatches.append(column)
                self.check('BASE_ledger_and_daily_arithmetic:' + '|'.join(key), not mismatches and (occupancy <= 1).all()
                           and saved.transaction_cost_usd.eq(0).all(), failed_fields=mismatches)
                paths[key] = dates, spot, executed, held, gross, rebuilt, saved
                base_windows[key] = {phase: window_from_path(dates, executed, held, gross, rebuilt, start, end)
                                     for phase, start, end in periods}
                local = checkpoints.loc[checkpoints.pair.eq(pair) & checkpoints.strategy.eq(strategy)]
                errors = []; trades = {int(t.trade_id): t for t in saved.itertuples(index=False)}
                for row in local.itertuples(index=False):
                    trade = trades[row.trade_id]; m = int(trade.entry_idx + row.checkpoint_age)
                    values = gross[int(trade.entry_idx) + 1:m + 1] * 100
                    pnl = float(values.sum()); peak = max(0., float(np.cumsum(values).max()))
                    recent = float(gross[m - 4:m + 1].sum() * 100)
                    directional = trade.position * direction20[m]
                    computed = {'known_cumulative_simple_pnl_pct': pnl, 'known_running_peak_pct': peak,
                                'known_drawdown_from_peak_pp': peak - pnl,
                                'known_giveback_fraction': (peak - pnl) / peak if peak > TOL else np.nan,
                                'known_recent5_signed_return_pct': recent, 'current_direction_return20_pct': directional}
                    good = m == row.checkpoint_idx and dates[m] == row.checkpoint_signal_date
                    active = bool(not trade.closed or m < trade.exit_idx)
                    next_observed = m + 1 < len(dates)
                    if next_observed:
                        technical = bool(active and executed[m + 1] != trade.position)
                    else:
                        # Executed holdings contain no observation of the last
                        # target. This pending-only distinction has no fill or
                        # realized residual, so disclose rather than fabricate
                        # an independent verification of that final target.
                        technical = bool(row.technical_exit_already_signaled)
                        terminal_target_unobservable.append([pair, strategy, int(trade.trade_id), int(row.checkpoint_age)])
                    actionable = bool(active and not technical and next_observed)
                    pending_risk = bool(active and not technical and not next_observed)
                    observed_closed = bool(trade.eligible_closed and actionable and m + 2 <= trade.exit_idx)
                    qualification = {'original_active_at_checkpoint': active,
                                     'technical_exit_already_signaled': technical,
                                     'next_close_observed': next_observed, 'actionable_checkpoint': actionable,
                                     'risk_exit_pending': pending_risk, 'residual_observed_closed': observed_closed,
                                     'original_closed': bool(trade.closed), 'original_eligible_closed': bool(trade.eligible_closed)}
                    good &= all(getattr(row, name) == value for name, value in qualification.items())
                    good &= (row.entry_idx == trade.entry_idx and row.entry_signal_idx == trade.signal_idx
                             and row.original_exit_idx == trade.exit_idx and row.position == trade.position
                             and row.entry_date == trade.entry_date and row.original_exit_date == trade.exit_date
                             and missing_equal(row.hypothetical_exit_idx, m + 1 if next_observed else np.nan))
                    good &= all(same_number(getattr(row, n), v) for n, v in computed.items())
                    flags = predicates(pnl, peak, recent, directional)
                    good &= all(missing_equal(getattr(row, n), v) for n, v in flags.items())
                    if trade.eligible_closed and active and next_observed:
                        residual_count += 1
                        good &= same_number(row.label_remaining_simple_return_pct,
                                            gross[m + 2:int(trade.exit_idx) + 1].sum() * 100)
                        good &= same_number(row.label_next_close_return_pct, gross[m + 1] * 100)
                        good &= same_number(row.label_remaining_observations, max(0, int(trade.exit_idx) - m - 1))
                    else:
                        good &= pd.isna(row.label_remaining_simple_return_pct) and pd.isna(row.label_next_close_return_pct)
                    if not good:
                        errors.append([int(row.trade_id), int(row.checkpoint_age)])
                    checkpoint_lookup[pair, strategy, int(row.trade_id), int(row.checkpoint_age)] = row
                    checkpoint_count += 1
                self.check('all_checkpoint_features_predicates_residuals:' + '|'.join(key), not errors, mismatches=errors[:10], rows=len(local))
                # Rebuild all D summaries, including zero and ALL rows, from episodes.
                for age in AGES:
                    age_rows = local.loc[local.checkpoint_age.eq(age)]
                    for condition in ['ALL', *CONDITIONS]:
                        use = np.ones(len(age_rows), dtype=bool)
                        if condition != 'ALL':
                            for part in condition.split('&'):
                                use &= age_rows[part].eq(True).to_numpy()
                        for block, (start, end) in DISCOVERY.items():
                            chosen = age_rows.loc[use & age_rows.residual_observed_closed
                                & age_rows.entry_date.ge(start) & age_rows.original_exit_date.le(end)]
                            values = np.sort(chosen.label_remaining_simple_return_pct.to_numpy())
                            negatives = values[values < -TOL]
                            expected = {'closed_n': len(values), 'remaining_return_sum_pct': values.sum(),
                                        'remaining_without_worst1_pct': values.sum() - negatives[:1].sum(),
                                        'remaining_without_worst3_pct': values.sum() - negatives[:3].sum()}
                            original = discovered[pair, strategy, age, condition][block]
                            if not all(same_number(original[n], v) for n, v in expected.items()):
                                errors.append(['summary', age, condition, block])
                            discovery_rows += 1
                self.check('D_summary_independent_episode_slices:' + '|'.join(key), not errors, mismatches=errors[:10])
            print('Independently reconstructed BASE/checkpoints:', pair, flush=True)

        metric_index = metrics.set_index(['candidate_id', 'variant', 'split'])
        decision_groups = {cid: group for cid, group in decisions.groupby('candidate_id', sort=False)}
        total_decisions = total_triggers = window_count = 0
        for index, item in enumerate(items):
            cid, pair, strategy = item['candidate_id'], item['pair'], item['strategy']
            age = item['spec']['checkpoint_age']; condition = item['spec'].get('condition', 'ALL')
            condition = '&'.join(condition) if isinstance(condition, list) else condition
            dates, spot, base_exec, base_held, base_gross, base_ledger, saved = paths[pair, strategy]
            table = decision_groups[cid]; errors = []; expected_exec = np.zeros(len(dates), dtype=int)
            normal = table.loc[~table.entry_pending].set_index('entry_idx', drop=False)
            if normal.index.duplicated().any() or set(normal.index) != set(saved.entry_idx):
                errors.append('BASE_episode_coverage')
            if not table.accepted.all() or not table.checkpoint_age.eq(age).all() or not table.condition.eq(condition).all():
                errors.append('spec_or_rejected_entry')
            pending = table.loc[table.entry_pending]
            if not (len(pending) <= 1 and pending.signal_idx.eq(len(dates) - 1).all()
                    and pending.entry_idx.isna().all() and ~pending.early_exit.any() and ~pending.evaluated.any()):
                errors.append('terminal_pending_entry')
            for trade in saved.itertuples(index=False):
                d = normal.loc[trade.entry_idx]
                ck = checkpoint_lookup.get((pair, strategy, int(trade.trade_id), age))
                snapshot = ck is not None and bool(ck.original_active_at_checkpoint)
                evaluated = snapshot and not bool(ck.technical_exit_already_signaled)
                expected_state = state({n: getattr(ck, n) for n in NAMES}, condition) if evaluated else 'NOT_EVALUATED'
                trigger = evaluated and expected_state == 'TRUE'
                m = int(trade.entry_idx) + age
                risk = m if trigger else np.nan
                actual_exit = (m + 1 if m + 1 < len(dates) else np.nan) if trigger else (trade.exit_idx if trade.closed else np.nan)
                good = (d.signal_idx == trade.signal_idx and d.position == trade.position and d.evaluated == evaluated
                        and d.condition_state == expected_state and d.early_exit == trigger
                        and missing_equal(d.checkpoint_idx, m if snapshot else np.nan)
                        and missing_equal(d.risk_signal_idx, risk) and missing_equal(d.actual_exit_idx, actual_exit)
                        and missing_equal(d.original_exit_idx, trade.exit_idx if trade.closed else np.nan)
                        and d.risk_pending == (trigger and m + 1 == len(dates)))
                for col, source in [('known_pnl_pct', 'known_cumulative_simple_pnl_pct'),
                                    ('running_peak_pct', 'known_running_peak_pct'), ('giveback_fraction', 'known_giveback_fraction'),
                                    ('signed_recent5_pct', 'known_recent5_signed_return_pct'),
                                    ('direction_return_20_pct', 'current_direction_return20_pct')]:
                    good &= same_number(d[col], getattr(ck, source) if snapshot else np.nan)
                for prefix in ('signal', 'entry', 'checkpoint', 'risk_signal', 'original_end_signal', 'original_exit', 'actual_exit'):
                    idx = d[prefix + '_idx']; date = pd.to_datetime(d[prefix + '_date'])
                    good &= pd.isna(date) if pd.isna(idx) else 0 <= idx < len(dates) and date == dates[int(idx)]
                if not good:
                    errors.append(int(trade.trade_id))
                stop = int(actual_exit) if np.isfinite(actual_exit) else len(dates)
                expected_exec[int(trade.entry_idx):stop] = int(trade.position)
                total_triggers += int(trigger); total_decisions += 1
            self.check('candidate_decisions_match_known_checkpoint:' + cid, not errors, mismatches=errors[:10], episodes=len(table))
            held = np.r_[0, expected_exec[:-1]]; gross = held * spot
            filtered_ledger = ledger_from_executed(dates, expected_exec, gross)
            effect = 0.
            for trade in saved.itertuples(index=False):
                d = normal.loc[trade.entry_idx]
                if d.early_exit and pd.notna(d.actual_exit_idx):
                    effect -= base_gross[int(d.actual_exit_idx) + 1:int(trade.exit_idx) + 1].sum() * 100
            self.check('candidate_direct_BASE_tail_effect:' + cid, same_number((gross - base_gross).sum() * 100, effect))
            for phase, start, end in periods:
                for variant in ('BASE', 'FILTER'):
                    calculated = base_windows[pair, strategy][phase] if variant == 'BASE' else \
                        window_from_path(dates, expected_exec, held, gross, filtered_ledger, start, end)
                    calculated['base_closed_n'] = base_windows[pair, strategy][phase]['closed_n']
                    calculated['retention'] = calculated['closed_n'] / calculated['base_closed_n'] if calculated['base_closed_n'] else np.nan
                    row = metric_index.loc[cid, variant, phase]
                    mismatches = [name for name, value in calculated.items() if
                        (pd.Timestamp(row[name]) != value if name.endswith('_date') else not same_number(row[name], value))]
                    self.check('independent_window:' + cid + ':' + variant + ':' + phase,
                               not mismatches, failed_fields=mismatches)
                    window_count += 1
            if (index + 1) % 100 == 0:
                print('All decisions and windows audited:', index + 1, '/', len(items), flush=True)

        daily = self.read_csv('is_frozen_daily.csv.gz'); frozen_metrics = self.read_csv('is_frozen_metrics.csv')
        required = ['candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'variant', 'split',
                    'date', 'gross_daily_return', 'net_daily_return', 'executed', 'held']
        self.check('empty_frozen_daily_preserves_schema', daily.empty and list(daily) == required)
        self.check('empty_frozen_metrics_preserves_schema', frozen_metrics.empty and list(frozen_metrics) == list(metrics))
        forbidden = ['is_frozen_trades.csv.gz', 'is_frozen_decisions.csv.gz', 'oos_status.json', 'oos_daily.csv.gz',
                     'oos_metrics.csv', 'oos_anniversary_metrics.csv', 'oos_fresh_start_metrics.csv',
                     'candidate_assessment.csv', 'frozen_full_trade_ledger.csv.gz', 'frozen_full_episode_decisions.csv.gz']
        self.check('no_fabricated_frozen_or_OOS_data', all(not (self.campaign / name).exists() for name in forbidden))
        self.check('all_registered_evidence_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(campaign_version=protocol['version'], accounts=180, candidate_accounts=int((coverage.candidate_count > 0).sum()),
                            candidates=len(items), original_SCREEN_passes=int(selection['pass'].sum()), frozen_candidates=0,
                            checkpoint_rows_independently_recomputed=checkpoint_count,
                            residual_rows_independently_recomputed=residual_count,
                            D_summary_rows_independently_recomputed=discovery_rows,
                            pending_terminal_target_limit={'count': len(terminal_target_unobservable),
                                'episodes': terminal_target_unobservable,
                                'scope': 'Only the final unexecuted BASE target cannot be independently recovered from an execution ledger. Its technical-priority flag is checked for agreement with saved decisions, not independently reconstructed. These checkpoints have no next-close fill, residual label, or realized PnL effect.'},
                            executed_episode_decisions_checked=total_decisions, early_exit_decisions=total_triggers,
                            independent_metric_windows=window_count, strategy_engine_replays=0,
                            OOS_results_opened=False, holdout_returns_computed=False,
                            proposal_outcome_blocks_decoded=list(DISCOVERY),
                            raw_prices_computed_through=protocol['is_end'],
                            frozen_prefix_check='Not applicable: zero frozen candidates; no nonempty frozen daily path is present.',
                            tests={'rerun': False, 'parent_reported_tests': 87, 'parent_reported_passed': 87,
                                   'parent_reported_seconds': 10.033, 'engine_prefix_subchecks': 693,
                                   'engine_observation_synthetic_crosschecks': 75,
                                   'note': 'Prior parent test run; this audit does not claim to have rerun the suite.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/holding_filter_v3')
    args = parser.parse_args()
    audit = HoldingAudit(args.output.resolve()); started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_state_filter_research.py', ROOT / 'verify_state_filter_supplement.py'):
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
    output = audit.campaign / 'independent_validation.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(output), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
