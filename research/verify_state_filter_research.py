"""Read-only audit of the completed v2 discovery/validation/confirmation campaign.

Only the audit JSON is written. No strategy is rerun, selected, or altered; no
OOS result is opened. Phase gates, trade attribution, and window arithmetic are
recomputed independently from saved evidence. The original full-IS screen is
also reapplied as an explicit compatibility check.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
from fx_baselines import CURRENCIES, STRATEGIES, _workbook_rows
from fx_filter_evaluation import SCREEN, screen_candidates

TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_string(value):
    return '' if pd.isna(value) else str(value)


def same_number(a, b, atol=1e-8):
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=1e-10, equal_nan=True))


def ratio_pass(base, filtered):
    return bool(filtered.gross_simple_return_pct * -base.gross_max_drawdown_pp >
                base.gross_simple_return_pct * -filtered.gross_max_drawdown_pp + TOL)


def independent_phase(base, filtered, changed, discovery=False):
    finite = np.isfinite([base.win_rate, filtered.win_rate, base.payoff_ratio, filtered.payoff_ratio]).all()
    delta = filtered.gross_simple_return_pct - base.gross_simple_return_pct
    return {
        'annualized': bool(filtered.gross_annualized_simple_pct >= 1 - TOL),
        'return_improvement': bool(delta >= 5 - TOL if discovery else delta > TOL),
        'return_drawdown': ratio_pass(base, filtered),
        'quality': bool(finite and filtered.win_rate >= base.win_rate + (.02 if discovery else 0) - TOL
                        and filtered.payoff_ratio >= base.payoff_ratio * (1.1 if discovery else 1) - TOL),
        'drawdown': bool(filtered.gross_max_drawdown_pp >= base.gross_max_drawdown_pp - 2 - TOL),
        'sample': bool(filtered.closed_n >= (30 if discovery else 20) and filtered.closed_n >= base.closed_n * .5 - TOL),
        'outcome_sample': bool(discovery or (filtered.win_n >= 5 and filtered.loss_n >= 5)),
        'affected_sample': bool(changed >= (20 if discovery else 5)),
    }


def independent_quality(values):
    values = np.asarray(values, dtype=float)
    wins, losses = values[values > TOL], values[values < -TOL]
    return {
        'closed_n': len(values), 'win_n': len(wins), 'loss_n': len(losses),
        'flat_n': len(values) - len(wins) - len(losses),
        'win_rate': len(wins) / len(values) if len(values) else np.nan,
        'payoff_ratio': wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan,
        'mean_win_pct': wins.mean() if len(wins) else np.nan,
        'mean_loss_abs_pct': -losses.mean() if len(losses) else np.nan,
        'expectancy_pct': values.mean() if len(values) else np.nan,
    }


def independent_effect(base_daily, filter_daily, base_ledger, start, end):
    """Direct date slicing per original episode; no production prefix-sum helper."""
    dates = pd.DatetimeIndex(base_daily.date)
    delta = (filter_daily.gross_daily_return.to_numpy() - base_daily.gross_daily_return.to_numpy()) * 100
    changed = filter_daily.held.to_numpy() != base_daily.held.to_numpy()
    window = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    contributions, changed_closed, changed_all = [], 0, 0
    for trade in base_ledger.itertuples(index=False):
        interval = window & (dates > trade.entry_date) & (dates <= trade.exit_date)
        contributions.append(float(delta[interval].sum()))
        used = bool(changed[interval].any())
        changed_all += int(used)
        changed_closed += int(used and trade.eligible_closed and trade.entry_date >= pd.Timestamp(start)
                              and trade.exit_date <= pd.Timestamp(end))
    total = float(delta[window].sum())
    if not same_number(sum(contributions), total):
        raise AssertionError('Independent date-slice episode effects do not sum to the daily delta.')
    best = sorted((value for value in contributions if value > 0), reverse=True)[:3]
    return {'delta_pp': total, 'changed_closed_n': changed_closed, 'changed_episode_n': changed_all,
            'delta_without_best3_pp': total - sum(best)}


class Audit:
    def __init__(self, campaign):
        self.campaign = campaign
        self.checks = []
        self.files = {}
        self.details = {}

    def check(self, name, passed, **detail):
        self.checks.append({'name': name, 'passed': bool(passed), **detail})
        if not passed:
            print('FAILED:', name, detail, flush=True)

    def register(self, path):
        path = Path(path).resolve()
        self.files[str(path)] = sha(path)
        return path

    def read_json(self, name):
        return json.loads(self.register(self.campaign / name).read_text(encoding='utf-8'))

    def read_csv(self, name):
        return pd.read_csv(self.register(self.campaign / name), float_precision='round_trip')

    def hashes(self, label, values, directory=ROOT):
        for name, expected in values.items():
            path = self.register(directory / name)
            self.check(label + ':' + name, sha(path) == expected)

    def values(self, label, actual, expected, columns, numeric=()):
        actual = actual.set_index('candidate_id').sort_index()
        expected = expected.set_index('candidate_id').sort_index()
        self.check(label + ':identities', actual.index.equals(expected.index))
        if not actual.index.equals(expected.index):
            return
        for column in columns:
            if column in numeric:
                good = np.isclose(actual[column].astype(float), expected[column].astype(float),
                                  atol=1e-8, rtol=1e-10, equal_nan=True)
            else:
                good = actual[column].map(clean_string).eq(expected[column].map(clean_string)).to_numpy()
            self.check(label + ':' + column, good.all(), mismatches=list(actual.index[~good][:5]))

    def run(self):
        protocol = self.read_json('protocol.json')
        lock = self.read_json('validation_locked.json')
        fit_status = self.read_json('fit_status.json')
        preaudit = self.read_json('preconfirmation_integrity_audit.json')
        freeze = self.read_json('frozen_selection.json')
        status = self.read_json('development_status.json')
        proposals = self.read_json('discovery_proposals.json')['candidates']
        effects = self.read_json('confirmation_effect_evidence.json')
        self.hashes('source', protocol['source_sha256'])
        self.hashes('input', protocol['input_sha256'])
        self.hashes('observation', protocol['observation_source_sha256'])
        self.hashes('locked_fit_output', lock['fit_output_sha256'], self.campaign)
        self.hashes('frozen_is_output', freeze['is_output_sha256'], self.campaign)
        for name, record in [('lock', lock), ('freeze', freeze)]:
            self.check(name + ':protocol_hash', record['protocol_sha256'] == sha(self.campaign / 'protocol.json'))
            self.check(name + ':source_manifest', record['source_sha256'] == protocol['source_sha256'])
            self.check(name + ':input_manifest', record['input_sha256'] == protocol['input_sha256'])
        for name, record, field in [('fit_status', fit_status, 'lock_sha256'), ('preaudit', preaudit, 'lock_sha256'),
                                    ('freeze', freeze, 'preconfirmation_lock_sha256')]:
            self.check(name + ':lock_hash', record[field] == sha(self.campaign / 'validation_locked.json'))
        self.check('development:freeze_hash', status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json'))
        self.check('chronological_lock_audit_freeze', pd.Timestamp(lock['created_utc']) < pd.Timestamp(preaudit['created_utc']) < pd.Timestamp(freeze['created_utc']))
        self.check('protocol_precedes_lock', pd.Timestamp(protocol['created_utc']) < pd.Timestamp(lock['created_utc']))
        self.check('statuses_passed', all(item['status'] == 'passed' for item in (fit_status, preaudit, status)))
        self.check('preaudit_checks_passed', all(preaudit['checks'].values()))
        self.check('unopened_at_stage_boundaries', not lock['confirmation_computed'] and not lock['oos_computed']
                   and not freeze['oos_evaluated_at_freeze'] and not status['oos_evaluated'])
        self.check('exact_scope', protocol['scope']['pairs'] == list(CURRENCIES) and protocol['scope']['strategies'] == STRATEGIES
                   and protocol['scope']['baseline_accounts'] == 180)
        self.check('zero_cost_simple_protocol', protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
                   and protocol['accounting'] == 'fixed_initial_capital_simple')
        self.check('original_SCREEN_unchanged', protocol['screen'] == SCREEN)
        self.check('chronological_cutoffs', protocol['discovery'][1:] == ['2005-01-03', '2014-12-31']
                   and protocol['validation'][1:] == ['2015-01-01', '2019-12-31']
                   and protocol['confirmation'][1:] == ['2020-01-01', '2023-05-29'] and fit_status['cutoff'] == '2019-12-31'
                   and status['compute_cutoff'] == '2023-05-29')
        self.check('three_year_OOS_definition', protocol['oos_start'] == '2023-05-30' and protocol['oos_end'] == '2026-05-29')

        observation_path = ROOT / next(iter(protocol['observation_source_sha256']))
        cache_dir = observation_path.parent
        cache_validation = json.loads(self.register(cache_dir / 'entry_state_validation.json').read_text())
        amendment = json.loads(self.register(cache_dir / 'technical_validation_amendment.json').read_text())
        self.hashes('observation_cache', cache_validation['output_sha256'], cache_dir)
        self.check('observation_cache_audited', cache_validation['status'] == amendment['status'] == 'passed'
                   and amendment['amended_source_sha256'] == sha(ROOT / 'src/fx_is_states.py'))
        discovery = {}
        seen_accounts = set()
        with observation_path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                seen_accounts.add((row['pair'], row['strategy']))
                if row['block'] not in ('DISCOVERY_1', 'DISCOVERY_2', 'DEVELOPMENT') or int(row['state']) < 0:
                    continue
                identity = (row['pair'], row['strategy'], row['feature'], int(row['state']))
                discovery.setdefault(identity, {})[row['block']] = {field: float(row[field]) for field in
                    ('closed_n', 'return_sum_pct', 'return_without_worst_trade_pct')}
        self.check('discovery_considered_all_180_accounts', seen_accounts == {(p, s) for p in CURRENCIES for s in STRATEGIES})
        expected_proposals = {}
        for (pair, strategy, feature, state), blocks in discovery.items():
            if all(blocks[b]['closed_n'] >= 10 and blocks[b]['return_sum_pct'] < 0 for b in ('DISCOVERY_1', 'DISCOVERY_2')) and blocks['DEVELOPMENT']['return_without_worst_trade_pct'] < 0:
                identifier = f'{pair}|{strategy}|STATE_{feature}_{state}'
                expected_proposals[identifier] = blocks
        proposal_lookup = {row['candidate_id']: row for row in proposals}
        self.check('discovery_proposal_exact_set', len(proposal_lookup) == len(proposals)
                   and set(proposal_lookup) == set(expected_proposals) and len(proposals) == fit_status['proposals'])
        self.check('discovery_evidence_matches_cache', all(row['discovery_evidence'] == expected_proposals.get(identifier)
                   for identifier, row in proposal_lookup.items()))
        self.check('proposal_specs_fixed_single_known_state', all(row['spec'] == {'id': f"STATE_{row['spec']['feature']}_{row['spec']['state']}",
                   'feature': row['spec']['feature'], 'state': row['spec']['state']}
                   and row['spec']['feature'] in protocol['state_bins'] and row['spec']['state'] in (0, 1, 2, 3)
                   for row in proposals))

        fit_metrics, fit_selection = self.read_csv('fit_metrics.csv'), self.read_csv('fit_selection.csv')
        self.check('fit_metrics_exact_grid', not fit_metrics.duplicated(['candidate_id', 'split', 'variant']).any()
                   and set(zip(fit_metrics.candidate_id, fit_metrics.split, fit_metrics.variant)) ==
                   {(c, p, v) for c in proposal_lookup for p in ('DISCOVERY', 'VALIDATION', 'FIT') for v in ('BASE', 'FILTER')})
        self.check('fit_dates_no_confirmation', pd.to_datetime(fit_metrics.end_date).le('2019-12-31').all())
        # Only state labels and dates are read from this ledger: no later outcome values are needed.
        state_columns = [name + '_state' for name in protocol['state_bins']]
        base_entries = pd.read_csv(cache_dir / 'is_entry_state_ledger.csv.gz', usecols=[
            'pair', 'strategy', 'entry_date', 'exit_date', 'eligible_closed', *state_columns],
            parse_dates=['entry_date', 'exit_date'])
        entry_groups = {(p, s): sub for (p, s), sub in base_entries.groupby(['pair', 'strategy'])}
        fit_lookup = fit_metrics.set_index(['candidate_id', 'split', 'variant'])
        rebuilt = []
        for candidate, proposal in proposal_lookup.items():
            item = {'candidate_id': candidate, 'pair': proposal['pair'], 'strategy': proposal['strategy']}
            sub = entry_groups[proposal['pair'], proposal['strategy']]
            rejected = sub[proposal['spec']['feature'] + '_state'].eq(proposal['spec']['state'])
            passes = []
            for phase, start, end in (protocol['discovery'], protocol['validation']):
                changed = int((rejected & sub.eligible_closed & sub.entry_date.ge(start) & sub.entry_date.le(end) & sub.exit_date.le(end)).sum())
                flags = independent_phase(fit_lookup.loc[(candidate, phase, 'BASE')], fit_lookup.loc[(candidate, phase, 'FILTER')], changed, phase == 'DISCOVERY')
                prefix = phase.lower()
                item[prefix + '_changed_n'] = changed
                item[prefix + '_pass'] = all(flags.values())
                item[prefix + '_failed'] = '|'.join(k for k, value in flags.items() if not value)
                passes.append(all(flags.values()))
            item['fit_pass'] = all(passes)
            item['validation_annualized_pct'] = fit_lookup.loc[(candidate, 'VALIDATION', 'FILTER'), 'gross_annualized_simple_pct']
            item['fit_delta_pp'] = fit_lookup.loc[(candidate, 'FIT', 'FILTER'), 'gross_simple_return_pct'] - fit_lookup.loc[(candidate, 'FIT', 'BASE'), 'gross_simple_return_pct']
            rebuilt.append(item)
        rebuilt = pd.DataFrame(rebuilt).sort_values(['validation_annualized_pct', 'fit_delta_pp', 'candidate_id'], ascending=[False, False, True], kind='stable')
        rebuilt['preconfirmation_rank'] = np.nan
        winners = rebuilt.loc[rebuilt.fit_pass].drop_duplicates(['pair', 'strategy'])
        rebuilt.loc[winners.index, 'preconfirmation_rank'] = np.arange(1, len(winners) + 1)
        columns = ['discovery_changed_n', 'validation_changed_n', 'discovery_pass', 'validation_pass', 'fit_pass',
                   'discovery_failed', 'validation_failed', 'validation_annualized_pct', 'fit_delta_pp', 'preconfirmation_rank']
        self.values('DV_independent_rescreen', fit_selection, rebuilt, columns,
                    numeric=['discovery_changed_n', 'validation_changed_n', 'validation_annualized_pct', 'fit_delta_pp', 'preconfirmation_rank'])
        self.check('DV_all_candidate_sort_order', fit_selection.candidate_id.tolist() == rebuilt.candidate_id.tolist())
        expected_locked = rebuilt.loc[rebuilt.preconfirmation_rank.notna()].sort_values('preconfirmation_rank')
        locked_ids = [row['candidate_id'] for row in lock['candidates']]
        self.check('DV_top1_exact_lock_and_rank', locked_ids == expected_locked.candidate_id.tolist()
                   and [row['preconfirmation_rank'] for row in lock['candidates']] == expected_locked.preconfirmation_rank.tolist())
        self.check('lock_specs_match_discovery', all(row['spec'] == proposal_lookup[row['candidate_id']]['spec'] for row in lock['candidates']))
        self.check('four_locked_accounts', len(locked_ids) == fit_status['locked_candidates'] == status['preconfirmation_locked'] == 4)

        metrics, recent = self.read_csv('confirmation_full_metrics.csv'), self.read_csv('confirmation_recent_metrics.csv')
        selection = self.read_csv('confirmation_selection.csv')
        daily, fit_daily = self.read_csv('locked_is_daily.csv.gz'), self.read_csv('fit_locked_daily.csv.gz')
        ledger, decisions = self.read_csv('locked_is_trade_ledger.csv.gz'), self.read_csv('locked_is_decisions.csv.gz')
        for name, frame in [('full_metrics', metrics), ('recent_metrics', recent), ('selection', selection), ('daily', daily),
                            ('fit_daily', fit_daily), ('ledger', ledger), ('decisions', decisions)]:
            self.check('locked_candidates_only:' + name, set(frame.candidate_id) == set(locked_ids))
        self.check('locked_candidates_only:effects', len(effects) == len(locked_ids) and {r['candidate_id'] for r in effects} == set(locked_ids))
        for frame in (daily, fit_daily):
            frame['date'] = pd.to_datetime(frame.date)
        for column in ('signal_date', 'entry_date', 'exit_date'):
            ledger[column] = pd.to_datetime(ledger[column])
        for name, frame in [('fit', fit_daily), ('IS', daily)]:
            self.check('daily_unique:' + name, not frame.duplicated(['candidate_id', 'variant', 'date']).any())
            self.check('daily_zero_cost:' + name, np.array_equal(frame.gross_daily_return, frame.net_daily_return))
            self.check('daily_cutoff:' + name, frame.date.max() <= pd.Timestamp('2019-12-31' if name == 'fit' else protocol['is_end']))
        for name, frame in [('fit', fit_metrics), ('IS', metrics), ('recent', recent)]:
            self.check('metric_zero_cost:' + name, np.array_equal(frame.gross_simple_return_pct, frame.net_simple_return_pct)
                       and np.array_equal(frame.gross_max_drawdown_pp, frame.net_max_drawdown_pp) and frame.cost_drag_pp.eq(0).all())
        self.check('ledger_zero_cost', ledger.transaction_cost_usd.eq(0).all()
                   and np.array_equal(ledger.gross_simple_return_pct, ledger.net_trade_return_pct))

        # Raw quote dates only; discard every date after the IS cutoff before use.
        calendars = {p: [] for p in CURRENCIES}
        workbook = ROOT / next(name for name in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in name)
        with zipfile.ZipFile(workbook) as archive:
            rows = _workbook_rows(archive, 'xl/worksheets/sheet6.xml', {'B', 'D', 'E'})
            next(rows)
            for row in rows:
                if row.get('D') not in calendars or row.get('E') is None:
                    continue
                date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(row['B']))
                if date <= pd.Timestamp(protocol['is_end']):
                    calendars[row['D']].append(date)
        calendars = {p: pd.DatetimeIndex(sorted(dates)) for p, dates in calendars.items()}
        reconstructed_ledgers, daily_groups = {}, {}
        for (candidate, variant), frame in daily.groupby(['candidate_id', 'variant']):
            frame = frame.sort_values('date').reset_index(drop=True)
            daily_groups[candidate, variant] = frame
            pair = frame.pair.iloc[0]
            calendar = calendars[pair]
            expected_dates = calendar[(calendar >= protocol['is_start']) & (calendar <= protocol['is_end'])]
            self.check('daily_calendar:' + candidate + ':' + variant, pd.DatetimeIndex(frame.date).equals(expected_dates))
            history = fit_daily.loc[fit_daily.candidate_id.eq(candidate) & fit_daily.variant.eq(variant)].sort_values('date')
            prefix = frame.loc[frame.date.le('2019-12-31')]
            self.check('DV_prefix:' + candidate + ':' + variant, pd.DatetimeIndex(history.date).equals(pd.DatetimeIndex(prefix.date))
                       and np.array_equal(history.executed, prefix.executed) and np.array_equal(history.held, prefix.held)
                       and np.allclose(history.gross_daily_return, prefix.gross_daily_return, atol=1e-14, rtol=0))
            self.check('next_close_held:' + candidate + ':' + variant, frame.held.iloc[0] == 0
                       and np.array_equal(frame.held.to_numpy()[1:], frame.executed.to_numpy()[:-1]))
            sub = ledger.loc[ledger.candidate_id.eq(candidate) & ledger.variant.eq(variant)].sort_values('entry_date').copy()
            executed = frame.executed.to_numpy()
            changes = np.flatnonzero(executed != np.r_[0, executed[:-1]])
            episodes = []
            for i, entry in enumerate(changes):
                if executed[entry] == 0:
                    continue
                closed = i + 1 < len(changes)
                exit_index = changes[i + 1] if closed else len(frame) - 1
                episodes.append((frame.date.iloc[entry], frame.date.iloc[exit_index], int(executed[entry]), closed, bool(closed and exit_index > entry)))
            stored = list(sub[['entry_date', 'exit_date', 'position', 'closed', 'eligible_closed']].itertuples(index=False, name=None))
            self.check('independent_execution_episodes:' + candidate + ':' + variant, stored == episodes)
            expected_entry = calendar.get_indexer(sub.entry_date)
            expected_exit = calendar.get_indexer(sub.exit_date)
            self.check('ledger_indices:' + candidate + ':' + variant, np.array_equal(sub.entry_idx, expected_entry)
                       and np.array_equal(sub.exit_idx, expected_exit) and np.array_equal(sub.signal_idx, expected_entry - 1)
                       and pd.DatetimeIndex(sub.signal_date).equals(calendar[expected_entry - 1]))
            values = []
            for trade in sub.itertuples(index=False):
                values.append(frame.loc[frame.date.gt(trade.entry_date) & frame.date.le(trade.exit_date), 'gross_daily_return'].sum() * 100)
            self.check('ledger_simple_return_from_daily:' + candidate + ':' + variant,
                       np.allclose(sub.gross_simple_return_pct, values, atol=1e-8, rtol=1e-10))
            sub['recomputed_simple_pct'] = values
            reconstructed_ledgers[candidate, variant] = sub

        all_metrics = pd.concat([metrics, recent, fit_metrics.loc[fit_metrics.candidate_id.isin(locked_ids)]], ignore_index=True)
        recomputed_windows = 0
        for row in all_metrics.itertuples(index=False):
            frame = daily_groups[row.candidate_id, row.variant]
            sub = reconstructed_ledgers[row.candidate_id, row.variant]
            start, end = pd.Timestamp(row.start_date), pd.Timestamp(row.end_date)
            part = frame.loc[frame.date.between(start, end)]
            calendar = calendars[row.pair]
            first = int(calendar.get_loc(start))
            anchor = calendar[max(0, first - 1)]
            years = (end - anchor).days / 365.25
            returns = part.gross_daily_return.to_numpy() * 100
            cumulative = np.r_[0., returns.cumsum()]
            entered = sub.loc[sub.entry_date.between(start, end)]
            closed = entered.loc[entered.eligible_closed & entered.exit_date.le(end)]
            carry = sub.loc[sub.entry_date.lt(start) & sub.exit_date.ge(start)]
            recomputed = {'gross_simple_return_pct': returns.sum(), 'gross_annualized_simple_pct': returns.sum() / years,
                          'gross_max_drawdown_pp': (cumulative - np.maximum.accumulate(cumulative)).min(),
                          'calendar_years': years, 'observations': len(part), 'entry_n': len(entered), 'carry_in_n': len(carry),
                          'censored_entry_n': int((~entered.closed | entered.exit_date.gt(end)).sum()),
                          **independent_quality(closed.recomputed_simple_pct)}
            failures = [field for field, value in recomputed.items() if not same_number(getattr(row, field), value)]
            self.check('independent_window:' + row.candidate_id + ':' + row.variant + ':' + row.split,
                       not failures and pd.Timestamp(row.anchor_date) == anchor, failed_fields=failures)
            recomputed_windows += 1

        effect_lookup = {row['candidate_id']: row for row in effects}
        independent_effects = {}
        periods = [('IS', protocol['is_start'], protocol['is_end']), protocol['confirmation'], ('RECENT', '2015-01-01', protocol['is_end'])]
        for candidate in locked_ids:
            base, filtered = daily_groups[candidate, 'BASE'], daily_groups[candidate, 'FILTER']
            self.check('entry_rejection_only:' + candidate, ((filtered.held == base.held) | filtered.held.eq(0)).all()
                       and np.allclose(filtered.gross_daily_return, np.where(filtered.held.eq(0), 0., base.gross_daily_return), atol=1e-14, rtol=0))
            independent_effects[candidate] = {}
            for phase, start, end in periods:
                result = independent_effect(base, filtered, reconstructed_ledgers[candidate, 'BASE'], start, end)
                independent_effects[candidate][phase] = result
                self.check('independent_episode_effect:' + candidate + ':' + phase,
                           all(same_number(value, effect_lookup[candidate][phase][field]) for field, value in result.items()))

        original = screen_candidates(metrics)
        original_compare = original.rename(columns={'pass': 'original_is_pass'})
        compare_columns = [c for c in original_compare if c not in ('candidate_id', 'pass', 'freeze_rank', 'frozen')]
        numeric = [c for c in compare_columns if pd.api.types.is_numeric_dtype(original_compare[c]) and not pd.api.types.is_bool_dtype(original_compare[c])]
        self.values('original_full_IS_rescreen', selection, original_compare, compare_columns, numeric=numeric)
        selection_index = selection.set_index('candidate_id')
        recent_lookup = recent.set_index(['candidate_id', 'split', 'variant'])
        expected_final = []
        for candidate in locked_ids:
            saved = selection_index.loc[candidate]
            b, f = recent_lookup.loc[(candidate, 'CONFIRMATION', 'BASE')], recent_lookup.loc[(candidate, 'CONFIRMATION', 'FILTER')]
            cf = independent_phase(b, f, independent_effects[candidate]['CONFIRMATION']['changed_closed_n'])
            b, f = recent_lookup.loc[(candidate, 'RECENT', 'BASE')], recent_lookup.loc[(candidate, 'RECENT', 'FILTER')]
            robust = {'recent_absolute': f.gross_annualized_simple_pct >= 1 - TOL,
                      'recent_delta': f.gross_simple_return_pct > b.gross_simple_return_pct + TOL,
                      'recent_ratio': ratio_pass(b, f),
                      'full_effect_sample': independent_effects[candidate]['IS']['changed_closed_n'] >= 20,
                      'full_top3_robust': independent_effects[candidate]['IS']['delta_without_best3_pp'] > TOL,
                      'recent_top3_robust': independent_effects[candidate]['RECENT']['delta_without_best3_pp'] > TOL}
            self.check('confirmation_gates:' + candidate, bool(saved.confirmation_pass) == all(cf.values())
                       and clean_string(saved.confirmation_failed) == '|'.join(k for k, value in cf.items() if not value))
            self.check('recent_robustness_gates:' + candidate, bool(saved.robustness_pass) == all(robust.values())
                       and clean_string(saved.robustness_failed) == '|'.join(k for k, value in robust.items() if not value))
            passed = bool(saved.original_is_pass and all(cf.values()) and all(robust.values()))
            self.check('final_intersection:' + candidate, bool(saved['pass']) == bool(saved.frozen) == passed)
            if passed:
                expected_final.append(candidate)
        self.check('confirmation_keeps_lock_order', selection.candidate_id.tolist() == locked_ids
                   and selection.preconfirmation_rank.tolist() == list(range(1, len(locked_ids) + 1)))
        self.check('freeze_exact_survivors_no_fallback', [r['candidate_id'] for r in freeze['candidates']] == expected_final
                   and freeze['frozen_candidates'] == status['frozen_candidates'] == len(expected_final))
        self.check('zero_survivors', not expected_final and not selection.frozen.any()
                   and selection.freeze_rank.isna().all())
        self.check('empty_frozen_outputs', self.read_csv('is_frozen_metrics.csv').empty and self.read_csv('is_frozen_daily.csv.gz').empty)
        absent = ['oos_status.json', 'oos_daily.csv.gz', 'oos_metrics.csv', 'oos_anniversary_metrics.csv',
                  'oos_fresh_start_metrics.csv', 'candidate_assessment.csv', 'frozen_full_trade_ledger.csv.gz', 'frozen_full_episode_decisions.csv.gz']
        self.check('OOS_not_opened_no_artifacts', all(not (self.campaign / name).exists() for name in absent))
        self.details.update(accounts_considered=180, discovery_proposals=len(proposals), locked_candidates=len(locked_ids),
                            frozen_candidates=len(expected_final), recomputed_metric_windows=recomputed_windows,
                            independently_recomputed_episode_effect_windows=len(locked_ids) * len(periods),
                            independent_episode_effects=independent_effects,
                            confirmation_rejections=[{'candidate_id': c, 'original_is_failed': clean_string(selection_index.loc[c, 'failed_criteria']),
                                'confirmation_failed': clean_string(selection_index.loc[c, 'confirmation_failed']),
                                'robustness_failed': clean_string(selection_index.loc[c, 'robustness_failed'])} for c in locked_ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'reproduced/state_filter_v2')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-tests', action='store_true', help='Omit test execution; the JSON explicitly records this limitation.')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output.resolve() if args.output else campaign / 'independent_validation.json'
    if output.name in {p.name for p in campaign.iterdir() if p.name != 'independent_validation.json'} or output.suffix.lower() != '.json':
        raise ValueError('Audit output must be a separate JSON, never an existing campaign artifact.')
    audit = Audit(campaign)
    started = datetime.now(timezone.utc).isoformat()
    error = None
    try:
        audit.run()
    except Exception:
        error = traceback.format_exc()
        audit.check('audit_execution_completed', False, error=error)
    tests = {'executed': False}
    if not args.skip_tests:
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        log = result.stdout + result.stderr
        match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', log)
        tests = {'executed': True, 'command': 'python -B -m unittest discover -s tests -v', 'exit_code': result.returncode,
                 'tests_run': int(match.group(1)) if match else None, 'elapsed_seconds': float(match.group(2)) if match else None,
                 'test_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'tests').glob('test_*.py'))},
                 'output': log}
        audit.check('all_repository_tests', result.returncode == 0 and match is not None)
    for path, original in audit.files.items():
        audit.check('read_only_preserved:' + str(Path(path).relative_to(ROOT)), sha(path) == original)
    passed = sum(check['passed'] for check in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
              'scope': 'Completed v2 IS-only campaign. No OOS result opened, no strategy recomputation or selection mutation.',
              'method': 'Independent D/V phase gates and state-based effect counts; original SCREEN reapplication; direct daily/episode date slicing for simple sums, drawdowns, cohort quality and top-three contribution removal.',
              'checks_passed': passed, 'checks_total': len(audit.checks), 'verified_file_count': len(audit.files),
              'verified_files_sha256': {str(Path(p).relative_to(ROOT)).replace('\\', '/'): value for p, value in audit.files.items()},
              'details': audit.details, 'tests': tests, 'checks': audit.checks}
    if error:
        result['error'] = error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks_passed': passed, 'checks_total': len(audit.checks),
                      'verified_files': len(audit.files), 'tests_run': tests.get('tests_run'), 'output': str(output)}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
