"""Independent IS-only audit of the fixed three-pair entry-interaction study.

This reads saved IS evidence and price cells at/before IS_END. It does not call
the candidate engine, the BASE signal generators, or any OOS entry point. Only
independent_validation.json is written; screening artifacts remain immutable.
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
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_state_filter_research import Audit, ROOT, sha, same_number, independent_quality
from verify_state_filter_supplement import ledger_from_executed, window_from_path
from fx_baselines import CURRENCIES, STRATEGIES
from fx_filter_evaluation import SCREEN, screen_candidates

TOL = 1e-10
INTERACTIONS = (
    ('efficiency_volatility', 'efficiency_63', 'vol_ratio_20_63'),
    ('efficiency_directional_stretch', 'efficiency_63', 'direction_return_20_vol'),
    ('dollar_carry_conflict', 'direction_dxy_d20_pct', 'direction_spread_bp'),
)
BINS = {
    'vol_ratio_20_63': [.8, 1., 1.25], 'realized_vol_20_pct': [5., 10., 15.],
    'efficiency_63': [.1, .2, .4], 'direction_return_20_vol': [-1., 0., 1.],
    'direction_distance_ma_60_vol': [-1., 0., 1.], 'K_vix': [15., 20., 30.],
    'K_vix_d5': [-3., 0., 3.], 'direction_spread_bp': [-100., 0., 100.],
    'direction_spread_d20_bp': [-10., 0., 10.], 'direction_dxy_d20_pct': [-1., 0., 1.],
    'direction_commodity_d20_pct': [-1., 0., 1.], 'base_pnl_63_pct': [-2., 0., 2.],
    'base_pnl_252_pct': [-5., 0., 5.],
}
BLOCKS = (('D1', '2005-01-03', '2009-12-31'), ('D2', '2010-01-01', '2014-12-31'),
          ('D', '2005-01-03', '2014-12-31'))
CELL_KEYS = ['pair', 'strategy', 'interaction_id', 'state_a', 'state_b']
SOURCE_LEDGER = ROOT / 'reproduced/filter_is_diagnostics_v2/is_entry_state_ledger.csv.gz'
OBS = ROOT / 'reproduced/entry_interaction_observations_v5'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def truth(value):
    return not pd.isna(value) and str(value).lower() in ('true', '1', '1.0')


def codes(values, edges):
    """Independent interval-membership implementation, including exact edges."""
    values = np.asarray(values, float)
    result = sum((values >= edge).astype(int) for edge in edges)
    result[~np.isfinite(values)] = -1
    return result


def describe(values):
    values = np.asarray(values, float)
    q = independent_quality(values)
    losses = np.sort(values[values < -TOL])
    total = float(values.sum())
    return {**q, 'return_sum_pct': total,
            'return_without_worst1_pct': total - float(losses[:1].sum()),
            'return_without_worst3_pct': total - float(losses[:3].sum()),
            'largest_loss_pct': float(losses[0]) if len(losses) else np.nan,
            'largest_loss_share': float(losses[0] / losses.sum()) if len(losses) else np.nan,
            'low_sample_flag': len(values) < 20,
            'low_outcome_sample_flag': q['win_n'] < 20 or q['loss_n'] < 20}


def spec_for(a, x, b, y):
    conditions = sorted([{'feature': a, 'state': int(x)}, {'feature': b, 'state': int(y)}],
                        key=lambda c: c['feature'])
    return {'id': 'AND_' + '__'.join(f"{c['feature']}_{c['state']}" for c in conditions),
            'conditions': conditions}


def load_D_ledger(path):
    """Do not convert post-D/open/cross-D trade outcomes to numbers."""
    rows, skipped = [], 0
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for r in csv.DictReader(stream):
            if not ('2005-01-03' <= r['entry_date'] <= '2014-12-31') \
                    or r['eligible_closed'] != 'True' or r['closed'] != 'True' or r['exit_date'] > '2014-12-31':
                skipped += 1
                continue
            row = {c: r[c] for c in ['pair', 'strategy', 'family', 'trade_id', 'signal_date', 'entry_date', 'exit_date']}
            row.update({c: float(r[c]) if r[c] else np.nan for c in
                        ['position', 'signal_idx', 'entry_idx', 'exit_idx', 'gross_simple_return_pct',
                         'transaction_cost_usd', *BINS]})
            row.update({c + '_state': int(r[c + '_state']) for c in BINS})
            rows.append(row)
    return pd.DataFrame(rows), skipped


def read_IS_quotes(path, cutoff):
    """Read date/currency first; never decode a price value after the cutoff."""
    quotes = {p: [] for p in CURRENCIES}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml'))]

        def value(cell):
            if cell is None:
                return None
            kind = cell.get('t', '')
            raw = cell.find(NS + 'v')
            if kind == 'inlineStr':
                return ''.join(node.text or '' for node in cell.iter(NS + 't'))
            if raw is None or not raw.text:
                return None
            if kind == 's':
                return shared[int(raw.text)]
            return raw.text if kind in ('str', 'e') else float(raw.text)

        with archive.open('xl/worksheets/sheet6.xml') as stream:
            for _, element in ET.iterparse(stream, events=('end',)):
                if element.tag != NS + 'row':
                    continue
                cells = {''.join(ch for ch in cell.get('r', '') if ch.isalpha()): cell
                         for cell in element.findall(NS + 'c')}
                serial = value(cells.get('B'))
                if isinstance(serial, (float, int)) and np.isfinite(serial):
                    day = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(serial))
                    if day <= cutoff:
                        pair = value(cells.get('D'))
                        if pair in quotes:
                            price = value(cells.get('E'))
                            if price is not None:
                                quotes[pair].append((day, float(price)))
                element.clear()
    return {p: pd.DataFrame(sorted(rows), columns=['date', 'price']) for p, rows in quotes.items()}


class InteractionAudit(Audit):
    def frame(self, path):
        return pd.read_csv(self.register(path), float_precision='round_trip')

    def document(self, path):
        return json.loads(self.register(path).read_text(encoding='utf-8'))

    def compare_frames(self, name, saved, computed, keys):
        """Compare every computed column; unrelated descriptive columns are allowed."""
        missing = set(computed).difference(saved)
        self.check(name + ':schema', not missing, missing=sorted(missing))
        if missing:
            return
        saved, computed = saved.sort_values(keys).reset_index(drop=True), computed.sort_values(keys).reset_index(drop=True)
        self.check(name + ':unique_grid', not saved.duplicated(keys).any() and not computed.duplicated(keys).any()
                   and len(saved) == len(computed), saved_rows=len(saved), computed_rows=len(computed))
        if len(saved) != len(computed):
            return
        errors = []
        for c in computed:
            a, b = saved[c], computed[c]
            if pd.api.types.is_datetime64_any_dtype(b):
                okay = pd.to_datetime(a).fillna(pd.Timestamp('1900-01-01')).equals(b.fillna(pd.Timestamp('1900-01-01')))
            elif pd.api.types.is_numeric_dtype(b):
                okay = bool(np.isclose(pd.to_numeric(a).astype(float), b.astype(float), atol=1e-8, rtol=1e-10, equal_nan=True).all())
            else:
                okay = a.fillna('').astype(str).equals(b.fillna('').astype(str))
            if not okay:
                errors.append(c)
        self.check(name + ':all_values', not errors, mismatched_fields=errors)

    def discovery(self, observation_protocol):
        frame, skipped = load_D_ledger(self.register(SOURCE_LEDGER))
        self.check('D_guarded_closed_cohort', not frame.duplicated(['pair', 'strategy', 'trade_id']).any()
                   and set(zip(frame.pair, frame.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES))
                   and np.isfinite(frame.gross_simple_return_pct).all() and frame.transaction_cost_usd.eq(0).all()
                   and frame.signal_date.lt(frame.entry_date).all() and frame.entry_date.lt(frame.exit_date).all())
        for feature, edges in BINS.items():
            self.check('D_fixed_states:' + feature,
                       np.array_equal(codes(frame[feature], edges), frame[feature + '_state']))
        partitions, comparisons, coverage, support, proposals = [], [], [], [], []
        partition_conservation = True
        for (pair, strategy), account in frame.groupby(['pair', 'strategy'], sort=True):
            cohorts = {block: account.loc[account.entry_date.ge(start) & account.exit_date.le(end)]
                       for block, start, end in BLOCKS}
            for block, cohort in cohorts.items():
                coverage.append({'pair': pair, 'strategy': strategy, 'block': block, **describe(cohort.gross_simple_return_pct)})
            for mechanism, a, b in INTERACTIONS:
                for x, y in itertools.product(range(4), repeat=2):
                    key = {'pair': pair, 'strategy': strategy, 'interaction_id': mechanism,
                           'feature_a': a, 'state_a': x, 'feature_b': b, 'state_b': y}
                    joint = {}
                    for block, cohort in cohorts.items():
                        aa, bb = codes(cohort[a], BINS[a]), codes(cohort[b], BINS[b])
                        known = (aa >= 0) & (bb >= 0)
                        am, bm = aa == x, bb == y
                        masks = {'joint': known & am & bm, 'A_only': known & am & ~bm,
                                 'B_only': known & ~am & bm, 'neither': known & ~am & ~bm, 'UNKNOWN': ~known}
                        values = cohort.gross_simple_return_pct.to_numpy(float)
                        partition_conservation &= np.array_equal(sum(m.astype(int) for m in masks.values()), np.ones(len(values), int))
                        partition_conservation &= same_number(sum(values[m].sum() for m in masks.values()), values.sum())
                        joint[block] = describe(values[masks['joint']])
                        for group, mask in masks.items():
                            partitions.append({**key, 'block': block, 'group': group, 'cohort_n': len(values),
                                               'known_features_n': int(known.sum()), **describe(values[mask])})
                        for group, mask in {'A_all': am, 'B_all': bm, 'retained': ~masks['joint']}.items():
                            comparisons.append({**key, 'block': block, 'group': group, 'cohort_n': len(values),
                                                'known_features_n': int(known.sum()), **describe(values[mask])})
                    nominated = all(joint[k]['closed_n'] >= 10 and joint[k]['return_sum_pct'] < -TOL for k in ('D1', 'D2')) \
                        and joint['D']['return_sum_pct'] < -TOL
                    support.append({**key, 'D_repeated_negative_support10': bool(nominated),
                                    'D_repeated_negative_support20': bool(nominated and all(joint[k]['closed_n'] >= 20 for k in ('D1', 'D2'))),
                                    **{f'{block}_{n}': values[n] for block, values in joint.items() for n in
                                       ('closed_n', 'return_sum_pct', 'win_rate', 'payoff_ratio', 'expectancy_pct',
                                        'return_without_worst1_pct', 'return_without_worst3_pct')}})
                    if nominated:
                        spec = spec_for(a, x, b, y)
                        proposals.append({'candidate_id': '|'.join((pair, strategy, spec['id'])),
                                          'pair': pair, 'strategy': strategy, 'spec': spec, 'interaction_id': mechanism,
                                          'discovery_evidence': {k: {n: v[n] for n in ('closed_n', 'return_sum_pct',
                                              'return_without_worst1_pct', 'return_without_worst3_pct')} for k, v in joint.items()}})
        self.check('five_way_partition_count_and_PnL_conservation', partition_conservation)
        for filename, records, keys in [
                ('interaction_state_summary.csv', partitions, CELL_KEYS + ['block', 'group']),
                ('interaction_comparison_summary.csv', comparisons, CELL_KEYS + ['block', 'group']),
                ('interaction_account_coverage.csv', coverage, ['pair', 'strategy', 'block']),
                ('interaction_support_diagnostics.csv', support, CELL_KEYS)]:
            self.compare_frames(filename, self.frame(OBS / filename), pd.DataFrame(records), keys)
        self.details.update(D_closed_episodes=len(frame), later_or_nonclosed_outcomes_skipped_before_D_numeric_parse=skipped,
                            D_proposal_numeric_outcome_blocks=['D1', 'D2', 'D'], discovery_cells=len(support),
                            discovery_partition_rows=len(partitions), discovery_comparison_rows=len(comparisons),
                            cross_half_policy='Eligible trades crossing 2009/2010 belong to combined D only; D is not independent of D1/D2.')
        print('Independent D partitions and proposals reconstructed:', len(proposals), flush=True)
        return proposals

    def run(self):
        protocol, status = self.read_json('protocol.json'), self.read_json('development_status.json')
        freeze, hypotheses = self.read_json('frozen_selection.json'), self.read_json('is_hypotheses.json')
        op = self.document(OBS / 'interaction_observation_protocol.json')
        ov = self.document(OBS / 'interaction_observation_validation.json')
        self.check('exact_campaign_scope_clock_and_original_SCREEN', protocol['version'] == 'entry_interaction_filter_v5'
                   and protocol['is_start'] == '2005-01-03' and protocol['is_end'] == '2023-05-29'
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0 and protocol['screen'] == SCREEN
                   and protocol['initial_capital_usd'] == 100000 and protocol['accounting'] == 'fixed_initial_capital_simple'
                   and set(protocol['scope']['pairs']) == set(CURRENCIES)
                   and set(protocol['scope']['strategies']) == set(STRATEGIES) and protocol['scope']['baseline_accounts'] == 180)
        self.check('completed_preOOS_development', status['status'] == 'passed' and status['is_screen_completed']
                   and status['source_integrity_pass'] and status['compute_cutoff'] == protocol['is_end']
                   and status['oos_evaluated'] is False and status['holdout_returns_computed'] is False)
        self.check('fixed_three_interactions_all_48_specs', protocol['interaction_registry'] == [list(i) for i in INTERACTIONS]
                   and protocol['candidate_specs'] == [spec_for(a, x, b, y) for _, a, b in INTERACTIONS
                                                       for x in range(4) for y in range(4)]
                   and protocol['state_bins'] == op['state_bins'] == BINS)
        self.check('observation_scope_and_hash', op['compute_cutoff'] == ov['compute_cutoff'] == '2014-12-31'
                   and op['oos_computed'] is ov['oos_computed'] is False and ov['status'] == 'passed'
                   and ov['accounts'] == op['accounts'] == 180 and op['cost_bp'] == 0
                   and ov['protocol_sha256'] == sha(OBS / 'interaction_observation_protocol.json'))
        self.check('freeze_chain_and_no_prior_OOS', freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
                   and freeze['oos_evaluated_at_freeze'] is False)
        self.check('observation_proposal_replay_freeze_chronology', pd.Timestamp(op['created_utc']) < pd.Timestamp(ov['finished_utc'])
                   < pd.Timestamp(protocol['created_utc']) < pd.Timestamp(hypotheses['created_utc'])
                   <= pd.Timestamp(status['started_utc']) < pd.Timestamp(freeze['created_utc']) <= pd.Timestamp(status['finished_utc']))
        for key in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest_agreement:' + key, freeze[key] == protocol[key])
            self.hashes(key, protocol[key]); self.hashes('observation_' + key, op[key])
        self.hashes('observation_link', protocol['observation_source_sha256'])
        self.hashes('observation_outputs', ov['output_sha256'], OBS)
        self.hashes('IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.check('every_IS_artifact_hashed', {p.name for p in self.campaign.glob('is_*') if p.is_file()} == set(freeze['is_output_sha256']))
        self.check('no_OOS_artifacts_present', all(not (self.campaign / name).exists() for name in
                   ('oos_status.json', 'oos_daily.csv.gz', 'oos_metrics.csv', 'oos_anniversary_metrics.csv',
                    'oos_fresh_start_metrics.csv', 'candidate_assessment.csv', 'frozen_full_trade_ledger.csv.gz',
                    'frozen_full_episode_decisions.csv.gz')))
        nominated = self.discovery(op)
        items = hypotheses['candidates']
        expected = {r['candidate_id']: r for r in nominated}
        self.check('all_and_only_D_supported_proposals', len(items) == len(expected)
                   and {r['candidate_id'] for r in items} == set(expected), expected=len(expected), actual=len(items))
        for h in items:
            r = expected[h['candidate_id']]
            self.check('D_proposal_spec:' + h['candidate_id'], h['spec'] == r['spec'] and h['filter_id'] == r['spec']['id']
                       and h['pair'] == r['pair'] and h['strategy'] == r['strategy'] and h['is_new'] is True
                       and h['interaction_id'] == r['interaction_id']
                       and all(same_number(h['discovery_evidence'][block][field], value)
                               for block, values in r['discovery_evidence'].items() for field, value in values.items()))
        metrics, selection = self.read_csv('is_all_metrics.csv'), self.read_csv('is_selection.csv')
        coverage = self.read_csv('is_account_coverage.csv')
        candidate_ids = set(expected)
        counts = pd.Series([(h['pair'], h['strategy']) for h in items], dtype=object).value_counts()
        self.check('all_180_account_coverage_and_proposal_counts', len(coverage) == 180
                   and set(zip(coverage.pair, coverage.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES))
                   and coverage.compute_end.eq(protocol['is_end']).all()
                   and all(r.candidate_count == counts.get((r.pair, r.strategy), 0) for r in coverage.itertuples())
                   and coverage.candidate_count.sum() == len(items))
        periods = [('IS', protocol['is_start'], protocol['is_end']), *protocol['is_blocks']]
        self.check('full_IS_original_screen_coverage_every_proposal', set(selection.candidate_id) == candidate_ids
                   and not metrics.duplicated(['candidate_id', 'variant', 'split']).any()
                   and set(zip(metrics.candidate_id, metrics.variant, metrics.split))
                   == set(itertools.product(candidate_ids, ('BASE', 'FILTER'), [p[0] for p in periods])))
        rescreened = screen_candidates(metrics) if len(items) else selection.copy()
        self.compare_frames('original_SCREEN_and_ranking', selection, rescreened, ['candidate_id'])
        self.check('saved_IS_sort_order', selection.candidate_id.tolist() == rescreened.candidate_id.tolist())
        winners = rescreened.loc[rescreened.frozen.map(truth)].sort_values('freeze_rank')
        self.check('freeze_exact_ranked_original_SCREEN_winners', [r['candidate_id'] for r in freeze['candidates']] == winners.candidate_id.tolist()
                   and freeze['frozen_candidates'] == status['frozen_candidates'] == len(winners)
                   and freeze['all_passing_versions'] == status['all_passing_versions'] == int(rescreened['pass'].sum())
                   and freeze['all_candidate_accounts_tested'] == status['candidates_evaluated'] == len(items)
                   and all(r['spec'] == expected[r['candidate_id']]['spec'] for r in freeze['candidates']))
        for item in freeze['candidates']:
            saved = winners.set_index('candidate_id').loc[item['candidate_id']]
            self.check('freeze_record_matches_selection:' + item['candidate_id'], all(
                same_number(item[c], saved[c]) if isinstance(saved[c], (float, int, np.number, bool, np.bool_))
                else (('' if item[c] is None else str(item[c])) == ('' if pd.isna(saved[c]) else str(saved[c])))
                for c in rescreened.columns if c != 'candidate_id'))
        if items:
            self.audit_paths(protocol, items, metrics, winners, periods)
        else:
            self.check('no_candidate_decisions_when_no_proposals', not (self.campaign / 'is_all_decisions.csv.gz').exists())
            self.check('empty_frozen_data', self.read_csv('is_frozen_daily.csv.gz').empty and self.read_csv('is_frozen_metrics.csv').empty)
        self.check('all_registered_inputs_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(campaign_version=protocol['version'], accounts=180, candidates=len(items),
                            candidate_accounts=int(coverage.candidate_count.gt(0).sum()),
                            original_SCREEN_passes=int(rescreened['pass'].sum()), frozen_candidates=len(winners),
                            strategy_engine_replays=0, OOS_results_opened=False, holdout_returns_computed=False,
                            post_IS_price_values_decoded=0, raw_prices_computed_through=protocol['is_end'],
                            audit_scope='IS-only pre-OOS audit. Historical reuse is not converted into untouched evidence.')

    def audit_paths(self, protocol, items, metrics, winners, periods):
        ledger = self.frame(SOURCE_LEDGER)
        for c in ('signal_date', 'entry_date', 'exit_date'):
            ledger[c] = pd.to_datetime(ledger[c])
        self.check('IS_ledger_scope_unique', not ledger.duplicated(['pair', 'strategy', 'signal_idx']).any()
                   and ledger.exit_date.le(protocol['is_end']).all()
                   and set(zip(ledger.pair, ledger.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES)))
        for c, edges in BINS.items():
            self.check('full_IS_cached_states:' + c, np.array_equal(codes(ledger[c], edges), ledger[c + '_state']))
        for date_col, source_cols in {
                'K_dxy_obs_date': ['K_dxy_d20_pct'], 'K_rate_obs_date': ['K_spread_bp', 'K_spread_d20_bp']}.items():
            known = np.isfinite(ledger[source_cols]).any(axis=1)
            age = ledger.signal_date - pd.to_datetime(ledger[date_col])
            self.check('macro_strict_prior_freshness:' + date_col,
                       bool((age[known].notna() & age[known].gt(pd.Timedelta(0)) & age[known].le(pd.Timedelta(days=7))).all()))
        decisions = self.read_csv('is_all_decisions.csv.gz')
        self.check('complete_unique_decision_candidates', set(decisions.candidate_id) == {h['candidate_id'] for h in items}
                   and not decisions.duplicated(['candidate_id', 'signal_idx']).any())
        workbook = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in p and p.endswith('.xlsx'))
        quotes = read_IS_quotes(self.register(workbook), pd.Timestamp(protocol['is_end']))
        paths, windows = {}, {}
        for pair, q in quotes.items():
            dates = pd.DatetimeIndex(q.date); prices = q.price.to_numpy(float)
            self.check('IS_price_calendar:' + pair, dates.is_unique and dates.is_monotonic_increasing
                       and dates[-1] <= pd.Timestamp(protocol['is_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1.]
            p = pd.Series(prices)
            ret = p.pct_change()
            vol20 = ret.rolling(20).std() * np.sqrt(252) * 100
            vol63 = ret.rolling(63).std() * np.sqrt(252) * 100
            path_length = p.diff().abs().rolling(63).sum()
            efficiency = (p.diff(63).abs() / path_length.where(path_length.ne(0))).mask(path_length.eq(0), 0.)
            directional = p.pct_change(20) * 100 / (vol20 * np.sqrt(20 / 252))
            for strategy in STRATEGIES:
                key = (pair, strategy)
                saved = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)].sort_values('entry_idx')
                executed = np.zeros(len(dates), dtype=int); occupancy = executed.copy()
                for row in saved.itertuples(index=False):
                    stop = int(row.exit_idx) if row.closed else len(dates)
                    executed[int(row.entry_idx):stop] = int(row.position)
                    occupancy[int(row.entry_idx):stop] += 1
                held = np.r_[0, executed[:-1]]; gross = held * spot
                rebuilt = ledger_from_executed(dates, executed, gross)
                self.compare_frames('BASE_ledger:' + '|'.join(key), saved, rebuilt, ['entry_idx'])
                self.check('BASE_no_overlap_zero_cost:' + '|'.join(key), (occupancy <= 1).all() and saved.transaction_cost_usd.eq(0).all())
                idx = saved.signal_idx.to_numpy(int); direction = saved.position.to_numpy(float)
                independent_features = {'efficiency_63': efficiency.to_numpy()[idx],
                                        'vol_ratio_20_63': (vol20 / vol63).to_numpy()[idx],
                                        'direction_return_20_vol': directional.to_numpy()[idx] * direction,
                                        'direction_dxy_d20_pct': saved.K_dxy_d20_pct.to_numpy() * direction,
                                        'direction_spread_bp': saved.K_spread_bp.to_numpy() * direction}
                self.check('signal_clock_feature_formula:' + '|'.join(key), all(np.allclose(saved[c], v, atol=1e-8, rtol=1e-10, equal_nan=True)
                           for c, v in independent_features.items()))
                paths[key] = dates, spot, executed, held, gross, rebuilt, saved, independent_features
                windows[key] = {phase: window_from_path(dates, executed, held, gross, rebuilt, start, end)
                                for phase, start, end in periods}
        metric_index = metrics.set_index(['candidate_id', 'variant', 'split'])
        independent_metrics, frozen_daily, frozen_ledgers, terminal = [], [], [], []
        frozen_ids = set(winners.candidate_id)
        decisions_checked = rejected_count = 0
        for number, h in enumerate(items):
            cid, key = h['candidate_id'], (h['pair'], h['strategy'])
            dates, spot, base_exec, base_held, base_returns, base_ledger, saved, independent_features = paths[key]
            rows = decisions.loc[decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            filled = rows.loc[~rows.entry_pending.map(truth)].copy()
            self.check('original_executed_episodes_complete:' + cid, len(filled) == len(saved)
                       and np.array_equal(filled.signal_idx.to_numpy(), saved.signal_idx.to_numpy()))
            original_target = np.zeros(len(dates), int); filtered_target = original_target.copy(); used = original_target.copy()
            by_signal = saved.set_index('signal_idx')
            errors = []
            for ordinal, row in enumerate(rows.itertuples(index=False)):
                start = int(row.signal_idx)
                stop = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
                pending = start + 1 == len(dates)
                good = (0 <= start < stop <= len(dates) and row.original_episode == ordinal
                        and dates[start] == pd.Timestamp(row.signal_date) and row.position in (-1, 1)
                        and bool(row.entry_pending) == pending)
                if pending:
                    good &= pd.isna(row.entry_idx) and pd.isna(row.entry_date)
                    terminal.append({'candidate_id': cid, 'signal_idx': start})
                else:
                    trade = by_signal.loc[start]
                    good &= row.entry_idx == start + 1 and pd.Timestamp(row.entry_date) == dates[start + 1]
                    good &= row.position == trade.position
                    # A close on the last quote is still a real close; an open
                    # trade may already have its technical exit pending there.
                    if trade.closed:
                        good &= stop == int(trade.exit_idx) - 1
                    else:
                        good &= stop >= len(dates) - 1
                if stop < len(dates):
                    good &= pd.Timestamp(row.original_end_signal_date) == dates[stop]
                else:
                    good &= pd.isna(row.original_end_signal_date)
                states = []
                for index, condition in enumerate(h['spec']['conditions'], 1):
                    prefix = f'condition_{index}_'
                    feature = condition['feature']; value = getattr(row, prefix + 'value')
                    state = int(codes([value], BINS[feature])[0]); states.append(state)
                    good &= getattr(row, prefix + 'feature') == feature
                    good &= getattr(row, prefix + 'rejected_state') == condition['state']
                    good &= getattr(row, prefix + 'state') == state
                    good &= bool(getattr(row, prefix + 'matches')) == (state == condition['state'])
                    if not pending:
                        good &= same_number(value, by_signal.loc[start, feature])
                known = all(state >= 0 for state in states)
                rejected = known and all(state == c['state'] for state, c in zip(states, h['spec']['conditions']))
                good &= bool(row.conditions_known) == known and bool(row.rejected) == rejected and bool(row.accepted) != rejected
                good &= row.status == ('UNKNOWN' if not known else 'REJECT' if rejected else 'KEEP')
                good &= row.condition == '&'.join(f"{c['feature']}={c['state']}" for c in h['spec']['conditions'])
                good &= pd.isna(row.actual_entry_idx) if rejected or pending else row.actual_entry_idx == start + 1
                used[start:stop] += 1; original_target[start:stop] = int(row.position)
                if not rejected:
                    filtered_target[start:stop] = int(row.position)
                rejected_count += int(rejected); decisions_checked += 1
                if not good:
                    errors.append(start)
            original_exec = np.r_[0, original_target[:-1]]
            self.check('all_AND_decisions_and_next_close_clock:' + cid, not errors and (used <= 1).all()
                       and np.array_equal(original_exec, base_exec), mismatched_signals=errors[:10])
            executed = np.r_[0, filtered_target[:-1]]; held = np.r_[0, executed[:-1]]; gross = held * spot
            filtered_ledger = ledger_from_executed(dates, executed, gross)
            if filtered_ledger.empty:
                filtered_ledger = base_ledger.iloc[:0].copy()
            self.check('whole_original_episode_rejection:' + cid,
                       set(filtered_ledger.entry_idx).issubset(set(base_ledger.entry_idx))
                       and np.isin(executed, [-1, 0, 1]).all()
                       and np.all((executed == 0) | (executed == base_exec)))
            for phase, start, end in periods:
                for variant in ('BASE', 'FILTER'):
                    calculated = dict(windows[key][phase]) if variant == 'BASE' else window_from_path(dates, executed, held, gross, filtered_ledger, start, end)
                    calculated['base_closed_n'] = windows[key][phase]['closed_n']
                    calculated['retention'] = calculated['closed_n'] / calculated['base_closed_n'] if calculated['base_closed_n'] else np.nan
                    saved_row = metric_index.loc[cid, variant, phase]
                    failed = [n for n, v in calculated.items() if
                              (pd.Timestamp(saved_row[n]) != v if n.endswith('_date') else not same_number(saved_row[n], v))]
                    self.check('independent_IS_window:' + cid + ':' + variant + ':' + phase, not failed, failed_fields=failed)
                    independent_metrics.append({**{n: h[n] for n in ('candidate_id', 'pair', 'strategy', 'filter_id')},
                                                'is_new': True, 'variant': variant, 'split': phase, **calculated})
            if cid in frozen_ids:
                mask = (dates >= pd.Timestamp(protocol['is_start'])) & (dates <= pd.Timestamp(protocol['is_end']))
                for variant, e, held_v, g, trade_table in [('BASE', base_exec, base_held, base_returns, base_ledger),
                                                         ('FILTER', executed, held, gross, filtered_ledger)]:
                    frozen_daily.append(pd.DataFrame({'candidate_id': cid, 'variant': variant, 'split': 'IS',
                        'date': dates[mask], 'executed': e[mask], 'held': held_v[mask],
                        'gross_daily_return': g[mask], 'net_daily_return': g[mask]}))
                    frozen_ledgers.append(trade_table.assign(candidate_id=cid, variant=variant))
            if (number + 1) % 50 == 0:
                print('Independent episode paths audited:', number + 1, '/', len(items), flush=True)
        rebuilt_metrics = pd.DataFrame(independent_metrics)
        independent_selection = screen_candidates(rebuilt_metrics)
        self.compare_frames('reconstructed_paths_SCREEN_rank_freeze', self.read_csv('is_selection.csv'), independent_selection, ['candidate_id'])
        daily, frozen_metrics = self.read_csv('is_frozen_daily.csv.gz'), self.read_csv('is_frozen_metrics.csv')
        if frozen_ids:
            self.compare_frames('frozen_daily_from_independent_paths', daily, pd.concat(frozen_daily, ignore_index=True), ['candidate_id', 'variant', 'date'])
            self.compare_frames('frozen_metrics_from_independent_paths', frozen_metrics,
                                rebuilt_metrics.loc[rebuilt_metrics.candidate_id.isin(frozen_ids) & rebuilt_metrics.split.eq('IS')],
                                ['candidate_id', 'variant', 'split'])
            self.compare_frames('frozen_ledger_from_independent_paths', self.read_csv('is_frozen_trades.csv.gz'),
                                pd.concat(frozen_ledgers, ignore_index=True), ['candidate_id', 'variant', 'entry_idx'])
            self.compare_frames('frozen_decisions_exact_subset', self.read_csv('is_frozen_decisions.csv.gz'),
                                decisions.loc[decisions.candidate_id.isin(frozen_ids)], ['candidate_id', 'signal_idx'])
        else:
            self.check('zero_freeze_no_fabricated_paths', daily.empty and frozen_metrics.empty
                       and not (self.campaign / 'is_frozen_trades.csv.gz').exists()
                       and not (self.campaign / 'is_frozen_decisions.csv.gz').exists())
        signatures = {h['candidate_id']: tuple(decisions.loc[decisions.candidate_id.eq(h['candidate_id']) & decisions.rejected.map(truth), 'signal_idx'].astype(int))
                      for h in items}
        unique_paths = len({(h['pair'], h['strategy'], signatures[h['candidate_id']]) for h in items})
        self.details.update(executed_BASE_accounts_reconstructed=180, original_episode_decisions_checked=decisions_checked,
                            rejected_episode_decisions=rejected_count, independent_metric_windows=len(rebuilt_metrics),
                            unique_account_rejection_paths_including_pending=unique_paths,
                            duplicate_paths_are_not_independent_support=True,
                            pending_terminal_signal_limit={'candidate_rows': len(terminal), 'rows': terminal,
                                'scope': 'The last unexecuted original signal is absent from the executed BASE ledger. Its saved feature values, codes, AND truth and pending status are crosschecked internally; no claim is made to independently recover that original signal or macro value. It creates no executed trade or realized return.'},
                            feature_validation_scope='All executed signal states match the immutable IS entry cache. Technical selected features also recomputed from IS-only source prices; macro direction/sign and strictly prior provenance checked against the cache. No new macro-data extraction is claimed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/entry_interaction_filter_v5')
    args = parser.parse_args()
    audit = InteractionAudit(args.output.resolve()); started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_state_filter_research.py', ROOT / 'verify_state_filter_supplement.py'):
        audit.register(source)
    try:
        audit.run()
    except Exception:
        audit.check('audit_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(r['passed'] for r in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
              'checks_total': len(audit.checks), 'checks_passed': passed, 'checks': audit.checks,
              'verified_files_sha256': audit.files, 'details': audit.details}
    destination = audit.campaign / 'independent_validation.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(destination), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
