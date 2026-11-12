"""Descriptive D-only entry interactions from the existing immutable IS cache.

Three predeclared feature pairs, all 16 known-state cells, all 180 accounts.
This computes trade-cohort observations, never a filtered path or OOS result.
Future/nonclosed/cross-D outcomes are skipped before numeric return parsing.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from fx_baselines import CURRENCIES, STRATEGIES
from fx_is_states import STATE_BINS, MACRO_PROVENANCE

ALLOWED_INTERACTIONS = (
    ('efficiency_volatility', 'efficiency_63', 'vol_ratio_20_63'),
    ('efficiency_directional_stretch', 'efficiency_63', 'direction_return_20_vol'),
    ('dollar_carry_conflict', 'direction_dxy_d20_pct', 'direction_spread_bp'),
)
BLOCKS = (('D1', '2005-01-03', '2009-12-31'), ('D2', '2010-01-01', '2014-12-31'),
          ('D', '2005-01-03', '2014-12-31'))
PARTITION_GROUPS = ('joint', 'A_only', 'B_only', 'neither', 'UNKNOWN')
COMPARISON_GROUPS = ('A_all', 'B_all', 'retained')
SOURCE = ROOT / 'reproduced/filter_is_diagnostics_v2'
TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_manifest(values):
    for name, digest in values.items():
        if sha(ROOT / name) != digest:
            raise AssertionError('Immutable source/input changed: ' + name)


def source_chain():
    original = json.loads((SOURCE / 'entry_state_protocol.json').read_text(encoding='utf-8'))
    validation = json.loads((SOURCE / 'entry_state_validation.json').read_text(encoding='utf-8'))
    amendment = json.loads((SOURCE / 'technical_validation_amendment.json').read_text(encoding='utf-8'))
    if original['compute_cutoff'] != '2023-05-29' or original['oos_computed'] or original['cost_bp'] != 0:
        raise AssertionError('Unexpected source observation scope.')
    if validation['status'] != 'passed' or amendment['status'] != 'passed':
        raise AssertionError('Source observation validation/amendment failed.')
    if amendment['original_protocol_sha256'] != sha(SOURCE / 'entry_state_protocol.json'):
        raise AssertionError('Source protocol differs from its defensive amendment.')
    if original['source_sha256']['src/fx_is_states.py'] != amendment['original_source_sha256']:
        raise AssertionError('Original state-source chain differs.')
    if sha(ROOT / amendment['original_source_snapshot']) != amendment['original_source_sha256']:
        raise AssertionError('Initial state-source archive differs.')
    source = dict(original['source_sha256'])
    source['src/fx_is_states.py'] = amendment['amended_source_sha256']
    check_manifest(source); check_manifest(original['input_sha256']); check_manifest(amendment['preserved_files_sha256'])
    for name, digest in validation['output_sha256'].items():
        if sha(SOURCE / name) != digest:
            raise AssertionError('Source cache changed: ' + name)
    inputs = dict(original['input_sha256'])
    for name in ('entry_state_protocol.json', 'entry_state_validation.json', 'technical_validation_amendment.json',
                 'is_entry_state_ledger.csv.gz', 'is_entry_state_summary.csv', 'is_entry_cohort_coverage.csv',
                 'source_snapshots/fx_is_states_initial.py'):
        path = SOURCE / name
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    source['build_entry_interaction_observations.py'] = sha(__file__)
    return source, inputs


def quality(values):
    values = np.asarray(values, float)
    wins, losses = values[values > TOL], values[values < -TOL]
    worst = np.sort(losses)
    total = float(values.sum())
    return {'closed_n': len(values), 'win_n': len(wins), 'loss_n': len(losses),
            'flat_n': len(values) - len(wins) - len(losses),
            'win_rate': len(wins) / len(values) if len(values) else np.nan,
            'payoff_ratio': float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
            'mean_win_pct': float(wins.mean()) if len(wins) else np.nan,
            'mean_loss_abs_pct': float(-losses.mean()) if len(losses) else np.nan,
            'expectancy_pct': float(values.mean()) if len(values) else np.nan,
            'return_sum_pct': total,
            'return_without_worst1_pct': total - float(worst[:1].sum()),
            'return_without_worst3_pct': total - float(worst[:3].sum()),
            'largest_loss_pct': float(worst[0]) if len(worst) else np.nan,
            'largest_loss_share': float(worst[0] / losses.sum()) if len(worst) else np.nan,
            'low_sample_flag': len(values) < 20,
            'low_outcome_sample_flag': len(wins) < 20 or len(losses) < 20}


def load_discovery():
    source_columns = sorted(set(itertools.chain.from_iterable(MACRO_PROVENANCE.values())))
    numeric = ['position', 'signal_idx', 'entry_idx', 'exit_idx', 'holding_observations',
               'gross_simple_return_pct', 'transaction_cost_usd', *STATE_BINS, *source_columns]
    strings = ['pair', 'strategy', 'family', 'trade_id', 'signal_date', 'entry_date', 'exit_date',
               *MACRO_PROVENANCE]
    counts = {'source_rows': 0, 'late_entry_rows_skipped_before_return_parse': 0,
              'early_entry_rows_skipped_before_return_parse': 0,
              'nonclosed_or_cross_D_rows_skipped_before_return_parse': 0, 'decoded_D_closed_rows': 0}
    rows = []
    with gzip.open(SOURCE / 'is_entry_state_ledger.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            counts['source_rows'] += 1
            if row['entry_date'] > '2014-12-31':
                counts['late_entry_rows_skipped_before_return_parse'] += 1
                continue
            if row['entry_date'] < '2005-01-03':
                counts['early_entry_rows_skipped_before_return_parse'] += 1
                continue
            if row['eligible_closed'] != 'True' or row['closed'] != 'True' or row['exit_date'] > '2014-12-31':
                counts['nonclosed_or_cross_D_rows_skipped_before_return_parse'] += 1
                continue
            record = {name: row[name] for name in strings}
            record.update({name: float(row[name]) if row[name] else np.nan for name in numeric})
            record.update({name + '_state': int(row[name + '_state']) for name in STATE_BINS})
            rows.append(record)
            counts['decoded_D_closed_rows'] += 1
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.gross_simple_return_pct).all() or not frame.transaction_cost_usd.eq(0).all():
        raise AssertionError('Nonfinite outcome or nonzero costs in discovery.')
    if set(zip(frame.pair, frame.strategy)) != set(itertools.product(CURRENCIES, STRATEGIES)):
        raise AssertionError('The discovery cache does not cover all 180 accounts.')
    if frame.duplicated(['pair', 'strategy', 'trade_id']).any():
        raise AssertionError('Duplicated original BASE episode.')
    signal = pd.to_datetime(frame.signal_date)
    if not ((signal < pd.to_datetime(frame.entry_date)) &
            (pd.to_datetime(frame.entry_date) < pd.to_datetime(frame.exit_date))).all():
        raise AssertionError('Entry signal/fill/true close chronology differs.')
    for feature, edges in STATE_BINS.items():
        values = frame[feature].to_numpy(float)
        expected = np.searchsorted(edges, values, side='right')
        expected[~np.isfinite(values)] = -1
        if not np.array_equal(expected, frame[feature + '_state'].to_numpy()):
            raise AssertionError('Fixed state does not match its signal feature: ' + feature)
    for observation, sources in MACRO_PROVENANCE.items():
        known = np.zeros(len(frame), bool)
        for feature in sources:
            known |= np.isfinite(frame[feature].to_numpy(float))
        age = signal - pd.to_datetime(frame[observation])
        if not (age[known].notna() & age[known].gt(pd.Timedelta(0)) & age[known].le(pd.Timedelta(days=7))).all():
            raise AssertionError('Finite macro value lacks strictly prior fresh provenance: ' + observation)
    counts['fixed_state_cells_validated'] = len(frame) * len(STATE_BINS)
    counts['macro_strict_prior_validated'] = True
    return frame, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/entry_interaction_observations_v5')
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve existing observation evidence; choose a new empty directory.')
    sources, inputs = source_chain()
    protocol = {'version': 'entry_interaction_observations_v5', 'created_utc': now(),
                'compute_cutoff': '2014-12-31', 'source_cache_cutoff': '2023-05-29', 'cost_bp': 0,
                'accounts': 180, 'pairs': list(CURRENCIES), 'strategies': list(STRATEGIES),
                'allowed_interactions': [{'interaction_id': i, 'feature_a': a, 'feature_b': b}
                                         for i, a, b in ALLOWED_INTERACTIONS],
                'state_bins': STATE_BINS, 'allowed_states': [0, 1, 2, 3], 'blocks': BLOCKS,
                'maximum_features_per_interaction': 2,
                'clock': 'Signal-close prices and already-earned BASE PnL; macro observation strictly earlier than signal and no older than seven calendar days. Original entry fills next quote close.',
                'cohort': 'Original BASE entry and eligible true close both inside each block. Cross-D1/D2 trades appear in combined D only. Open and cross-D-end rows are never decoded as numerical outcomes.',
                'outcome_parse_guard': 'Skip entries after 2014-12-31, pre-D entries, nonclosed entries and exits after D before float conversion of any return field. No later outcome numbers are parsed.',
                'partition_groups': {'joint': 'A==a AND B==b', 'A_only': 'A==a AND B known but not b',
                    'B_only': 'B==b AND A known but not a', 'neither': 'Both known and neither selected state',
                    'UNKNOWN': 'Either feature unknown; always retained by a potential joint rule'},
                'comparison_groups': {'A_all': 'All A==a, including B unknown', 'B_all': 'All B==b, including A unknown',
                                      'retained': 'Every original episode outside the known joint cell, including UNKNOWN'},
                'overlap': 'Five partition groups disjoint within a cell. Comparison groups overlap them. Cells from different feature pairs and accounts can overlap; these are not independent samples.',
                'proposal_support': {'D1_min_joint_closed_n': 10, 'D2_min_joint_closed_n': 10,
                    'D1_D2_joint_return_sum_pct_each_lt': -TOL, 'D_joint_return_sum_pct_lt': -TOL,
                    'concentration_and_complement_differences': 'Diagnostics only, never additional gates'},
                'mechanisms': {'efficiency_volatility': 'Distinguish path directionality from recent volatility acceleration; choppy expanding ranges may impair trend/breakout entries.',
                    'efficiency_directional_stretch': 'Distinguish a strong opposite-direction move on an efficient path from a noisy pullback; relevant to mean reversion versus momentum.',
                    'dollar_carry_conflict': 'Distinguish broad dollar conflict from directional interest-spread background. Positive direction_DXY is adverse to the foreign-currency position. No carry return is credited.'},
                'multiplicity_and_status': 'Exploratory reuse of previously observed IS. Exactly three feature pairs, all 16 cells each, no feature-set expansion from outcomes, no p-values or causal claim, no filtered paths. Prior three-year holdout has historical exposure and is not accessed here.',
                'oos_computed': False, 'source_sha256': sources, 'input_sha256': inputs}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'interaction_observation_protocol.json', protocol)
    frame, counts = load_discovery()
    frame.to_csv(output / 'discovery_closed_entry_ledger.csv.gz', index=False)
    summary, comparisons, coverage, support = [], [], [], []
    partitions_checked = 0
    for pair in CURRENCIES:
        for strategy in STRATEGIES:
            account = frame.loc[frame.pair.eq(pair) & frame.strategy.eq(strategy)]
            family = account.family.iloc[0]
            by_block = {block: account.loc[account.entry_date.ge(start) & account.exit_date.le(end)]
                        for block, start, end in BLOCKS}
            for block, cohort in by_block.items():
                coverage.append({'pair': pair, 'strategy': strategy, 'family': family, 'block': block,
                                 **quality(cohort.gross_simple_return_pct)})
            for interaction, feature_a, feature_b in ALLOWED_INTERACTIONS:
                for state_a, state_b in itertools.product(range(4), repeat=2):
                    key = {'pair': pair, 'strategy': strategy, 'family': family, 'interaction_id': interaction,
                           'feature_a': feature_a, 'state_a': state_a, 'feature_b': feature_b, 'state_b': state_b}
                    cell = {}
                    for block, cohort in by_block.items():
                        aa = cohort[feature_a + '_state'].to_numpy(); bb = cohort[feature_b + '_state'].to_numpy()
                        values = cohort.gross_simple_return_pct.to_numpy()
                        known = (aa >= 0) & (bb >= 0); a = aa == state_a; b = bb == state_b
                        groups = {'joint': known & a & b, 'A_only': known & a & ~b, 'B_only': known & ~a & b,
                                  'neither': known & ~a & ~b, 'UNKNOWN': ~known}
                        if not np.array_equal(sum(mask.astype(int) for mask in groups.values()), np.ones(len(cohort), int)):
                            raise AssertionError('Partition lost or double-counted a trade.')
                        if not np.isclose(sum(values[mask].sum() for mask in groups.values()), values.sum(), atol=1e-10, rtol=1e-12):
                            raise AssertionError('Partition PnL does not reconcile to BASE cohort.')
                        partitions_checked += 1
                        cell[block] = quality(values[groups['joint']])
                        for group, mask in groups.items():
                            summary.append({**key, 'block': block, 'group': group, 'cohort_n': len(cohort),
                                            'known_features_n': int(known.sum()), **quality(values[mask])})
                        for group, mask in {'A_all': a, 'B_all': b, 'retained': ~groups['joint']}.items():
                            comparisons.append({**key, 'block': block, 'group': group, 'cohort_n': len(cohort),
                                                'known_features_n': int(known.sum()), **quality(values[mask])})
                    repeated = all(cell[b]['closed_n'] >= 10 and cell[b]['return_sum_pct'] < -TOL for b in ('D1', 'D2'))
                    repeated = bool(repeated and cell['D']['return_sum_pct'] < -TOL)
                    support.append({**key, 'D_repeated_negative_support10': repeated,
                                    'D_repeated_negative_support20': bool(repeated and all(cell[b]['closed_n'] >= 20 for b in ('D1', 'D2'))),
                                    **{f'{block}_{name}': values[name] for block, values in cell.items()
                                       for name in ('closed_n', 'return_sum_pct', 'win_rate', 'payoff_ratio', 'expectancy_pct',
                                                    'return_without_worst1_pct', 'return_without_worst3_pct')}})
        print('D interaction observations:', pair, flush=True)
    pd.DataFrame(summary).to_csv(output / 'interaction_state_summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'interaction_comparison_summary.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'interaction_account_coverage.csv', index=False)
    support = pd.DataFrame(support)
    support.to_csv(output / 'interaction_support_diagnostics.csv', index=False)
    check_manifest(sources); check_manifest(inputs)
    validation = {'status': 'passed', 'finished_utc': now(), 'accounts': 180,
                  'compute_cutoff': '2014-12-31', 'source_cache_cutoff': '2023-05-29', 'oos_computed': False,
                  'protocol_sha256': sha(output / 'interaction_observation_protocol.json'),
                  'source_integrity_pass': True, 'later_outcome_numbers_decoded': 0, 'outcome_parse_counts': counts,
                  'state_cells': len(support), 'summary_rows': len(summary), 'comparison_rows': len(comparisons),
                  'coverage_rows': len(coverage), 'partition_count_and_PnL_checks': partitions_checked,
                  'support10_cells': int(support.D_repeated_negative_support10.sum()),
                  'support20_cells': int(support.D_repeated_negative_support20.sum()),
                  'support10_by_interaction': support.groupby('interaction_id').D_repeated_negative_support10.sum().astype(int).to_dict(),
                  'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir())
                                    if p.is_file() and p.name != 'interaction_observation_protocol.json'}}
    save_json(output / 'interaction_observation_validation.json', validation)
    print('Complete:', len(frame), 'D closed episodes;', validation['support10_cells'], 'descriptive supported cells.', flush=True)


if __name__ == '__main__':
    main()
