"""Independently verify the immutable D-only long/short observation probe.

Only complete original BASE outcomes with entry and true close inside D are
converted to numbers. D1/D2 censor cross-boundary trades; combined D retains
them. No strategy, filtered path, later IS outcome or OOS result is evaluated.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import gzip
import itertools
import json
import traceback

import numpy as np
import pandas as pd

from build_entry_interaction_observations import ROOT, SOURCE, source_chain, sha, check_manifest
from fx_baselines import CURRENCIES, STRATEGIES

PROBE = ROOT / 'reproduced/direction_probe_v5'
BLOCKS = (('D1', '2005-01-03', '2009-12-31'), ('D2', '2010-01-01', '2014-12-31'),
          ('D', '2005-01-03', '2014-12-31'))
TOL = 1e-10


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def quality(values):
    values = np.asarray(values, float)
    wins, losses = values[values > TOL], values[values < -TOL]
    total = float(values.sum())
    return {'closed_n': len(values), 'win_n': len(wins), 'loss_n': len(losses),
            'flat_n': len(values) - len(wins) - len(losses), 'return_sum_pct': total,
            'expectancy_pct': float(values.mean()) if len(values) else np.nan,
            'win_rate': len(wins) / len(values) if len(values) else np.nan,
            'payoff_ratio': float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
            'mean_win_pct': float(wins.mean()) if len(wins) else np.nan,
            'mean_loss_abs_pct': float(-losses.mean()) if len(losses) else np.nan,
            # Preserve the probe's literal diagnostic: remove the minimum
            # trade, including a positive minimum in an all-winning cohort.
            'return_without_worst_trade_pct': total - float(values.min()) if len(values) else np.nan}


def decode_D_only():
    rows, audit = [], {'source_rows': 0, 'D_entry_metadata_rows': 0, 'late_entry_rows_skipped_before_return_parse': 0,
                       'early_entry_rows_skipped_before_return_parse': 0,
                       'ineligible_or_cross_D_rows_skipped_before_return_parse': 0, 'D_outcomes_decoded': 0}
    with gzip.open(SOURCE / 'is_entry_state_ledger.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            audit['source_rows'] += 1
            if row['entry_date'] > '2014-12-31':
                audit['late_entry_rows_skipped_before_return_parse'] += 1
                continue
            if row['entry_date'] < '2005-01-03':
                audit['early_entry_rows_skipped_before_return_parse'] += 1
                continue
            audit['D_entry_metadata_rows'] += 1
            # Metadata of censored entries is needed for entered_n, but their
            # outcome is never numerically decoded or included in any quality.
            record = {name: row[name] for name in ('pair', 'strategy', 'family', 'trade_id', 'signal_date', 'entry_date', 'exit_date')}
            record['position'] = int(float(row['position']))
            record['eligible_D'] = (row['eligible_closed'] == 'True' and row['closed'] == 'True'
                                    and row['entry_date'] < row['exit_date'] <= '2014-12-31')
            if record['position'] not in (-1, 1):
                raise AssertionError('Unknown signed direction.')
            if record['eligible_D']:
                record['gross_simple_return_pct'] = float(row['gross_simple_return_pct'])
                if not np.isfinite(record['gross_simple_return_pct']):
                    raise AssertionError('Nonfinite complete-D outcome.')
                if float(row['transaction_cost_usd']) != 0:
                    raise AssertionError('Discovery source contains costs.')
                audit['D_outcomes_decoded'] += 1
            else:
                record['gross_simple_return_pct'] = np.nan
                audit['ineligible_or_cross_D_rows_skipped_before_return_parse'] += 1
            rows.append(record)
    return pd.DataFrame(rows), audit


def independently_summarize(frame):
    rows, comparisons = [], []
    for pair, strategy in itertools.product(CURRENCIES, STRATEGIES):
        account = frame.loc[frame.pair.eq(pair) & frame.strategy.eq(strategy)]
        family = account.family.iloc[0]
        for position in (-1, 1):
            directional = account.loc[account.position.eq(position)]
            identity = {'pair': pair, 'strategy': strategy, 'family': family, 'position': position,
                        'direction': 'LONG_FOREIGN_SHORT_USD' if position == 1 else 'SHORT_FOREIGN_LONG_USD'}
            blocks = {}
            for block, start, end in BLOCKS:
                entered = directional.loc[directional.entry_date.between(start, end)]
                closed = entered.loc[entered.eligible_D & entered.exit_date.le(end)]
                stats = quality(closed.gross_simple_return_pct)
                blocks[block] = {'entered_n': len(entered), **stats,
                                 'censored_or_ineligible_entry_n': len(entered) - stats['closed_n']}
                rows.append({**identity, 'block': block, **blocks[block]})
            cross = directional.loc[directional.eligible_D & directional.entry_date.le('2009-12-31')
                                    & directional.exit_date.ge('2010-01-01')]
            selected_fields = ('entered_n', 'closed_n', 'censored_or_ineligible_entry_n', 'win_n', 'loss_n',
                               'return_sum_pct', 'expectancy_pct', 'win_rate', 'payoff_ratio', 'return_without_worst_trade_pct')
            fields = {block.lower() + '_' + field: blocks[block][field]
                      for field in selected_fields for block in ('D', 'D1', 'D2')}
            negative = bool(all(blocks[b]['return_sum_pct'] < -TOL for b in ('D1', 'D2')))
            supported = bool(all(blocks[b]['closed_n'] >= 10 for b in ('D1', 'D2')))
            comparisons.append({**identity, **fields,
                'negative_in_both_discovery_blocks': negative,
                'each_discovery_block_at_least_10_closed': supported,
                'repeated_negative_with_descriptive_sample_support': negative and supported,
                'combined_D_negative': blocks['D']['return_sum_pct'] < -TOL,
                'combined_D_negative_without_worst_trade': blocks['D']['return_sum_pct'] < -TOL
                    and blocks['D']['return_without_worst_trade_pct'] < -TOL,
                'cross_D1_D2_closed_n': len(cross),
                'cross_D1_D2_return_sum_pct': float(cross.gross_simple_return_pct.sum())})
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


class Verification:
    def __init__(self):
        self.checks = []
        self.max_differences = {}

    def check(self, name, passed, **detail):
        self.checks.append({'name': name, 'passed': bool(passed), **detail})
        if not passed:
            print('FAILED', name, detail, flush=True)

    def compare(self, name, saved, rebuilt, keys):
        a, b = saved.set_index(keys).sort_index(), rebuilt.set_index(keys).sort_index()
        self.check(name + ':schema', set(a) == set(b))
        self.check(name + ':identities', a.index.equals(b.index) and not a.index.duplicated().any())
        if not a.index.equals(b.index):
            return
        biggest = 0.
        for key in a.index:
            failed = []
            for column in a:
                first, second = a.loc[key, column], b.loc[key, column]
                if pd.api.types.is_numeric_dtype(a[column]):
                    good = bool(np.isclose(first, second, atol=1e-10, rtol=1e-12, equal_nan=True))
                    if np.isfinite(first) and np.isfinite(second):
                        biggest = max(biggest, abs(float(first) - float(second)))
                else:
                    good = first == second or pd.isna(first) and pd.isna(second)
                if not good:
                    failed.append({'field': column, 'saved': first, 'independent': second})
            self.check(name + ':' + '|'.join(map(str, key)), not failed, differences=failed)
        self.max_differences[name] = biggest


def describe_cases(frame, summary, comparison):
    examples = [
        ('EUR', 'BB_MR_10', 'repeated_D_direction_contrast',
         'Long-foreign losses and short-foreign gains repeat in both discovery halves. This is a descriptive direction contrast, not a claim that a permanent USD preference will persist.'),
        ('NZD', 'RSI_MOM_20', 'repeated_D_direction_contrast',
         'Long-foreign losses and short-foreign gains repeat in both discovery halves, with at least 39 long entries per half. Removing the largest D loss is a concentration diagnostic, not a selection rule.'),
        ('JPY', 'RSI_MOM_7', 'direction_sign_reversal_counterexample',
         'Long-foreign is profitable in D1 and loses in D2; short-foreign changes the other way. Historical direction averages can reflect regime differences and should not be treated as a timeless directional mechanism.'),
    ]
    cases = []
    for pair, strategy, kind, explanation in examples:
        rows = summary.loc[summary.pair.eq(pair) & summary.strategy.eq(strategy)]
        original = frame.loc[frame.pair.eq(pair) & frame.strategy.eq(strategy) & frame.eligible_D]
        cases.append({'pair': pair, 'strategy': strategy, 'kind': kind, 'interpretation': explanation,
                      'blocks': rows.to_dict('records'),
                      'worst_three_D_original_episodes': original.sort_values('gross_simple_return_pct')
                          [['trade_id', 'signal_date', 'entry_date', 'exit_date', 'position', 'gross_simple_return_pct']].head(3).to_dict('records')})
    cross = frame.loc[frame.pair.eq('AUD') & frame.strategy.eq('TSMOM_63') & frame.position.eq(1)
                     & frame.eligible_D & frame.entry_date.le('2009-12-31') & frame.exit_date.ge('2010-01-01')]
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'D-only original complete BASE trades, zero cost',
            'compute_cutoff': '2014-12-31', 'oos_computed': False, 'cases': cases,
            'cross_boundary_counterexample': {
                'pair': 'AUD', 'strategy': 'TSMOM_63', 'position': 1,
                'fact': 'D1 and D2 each have at least ten closed trades and negative sums, yet combined D is positive because it includes a winning trade crossing 2009/2010. This direction must not be proposed under the D-negative requirement.',
                'blocks': summary.loc[summary.pair.eq('AUD') & summary.strategy.eq('TSMOM_63') & summary.position.eq(1)].to_dict('records'),
                'comparison': comparison.loc[comparison.pair.eq('AUD') & comparison.strategy.eq('TSMOM_63') & comparison.position.eq(1)].iloc[0].to_dict(),
                'crossing_episodes': cross[['trade_id', 'signal_date', 'entry_date', 'exit_date', 'position', 'gross_simple_return_pct']].to_dict('records')},
            'limits': ['Examples selected after descriptive discovery; they are not untouched validation.',
                       'Long/short cohorts are mutually exclusive within an account but share the same market history; accounts and rules are not independent experiments.',
                       'D includes boundary-spanning closed trades, so D totals need not equal D1 plus D2.',
                       'No direction filter or return curve is simulated here, and worst-trade removal is never an extra proposal gate.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/direction_observation_validation_v6')
    args = parser.parse_args(); output = args.output.resolve()
    if (output / 'validation.json').exists():
        if json.loads((output / 'validation.json').read_text(encoding='utf-8'))['status'] == 'passed':
            raise ValueError('Preserve a completed audit; use a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(); audit = Verification()
    sources, inputs, access, counts = {}, {}, {}, {}
    try:
        sources, inputs = source_chain()
        sources['verify_direction_observations.py'] = sha(__file__)
        for name in ('direction_probe.json', 'direction_summary.csv', 'direction_comparison.csv'):
            file = PROBE / name; inputs[file.relative_to(ROOT).as_posix()] = sha(file)
        probe = json.loads((PROBE / 'direction_probe.json').read_text(encoding='utf-8'))
        check_manifest(probe['source_sha256'])
        for name, digest in probe['output_sha256'].items():
            audit.check('probe_hash:' + name, sha(PROBE / name) == digest)
        audit.check('probe_zero_cost_and_D_only_metadata', probe['cost_bp'] == 0
                    and probe['access_audit']['allowed_outcome_end'] == '2014-12-31'
                    and probe['access_audit']['later_outcomes_read'] is False and probe['access_audit']['OOS_read'] is False)
        frame, access = decode_D_only()
        audit.check('all_180_accounts', set(zip(frame.pair, frame.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES)))
        audit.check('unique_original_episode', not frame.duplicated(['pair', 'strategy', 'trade_id']).any())
        audit.check('D_metadata_and_outcome_parse_counts', access['D_entry_metadata_rows'] == probe['access_audit']['D_entry_metadata_rows']
                    and access['D_outcomes_decoded'] == probe['access_audit']['parsed_eligible_closed_trade_outcomes']
                    and access['late_entry_rows_skipped_before_return_parse'] == probe['access_audit']['rows_skipped_before_outcome_conversion'])
        summary, comparison = independently_summarize(frame)
        old_summary = pd.read_csv(PROBE / 'direction_summary.csv', float_precision='round_trip')
        old_comparison = pd.read_csv(PROBE / 'direction_comparison.csv', float_precision='round_trip')
        audit.check('complete_summary_grid', len(summary) == 1080 and len(old_summary) == 1080)
        audit.check('complete_direction_grid', len(comparison) == 360 and len(old_comparison) == 360)
        audit.compare('summary', old_summary, summary, ['pair', 'strategy', 'position', 'block'])
        audit.compare('comparison', old_comparison, comparison, ['pair', 'strategy', 'position'])
        support = comparison.repeated_negative_with_descriptive_sample_support
        proposed = support & comparison.combined_D_negative
        counts = {'accounts': 180, 'account_direction_groups': len(comparison),
                  'D_eligible_closed_trades': int(summary.loc[summary.block.eq('D'), 'closed_n'].sum()),
                  'negative_in_both_subperiods_groups': int(comparison.negative_in_both_discovery_blocks.sum()),
                  'repeated_negative_with_at_least_10_per_subperiod_groups': int(support.sum()),
                  'distinct_accounts_with_supported_repeated_negative_direction': len(comparison.loc[support, ['pair', 'strategy']].drop_duplicates()),
                  'supported_groups_also_D_negative': int(proposed.sum()),
                  'supported_groups_also_D_negative_without_worst_trade': int((proposed & comparison.combined_D_negative_without_worst_trade).sum())}
        audit.check('all_original_probe_counts', counts == probe['counts'], independently_computed=counts)
        audit.check('proposal_count_97_without_new_concentration_gate', int(proposed.sum()) == 97)
        for row in comparison.itertuples(index=False):
            audit.check('D_boundary_partition:' + row.pair + '|' + row.strategy + '|' + str(row.position),
                        row.d_closed_n == row.d1_closed_n + row.d2_closed_n + row.cross_D1_D2_closed_n
                        and np.isclose(row.d_return_sum_pct, row.d1_return_sum_pct + row.d2_return_sum_pct + row.cross_D1_D2_return_sum_pct,
                                       atol=1e-10, rtol=1e-12))
        aud = comparison.loc[comparison.pair.eq('AUD') & comparison.strategy.eq('TSMOM_63') & comparison.position.eq(1)].iloc[0]
        audit.check('AUD_TSMOM63_cross_boundary_exception', aud.repeated_negative_with_descriptive_sample_support
                    and not aud.combined_D_negative and aud.cross_D1_D2_closed_n == 1 and aud.cross_D1_D2_return_sum_pct > 0)
        check_manifest(sources); check_manifest(inputs)
        audit.check('source_and_input_bytes_unchanged', True)
        summary.to_csv(output / 'recomputed_direction_summary.csv', index=False)
        comparison.to_csv(output / 'recomputed_direction_comparison.csv', index=False)
        cases = describe_cases(frame, summary, comparison)
        cases['source_sha256'] = {name: digest for name, digest in inputs.items()
                                  if name.startswith('reproduced/direction_probe_v5/') or name.endswith('is_entry_state_ledger.csv.gz')}
        save(output / 'cases.json', cases)
    except Exception:
        audit.check('verification_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(row['passed'] for row in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'accounts': 180,
              'compute_cutoff': '2014-12-31', 'source_cache_cutoff': '2023-05-29', 'oos_computed': False,
              'source_sha256': sources, 'input_sha256': inputs, 'access_audit': access,
              'later_outcome_numbers_decoded': 0, 'original_probe_modified': False,
              'numerical_tolerance': {'absolute': 1e-10, 'relative': 1e-12},
              'worst_trade_diagnostic_definition': 'Sum minus the minimum one trade, including a positive minimum; empty cohort is NaN. Diagnostic only, never an extra proposal gate.',
              'maximum_absolute_numeric_difference_by_table': audit.max_differences,
              'independent_counts': counts, 'checks_total': len(audit.checks), 'checks_passed': passed,
              'checks': audit.checks,
              'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'validation.json'}}
    save(output / 'validation.json', result)
    print(result['status'], passed, '/', len(audit.checks), 'max absolute numeric differences', audit.max_differences, flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
