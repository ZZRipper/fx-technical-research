"""Build observation-first diagnostics for all 20 fixed rules and nine currencies.

python run_observation_research.py
python run_observation_research.py --report-only

Outputs are isolated under reproduced/observation_v1. No original research
account, rule, filter, dashboard, or input file is modified.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'reproduced' / 'observation_v1' / '.matplotlib'))

import numpy as np
import pandas as pd

from fx_baselines import CURRENCIES, STRATEGIES, canonical_prices_from_raw, reconstruct_fixed_baselines
from fx_economics import read_native_features_from_raw, rebuild_economic_known
from fx_evidence import verify_baselines, verify_known_features
from fx_observation import analyze_baselines
from fx_observation_features import build_entry_features, FEATURE_COLUMNS


FILES = {
    'daily': 'daily_accounts.csv.gz', 'accounts': 'account_summary.csv',
    'periods': 'calendar_periods.csv', 'rolling': 'rolling_windows.csv',
    'drawdowns': 'drawdown_episodes.csv', 'trades': 'enriched_trade_ledger.csv.gz',
    'attribution': 'window_trade_attribution.csv.gz', 'rule_summary': 'rule_summary.csv',
    'contrasts': 'entry_feature_contrasts.csv', 'coverage': 'entry_feature_coverage.csv',
}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    return {
        'version': 'observation_v1', 'user_scope': 'All 20 technical rules across all nine currencies',
        'pairs': list(CURRENCIES), 'strategies': STRATEGIES,
        'account_count': 180, 'sample_start': '2005-01-03', 'sample_end': '2026-05-29',
        'primary_accounting': 'fixed_initial_capital_simple', 'initial_capital_usd': 100000,
        'reference_accounting': 'compound_with_identical_positions',
        'costs_carry_financing_interest': 0,
        'calendar_periods': ['year', 'half_year'],
        'rolling_window': 252,
        'window_selection': 'Greedy descending absolute simple return; no overlaps within an account; at most three positive and three negative complete windows. Ex-post labels only.',
        'window_attribution': 'Clip every overlapping original trade to actual earned-return days; includes carry-in and open trades; sum equals window daily PnL.',
        'drawdown': 'Fixed-capital pp from most recent high plateau to first recovery; unrecovered tail retained.',
        'excursions': 'MFE/MAE on close-observed strategy-return path, with entry zero; no intraday extremes.',
        'cohort_policy': 'Entry in period and eligible close by period end; crossing/open trades separately counted. Calendar PnL includes all positions.',
        'feature_policy': 'Price features at signal close; macro observations strictly before signal date, at most seven calendar days old.',
        'entry_features': list(FEATURE_COLUMNS),
        'feature_comparison': 'WIN versus LOSS mean, median and pooled within-group standardized mean difference; display low sample when either group has fewer than 20 observations.',
        'evidence_status': 'retrospective exploratory observation',
        'new_untouched_out_of_sample': False, 'filter_optimization_performed': False,
        'notes': ['No new entry or exit rule is implemented by these diagnostics.',
                  'No p-values, fitted cutoffs or performance-optimized regime labels.',
                  'The price efficiency diagnostic uses absolute price changes, whereas the existing REGIME_RULE_FIXED signal uses log-price efficiency.',
                  '2026 is a partial year; H.10 is outside this main-history observation scope.'],
    }


def _load_tables(output):
    tables = {key: pd.read_csv(output / name) for key, name in FILES.items()}
    for frame in tables.values():
        for column in frame:
            if column == 'date' or column.endswith('_date') or column in ('sample_start', 'sample_end'):
                frame[column] = pd.to_datetime(frame[column])
    return tables


def _render(output, tables):
    from fx_observation_report import render_report
    hypotheses_path = output / 'hypotheses.json'
    hypotheses = json.loads(hypotheses_path.read_text(encoding='utf-8')) if hypotheses_path.exists() else []
    metadata = {
        'built_utc': datetime.now(timezone.utc).isoformat(), 'hypotheses': hypotheses,
        'csv_files': {**{name: name for name in FILES.values()},
                      '验证记录': 'validation_summary.json', '观察定义': 'observation_protocol.json',
                      **({'候选假设与反例': 'hypotheses.json'} if hypotheses_path.exists() else {})},
    }
    print('Rendering 20 strategy figures, PDF atlas and Chinese report...', flush=True)
    rendered = render_report(output, *(tables[k] for k in ('daily', 'accounts', 'periods', 'rolling', 'drawdowns', 'trades', 'contrasts')), metadata)
    _write_json(output / 'report_build.json', rendered)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced' / 'observation_v1')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        _render(output, _load_tables(output))
        return 0
    summary = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'new_untouched_out_of_sample': False, 'filter_optimization_performed': False}
    _write_json(output / 'validation_summary.json', summary)
    _write_json(output / 'observation_protocol.json', protocol())
    try:
        print('Reading original price workbook and rebuilding fixed baselines...', flush=True)
        prices, _ = canonical_prices_from_raw()
        baseline = reconstruct_fixed_baselines(prices)
        checks, rows = verify_baselines(baseline[0], baseline[2], baseline[3])
        checks.to_csv(output / 'baseline_reconciliation.csv', index=False)
        if not checks['pass'].all():
            raise AssertionError('Baseline reference reconciliation failed.')
        print(f'Baseline: {len(checks)} checks passed, {rows:,} daily observations reconciled.', flush=True)
        tables = analyze_baselines(baseline)
        print('Rebuilding strictly prior economic inputs from the feature workbook...', flush=True)
        known = rebuild_economic_known(read_native_features_from_raw(), prices.date)
        checks_known = verify_known_features(known)
        checks_known.to_csv(output / 'feature_reconciliation.csv', index=False)
        if not checks_known['pass'].all():
            raise AssertionError('Known feature reference reconciliation failed.')
        print('Comparing entry-known features for all WIN/LOSS/FLAT cohorts...', flush=True)
        tables['trades'], tables['contrasts'], tables['coverage'] = build_entry_features(prices, tables['trades'], known)
        if len(tables['coverage']) != 180 * 4 or len(tables['contrasts']) != 180 * 4 * len(FEATURE_COLUMNS):
            raise AssertionError('Incomplete entry feature account/cohort coverage.')
        print('Writing observation tables...', flush=True)
        for key, filename in FILES.items():
            tables[key].to_csv(output / filename, index=False)
        # Current simple outputs are independently generated by the existing engine.
        simple_reference_path = ROOT / 'reproduced/simple_interest/account_metrics.csv'
        simple_reference_checks = []
        if simple_reference_path.exists():
            reference = pd.read_csv(simple_reference_path)
            reference = reference.loc[reference.dataset.eq('main') & reference.variant.eq('BASE') & reference.strategy.isin(STRATEGIES)]
            comparison = tables['accounts'].merge(reference, on=['pair', 'strategy'], suffixes=('', '_reference'), validate='one_to_one')
            if len(comparison) != 180 or len(reference) != 180:
                raise AssertionError('Stored simple baseline coverage differs.')
            for actual, expected in [('simple_return_pct', 'total_return_pct'),
                                     ('simple_max_drawdown_pp', 'max_drawdown_initial_capital_pp'),
                                     ('closed_trades', 'eligible_closed_n'),
                                     ('win_rate', 'win_rate_reference'), ('payoff_ratio', 'payoff_ratio_reference')]:
                passed = bool(np.allclose(comparison[actual], comparison[expected], atol=1e-8, rtol=0, equal_nan=True))
                simple_reference_checks.append({'metric': actual, 'accounts': len(comparison), 'pass': passed,
                                                'max_abs_error': float((comparison[actual] - comparison[expected]).abs().max())})
            pd.DataFrame(simple_reference_checks).to_csv(output / 'simple_baseline_reconciliation.csv', index=False)
            if not all(row['pass'] for row in simple_reference_checks):
                raise AssertionError('Stored simple baseline statistics differ.')
        inputs = list((ROOT / 'src').glob('fx_*.py')) + [Path(__file__), ROOT / 'requirements.txt']
        inputs += list((ROOT / 'FX_Research_Datasets/data/raw').glob('*.xlsx'))
        inputs += [ROOT / 'FX_PnL_Explorer.html', ROOT / 'FX_Research_Datasets/data/clean/fixed_baseline_metrics_rebuilt.csv',
                   ROOT / 'FX_Research_Datasets/data/clean/economic_known_inputs_rebuilt.csv.gz']
        if simple_reference_path.exists():
            inputs.append(simple_reference_path)
        summary.update({
            'status': 'passed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'accounts': len(tables['accounts']), 'strategies': tables['accounts'].strategy.nunique(),
            'currencies': tables['accounts'].pair.nunique(),
            'daily_observations': len(tables['daily']), 'all_trade_episodes': len(tables['trades']),
            'eligible_closed_trades': int(tables['trades'].eligible_closed.sum()),
            'baseline_reconciliation_checks': len(checks), 'baseline_checks_pass': True,
            'baseline_daily_records_reconciled': rows,
            'economic_feature_checks': len(checks_known), 'economic_feature_checks_pass': True,
            'calendar_additivity_and_compounding_pass': True, 'trade_partition_and_excursions_pass': True,
            'drawdown_maximum_reconciliation_pass': True, 'window_trade_attribution_pass': True,
            'entry_timing_checks_pass': True,
            'stored_simple_baseline_checks': simple_reference_checks,
            'table_rows': {key: len(value) for key, value in tables.items()},
            'source_sha256': {p.relative_to(ROOT).as_posix(): _hash(p) for p in inputs},
            'output_sha256': {name: _hash(output / name) for name in FILES.values()},
        })
        _write_json(output / 'validation_summary.json', summary)
        if not args.no_report:
            _render(output, tables)
        print(f'Observation research complete: {output}', flush=True)
        return 0
    except Exception as error:
        summary.update(status='failed', error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output / 'validation_summary.json', summary)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
