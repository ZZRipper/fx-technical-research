"""IS-first filter campaign with a frozen, three-year historical holdout.

python -B run_filter_research.py --stage develop
python -B run_filter_research.py --stage oos
python -B run_filter_research.py --stage report

The develop process truncates prices and native features before computing any
strategy outcomes. The oos process accepts only the frozen IS-selected identities.
This is a historical isolation exercise: the holdout was previously observed.
"""

from __future__ import annotations

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
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'reproduced/filter_oos_v1/.matplotlib'))

import numpy as np
import pandas as pd

from fx_baselines import CURRENCIES, STRATEGIES, RULES, canonical_prices_from_raw, complete_signals, fixed_execute
from fx_economics import read_native_features_from_raw, rebuild_economic_known
from fx_observation_features import _price_features
from fx_filter_candidates import specs, run_candidate
from fx_filter_evaluation import (
    IS_BLOCKS, SCREEN, OOS_SCREEN, evaluate_window, simple_ledger, daily_account,
    screen_candidates, delayed_schedule, fresh_start,
)

IS_START, IS_END = '2005-01-03', '2023-05-29'
OOS_START, OOS_END = '2023-05-30', '2026-05-29'
COST_BP = 0.0
CONTROLS = {'RATE_USD_CONFLICT', 'COMMODITY_USD'}
OOS_YEARS = (
    ('OOS_YEAR_1', '2023-05-30', '2024-05-29'),
    ('OOS_YEAR_2', '2024-05-30', '2025-05-29'),
    ('OOS_YEAR_3', '2025-05-30', '2026-05-29'),
)
DECISION_SOURCES = [
    'run_filter_research.py', 'src/fx_filter_candidates.py', 'src/fx_filter_evaluation.py',
    'src/fx_baselines.py', 'src/fx_economics.py', 'src/fx_observation_features.py',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_hashes():
    return {name: sha(ROOT / name) for name in DECISION_SOURCES}


def input_hashes():
    data = ROOT / 'FX_Research_Datasets/data/raw'
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(data.glob('*.xlsx'))}


def protocol_document():
    return {
        'version': 'filter_oos_v1', 'created_utc': now(),
        'is_start': IS_START, 'is_end': IS_END, 'oos_start': OOS_START, 'oos_end': OOS_END,
        'oos_duration': 'Three calendar years immediately following IS, ending at the supplied workbook endpoint.',
        'scope': {'pairs': list(CURRENCIES), 'strategies': STRATEGIES, 'baseline_accounts': 180},
        'initial_capital_usd': 100000, 'accounting': 'fixed_initial_capital_simple',
        'cost_bp': COST_BP, 'reference_cost_bp': 0.0,
        'cost_instruction': 'User explicitly requested no costs. No cost sensitivity is used this round.',
        'carry_financing_cash_interest': 0.0,
        'prior_holdout_exposure': 'The 2023-2026 history was already plotted and inspected in earlier work. This is a development-isolated historical test, not virgin or newly collected OOS.',
        'development_access': 'After workbook parsing, price and native feature rows after IS_END are removed before indicator/return/feature calculations. No full-sample outcome artifact is used as a development input.',
        'screen': SCREEN, 'oos_screen': OOS_SCREEN, 'is_blocks': IS_BLOCKS, 'oos_anniversary_years': OOS_YEARS,
        'candidate_specs': specs(), 'legacy_comparator_ids': sorted(CONTROLS),
        'family_application': 'Each spec is applied to every compatible rule and currency; commodity comparator only AUD/CAD/NOK. No performance-dependent subset.',
        'boundary_policy': 'Main evaluation preserves continuous positions across IS/OOS; PnL display resets only. Quality uses entries and closes inside each interval. Carry-in and censored trades are reported separately.',
        'fresh_start_sensitivity': 'Zero prior positions; only original BASE episodes with execution entry on/after OOS_START may enter. No mid-episode re-entry.',
        'schedule_delay_sensitivity': 'IS-only diagnostic: lag the fixed target schedule by one additional quote for both BASE and FILTER. Decisions are not recomputed under delayed fills; this is not an operational latency simulator and not a selection gate.',
        'holdout_rule': 'Freeze one highest IS-ranked passing NEW filter per account, then evaluate every frozen candidate exactly as specified. No OOS-based re-ranking, rule revision or deletion of failures.',
        'source_sha256': source_hashes(), 'input_sha256': input_hashes(),
    }


def assert_hashes(expected):
    bad = [name for name, expected_hash in expected.items() if sha(ROOT / name) != expected_hash]
    if bad:
        raise ValueError('Frozen decision sources or inputs changed: ' + ', '.join(bad))


def prepare(end):
    print(f'Reading source workbooks; compute cutoff={end}...', flush=True)
    raw, _ = canonical_prices_from_raw()
    prices = raw.loc[raw.date.le(pd.Timestamp(end))].copy()
    del raw
    native = read_native_features_from_raw()
    native = {key: value.loc[value.index <= pd.Timestamp(end)].copy() for key, value in native.items()}
    if prices.date.max() > pd.Timestamp(end) or any(len(v) and v.index.max() > pd.Timestamp(end) for v in native.values()):
        raise AssertionError('Data cutoff violated before feature calculations.')
    price_features = _price_features(prices)
    known = rebuild_economic_known(native, prices.date)
    features = price_features.merge(known, on=['pair', 'date'], validate='one_to_one')
    prepared = {}
    for pair in CURRENCIES:
        frame = features.loc[features.pair.eq(pair)].sort_values('date').reset_index(drop=True)
        dates = pd.DatetimeIndex(frame.date)
        values = frame.price_USD_per_foreign.to_numpy(float)
        targets, _ = complete_signals(values)
        first_execution = int(dates.searchsorted(pd.Timestamp(IS_START)))
        for strategy in STRATEGIES:
            targets[strategy] = targets[strategy].copy()
            targets[strategy][:max(first_execution - 1, 0)] = 0
        prepared[pair] = (values, dates, frame, targets)
    return prepared


def applicable(spec, pair, strategy):
    if spec['id'] == 'BASE':
        return False
    if spec['id'] == 'COMMODITY_USD' and pair not in ('AUD', 'CAD', 'NOK'):
        return False
    return spec['families'] == 'all' or RULES[strategy][1] in spec['families']


def identity(pair, strategy, spec):
    return {'candidate_id': f"{pair}|{strategy}|{spec['id']}", 'pair': pair, 'strategy': strategy,
            'filter_id': spec['id'], 'filter_name': spec['label'], 'is_new': spec['id'] not in CONTROLS}


def metric_rows(key, run, dates, periods, variant):
    ledger = simple_ledger(run, dates, COST_BP)
    return [{**key, 'variant': variant, 'split': label,
             **evaluate_window(run, dates, start, end, COST_BP, ledger)} for label, start, end in periods]


def daily_rows(key, run, dates, periods, variant):
    gross, net, _ = daily_account(run, COST_BP)
    frames = []
    for label, start, end in periods:
        mask = (dates >= start) & (dates <= end)
        frames.append(pd.DataFrame({**key, 'variant': variant, 'split': label, 'date': dates[mask],
                                    'gross_daily_return': gross[mask], 'net_daily_return': net[mask],
                                    'executed': np.asarray(run['executed'])[mask], 'held': np.asarray(run['held'])[mask]}))
    return pd.concat(frames, ignore_index=True)


def add_retention(metrics):
    if metrics.empty:
        return metrics
    baseline = metrics.loc[metrics.variant.eq('BASE'), ['candidate_id', 'split', 'closed_n']].rename(columns={'closed_n': 'base_closed_n'})
    frame = metrics.merge(baseline, on=['candidate_id', 'split'], validate='many_to_one')
    frame['retention'] = frame.closed_n / frame.base_closed_n.replace(0, np.nan)
    return frame


def develop(output):
    protocol_path, freeze_path = output / 'protocol.json', output / 'frozen_selection.json'
    if freeze_path.exists():
        raise ValueError('This campaign is already frozen. Continue --stage oos/report; do not replace the selection.')
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        assert_hashes(protocol['source_sha256'])
        assert_hashes(protocol['input_sha256'])
    else:
        protocol = protocol_document()
        save_json(protocol_path, protocol)
    status = {'status': 'running', 'started_utc': now(), 'oos_evaluated': False,
              'protocol_sha256': sha(protocol_path), 'compute_cutoff': IS_END}
    save_json(output / 'development_status.json', status)
    prepared = prepare(IS_END)
    rows, coverage = [], []
    periods = [('IS', IS_START, IS_END), *IS_BLOCKS]
    registry = protocol['candidate_specs']
    for pair in CURRENCIES:
        print(f'IS only: {pair}, all 20 rules and compatible fixed mechanisms...', flush=True)
        values, dates, features, targets = prepared[pair]
        for strategy in STRATEGIES:
            base = fixed_execute(values, targets[strategy])
            base_stats = metric_rows({}, base, dates, periods, 'BASE')
            count = 0
            for spec in registry:
                if not applicable(spec, pair, strategy):
                    continue
                key = identity(pair, strategy, spec)
                filtered, decisions = run_candidate(values, dates, base, features, spec, pair, strategy)
                rows.extend({**row, **key} for row in base_stats)
                rows.extend(metric_rows(key, filtered, dates, periods, 'FILTER'))
                count += 1
            coverage.append({'pair': pair, 'strategy': strategy, 'candidate_count': count, 'compute_end': str(dates[-1].date())})
    metrics = add_retention(pd.DataFrame(rows))
    metrics.to_csv(output / 'is_all_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'is_account_coverage.csv', index=False)
    selection = screen_candidates(metrics)
    labels = {spec['id']: spec['label'] for spec in registry}
    selection['filter_name'] = selection.filter_id.map(labels)
    selection.to_csv(output / 'is_selection.csv', index=False)
    frozen_rows = selection.loc[selection.frozen].sort_values('freeze_rank')
    print(f'IS screen: {len(selection)} candidate accounts; {int(selection["pass"].sum())} passes including controls; {len(frozen_rows)} new account winners frozen.', flush=True)
    selected_daily, selected_metrics, selected_trades, selected_decisions, delay_rows = [], [], [], [], []
    registry_map = {item['id']: item for item in registry}
    for row in frozen_rows.itertuples(index=False):
        values, dates, features, targets = prepared[row.pair]
        base = fixed_execute(values, targets[row.strategy])
        spec = registry_map[row.filter_id]
        key = identity(row.pair, row.strategy, spec)
        filtered, decisions = run_candidate(values, dates, base, features, spec, row.pair, row.strategy)
        for variant, run in [('BASE', base), ('FILTER', filtered)]:
            selected_daily.append(daily_rows(key, run, dates, [('IS', IS_START, IS_END)], variant))
            selected_metrics.extend(metric_rows(key, run, dates, [('IS', IS_START, IS_END)], variant))
            ledger = simple_ledger(run, dates, COST_BP).assign(**key, variant=variant)
            selected_trades.append(ledger)
            delay_rows.extend(metric_rows(key, delayed_schedule(run), dates, [('IS_SCHEDULE_DELAY', IS_START, IS_END)], variant))
        selected_decisions.append(decisions.assign(**key))
    daily_columns = ['candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'variant', 'split', 'date', 'gross_daily_return', 'net_daily_return', 'executed', 'held']
    pd.concat(selected_daily, ignore_index=True).to_csv(output / 'is_frozen_daily.csv.gz', index=False) if selected_daily else pd.DataFrame(columns=daily_columns).to_csv(output / 'is_frozen_daily.csv.gz', index=False)
    main_metrics = add_retention(pd.DataFrame(selected_metrics)) if selected_metrics else metrics.iloc[:0].copy()
    main_metrics.to_csv(output / 'is_frozen_metrics.csv', index=False)
    if selected_trades:
        pd.concat(selected_trades, ignore_index=True).to_csv(output / 'is_frozen_trades.csv.gz', index=False)
        pd.concat(selected_decisions, ignore_index=True).to_csv(output / 'is_frozen_decisions.csv.gz', index=False)
        add_retention(pd.DataFrame(delay_rows)).to_csv(output / 'is_schedule_delay_diagnostic.csv', index=False)
    assert_hashes(protocol['source_sha256'])
    records = []
    for row in frozen_rows.to_dict('records'):
        records.append({**row, 'spec': registry_map[row['filter_id']]})
    freeze = {
        'created_utc': now(), 'protocol_sha256': sha(protocol_path),
        'selection_basis': 'IS only; all required criteria passed; one new filter per pair/strategy; fixed ranking',
        'oos_evaluated_at_freeze': False, 'is_end': IS_END, 'oos_start': OOS_START,
        'source_sha256': protocol['source_sha256'], 'input_sha256': protocol['input_sha256'],
        'is_output_sha256': {path.name: sha(path) for path in sorted(output.glob('is_*')) if path.is_file()},
        'candidates': records, 'all_candidate_accounts_tested': len(selection),
        'all_passing_versions': int(selection['pass'].sum()), 'frozen_candidates': len(records),
    }
    save_json(freeze_path, freeze)
    status.update(status='passed', finished_utc=now(), is_screen_completed=True,
                  source_integrity_pass=True, accounts_covered=len(coverage), candidates_evaluated=len(selection),
                  all_passing_versions=int(selection['pass'].sum()), frozen_candidates=len(records),
                  freeze_sha256=sha(freeze_path), holdout_returns_computed=False)
    save_json(output / 'development_status.json', status)


def verify_is_prefix(frozen_daily, key, run, dates, variant):
    reference = frozen_daily.loc[frozen_daily.candidate_id.eq(key['candidate_id']) & frozen_daily.variant.eq(variant)].sort_values('date')
    mask = (dates >= IS_START) & (dates <= IS_END)
    if not pd.DatetimeIndex(pd.to_datetime(reference.date)).equals(dates[mask]):
        raise AssertionError('Full replay changes the frozen IS calendar.')
    if not np.array_equal(reference.held.to_numpy(), np.asarray(run['held'])[mask]):
        raise AssertionError('Full replay changes frozen IS positions.')
    if not np.array_equal(reference.executed.to_numpy(), np.asarray(run['executed'])[mask]):
        raise AssertionError('Full replay changes frozen IS executions.')
    if not np.allclose(reference.gross_daily_return, np.asarray(run['gross_return'])[mask], atol=1e-14, rtol=0):
        raise AssertionError('Full replay changes frozen IS returns.')


def evaluate_oos(output):
    protocol_path, freeze_path = output / 'protocol.json', output / 'frozen_selection.json'
    protocol, freeze = read_json(protocol_path), read_json(freeze_path)
    if sha(protocol_path) != freeze['protocol_sha256']:
        raise ValueError('Protocol differs from the frozen selection.')
    assert_hashes(freeze['source_sha256'])
    assert_hashes(freeze['input_sha256'])
    for name, expected in freeze['is_output_sha256'].items():
        if sha(output / name) != expected:
            raise ValueError('IS evidence changed after freeze: ' + name)
    if not freeze['candidates']:
        print('No IS-qualified new candidate: OOS remains unevaluated.', flush=True)
        return
    existing = output / 'oos_status.json'
    if existing.exists() and read_json(existing).get('status') == 'passed':
        print('Frozen OOS results already exist; no second evaluation.', flush=True)
        return
    status = {'status': 'running', 'opened_utc': now(), 'freeze_sha256': sha(freeze_path),
              'protocol_sha256': sha(protocol_path), 'oos_evaluated': False,
              'freeze_created_before_oos': pd.Timestamp(freeze['created_utc']) < pd.Timestamp.now(tz='UTC'),
              'candidate_count': len(freeze['candidates']), 'retuning_permitted': False}
    save_json(existing, status)
    prepared = prepare(OOS_END)
    frozen_daily = pd.read_csv(output / 'is_frozen_daily.csv.gz', parse_dates=['date'])
    daily, metrics, anniversary, sensitivity, ledgers, decisions_all = [], [], [], [], [], []
    for row in freeze['candidates']:
        print(f"Frozen OOS {int(row['freeze_rank'])}/{len(freeze['candidates'])}: {row['candidate_id']}", flush=True)
        pair, strategy, spec = row['pair'], row['strategy'], row['spec']
        values, dates, features, targets = prepared[pair]
        base = fixed_execute(values, targets[strategy])
        filtered, decisions = run_candidate(values, dates, base, features, spec, pair, strategy)
        key = identity(pair, strategy, spec)
        for variant, run in [('BASE', base), ('FILTER', filtered)]:
            verify_is_prefix(frozen_daily, key, run, dates, variant)
            daily.append(daily_rows(key, run, dates, [('OOS', OOS_START, OOS_END)], variant))
            metrics.extend(metric_rows(key, run, dates, [('OOS', OOS_START, OOS_END)], variant))
            anniversary.extend(metric_rows(key, run, dates, OOS_YEARS, variant))
            cash_run = fresh_start(base, run, dates, OOS_START)
            sensitivity.extend(metric_rows(key, cash_run, dates, [('OOS_FRESH_START', OOS_START, OOS_END)], variant))
            ledgers.append(simple_ledger(run, dates, COST_BP).assign(**key, variant=variant))
        decisions_all.append(decisions.assign(**key))
    metrics, anniversary, sensitivity = [add_retention(pd.DataFrame(rows)) for rows in (metrics, anniversary, sensitivity)]
    assessment = []
    for row in freeze['candidates']:
        sub = metrics.loc[metrics.candidate_id.eq(row['candidate_id'])].set_index('variant')
        b, f = sub.loc['BASE'], sub.loc['FILTER']
        years = anniversary.loc[anniversary.candidate_id.eq(row['candidate_id']) & anniversary.variant.eq('FILTER')]
        delta = float(f.net_simple_return_pct - b.net_simple_return_pct)
        flags = {
            'annualized_pass': bool(f.net_annualized_simple_pct >= OOS_SCREEN['min_net_annualized_simple_pct'] - 1e-10),
            'baseline_improvement_pass': bool(delta >= OOS_SCREEN['min_net_return_improvement_pp'] - 1e-10),
            'drawdown_pass': bool(f.net_max_drawdown_pp >= OOS_SCREEN['worst_net_drawdown_pp'] - 1e-10),
            'positive_years_pass': bool((years.net_simple_return_pct > 1e-10).sum() >= OOS_SCREEN['min_positive_anniversary_years']),
            'sample_pass': bool(f.closed_n >= OOS_SCREEN['min_closed_trades']),
        }
        quality_pass = bool(f.closed_n >= 20 and np.isfinite([f.win_rate, b.win_rate, f.payoff_ratio, b.payoff_ratio]).all()
                            and f.win_rate >= b.win_rate - 1e-10 and f.payoff_ratio >= b.payoff_ratio - 1e-10)
        assessment.append({
            **{k: row[k] for k in ('candidate_id', 'pair', 'strategy', 'filter_id', 'freeze_rank')},
            'is_return_pct': row['is_gross_return_pct'], 'oos_return_pct': float(f.net_simple_return_pct),
            'oos_base_return_pct': float(b.net_simple_return_pct), 'oos_improvement_pp': delta,
            'oos_annualized_simple_pct': float(f.net_annualized_simple_pct),
            'oos_max_drawdown_pp': float(f.net_max_drawdown_pp), 'oos_win_rate': float(f.win_rate),
            'oos_base_win_rate': float(b.win_rate), 'oos_payoff_ratio': float(f.payoff_ratio),
            'oos_base_payoff_ratio': float(b.payoff_ratio), 'oos_closed_n': int(f.closed_n),
            'oos_positive_years': int((years.net_simple_return_pct > 1e-10).sum()),
            **flags, 'oos_curve_pass': bool(all(flags.values())), 'oos_quality_confirmed': quality_pass,
            'failed_criteria': '|'.join(k for k, value in flags.items() if not value),
        })
    assessment = pd.DataFrame(assessment).sort_values('freeze_rank')
    metrics = metrics.merge(assessment[['candidate_id', 'oos_curve_pass', 'oos_quality_confirmed']], on='candidate_id', validate='many_to_one')
    metrics.to_csv(output / 'oos_metrics.csv', index=False)
    anniversary.to_csv(output / 'oos_anniversary_metrics.csv', index=False)
    sensitivity.to_csv(output / 'oos_fresh_start_metrics.csv', index=False)
    assessment.to_csv(output / 'candidate_assessment.csv', index=False)
    pd.concat(daily, ignore_index=True).to_csv(output / 'oos_daily.csv.gz', index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(output / 'frozen_full_trade_ledger.csv.gz', index=False)
    pd.concat(decisions_all, ignore_index=True).to_csv(output / 'frozen_full_episode_decisions.csv.gz', index=False)
    assert_hashes(freeze['source_sha256'])
    status.update(status='passed', finished_utc=now(), oos_evaluated=True, source_integrity_pass=True,
                  is_prefix_positions_and_returns_match=True, is_screen_completed=True,
                  curve_passing_candidates=int(assessment.oos_curve_pass.sum()),
                  quality_confirmed_candidates=int(assessment.oos_quality_confirmed.sum()),
                  both_curve_and_quality_candidates=int((assessment.oos_curve_pass & assessment.oos_quality_confirmed).sum()),
                  output_sha256={p.name: sha(p) for p in output.glob('*.csv*') if not p.name.startswith('is_')})
    save_json(existing, status)
    print(assessment.to_string(index=False), flush=True)


def report(output):
    from fx_filter_report import render_filter_report
    protocol = read_json(output / 'protocol.json')
    freeze = read_json(output / 'frozen_selection.json')
    validation = read_json(output / 'development_status.json')
    validation.update(protocol_sha256=sha(output / 'protocol.json'), freeze_sha256=sha(output / 'frozen_selection.json'))
    selection = pd.read_csv(output / 'is_selection.csv')
    daily = pd.read_csv(output / 'is_frozen_daily.csv.gz', parse_dates=['date'])
    metrics = pd.read_csv(output / 'is_frozen_metrics.csv')
    if (output / 'oos_status.json').exists() and read_json(output / 'oos_status.json').get('status') == 'passed':
        validation.update(read_json(output / 'oos_status.json'))
        daily = pd.concat([daily, pd.read_csv(output / 'oos_daily.csv.gz', parse_dates=['date'])], ignore_index=True)
        metrics = pd.concat([metrics, pd.read_csv(output / 'oos_metrics.csv')], ignore_index=True)
    validation['frozen_candidates'] = len(freeze['candidates'])
    for column in ('start_date', 'end_date', 'anchor_date'):
        if column in metrics:
            metrics[column] = pd.to_datetime(metrics[column])
    built = render_filter_report(output, daily, metrics, selection, protocol, validation)
    save_json(output / 'report_build.json', built)
    print('Report:', output / built['report'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['develop', 'oos', 'report'], required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/filter_oos_v1')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        {'develop': develop, 'oos': evaluate_oos, 'report': report}[args.stage](output)
    except Exception as error:
        save_json(output / f'{args.stage}_error.json', {'failed_utc': now(), 'error': str(error), 'stage': args.stage})
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
