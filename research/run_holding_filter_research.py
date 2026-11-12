"""Observed exact-checkpoint exits, original IS screen, frozen historical OOS.

The discovery parser consumes only 2005-2014 outcome rows. Concentration is
reported rather than added to the user's original comprehensive screen.
"""
from pathlib import Path
import argparse
import csv
from itertools import combinations
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_baselines import CURRENCIES, STRATEGIES, fixed_execute
from fx_filter_evaluation import IS_BLOCKS, SCREEN, OOS_SCREEN, simple_ledger, screen_candidates
from fx_holding_filters import run_holding_filter

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/holding_filter_v3'
OBSERVATIONS = ROOT / 'reproduced/holding_observations_v3'
DISCOVERY_BLOCKS = {'D1', 'D2', 'D'}
CONDITION_ORDER = ('pnl_negative', 'pnl_positive', 'giveback_half', 'recent5_negative', 'opposite20')
ALLOWED_CONDITIONS = set(CONDITION_ORDER)
AGES = {5, 10, 20, 40, 63}


def registered_specs():
    conditions = [(c,) for c in CONDITION_ORDER] + [c for c in combinations(CONDITION_ORDER, 2)
                  if set(c) != {'pnl_negative', 'pnl_positive'}]
    return [{'id': f"CHECK_{age}_{'_AND_'.join(c)}", 'checkpoint_age': age,
             'condition': c[0] if len(c) == 1 else list(c)} for age in sorted(AGES) for c in conditions]


def identity(pair, strategy, spec):
    condition = spec['condition'] if isinstance(spec['condition'], str) else '&'.join(spec['condition'])
    return {'candidate_id': f"{pair}|{strategy}|{spec['id']}", 'pair': pair, 'strategy': strategy,
            'filter_id': spec['id'], 'filter_name': f"Exit at age {spec['checkpoint_age']} if {condition}", 'is_new': True}


def proposals(path):
    """Ignore later blocks BEFORE parsing outcome values; one fixed condition/age."""
    evidence = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['block'] not in DISCOVERY_BLOCKS or row['condition'] == 'ALL':
                continue
            condition = row['condition'].split('&')
            if len(condition) not in (1, 2) or not set(condition) <= ALLOWED_CONDITIONS:
                raise ValueError('Unregistered holding condition.')
            if len(set(condition)) != len(condition) or {'pnl_negative', 'pnl_positive'} <= set(condition):
                raise ValueError('Duplicate or contradictory condition.')
            if condition != sorted(condition, key=CONDITION_ORDER.index):
                raise ValueError('Condition order differs from the registered observation protocol.')
            age = int(row['checkpoint_age'])
            if age not in AGES or row['pair'] not in CURRENCIES or row['strategy'] not in STRATEGIES:
                raise ValueError('Unregistered account/checkpoint.')
            key = (row['pair'], row['strategy'], age, row['condition'])
            block = evidence.setdefault(key, {})
            if row['block'] in block:
                raise ValueError('Duplicate discovery observation.')
            block[row['block']] = {field: float(row[field]) for field in (
                'closed_n', 'remaining_return_sum_pct', 'remaining_without_worst1_pct', 'remaining_without_worst3_pct')}
            values = block[row['block']]
            if not np.isfinite(list(values.values())).all() or values['closed_n'] < 0 or values['closed_n'] != int(values['closed_n']):
                raise ValueError('Invalid observation counts or residual sums.')
    result = []
    for (pair, strategy, age, condition), blocks in sorted(evidence.items()):
        if set(blocks) != DISCOVERY_BLOCKS:
            raise ValueError('Missing discovery block.')
        # Repeated negative executable tails motivate a hypothesis, not a success claim.
        if blocks['D']['closed_n'] < blocks['D1']['closed_n'] + blocks['D2']['closed_n']:
            raise ValueError('Combined discovery contains fewer closed trades than its disjoint halves.')
        if not all(blocks[b]['closed_n'] >= 10 and blocks[b]['remaining_return_sum_pct'] < -1e-10
                   for b in ('D1', 'D2')):
            continue
        if blocks['D']['remaining_return_sum_pct'] >= -1e-10:
            continue
        parts = condition.split('&')
        spec = {'id': f"CHECK_{age}_{condition.replace('&', '_AND_')}", 'checkpoint_age': age,
                'condition': parts[0] if len(parts) == 1 else parts}
        result.append({**identity(pair, strategy, spec), 'spec': spec, 'discovery_evidence': blocks})
    return result


def check_observations():
    validation = core.read_json(OBSERVATIONS / 'holding_observation_validation.json')
    if validation['status'] != 'passed':
        raise ValueError('Holding observation audit did not pass.')
    protocol = core.read_json(OBSERVATIONS / 'holding_observation_protocol.json')
    if validation['protocol_sha256'] != core.sha(OBSERVATIONS / 'holding_observation_protocol.json'):
        raise ValueError('Observation protocol changed.')
    if validation['accounts'] != 180 or validation['compute_cutoff'] != core.IS_END or validation['oos_computed'] is not False or protocol['cost_bp'] != 0:
        raise ValueError('Observation scope/cutoff differs from this campaign.')
    core.assert_hashes(protocol['source_sha256'])
    core.assert_hashes(protocol['input_sha256'])
    for name, digest in validation['output_sha256'].items():
        if core.sha(OBSERVATIONS / name) != digest:
            raise ValueError('Observation cache changed: ' + name)
    return {p.relative_to(ROOT).as_posix(): core.sha(p) for p in
            [OBSERVATIONS / 'holding_observation_protocol.json', OBSERVATIONS / 'holding_observation_validation.json',
             *[OBSERVATIONS / n for n in validation['output_sha256']]]}


def develop(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing campaign; use a new empty directory.')
    observation_hashes = check_observations()
    source = OBSERVATIONS / 'holding_state_summary.csv'
    protocol = core.protocol_document()
    protocol.update(version='holding_filter_v3', created_utc=core.now(), legacy_comparator_ids=[], candidate_specs=registered_specs(),
        proposal_rule='Only D1 (2005-09), D2 (2010-14), and D (2005-14) outcomes are parsed. Each half must have >=10 closed actionable original episodes and negative executable remaining-return sum; combined discovery must also be negative. Worst-trade concentration is diagnostic only.',
        checkpoint_rule='Exactly one check at the specified earned-return age. If the original technical target still holds and the condition is true, issue an exit at this close, execute next close, and keep that execution-day return. Never re-enter the original episode. False/unknown retains the original episode without later retries.',
        family_application='All 180 BASE accounts are observed. Every discovery-supported exact-checkpoint hypothesis is replayed on its originating account; zero-proposal accounts remain in the coverage table.',
        observation_source_sha256=observation_hashes,
        schedule_delay_sensitivity='No extra schedule-delay experiment in this campaign.',
        holdout_rule='Original SCREEN and original within-account ranking, then freeze one passing candidate per account. Evaluate every frozen candidate under unchanged OOS_SCREEN. No OOS-based rule revisions, substitutions or deletions.',
        source_sha256={**core.source_hashes(), **{n: core.sha(ROOT / n) for n in
            ['run_holding_filter_research.py', 'src/fx_holding_filters.py', 'build_holding_observations.py']}},
        input_sha256={**core.input_hashes(), **observation_hashes})
    output.mkdir(parents=True, exist_ok=True)
    core.save_json(output / 'protocol.json', protocol)
    hypotheses = proposals(source)
    registry = {h['filter_id']: h['spec'] for h in hypotheses}
    # Freeze the observation-derived proposals before computing filtered paths.
    core.save_json(output / 'is_hypotheses.json', {'created_utc': core.now(), 'candidates': hypotheses})
    status = {'status': 'running', 'started_utc': core.now(), 'oos_evaluated': False, 'compute_cutoff': core.IS_END}
    core.save_json(output / 'development_status.json', status)
    print('Discovery-supported checkpoint hypotheses:', len(hypotheses), flush=True)
    prepared = core.prepare(core.IS_END) if hypotheses else {}
    cache, rows, triggers = {}, [], []
    for index, item in enumerate(hypotheses):
        pair, strategy, spec = item['pair'], item['strategy'], item['spec']
        values, dates, features, targets = prepared[pair]
        if (pair, strategy) not in cache:
            base = fixed_execute(values, targets[strategy])
            cache[pair, strategy] = base
        base = cache[pair, strategy]
        filtered, decisions = run_holding_filter(values, dates, base, features, spec, pair, strategy)
        key = identity(pair, strategy, spec)
        periods = [('IS', core.IS_START, core.IS_END), *IS_BLOCKS]
        rows.extend(core.metric_rows(key, base, dates, periods, 'BASE'))
        rows.extend(core.metric_rows(key, filtered, dates, periods, 'FILTER'))
        triggers.append(decisions.assign(**key))
        if (index + 1) % 100 == 0:
            print('IS replay', index + 1, '/', len(hypotheses), flush=True)
    counts = pd.Series([h['pair'] + '|' + h['strategy'] for h in hypotheses], dtype=object).value_counts()
    pd.DataFrame([{'pair': p, 'strategy': s, 'candidate_count': int(counts.get(p + '|' + s, 0)),
                   'compute_end': core.IS_END} for p in CURRENCIES for s in STRATEGIES]).to_csv(output / 'is_account_coverage.csv', index=False)
    if rows:
        metrics = core.add_retention(pd.DataFrame(rows))
        selection = screen_candidates(metrics)
        selection['filter_name'] = selection.filter_id.map({h['filter_id']: h['filter_name'] for h in hypotheses})
        pd.concat(triggers, ignore_index=True).to_csv(output / 'is_all_decisions.csv.gz', index=False)
    else:
        metrics = pd.DataFrame(columns=['candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'variant', 'split'])
        selection = pd.DataFrame(columns=['candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'pass', 'frozen', 'freeze_rank'])
    metrics.to_csv(output / 'is_all_metrics.csv', index=False)
    selection.to_csv(output / 'is_selection.csv', index=False)
    frozen_rows = selection.loc[selection.frozen.eq(True)].sort_values('freeze_rank')
    daily, main, ledgers, decisions_out = [], [], [], []
    for row in frozen_rows.to_dict('records'):
        pair, strategy, spec = row['pair'], row['strategy'], registry[row['filter_id']]
        values, dates, features, targets = prepared[pair]
        base = cache[pair, strategy]
        filtered, decision = run_holding_filter(values, dates, base, features, spec, pair, strategy)
        key = identity(pair, strategy, spec)
        for variant, run in [('BASE', base), ('FILTER', filtered)]:
            daily.append(core.daily_rows(key, run, dates, [('IS', core.IS_START, core.IS_END)], variant))
            main.extend(core.metric_rows(key, run, dates, [('IS', core.IS_START, core.IS_END)], variant))
            ledgers.append(simple_ledger(run, dates).assign(**key, variant=variant))
        decisions_out.append(decision.assign(**key))
    empty_columns = ['candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'variant', 'split',
                     'date', 'gross_daily_return', 'net_daily_return', 'executed', 'held']
    (pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=empty_columns)).to_csv(output / 'is_frozen_daily.csv.gz', index=False)
    (core.add_retention(pd.DataFrame(main)) if main else metrics.iloc[:0]).to_csv(output / 'is_frozen_metrics.csv', index=False)
    if ledgers:
        pd.concat(ledgers, ignore_index=True).to_csv(output / 'is_frozen_trades.csv.gz', index=False)
        pd.concat(decisions_out, ignore_index=True).to_csv(output / 'is_frozen_decisions.csv.gz', index=False)
    core.assert_hashes(protocol['source_sha256']); core.assert_hashes(protocol['input_sha256'])
    records = [{**r, 'spec': registry[r['filter_id']]} for r in frozen_rows.to_dict('records')]
    core.save_json(output / 'frozen_selection.json', {
        'created_utc': core.now(), 'protocol_sha256': core.sha(output / 'protocol.json'),
        'selection_basis': protocol['holdout_rule'], 'oos_evaluated_at_freeze': False,
        'source_sha256': protocol['source_sha256'], 'input_sha256': protocol['input_sha256'],
        'is_output_sha256': {p.name: core.sha(p) for p in sorted(output.glob('is_*')) if p.is_file()},
        'candidates': records, 'all_candidate_accounts_tested': len(selection),
        'all_passing_versions': int(selection['pass'].sum()), 'frozen_candidates': len(records)})
    status.update(status='passed', finished_utc=core.now(), is_screen_completed=True, source_integrity_pass=True,
        accounts_covered=180, candidates_evaluated=len(selection), all_passing_versions=int(selection['pass'].sum()),
        frozen_candidates=len(records), freeze_sha256=core.sha(output / 'frozen_selection.json'), holdout_returns_computed=False)
    core.save_json(output / 'development_status.json', status)
    print('Original IS screen:', int(selection['pass'].sum()), 'passes;', len(records), 'frozen accounts.', flush=True)


def oos(output):
    status = core.read_json(output / 'development_status.json')
    if status['status'] != 'passed' or status['freeze_sha256'] != core.sha(output / 'frozen_selection.json'):
        raise ValueError('A valid completed IS freeze is required.')
    old_engine, old_identity = core.run_candidate, core.identity
    try:
        core.run_candidate, core.identity = run_holding_filter, identity
        core.evaluate_oos(output)
    finally:
        core.run_candidate, core.identity = old_engine, old_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['develop', 'oos', 'report'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(); output = args.output.resolve()
    if args.stage == 'develop':
        develop(output)
    elif args.stage == 'oos':
        oos(output)
    else:
        from build_holding_filter_report import build_report
        build_report(output)


if __name__ == '__main__':
    main()
