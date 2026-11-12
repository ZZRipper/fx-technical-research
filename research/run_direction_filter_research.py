"""Observed long/short asymmetry, original IS screen, frozen historical OOS.

Fixed direction hypotheses are nominated only from the previously saved D-only
observation table. Existing results and original quality gates remain unchanged.
"""
from pathlib import Path
import argparse
import csv
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_baselines import CURRENCIES, STRATEGIES, fixed_execute
from fx_filter_evaluation import IS_BLOCKS, simple_ledger, screen_candidates
from fx_direction_filter import run_direction_filter

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/direction_filter_v6'
OBSERVATIONS = ROOT / 'reproduced/direction_probe_v5'
OBS_VALIDATION = ROOT / 'reproduced/direction_observation_validation_v6/validation.json'
DISCOVERY_BLOCKS = {'D1', 'D2', 'D'}


def registered_specs():
    return [{'id': 'REJECT_SHORT', 'reject_position': -1},
            {'id': 'REJECT_LONG', 'reject_position': 1}]


def identity(pair, strategy, spec):
    name = ('Reject short foreign / long USD entries; retain original long entries'
            if spec['reject_position'] == -1 else
            'Reject long foreign / short USD entries; retain original short entries')
    return {'candidate_id': f"{pair}|{strategy}|{spec['id']}", 'pair': pair, 'strategy': strategy,
            'filter_id': spec['id'], 'filter_name': name, 'is_new': True}


def proposals(path):
    """Later blocks are skipped before any outcome conversion."""
    evidence = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['block'] not in DISCOVERY_BLOCKS:
                continue
            pair, strategy = row['pair'], row['strategy']
            if pair not in CURRENCIES or strategy not in STRATEGIES or row['position'] not in ('-1', '1'):
                raise ValueError('Unregistered account or noncanonical position.')
            position = int(row['position'])
            blocks = evidence.setdefault((pair, strategy, position), {})
            if row['block'] in blocks:
                raise ValueError('Duplicate discovery direction block.')
            values = {k: float(row[k]) for k in ('closed_n', 'return_sum_pct')}
            diagnostic = row['return_without_worst_trade_pct']
            values['return_without_worst_trade_pct'] = float(diagnostic) if diagnostic else np.nan
            if not np.isfinite([values['closed_n'], values['return_sum_pct']]).all() or values['closed_n'] < 0 or values['closed_n'] != int(values['closed_n']):
                raise ValueError('Invalid discovery count or return.')
            empty = values['closed_n'] == 0
            worst = values['return_without_worst_trade_pct']
            if (empty and (values['return_sum_pct'] != 0 or not np.isnan(worst))) or (not empty and not np.isfinite(worst)):
                raise ValueError('Worst-trade diagnostic must be missing exactly for empty groups.')
            blocks[row['block']] = values
    result = []
    registry = {s['reject_position']: s for s in registered_specs()}
    for (pair, strategy, position), blocks in sorted(evidence.items()):
        if set(blocks) != DISCOVERY_BLOCKS:
            raise ValueError('Missing discovery direction block.')
        if blocks['D']['closed_n'] < blocks['D1']['closed_n'] + blocks['D2']['closed_n']:
            raise ValueError('Combined D has fewer closed trades than its disjoint halves.')
        if not all(blocks[b]['closed_n'] >= 10 and blocks[b]['return_sum_pct'] < -1e-10 for b in ('D1', 'D2')):
            continue
        if blocks['D']['return_sum_pct'] >= -1e-10:
            continue
        spec = registry[position]
        result.append({**identity(pair, strategy, spec), 'spec': dict(spec), 'discovery_evidence': blocks})
    return result


def check_observations():
    validation = core.read_json(OBS_VALIDATION)
    if validation['status'] != 'passed' or validation['accounts'] != 180 or validation['compute_cutoff'] != '2014-12-31' or validation['oos_computed'] is not False:
        raise ValueError('Direction observation audit/scope/cutoff failed.')
    core.assert_hashes(validation['source_sha256']); core.assert_hashes(validation['input_sha256'])
    probe = core.read_json(OBSERVATIONS / 'direction_probe.json')
    if probe['cost_bp'] != 0 or probe['access_audit']['later_outcomes_read'] or probe['access_audit']['OOS_read']:
        raise ValueError('Direction probe scope or costs differ.')
    core.assert_hashes(probe['source_sha256'])
    for name, digest in probe['output_sha256'].items():
        if core.sha(OBSERVATIONS / name) != digest:
            raise ValueError('Earlier direction probe changed: ' + name)
    paths = [OBS_VALIDATION, OBSERVATIONS / 'direction_probe.json',
             *[OBSERVATIONS / n for n in probe['output_sha256']]]
    return {**validation['source_sha256'], **validation['input_sha256'],
            **{p.relative_to(ROOT).as_posix(): core.sha(p) for p in paths}}


def develop(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing campaign; use a new empty directory.')
    observation_hashes = check_observations()
    source = OBSERVATIONS / 'direction_summary.csv'
    protocol = core.protocol_document()
    protocol.update(version='direction_filter_v6', created_utc=core.now(), legacy_comparator_ids=[], candidate_specs=registered_specs(),
        proposal_rule='Only D1 (2005-09), D2 (2010-14), and D (2005-14) direction rows are parsed. Each half has >=10 eligible closed original trades and negative sum; combined D sum also negative. Worst-trade removal is diagnostic, not a gate.',
        entry_rule='Reject all original TARGET episodes with the fixed rejected position, +1 long foreign/short USD or -1 short foreign/long USD. Other original episodes and technical exit timing are unchanged; no mid-episode entry. Next-close fills and fixed initial capital remain unchanged.',
        family_application='All 180 BASE accounts observed for both position directions. Every D-supported direction hypothesis replayed on its originating account; zero-proposal accounts remain in coverage.',
        hypothesis_origin='The direction probe was saved before the preceding interaction campaign OOS evaluation. Nominations use only that earlier D-only table and its independent D revalidation. No outcome from a prior OOS is a nomination input. IS has been repeatedly examined and the last three-year history is reused historical verification, not untouched evidence.',
        direction_definition={'+1': 'long foreign / short USD', '-1': 'short foreign / long USD'},
        observation_source_sha256=observation_hashes,
        schedule_delay_sensitivity='No extra schedule-delay experiment in this campaign.',
        holdout_rule='Original SCREEN and original within-account ranking, then freeze one passing candidate per account. Evaluate every frozen candidate under unchanged OOS_SCREEN. No OOS-based rule revisions, substitutions or deletions.',
        source_sha256={**core.source_hashes(), **{n: core.sha(ROOT / n) for n in
            ['run_direction_filter_research.py', 'src/fx_direction_filter.py', 'verify_direction_observations.py']}},
        input_sha256={**core.input_hashes(), **observation_hashes})
    output.mkdir(parents=True, exist_ok=True)
    core.save_json(output / 'protocol.json', protocol)
    hypotheses = proposals(source)
    registry = {h['filter_id']: h['spec'] for h in hypotheses}
    # Freeze the observation-derived proposals before computing filtered paths.
    core.save_json(output / 'is_hypotheses.json', {'created_utc': core.now(), 'candidates': hypotheses})
    status = {'status': 'running', 'started_utc': core.now(), 'oos_evaluated': False, 'compute_cutoff': core.IS_END}
    core.save_json(output / 'development_status.json', status)
    print('Discovery-supported direction hypotheses:', len(hypotheses), flush=True)
    prepared = core.prepare(core.IS_END) if hypotheses else {}
    cache, rows, triggers = {}, [], []
    for index, item in enumerate(hypotheses):
        pair, strategy, spec = item['pair'], item['strategy'], item['spec']
        values, dates, features, targets = prepared[pair]
        if (pair, strategy) not in cache:
            base = fixed_execute(values, targets[strategy])
            cache[pair, strategy] = base
        base = cache[pair, strategy]
        filtered, decisions = run_direction_filter(values, dates, base, features, spec, pair, strategy)
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
        filtered, decision = run_direction_filter(values, dates, base, features, spec, pair, strategy)
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
        core.run_candidate, core.identity = run_direction_filter, identity
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
        core.report(output)


if __name__ == '__main__':
    main()
