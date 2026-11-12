"""D-observed two-state entry rejection under the original comprehensive screen.

Three fixed feature pairs, two known conditions AND, unchanged zero-cost BASE
execution. D-only nominations are saved before full IS replay; one original-
SCREEN passing rule per account freezes before reused historical OOS.
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
from fx_is_states import STATE_BINS
from fx_entry_interaction_filters import run_entry_interaction_filter
from build_entry_interaction_observations import ALLOWED_INTERACTIONS

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/entry_interaction_filter_v5'
OBSERVATIONS = ROOT / 'reproduced/entry_interaction_observations_v5'
DISCOVERY_BLOCKS = {'D1', 'D2', 'D'}


def make_spec(feature_a, state_a, feature_b, state_b):
    conditions = sorted([{'feature': feature_a, 'state': state_a},
                         {'feature': feature_b, 'state': state_b}], key=lambda x: x['feature'])
    identifier = 'AND_' + '__'.join(f"{c['feature']}_{c['state']}" for c in conditions)
    return {'id': identifier, 'conditions': conditions}


def registered_specs():
    return [make_spec(a, x, b, y) for _, a, b in ALLOWED_INTERACTIONS
            for x in range(4) for y in range(4)]


def identity(pair, strategy, spec):
    labels = []
    for condition in spec['conditions']:
        feature, state = condition['feature'], condition['state']
        bounds = [-np.inf, *STATE_BINS[feature], np.inf]
        labels.append(f"{feature} in [{bounds[state]:g}, {bounds[state + 1]:g})")
    name = 'Reject when ' + ' AND '.join(labels)
    return {'candidate_id': f"{pair}|{strategy}|{spec['id']}", 'pair': pair, 'strategy': strategy,
            'filter_id': spec['id'], 'filter_name': name, 'is_new': True}


def proposals(path):
    """Read only the fixed D joint cohorts; later outcomes are ignored pre-parse."""
    registry = {name: (a, b) for name, a, b in ALLOWED_INTERACTIONS}
    evidence = {}
    fields = ('closed_n', 'return_sum_pct', 'return_without_worst1_pct', 'return_without_worst3_pct')
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['block'] not in DISCOVERY_BLOCKS or row['group'] != 'joint':
                continue
            pair, strategy, mechanism = row['pair'], row['strategy'], row['interaction_id']
            if pair not in CURRENCIES or strategy not in STRATEGIES or mechanism not in registry:
                raise ValueError('Unregistered account or entry interaction.')
            a, b = row['feature_a'], row['feature_b']
            if (a, b) != registry[mechanism]:
                raise ValueError('Features differ from registered mechanism order.')
            x, y = int(row['state_a']), int(row['state_b'])
            if str(x) != row['state_a'] or str(y) != row['state_b'] or x not in range(4) or y not in range(4):
                raise ValueError('Joint states must be canonical integers 0..3.')
            key = (pair, strategy, mechanism, x, y)
            blocks = evidence.setdefault(key, {})
            if row['block'] in blocks:
                raise ValueError('Duplicate discovery joint cohort.')
            values = {field: float(row[field]) for field in fields}
            if not np.isfinite(list(values.values())).all() or values['closed_n'] < 0 or values['closed_n'] != int(values['closed_n']):
                raise ValueError('Invalid discovery counts or sums.')
            blocks[row['block']] = values
    result = []
    for (pair, strategy, mechanism, x, y), blocks in sorted(evidence.items()):
        if set(blocks) != DISCOVERY_BLOCKS:
            raise ValueError('Missing discovery block.')
        if blocks['D']['closed_n'] < blocks['D1']['closed_n'] + blocks['D2']['closed_n']:
            raise ValueError('Combined D has fewer closed trades than its disjoint halves.')
        if not all(blocks[b]['closed_n'] >= 10 and blocks[b]['return_sum_pct'] < -1e-10 for b in ('D1', 'D2')):
            continue
        if blocks['D']['return_sum_pct'] >= -1e-10:
            continue
        a, b = registry[mechanism]
        spec = make_spec(a, x, b, y)
        result.append({**identity(pair, strategy, spec), 'interaction_id': mechanism,
                       'spec': spec, 'discovery_evidence': blocks})
    return result


def check_observations():
    validation_path = OBSERVATIONS / 'interaction_observation_validation.json'
    protocol_path = OBSERVATIONS / 'interaction_observation_protocol.json'
    validation, protocol = core.read_json(validation_path), core.read_json(protocol_path)
    if validation['status'] != 'passed' or validation['protocol_sha256'] != core.sha(protocol_path):
        raise ValueError('Interaction observation audit or protocol hash failed.')
    if validation['accounts'] != 180 or validation['compute_cutoff'] != '2014-12-31' or validation['oos_computed'] is not False or protocol['cost_bp'] != 0:
        raise ValueError('Observation scope/cutoff/cost differs from this campaign.')
    core.assert_hashes(protocol['source_sha256']); core.assert_hashes(protocol['input_sha256'])
    for name, digest in validation['output_sha256'].items():
        if core.sha(OBSERVATIONS / name) != digest:
            raise ValueError('Interaction observations changed: ' + name)
    return {p.relative_to(ROOT).as_posix(): core.sha(p) for p in
            [protocol_path, validation_path, *[OBSERVATIONS / n for n in validation['output_sha256']]]}


def develop(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing campaign; use a new empty directory.')
    observation_hashes = check_observations()
    source = OBSERVATIONS / 'interaction_state_summary.csv'
    protocol = core.protocol_document()
    protocol.update(version='entry_interaction_filter_v5', created_utc=core.now(), legacy_comparator_ids=[], candidate_specs=registered_specs(),
        proposal_rule='Only joint rows in D1 (2005-09), D2 (2010-14), and D (2005-14) are parsed. Each half must have >=10 eligible closed original trades and negative sum; combined D sum must also be negative. Complement quality and concentration are descriptive, not additional gates.',
        entry_rule='At each original BASE signal, reject the entire original target episode only if both fixed entry states are known and match. Keep UNKNOWN. Never re-enter during the rejected original episode; technical exits and reversals are unchanged. Both conditions use the original unfiltered BASE history.',
        family_application='All 180 BASE accounts are observed for exactly three predeclared feature pairs and all sixteen state intersections each. Every D-supported joint hypothesis is replayed on its originating account; zero-proposal accounts remain covered.',
        hypothesis_origin='New interaction hypotheses after prior entry and holding IS campaigns. IS has been repeatedly observed; the historical 2023-2026 OOS was exposed in prior research. This is disclosed repeated historical verification, not untouched independent evidence.',
        interaction_registry=ALLOWED_INTERACTIONS, state_bins=STATE_BINS,
        observation_source_sha256=observation_hashes,
        schedule_delay_sensitivity='No extra schedule-delay experiment in this campaign.',
        holdout_rule='Original SCREEN and original within-account ranking, then freeze one passing candidate per account. Evaluate every frozen candidate under unchanged OOS_SCREEN. No OOS-based rule revisions, substitutions or deletions.',
        source_sha256={**core.source_hashes(), **{n: core.sha(ROOT / n) for n in
            ['run_entry_interaction_research.py', 'src/fx_entry_interaction_filters.py', 'src/fx_state_filters.py', 'src/fx_is_states.py', 'build_entry_interaction_observations.py']}},
        input_sha256={**core.input_hashes(), **observation_hashes})
    output.mkdir(parents=True, exist_ok=True)
    core.save_json(output / 'protocol.json', protocol)
    hypotheses = proposals(source)
    registry = {h['filter_id']: h['spec'] for h in hypotheses}
    # Freeze the observation-derived proposals before computing filtered paths.
    core.save_json(output / 'is_hypotheses.json', {'created_utc': core.now(), 'candidates': hypotheses})
    status = {'status': 'running', 'started_utc': core.now(), 'oos_evaluated': False, 'compute_cutoff': core.IS_END}
    core.save_json(output / 'development_status.json', status)
    print('Discovery-supported joint-entry hypotheses:', len(hypotheses), flush=True)
    prepared = core.prepare(core.IS_END) if hypotheses else {}
    cache, rows, triggers = {}, [], []
    for index, item in enumerate(hypotheses):
        pair, strategy, spec = item['pair'], item['strategy'], item['spec']
        values, dates, features, targets = prepared[pair]
        if (pair, strategy) not in cache:
            base = fixed_execute(values, targets[strategy])
            cache[pair, strategy] = base
        base = cache[pair, strategy]
        filtered, decisions = run_entry_interaction_filter(values, dates, base, features, spec, pair, strategy)
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
        filtered, decision = run_entry_interaction_filter(values, dates, base, features, spec, pair, strategy)
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
        core.run_candidate, core.identity = run_entry_interaction_filter, identity
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
