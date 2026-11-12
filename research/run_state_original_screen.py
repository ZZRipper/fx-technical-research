"""Complete the original SCREEN evaluation of the fixed v2 D-only state list.

Prior v2 extra D/V/confirmation criteria and outcomes remain immutable. This
separate campaign has no new thresholds and uses all 1201 original hypotheses.
"""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_baselines import CURRENCIES, STRATEGIES, fixed_execute
from fx_filter_evaluation import IS_BLOCKS, simple_ledger, screen_candidates
from fx_is_states import STATE_BINS
from fx_state_filters import run_state_filter
from run_state_filter_research import proposal_rows

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/state_original_screen_v7'
ORIGINAL = ROOT / 'reproduced/state_filter_v2'
REVIEW = ROOT / 'reproduced/state_original_screen_scope_review/review.json'
PRIOR_ID = 'JPY|BB_MR_20|STATE_efficiency_63_1'


def registered_specs():
    return [{'id': f'STATE_{feature}_{state}', 'feature': feature, 'state': state}
            for feature in STATE_BINS for state in range(4)]


def identity(pair, strategy, spec):
    bounds = [-np.inf, *STATE_BINS[spec['feature']], np.inf]
    state = spec['state']
    name = f"Reject when {spec['feature']} in [{bounds[state]:g}, {bounds[state+1]:g})"
    return {'candidate_id': f"{pair}|{strategy}|{spec['id']}", 'pair': pair, 'strategy': strategy,
            'filter_id': spec['id'], 'filter_name': name, 'is_new': True}


def proposals(path):
    document = core.read_json(path)
    original = document['candidates']
    registry = {spec['id']: spec for spec in registered_specs()}
    ids = set(); result = []
    for item in original:
        spec = item['spec']; pair, strategy = item['pair'], item['strategy']
        if pair not in CURRENCIES or strategy not in STRATEGIES or spec.get('id') not in registry or spec != registry[spec['id']]:
            raise ValueError('A saved proposal differs from the fixed original state registry.')
        if isinstance(spec['state'], bool) or not isinstance(spec['state'], int):
            raise ValueError('Original state must be an integer, never bool.')
        key = identity(pair, strategy, spec)
        if item['candidate_id'] != key['candidate_id'] or item['filter_id'] != spec['id'] or item.get('is_new') is not True or key['candidate_id'] in ids:
            raise ValueError('Duplicate or altered original hypothesis identity.')
        blocks = item['discovery_evidence']
        if set(blocks) != {'DISCOVERY_1', 'DISCOVERY_2', 'DEVELOPMENT'}:
            raise ValueError('Saved proposal lacks exact original D evidence.')
        for values in blocks.values():
            nums = [values[k] for k in ('closed_n', 'return_sum_pct', 'return_without_worst_trade_pct')]
            if not np.isfinite(nums).all() or values['closed_n'] < 0 or values['closed_n'] != int(values['closed_n']):
                raise ValueError('Nonfinite or invalid original D support.')
        if not all(blocks[b]['closed_n'] >= 10 and blocks[b]['return_sum_pct'] < 0 for b in ('DISCOVERY_1', 'DISCOVERY_2')) or blocks['DEVELOPMENT']['return_without_worst_trade_pct'] >= 0:
            raise ValueError('Saved proposal no longer meets its original D rule.')
        ids.add(key['candidate_id'])
        result.append({**item, **key, 'spec': dict(spec)})
    return result


def check_observations():
    review = core.read_json(REVIEW)
    if review['status'] != 'passed' or review['original_proposal_count'] != 1201 or review['no_new_filtered_returns_computed'] is not True or review['OOS_results_opened'] is not False:
        raise ValueError('Fixed-candidate scope review failed.')
    core.assert_hashes(review['source_sha256']); core.assert_hashes(review['input_sha256'])
    saved = core.read_json(ORIGINAL / 'discovery_proposals.json')['candidates']
    regenerated = proposal_rows(ROOT / 'reproduced/filter_is_diagnostics_v2/is_entry_state_summary.csv')
    if saved != regenerated or len(saved) != 1201:
        raise ValueError('Original proposal list/order/evidence differs from its frozen D-only construction.')
    validated = proposals(ORIGINAL / 'discovery_proposals.json')
    if len(validated) != 1201:
        raise ValueError('The entire fixed original hypothesis list is required.')
    return {**review['source_sha256'], **review['input_sha256'], REVIEW.relative_to(ROOT).as_posix(): core.sha(REVIEW)}


def develop(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing campaign; use a new empty directory.')
    observation_hashes = check_observations()
    source = ORIGINAL / 'discovery_proposals.json'
    protocol = core.protocol_document()
    protocol.update(version='state_original_screen_v7', created_utc=core.now(), legacy_comparator_ids=[], candidate_specs=registered_specs(),
        proposal_rule='Exactly the 1201 proposals in the immutable v2 discovery_proposals.json, including original ordering/specs/D-only evidence. Original D1/D2 >=10 closed and negative sum, combined D return after removing worst one still negative. No new proposal generation or feature/threshold search.',
        entry_rule='Use the unchanged fx_state_filters single-state engine. At each original BASE signal reject the complete original TARGET episode if its fixed feature state matches; UNKNOWN retained. No mid-episode entry, unchanged original technical exits, next-close fills, zero costs.',
        family_application='All 180 BASE accounts retained in coverage; every original D proposal evaluated on its originating account. The original 1201 list is a fixed candidate set, not the complete universe of all feature/account bins.',
        hypothesis_origin='Evaluation-scope completion for existing v2 D-only proposals: all 1201 had D/V/FIT metrics, only four v2 locked candidates had full-IS/confirmation paths. This separate campaign evaluates every fixed proposal using the user original SCREEN. Original v2 strict stages, locks and failures are not rewritten. Later repeated IS and historical OOS exposure are disclosed.',
        candidate_novelty='Existing hypotheses, newly completed original-standard evaluation; is_new is the original SCREEN engine eligibility flag, not a claim of unseen hypotheses.',
        original_proposal_count=1201, original_v2_full_IS_count=4, original_v2_missing_full_IS_count=1197,
        previously_evaluated_exact_candidates=[{'candidate_id': PRIOR_ID, 'spec': {'id': 'STATE_efficiency_63_1', 'feature': 'efficiency_63', 'state': 1}, 'source_campaign': 'state_filter_supplement_v1', 'policy': 'If selected under unchanged ranking, retain it. Any same historical OOS replay is verification of existing evidence, not new independent evidence; no exclusion or substitution based on prior outcomes.'}],
        state_bins=STATE_BINS, observation_source_sha256=observation_hashes,
        schedule_delay_sensitivity='No extra schedule-delay experiment in this campaign.',
        holdout_rule='Original SCREEN and original within-account ranking applied to the entire fixed 1201 set. Freeze one passing candidate per account; evaluate every frozen candidate unchanged, preserving all failures. No old outcome file is read to nominate or rank.',
        source_sha256={**core.source_hashes(), **{n: core.sha(ROOT / n) for n in
            ['run_state_original_screen.py', 'src/fx_state_filters.py', 'src/fx_is_states.py', 'run_state_filter_research.py']}},
        input_sha256={**core.input_hashes(), **observation_hashes})
    output.mkdir(parents=True, exist_ok=True)
    core.save_json(output / 'protocol.json', protocol)
    hypotheses = proposals(source)
    registry = {h['filter_id']: h['spec'] for h in hypotheses}
    # Freeze the observation-derived proposals before computing filtered paths.
    core.save_json(output / 'is_hypotheses.json', {'created_utc': core.now(), 'candidates': hypotheses})
    status = {'status': 'running', 'started_utc': core.now(), 'oos_evaluated': False, 'compute_cutoff': core.IS_END}
    core.save_json(output / 'development_status.json', status)
    print('Fixed original D-only state proposals:', len(hypotheses), flush=True)
    prepared = core.prepare(core.IS_END) if hypotheses else {}
    cache, rows, triggers = {}, [], []
    for index, item in enumerate(hypotheses):
        pair, strategy, spec = item['pair'], item['strategy'], item['spec']
        values, dates, features, targets = prepared[pair]
        if (pair, strategy) not in cache:
            base = fixed_execute(values, targets[strategy])
            cache[pair, strategy] = base
        base = cache[pair, strategy]
        filtered, decisions = run_state_filter(values, dates, base, features, spec, pair, strategy)
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
        filtered, decision = run_state_filter(values, dates, base, features, spec, pair, strategy)
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
        core.run_candidate, core.identity = run_state_filter, identity
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
