"""User-approved expectancy/profit-factor evaluation of fixed v3/v4 hypotheses.

The original campaigns, review and evaluation modules remain immutable. This
campaign changes only the two quality gates explicitly approved by the user.
"""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
import run_holding_filter_research as conditional
import run_timecap_filter_research as capped
from fx_baselines import CURRENCIES, STRATEGIES, fixed_execute
from fx_filter_evaluation import IS_BLOCKS, simple_ledger, fresh_start
from fx_tradeoff_evaluation import (TRADEOFF_SCREEN, TRADEOFF_OOS_SCREEN,
    add_trade_quality, screen_tradeoff_candidates, assess_tradeoff_oos)

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/holding_tradeoff_v8'
REVIEW = ROOT / 'reproduced/holding_quality_tradeoff_review'
CAMPAIGNS = ('holding_filter_v3', 'timecap_filter_v4')
COUNTS = {'holding_filter_v3': 391, 'timecap_filter_v4': 101}
USER_APPROVAL = '允许取舍，但每笔期望收益与总盈利/总亏损比都须改善'
OOS_OUTPUTS = ('oos_metrics.csv', 'oos_anniversary_metrics.csv', 'oos_fresh_start_metrics.csv',
               'candidate_assessment.csv', 'oos_daily.csv.gz', 'frozen_full_trade_ledger.csv.gz',
               'frozen_full_episode_decisions.csv.gz')


def family(spec):
    if spec['id'].startswith('TIME_CAP_'):
        registry, name = capped.registered_specs(), CAMPAIGNS[1]
    elif spec['id'].startswith('CHECK_'):
        registry, name = conditional.registered_specs(), CAMPAIGNS[0]
    else:
        raise ValueError('Unregistered original hypothesis family.')
    if spec not in registry or isinstance(spec.get('checkpoint_age'), bool):
        raise ValueError('Spec differs from the original fixed registry.')
    return name


def identity(pair, strategy, spec):
    name = family(spec)
    if pair not in CURRENCIES or strategy not in STRATEGIES:
        raise ValueError('Unregistered account.')
    origin = capped if name == CAMPAIGNS[1] else conditional
    return {**origin.identity(pair, strategy, spec), 'source_campaign': name}


def run_candidate(prices, dates, base, features, spec, pair='', strategy=''):
    engine = capped.run_holding_filter if family(spec) == CAMPAIGNS[1] else conditional.run_holding_filter
    return engine(prices, dates, base, features, spec, pair, strategy)


def original_pool():
    """Read only the bound old IS hypotheses/review; never historical OOS files."""
    review = core.read_json(REVIEW / 'review.json')
    audit = core.read_json(REVIEW / 'final_review_validation.json')
    if not (audit['status'] == 'passed' and audit['versions'] == 492
            and review['OOS_not_evaluated_for_review'] is True
            and audit['review_json_sha256'] == core.sha(REVIEW / 'review.json')
            and audit['review_csv_sha256'] == review['review_csv_sha256'] == core.sha(REVIEW / 'review.csv')):
        raise ValueError('Saved IS-only review provenance changed.')
    core.assert_hashes(review['input_sha256'])
    inputs = dict(review['input_sha256'])
    for filename in ('review.json', 'review.csv', 'final_review_validation.json'):
        path = REVIEW / filename
        inputs[path.relative_to(ROOT).as_posix()] = core.sha(path)
    hypotheses, old_metrics, old_selections = [], [], []
    for name in CAMPAIGNS:
        path = ROOT / 'reproduced' / name
        protocol = core.read_json(path / 'protocol.json')
        freeze = core.read_json(path / 'frozen_selection.json')
        status = core.read_json(path / 'development_status.json')
        pre = core.read_json(path / 'independent_validation.json')
        if not (status['status'] == pre['status'] == 'passed' and status['oos_evaluated'] is False
                and status['freeze_sha256'] == core.sha(path / 'frozen_selection.json')
                and freeze['protocol_sha256'] == core.sha(path / 'protocol.json')
                and pre['checks_total'] == pre['checks_passed'] == len(pre['checks'])
                and all(row['passed'] for row in pre['checks']) and not freeze['candidates']):
            raise ValueError('Old IS campaign status or audit changed: ' + name)
        for key in ('source_sha256', 'input_sha256'):
            if freeze[key] != protocol[key]:
                raise ValueError('Old source/input manifest disagreement.')
            core.assert_hashes(protocol[key]); inputs.update(protocol[key])
        for filename, digest in freeze['is_output_sha256'].items():
            if core.sha(path / filename) != digest:
                raise ValueError('Old frozen IS output changed: ' + filename)
        for filename, digest in pre['verified_files_sha256'].items():
            if core.sha(Path(filename)) != digest:
                raise ValueError('Old independently audited evidence changed: ' + filename)
        for filename in ('protocol.json', 'frozen_selection.json', 'development_status.json',
                         'independent_validation.json', *freeze['is_output_sha256']):
            source = path / filename
            inputs[source.relative_to(ROOT).as_posix()] = core.sha(source)
        items = core.read_json(path / 'is_hypotheses.json')['candidates']
        if len(items) != COUNTS[name]:
            raise ValueError('Original candidate count changed.')
        for item in items:
            expected = identity(item['pair'], item['strategy'], item['spec'])
            if family(item['spec']) != name or any(item[k] != v for k, v in expected.items() if k != 'source_campaign'):
                raise ValueError('Original hypothesis identity changed.')
            hypotheses.append({**item, 'source_campaign': name})
        old_metrics.append(pd.read_csv(path / 'is_all_metrics.csv', float_precision='round_trip').assign(source_campaign=name))
        old_selections.append(pd.read_csv(path / 'is_selection.csv', float_precision='round_trip').assign(source_campaign=name))
    ids = [x['candidate_id'] for x in hypotheses]
    saved_review = pd.read_csv(REVIEW / 'review.csv', float_precision='round_trip')
    if len(ids) != len(set(ids)) or set(ids) != set(saved_review.candidate_id):
        raise ValueError('The complete fixed review universe must be preserved.')
    return hypotheses, pd.concat(old_metrics, ignore_index=True), pd.concat(old_selections, ignore_index=True), saved_review, inputs


def compare_old_metrics(actual, previous):
    keys = ['candidate_id', 'variant', 'split']
    a, b = [frame.sort_values(keys).reset_index(drop=True) for frame in (actual, previous)]
    if len(a) != len(b) or a[keys].to_dict('records') != b[keys].to_dict('records'):
        raise AssertionError('Original IS metric keys changed.')
    for name in b:
        if pd.api.types.is_numeric_dtype(b[name]):
            if not np.allclose(a[name].to_numpy(float), b[name].to_numpy(float), equal_nan=True, rtol=1e-10, atol=1e-8):
                raise AssertionError('Original IS numeric metric changed: ' + name)
        elif not a[name].fillna('').astype(str).eq(b[name].fillna('').astype(str)).all():
            raise AssertionError('Original IS metric metadata changed: ' + name)


def develop(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve existing campaign; choose a new empty directory.')
    hypotheses, old_metrics, old_selection, review, inputs = original_pool()
    protocol = core.protocol_document()
    sources = {**core.source_hashes(), **{name: core.sha(ROOT / name) for name in
        ('run_holding_tradeoff_research.py', 'src/fx_tradeoff_evaluation.py',
         'run_holding_filter_research.py', 'run_timecap_filter_research.py',
         'src/fx_holding_filters.py', 'src/fx_timecap_filters.py')}}
    specs = {x['filter_id']: x['spec'] for x in hypotheses}
    protocol.update(version='holding_tradeoff_v8', created_utc=core.now(),
        user_authorization={'instruction_zh': USER_APPROVAL, 'recorded_utc': core.now(),
            'scope': 'Replace the two quality requirements for the fixed 492 previously discussed v3/v4 candidates; retain every other original gate and ranking.'},
        screen=TRADEOFF_SCREEN, oos_screen=TRADEOFF_OOS_SCREEN,
        legacy_comparator_ids=[], candidate_specs=list(specs.values()),
        source_campaigns=list(CAMPAIGNS), original_candidate_counts=COUNTS,
        hypothesis_origin='Exactly the saved 391 conditional checkpoint and 101 unconditional cap D-only hypotheses. Existing IS-only review predates this authorization. No new hypotheses, ages, conditions or bin fitting.',
        proposal_rule='Reuse the entire original v3/v4 proposal lists including exact original specs and D1/D2/D evidence. Both D halves have >=10 closed actionable episodes and negative executable tails; combined D negative. Concentration remains diagnostic.',
        family_application='All 180 original accounts retained. Rank the fixed 492 versions together using the original rank; one highest-ranked passing version per account.',
        checkpoint_rule='Unchanged original checkpoint engines: earned-return age, signal-close exit, next-close execution retaining that return, natural exits first, no reentry within original episode.',
        quality_rule='Expectancy and gross profit factor must both be finite for BASE/FILTER and strictly improve (>0 delta, no fitted minimum). Zero loss denominator is undefined; zero-win numerator is zero. OOS quality additionally requires >=20 eligible closed FILTER trades.',
        criterion_change='Explicit user-approved change from separate win-rate/payoff gates to expectancy/profit-factor gates. Old failures remain old failures. Original win/payoff results remain diagnostics.',
        prior_holdout_exposure='Historical 2023-05-30 to 2026-05-29 observations have already been seen in other campaigns. This is repeated historical verification, not untouched independent evidence. The original v3/v4 campaigns stopped with no IS-frozen candidates and no OOS evaluation.',
        schedule_delay_sensitivity='No extra schedule-delay experiment.',
        holdout_rule='Freeze all passing per-account winners under the approved new quality rule and unchanged original ranking before OOS. Evaluate every frozen candidate once; no OOS-driven revisions, substitutions, deletions, or re-ranking.',
        source_sha256=sources, input_sha256={**core.input_hashes(), **inputs})
    output.mkdir(parents=True, exist_ok=True)
    core.save_json(output / 'protocol.json', protocol)
    core.save_json(output / 'is_hypotheses.json', {'created_utc': core.now(), 'candidates': hypotheses})
    status = {'status': 'running', 'started_utc': core.now(), 'compute_cutoff': core.IS_END, 'oos_evaluated': False}
    core.save_json(output / 'development_status.json', status)
    prepared = core.prepare(core.IS_END)
    cache, rows, decisions_all = {}, [], []
    for i, item in enumerate(hypotheses, 1):
        pair, strategy, spec = item['pair'], item['strategy'], item['spec']
        values, dates, features, targets = prepared[pair]
        if (pair, strategy) not in cache:
            cache[pair, strategy] = fixed_execute(values, targets[strategy])
        base = cache[pair, strategy]
        filtered, decisions = run_candidate(values, dates, base, features, spec, pair, strategy)
        key = identity(pair, strategy, spec)
        for variant, run in [('BASE', base), ('FILTER', filtered)]:
            rows.extend(core.metric_rows(key, run, dates, [('IS', core.IS_START, core.IS_END), *IS_BLOCKS], variant))
        decisions_all.append(decisions.assign(**key))
        if i % 100 == 0:
            print('IS replay:', i, '/', len(hypotheses), flush=True)
    metrics = add_trade_quality(core.add_retention(pd.DataFrame(rows)))
    compare_old_metrics(metrics, old_metrics)
    selection = screen_tradeoff_candidates(metrics)
    observed = selection.set_index('candidate_id'); prior = review.set_index('candidate_id')
    for name, original in [('pass', 'diagnostic_alternative_criteria_satisfied'), ('frozen', 'diagnostic_one_per_account_suggested')]:
        if not observed[name].eq(prior.loc[observed.index, original]).all():
            raise AssertionError('Approved screen no longer agrees with the saved review: ' + name)
    frozen = selection.loc[selection.frozen].sort_values('freeze_rank')
    review_order = review.loc[review.diagnostic_one_per_account_suggested].sort_values('diagnostic_suggestion_order').candidate_id.tolist()
    if frozen.candidate_id.tolist() != review_order:
        raise AssertionError('Cross-family rank differs from the unchanged review rank.')
    counts = pd.Series([x['pair'] + '|' + x['strategy'] for x in hypotheses]).value_counts()
    pd.DataFrame([{'pair': p, 'strategy': s, 'candidate_count': int(counts.get(p+'|'+s, 0)), 'compute_end': core.IS_END}
                  for p in CURRENCIES for s in STRATEGIES]).to_csv(output / 'is_account_coverage.csv', index=False)
    metrics.to_csv(output / 'is_all_metrics.csv', index=False)
    selection.to_csv(output / 'is_selection.csv', index=False)
    old_selection.to_csv(output / 'is_original_selection.csv', index=False)
    pd.concat(decisions_all, ignore_index=True).to_csv(output / 'is_all_decisions.csv.gz', index=False)
    daily, main, ledgers, decisions_out = [], [], [], []
    for row in frozen.to_dict('records'):
        pair, strategy, spec = row['pair'], row['strategy'], specs[row['filter_id']]
        values, dates, features, targets = prepared[pair]; base = cache[pair, strategy]
        filtered, decisions = run_candidate(values, dates, base, features, spec, pair, strategy)
        key = identity(pair, strategy, spec)
        for variant, run in [('BASE', base), ('FILTER', filtered)]:
            daily.append(core.daily_rows(key, run, dates, [('IS', core.IS_START, core.IS_END)], variant))
            main.extend(core.metric_rows(key, run, dates, [('IS', core.IS_START, core.IS_END)], variant))
            ledgers.append(simple_ledger(run, dates).assign(**key, variant=variant))
        decisions_out.append(decisions.assign(**key))
    if not len(frozen):
        raise AssertionError('The approved fixed review should retain its IS-qualified candidates.')
    pd.concat(daily, ignore_index=True).to_csv(output / 'is_frozen_daily.csv.gz', index=False)
    add_trade_quality(core.add_retention(pd.DataFrame(main))).to_csv(output / 'is_frozen_metrics.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(output / 'is_frozen_trades.csv.gz', index=False)
    pd.concat(decisions_out, ignore_index=True).to_csv(output / 'is_frozen_decisions.csv.gz', index=False)
    core.assert_hashes(protocol['source_sha256']); core.assert_hashes(protocol['input_sha256'])
    records = [{**r, 'spec': specs[r['filter_id']]} for r in frozen.to_dict('records')]
    core.save_json(output / 'frozen_selection.json', {'created_utc': core.now(), 'protocol_sha256': core.sha(output / 'protocol.json'),
        'selection_basis': protocol['holdout_rule'], 'oos_evaluated_at_freeze': False,
        'source_sha256': protocol['source_sha256'], 'input_sha256': protocol['input_sha256'],
        'is_output_sha256': {p.name: core.sha(p) for p in sorted(output.glob('is_*')) if p.is_file()},
        'candidates': records, 'all_candidate_accounts_tested': len(selection),
        'all_passing_versions': int(selection['pass'].sum()), 'frozen_candidates': len(records)})
    status.update(status='passed', finished_utc=core.now(), is_screen_completed=True, source_integrity_pass=True,
        accounts_covered=180, candidates_evaluated=len(selection), all_passing_versions=int(selection['pass'].sum()),
        frozen_candidates=len(records), freeze_sha256=core.sha(output / 'frozen_selection.json'), holdout_returns_computed=False,
        all_old_IS_metrics_match=True, approved_review_selection_and_order_match=True)
    core.save_json(output / 'development_status.json', status)
    print('Approved IS screen:', int(selection['pass'].sum()), 'passes;', len(records), 'frozen.', flush=True)


def oos(output):
    protocol, freeze = core.read_json(output / 'protocol.json'), core.read_json(output / 'frozen_selection.json')
    status = core.read_json(output / 'development_status.json')
    pre = core.read_json(output / 'independent_validation.json')
    if not (status['status'] == pre['status'] == 'passed' and pre['checks_total'] == pre['checks_passed']
            and all(c['passed'] for c in pre['checks'])
            and status['freeze_sha256'] == core.sha(output / 'frozen_selection.json')
            and freeze['protocol_sha256'] == core.sha(output / 'protocol.json')
            and protocol['screen'] == TRADEOFF_SCREEN and protocol['oos_screen'] == TRADEOFF_OOS_SCREEN
            and freeze['created_utc'] < pre['started_utc'] <= pre['finished_utc'] < core.now()
            and len(freeze['candidates']) == status['frozen_candidates'] == freeze['frozen_candidates']):
        raise ValueError('A verified, unchanged authorized IS freeze is required.')
    for key in ('source_sha256', 'input_sha256'):
        core.assert_hashes(freeze[key])
    for name, digest in freeze['is_output_sha256'].items():
        if core.sha(output / name) != digest:
            raise ValueError('Frozen IS output changed: ' + name)
    for name, digest in pre['verified_files_sha256'].items():
        if core.sha(Path(name)) != digest:
            raise ValueError('Pre-audit evidence changed: ' + name)
    if (output / 'oos_status.json').exists():
        raise ValueError('OOS was already opened; preserve its record and do not replay.')
    status = {'status': 'running', 'opened_utc': core.now(), 'freeze_sha256': core.sha(output / 'frozen_selection.json'),
        'protocol_sha256': core.sha(output / 'protocol.json'), 'preaudit_sha256': core.sha(output / 'independent_validation.json'),
        'oos_evaluated': False, 'freeze_created_before_oos': True, 'candidate_count': len(freeze['candidates']), 'retuning_permitted': False}
    core.save_json(output / 'oos_status.json', status)
    prepared = core.prepare(core.OOS_END)
    frozen_daily = pd.read_csv(output / 'is_frozen_daily.csv.gz', parse_dates=['date'])
    daily, rows, annual, fresh, ledgers, decisions_all = [], [], [], [], [], []
    for row in freeze['candidates']:
        pair, strategy, spec = row['pair'], row['strategy'], row['spec']
        print('Frozen OOS:', int(row['freeze_rank']), row['candidate_id'], flush=True)
        values, dates, features, targets = prepared[pair]
        base = fixed_execute(values, targets[strategy])
        filtered, decisions = run_candidate(values, dates, base, features, spec, pair, strategy)
        key = identity(pair, strategy, spec)
        for variant, run in [('BASE', base), ('FILTER', filtered)]:
            core.verify_is_prefix(frozen_daily, key, run, dates, variant)
            daily.append(core.daily_rows(key, run, dates, [('OOS', core.OOS_START, core.OOS_END)], variant))
            rows.extend(core.metric_rows(key, run, dates, [('OOS', core.OOS_START, core.OOS_END)], variant))
            annual.extend(core.metric_rows(key, run, dates, core.OOS_YEARS, variant))
            fresh.extend(core.metric_rows(key, fresh_start(base, run, dates, core.OOS_START), dates,
                                          [('OOS_FRESH_START', core.OOS_START, core.OOS_END)], variant))
            ledgers.append(simple_ledger(run, dates).assign(**key, variant=variant))
        decisions_all.append(decisions.assign(**key))
    metrics, anniversary, sensitivity = [add_trade_quality(core.add_retention(pd.DataFrame(x))) for x in (rows, annual, fresh)]
    assessments = []
    for row in freeze['candidates']:
        sub = metrics.loc[metrics.candidate_id.eq(row['candidate_id'])].set_index('variant')
        years = anniversary.loc[anniversary.candidate_id.eq(row['candidate_id']) & anniversary.variant.eq('FILTER')]
        result = assess_tradeoff_oos(sub.loc['BASE'], sub.loc['FILTER'], years)
        assessments.append({**{k: row[k] for k in ('candidate_id', 'pair', 'strategy', 'filter_id', 'freeze_rank', 'source_campaign')},
                            'is_return_pct': row['is_gross_return_pct'], **result})
    assessment = pd.DataFrame(assessments).sort_values('freeze_rank')
    metrics = metrics.merge(assessment[['candidate_id', 'oos_curve_pass', 'oos_quality_confirmed', 'original_oos_quality_confirmed']],
                            on='candidate_id', validate='many_to_one')
    for name, frame in [('oos_metrics.csv', metrics), ('oos_anniversary_metrics.csv', anniversary),
                        ('oos_fresh_start_metrics.csv', sensitivity), ('candidate_assessment.csv', assessment),
                        ('oos_daily.csv.gz', pd.concat(daily, ignore_index=True)),
                        ('frozen_full_trade_ledger.csv.gz', pd.concat(ledgers, ignore_index=True)),
                        ('frozen_full_episode_decisions.csv.gz', pd.concat(decisions_all, ignore_index=True))]:
        frame.to_csv(output / name, index=False)
    core.assert_hashes(freeze['source_sha256']); core.assert_hashes(freeze['input_sha256'])
    status.update(status='passed', finished_utc=core.now(), oos_evaluated=True, source_integrity_pass=True,
        is_prefix_positions_and_returns_match=True, is_screen_completed=True,
        curve_passing_candidates=int(assessment.oos_curve_pass.sum()), quality_confirmed_candidates=int(assessment.oos_quality_confirmed.sum()),
        both_curve_and_quality_candidates=int((assessment.oos_curve_pass & assessment.oos_quality_confirmed).sum()),
        output_sha256={name: core.sha(output / name) for name in OOS_OUTPUTS})
    core.save_json(output / 'oos_status.json', status)
    print(assessment.to_string(index=False), flush=True)


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
        from build_holding_tradeoff_report import build_report
        build_report(output)


if __name__ == '__main__':
    main()
