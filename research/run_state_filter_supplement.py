"""Separate original-standard evaluation of one already locked v2 candidate.

Preparation alone does not freeze anything or compute OOS returns. Explicit stages:
  python -B run_state_filter_supplement.py --stage freeze
  python -B run_state_filter_supplement.py --stage oos
  python -B run_state_filter_supplement.py --stage report

The strict v2 result remains a failure. This supplement keeps its original rule,
uses the original SCREEN/OOS_SCREEN, and discloses the extra v2 diagnostics and
previous exposure to the historical holdout. It cannot select a replacement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
import run_state_filter_research as state
from fx_filter_evaluation import SCREEN, OOS_SCREEN, IS_BLOCKS, screen_candidates
from fx_state_filters import run_state_filter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reproduced/state_filter_v2'
DEFAULT = ROOT / 'reproduced/state_filter_supplement_v1'
CANDIDATE_ID = 'JPY|BB_MR_20|STATE_efficiency_63_1'
EXPECTED_SPEC = {'id': 'STATE_efficiency_63_1', 'feature': 'efficiency_63', 'state': 1}
NOTICE = (
    '本目录是原综合标准下的独立补充分析，只评估事先指定、已在 v2 锁定的 JPY BB_MR_20 候选。'
    '严格 v2 仍未通过：确认期结果样本不足，近期改善在剔除最有帮助的三个原交易段后不稳健。'
    '集中度与确认期亏损笔数在此作为披露诊断，不改变 v2 结论，也不改变过滤状态或阈值。'
    '2023—2026 历史 OOS 曾在前序研究中暴露；本次不是全新、未见样本。'
)
IS_EVIDENCE_NAMES = (
    'confirmation_selection.csv', 'confirmation_full_metrics.csv',
    'confirmation_recent_metrics.csv', 'confirmation_effect_evidence.json',
    'locked_is_daily.csv.gz',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assert_manifest(manifest, label):
    """Hash bytes only; workbook outcomes are not parsed by preparation."""
    require(isinstance(manifest, dict) and bool(manifest), label + ' manifest is empty.')
    for name, digest in manifest.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), label + ' path escapes the repository.')
        require(path.is_file() and core.sha(path) == digest, label + ' changed: ' + name)


def output_path(output):
    output = Path(output).resolve()
    require(output.parent == ROOT / 'reproduced' and output.name.startswith('state_filter_supplement'),
            'Use a separate reproduced/state_filter_supplement* directory.')
    return output


def verify_v2():
    """Read-only validation of the source campaign and its independent audit."""
    protocol, lock = state.load_locked(SOURCE)
    freeze = core.read_json(SOURCE / 'frozen_selection.json')
    status = core.read_json(SOURCE / 'development_status.json')
    audit = core.read_json(SOURCE / 'independent_validation.json')
    require(protocol['screen'] == SCREEN and protocol['oos_screen'] == OOS_SCREEN,
            'Original SCREEN/OOS_SCREEN changed.')
    require(protocol['cost_bp'] == protocol['reference_cost_bp'] == core.COST_BP == 0,
            'The supplement uses the original zero-cost accounting.')
    require((protocol['is_start'], protocol['is_end'], protocol['oos_start'], protocol['oos_end']) ==
            (core.IS_START, core.IS_END, core.OOS_START, core.OOS_END), 'Research boundaries changed.')
    require(status['status'] == 'passed' and status['is_screen_completed'] is True and
            status['compute_cutoff'] == core.IS_END, 'v2 IS processing is incomplete.')
    require(core.sha(SOURCE / 'frozen_selection.json') == status['freeze_sha256'], 'v2 freeze changed.')
    require(freeze['protocol_sha256'] == core.sha(SOURCE / 'protocol.json') and
            freeze['preconfirmation_lock_sha256'] == core.sha(SOURCE / 'validation_locked.json'),
            'v2 protocol/lock link changed.')
    for record in (lock, freeze):
        require(record['source_sha256'] == protocol['source_sha256'] and
                record['input_sha256'] == protocol['input_sha256'], 'v2 manifests disagree.')
    require(freeze['frozen_candidates'] == status['frozen_candidates'] == 0 and
            freeze['candidates'] == [], 'Expected strict v2 zero-survivor record changed.')
    require(freeze['oos_evaluated_at_freeze'] is False and status['oos_evaluated'] is False,
            'Source v2 OOS state changed; review separately.')
    for name, digest in freeze['is_output_sha256'].items():
        require(Path(name).name == name and core.sha(SOURCE / name) == digest,
                'v2 final IS evidence changed: ' + name)
    require(audit['status'] == 'passed' and audit['checks_total'] > 0 and
            audit['checks_passed'] == audit['checks_total'] and
            len(audit['checks']) == audit['checks_total'] and
            all(item['passed'] is True for item in audit['checks']), 'Independent audit did not pass.')
    require(core.sha(ROOT / 'verify_state_filter_research.py') == audit['audit_script_sha256'],
            'Independent audit source changed.')
    tests = audit['tests']
    require(tests['executed'] is True and tests['exit_code'] == 0 and tests['tests_run'] > 0,
            'Independent audit did not include passing tests.')
    assert_manifest(tests['test_source_sha256'], 'Audited test source')
    assert_manifest(audit['verified_files_sha256'], 'Independent audit evidence')
    require(len(audit['verified_files_sha256']) == audit['verified_file_count'], 'Audit file count disagrees.')
    for name in IS_EVIDENCE_NAMES:
        rel = (SOURCE / name).relative_to(ROOT).as_posix()
        require(rel in audit['verified_files_sha256'], 'Required IS evidence was not independently audited: ' + name)
    matches = [r for r in lock['candidates'] if r['candidate_id'] == CANDIDATE_ID]
    require(len(matches) == 1 and matches[0]['spec'] == EXPECTED_SPEC,
            'The explicitly designated original locked spec is missing or changed.')
    require(matches[0]['discovery_pass'] is True and matches[0]['validation_pass'] is True,
            'The designated candidate did not pass preconfirmation phases.')
    return protocol, lock, freeze, audit, deepcopy(matches[0])


def selected_csv(name):
    """Only the predesignated ID is retained; no competing candidate is ranked."""
    frames = []
    for chunk in pd.read_csv(SOURCE / name, chunksize=20000):
        frames.append(chunk.loc[chunk.candidate_id.eq(CANDIDATE_ID)].copy())
    result = pd.concat(frames, ignore_index=True)
    require(not result.empty, 'Missing designated candidate in ' + name)
    return result


def validate_daily_metrics(daily, metrics):
    """Check copies against saved metrics, preserving actual-day boundary PnL."""
    require(set(daily.candidate_id) == set(metrics.candidate_id) == {CANDIDATE_ID}, 'Candidate substitution.')
    require(set(daily.split) == {'IS'} and set(daily.variant) == {'BASE', 'FILTER'}, 'Unexpected daily scope.')
    require(not daily.duplicated(['variant', 'date']).any(), 'Duplicate IS daily rows.')
    require(daily.date.between(core.IS_START, core.IS_END).all(), 'Post-IS daily observation.')
    require(np.isfinite(daily[['gross_daily_return', 'net_daily_return']].to_numpy(float)).all(), 'Nonfinite IS return.')
    require(np.allclose(daily.gross_daily_return, daily.net_daily_return, atol=1e-14, rtol=0), 'Nonzero cost detected.')
    require(daily.executed.isin([-1, 0, 1]).all() and daily.held.isin([-1, 0, 1]).all(), 'Invalid IS positions.')
    periods = [('IS', core.IS_START, core.IS_END), *IS_BLOCKS]
    expected = {(name, variant) for name, _, _ in periods for variant in ('BASE', 'FILTER')}
    require(not metrics.duplicated(['split', 'variant']).any() and
            set(metrics[['split', 'variant']].itertuples(index=False, name=None)) == expected,
            'Expected exactly both variants and four original IS periods.')
    calendars = [pd.DatetimeIndex(daily.loc[daily.variant.eq(v)].sort_values('date').date)
                 for v in ('BASE', 'FILTER')]
    require(calendars[0].equals(calendars[1]), 'BASE/FILTER IS calendars differ.')
    require(str(calendars[0][0].date()) == core.IS_START and
            str(calendars[0][-1].date()) == core.IS_END, 'Incomplete saved IS calendar.')
    for variant in ('BASE', 'FILTER'):
        path = daily.loc[daily.variant.eq(variant)].sort_values('date')
        require(np.array_equal(path.held.to_numpy()[1:], path.executed.to_numpy()[:-1]), 'Position timing changed.')
        for split, start, end in periods:
            values = path.loc[path.date.between(start, end), 'gross_daily_return'].to_numpy(float)
            row = metrics.loc[metrics.split.eq(split) & metrics.variant.eq(variant)].iloc[0]
            curve = np.r_[0., np.cumsum(values) * 100.]
            require(int(row.observations) == len(values) and
                    np.isclose(values.sum() * 100, row.gross_simple_return_pct, atol=1e-10, rtol=0) and
                    np.isclose(np.min(curve - np.maximum.accumulate(curve)), row.gross_max_drawdown_pp, atol=1e-10, rtol=0),
                    'Saved IS path/metric mismatch: ' + split + ' ' + variant)


def load_is_evidence(locked):
    saved = selected_csv('confirmation_selection.csv')
    require(len(saved) == 1, 'Duplicate designated confirmation row.')
    saved_row = saved.iloc[0].to_dict()
    require(saved_row['original_is_pass'] == True and saved_row['pass'] == False and
            saved_row['frozen'] == False, 'Original standard or strict v2 status changed.')
    metrics = selected_csv('confirmation_full_metrics.csv')
    daily = selected_csv('locked_is_daily.csv.gz')
    daily['date'] = pd.to_datetime(daily.date)
    validate_daily_metrics(daily, metrics)
    selection = screen_candidates(metrics)
    require(len(selection) == 1 and selection.iloc[0]['pass'] == True and
            selection.iloc[0].freeze_rank == 1, 'The original SCREEN no longer passes.')
    # Check original gates and reported IS statistics against the audited selection.
    for field in selection:
        if field in {'pass', 'frozen', 'freeze_rank'}:
            continue
        actual, expected = selection.iloc[0][field], saved_row[field]
        if pd.isna(actual) or pd.isna(expected):
            require((pd.isna(actual) and pd.isna(expected)) or field == 'failed_criteria' and
                    actual == '' and pd.isna(expected), 'Selection field changed: ' + field)
        elif isinstance(actual, (float, np.floating)):
            require(np.isclose(actual, expected, atol=1e-10, rtol=0), 'Selection statistic changed: ' + field)
        else:
            require(actual == expected, 'Selection field changed: ' + field)
    selection['filter_name'] = locked['filter_name']
    selection['original_is_pass'] = True
    selection['strict_v2_pass'] = False
    selection['strict_v2_confirmation_failed'] = saved_row['confirmation_failed']
    selection['strict_v2_robustness_failed'] = saved_row['robustness_failed']
    selection['preconfirmation_rank'] = locked['preconfirmation_rank']
    recent = selected_csv('confirmation_recent_metrics.csv')
    effects = [r for r in core.read_json(SOURCE / 'confirmation_effect_evidence.json') if r['candidate_id'] == CANDIDATE_ID]
    require(len(effects) == 1, 'Missing/duplicate original episode diagnostics.')
    diagnostics = {
        'candidate_id': CANDIDATE_ID, 'notice_zh': NOTICE, 'original_is_pass': True,
        'strict_v2_failed': True, 'strict_v2_pass': False, 'strict_v2_frozen_candidates': 0,
        'strict_v2_confirmation_failed': saved_row['confirmation_failed'],
        'strict_v2_robustness_failed': saved_row['robustness_failed'],
        'confirmation_metrics': recent.loc[recent.split.eq('CONFIRMATION')].to_dict('records'),
        'recent_metrics': recent.loc[recent.split.eq('RECENT')].to_dict('records'),
        'effects': effects[0], 'diagnostic_only': True,
    }
    return selection, metrics, daily, diagnostics


def freeze(output):
    output = output_path(output)
    require(not output.exists() or not any(output.iterdir()), 'Supplement directory already contains a record; never replace it.')
    original, lock, strict_freeze, audit, locked = verify_v2()
    selection, metrics, daily, diagnostics = load_is_evidence(locked)
    evidence_hashes = dict(audit['verified_files_sha256'])
    for name in ['independent_validation.json', 'protocol.json', 'validation_locked.json', 'frozen_selection.json', 'development_status.json']:
        evidence_hashes[(SOURCE / name).relative_to(ROOT).as_posix()] = core.sha(SOURCE / name)
    protocol = deepcopy(original)
    protocol.update(
        version='state_filter_supplement_v1', created_utc=core.now(),
        candidate_specs=[deepcopy(locked['spec'])], designated_candidate_id=CANDIDATE_ID,
        scope={'pairs': ['JPY'], 'strategies': ['BB_MR_20'], 'baseline_accounts': 1},
        original_search_scope=original['scope'], supplement=True, notice_zh=NOTICE,
        strict_v2_result_unchanged=True, strict_v2_pass=False,
        strict_v2_temporal_protocol=protocol.pop('temporal_protocol'),
        source_campaign=SOURCE.relative_to(ROOT).as_posix(), source_campaign_evidence_sha256=evidence_hashes,
        source_independent_audit={'status': audit['status'], 'checks_passed': audit['checks_passed'],
                                  'checks_total': audit['checks_total'], 'tests_run': audit['tests']['tests_run']},
        supplement_reason='Evaluate the explicitly designated, previously locked original-SCREEN passer separately. Extra v2 concentration and confirmation outcome-count gates remain disclosed diagnostics; strict v2 is not relabelled.',
        development_access='Use only independently audited, saved full-IS daily paths and metrics for the single designated candidate. No new threshold, fallback, ranking or OOS return is used for this supplemental freeze.',
        family_application='One explicitly designated v2-locked candidate; original rule and state bin unchanged.',
        holdout_rule='Freeze the sole designated original-SCREEN passer at rank 1. Evaluate it under unchanged OOS_SCREEN; retain every outcome and diagnostic. No fallback, retuning or replacement.',
        schedule_delay_sensitivity='No additional IS delay experiment in this supplement.',
    )
    protocol['source_sha256'].update({
        'run_state_filter_supplement.py': core.sha(__file__),
        'verify_state_filter_research.py': audit['audit_script_sha256'],
        **audit['tests']['test_source_sha256'],
    })
    assert_manifest(evidence_hashes, 'Source campaign evidence')
    output.mkdir(parents=True, exist_ok=True)
    # Protocol is committed before any supplement selection artifact or OOS computation.
    core.save_json(output / 'protocol.json', protocol)
    selection.to_csv(output / 'is_selection.csv', index=False)
    metrics.to_csv(output / 'is_all_metrics.csv', index=False)
    metrics.loc[metrics.split.eq('IS')].to_csv(output / 'is_frozen_metrics.csv', index=False)
    daily.to_csv(output / 'is_frozen_daily.csv.gz', index=False)
    core.save_json(output / 'supplement_diagnostics.json', diagnostics)
    assert_manifest(protocol['source_sha256'], 'Supplement source')
    assert_manifest(evidence_hashes, 'Source campaign evidence')
    record = {**selection.iloc[0].to_dict(), 'spec': deepcopy(locked['spec'])}
    frozen = {
        'created_utc': core.now(), 'protocol_sha256': core.sha(output / 'protocol.json'),
        'selection_basis': protocol['holdout_rule'], 'supplement': True, 'strict_v2_pass': False,
        'oos_evaluated_at_freeze': False, 'is_end': core.IS_END, 'oos_start': core.OOS_START,
        'source_sha256': protocol['source_sha256'], 'input_sha256': protocol['input_sha256'],
        'source_campaign_evidence_sha256': evidence_hashes,
        'is_output_sha256': {p.name: core.sha(p) for p in sorted(output.glob('is_*'))},
        'supplement_diagnostics_sha256': core.sha(output / 'supplement_diagnostics.json'),
        'candidates': [record], 'all_candidate_accounts_tested': 1,
        'all_passing_versions': 1, 'frozen_candidates': 1,
    }
    core.save_json(output / 'frozen_selection.json', frozen)
    core.save_json(output / 'development_status.json', {
        'status': 'passed', 'finished_utc': core.now(), 'is_screen_completed': True,
        'source_integrity_pass': True, 'accounts_covered': 1, 'candidates_evaluated': 1,
        'all_passing_versions': 1, 'frozen_candidates': 1, 'supplement': True, 'strict_v2_pass': False,
        'freeze_sha256': core.sha(output / 'frozen_selection.json'),
        'protocol_sha256': core.sha(output / 'protocol.json'), 'compute_cutoff': core.IS_END,
        'holdout_returns_computed': False, 'oos_evaluated': False,
    })
    print('Supplement frozen:', CANDIDATE_ID, '; strict v2 result remains failed; OOS not computed.')


def validate_frozen(output):
    output = output_path(output)
    protocol = core.read_json(output / 'protocol.json')
    frozen = core.read_json(output / 'frozen_selection.json')
    status = core.read_json(output / 'development_status.json')
    require(protocol['version'] == 'state_filter_supplement_v1' and protocol['supplement'] is True and
            protocol['screen'] == SCREEN and protocol['oos_screen'] == OOS_SCREEN, 'Invalid supplement protocol.')
    require(core.sha(output / 'protocol.json') == frozen['protocol_sha256'] == status['protocol_sha256'] and
            core.sha(output / 'frozen_selection.json') == status['freeze_sha256'], 'Supplement freeze/protocol changed.')
    require(status['status'] == 'passed' and status['is_screen_completed'] is True and
            status['compute_cutoff'] == core.IS_END and status['frozen_candidates'] == 1,
            'Valid completed supplemental freeze is required.')
    require(frozen['oos_evaluated_at_freeze'] is False and frozen['frozen_candidates'] == 1 and
            len(frozen['candidates']) == 1, 'Supplement frozen membership changed.')
    row = frozen['candidates'][0]
    require(row['candidate_id'] == CANDIDATE_ID and row['spec'] == EXPECTED_SPEC and row['freeze_rank'] == 1 and
            row['pass'] is True and row['original_is_pass'] is True and row['strict_v2_pass'] is False,
            'Supplement candidate/spec/status changed.')
    for manifest in ('source_sha256', 'input_sha256', 'source_campaign_evidence_sha256'):
        require(frozen[manifest] == protocol[manifest], 'Supplement manifest disagreement.')
        assert_manifest(frozen[manifest], manifest)
    for name, digest in frozen['is_output_sha256'].items():
        require(Path(name).name == name and core.sha(output / name) == digest, 'Frozen IS evidence changed: ' + name)
    require(core.sha(output / 'supplement_diagnostics.json') == frozen['supplement_diagnostics_sha256'],
            'Supplement diagnostics changed.')
    return protocol, frozen, status


def oos(output):
    output = output_path(output)
    validate_frozen(output)
    original_engine, original_key = core.run_candidate, core.identity
    try:
        core.run_candidate, core.identity = run_state_filter, state.key
        core.evaluate_oos(output)
    finally:
        core.run_candidate, core.identity = original_engine, original_key


def report(output):
    output = output_path(output)
    validate_frozen(output)
    # Separate renderer makes the supplement and strict-v2 failure conspicuous.
    from build_state_filter_supplement_report import build_report
    build_report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze', 'oos', 'report'], required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    {'freeze': freeze, 'oos': oos, 'report': report}[args.stage](args.output)


if __name__ == '__main__':
    main()
