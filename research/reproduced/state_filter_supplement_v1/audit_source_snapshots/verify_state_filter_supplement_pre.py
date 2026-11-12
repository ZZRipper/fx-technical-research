"""Independent read-only pre/post audit for the fixed JPY supplement.

Pre-OOS mode hashes source workbooks as bytes but never parses future prices or
opens OOS artifacts. Only a separate audit JSON is written; frozen inputs and
the strict-v2 decision remain unchanged. Tests are not rerun by this auditor.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_state_filter_research import Audit, ROOT, sha, same_number
from fx_filter_evaluation import SCREEN, OOS_SCREEN, screen_candidates

IDENTIFIER = 'JPY|BB_MR_20|STATE_efficiency_63_1'
SPEC = {'id': 'STATE_efficiency_63_1', 'feature': 'efficiency_63', 'state': 1}
OOS_FILES = ('oos_status.json', 'oos_daily.csv.gz', 'oos_metrics.csv', 'oos_anniversary_metrics.csv',
             'oos_fresh_start_metrics.csv', 'candidate_assessment.csv', 'frozen_full_trade_ledger.csv.gz',
             'frozen_full_episode_decisions.csv.gz')


class SupplementAudit(Audit):
    def json_path(self, path):
        return json.loads(self.register(path).read_text(encoding='utf-8'))

    def frame_path(self, path):
        return pd.read_csv(self.register(path), float_precision='round_trip')

    def same_frame(self, label, actual, expected, keys):
        actual, expected = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-10)
        except AssertionError as error:
            self.check(label, False, error=str(error))
        else:
            self.check(label, True, rows=len(actual))

    def common(self):
        protocol = self.read_json('protocol.json')
        freeze = self.read_json('frozen_selection.json')
        status = self.read_json('development_status.json')
        diagnostics = self.read_json('supplement_diagnostics.json')
        source = ROOT / protocol['source_campaign']
        old_protocol = self.json_path(source / 'protocol.json')
        old_lock = self.json_path(source / 'validation_locked.json')
        old_freeze = self.json_path(source / 'frozen_selection.json')
        old_status = self.json_path(source / 'development_status.json')
        old_audit = self.json_path(source / 'independent_validation.json')
        for manifest in ('source_sha256', 'input_sha256', 'source_campaign_evidence_sha256'):
            self.check('manifest_agreement:' + manifest, freeze[manifest] == protocol[manifest])
            self.hashes(manifest, protocol[manifest])
        self.hashes('observation', protocol['observation_source_sha256'])
        self.hashes('frozen_IS', freeze['is_output_sha256'], self.campaign)
        self.hashes('original_v2_IS', old_freeze['is_output_sha256'], source)
        self.hashes('original_v2_lock_outputs', old_lock['fit_output_sha256'], source)
        self.check('protocol_freeze_link', sha(self.campaign / 'protocol.json') == freeze['protocol_sha256'] == status['protocol_sha256'])
        self.check('freeze_status_link', sha(self.campaign / 'frozen_selection.json') == status['freeze_sha256'])
        self.check('diagnostics_hash', sha(self.campaign / 'supplement_diagnostics.json') == freeze['supplement_diagnostics_sha256'])
        self.check('v2_freeze_status_link', sha(source / 'frozen_selection.json') == old_status['freeze_sha256'])
        self.check('v2_protocol_lock_links', old_freeze['protocol_sha256'] == old_lock['protocol_sha256'] == sha(source / 'protocol.json')
                   and old_freeze['preconfirmation_lock_sha256'] == sha(source / 'validation_locked.json'))
        self.check('v2_audit_passed', old_audit['status'] == 'passed' and old_audit['checks_passed'] == old_audit['checks_total']
                   and len(old_audit['checks']) == old_audit['checks_total'] and all(r['passed'] for r in old_audit['checks']))
        self.check('v2_audit_source_matches', sha(ROOT / 'verify_state_filter_research.py') == old_audit['audit_script_sha256'])
        self.hashes('audited_test_sources', old_audit['tests']['test_source_sha256'])
        self.hashes('audited_v2_evidence', old_audit['verified_files_sha256'])
        self.check('v2_tests_passed', old_audit['tests']['executed'] and old_audit['tests']['exit_code'] == 0 and old_audit['tests']['tests_run'] == 76)
        self.check('exact_original_SCREEN', protocol['screen'] == old_protocol['screen'] == SCREEN)
        self.check('exact_original_OOS_SCREEN', protocol['oos_screen'] == old_protocol['oos_screen'] == OOS_SCREEN)
        self.check('cost_zero_simple', protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
                   and protocol['accounting'] == 'fixed_initial_capital_simple')
        self.check('dates_unchanged', all(protocol[k] == old_protocol[k] for k in ('is_start', 'is_end', 'oos_start', 'oos_end', 'oos_anniversary_years', 'is_blocks')))
        self.check('original_state_bins', protocol['state_bins'] == old_protocol['state_bins'])
        self.check('sole_designated_scope', protocol['scope'] == {'pairs': ['JPY'], 'strategies': ['BB_MR_20'], 'baseline_accounts': 1}
                   and protocol['designated_candidate_id'] == IDENTIFIER and protocol['candidate_specs'] == [SPEC])
        locked = [r for r in old_lock['candidates'] if r['candidate_id'] == IDENTIFIER]
        self.check('previously_locked_same_spec', len(locked) == 1 and locked[0]['spec'] == SPEC)
        self.check('one_frozen_candidate', len(freeze['candidates']) == freeze['frozen_candidates'] == status['frozen_candidates'] == 1)
        candidate = freeze['candidates'][0]
        self.check('candidate_spec_and_rank_unchanged', candidate['candidate_id'] == IDENTIFIER and candidate['spec'] == SPEC
                   and candidate['preconfirmation_rank'] == locked[0]['preconfirmation_rank'] and candidate['freeze_rank'] == 1)
        self.check('strict_v2_failure_preserved', old_freeze['candidates'] == [] and old_freeze['frozen_candidates'] == old_status['frozen_candidates'] == 0
                   and protocol['strict_v2_pass'] is False and freeze['strict_v2_pass'] is False and candidate['strict_v2_pass'] is False
                   and status['strict_v2_pass'] is False and diagnostics['strict_v2_pass'] is False and diagnostics['strict_v2_failed'] is True)
        self.check('v2_outcomes_not_relabelled', diagnostics['strict_v2_confirmation_failed'] == candidate['strict_v2_confirmation_failed'] == 'outcome_sample'
                   and diagnostics['strict_v2_robustness_failed'] == candidate['strict_v2_robustness_failed'] == 'recent_top3_robust')
        self.check('historical_exposure_disclosed', protocol['prior_holdout_exposure'] == old_protocol['prior_holdout_exposure']
                   and 'not independent untouched OOS' in protocol['prior_holdout_exposure'])
        self.check('supplement_status_passed', status['status'] == 'passed' and status['is_screen_completed'] and status['compute_cutoff'] == '2023-05-29')
        self.check('freeze_precedes_this_audit', pd.Timestamp(old_lock['created_utc']) < pd.Timestamp(protocol['created_utc'])
                   < pd.Timestamp(freeze['created_utc']) < pd.Timestamp.now(tz='UTC'))
        self.check('freeze_records_no_OOS_computation', freeze['oos_evaluated_at_freeze'] is False
                   and status['holdout_returns_computed'] is False and status['oos_evaluated'] is False)
        selection, metrics, daily = self.read_csv('is_selection.csv'), self.read_csv('is_all_metrics.csv'), self.read_csv('is_frozen_daily.csv.gz')
        self.check('sole_candidate_in_IS_outputs', all(set(frame.candidate_id) == {IDENTIFIER} for frame in (selection, metrics, daily)))
        old_metrics = self.frame_path(source / 'confirmation_full_metrics.csv')
        old_daily = self.frame_path(source / 'locked_is_daily.csv.gz')
        old_selection = self.frame_path(source / 'confirmation_selection.csv')
        original = old_selection.loc[old_selection.candidate_id.eq(IDENTIFIER)].iloc[0]
        self.check('v2_saved_original_pass_strict_fail', bool(original.original_is_pass) and not bool(original['pass']) and not bool(original.frozen))
        self.same_frame('copied_IS_metrics_exact', metrics, old_metrics.loc[old_metrics.candidate_id.eq(IDENTIFIER)], ['split', 'variant'])
        self.same_frame('copied_IS_daily_exact', daily, old_daily.loc[old_daily.candidate_id.eq(IDENTIFIER)], ['variant', 'date'])
        self.same_frame('frozen_IS_metrics_subset', self.read_csv('is_frozen_metrics.csv'), metrics.loc[metrics.split.eq('IS')], ['variant'])
        rebuilt = screen_candidates(metrics)
        numeric = [c for c in rebuilt if pd.api.types.is_numeric_dtype(rebuilt[c]) and not pd.api.types.is_bool_dtype(rebuilt[c])]
        self.values('original_SCREEN_rescreen', selection, rebuilt, [c for c in rebuilt if c != 'candidate_id'], numeric=numeric)
        self.check('original_SCREEN_pass_and_freeze', len(rebuilt) == 1 and bool(rebuilt.iloc[0]['pass']) and bool(rebuilt.iloc[0].frozen)
                   and candidate['pass'] is True and candidate['original_is_pass'] is True)
        self.check('IS_only_dates', pd.to_datetime(daily.date).max() == pd.Timestamp('2023-05-29')
                   and pd.to_datetime(metrics.end_date).le('2023-05-29').all())
        self.check('IS_zero_cost_daily', np.array_equal(daily.gross_daily_return, daily.net_daily_return))
        for row in metrics.itertuples(index=False):
            path = daily.loc[daily.variant.eq(row.variant)].sort_values('date')
            part = path.loc[pd.to_datetime(path.date).between(row.start_date, row.end_date)]
            returns = part.gross_daily_return.to_numpy() * 100
            curve = np.r_[0., returns.cumsum()]
            self.check('IS_simple_sum_DD:' + row.split + ':' + row.variant,
                       same_number(returns.sum(), row.gross_simple_return_pct) and same_number((curve - np.maximum.accumulate(curve)).min(), row.gross_max_drawdown_pp)
                       and len(returns) == row.observations)
        self.details.update(candidate_id=IDENTIFIER, spec=SPEC, original_SCREEN_pass=True, strict_v2_pass=False,
                            inherited_tests={'tests_run': old_audit['tests']['tests_run'], 'exit_code': old_audit['tests']['exit_code'],
                                             'rerun': False, 'reason': 'No decision source changed; all audited test-source hashes checked.'},
                            protocol_sha256=sha(self.campaign / 'protocol.json'), freeze_sha256=sha(self.campaign / 'frozen_selection.json'))
        return protocol, freeze, status

    def pre(self):
        self.common()
        self.check('OOS_artifacts_absent_before_opening', all(not (self.campaign / name).exists() for name in OOS_FILES))
        self.details.update(OOS_results_opened=False, raw_workbooks_parsed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pre'], required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'reproduced/state_filter_supplement_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else args.campaign.resolve() / 'pre_oos_independent_audit.json'
    if output.name in {'protocol.json', 'frozen_selection.json', 'development_status.json', 'supplement_diagnostics.json'}:
        raise ValueError('Audit must not overwrite decision evidence.')
    audit = SupplementAudit(args.campaign.resolve())
    started = datetime.now(timezone.utc).isoformat()
    try:
        audit.pre()
    except Exception:
        audit.check('audit_completed', False, error=traceback.format_exc())
    for path, digest in audit.files.items():
        audit.check('read_only_preserved:' + str(Path(path).relative_to(ROOT)), sha(path) == digest)
    passed = sum(row['passed'] for row in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'stage': args.stage,
              'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'audit_script_sha256': sha(__file__), 'checks_passed': passed, 'checks_total': len(audit.checks),
              'verified_file_count': len(audit.files), 'verified_files_sha256': {Path(p).relative_to(ROOT).as_posix(): h for p, h in audit.files.items()},
              'details': audit.details, 'checks': audit.checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': f'{passed}/{len(audit.checks)}', 'files': len(audit.files), 'output': str(output)}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
