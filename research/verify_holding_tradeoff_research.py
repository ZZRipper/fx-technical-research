"""Independent pre-OOS audit of the authorized fixed v3/v4 quality tradeoff.

The old passed audits establish the unchanged D-only proposal evidence. This
audit checks their hashes, independently rebuilds every BASE/checkpoint path
from IS prices and the original BASE ledger, and reapplies the approved gates.
No strategy/filter engine or OOS result is used. Only the new audit JSON is
written. Functions here are also suitable for a separate, later post auditor.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import itertools
import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_entry_interaction_research import InteractionAudit, read_IS_quotes
from verify_state_filter_research import ROOT, sha, same_number
from verify_state_filter_supplement import ledger_from_executed, window_from_path
from fx_baselines import CURRENCIES, STRATEGIES
from fx_filter_evaluation import SCREEN, IS_BLOCKS, OOS_SCREEN

TOL = 1e-10
CAMPAIGNS = ('holding_filter_v3', 'timecap_filter_v4')
COUNTS = {'holding_filter_v3': 391, 'timecap_filter_v4': 101}
REVIEW = ROOT / 'reproduced/holding_quality_tradeoff_review'
BASE_LEDGER = ROOT / 'reproduced/holding_observations_v3/is_base_trade_ledger.csv.gz'
ACTIVE_GATES = ('positive_and_annualized_pass', 'return_improvement_pass',
                'expectancy_pass', 'profit_factor_pass', 'drawdown_pass',
                'return_to_drawdown_pass', 'sample_pass', 'block_positive_pass',
                'block_improvement_pass')
ORIGINAL_GATES = ('positive_and_annualized_pass', 'return_improvement_pass',
                  'win_rate_pass', 'payoff_pass', 'drawdown_pass',
                  'return_to_drawdown_pass', 'sample_pass', 'block_positive_pass',
                  'block_improvement_pass')
IDENTITY = ('candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'source_campaign')


def quality_extras(row):
    """Zero wins contribute zero; a zero loss denominator gives undefined PF."""
    profit = float(row['win_n'] * row['mean_win_pct']) if row['win_n'] else 0.
    loss = float(row['loss_n'] * row['mean_loss_abs_pct']) if row['loss_n'] else 0.
    factor = profit / loss if np.isfinite([profit, loss]).all() and loss > 0 else np.nan
    if not np.isfinite(factor):
        factor = np.nan
    return {'gross_profit_sum_pct': profit, 'gross_loss_abs_sum_pct': loss,
            'gross_profit_factor': factor,
            'trade_quality_finite': bool(np.isfinite([row['expectancy_pct'], factor]).all())}


def independent_screen(metrics):
    """Compute all gates/rank without calling either production screen."""
    rows = []
    for cid, group in metrics.groupby('candidate_id', sort=False):
        table = group.set_index(['split', 'variant'])
        b, f = table.loc['IS', 'BASE'], table.loc['IS', 'FILTER']
        blocks = np.array([table.loc[(block, 'FILTER'), 'gross_simple_return_pct'] for block, _, _ in IS_BLOCKS])
        delta_blocks = blocks - np.array([table.loc[(block, 'BASE'), 'gross_simple_return_pct'] for block, _, _ in IS_BLOCKS])
        delta = float(f.gross_simple_return_pct - b.gross_simple_return_pct)
        wr = float((f.win_rate - b.win_rate) * 100)
        payoff = float(f.payoff_ratio - b.payoff_ratio)
        relative = float(f.payoff_ratio / b.payoff_ratio - 1) if np.isfinite(b.payoff_ratio) and b.payoff_ratio > 0 else np.nan
        retention = float(f.closed_n / b.closed_n) if b.closed_n else 0.
        bdd, fdd = -float(b.gross_max_drawdown_pp), -float(f.gross_max_drawdown_pp)
        original = {
            'positive_and_annualized_pass': bool(f.gross_simple_return_pct > 0 and f.gross_annualized_simple_pct >= 1 - TOL),
            'return_improvement_pass': bool(delta >= 5 - TOL),
            'win_rate_pass': bool(np.isfinite(wr) and wr >= 2 - TOL),
            'payoff_pass': bool(np.isfinite(relative) and relative >= .1 - TOL),
            'drawdown_pass': bool(f.gross_max_drawdown_pp >= b.gross_max_drawdown_pp - 2 - TOL),
            'return_to_drawdown_pass': bool(np.isfinite([bdd, fdd]).all() and bdd >= 0 and fdd >= 0
                and f.gross_simple_return_pct * bdd > b.gross_simple_return_pct * fdd + TOL),
            'sample_pass': bool(f.closed_n >= 50 and retention >= .5 - TOL),
            'block_positive_pass': bool(np.count_nonzero(blocks > TOL) >= 2 and blocks[-1] > TOL),
            'block_improvement_pass': bool(np.count_nonzero(delta_blocks > TOL) >= 2 and delta_blocks[-1] > TOL),
        }
        be, fe = float(b.expectancy_pct), float(f.expectancy_pct)
        bp, fp = quality_extras(b)['gross_profit_factor'], quality_extras(f)['gross_profit_factor']
        efinite, pfinite = bool(np.isfinite([be, fe]).all()), bool(np.isfinite([bp, fp]).all())
        flags = {name: original[name] for name in ACTIVE_GATES if name in original}
        flags.update(expectancy_pass=bool(efinite and fe > be), profit_factor_pass=bool(pfinite and fp > bp))
        rows.append({**{name: f[name] for name in IDENTITY if name != 'candidate_id'}, 'candidate_id': cid,
            'gross_return_improvement_pp': delta, 'win_rate_improvement_pp': wr,
            'payoff_absolute_improvement': payoff, 'payoff_relative_improvement': relative,
            'retention': retention, 'is_gross_return_pct': float(f.gross_simple_return_pct),
            'is_net_return_pct': float(f.net_simple_return_pct), 'is_annualized_simple_pct': float(f.gross_annualized_simple_pct),
            'is_win_rate': float(f.win_rate), 'is_payoff_ratio': float(f.payoff_ratio), 'is_closed_n': int(f.closed_n),
            'is_max_drawdown_pp': float(f.gross_max_drawdown_pp), 'positive_blocks': int(np.count_nonzero(blocks > TOL)),
            'improved_blocks': int(np.count_nonzero(delta_blocks > TOL)),
            'original_win_rate_pass': original['win_rate_pass'], 'original_payoff_pass': original['payoff_pass'],
            'original_standard_pass': bool(all(original.values())),
            'original_failed_criteria': '|'.join(name for name in ORIGINAL_GATES if not original[name]),
            'quality_metrics_finite': efinite and pfinite,
            'expectancy_improvement_pp_per_trade': fe - be if efinite else np.nan,
            'profit_factor_improvement': fp - bp if pfinite else np.nan,
            'base_is_expectancy_pct': be, 'filter_is_expectancy_pct': fe,
            'base_is_gross_profit_factor': bp, 'filter_is_gross_profit_factor': fp,
            **flags, 'pass': bool(all(flags.values())),
            'failed_criteria': '|'.join(name for name in ACTIVE_GATES if not flags[name])})
    result = pd.DataFrame(rows).sort_values(
        ['gross_return_improvement_pp', 'payoff_absolute_improvement', 'win_rate_improvement_pp', 'candidate_id'],
        ascending=[False, False, False, True], kind='stable').reset_index(drop=True)
    for test, rank, frozen in [('original_standard_pass', 'original_freeze_rank', 'original_frozen'),
                               ('pass', 'freeze_rank', 'frozen')]:
        result[rank] = np.nan
        winners = result.loc[result[test] & result.is_new].drop_duplicates(['pair', 'strategy'])
        result.loc[winners.index, rank] = np.arange(1, len(winners) + 1)
        result[frozen] = result[rank].notna()
    return result


def augmented_ledger(calendar, executed, returns):
    ledger = ledger_from_executed(calendar, executed, returns)
    if ledger.empty:
        raise ValueError('Holding exits must preserve original entries, including nonempty BASE scope.')
    ledger.insert(0, 'trade_id', np.arange(1, len(ledger) + 1))
    ledger['return_pct'] = ledger.trade_return * 100
    # The old engine's compatibility fields retain compounded dollar equity;
    # simple_ledger adds net_pnl_usd separately for fixed-principal accounting.
    compounded_daily_dollars = 100000 * np.r_[1., np.cumprod(1 + returns)[:-1]] * returns
    legacy_dollars = [float(compounded_daily_dollars[int(row.entry_idx) + 1:int(row.exit_idx) + 1].sum())
                      for row in ledger.itertuples(index=False)]
    ledger['pnl_usd'] = legacy_dollars
    ledger['price_pnl_usd'] = legacy_dollars
    ledger['net_pnl_usd'] = ledger.gross_simple_return_pct * 1000
    ledger['transaction_cost_usd'] = 0.
    ledger['net_trade_return_pct'] = ledger.gross_simple_return_pct
    return ledger


def rebuild_checkpoint_path(calendar, prices, base_gross, base_ledger, decisions, spec):
    """Independent exact-checkpoint truncation, including technical priority.

    A final, unexecuted original signal is not recoverable from a filled-trade
    ledger. That terminal-only fact is taken from the hash-bound decision table,
    explicitly returned as a limitation, and cannot change observed returns.
    """
    n = len(calendar)
    age = int(spec['checkpoint_age'])
    condition = spec.get('condition', 'ALL')
    condition = '&'.join(condition) if isinstance(condition, list) else condition
    normal = decisions.loc[~decisions.entry_pending].set_index('entry_idx', drop=False)
    if normal.index.duplicated().any() or set(normal.index) != set(base_ledger.entry_idx):
        raise ValueError('Original filled episodes do not match the BASE ledger.')
    executed = np.zeros(n, dtype=int)
    expected, terminal = [], []

    def with_dates(row):
        for prefix in ('signal', 'entry', 'checkpoint', 'risk_signal', 'original_end_signal', 'original_exit', 'actual_exit'):
            index = row[prefix + '_idx']
            row[prefix + '_date'] = pd.NaT if pd.isna(index) else calendar[int(index)]
        return row

    for number, trade in enumerate(base_ledger.itertuples(index=False)):
        entry, position = int(trade.entry_idx), int(trade.position)
        source = normal.loc[entry]
        if trade.closed:
            end, natural = int(trade.exit_idx) - 1, int(trade.exit_idx)
        else:
            end, natural = source.original_end_signal_idx, np.nan
            if pd.notna(end):
                if end != n - 1:
                    raise ValueError('An unfilled natural exit can only be the terminal signal.')
                terminal.append({'entry_idx': entry, 'kind': 'pending_original_exit_or_reversal'})
        m = entry + age
        snapshot = bool(m < n and (pd.isna(end) or m <= end))
        evaluated = bool(snapshot and (pd.isna(end) or m < end))
        pnl = peak = giveback = recent = direction = np.nan
        if snapshot:
            earned = base_gross[entry + 1:m + 1] * 100
            pnl, peak = float(earned.sum()), max(0., float(np.cumsum(earned).max()))
            giveback = (peak - pnl) / peak if peak > TOL else np.nan
            recent = float(base_gross[m - 4:m + 1].sum() * 100)
            if m >= 20:
                direction = float(position * (prices[m] / prices[m - 20] - 1) * 100)
        known = 'NOT_EVALUATED'
        if evaluated:
            predicates = {'pnl_negative': pnl < -TOL, 'pnl_positive': pnl > TOL,
                'giveback_half': peak > TOL and pnl <= peak / 2 + TOL,
                'recent5_negative': recent < -TOL if np.isfinite(recent) else None,
                'opposite20': direction < -TOL if np.isfinite(direction) else None}
            values = [True] if condition == 'ALL' else [predicates[name] for name in condition.split('&')]
            known = 'UNKNOWN' if any(value is None for value in values) else 'TRUE' if all(values) else 'FALSE'
        trigger = known == 'TRUE'
        risk = m if trigger else np.nan
        actual = (m + 1 if m + 1 < n else np.nan) if trigger else natural
        risk_pending = bool(trigger and pd.isna(actual))
        exit_pending = bool(risk_pending or (not trigger and pd.notna(end) and pd.isna(natural)))
        status = ('EARLY_EXIT_PENDING' if risk_pending else 'EARLY_EXIT') if trigger else (
            'UNKNOWN' if known == 'UNKNOWN' else 'KEEP' if evaluated else 'TECHNICAL_EXIT' if pd.notna(end) else 'NOT_REACHED')
        expected.append(with_dates({'original_episode': number, 'signal_idx': entry - 1, 'entry_idx': entry,
            'entry_pending': False, 'position': position, 'accepted': True, 'checkpoint_age': age, 'condition': condition,
            'checkpoint_idx': m if snapshot else np.nan, 'evaluated': evaluated, 'condition_state': known,
            'known_pnl_pct': pnl, 'running_peak_pct': peak, 'giveback_fraction': giveback,
            'signed_recent5_pct': recent, 'direction_return_20_pct': direction, 'early_exit': trigger,
            'risk_signal_idx': risk, 'risk_pending': risk_pending, 'original_end_signal_idx': end,
            'original_exit_idx': natural, 'actual_exit_idx': actual, 'exit_pending': exit_pending, 'status': status}))
        stop = int(actual) if pd.notna(actual) else n
        executed[entry:stop] = position
    pending = decisions.loc[decisions.entry_pending]
    if len(pending) > 1:
        raise ValueError('Multiple pending original entries are impossible.')
    for row in pending.itertuples(index=False):
        if row.signal_idx != n - 1 or pd.notna(row.entry_idx) or row.position not in (-1, 1):
            raise ValueError('Invalid terminal original entry.')
        terminal.append({'signal_idx': n - 1, 'kind': 'pending_original_entry'})
        expected.append(with_dates({'original_episode': len(base_ledger), 'signal_idx': n - 1, 'entry_idx': np.nan,
            'entry_pending': True, 'position': int(row.position), 'accepted': True, 'checkpoint_age': age,
            'condition': condition, 'checkpoint_idx': np.nan, 'evaluated': False, 'condition_state': 'NOT_EVALUATED',
            'known_pnl_pct': np.nan, 'running_peak_pct': np.nan, 'giveback_fraction': np.nan,
            'signed_recent5_pct': np.nan, 'direction_return_20_pct': np.nan, 'early_exit': False,
            'risk_signal_idx': np.nan, 'risk_pending': False, 'original_end_signal_idx': np.nan,
            'original_exit_idx': np.nan, 'actual_exit_idx': np.nan, 'exit_pending': False, 'status': 'PENDING_ENTRY'}))
    return executed, pd.DataFrame(expected), terminal


class TradeoffAudit(InteractionAudit):
    def old_evidence(self, protocol):
        review = self.document(REVIEW / 'review.json')
        review_audit = self.document(REVIEW / 'final_review_validation.json')
        review_table = self.frame(REVIEW / 'review.csv')
        self.check('historical_review_immutable_IS_only', review['OOS_not_evaluated_for_review'] is True
            and review['strategy_recomputed'] is False and review['user_preference_pending'] is True
            and review_audit['status'] == 'passed' and review_audit['versions'] == 492
            and review_audit['review_json_sha256'] == sha(REVIEW / 'review.json')
            and review_audit['review_csv_sha256'] == review['review_csv_sha256'] == sha(REVIEW / 'review.csv'))
        self.hashes('review_inputs', review['input_sha256'])
        items, metrics, selections, decisions, required_inputs = [], [], [], [], {}
        for filename in ('review.json', 'review.csv', 'final_review_validation.json'):
            p = REVIEW / filename
            required_inputs[p.relative_to(ROOT).as_posix()] = sha(p)
        old_audit_counts = {}
        for campaign in CAMPAIGNS:
            directory = ROOT / 'reproduced' / campaign
            old_protocol = self.document(directory / 'protocol.json')
            freeze = self.document(directory / 'frozen_selection.json')
            status = self.document(directory / 'development_status.json')
            pre = self.document(directory / 'independent_validation.json')
            self.check('original_audit_passed:' + campaign, pre['status'] == status['status'] == 'passed'
                and pre['checks_total'] == pre['checks_passed'] == len(pre['checks']) and all(x['passed'] for x in pre['checks'])
                and status['oos_evaluated'] is False and not freeze['candidates']
                and freeze['protocol_sha256'] == sha(directory / 'protocol.json')
                and status['freeze_sha256'] == sha(directory / 'frozen_selection.json')
                and pre['audit_script_sha256'] == sha(ROOT / 'verify_holding_filter_research.py'))
            old_audit_counts[campaign] = pre['checks_total']
            for key in ('source_sha256', 'input_sha256'):
                self.check('original_manifest_agreement:' + campaign + ':' + key, old_protocol[key] == freeze[key])
                self.hashes('original_' + campaign + '_' + key, old_protocol[key])
                required_inputs.update(old_protocol[key])
            self.hashes('original_audited_evidence:' + campaign, pre['verified_files_sha256'])
            self.hashes('original_frozen_IS:' + campaign, freeze['is_output_sha256'], directory)
            for name in ('protocol.json', 'frozen_selection.json', 'development_status.json', 'independent_validation.json', *freeze['is_output_sha256']):
                p = directory / name
                required_inputs[p.relative_to(ROOT).as_posix()] = sha(p)
            old_items = self.document(directory / 'is_hypotheses.json')['candidates']
            self.check('fixed_original_candidate_count:' + campaign, len(old_items) == COUNTS[campaign])
            items.extend({**item, 'source_campaign': campaign} for item in old_items)
            metrics.append(self.frame(directory / 'is_all_metrics.csv').assign(source_campaign=campaign))
            selections.append(self.frame(directory / 'is_selection.csv').assign(source_campaign=campaign))
            decisions.append(self.frame(directory / 'is_all_decisions.csv.gz').assign(source_campaign=campaign))
        self.check('new_protocol_binds_all_original_and_review_evidence', all(protocol['input_sha256'].get(name) == value
                   for name, value in required_inputs.items()), required_files=len(required_inputs))
        self.details['original_pre_audits_reused_not_rerun'] = old_audit_counts
        return items, pd.concat(metrics, ignore_index=True), pd.concat(selections, ignore_index=True), pd.concat(decisions, ignore_index=True), review_table

    def run(self):
        protocol = self.read_json('protocol.json')
        status = self.read_json('development_status.json')
        freeze = self.read_json('frozen_selection.json')
        hypotheses = self.read_json('is_hypotheses.json')
        self.check('completed_IS_only_stage', protocol['version'] == 'holding_tradeoff_v8'
            and status['status'] == 'passed' and status['compute_cutoff'] == '2023-05-29'
            and status['oos_evaluated'] is False and status['holdout_returns_computed'] is False)
        self.check('authorized_quality_change_only', bool(protocol.get('user_authorization', {}).get('instruction_zh'))
            and protocol['screen']['expectancy_strict_improvement'] is True
            and protocol['screen']['gross_profit_factor_strict_improvement'] is True
            and protocol['screen']['quality_finiteness_required'] is True
            and protocol['screen']['quality_comparison_tolerance'] == 0
            and protocol['screen']['replaced_original_quality_gates'] == ['win_rate_pass', 'payoff_pass']
            and protocol['screen']['original_quality_gates_diagnostic_only'] is True
            and 'min_win_rate_improvement_pp' not in protocol['screen']
            and 'min_payoff_relative_improvement' not in protocol['screen'])
        self.check('every_other_original_SCREEN_setting_unchanged', all(protocol['screen'].get(name) == value
            for name, value in SCREEN.items() if name not in ('min_win_rate_improvement_pp', 'min_payoff_relative_improvement')))
        self.check('prewritten_OOS_curve_and_new_quality_settings', all(protocol['oos_screen'].get(name) == value
            for name, value in OOS_SCREEN.items() if name != 'quality_confirmation')
            and protocol['oos_screen']['min_quality_closed_trades'] == 20
            and protocol['oos_screen']['expectancy_strict_improvement'] is True
            and protocol['oos_screen']['gross_profit_factor_strict_improvement'] is True
            and protocol['oos_screen']['quality_comparison_tolerance'] == 0)
        self.check('dates_cost_and_universe', protocol['is_start'] == '2005-01-03' and protocol['is_end'] == '2023-05-29'
            and protocol['oos_start'] == '2023-05-30' and protocol['oos_end'] == '2026-05-29'
            and protocol['is_blocks'] == [list(row) for row in IS_BLOCKS]
            and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
            and protocol['accounting'] == 'fixed_initial_capital_simple' and protocol['initial_capital_usd'] == 100000
            and set(protocol['scope']['pairs']) == set(CURRENCIES) and set(protocol['scope']['strategies']) == set(STRATEGIES)
            and protocol['scope']['baseline_accounts'] == 180 and protocol['source_campaigns'] == list(CAMPAIGNS)
            and protocol['original_candidate_counts'] == COUNTS)
        self.check('protocol_freeze_status_chain', freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
            and status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json') and freeze['oos_evaluated_at_freeze'] is False)
        self.check('authorization_then_hypotheses_then_IS_then_freeze', pd.Timestamp(protocol['created_utc'])
            <= pd.Timestamp(protocol['user_authorization']['recorded_utc']) <= pd.Timestamp(hypotheses['created_utc'])
            <= pd.Timestamp(status['started_utc']) < pd.Timestamp(freeze['created_utc']) <= pd.Timestamp(status['finished_utc']))
        for key in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest_agrees:' + key, freeze[key] == protocol[key])
            self.hashes(key, protocol[key])
        self.hashes('frozen_IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.check('complete_frozen_IS_manifest', set(freeze['is_output_sha256'])
                   == {p.name for p in self.campaign.glob('is_*') if p.is_file()})
        self.check('OOS_not_opened_before_pre_audit', not any((self.campaign / name).exists() for name in
            ('oos_status.json', 'oos_metrics.csv', 'oos_daily.csv.gz', 'candidate_assessment.csv',
             'oos_anniversary_metrics.csv', 'oos_fresh_start_metrics.csv', 'frozen_full_trade_ledger.csv.gz',
             'frozen_full_episode_decisions.csv.gz')))

        items, old_metrics, old_selection, old_decisions, review = self.old_evidence(protocol)
        self.check('all_492_original_hypotheses_order_specs_D_evidence_unchanged', hypotheses['candidates'] == items
            and len(items) == len({x['candidate_id'] for x in items}) == 492)
        registry = {item['filter_id']: item['spec'] for item in items}
        self.check('registry_exact_original_specs', protocol['candidate_specs'] == list(registry.values()))
        metrics = self.read_csv('is_all_metrics.csv')
        selection = self.read_csv('is_selection.csv')
        decisions = self.read_csv('is_all_decisions.csv.gz')
        coverage = self.read_csv('is_account_coverage.csv')
        self.compare_frames('old_metric_rows_unchanged', metrics, old_metrics, ['candidate_id', 'variant', 'split'])
        self.compare_frames('old_selection_preserved_as_history', self.read_csv('is_original_selection.csv'), old_selection, ['candidate_id'])
        self.compare_frames('every_old_episode_decision_unchanged', decisions, old_decisions, ['candidate_id', 'original_episode'])
        all_ids = {item['candidate_id'] for item in items}
        expected_grid = set(itertools.product(all_ids, ('BASE', 'FILTER'), ('IS', *[x[0] for x in IS_BLOCKS])))
        self.check('complete_unique_candidate_metric_grid', len(metrics) == 492 * 8
            and not metrics.duplicated(['candidate_id', 'variant', 'split']).any()
            and set(zip(metrics.candidate_id, metrics.variant, metrics.split)) == expected_grid
            and set(selection.candidate_id) == set(decisions.candidate_id) == all_ids)
        counts = pd.Series([(item['pair'], item['strategy']) for item in items]).value_counts()
        self.check('all_180_accounts_with_zero_proposal_coverage', len(coverage) == 180
            and set(zip(coverage.pair, coverage.strategy)) == set(itertools.product(CURRENCIES, STRATEGIES))
            and coverage.compute_end.eq(protocol['is_end']).all()
            and all(row.candidate_count == counts.get((row.pair, row.strategy), 0) for row in coverage.itertuples(index=False)))
        extras = pd.DataFrame([quality_extras(row) for row in metrics.to_dict('records')])
        self.compare_frames('new_quality_arithmetic_all_windows', metrics,
            pd.concat([metrics[['candidate_id', 'variant', 'split']].reset_index(drop=True), extras], axis=1),
            ['candidate_id', 'variant', 'split'])
        rescreened = independent_screen(metrics)
        self.compare_frames('independent_approved_SCREEN_all_flags_rank_fields', selection, rescreened, ['candidate_id'])
        self.check('selection_original_total_order', selection.candidate_id.tolist() == rescreened.candidate_id.tolist())
        winners = rescreened.loc[rescreened.frozen].sort_values('freeze_rank')
        self.check('review_approved_flags_and_order_reproduced', review.set_index('candidate_id').loc[rescreened.candidate_id,
            'diagnostic_alternative_criteria_satisfied'].tolist() == rescreened['pass'].tolist()
            and review.loc[review.diagnostic_one_per_account_suggested].sort_values('diagnostic_suggestion_order').candidate_id.tolist()
            == winners.candidate_id.tolist())
        frozen = pd.DataFrame(freeze['candidates'])
        self.compare_frames('frozen_candidates_exact_cross_family_winners', frozen, winners, ['candidate_id'])
        self.check('freeze_specs_order_counts', [x['candidate_id'] for x in freeze['candidates']] == winners.candidate_id.tolist()
            and all(x['spec'] == registry[x['filter_id']] for x in freeze['candidates'])
            and len(winners) == freeze['frozen_candidates'] == status['frozen_candidates']
            and int(rescreened['pass'].sum()) == freeze['all_passing_versions'] == status['all_passing_versions']
            and freeze['all_candidate_accounts_tested'] == status['candidates_evaluated'] == 492)

        ledger = self.frame(BASE_LEDGER)
        for name in ('signal_date', 'entry_date', 'exit_date'):
            ledger[name] = pd.to_datetime(ledger[name])
        self.check('original_BASE_ledger_scope_and_IS_only', set(zip(ledger.pair, ledger.strategy))
            == set(itertools.product(CURRENCIES, STRATEGIES)) and ledger.exit_date.le(protocol['is_end']).all()
            and not ledger.duplicated(['pair', 'strategy', 'entry_idx']).any())
        for table, names, label in [(ledger, ('closed', 'eligible_closed'), 'BASE'),
                (decisions, ('accepted', 'entry_pending', 'evaluated', 'early_exit', 'risk_pending', 'exit_pending'), 'decisions')]:
            self.check('explicit_nonmissing_booleans:' + label, all(table[name].notna().all()
                and table[name].isin([True, False]).all() for name in names))
        workbook = ROOT / next(name for name in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in name)
        quotes = read_IS_quotes(workbook, pd.Timestamp(protocol['is_end']))
        periods = [('IS', protocol['is_start'], protocol['is_end']), *IS_BLOCKS]
        paths, base_metrics = {}, {}
        for pair in CURRENCIES:
            frame = quotes[pair]
            dates, prices = pd.DatetimeIndex(frame.date), frame.price.to_numpy(float)
            self.check('IS_raw_price_calendar:' + pair, dates.is_unique and dates.is_monotonic_increasing
                and dates[-1] == pd.Timestamp(protocol['is_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1]
            for strategy in STRATEGIES:
                saved = ledger.loc[ledger.pair.eq(pair) & ledger.strategy.eq(strategy)].sort_values('entry_idx')
                executed, occupancy = np.zeros(len(dates), int), np.zeros(len(dates), int)
                for trade in saved.itertuples(index=False):
                    stop = int(trade.exit_idx) if trade.closed else len(dates)
                    executed[int(trade.entry_idx):stop] = int(trade.position)
                    occupancy[int(trade.entry_idx):stop] += 1
                held = np.r_[0, executed[:-1]]
                gross = held * spot
                rebuilt = augmented_ledger(dates, executed, gross)
                self.check('BASE_no_overlapping_episodes:' + pair + '|' + strategy, (occupancy <= 1).all())
                self.compare_frames('BASE_price_ledger:' + pair + '|' + strategy, saved, rebuilt, ['entry_idx'])
                paths[pair, strategy] = dates, prices, spot, executed, held, gross, rebuilt
                base_metrics[pair, strategy] = {name: window_from_path(dates, executed, held, gross, rebuilt, start, end)
                                               for name, start, end in periods}

        by_decisions = {cid: group for cid, group in decisions.groupby('candidate_id', sort=False)}
        metric_index = metrics.set_index(['candidate_id', 'variant', 'split'])
        frozen_ids = set(winners.candidate_id)
        daily = self.read_csv('is_frozen_daily.csv.gz')
        trades = self.read_csv('is_frozen_trades.csv.gz')
        selected_decisions = self.read_csv('is_frozen_decisions.csv.gz')
        frozen_metrics = self.read_csv('is_frozen_metrics.csv')
        self.check('nonempty_frozen_grids_exact_winners', all(set(frame.candidate_id) == frozen_ids
            for frame in (daily, trades, selected_decisions, frozen_metrics))
            and daily.split.eq('IS').all() and frozen_metrics.split.eq('IS').all())
        self.compare_frames('frozen_metrics_subset_all_metrics', frozen_metrics,
            metrics.loc[metrics.candidate_id.isin(frozen_ids) & metrics.split.eq('IS')], ['candidate_id', 'variant', 'split'])
        self.compare_frames('frozen_decisions_subset_all_decisions', selected_decisions,
            decisions.loc[decisions.candidate_id.isin(frozen_ids)], ['candidate_id', 'original_episode'])
        total_windows = total_episodes = total_triggers = 0
        limitations, path_equivalence = [], {}
        for number, item in enumerate(items, 1):
            cid, pair, strategy = item['candidate_id'], item['pair'], item['strategy']
            dates, prices, spot, base_exec, base_held, base_gross, base_ledger = paths[pair, strategy]
            table = by_decisions[cid]
            self.check('candidate_identity:' + cid, all(table[name].eq(item[name]).all() for name in IDENTITY)
                and all(metrics.loc[metrics.candidate_id.eq(cid), name].eq(item[name]).all() for name in IDENTITY))
            executed, computed, terminal = rebuild_checkpoint_path(dates, prices, base_gross, base_ledger, table, item['spec'])
            self.compare_frames('independent_checkpoint_decisions:' + cid, table, computed, ['original_episode'])
            limitations.extend({'candidate_id': cid, **row} for row in terminal)
            total_episodes += len(computed)
            total_triggers += int(computed.early_exit.sum())
            held = np.r_[0, executed[:-1]]
            gross = held * spot
            rebuilt = augmented_ledger(dates, executed, gross)
            effect = 0.
            for row in computed.loc[computed.early_exit & computed.actual_exit_idx.notna()].itertuples(index=False):
                original = base_ledger.loc[base_ledger.entry_idx.eq(row.entry_idx)].iloc[0]
                effect -= base_gross[int(row.actual_exit_idx) + 1:int(original.exit_idx) + 1].sum() * 100
            self.check('independent_removed_tail_conservation:' + cid, same_number((gross - base_gross).sum() * 100, effect))
            path_key = (pair, strategy, executed.tobytes())
            path_equivalence.setdefault(path_key, []).append(cid)
            for phase, start, end in periods:
                for variant in ('BASE', 'FILTER'):
                    calculated = dict(base_metrics[pair, strategy][phase]) if variant == 'BASE' else \
                        window_from_path(dates, executed, held, gross, rebuilt, start, end)
                    calculated.update(quality_extras(calculated))
                    calculated['base_closed_n'] = base_metrics[pair, strategy][phase]['closed_n']
                    calculated['retention'] = calculated['closed_n'] / calculated['base_closed_n'] if calculated['base_closed_n'] else np.nan
                    row = metric_index.loc[cid, variant, phase]
                    errors = [name for name, value in calculated.items() if
                        (pd.Timestamp(row[name]) != value if name.endswith('_date') else not same_number(row[name], value))]
                    self.check('independent_window:' + cid + ':' + variant + ':' + phase, not errors, failed_fields=errors)
                    total_windows += 1
            if cid in frozen_ids:
                mask = dates >= pd.Timestamp(protocol['is_start'])
                key = {name: item[name] for name in IDENTITY}
                for variant, ex, hd, ret, book in [('BASE', base_exec, base_held, base_gross, base_ledger),
                                                  ('FILTER', executed, held, gross, rebuilt)]:
                    expected_daily = pd.DataFrame({'date': dates[mask], 'executed': ex[mask], 'held': hd[mask],
                        'gross_daily_return': ret[mask], 'net_daily_return': ret[mask]}).assign(**key, variant=variant, split='IS')
                    self.compare_frames('frozen_daily_path:' + cid + ':' + variant,
                        daily.loc[daily.candidate_id.eq(cid) & daily.variant.eq(variant)], expected_daily, ['date'])
                    self.compare_frames('frozen_real_ledger:' + cid + ':' + variant,
                        trades.loc[trades.candidate_id.eq(cid) & trades.variant.eq(variant)], book.assign(**key, variant=variant), ['entry_idx'])
            if number % 100 == 0:
                print('Independent IS paths/windows:', number, '/', len(items), flush=True)
        self.check('every_registered_input_unchanged_at_finish', all(sha(name) == digest for name, digest in self.files.items()))
        self.details.update(campaign_version=protocol['version'], candidates=len(items), BASE_accounts=180,
            candidate_accounts=int((coverage.candidate_count > 0).sum()), original_standard_passes=int(rescreened.original_standard_pass.sum()),
            approved_standard_passes=int(rescreened['pass'].sum()), frozen_candidates=len(winners),
            source_candidate_counts=COUNTS, independent_metric_windows=total_windows,
            independent_episode_decisions=total_episodes, early_exit_decisions=total_triggers,
            independently_rebuilt_frozen_daily_candidates=len(frozen_ids),
            distinct_IS_executed_paths_by_account=len(path_equivalence),
            duplicate_path_groups=[ids for ids in path_equivalence.values() if len(ids) > 1],
            repeated_hypothesis_note='492 registered versions are not 492 independent trials; age-5 cumulative/recent5 predicates and some realized paths overlap. No deduplication changes the approved pool or rank.',
            pending_terminal_original_signal_limit={'rows': len(limitations), 'cases': limitations,
                'scope': 'Original BASE signal generation is inherited from hash-bound, previously audited IS evidence. A terminal unfilled original entry/exit/reversal cannot be reconstructed from executions alone; its saved signal is structurally checked and disclosed. No such distinction changes observed IS PnL. All filled paths, checkpoint facts, actual exits and returns are independently rebuilt.'},
            quality_definition='Eligible closed trades entered and closed within each window; flats included in expectancy denominator. Gross PF = gross wins/gross absolute losses, zero wins = 0, zero losses undefined. Strict finite FILTER > BASE for both quality metrics, no selection epsilon.',
            ledger_dollar_compatibility='pnl_usd/price_pnl_usd are independently checked legacy compounded-dollar fields. net_pnl_usd is fixed-principal simple dollars; the SCREEN and PF/expectancy use gross_simple_return_pct, not legacy dollar fields.',
            fresh_sample_warning='Earlier exits can add eligible closed trades near boundaries; formal SCREEN uses actual closed cohorts. This audit does not add paired-cohort gates.',
            strategy_engine_replays=0, old_pre_audits_rerun=False, OOS_results_opened=False,
            holdout_returns_computed=False, raw_prices_decoded_through=protocol['is_end'],
            historical_review_pending_flag='Preserved as the original review snapshot; current authorization is recorded in the new protocol.',
            historical_OOS_exposure=protocol['prior_holdout_exposure'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/holding_tradeoff_v8')
    args = parser.parse_args()
    campaign = args.output.resolve()
    status = json.loads((campaign / 'development_status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'passed':
        raise SystemExit('The IS develop stage must pass before this audit may run.')
    if (campaign / 'oos_status.json').exists():
        raise SystemExit('This is a pre-OOS audit; preserve its record after OOS has opened.')
    audit = TradeoffAudit(campaign)
    started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_entry_interaction_research.py', ROOT / 'verify_state_filter_research.py',
                   ROOT / 'verify_state_filter_supplement.py'):
        audit.register(source)
    try:
        audit.run()
    except Exception:
        audit.check('audit_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(row['passed'] for row in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
        'checks_total': len(audit.checks), 'checks_passed': passed, 'checks': audit.checks,
        'verified_files_sha256': audit.files, 'details': audit.details}
    output = campaign / 'independent_validation.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(output), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
