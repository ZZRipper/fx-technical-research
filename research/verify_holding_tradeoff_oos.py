"""Independent post-OOS audit of all frozen v8 holding-quality candidates.

The passed status is checked before any OOS table or raw future price is read.
Raw prices plus saved original BASE execution episodes determine the independent
checkpoint exits, daily paths, closed cohorts and assessment. No production
filter, evaluator, strategy generator or old OOS result is called/read.
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
from verify_entry_interaction_research import InteractionAudit
from verify_state_filter_research import ROOT, sha, same_number
from verify_state_filter_supplement import ledger_from_executed, window_from_path
from verify_entry_interaction_oos import read_frozen_quotes

TOL = 1e-10
DEFAULT = ROOT / 'reproduced/holding_tradeoff_v8'
IDENTITY = ('candidate_id', 'pair', 'strategy', 'filter_id', 'filter_name', 'is_new', 'source_campaign')
CURVE = {'min_net_annualized_simple_pct': 1., 'min_net_return_improvement_pp': 0.,
         'worst_net_drawdown_pp': -15., 'min_positive_anniversary_years': 2, 'min_closed_trades': 10}


def independent_assessment(item, base, filtered, annual):
    """Independent approved flag formula; no production evaluation import."""
    positive = int(np.count_nonzero(annual.net_simple_return_pct.to_numpy(float) > TOL))
    delta = float(filtered.net_simple_return_pct - base.net_simple_return_pct)
    curve = {
        'annualized_pass': bool(filtered.net_annualized_simple_pct >= 1. - TOL),
        'baseline_improvement_pass': bool(delta >= -TOL),
        'drawdown_pass': bool(filtered.net_max_drawdown_pp >= -15. - TOL),
        'positive_years_pass': bool(positive >= 2),
        'sample_pass': bool(filtered.closed_n >= 10),
    }
    be, fe = float(base.expectancy_pct), float(filtered.expectancy_pct)
    bp, fp = float(base.gross_profit_factor), float(filtered.gross_profit_factor)
    finite_e, finite_p = bool(np.isfinite([be, fe]).all()), bool(np.isfinite([bp, fp]).all())
    quality = {
        'quality_sample_pass': bool(filtered.closed_n >= 20),
        'quality_finiteness_pass': finite_e and finite_p,
        'expectancy_pass': bool(finite_e and fe > be),
        'profit_factor_pass': bool(finite_p and fp > bp),
    }
    original = bool(filtered.closed_n >= 20 and np.isfinite([filtered.win_rate, base.win_rate,
        filtered.payoff_ratio, base.payoff_ratio]).all() and filtered.win_rate >= base.win_rate - TOL
        and filtered.payoff_ratio >= base.payoff_ratio - TOL)
    return {
        **{name: item[name] for name in (*IDENTITY, 'freeze_rank')},
        'is_return_pct': item['is_gross_return_pct'],
        'oos_return_pct': float(filtered.net_simple_return_pct), 'oos_base_return_pct': float(base.net_simple_return_pct),
        'oos_improvement_pp': delta, 'oos_annualized_simple_pct': float(filtered.net_annualized_simple_pct),
        'oos_max_drawdown_pp': float(filtered.net_max_drawdown_pp), 'oos_win_rate': float(filtered.win_rate),
        'oos_base_win_rate': float(base.win_rate), 'oos_payoff_ratio': float(filtered.payoff_ratio),
        'oos_base_payoff_ratio': float(base.payoff_ratio), 'oos_closed_n': int(filtered.closed_n),
        'oos_positive_years': positive, 'oos_expectancy_pct': fe, 'oos_base_expectancy_pct': be,
        'oos_gross_profit_factor': fp, 'oos_base_gross_profit_factor': bp,
        'oos_expectancy_improvement_pp_per_trade': fe - be if finite_e else np.nan,
        'oos_profit_factor_improvement': fp - bp if finite_p else np.nan,
        **curve, **quality, 'oos_curve_pass': bool(all(curve.values())),
        'oos_quality_confirmed': bool(all(quality.values())), 'original_oos_quality_confirmed': original,
        'failed_criteria': '|'.join(name for name, passed in curve.items() if not passed),
        'quality_failed_criteria': '|'.join(name for name, passed in quality.items() if not passed),
    }


class HoldingTradeoffPostAudit(InteractionAudit):
    def run(self):
        status = self.read_json('oos_status.json')
        if status.get('status') != 'passed' or status.get('oos_evaluated') is not True:
            raise ValueError('A completed authorized OOS is required before outcome access.')
        protocol, freeze = self.read_json('protocol.json'), self.read_json('frozen_selection.json')
        pre = self.read_json('independent_validation.json')
        valid_pre = (pre['status'] == 'passed' and pre['checks_total'] == pre['checks_passed'] == len(pre['checks'])
            and all(row['passed'] for row in pre['checks'])
            and pre['audit_script_sha256'] == sha(ROOT / 'verify_holding_tradeoff_research.py')
            and pre['details']['OOS_results_opened'] is False and pre['details']['candidates'] == 492)
        self.check('passed_preaudit_and_exact_helper_source', valid_pre)
        if not valid_pre:
            raise ValueError('An unchanged successful independent pre-audit is required.')
        # Pure independent helpers; imports do not execute the pre-audit entry point.
        from verify_holding_tradeoff_research import augmented_ledger, quality_extras, rebuild_checkpoint_path
        self.hashes('preaudit_evidence_unchanged', pre['verified_files_sha256'])
        self.check('freeze_preaudit_then_authorized_OOS',
            pd.Timestamp(freeze['created_utc']) < pd.Timestamp(pre['started_utc']) <= pd.Timestamp(pre['finished_utc'])
            < pd.Timestamp(status['opened_utc']) <= pd.Timestamp(status['finished_utc'])
            and status['freeze_created_before_oos'] is True and status['retuning_permitted'] is False)
        self.check('protocol_freeze_preaudit_status_bindings',
            status['protocol_sha256'] == freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
            and status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
            and status['preaudit_sha256'] == sha(self.campaign / 'independent_validation.json')
            and status['source_integrity_pass'] is True and status['is_prefix_positions_and_returns_match'] is True)
        self.check('approved_unchanged_curve_and_strict_quality_protocol', protocol['version'] == 'holding_tradeoff_v8'
            and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
            and protocol['initial_capital_usd'] == 100000
            and protocol['is_start'] == '2005-01-03' and protocol['is_end'] == '2023-05-29'
            and protocol['oos_start'] == '2023-05-30' and protocol['oos_end'] == '2026-05-29'
            and all(protocol['oos_screen'].get(key) == value for key, value in CURVE.items())
            and protocol['oos_screen']['min_quality_closed_trades'] == 20
            and protocol['oos_screen']['quality_comparison_tolerance'] == 0
            and protocol['oos_screen']['expectancy_strict_improvement'] is True
            and protocol['oos_screen']['gross_profit_factor_strict_improvement'] is True
            and protocol['oos_screen']['quality_finiteness_required'] is True)
        self.check('fixed_three_anniversary_windows', protocol['oos_anniversary_years'] == [
            ['OOS_YEAR_1', '2023-05-30', '2024-05-29'], ['OOS_YEAR_2', '2024-05-30', '2025-05-29'],
            ['OOS_YEAR_3', '2025-05-30', '2026-05-29']])
        for name in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest:' + name, protocol[name] == freeze[name])
            self.hashes(name, freeze[name])
        self.hashes('frozen_IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.hashes('completed_OOS_outputs', status['output_sha256'], self.campaign)
        items = freeze['candidates']; ids = [item['candidate_id'] for item in items]
        self.check('complete_21_original_rank_freeze', len(items) == 21 == status['candidate_count'] == freeze['frozen_candidates']
            and len(ids) == len(set(ids)) and [int(item['freeze_rank']) for item in items] == list(range(1, 22))
            and len({(item['pair'], item['strategy']) for item in items}) == 21)

        daily, prior_daily = self.read_csv('oos_daily.csv.gz'), self.read_csv('is_frozen_daily.csv.gz')
        daily.date, prior_daily.date = pd.to_datetime(daily.date), pd.to_datetime(prior_daily.date)
        ledger, prior_ledger = self.read_csv('frozen_full_trade_ledger.csv.gz'), self.read_csv('is_frozen_trades.csv.gz')
        for table in (ledger, prior_ledger):
            for name in ('signal_date', 'entry_date', 'exit_date'):
                table[name] = pd.to_datetime(table[name])
        decisions, prior_decisions = self.read_csv('frozen_full_episode_decisions.csv.gz'), self.read_csv('is_frozen_decisions.csv.gz')
        metrics, annual, fresh = self.read_csv('oos_metrics.csv'), self.read_csv('oos_anniversary_metrics.csv'), self.read_csv('oos_fresh_start_metrics.csv')
        prior_metrics, assessment = self.read_csv('is_frozen_metrics.csv'), self.read_csv('candidate_assessment.csv')
        tables = (daily, prior_daily, ledger, prior_ledger, decisions, prior_decisions, metrics, annual, fresh, prior_metrics, assessment)
        self.check('all_and_only_frozen_candidates_in_every_table', all(set(table.candidate_id) == set(ids) for table in tables))
        self.check('assessment_retains_original_IS_order', assessment.candidate_id.tolist() == ids
            and assessment.freeze_rank.tolist() == [item['freeze_rank'] for item in items])
        self.check('result_dates_and_unique_grids', daily.date.between(protocol['oos_start'], protocol['oos_end']).all()
            and prior_daily.date.between(protocol['is_start'], protocol['is_end']).all()
            and not daily.duplicated(['candidate_id', 'variant', 'date']).any()
            and not ledger.duplicated(['candidate_id', 'variant', 'entry_idx']).any()
            and not decisions.duplicated(['candidate_id', 'signal_idx']).any())
        for label, table in [('IS', prior_decisions), ('FULL', decisions)]:
            self.check('explicit_boolean_decisions:' + label, all(table[name].notna().all() and table[name].isin([True, False]).all()
                for name in ('accepted', 'entry_pending', 'evaluated', 'early_exit', 'risk_pending', 'exit_pending')))

        source = ROOT / next(name for name in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in name and name.endswith('.xlsx'))
        quotes = read_frozen_quotes(self.register(source), pd.Timestamp(protocol['oos_end']), {item['pair'] for item in items})
        rebuilt_daily, rebuilt_prefix, rebuilt_ledgers, rebuilt_prefix_ledgers = [], [], [], []
        metric_rows, annual_rows, fresh_rows, prefix_rows = [], [], [], []
        limitations = []; total_decisions = total_triggers = 0
        for number, item in enumerate(items, 1):
            cid, pair, strategy = item['candidate_id'], item['pair'], item['strategy']
            key = {name: item[name] for name in IDENTITY}
            q = quotes[pair]; dates = pd.DatetimeIndex(q.date); prices = q.price.to_numpy(float)
            self.check('raw_price_calendar:' + cid, dates.is_unique and dates.is_monotonic_increasing
                and dates[-1] == pd.Timestamp(protocol['oos_end']) and np.isfinite(prices).all() and (prices > 0).all())
            spot = np.r_[0., prices[1:] / prices[:-1] - 1.]
            original = ledger.loc[ledger.candidate_id.eq(cid) & ledger.variant.eq('BASE')].sort_values('entry_idx')
            base_exec, occupied = np.zeros(len(dates), int), np.zeros(len(dates), int)
            for trade in original.itertuples(index=False):
                stop = int(trade.exit_idx) if trade.closed else len(dates)
                base_exec[int(trade.entry_idx):stop] = int(trade.position)
                occupied[int(trade.entry_idx):stop] += 1
            base_held = np.r_[0, base_exec[:-1]]; base_gross = base_held * spot
            base_book = augmented_ledger(dates, base_exec, base_gross)
            self.check('original_BASE_episode_occupancy:' + cid, (occupied <= 1).all())
            self.compare_frames('independent_full_BASE_ledger:' + cid, original, base_book, ['entry_idx'])
            rows = decisions.loc[decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            previous = prior_decisions.loc[prior_decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            self.check('frozen_identity_and_family:' + cid,
                all(rows[name].eq(item[name]).all() for name in IDENTITY)
                and item['spec'] in protocol['candidate_specs']
                and item['source_campaign'] in ('holding_filter_v3', 'timecap_filter_v4'))
            executed, computed, terminal = rebuild_checkpoint_path(dates, prices, base_gross, base_book, rows, item['spec'])
            self.compare_frames('independent_full_checkpoint_decisions:' + cid, rows, computed, ['original_episode'])
            limitations.extend({'candidate_id': cid, 'scope': 'FULL', **row} for row in terminal)
            total_decisions += len(computed); total_triggers += int(computed.early_exit.sum())
            held = np.r_[0, executed[:-1]]; gross = held * spot
            book = augmented_ledger(dates, executed, gross)
            effect = 0.
            for row in computed.loc[computed.early_exit & computed.actual_exit_idx.notna()].itertuples(index=False):
                trade = base_book.loc[base_book.entry_idx.eq(row.entry_idx)].iloc[0]
                effect -= base_gross[int(row.actual_exit_idx) + 1:int(trade.exit_idx) + 1].sum() * 100
            self.check('removed_BASE_tail_conservation_execution_day_retained:' + cid,
                same_number((gross - base_gross).sum() * 100, effect))
            paths = {'BASE': (base_exec, base_held, base_gross, base_book), 'FILTER': (executed, held, gross, book)}

            # Rebuild the truncated IS independently; do not copy a future exit
            # or checkpoint result back into its earlier open/pending episode.
            n_is = int((dates <= pd.Timestamp(protocol['is_end'])).sum())
            prefix_dates = dates[:n_is]
            prefix_base = augmented_ledger(prefix_dates, base_exec[:n_is], base_gross[:n_is])
            prefix_exec, prefix_decisions, terminal = rebuild_checkpoint_path(prefix_dates, prices[:n_is],
                base_gross[:n_is], prefix_base, previous, item['spec'])
            self.compare_frames('independent_truncated_IS_checkpoint_decisions:' + cid, previous, prefix_decisions, ['original_episode'])
            self.check('causal_IS_execution_prefix:' + cid, np.array_equal(prefix_exec, executed[:n_is]))
            limitations.extend({'candidate_id': cid, 'scope': 'IS', **row} for row in terminal)
            prefix_paths = {}
            for variant, (ex, hd, ret, full_book) in paths.items():
                rebuilt_ledgers.append(full_book.assign(**key, variant=variant))
                prefix_book = prefix_base if variant == 'BASE' else augmented_ledger(prefix_dates, prefix_exec, gross[:n_is])
                rebuilt_prefix_ledgers.append(prefix_book.assign(**key, variant=variant))
                prefix_paths[variant] = ex[:n_is], hd[:n_is], ret[:n_is], prefix_book
                for split, start, end, collector in [('IS', protocol['is_start'], protocol['is_end'], rebuilt_prefix),
                                                     ('OOS', protocol['oos_start'], protocol['oos_end'], rebuilt_daily)]:
                    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                    collector.append(pd.DataFrame({'date': dates[mask], 'executed': ex[mask], 'held': hd[mask],
                        'gross_daily_return': ret[mask], 'net_daily_return': ret[mask]}).assign(**key, variant=variant, split=split))
            for periods, collector, calendar, local_paths in [
                    ([('IS', protocol['is_start'], protocol['is_end'])], prefix_rows, prefix_dates, prefix_paths),
                    ([('OOS', protocol['oos_start'], protocol['oos_end'])], metric_rows, dates, paths),
                    (protocol['oos_anniversary_years'], annual_rows, dates, paths)]:
                for split, start, end in periods:
                    values = {variant: window_from_path(calendar, *local_paths[variant], start, end) for variant in ('BASE', 'FILTER')}
                    for variant, value in values.items():
                        value.update(quality_extras(value))
                        count = values['BASE']['closed_n']
                        collector.append({**key, 'variant': variant, 'split': split, **value,
                            'base_closed_n': count, 'retention': value['closed_n'] / count if count else np.nan})
            allowed = np.zeros(len(dates), bool)
            for trade in base_book.itertuples(index=False):
                if trade.entry_date >= pd.Timestamp(protocol['oos_start']):
                    stop = int(trade.exit_idx) if trade.closed else len(dates)
                    allowed[int(trade.entry_idx):stop] = True
            cash_values = {}
            for variant, (ex, _, _, _) in paths.items():
                cash_exec = np.where(allowed, ex, 0); cash_held = np.r_[0, cash_exec[:-1]]; cash_return = cash_held * spot
                cash_book = ledger_from_executed(dates, cash_exec, cash_return)
                if cash_book.empty:
                    cash_book = base_book.iloc[:0].copy()
                cash_values[variant] = window_from_path(dates, cash_exec, cash_held, cash_return, cash_book,
                                                       protocol['oos_start'], protocol['oos_end'])
            for variant, value in cash_values.items():
                value.update(quality_extras(value))
                count = cash_values['BASE']['closed_n']
                fresh_rows.append({**key, 'variant': variant, 'split': 'OOS_FRESH_START', **value,
                    'base_closed_n': count, 'retention': value['closed_n'] / count if count else np.nan})
            print('Independently audited full holding path:', number, '/', len(items), cid, flush=True)

        for label, saved, rebuilt, keys in [
                ('full_real_ledger', ledger, rebuilt_ledgers, ['candidate_id', 'variant', 'entry_idx']),
                ('truncated_IS_real_ledger', prior_ledger, rebuilt_prefix_ledgers, ['candidate_id', 'variant', 'entry_idx']),
                ('all_frozen_IS_daily_prefixes', prior_daily, rebuilt_prefix, ['candidate_id', 'variant', 'date']),
                ('all_frozen_continuous_OOS_daily_paths', daily, rebuilt_daily, ['candidate_id', 'variant', 'date'])]:
            self.compare_frames(label, saved, pd.concat(rebuilt, ignore_index=True), keys)
        full_metrics, years_metrics, cash_metrics, is_metrics = map(pd.DataFrame, (metric_rows, annual_rows, fresh_rows, prefix_rows))
        for label, saved, rebuilt in [('OOS_all_metrics', metrics, full_metrics), ('anniversary_all_metrics', annual, years_metrics),
                                      ('fresh_start_all_metrics', fresh, cash_metrics), ('IS_prefix_all_metrics', prior_metrics, is_metrics)]:
            self.compare_frames(label, saved, rebuilt, ['candidate_id', 'variant', 'split'])
        expected = []
        for item in items:
            cid = item['candidate_id']; account = full_metrics.loc[full_metrics.candidate_id.eq(cid)].set_index('variant')
            years = years_metrics.loc[years_metrics.candidate_id.eq(cid) & years_metrics.variant.eq('FILTER')]
            expected.append(independent_assessment(item, account.loc['BASE'], account.loc['FILTER'], years))
            independently_rebuilt_IS = is_metrics.loc[is_metrics.candidate_id.eq(cid) & is_metrics.variant.eq('FILTER')].iloc[0]
            self.check('frozen_IS_return_against_independent_prefix:' + cid,
                same_number(item['is_gross_return_pct'], independently_rebuilt_IS.gross_simple_return_pct))
        expected = pd.DataFrame(expected)
        self.compare_frames('independent_curve_new_quality_original_quality_and_reasons', assessment, expected, ['candidate_id'])
        flags = ['oos_curve_pass', 'oos_quality_confirmed', 'original_oos_quality_confirmed']
        self.compare_frames('metric_quality_flag_join', metrics[['candidate_id', 'variant', *flags]],
            full_metrics[['candidate_id', 'variant']].merge(expected[['candidate_id', *flags]], on='candidate_id'), ['candidate_id', 'variant'])
        both = expected.oos_curve_pass & expected.oos_quality_confirmed
        self.check('saved_status_pass_counts_independently_reproduced',
            status['curve_passing_candidates'] == int(expected.oos_curve_pass.sum())
            and status['quality_confirmed_candidates'] == int(expected.oos_quality_confirmed.sum())
            and status['both_curve_and_quality_candidates'] == int(both.sum()))
        self.check('registered_evidence_unchanged_at_finish', all(sha(path) == digest for path, digest in self.files.items()))
        self.details.update(candidates=len(ids), candidate_ids_in_frozen_order=ids,
            curve_passes=int(expected.oos_curve_pass.sum()), quality_confirmed=int(expected.oos_quality_confirmed.sum()),
            both_curve_and_quality=int(both.sum()), original_quality_diagnostic_confirmed=int(expected.original_oos_quality_confirmed.sum()),
            full_episode_decisions_checked=total_decisions, early_exit_decisions=total_triggers,
            independently_recomputed_metric_windows=len(full_metrics) + len(years_metrics) + len(cash_metrics),
            independently_recomputed_IS_prefix_metric_windows=len(is_metrics), all_frozen_IS_paths_match=True,
            frozen_currencies_decoded=sorted(quotes), raw_prices_decoded_through=protocol['oos_end'],
            terminal_original_signal_limitations={'count': len(limitations), 'cases': limitations,
                'scope': 'Only terminal unexecuted original BASE entry/exit/reversal facts are inherited from hash-bound decisions and structurally checked. Filled BASE episodes come from the saved execution ledger. No terminal-only distinction changes observed PnL. All checkpoint facts, predicate results, risk execution clocks and observed paths are independently reconstructed.'},
            quality_definition='Eligible closed trades entered and genuinely closed in each window; flat trades remain in expectancy denominator. Gross PF=sum wins/absolute sum losses; missing wins=0, missing losses undefined. Finite FILTER > BASE for both, no selection epsilon; FILTER closed_n >=20 for new OOS quality.',
            execution_clock='Risk signal at checkpoint m executes at close m+1; m+1 return retained, deleted tail starts m+2. Original-episode lock prevents reentry.',
            accounting_compatibility='net_pnl_usd and all window research returns use fixed-principal simple PnL. Legacy ledger pnl_usd/price_pnl_usd fields retain independently reconstructed compound-equity accounting.',
            strategy_engine_replays=0, production_filter_calls=0, production_quality_evaluator_calls=0,
            old_OOS_results_opened=False, OOS_results_opened=True,
            historical_evidence='Reused historical OOS, not independent untouched evidence; all original frozen candidates preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(); campaign = args.output.resolve()
    status_file = campaign / 'oos_status.json'
    if not status_file.exists():
        raise SystemExit('Wait for completed authorized OOS; no audit output written.')
    status = json.loads(status_file.read_text(encoding='utf-8'))
    if status.get('status') != 'passed' or status.get('oos_evaluated') is not True:
        raise SystemExit('Wait for oos_status passed; no outcome tables or audit output opened.')
    audit = HoldingTradeoffPostAudit(campaign); started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_holding_tradeoff_research.py', ROOT / 'verify_entry_interaction_research.py',
                   ROOT / 'verify_entry_interaction_oos.py', ROOT / 'verify_state_filter_research.py',
                   ROOT / 'verify_state_filter_supplement.py'):
        audit.register(source)
    try:
        audit.run()
    except Exception:
        audit.check('audit_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(check['passed'] for check in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
        'checks_total': len(audit.checks), 'checks_passed': passed, 'checks': audit.checks,
        'verified_files_sha256': audit.files, 'details': audit.details}
    destination = campaign / 'post_oos_independent_validation.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(destination), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
