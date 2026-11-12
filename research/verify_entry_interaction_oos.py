"""Post-evaluation arithmetic audit for the frozen entry-interaction campaign.

Reads this campaign's OOS outputs only after its status is passed. Never runs a
strategy, changes a rule, ranks on OOS, or opens a previous campaign's OOS.
Run only after the campaign's one-shot OOS evaluation has been authorized.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from verify_entry_interaction_research import (
    InteractionAudit, ROOT, BINS, TOL, NS, sha, same_number, truth, codes,
    ledger_from_executed, window_from_path,
)


def read_frozen_quotes(path, cutoff, pairs):
    """Decode price cells only for frozen currencies and through OOS_END."""
    quotes = {pair: [] for pair in pairs}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml'))]

        def value(cell):
            if cell is None:
                return None
            kind = cell.get('t', ''); raw = cell.find(NS + 'v')
            if kind == 'inlineStr':
                return ''.join(node.text or '' for node in cell.iter(NS + 't'))
            if raw is None or not raw.text:
                return None
            if kind == 's':
                return shared[int(raw.text)]
            return raw.text if kind in ('str', 'e') else float(raw.text)

        with archive.open('xl/worksheets/sheet6.xml') as stream:
            for _, element in ET.iterparse(stream, events=('end',)):
                if element.tag != NS + 'row':
                    continue
                cells = {''.join(ch for ch in cell.get('r', '') if ch.isalpha()): cell for cell in element.findall(NS + 'c')}
                serial = value(cells.get('B'))
                if isinstance(serial, (int, float)) and np.isfinite(serial):
                    day = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(serial))
                    if day <= cutoff:
                        pair = value(cells.get('D'))
                        if pair in quotes:
                            price = value(cells.get('E'))
                            if price is not None:
                                quotes[pair].append((day, float(price)))
                element.clear()
    return {pair: pd.DataFrame(sorted(rows), columns=['date', 'price']) for pair, rows in quotes.items()}


class PostAudit(InteractionAudit):
    def run(self):
        # Status is the first result document opened; no outcome table is read
        # if the authorized one-shot evaluation has not finished successfully.
        status = self.read_json('oos_status.json')
        if status.get('status') != 'passed' or status.get('oos_evaluated') is not True:
            raise ValueError('Completed OOS status is required before result access.')
        protocol, freeze = self.read_json('protocol.json'), self.read_json('frozen_selection.json')
        pre = self.read_json('independent_validation.json')
        self.check('preaudit_passed_exact_source', pre['status'] == 'passed'
                   and pre['checks_total'] == pre['checks_passed'] == len(pre['checks'])
                   and all(c['passed'] for c in pre['checks'])
                   and pre['audit_script_sha256'] == sha(ROOT / 'verify_entry_interaction_research.py')
                   and pre['details']['OOS_results_opened'] is False)
        self.hashes('preaudit_evidence_unchanged', pre['verified_files_sha256'])
        self.check('freeze_preaudit_then_OOS', pd.Timestamp(freeze['created_utc']) < pd.Timestamp(pre['started_utc'])
                   <= pd.Timestamp(pre['finished_utc']) < pd.Timestamp(status['opened_utc']) <= pd.Timestamp(status['finished_utc'])
                   and status['freeze_created_before_oos'] is True and status['retuning_permitted'] is False)
        self.check('OOS_binds_exact_freeze_protocol', status['freeze_sha256'] == sha(self.campaign / 'frozen_selection.json')
                   and status['protocol_sha256'] == freeze['protocol_sha256'] == sha(self.campaign / 'protocol.json')
                   and status['candidate_count'] == len(freeze['candidates'])
                   and status['source_integrity_pass'] and status['is_prefix_positions_and_returns_match'])
        self.check('unchanged_zero_cost_dates', protocol['version'] == 'entry_interaction_filter_v5'
                   and protocol['cost_bp'] == protocol['reference_cost_bp'] == 0
                   and protocol['oos_start'] == '2023-05-30' and protocol['oos_end'] == '2026-05-29')
        for name in ('source_sha256', 'input_sha256'):
            self.check('freeze_manifest:' + name, protocol[name] == freeze[name])
            self.hashes(name, freeze[name])
        self.hashes('frozen_IS_outputs', freeze['is_output_sha256'], self.campaign)
        self.hashes('OOS_outputs', status['output_sha256'], self.campaign)
        items = freeze['candidates']; ids = [r['candidate_id'] for r in items]
        if not items:
            raise AssertionError('A completed OOS campaign must have frozen candidates.')
        daily = self.read_csv('oos_daily.csv.gz'); daily.date = pd.to_datetime(daily.date)
        is_daily = self.read_csv('is_frozen_daily.csv.gz'); is_daily.date = pd.to_datetime(is_daily.date)
        full_ledger = self.read_csv('frozen_full_trade_ledger.csv.gz')
        for col in ('signal_date', 'entry_date', 'exit_date'):
            full_ledger[col] = pd.to_datetime(full_ledger[col])
        decisions = self.read_csv('frozen_full_episode_decisions.csv.gz')
        is_decisions = self.read_csv('is_frozen_decisions.csv.gz')
        metrics = self.read_csv('oos_metrics.csv')
        annual = self.read_csv('oos_anniversary_metrics.csv')
        fresh = self.read_csv('oos_fresh_start_metrics.csv')
        assessment = self.read_csv('candidate_assessment.csv')
        self.check('all_frozen_candidates_retained_in_every_OOS_table', all(set(frame.candidate_id) == set(ids)
                   for frame in (daily, full_ledger, decisions, metrics, annual, fresh, assessment)))
        self.check('assessment_preserves_freeze_order', assessment.candidate_id.tolist() == ids
                   and assessment.freeze_rank.tolist() == [r['freeze_rank'] for r in items])
        self.check('OOS_dates_and_unique_keys', daily.date.between(protocol['oos_start'], protocol['oos_end']).all()
                   and not daily.duplicated(['candidate_id', 'variant', 'date']).any()
                   and not full_ledger.duplicated(['candidate_id', 'variant', 'entry_idx']).any()
                   and not decisions.duplicated(['candidate_id', 'signal_idx']).any())
        source = ROOT / next(p for p in protocol['input_sha256'] if 'FX_Technical_PIT_Dataset' in p and p.endswith('.xlsx'))
        quotes = read_frozen_quotes(self.register(source), pd.Timestamp(protocol['oos_end']), {item['pair'] for item in items})
        rebuilt_daily, rebuilt_IS, rebuilt_ledgers, rebuilt_metrics, rebuilt_annual, rebuilt_fresh = [], [], [], [], [], []
        decision_count = 0
        for item in items:
            cid, pair = item['candidate_id'], item['pair']
            q = quotes[pair]; dates = pd.DatetimeIndex(q.date); price = q.price.to_numpy(float)
            spot = np.r_[0., price[1:] / price[:-1] - 1.]
            self.check('raw_full_calendar:' + cid, dates.is_unique and dates.is_monotonic_increasing
                       and np.isfinite(price).all() and (price > 0).all() and dates[-1] <= pd.Timestamp(protocol['oos_end']))
            source_base = full_ledger.loc[full_ledger.candidate_id.eq(cid) & full_ledger.variant.eq('BASE')].sort_values('entry_idx')
            base_executed = np.zeros(len(dates), int); occupancy = base_executed.copy()
            for trade in source_base.itertuples(index=False):
                stop = int(trade.exit_idx) if trade.closed else len(dates)
                base_executed[int(trade.entry_idx):stop] = int(trade.position)
                occupancy[int(trade.entry_idx):stop] += 1
            self.check('BASE_episodes_disjoint:' + cid, (occupancy <= 1).all())
            p = pd.Series(price); vol20 = p.pct_change().rolling(20).std() * np.sqrt(252) * 100
            vol63 = p.pct_change().rolling(63).std() * np.sqrt(252) * 100
            length = p.diff().abs().rolling(63).sum()
            technical = {'efficiency_63': (p.diff(63).abs() / length.where(length.ne(0))).mask(length.eq(0), 0.).to_numpy(),
                         'vol_ratio_20_63': (vol20 / vol63.where(vol63.gt(0))).to_numpy(),
                         'direction_return_20_vol': (p.pct_change(20) * 100 / (vol20 * np.sqrt(20 / 252)).where(vol20.gt(0))).to_numpy()}
            rows = decisions.loc[decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            prior = is_decisions.loc[is_decisions.candidate_id.eq(cid)].sort_values('signal_idx')
            prefix = rows.loc[pd.to_datetime(rows.signal_date).le(protocol['is_end'])]
            stable = ['candidate_id', 'signal_idx', 'signal_date', 'position', 'condition', 'conditions_known', 'rejected', 'accepted', 'status']
            stable += [c for c in rows if c.startswith('condition_')]
            self.compare_frames('all_frozen_IS_entry_decisions_unchanged:' + cid, prefix[stable], prior[stable], ['candidate_id', 'signal_idx'])
            original_target = np.zeros(len(dates), int); filtered_target = original_target.copy(); occupied = original_target.copy()
            error_signals, unsupported = [], set()
            base_by_signal = source_base.set_index('signal_idx')
            for ordinal, row in enumerate(rows.itertuples(index=False)):
                begin = int(row.signal_idx); end = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
                pending = begin + 1 == len(dates)
                okay = (0 <= begin < end <= len(dates) and dates[begin] == pd.Timestamp(row.signal_date)
                        and ordinal == row.original_episode and row.position in (-1, 1) and bool(row.entry_pending) == pending)
                states = []
                for j, cond in enumerate(item['spec']['conditions'], 1):
                    stem = f'condition_{j}_'; feature = cond['feature']; value = getattr(row, stem + 'value')
                    if feature in technical:
                        expected_value = technical[feature][begin] * (row.position if feature.startswith('direction_') else 1)
                        okay &= same_number(value, expected_value)
                    else:
                        # Future macro interactions need a dedicated independent
                        # extraction; do not silently claim that coverage.
                        unsupported.add(feature)
                    state = int(codes([value], BINS[feature])[0]); states.append(state)
                    okay &= getattr(row, stem + 'feature') == feature and getattr(row, stem + 'rejected_state') == cond['state']
                    okay &= getattr(row, stem + 'state') == state and bool(getattr(row, stem + 'matches')) == (state == cond['state'])
                known = all(s >= 0 for s in states)
                reject = known and all(s == c['state'] for s, c in zip(states, item['spec']['conditions']))
                okay &= bool(row.conditions_known) == known and bool(row.rejected) == reject and bool(row.accepted) != reject
                okay &= row.status == ('UNKNOWN' if not known else 'REJECT' if reject else 'KEEP')
                okay &= row.condition == '&'.join(f"{c['feature']}={c['state']}" for c in item['spec']['conditions'])
                if not pending:
                    trade = base_by_signal.loc[begin]
                    okay &= row.entry_idx == begin + 1 and pd.Timestamp(row.entry_date) == dates[begin + 1]
                    okay &= row.position == trade.position and (end == int(trade.exit_idx) - 1 if trade.closed else end >= len(dates) - 1)
                else:
                    okay &= pd.isna(row.entry_idx) and pd.isna(row.entry_date)
                okay &= pd.isna(row.actual_entry_idx) if reject or pending else row.actual_entry_idx == begin + 1
                occupied[begin:end] += 1; original_target[begin:end] = row.position
                if not reject:
                    filtered_target[begin:end] = row.position
                if not okay:
                    error_signals.append(begin)
                decision_count += 1
            self.check('full_signal_features_AND_clock:' + cid, not error_signals and not unsupported
                       and (occupied <= 1).all() and np.array_equal(np.r_[0, original_target[:-1]], base_executed),
                       mismatched_signals=error_signals[:10], unsupported_feature_extraction=sorted(unsupported))
            variant_paths = {}
            for variant, executed in [('BASE', base_executed), ('FILTER', np.r_[0, filtered_target[:-1]])]:
                held = np.r_[0, executed[:-1]]; returns = held * spot
                ledger = ledger_from_executed(dates, executed, returns)
                if ledger.empty:
                    ledger = source_base.iloc[:0].copy()
                variant_paths[variant] = executed, held, returns, ledger
                rebuilt_ledgers.append(ledger.assign(candidate_id=cid, variant=variant))
                for split, start, end, collector in [('IS', protocol['is_start'], protocol['is_end'], rebuilt_IS),
                                                     ('OOS', protocol['oos_start'], protocol['oos_end'], rebuilt_daily)]:
                    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                    collector.append(pd.DataFrame({'candidate_id': cid, 'variant': variant, 'split': split, 'date': dates[mask],
                        'executed': executed[mask], 'held': held[mask], 'gross_daily_return': returns[mask], 'net_daily_return': returns[mask]}))
            for periods, collector in [([('OOS', protocol['oos_start'], protocol['oos_end'])], rebuilt_metrics),
                                       (protocol['oos_anniversary_years'], rebuilt_annual)]:
                for split, start, end in periods:
                    calculated = {v: window_from_path(dates, *variant_paths[v], start, end) for v in ('BASE', 'FILTER')}
                    for variant, values in calculated.items():
                        n = calculated['BASE']['closed_n']
                        collector.append({'candidate_id': cid, 'variant': variant, 'split': split,
                                          'base_closed_n': n, 'retention': values['closed_n'] / n if n else np.nan, **values})
            allowed = np.zeros(len(dates), bool)
            for trade in variant_paths['BASE'][3].itertuples(index=False):
                if trade.entry_date >= pd.Timestamp(protocol['oos_start']):
                    end = int(trade.exit_idx) if trade.closed else len(dates)
                    allowed[int(trade.entry_idx):end] = True
            fresh_values = {}
            for variant, (executed, _, _, _) in variant_paths.items():
                cash = np.where(allowed, executed, 0); held = np.r_[0, cash[:-1]]; returns = held * spot
                ledger = ledger_from_executed(dates, cash, returns)
                if ledger.empty:
                    ledger = source_base.iloc[:0].copy()
                fresh_values[variant] = window_from_path(dates, cash, held, returns, ledger, protocol['oos_start'], protocol['oos_end'])
            for variant, values in fresh_values.items():
                n = fresh_values['BASE']['closed_n']
                rebuilt_fresh.append({'candidate_id': cid, 'variant': variant, 'split': 'OOS_FRESH_START',
                                      'base_closed_n': n, 'retention': values['closed_n'] / n if n else np.nan, **values})
        self.compare_frames('full_ledger_source_price_arithmetic', full_ledger, pd.concat(rebuilt_ledgers, ignore_index=True), ['candidate_id', 'variant', 'entry_idx'])
        self.check('full_ledger_zero_cost_dollar_identity', full_ledger.transaction_cost_usd.eq(0).all()
                   and np.allclose(full_ledger.gross_simple_return_pct, full_ledger.net_trade_return_pct, atol=1e-8, rtol=1e-10)
                   and np.allclose(full_ledger.net_pnl_usd, full_ledger.net_trade_return_pct * 1000, atol=1e-5, rtol=1e-10))
        self.compare_frames('IS_prefix_independently_rebuilt', is_daily, pd.concat(rebuilt_IS, ignore_index=True), ['candidate_id', 'variant', 'date'])
        self.compare_frames('OOS_daily_independently_rebuilt', daily, pd.concat(rebuilt_daily, ignore_index=True), ['candidate_id', 'variant', 'date'])
        computed, computed_annual, computed_fresh = map(pd.DataFrame, [rebuilt_metrics, rebuilt_annual, rebuilt_fresh])
        for name, saved, rebuilt in [('OOS_all_metrics', metrics, computed), ('all_anniversary_metrics', annual, computed_annual), ('fresh_start_all_metrics', fresh, computed_fresh)]:
            self.compare_frames(name, saved, rebuilt, ['candidate_id', 'variant', 'split'])
        computed_assessments = []
        gate = protocol['oos_screen']
        for item in items:
            cid = item['candidate_id']; account = computed.loc[computed.candidate_id.eq(cid)].set_index('variant')
            b, f = account.loc['BASE'], account.loc['FILTER']
            positive = int((computed_annual.loc[computed_annual.candidate_id.eq(cid) & computed_annual.variant.eq('FILTER'), 'net_simple_return_pct'] > TOL).sum())
            delta = float(f.net_simple_return_pct - b.net_simple_return_pct)
            flags = {'annualized_pass': bool(f.net_annualized_simple_pct >= gate['min_net_annualized_simple_pct'] - TOL),
                     'baseline_improvement_pass': bool(delta >= gate['min_net_return_improvement_pp'] - TOL),
                     'drawdown_pass': bool(f.net_max_drawdown_pp >= gate['worst_net_drawdown_pp'] - TOL),
                     'positive_years_pass': positive >= gate['min_positive_anniversary_years'],
                     'sample_pass': bool(f.closed_n >= gate['min_closed_trades'])}
            quality = bool(f.closed_n >= 20 and np.isfinite([b.win_rate, f.win_rate, b.payoff_ratio, f.payoff_ratio]).all()
                           and f.win_rate >= b.win_rate - TOL and f.payoff_ratio >= b.payoff_ratio - TOL)
            computed_assessments.append({**{k: item[k] for k in ('candidate_id', 'pair', 'strategy', 'filter_id', 'freeze_rank')},
                'is_return_pct': item['is_gross_return_pct'], 'oos_return_pct': f.net_simple_return_pct,
                'oos_base_return_pct': b.net_simple_return_pct, 'oos_improvement_pp': delta,
                'oos_annualized_simple_pct': f.net_annualized_simple_pct, 'oos_max_drawdown_pp': f.net_max_drawdown_pp,
                'oos_win_rate': f.win_rate, 'oos_base_win_rate': b.win_rate, 'oos_payoff_ratio': f.payoff_ratio,
                'oos_base_payoff_ratio': b.payoff_ratio, 'oos_closed_n': int(f.closed_n), 'oos_positive_years': positive,
                **flags, 'oos_curve_pass': all(flags.values()), 'oos_quality_confirmed': quality,
                'failed_criteria': '|'.join(k for k, value in flags.items() if not value)})
        expected = pd.DataFrame(computed_assessments)
        self.compare_frames('all_prespecified_OOS_curve_quality_assessments', assessment, expected, ['candidate_id'])
        self.check('OOS_status_result_counts', status['curve_passing_candidates'] == int(expected.oos_curve_pass.sum())
                   and status['quality_confirmed_candidates'] == int(expected.oos_quality_confirmed.sum())
                   and status['both_curve_and_quality_candidates'] == int((expected.oos_curve_pass & expected.oos_quality_confirmed).sum()))
        self.compare_frames('OOS_metrics_assessment_flags', metrics[['candidate_id', 'variant', 'oos_curve_pass', 'oos_quality_confirmed']],
                            computed[['candidate_id', 'variant']].merge(expected[['candidate_id', 'oos_curve_pass', 'oos_quality_confirmed']], on='candidate_id'),
                            ['candidate_id', 'variant'])
        self.check('all_evidence_unchanged_at_finish', all(sha(p) == digest for p, digest in self.files.items()))
        self.details.update(candidate_ids_in_frozen_order=ids, candidates=len(ids), full_episode_decisions_checked=decision_count,
                            recomputed_metric_windows=len(computed) + len(computed_annual) + len(computed_fresh),
                            all_frozen_IS_paths_match=True, current_campaign_OOS_only=True, strategy_engine_replays=0,
                            source_price_cutoff=protocol['oos_end'], curve_passes=int(expected.oos_curve_pass.sum()),
                            quality_confirmed=int(expected.oos_quality_confirmed.sum()),
                            raw_BASE_signal_generation='Not rerun; original episodes are verified from saved full BASE ledger and independent source-price accounting.',
                            reuse_disclosure='Repeated historical OOS verification; this audit does not make the sample untouched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/entry_interaction_filter_v5')
    args = parser.parse_args(); audit = PostAudit(args.output.resolve())
    started = datetime.now(timezone.utc).isoformat()
    for source in (__file__, ROOT / 'verify_entry_interaction_research.py', ROOT / 'verify_state_filter_research.py', ROOT / 'verify_state_filter_supplement.py'):
        audit.register(source)
    try:
        audit.run()
    except Exception:
        audit.check('audit_completed_without_exception', False, traceback=traceback.format_exc())
    passed = sum(r['passed'] for r in audit.checks)
    result = {'status': 'passed' if passed == len(audit.checks) else 'failed', 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
              'checks_total': len(audit.checks), 'checks_passed': passed, 'checks': audit.checks,
              'verified_files_sha256': audit.files, 'details': audit.details}
    path = audit.campaign / 'post_oos_independent_validation.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(result['status'], passed, '/', len(audit.checks), str(path), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
