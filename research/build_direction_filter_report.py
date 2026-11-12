"""Render saved direction-filter evidence without recomputing or selecting rules.

The independent presentation is scoped to direction_filter_v6. Completed OOS
artifacts are read only after that campaign's oos_status is passed. Every frozen
candidate keeps its saved IS rank, including any failed historical OOS result.
"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/direction_filter_v6'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(DEFAULT / '.matplotlib_report_cache'))
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_filter_report import _CSS, _table


GATES = {
    'positive_and_annualized_pass': '正收益且简单年化至少 1%',
    'return_improvement_pass': '收益改善至少 5 pp',
    'win_rate_pass': '胜率改善至少 2 pp',
    'payoff_pass': '盈亏比相对改善至少 10%',
    'drawdown_pass': '最大回撤恶化不超过 2 pp',
    'return_to_drawdown_pass': '收益 / 回撤比改善',
    'sample_pass': '至少 50 笔平仓且保留至少 50%',
    'block_positive_pass': '至少两段正收益且末段为正',
    'block_improvement_pass': '至少两段优于 BASE 且末段改善',
}
D_COLUMNS = ['block', 'position', 'closed_n', 'return_sum_pct', 'win_rate',
             'payoff_ratio', 'expectancy_pct', 'return_without_worst_trade_pct',
             'censored_or_ineligible_entry_n']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self, output):
        self.output, self.hashes = output, {}

    def path(self, name):
        path = self.output / name
        if not path.is_file():
            raise ValueError('Required completed report input is missing: ' + name)
        self.hashes[name] = sha(path)
        return path

    def csv(self, name):
        return pd.read_csv(self.path(name))

    def json(self, name):
        return core.read_json(self.path(name))

    def unchanged(self):
        for name, digest in self.hashes.items():
            if sha(self.output / name) != digest:
                raise ValueError('Report input changed during rendering: ' + name)


def decision_counts(decisions, selection, hypotheses):
    if set(decisions.candidate_id) != set(selection.candidate_id):
        raise ValueError('Decisions must cover every saved candidate exactly.')
    if decisions.duplicated(['candidate_id', 'signal_idx']).any():
        raise ValueError('Duplicate original signal in saved decisions.')
    records = []
    for candidate_id, group in decisions.groupby('candidate_id', sort=False):
        spec = hypotheses[candidate_id]['spec']
        rejected = group.position.eq(spec['reject_position'])
        if not (group.reject_position.eq(spec['reject_position']).all()
                and group.rejected.eq(rejected).all()
                and group.accepted.eq(~rejected).all()
                and group.status.eq(np.where(rejected, 'REJECT', 'KEEP')).all()):
            raise ValueError('Saved direction decision does not match the fixed spec.')
        if group.pair.nunique() != 1 or group.strategy.nunique() != 1:
            raise ValueError('A candidate spans more than one original account.')
        if group.loc[rejected, ['actual_entry_idx', 'actual_exit_idx']].notna().any().any():
            raise ValueError('Rejected original episodes must not have filter fills.')
        records.append({
            'candidate_id': candidate_id, 'pair': str(group.pair.iloc[0]),
            'strategy': str(group.strategy.iloc[0]), 'reject_position': spec['reject_position'],
            'original_episode_n': len(group), 'accepted_n': int((~rejected).sum()),
            'rejected_n': int(rejected.sum()), 'unknown_n': 0,
            'original_long_n': int(group.position.eq(1).sum()),
            'original_short_n': int(group.position.eq(-1).sum()),
            'pending_original_entry_n': int(group.entry_pending.sum()),
            'rejected_pending_entry_n': int((group.entry_pending & rejected).sum()),
            'accepted_observed_entry_n': int(group.actual_entry_idx.notna().sum()),
            'accepted_observed_exit_n': int(group.actual_exit_idx.notna().sum()),
            'accepted_pending_exit_n': int(group.exit_pending.sum()),
        })
    return pd.DataFrame(records)


def cases_html(cases):
    blocks = []
    for case in cases['cases']:
        blocks.append('<details><summary>' + escape(case['pair'] + ' · ' + case['strategy'] + ' · ' + case['kind'])
                      + '</summary><p>' + escape(case['interpretation']) + '</p>'
                      + _table(pd.DataFrame(case['blocks']), D_COLUMNS) + '</details>')
    crossing = cases['cross_boundary_counterexample']
    blocks.append('<details open><summary>跨年持仓反例：' + escape(crossing['pair'] + ' · ' + crossing['strategy'])
                  + '</summary><p>' + escape(crossing['fact']) + '</p>'
                  + _table(pd.DataFrame(crossing['blocks']), D_COLUMNS) + '</details>')
    blocks.append('<ul>' + ''.join('<li>' + escape(str(item)) + '</li>' for item in cases.get('limits', [])) + '</ul>')
    return ''.join(blocks)


def validate_artifacts(output, base_build, frozen_count):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.hrefs = set(), []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs:
                self.ids.add(attrs['id'])
            for key in ('href', 'src'):
                if key in attrs:
                    self.hrefs.append(attrs[key])

    documents = {}
    for name in ['direction_report.html', base_build['report']]:
        parser = Links()
        contents = (output / name).read_text(encoding='utf-8')
        if '\ufffd' in contents or '<html' not in contents or '</html>' not in contents:
            raise ValueError('Invalid UTF-8 HTML: ' + name)
        parser.feed(contents)
        documents[name] = parser
    local_links = 0
    for name, document in documents.items():
        for href in document.hrefs:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                continue
            target = unquote(parts.path) or name
            # This manifest is written immediately after the validation succeeds.
            if target != 'direction_report_build.json' and not (output / target).is_file():
                raise ValueError('Broken local report link: ' + href)
            if parts.fragment and target in documents and parts.fragment not in documents[target].ids:
                raise ValueError('Missing report anchor: ' + href)
            local_links += 1
    from PIL import Image
    sizes = {}
    for name in base_build['figures']:
        with Image.open(output / name) as picture:
            sizes[name] = list(picture.size)
            if picture.format != 'PNG' or min(picture.size) < 400:
                raise ValueError('Invalid PNG: ' + name)
            picture.verify()
    pdf = (output / base_build['atlas']).read_bytes()
    pages = len(re.findall(rb'/Type\s*/Page\b', pdf))
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-100:] or pages != max(1, frozen_count):
        raise ValueError('PDF structure/page count differs from the complete freeze.')
    return {'status': 'passed', 'html_documents': len(documents), 'local_links_checked': local_links,
            'png_dimensions': sizes, 'pdf_pages': pages,
            'pdf_check': 'PDF header, EOF, Page object count; visual review is separate.'}


def build_report(output_dir=DEFAULT):
    output = Path(output_dir).resolve()
    inputs = Inputs(output)
    development = inputs.json('development_status.json')
    if development.get('status') != 'passed' or development.get('is_screen_completed') is not True:
        raise ValueError('Wait until development_status passed.')
    protocol = inputs.json('protocol.json')
    if protocol.get('version') != 'direction_filter_v6' or protocol.get('cost_bp') != 0:
        raise ValueError('Only the new zero-cost direction campaign is supported.')
    freeze = inputs.json('frozen_selection.json')
    if freeze['protocol_sha256'] != inputs.hashes['protocol.json'] or development['freeze_sha256'] != inputs.hashes['frozen_selection.json']:
        raise ValueError('Protocol, development and freeze hashes disagree.')
    for name, digest in freeze['is_output_sha256'].items():
        if sha(inputs.path(name)) != digest:
            raise ValueError('Frozen IS evidence changed: ' + name)
    selection = inputs.csv('is_selection.csv')
    metrics = inputs.csv('is_all_metrics.csv')
    decisions = inputs.csv('is_all_decisions.csv.gz')
    coverage = inputs.csv('is_account_coverage.csv')
    hypotheses = {row['candidate_id']: row for row in inputs.json('is_hypotheses.json')['candidates']}
    if set(selection.candidate_id) != set(hypotheses) or selection.candidate_id.duplicated().any():
        raise ValueError('Selection and nominated direction hypotheses differ.')
    if len(coverage) != protocol['scope']['baseline_accounts'] or coverage.duplicated(['pair', 'strategy']).any() or int(coverage.candidate_count.sum()) != len(selection):
        raise ValueError('Coverage must include all original accounts and saved versions.')
    frozen = selection.loc[selection.frozen].sort_values('freeze_rank')
    expected_order = [row['candidate_id'] for row in sorted(freeze['candidates'], key=lambda row: row['freeze_rank'])]
    if list(frozen.candidate_id) != expected_order:
        raise ValueError('Frozen order differs from saved selection.')
    oos_status = inputs.json('oos_status.json') if (output / 'oos_status.json').exists() else {}
    oos_completed = oos_status.get('status') == 'passed'
    assessment = pd.DataFrame()
    if oos_completed:
        if oos_status['freeze_sha256'] != inputs.hashes['frozen_selection.json']:
            raise ValueError('OOS status refers to a different freeze.')
        for name, digest in oos_status['output_sha256'].items():
            if sha(inputs.path(name)) != digest:
                raise ValueError('Completed OOS evidence changed: ' + name)
        assessment = inputs.csv('candidate_assessment.csv').sort_values('freeze_rank')
        if list(assessment.candidate_id) != expected_order:
            raise ValueError('Completed OOS must include every frozen candidate in saved IS order.')
        for name in ('oos_fresh_start_metrics.csv', 'failure_attribution.csv',
                     'failure_episode_attribution.csv', 'failure_diagnostics.json'):
            if (output / name).exists():
                inputs.path(name)

    d_name = '../direction_probe_v5/direction_summary.csv'
    d_summary = inputs.csv(d_name)
    if set(d_summary.block) != {'D1', 'D2', 'D'} or len(d_summary) != len(coverage) * 2 * 3:
        raise ValueError('Direction comparison must contain only D1/D2/D and both sides of every account.')
    if d_summary.duplicated(['pair', 'strategy', 'position', 'block']).any():
        raise ValueError('Duplicate D direction cohort.')
    cases_name = '../direction_observation_validation_v6/cases.json'
    cases = inputs.json(cases_name)
    audit_name = '../direction_observation_validation_v6/validation.json'
    d_audit = inputs.json(audit_name)
    for record in (cases, d_audit):
        if record.get('compute_cutoff') != '2014-12-31' or record.get('oos_computed') is not False:
            raise ValueError('Direction observation narrative must remain D-only.')
    if d_audit['status'] != 'passed' or d_audit['checks_passed'] != d_audit['checks_total']:
        raise ValueError('D-only independent audit has not passed.')
    summary_key = 'reproduced/direction_probe_v5/direction_summary.csv'
    if d_audit['input_sha256'][summary_key] != inputs.hashes[d_name]:
        raise ValueError('D comparison differs from the independently verified table.')
    audits = []
    for name in ['independent_validation.json'] + (['post_oos_independent_validation.json'] if oos_completed else []):
        if (output / name).exists():
            audits.append((name, inputs.json(name)))
    counts = decision_counts(decisions, selection, hypotheses)
    failures = pd.DataFrame([{'gate': gate, 'criterion': label,
                             'failed_versions': int(selection[gate].eq(False).sum()),
                             'passed_versions': int(selection[gate].eq(True).sum())}
                            for gate, label in GATES.items()])
    by_pair = coverage.groupby('pair', sort=False).agg(accounts=('strategy', 'size'),
        candidate_versions=('candidate_count', 'sum'),
        accounts_with_candidates=('candidate_count', lambda values: int(values.gt(0).sum()))).reset_index()
    nomination_sides = pd.DataFrame([{'reject_position': position,
        'direction': '拒绝做空外币 / 做多USD' if position == -1 else '拒绝做多外币 / 做空USD',
        'candidate_versions': sum(item['spec']['reject_position'] == position for item in hypotheses.values())}
        for position in [-1, 1]])

    # The common renderer only consumes saved paths. No execution/research API is called.
    core.report(output)
    base_build = core.read_json(output / 'report_build.json')
    if bool(base_build['oos_evaluated']) != oos_completed:
        raise ValueError('OOS completion changed during rendering; rerun a consistent snapshot.')
    inputs.unchanged()
    counts.to_csv(output / 'is_direction_decision_counts.csv', index=False)
    failures.to_csv(output / 'is_direction_gate_failures.csv', index=False)
    members = selection.assign(_saved_order=np.arange(len(selection))).sort_values(
        ['freeze_rank', '_saved_order'], na_position='last', kind='stable')
    links, cards = [], []
    for order, row in enumerate(members.to_dict('records'), 1):
        candidate_id = row['candidate_id']
        anchor = f'version-{order:04d}'
        plot = (f'<a href="{base_build["report"]}#candidate-{int(row["freeze_rank"])}">冻结 #{int(row["freeze_rank"])} 图表</a>'
                if row['frozen'] else '未冻结')
        links.append('<li><a href="#' + anchor + '">' + escape(candidate_id) + '</a> · ' + plot + '</li>')
        original = selection.loc[selection.candidate_id.eq(candidate_id)]
        observed = d_summary.loc[d_summary.pair.eq(row['pair']) & d_summary.strategy.eq(row['strategy'])]
        cards.append(f'<details id="{anchor}"><summary>{escape(candidate_id)} · {"原IS标准通过" if row["pass"] else "原IS标准未通过"}</summary>'
            + '<p>' + escape(row['filter_name']) + ' · ' + plot + '</p>'
            + _table(original, ['is_gross_return_pct', 'is_annualized_simple_pct', 'gross_return_improvement_pp',
                               'win_rate_improvement_pp', 'payoff_relative_improvement', 'is_closed_n', 'retention', 'failed_criteria'])
            + '<p>D-only 双方向完整闭合交易对照（position +1=做多外币，−1=做空外币）：</p>'
            + _table(observed, D_COLUMNS)
            + '<p>全部 IS 原交易段的处理计数：</p>'
            + _table(counts.loc[counts.candidate_id.eq(candidate_id)]) + '</details>')
    if oos_completed:
        curve = int(assessment.oos_curve_pass.sum())
        quality = int(assessment.oos_quality_confirmed.sum())
        both = int((assessment.oos_curve_pass & assessment.oos_quality_confirmed).sum())
        outcome = f'全部 {len(frozen)} 个冻结候选均已完成本轮历史 OOS：曲线标准通过 {curve} 个，交易质量确认 {quality} 个，同时满足两者 {both} 个。'
        if both == 0:
            outcome += ' 本轮综合目标尚未达成。'
    else:
        outcome = f'原 IS 综合标准通过 {int(selection["pass"].sum())} 个版本，冻结 {len(frozen)} 个账户；本报告尚不展示 OOS。'
    audit_html = '<p><a href="' + audit_name + '">D方向独立核验</a>：' + str(d_audit['checks_passed']) + '/' + str(d_audit['checks_total']) + ' 项通过。</p>'
    for name, record in audits:
        audit_html += '<p><a href="' + name + '">' + escape(name) + '</a>：status=' + escape(str(record.get('status'))) + '，checks=' + escape(str(record.get('checks_passed', '—'))) + '/' + escape(str(record.get('checks_total', '—'))) + '。</p>'
    frozen_links = ''.join('<li><a href="' + base_build['report'] + '#candidate-' + str(int(row.freeze_rank)) + '">'
                          + escape(str(row.candidate_id)) + '</a></li>' for row in frozen.itertuples(index=False))
    oos_columns = ['freeze_rank', 'candidate_id', 'oos_return_pct', 'oos_base_return_pct', 'oos_improvement_pp',
        'oos_annualized_simple_pct', 'oos_max_drawdown_pp', 'OOS胜率%', 'BASE胜率%',
        'oos_payoff_ratio', 'oos_base_payoff_ratio', 'oos_closed_n', 'oos_positive_years',
        'oos_curve_pass', 'oos_quality_confirmed', 'failed_criteria']
    if oos_completed:
        display = assessment.rename(columns={'failed_criteria': 'oos_failed_criteria'}).copy()
        display['OOS胜率%'] = display.oos_win_rate * 100
        display['BASE胜率%'] = display.oos_base_win_rate * 100
        oos_html = _table(display, [name if name != 'failed_criteria' else 'oos_failed_criteria' for name in oos_columns])
        for row in assessment.itertuples(index=False):
            oos_html += (f'<p>{escape(row.candidate_id)}：OOS单利 {row.oos_return_pct:+.4f}% ，BASE {row.oos_base_return_pct:+.4f}% ，'
                         f'差额 {row.oos_improvement_pp:+.4f} pp；胜率 {row.oos_win_rate*100:.4f}% 对 BASE {row.oos_base_win_rate*100:.4f}%。'
                         '数值差异大小不改变预写标准；具体曲线与质量判定以上表原记录为准。</p>')
    else:
        oos_html = '<p>未读取本轮未完成的 OOS 结果。</p>'
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>方向过滤：完整研究记录</title><style>{_CSS}</style></head><body>
<header><p class="eyebrow">DIRECTION FILTER · SAVED EVIDENCE ONLY</p><h1>方向过滤：是否应保留原策略的两个方向？</h1><p class="note caution"><b>{escape(outcome)}</b></p>
<p class="lead">观察覆盖20条技术规则 × 9个币种。候选仅来自2005—2014的原始多空交易对照；逐项应用原综合标准，成本保持为0。固定拒绝某一方向是一种静态方向限制，不能单凭历史差异解释为稳定的经济因果关系。</p>
<div class="stats"><div class="stat"><strong>{len(coverage)}</strong><span>原BASE账户</span></div><div class="stat"><strong>{len(coverage)*2}</strong><span>D中观察的账户方向</span></div><div class="stat"><strong>{len(selection)}</strong><span>D支持的方向候选</span></div><div class="stat"><strong>{len(frozen)}</strong><span>按原标准冻结</span></div></div>
<p><a href="{base_build['report']}">全部冻结候选IS/OOS图表</a> · <a href="{base_build['atlas']}">完整PDF图册</a> · <a href="#versions">全部候选版本</a> · <a href="#evidence">来源与审计</a></p></header><main>
<section><h2>方向定义与执行时序</h2><p>价格均换算为USD per foreign：position +1 是做多外币、做空USD；−1 是做空外币、做多USD。候选只拒绝预先指定的一个方向，保留另一方向的原始入场和技术退出。拒绝的原交易段不会中途重入。</p>
<p>信号收盘t形成指令，下一报价收盘t+1成交，首次策略收益为t+2。自然退出执行日仍获得旧仓当日收益；期末已知指令保留为pending，未虚造成交或强制平仓。固定本金单利，未计carry、融资及交易成本。</p>
{_table(nomination_sides)}<p>没有波动、宏观或价格阈值；没有 UNKNOWN 状态。两种固定spec只有 reject_position=−1 和 +1，方向的好坏不随本轮未来结果更换。</p></section>
<section><h2>提案来源：D1 / D2均为负，且整个D为负</h2><p>D1=2005—2009，D2=2010—2014，D=2005—2014。每个半段该方向至少10笔合格闭合交易，收益和都为负，且整个D收益和也为负，才进入IS重放。删除最大单笔亏损仅为集中度诊断，没有新增门槛。</p>
<p>D1/D2统计要求交易在该段内开仓并自然平仓。跨2009/2010的交易可能只进入整个D，因此D不是两个半段交易收益的机械相加。账户日PnL仍逐日归属，与闭合交易cohort统计不同；提案观察不能替代完整IS曲线。</p>
<p>D独立核验观察到 {d_audit['independent_counts']['repeated_negative_with_at_least_10_per_subperiod_groups']} 个方向在两个半段满足样本和负收益要求，其中 {d_audit['independent_counts']['supported_groups_also_D_negative']} 个整个D也为负。后者全部进入本轮，没有按本轮OOS结果挑选。</p>
<p><a href="{d_name}">全部360个方向的D1/D2/D原观察</a> · <a href="{cases_name}">独立案例与反例</a></p>{cases_html(cases)}</section>
<section><h2>覆盖范围与原门槛失败分布</h2>{_table(by_pair)}<details><summary>全部180个账户，包含零提案账户</summary>{_table(coverage)}</details>
<p>以下仅计数保存的原门槛布尔结果，同一版本可以同时未通过多项。通过标准及账户内排名均未改变。</p>{_table(failures)}<p><a href="is_direction_gate_failures.csv">全部门槛计数CSV</a></p></section>
<section><h2>全部冻结结果：保持IS顺序</h2><ol>{frozen_links}</ol>{_table(frozen, ['freeze_rank','candidate_id','is_gross_return_pct','gross_return_improvement_pp','win_rate_improvement_pp','payoff_relative_improvement','is_closed_n','retention'])}
<p>OOS曲线标准与交易质量确认分别展示。正收益、某项改善或较小回撤不等于综合通过；未通过者完整保留。</p>{oos_html}</section>
<section><h2>原交易段保留与拒绝</h2><p>计数来自保存的全部IS决定，包含期末尚未成交的原始entry指令。实际entry/exit另列；它们与用于胜率的合格闭合交易数有不同含义。这里只核对固定方向规则并统计作用范围，不重新执行策略。</p>{_table(counts)}<p><a href="is_direction_decision_counts.csv">完整计数CSV</a></p></section>
<section id="versions"><h2>全部版本：冻结顺序优先，其余保留原表顺序</h2><ol>{''.join(links)}</ol>{''.join(cards)}</section>
<section id="evidence"><h2>来源与复核</h2>{audit_html}<p>报告只读取已保存结果，不调整阈值、筛选或交易决策。方向观察在此前交互过滤轮的OOS前已保存；本轮使用原D观察提案，但IS已反复研究，最后三年也属于复用的历史验证，不能称为全新未触碰证据。</p>
<p><a href="protocol.json">本轮协议</a> · <a href="frozen_selection.json">冻结记录</a> · <a href="is_selection.csv">完整IS筛选</a> · <a href="direction_report_build.json">报告输入、输出与源码SHA256</a></p></section></main></body></html>'''
    (output / 'direction_report.html').write_text(html, encoding='utf-8')
    inputs.unchanged()
    validation = validate_artifacts(output, base_build, len(frozen))
    products = ['direction_report.html', 'is_direction_decision_counts.csv', 'is_direction_gate_failures.csv',
                'report_build.json', base_build['report'], base_build['atlas'], *base_build['figures']]
    manifest = {
        'created_utc': core.now(), 'report': 'direction_report.html', 'base_report': base_build['report'],
        'atlas': base_build['atlas'], 'figures': base_build['figures'], 'accounts': len(coverage),
        'observed_account_directions': len(coverage) * 2, 'candidate_versions': len(selection),
        'original_standard_passing_versions': int(selection['pass'].sum()), 'frozen_candidates': len(frozen),
        'oos_evaluated': oos_completed, 'ordering': 'Saved freeze_rank first, then original is_selection row order.',
        'input_paths_relative_to': str(output), 'input_sha256': inputs.hashes,
        'output_sha256': {name: sha(output / name) for name in products},
        'source_sha256': {name: sha(ROOT / name) for name in ['build_direction_filter_report.py', 'run_filter_research.py', 'src/fx_filter_report.py']},
        'artifact_validation': validation, 'strategy_recomputed': False, 'selection_changed': False,
        'old_OOS_read': False, 'D_only_observation_cutoff': '2014-12-31',
    }
    core.save_json(output / 'direction_report_build.json', manifest)
    print(json.dumps({name: manifest[name] for name in ['report', 'accounts', 'candidate_versions', 'frozen_candidates', 'oos_evaluated']}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    build_report(args.output)


if __name__ == '__main__':
    main()
