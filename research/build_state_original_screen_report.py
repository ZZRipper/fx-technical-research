"""Report a fixed old proposal set's completed original-SCREEN evaluation.

This presentation reads only saved v7 results and the old discovery proposal
identities/evidence. It neither executes a filter nor reads old OOS results.
New OOS output is shown only after the v7 oos_status is passed. The former v2
temporal protocol and its result are not retroactively replaced by this round.
"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import hashlib
import io
import json
import os
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/state_original_screen_v7'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(DEFAULT / '.matplotlib_report_cache'))
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_filter_report import _CSS, _table


PREVIOUSLY_EVALUATED = 'JPY|BB_MR_20|STATE_efficiency_63_1'
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_expression(spec, bins):
    labels = {
        'direction_dxy_d20_pct': 'position × DXY原生20步变化%',
        'direction_distance_ma_60_vol': 'position × (价格/MA60−1)×100 / (20日年化波动率% × sqrt(20/252))',
    }
    expression = labels.get(spec['feature'], spec['feature'])
    edges = [-np.inf, *bins[spec['feature']], np.inf]
    lower, upper = edges[spec['state']:spec['state'] + 2]
    if not np.isfinite(lower):
        return expression + f' < {upper:g}'
    if not np.isfinite(upper):
        return expression + f' ≥ {lower:g}'
    return f'{lower:g} ≤ ' + expression + f' < {upper:g}'


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
        raise ValueError('Original-episode decisions must cover every saved candidate.')
    if decisions.duplicated(['candidate_id', 'signal_idx']).any():
        raise ValueError('Duplicate candidate/original-signal decision.')
    records = []
    for candidate_id, group in decisions.groupby('candidate_id', sort=False):
        spec = hypotheses[candidate_id]['spec']
        rejected = group.state.eq(spec['state'])
        unknown = group.state.eq(-1)
        expected_status = np.where(unknown, 'UNKNOWN', np.where(rejected, 'REJECT', 'KEEP'))
        if not (group.feature.eq(spec['feature']).all()
                and group.rejected_state.eq(spec['state']).all()
                and group.state.isin([-1, 0, 1, 2, 3]).all()
                and group.accepted.eq(~rejected).all()
                and group.status.eq(expected_status).all()):
            raise ValueError('Saved state decisions disagree with the unchanged single-state spec.')
        if group.pair.nunique() != 1 or group.strategy.nunique() != 1:
            raise ValueError('Candidate spans multiple original accounts.')
        if group.loc[rejected | group.entry_pending, 'actual_entry_idx'].notna().any():
            raise ValueError('Rejected or terminal-pending entries cannot have an actual fill.')
        records.append({
            'candidate_id': candidate_id, 'pair': str(group.pair.iloc[0]), 'strategy': str(group.strategy.iloc[0]),
            'feature': spec['feature'], 'rejected_state': spec['state'],
            'original_episode_n': len(group), 'rejected_n': int(rejected.sum()),
            'accepted_n': int((~rejected).sum()), 'unknown_retained_n': int(unknown.sum()),
            'known_retained_n': int((~unknown & ~rejected).sum()),
            'pending_original_entry_n': int(group.entry_pending.sum()),
            'rejected_pending_entry_n': int((rejected & group.entry_pending).sum()),
            'accepted_observed_entry_n': int(group.actual_entry_idx.notna().sum()),
        })
    return pd.DataFrame(records)


def write_passed_pdf(output, inputs, frozen, assessment, protocol):
    """Render a convenient subset from saved flags, retaining original IS rank."""
    passed = assessment.loc[assessment.oos_curve_pass & assessment.oos_quality_confirmed]
    candidates = frozen.loc[frozen.candidate_id.isin(passed.candidate_id)].sort_values('freeze_rank')
    if candidates.empty:
        return None
    ids = set(candidates.candidate_id)
    daily = pd.concat([inputs.csv('is_frozen_daily.csv.gz'), inputs.csv('oos_daily.csv.gz')], ignore_index=True)
    metrics = pd.concat([inputs.csv('is_frozen_metrics.csv'), inputs.csv('oos_metrics.csv')], ignore_index=True)
    daily = daily.loc[daily.candidate_id.isin(ids)]
    metrics = metrics.loc[metrics.candidate_id.isin(ids)]
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from fx_filter_report import _plot_candidate, _ordered_metrics

    class SummaryPages:
        def __init__(self, pdf):
            self.pdf = pdf

        def savefig(self, figure, **kwargs):
            # Only this new summary's figure text changes; no shared source edits.
            for label in figure.texts:
                if label.get_text().startswith('Displayed in frozen IS rank order'):
                    label.set_text('Saved curve AND quality pass summary; original IS rank retained. Complete atlas includes every frozen candidate.')
            self.pdf.savefig(figure, **kwargs)

    name = 'passed_candidates.pdf'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.unicode_minus': False}), PdfPages(output / name) as pdf:
        pdf.infodict().update(Title='Saved curve and quality pass summary; original IS order',
                             Subject='Presentation subset only; no candidate substitution or portfolio construction')
        writer = SummaryPages(pdf)
        for row in candidates.to_dict('records'):
            candidate_id = row['candidate_id']
            # The PNG copy is discarded; the shared plotter writes the vector PDF page.
            with io.BytesIO() as temporary_png:
                _plot_candidate(row, daily.loc[daily.candidate_id.eq(candidate_id)],
                    _ordered_metrics(metrics, candidate_id), temporary_png, writer,
                    float(protocol['initial_capital_usd']), True)
    contents = (output / name).read_bytes()
    pages = len(re.findall(rb'/Type\s*/Page\b', contents))
    if not contents.startswith(b'%PDF-') or b'%%EOF' not in contents[-100:] or pages != len(candidates):
        raise ValueError('Passed summary PDF page count or structure differs from its saved flags.')
    return {'file': name, 'pages': pages, 'freeze_ranks': candidates.freeze_rank.astype(int).tolist(),
            'candidate_ids': candidates.candidate_id.tolist(),
            'definition': 'Saved oos_curve_pass AND oos_quality_confirmed; original IS rank, complete atlas retained.',
            'strategy_recomputed': False}


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
    for name in ['state_original_screen_report.html', base_build['report']]:
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
            if target != 'state_original_screen_report_build.json' and not (output / target).is_file():
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
    if protocol.get('version') != 'state_original_screen_v7' or protocol.get('cost_bp') != 0:
        raise ValueError('Only the new zero-cost original-screen state campaign is supported.')
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
    original_name = '../state_filter_v2/discovery_proposals.json'
    original = {row['candidate_id']: row for row in inputs.json(original_name)['candidates']}
    original_key = 'reproduced/state_filter_v2/discovery_proposals.json'
    if protocol['input_sha256'].get(original_key) != inputs.hashes[original_name]:
        raise ValueError('The old proposal source differs from this campaign protocol.')
    if len(original) != 1201 or set(original) != set(hypotheses) or set(selection.candidate_id) != set(hypotheses):
        raise ValueError('All and only the 1201 archived discovery proposals must be evaluated.')
    if list(original) != list(hypotheses):
        raise ValueError('The fixed archived proposal order changed.')
    if selection.candidate_id.duplicated().any():
        raise ValueError('Duplicate saved candidate identity.')
    for candidate_id, row in hypotheses.items():
        if (row['spec'] != original[candidate_id]['spec'] or
                row['discovery_evidence'] != original[candidate_id]['discovery_evidence']):
            raise ValueError('An old proposal spec or D evidence changed in the new evaluation: ' + candidate_id)
    if len(coverage) != protocol['scope']['baseline_accounts'] or coverage.duplicated(['pair', 'strategy']).any() or int(coverage.candidate_count.sum()) != len(selection):
        raise ValueError('Coverage must include all original accounts and saved versions.')
    frozen = selection.loc[selection.frozen].sort_values('freeze_rank')
    expected_order = [row['candidate_id'] for row in sorted(freeze['candidates'], key=lambda row: row['freeze_rank'])]
    if list(frozen.candidate_id) != expected_order:
        raise ValueError('Frozen order differs from saved IS selection.')
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
            raise ValueError('OOS must include every frozen candidate in saved IS order.')
        for name in ('oos_fresh_start_metrics.csv', 'failure_attribution.csv',
                     'failure_episode_attribution.csv', 'failure_diagnostics.json'):
            if (output / name).exists():
                inputs.path(name)
    audits = []
    for name in ['independent_validation.json'] + (['post_oos_independent_validation.json'] if oos_completed else []):
        if (output / name).exists():
            audits.append((name, inputs.json(name)))
    counts = decision_counts(decisions, selection, hypotheses)
    failures = pd.DataFrame([{'gate': gate, 'criterion': label,
        'failed_versions': int(selection[gate].eq(False).sum()),
        'passed_versions': int(selection[gate].eq(True).sum())} for gate, label in GATES.items()])
    by_pair = coverage.groupby('pair', sort=False).agg(accounts=('strategy', 'size'),
        candidate_versions=('candidate_count', 'sum'),
        accounts_with_candidates=('candidate_count', lambda values: int(values.gt(0).sum()))).reset_index()
    registry = pd.DataFrame([{'candidate_id': cid, 'pair': row['pair'], 'strategy': row['strategy'],
        'feature': row['spec']['feature'], 'state': row['spec']['state']} for cid, row in hypotheses.items()])
    by_feature = registry.groupby('feature', sort=True).agg(candidate_versions=('candidate_id', 'size'),
        currencies=('pair', 'nunique'), strategies=('strategy', 'nunique')).reset_index()
    boundary_rows = []
    for feature in sorted(registry.feature.unique()):
        edges = [-np.inf, *protocol['state_bins'][feature], np.inf]
        for state in range(4):
            boundary_rows.append({'feature': feature, 'state': state, 'interval': f'[{edges[state]:g}, {edges[state+1]:g})'})
    periods = metrics[['split', 'start_date', 'end_date']].drop_duplicates().sort_values(['start_date', 'end_date', 'split'])

    # Common presentation consumes only saved paths; no execution/research API is called.
    core.report(output)
    base_build = core.read_json(output / 'report_build.json')
    if bool(base_build['oos_evaluated']) != oos_completed:
        raise ValueError('OOS completion changed during rendering; rerun a consistent snapshot.')
    passed_pdf = write_passed_pdf(output, inputs, frozen, assessment, protocol) if oos_completed else None
    inputs.unchanged()
    counts.to_csv(output / 'is_state_decision_counts.csv', index=False)
    failures.to_csv(output / 'is_state_gate_failures.csv', index=False)
    members = selection.assign(_saved_order=np.arange(len(selection))).sort_values(
        ['freeze_rank', '_saved_order'], na_position='last', kind='stable')
    links, cards = [], []
    for order, row in enumerate(members.to_dict('records'), 1):
        candidate_id = row['candidate_id']
        anchor = f'version-{order:04d}'
        plot = (f'<a href="{base_build["report"]}#candidate-{int(row["freeze_rank"])}">冻结 #{int(row["freeze_rank"])} 图表</a>'
                if row['frozen'] else '未冻结')
        repeated = ' · 既有历史证据复核' if candidate_id == PREVIOUSLY_EVALUATED else ''
        links.append('<li><a href="#' + anchor + '">' + escape(candidate_id) + '</a> · ' + plot + repeated + '</li>')
        selected = selection.loc[selection.candidate_id.eq(candidate_id)]
        evidence = pd.DataFrame([{'block': block, **values}
            for block, values in original[candidate_id].get('discovery_evidence', {}).items()])
        cards.append(f'<details id="{anchor}"><summary>{escape(candidate_id)} · {"原IS标准通过" if row["pass"] else "原IS标准未通过"}{repeated}</summary>'
            + '<p>' + escape(row['filter_name']) + ' · ' + plot + '</p>'
            + _table(selected, ['is_gross_return_pct', 'is_annualized_simple_pct', 'gross_return_improvement_pp',
                               'win_rate_improvement_pp', 'payoff_relative_improvement', 'is_closed_n', 'retention', 'failed_criteria'])
            + '<p>旧提案原存D证据（完整闭合交易cohort）：</p>' + _table(evidence)
            + '<p>本轮全部IS原交易段处理计数：</p>' + _table(counts.loc[counts.candidate_id.eq(candidate_id)]) + '</details>')
    if oos_completed:
        curve = int(assessment.oos_curve_pass.sum())
        quality = int(assessment.oos_quality_confirmed.sum())
        both = int((assessment.oos_curve_pass & assessment.oos_quality_confirmed).sum())
        outcome = f'全部 {len(frozen)} 个冻结候选已完成本轮历史OOS：曲线通过 {curve} 个，质量确认 {quality} 个，同时满足两者 {both} 个。'
        if both == 0:
            outcome += ' 本轮综合目标尚未达成。'
        display = assessment.rename(columns={'failed_criteria': 'oos_failed_criteria'}).copy()
        display['OOS胜率%'] = display.oos_win_rate * 100
        display['BASE胜率%'] = display.oos_base_win_rate * 100
        display['既有历史复核'] = display.candidate_id.eq(PREVIOUSLY_EVALUATED)
        quality_reasons = []
        for row in assessment.itertuples(index=False):
            reasons = []
            if not row.oos_quality_confirmed:
                if row.oos_closed_n < 20:
                    reasons.append('合格闭合少于20笔')
                numbers = [row.oos_win_rate, row.oos_base_win_rate, row.oos_payoff_ratio, row.oos_base_payoff_ratio]
                if not np.isfinite(numbers).all():
                    reasons.append('胜率或盈亏比未定义')
                else:
                    if row.oos_win_rate < row.oos_base_win_rate - 1e-10:
                        reasons.append('胜率低于BASE')
                    if row.oos_payoff_ratio < row.oos_base_payoff_ratio - 1e-10:
                        reasons.append('盈亏比低于BASE')
            quality_reasons.append('；'.join(reasons) if reasons else '—')
        display['质量未通过原因'] = quality_reasons
        oos_html = _table(display, ['freeze_rank', 'candidate_id', '既有历史复核', 'oos_return_pct', 'oos_base_return_pct',
            'oos_improvement_pp', 'oos_annualized_simple_pct', 'oos_max_drawdown_pp', 'OOS胜率%', 'BASE胜率%',
            'oos_payoff_ratio', 'oos_base_payoff_ratio', 'oos_closed_n', 'oos_positive_years',
            'oos_curve_pass', 'oos_quality_confirmed', 'oos_failed_criteria', '质量未通过原因'])
    else:
        outcome = f'原IS标准通过 {int(selection["pass"].sum())} 个版本，按原排名冻结 {len(frozen)} 个账户；本报告尚不展示OOS。'
        oos_html = '<p>未读取本轮未完成的OOS结果。</p>'
    passed_html = ''
    if passed_pdf:
        passed = assessment.loc[assessment.candidate_id.isin(passed_pdf['candidate_ids'])].sort_values('freeze_rank')
        full_is = metrics.loc[metrics.split.eq('IS') & metrics.variant.eq('FILTER'),
                              ['candidate_id', 'gross_simple_return_pct', 'gross_max_drawdown_pp', 'closed_n']]
        passed = passed.merge(full_is.rename(columns={'gross_simple_return_pct': 'IS单利收益%',
            'gross_max_drawdown_pp': 'IS最大回撤pp', 'closed_n': 'IS合格平仓数'}), on='candidate_id', validate='one_to_one')
        rule_cards = []
        explanations = {
            ('direction_dxy_d20_pct', 3): '该状态与较强美元变化同原持仓方向发生冲突相关，可能是技术入场的环境限制；历史关联不证明因果机制。',
            ('direction_dxy_d20_pct', 1): '该状态对应美元变化略有利于原持仓，是一个局部分箱现象；目前不支持可信的单调经济解释。',
            ('direction_distance_ma_60_vol', 0): '该状态可能对应价格远离较长均线时过早逆势。D合计剔除最差一笔仍为负，不代表每一个半段剔除最差一笔后都为负。',
        }
        for row in passed.itertuples(index=False):
            candidate = original[row.candidate_id]
            spec = candidate['spec']
            evidence = pd.DataFrame([{'block': block, **values} for block, values in candidate['discovery_evidence'].items()])
            rule_cards.append('<article><h3>原冻结顺序 ' + str(int(row.freeze_rank)) + ' · '
                + escape(candidate['pair'] + ' / ' + candidate['strategy']) + '</h3><p><b>完整拒绝条件：'
                + escape(state_expression(spec, protocol['state_bins'])) + '</b>。</p><p>'
                + escape(explanations.get((spec['feature'], spec['state']), '固定状态的历史关联，需要保留机制解释的不确定性。'))
                + '</p>' + _table(evidence)
                + '<p><a href="' + base_build['report'] + '#candidate-' + str(int(row.freeze_rank))
                + '">完整IS/OOS曲线与指标</a> · <a href="figures/frozen_' + f'{int(row.freeze_rank):04d}'
                + '.png">单独PNG</a></p></article>')
        passed_html = ('<section id="passed-summary"><h2>满足预声明曲线及质量要求的便捷摘要</h2><p><a href="passed_candidates.pdf">'
            + str(passed_pdf['pages']) + '页通过项图册</a>保留原冻结顺序'
            + ' / '.join(map(str, passed_pdf['freeze_ranks']))
            + '。这是保存结果的阅读摘要，不是新排名、替换候选或策略组合；完整图册仍保留全部'
            + str(len(frozen)) + '个冻结者及其失败。</p>'
            + _table(passed, ['freeze_rank','candidate_id','IS单利收益%','IS最大回撤pp','IS合格平仓数',
                             'oos_return_pct','oos_base_return_pct','oos_max_drawdown_pp','oos_closed_n'])
            + '<p class="note caution">原IS回撤门槛限制相对BASE的恶化，不保证绝对回撤低；IS最大回撤与OOS最大回撤必须一起阅读。'
            + '这几项OOS合格闭合交易仍很少，且历史区间已重复使用。满足本轮预写要求不等于稳定机制或新增独立样本外证据。</p></section>')
        passed_html = passed_html[:-len('</section>')] + (
            '<p>position=+1为做多外币、做空USD，−1为反向。信号日读取条件，满足则跳过整个原交易段；UNKNOWN放行，'
            '保留交易仍下一收盘执行并沿原技术退出。DXY是源序列20个原生观察步的变化，宏观观察日严格早于信号日，'
            '距信号日最多7个自然日；这不能证明源数据原始公布时间或没有历史修订。</p>'
            + ''.join(rule_cards) + '</section>')
    audit_html = ''.join('<p><a href="' + name + '">' + escape(name) + '</a>：status=' + escape(str(record.get('status')))
        + '，checks=' + escape(str(record.get('checks_passed', '—'))) + '/' + escape(str(record.get('checks_total', '—'))) + '。</p>'
        for name, record in audits) or '<p>独立审计记录尚未保存；保存后以相同入口重绘纳入。</p>'
    frozen_links = ''.join('<li><a href="' + base_build['report'] + '#candidate-' + str(int(row.freeze_rank)) + '">'
        + escape(str(row.candidate_id)) + '</a>' + (' · 既有历史证据复核，非新增独立OOS' if row.candidate_id == PREVIOUSLY_EVALUATED else '')
        + '</li>' for row in frozen.itertuples(index=False))
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单状态旧提案：原综合标准评价补齐</title><style>{_CSS}</style></head><body>
<header><p class="eyebrow">ARCHIVED SINGLE-STATE PROPOSALS · ORIGINAL SCREEN</p><h1>补齐旧1201个单状态提案的原标准评价</h1><p class="note caution"><b>{escape(outcome)}</b></p>
<p class="lead">这轮固定使用旧v2已封存的全部1201个提案，补齐它们在完整IS及原综合标准下的评价。没有新增状态、阈值或提案；没有根据任何旧OOS结果排除候选、改排序或替换冻结者。</p>
<div class="stats"><div class="stat"><strong>{len(coverage)}</strong><span>原BASE账户</span></div><div class="stat"><strong>{len(selection)}</strong><span>原封存候选版本</span></div><div class="stat"><strong>{int(selection['pass'].sum())}</strong><span>原IS综合标准通过</span></div><div class="stat"><strong>{len(frozen)}</strong><span>按原排名冻结</span></div></div>
<p><a href="{base_build['report']}">全部冻结候选IS/OOS图表</a> · <a href="{base_build['atlas']}">完整PDF图册</a> · {('<a href="#passed-summary">3个通过项的完整规则</a> · <a href="passed_candidates.pdf">3页通过项PDF</a> · ' if passed_pdf and passed_pdf['pages'] == 3 else '<a href="#passed-summary">通过项摘要</a> · ' if passed_pdf else '')}<a href="#versions">全部1201版本</a> · <a href="#evidence">来源与审计</a></p></header><main>
<section><h2>评价范围补齐与旧协议结论</h2><p>旧v2先经过额外的D/V筛选与锁定，再对锁定者应用确认和其他稳健性检查。旧提案中仅 {protocol.get('original_v2_full_IS_count', 4)} 个已获得完整IS评价，另 {protocol.get('original_v2_missing_full_IS_count', 1197)} 个缺少该评价。本轮将同一封存提案集全部交给原SCREEN及其原账户内排序。两种预写流程评价范围不同；本轮通过不能倒改旧v2额外协议的不通过结论。</p>
<p>保存表中的 is_new 沿用原SCREEN执行接口的资格标记，不表示候选假说从未研究过；新增的是这批旧提案在原标准下的完整评价。</p>
<p class="note caution">JPY · BB_MR_20 · STATE_efficiency_63_1 已在此前单独补充验证。若本轮再次冻结，它属于同一规则、同一历史区间的证据复核，不能计作新增独立OOS成功；也不能因已知的旧表现而删除或改变排名。本报告不读取旧OOS数值。</p>
<p>IS已被多轮研究，最后三年也属于复用的历史验证。保留当轮提案—筛选—冻结—评价的记录，不把复用历史声称为全新未触碰样本外证据。成本保持0，原SCREEN与OOS_SCREEN均不变。</p></section>
<section><h2>固定区间与交易口径</h2>{_table(periods)}<p>完整IS为2005-01-03至2023-05-29；后续本轮OOS为2023-05-30至2026-05-29。图形分别归零显示，持仓仍连续；跨期收益按实际收益日归属。胜率和盈亏比只使用该区间内开仓、真实平仓且至少赚取一个报价收益的交易，不能与账户日PnL混为一个cohort。</p>
<p>旧提案仅来自2005—2014的固定状态观察：两个不重叠半段各至少10笔闭合交易且收益为负，整个D剔除最差一笔后仍为负。这里复用已封存列表和证据，报告不重新提案或计算filter。</p></section>
<section><h2>原单状态规则与固定边界</h2><p>每个原TARGET非零连续段只在最初信号日读取已知状态；等于spec指定状态则整段拒绝，其他状态保留。UNKNOWN=-1始终保留，拒绝后不中途重入，保留段沿原技术退出。信号t、下一报价收盘t+1执行、首次收益t+2；期末pending不虚造成交。</p>
<p>已知BASE过去表现特征使用原未过滤BASE的已赚日收益，包含当时已知浮动部分；不会使用未闭合交易的最终收益。状态区间左闭右开，精确落在边界的数进入右侧状态。</p>
{_table(by_feature)}<details><summary>固定状态边界（0至3；UNKNOWN保留）</summary>{_table(pd.DataFrame(boundary_rows))}</details></section>
<section><h2>全部账户覆盖与原门槛失败</h2>{_table(by_pair)}<details><summary>全部180个账户，含零提案账户</summary>{_table(coverage)}</details><p>以下仅统计原筛选表中的布尔结果；同一版本可同时未通过多项。所有版本继续保留，不以报告统计重新筛选。</p>{_table(failures)}<p><a href="is_state_gate_failures.csv">门槛计数CSV</a></p></section>
{passed_html}<section><h2>全部冻结结果：固定IS顺序</h2><ol>{frozen_links}</ol>{_table(frozen, ['freeze_rank','candidate_id','is_gross_return_pct','gross_return_improvement_pp','win_rate_improvement_pp','payoff_relative_improvement','is_closed_n','retention'])}
<p>每个原账户按原IS排名最多冻结一个通过版本。OOS曲线与交易质量分别列示，未通过者完整保留，不能用正收益或某项改善替代综合标准。</p>{oos_html}</section>
<section><h2>拒绝、保留与UNKNOWN</h2><p>以下统计保存的原交易段决定，pending另外列示；accepted包含UNKNOWN保留段。相同原交易会出现在多个候选版本中，因此跨版本相加不是独立交易样本数。这些计数不会改变排名或标准。</p>
<details><summary>全部1201个版本的决定计数</summary>{_table(counts)}</details><p><a href="is_state_decision_counts.csv">完整计数CSV</a></p></section>
<section id="versions"><h2>全部1201版本：冻结顺序优先，其余保持原表顺序</h2><ol>{''.join(links)}</ol>{''.join(cards)}</section>
<section id="evidence"><h2>来源与复核</h2>{audit_html}<p><a href="{original_name}">旧v2封存提案及D证据</a> · <a href="is_hypotheses.json">本轮固定候选集</a> · <a href="protocol.json">本轮协议</a> · <a href="frozen_selection.json">冻结记录</a> · <a href="is_selection.csv">完整IS筛选</a> · <a href="state_original_screen_report_build.json">报告输入、输出与源码SHA256</a></p></section></main></body></html>'''
    (output / 'state_original_screen_report.html').write_text(html, encoding='utf-8')
    inputs.unchanged()
    validation = validate_artifacts(output, base_build, len(frozen))
    products = ['state_original_screen_report.html', 'is_state_decision_counts.csv', 'is_state_gate_failures.csv',
                'report_build.json', base_build['report'], base_build['atlas'], *base_build['figures']]
    if passed_pdf:
        products.append(passed_pdf['file'])
    manifest = {
        'created_utc': core.now(), 'report': 'state_original_screen_report.html', 'base_report': base_build['report'],
        'atlas': base_build['atlas'], 'figures': base_build['figures'], 'accounts': len(coverage),
        'candidate_versions': len(selection), 'fixed_old_proposal_specs_match': True,
        'original_standard_passing_versions': int(selection['pass'].sum()), 'frozen_candidates': len(frozen),
        'oos_evaluated': oos_completed, 'ordering': 'Saved freeze_rank first, then original is_selection row order.',
        'oos_pass_counts': ({'curve': curve, 'quality': quality, 'both': both, 'not_both': len(assessment) - both}
                            if oos_completed else None),
        'passed_summary_pdf': passed_pdf,
        'previously_evaluated_candidate_id': PREVIOUSLY_EVALUATED,
        'previously_evaluated_candidate_frozen_again': PREVIOUSLY_EVALUATED in expected_order,
        'prior_v2_protocol_conclusion_unchanged': True, 'input_paths_relative_to': str(output),
        'input_sha256': inputs.hashes, 'output_sha256': {name: sha(output / name) for name in products},
        'source_sha256': {name: sha(ROOT / name) for name in ['build_state_original_screen_report.py', 'run_filter_research.py', 'src/fx_filter_report.py']},
        'artifact_validation': validation, 'strategy_recomputed': False, 'selection_changed': False,
        'old_OOS_read': False,
    }
    core.save_json(output / 'state_original_screen_report_build.json', manifest)
    print(json.dumps({name: manifest[name] for name in ['report', 'accounts', 'candidate_versions', 'frozen_candidates', 'oos_evaluated']}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    build_report(args.output)


if __name__ == '__main__':
    main()
