"""Presentation-only expansion of the completed joint-entry IS/OOS campaign.

Uses saved selections, metrics and original-episode decisions. It never executes
a strategy or changes a gate. New-campaign OOS artifacts are read only after a
passed oos_status, and every frozen candidate stays in its saved IS order.
"""
from pathlib import Path
from html import escape
import argparse
import hashlib
from html.parser import HTMLParser
import json
import os
import re
import sys
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'reproduced/entry_interaction_filter_v5'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(DEFAULT / '.matplotlib_report_cache'))
import numpy as np
import pandas as pd
import run_filter_research as core


GATES = (
    'positive_and_annualized_pass', 'return_improvement_pass', 'win_rate_pass',
    'payoff_pass', 'drawdown_pass', 'return_to_drawdown_pass', 'sample_pass',
    'block_positive_pass', 'block_improvement_pass',
)
GATE_NAMES = {
    'positive_and_annualized_pass': '正收益且简单年化 ≥ 1%', 'return_improvement_pass': '收益改善 ≥ 5 pp',
    'win_rate_pass': '胜率改善 ≥ 2 pp', 'payoff_pass': '盈亏比相对改善 ≥ 10%',
    'drawdown_pass': '最大回撤恶化 ≤ 2 pp', 'return_to_drawdown_pass': '收益 / 回撤比改善',
    'sample_pass': '至少 50 笔且保留 ≥ 50%', 'block_positive_pass': '至少两段正收益且末段为正',
    'block_improvement_pass': '至少两段优于 BASE 且末段改善',
}
MECHANISMS = {
    'efficiency_volatility': '价格路径效率 × 当前短期/中期波动比',
    'efficiency_directional_stretch': '价格路径效率 × 相对原入场方向的20日价格伸展',
    'dollar_carry_conflict': '相对入场方向的美元变化 × 相对入场方向的利差',
}


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
                raise ValueError('Report input changed during rendering; rerun: ' + name)


def decision_diagnostics(decisions, selection, metrics):
    if set(decisions.candidate_id) != set(selection.candidate_id):
        raise ValueError('Decisions must cover exactly every saved candidate.')
    if decisions.duplicated(['candidate_id', 'signal_idx']).any():
        raise ValueError('Duplicate original episode signal in decisions.')
    rows, original_by_account = [], {}
    for cid, group in decisions.groupby('candidate_id', sort=False):
        group = group.sort_values('signal_idx')
        account = (str(group.pair.iloc[0]), str(group.strategy.iloc[0]))
        if group.pair.nunique() != 1 or group.strategy.nunique() != 1:
            raise ValueError('One candidate spans multiple accounts.')
        original = tuple(group[['signal_idx', 'position', 'entry_pending']].itertuples(index=False, name=None))
        if account in original_by_account and original_by_account[account] != original:
            raise ValueError('Candidate versions do not share the same original episodes.')
        original_by_account[account] = original
        known = group.conditions_known.eq(True)
        rejected = group.rejected.eq(True)
        accepted = group.accepted.eq(True)
        if not (accepted.eq(~rejected).all() and (~known & rejected).sum() == 0 and
                group.status.eq('UNKNOWN').equals(~known)):
            raise ValueError('Rejected, accepted and UNKNOWN decisions disagree.')
        signature = tuple(group.loc[rejected, 'signal_idx'].astype(int))
        executed_signature = tuple(group.loc[rejected & ~group.entry_pending, 'signal_idx'].astype(int))
        encode = lambda values: hashlib.sha256(json.dumps(values, separators=(',', ':')).encode('ascii')).hexdigest()
        rows.append({
            'candidate_id': cid, 'pair': account[0], 'strategy': account[1],
            'original_episode_n': len(group), 'rejected_n': int(rejected.sum()),
            'accepted_n': int(accepted.sum()), 'unknown_retained_n': int((~known & accepted).sum()),
            'known_retained_n': int((known & accepted).sum()), 'pending_entry_n': int(group.entry_pending.sum()),
            'rejected_pending_entry_n': int((rejected & group.entry_pending).sum()),
            'rejection_set_sha256': encode(signature), 'executed_rejection_set_sha256': encode(executed_signature),
        })
    counts = pd.DataFrame(rows)
    group_keys = ['pair', 'strategy', 'rejection_set_sha256']
    equivalents = counts.copy()
    groups = sorted(set(equivalents[group_keys].itertuples(index=False, name=None)))
    identifiers = {key: f'IS_REJECTION_SET_{index + 1:04d}' for index, key in enumerate(groups)}
    equivalents['equivalence_group'] = [identifiers[key] for key in equivalents[group_keys].itertuples(index=False, name=None)]
    equivalents['group_version_n'] = equivalents.groupby('equivalence_group').candidate_id.transform('size')
    checks = 0
    comparison_columns = ['gross_simple_return_pct', 'gross_max_drawdown_pp', 'win_rate', 'payoff_ratio',
                          'closed_n', 'entry_n', 'retention']
    for _, group in equivalents.groupby('equivalence_group', sort=False):
        reference = None
        for cid in group.candidate_id:
            path_metrics = metrics.loc[metrics.candidate_id.eq(cid)].sort_values(['split', 'variant'])
            current = path_metrics[comparison_columns].to_numpy(float)
            if reference is not None:
                if not np.allclose(reference, current, atol=1e-10, rtol=0, equal_nan=True):
                    raise ValueError('Identical rejected-episode sets have different saved IS metrics.')
                checks += 1
            reference = current
    duplicates = equivalents.loc[equivalents.group_version_n.gt(1)]
    info = {
        'candidate_versions': len(equivalents), 'distinct_rejected_episode_sets': len(groups),
        'duplicate_groups': duplicates.equivalence_group.nunique(), 'versions_in_duplicate_groups': len(duplicates),
        'identical_group_metric_checks': checks,
        'definition': 'Within one original account, identical complete rejected signal sets including terminal pending instructions. All versions remain listed; no equivalence group changes selection.',
        'pending_distinction': 'An additional executed_rejection_set_sha256 excludes pending entries. Equal complete sets are sufficient for equal IS target/execution paths; different terminal pending instructions remain separate.',
    }
    return counts, equivalents, info


def _table(frame, columns=None):
    from fx_filter_report import _table as render
    return render(frame, columns)


def _case_section(record):
    if record is None:
        return '<p>D-only 案例说明尚未保存；本版不自行挑选事后成功或失败案例。完成后运行相同脚本即可补入。</p>'
    blocks = ['<p>以下是独立整理的 2005—2014 观察案例及反例，不用于重新选择候选。joint为两条件交集；A_only/B_only仅满足一项；retained为不拒绝的原交易。收益均是原BASE完整闭合交易的单利和，不是过滤后账户曲线。</p>']
    fields = ['block', 'group', 'closed_n', 'return_sum_pct', 'win_rate', 'payoff_ratio',
              'expectancy_pct', 'return_without_worst1_pct', 'return_without_worst3_pct']
    for case in record.get('cases', []):
        blocks.append('<details><summary>' + escape(str(case.get('title', case['case_id']))) + '</summary><p>'
                      + escape(str(case.get('hypothesis_not_causal_claim', ''))) + '</p>'
                      + _table(pd.DataFrame(case.get('blocks', [])), fields) + '</details>')
    for case in record.get('counterexamples', []):
        blocks.append('<details><summary>反例：' + escape(str(case.get('kind', ''))) + ' · '
                      + escape(str(case.get('pair', ''))) + ' ' + escape(str(case.get('strategy', '')))
                      + '</summary><p>' + escape(str(case.get('fact', ''))) + '</p>'
                      + _table(pd.DataFrame(case.get('blocks', [])), fields) + '</details>')
    blocks.append('<details><summary>解释边界与重叠诊断</summary><ul>'
                  + ''.join('<li>' + escape(str(item)) + '</li>' for item in record.get('interpretation_limits', []))
                  + '</ul><pre>' + escape(json.dumps(record.get('overlap_diagnostics', {}), ensure_ascii=False, indent=2)) + '</pre></details>')
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
    for name in ['interaction_report.html', base_build['report']]:
        parser = Links()
        text = (output / name).read_text(encoding='utf-8')
        if '\ufffd' in text or '<html' not in text or '</html>' not in text:
            raise ValueError('Invalid UTF-8 HTML document: ' + name)
        parser.feed(text)
        documents[name] = parser
    local_links = 0
    for name, document in documents.items():
        for href in document.hrefs:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                continue
            path = unquote(parts.path) or name
            if not (output / path).is_file():
                raise ValueError('Broken local report link: ' + href)
            if parts.fragment and path in documents and parts.fragment not in documents[path].ids:
                raise ValueError('Missing report anchor: ' + href)
            local_links += 1
    from PIL import Image
    sizes = {}
    for name in base_build['figures']:
        with Image.open(output / name) as picture:
            sizes[name] = list(picture.size)
            if picture.format != 'PNG' or min(picture.size) < 400:
                raise ValueError('Invalid or undersized PNG: ' + name)
            picture.verify()
    pdf = (output / base_build['atlas']).read_bytes()
    pages = len(re.findall(rb'/Type\s*/Page\b', pdf))
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-100:] or pages != max(1, frozen_count):
        raise ValueError('PDF structure/page count differs from the frozen atlas.')
    return {'status': 'passed', 'html_documents': len(documents), 'local_links_checked': local_links,
            'png_dimensions': sizes, 'pdf_pages': pages, 'pdf_check': 'PDF header, EOF and Page object count; visual review remains separate.'}


def build_report(output_dir=DEFAULT):
    output = Path(output_dir).resolve()
    inputs = Inputs(output)
    development = inputs.json('development_status.json')
    if development.get('status') != 'passed' or development.get('is_screen_completed') is not True:
        raise ValueError('Wait until development_status is passed before building this report.')
    protocol = inputs.json('protocol.json')
    if protocol.get('version') != 'entry_interaction_filter_v5' or protocol.get('cost_bp') != 0:
        raise ValueError('Only the new zero-cost entry-interaction campaign is supported.')
    freeze = inputs.json('frozen_selection.json')
    if freeze['protocol_sha256'] != inputs.hashes['protocol.json'] or development['freeze_sha256'] != inputs.hashes['frozen_selection.json']:
        raise ValueError('Protocol/freeze/development hash chain differs.')
    selection = inputs.csv('is_selection.csv')
    metrics = inputs.csv('is_all_metrics.csv')
    decisions = inputs.csv('is_all_decisions.csv.gz')
    coverage = inputs.csv('is_account_coverage.csv')
    hypotheses = inputs.json('is_hypotheses.json')
    for name in ('is_frozen_daily.csv.gz', 'is_frozen_metrics.csv'):
        inputs.path(name)
    if coverage.duplicated(['pair', 'strategy']).any() or len(coverage) != protocol['scope']['baseline_accounts'] or int(coverage.candidate_count.sum()) != len(selection):
        raise ValueError('Saved coverage/candidate count disagrees.')
    if set(selection.candidate_id) != {row['candidate_id'] for row in hypotheses['candidates']}:
        raise ValueError('Saved hypotheses and selection differ.')
    frozen = selection.loc[selection.frozen].sort_values('freeze_rank')
    if list(frozen.candidate_id) != [row['candidate_id'] for row in sorted(freeze['candidates'], key=lambda row: row['freeze_rank'])]:
        raise ValueError('Frozen candidates/order differ from saved selection.')
    for name, digest in freeze['is_output_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Frozen IS evidence changed: ' + name)

    oos_status = inputs.json('oos_status.json') if (output / 'oos_status.json').exists() else {}
    oos_completed = oos_status.get('status') == 'passed'
    assessment = pd.DataFrame()
    if oos_completed:
        if oos_status['freeze_sha256'] != inputs.hashes['frozen_selection.json']:
            raise ValueError('OOS refers to a different freeze.')
        for name, digest in oos_status['output_sha256'].items():
            if sha(inputs.path(name)) != digest:
                raise ValueError('Saved completed OOS evidence changed: ' + name)
        assessment = inputs.csv('candidate_assessment.csv').sort_values('freeze_rank')
        if list(assessment.candidate_id) != list(frozen.candidate_id):
            raise ValueError('Completed OOS must show every frozen candidate in IS order.')
    counts, equivalents, equivalence_info = decision_diagnostics(decisions, selection, metrics)
    failures = pd.DataFrame([{'gate': name, 'criterion': GATE_NAMES[name],
                             'failed_versions': int((~selection[name]).sum()),
                             'failed_fraction': float((~selection[name]).mean())} for name in GATES])
    case_name = ('interaction_discovery_cases.json' if (output / 'interaction_discovery_cases.json').exists()
                 else '../entry_interaction_probe_v5/cases.json')
    case_record = inputs.json(case_name) if (output / case_name).exists() else None
    if case_record is not None and (case_record.get('outcome_cutoff') != '2014-12-31' or case_record.get('OOS_results_opened') is not False):
        raise ValueError('Case narrative must use only D outcomes without OOS results.')
    independent = inputs.json('independent_validation.json') if (output / 'independent_validation.json').exists() else None
    post_audit = inputs.json('post_oos_independent_validation.json') if oos_completed and (output / 'post_oos_independent_validation.json').exists() else None
    attribution_name = '../entry_interaction_probe_v5/frozen_IS_episode_attribution.json'
    attribution = inputs.json(attribution_name) if (output / attribution_name).exists() else None
    # Existing orchestration only renders saved paths. It cannot execute a rule.
    core.report(output)
    base_build = core.read_json(output / 'report_build.json')
    if bool(base_build['oos_evaluated']) != oos_completed:
        raise ValueError('OOS completion changed while building; rerun for one consistent snapshot.')
    inputs.unchanged()

    counts.to_csv(output / 'interaction_decision_counts.csv', index=False)
    equivalents.to_csv(output / 'interaction_equivalent_paths.csv', index=False)
    failures.to_csv(output / 'interaction_gate_failures.csv', index=False)
    members = selection.merge(equivalents, on=['candidate_id', 'pair', 'strategy'], validate='one_to_one')
    members['_saved_order'] = np.arange(len(members))
    members = members.sort_values(['freeze_rank', '_saved_order'], na_position='last', kind='stable')
    specs = {row['candidate_id']: row for row in hypotheses['candidates']}
    mechanism_rows, boundary_rows = [], []
    for mechanism, feature_a, feature_b in protocol['interaction_registry']:
        candidate_ids = {cid for cid, row in specs.items() if row['interaction_id'] == mechanism}
        mechanism_rows.append({'mechanism': mechanism, 'meaning': MECHANISMS.get(mechanism, mechanism),
                               'features': feature_a + ' AND ' + feature_b, 'registered_cells_per_account': 16,
                               'D_supported_versions': len(candidate_ids)})
    used_features = sorted({item for _, a, b in protocol['interaction_registry'] for item in (a, b)})
    for feature in used_features:
        edges = [-np.inf, *protocol['state_bins'][feature], np.inf]
        for code in range(4):
            boundary_rows.append({'feature': feature, 'state': code, 'interval': f'[{edges[code]:g}, {edges[code + 1]:g})'})
    by_pair = coverage.groupby('pair', sort=False).agg(accounts=('strategy', 'size'), candidate_versions=('candidate_count', 'sum'),
                                                     accounts_with_candidates=('candidate_count', lambda values: int(values.gt(0).sum()))).reset_index()
    # All versions get stable local links; only saved frozen versions get plot links.
    links, cards = [], []
    for order, row in enumerate(members.to_dict('records'), 1):
        anchor = f'version-{order:04d}'
        plot = (f'<a href="{base_build["report"]}#candidate-{int(row["freeze_rank"])}">冻结 #{int(row["freeze_rank"])} 图表</a>'
                if row['frozen'] else '未冻结')
        links.append(f'<li><a href="#{anchor}">{escape(row["candidate_id"])}</a> · {plot}</li>')
        original = selection.loc[selection.candidate_id.eq(row['candidate_id'])]
        diagnostic = equivalents.loc[equivalents.candidate_id.eq(row['candidate_id'])]
        evidence = pd.DataFrame([{'block': block, **values} for block, values in specs[row['candidate_id']]['discovery_evidence'].items()])
        cards.append(f'<details id="{anchor}"><summary>{escape(row["candidate_id"])} · {"IS 原标准通过" if row["pass"] else "IS 原标准未通过"}</summary>'
                     f'<p>{escape(row["filter_name"])} · {plot}</p>'
                     + _table(original, ['is_gross_return_pct', 'is_annualized_simple_pct', 'gross_return_improvement_pp',
                                         'win_rate_improvement_pp', 'payoff_relative_improvement', 'is_closed_n', 'retention', 'failed_criteria'])
                     + _table(diagnostic, ['original_episode_n', 'rejected_n', 'accepted_n', 'unknown_retained_n',
                                           'pending_entry_n', 'rejected_pending_entry_n', 'equivalence_group', 'group_version_n'])
                     + '<p>D 提名证据（闭合交易同期口径；不等于逐日账户收益）：</p>' + _table(evidence) + '</details>')
    status_text = ('新目录 OOS 已完成，全部冻结结果如下；排序仍为 IS freeze_rank。' if oos_completed else
                   '当前仅展示完成并冻结的 IS 结果；没有读取尚未完成的 OOS 结果。')
    outcome_text = ''
    if oos_completed:
        curve_count, quality_count = int(assessment.oos_curve_pass.sum()), int(assessment.oos_quality_confirmed.sum())
        outcome_text = f'<p class="note caution"><b>{len(assessment)}个冻结候选中，{curve_count}个通过原OOS曲线标准，{quality_count}个通过额外交易质量确认。</b></p>'
        for row in assessment.itertuples(index=False):
            description = (f'{row.pair} {row.strategy}：OOS FILTER {row.oos_return_pct:.2f}%，BASE {row.oos_base_return_pct:.2f}%，'
                           f'相对变化 {row.oos_improvement_pp:+.2f} pp；胜率 {row.oos_win_rate * 100:.2f}% 对 BASE {row.oos_base_win_rate * 100:.2f}%。')
            if row.oos_return_pct > 0 and row.oos_improvement_pp < 0:
                description += ' 正收益、较小回撤或正收益年份不能替代相对BASE改善；该filter未验证通过。'
            elif row.oos_return_pct < 0:
                description += ' 相对BASE少亏仍是绝对亏损，不能据此称filter验证通过。'
            outcome_text += '<p>' + escape(description) + '</p>'
    audit_text = ('尚未保存独立审计结果。' if independent is None else
                  '独立审计状态：' + str(independent.get('status')) + '；详见 <a href="independent_validation.json">原审计 JSON</a>。')
    if post_audit is not None:
        audit_text += ' OOS后独立审计状态：' + escape(str(post_audit.get('status'))) + '；详见 <a href="post_oos_independent_validation.json">OOS后审计 JSON</a>。'
    attribution_html = ''
    if attribution is not None:
        if (attribution.get('status') != 'passed' or attribution.get('OOS_results_opened') is not False or
                attribution.get('compute_cutoff') != protocol['is_end']):
            raise ValueError('Independent IS contribution diagnostics have not passed.')
        attribution_html = '<section><h2>冻结候选的IS拒绝贡献与集中度</h2><p>避开的亏损和放弃的盈利来自原BASE被拒交易的实际区间收益归属。剔除最有帮助的三笔仅作限制披露，不新增筛选门槛，也不替换冻结者。<a href="' + attribution_name + '">独立归因原始 JSON</a></p>'
        attributed = {item['candidate_id']: item for item in attribution['candidates']}
        if set(attributed) != set(frozen.candidate_id):
            raise ValueError('IS contribution diagnostics must cover exactly the frozen candidates.')
        for cid in frozen.candidate_id:
            candidate = attributed[cid]
            attribution_html += '<h3>' + escape(candidate['candidate_id']) + '</h3>'
            period_rows = [{'period': values['block'], **values} for values in candidate['periods']]
            attribution_html += _table(pd.DataFrame(period_rows), ['period','avoided_loss_pp','foregone_gain_pp','net_improvement_pp',
                                                                  'top3_share_of_avoided_loss','delta_without_top3_helpful_pp'])
            whole = next(values for values in period_rows if values['period'] == 'IS')
            validation = next(values for values in period_rows if values['period'] == 'V')
            attribution_html += '<p>' + escape(f"IS前三笔有利作用占避损 {whole['top3_share_of_avoided_loss'] * 100:.2f}%；剔除后净改善 {whole['delta_without_top3_helpful_pp']:+.3f} pp。V期净改善 {validation['net_improvement_pp']:+.3f} pp。") + '</p>'
        attribution_html += '</section>'
    from fx_filter_report import _CSS
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>两个入场状态交互：完整研究报告</title><style>{_CSS}pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}.versions{{max-height:440px;overflow:auto}}</style></head><body>
<header><p class="eyebrow">TWO ENTRY STATES · AND REJECTION · ZERO COST</p><h1>两个入场状态交互：完整研究报告</h1>
<p class="lead">同一原始信号时刻的两个已知状态同时匹配，才拒绝原技术持仓整段。未知状态保留，两个条件都读取未过滤 BASE 历史。原综合标准保持不变。</p>
<div class="stats"><div class="stat"><strong>{len(coverage)}</strong><span>覆盖 BASE 账户</span></div><div class="stat"><strong>{len(selection)}</strong><span>D 提名并重放版本</span></div><div class="stat"><strong>{int(selection['pass'].sum())}</strong><span>原 IS 标准通过</span></div><div class="stat"><strong>{len(frozen)}</strong><span>按原规则冻结</span></div></div>
<p class="note">{escape(status_text)}</p>{outcome_text}<p>历史 IS 已反复研究，2023—2026 OOS 也曾在前序研究中暴露；本报告不宣称全新的未触碰样本外证据。</p>
<p><a href="{base_build['report']}">BASE/FILTER 基础报告与全部冻结图</a> · <a href="{base_build['atlas']}">PDF 图册</a> · <a href="#versions">全部版本</a> · <a href="#manifest">证据清单</a></p></header>
<main><section><h2>覆盖与完整候选范围</h2>{_table(by_pair)}<p>零提名账户也保留在180账户覆盖表中。每账户预先观察3种机制×16个联合状态；仅D支持的联合状态进入重放。</p><details><summary>全部180账户覆盖</summary>{_table(coverage)}</details></section>
<section><h2>三个固定机制与状态边界</h2>{_table(pd.DataFrame(mechanism_rows))}<p>状态0—3采用左闭右开区间；UNKNOWN=-1不拒绝。拒绝是AND，不能将两次过滤串联成OR。</p>{_table(pd.DataFrame(boundary_rows))}</section>
<section><h2>原综合门槛失败分布</h2><p>同一版本可同时失败多项，数量不能相加解释为版本总数。以下逐项读取原selection布尔列，没有替换胜率或盈亏比要求。</p>{_table(failures)}</section>
<section><h2>冻结候选：保存的IS顺序</h2>{_table(frozen, ['freeze_rank','candidate_id','filter_name','is_annualized_simple_pct','is_win_rate','is_payoff_ratio','is_closed_n'])}
{_table(assessment, ['freeze_rank','pair','strategy','oos_return_pct','oos_base_return_pct','oos_improvement_pp','oos_annualized_simple_pct','oos_max_drawdown_pp','oos_win_rate','oos_base_win_rate','oos_payoff_ratio','oos_base_payoff_ratio','oos_closed_n','oos_positive_years','oos_curve_pass','oos_quality_confirmed']) if oos_completed else '<p>新OOS尚未完成，不展示OOS指标。</p>'}</section>
<section><h2>拒绝集合与等价IS路径</h2><p>{equivalence_info['candidate_versions']}个版本对应{equivalence_info['distinct_rejected_episode_sets']}个同账户拒绝集合；{equivalence_info['duplicate_groups']}组含多个版本，共{equivalence_info['versions_in_duplicate_groups']}个版本。所有版本继续保留，等价分组不改变筛选。</p>
<p>完整集合包含末日pending指令；另存executed_rejection_set_sha256排除尚未成交的entry。相同完整拒绝集合的IS指标已核对一致。UNKNOWN保留数包含在accepted中，pending另列。</p>
{_table(equivalents.loc[equivalents.group_version_n.gt(1)], ['candidate_id','equivalence_group','group_version_n','rejected_n','unknown_retained_n'])}
<p><a href="interaction_decision_counts.csv">全部决定计数</a> · <a href="interaction_equivalent_paths.csv">全部等价分组及哈希</a></p></section>
<section><h2>D-only案例与反例</h2>{('<p><a href="' + case_name + '">独立D-only案例原始JSON</a></p>') if case_record is not None else ''}{_case_section(case_record)}</section>{attribution_html}
<section id="versions"><h2>全部候选版本：冻结顺序优先，其余保留原表顺序</h2><ol class="versions">{''.join(links)}</ol>{''.join(cards)}</section>
<section id="manifest"><h2>来源与复核</h2><p>{audit_text}</p><p>所有图表仅使用本目录保存结果。报告没有重新运行策略、修改原标准或选择替代候选。</p><ul>
<li><a href="protocol.json">冻结前协议</a> · <a href="frozen_selection.json">冻结记录</a> · <a href="is_selection.csv">完整IS筛选表</a></li>
<li><a href="interaction_report_build.json">报告输入、输出与源码 SHA256</a> · <a href="interaction_gate_failures.csv">原门槛失败分布</a></li></ul></section></main></body></html>'''
    (output / 'interaction_report.html').write_text(html, encoding='utf-8')
    inputs.unchanged()
    artifact_checks = validate_artifacts(output, base_build, len(frozen))
    products = ['interaction_report.html', 'interaction_decision_counts.csv', 'interaction_equivalent_paths.csv',
                'interaction_gate_failures.csv', 'report_build.json', base_build['report'], base_build['atlas'], *base_build['figures']]
    manifest = {
        'created_utc': core.now(), 'report': 'interaction_report.html', 'base_report': base_build['report'],
        'atlas': base_build['atlas'], 'figures': base_build['figures'], 'accounts': len(coverage),
        'candidate_versions': len(selection), 'original_standard_passing_versions': int(selection['pass'].sum()),
        'frozen_candidates': len(frozen), 'oos_evaluated': oos_completed, 'ordering': 'Saved freeze_rank first, then original is_selection row order.',
        'equivalence': equivalence_info, 'D_cases_present': case_record is not None,
        'artifact_validation': artifact_checks, 'IS_episode_attribution_present': attribution is not None,
        'post_OOS_independent_audit_present': post_audit is not None,
        'input_paths_relative_to': str(output), 'input_sha256': inputs.hashes, 'output_sha256': {name: sha(output / name) for name in products},
        'source_sha256': {name: sha(ROOT / name) for name in ['build_entry_interaction_report.py','run_filter_research.py','src/fx_filter_report.py']},
        'strategy_recomputed': False, 'selection_changed': False, 'old_OOS_read': False,
    }
    core.save_json(output / 'interaction_report_build.json', manifest)
    print(json.dumps({name: manifest[name] for name in ('report','accounts','candidate_versions','frozen_candidates','oos_evaluated','D_cases_present')}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    build_report(args.output)


if __name__ == '__main__':
    main()
