"""Standalone report for the fixed JPY original-criteria supplement.

This renderer never runs or imports strategy decisions. All research inputs are
direct children of --output. OOS artifacts are read only after that directory's
oos_status.json records status='passed'. Strict v2 and prior OOS are not read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "reproduced/state_filter_supplement_v1"
CANDIDATE_ID = "JPY|BB_MR_20|STATE_efficiency_63_1"
REPORT = "state_filter_supplement_report.html"
FIGURE = "figures/jpy_is_oos_comparison.png"
PDF = "jpy_is_oos_comparison.pdf"
FAIL_LABELS = {
    "annualized_pass": "单利年化门槛",
    "baseline_improvement_pass": "相对 BASE 的收益改善",
    "drawdown_pass": "最大回撤门槛",
    "positive_years_pass": "三个周年中正收益的期数",
    "sample_pass": "合格闭合交易数量",
    "outcome_sample": "确认期盈利和亏损各至少 5 笔",
    "recent_top3_robust": "RECENT 去掉最有利 3 个原始交易后改善仍为正",
}


def _truth(value):
    return value is not None and not pd.isna(value) and str(value).lower() in {"true", "1", "1.0"}


def _fmt(value, digits=2, suffix=""):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.{digits}f}{suffix}" if np.isfinite(number) else "—"


def _percent(value):
    try:
        return _fmt(float(value) * 100, suffix="%")
    except (TypeError, ValueError):
        return "—"


def _flag(value):
    if value is None or pd.isna(value):
        return "待验证"
    return "通过" if _truth(value) else "未通过"


def _table(headers, rows):
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>" for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}">待验证：尚无本阶段已完成的结果。</td></tr>'
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class Inputs:
    def __init__(self, directory):
        self.directory = directory
        self.hashes = {}

    def _read(self, name, required=False):
        if Path(name).name != name:
            raise ValueError("Only direct children of the supplement directory may be read.")
        path = self.directory / name
        if not path.exists():
            if required:
                raise ValueError(f"Required supplement artifact is missing: {name}")
            return None
        raw = path.read_bytes()
        self.hashes[name] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, name, default=None, required=False):
        raw = self._read(name, required)
        return json.loads(raw.decode("utf-8-sig")) if raw is not None else default

    def csv(self, name, required=False):
        raw = self._read(name, required)
        if raw is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.BytesIO(raw), compression="gzip" if name.endswith(".gz") else None)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


def _validate_candidate(frame, name, required=True):
    if frame.empty:
        if required:
            raise ValueError(f"Empty required supplement artifact: {name}")
        return
    if "candidate_id" not in frame or set(frame.candidate_id) != {CANDIDATE_ID}:
        raise ValueError(f"{name} must contain exactly the predetermined JPY candidate.")


def _metric(frame, split, variant):
    rows = frame.loc[frame.split.eq(split) & frame.variant.eq(variant)] if not frame.empty else pd.DataFrame()
    if len(rows) != 1:
        raise ValueError(f"Expected one saved metric for {split}/{variant}, got {len(rows)}.")
    return rows.iloc[0].to_dict()


def _metric_table(frame, splits):
    rows = []
    for split in splits:
        for variant in ("BASE", "FILTER"):
            saved = _metric(frame, split, variant)
            rows.append([split, str(saved.get("start_date", ""))[:10], str(saved.get("end_date", ""))[:10], variant, _fmt(saved.get("gross_simple_return_pct"), suffix="%"),
                         _fmt(saved.get("gross_annualized_simple_pct"), suffix="%"), _fmt(saved.get("gross_max_drawdown_pp")),
                         _percent(saved.get("win_rate")), _fmt(saved.get("payoff_ratio")),
                         _fmt(saved.get("closed_n"), 0), _fmt(saved.get("win_n"), 0), _fmt(saved.get("loss_n"), 0),
                         _percent(saved.get("retention")), _fmt(saved.get("carry_in_n"), 0), _fmt(saved.get("censored_entry_n"), 0),
                         "是" if _truth(saved.get("low_sample_flag")) else "否"])
    return _table(["区间", "实际起始日", "实际结束日", "版本", "单利累计", "单利年化", "最大回撤 pp", "胜率", "盈亏比", "闭合笔数", "盈利笔数", "亏损笔数", "闭合保留率", "跨界交易数", "期内开仓未平数", "闭合少于 20 笔"], rows)


def _observation_section(explanation):
    if not explanation:
        return ""
    periods = {row["period"]: row for row in explanation.get("periods", [])}
    full = periods.get("IS", {})
    rows = []
    for trade in full.get("top3_helpful_rejected_episodes", []):
        rows.append([str(trade.get("entry_date", ""))[:10], str(trade.get("exit_date", ""))[:10],
                     trade.get("direction", trade.get("position", "")), _fmt(trade.get("gross_simple_return_pct"), suffix="%"),
                     _fmt(trade.get("avoided_loss_contribution_pp")), _fmt(trade.get("holding_observations"), 0),
                     _fmt(trade.get("entry_efficiency_63"), 4)])
    html = '<section><h2>样本内改善的来源：三笔大亏与贡献集中</h2>'
    html += ('<p>随本补充保存的 IS 观察记录显示：全 IS 的 FILTER−BASE 累计改善为 <strong>'
             + _fmt(full.get("calendar_delta_filter_minus_base_pp"), 3) + ' pp</strong>，其中最有利的三笔被拒原始交易贡献 '
             + _fmt(full.get("best3_avoided_loss_contribution_pp"), 3) + ' pp，占净改善的 <strong>'
             + _percent(full.get("best3_share_of_net_filter_improvement")) + '</strong>。这解释了样本内结果对少数大亏交易的依赖。</p>')
    html += _table(["原始开仓日", "原始平仓日", "方向", "原始交易单利", "避开亏损贡献 pp", "持有报价日", "入场 efficiency_63"], rows)
    concentration = []
    for label in ("IS", "RECENT"):
        period = periods.get(label, {})
        if period:
            concentration.append([label, _fmt(period.get("calendar_delta_filter_minus_base_pp")),
                                  _percent(period.get("best3_share_of_net_filter_improvement")),
                                  _fmt(period.get("delta_excluding_best3_helpful_episodes_pp"))])
    html += _table(["区间", "净改善 pp", "前三笔贡献占净改善", "去掉前三笔后改善 pp"], concentration)
    html += '<p class="muted">贡献占比可以超过 100%：这表示其余交易对改善有负贡献。按期内开平交易分组求和与连续持仓的日历区间账户收益口径不同，不能混用。上述都是 IS 事实解释，不是新的筛选条件。</p>'
    hypothesis = explanation.get("candidate_mechanism_hypothesis", {})
    html += '<h3>待验证的机制假设</h3><p>' + escape(str(hypothesis.get("statement", "尚无已验证机制结论。"))) + '</p>'
    for key, label in (("observed_support", "观察支持"), ("limits", "解释限制"), ("falsifiable_follow_up", "后续可检验问题")):
        value = hypothesis.get(key)
        if value:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            html += '<p><strong>' + label + '：</strong>' + escape(text) + '</p>'
    html += '<p class="note">这些大亏事件说明本样本中改善来自何处；它们不能证明该 efficiency 区间能稳定预测未来亏损，也不构成因果确认。机制假设不改变本次已冻结规则或 OOS 判定。</p></section>'
    return html


def _plots(output, protocol, is_daily, is_metrics, oos_daily, oos_metrics, oos_complete):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_report_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True)
    windows = [("IS", protocol["is_start"], protocol["is_end"], is_daily, is_metrics),
               ("OOS", protocol["oos_start"], protocol["oos_end"], oos_daily, oos_metrics)]
    checks = []
    for ax, (split, start, end, daily, metrics) in zip(axes, windows):
        ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
        ax.set_title("IS: original comprehensive criteria" if split == "IS" else "OOS: reused historical evaluation")
        ax.set_ylabel("Cumulative simple return (%)")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4) if split == "IS" else mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if split == "IS" else "%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="#9ba7ae", linewidth=.7)
        ax.grid(alpha=.17)
        if split == "OOS" and not oos_complete:
            ax.text(.5, .5, "PENDING\nNo completed OOS results read", ha="center", va="center", transform=ax.transAxes, color="#687d88")
            continue
        if set(daily.split) != {split}:
            raise ValueError(f"Saved {split} daily artifact contains another split.")
        for variant, color in (("BASE", "#798893"), ("FILTER", "#007f84")):
            part = daily.loc[daily.variant.eq(variant)].copy()
            part["date"] = pd.to_datetime(part.date)
            part = part.sort_values("date")
            if part.empty or part.date.duplicated().any() or not part.date.between(start, end).all():
                raise ValueError(f"Invalid saved {split}/{variant} dates.")
            saved = _metric(metrics, split, variant)
            curve = part.gross_daily_return.to_numpy(float).cumsum() * 100
            if not np.isfinite(curve).all():
                raise ValueError(f"Nonfinite {split}/{variant} returns.")
            difference = float(curve[-1] - saved["gross_simple_return_pct"])
            if not np.isclose(difference, 0, atol=1e-8, rtol=0):
                raise ValueError(f"Daily/metric endpoint mismatch: {split}/{variant}.")
            checks.append({"split": split, "variant": variant, "absolute_error_pp": abs(difference)})
            anchor = pd.Timestamp(saved.get("anchor_date", start))
            dates = pd.DatetimeIndex([anchor]).append(pd.DatetimeIndex(part.date))
            ax.plot(dates, np.r_[0., curve], color=color, linewidth=1.6, label=f"{variant}: {curve[-1]:+.2f}%")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("JPY | BB_MR_20 | Reject signal-day efficiency_63 in [0.1, 0.2)\nFixed-candidate supplement; original IS pass; strict v2 fail", fontsize=12)
    fig.text(.5, .012, "Cost = 0; fixed capital simple returns. Display resets only; positions stay continuous. Historical OOS was previously observed.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, .89))
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / FIGURE, dpi=160, facecolor="white")
    with PdfPages(output / PDF) as document:
        document.infodict()["Title"] = "Fixed JPY state-filter supplement: IS and historical OOS"
        document.infodict()["Subject"] = "Original IS criteria pass; strict v2 fails; zero-cost fixed-capital simple returns"
        document.savefig(fig, facecolor="white")
    plt.close(fig)
    return checks


def _criteria(protocol):
    screen = protocol.get("screen", {})
    oos = protocol.get("oos_screen", {})
    original = f"单利年化 ≥{screen.get('min_gross_annualized_simple_pct', 1):g}%；累计相对 BASE 改善 ≥{screen.get('min_return_improvement_pp', 5):g} pp；胜率改善 ≥{screen.get('min_win_rate_improvement_pp', 2):g} pp；盈亏比相对改善 ≥{screen.get('min_payoff_relative_improvement', .1)*100:g}%；收益/回撤比改善；回撤恶化 ≤{screen.get('max_drawdown_deterioration_pp', 2):g} pp；闭合 ≥{screen.get('min_closed_trades', 50)} 笔、保留率 ≥{screen.get('min_closed_retention', .5)*100:g}%；至少两个 IS 分段收益为正、至少两个分段改善，末段须同时为正并改善。"
    future = f"单利年化 ≥{oos.get('min_net_annualized_simple_pct', 1):g}%；相对 BASE 累计改善 ≥{oos.get('min_net_return_improvement_pp', 0):g} pp；最大回撤 ≥{oos.get('worst_net_drawdown_pp', -15):g} pp；三个周年至少 {oos.get('min_positive_anniversary_years', 2)} 个正收益；闭合交易 ≥{oos.get('min_closed_trades', 10)} 笔。"
    return _table(["评价层次", "条件"], [("用户原综合 IS 标准", original), ("预写历史 OOS 曲线标准", future), ("OOS 质量另行确认", "闭合交易至少 20 笔，且胜率与盈亏比均不低于 BASE；曲线通过与质量确认分别展示。")])


def build_report(output_dir: Path) -> dict:
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError(f"Supplement directory does not exist: {output}")
    inputs = Inputs(output)
    protocol = inputs.json("protocol.json", required=True)
    if float(protocol.get("cost_bp", 0)) != 0:
        raise ValueError("The supplement report requires zero costs.")
    selection = inputs.csv("is_selection.csv", required=True)
    is_daily = inputs.csv("is_frozen_daily.csv.gz", required=True)
    is_metrics = inputs.csv("is_frozen_metrics.csv", required=True)
    diagnostics = inputs.json("supplement_diagnostics.json", {})
    explanation = inputs.json("jpy_observation_explanation.json", {})
    for name, frame in (("is_selection.csv", selection), ("is_frozen_daily.csv.gz", is_daily), ("is_frozen_metrics.csv", is_metrics)):
        _validate_candidate(frame, name)
    if len(selection) != 1:
        raise ValueError("The supplement must retain exactly one predetermined selection row.")
    chosen = selection.iloc[0].to_dict()
    if not _truth(chosen.get("pass")) or not _truth(chosen.get("frozen")) or int(chosen.get("freeze_rank", 0)) != 1:
        raise ValueError("The fixed candidate must pass the original IS screen and be frozen at rank 1.")
    if _truth(chosen.get("strict_v2_pass")):
        raise ValueError("The supplement must not relabel the strict v2 failure as a pass.")
    status = inputs.json("oos_status.json", {})
    oos_complete = status.get("status") == "passed"
    oos_daily = oos_metrics = assessment = annual = fresh = pd.DataFrame()
    if oos_complete:
        oos_daily = inputs.csv("oos_daily.csv.gz", required=True)
        oos_metrics = inputs.csv("oos_metrics.csv", required=True)
        assessment = inputs.csv("candidate_assessment.csv", required=True)
        annual = inputs.csv("oos_anniversary_metrics.csv", required=True)
        fresh = inputs.csv("oos_fresh_start_metrics.csv", required=True)
        for name, frame in (("oos_daily.csv.gz", oos_daily), ("oos_metrics.csv", oos_metrics), ("candidate_assessment.csv", assessment), ("oos_anniversary_metrics.csv", annual), ("oos_fresh_start_metrics.csv", fresh)):
            _validate_candidate(frame, name)
        if len(assessment) != 1:
            raise ValueError("Expected one fixed-candidate OOS assessment.")
        required_flags = {"annualized_pass", "baseline_improvement_pass", "drawdown_pass", "positive_years_pass", "sample_pass", "oos_curve_pass", "oos_quality_confirmed"}
        if not required_flags.issubset(assessment.columns) or assessment[list(required_flags)].isna().any().any():
            raise ValueError("Saved OOS assessment lacks a completed verdict.")
        for name, digest in status.get("output_sha256", {}).items():
            if name in inputs.hashes and inputs.hashes[name] != digest:
                raise ValueError(f"Saved OOS status hash differs: {name}")
    for daily in (is_daily, oos_daily):
        if not daily.empty and not np.allclose(daily.gross_daily_return, daily.net_daily_return, atol=1e-12, rtol=0):
            raise ValueError("Zero-cost daily gross/net returns must match.")
    all_metrics = pd.concat([is_metrics, oos_metrics], ignore_index=True)
    checks = _plots(output, protocol, is_daily, is_metrics, oos_daily, oos_metrics, oos_complete)
    result = assessment.iloc[0].to_dict() if oos_complete else {}
    curve_pass = _truth(result.get("oos_curve_pass")) if oos_complete else None
    quality_pass = _truth(result.get("oos_quality_confirmed")) if oos_complete else None
    if oos_complete:
        headline = f"固定 JPY 候选的补充历史 OOS 评估已完成：预写曲线标准{_flag(curve_pass)}，交易质量另行确认{_flag(quality_pass)}。"
        if not curve_pass:
            headline += "本候选未达到本次补充评估的预写 OOS 曲线目标。"
    else:
        headline = "固定 JPY 候选已按用户原综合 IS 标准通过并冻结；补充 OOS 待验证。尚未读取 OOS 每日曲线或结果。"
    body = [f'<section class="outcome"><h2>本次补充评估状态</h2><p>{escape(headline)}</p></section>',
            '<section><h2>为何单独补充</h2><p>本报告只评估固定候选 <strong>JPY · BB_MR_20 · STATE_efficiency_63_1</strong>。规则参数在 C 前已经锁定；<strong>看到 C 确认结果后</strong>，按用户此前确认的原综合标准，决定只对这个原先已锁定的 JPY 候选进行<strong>单独补充验证</strong>。因此，规则锁定发生在 C 前，补充晋级决策发生在 C 后，不能将本次补充称为统计独立验证。</p><p>该候选满足用户原综合 IS 标准；严格 v2 的额外时间确认与稳健性要求仍未通过：确认期 FILTER 只有 <strong>4 笔闭合亏损交易，少于 5 笔</strong>，且 RECENT 的改善集中于最有利的前三个原始交易，去掉其贡献后改善不再为正。这些额外门槛在本补充中用于披露限制；不把严格 v2 失败改成通过，不回写严格 v2 的筛选、冻结或结论，也不会在看到本次 OOS 后改规则、改阈值或补选其他候选。</p><p class="note">三年 OOS 是此前已经观察过的历史区间。本次运行隔离开发与评价流程，但属于历史重复验证，不能称为从未看过的新样本或新增的独立证据。</p></section>',
            '<section><h2>固定规则与收益口径</h2><p>BASE 使用 JPY 的 BB_MR_20。每个原始交易段在其<strong>信号日</strong>检查 efficiency_63：若 <code>0.1 ≤ efficiency_63 &lt; 0.2</code>，拒绝该整个原始交易段；区间外保留，特征缺失也保留。没有交易段内重新入场或根据后续收益改变拒绝条件。</p>',
            f'<p>固定初始本金 ${float(protocol.get("initial_capital_usd", 100000)):,.0f}，成本为 0，每日收益按初始本金累加；单利年化为累计收益除以实际日历年数。信号 t → t+1 收盘执行 → t+2 首次收益。IS 为 {escape(protocol["is_start"])}—{escape(protocol["is_end"])}，历史 OOS 为 {escape(protocol["oos_start"])}—{escape(protocol["oos_end"])}。</p>',
            '<p>主口径跨 IS/OOS 保持持仓连续。图中两个面板各自将展示的累计 PnL 归零，并使用相同纵轴尺度；不因此强制平仓。最大回撤是负 pp。胜率、盈亏比只统计在本区间内开仓并闭合的合格交易，跨界及未闭合交易的每日收益仍计入主曲线。</p>',
            _criteria(protocol), '</section>',
            '<section><h2>IS 与 OOS 主曲线</h2>',
            f'<p><a href="{PDF}">下载单页 PDF</a> · <a href="{FIGURE}">下载 PNG</a></p><img src="{FIGURE}" alt="JPY BASE/FILTER: IS and historical OOS cumulative simple return comparison">',
            _metric_table(all_metrics, ["IS", "OOS"] if oos_complete else ["IS"]), '</section>']
    if oos_complete:
        rows = []
        for key in ("annualized_pass", "baseline_improvement_pass", "drawdown_pass", "positive_years_pass", "sample_pass"):
            rows.append([FAIL_LABELS[key], _flag(result.get(key))])
        rows += [("全部预写 OOS 曲线条件", _flag(curve_pass)), ("OOS 胜率与盈亏比质量确认", _flag(quality_pass))]
        failed = [FAIL_LABELS.get(item, item) for item in str(result.get("failed_criteria", "")).split("|") if item and item != "nan"]
        body += ['<section><h2>预写 OOS 标准逐项结果</h2>', _table(["条件", "结果"], rows),
                 '<p>未通过的曲线条件：' + escape("；".join(failed) or "无") + '。</p><p class="muted">质量确认不代替累计收益、回撤、分年稳定性与样本量等曲线标准。少于 20 笔闭合交易时，交易质量仍视为低样本。</p></section>']
    else:
        body.append('<section><h2>预写 OOS 标准逐项结果</h2><p>待验证。只有本目录 oos_status.json 的状态为 passed，才读取已保存 OOS 结果。</p></section>')
    body.append('<section><h2>全部 OOS 周年区间</h2><p>按时间顺序保留三个预定周年区间及 BASE/FILTER 两个版本；这些是主 OOS 的分段诊断，不是另外三份独立样本。</p>')
    if oos_complete:
        expected = protocol.get("oos_anniversary_years", [])
        year_order = [item[0] for item in expected] if expected else sorted(annual.split.unique())
        if len(year_order) != 3 or set(annual.split) != set(year_order):
            raise ValueError("Expected all three predetermined OOS anniversary intervals.")
        body.append(_metric_table(annual, year_order))
    else:
        body.append('<p class="muted">待验证；尚未读取周年结果。</p>')
    body.append('</section><section><h2>Fresh-start 边界诊断</h2><p>主结果连续持仓。此处另从 OOS 起点空仓开始，只接受此后新开的原始 BASE 交易段，禁止在跨界旧交易段中途重新进入。它用于显示边界假设的影响，不能用其中较好的结果替换主口径或调整门槛。</p>')
    if oos_complete:
        body.append(_metric_table(fresh, ["OOS_FRESH_START"]))
        differences = []
        for variant in ("BASE", "FILTER"):
            continuous, empty_start = _metric(oos_metrics, "OOS", variant), _metric(fresh, "OOS_FRESH_START", variant)
            differences.append([variant, _fmt(continuous["gross_simple_return_pct"], suffix="%"), _fmt(empty_start["gross_simple_return_pct"], suffix="%"), _fmt(empty_start["gross_simple_return_pct"] - continuous["gross_simple_return_pct"]), _fmt(continuous["gross_max_drawdown_pp"]), _fmt(empty_start["gross_max_drawdown_pp"])])
        body.append(_table(["版本", "连续持仓累计", "空仓起步累计", "空仓−连续 pp", "连续回撤 pp", "空仓回撤 pp"], differences))
    else:
        body.append('<p class="muted">待验证；尚未读取 fresh-start 结果。</p>')
    body.append('</section>')
    if diagnostics:
        body.append('<section><h2>随本补充保存的严格 v2 诊断</h2><p>以下只使用本补充目录附带的记录。总闭合交易数不等于机制实际作用的样本数；受影响交易与贡献集中度单独列示。</p>')
        confirmation_records = diagnostics.get("confirmation_metrics", [])
        if isinstance(confirmation_records, dict):
            confirmation_records = list(confirmation_records.values())
        confirmation = pd.DataFrame(confirmation_records)
        if not confirmation.empty:
            if "split" not in confirmation:
                confirmation["split"] = "CONFIRMATION"
            body.append(_metric_table(confirmation, ["CONFIRMATION"]))
        effects = diagnostics.get("effects", {})
        rows = []
        for split in ("IS", "CONFIRMATION", "RECENT"):
            saved = effects.get(split, {}) if isinstance(effects, dict) else {}
            if saved:
                rows.append([split, _fmt(saved.get("delta_pp")), _fmt(saved.get("changed_closed_n"), 0), _fmt(saved.get("changed_episode_n"), 0), _fmt(saved.get("delta_without_best3_pp"))])
        if rows:
            body.append(_table(["区间", "FILTER−BASE pp", "受影响闭合原始交易", "全部受影响原始交易", "去掉最有利 3 个后改善 pp"], rows))
        body.append('<details><summary>查看随附诊断原始记录</summary><pre>' + escape(json.dumps(diagnostics, ensure_ascii=False, indent=2)) + '</pre></details></section>')
    body.append(_observation_section(explanation))
    body.append('<section><h2>可复核文件</h2><p>这里只链接本补充目录的实际输入；不读取或改写严格 v2 或旧 OOS 产物。渲染输入哈希记于 <a href="supplement_report_build.json">报告 manifest</a>。</p><ul>')
    body.extend(f'<li><a href="{escape(name)}">{escape(name)}</a></li>' for name in inputs.hashes)
    body.append('</ul><details><summary>查看补充协议</summary><pre>' + escape(json.dumps(protocol, ensure_ascii=False, indent=2)) + '</pre></details></section>')
    generated = datetime.now(timezone.utc).isoformat()
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JPY 固定状态过滤：原综合标准补充评估</title><style>
*{box-sizing:border-box}body{margin:0;color:#193341;background:#f2f5f7;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:30px 26px 50px}h1{font-size:29px;line-height:1.4}h2{font-size:22px;margin:0 0 12px}p{max-width:1180px}section{background:white;border:1px solid #dce5ea;padding:24px;margin:22px 0}.outcome{border-left:5px solid #b87c26;background:#fff4de}.outcome p{font-weight:600;font-size:18px}.muted{color:#627784;font-size:13px}.note{border-left:4px solid #16878a;background:#eaf5f6;padding:12px 16px}.table-wrap{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:12px;min-width:640px}th,td{text-align:left;padding:9px 10px;border:1px solid #dce5ea;vertical-align:top}th{background:#edf3f6;white-space:nowrap}tr:nth-child(even){background:#f8fafb}img{width:100%;height:auto}a{color:#006c78}details{margin-top:16px;border-top:1px solid #dce5ea;padding-top:12px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}footer{font-size:12px;color:#627784}@media(max-width:650px){main{padding:18px 12px}section{padding:18px 12px}h1{font-size:24px}}
</style></head><body><main><h1>JPY 固定状态过滤：原综合标准补充评估</h1><p class="muted">单一固定候选 · 成本 0 · 固定本金单利 · 已观察历史重复验证</p>'''
    html += "\n".join(body) + f'<footer>生成时间：{escape(generated)} UTC。单页 PDF 为主曲线图；完整解释和诊断见本 HTML。</footer></main></body></html>'
    (output / REPORT).write_text(html, encoding="utf-8")
    built = {"report": REPORT, "figure": FIGURE, "figures": [FIGURE], "pdf": PDF, "pdf_pages": 1,
             "candidate_id": CANDIDATE_ID, "original_is_pass": True, "strict_v2_pass": False,
             "oos_evaluated": oos_complete, "oos_curve_pass": curve_pass, "oos_quality_confirmed": quality_pass,
             "counts": {"frozen_candidates": 1, "oos_candidates": int(oos_complete),
                        "curve_pass": int(curve_pass) if oos_complete else None,
                        "quality_confirmed": int(quality_pass) if oos_complete else None},
             "daily_metric_endpoint_checks": checks, "input_sha256": inputs.hashes,
             "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "generated_utc": generated}
    (output / "supplement_report_build.json").write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
