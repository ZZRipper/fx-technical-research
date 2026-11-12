"""Offline, descriptive reports for all reconstructed technical BASE accounts.

The renderer accepts computed tables; it neither fits a filter nor changes a
position.  All links are relative and all plotting uses the noninteractive Agg
backend.  English plot labels avoid dependence on a particular Chinese font.
"""

from html import escape
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from fx_baselines import CURRENCIES, RULES, STRATEGIES


_BLUE = "#126c91"
_GRAY = "#7d8792"
_RED = "#b3494f"
_GREEN = "#25856b"

_LABELS = {
    "pair": "币种", "strategy": "策略", "period": "区间", "period_type": "区间类型",
    "start": "起始日", "end": "结束日", "start_date": "起始日", "end_date": "结束日",
    "simple_return_pct": "单利收益 %", "compound_return_pct": "复利收益 %",
    "simple_pnl_usd": "单利 PnL / USD", "compound_pnl_usd": "复利 PnL / USD",
    "closed_trades": "已平仓交易数", "win_rate": "胜率 %", "payoff_ratio": "盈亏比",
    "simple_max_drawdown_pp": "最大单利回撤 pp", "observations": "报价日数",
    "active_fraction": "持仓日比例 %", "active_pct": "持仓日比例 %",
    "rank": "排序", "kind": "类型", "horizon": "窗口报价日数", "window": "窗口报价日数",
    "feature": "特征", "state": "状态", "value": "数值", "count": "样本数",
    "entry_date": "开仓日", "exit_date": "平仓日", "position": "方向",
    "holding_observations": "持有报价日数", "closed": "已平仓",
    "peak_date": "前峰日期", "trough_date": "谷底日期", "recovery_date": "恢复日期",
    "depth_pp": "回撤深度 pp", "duration_observations": "持续报价日数",
    "active_days": "持仓日数", "complete_calendar_period": "完整日历区间",
    "cohort_closed_trades": "期内开仓并平仓数", "cohort_win_rate": "区间交易胜率 %",
    "cohort_payoff_ratio": "区间交易盈亏比", "cohort_censored_trades": "期内开仓未平数",
    "decline_start_date": "开始回落日期", "observed_end_date": "观测结束日",
    "recovered": "已恢复", "decline_observations": "下降报价日数",
    "recovery_observations": "恢复报价日数", "positive_years": "正收益年数",
    "observed_years": "观测年数", "positive_year_fraction": "正收益年份 / 观测年份",
    "largest_loss_share_of_gross_losses": "最大亏损交易占总亏损 %",
    "top3_loss_share_of_gross_losses": "前三亏损交易占总亏损 %",
    "losers_with_prior_gain_fraction": "亏损交易曾有浮盈比例 %",
    "mfe_simple_pct": "交易中最大浮盈 %", "mae_simple_pct": "交易中最深浮亏 %",
    "giveback_simple_pct": "从最大浮盈回吐 pp", "loser_with_prior_gain": "曾浮盈后亏损",
    "trade_id": "交易编号", "sample_role": "展示理由", "signal_date": "信号日",
    "coverage": "特征覆盖率 %", "n_feature_win": "有特征赢家数",
    "n_feature_loss": "有特征输家数", "win_mean": "赢家特征均值",
    "loss_mean": "输家特征均值", "win_median": "赢家特征中位数",
    "loss_median": "输家特征中位数", "standardized_mean_difference": "赢家−输家 / 合并标准差",
    "low_sample_flag": "低样本", "cohort": "开仓 cohort",
}

_FEATURE_LABELS = {
    "spot_return_20_pct": "20日价格收益 %", "spot_return_63_pct": "63日价格收益 %",
    "realized_vol_20_pct": "20日年化实现波动 %", "realized_vol_63_pct": "63日年化实现波动 %",
    "efficiency_20": "20日价格路径效率", "efficiency_63": "63日价格路径效率",
    "distance_ma_20_pct": "偏离20日均线 %", "distance_ma_60_pct": "偏离60日均线 %",
    "vol_ratio_20_63": "20日 / 63日波动比",
    "direction_spot_return_20_pct": "持仓方向 × 20日价格收益 %",
    "direction_spot_return_63_pct": "持仓方向 × 63日价格收益 %",
    "direction_distance_ma_20_pct": "持仓方向 × 偏离20日均线 %",
    "direction_distance_ma_60_pct": "持仓方向 × 偏离60日均线 %",
    "K_vix": "前日已知 VIX", "K_vix_d5": "前日已知 VIX 5观测变化",
    "K_spread_bp": "本国−美国2年期利差 bp", "K_spread_d20_bp": "利差20观测变化 bp",
    "K_dxy_d20_pct": "DXY 20观测变化 %", "direction_spread_bp": "持仓方向 × 利差 bp",
    "direction_spread_d20_bp": "持仓方向 × 利差变化 bp",
    "direction_dxy_d20_pct": "持仓方向 × DXY变化 %",
    "commodity_d20_pct": "适用商品20观测变化 %",
    "direction_commodity_d20_pct": "持仓方向 × 适用商品变化 %",
}

_CSS = """
:root{color-scheme:light;--ink:#173149;--muted:#5c6a77;--line:#dce4e9;--blue:#126c91}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f7f9;color:var(--ink);font:15px/1.75 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}
a{color:#075c83;text-underline-offset:3px}header,main,footer{max-width:1320px;margin:auto;padding:26px 34px}header{padding-top:48px;padding-bottom:30px}h1{font-size:34px;line-height:1.3;margin:8px 0 14px}h2{font-size:25px;margin:0 0 15px}h3{font-size:18px;margin:22px 0 9px}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--blue);font-weight:750}.lead{max-width:960px;font-size:17px;color:var(--muted)}
section{background:white;border:1px solid var(--line);border-radius:12px;margin:0 0 24px;padding:27px;scroll-margin-top:18px}.note{background:#eef6fa;border-left:4px solid var(--blue);padding:14px 18px;margin:16px 0}.caution{background:#fff6e8;border-color:#bb8540}.muted,figcaption{color:var(--muted);font-size:13px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.stat{background:white;border:1px solid var(--line);padding:15px 18px;border-radius:8px}.stat strong{display:block;font-size:25px}.stat span{font-size:12px;color:var(--muted)}
nav{display:flex;flex-wrap:wrap;gap:8px 15px;margin:16px 0}nav a{font-size:12px;padding:3px 8px;background:#edf3f7;border-radius:4px}figure{margin:16px 0 22px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:white}figcaption{padding-top:8px}.table-wrap{overflow-x:auto;margin:12px 0 18px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:8px 10px;border-bottom:1px solid #e6ecf0;text-align:right;font-variant-numeric:tabular-nums}th{background:#edf3f7;color:#3c5364;font-weight:650}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#f6f9fb}.pos{color:#22745d}.neg{color:#ac3941}.table-note{font-size:12px;color:var(--muted)}details{border-top:1px solid var(--line);padding-top:12px;margin-top:16px}summary{cursor:pointer;font-weight:650}code{background:#edf3f7;border-radius:3px;padding:2px 4px}ul,ol{padding-left:23px}.hypothesis{border-left:3px solid #8aa9bb;padding-left:17px;margin:19px 0}.back{float:right;font-size:12px}.formula{font-family:ui-monospace,Consolas,monospace;overflow-wrap:anywhere}footer{color:var(--muted);font-size:12px;padding-bottom:40px}
@media(max-width:720px){header,main,footer{padding:20px 14px}header{padding-top:28px}h1{font-size:28px}section{padding:18px 14px}.stats{grid-template-columns:repeat(2,1fr)}.stat strong{font-size:22px}h2{font-size:21px}nav{gap:6px}nav a{font-size:11px}}
@media print{body{background:white}header,main,footer{padding:10px;max-width:none}section{border:0;page-break-before:auto;padding:12px 0}nav,.back{display:none}.table-wrap{overflow:visible}table{white-space:normal}figure{break-inside:avoid}.strategy{break-before:page}a{color:inherit;text-decoration:none}details{display:block}details>*{display:block}summary{display:none}}
"""


def _scalar(value, column=""):
    if value is None or value is pd.NaT or (np.isscalar(value) and pd.isna(value)):
        return "—"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if column in {"win_rate", "cohort_win_rate", "active_fraction", "coverage", "largest_loss_share_of_gross_losses", "top3_loss_share_of_gross_losses", "losers_with_prior_gain_fraction"}:
            value *= 100
        if column.endswith("_usd"):
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def _table(frame, columns=None, limit=None, labels=None):
    """Escape every supplied value; schema-specific selection happens upstream."""
    if frame is None or frame.empty:
        return '<p class="muted">本项没有符合定义的记录。</p>'
    columns = [column for column in (columns or list(frame.columns)) if column in frame]
    if not columns:
        return '<p class="muted">本项没有可显示的字段。</p>'
    selected = frame if limit is None else frame.head(limit)
    names = {**_LABELS, **(labels or {})}
    header = "".join(f"<th>{escape(names.get(c, c))}</th>" for c in columns)
    rows = []
    for values in selected[columns].itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, values):
            color = ""
            if isinstance(value, (int, float, np.number)) and ("return" in column or "pnl" in column):
                color = ' class="pos"' if value > 0 else (' class="neg"' if value < 0 else "")
            cells.append(f"<td{color}>{escape(_scalar(value, column))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = ""
    if limit is not None and len(frame) > limit:
        note = f'<p class="table-note">此处显示前 {limit:,} 行；完整 {len(frame):,} 行见对应 CSV。</p>'
    return '<div class="table-wrap"><table><thead><tr>' + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + note


def _for_strategy(frame, strategy):
    if frame is None or frame.empty or "strategy" not in frame:
        return pd.DataFrame()
    return frame.loc[frame.strategy == strategy].copy()


def _required(frame, columns, name):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _plot_strategy(strategy, daily, accounts, rolling, destination, pdf):
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.7), sharex=True, sharey=True)
    rule = RULES.get(strategy, ("", "", ""))
    fig.suptitle(strategy + "  |  BASE technical accounts", x=.065, y=.97, ha="left", fontsize=21, fontweight="bold", color="#173149")
    fig.text(.065, .934, textwrap.fill(rule[2], 135), ha="left", fontsize=10, color="#536878")
    account_map = accounts.set_index("pair")
    for axis, pair in zip(axes.flat, CURRENCIES):
        group = daily.loc[daily.pair == pair].sort_values("date")
        dates = pd.to_datetime(group.date)
        axis.plot(dates, group.simple_pnl_usd.to_numpy() / 1000, color=_BLUE, linewidth=1.05)
        axis.plot(dates, group.compound_pnl_usd.to_numpy() / 1000, color=_GRAY, linewidth=.75, linestyle="--", alpha=.85)
        if not rolling.empty and {"pair", "kind", "start_date", "end_date", "simple_return_pct"}.issubset(rolling.columns):
            windows = rolling.loc[rolling.pair == pair]
            for kind, color, ascending in [("positive", _GREEN, False), ("negative", _RED, True)]:
                chosen = windows.loc[windows.kind == kind].sort_values("simple_return_pct", ascending=ascending)
                if not chosen.empty:
                    span = chosen.iloc[0]
                    axis.axvspan(pd.Timestamp(span.start_date), pd.Timestamp(span.end_date), color=color, alpha=.1, linewidth=0)
        axis.axhline(0, color="#bbc8d1", linewidth=.65)
        row = account_map.loc[pair]
        axis.set_title(f"{pair}   simple {row.simple_return_pct:+.1f}%   |   max DD {row.simple_max_drawdown_pp:.1f} pp", loc="left", fontsize=10, pad=10)
        axis.xaxis.set_major_locator(mdates.YearLocator(5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(axis="y", color="#e5ebef", linewidth=.6)
        axis.tick_params(labelsize=8, colors="#546978")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#bdcad3")
    for axis in axes[:, 0]:
        axis.set_ylabel("Cumulative PnL (USD thousands)", fontsize=9, color="#536878")
    handles = [Line2D([0], [0], color=_BLUE, lw=1.8, label="Simple: fixed initial USD 100,000"), Line2D([0], [0], color=_GRAY, lw=1.2, ls="--", label="Compound: changing account equity")]
    if not rolling.empty:
        handles.extend([Patch(facecolor=_GREEN, alpha=.2, label="Strongest selected 252-quote window"), Patch(facecolor=_RED, alpha=.2, label="Weakest selected 252-quote window")])
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.06, .042), ncol=2, frameon=False, fontsize=9)
    fig.text(.065, .024, "Shaded windows are hindsight diagnostics, never entry conditions. Independent accounts; same vertical scale within this page.", fontsize=8, color="#667784")
    fig.text(.065, .012, "Gross spot PnL, zero transaction costs/carry/financing. Previously examined history; not fresh out-of-sample evidence.", fontsize=8, color="#667784")
    fig.subplots_adjust(left=.065, right=.98, top=.885, bottom=.115, hspace=.29, wspace=.15)
    try:
        fig.savefig(destination, dpi=145, facecolor="white")
        pdf.savefig(fig, facecolor="white")
    finally:
        plt.close(fig)


def _period_sections(periods, strategy):
    selected = _for_strategy(periods, strategy)
    if selected.empty:
        return ""
    columns = [c for c in selected.columns if c != "strategy"]
    annual = selected.loc[selected.period_type == "year"]
    heatmap = ""
    if not annual.empty:
        values = annual.pivot(index="period", columns="pair", values="simple_return_pct").reindex(columns=CURRENCIES).sort_index()
        full = annual.groupby("period").complete_calendar_period.all()
        headers = "<th>年份</th>" + "".join(f"<th>{pair}</th>" for pair in CURRENCIES)
        rows = []
        for period, returns in values.iterrows():
            suffix = " *" if not full.get(period, False) else ""
            cells = [f"<td>{escape(str(period) + suffix)}</td>"]
            for value in returns:
                if pd.isna(value):
                    cells.append("<td>—</td>")
                else:
                    alpha = min(abs(float(value)) / 20, 1) * .2
                    rgb = "37,133,107" if value >= 0 else "179,73,79"
                    cells.append(f'<td style="background:rgba({rgb},{alpha:.3f})">{value:+.2f}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        heatmap = '<details><summary>年度收益矩阵：每年 × 每个币种</summary><p class="table-note">单位为初始本金收益 %。绿色为正、红色为负；颜色强度统一按绝对收益 20% 封顶。带 * 的年份不是完整日历年，不能与整年收益直接比较。</p><div class="table-wrap"><table><thead><tr>' + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>"
    return heatmap + "<details><summary>日历年 / 半年度完整指标</summary><p class=\"table-note\">收益包含区间内全部持仓日收益，包括跨期持仓；交易统计仅包含区间内开仓并于区间末前实际平仓且持有至少一个收益观测的交易。两种统计总体不同。</p>" + _table(selected, columns) + "</details>"


def _window_digest(rolling, strategy):
    selected = _for_strategy(rolling, strategy)
    if selected.empty:
        return ""
    rows = []
    for pair in CURRENCIES:
        windows = selected.loc[selected.pair == pair]
        row = {"pair": pair}
        for kind, label, ascending in [("positive", "positive", False), ("negative", "negative", True)]:
            chosen = windows.loc[windows.kind == kind].sort_values("simple_return_pct", ascending=ascending)
            if chosen.empty:
                row[label + "_period"] = "没有所选窗口"
                row[label + "_return_pct"] = np.nan
            else:
                best = chosen.iloc[0]
                row[label + "_period"] = f"{pd.Timestamp(best.start_date):%Y-%m-%d} → {pd.Timestamp(best.end_date):%Y-%m-%d}"
                row[label + "_return_pct"] = best.simple_return_pct
        rows.append(row)
    return '<h3>各币种最强 / 最弱的所选 252 日窗口</h3><p class="table-note">每个窗口覆盖 252 个有效报价日；按绝对单利收益排序后贪婪选择不重叠区间，每种盈亏符号最多保留 3 个。本表各显示 1 个；完整所选窗口见下方。区间是事后识别，图中的淡绿 / 淡红背景对应本表。</p>' + _table(pd.DataFrame(rows), labels={"positive_period": "最强正收益窗口", "positive_return_pct": "单利收益 %", "negative_period": "最弱负收益窗口", "negative_return_pct": "单利收益 %"})


def _diagnostics(title, frame, strategy, note, columns=None, limit=None):
    selected = _for_strategy(frame, strategy)
    return f"<details><summary>{escape(title)}</summary><p class=\"table-note\">{escape(note)}</p>" + _table(selected, columns, limit) + "</details>"


def _drawdown_section(drawdowns, strategy):
    selected = _for_strategy(drawdowns, strategy)
    if selected.empty:
        return ""
    shown = pd.concat([selected.loc[selected.pair == pair].nlargest(3, "depth_pp") for pair in CURRENCIES], ignore_index=True)
    columns = ["pair", "peak_date", "decline_start_date", "trough_date", "recovery_date", "observed_end_date", "recovered", "depth_pp", "duration_observations", "decline_observations", "recovery_observations"]
    return '<details><summary>峰谷回撤与恢复：每个币种最深的 3 段</summary><p class="table-note">此处回撤深度按正的损失幅度显示，单位为初始本金百分点；账户摘要中的最大回撤以负数显示。未恢复区间观测至样本末。持续报价日数从峰值算到恢复日或样本末，下降与恢复日数分列；全部回撤见 CSV。</p>' + _table(shown, columns) + "</details>"


def _trade_section(trades, accounts, strategy):
    selected = _for_strategy(trades, strategy)
    if selected.empty or "simple_return_pct" not in selected:
        return ""
    eligible = selected.loc[selected.eligible_closed.astype(bool)] if "eligible_closed" in selected else selected.loc[selected.closed.astype(bool) & selected.holding_observations.gt(0)]
    pieces = []
    for pair in CURRENCIES:
        group = eligible.loc[eligible.pair == pair]
        wins = group.loc[group.simple_return_pct > 1e-10].nlargest(2, "simple_return_pct").assign(sample_role="最大盈利交易")
        losses = group.loc[group.simple_return_pct < -1e-10].nsmallest(2, "simple_return_pct").assign(sample_role="最大亏损交易")
        pieces.extend([wins, losses])
    shown = pd.concat(pieces, ignore_index=True)
    concentration = _for_strategy(accounts, strategy)
    concentration_columns = ["pair", "largest_loss_share_of_gross_losses", "top3_loss_share_of_gross_losses", "losers_with_prior_gain_fraction"]
    columns = ["pair", "sample_role", "trade_id", "signal_date", "entry_date", "exit_date", "position", "holding_observations", "simple_return_pct", "mfe_simple_pct", "mae_simple_pct", "giveback_simple_pct", "loser_with_prior_gain"]
    return '<details><summary>交易段：亏损集中度、浮盈回吐与代表性大额盈亏</summary><p class="table-note">集中度分母是全部已平仓亏损交易的亏损绝对值之和，不是账户净亏损。每个币种展示 2 笔最大盈利与 2 笔最大亏损的已平仓交易，属于后见例子，不能单独代表总体。MFE / MAE 仅为可观测收盘时点的交易内最大浮盈 / 最深浮亏，含开仓零点，并非日内止损可成交价格；“曾浮盈”也未说明能提前识别回吐。</p>' + _table(concentration, concentration_columns) + _table(shown, columns) + '<p class="table-note">未平仓交易截至样本末的每日盈亏仍计入账户净值，但未进入上表与已平仓胜率。完整账本、所有交易的特征及窗口内截取后的交易贡献见下载数据。</p></details>'


def _contrast_section(contrasts, strategy):
    selected = _for_strategy(contrasts, strategy)
    if selected.empty:
        return ""
    if "cohort" in selected:
        selected = selected.loc[selected.cohort == "FULL"].copy()
    selected["feature"] = selected.feature.map(lambda value: _FEATURE_LABELS.get(value, value))
    columns = ["pair", "feature", "coverage", "n_feature_win", "n_feature_loss", "win_mean", "loss_mean", "win_median", "loss_median", "standardized_mean_difference", "low_sample_flag"]
    return '<details><summary>入场特征对照：赢家与输家的全部 23 项特征</summary><p class="table-note">展示 FULL 开仓 cohort 的完整特征，不按效果量筛选变量；分时期 2005–2014、2015–2019、2020至样本末的对照见 CSV。这些时期也都已看过。特征取信号日或更早资料，宏观观测日必须严格早于信号日且不超过 7 个日历日；这仍不等于已掌握原始发布时间或历史修订版本。</p><p class="table-note">覆盖率分母为适用该特征的合格已平仓交易。商品变量仅覆盖适用币种，其他币种的“—”表示不适用。效果量为赢家均值减输家均值，再除以合并标准差；任一特征有效赢家或输家少于 20 笔标为低样本。“持仓方向 × DXY变化”大于零表示与外币仓位方向冲突。正负差异均非因果、预测能力或可交易阈值的证据。</p>' + _table(selected, columns) + "</details>"


def _hypotheses(metadata):
    hypotheses = metadata.get("hypotheses", [])
    if not hypotheses:
        return "<p>先从每个策略的持续盈利段、持续亏损段和峰谷回撤识别问题，再查阅状态对照表。单个区间的事后解释尚不能确定可交易的过滤条件。</p>"
    blocks = []
    labels = {"title": "候选假设", "strategy": "策略", "strategies": "适用策略", "observation": "观测", "evidence": "证据", "mechanism": "可能机制", "candidate": "候选过滤方向", "condition": "候选条件", "test": "待检验设计", "limitation": "限制", "next_step": "下一步"}
    for index, hypothesis in enumerate(hypotheses, 1):
        if isinstance(hypothesis, str):
            content = "<p>" + escape(hypothesis) + "</p>"
        else:
            pieces = []
            for key, value in hypothesis.items():
                if isinstance(value, (list, tuple)):
                    value = "；".join(str(item) for item in value)
                elif isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False)
                pieces.append(f"<p><b>{escape(labels.get(key, key))}：</b>{escape(str(value))}</p>")
            content = "".join(pieces)
        blocks.append(f'<div class="hypothesis"><h3>H{index:02d}</h3>{content}</div>')
    return "".join(blocks)


def _csv_links(output_dir, metadata):
    configured = metadata.get("csv_files")
    if isinstance(configured, dict):
        items = [(str(label), str(path)) for label, path in configured.items()]
    elif isinstance(configured, (list, tuple)):
        items = [(str(path), str(path)) for path in configured]
    else:
        items = [(path.name, path.relative_to(output_dir).as_posix()) for path in sorted(output_dir.glob("*.csv*"))]
    links = []
    for label, relative in items:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Report CSV link must be relative to its output directory: {relative}")
        links.append(f'<li><a href="{escape(relative.replace(chr(92), "/"), quote=True)}">{escape(label)}</a></li>')
    return "<ul>" + "".join(links) + "</ul>" if links else '<p class="muted">CSV 与本报告保存在同一输出目录。</p>'


def render_report(output_dir: Path, daily: pd.DataFrame, accounts: pd.DataFrame,
                  periods: pd.DataFrame, rolling: pd.DataFrame, drawdowns: pd.DataFrame,
                  trades: pd.DataFrame, contrasts: pd.DataFrame, metadata: dict) -> dict:
    """Write twenty nine-currency figures, a twenty-page PDF, and a Chinese report.

    Returns relative artifact paths and SHA-256 values for the caller's manifest.
    Existing historical dashboards and research outputs are never accessed here.
    """
    _required(daily, ["pair", "strategy", "date", "daily_return", "held", "simple_pnl_usd", "compound_pnl_usd"], "daily")
    _required(accounts, ["pair", "strategy", "simple_return_pct", "compound_return_pct", "closed_trades", "win_rate", "payoff_ratio", "simple_max_drawdown_pp"], "accounts")
    expected = {(pair, strategy) for pair in CURRENCIES for strategy in STRATEGIES}
    actual = set(accounts[["pair", "strategy"]].itertuples(index=False, name=None))
    daily_keys = set(daily[["pair", "strategy"]].drop_duplicates().itertuples(index=False, name=None))
    if actual != expected or daily_keys != expected or accounts.duplicated(["pair", "strategy"]).any():
        raise ValueError("The report requires exactly the 20 technical rules x 9 currencies, with one account row per pair/rule.")
    if daily.duplicated(["pair", "strategy", "date"]).any():
        raise ValueError("Daily curves contain duplicate account/date observations.")
    if not np.isfinite(daily[["daily_return", "simple_pnl_usd", "compound_pnl_usd"]].to_numpy(dtype=float)).all():
        raise ValueError("Daily curves contain nonfinite accounting values.")
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "pnl_atlas.pdf"
    html_path = output_dir / "observation_report.html"
    figure_paths = []
    overview_rows = []
    sections = []
    dates = pd.to_datetime(daily.date)
    start, end = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": False}), PdfPages(pdf_path) as pdf:
        pdf.infodict().update(Title="FX BASE PnL Atlas: 20 rules x 9 currencies", Author="FX Technical Research", Subject="Historical, descriptive technical-strategy diagnostics; fixed-initial-capital and compound PnL")
        for strategy in STRATEGIES:
            account_rows = _for_strategy(accounts, strategy)
            account_rows["pair"] = pd.Categorical(account_rows.pair, categories=CURRENCIES, ordered=True)
            account_rows = account_rows.sort_values("pair")
            figure_path = figure_dir / f"{strategy}.png"
            _plot_strategy(strategy, _for_strategy(daily, strategy), account_rows, _for_strategy(rolling, strategy), figure_path, pdf)
            relative = figure_path.relative_to(output_dir).as_posix()
            figure_paths.append(relative)
            positive = int((account_rows.simple_return_pct > 0).sum())
            overview_rows.append({"strategy": strategy, "positive_currencies": f"{positive} / 9", "median_simple_return_pct": account_rows.simple_return_pct.median(), "min_simple_return_pct": account_rows.simple_return_pct.min(), "max_simple_return_pct": account_rows.simple_return_pct.max(), "median_drawdown_pp": account_rows.simple_max_drawdown_pp.median()})
            if {"positive_years", "observed_years"}.issubset(account_rows.columns):
                account_rows["positive_year_fraction"] = account_rows.positive_years.astype(str) + " / " + account_rows.observed_years.astype(str)
            account_columns = ["pair", "simple_return_pct", "compound_return_pct", "simple_max_drawdown_pp", "closed_trades", "win_rate", "payoff_ratio", "positive_year_fraction"]
            sections.append(f'<section class="strategy" id="{strategy}"><a class="back" href="#contents">返回目录 ↑</a><p class="eyebrow">{escape(RULES[strategy][0])} · {escape(RULES[strategy][1])}</p><h2>{escape(strategy)}</h2><p>{escape(RULES[strategy][2])}</p><p>全期单利为正：<b>{positive} / 9</b> 个币种；各账户的路径和阶段表现如下。全期正收益不代表每个阶段均有效。</p><figure><a href="{relative}"><img src="{relative}" loading="lazy" width="2392" height="1696" alt="{strategy} 的 9 个币种独立账户单利与复利累计 PnL"></a><figcaption>蓝色实线：固定初始本金单利；灰色虚线：随账户权益复利。同页九个面板采用相同纵轴，单位为千美元。点击打开原图。</figcaption></figure>'
                + _table(account_rows, account_columns)
                + _window_digest(rolling, strategy)
                + _period_sections(periods, strategy)
                + _diagnostics("有效 / 失效窗口：完整所选区间", rolling, strategy, "按绝对单利结果从大到小联合选择不重叠窗口，各正负符号分别计 rank、各最多 3 个，不保证每个账户都有六个区间。非重叠仅减少重复叙述，不构成独立统计试验。", columns=["pair", "kind", "rank", "start_date", "end_date", "observations", "active_days", "simple_return_pct", "compound_return_pct", "cohort_closed_trades", "cohort_win_rate", "cohort_payoff_ratio"])
                + _drawdown_section(drawdowns, strategy)
                + _trade_section(trades, accounts, strategy)
                + _contrast_section(contrasts, strategy)
                + "</section>")
    overview = pd.DataFrame(overview_rows)
    navigation = "".join(f'<a href="#{strategy}">{strategy}</a>' for strategy in STRATEGIES)
    generated = metadata.get("built_utc", metadata.get("generated_utc", ""))
    body = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FX 全策略 PnL 观测报告</title><style>{_CSS}</style></head><body>
<header><p class="eyebrow">FX TECHNICAL RESEARCH · BASE OBSERVATIONS</p><h1>从 PnL 路径寻找有效与失效时期</h1><p class="lead">覆盖全部 20 条技术规则与 9 个币种，以固定初始本金单利为主、原复利为参照。先记录哪些时期赚钱或亏钱，再提出有待单独检验的 filter 假设。</p><div class="stats"><div class="stat"><strong>20 × 9</strong><span>技术规则 × 独立币种账户</span></div><div class="stat"><strong>{len(daily):,}</strong><span>账户日观测</span></div><div class="stat"><strong>$100,000</strong><span>每个账户固定初始本金</span></div><div class="stat"><strong>{start[:4]}–{end[:4]}</strong><span>{start} 至 {end}</span></div></div><p><a href="pnl_atlas.pdf">下载 20 页 PnL 曲线图册 PDF</a> · <a href="#methods">口径与证据边界</a> · <a href="#hypotheses">候选 filter 假设</a> · <a href="#files">下载数据表</a></p></header>
<main><section id="methods"><h2>如何阅读这份观测</h2><p class="formula">日单利 PnL = 100,000 × 当日实际持仓方向 × FX 简单日收益；累计 PnL = 日 PnL 逐日相加。</p><p>复利参照使用同一持仓序列，将每日策略收益连乘到变化的权益上。所有账户独立出资，跨币种并列不构成组合。收益均为现货毛收益，未扣交易成本、carry、融资或现金利息。</p><p>策略信号、成交和首个收益依次位于相邻的三个有效报价日；图形分析不会修改原信号或持仓。年度、半年度和滚动诊断只划分收益观察区间，不重启策略。</p><div class="note caution"><b>“有效 / 失效”是回看本样本得到的描述标签。</b> 盈亏窗口的终点、区间排名和回撤谷底当时尚不可知，不能直接充当交易条件。主样本已被用于研究；本报告及候选解释均不构成新的、未触碰的样本外证据。筛选假设需要可在信号日取得的信息、预先固定的规则和独立验证。</div><p class="muted">胜率分母为有持仓收益观测且真实平仓的交易，平手交易仍在分母内；没有赢家或输家时盈亏比未定义。最大单利回撤用相对初始本金的百分点表示，不是相对峰值权益的百分比。未恢复回撤与尚未平仓交易分别处理。</p></section>
<section id="contents"><h2>全部策略概览</h2><p>表中位数和区间汇总九个独立账户，不代表组合收益。“盈利币种数”只描述全期结果；判断阶段性失效应继续查看路径和分段表。</p>{_table(overview, labels={"positive_currencies":"全期盈利币种数", "median_simple_return_pct":"单利收益中位数 %", "min_simple_return_pct":"最低单利收益 %", "max_simple_return_pct":"最高单利收益 %", "median_drawdown_pp":"最大回撤中位数 pp"})}<nav aria-label="策略目录">{navigation}</nav></section>
<section id="hypotheses"><h2>从观测到候选 filter</h2><p>以下内容用于提出下一轮可被否证的研究问题。它们尚未改写 BASE 策略，也没有被表示为已通过验证的交易规则。</p>{_hypotheses(metadata)}</section>
{''.join(sections)}
<section id="files"><h2>完整数据与图册</h2><p><a href="pnl_atlas.pdf">pnl_atlas.pdf：20 页，每个策略一页、每页 9 个币种</a>；各图 PNG 位于 <code>figures/</code>。下列 CSV 保留可追溯的账户、区间与状态证据。</p>{_csv_links(output_dir, metadata)}<p>本页面、PNG、PDF 与数据表可在本地离线查看，无需服务器或联网。图形和表格由相同的重建账户数据生成。</p></section></main><footer>生成标识：{escape(str(generated))}。代码入口：src/fx_observation_report.py。本报告是历史研究观测，不是交易建议或新增样本外验证。</footer></body></html>'''
    html_path.write_text(body, encoding="utf-8")
    paths = [html_path, pdf_path] + [output_dir / path for path in figure_paths]
    return {
        "report": html_path.name, "atlas": pdf_path.name, "figures": figure_paths,
        "strategy_figures": len(figure_paths), "pdf_pages": len(STRATEGIES),
        "accounts": len(accounts), "daily_observations": len(daily),
        "sha256": {path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
