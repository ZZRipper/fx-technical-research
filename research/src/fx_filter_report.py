"""Render frozen IS-selected filter candidates without ranking on OOS results.

Only the caller's frozen candidate order controls presentation. Daily plots reset
displayed PnL at the IS/OOS split; they do not reset the underlying positions.
The historical OOS interval is an isolated evaluation for this round, not a
claim that the market observations have never previously been examined.
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
import numpy as np
import pandas as pd


_BASE = "#657889"
_FILTER = "#087e8b"
_CSS = """
*{box-sizing:border-box}body{margin:0;color:#173149;background:#f3f6f8;font:15px/1.75 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}header,main,footer{max-width:1320px;margin:auto;padding:28px 32px}header{padding-top:44px}h1{font-size:32px;line-height:1.35;margin:8px 0 16px}h2{font-size:24px;margin:0 0 14px}h3{font-size:18px}a{color:#075c83;text-underline-offset:3px}p{margin:10px 0}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.12em;color:#087e8b}.lead{font-size:17px;color:#526776;max-width:1000px}section{padding:26px;background:white;border:1px solid #dce5eb;border-radius:10px;margin-bottom:24px;scroll-margin-top:20px}.note{padding:14px 18px;background:#eef7fa;border-left:4px solid #087e8b}.caution{background:#fff5e6;border-color:#b48135}.muted,figcaption{font-size:13px;color:#657889}.table-wrap{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:9px 10px;text-align:right;border-bottom:1px solid #e1e9ee;font-variant-numeric:tabular-nums}th{background:#edf3f7}th:first-child,td:first-child{text-align:left}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.stage{padding:14px;background:#edf3f7;border-radius:7px}.stage strong{display:block;font-size:15px}.stage span{font-size:12px}.hash{font-family:ui-monospace,Consolas,monospace;word-break:break-all;font-size:12px}.stats{display:flex;flex-wrap:wrap;gap:20px;margin:20px 0}.stat{padding:13px 18px;background:white;border:1px solid #dce5eb;border-radius:7px}.stat strong{font-size:24px;display:block}.stat span{font-size:12px;color:#657889}figure{margin:20px 0}figure img{display:block;width:100%;height:auto}figcaption{padding-top:7px}details{border-top:1px solid #dce5eb;padding-top:12px;margin-top:15px}summary{cursor:pointer;font-weight:650}nav{display:flex;flex-wrap:wrap;gap:8px 14px}nav a{font-size:12px;padding:3px 7px;background:#edf3f7;border-radius:4px}.back{float:right;font-size:12px}.negative{color:#aa3943}.positive{color:#22745d}code{background:#edf3f7;padding:2px 4px}footer{font-size:12px;color:#657889;padding-bottom:40px}
@media(max-width:700px){header,main,footer{padding:20px 14px}h1{font-size:27px}section{padding:18px 14px}.pipeline{grid-template-columns:repeat(2,1fr)}h2{font-size:21px}}
@media print{body{background:white}header,main,footer{max-width:none;padding:10px}section{border:0;padding:10px 0}.candidate{break-before:page}figure{break-inside:avoid}nav,.back{display:none}.table-wrap{overflow:visible}table{white-space:normal}a{color:inherit;text-decoration:none}}
"""

_LABELS = {
    "candidate_id": "冻结候选", "freeze_rank": "IS 冻结顺序", "pair": "币种",
    "strategy": "技术规则", "filter_name": "过滤条件", "split": "区间", "variant": "账户",
    "gross_simple_return_pct": "单利累计 %", "net_simple_return_pct": "净单利累计 %",
    "gross_annualized_simple_pct": "简单年化 %", "net_annualized_simple_pct": "净单利年化 %",
    "gross_max_drawdown_pp": "最大回撤 pp", "net_max_drawdown_pp": "净最大回撤 pp",
    "win_rate": "交易胜率 %", "payoff_ratio": "交易盈亏比", "closed_n": "已平仓数",
    "net_win_rate": "净交易胜率 %", "net_payoff_ratio": "净交易盈亏比",
    "retention": "已平仓交易保留比例 %", "low_sample_flag": "不足20笔平仓",
    "oos_curve_pass": "预写OOS曲线标准", "quality_confirmation": "交易质量确认",
    "oos_quality_confirmed": "OOS交易质量确认", "carry_in_n": "跨界持仓数",
    "censored_entry_n": "期内开仓未平数", "filter_id": "过滤条件", "frozen": "已冻结",
    "pass": "IS 综合通过", "is_new": "新增过滤版本", "failed_criteria": "IS未通过项目",
    "is_gross_return_pct": "IS单利累计 %", "is_net_return_pct": "IS净单利累计 %",
    "gross_return_improvement_pp": "IS收益改善 pp", "win_rate_improvement_pp": "IS胜率改善 pp",
    "payoff_relative_improvement": "IS盈亏比相对改善 %", "is_closed_n": "IS已平仓数",
    "start_date": "起始日", "end_date": "结束日", "observations": "报价日数",
    "is_filter_simple_return_pct": "IS FILTER 累计 %",
    "oos_filter_simple_return_pct": "OOS FILTER 累计 %",
    "oos_base_simple_return_pct": "OOS BASE 累计 %",
    "fresh_base_return_pct": "现金起步 BASE 累计 %",
    "fresh_filter_return_pct": "现金起步 FILTER 累计 %",
    "fresh_base_drawdown_pp": "现金起步 BASE 回撤 pp",
    "fresh_filter_drawdown_pp": "现金起步 FILTER 回撤 pp",
    "continuous_filter_return_pct": "连续持仓 FILTER 累计 %",
    "action": "处理方式", "original_episode_count": "涉及原交易段数",
    "baseline_return_pp": "BASE 贡献 pp", "filtered_return_pp": "FILTER 贡献 pp",
    "removed_losing_contribution_pp": "避开的亏损 pp",
    "foregone_profitable_contribution_pp": "放弃的盈利 pp", "net_delta_pp": "净改变 pp",
}


def _scalar(value, column=""):
    if value is None or value is pd.NaT or (np.isscalar(value) and pd.isna(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if column in {"win_rate", "net_win_rate", "retention", "payoff_relative_improvement"}:
            value *= 100
        return f"{value:,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _table(frame, columns=None):
    if frame.empty:
        return '<p class="muted">没有符合本轮冻结协议的记录。</p>'
    columns = [column for column in (columns or list(frame.columns)) if column in frame]
    header = "".join(f"<th>{escape(_LABELS.get(column, column))}</th>" for column in columns)
    rows = []
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, values):
            css = ""
            if isinstance(value, (int, float, np.number)) and ("return" in column or "annualized" in column):
                css = ' class="positive"' if value > 0 else (' class="negative"' if value < 0 else "")
            cells.append(f"<td{css}>{escape(_scalar(value, column))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _required(frame, columns, name):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _json_section(title, record):
    rows = "".join(f"<tr><th>{escape(str(key))}</th><td class=\"hash\">{escape(_scalar(value))}</td></tr>" for key, value in record.items())
    return f"<details><summary>{escape(title)}</summary><div class=\"table-wrap\"><table>{rows}</table></div></details>"


def _criteria_explainer(protocol):
    screen, oos = protocol.get("screen", {}), protocol.get("oos_screen", {})
    entries = []
    descriptions = [
        ("min_gross_annualized_simple_pct", "IS 简单年化至少 {value}%", 1),
        ("min_return_improvement_pp", "IS 累计收益比 BASE 至少改善 {value} pp", 1),
        ("min_win_rate_improvement_pp", "IS 交易胜率至少改善 {value} pp", 1),
        ("min_payoff_relative_improvement", "IS 盈亏比至少相对提高 {value}%", 100),
        ("max_drawdown_deterioration_pp", "IS 最大回撤恶化不超过 {value} pp", 1),
        ("min_closed_trades", "IS 至少 {value} 笔合格已平仓交易", 1),
        ("min_closed_retention", "IS 合格已平仓交易保留比例至少 {value}%", 100),
        ("min_positive_is_blocks", "IS 三个预设分段中至少 {value} 段正收益", 1),
        ("min_improved_is_blocks", "IS 三个预设分段中至少 {value} 段优于 BASE", 1),
    ]
    for key, description, scale in descriptions:
        if key in screen:
            entries.append(description.format(value=f"{float(screen[key]) * scale:g}"))
    if screen.get("last_is_block_positive_and_improved"):
        entries.append("最后一个 IS 分段必须同时正收益且优于 BASE")
    if screen.get("return_to_drawdown_must_improve"):
        entries.append("IS 收益 / 回撤比必须优于 BASE：累计单利收益除以最大回撤的绝对值；零回撤边界按预写筛选实现处理")
    oos_entries = []
    for key, description in [
        ("min_net_annualized_simple_pct", "OOS 简单年化至少 {value}%"),
        ("min_net_return_improvement_pp", "OOS 累计收益比 BASE 至少改善 {value} pp"),
        ("worst_net_drawdown_pp", "OOS 最大回撤不低于 {value} pp"),
        ("min_positive_anniversary_years", "三个 OOS 周年区间中至少 {value} 段正收益"),
        ("min_closed_trades", "OOS 至少 {value} 笔合格已平仓交易"),
    ]:
        if key in oos:
            oos_entries.append(description.format(value=f"{float(oos[key]):g}"))
    result = ""
    if entries:
        result += "<h3>IS 综合门槛（需同时满足）</h3><ul>" + "".join(f"<li>{escape(entry)}</li>" for entry in entries) + "</ul>"
    if oos_entries:
        result += "<h3>冻结后评价 OOS 的预写曲线标准</h3><ul>" + "".join(f"<li>{escape(entry)}</li>" for entry in oos_entries) + "</ul>"
    if "quality_confirmation" in oos:
        result += '<p class="muted">交易质量另行确认：OOS 的胜率和盈亏比均不劣于 BASE，且至少有 20 笔合格已平仓交易。该确认不授权根据 OOS 修改阈值或重选候选。</p>'
    return result


def _ordered_metrics(metrics, candidate_id):
    selected = metrics.loc[metrics.candidate_id.eq(candidate_id)].copy()
    selected["_split_order"] = selected.split.map({"IS": 0, "OOS": 1})
    selected["_variant_order"] = selected.variant.map({"BASE": 0, "FILTER": 1})
    return selected.sort_values(["_split_order", "_variant_order"], kind="stable").drop(columns=["_split_order", "_variant_order"])


def _outcome_text(frozen_count, validation):
    if not validation.get("oos_evaluated", False):
        return ""
    curve_count = validation.get("curve_passing_candidates")
    quality_count = validation.get("quality_confirmed_candidates")
    both_count = validation.get("both_curve_and_quality_candidates")
    if curve_count is None:
        return f"本轮 {frozen_count} 个冻结候选已评估 OOS；验证记录尚未提供曲线标准通过数，不能据此宣称目标达成。"
    message = f"本轮 {frozen_count} 个冻结候选中，{int(curve_count)} 个满足预写 OOS 曲线标准。"
    if quality_count is not None:
        message += f"另有 {int(quality_count)} 个通过交易质量确认"
        message += f"，同时满足两者的为 {int(both_count)} 个。" if both_count is not None else "。"
    if int(curve_count) == 0:
        message += "本轮目标尚未达成；交易质量单项确认不能替代曲线标准。"
    elif both_count is not None and int(both_count) == 0:
        message += "尚无候选同时满足曲线与交易质量要求，综合目标尚未达成。"
    return message


def _candidate_summary(frozen, metrics, oos_evaluated):
    if frozen.empty:
        return ""
    rows = []
    for candidate in frozen.to_dict("records"):
        selected = metrics.loc[metrics.candidate_id.eq(candidate["candidate_id"])].set_index(["split", "variant"])
        row = {key: candidate[key] for key in ("freeze_rank", "pair", "strategy", "filter_name")}
        row["is_filter_simple_return_pct"] = selected.loc[("IS", "FILTER"), "gross_simple_return_pct"]
        if oos_evaluated:
            filtered = selected.loc[("OOS", "FILTER")]
            row.update(
                oos_filter_simple_return_pct=filtered.gross_simple_return_pct,
                oos_base_simple_return_pct=selected.loc[("OOS", "BASE"), "gross_simple_return_pct"],
                oos_curve_pass=filtered.get("oos_curve_pass"),
                oos_quality_confirmed=filtered.get("oos_quality_confirmed"),
            )
        rows.append(row)
    heading = "全部冻结候选的 IS / OOS 结果" if oos_evaluated else "全部冻结候选的 IS 结果"
    note = "以下保持原 IS 冻结顺序，包含 OOS 未通过者；不按 OOS 盈亏排序。" if oos_evaluated else "OOS 尚未运行，本表只列冻结候选的 IS 结果。"
    return f"<h3>{heading}</h3><p class=\"muted\">{note}</p>" + _table(pd.DataFrame(rows))


def _fresh_start_section(output_dir, frozen, metrics):
    path = output_dir / "oos_fresh_start_metrics.csv"
    if not path.is_file() or frozen.empty:
        return ""
    frame = pd.read_csv(path)
    _required(frame, ["candidate_id", "variant", "gross_simple_return_pct", "gross_max_drawdown_pp"], "fresh-start diagnostics")
    if frame.duplicated(["candidate_id", "variant"]).any():
        raise ValueError("Fresh-start diagnostics contain duplicate candidate/variant rows.")
    frame = frame.set_index(["candidate_id", "variant"])
    rows = []
    for candidate in frozen.to_dict("records"):
        key = candidate["candidate_id"]
        base, filtered = frame.loc[(key, "BASE")], frame.loc[(key, "FILTER")]
        continuous = metrics.loc[metrics.candidate_id.eq(key) & metrics.split.eq("OOS") & metrics.variant.eq("FILTER")].iloc[0]
        rows.append({
            "freeze_rank": int(candidate["freeze_rank"]), "pair": candidate["pair"], "strategy": candidate["strategy"],
            "continuous_filter_return_pct": continuous.gross_simple_return_pct,
            "fresh_base_return_pct": base.gross_simple_return_pct,
            "fresh_filter_return_pct": filtered.gross_simple_return_pct,
            "fresh_base_drawdown_pp": base.gross_max_drawdown_pp,
            "fresh_filter_drawdown_pp": filtered.gross_max_drawdown_pp,
        })
    return '<section id="fresh-start"><h2>边界敏感性：OOS 现金起步</h2><p>主结果保持跨 IS/OOS 持仓连续。本表另从 OOS 起点空仓开始，仅允许此后新开的原始 BASE 交易段进入，禁止在跨界旧交易段中途入场。边界处理可能显著改变累计收益；现金起步仍有自身回撤，不能以其中较好的数值替换连续持仓结果或修改预写门槛。</p><p class="muted">全部冻结候选仍按 IS freeze rank 排列；这是已预写的诊断，不用于 OOS 重选。完整记录：<a href="oos_fresh_start_metrics.csv">oos_fresh_start_metrics.csv</a>。</p>' + _table(pd.DataFrame(rows)) + "</section>"


def _failure_section(output_dir, frozen, metrics):
    summary_path = output_dir / "failure_attribution.csv"
    explanation_path = output_dir / "failure_diagnostics.json"
    if frozen.empty or not summary_path.is_file() or not explanation_path.is_file():
        return ""
    frame = pd.read_csv(summary_path)
    explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
    _required(frame, ["candidate_id", "split", "action", "baseline_return_pp", "filtered_return_pp", "net_delta_pp"], "failure attribution")
    if set(frame.candidate_id) != set(frozen.candidate_id) or frame.duplicated(["candidate_id", "split", "action"]).any():
        raise ValueError("Failure attribution must cover exactly the frozen candidates with unique action rows.")
    reference = metrics.set_index(["candidate_id", "split", "variant"])
    groups = frame.groupby(["candidate_id", "split"], sort=False)
    if set(groups.groups) != {(key, split) for key in frozen.candidate_id for split in ("IS", "OOS")}:
        raise ValueError("Failure attribution is missing frozen IS/OOS groups.")
    for (key, split), group in groups:
        base = float(reference.loc[(key, split, "BASE"), "gross_simple_return_pct"])
        filtered = float(reference.loc[(key, split, "FILTER"), "gross_simple_return_pct"])
        actual = group[["baseline_return_pp", "filtered_return_pp", "net_delta_pp"]].sum().to_numpy(float)
        if not np.allclose(actual, [base, filtered, filtered - base], atol=1e-8, rtol=0):
            raise ValueError("Saved failure attribution does not reconcile with the report's frozen metrics.")
    rank = frozen.set_index("candidate_id").freeze_rank
    frame["freeze_rank"] = frame.candidate_id.map(rank).astype(int)
    frame["_split_order"] = frame.split.map({"IS": 0, "OOS": 1})
    frame["_action_order"] = frame.action.map({"entry_rejected": 0, "early_exit": 1, "retained": 2})
    if frame._action_order.isna().any():
        raise ValueError("Unknown failure attribution action.")
    frame = frame.sort_values(["freeze_rank", "_split_order", "_action_order"], kind="stable")
    action_records = frame.set_index(["candidate_id", "split", "action"])
    frame["action"] = frame.action.map({"entry_rejected": "入场拒绝", "early_exit": "提前退出", "retained": "保留收益"})
    columns = ["freeze_rank", "pair", "strategy", "split", "action", "original_episode_count", "baseline_return_pp", "filtered_return_pp", "removed_losing_contribution_pp", "foregone_profitable_contribution_pp", "net_delta_pp"]
    cases = {case["candidate_id"]: case for case in explanation.get("cases", [])}
    if set(cases) != set(frozen.candidate_id):
        raise ValueError("Failure explanations must cover the complete frozen set.")
    blocks = []
    for candidate in frozen.to_dict("records"):
        case = cases[candidate["candidate_id"]]
        facts = "".join(f"<li>{escape(str(fact))}</li>" for fact in case.get("facts_zh", []))
        questions = "".join(f"<li>{escape(str(question))}</li>" for question in case.get("next_is_only_questions", []))
        mechanism_note = ""
        if candidate.get("filter_id") == "VOL_TIME_20":
            key = candidate["candidate_id"]
            entry = action_records.loc[(key, "IS", "entry_rejected")]
            early_is = action_records.loc[(key, "IS", "early_exit")]
            early_oos = action_records.loc[(key, "OOS", "early_exit")]
            closed = int(reference.loc[(key, "IS", "FILTER"), "closed_n"])
            mechanism_note = f'<p class="note caution">机制作用样本需要单独看：该账户 IS 保留 {closed} 笔合格已平仓交易；入场拒绝贡献 {entry.net_delta_pp:+.2f} pp，提前退出仅贡献 {early_is.net_delta_pp:+.2f} pp、影响 {int(early_is.original_episode_count)} 个原交易段。OOS 提前退出影响 {int(early_oos.original_episode_count)} 个原交易段，贡献 {early_oos.net_delta_pp:+.2f} pp。账户总交易数不能当作时间退出机制的作用样本数，整体 IS 改善不等于该机制已得到充分验证。</p>'
        blocks.append(f'<details><summary>IS顺序 {int(candidate["freeze_rank"])} · {escape(str(candidate["pair"]))} · {escape(str(candidate["strategy"]))}</summary><ul>{facts}</ul>{mechanism_note}<p><b>下一轮仅用 IS 可检验的问题，未修改本轮规则：</b></p><ul>{questions}</ul></details>')
    episode_link = ' · <a href="failure_episode_attribution.csv">逐交易段归因 CSV</a>' if (output_dir / "failure_episode_attribution.csv").is_file() else ""
    return '<section id="failure-attribution"><h2>失败归因：避开的亏损与放弃的盈利</h2><p>以下仅拆分已经保存的 IS/OOS 结果，保持冻结顺序，不重新运行或调整 filter。分类单位是原 BASE 交易段在某个区间、某类处理下被删去的收益净和：段内正负日先相加，再判定避开亏损或放弃盈利；不是挑出每日涨跌相加。提前退出当日的旧仓收益仍计入保留段，从下一收益日才计入提前退出删除段。</p><p class="muted">同一原交易可先保留、后提前退出，因此不同处理方式的交易段数量不能直接相加为总交易数。账户合格已平仓总数也不能替代每种机制真正改变的交易段样本数。这里是收益拆分，不证明经济因果关系；观察到 OOS 失效后提出的问题不能把同一 OOS 再变成未触碰验证集。</p>' + _table(frame, columns) + "".join(blocks) + '<p>下载：<a href="failure_attribution.csv">归因汇总 CSV</a>' + episode_link + ' · <a href="failure_diagnostics.json">定义、事实与核对 JSON</a>。</p>' + _json_section("归因核对记录", explanation.get("validation", {})) + "</section>"


def _plot_candidate(candidate, daily, metrics, destination, pdf, capital, oos_evaluated):
    splits = ["IS", "OOS"] if oos_evaluated else ["IS"]
    figure, axes = plt.subplots(1, len(splits), figsize=(16.5, 8.2), squeeze=False, sharey=True)
    name = f"{candidate['pair']} | {candidate['strategy']} | {candidate['filter_name']}"
    figure.suptitle(f"IS freeze rank {int(candidate['freeze_rank'])}: {name}", x=.065, y=.96, ha="left", fontsize=16, fontweight="bold", color="#173149")
    figure.text(.065, .912, "Fixed initial capital, zero costs. Each displayed split starts at zero; underlying positions continue.", fontsize=10, color="#536878")
    for axis, split in zip(axes.flat, splits):
        selected = daily.loc[daily.split.eq(split)]
        for variant, color in [("BASE", _BASE), ("FILTER", _FILTER)]:
            values = selected.loc[selected.variant.eq(variant)].sort_values("date")
            dates = pd.DatetimeIndex(pd.to_datetime(values.date))
            anchor = pd.Timestamp(metrics.loc[metrics.split.eq(split) & metrics.variant.eq(variant), "anchor_date"].iloc[0])
            dates = dates.insert(0, anchor)
            axis.plot(dates, np.r_[0., np.cumsum(values.gross_daily_return.to_numpy(dtype=float))] * capital / 1000, color=color, linewidth=1.35, label=variant)
        span = pd.to_datetime(selected.date)
        title = "IS: protocol screen" if split == "IS" else "OOS: frozen one-shot evaluation"
        axis.set_title(f"{title}\n{span.min():%Y-%m-%d} to {span.max():%Y-%m-%d}", loc="left", fontsize=11, pad=13)
        axis.axhline(0, color="#bac7d1", linewidth=.7)
        axis.grid(axis="y", color="#e5ebef", linewidth=.65)
        if split == "IS":
            axis.xaxis.set_major_locator(mdates.YearLocator(base=4))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(labelsize=9, colors="#546978")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#bdcad3")
    axes[0, 0].set_ylabel("Window cumulative PnL (USD thousands)", fontsize=10, color="#536878")
    figure.legend(handles=[Line2D([0], [0], color=_BASE, lw=1.8, label="BASE"), Line2D([0], [0], color=_FILTER, lw=1.8, label="FILTER")], loc="lower left", bbox_to_anchor=(.06, .055), ncol=2, frameon=False, fontsize=10)
    figure.text(.065, .035, "Displayed in frozen IS rank order, including OOS failures. No OOS ranking or candidate reselection.", fontsize=9, color="#657889")
    figure.text(.065, .016, "Historical isolation for this round; previously examined market history is not untouched OOS. Zero costs/carry/financing.", fontsize=9, color="#657889")
    figure.subplots_adjust(left=.065, right=.975, top=.83, bottom=.18, wspace=.15)
    try:
        figure.savefig(destination, dpi=145, facecolor="white")
        pdf.savefig(figure, facecolor="white")
    finally:
        plt.close(figure)


def _empty_pdf(pdf, validation):
    figure = plt.figure(figsize=(11.7, 8.3))
    figure.text(.09, .83, "IS screen: no frozen candidates", fontsize=24, fontweight="bold", color="#173149")
    message = "No new filter was eligible for the caller's frozen set after the prewritten IS screen. No OOS candidate result is constructed or shown. This page records the stopped gate; it is not a PnL curve."
    figure.text(.09, .69, textwrap.fill(message, 90), fontsize=13, linespacing=1.7, color="#536878")
    figure.text(.09, .42, "Protocol -> IS screen -> STOP\nNo freeze set -> no one-shot OOS evaluation", fontsize=16, linespacing=1.7, color="#087e8b")
    figure.text(.09, .14, "Historical isolation for this round does not imply previously untouched market history.", fontsize=10, color="#657889")
    try:
        pdf.savefig(figure, facecolor="white")
    finally:
        plt.close(figure)


def render_filter_report(output_dir, daily, metrics, selection, protocol, validation) -> dict:
    """Render the caller's frozen candidates in IS freeze order, never OOS order.

    Data contract is deliberately explicit: no fallback can infer an approved
    candidate from its performance. The caller supplies the full IS screen;
    ``frozen`` and ``freeze_rank`` identify the already-fixed presentation set.
    """
    _required(selection, ["candidate_id", "freeze_rank", "frozen", "pair", "strategy"], "selection")
    if "filter_name" not in selection and "filter_id" not in selection:
        raise ValueError("selection must provide filter_name or filter_id.")
    selection = selection.copy()
    if "filter_name" not in selection:
        selection["filter_name"] = selection.filter_id
    if selection.frozen.isna().any() or not selection.frozen.isin([True, False]).all():
        raise ValueError("selection.frozen must contain booleans.")
    selection["frozen"] = selection.frozen.astype(bool)
    if not selection.frozen.eq(selection.freeze_rank.notna()).all():
        raise ValueError("Only frozen IS candidates may carry a freeze_rank.")
    frozen = selection.loc[selection.frozen].copy()
    frozen["freeze_rank"] = pd.to_numeric(frozen.freeze_rank, errors="raise")
    if frozen.candidate_id.isna().any() or frozen.candidate_id.duplicated().any() or frozen.freeze_rank.duplicated().any() or frozen.freeze_rank.isna().any():
        raise ValueError("Frozen candidate IDs and freeze ranks must be nonmissing and unique.")
    if len(frozen) and ((frozen.freeze_rank <= 0).any() or (frozen.freeze_rank % 1 != 0).any()):
        raise ValueError("Freeze ranks must be positive integers.")
    if frozen.duplicated(["pair", "strategy"]).any():
        raise ValueError("The protocol allows at most one frozen filter per pair/strategy.")
    for flag in ("pass", "is_new"):
        if flag in frozen and not frozen[flag].eq(True).all():
            raise ValueError("Only new filters that passed the IS screen may be frozen.")
    frozen = frozen.sort_values("freeze_rank", kind="stable")
    oos_evaluated = validation.get("oos_evaluated", False)
    if not isinstance(oos_evaluated, bool):
        raise ValueError("validation.oos_evaluated must be a boolean.")
    if frozen.empty and oos_evaluated:
        raise ValueError("An empty freeze set cannot have an evaluated OOS candidate set.")
    if float(protocol.get("cost_bp", 0.)) != 0.:
        raise ValueError("This research round is explicitly zero-cost.")
    if len(frozen):
        _required(daily, ["candidate_id", "split", "variant", "date", "gross_daily_return", "net_daily_return"], "daily")
        _required(metrics, ["candidate_id", "split", "variant", "anchor_date", "gross_simple_return_pct", "gross_annualized_simple_pct", "gross_max_drawdown_pp", "win_rate", "payoff_ratio", "closed_n", "retention", "low_sample_flag"], "metrics")
        ids = set(frozen.candidate_id)
        metrics = metrics.loc[metrics.candidate_id.isin(ids) & metrics.split.isin(["IS", "OOS"])].copy()
        if set(daily.candidate_id) != ids or set(metrics.candidate_id) != ids:
            raise ValueError("Report paths must cover exactly the caller's frozen candidates.")
        splits = {"IS", "OOS"} if oos_evaluated else {"IS"}
        expected = {(candidate, split, variant) for candidate in ids for split in splits for variant in ("BASE", "FILTER")}
        for frame, label in [(daily, "daily"), (metrics, "metrics")]:
            actual = set(frame[["candidate_id", "split", "variant"]].drop_duplicates().itertuples(index=False, name=None))
            if actual != expected:
                raise ValueError(f"{label} must cover each frozen candidate's BASE/FILTER and evaluated splits exactly.")
        if metrics.duplicated(["candidate_id", "split", "variant"]).any():
            raise ValueError("Metrics contain duplicate candidate/split/variant rows.")
        if daily.duplicated(["candidate_id", "split", "variant", "date"]).any():
            raise ValueError("Daily paths contain duplicate observations.")
        if not np.isfinite(daily[["gross_daily_return", "net_daily_return"]].to_numpy(dtype=float)).all():
            raise ValueError("Daily paths contain nonfinite returns.")
        if not np.allclose(daily.gross_daily_return, daily.net_daily_return, rtol=0, atol=1e-14):
            raise ValueError("Zero-cost daily gross and net returns must agree.")
        bounds = {
            "IS": (pd.Timestamp(protocol.get("is_start", "2005-01-03")), pd.Timestamp(protocol.get("is_end", "2023-05-29"))),
            "OOS": (pd.Timestamp(protocol.get("oos_start", "2023-05-30")), pd.Timestamp(protocol.get("oos_end", "2026-05-29"))),
        }
        for (candidate_id, split), group in daily.groupby(["candidate_id", "split"], sort=False):
            base_dates = pd.DatetimeIndex(pd.to_datetime(group.loc[group.variant.eq("BASE"), "date"])).sort_values()
            filter_dates = pd.DatetimeIndex(pd.to_datetime(group.loc[group.variant.eq("FILTER"), "date"])).sort_values()
            if not base_dates.equals(filter_dates) or base_dates.hasnans:
                raise ValueError("BASE/FILTER calendars must agree within each split.")
            lo, hi = bounds[split]
            if base_dates.min() < lo or base_dates.max() > hi:
                raise ValueError("Daily plot dates fall outside the prewritten split.")
            anchors = pd.to_datetime(metrics.loc[metrics.candidate_id.eq(candidate_id) & metrics.split.eq(split), "anchor_date"])
            if anchors.isna().any() or anchors.nunique() != 1 or anchors.max() > base_dates.min():
                raise ValueError("BASE/FILTER must share a valid preceding-close anchor.")
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    capital = float(protocol.get("initial_capital_usd", 100000.))
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Initial capital must be positive and finite.")
    figure_paths, blocks = [], []
    pdf_path = output_dir / "pnl_comparison.pdf"
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": False}), PdfPages(pdf_path) as pdf:
        pdf.infodict().update(Title="Frozen IS-selected FX filters: historical isolated evaluation", Subject="All candidates in frozen IS order; no OOS selection")
        if frozen.empty:
            _empty_pdf(pdf, validation)
        for candidate in frozen.to_dict("records"):
            candidate_id = candidate["candidate_id"]
            rank = int(candidate["freeze_rank"])
            figure_path = figure_dir / f"frozen_{rank:04d}.png"
            candidate_metrics = _ordered_metrics(metrics, candidate_id)
            _plot_candidate(candidate, daily.loc[daily.candidate_id.eq(candidate_id)], candidate_metrics, figure_path, pdf, capital, oos_evaluated)
            relative = figure_path.relative_to(output_dir).as_posix()
            figure_paths.append(relative)
            columns = ["split", "variant", "gross_simple_return_pct", "gross_annualized_simple_pct", "gross_max_drawdown_pp", "win_rate", "payoff_ratio", "closed_n", "retention", "carry_in_n", "censored_entry_n", "low_sample_flag", "oos_curve_pass", "oos_quality_confirmed", "quality_confirmation"]
            blocks.append(f'<section class="candidate" id="candidate-{rank}"><a class="back" href="#frozen">返回冻结清单 ↑</a><p class="eyebrow">IS FREEZE RANK {rank}</p><h2>{escape(str(candidate["pair"]))} · {escape(str(candidate["strategy"]))}</h2><p><b>{escape(str(candidate["filter_name"]))}</b> · 候选编号 <code>{escape(str(candidate_id))}</code></p><figure><a href="{relative}"><img src="{relative}" loading="lazy" alt="冻结候选 {rank} 的 BASE 与 FILTER 零成本单利 PnL 比较"></a><figcaption>灰色为 BASE，蓝绿色为 FILTER；均为固定本金、零成本单利 PnL。各区间从真实前一收盘锚点归零显示，底层策略持仓沿时间连续。OOS 的亏损、劣化和低样本结果均保留。</figcaption></figure>' + _table(candidate_metrics, columns) + _json_section("冻结候选记录", candidate) + "</section>")
    navigation = "".join(f'<a href="#candidate-{int(row.freeze_rank)}">{int(row.freeze_rank)} · {escape(str(row.pair))} · {escape(str(row.strategy))}</a>' for row in frozen.itertuples(index=False))
    status = "已按冻结集合完成 OOS 一次评估" if oos_evaluated else ("没有可冻结的新增 IS 达标候选，流程在 IS 后停止" if frozen.empty else "候选已冻结；尚未运行 OOS")
    outcome = _outcome_text(len(frozen), validation)
    outcome_banner = '<div class="note caution"><b>' + escape(outcome) + "</b></div>" if outcome else ""
    if outcome:
        status += "；" + outcome.rstrip("。")
    candidate_summary = _candidate_summary(frozen, metrics, oos_evaluated)
    fresh_start_section = _fresh_start_section(output_dir, frozen, metrics) if oos_evaluated else ""
    failure_section = _failure_section(output_dir, frozen, metrics) if oos_evaluated else ""
    empty_note = '<div class="note caution"><b>本轮无可冻结的新增 IS 达标候选。</b> 因此没有构造 OOS 候选收益，也不从 OOS 反选策略。PDF 仅记录本轮停止状态。</div>' if frozen.empty else ""
    oos_start, oos_end = str(protocol.get("oos_start", "2023-05-30")), str(protocol.get("oos_end", "2026-05-29"))
    protocol_hash = validation.get("protocol_sha256", "未提供")
    freeze_hash = validation.get("freeze_sha256", "未生成 / 未提供")
    passing_count = int(selection["pass"].eq(True).sum()) if "pass" in selection else None
    screen_note = f"IS 共评估 {len(selection):,} 个规则 / 币种 / filter 组合；综合通过 {passing_count:,} 个。" if passing_count is not None else f"IS 筛选表包含 {len(selection):,} 个组合。"
    screen_table = _table(selection, ["candidate_id", "pair", "strategy", "filter_name", "is_new", "pass", "frozen", "freeze_rank", "is_gross_return_pct", "gross_return_improvement_pp", "win_rate_improvement_pp", "payoff_relative_improvement", "is_closed_n", "retention", "failed_criteria"])
    criteria = _criteria_explainer(protocol) + _json_section("IS 预写综合标准原记录", protocol.get("screen", {})) + _json_section("OOS 预写评价标准原记录（不用于重选）", protocol.get("oos_screen", {}))
    file_links = "".join(f'<li><a href="{escape(path.relative_to(output_dir).as_posix(), quote=True)}">{escape(path.name)}</a></li>' for path in sorted(output_dir.glob("*.csv*")))
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FX filter：本轮隔离历史测试</title><style>{_CSS}</style></head><body>
<header><p class="eyebrow">FROZEN IS SELECTION · HISTORICAL ISOLATION</p><h1>FX filter：本轮隔离历史测试</h1>{outcome_banner}<p class="lead">全部 20 条技术规则 × 9 个币种组成 BASE 研究范围。先写协议，用前段 IS 筛选；只有达到综合标准的新增候选才进入冻结清单，然后一次评估最后三年 OOS。这里的隔离针对本轮流程，这段市场历史并非从未看过。</p><div class="stats"><div class="stat"><strong>20 × 9</strong><span>BASE 研究范围</span></div><div class="stat"><strong>{len(frozen)}</strong><span>IS 达标并冻结候选</span></div><div class="stat"><strong>{escape(oos_start)}</strong><span>OOS 起始日</span></div><div class="stat"><strong>{escape(oos_end)}</strong><span>OOS 结束日</span></div></div><p><a href="pnl_comparison.pdf">下载全部冻结候选 PnL 图册</a> · <a href="#frozen">冻结清单</a> · <a href="#evidence">协议及验证证据</a></p></header>
<main><section id="flow"><h2>先筛选，再冻结，再评估</h2><div class="pipeline"><div class="stage"><strong>1. Protocol prewritten</strong><span>预先记录规则、IS 标准与 OOS 边界</span></div><div class="stage"><strong>2. IS screen</strong><span>仅用 IS 结果应用综合标准</span></div><div class="stage"><strong>3. Freeze SHA</strong><span>固定候选身份、参数和 IS 顺序</span></div><div class="stage"><strong>4. OOS one-shot</strong><span>固定集合进入 OOS，完整保留结果</span></div></div><p><b>当前状态：{escape(status)}。</b></p><p>预写协议 SHA-256：<span class="hash">{escape(str(protocol_hash))}</span></p><p>冻结集合 SHA-256：<span class="hash">{escape(str(freeze_hash))}</span></p><div class="note caution">本报告严格按调用方提供的 <code>freeze_rank</code> 排序。OOS 收益、胜率、回撤或最终成败均不参与候选选择、图册顺序或展示取舍。报告不把历史隔离声称为全新的未触碰样本外证据。</div>{empty_note}{criteria}</section>
<section id="accounting"><h2>比较口径：固定本金单利，零成本</h2><p>每个独立账户使用固定初始本金 ${capital:,.0f}。日 PnL 为固定本金乘实际持仓方向及现货日收益，逐日相加。本轮不计交易成本、carry、融资或现金利息。简单年化为区间单利收益除以实际日历年数，不是 CAGR。</p><p>IS 与 OOS 图形分别重置累计 PnL 为零；这只是显示归零，不能解释为在分界日清仓或重新训练。跨界持仓在 OOS 内发生的日收益仍进入 OOS 账户表现。最大回撤以初始本金百分点表示。交易胜率、盈亏比与已平仓数仅覆盖区间内开仓且在区间末前真实平仓、至少获得一个收益观测的交易；跨界持仓和期末未平仓交易另列。保留比例为 FILTER 合格已平仓数除以同区间 BASE 数量。</p><p class="muted">少于 20 笔合格已平仓交易标为低样本；OOS 曲线标准与额外交易质量确认分别列示。未定义统计显示“—”，不代表零。全部数值与冻结标准均来自计算模块，报告不重选阈值或补造 OOS。</p></section>
<section id="frozen"><h2>完整冻结清单：固定 IS 顺序</h2><p>{escape(screen_note)} 每个币种 / 技术规则最多冻结一个 IS 排名最高的新增达标 filter。其他通过版本和未通过版本保留在完整 IS 表，不凭 OOS 替换冻结者。</p>{_table(frozen, ["freeze_rank", "candidate_id", "pair", "strategy", "filter_name"])}{candidate_summary}<nav>{navigation}</nav><details><summary>完整 IS 筛选记录：含未通过与通过但未冻结版本</summary>{screen_table}</details></section>
{fresh_start_section}
{failure_section}
{''.join(blocks)}
<section id="evidence"><h2>协议、冻结与一次评估证据</h2><p>以下字段原样来自本轮运行器。哈希和状态用于定位实际记录；报告本身不另行认证缺失的冻结时刻或历史访问隔离。</p>{_json_section("预写协议", protocol)}{_json_section("本轮验证记录", validation)}</section>
<section id="files"><h2>完整输出</h2><p><a href="pnl_comparison.pdf">pnl_comparison.pdf</a> 包含所有冻结候选，顺序与上表一致；PNG 保存在 <code>figures/</code>。</p><ul>{file_links}</ul><p class="muted">页面与图册均可离线查看。全期观测报告和既有 dashboard 未被本报告修改。</p></section></main><footer>研究说明：这是预写规则下的历史隔离评估。失败与不确定性是结果的一部分，OOS 不用于当轮重新选择候选。</footer></body></html>'''
    html_path = output_dir / "filter_research_report.html"
    html_path.write_text(html, encoding="utf-8")
    paths = [html_path, pdf_path] + [output_dir / relative for relative in figure_paths]
    return {
        "report": html_path.name, "atlas": pdf_path.name, "figures": figure_paths,
        "frozen_candidates": len(frozen), "pdf_pages": max(1, len(frozen)),
        "oos_evaluated": oos_evaluated, "ordering": "IS freeze_rank only",
        "diagnostic_input_sha256": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("oos_fresh_start_metrics.csv", "failure_attribution.csv", "failure_episode_attribution.csv", "failure_diagnostics.json")
            if oos_evaluated and (output_dir / name).is_file()
        },
        "sha256": {path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
