"""Render the saved D/V/C state-filter campaign, without running decisions.

Only files inside --output are read. OOS result files are inaccessible to this
renderer until that directory contains a completed oos_status.json. This file
is a presentation tool and is deliberately separate from decision sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
DEFAULT = ROOT / "reproduced/state_filter_v2"
FAIL_LABELS = {
    "annualized": "单利年化至少 1%",
    "return_improvement": "收益改善",
    "return_drawdown": "收益/回撤比改善",
    "quality": "胜率与盈亏比",
    "drawdown": "回撤恶化不超过 2 pp",
    "sample": "闭合交易数与保留率",
    "outcome_sample": "盈利与亏损交易各至少 5 笔",
    "affected_sample": "实际受影响的闭合原始交易数量",
    "positive_and_annualized_pass": "正收益且单利年化至少 1%",
    "return_improvement_pass": "全 IS 收益改善至少 5 pp",
    "win_rate_pass": "胜率至少改善 2 pp",
    "payoff_pass": "盈亏比至少改善 10%",
    "drawdown_pass": "回撤恶化不超过 2 pp",
    "return_to_drawdown_pass": "收益/回撤比改善",
    "sample_pass": "至少 50 笔闭合交易且保留至少 50%",
    "block_positive_pass": "至少两段为正且末段为正",
    "block_improvement_pass": "至少两段改善且末段改善",
    "recent_absolute": "RECENT 单利年化至少 1%",
    "recent_delta": "RECENT 收益改善为正",
    "recent_ratio": "RECENT 收益/回撤比改善",
    "full_effect_sample": "全 IS 至少 20 个受影响闭合原始交易",
    "full_top3_robust": "全 IS 去除最有利 3 个原始交易后改善仍为正",
    "recent_top3_robust": "RECENT 去除最有利 3 个原始交易后改善仍为正",
}


def truth(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "1.0"}


def fmt(value, digits=2, suffix="") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}{suffix}"


def pct(value) -> str:
    try:
        return fmt(float(value) * 100, suffix="%")
    except (TypeError, ValueError):
        return "—"


def flag(value, present=True) -> str:
    return ("通过" if truth(value) else "未通过") if present else "待完成"


def failures(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item for item in str(value).split("|") if item]


def failure_text(value) -> str:
    return "；".join(FAIL_LABELS.get(item, item) for item in failures(value)) or "—"


def table(headers, rows, css="") -> str:
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>" for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}">暂无已完成记录</td></tr>'
    return f'<div class="table-wrap"><table class="{css}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class SavedInputs:
    """Read named local artifacts and record precisely which bytes were used."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.hashes: dict[str, str] = {}

    def payload(self, name):
        if Path(name).name != name:
            raise ValueError("Report inputs must be direct children of output.")
        path = self.directory / name
        if not path.exists():
            return None
        raw = path.read_bytes()
        self.hashes[name] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, name, default=None):
        raw = self.payload(name)
        return json.loads(raw.decode("utf-8-sig")) if raw is not None else default

    def csv(self, name):
        raw = self.payload(name)
        if raw is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.BytesIO(raw), compression="gzip" if name.endswith(".gz") else None)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


def metric_row(frame, candidate_id, split, variant):
    if frame.empty:
        return {}
    rows = frame.loc[frame.candidate_id.eq(candidate_id) & frame.split.eq(split) & frame.variant.eq(variant)]
    if len(rows) > 1:
        raise ValueError(f"Duplicate saved metric: {candidate_id}, {split}, {variant}")
    return rows.iloc[0].to_dict() if len(rows) else {}


def plot_candidate(output, identity, daily, metrics, protocol, complete, oos_daily, oos_metrics):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_report_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    candidate_id = identity["candidate_id"]
    rank = int(identity["preconfirmation_rank"])
    is_start, is_end = protocol.get("is_start", "2005-01-03"), protocol.get("is_end", "2023-05-29")
    confirmation = protocol.get("confirmation", ["CONFIRMATION", "2020-01-01", is_end])
    windows = [("IS" if complete else "FIT", is_start, is_end if complete else "2019-12-31", daily, metrics),
               ("CONFIRMATION", confirmation[1], confirmation[2], daily if complete else pd.DataFrame(), metrics)]
    if not oos_daily.empty and oos_daily.candidate_id.eq(candidate_id).any():
        windows.append(("OOS", protocol["oos_start"], protocol["oos_end"], oos_daily, oos_metrics))
    fig, axes = plt.subplots(1, len(windows), figsize=(7.2 * len(windows), 4.7), sharey=True)
    checked = 0
    for ax, (label, start, end, source, metric_source) in zip(np.atleast_1d(axes), windows):
        ax.set_title({"IS": "Full IS (D + V + C)", "FIT": "D + V only; full IS pending", "CONFIRMATION": "C: temporal confirmation", "OOS": "Reused historical OOS"}[label], fontsize=11)
        ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
        ax.axhline(0, color="#9ba6ac", linewidth=.7)
        ax.grid(alpha=.17)
        ax.set_ylabel("Cumulative simple return (%)")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4) if label in {"IS", "FIT"} else mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if label in {"IS", "FIT"} else "%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        if source.empty:
            ax.text(.5, .5, "PENDING: no confirmation results", ha="center", va="center", transform=ax.transAxes)
            continue
        rows = source.loc[source.candidate_id.eq(candidate_id)].copy()
        rows["date"] = pd.to_datetime(rows.date)
        rows = rows.loc[rows.date.between(start, end)]
        if rows.empty:
            ax.text(.5, .5, "No saved daily path", ha="center", va="center", transform=ax.transAxes)
            continue
        for variant, color in [("BASE", "#7b8790"), ("FILTER", "#007d81")]:
            part = rows.loc[rows.variant.eq(variant)].sort_values("date")
            if part.empty or part.date.duplicated().any():
                raise ValueError(f"Missing/duplicate daily curve: {candidate_id} {label} {variant}")
            saved = metric_row(metric_source, candidate_id, label, variant)
            curve = pd.to_numeric(part.gross_daily_return).cumsum().to_numpy() * 100
            if saved:
                if not np.isclose(curve[-1], saved["gross_simple_return_pct"], atol=1e-8, rtol=0):
                    raise ValueError(f"Saved daily/metric mismatch: {candidate_id} {label} {variant}")
                checked += 1
            anchor = pd.Timestamp(saved.get("anchor_date", start))
            xs = pd.DatetimeIndex([anchor]).append(pd.DatetimeIndex(part.date))
            ax.plot(xs, np.r_[0., curve], color=color, linewidth=1.6, label=f"{variant} {curve[-1]:+.2f}%")
        if label == "IS":
            for boundary, name in [("2015-01-01", "V"), (confirmation[1], "C")]:
                ax.axvline(pd.Timestamp(boundary), color="#9ba6ac", linestyle=":", linewidth=.9)
                ax.text(pd.Timestamp(boundary), .98, f" {name}", transform=ax.get_xaxis_transform(), va="top", fontsize=9)
        ax.legend(loc="best", fontsize=9, framealpha=.9)
    fig.suptitle(f"Pre-C rank {rank} | {identity['pair']} | {identity['strategy']}\n{identity.get('filter_name', identity['filter_id'])}", fontsize=12)
    fig.text(.5, .005, "Fixed initial capital; cost = 0. Each panel resets displayed PnL only; actual positions remain continuous.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .90))
    relative = f"state_figures/locked_{rank:04d}.png"
    (output / "state_figures").mkdir(exist_ok=True)
    fig.savefig(output / relative, dpi=150, facecolor="white")
    plt.close(fig)
    return relative, checked


def build_report(output_dir: Path) -> dict:
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError(f"Campaign directory does not exist: {output}")
    inputs = SavedInputs(output)
    protocol = inputs.json("protocol.json", {})
    if protocol.get("version", "state_filter_v2") != "state_filter_v2":
        raise ValueError("Expected a state_filter_v2 protocol.")
    if float(protocol.get("cost_bp", 0)) != 0:
        raise ValueError("This report is specified for the zero-cost campaign.")
    fit_status = inputs.json("fit_status.json", {})
    fit = inputs.csv("fit_selection.csv")
    fit_metrics = inputs.csv("fit_metrics.csv")
    lock = inputs.json("validation_locked.json", {})
    development = inputs.json("development_status.json", {})
    selection = inputs.csv("confirmation_selection.csv")
    full_metrics = inputs.csv("confirmation_full_metrics.csv")
    recent_metrics = inputs.csv("confirmation_recent_metrics.csv")
    evidence = inputs.json("confirmation_effect_evidence.json", [])
    complete = development.get("status") == "passed" and truth(development.get("is_screen_completed"))
    daily = inputs.csv("locked_is_daily.csv.gz" if complete else "fit_locked_daily.csv.gz")
    metrics = pd.concat([full_metrics, recent_metrics], ignore_index=True) if complete else fit_metrics
    locked = sorted(lock.get("candidates", []), key=lambda r: (float(r["preconfirmation_rank"]), r["candidate_id"]))
    locked_ids = {row["candidate_id"] for row in locked}
    selected = {row["candidate_id"]: row for row in selection.to_dict("records")}
    if complete and set(selected) != locked_ids:
        raise ValueError("Completed confirmation must retain every pre-C locked candidate.")
    survivors = {candidate_id for candidate_id, row in selected.items() if complete and truth(row.get("frozen"))}
    if complete and len(survivors) != int(development.get("frozen_candidates", len(survivors))):
        raise ValueError("Final survivor count differs from development_status.json.")
    oos_status, oos_daily, oos_metrics, assessment = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Do not inspect OOS result files before the explicit current-campaign gate.
    if (output / "oos_status.json").exists():
        oos_status = inputs.json("oos_status.json", {})
        if oos_status.get("status") == "passed" and truth(oos_status.get("oos_evaluated")):
            if not complete:
                raise ValueError("OOS result exists without completed IS confirmation.")
            oos_daily = inputs.csv("oos_daily.csv.gz")
            oos_metrics = inputs.csv("oos_metrics.csv")
            assessment = inputs.csv("candidate_assessment.csv")
            for frame in (oos_daily, oos_metrics, assessment):
                if not frame.empty and not set(frame.candidate_id).issubset(survivors):
                    raise ValueError("OOS contains a candidate that did not survive final IS.")
    for frame in (daily, oos_daily):
        if not frame.empty and not np.allclose(frame.gross_daily_return, frame.net_daily_return, atol=1e-12, rtol=0):
            raise ValueError("Gross and net paths differ despite the zero-cost protocol.")
    counts = {
        "proposals": int(fit_status.get("proposals", len(fit))),
        "discovery_pass": int(fit.discovery_pass.map(truth).sum()) if "discovery_pass" in fit else 0,
        "validation_pass": int(fit.validation_pass.map(truth).sum()) if "validation_pass" in fit else 0,
        "joint_pass": int(fit.fit_pass.map(truth).sum()) if "fit_pass" in fit else 0,
        "preconfirmation_locked": len(locked),
        "original_is_pass": sum(truth(row.get("original_is_pass")) for row in selected.values()) if complete else None,
        "confirmation_pass": sum(truth(row.get("confirmation_pass")) for row in selected.values()) if complete else None,
        "robustness_pass": sum(truth(row.get("robustness_pass")) for row in selected.values()) if complete else None,
        "final_is_survivors": len(survivors) if complete else None,
    }
    oos_complete = oos_status.get("status") == "passed" and truth(oos_status.get("oos_evaluated"))
    oos_label = "已完成历史 OOS 评估" if oos_complete else ("阶段未完成" if oos_status else "未打开")
    if complete and locked:
        outcome = f"IS 时间确认（C）已完成：{len(locked)} 个预确认锁定候选中，{len(survivors)} 个通过最终 IS 综合门槛。"
        if not survivors:
            outcome += "本轮没有可进入 OOS 的候选，研究目标尚未达成。"
    elif complete:
        outcome = "D/V 筛选与 IS 开发流程已完成，未产生预确认锁定候选，因此没有 C 候选曲线或可进入 OOS 的候选。"
    else:
        outcome = f"IS 时间确认（C）待完成；目前预确认锁定 {len(locked)} 个候选，最终 IS 结论待完成。"
    outcome += f" OOS：{oos_label}。"
    body = [f'<section class="outcome"><h2>本轮状态</h2><p>{escape(outcome)}</p></section>',
            '<p class="muted">本报告是开发隔离下的历史重复验证。C 是 IS 内的时间确认段；三年历史 OOS 此前已被观察过，即使后续评估也不是从未看过的新样本或新的独立证据。本报告不运行策略、不改阈值、不根据后段结果补选或重排。</p>']
    stages = [
        ("事先协议", "已保存" if protocol else "待完成"),
        ("D/V 筛选", "已完成" if fit_status.get("status") == "passed" else "待完成"),
        ("C 前锁定", f"{len(locked)} 个；保存哈希" if lock else "待完成"),
        ("IS 时间确认 C", "已完成" if complete and locked else ("无锁定候选，无需计算" if complete else "待完成")),
        ("最终 IS 冻结", f"{len(survivors)} 个" if complete else "待完成"),
        ("历史 OOS", oos_label),
    ]
    body += ["<h2>顺序与覆盖范围</h2>", table(["阶段", "当前状态"], stages)]
    scope = protocol.get("scope", {})
    body.append(f'<p>基础范围：{len(scope.get("strategies", [])) or 20} 条规则 × {len(scope.get("pairs", [])) or 9} 个币种，共 {scope.get("baseline_accounts", 180)} 个 BASE 账户。D 为 2005-01-03—2014-12-31，V 为 2015-01-01—2019-12-31，C 为 2020-01-01—2023-05-29。RECENT 为 V + C（2015-01-01—2023-05-29），全 IS 为 2005-01-03—2023-05-29。预定历史 OOS 为 2023-05-30—2026-05-29。</p>')
    body.append(table(["候选总数", "D 通过", "V 单独通过", "D 与 V 共同通过", "C 前锁定", "原始全 IS 通过", "C 门槛通过", "新增稳健性通过", "最终 IS 通过"], [[counts[key] if counts[key] is not None else "待完成" for key in counts]]))
    body.append('<p class="muted">D 与 V 通过数分别在全部提案中计数，并非两个相减的漏斗层级。共同通过后，每个币种/策略最多锁定一个，按 V 单利年化、截至 2019 年累计改善、候选 ID 排序。以下候选始终沿用 C 前排名；C 失败者完整保留且不启用备选。</p>')
    body.append("<h2>口径与预写门槛</h2>")
    body.append(f'<p>成本为 0；固定初始本金 ${float(protocol.get("initial_capital_usd", 100000)):,.0f}，每日收益按初始本金相加，图中累计收益 1% 对应 $1,000。单利年化为窗口累计收益除以实际日历年数。BASE/FILTER 都按 signal t → t+1 close 执行 → t+2 首次收益。跨 D/V/C/OOS 边界持续持仓，各图面板只将展示的累计 PnL 归零。最大回撤以负 pp 表示。胜率与盈亏比只统计期内开仓、期内闭合的合格交易，跨界与未闭合交易的每日收益仍进入曲线。</p>')
    screen = protocol.get("screen", {})
    gate_rows = [
        ("候选提出（仅 D）", "一个预定特征状态；UNKNOWN 不拒绝。2005–09、2010–14 两段各至少 10 笔闭合交易且状态收益和为负，合并 D 去掉最差一笔仍为负。"),
        ("D", "FILTER 单利年化 ≥1%；累计改善 ≥5 pp；胜率改善 ≥2 pp；盈亏比改善 ≥10%；收益/回撤比改善；回撤恶化 ≤2 pp；闭合 ≥30、保留率 ≥50%；受影响闭合原始交易 ≥20。"),
        ("V 与 C（分别考核）", "FILTER 单利年化 ≥1%；累计改善 >0；收益/回撤比改善；胜率和盈亏比不差于 BASE；回撤恶化 ≤2 pp；闭合 ≥20、保留率 ≥50%；盈利与亏损各 ≥5；受影响闭合原始交易 ≥5。"),
        ("原始全 IS 门槛", f"单利年化 ≥{screen.get('min_gross_annualized_simple_pct', 1):g}%；累计改善 ≥{screen.get('min_return_improvement_pp', 5):g} pp；胜率改善 ≥{screen.get('min_win_rate_improvement_pp', 2):g} pp；盈亏比改善 ≥{screen.get('min_payoff_relative_improvement', .1)*100:g}%；收益/回撤比改善；回撤恶化 ≤{screen.get('max_drawdown_deterioration_pp', 2):g} pp；闭合 ≥{screen.get('min_closed_trades', 50)}、保留率 ≥{screen.get('min_closed_retention', .5)*100:g}%；三段至少两段为正、至少两段改善，且末段同时为正并改善。"),
        ("新增稳健性", "RECENT 单利年化 ≥1%、改善 >0、收益/回撤比改善；全 IS 与 RECENT 各自去掉最有利的 3 个原始交易对改善的贡献后，改善仍 >0；全 IS 至少 20 个受影响闭合原始交易。"),
        ("最终准入", "原始全 IS 门槛、C 门槛与新增稳健性必须全部通过；仅最终通过者才可冻结后一次评估历史 OOS。"),
    ]
    body.append(table(["阶段", "预写条件"], gate_rows))
    body.append('<p class="muted">收益/回撤比改善沿用引擎的交叉乘积比较，避免用零回撤作分母。稳健性删除的是对 BASE→FILTER 改善贡献最大的原始交易段，不是逐日挑选收益。原始全 IS 门槛与新增门槛分开展示，不能把新增失败表述为原始门槛也失败。</p>')
    body.append("<h2>全部预确认锁定候选</h2>")
    overview = []
    for row in locked:
        cid = row["candidate_id"]
        result = selected.get(cid, {})
        full_b, full_f = [metric_row(full_metrics, cid, "IS", variant) for variant in ("BASE", "FILTER")]
        c_b, c_f = [metric_row(recent_metrics, cid, "CONFIRMATION", variant) for variant in ("BASE", "FILTER")]
        overview.append([int(row["preconfirmation_rank"]), row["pair"], row["strategy"], row["filter_id"],
                         fmt(full_b.get("gross_simple_return_pct"), suffix="%"), fmt(full_f.get("gross_simple_return_pct"), suffix="%"),
                         fmt(c_b.get("gross_simple_return_pct"), suffix="%"), fmt(c_f.get("gross_simple_return_pct"), suffix="%"),
                         flag(result.get("original_is_pass"), complete), flag(result.get("confirmation_pass"), complete),
                         flag(result.get("robustness_pass"), complete), flag(result.get("pass"), complete)])
    body.append(table(["C 前排名", "币种", "策略", "过滤状态", "全 IS BASE", "全 IS FILTER", "C BASE", "C FILTER", "原始全 IS", "C 门槛", "新增稳健性", "最终 IS"], overview))
    original_pass_ids = [row["candidate_id"] for row in locked if truth(selected.get(row["candidate_id"], {}).get("original_is_pass"))]
    if complete and original_pass_ids:
        body.append('<p><strong>原始全 IS 门槛通过者：</strong>' + escape("；".join(original_pass_ids)) + "。其 C 门槛与新增稳健性结果独立列示，不抹去原始门槛已达标的事实。</p>")
    reason_rows = []
    for row in locked:
        result = selected.get(row["candidate_id"], {})
        reason_rows.append([int(row["preconfirmation_rank"]), row["candidate_id"],
                            failure_text(result.get("failed_criteria")) if complete else "待完成",
                            failure_text(result.get("confirmation_failed")) if complete else "待完成",
                            failure_text(result.get("robustness_failed")) if complete else "待完成"])
    body.append(table(["C 前排名", "候选", "原始全 IS 失败条件", "C 失败条件", "新增稳健性失败条件"], reason_rows))
    if not locked:
        body.append("<p>没有预确认锁定候选；下方仍保留全部提案和 D/V 失败分布，不制造 C 或 OOS 曲线。</p>")
    evidence_lookup = {row["candidate_id"]: row for row in evidence}
    figures, endpoint_checks = [], 0
    for row in locked:
        cid, rank = row["candidate_id"], int(row["preconfirmation_rank"])
        image, checked = plot_candidate(output, row, daily, metrics, protocol, complete, oos_daily, oos_metrics)
        figures.append(image)
        endpoint_checks += checked
        body.append(f'<section class="candidate"><h3>#{rank} {escape(row["pair"])} · {escape(row["strategy"])}</h3><p><code>{escape(row["filter_id"])}</code></p><a href="{image}"><img src="{image}" alt="Pre-C rank {rank}: BASE and FILTER cumulative simple returns" loading="lazy"></a>')
        metric_rows = []
        display_splits = ["IS", "CONFIRMATION", "RECENT"] if complete else ["DISCOVERY", "VALIDATION", "FIT"]
        for split in display_splits:
            for variant in ("BASE", "FILTER"):
                saved = metric_row(metrics, cid, split, variant)
                if not saved:
                    continue
                metric_rows.append([split, variant, fmt(saved.get("gross_simple_return_pct"), suffix="%"),
                                    fmt(saved.get("gross_annualized_simple_pct"), suffix="%"), fmt(saved.get("gross_max_drawdown_pp")),
                                    fmt(saved.get("closed_n"), 0), fmt(saved.get("win_n"), 0), fmt(saved.get("loss_n"), 0),
                                    pct(saved.get("win_rate")), fmt(saved.get("payoff_ratio")), pct(saved.get("retention"))])
        if oos_complete:
            for variant in ("BASE", "FILTER"):
                saved = metric_row(oos_metrics, cid, "OOS", variant)
                if saved:
                    metric_rows.append(["OOS", variant, fmt(saved.get("gross_simple_return_pct"), suffix="%"), fmt(saved.get("gross_annualized_simple_pct"), suffix="%"), fmt(saved.get("gross_max_drawdown_pp")), fmt(saved.get("closed_n"), 0), fmt(saved.get("win_n"), 0), fmt(saved.get("loss_n"), 0), pct(saved.get("win_rate")), fmt(saved.get("payoff_ratio")), pct(saved.get("retention"))])
        body.append(table(["区间", "版本", "单利累计", "单利年化", "最大回撤 pp", "闭合笔数", "盈利笔数", "亏损笔数", "胜率", "盈亏比", "保留率"], metric_rows))
        effect_rows = []
        for split in ("IS", "CONFIRMATION", "RECENT"):
            saved = evidence_lookup.get(cid, {}).get(split, {})
            if saved:
                effect_rows.append([split, fmt(saved.get("delta_pp")), fmt(saved.get("changed_closed_n"), 0), fmt(saved.get("changed_episode_n"), 0), fmt(saved.get("delta_without_best3_pp"))])
        if effect_rows:
            body.append(table(["区间", "FILTER−BASE pp", "受影响闭合原始交易", "全部受影响原始交易", "去掉最有利 3 个后改善 pp"], effect_rows))
        body.append("</section>")
    body.append('<p class="muted">闭合交易总数不等于过滤机制的作用样本数。上表另列实际受影响的原始交易；“全部受影响”包含跨界及期末尚未闭合的贡献，不能代替期内闭合样本门槛。RECENT 包含 C，属于明确写入协议的 IS 稳健性检查，不是另一份独立验证。</p>')
    if oos_complete:
        oos_rows = []
        assessment_lookup = {row["candidate_id"]: row for row in assessment.to_dict("records")}
        for row in locked:
            if row["candidate_id"] in survivors:
                saved = assessment_lookup.get(row["candidate_id"], {})
                oos_rows.append([int(row["preconfirmation_rank"]), row["candidate_id"], flag(saved.get("oos_curve_pass"), bool(saved)), flag(saved.get("oos_quality_confirmed"), bool(saved))])
        body += ["<h2>历史 OOS 全部冻结候选</h2>", table(["C 前排名", "候选", "预写曲线标准", "质量确认"], oos_rows)]
    body.append("<h2>D/V 失败分布与完整提案</h2>")
    distribution = []
    for label, column in (("D", "discovery_failed"), ("V", "validation_failed")):
        counter = Counter(item for value in fit.get(column, pd.Series(dtype=object)) for item in failures(value))
        distribution.extend((label, FAIL_LABELS.get(key, key), count) for key, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    body.append(table(["阶段", "未满足条件", "候选数"], distribution))
    body.append('<p class="muted">同一候选可以同时未满足多个条件，失败分布不能相加为独立淘汰人数。完整提案按币种、策略、候选 ID 排序，包含共同通过但未获账户内最高排名的版本；不按 C 或 OOS 收益选择展示。</p>')
    all_rows = []
    if not fit.empty:
        for row in fit.sort_values(["pair", "strategy", "candidate_id"]).to_dict("records"):
            all_rows.append([row["pair"], row["strategy"], row["filter_id"], flag(row.get("discovery_pass")), flag(row.get("validation_pass")), flag(row.get("fit_pass")), fmt(row.get("preconfirmation_rank"), 0), failure_text(row.get("discovery_failed")), failure_text(row.get("validation_failed"))])
    body.append(f'<details><summary>查看全部 {len(fit)} 个提案及其 D/V 结果</summary>' + table(["币种", "策略", "过滤状态", "D", "V", "共同通过", "C 前排名", "D 失败条件", "V 失败条件"], all_rows) + "</details>")
    body.append("<h2>当前阶段证据与下载</h2>")
    body.append("<p>以下仅链接本轮目录内实际读取的文件。报告哈希清单只记录渲染输入，不替代研究引擎对冻结源与数据的完整性验证。</p>")
    body.append("<ul>" + "".join(f'<li><a href="{escape(name)}">{escape(name)}</a></li>' for name in inputs.hashes) + "</ul>")
    body.append(table(["记录", "SHA-256"], [["protocol.json", inputs.hashes.get("protocol.json", "待保存")], ["C 前锁定", inputs.hashes.get("validation_locked.json", "待保存")], ["最终 IS 冻结（开发状态记录）", development.get("freeze_sha256", "待完成")]]))
    body.append('<details><summary>查看保存的时间协议与阶段状态</summary><pre>' + escape(json.dumps({"temporal_protocol": protocol.get("temporal_protocol", {}), "fit_status": fit_status, "development_status": development, "oos_status": oos_status or "NOT OPENED"}, ensure_ascii=False, indent=2)) + "</pre></details>")
    now = datetime.now(timezone.utc).isoformat()
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态过滤研究：D/V/C 历史时间确认</title><style>
body{margin:0;background:#f3f5f6;color:#1c303b;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:32px 24px 60px}h1{font-size:28px;margin:0 0 8px}h2{margin-top:32px;font-size:21px}h3{font-size:18px}p{max-width:1180px}.muted{color:#556a76}.outcome{background:#fff3d8;border-left:5px solid #b37512;padding:12px 22px;margin:22px 0}.outcome h2{margin-top:0}.outcome p{font-size:18px;font-weight:600}.table-wrap{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;min-width:680px;width:100%;font-size:13px;background:white}th,td{text-align:left;padding:9px 10px;border:1px solid #dce4e8;vertical-align:top}th{background:#e9f0f2;white-space:nowrap}td{overflow-wrap:anywhere}tr:nth-child(even){background:#f8fafb}.candidate{background:white;padding:14px 20px 24px;border:1px solid #dce4e8;margin:22px 0}img{width:100%;height:auto}a{color:#006c76}code{overflow-wrap:anywhere}details{background:white;border:1px solid #dce4e8;padding:12px 16px;margin:16px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}footer{color:#607480;margin-top:30px;font-size:12px}@media(max-width:650px){main{padding:20px 12px}h1{font-size:23px}.candidate{padding:8px}}
</style></head><body><main><h1>状态过滤研究：D/V/C 历史时间确认</h1><p class="muted">开发隔离的历史重复验证 · 零成本 · 固定本金单利 · 全部预确认锁定候选</p>'''
    html += "\n".join(body) + f'<footer>生成于 {escape(now)} UTC。图中每个面板使用相同纵轴尺度；面板内 PnL 从零起算。未完成阶段不补造结果。</footer></main></body></html>'
    (output / "state_research_report.html").write_text(html, encoding="utf-8")
    result = {"report": "state_research_report.html", "figures": figures, "counts": counts,
              "confirmation_completed": complete, "oos_status_exists": bool(oos_status), "oos_evaluated": oos_complete,
              "daily_metric_endpoint_checks": endpoint_checks, "input_sha256": inputs.hashes,
              "generated_utc": now, "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "state_report_build.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = build_report(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
