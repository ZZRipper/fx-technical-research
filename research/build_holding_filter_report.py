"""Saved-artifact report for zero-cost EXACT holding-checkpoint research.

No strategy is executed. Only the selected campaign directory is read. The
existing presentation renderer supplies the frozen-order atlas; this wrapper
adds original-episode quality and contribution diagnostics without new gates.
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
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "reproduced/holding_filter_v3"
REPORT = "holding_filter_report.html"
TITLE = "Holding EXACT checkpoint：零成本持仓退出研究"
TOL = 1e-10
KEYS = ["candidate_id", "freeze_rank", "pair", "strategy", "filter_id", "split"]
QUALITY_COLUMNS = KEYS + ["cohort", "variant", "closed_n", "win_n", "loss_n", "flat_n", "win_rate", "payoff_ratio", "mean_win_pct", "mean_loss_abs_pct", "simple_return_sum_pct"]
EPISODE_COLUMNS = KEYS + ["original_entry_idx", "original_entry_date", "original_exit_date", "original_closed_at_snapshot", "filtered_exit_date", "baseline_return_pp", "filtered_return_pp", "delta_pp", "return_days", "changed_return_days", "risk_signal_in_window", "early_exit_fill_in_window"]
EFFECT_COLUMNS = KEYS + ["baseline_return_pp", "filtered_return_pp", "delta_pp", "original_episodes_with_returns", "affected_original_episodes", "evaluated_checkpoints_in_window", "risk_signals_in_window", "filled_early_exits_in_window", "pending_risk_exits_at_snapshot", "helpful_original_episodes", "harmful_original_episodes", "positive_contributions_pp", "negative_contributions_pp", "largest_helpful_contribution_pp", "top3_helpful_contributions_pp", "top3_share_of_net_improvement", "delta_without_best3_pp"]
EQUIVALENT_COLUMNS = ["pair", "strategy", "candidate_id", "filter_id", "pass", "frozen", "freeze_rank", "signal_path_id", "executed_path_id", "same_signal_path_versions", "same_executed_path_versions", "exit_signals", "filled_early_exits", "pending_risk_exits", "diagnostic_only"]
COHORT_LABELS = {"SCREEN_ACTUAL": "正式口径：实际合格闭合", "PAIRED_BOTH_CLOSED": "配对：相同原交易双闭合", "FILTER_NEWLY_CLOSED": "FILTER 新增闭合记录", "BASE_ONLY_CLOSED": "仅 BASE 合格闭合"}
FAIL_LABELS = {
    "positive_and_annualized_pass": "正收益且达到单利年化要求",
    "return_improvement_pass": "累计收益改善",
    "win_rate_pass": "胜率改善",
    "payoff_pass": "盈亏比改善",
    "drawdown_pass": "回撤恶化限制",
    "return_to_drawdown_pass": "收益/回撤比改善",
    "sample_pass": "闭合笔数与闭合数量比",
    "block_positive_pass": "分段正收益及末段正收益",
    "block_improvement_pass": "分段改善及末段改善",
}


def truth(value):
    return value is not None and not pd.isna(value) and str(value).lower() in {"true", "1", "1.0"}


def fmt(value, digits=2):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.{digits}f}" if np.isfinite(number) else "—"


def percent(value):
    try:
        return fmt(float(value) * 100) + "%" if pd.notna(value) else "—"
    except (TypeError, ValueError):
        return "—"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + (body or f'<tr><td colspan="{len(headers)}">本阶段无记录</td></tr>') + "</tbody></table></div>"


class Inputs:
    def __init__(self, directory):
        self.directory = directory
        self.hashes = {}

    def raw(self, name, required=False):
        if Path(name).name != name:
            raise ValueError("Research inputs must be direct children of the campaign directory.")
        path = self.directory / name
        if not path.exists():
            if required:
                raise ValueError(f"Missing saved artifact: {name}")
            return None
        raw = path.read_bytes()
        self.hashes[name] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, name, default=None, required=False):
        raw = self.raw(name, required)
        return json.loads(raw.decode("utf-8-sig")) if raw is not None else default

    def csv(self, name, required=False):
        raw = self.raw(name, required)
        if raw is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.BytesIO(raw), compression="gzip" if name.endswith(".gz") else None)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


def quality(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("A closed-trade quality cohort contains nonfinite returns.")
    wins, losses = values[values > TOL], values[values < -TOL]
    return {"closed_n": len(values), "win_n": len(wins), "loss_n": len(losses),
            "flat_n": len(values) - len(wins) - len(losses),
            "win_rate": float(len(wins) / len(values)) if len(values) else np.nan,
            "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
            "mean_win_pct": float(wins.mean()) if len(wins) else np.nan,
            "mean_loss_abs_pct": float(-losses.mean()) if len(losses) else np.nan,
            "simple_return_sum_pct": float(values.sum())}


def metric(frame, candidate, split, variant):
    rows = frame.loc[frame.candidate_id.eq(candidate) & frame.split.eq(split) & frame.variant.eq(variant)]
    if len(rows) != 1:
        raise ValueError(f"Expected one official metric: {candidate}, {split}, {variant}.")
    return rows.iloc[0]


def assert_close(actual, expected, label, errors):
    actual, expected = float(actual), float(expected)
    if np.isnan(actual) and np.isnan(expected):
        return
    if not np.isfinite([actual, expected]).all() or not np.isclose(actual, expected, atol=1e-8, rtol=0):
        raise ValueError(f"Saved-artifact diagnostic failed reconciliation: {label}: {actual} != {expected}")
    errors.append(abs(actual - expected))


def ledger_rows(ledger, candidate, variant):
    rows = ledger.loc[ledger.candidate_id.eq(candidate) & ledger.variant.eq(variant)].copy()
    if rows.entry_idx.isna().any() or rows.entry_idx.duplicated().any():
        raise ValueError("Original entry_idx must uniquely identify each executed episode.")
    rows["entry_idx"] = rows.entry_idx.astype(int)
    for column in ("entry_date", "exit_date"):
        rows[column] = pd.to_datetime(rows[column], errors="raise")
        if rows[column].isna().any():
            raise ValueError("Saved ledger entry/observed exit dates cannot be missing.")
    return rows.set_index("entry_idx", drop=False).sort_index()


def eligible(rows, start, end):
    return rows.loc[rows.entry_date.between(start, end) & rows.eligible_closed.map(truth) & rows.exit_date.le(end)]


def in_window(rows, date_column, start, end):
    if date_column not in rows:
        return pd.Series(False, index=rows.index)
    return pd.to_datetime(rows[date_column], errors="coerce").between(start, end)


def derive_diagnostics(frozen, daily, metrics, is_ledger, is_decisions, oos_ledger, oos_decisions, protocol, oos_complete):
    quality_rows, episodes, effects, errors = [], [], [], []
    splits = [("IS", protocol["is_start"], protocol["is_end"], is_ledger, is_decisions)]
    if oos_complete:
        splits.append(("OOS", protocol["oos_start"], protocol["oos_end"], oos_ledger, oos_decisions))
    for candidate in frozen.sort_values("freeze_rank").to_dict("records"):
        cid = candidate["candidate_id"]
        for split, start, end, ledger, decisions in splits:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            key = {name: candidate[name] for name in KEYS if name != "split"}
            key["split"] = split
            base, filtered = [ledger_rows(ledger, cid, variant) for variant in ("BASE", "FILTER")]
            if set(base.index) != set(filtered.index):
                raise ValueError("A pure early-exit filter must preserve all original execution entries.")
            if not base.position.eq(filtered.position).all() or not base.entry_date.eq(filtered.entry_date).all():
                raise ValueError("BASE/FILTER entry identity differs; cannot form original-episode pairs.")
            actual_base, actual_filter = eligible(base, start, end), eligible(filtered, start, end)
            common = actual_base.index.intersection(actual_filter.index)
            new = actual_filter.index.difference(actual_base.index)
            base_only = actual_base.index.difference(actual_filter.index)
            cohorts = [("SCREEN_ACTUAL", "BASE", actual_base), ("SCREEN_ACTUAL", "FILTER", actual_filter),
                       ("PAIRED_BOTH_CLOSED", "BASE", actual_base.loc[common]), ("PAIRED_BOTH_CLOSED", "FILTER", actual_filter.loc[common]),
                       ("FILTER_NEWLY_CLOSED", "FILTER", actual_filter.loc[new]), ("BASE_ONLY_CLOSED", "BASE", actual_base.loc[base_only])]
            for cohort, variant, rows in cohorts:
                q = quality(rows.gross_simple_return_pct)
                quality_rows.append({**key, "cohort": cohort, "variant": variant, **q})
                if cohort == "SCREEN_ACTUAL":
                    saved = metric(metrics, cid, split, variant)
                    for field in ("closed_n", "win_n", "loss_n", "flat_n", "win_rate", "payoff_ratio"):
                        assert_close(q[field], saved[field], f"{cid} {split} {variant} {field}", errors)
            observed = decisions.loc[decisions.candidate_id.eq(cid)].copy()
            executed_decisions = observed.loc[observed.entry_idx.notna()].copy()
            executed_decisions["entry_idx"] = executed_decisions.entry_idx.astype(int)
            if executed_decisions.entry_idx.duplicated().any() or set(executed_decisions.entry_idx) != set(base.index):
                raise ValueError("Saved decisions must cover each original executed episode once.")
            decision_index = executed_decisions.set_index("entry_idx")
            sub = daily.loc[daily.candidate_id.eq(cid) & daily.split.eq(split)].copy()
            sub["date"] = pd.to_datetime(sub.date)
            by_variant = {variant: sub.loc[sub.variant.eq(variant)].sort_values("date") for variant in ("BASE", "FILTER")}
            dates = pd.DatetimeIndex(by_variant["BASE"].date)
            if dates.has_duplicates or not dates.equals(pd.DatetimeIndex(by_variant["FILTER"].date)):
                raise ValueError("Saved BASE/FILTER calendars must match without duplicates.")
            if len(dates) == 0 or dates.min() < start or dates.max() > end:
                raise ValueError("Saved daily split is empty or extends beyond its prewritten bounds.")
            br, fr = [by_variant[variant].gross_daily_return.to_numpy(float) * 100 for variant in ("BASE", "FILTER")]
            bh, fh = [by_variant[variant].held.to_numpy(int) for variant in ("BASE", "FILTER")]
            if np.any((fh != 0) & (fh != bh)):
                raise ValueError("A pure early exit must never add or reverse BASE exposure.")
            coverage = np.zeros(len(dates), dtype=int)
            local = []
            for original in base.itertuples(index=False):
                # entry+1 .. exit inclusive, including the old position's return
                # on a reversal date. Using saved dates avoids index resets.
                left = int(dates.searchsorted(original.entry_date, side="right"))
                right = int(dates.searchsorted(original.exit_date, side="right"))
                if left >= right:
                    continue
                coverage[left:right] += 1
                decision = decision_index.loc[int(original.entry_idx)]
                risk_date = pd.to_datetime(decision.get("risk_signal_date"), errors="coerce")
                fill_date = pd.to_datetime(decision.get("actual_exit_date"), errors="coerce")
                b_return, f_return = float(br[left:right].sum()), float(fr[left:right].sum())
                item = {**key, "original_entry_idx": int(original.entry_idx), "original_entry_date": original.entry_date,
                        "original_exit_date": original.exit_date, "original_closed_at_snapshot": truth(original.closed),
                        "filtered_exit_date": filtered.loc[original.entry_idx, "exit_date"],
                        "baseline_return_pp": b_return, "filtered_return_pp": f_return, "delta_pp": f_return - b_return,
                        "return_days": right - left, "changed_return_days": int((bh[left:right] != fh[left:right]).sum()),
                        "risk_signal_in_window": bool(pd.notna(risk_date) and start <= risk_date <= end),
                        "early_exit_fill_in_window": bool(truth(decision.get("early_exit")) and pd.notna(fill_date) and start <= fill_date <= end)}
                local.append(item)
            if np.any(coverage > 1) or np.any((bh != 0) & (coverage != 1)):
                raise ValueError("Original BASE episodes do not partition the held return days.")
            if np.any((bh == 0) & ((br != 0) | (fr != 0))):
                raise ValueError("A cash day has nonzero BASE or FILTER gross return.")
            for variant, series in (("BASE", br), ("FILTER", fr)):
                assert_close(series.sum(), metric(metrics, cid, split, variant).gross_simple_return_pct, f"{cid} {split} {variant} daily total", errors)
            total = float(fr.sum() - br.sum())
            assert_close(sum(row["delta_pp"] for row in local), total, f"{cid} {split} episode delta", errors)
            assert_close(sum(row["baseline_return_pp"] for row in local), br.sum(), f"{cid} {split} BASE episode total", errors)
            assert_close(sum(row["filtered_return_pp"] for row in local), fr.sum(), f"{cid} {split} FILTER episode total", errors)
            deltas = np.array([row["delta_pp"] for row in local], dtype=float)
            positive, negative = deltas[deltas > TOL], deltas[deltas < -TOL]
            top3 = float(np.sort(positive)[-3:].sum())
            risk_mask = in_window(observed, "risk_signal_date", start, end)
            fill_mask = in_window(observed, "actual_exit_date", start, end)
            evaluated_mask = in_window(observed, "checkpoint_date", start, end)
            evaluated = observed.get("evaluated", pd.Series(False, index=observed.index)).map(truth)
            early = observed.get("early_exit", pd.Series(False, index=observed.index)).map(truth)
            pending = observed.get("risk_pending", observed.get("risk_exit_pending", pd.Series(False, index=observed.index))).map(truth)
            effects.append({**key, "baseline_return_pp": float(br.sum()), "filtered_return_pp": float(fr.sum()), "delta_pp": total,
                            "original_episodes_with_returns": len(local), "affected_original_episodes": sum(row["changed_return_days"] > 0 for row in local),
                            "evaluated_checkpoints_in_window": int((evaluated_mask & evaluated).sum()),
                            "risk_signals_in_window": int(risk_mask.sum()), "filled_early_exits_in_window": int((fill_mask & early).sum()),
                            "pending_risk_exits_at_snapshot": int((pending & risk_mask).sum()),
                            "helpful_original_episodes": len(positive), "harmful_original_episodes": len(negative),
                            "positive_contributions_pp": float(positive.sum()), "negative_contributions_pp": float(negative.sum()),
                            "largest_helpful_contribution_pp": float(positive.max()) if len(positive) else 0.,
                            "top3_helpful_contributions_pp": top3, "top3_share_of_net_improvement": top3 / total if total > TOL else np.nan,
                            "delta_without_best3_pp": total - top3})
            episodes.extend(local)
    return (pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS), pd.DataFrame(episodes, columns=EPISODE_COLUMNS),
            pd.DataFrame(effects, columns=EFFECT_COLUMNS), {"checks": len(errors), "max_absolute_error": max(errors, default=0.)})


def unconditional(protocol):
    specs = protocol.get("candidate_specs", [])
    return str(protocol.get("version", "")).startswith("timecap") or bool(specs and all("condition" not in item for item in specs))


def campaign_title(protocol):
    return "Time-cap EXACT checkpoint：零成本持仓时间上限研究" if unconditional(protocol) else TITLE


def method_section(protocol):
    text = '''<section id="holding-method"><h2>本轮方法：EXACT checkpoint，成本为 0</h2>
<p>每个候选固定一个持仓年龄：5、10、20、40 或 63 个已赚取收益的报价日。原始执行入场在 e 收盘，checkpoint 位于 e+k 收盘；只在这一时点判断一次。条件为 false 或 UNKNOWN 后不再检查，也不推广为 age≥k 的每日首次触发策略。具体条件和阈值以本轮冻结候选记录为准。</p>
<p>累计 PnL、截至当时的峰值回撤、最近 5 日方向收益和可用方向冲突都只能使用 checkpoint 时已知信息；完整交易最终 MFE、最终输赢和剩余收益不是触发输入。若触发，在下一收盘执行退出，执行日收益仍属于旧仓，可避免的尾收益从再下一收益日起算。原技术规则已经安排的退出优先；退出后不在同一原始 episode 中重新入场。样本末尚未执行的指令保持 pending，不制造期末平仓。</p>
<p>全部 20 条规则 × 9 个币种是基础研究范围；只按用户原 SCREEN 筛选与排序。下面的作用样本、配对质量和前三贡献集中度<strong>仅解释结果，不增加硬门槛</strong>，也不改写正式指标。三年 OOS 已经被观察过，属于再次使用的历史评价，不能称为全新、未触碰或统计独立的样本外证据。</p></section>'''
    if unconditional(protocol):
        begin = text.index("<p>")
        finish = text.index("</p>", begin) + len("</p>")
        text = text[:begin] + '<p>每个候选固定一个持仓年龄：5、10、20、40 或 63 个已赚取收益的报价日。原始执行入场在 e 收盘，checkpoint 位于 e+k 收盘；若原技术持仓仍有效，就不分盈亏发出一次退出指令。本轮观察的是 ALL cohort，不使用额外 PnL、回撤或方向条件。</p>' + text[finish:]
        begin = text.index("<p>", begin + 3)
        finish = text.index("</p>", begin) + len("</p>")
        text = text[:begin] + '<p>checkpoint 收盘发出退出指令，在下一收盘执行，执行日收益仍属于旧仓，可避免的尾收益从再下一收益日起算。原技术规则已经安排的退出优先；退出后不在同一原始 episode 中重新入场。样本末尚未执行的指令保持 pending，不制造期末平仓。</p>' + text[finish:]
    return text


def equivalent_paths(decisions, selection, all_metrics):
    if set(decisions.candidate_id) != set(selection.candidate_id):
        raise ValueError("All-candidate decisions must match the complete IS selection.")
    rows = []
    identity = selection.set_index("candidate_id")
    for cid, group in decisions.groupby("candidate_id", sort=True):
        selected = identity.loc[cid]
        early = group.loc[group.early_exit.map(truth)].copy()
        if early[["entry_idx", "risk_signal_idx"]].isna().any().any() or early.entry_idx.duplicated().any():
            raise ValueError("Each exit signal must refer to one observed original entry.")
        pairs = sorted((int(row.entry_idx), int(row.risk_signal_idx)) for row in early.itertuples(index=False))
        filled = early.loc[early.actual_exit_idx.notna()]
        filled_pairs = sorted((int(row.entry_idx), int(row.risk_signal_idx)) for row in filled.itertuples(index=False))
        def path_id(values):
            payload = json.dumps([str(selected.pair), str(selected.strategy), values], separators=(",", ":"))
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        rows.append({"pair": selected.pair, "strategy": selected.strategy, "candidate_id": cid,
                     "filter_id": selected.filter_id, "pass": truth(selected["pass"]), "frozen": truth(selected.frozen), "freeze_rank": selected.freeze_rank,
                     "signal_path_id": path_id(pairs), "executed_path_id": path_id(filled_pairs),
                     "exit_signals": len(early), "filled_early_exits": len(filled),
                     "pending_risk_exits": int(early.risk_pending.map(truth).sum()), "diagnostic_only": True})
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=EQUIVALENT_COLUMNS), {"versions": 0, "unique_signal_paths": 0, "unique_executed_paths": 0, "redundant_versions_by_signal_path": 0, "max_within_path_is_return_range_pp": 0.}
    result["same_signal_path_versions"] = result.groupby("signal_path_id").candidate_id.transform("size")
    result["same_executed_path_versions"] = result.groupby("executed_path_id").candidate_id.transform("size")
    main = all_metrics.loc[all_metrics.split.eq("IS") & all_metrics.variant.eq("FILTER"), ["candidate_id", "gross_simple_return_pct"]]
    compared = result.merge(main, on="candidate_id", validate="one_to_one")
    if len(compared) != len(result):
        raise ValueError("All candidate paths require saved IS FILTER metrics.")
    ranges = compared.groupby("executed_path_id").gross_simple_return_pct.agg(lambda x: float(x.max() - x.min()))
    if (ranges > 1e-8).any():
        raise ValueError("Equivalent saved executed paths disagree on IS returns.")
    summary = {"versions": len(result), "unique_signal_paths": int(result.signal_path_id.nunique()),
               "unique_executed_paths": int(result.executed_path_id.nunique()),
               "redundant_versions_by_signal_path": int(len(result) - result.signal_path_id.nunique()),
               "max_within_path_is_return_range_pp": float(ranges.max())}
    return result[EQUIVALENT_COLUMNS].sort_values(["pair", "strategy", "filter_id"]), summary


def observation_sections(hypotheses, coverage, equivalents, equivalence, protocol):
    candidates = hypotheses.get("candidates", [])
    proposed_accounts = int(coverage.candidate_count.gt(0).sum()) if "candidate_count" in coverage else 0
    sections = [f'<section id="holding-observations"><h2>观察覆盖与 D-only 提案依据</h2><p>本轮已保存 {len(coverage)} 个 BASE 账户的覆盖记录，{proposed_accounts} 个账户提出候选，{len(coverage) - proposed_accounts} 个账户没有候选。提出候选只读取 D1（2005–09）、D2（2010–14）及合并 D（2005–14）的结果；每半段至少 10 个合格可执行原始交易，剩余可避免收益之和均为负，合并 D 也为负。前三笔集中度仅诊断，不是附加淘汰条件。</p><p>这只是提出待检验规则的 IS 观察，后续必须对 EXACT checkpoint 规则完整回放并满足原 SCREEN；不能把观察到的负尾收益当成已实现的合格策略收益。</p>']
    if not coverage.empty:
        sections.append('<details><summary>全部 BASE 账户覆盖（含零候选账户）</summary>' + table(["币种", "策略", "提案版本数", "计算截止"], [[row.pair, row.strategy, int(row.candidate_count), row.compute_end] for row in coverage.sort_values(["pair", "strategy"]).itertuples(index=False)]) + '</details>')
    evidence = []
    for row in sorted(candidates, key=lambda item: (item["pair"], item["strategy"], item["filter_id"])):
        fields = [row["pair"], row["strategy"], row["filter_id"]]
        for period in ("D1", "D2", "D"):
            saved = row.get("discovery_evidence", {}).get(period, {})
            fields += [fmt(saved.get("closed_n"), 0), fmt(saved.get("remaining_return_sum_pct")), fmt(saved.get("remaining_without_worst3_pct"))]
        evidence.append(fields)
    sections.append(f'<details><summary>全部 {len(candidates)} 个提案的已保存 D-only 依据</summary>' + table(["币种", "策略", "规则", "D1 笔数", "D1 可避免尾收益 %", "D1 去掉最亏3笔后 %", "D2 笔数", "D2 可避免尾收益 %", "D2 去掉最亏3笔后 %", "D 笔数", "D 可避免尾收益 %", "D 去掉最亏3笔后 %"], evidence) + '</details></section>')
    overlap_note = '不同固定年龄可能在本样本产生相同退出路径。' if unconditional(protocol) else '在 age=5 时，pnl_negative 与 recent5_negative 都检查前 5 个已赚收益之和是否为负，两者的 AND 在数学上同义。其他组合也可能在本样本产生相同路径。'
    sections.append(f'<section id="holding-equivalents"><h2>重叠规则与相同 IS 退出路径</h2><p><strong>{equivalence.get("versions", 0)} 是登记版本数，不是独立试验数量。</strong>{overlap_note}</p><p>仅在同一币种/策略内，按全部 early_exit 的 <code>(entry_idx, risk_signal_idx)</code> 集合归组：得到 <strong>{equivalence.get("unique_signal_paths", 0)} 个退出信号路径类</strong>；只看已经执行的退出，得到 <strong>{equivalence.get("unique_executed_paths", 0)} 个执行路径类</strong>。末日 pending 信号单列。路径类数量仍不能解释为统计独立证据。</p><p>这项归组只解释重叠；源、参数、原筛选和冻结都已固定，不在报告中去重或改变排名。相同执行路径的已保存 IS 收益已核对一致。完整映射见 <a href="equivalent_rules.csv">equivalent_rules.csv</a>。</p>')
    sections.append('<details><summary>全部版本的路径归组</summary>' + table(["币种", "策略", "规则", "信号路径 ID", "同信号路径版本数", "退出信号数", "已执行退出", "待执行退出"], [[row.pair, row.strategy, row.filter_id, row.signal_path_id[:16], int(row.same_signal_path_versions), int(row.exit_signals), int(row.filled_early_exits), int(row.pending_risk_exits)] for row in equivalents.itertuples(index=False)]) + '</details></section>')
    return "\n".join(sections)


def diagnostic_sections(frozen, paired, effects):
    blocks = ['''<section id="paired-explainer"><h2>质量分母与原始交易配对</h2><p>正式 SCREEN 使用实际 FILTER 账本中期内开仓、期内闭合的合格交易。提前退出可能把 BASE 尚未闭合或跨期的交易变为 FILTER 本期已闭合，改变质量统计分母。另用原 <code>entry_idx</code> 配对，只比较两个版本都在期内合格闭合的同一原始交易。这是事后诊断样本，不能成为实时退出条件，也不替代正式 SCREEN。</p><p>“FILTER 新增闭合记录”单列数量和单利收益，并不是新增独立样本。闭合数量比是 FILTER 合格闭合数 / BASE 合格闭合数，允许超过 100%，不截断。闭合交易收益和不必等于日历账户累计收益：跨界和未闭合贡献仍进入账户曲线。</p><p>作用归因以原 BASE 的 entry+1 至 exit（含退出日）为区间，将两版本已保存的每日收益归回同一原始交易，反转日旧仓收益不会分给新仓。贡献先在交易段内求和再判断利弊，不能逐日挑出有利部分；全部贡献与主账户 FILTER−BASE 严格核对。</p></section>''']
    for candidate in frozen.sort_values("freeze_rank").to_dict("records"):
        cid, rank = candidate["candidate_id"], int(candidate["freeze_rank"])
        q = paired.loc[paired.candidate_id.eq(cid)]
        e = effects.loc[effects.candidate_id.eq(cid)]
        blocks.append(f'<section id="holding-diagnostic-{rank}"><h2>IS 顺序 {rank} · {escape(candidate["pair"])} · {escape(candidate["strategy"])}：作用与配对诊断</h2><p><code>{escape(candidate["filter_id"])}</code>；候选顺序沿用冻结顺序，不按 OOS 排序。</p>')
        rows = [[row.split, COHORT_LABELS[row.cohort], row.variant, int(row.closed_n), int(row.win_n), int(row.loss_n), percent(row.win_rate), fmt(row.payoff_ratio), fmt(row.simple_return_sum_pct)] for row in q.itertuples(index=False)]
        blocks.append(table(["区间", "样本定义", "版本", "合格闭合数", "盈利数", "亏损数", "胜率", "盈亏比", "闭合交易单利和 %"], rows))
        rows = [[row.split, int(row.evaluated_checkpoints_in_window), int(row.risk_signals_in_window), int(row.filled_early_exits_in_window), int(row.pending_risk_exits_at_snapshot), int(row.affected_original_episodes), fmt(row.delta_pp), int(row.helpful_original_episodes), int(row.harmful_original_episodes), fmt(row.top3_helpful_contributions_pp), percent(row.top3_share_of_net_improvement), fmt(row.delta_without_best3_pp)] for row in e.itertuples(index=False)]
        blocks.append(table(["区间", "当期实际判定 checkpoint", "当期退出信号", "当期提前退出执行", "末尾待执行", "当期受影响原交易", "FILTER−BASE pp", "有利原交易", "不利原交易", "前三有利贡献 pp", "前三占净改善", "剔除前三后改善 pp"], rows))
        blocks.append('<p class="muted">信号/执行按事件日期计数，受影响交易按当期实际收益归因；跨界时这些数量不必相等。原始交易贡献占净改善可超过 100%；净改善不为正时该占比不定义。总闭合交易数不能代替真正作用样本，集中度不构成额外淘汰门槛。</p></section>')
    return "\n".join(blocks)


def failure_section(selection, is_metrics):
    count = Counter()
    for value in selection.get("failed_criteria", pd.Series(dtype=object)).fillna(""):
        count.update(item for item in str(value).split("|") if item)
    rows = [(FAIL_LABELS.get(name, name), number) for name, number in sorted(count.items(), key=lambda item: (-item[1], item[0]))]
    accounts = len(is_metrics[["pair", "strategy"]].drop_duplicates()) if {"pair", "strategy"}.issubset(is_metrics.columns) else 0
    passed = int(selection["pass"].map(truth).sum()) if "pass" in selection else 0
    return f'<section id="holding-screen-distribution"><h2>完整 IS 筛选与失败分布</h2><p>已保存 {len(selection):,} 个候选账户组合，涵盖 {accounts} 个基础账户；原 SCREEN 综合通过 {passed} 个。全部候选记录保留在原 IS 筛选表，不用配对或集中度诊断另加门槛。</p>' + table(["未满足的原 SCREEN 条件", "候选数"], rows) + '<p class="muted">同一候选可以有多个失败原因，因此这些计数不能相加为独立淘汰人数。完整记录见 <a href="is_selection.csv">is_selection.csv</a> 与 <a href="is_all_metrics.csv">is_all_metrics.csv</a>。</p></section>'


def save_manifest(output, inputs, built):
    built.update(input_sha256=inputs.hashes, generated_utc=datetime.now(timezone.utc).isoformat(),
                 renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    paths = [REPORT, "filter_research_report.html", "pnl_comparison.pdf", "paired_quality.csv", "episode_effects.csv", "effect_summary.csv", "equivalent_rules.csv", *built.get("figures", [])]
    built["output_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in dict.fromkeys(paths) if (output / name).is_file()}
    (output / "holding_report_build.json").write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")
    return built


def build_report(output_dir: Path = DEFAULT) -> dict:
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError(f"Campaign directory does not exist: {output}")
    inputs = Inputs(output)
    protocol = inputs.json("protocol.json", {})
    title = campaign_title(protocol)
    development = inputs.json("development_status.json", {})
    completed = development.get("status") == "passed" and truth(development.get("is_screen_completed"))
    if not completed:
        html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + title + '</title><body><main><h1>' + title + '</h1><p><b>IS 筛选尚未完成；未读取 OOS 状态或结果，未制造冻结候选。</b></p>' + method_section(protocol) + '</main></body></html>'
        (output / REPORT).write_text(html, encoding="utf-8")
        return save_manifest(output, inputs, {"report": REPORT, "figures": [], "is_completed": False, "oos_evaluated": False, "frozen_candidates": None, "diagnostics_completed": False})
    if not protocol or float(protocol.get("cost_bp", 0)) != 0:
        raise ValueError("A completed zero-cost checkpoint protocol is required.")
    selection = inputs.csv("is_selection.csv", required=True)
    all_metrics = inputs.csv("is_all_metrics.csv", required=True)
    all_decisions = inputs.csv("is_all_decisions.csv.gz", required=True)
    coverage = inputs.csv("is_account_coverage.csv", required=True)
    hypotheses = inputs.json("is_hypotheses.json", {}, required=True)
    freeze = inputs.json("frozen_selection.json", required=True)
    if development.get("freeze_sha256") and development["freeze_sha256"] != inputs.hashes["frozen_selection.json"]:
        raise ValueError("Saved final freeze hash differs from development status.")
    if freeze.get("protocol_sha256") and freeze["protocol_sha256"] != inputs.hashes["protocol.json"]:
        raise ValueError("Frozen protocol hash differs from the saved protocol.")
    if "frozen" not in selection or "freeze_rank" not in selection:
        raise ValueError("Saved IS selection does not identify the frozen candidates.")
    selection["frozen"] = selection.frozen.map(truth)
    frozen = selection.loc[selection.frozen].sort_values("freeze_rank")
    ids = set(frozen.candidate_id)
    if ids != {row["candidate_id"] for row in freeze.get("candidates", [])}:
        raise ValueError("IS selection and frozen candidate list differ.")
    is_daily = inputs.csv("is_frozen_daily.csv.gz", required=bool(ids))
    is_metrics = inputs.csv("is_frozen_metrics.csv", required=bool(ids))
    is_ledger = inputs.csv("is_frozen_trades.csv.gz", required=bool(ids))
    is_decisions = inputs.csv("is_frozen_decisions.csv.gz", required=bool(ids))
    for name, expected in freeze.get("is_output_sha256", {}).items():
        if name in inputs.hashes and inputs.hashes[name] != expected:
            raise ValueError(f"Frozen IS artifact has changed: {name}")
    status = inputs.json("oos_status.json", {})
    if not ids and status:
        raise ValueError("An empty freeze set must not open an OOS evaluation stage.")
    oos_complete = status.get("status") == "passed"
    oos_daily = oos_metrics = oos_ledger = oos_decisions = pd.DataFrame()
    if oos_complete:
        if not ids:
            raise ValueError("There are no frozen candidates, but OOS claims completion.")
        oos_daily = inputs.csv("oos_daily.csv.gz", required=True)
        oos_metrics = inputs.csv("oos_metrics.csv", required=True)
        oos_ledger = inputs.csv("frozen_full_trade_ledger.csv.gz", required=True)
        oos_decisions = inputs.csv("frozen_full_episode_decisions.csv.gz", required=True)
        inputs.csv("candidate_assessment.csv", required=True)
        inputs.csv("oos_anniversary_metrics.csv", required=True)
        inputs.csv("oos_fresh_start_metrics.csv", required=True)
        for name, expected in status.get("output_sha256", {}).items():
            if name in inputs.hashes and inputs.hashes[name] != expected:
                raise ValueError(f"Saved OOS artifact has changed: {name}")
    for frame, name in ((is_daily, "IS daily"), (is_metrics, "IS metrics"), (is_ledger, "IS ledger"), (is_decisions, "IS decisions"), (oos_daily, "OOS daily"), (oos_metrics, "OOS metrics"), (oos_ledger, "OOS ledger"), (oos_decisions, "OOS decisions")):
        if not frame.empty and set(frame.candidate_id) != ids:
            raise ValueError(f"{name} must cover exactly the saved frozen candidates.")
    daily = pd.concat([is_daily, oos_daily], ignore_index=True)
    metrics = pd.concat([is_metrics, oos_metrics], ignore_index=True)
    equivalents, equivalence = equivalent_paths(all_decisions, selection, all_metrics)
    if ids:
        paired, episodes, effects, reconciliation = derive_diagnostics(frozen, daily, metrics, is_ledger, is_decisions, oos_ledger, oos_decisions, protocol, oos_complete)
    else:
        paired, episodes, effects = pd.DataFrame(columns=QUALITY_COLUMNS), pd.DataFrame(columns=EPISODE_COLUMNS), pd.DataFrame(columns=EFFECT_COLUMNS)
        reconciliation = {"checks": 0, "max_absolute_error": 0.}
    paired.to_csv(output / "paired_quality.csv", index=False)
    episodes.to_csv(output / "episode_effects.csv", index=False)
    effects.to_csv(output / "effect_summary.csv", index=False)
    equivalents.to_csv(output / "equivalent_rules.csv", index=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_report_cache"))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    # Presentation-only dependency; never import the runner or execution engine.
    from fx_filter_report import render_filter_report
    validation = dict(development)
    if oos_complete:
        validation.update(status)
    validation.update(oos_evaluated=oos_complete, protocol_sha256=inputs.hashes["protocol.json"], freeze_sha256=inputs.hashes["frozen_selection.json"], frozen_candidates=len(frozen))
    base = render_filter_report(output, daily, metrics, selection, protocol, validation)
    html = (output / base["report"]).read_text(encoding="utf-8")
    html = html.replace("FX filter：本轮隔离历史测试", title)
    html = html.replace("FROZEN IS SELECTION · HISTORICAL ISOLATION", "EXACT HOLDING CHECKPOINT · ZERO COST · REUSED HISTORY")
    additions = method_section(protocol) + failure_section(selection, all_metrics) + observation_sections(hypotheses, coverage, equivalents, equivalence, protocol)
    if ids:
        additions += diagnostic_sections(frozen, paired, effects)
    else:
        additions += '<section id="holding-diagnostics-unavailable"><h2>冻结候选配对质量与作用归因：不适用</h2><p>本轮没有原 SCREEN 达标并冻结的候选，也没有保存冻结候选逐日曲线、交易与决策。paired_quality.csv、episode_effects.csv 和 effect_summary.csv 仅保留列定义，未编造质量样本或收益归因。本报告没有为展示曲线重跑策略；PDF 只是一页 IS 停止状态说明。</p></section>'
    if not ids:
        additions = '<section class="note caution"><h2>本轮无 IS 达标冻结候选</h2><p>流程停在 IS；没有打开 OOS 结果，也没有构造 OOS 曲线。下面保留完整原 SCREEN 失败分布。</p></section>' + additions
    elif not oos_complete:
        additions = '<section class="note"><h2>历史 OOS 待完成</h2><p>本报告只使用已冻结 IS 结果；没有读取未完成的 OOS 曲线、交易或指标。</p></section>' + additions
    html = html.replace("<main>", "<main>" + additions, 1)
    if not ids:
        html = html.replace("下载全部冻结候选 PnL 图册", "下载 IS 停止状态页（无收益曲线）")
    links = '<section id="holding-derived-files"><h2>本报告新增的解释性产物</h2><p><a href="paired_quality.csv">paired_quality.csv</a> · <a href="episode_effects.csv">episode_effects.csv</a> · <a href="effect_summary.csv">effect_summary.csv</a> · <a href="holding_report_build.json">holding_report_build.json</a></p><p>所有诊断仅由当前目录已保存的 ledger、decisions 和 daily 推导。正式指标未改写，未重跑策略或新增筛选条件。</p><pre>' + escape(json.dumps(reconciliation, ensure_ascii=False, indent=2)) + '</pre></section>'
    html = html.replace("</main>", links + "</main>", 1)
    (output / REPORT).write_text(html, encoding="utf-8")
    return save_manifest(output, inputs, {"report": REPORT, "base_report": base["report"], "atlas": base["atlas"], "figures": base["figures"], "pdf_pages": base["pdf_pages"],
                        "is_completed": True, "oos_evaluated": oos_complete, "frozen_candidates": len(frozen), "diagnostics_completed": True,
                        "ordering": "saved IS freeze_rank only", "paired_quality_rows": len(paired), "episode_effect_rows": len(episodes),
                        "effect_summary_rows": len(effects), "paired_quality_available": bool(ids), "equivalence_diagnostic": equivalence,
                        "reconciliation": reconciliation, "base_renderer_sha256": hashlib.sha256((ROOT / "src/fx_filter_report.py").read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
