"""Attribute the frozen experiment using saved observations only; never refit."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ACTIONS = ("entry_rejected", "early_exit", "retained")
IDENTITY = ["candidate_id", "pair", "strategy", "filter_id"]
TOL = 1e-8
INPUTS = ("is_frozen_daily.csv.gz", "oos_daily.csv.gz", "frozen_full_trade_ledger.csv.gz",
          "frozen_full_episode_decisions.csv.gz", "is_frozen_metrics.csv", "oos_metrics.csv",
          "candidate_assessment.csv", "is_all_metrics.csv")


def check_close(actual, expected, label):
    error = abs(float(actual) - float(expected))
    if not np.isfinite(error) or error > TOL:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    return error


def read_table(path):
    frame = pd.read_csv(path)
    for name in frame.columns:
        if name == "date" or name.endswith("_date"):
            frame[name] = pd.to_datetime(frame[name])
    return frame


def build_attribution(daily, ledger, decisions, metrics):
    """Partition earned returns, with exit-day returns owned by the old position."""
    if daily.duplicated(["candidate_id", "variant", "date"]).any():
        raise AssertionError("Duplicate daily candidate/variant/date observations.")
    if not np.isfinite(daily[["gross_daily_return", "net_daily_return"]].to_numpy()).all():
        raise AssertionError("Nonfinite daily returns.")
    # The saved primary experiment is cost-free. Cost-sensitive entry-day flows
    # require a separate ownership convention and must not be silently included.
    if not np.allclose(daily.gross_daily_return, daily.net_daily_return, atol=1e-12, rtol=0):
        raise AssertionError("This attribution requires the saved zero-cost primary daily paths.")
    if metrics.duplicated(["candidate_id", "variant", "split"]).any():
        raise AssertionError("Duplicate main metrics.")
    detail, audit = [], []
    for cid, observations in daily.groupby("candidate_id", sort=False):
        base = observations.loc[observations.variant.eq("BASE")].sort_values("date").reset_index(drop=True)
        filt = observations.loc[observations.variant.eq("FILTER")].sort_values("date").reset_index(drop=True)
        if not len(base) or not base[["date", "split"]].equals(filt[["date", "split"]]):
            raise AssertionError(f"BASE/FILTER calendars differ: {cid}")
        dates = pd.DatetimeIndex(base.date)
        identity = {name: base[name].iloc[0] for name in IDENTITY}
        trades = ledger.loc[ledger.candidate_id.eq(cid) & ledger.variant.eq("BASE")].copy()
        actual_trades = ledger.loc[ledger.candidate_id.eq(cid) & ledger.variant.eq("FILTER")].copy()
        dec = decisions.loc[decisions.candidate_id.eq(cid)].copy()
        if trades.entry_idx.duplicated().any() or dec.entry_idx.duplicated().any():
            raise AssertionError(f"Duplicate original episode entry: {cid}")
        if set(trades.entry_idx) != set(dec.entry_idx):
            raise AssertionError(f"Original ledger/decision episode mismatch: {cid}")
        dec = dec.set_index("entry_idx")
        actual_trades = actual_trades.set_index("entry_idx", verify_integrity=True)
        if set(actual_trades.index) != set(dec.index[dec.accepted]):
            raise AssertionError(f"Accepted decision/FILTER ledger mismatch: {cid}")
        coordinates = []
        for index_field, date_field in (("entry_idx", "entry_date"), ("exit_idx", "exit_date")):
            locations = dates.get_indexer(pd.DatetimeIndex(trades[date_field]))
            if np.any(locations < 0):
                raise AssertionError(f"Trade date missing from saved calendar: {cid}")
            coordinates.extend((trades[index_field].to_numpy(int) - locations).tolist())
        if not coordinates or len(set(coordinates)) != 1:
            raise AssertionError(f"Original index/date coordinates disagree: {cid}")
        offset = coordinates[0]
        global_indices = np.arange(len(dates)) + offset
        owner = np.full(len(dates), -1, dtype=int)
        expected_filter_held = np.zeros(len(dates), dtype=int)
        bret, fret = base.net_daily_return.to_numpy(float), filt.net_daily_return.to_numpy(float)
        for trade in trades.itertuples(index=False):
            decision = dec.loc[trade.entry_idx]
            if (decision.position != trade.position or decision.signal_idx != trade.signal_idx
                    or decision.entry_date != trade.entry_date):
                raise AssertionError(f"Episode identity mismatch: {cid}/{trade.trade_id}")
            if bool(trade.closed) != (not bool(decision.original_open)):
                raise AssertionError("Original open/closed status mismatch.")
            if trade.closed and (int(decision.original_exit_idx) != trade.exit_idx
                                 or decision.original_exit_date != trade.exit_date):
                raise AssertionError("Original exit boundary mismatch.")
            selection = (global_indices > trade.entry_idx) & (global_indices <= trade.exit_idx)
            if np.any(owner[selection] != -1):
                raise AssertionError("Original earned-return episodes overlap.")
            owner[selection] = int(trade.trade_id)
            if not np.all(base.held.to_numpy()[selection] == trade.position):
                raise AssertionError("BASE held positions disagree with episode interval.")
            accepted = bool(decision.accepted)
            actual_exit = int(decision.actual_exit_idx) if pd.notna(decision.actual_exit_idx) else None
            if accepted:
                actual = actual_trades.loc[trade.entry_idx]
                if bool(actual.closed) != (actual_exit is not None):
                    raise AssertionError("FILTER ledger/decision closure mismatch.")
                if actual_exit is not None and (int(actual.exit_idx) != actual_exit
                                               or actual.exit_date != decision.actual_exit_date):
                    raise AssertionError("FILTER ledger/decision exit mismatch.")
                if actual.position != trade.position:
                    raise AssertionError("FILTER changes original trade direction.")
                keep = selection & ((global_indices <= actual_exit) if actual_exit is not None else True)
                expected_filter_held[keep] = trade.position
            else:
                keep = np.zeros(len(dates), dtype=bool)
                if actual_exit is not None:
                    raise AssertionError("Rejected episode cannot have an actual exit.")
            action_masks = {"entry_rejected": selection if not accepted else np.zeros(len(dates), bool),
                            "early_exit": selection & ~keep if accepted else np.zeros(len(dates), bool),
                            "retained": keep}
            for action, mask in action_masks.items():
                if action == "early_exit" and mask.any() and not bool(decision.early_exit):
                    raise AssertionError("Removed tail is missing its early-exit decision.")
                for split in ("IS", "OOS"):
                    chosen = mask & base.split.eq(split).to_numpy()
                    if not chosen.any():
                        continue
                    baseline_pp, filtered_pp = float(bret[chosen].sum() * 100), float(fret[chosen].sum() * 100)
                    delta = filtered_pp - baseline_pp
                    if action == "retained":
                        check_close(delta, 0., "Retained-return difference")
                        removed = 0.
                    else:
                        check_close(filtered_pp, 0., "Removed segment FILTER return")
                        removed = baseline_pp
                    first_split = dates[base.split.eq(split).to_numpy()][0]
                    decision_date = decision.signal_date if action == "entry_rejected" else decision.risk_signal_date
                    detail.append({**identity, "split": split, "action": action,
                        "original_episode": int(decision.original_episode), "base_trade_id": int(trade.trade_id),
                        "entry_date": trade.entry_date, "original_exit_date": trade.exit_date,
                        "original_closed": bool(trade.closed), "actual_exit_date": decision.actual_exit_date,
                        "segment_start_date": dates[chosen][0], "segment_end_date": dates[chosen][-1],
                        "return_day_count": int(chosen.sum()), "baseline_return_pp": baseline_pp,
                        "filtered_return_pp": filtered_pp, "net_delta_pp": delta,
                        "removed_losing_contribution_pp": max(-removed, 0.),
                        "foregone_profitable_contribution_pp": max(removed, 0.),
                        "avoided_losing_segment_count": int(action != "retained" and removed < -TOL),
                        "foregone_profitable_segment_count": int(action != "retained" and removed > TOL),
                        "near_zero_removed_segment_count": int(action != "retained" and abs(removed) <= TOL),
                        "carry_in": bool(trade.entry_date < first_split),
                        "decision_before_split": bool(pd.notna(decision_date) and decision_date < first_split)})
        if not np.array_equal(expected_filter_held, filt.held.to_numpy(int)):
            raise AssertionError(f"Replayed decision holding intervals differ from saved FILTER: {cid}")
        if np.any((base.held.to_numpy() != 0) & (owner < 0)):
            raise AssertionError("Some active BASE returns have no original episode owner.")
        if np.any((owner < 0) & ((np.abs(bret) > 1e-12) | (np.abs(fret) > 1e-12))):
            raise AssertionError("Nonzero return outside original episodes.")
        audit.append({"candidate_id": cid, "original_episodes": len(trades),
                      "accepted_episodes": len(actual_trades), "calendar_observations": len(dates),
                      "index_offset": offset, "holding_and_join_checks_pass": True})
    details = pd.DataFrame(detail)
    sums = ["return_day_count", "baseline_return_pp", "filtered_return_pp", "net_delta_pp",
            "removed_losing_contribution_pp", "foregone_profitable_contribution_pp",
            "avoided_losing_segment_count", "foregone_profitable_segment_count", "near_zero_removed_segment_count"]
    rows, errors = [], []
    for cid, group in details.groupby("candidate_id", sort=False):
        identity = {name: group[name].iloc[0] for name in IDENTITY}
        for split in ("IS", "OOS"):
            for action in ACTIONS:
                selected = group.loc[group.split.eq(split) & group.action.eq(action)]
                rows.append({**identity, "split": split, "action": action,
                             **{name: selected[name].sum() for name in sums},
                             "original_episode_count": selected.original_episode.nunique(),
                             "carry_in_episode_count": selected.loc[selected.carry_in, "original_episode"].nunique(),
                             "decision_before_split_count": int(selected.decision_before_split.sum())})
            selected = group.loc[group.split.eq(split)]
            observed = daily.loc[daily.candidate_id.eq(cid) & daily.split.eq(split)]
            main = metrics.loc[metrics.candidate_id.eq(cid) & metrics.split.eq(split)].set_index("variant")
            for variant, column in (("BASE", "baseline_return_pp"), ("FILTER", "filtered_return_pp")):
                source = observed.loc[observed.variant.eq(variant)]
                errors.append(check_close(len(source), main.loc[variant, "observations"], "Main observation count"))
                errors.append(check_close(selected[column].sum(), source.net_daily_return.sum() * 100, "Daily total"))
                errors.append(check_close(selected[column].sum(), main.loc[variant, "net_simple_return_pct"], "Main metric total"))
            delta = main.loc["FILTER", "net_simple_return_pct"] - main.loc["BASE", "net_simple_return_pct"]
            errors.append(check_close(selected.net_delta_pp.sum(), delta, "Split delta conservation"))
            errors.append(check_close(selected.removed_losing_contribution_pp.sum()
                                      - selected.foregone_profitable_contribution_pp.sum(), delta, "Removal benefit identity"))
    return pd.DataFrame(rows), details, {"pass": True, "reconciliation_checks": len(errors),
        "maximum_absolute_error_pp": max(errors, default=0.), "candidate_checks": audit,
        "cross_split_original_episode_count": int((details.groupby(["candidate_id", "original_episode"]).split.nunique() > 1).sum())}


QUESTIONS = {
    "MOM_NO_PROGRESS_5": "仅用 IS 非重叠时间块检查早退后尾段的避免亏损和放弃盈利，判断改善是否依赖少数行情，且保留恢复盈利的反例。",
    "VOL_NOT_EXPANDING": "仅用 IS 分期比较被拒与保留交易的收益和样本量，区分降低损失与持续产生正收益。",
    "ENTRY_NOT_EXTENDED": "仅用 IS 检查延伸状态与后续收益的关系是否跨时间块稳定，并核查保留交易的绝对收益水平。",
    "VOL_TIME_20": "仅用 IS 分开核算入场拒绝和超时退出的贡献，检验组合改善是否主要来自其中一个机制及其时间稳定性。",
    "MOM_ALIGNED_63": "仅用 IS 审查被拒反向信号中后来形成大盈利的完整反例，判断较长趋势确认是否错过新趋势开端。",
    "VOL_TREND_CONFLICT": "仅用 IS 检查波动与趋势条件拒绝交易的重合情况和分期贡献，避免把组合改善视为两个独立证据。",
}


def diagnostics(summary, details, assessment, metrics, late, validation, hashes):
    cases = []
    for row in assessment.sort_values("freeze_rank").itertuples(index=False):
        case = {name: getattr(row, name) for name in IDENTITY}
        case.update({"freeze_rank": int(row.freeze_rank), "failed_oos_criteria": str(row.failed_criteria).split("|"),
                     "facts_zh": [], "next_is_only_questions": [QUESTIONS[row.filter_id]], "splits": {}})
        for split in ("IS", "OOS"):
            selected = summary.loc[summary.candidate_id.eq(row.candidate_id) & summary.split.eq(split)]
            main = metrics.loc[metrics.candidate_id.eq(row.candidate_id) & metrics.split.eq(split)].set_index("variant")
            values = {name: float(selected[name].sum()) for name in ("baseline_return_pp", "filtered_return_pp", "net_delta_pp",
                        "removed_losing_contribution_pp", "foregone_profitable_contribution_pp")}
            values.update({name: int(selected[name].sum()) for name in
                           ("avoided_losing_segment_count", "foregone_profitable_segment_count", "near_zero_removed_segment_count")})
            values["filter_annualized_simple_pct"] = float(main.loc["FILTER", "net_annualized_simple_pct"])
            values["filter_closed_trades"] = int(main.loc["FILTER", "closed_n"])
            values["actions"] = selected.to_dict("records")
            case["splits"][split] = values
            case["facts_zh"].append(
                f"{split}：BASE {values['baseline_return_pp']:.2f}%，FILTER {values['filtered_return_pp']:.2f}%，净改善 {values['net_delta_pp']:+.2f} 个百分点；"
                f"删除收益段避免亏损 {values['removed_losing_contribution_pp']:.2f} 个百分点"
                f"（{values['avoided_losing_segment_count']} 段），同时放弃盈利 {values['foregone_profitable_contribution_pp']:.2f} 个百分点"
                f"（{values['foregone_profitable_segment_count']} 段）。")
        out = case["splits"]["OOS"]
        if out["net_delta_pp"] < -TOL:
            case["facts_zh"].append("OOS 放弃的盈利超过避免的亏损，过滤相对 BASE 的净贡献为负；这与绝对收益是否为正是两个问题。")
        else:
            case["facts_zh"].append("OOS 过滤仍改善 BASE，但相对改善没有使该账户通过全部预先规定的绝对收益、回撤及年度稳定性条件。")
        own = details.loc[details.candidate_id.eq(row.candidate_id) & details.split.eq("OOS")]
        carry_delta = float(own.loc[own.carry_in, "net_delta_pp"].sum())
        case["oos_carry_in_delta_pp"] = carry_delta
        case["facts_zh"].append(f"OOS 由边界前已入场交易贡献的过滤净差额为 {carry_delta:+.2f} 个百分点，按 OOS 实际收益日计入。")
        recent = late.loc[late.candidate_id.eq(row.candidate_id)]
        if len(recent):
            value = float(recent.iloc[0].net_annualized_simple_pct)
            case["is_late_annualized_simple_pct"] = value
            case["facts_zh"].append(f"已保存的 IS_LATE 段单利年化为 {value:.2f}%；全 IS 达标不代表临近留出期的绝对收益同样强。")
        cases.append(case)
    return {"schema_version": 1, "purpose": "冻结候选的后见失效诊断；不选择、替换候选或更改规则及阈值。",
        "definitions": {
            "primary_accounting": "零成本固定初始本金单利；收益贡献单位为初始本金百分点。",
            "attribution_unit": "原 BASE episode × split × action 的实际收益段净和；不以完整交易最终输赢标签分类跨界收益。",
            "entry_rejected": "原交易被拒绝后原持仓区间的收益；首个收益日在原成交日之后。",
            "early_exit": "实际提前退出日之后、原技术退出日及之前的收益；实际退出日收益仍属于 retained。",
            "retained": "保留的原持仓收益，包括被提前退出交易在退出日及以前的实际收益。",
            "removed_losing_contribution_pp": "被删收益段净和为负时的绝对值；该段内正负日先相加。",
            "foregone_profitable_contribution_pp": "被删收益段净和为正时的金额；该段内正负日先相加。",
            "counts": "action 内 original_episode_count 去重；同一原交易可先 retained 后 early_exit，跨 action 不可直接相加为交易总数。",
            "open_trades": "期末仍开放的原交易按已保存的收益日归属，不补强制退出。"},
        "validation": validation, "source_sha256": hashes,
        "cases": cases,
        "next_is_only_questions": [
            "首先仅用 IS 滚动或非重叠分段检查绝对收益与过滤贡献稳定性；六个冻结账户中五个已保存 IS_LATE 段年化低于 1%，不能把全 IS 累计改善等同近期可持续收益。",
            "分开审查避免亏损与放弃盈利、入场拒绝与退出尾段，并保留跨界持仓及相反案例；只提出机制问题，不根据本 OOS 搜索阈值。",
            "本 OOS 已经查看，后续修改后的独立确认需要新的未查看数据；本归因不能把本次失败转化成新的样本外成功。"],
        "frozen_decisions_or_parameters_changed": False}


def self_test():
    """One cross-split early exit and one rejected open reversal, hand calculated."""
    dates = pd.bdate_range("2023-05-24", periods=10)
    common = {"candidate_id": "TEST", "pair": "EUR", "strategy": "MA_20_60", "filter_id": "TEST"}
    returns = np.array([0., .01, -.02, -.03, .04, -.01, -.02, .01, -.03, .05])
    held = np.array([0, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    fheld = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    frames = []
    for variant, holdings, values in (("BASE", held, returns), ("FILTER", fheld, np.r_[returns[:3], np.zeros(7)])):
        frames.append(pd.DataFrame({**common, "variant": variant, "date": dates,
            "split": ["IS"] * 4 + ["OOS"] * 6, "held": holdings,
            "gross_daily_return": values, "net_daily_return": values}))
    daily = pd.concat(frames, ignore_index=True)
    base0 = {**common, "variant": "BASE", "trade_id": 1, "signal_idx": 99,
             "entry_idx": 100, "exit_idx": 105, "entry_date": dates[0], "exit_date": dates[5], "position": 1, "closed": True}
    base1 = {**common, "variant": "BASE", "trade_id": 2, "signal_idx": 104,
             "entry_idx": 105, "exit_idx": 109, "entry_date": dates[5], "exit_date": dates[9], "position": -1, "closed": False}
    actual = {**base0, "variant": "FILTER", "exit_idx": 102, "exit_date": dates[2]}
    ledger = pd.DataFrame([base0, base1, actual])
    decisions = pd.DataFrame([
        {**common, "original_episode": 0, "entry_idx": 100, "signal_idx": 99, "entry_date": dates[0],
         "position": 1, "accepted": True, "original_open": False, "original_exit_idx": 105,
         "original_exit_date": dates[5], "actual_exit_idx": 102, "actual_exit_date": dates[2],
         "early_exit": True, "signal_date": dates[0] - pd.offsets.BDay(), "risk_signal_date": dates[1]},
        {**common, "original_episode": 1, "entry_idx": 105, "signal_idx": 104, "entry_date": dates[5],
         "position": -1, "accepted": False, "original_open": True, "original_exit_idx": np.nan,
         "original_exit_date": pd.NaT, "actual_exit_idx": np.nan, "actual_exit_date": pd.NaT,
         "early_exit": False, "signal_date": dates[4], "risk_signal_date": pd.NaT}])
    metrics = daily.groupby(["candidate_id", "variant", "split"], as_index=False).agg(
        net_simple_return_pct=("net_daily_return", lambda x: x.sum() * 100), observations=("date", "size"))
    summary, detail, validation = build_attribution(daily, ledger, decisions, metrics)
    check_close(summary.loc[summary.split.eq("IS"), "net_delta_pp"].sum(), 3., "IS hand delta")
    check_close(summary.loc[summary.split.eq("OOS"), "net_delta_pp"].sum(), -4., "OOS hand delta")
    retained = detail.loc[detail.action.eq("retained")].iloc[0]
    assert retained.segment_end_date == dates[2] and retained.return_day_count == 2
    assert detail.loc[detail.action.eq("early_exit"), "segment_start_date"].min() == dates[3]
    assert detail.loc[detail.action.eq("entry_rejected"), "segment_start_date"].min() == dates[6]
    check_close(detail.loc[detail.split.eq("OOS"), "foregone_profitable_contribution_pp"].sum(), 4., "Cross-split sign attribution")
    assert validation["cross_split_original_episode_count"] == 1
    try:
        build_attribution(daily, ledger, pd.concat([decisions, decisions.iloc[:1]]), metrics)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate decision join was not rejected.")
    return {"pass": True, "checks": ["unique joins", "cross-split signed contributions", "exit-day return retained",
            "reversal-day old-position ownership", "open rejected trade not liquidated", "main-metric conservation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "reproduced/filter_oos_v1")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    tests = self_test()
    if args.self_test_only:
        print(json.dumps(tests))
        return
    directory = args.directory.resolve()
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in INPUTS}
    daily = pd.concat([read_table(directory / name) for name in INPUTS[:2]], ignore_index=True)
    ledger = read_table(directory / "frozen_full_trade_ledger.csv.gz")
    decisions = read_table(directory / "frozen_full_episode_decisions.csv.gz")
    metrics = pd.concat([read_table(directory / name) for name in ("is_frozen_metrics.csv", "oos_metrics.csv")], ignore_index=True)
    assessment = read_table(directory / "candidate_assessment.csv")
    if len(assessment) != 6 or assessment.candidate_id.duplicated().any() or set(assessment.candidate_id) != set(daily.candidate_id):
        raise AssertionError("Expected the six saved frozen candidates, without substitutions.")
    summary, detail, validation = build_attribution(daily, ledger, decisions, metrics)
    late = pd.read_csv(directory / "is_all_metrics.csv", usecols=["candidate_id", "variant", "split", "net_annualized_simple_pct"])
    late = late.loc[late.candidate_id.isin(assessment.candidate_id) & late.variant.eq("FILTER") & late.split.eq("IS_LATE")]
    validation["self_test"] = tests
    validation["summary_rows"] = len(summary)
    validation["episode_segment_rows"] = len(detail)
    result = diagnostics(summary, detail, assessment, metrics, late, validation, hashes)
    if hashes != {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in INPUTS}:
        raise AssertionError("An input source changed during attribution; nothing was written.")
    summary.to_csv(directory / "failure_attribution.csv", index=False)
    detail.to_csv(directory / "failure_episode_attribution.csv", index=False)
    (directory / "failure_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"pass": True, "candidates": len(assessment), **validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
