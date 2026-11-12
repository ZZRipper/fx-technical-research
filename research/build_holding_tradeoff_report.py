"""Presentation of the authorized expectancy/profit-factor holding campaign.

Only completed, saved campaign artifacts are read. This renderer does not run
strategies, select candidates, change gates, or open prior-campaign OOS results.
The old IS-only discussion remains immutable; the new protocol records the
user-authorized change in quality criteria. Every frozen candidate is plotted
in saved IS order, including every subsequent failure.
"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from datetime import datetime, timezone
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "reproduced/holding_tradeoff_v8"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reproduced/holding_tradeoff_report_cache"))
import numpy as np
import pandas as pd
from fx_filter_report import _CSS


GATES = {
    "positive_and_annualized_pass": "正收益且简单年化至少 1%",
    "return_improvement_pass": "累计收益改善至少 5 pp",
    "expectancy_pass": "每笔期望值严格优于 BASE",
    "profit_factor_pass": "毛利润因子严格优于 BASE",
    "drawdown_pass": "最大回撤恶化不超过 2 pp",
    "return_to_drawdown_pass": "收益／回撤比改善",
    "sample_pass": "至少 50 笔合格闭合且保留至少 50%",
    "block_positive_pass": "至少两段正收益且最后一段为正",
    "block_improvement_pass": "至少两段优于 BASE 且最后一段改善",
}
LABELS = {
    "freeze_rank": "IS冻结顺序", "candidate_id": "候选版本", "source_campaign": "原研究",
    "pair": "货币", "strategy": "原规则", "filter_name": "退出规则", "split": "区间", "variant": "账户",
    "gross_simple_return_pct": "单利收益%", "gross_annualized_simple_pct": "简单年化%",
    "gross_max_drawdown_pp": "最大回撤pp", "closed_n": "合格闭合数", "win_rate": "胜率%",
    "payoff_ratio": "盈亏比", "expectancy_pct": "每笔期望值%", "gross_profit_factor": "毛利润因子",
    "gross_profit_sum_pct": "盈利合计pp", "gross_loss_abs_sum_pct": "亏损绝对值合计pp",
    "retention": "闭合保留比例%", "carry_in_n": "跨界持仓数", "censored_entry_n": "期内开仓未闭合数",
    "original_standard_pass": "旧IS标准通过", "original_failed_criteria": "旧IS未通过项目",
    "original_win_rate_pass": "旧胜率门槛", "original_payoff_pass": "旧盈亏比门槛",
    "pass": "新IS标准通过", "frozen": "本轮冻结", "failed_criteria": "新IS未通过项目",
    "is_gross_return_pct": "IS单利收益%", "gross_return_improvement_pp": "IS收益改善pp",
    "is_annualized_simple_pct": "IS简单年化%", "is_max_drawdown_pp": "IS最大回撤pp",
    "is_closed_n": "IS合格闭合数", "win_rate_improvement_pp": "IS胜率改善pp",
    "payoff_relative_improvement": "IS盈亏比相对改善%",
    "base_is_expectancy_pct": "BASE每笔期望值%", "filter_is_expectancy_pct": "FILTER每笔期望值%",
    "base_is_gross_profit_factor": "BASE毛利润因子", "filter_is_gross_profit_factor": "FILTER毛利润因子",
    "expectancy_improvement_pp_per_trade": "每笔期望改善pp", "profit_factor_improvement": "毛利润因子改善",
    "expectancy_pass": "新期望值门槛", "profit_factor_pass": "新毛利润因子门槛",
    "oos_curve_pass": "OOS曲线通过", "oos_quality_confirmed": "OOS新质量确认",
    "original_oos_quality_confirmed": "OOS旧质量诊断", "oos_failed_criteria": "OOS曲线未通过项",
    "quality_failed_criteria": "OOS新质量未通过项", "oos_return_pct": "OOS单利收益%",
    "oos_base_return_pct": "OOS BASE收益%", "oos_improvement_pp": "OOS收益改善pp",
    "oos_annualized_simple_pct": "OOS简单年化%", "oos_max_drawdown_pp": "OOS最大回撤pp",
    "oos_closed_n": "OOS合格闭合数", "oos_positive_years": "正收益周年数",
    "oos_base_expectancy_pct": "OOS BASE每笔期望值%", "oos_expectancy_pct": "OOS FILTER每笔期望值%",
    "oos_base_gross_profit_factor": "OOS BASE毛利润因子", "oos_gross_profit_factor": "OOS FILTER毛利润因子",
    "oos_base_win_rate": "OOS BASE胜率%", "oos_win_rate": "OOS FILTER胜率%",
    "oos_base_payoff_ratio": "OOS BASE盈亏比", "oos_payoff_ratio": "OOS FILTER盈亏比",
}
METRICS = ["split", "variant", "gross_simple_return_pct", "gross_annualized_simple_pct",
           "gross_max_drawdown_pp", "closed_n", "expectancy_pct", "gross_profit_factor",
           "win_rate", "payoff_ratio", "retention", "carry_in_n", "censored_entry_n"]
PERCENT_FRACTIONS = {"win_rate", "oos_win_rate", "oos_base_win_rate", "retention", "payoff_relative_improvement"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def scalar(value, name=""):
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if name in PERCENT_FRACTIONS:
            value *= 100
        precision = 4 if "expectancy" in name or "factor" in name or "annualized" in name else 2
        return f"{value:,.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if name in {"failed_criteria", "original_failed_criteria", "oos_failed_criteria", "quality_failed_criteria"}:
        labels = {**GATES, "win_rate_pass": "胜率改善至少2pp", "payoff_pass": "盈亏比相对改善至少10%"}
        if name == "oos_failed_criteria":
            labels = {"annualized_pass": "简单年化至少1%", "baseline_improvement_pass": "收益不低于BASE",
                      "drawdown_pass": "最大回撤不低于−15pp", "positive_years_pass": "至少2个正收益周年", "sample_pass": "至少10笔合格闭合"}
        elif name == "quality_failed_criteria":
            labels = {"quality_sample_pass": "至少20笔合格闭合", "quality_finiteness_pass": "两账户期望值和毛利润因子均有定义",
                      "expectancy_pass": "每笔期望值严格改善", "profit_factor_pass": "毛利润因子严格改善"}
        return "；".join(labels.get(part, part) for part in str(value).split("|") if part) or "—"
    return str(value)


def table(frame, columns=None):
    if frame.empty:
        return '<p class="muted">无记录。</p>'
    columns = [c for c in (columns or list(frame)) if c in frame]
    head = "".join("<th>" + escape(LABELS.get(c, c)) + "</th>" for c in columns)
    rows = ["<tr>" + "".join("<td>" + escape(scalar(v, c)) + "</td>" for c, v in zip(columns, row)) + "</tr>"
            for row in frame[columns].itertuples(index=False, name=None)]
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


class Inputs:
    def __init__(self, directory):
        self.directory, self.hashes = directory, {}

    def path(self, name):
        path = self.directory / name
        if not path.is_file():
            raise ValueError("Missing completed report input: " + str(name))
        self.hashes[str(name)] = sha(path)
        return path

    def csv(self, name):
        return pd.read_csv(self.path(name), float_precision="round_trip")

    def json(self, name):
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def verify(self, manifest):
        for name, digest in manifest.items():
            if sha(self.path(name)) != digest:
                raise ValueError("Saved result hash differs: " + name)

    def unchanged(self):
        if any(sha(self.directory / name) != digest for name, digest in self.hashes.items()):
            raise ValueError("Report inputs changed during rendering; rerun a consistent snapshot")


def ordered_metrics(metrics, candidate_id):
    frame = metrics.loc[metrics.candidate_id.eq(candidate_id)].copy()
    frame["_split"] = frame.split.map({"IS": 0, "OOS": 1})
    frame["_variant"] = frame.variant.map({"BASE": 0, "FILTER": 1})
    return frame.sort_values(["_split", "_variant"]).drop(columns=["_split", "_variant"])


def rule_text(spec):
    conditions = {
        "pnl_negative": "已赚累计单利收益为负", "pnl_positive": "已赚累计单利收益为正",
        "giveback_half": "历史已赚峰值为正且当前累计收益已回吐至少一半",
        "recent5_negative": "最近5个报价的持仓方向收益合计为负",
        "opposite20": "当前价格过去20报价的变化与原持仓方向相反",
    }
    age = int(spec["checkpoint_age"])
    condition = spec.get("condition", "ALL")
    if condition == "ALL":
        return f"原交易赚取恰好 {age} 个报价收益后，在该收盘发出退出信号；下一报价收盘退出。"
    names = condition if isinstance(condition, list) else condition.split("&")
    return f"原交易赚取恰好 {age} 个报价收益后，仅检查一次：" + "，且".join(conditions.get(n, n) for n in names) + "；满足则下一报价收盘退出。"


def plot_candidate(output, row, daily, metrics, pdf, capital, oos_completed):
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9), gridspec_kw={"height_ratios": [1.65, 1]})
    rank, cid = int(row["freeze_rank"]), row["candidate_id"]
    fig.suptitle(f'Frozen IS rank {rank}  |  {row["pair"]} / {row["strategy"]}', fontsize=17, fontweight="bold", y=.98)
    fig.text(.065, .929, str(row["filter_name"]), fontsize=10)
    fig.text(.065, .906, f'Fixed USD {capital:,.0f}; zero cost; expectancy AND gross profit factor are the current quality criteria.', fontsize=9, color="#364f59")
    checks = []
    for col, split in enumerate(("IS", "OOS")):
        top, bottom = axes[:, col]
        if split == "OOS" and not oos_completed:
            for ax in (top, bottom):
                ax.set_axis_off()
            top.text(.5, .5, "OOS has not been evaluated", ha="center", va="center", transform=top.transAxes)
            continue
        for variant, color in (("BASE", "#738593"), ("FILTER", "#087f8c")):
            part = daily.loc[daily.split.eq(split) & daily.variant.eq(variant)].sort_values("date")
            match = metrics.loc[metrics.split.eq(split) & metrics.variant.eq(variant)]
            if len(match) != 1 or part.empty or part.date.duplicated().any():
                raise ValueError("Missing unique daily/metric plot group: " + cid + ":" + split + ":" + variant)
            stat = match.iloc[0]
            dates = pd.DatetimeIndex(pd.to_datetime(part.date))
            values = part.gross_daily_return.to_numpy(float)
            if not np.isfinite(values).all() or not np.array_equal(values, part.net_daily_return.to_numpy(float)):
                raise ValueError("Nonfinite or non-zero-cost saved daily returns")
            cumulative = np.r_[0., np.cumsum(values) * 100]
            drawdown = cumulative - np.maximum.accumulate(cumulative)
            if not np.allclose([cumulative[-1], drawdown.min()], [stat.gross_simple_return_pct, stat.gross_max_drawdown_pp], atol=1e-8, rtol=1e-10):
                raise ValueError("Plotted endpoint or drawdown disagrees with saved metrics: " + cid)
            anchor = pd.Timestamp(stat.anchor_date)
            if not anchor < dates[0]:
                raise ValueError("Chart anchor must precede its first earned quote")
            chart_dates = dates.insert(0, anchor)
            top.plot(chart_dates, cumulative * capital / 100, color=color, lw=1.3, label=f"{variant}: {cumulative[-1]:+.2f}%")
            bottom.plot(chart_dates, drawdown, color=color, lw=1.15, label=f"{variant}: {drawdown.min():.2f} pp")
            checks.append({"candidate_id": cid, "split": split, "variant": variant,
                           "endpoint_pct": float(cumulative[-1]), "max_drawdown_pp": float(drawdown.min()), "matches_saved": True})
        top.set_title(split + " | continuous position; displayed PnL resets", fontsize=10)
        top.set_ylabel("Cumulative fixed-capital PnL (USD)")
        bottom.set_ylabel("Drawdown (initial-capital pp)")
        for ax in (top, bottom):
            ax.axhline(0, color="#b3c0c7", lw=.7)
            ax.grid(alpha=.16)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="best", fontsize=8)
            if split == "OOS":
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            else:
                locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    grid = []
    for stat in metrics.itertuples(index=False):
        grid.append([str(stat.split), str(stat.variant), f"{stat.gross_simple_return_pct:.2f}", f"{stat.gross_annualized_simple_pct:.4f}",
                     f"{stat.gross_max_drawdown_pp:.2f}", str(int(stat.closed_n)), scalar(stat.expectancy_pct, "expectancy_pct"),
                     scalar(stat.gross_profit_factor, "gross_profit_factor"), scalar(stat.win_rate, "win_rate"), scalar(stat.payoff_ratio)])
    table_ax = fig.add_axes([.06, .071, .88, .13]); table_ax.set_axis_off()
    tab = table_ax.table(cellText=grid, colLabels=["Period", "Account", "Return %", "Ann. %", "DD pp", "Closed", "Exp. %/trade", "Gross PF", "Win %", "Payoff"],
                        loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.4)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#d5dfe4")
        if r == 0:
            cell.set_facecolor("#eaf0f3")
    fig.text(.065, .045, "Checkpoint age counts earned quote returns; signal at checkpoint close, fill at next close, including that execution-day return.", fontsize=8, color="#4b606b")
    fig.text(.065, .027, "Every frozen candidate is retained in IS order. Win rate/payoff remain diagnostics. Reused historical OOS is not independent new evidence.", fontsize=8, color="#4b606b")
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.265, wspace=.25, hspace=.31)
    name = f"tradeoff_figures/frozen_{rank:04d}.png"
    fig.savefig(output / name, dpi=145, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    return name, checks


def validate_artifacts(output, report, atlas, figures, expected_pages):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.ids, self.links = set(), []

        def handle_starttag(self, tag, attributes):
            attributes = dict(attributes)
            if "id" in attributes:
                self.ids.add(attributes["id"])
            self.links.extend(attributes[k] for k in ("href", "src") if k in attributes)
    parser = Links(); contents = (output / report).read_text(encoding="utf-8")
    if "\ufffd" in contents or "</html>" not in contents:
        raise ValueError("Invalid report HTML encoding or structure")
    parser.feed(contents)
    for link in parser.links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc:
            continue
        path = unquote(parts.path) or report
        if path != "holding_tradeoff_report_build.json" and not (output / path).is_file():
            raise ValueError("Missing local report link: " + link)
        if path == report and parts.fragment and parts.fragment not in parser.ids:
            raise ValueError("Missing local report anchor: " + link)
    from PIL import Image
    for name in figures:
        with Image.open(output / name) as pic:
            if pic.format != "PNG" or min(pic.size) < 400:
                raise ValueError("Invalid report PNG: " + name)
            pic.verify()
    data = (output / atlas).read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-100:] or pages != max(1, expected_pages):
        raise ValueError("Atlas page count or PDF structure differs from complete frozen set")
    return {"status": "passed", "local_links_checked": len(parser.links), "png_files": len(figures), "pdf_pages": pages}


def build_report(output_dir=DEFAULT):
    output = Path(output_dir).resolve(); inputs = Inputs(output)
    development, protocol = inputs.json("development_status.json"), inputs.json("protocol.json")
    if development.get("status") != "passed" or development.get("is_screen_completed") is not True:
        raise ValueError("Wait for completed IS development before rendering")
    if protocol.get("version") != "holding_tradeoff_v8" or protocol.get("cost_bp") != 0:
        raise ValueError("Only the authorized zero-cost holding_tradeoff_v8 campaign is supported")
    freeze = inputs.json("frozen_selection.json")
    if freeze["protocol_sha256"] != inputs.hashes["protocol.json"] or development["freeze_sha256"] != inputs.hashes["frozen_selection.json"]:
        raise ValueError("Protocol/freeze/development chain differs")
    inputs.verify(freeze["is_output_sha256"])
    selection, metrics = inputs.csv("is_selection.csv"), inputs.csv("is_all_metrics.csv")
    original_selection = inputs.csv("is_original_selection.csv")
    hypotheses = inputs.json("is_hypotheses.json")["candidates"]
    members = {row["candidate_id"]: row for row in hypotheses}
    coverage = inputs.csv("is_account_coverage.csv")
    review_name = "../holding_quality_tradeoff_review/review.json"
    review, reviewed = inputs.json(review_name), inputs.csv("../holding_quality_tradeoff_review/review.csv")
    if review.get("OOS_not_evaluated_for_review") is not True or len(reviewed) != 492:
        raise ValueError("Expected original IS-only 492-version discussion record")
    if review["review_csv_sha256"] != inputs.hashes["../holding_quality_tradeoff_review/review.csv"]:
        raise ValueError("The original review CSV differs from its saved manifest")
    for name in ("review.json", "review.csv"):
        expected = protocol["input_sha256"]["reproduced/holding_quality_tradeoff_review/" + name]
        if inputs.hashes["../holding_quality_tradeoff_review/" + name] != expected:
            raise ValueError("The new protocol refers to a different historical discussion")
    if len(selection) != 492 or len(members) != 492 or set(selection.candidate_id) != set(members) or set(members) != set(reviewed.candidate_id):
        raise ValueError("Report must retain exactly all 492 saved original versions")
    source_counts = selection.source_campaign.value_counts().to_dict()
    if source_counts != {"holding_filter_v3": 391, "timecap_filter_v4": 101}:
        raise ValueError("Original source counts differ")
    if len(original_selection) != 492 or original_selection.candidate_id.duplicated().any() or set(original_selection.candidate_id) != set(members):
        raise ValueError("The saved original-standard table must retain every fixed version")
    old_index = original_selection.set_index("candidate_id").reindex(selection.candidate_id)
    for original, current in (("pass", "original_standard_pass"), ("win_rate_pass", "original_win_rate_pass"),
                              ("payoff_pass", "original_payoff_pass"), ("failed_criteria", "original_failed_criteria")):
        if not old_index[original].fillna("").reset_index(drop=True).equals(selection[current].fillna("").reset_index(drop=True)):
            raise ValueError("Displayed old-standard diagnostic differs from original saved selection: " + current)
    for campaign in source_counts:
        old = inputs.json("../" + campaign + "/is_hypotheses.json")["candidates"]
        for row in old:
            if row["candidate_id"] not in members or members[row["candidate_id"]]["spec"] != row["spec"]:
                raise ValueError("An original checkpoint/condition spec changed")
    if len(coverage) != 180 or coverage.duplicated(["pair", "strategy"]).any() or coverage.candidate_count.sum() != 492:
        raise ValueError("Expected all 180 accounts, including those without proposals")
    frozen = selection.loc[selection.frozen.eq(True)].sort_values("freeze_rank")
    frozen_ids = frozen.candidate_id.tolist()
    tradeoff_counts = {"win_rate_actually_lower": int(frozen.win_rate_improvement_pp.lt(0).sum()),
                       "payoff_actually_lower": int(frozen.payoff_relative_improvement.lt(0).sum()),
                       "old_win_rate_gate_failed": int(frozen.original_win_rate_pass.eq(False).sum()),
                       "old_payoff_gate_failed": int(frozen.original_payoff_pass.eq(False).sum())}
    if frozen_ids != [row["candidate_id"] for row in freeze["candidates"]]:
        raise ValueError("Frozen identities/order differ from the saved IS selection")
    daily = inputs.csv("is_frozen_daily.csv.gz")
    plot_metrics = inputs.csv("is_frozen_metrics.csv")
    oos_completed = False; assessment = pd.DataFrame(); anniversaries = pd.DataFrame(); fresh = pd.DataFrame()
    if (output / "oos_status.json").exists():
        status = inputs.json("oos_status.json")
        if status.get("status") == "passed":
            if status["freeze_sha256"] != inputs.hashes["frozen_selection.json"]:
                raise ValueError("Completed OOS refers to a different freeze")
            inputs.verify(status["output_sha256"])
            assessment = inputs.csv("candidate_assessment.csv").sort_values("freeze_rank")
            if assessment.candidate_id.tolist() != frozen_ids:
                raise ValueError("OOS must retain every frozen candidate in original IS order")
            daily = pd.concat([daily, inputs.csv("oos_daily.csv.gz")], ignore_index=True)
            plot_metrics = pd.concat([plot_metrics, inputs.csv("oos_metrics.csv")], ignore_index=True)
            anniversaries = inputs.csv("oos_anniversary_metrics.csv")
            fresh = inputs.csv("oos_fresh_start_metrics.csv")
            oos_completed = True
    if set(daily.candidate_id) != set(frozen_ids) or set(plot_metrics.candidate_id) != set(frozen_ids):
        raise ValueError("Figure inputs differ from the complete frozen set")
    expected_splits = {"IS", "OOS"} if oos_completed else {"IS"}
    if set(daily.split) != expected_splits or set(plot_metrics.split) != expected_splits:
        raise ValueError("Figure inputs contain an unexpected evaluation period")
    required = {"expectancy_pct", "gross_profit_factor", "gross_simple_return_pct", "gross_max_drawdown_pp"}
    if not required.issubset(plot_metrics):
        raise ValueError("Report metrics must include explicit saved expectancy and gross profit factor")
    audits = []
    for name in ("independent_validation.json", "post_oos_independent_validation.json"):
        if (output / name).exists() and (not name.startswith("post_") or oos_completed):
            audits.append((name, inputs.json(name)))

    failure_counts = pd.DataFrame([{"gate": name, "criterion": label, "failed_versions": int(selection[name].eq(False).sum()),
                                   "passed_versions": int(selection[name].eq(True).sum())} for name, label in GATES.items()])
    parallel_columns = ["candidate_id", "source_campaign", "pair", "strategy", "filter_name", "original_standard_pass",
                        "original_win_rate_pass", "original_payoff_pass", "original_failed_criteria", "pass", "frozen", "freeze_rank",
                        "base_is_expectancy_pct", "filter_is_expectancy_pct", "base_is_gross_profit_factor", "filter_is_gross_profit_factor",
                        "expectancy_improvement_pp_per_trade", "profit_factor_improvement", "gross_return_improvement_pp",
                        "win_rate_improvement_pp", "payoff_relative_improvement", "is_closed_n", "retention", "failed_criteria"]
    parallel = selection[parallel_columns].copy()
    failed = parallel.loc[parallel["pass"].eq(False)]
    parallel.to_csv(output / "report_all_492_criteria.csv", index=False)
    failed.to_csv(output / "report_IS_failures.csv", index=False)
    failure_counts.to_csv(output / "report_gate_failure_counts.csv", index=False)
    atlas, report = "holding_tradeoff_atlas.pdf", "holding_tradeoff_report.html"
    (output / "tradeoff_figures").mkdir(exist_ok=True)
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    figures, chart_checks = [], []
    capital = float(protocol["initial_capital_usd"])
    passing_ids = set(assessment.loc[assessment.oos_curve_pass & assessment.oos_quality_confirmed, "candidate_id"]) if oos_completed else set()
    passed_pdf_name = "holding_tradeoff_passed_summary.pdf" if passing_ids else None
    class PdfTargets:
        def __init__(self, *writers):
            self.writers = writers

        def savefig(self, *args, **kwargs):
            for writer in self.writers:
                writer.savefig(*args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": False}))
        pdf = stack.enter_context(PdfPages(output / atlas))
        passed_pdf = stack.enter_context(PdfPages(output / passed_pdf_name)) if passed_pdf_name else None
        if frozen.empty:
            fig = plt.figure(figsize=(12, 8)); fig.text(.1, .7, "No candidate was frozen under the authorized criteria.")
            pdf.savefig(fig); plt.close(fig)
        for row in frozen.to_dict("records"):
            cid = row["candidate_id"]
            writer = PdfTargets(pdf, passed_pdf) if cid in passing_ids else pdf
            name, checks = plot_candidate(output, row, daily.loc[daily.candidate_id.eq(cid)], ordered_metrics(plot_metrics, cid), writer, capital, oos_completed)
            figures.append(name); chart_checks.extend(checks)
    oos_summary = "OOS 尚未完成，本报告只展示已完成的 IS 与冻结记录。"
    display = pd.DataFrame(); oos_counts = None
    if oos_completed:
        curve = int(assessment.oos_curve_pass.sum()); quality = int(assessment.oos_quality_confirmed.sum())
        both = int((assessment.oos_curve_pass & assessment.oos_quality_confirmed).sum())
        oos_counts = {"curve": curve, "quality": quality, "both": both, "not_both": len(assessment) - both}
        oos_summary = f"全部 {len(frozen)} 个冻结候选完成本轮历史 OOS：曲线通过 {curve} 个，新质量确认 {quality} 个，同时通过 {both} 个。"
        if both == 0:
            oos_summary += " 本轮尚无候选达到两类要求。"
        display = assessment.rename(columns={"failed_criteria": "oos_failed_criteria"}).copy()
        display.to_csv(output / "report_OOS_complete_results.csv", index=False)
        display.loc[~(display.oos_curve_pass & display.oos_quality_confirmed)].to_csv(output / "report_OOS_failures.csv", index=False)
    cards, links = [], []
    for row, figure in zip(frozen.to_dict("records"), figures):
        cid, rank = row["candidate_id"], int(row["freeze_rank"])
        selected = ordered_metrics(plot_metrics, cid)
        explanation = rule_text(members[cid]["spec"])
        links.append(f'<li><a href="#candidate-{rank}">{escape(cid)}</a></li>')
        outcome_table = table(display.loc[display.candidate_id.eq(cid)], ["oos_curve_pass", "oos_quality_confirmed", "original_oos_quality_confirmed",
                              "oos_failed_criteria", "quality_failed_criteria"]) if oos_completed else ""
        cards.append(f'<article id="candidate-{rank}"><h3>冻结 #{rank} · {escape(row["pair"])} / {escape(row["strategy"])}</h3>'
                     + '<p><code>' + escape(cid) + '</code> · 来源：' + escape(str(row["source_campaign"])) + '</p><p>' + escape(explanation) + '</p>'
                     + '<img loading="lazy" style="width:100%;height:auto" src="' + figure + '" alt="冻结候选的IS及OOS累计单利PnL、回撤和交易质量">'
                     + table(selected, METRICS) + outcome_table + '</article>')
    comparison = pd.DataFrame([
        {"项目": "IS交易质量", "旧标准（保留诊断）": "胜率至少 +2 pp；盈亏比至少相对 +10%", "本轮用户授权标准": "每笔期望值与毛利润因子均有限且严格优于 BASE"},
        {"项目": "OOS质量确认", "旧标准（保留诊断）": "至少20笔；胜率及盈亏比均不劣于 BASE", "本轮用户授权标准": "至少20笔；每笔期望值和毛利润因子均有限且严格优于 BASE"},
        {"项目": "其他IS要求", "旧标准（保留诊断）": "正收益、年化、收益改善、回撤、收益/回撤、样本和三段一致性", "本轮用户授权标准": "原要求及数值门槛全部保留"},
        {"项目": "排序", "旧标准（保留诊断）": "收益改善↓、盈亏比绝对改善↓、胜率改善↓、candidate_id↑", "本轮用户授权标准": "排序不变；两轮合并后每账户最多冻结一条"},
        {"项目": "OOS曲线", "旧标准（保留诊断）": "年化≥1%、优于或等于BASE、回撤≥−15pp、至少2个正周年、≥10笔闭合", "本轮用户授权标准": "全部保留；曲线与质量分开报告"},
    ])
    audit_html = "".join('<p><a href="' + name + '">' + name + '</a>：' + escape(str(record.get("status"))) + '，'
                          + str(record.get("checks_passed", "—")) + '/' + str(record.get("checks_total", "—")) + '项。</p>' for name, record in audits)
    if not audit_html:
        audit_html = '<p>独立审计记录尚未保存；本报告的图表检查不替代独立数值审计。</p>'
    oos_columns = ["freeze_rank", "candidate_id", "oos_return_pct", "oos_base_return_pct", "oos_improvement_pp", "oos_annualized_simple_pct",
                   "oos_max_drawdown_pp", "oos_closed_n", "oos_positive_years", "oos_base_expectancy_pct", "oos_expectancy_pct",
                   "oos_base_gross_profit_factor", "oos_gross_profit_factor", "oos_base_win_rate", "oos_win_rate", "oos_base_payoff_ratio", "oos_payoff_ratio",
                   "oos_curve_pass", "oos_quality_confirmed", "original_oos_quality_confirmed", "oos_failed_criteria", "quality_failed_criteria"]
    oos_html = (table(display, oos_columns) + '<p><a href="report_OOS_complete_results.csv">全部冻结OOS结果CSV</a> · '
                + '<a href="report_OOS_failures.csv">所有曲线或新质量未通过结果CSV</a></p>') if oos_completed else '<p>没有读取未完成的OOS结果。</p>'
    focus_id = "AUD|DONCHIAN_BO_20|TIME_CAP_10"
    focus = frozen.loc[frozen.candidate_id.eq(focus_id)]
    focus_html = ""
    if len(focus) == 1:
        focus_rank = int(focus.iloc[0].freeze_rank)
        focus_result = ""
        if oos_completed:
            item = assessment.loc[assessment.candidate_id.eq(focus_id)].iloc[0]
            focus_result = (f"本轮OOS单利收益 {item.oos_return_pct:+.2f}%，BASE {item.oos_base_return_pct:+.2f}%，"
                            f"收益改善 {item.oos_improvement_pp:+.2f} pp；曲线通过：{scalar(item.oos_curve_pass)}，新质量确认：{scalar(item.oos_quality_confirmed)}。")
        focus_html = (f'<p class="note"><b>用户指定关注：<a href="#candidate-{focus_rank}">AUD / DONCHIAN_BO_20 / TIME_CAP_10（冻结 #{focus_rank}）</a></b>。'
                      + escape(focus_result) + '累计赚取10个报价收益后，在该收盘发退出信号，下一报价收盘执行且仍承担执行日收益；这不是10个自然日持仓上限。排序未改变。</p>')
    passed_html = ""
    if passing_ids:
        passed_rows = assessment.loc[assessment.candidate_id.isin(passing_ids)].sort_values("freeze_rank")
        brief, rules, last_negative = [], [], []
        for item in passed_rows.itertuples(index=False):
            is_stat = plot_metrics.loc[plot_metrics.candidate_id.eq(item.candidate_id) & plot_metrics.split.eq("IS") & plot_metrics.variant.eq("FILTER")].iloc[0]
            years = anniversaries.loc[anniversaries.candidate_id.eq(item.candidate_id) & anniversaries.variant.eq("FILTER")].sort_values("start_date")
            if len(years) != 3:
                raise ValueError("Passed summary requires all three saved anniversary windows")
            record = {"freeze_rank": int(item.freeze_rank), "candidate_id": item.candidate_id,
                      "IS最大回撤pp": float(is_stat.gross_max_drawdown_pp), "OOS年化%": f"{item.oos_annualized_simple_pct:.4f}",
                      "oos_return_pct": item.oos_return_pct, "oos_max_drawdown_pp": item.oos_max_drawdown_pp,
                      "oos_closed_n": item.oos_closed_n, "oos_positive_years": item.oos_positive_years}
            for i, year in enumerate(years.itertuples(index=False), 1):
                record[f"第{i}周年收益%"] = f"{year.gross_simple_return_pct:+.3f}"
            brief.append(record)
            if years.iloc[-1].gross_simple_return_pct < 0:
                last_negative.append(f"{item.pair} / {item.strategy} {years.iloc[-1].gross_simple_return_pct:+.3f}%")
            rules.append('<li><a href="#candidate-' + str(int(item.freeze_rank)) + '">' + escape(item.pair + ' / ' + item.strategy)
                         + '</a>：' + escape(rule_text(members[item.candidate_id]["spec"])) + '</li>')
        all_two = passed_rows.oos_positive_years.eq(2).all()
        weakness = (f"这{len(passed_rows)}项均只有2/3个周年盈利。" if all_two else "每项的正收益周年数和负周年完整列在下表。")
        if last_negative:
            weakness += "最后周年仍为负：" + "；".join(last_negative) + "。"
        bb = passed_rows.loc[passed_rows.candidate_id.eq("JPY|BB_MR_20|TIME_CAP_5")]
        if len(bb):
            weakness += f"JPY / BB_MR_20 的OOS简单年化为 {bb.iloc[0].oos_annualized_simple_pct:.4f}%，接近1%门槛。"
        passed_html = ('<section id="passed-summary"><h2>已保存的曲线与新质量同时通过项</h2><p><a href="' + passed_pdf_name + '">'
                       + str(len(passed_rows)) + '页便捷摘要PDF</a>保留原冻结顺序 '
                       + ' / '.join(str(int(v)) for v in passed_rows.freeze_rank) + '；完整图册仍保留全部'
                       + str(len(frozen)) + '项及失败。这是阅读摘要，不产生新排名或替换候选。</p><p class="note caution">'
                       + escape(weakness) + '原IS回撤要求是相对BASE的限制，不保证绝对回撤低；通过门槛不能称作曲线普遍平稳，也不构成新的独立OOS证据。</p>'
                       + table(pd.DataFrame(brief)) + '<ul>' + ''.join(rules) + '</ul></section>')
    diagnostics = ""
    if oos_completed:
        for frame in (anniversaries, fresh):
            frame["_rank"] = frame.candidate_id.map(frozen.set_index("candidate_id").freeze_rank)
            frame.sort_values(["_rank", "split", "variant"], inplace=True); frame.drop(columns="_rank", inplace=True)
        diagnostics = '<section><h2>全部冻结候选的边界与周年诊断</h2><p>现金起步仅允许OOS后新开的原BASE交易段进入，禁止中途加入旧段。它只作诊断，不能替换连续持仓主结果或重选。</p>' \
            + '<details><summary>三个预设周年窗口</summary>' + table(anniversaries, ["candidate_id", *METRICS]) + '</details>' \
            + '<details><summary>OOS现金起步</summary>' + table(fresh, ["candidate_id", *METRICS]) + '</details></section>'
    body = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>持仓退出研究：期望值与毛利润因子</title><style>{_CSS}</style></head><body>
<header><p class="eyebrow">HOLDING TRADEOFF V8 · AUTHORIZED QUALITY CRITERIA</p><h1>允许胜率与盈亏比取舍，同时改善期望值与毛利润因子</h1>
<p class="note caution"><b>{escape(oos_summary)}</b></p><p class="lead">用户已批准这项交易质量取舍。本轮重新应用获准标准，固定使用此前保存的492个持仓退出版本，保留其他原门槛和原排序。旧v3、v4和讨论记录保持原样，其原标准不通过结论不被倒改。</p>
<div class="stats"><div class="stat"><strong>180</strong><span>原BASE账户</span></div><div class="stat"><strong>492</strong><span>已有固定版本</span></div><div class="stat"><strong>{int(selection['pass'].sum())}</strong><span>新IS标准通过版本</span></div><div class="stat"><strong>{len(frozen)}</strong><span>冻结账户</span></div></div>
<p><a href="{atlas}">完整PDF图册</a> · <a href="#frozen">全部冻结曲线</a> · <a href="#all-versions">全部492版与失败</a> · <a href="#evidence">协议与审计</a></p></header><main>
{focus_html}
<section><h2>用户授权的改变与保持的要求</h2>{table(comparison)}<p>期望值 = 合格闭合交易单利收益之和 / 合格闭合笔数，包含平手交易；毛利润因子 = 盈利总和 / 亏损绝对值总和。它与平均盈利/平均亏损的“盈亏比”不同；分母为零或未定义不能视作通过。两项均采用严格改善，不另拟合最小改善阈值。</p>
<p>新版质量允许胜率或平均盈亏比下降；报告仍完整展示这两项及旧标准判定。原排序仍将盈亏比和胜率改善作为次级排序键，这不等于恢复旧质量硬门槛。</p>
<p class="note caution">此标准改变发生在旧IS结果和多轮历史研究已经可见之后，是明确授权的另一轮研究。最后三年仍是复用历史OOS，不能宣称新收集、未触碰或独立验证。所有冻结候选及失败保留，不以OOS修改规则、阈值或顺序。</p></section>
<section><h2>固定来源、时钟与账户口径</h2><p>391个条件检查版本来自holding_filter_v3，101个无条件时间上限来自timecap_filter_v4；合并保留全部492个版本，未新增参数。旧IS讨论产生60个替代标准诊断通过版本和21个账户建议；本轮以保存的实际筛选和冻结结果为准，报告不会硬编码21个结果。</p>
<p>检查年龄固定为5、10、20、40、63个已经赚取的报价收益；每段每个候选只在精确年龄检查一次。该日原BASE技术退出/反转已发出则不额外退出；条件未知保留。信号收盘t发出，t+1收盘退出并赚取该日收益，可避免的原BASE尾部从t+2起，直到原段结束前禁止重入。</p>
<p>IS：2005-01-03—2023-05-29；OOS：2023-05-30—2026-05-29。固定本金${capital:,.0f}，成本0，单利逐日相加；不含carry、融资或现金利息。图形IS/OOS分别归零，持仓连续。交易质量只用区间内入场并真实平仓的完整交易cohort；跨界及期末未闭合收益仍进入按实际日计算的账户PnL。</p>
<details><summary>全部180账户覆盖，含无候选账户</summary>{table(coverage)}</details></section>
<section><h2>新旧质量标准与完整失败计数</h2><p>旧标准通过 {int(selection.original_standard_pass.sum())} 个版本；新标准通过 {int(selection['pass'].sum())} 个版本。同一版本可失败于多项，计数不可相加视作独立样本。</p>{table(failure_counts)}
<p>在本轮{len(frozen)}个冻结者中，实际IS胜率下降{tradeoff_counts['win_rate_actually_lower']}个、实际盈亏比下降{tradeoff_counts['payoff_actually_lower']}个；旧胜率门槛未达到{tradeoff_counts['old_win_rate_gate_failed']}个、旧盈亏比门槛未达到{tradeoff_counts['old_payoff_gate_failed']}个。未达到旧改善幅度，并不一定意味着指标实际下降；这些类别可以重叠。</p>
<p><a href="report_gate_failure_counts.csv">全部门槛计数CSV</a> · <a href="report_all_492_criteria.csv">全部492版新旧标准CSV</a> · <a href="report_IS_failures.csv">所有IS未通过版本CSV</a></p></section>
<section><h2>完整OOS判定，保留IS冻结顺序</h2>{oos_html}</section>
{passed_html}<section id="frozen"><h2>每一个冻结候选的IS／OOS曲线</h2><ol>{''.join(links)}</ol>{''.join(cards)}</section>{diagnostics}
<section id="all-versions"><h2>全部492个固定版本</h2><p>下表保持保存的IS筛选顺序；新旧标准同时展示。通过但未获账户首位的版本也保留。</p><details><summary>完整新旧标准并列表</summary>{table(parallel)}</details><details><summary>全部新IS标准未通过版本（{len(failed)}）</summary>{table(failed)}</details></section>
<section id="evidence"><h2>冻结记录与独立审计</h2>{audit_html}<p><a href="protocol.json">本轮协议与用户授权说明</a> · <a href="frozen_selection.json">冻结记录</a> · <a href="is_hypotheses.json">固定492版本</a> · <a href="{review_name}">旧IS-only讨论原记录</a> · <a href="holding_tradeoff_report_build.json">报告输入／输出／源码哈希</a></p><p>旧讨论文件仍保留当时“等待用户选择”的状态；后续授权由本轮协议记录，未重写旧证据。</p></section></main></body></html>'''
    (output / report).write_text(body, encoding="utf-8")
    inputs.unchanged()
    validation = validate_artifacts(output, report, atlas, figures, len(frozen))
    products = [report, atlas, *figures, "report_all_492_criteria.csv", "report_IS_failures.csv", "report_gate_failure_counts.csv"]
    if oos_completed:
        products.extend(["report_OOS_complete_results.csv", "report_OOS_failures.csv"])
    if passed_pdf_name:
        subset = (output / passed_pdf_name).read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page\b", subset))
        if not subset.startswith(b"%PDF-") or b"%%EOF" not in subset[-100:] or pages != len(passing_ids):
            raise ValueError("Passed-summary PDF differs from the saved curve AND quality subset")
        validation["passed_summary_pages"] = pages
        products.append(passed_pdf_name)
    manifest = {"created_utc": now(), "report": report, "atlas": atlas, "figures": figures,
                "candidate_versions": len(selection), "source_counts": source_counts, "accounts": len(coverage),
                "original_standard_passing_versions": int(selection.original_standard_pass.sum()),
                "authorized_standard_passing_versions": int(selection['pass'].sum()), "frozen_candidates": len(frozen),
                "frozen_IS_tradeoff_diagnostics": tradeoff_counts,
                "frozen_candidate_ids": frozen_ids, "oos_evaluated": oos_completed, "oos_pass_counts": oos_counts,
                "passed_summary": {"file": passed_pdf_name, "candidate_ids": [cid for cid in frozen_ids if cid in passing_ids],
                                   "freeze_ranks": frozen.loc[frozen.candidate_id.isin(passing_ids), "freeze_rank"].astype(int).tolist()},
                "plot_checks": chart_checks, "artifact_validation": validation, "input_paths_relative_to": str(output),
                "input_sha256": inputs.hashes, "output_sha256": {name: sha(output / name) for name in products},
                "source_sha256": {"build_holding_tradeoff_report.py": sha(__file__), "src/fx_filter_report.py": sha(ROOT / "src/fx_filter_report.py")},
                "strategy_recomputed": False, "selection_changed": False, "old_OOS_results_read": False,
                "ordering": "Every frozen candidate in saved IS freeze_rank order; every original version in saved selection order"}
    (output / "holding_tradeoff_report_build.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("report", "candidate_versions", "frozen_candidates", "oos_evaluated", "oos_pass_counts")}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    build_report(parser.parse_args().output)


if __name__ == "__main__":
    main()
