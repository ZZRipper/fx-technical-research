"""One authorized IS-only discussion plot for the fixed AUD TIME_CAP_10 rule.

Reconstruct positions from saved v4 decisions and source AUD prices at/before
IS_END. No signal generation, selection, parameter search or OOS evaluation is
performed. The original v4 failure remains unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "reproduced/timecap_filter_v4"
OUTPUT = ROOT / "reproduced/holding_quality_tradeoff_review"
CANDIDATE = "AUD|DONCHIAN_BO_20|TIME_CAP_10"
PAIR, STRATEGY, FILTER = CANDIDATE.split("|")
SPEC = {"id": FILTER, "checkpoint_age": 10}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value):
    return value is not None and not pd.isna(value) and str(value).lower() in {"true", "1", "1.0"}


def aud_is_prices(path, cutoff):
    """Decode price cells only after checking source-row date and currency.

    The workbook is hashed as opaque bytes. Date/currency metadata may be
    scanned beyond cutoff, but no post-cutoff price cell is decoded or returned.
    """
    rows = []
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root]

        def value(cell):
            if cell is None:
                return None
            kind = cell.get("t", "")
            raw = cell.find(NS + "v")
            if kind == "inlineStr":
                return "".join(node.text or "" for node in cell.iter(NS + "t"))
            if raw is None or not raw.text:
                return None
            if kind == "s":
                return shared[int(raw.text)]
            if kind in {"str", "e"}:
                return raw.text
            return float(raw.text)

        with archive.open("xl/worksheets/sheet6.xml") as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != NS + "row":
                    continue
                cells = {"".join(ch for ch in cell.get("r", "") if ch.isalpha()): cell for cell in element.findall(NS + "c")}
                serial = value(cells.get("B"))
                if isinstance(serial, (int, float)) and np.isfinite(serial):
                    day = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(serial))
                    if day <= cutoff and value(cells.get("D")) == PAIR:
                        price = value(cells.get("E"))
                        if price is not None:
                            rows.append((day, float(price)))
                element.clear()
    frame = pd.DataFrame(rows, columns=["date", "price"]).sort_values("date").reset_index(drop=True)
    if frame.empty or frame.date.duplicated().any() or frame.date.max() > cutoff:
        raise AssertionError("Invalid IS-only source price calendar.")
    if not np.isfinite(frame.price).all() or not frame.price.gt(0).all():
        raise AssertionError("Invalid source AUD prices.")
    return frame


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def build_review():
    protected_names = ["protocol.json", "development_status.json", "frozen_selection.json", "is_selection.csv", "is_hypotheses.json", "is_all_metrics.csv", "is_all_decisions.csv.gz"]
    protected_before = {name: sha(CAMPAIGN / name) for name in protected_names}
    protocol = json.loads((CAMPAIGN / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((CAMPAIGN / "frozen_selection.json").read_text(encoding="utf-8"))
    hypotheses = json.loads((CAMPAIGN / "is_hypotheses.json").read_text(encoding="utf-8"))
    if protocol["version"] != "timecap_filter_v4" or protocol["cost_bp"] != 0 or SPEC not in protocol["candidate_specs"]:
        raise AssertionError("The review must use the unchanged registered v4 zero-cost rule.")
    proposal = [row for row in hypotheses["candidates"] if row["candidate_id"] == CANDIDATE]
    if len(proposal) != 1 or proposal[0]["spec"] != SPEC:
        raise AssertionError("The fixed review candidate is not the saved proposal.")
    for name, digest in frozen["is_output_sha256"].items():
        if name in protected_before and protected_before[name] != digest:
            raise AssertionError("Saved v4 input changed: " + name)
    selection = pd.read_csv(CAMPAIGN / "is_selection.csv")
    selected = selection.loc[selection.candidate_id.eq(CANDIDATE)]
    if len(selected) != 1 or truth(selected.iloc[0]["pass"]) or truth(selected.iloc[0].frozen) or selected.iloc[0].failed_criteria != "payoff_pass":
        raise AssertionError("The discussion label must match the saved original-criteria failure.")
    expected = pd.read_csv(CAMPAIGN / "is_all_metrics.csv")
    expected = expected.loc[expected.candidate_id.eq(CANDIDATE)].copy()
    decisions = pd.read_csv(CAMPAIGN / "is_all_decisions.csv.gz")
    decisions = decisions.loc[decisions.candidate_id.eq(CANDIDATE)].sort_values("signal_idx").copy()
    if len(expected) != 8 or decisions.empty or not decisions.condition.eq("ALL").all() or not decisions.checkpoint_age.eq(10).all():
        raise AssertionError("Unexpected saved review metrics or decisions.")
    input_hashes = {f"reproduced/timecap_filter_v4/{name}": digest for name, digest in protected_before.items()}
    workbook_names = [name for name in protocol["input_sha256"] if "FX_Technical_PIT_Dataset" in name and name.endswith(".xlsx")]
    if len(workbook_names) != 1:
        raise AssertionError("Expected one registered raw price workbook.")
    workbook_name = workbook_names[0]
    ledger_name = "reproduced/holding_observations_v3/is_base_trade_ledger.csv.gz"
    for name in (workbook_name, ledger_name):
        actual_hash = sha(ROOT / name)
        if actual_hash != protocol["input_sha256"][name]:
            raise AssertionError("Registered research input changed: " + name)
        input_hashes[name] = actual_hash
    cutoff = pd.Timestamp(protocol["is_end"])
    price_frame = aud_is_prices(ROOT / workbook_name, cutoff)
    dates = pd.DatetimeIndex(price_frame.date)
    prices = price_frame.price.to_numpy(float)
    target = np.zeros(len(dates), dtype=np.int8)
    used = np.zeros(len(dates), dtype=np.int8)
    for row in decisions.itertuples(index=False):
        start = int(row.signal_idx)
        stop = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
        if not (0 <= start < stop <= len(dates)) or dates[start] != pd.Timestamp(row.signal_date):
            raise AssertionError("Saved signal indices do not match the independently read source calendar.")
        used[start:stop] += 1
        target[start:stop] = int(row.position)
    if np.any(used > 1):
        raise AssertionError("Original TARGET episodes overlap.")
    filtered_target = target.copy()
    for row in decisions.loc[decisions.early_exit.map(truth)].itertuples(index=False):
        signal = int(row.risk_signal_idx)
        stop = int(row.original_end_signal_idx) if pd.notna(row.original_end_signal_idx) else len(dates)
        if signal != int(row.entry_idx) + SPEC["checkpoint_age"] or dates[signal] != pd.Timestamp(row.risk_signal_date):
            raise AssertionError("Saved checkpoint is not the registered exact holding age.")
        if pd.notna(row.actual_exit_idx) and int(row.actual_exit_idx) != signal + 1:
            raise AssertionError("Saved risk exit violates the next-close clock.")
        filtered_target[signal:stop] = 0
    sys.path.insert(0, str(ROOT / "src"))
    from fx_baselines import fixed_execute
    from fx_filter_evaluation import evaluate_window, simple_ledger
    accounting_sources = ["src/fx_baselines.py", "src/fx_filter_evaluation.py"]
    for name in accounting_sources:
        digest = sha(ROOT / name)
        if digest != protocol["source_sha256"][name]:
            raise AssertionError("Registered accounting source changed: " + name)
        input_hashes[name] = digest
    runs = {"BASE": fixed_execute(prices, target, cost_bp=0., capital=protocol["initial_capital_usd"]),
            "FILTER": fixed_execute(prices, filtered_target, cost_bp=0., capital=protocol["initial_capital_usd"])}
    ledgers = {variant: simple_ledger(run, dates) for variant, run in runs.items()}
    original_ledger = pd.read_csv(ROOT / ledger_name)
    original_ledger = original_ledger.loc[original_ledger.pair.eq(PAIR) & original_ledger.strategy.eq(STRATEGY)].sort_values("entry_idx").reset_index(drop=True)
    replayed_ledger = ledgers["BASE"].sort_values("entry_idx").reset_index(drop=True)
    if len(original_ledger) != len(replayed_ledger):
        raise AssertionError("Reconstructed BASE episode count differs from the saved IS ledger.")
    errors = []
    checks = 0
    def compare(actual, saved, label):
        nonlocal checks
        checks += 1
        if isinstance(actual, (bool, np.bool_)):
            if bool(actual) != truth(saved):
                raise AssertionError(label)
        elif isinstance(actual, (pd.Timestamp, datetime)):
            if pd.Timestamp(actual) != pd.Timestamp(saved):
                raise AssertionError(label)
        elif isinstance(actual, (int, float, np.number)):
            actual, saved = float(actual), float(saved)
            if np.isnan(actual) and np.isnan(saved):
                return
            if not np.isfinite([actual, saved]).all() or not np.isclose(actual, saved, rtol=0., atol=1e-8):
                raise AssertionError(f"{label}: {actual} != {saved}")
            errors.append(abs(actual - saved))
        elif str(actual) != str(saved):
            raise AssertionError(label)
    for column in ("signal_idx", "entry_idx", "exit_idx", "position", "holding_observations", "closed", "eligible_closed", "gross_simple_return_pct"):
        for actual, saved in zip(replayed_ledger[column], original_ledger[column]):
            compare(actual, saved, "BASE ledger " + column)
    periods = [("IS", protocol["is_start"], protocol["is_end"]), *protocol["is_blocks"]]
    comparisons = []
    for split, start, end in periods:
        computed = {variant: evaluate_window(run, dates, start, end, cost_bp=0., ledger=ledgers[variant]) for variant, run in runs.items()}
        for variant, actual in computed.items():
            base_closed = computed["BASE"]["closed_n"]
            actual.update(base_closed_n=base_closed, retention=actual["closed_n"] / base_closed if base_closed else np.nan)
            saved = expected.loc[expected.split.eq(split) & expected.variant.eq(variant)]
            if len(saved) != 1:
                raise AssertionError("Missing unique saved interval/variant.")
            saved = saved.iloc[0]
            for field, value in actual.items():
                if field not in saved:
                    raise AssertionError("A recomputed official metric is absent from saved v4: " + field)
                compare(value, saved[field], f"{split}/{variant}/{field}")
            comparisons.append({"split": split, "variant": variant, "all_metrics_match": True, "metrics": actual})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle("IS only; original criteria fail (payoff); discussion only", color="#9e3434", fontsize=17, fontweight="bold", y=.976)
    fig.text(.065, .918, "AUD | DONCHIAN_BO_20 | TIME_CAP_10\nFixed USD 100,000 initial capital; zero costs; parameters and original SCREEN unchanged.", fontsize=11, color="#334d5c")
    mask = (dates >= protocol["is_start"]) & (dates <= protocol["is_end"])
    window_dates = dates[mask]
    anchor = pd.Timestamp(expected.loc[expected.split.eq("IS") & expected.variant.eq("BASE"), "anchor_date"].iloc[0])
    plot_dates = window_dates.insert(0, anchor)
    chart_checks = []
    for variant, color in (("BASE", "#728695"), ("FILTER", "#087e8b")):
        cumulative = np.r_[0., runs[variant]["gross_return"][mask].cumsum() * 100]
        drawdown = cumulative - np.maximum.accumulate(cumulative)
        saved = expected.loc[expected.split.eq("IS") & expected.variant.eq(variant)].iloc[0]
        compare(cumulative[-1], saved.gross_simple_return_pct, variant + " chart endpoint")
        compare(drawdown.min(), saved.gross_max_drawdown_pp, variant + " chart drawdown")
        axes[0].plot(plot_dates, cumulative, color=color, linewidth=1.5, label=f"{variant}: {cumulative[-1]:+.2f}% total; {saved.gross_annualized_simple_pct:.2f}% simple annualized")
        axes[1].plot(plot_dates, drawdown, color=color, linewidth=1.3, label=f"{variant}: max drawdown {drawdown.min():.2f} pp")
        chart_checks.append({"variant": variant, "cumulative_return_pct": float(cumulative[-1]), "maximum_drawdown_pp": float(drawdown.min()), "saved_endpoint_matches": True, "saved_drawdown_matches": True})
    for ax in axes:
        ax.axhline(0, color="#aab8c0", linewidth=.7)
        ax.grid(alpha=.16)
        for boundary in ("2015-01-01", "2020-01-01"):
            ax.axvline(pd.Timestamp(boundary), color="#9daeb8", linewidth=1, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=10, framealpha=.92)
    axes[0].set_ylabel("Cumulative simple return (%)")
    axes[1].set_ylabel("Drawdown (initial-capital pp)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(base=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(anchor, pd.Timestamp(protocol["is_end"]))
    axes[1].set_xlabel("IS only: 2005-01-03 to 2023-05-29; dotted lines mark original IS subperiods")
    fig.text(.065, .025, "Original payoff requirement: at least +10%; observed change: -13.46%. This review does not change the rule, approve a new gate, or open OOS.", fontsize=9, color="#7b3434")
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.10, hspace=.16)
    png_name, pdf_name = "aud_donchian20_timecap10_is_review.png", "aud_donchian20_timecap10_is_review.pdf"
    fig.savefig(OUTPUT / png_name, dpi=160, facecolor="white")
    with PdfPages(OUTPUT / pdf_name) as pdf:
        pdf.infodict().update(Title="IS only; original criteria fail (payoff); discussion only", Subject=CANDIDATE)
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    protected_after = {name: sha(CAMPAIGN / name) for name in protected_names}
    if protected_before != protected_after:
        raise AssertionError("An original v4 artifact changed during the review.")
    validation = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "candidate_id": CANDIDATE,
                  "purpose": "IS-only discussion of the fixed candidate; no revision to original SCREEN or frozen selection",
                  "original_is_pass": False, "failed_criteria": ["payoff_pass"], "frozen": False,
                  "parameters": SPEC, "oos_computed": False, "oos_result_files_read": [], "oos_price_values_decoded": 0,
                  "source_price_scope": {"pair": PAIR, "first_quote": dates.min(), "last_quote": dates.max(), "quotes_including_warmup": len(dates)},
                  "reconstruction": "Original and filtered TARGET arrays reconstructed from the fixed candidate's saved original episode and risk-signal ranges; fixed_execute used only for zero-cost accounting. No strategy signal generator or selection function was called.",
                  "source_read_policy": "Read date/currency metadata first; decode price cells only for AUD at/before IS_END. Full-workbook hash is an opaque file-integrity check.",
                  "checked_fields": checks, "max_absolute_numeric_error": max(errors, default=0.), "base_ledger_episodes_checked": len(original_ledger),
                  "all_eight_saved_metric_rows_match": True, "period_metric_comparisons": comparisons, "chart_checks": chart_checks,
                  "input_sha256": input_hashes, "review_source_sha256": sha(Path(__file__)),
                  "v4_protected_inputs_unchanged": protected_before == protected_after, "v4_protected_sha256": protected_after,
                  "outputs": {png_name: sha(OUTPUT / png_name), pdf_name: sha(OUTPUT / pdf_name)}, "pdf_pages": 1}
    (OUTPUT / "review_plot_validation.json").write_text(json.dumps(clean(validation), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"output_dir": str(OUTPUT), "png": png_name, "pdf": pdf_name, "validation": "review_plot_validation.json", "checked_fields": checks, "max_absolute_numeric_error": max(errors, default=0.)}


if __name__ == "__main__":
    print(json.dumps(build_review(), ensure_ascii=False, indent=2))
