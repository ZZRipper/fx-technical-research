"""Read-only campaign audit, with an optional standalone JSON evidence output.

Reads only the named campaign's completed IS artifacts; OOS artifacts are read
only in --stage oos or when --stage auto finds a completed OOS status. Does not
run a candidate, tune a parameter, read previous full-sample results, or alter
the decision sources. Example: python -B verify_filter_research.py --stage auto
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from fx_baselines import CURRENCIES, STRATEGIES
from fx_filter_evaluation import screen_candidates


class Audit:
    def __init__(self):
        self.checks = []
        self.files = set()

    def check(self, name, passed, detail=None):
        row = {"check": name, "pass": bool(passed)}
        if detail is not None:
            row["detail"] = str(detail)
        self.checks.append(row)
        return bool(passed)

    def read_json(self, path):
        self.files.add(str(path.resolve()))
        return json.loads(path.read_text(encoding="utf-8"))

    def read_csv(self, path):
        self.files.add(str(path.resolve()))
        return pd.read_csv(path)

    def hashes(self, mapping, directory, label):
        for name, expected in mapping.items():
            path = directory / name
            self.files.add(str(path.resolve()))
            actual = digest(path) if path.is_file() else None
            self.check(f"{label}: {name}", actual == expected)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_numbers(actual, expected, tolerance=1e-8):
    return bool(np.isclose(np.asarray(actual, float), np.asarray(expected, float),
                           rtol=1e-10, atol=tolerance, equal_nan=True).all())


def compare_frames(audit, actual, expected, keys, columns, label):
    duplicate_free = not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    actual_keys = set(actual[keys].itertuples(index=False, name=None))
    expected_keys = set(expected[keys].itertuples(index=False, name=None))
    if not audit.check(label + ": exact keys", duplicate_free and actual_keys == expected_keys):
        return
    merged = actual.merge(expected, on=keys, suffixes=("_a", "_e"), validate="one_to_one")
    for column in columns:
        if column in keys:
            continue
        if not audit.check(label + ": column " + column, column in actual and column in expected):
            continue
        left, right = merged[column + "_a"], merged[column + "_e"]
        if pd.api.types.is_numeric_dtype(expected[column]):
            matches = same_numbers(left, right)
        else:
            # pandas reads an empty CSV text field as NaN; it is the serialized
            # representation of an empty failure-reason string on passing rows.
            matches = bool(left.fillna("").eq(right.fillna("")).all())
        audit.check(label + ": values " + column, matches)


def dates(frame, columns):
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def audit_daily(audit, frame, ids, start, end, label):
    keys = ["candidate_id", "variant", "date"]
    audit.check(label + ": unique account/date", not frame.duplicated(keys).any())
    actual = set(frame[["candidate_id", "variant"]].itertuples(index=False, name=None))
    audit.check(label + ": exact frozen account set", actual == {(c, v) for c in ids for v in ("BASE", "FILTER")})
    audit.check(label + ": date bounds", frame.date.between(pd.Timestamp(start), pd.Timestamp(end)).all())
    audit.check(label + ": finite daily returns", np.isfinite(frame[["gross_daily_return", "net_daily_return"]]).all().all())
    audit.check(label + ": zero costs gross equals net", same_numbers(frame.gross_daily_return, frame.net_daily_return, 0.))
    audit.check(label + ": signed unit positions", frame.executed.isin([-1, 0, 1]).all() and frame.held.isin([-1, 0, 1]).all())
    for candidate_id, account in frame.groupby("candidate_id", sort=False):
        calendars = [pd.DatetimeIndex(account.loc[account.variant.eq(v), "date"].sort_values()) for v in ("BASE", "FILTER")]
        audit.check(f"{label}: matched calendar {candidate_id}", calendars[0].equals(calendars[1]))


def independent_ledger(audit, ledger, daily, label):
    """Rebuild each episode's arithmetic return from archived daily positions."""
    ledger = dates(ledger, ["entry_date", "exit_date"])
    ledger["audit_return_pct"] = np.nan
    ledger["audit_holding_n"] = 0
    all_positions_match = True
    for (candidate_id, variant), rows in ledger.groupby(["candidate_id", "variant"], sort=False):
        path = daily.loc[daily.candidate_id.eq(candidate_id) & daily.variant.eq(variant)].sort_values("date")
        calendar = pd.DatetimeIndex(path.date)
        prefix = np.r_[0., np.cumsum(path.gross_daily_return.to_numpy(float))]
        starts = calendar.searchsorted(pd.DatetimeIndex(rows.entry_date), side="right")
        stops = calendar.searchsorted(pd.DatetimeIndex(rows.exit_date), side="right")
        ledger.loc[rows.index, "audit_return_pct"] = (prefix[stops] - prefix[starts]) * 100
        ledger.loc[rows.index, "audit_holding_n"] = stops - starts
        held = path.held.to_numpy()
        for row, first, last in zip(rows.itertuples(index=False), starts, stops):
            all_positions_match &= bool((held[first:last] == row.position).all())
    audit.check(label + ": holding segment positions", all_positions_match)
    audit.check(label + ": holding observations", same_numbers(ledger.audit_holding_n, ledger.holding_observations, 0.))
    audit.check(label + ": independent arithmetic return", same_numbers(ledger.audit_return_pct, ledger.gross_simple_return_pct))
    audit.check(label + ": eligible closed definition", ledger.eligible_closed.eq(ledger.closed & ledger.audit_holding_n.gt(0)).all())
    audit.check(label + ": zero-cost trade returns", same_numbers(ledger.audit_return_pct, ledger.net_trade_return_pct))
    audit.check(label + ": dollar accounting", same_numbers(ledger.audit_return_pct * 1000., ledger.net_pnl_usd, 1e-5))
    return ledger


def independent_metrics(audit, metrics, daily, ledger, label):
    """Recompute window arithmetic and complete-entry-cohort quality independently."""
    metrics = dates(metrics, ["start_date", "end_date", "anchor_date"])
    checks = {name: True for name in (
        "daily simple sums", "simple drawdowns", "calendar annualization",
        "closed cohort counts", "win rates including flat", "payoff ratios",
        "carry-in counts", "cash costs", "quote observations",
    )}
    for row in metrics.itertuples(index=False):
        account_path = daily.loc[daily.candidate_id.eq(row.candidate_id) & daily.variant.eq(row.variant)].sort_values("date")
        path = account_path.loc[account_path.date.between(row.start_date, row.end_date)]
        returns = path.gross_daily_return.to_numpy(float)
        total = returns.sum() * 100
        cumulative = np.r_[0., np.cumsum(returns) * 100]
        drawdown = (cumulative - np.maximum.accumulate(cumulative)).min()
        years = (row.end_date - row.anchor_date).days / 365.25
        trades = ledger.loc[ledger.candidate_id.eq(row.candidate_id) & ledger.variant.eq(row.variant)]
        complete = trades.loc[trades.entry_date.between(row.start_date, row.end_date)
                              & trades.closed & trades.audit_holding_n.gt(0)
                              & trades.exit_date.le(row.end_date)]
        outcomes = complete.audit_return_pct.to_numpy(float)
        wins, losses = outcomes[outcomes > 1e-10], outcomes[outcomes < -1e-10]
        win_rate = len(wins) / len(outcomes) if len(outcomes) else np.nan
        payoff = wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan
        carry = trades.loc[trades.entry_date.lt(row.start_date) & trades.exit_date.ge(row.start_date)]
        checks["daily simple sums"] &= same_numbers([row.gross_simple_return_pct, row.net_simple_return_pct], [total, total])
        checks["simple drawdowns"] &= same_numbers([row.gross_max_drawdown_pp, row.net_max_drawdown_pp], [drawdown, drawdown])
        checks["calendar annualization"] &= same_numbers([row.calendar_years, row.gross_annualized_simple_pct], [years, total / years if years else np.nan])
        checks["closed cohort counts"] &= row.closed_n == len(outcomes)
        checks["win rates including flat"] &= same_numbers(row.win_rate, win_rate)
        checks["payoff ratios"] &= same_numbers(row.payoff_ratio, payoff)
        checks["carry-in counts"] &= row.carry_in_n == len(carry)
        checks["cash costs"] &= row.cost_drag_pp == 0
        checks["quote observations"] &= row.observations == len(path)
    for name, passed in checks.items():
        audit.check(label + ": " + name, passed)


def audit_campaign(output, stage="auto"):
    audit = Audit()
    development = audit.read_json(output / "development_status.json")
    if not audit.check("IS development completed before outcome-table access", development.get("status") == "passed"):
        raise ValueError("Development is not passed; do not open IS/OOS outcome tables yet.")
    protocol = audit.read_json(output / "protocol.json")
    freeze = audit.read_json(output / "frozen_selection.json")
    audit.check("development freeze hash", development.get("freeze_sha256") == digest(output / "frozen_selection.json"))
    audit.check("protocol hash bound to freeze and development", digest(output / "protocol.json") == freeze["protocol_sha256"] == development["protocol_sha256"])
    audit.check("freeze prior to OOS evaluation", freeze.get("oos_evaluated_at_freeze") is False and development.get("holdout_returns_computed") is False)
    audit.check("exact declared split", [protocol[k] for k in ("is_start", "is_end", "oos_start", "oos_end")] == ["2005-01-03", "2023-05-29", "2023-05-30", "2026-05-29"])
    audit.check("zero-cost protocol", protocol.get("cost_bp") == 0 and protocol.get("reference_cost_bp") == 0)
    audit.check("source manifests equal", protocol["source_sha256"] == freeze["source_sha256"])
    audit.check("input manifests equal", protocol["input_sha256"] == freeze["input_sha256"])
    audit.hashes(protocol["source_sha256"], ROOT, "decision source hash")
    audit.hashes(protocol["input_sha256"], ROOT, "input hash")
    audit.hashes(freeze["is_output_sha256"], output, "frozen IS output hash")
    expected_accounts = {(pair, rule) for pair in CURRENCIES for rule in STRATEGIES}
    audit.check("protocol scope matches full fixed universe", protocol["scope"]["pairs"] == list(CURRENCIES)
                and protocol["scope"]["strategies"] == list(STRATEGIES)
                and protocol["scope"]["baseline_accounts"] == 180)
    coverage = audit.read_csv(output / "is_account_coverage.csv")
    audit.check("180 exact accounts: 20 rules x 9 currencies", len(coverage) == 180 and not coverage.duplicated(["pair", "strategy"]).any()
                and set(coverage[["pair", "strategy"]].itertuples(index=False, name=None)) == expected_accounts)
    audit.check("IS compute cutoff per account", pd.to_datetime(coverage.compute_end).le(pd.Timestamp(protocol["is_end"])).all())
    all_metrics = audit.read_csv(output / "is_all_metrics.csv")
    selection = audit.read_csv(output / "is_selection.csv")
    audit.check("all selection accounts covered", set(selection[["pair", "strategy"]].itertuples(index=False, name=None)) == expected_accounts)
    audit.check("candidate count matches coverage", selection.groupby(["pair", "strategy"]).size().sort_index().equals(coverage.set_index(["pair", "strategy"]).candidate_count.sort_index()))
    audit.check("only IS periods in development metrics", set(all_metrics.split) == {"IS", "IS_EARLY", "IS_MIDDLE", "IS_LATE"})
    audit.check("all IS metric endpoints at or before cutoff", pd.to_datetime(all_metrics.end_date).le(pd.Timestamp(protocol["is_end"])).all())
    audit.check("all IS metric starts inside development interval", pd.to_datetime(all_metrics.start_date).ge(pd.Timestamp(protocol["is_start"])).all())
    audit.check("all IS metric costs zero", all_metrics.cost_drag_pp.eq(0).all() and same_numbers(all_metrics.gross_simple_return_pct, all_metrics.net_simple_return_pct))
    regenerated = screen_candidates(all_metrics)
    compare_frames(audit, selection, regenerated, ["candidate_id"], list(regenerated.columns), "IS rescreen")
    frozen_expected = regenerated.loc[regenerated.frozen].sort_values("freeze_rank")
    ids = [row["candidate_id"] for row in freeze["candidates"]]
    audit.check("frozen exact ordered IS winners", ids == frozen_expected.candidate_id.tolist())
    audit.check("freeze count and all-passing count", len(ids) == freeze["frozen_candidates"]
                and int(regenerated["pass"].sum()) == freeze["all_passing_versions"]
                and len(regenerated) == freeze["all_candidate_accounts_tested"])
    registry = {spec["id"]: spec for spec in protocol["candidate_specs"]}
    audit.check("frozen specs equal prespecified registry", all(row["spec"] == registry[row["filter_id"]] for row in freeze["candidates"]))
    audit.check("freeze identity and rank metadata", all(
        row["pair"] == row["candidate_id"].split("|")[0]
        and row["strategy"] == row["candidate_id"].split("|")[1]
        and row["filter_id"] == row["candidate_id"].split("|")[2]
        and row["freeze_rank"] == expected.freeze_rank
        for row, expected in zip(freeze["candidates"], frozen_expected.itertuples(index=False))))
    is_daily = dates(audit.read_csv(output / "is_frozen_daily.csv.gz"), ["date"])
    is_metrics = audit.read_csv(output / "is_frozen_metrics.csv")
    audit_daily(audit, is_daily, ids, protocol["is_start"], protocol["is_end"], "IS daily")
    if ids:
        is_ledger = independent_ledger(audit, audit.read_csv(output / "is_frozen_trades.csv.gz"), is_daily, "IS ledger")
        audit.check("IS ledger contains no post-cutoff entry or exit", is_ledger.entry_date.le(pd.Timestamp(protocol["is_end"])).all()
                    and is_ledger.exit_date.le(pd.Timestamp(protocol["is_end"])).all())
        is_decisions = audit.read_csv(output / "is_frozen_decisions.csv.gz")
        decision_dates = [column for column in is_decisions if column.endswith("_date")]
        audit.check("IS decisions contain no post-cutoff dates", all(
            pd.to_datetime(is_decisions[column]).dropna().le(pd.Timestamp(protocol["is_end"])).all()
            for column in decision_dates))
        independent_metrics(audit, is_metrics, is_daily, is_ledger, "IS frozen metrics")
        selected_reference = all_metrics.loc[all_metrics.candidate_id.isin(ids) & all_metrics.split.eq("IS")]
        fields = ["gross_simple_return_pct", "gross_max_drawdown_pp", "closed_n", "win_rate", "payoff_ratio"]
        compare_frames(audit, is_metrics, selected_reference, ["candidate_id", "variant", "split"], fields, "IS selected versus development")
    oos_status_path = output / "oos_status.json"
    oos_status = audit.read_json(oos_status_path) if stage != "is" and oos_status_path.exists() else None
    read_oos = stage == "oos" or (stage == "auto" and oos_status is not None and oos_status.get("status") == "passed")
    if read_oos:
        if oos_status is None or oos_status.get("status") != "passed":
            raise ValueError("OOS audit requires a completed OOS status.")
        audit.check("OOS exact frozen digest", oos_status["freeze_sha256"] == digest(output / "frozen_selection.json"))
        audit.check("OOS exact protocol digest", oos_status["protocol_sha256"] == digest(output / "protocol.json"))
        audit.check("freeze timestamp before OOS opened", pd.Timestamp(freeze["created_utc"]) < pd.Timestamp(oos_status["opened_utc"]))
        audit.check("OOS completion after opening", pd.Timestamp(oos_status["opened_utc"]) <= pd.Timestamp(oos_status["finished_utc"]))
        audit.hashes(oos_status["output_sha256"], output, "OOS output hash")
        oos_daily = dates(audit.read_csv(output / "oos_daily.csv.gz"), ["date"])
        oos_metrics = audit.read_csv(output / "oos_metrics.csv")
        anniversary = audit.read_csv(output / "oos_anniversary_metrics.csv")
        assessment = audit.read_csv(output / "candidate_assessment.csv")
        audit_daily(audit, oos_daily, ids, protocol["oos_start"], protocol["oos_end"], "OOS daily")
        audit.check("OOS retains every frozen candidate in original order", assessment.candidate_id.tolist() == ids
                    and assessment.freeze_rank.tolist() == [row["freeze_rank"] for row in freeze["candidates"]])
        audit.check("OOS metrics exact frozen accounts", set(oos_metrics[["candidate_id", "variant"]].itertuples(index=False, name=None))
                    == {(identifier, variant) for identifier in ids for variant in ("BASE", "FILTER")}
                    and len(oos_metrics) == 2 * len(ids))
        audit.check("OOS three-year calendar bounds", pd.Timestamp(protocol["oos_start"]) + pd.DateOffset(years=3) - pd.Timedelta(days=1) == pd.Timestamp(protocol["oos_end"]))
        declared_years = protocol["oos_anniversary_years"]
        audit.check("anniversary windows partition exactly", declared_years == [
            ["OOS_YEAR_1", "2023-05-30", "2024-05-29"],
            ["OOS_YEAR_2", "2024-05-30", "2025-05-29"],
            ["OOS_YEAR_3", "2025-05-30", "2026-05-29"],
        ])
        expected_year_keys = {(i, v, y[0]) for i in ids for v in ("BASE", "FILTER") for y in declared_years}
        audit.check("all OOS anniversary accounts present once", len(anniversary) == len(expected_year_keys)
                    and set(anniversary[["candidate_id", "variant", "split"]].itertuples(index=False, name=None)) == expected_year_keys)
        combined = pd.concat([is_daily, oos_daily], ignore_index=True).sort_values(["candidate_id", "variant", "date"])
        execution_consistent = True
        for _, rows in combined.groupby(["candidate_id", "variant"], sort=False):
            execution_consistent &= np.array_equal(rows.held.to_numpy()[1:], rows.executed.to_numpy()[:-1])
        audit.check("continuous execution across IS/OOS boundary", execution_consistent)
        full_ledger = independent_ledger(audit, audit.read_csv(output / "frozen_full_trade_ledger.csv.gz"), combined, "full frozen ledger")
        independent_metrics(audit, oos_metrics, combined, full_ledger, "OOS metrics")
        independent_metrics(audit, anniversary, combined, full_ledger, "OOS anniversary metrics")
        bounds_ok = True
        for year, start, end in declared_years:
            rows = anniversary.loc[anniversary.split.eq(year)]
            bounds_ok &= pd.to_datetime(rows.start_date).ge(pd.Timestamp(start)).all() and pd.to_datetime(rows.end_date).le(pd.Timestamp(end)).all()
        audit.check("anniversary metric boundaries", bounds_ok)
        yearly = anniversary.groupby(["candidate_id", "variant"]).gross_simple_return_pct.sum()
        main = oos_metrics.set_index(["candidate_id", "variant"]).gross_simple_return_pct
        audit.check("anniversary arithmetic sums to OOS", same_numbers(yearly.sort_index(), main.sort_index()))
        criterion = protocol["oos_screen"]
        rebuilt = []
        for identifier in ids:
            account = oos_metrics.loc[oos_metrics.candidate_id.eq(identifier)].set_index("variant")
            base, filtered = account.loc["BASE"], account.loc["FILTER"]
            years = anniversary.loc[anniversary.candidate_id.eq(identifier) & anniversary.variant.eq("FILTER")]
            flags = {
                "annualized_pass": filtered.net_annualized_simple_pct >= criterion["min_net_annualized_simple_pct"] - 1e-10,
                "baseline_improvement_pass": filtered.net_simple_return_pct - base.net_simple_return_pct >= criterion["min_net_return_improvement_pp"] - 1e-10,
                "drawdown_pass": filtered.net_max_drawdown_pp >= criterion["worst_net_drawdown_pp"] - 1e-10,
                "positive_years_pass": int((years.net_simple_return_pct > 1e-10).sum()) >= criterion["min_positive_anniversary_years"],
                "sample_pass": filtered.closed_n >= criterion["min_closed_trades"],
            }
            quality_confirmed = filtered.closed_n >= 20 and np.isfinite([filtered.win_rate, base.win_rate, filtered.payoff_ratio, base.payoff_ratio]).all() and filtered.win_rate >= base.win_rate - 1e-10 and filtered.payoff_ratio >= base.payoff_ratio - 1e-10
            rebuilt.append({"candidate_id": identifier, **flags, "oos_curve_pass": all(flags.values()), "oos_quality_confirmed": quality_confirmed})
        comparison = pd.DataFrame(rebuilt)
        compare_frames(audit, assessment, comparison, ["candidate_id"], list(comparison.columns), "predeclared OOS assessment")
        audit.check("OOS status result counts", int(assessment.oos_curve_pass.sum()) == oos_status["curve_passing_candidates"]
                    and int(assessment.oos_quality_confirmed.sum()) == oos_status["quality_confirmed_candidates"])
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(row["pass"] for row in audit.checks) else "failed",
        "stage": "is_and_oos" if read_oos else "is_only", "read_only_decision_sources": True,
        "auditor_sha256": digest(Path(__file__).resolve()),
        "previous_full_sample_artifacts_read": False, "checks": len(audit.checks),
        "passed_checks": sum(row["pass"] for row in audit.checks),
        "failed_checks": [row for row in audit.checks if not row["pass"]],
        "verified_file_count": len(audit.files), "files_verified": sorted(audit.files), "check_details": audit.checks,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced/filter_oos_v1")
    parser.add_argument("--stage", choices=["is", "oos", "auto"], default="auto")
    parser.add_argument("--audit-json", type=Path, help="Optional audit evidence file; no campaign input is modified.")
    args = parser.parse_args()
    try:
        result = audit_campaign(args.output.resolve(), args.stage)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}", "checked_utc": datetime.now(timezone.utc).isoformat()}
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("check_details", "files_verified")}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
