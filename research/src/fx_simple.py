"""Fixed-initial-capital accounting over independently reconstructed FX positions.

The original Explorer is used for account identities, calendars and reconciliation
only. Daily returns come from the raw-price research engines, never differences
or ratios of the archived, rounded account curves.
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fx_baselines import canonical_prices_from_raw, reconstruct_fixed_baselines, trade_ledger
from fx_economics import read_native_features_from_raw, rebuild_economic_known, run_economic_filters
from fx_evidence import load_explorer
from fx_legacy import reconstruct_h10, reconstruct_v3, V3_STAGES
from fx_risk import run_vix_variants


ACCOUNTING_METHOD = "fixed_initial_capital_simple"
ROOT = Path(__file__).resolve().parents[1]


def simple_equity(daily_returns, capital=100000.0):
    """E_t = C0 + C0 * sum(r_s); r_s is held direction times spot return."""
    returns = np.asarray(daily_returns, dtype=float)
    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("Daily returns must be a finite vector.")
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Initial capital must be positive and finite.")
    return capital * (1.0 + np.cumsum(returns))


def simple_path_statistics(daily_returns, dates, anchor_day, capital=100000.0):
    """Report simple returns and two explicitly named drawdown conventions."""
    returns = np.asarray(daily_returns, dtype=float)
    dates = pd.DatetimeIndex(dates)
    equity = simple_equity(returns, capital)
    if not len(equity) or len(equity) != len(dates):
        raise ValueError("A nonempty aligned calendar is required.")
    with_anchor = np.r_[capital, equity]
    peaks = np.maximum.accumulate(with_anchor)
    drawdown_dollars = with_anchor - peaks
    years = (dates[-1] - pd.to_datetime(anchor_day, unit="D")).days / 365.25
    total_return = float(returns.sum())
    # Arithmetic NAV can be negative. Percentage drawdown can exceed 100% when
    # its running-peak denominator remains positive; only a nonpositive peak
    # makes that denominator invalid.
    equity_drawdown = float(np.min(drawdown_dollars / peaks) * 100.0) if np.all(peaks > 0) else None
    return {
        "total_return_pct": total_return * 100.0,
        "account_pnl_usd": capital * total_return,
        "terminal_equity_usd": capital * (1.0 + total_return),
        "annualized_simple_return_pct": total_return * 100.0 / years if years > 0 else None,
        "max_drawdown_initial_capital_pp": float(drawdown_dollars.min() / capital * 100.0),
        "max_drawdown_equity_pct": equity_drawdown,
        "calendar_years": float(years),
        "nonpositive_equity_observations": int(np.count_nonzero(equity <= 0)),
    }


def _trade_quality(trades, tolerance_pct):
    closed = np.asarray([row[2] for row in trades if row[4] and row[3] > 0], dtype=float)
    wins = closed[closed > tolerance_pct]
    losses = closed[closed < -tolerance_pct]
    average_win = float(wins.mean()) if len(wins) else None
    average_loss = float(-losses.mean()) if len(losses) else None
    return {
        "closed": len(closed), "win": len(wins) / len(closed) if len(closed) else None,
        "payoff": average_win / average_loss if average_win is not None and average_loss else None,
        "wins": len(wins), "losses": len(losses),
        "flat": len(closed) - len(wins) - len(losses),
        "avg_win_return_pct": average_win, "avg_loss_abs_return_pct": average_loss,
        "expectancy_return_pct": float(closed.mean()) if len(closed) else None,
        "median_trade_return_pct": float(np.median(closed)) if len(closed) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else None,
        "open": sum(not row[4] for row in trades), "low_sample": len(closed) < 20,
    }


def _epoch_day(date):
    return int(pd.Timestamp(date).value // (86400 * 10**9))


def _rebuild_trades(run, dates, account, capital):
    ledger = trade_ledger(run, dates, account["pair"], account["rule"])
    returns = np.asarray(run["gross_return"], dtype=float)
    result = []
    covered = np.zeros(len(returns), dtype=np.int8)
    max_segment_error = 0.0
    for trade in ledger.itertuples(index=False):
        selection = slice(int(trade.entry_idx) + 1, int(trade.exit_idx) + 1)
        segment_return = float(returns[selection].sum())
        covered[selection] += 1
        # Independent dollars-first addition verifies the trade's accounting.
        max_segment_error = max(max_segment_error, abs(float((capital * returns[selection]).sum()) - capital * segment_return))
        result.append([
            _epoch_day(trade.entry_date), _epoch_day(trade.exit_date) if trade.closed else -1,
            segment_return * 100.0, int(trade.holding_observations), int(trade.closed), int(trade.position),
        ])
    if np.any(covered > 1) or np.any((returns != 0) & (covered != 1)):
        raise AssertionError(f"Trade segments do not partition daily PnL: {account['id']}")
    return result, ledger, max_segment_error


def _source_hashes(root):
    paths = sorted((root / "FX_Research_Datasets/data/raw").glob("*.xlsx"))
    paths += [root / "FX_Research_Datasets/data/clean/h10_official_daily.csv"]
    paths += [root / "src" / name for name in (
        "fx_baselines.py", "fx_economics.py", "fx_risk.py", "fx_legacy.py", "fx_simple.py",
    )]
    paths += [root / "build_simple_data.py"]
    return {str(path.relative_to(root)).replace("\\", "/"): {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
    } for path in paths}


def build_simple_payload(root=None, progress=print):
    """Return the 912-account simple-interest payload, metrics and audit tables."""
    root = Path(root).resolve() if root is not None else ROOT
    archive = load_explorer(root)
    data = root / "FX_Research_Datasets/data"
    capital = float(archive["capital"])
    progress("Reconstructing raw workbook prices and 207 fixed baselines...", flush=True)
    prices, _ = canonical_prices_from_raw(data)
    _, baseline_trades, baseline_runs, baseline_dates, _ = reconstruct_fixed_baselines(prices)
    progress("Reconstructing native economic inputs, V5 and V8 positions...", flush=True)
    known = rebuild_economic_known(read_native_features_from_raw(data), prices.date)
    v5 = run_vix_variants(prices, known, baseline_runs, baseline_dates, baseline_trades)
    v8 = run_economic_filters(prices, known, baseline_runs, baseline_dates, baseline_trades)
    progress("Reconstructing V3 T cash-reset accounts and separate H.10 accounts...", flush=True)
    v3 = reconstruct_v3(prices, stages={"T": V3_STAGES["T"]})
    h10 = reconstruct_h10(data)
    payload = deepcopy(archive)
    payload["version"] = archive["version"] + "-SIMPLE-INITIAL-CAPITAL"
    payload["roundingUsd"] = 0.0
    payload["accounting"] = {
        "method": ACCOUNTING_METHOD, "initialCapitalUsd": capital,
        "dailyPnlFormula": "C0 * held_position * (price_t / price_previous - 1)",
        "equityFormula": "C0 + sum(daily_pnl)",
        "tradeReturnFormula": "sum(daily_strategy_return over the original trade segment)",
        "annualization": "simple total return divided by elapsed calendar years (365.25 days)",
        "drawdown": "primary: dollars below running peak divided by initial capital, in percentage points",
        "costs": "gross spot-price PnL; no transaction costs, financing, carry or cash interest",
        "precision": "float64 daily returns and equity, without decimal rounding",
        "timing": "signals and execution unchanged; signal t, execution close t+1, first return t+2",
        "historicalScreen": "Compound-account historical selection was not carried over or recomputed.",
    }
    payload["archivedSources"] = payload.pop("sources")
    payload["archivedSourcesStatus"] = "Historical compound-output provenance only; not simple-account outputs or computational inputs."
    payload["sources"] = _source_hashes(root)
    payload["datasets"]["technical"]["note"] = (
        "V3 T accounts reset to cash in 2020 with indicators warmed on earlier prices. "
        "Daily positions and new simple-return trade ledgers are reconstructed locally. "
        "The original archive had no V3 trade ledgers; those new ledgers are not original archived evidence. "
        "These paths must not be spliced with continuous main accounts."
    )
    for dataset in payload["datasets"].values():
        dataset["note"] += " PnL uses fixed initial capital and daily arithmetic addition."
    metrics_rows, checks = [], []
    expected_counts = {"main": 471, "technical": 234, "h10": 207}
    counts = dict(Counter(account["ds"] for account in archive["accounts"]))
    if counts != expected_counts or len(archive["accounts"]) != 912:
        raise AssertionError(f"Unexpected source account coverage: {counts}")
    progress("Building simple paths and reconciling all 912 archived compound paths...", flush=True)
    for source, account in zip(archive["accounts"], payload["accounts"]):
        pair, rule, variant, dataset = (account[name] for name in ("pair", "rule", "variant", "ds"))
        if dataset == "technical":
            run = v3["runs"][("T", pair, rule if variant == "BASE" else variant)]
            dates = run["dates"]
        elif dataset == "h10":
            run = h10["runs"][(pair, rule)]
            dates = run["dates"]
        elif variant == "BASE":
            run = baseline_runs[(pair, rule)]
            dates = baseline_dates[pair]
        elif variant.startswith("V5_"):
            run = v5["runs"][(pair, rule, variant.removeprefix("V5_"))]
            dates = run["dates"]
        else:
            run = v8["runs"][(pair, rule, variant)]
            dates = run["dates"]
        dates = pd.DatetimeIndex(dates)
        reference_dates = pd.DatetimeIndex(pd.to_datetime(archive["calendars"][account["cal"]], unit="D"))
        locations = dates.get_indexer(reference_dates)
        if np.any(locations < 0) or not np.all(np.diff(locations) == 1):
            raise AssertionError(f"Calendar mismatch: {account['id']}")
        full_returns = np.asarray(run["gross_return"], dtype=float)
        returns = full_returns[locations]
        if np.any(full_returns[:locations[0]] != 0) or returns[0] != 0 or source["equity"][0] != capital:
            raise AssertionError(f"Cash-reset/first-observation anchor mismatch: {account['id']}")
        rebuilt_compound = capital * np.asarray(run["equity"])[locations]
        compound_error = float(np.max(np.abs(rebuilt_compound - np.asarray(source["equity"], dtype=float))))
        equity = simple_equity(returns, capital)
        independent_dollars = capital + np.cumsum(capital * returns)
        dollar_path_error = float(np.max(np.abs(equity - independent_dollars)))
        increment_error = float(np.max(np.abs(np.diff(np.r_[capital, equity]) - capital * returns)))
        terminal_error = abs(float(equity[-1]) - capital * (1.0 + float(returns.sum())))
        trades, compound_ledger, segment_error = _rebuild_trades(run, dates, account, capital)
        trade_total_error = abs(sum(row[2] for row in trades) * capital / 100.0 - capital * float(returns.sum()))
        archived_trade_structure_match = None
        archived_trade_return_error = None
        if source["trades"] is not None:
            old_structure = [tuple(row[index] for index in (0, 1, 3, 4, 5)) for row in source["trades"]]
            new_structure = [tuple(row[index] for index in (0, 1, 3, 4, 5)) for row in trades]
            archived_trade_structure_match = old_structure == new_structure
            if len(source["trades"]) == len(compound_ledger):
                archived_trade_return_error = max(
                    (abs(float(old[2]) - float(new)) for old, new in zip(source["trades"], compound_ledger.return_pct)),
                    default=0.0,
                )
        account["equity"] = equity.tolist()
        account["daily_returns"] = returns.tolist()
        account["trades"] = trades
        account["quality_full"] = _trade_quality(trades, archive["flatTolerancePct"])
        account["source_metrics"] = simple_path_statistics(returns, reference_dates, account["anchor"], capital)
        account["accounting"] = ACCOUNTING_METHOD
        account["trade_quality_source"] = (
            "newly_reconstructed_V3_ledger_not_present_in_original_archive"
            if dataset == "technical" else "reconstructed_positions_and_original_episode_boundaries_simple_returns"
        )
        account["archived_screen_status"] = "not_applied_to_simple_account"
        quality = account["quality_full"]
        metrics_rows.append({
            "account_id": account["id"], "dataset": dataset, "pair": pair, "strategy": rule, "variant": variant,
            "accounting": ACCOUNTING_METHOD, "initial_capital_usd": capital,
            "start_date": reference_dates[0].strftime("%Y-%m-%d"), "end_date": reference_dates[-1].strftime("%Y-%m-%d"),
            "capital_anchor_date": pd.to_datetime(account["anchor"], unit="D").strftime("%Y-%m-%d"),
            "observations": len(returns), **account["source_metrics"],
            "eligible_closed_n": quality["closed"], "win_rate": quality["win"], "payoff_ratio": quality["payoff"],
            "expectancy_return_pct": quality["expectancy_return_pct"], "open_n": quality["open"],
            "trade_quality_source": account["trade_quality_source"],
        })
        check = {
            "account_id": account["id"], "dataset": dataset, "daily_observations": len(returns),
            "compound_reference_max_abs_error_usd": compound_error,
            "simple_dollars_first_path_max_abs_error_usd": dollar_path_error,
            "simple_daily_increment_max_abs_error_usd": increment_error,
            "simple_terminal_abs_error_usd": terminal_error,
            "simple_trade_segment_max_abs_error_usd": segment_error,
            "simple_all_trade_pnl_abs_error_usd": trade_total_error,
            "archived_trade_structure_match": archived_trade_structure_match,
            "archived_compound_trade_return_max_abs_error_pct": archived_trade_return_error,
            "new_v3_ledger": dataset == "technical",
        }
        check["pass"] = bool(
            compound_error <= float(archive["roundingUsd"]) + 1e-7
            and max(dollar_path_error, increment_error, terminal_error, segment_error, trade_total_error) < 1e-7
            and archived_trade_structure_match is not False
            and (archived_trade_return_error is None or archived_trade_return_error < 1e-8)
        )
        checks.append(check)
    checks_frame = pd.DataFrame(checks)
    failed = checks_frame.loc[~checks_frame["pass"]]
    summary = {
        "status": "passed" if failed.empty else "failed", "pass": failed.empty,
        "built_utc": datetime.now(timezone.utc).isoformat(), "accounting": payload["accounting"],
        "accounts": len(checks), "accounts_by_dataset": counts,
        "daily_observations": int(checks_frame.daily_observations.sum()),
        "all_original_account_ids_and_calendars_preserved": True,
        "all_compound_reference_paths_pass": bool((checks_frame.compound_reference_max_abs_error_usd <= archive["roundingUsd"] + 1e-7).all()),
        "compound_reference_max_abs_error_usd": float(checks_frame.compound_reference_max_abs_error_usd.max()),
        "compound_reference_tolerance_usd": float(archive["roundingUsd"]) + 1e-7,
        "simple_daily_increment_max_abs_error_usd": float(checks_frame.simple_daily_increment_max_abs_error_usd.max()),
        "simple_terminal_max_abs_error_usd": float(checks_frame.simple_terminal_abs_error_usd.max()),
        "simple_trade_segment_max_abs_error_usd": float(checks_frame.simple_trade_segment_max_abs_error_usd.max()),
        "simple_all_trade_pnl_max_abs_error_usd": float(checks_frame.simple_all_trade_pnl_abs_error_usd.max()),
        "archived_trade_ledgers_reconciled": int(checks_frame.archived_trade_structure_match.notna().sum()),
        "newly_reconstructed_v3_trade_ledgers": int(checks_frame.new_v3_ledger.sum()),
        "original_explorer_sha256": hashlib.sha256((root / "FX_PnL_Explorer.html").read_bytes()).hexdigest(),
        "historical_screen_status": "Not applied; compound-screen results were not transferred.",
        "failed_account_ids": failed.account_id.tolist(),
    }
    if len(failed):
        raise AssertionError(failed.to_dict("records"))
    payload["validation"] = {
        "all912CompoundPathsReconciled": True,
        "maxCompoundReferenceErrorUsd": summary["compound_reference_max_abs_error_usd"],
        "simpleAccountingValidated": True, "newV3TradeLedgers": 234,
    }
    return payload, pd.DataFrame(metrics_rows), checks_frame, summary


def write_simple_data(output_dir=None, root=None):
    root = Path(root).resolve() if root is not None else ROOT
    output = Path(output_dir) if output_dir is not None else root / "reproduced/simple_interest"
    payload, metrics, checks, summary = build_simple_payload(root)
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "explorer_payload.json.gz"
    with gzip.open(payload_path, "wt", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    metrics.to_csv(output / "account_metrics.csv", index=False)
    checks.to_csv(output / "reconciliation.csv", index=False)
    summary["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    summary["payload_bytes"] = payload_path.stat().st_size
    (output / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
