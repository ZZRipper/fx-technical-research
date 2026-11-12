"""Reconstructed V5 VIX-only episode filters and applied-state attribution."""

import numpy as np
import pandas as pd

try:
    from .fx_baselines import CURRENCIES, STRATEGIES, account_metrics, fixed_execute, trade_ledger
    from .fx_economics import _at_least
except ImportError:
    from fx_baselines import CURRENCIES, STRATEGIES, account_metrics, fixed_execute, trade_ledger
    from fx_economics import _at_least


def _rising_high_vix(observation):
    level = observation.get("K_vix", np.nan)
    change = observation.get("K_vix_d5", np.nan)
    if not np.isfinite([level, change]).all():
        return "UNKNOWN"
    return "REJECT" if _at_least(level, 30) and _at_least(change, 3) else "KEEP"


def run_vix_variants(prices, known, baseline_runs, price_dates, baseline_trades):
    """Replay 36 V5 accounts using original episodes and next-close exits."""
    metrics_rows, ledger_frames, decisions, runs = [], [], [], {}
    for pair in CURRENCIES:
        dates = pd.DatetimeIndex(price_dates[pair] if isinstance(price_dates, dict) else price_dates)
        values = prices.loc[prices["pair"].eq(pair)].set_index("date")["price_USD_per_foreign"].reindex(dates).to_numpy(float)
        observations = known.loc[known["pair"].eq(pair)].set_index("date").reindex(dates).to_dict("records")
        for strategy in ("BB_MR_20", "RSI_MR_14"):
            episodes = baseline_trades.loc[baseline_trades["pair"].eq(pair) & baseline_trades["strategy"].eq(strategy)]
            baseline = baseline_runs[(pair, strategy)]
            baseline_return = float((baseline["equity"][-1] - 1.0) * 100.0)
            for variant in ("ENTRY_ONLY", "RISK_EXIT"):
                executed = np.zeros(len(dates), dtype=np.int8)
                blocked, early_exits, unknown = 0, 0, 0
                by_entry = {}
                for original_episode, episode in enumerate(episodes.itertuples(index=False)):
                    entry = int(episode.entry_idx)
                    original_end = int(episode.exit_idx) if episode.closed else len(dates)
                    final_end = original_end
                    state = _rising_high_vix(observations[int(episode.signal_idx)])
                    risk_signal = None
                    if state == "REJECT":
                        blocked += 1
                    else:
                        unknown += int(state == "UNKNOWN")
                        if variant == "RISK_EXIT":
                            for monitor_index in range(entry, min(original_end - 1, len(dates) - 1)):
                                if _rising_high_vix(observations[monitor_index]) == "REJECT":
                                    final_end = monitor_index + 1
                                    risk_signal = dates[monitor_index]
                                    early_exits += 1
                                    break
                        executed[entry:final_end] = int(episode.position)
                    decision = {
                        "pair": pair, "base_strategy": strategy, "variant": variant,
                        "original_episode": original_episode, "signal_date": dates[int(episode.signal_idx)],
                        "entry_date": dates[entry], "position": int(episode.position),
                        "entry_state": state, "accepted": state != "REJECT",
                        "early_risk_exit": risk_signal is not None, "risk_signal_date": risk_signal,
                        "execution_exit_date": dates[final_end] if final_end < len(dates) else pd.NaT,
                        "original_exit_date": dates[original_end] if original_end < len(dates) else pd.NaT,
                    }
                    decisions.append(decision)
                    by_entry[entry] = decision
                targets = np.concatenate((executed[1:], executed[-1:]))
                run = fixed_execute(values, targets)
                run["dates"] = dates
                run["account_start_date"] = "2005-01-01"
                runs[(pair, strategy, variant)] = run
                ledger = trade_ledger(run, dates, pair, strategy)
                ledger["base_strategy"] = strategy
                ledger["variant"] = variant
                ledger["original_episode"] = ledger["entry_idx"].map(lambda index: by_entry[index]["original_episode"])
                ledger["early_risk_exit"] = ledger["entry_idx"].map(lambda index: by_entry[index]["early_risk_exit"])
                ledger_frames.append(ledger)
                stats = account_metrics(run, dates, ledger)
                metrics_rows.append({**stats, "pair": pair, "base_strategy": strategy, "variant": variant,
                                     "blocked_base_trades": blocked, "early_risk_exits": early_exits,
                                     "unknown_entry_n": unknown, "baseline_return_pct": baseline_return,
                                     "return_difference_pp": stats["total_return_pct"] - baseline_return})
    return {"metrics": pd.DataFrame(metrics_rows), "trades": pd.concat(ledger_frames, ignore_index=True),
            "runs": runs, "episode_decisions": pd.DataFrame(decisions)}


def conditional_daily(known, baseline_runs, price_dates):
    """Attribute active returns to K shifted two FX observations, the A clock.

    years_covered counts years with that state, as in the archived table;
    active_years_covered counts years with an actual position in that state.
    Attribution retains original account equity and includes UNKNOWN states.
    """
    records = []
    applied_states = {}
    for (pair, strategy), run in baseline_runs.items():
        if strategy not in STRATEGIES:
            continue
        dates = pd.DatetimeIndex(price_dates[pair] if isinstance(price_dates, dict) else price_dates)
        if pair not in applied_states:
            applied = known.loc[known["pair"].eq(pair)].set_index("date").reindex(dates).shift(2)
            levels = np.select([applied["K_vix"].lt(20), applied["K_vix"].lt(30), applied["K_vix"].ge(30)], ["L", "M", "H"], default="UNKNOWN")
            changes = np.select([applied["K_vix_d5"].ge(3), applied["K_vix_d5"].le(-3), applied["K_vix_d5"].notna()], ["U", "D", "F"], default="UNKNOWN")
            states = np.array(["UNKNOWN" if "UNKNOWN" in (level, change) else f"{level}_{change}" for level, change in zip(levels, changes)])
            applied_states[pair] = states
        states = applied_states[pair]
        evaluation = dates >= pd.Timestamp("2005-01-01")
        for state in pd.unique(states[evaluation]):
            state_selection = evaluation & (states == state)
            active = state_selection & (np.asarray(run["held"]) != 0)
            pnl = np.asarray(run["net_pnl_usd"])[active]
            returns = np.asarray(run["net_return"])[active]
            absolute_pnl = float(np.abs(pnl).sum())
            records.append({
                "pair": pair, "strategy": strategy, "lens": "vix9", "timing": "A", "state": state,
                "state_days": int(state_selection.sum()), "active_days": int(active.sum()),
                "years_covered": int(dates[state_selection].year.nunique()),
                "active_years_covered": int(dates[active].year.nunique()),
                "mean_return_bp_per_active_day": float(returns.mean() * 10000.0) if len(returns) else np.nan,
                "original_account_pnl_usd": float(pnl.sum()),
                "max_abs_day_share_pct": float(np.abs(pnl).max() / absolute_pnl * 100.0) if absolute_pnl > 0 else np.nan,
            })
    return pd.DataFrame(records)


def verify_vix_variants(result, root=None):
    """Validate V5 curves, quality and saved notebook metrics independently."""
    try:
        from .fx_evidence import load_explorer, explorer_account_dates, saved_notebook_table
    except ImportError:
        from fx_evidence import load_explorer, explorer_account_dates, saved_notebook_table

    archive = load_explorer(root)
    lookup = {(account["pair"], account["rule"], account["variant"]): account for account in archive["accounts"] if account["ds"] == "main"}
    checks = []
    for metric in result["metrics"].itertuples(index=False):
        identity = (metric.pair, metric.base_strategy, metric.variant)
        account = lookup[(metric.pair, metric.base_strategy, "V5_" + metric.variant)]
        run = result["runs"][identity]
        actual = pd.Series(np.asarray(run["equity"]) * archive["capital"], index=run["dates"]).reindex(explorer_account_dates(account, root)).to_numpy()
        expected = np.asarray(account["equity"], dtype=float)
        error = float(np.max(np.abs(actual - expected)))
        checks.append({"check": "V5 daily " + " ".join(identity), "rows": len(actual), "max_abs_error": error,
                       "pass": bool(np.isfinite(actual).all() and error <= 1.1e-6), "reference_source": account["id"]})
        for field, value in (("total_return_pct", account["source_metrics"]["total_return_pct"]),
                             ("max_drawdown_pct", account["source_metrics"]["max_drawdown_pct"]),
                             ("eligible_closed_n", account["quality_full"]["closed"]),
                             ("win_rate", account["quality_full"]["win"]),
                             ("payoff_ratio", account["quality_full"]["payoff"])):
            error = abs(getattr(metric, field) - value)
            checks.append({"check": "V5 " + field + " " + " ".join(identity), "rows": 1,
                           "max_abs_error": float(error), "pass": bool(error <= 1e-7), "reference_source": account["id"]})
    expected = saved_notebook_table(26, root=root)
    merged = result["metrics"].merge(expected, on=["pair", "base_strategy", "variant"], suffixes=("_new", "_reference"), validate="one_to_one")
    for field in ("blocked_base_trades", "early_risk_exits", "baseline_return_pct", "total_return_pct", "return_difference_pp", "max_drawdown_pct"):
        error = float((merged[field + "_new"] - merged[field + "_reference"]).abs().max())
        checks.append({"check": "V5 saved table " + field, "rows": len(merged), "max_abs_error": error,
                       "pass": bool(len(merged) == 36 and error <= 5.1e-7), "reference_source": "notebook cell 26"})
    return pd.DataFrame(checks)
