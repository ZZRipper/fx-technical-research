"""Small, fixed candidate registry and causal original-episode replay.

The caller supplies a truncated price calendar and aligned, contemporaneously
known features. This module never loads data, uses outcomes as features, chooses
thresholds, or restarts a rejected/exited technical episode. All return-based
exit rules use the actual signed daily spot-return path in simple-return units.
"""

from copy import deepcopy

import numpy as np
import pandas as pd

try:
    from .fx_baselines import RULES, fixed_execute
    from .fx_economics import _at_least, _predicate_state
except ImportError:
    from fx_baselines import RULES, fixed_execute
    from fx_economics import _at_least, _predicate_state


_TREND_FAMILIES = ["Momentum", "Trend following", "Breakout", "Ensemble"]
_MR_FAMILIES = ["Mean reversion"]


def specs():
    """Return fresh copies of 17 prespecified mechanisms; there is no grid."""
    def item(identifier, label, kind, families, parameters, mechanism):
        return {"id": identifier, "label": label, "kind": kind,
                "families": families, "parameters": parameters, "mechanism": mechanism}

    volatility = {"rule": "volatility_ratio", "maximum_ratio": 1.0}
    time20 = {"rule": "time_nonpositive", "minimum_observations": 20, "maximum_simple_return": 0.0}
    mr_trend = {"rule": "mr_efficiency", "maximum_efficiency": 0.4}
    candidates = [
        item("BASE", "Unfiltered technical baseline", "baseline", "all", {},
             "Preserve the original technical targets and account clock."),
        item("VOL_NOT_EXPANDING", "Entry volatility is not expanding", "entry", "all",
             {"entry": volatility}, "Avoid entry while short-horizon volatility exceeds its longer-horizon level."),
        item("VOL_NOT_SURGING", "Entry volatility is not surging", "entry", "all",
             {"entry": {"rule": "volatility_ratio", "maximum_ratio": 1.25}},
             "Avoid a pronounced short-horizon volatility increase."),
        item("VIX_RISING", "Reject elevated and rising VIX", "entry", "all",
             {"entry": {"rule": "vix_rising", "minimum_level": 20.0, "minimum_change_5": 3.0}},
             "Avoid entry during an already elevated and worsening observed risk state."),
        item("ENTRY_NOT_EXTENDED", "Entry move is not extended", "entry", _TREND_FAMILIES,
             {"entry": {"rule": "directional_extension", "lookback_observations": 20,
                        "maximum_volatility_units": 1.0, "annualization_observations": 252}},
             "Avoid following a directional move already larger than one 20-observation volatility unit."),
        item("MOM_ALIGNED_63", "Entry agrees with 63-observation direction", "entry", _TREND_FAMILIES,
             {"entry": {"rule": "momentum_alignment", "lookback_observations": 63,
                        "minimum_directional_return_pct_exclusive": 0.0}},
             "Avoid a short-horizon directional entry against the longer observed price move."),
        item("TREND_EFFICIENCY_63", "Require directional 63-observation path", "entry", _TREND_FAMILIES,
             {"entry": {"rule": "trend_efficiency", "minimum_efficiency": 0.2}},
             "Avoid trend entries when displacement is small relative to the traveled price path."),
        item("MR_NOT_TRENDING", "Avoid strongly directional mean-reversion entries", "entry", _MR_FAMILIES,
             {"entry": mr_trend}, "Avoid fading a strongly directional 63-observation price path."),
        item("MR_TIME_20", "Exit unrewarded mean reversion after 20 observations", "exit", _MR_FAMILIES,
             {"exit": time20}, "Limit persistence in a mean-reversion episode that has not made progress."),
        item("MR_TIME_40", "Exit unrewarded mean reversion after 40 observations", "exit", _MR_FAMILIES,
             {"exit": {"rule": "time_nonpositive", "minimum_observations": 40, "maximum_simple_return": 0.0}},
             "Allow more time for reversion while limiting persistently unrewarded exposure."),
        item("MR_ADVERSE_2VOL", "Exit mean reversion at two entry-volatility units of loss", "exit", _MR_FAMILIES,
             {"exit": {"rule": "adverse_entry_volatility", "loss_daily_volatility_units": 2.0,
                       "annualization_observations": 252}},
             "Limit adverse movement using volatility fixed at the original entry signal."),
        item("MOM_NO_PROGRESS_5", "Exit directional entry without progress after five observations", "exit", _TREND_FAMILIES,
             {"exit": {"rule": "time_nonpositive", "minimum_observations": 5, "maximum_simple_return": 0.0}},
             "Limit a directional episode that fails to generate early cumulative progress."),
        item("TRAIL_2VOL", "Trail earned gains by two entry-volatility units", "exit", "all",
             {"exit": {"rule": "trailing_entry_volatility", "activation_daily_volatility_units": 2.0,
                       "giveback_daily_volatility_units": 2.0, "annualization_observations": 252}},
             "After sufficient observed gains, limit giveback using volatility frozen at the entry signal."),
        item("VOL_TIME_20", "Stable-volatility entry and 20-observation reversion timeout", "entry_exit", _MR_FAMILIES,
             {"entry": volatility, "exit": time20}, "Combine entry volatility context with a no-progress time exit."),
        item("VOL_TREND_CONFLICT", "Stable-volatility nontrending reversion entry", "entry", _MR_FAMILIES,
             {"entry": {"rule": "all", "components": [volatility, mr_trend]}},
             "Avoid mean-reversion entries with expanding volatility or a strongly directional path."),
        item("RATE_USD_CONFLICT", "Reject the existing rate-change and dollar conflict", "entry", "all",
             {"entry": {"rule": "policy_usd", "directional_rate_change_maximum_bp": -10.0,
                        "directional_dxy_change_minimum_pct": 1.0}},
             "Retain the existing POLICY_USD predicate as a prespecified economic comparator."),
        item("COMMODITY_USD", "Reject the existing commodity and dollar conflict", "entry", "all",
             {"entry": {"rule": "commodity_usd", "applicable_pairs": ["AUD", "CAD", "NOK"],
                        "directional_commodity_change_maximum_pct_exclusive": 0.0,
                        "directional_dxy_change_minimum_pct": 1.0}},
             "For AUD/CAD/NOK, avoid a commodity move against the trade alongside a conflicting dollar move."),
    ]
    return [deepcopy(candidate) for candidate in candidates]


def _number(observation, name):
    value = observation.get(name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _entry_state(observation, direction, pair, parameters):
    if parameters is None:
        return "KEEP"
    rule = parameters["rule"]
    if rule == "all":
        states = [_entry_state(observation, direction, pair, component) for component in parameters["components"]]
        if "REJECT" in states:
            return "REJECT"
        return "UNKNOWN" if "UNKNOWN" in states else "KEEP"
    if rule in ("policy_usd", "commodity_usd"):
        required = ("K_spread_d20_bp", "K_dxy_d20_pct", "K_copper_d20_pct", "K_brent_d20_pct")
        known = {name: _number(observation, name) for name in required}
        return _predicate_state(known, pair, direction,
                                "POLICY_USD" if rule == "policy_usd" else "COMMODITY_USD")
    if rule == "volatility_ratio":
        value = _number(observation, "vol_ratio_20_63")
        if not np.isfinite(value):
            return "UNKNOWN"
        reject = value > parameters["maximum_ratio"]
    elif rule == "vix_rising":
        level, change = (_number(observation, key) for key in ("K_vix", "K_vix_d5"))
        if not np.isfinite([level, change]).all():
            return "UNKNOWN"
        reject = _at_least(level, parameters["minimum_level"]) and _at_least(change, parameters["minimum_change_5"])
    elif rule == "directional_extension":
        move, volatility = (_number(observation, key) for key in ("spot_return_20_pct", "realized_vol_20_pct"))
        if not np.isfinite([move, volatility]).all() or volatility <= 0:
            return "UNKNOWN"
        scale = volatility * np.sqrt(parameters["lookback_observations"] / parameters["annualization_observations"])
        reject = direction * move / scale > parameters["maximum_volatility_units"]
    elif rule == "momentum_alignment":
        value = _number(observation, "spot_return_63_pct")
        if not np.isfinite(value):
            return "UNKNOWN"
        reject = direction * value <= parameters["minimum_directional_return_pct_exclusive"]
    elif rule in ("trend_efficiency", "mr_efficiency"):
        value = _number(observation, "efficiency_63")
        if not np.isfinite(value):
            return "UNKNOWN"
        reject = (value < parameters["minimum_efficiency"] if rule == "trend_efficiency"
                  else value > parameters["maximum_efficiency"])
    else:
        raise ValueError(f"Unknown entry predicate {rule!r}.")
    return "REJECT" if reject else "KEEP"


def _exit_trigger(parameters, age, cumulative_return, running_peak, entry_daily_volatility):
    if parameters is None:
        return False
    rule = parameters["rule"]
    if rule == "time_nonpositive":
        return age >= parameters["minimum_observations"] and cumulative_return <= parameters["maximum_simple_return"]
    if not np.isfinite(entry_daily_volatility) or entry_daily_volatility <= 0:
        return False
    if rule == "adverse_entry_volatility":
        return cumulative_return <= -parameters["loss_daily_volatility_units"] * entry_daily_volatility
    if rule == "trailing_entry_volatility":
        return (running_peak >= parameters["activation_daily_volatility_units"] * entry_daily_volatility
                and running_peak - cumulative_return >= parameters["giveback_daily_volatility_units"] * entry_daily_volatility)
    raise ValueError(f"Unknown exit predicate {rule!r}.")


def _validate_inputs(prices, dates, base_run, features):
    prices = np.asarray(prices, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if prices.ndim != 1 or not len(prices) or len(prices) != len(dates) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must be a nonempty positive finite vector aligned with dates.")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or dates.tz is not None:
        raise ValueError("Dates must be unique increasing timezone-naive observations.")
    if not isinstance(features, pd.DataFrame) or len(features) != len(dates):
        raise ValueError("Features must have one row per supplied price observation.")
    if "date" in features:
        feature_dates = pd.DatetimeIndex(pd.to_datetime(features["date"]))
    elif isinstance(features.index, pd.DatetimeIndex):
        feature_dates = features.index
    elif features.index.equals(pd.RangeIndex(len(features))):
        feature_dates = dates
    else:
        raise ValueError("Features require an aligned date column, DatetimeIndex, or default positional index.")
    if not feature_dates.equals(dates):
        raise ValueError("Feature dates differ from the supplied price calendar.")
    target = np.asarray(base_run["target"])
    executed = np.asarray(base_run["executed"])
    if target.shape != prices.shape or executed.shape != prices.shape or not np.isin(target, [-1, 0, 1]).all():
        raise ValueError("Base targets/executions must be aligned signed unit positions.")
    if not np.array_equal(executed, np.r_[0, target[:-1]]):
        raise ValueError("Base execution must implement the existing next-close clock.")
    if "held" in base_run and not np.array_equal(base_run["held"], np.r_[0, executed[:-1]]):
        raise ValueError("Base held positions do not match prior-close execution.")
    return prices, dates, executed.astype(np.int8)


def run_candidate(prices, dates, base_run, features, spec, pair="", strategy=""):
    """Replay one fixed candidate, returning a normal run and original-episode decisions.

    Signal i acts at close i+1. A time exit observed after k earned returns is
    therefore executed one close later and still earns that execution-day return.
    Missing entry/entry-volatility inputs retain the base episode as UNKNOWN.
    Open exit indices are nullable; an observed last-day exit signal can be
    pending without fabricating a liquidation. Existing warmup/cash-reset targets
    are preserved. A family-inapplicable candidate is an unchanged baseline.
    """
    registry = {candidate["id"]: candidate for candidate in specs()}
    identifier = spec.get("id")
    if identifier not in registry:
        raise ValueError(f"Unknown candidate {identifier!r}.")
    specification = registry[identifier]
    if "parameters" in spec and spec["parameters"] != specification["parameters"]:
        raise ValueError("Candidate parameters differ from the fixed registry.")
    prices, dates, original_executed = _validate_inputs(prices, dates, base_run, features)
    if pair and "pair" in features and not features["pair"].eq(pair).all():
        raise ValueError("Feature rows contain a different or missing currency.")
    if strategy and strategy not in RULES:
        raise ValueError(f"Unknown technical strategy {strategy!r}.")
    applicable = (specification["families"] == "all" or not strategy
                  or RULES[strategy][1] in specification["families"])
    if identifier == "COMMODITY_USD" and pair not in specification["parameters"]["entry"]["applicable_pairs"]:
        applicable = False
    parameters = specification["parameters"] if applicable else {}
    entry_parameters, exit_parameters = parameters.get("entry"), parameters.get("exit")
    original_targets = np.asarray(base_run["target"])
    spot_returns = np.r_[0., prices[1:] / prices[:-1] - 1.]
    executed = np.zeros(len(dates), dtype=np.int8)
    changes = np.flatnonzero(original_executed != np.r_[0, original_executed[:-1]])
    records = []
    pending_last_target = False
    for offset, entry in enumerate(changes):
        direction = int(original_executed[entry])
        if direction == 0:
            continue
        original_closed = offset + 1 < len(changes)
        original_end = int(changes[offset + 1]) if original_closed else len(dates)
        signal = int(entry) - 1
        observation = features.iloc[signal]
        state = (_entry_state(observation, direction, pair, entry_parameters)
                 if applicable else "NOT_APPLICABLE")
        entry_daily_volatility = np.nan
        exit_state = "NOT_TRIGGERED" if exit_parameters else "NOT_APPLICABLE"
        if exit_parameters and exit_parameters["rule"] != "time_nonpositive":
            annual_volatility = _number(observation, "realized_vol_20_pct")
            if np.isfinite(annual_volatility) and annual_volatility > 0:
                entry_daily_volatility = annual_volatility / 100. / np.sqrt(exit_parameters["annualization_observations"])
            else:
                exit_state = "UNKNOWN"
                if state == "KEEP":
                    state = "UNKNOWN"
        accepted = state != "REJECT"
        final_end, risk_signal = original_end, None
        cumulative, running_peak = 0., 0.
        if accepted:
            # A signal immediately before the original exit cannot exit earlier.
            monitor_stop = original_end - 1 if original_closed else len(dates)
            if exit_parameters:
                for monitor in range(int(entry) + 1, monitor_stop):
                    cumulative += direction * spot_returns[monitor]
                    running_peak = max(running_peak, cumulative)
                    # Even at a truncated final quote, the technical target is
                    # already known. Its next-close exit/reversal has priority
                    # over a risk exit that could only happen at that same close.
                    if original_targets[monitor] != direction:
                        break
                    if _exit_trigger(exit_parameters, monitor - int(entry), cumulative,
                                     running_peak, entry_daily_volatility):
                        risk_signal = monitor
                        final_end = monitor + 1
                        exit_state = "PENDING" if final_end == len(dates) else "TRIGGERED"
                        pending_last_target |= final_end == len(dates)
                        break
            executed[int(entry):final_end] = direction
        else:
            exit_state = "BLOCKED"
        actual_closed = accepted and final_end < len(dates)
        risk_pending = risk_signal is not None and final_end == len(dates)
        records.append({
            "pair": pair, "strategy": strategy, "candidate_id": identifier,
            "original_episode": len(records), "accepted": bool(accepted), "state": state,
            "signal_idx": signal, "entry_idx": int(entry),
            "original_exit_idx": original_end if original_closed else pd.NA,
            "actual_entry_idx": int(entry) if accepted else pd.NA,
            "actual_exit_idx": final_end if actual_closed else pd.NA,
            "risk_signal_idx": risk_signal if risk_signal is not None else pd.NA,
            "signal_date": dates[signal], "entry_date": dates[int(entry)],
            "original_exit_date": dates[original_end] if original_closed else pd.NaT,
            "actual_exit_date": dates[final_end] if actual_closed else pd.NaT,
            "risk_signal_date": dates[risk_signal] if risk_signal is not None else pd.NaT,
            "position": direction, "original_open": not original_closed,
            "actual_open": bool(accepted and not actual_closed),
            "early_exit": bool(risk_signal is not None and actual_closed),
            "risk_exit_pending": bool(risk_pending), "exit_state": exit_state,
            "entry_daily_volatility": entry_daily_volatility,
            "risk_signal_age": risk_signal - int(entry) if risk_signal is not None else pd.NA,
            "simple_return_at_risk_signal": cumulative if risk_signal is not None else np.nan,
            "running_peak_at_risk_signal": running_peak if risk_signal is not None else np.nan,
        })
    target = np.r_[executed[1:], executed[-1]].astype(np.int8)
    terminal_original = int(original_targets[-1])
    if terminal_original != int(original_executed[-1]):
        # A final-quote instruction is observable even though its execution lies
        # outside the supplied sample. A new original episode gets a fresh gate;
        # a rejected/previously exited episode cannot lock that new episode.
        if terminal_original == 0:
            target[-1] = 0
        else:
            terminal_state = (_entry_state(features.iloc[-1], terminal_original, pair, entry_parameters)
                              if applicable else "NOT_APPLICABLE")
            target[-1] = 0 if terminal_state == "REJECT" else terminal_original
    elif pending_last_target:
        target[-1] = 0
    if identifier == "BASE" or not applicable:
        target = np.asarray(base_run["target"]).copy()
    run = fixed_execute(prices, target, cost_bp=float(base_run.get("one_way_cost_bp", 0.)),
                        capital=float(base_run.get("initial_capital_usd", 100000.)))
    run["dates"] = dates
    run["candidate_id"] = identifier
    if "account_start_date" in base_run:
        run["account_start_date"] = base_run["account_start_date"]
    columns = ["pair", "strategy", "candidate_id", "original_episode", "accepted", "state",
               "signal_idx", "entry_idx", "original_exit_idx", "actual_entry_idx", "actual_exit_idx",
               "risk_signal_idx", "signal_date", "entry_date", "original_exit_date", "actual_exit_date",
               "risk_signal_date", "position", "original_open", "actual_open", "early_exit",
               "risk_exit_pending", "exit_state", "entry_daily_volatility", "risk_signal_age",
               "simple_return_at_risk_signal", "running_peak_at_risk_signal"]
    decisions = pd.DataFrame(records, columns=columns)
    for name in ("original_exit_idx", "actual_entry_idx", "actual_exit_idx", "risk_signal_idx", "risk_signal_age"):
        decisions[name] = decisions[name].astype("Int64")
    for name in ("accepted", "original_open", "actual_open", "early_exit", "risk_exit_pending"):
        decisions[name] = decisions[name].astype(bool)
    return run, decisions
