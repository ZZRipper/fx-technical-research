"""Causal rejection of one fixed entry state, without outcome-dependent fitting.

The original target episodes define every decision, including a final signal
whose execution is still pending. Rejection never changes technical exit timing
or restarts the episode. UNKNOWN is retained. State edges come only from the
fixed fx_is_states registry; this module does not read files or choose states.
"""

import numpy as np
import pandas as pd

try:
    from .fx_baselines import fixed_execute
    from .fx_is_states import STATE_BINS, entry_features, state_codes
except ImportError:
    from fx_baselines import fixed_execute
    from fx_is_states import STATE_BINS, entry_features, state_codes


_FIXED_BINS = {name: tuple(edges) for name, edges in STATE_BINS.items()}
_FEATURE_INPUTS = (
    "date", "realized_vol_20_pct", "spot_return_20_pct", "distance_ma_60_pct",
    "K_spread_bp", "K_spread_d20_bp", "K_dxy_d20_pct", "K_copper_d20_pct", "K_brent_d20_pct",
)


def run_state_filter(prices, dates, base_run, features, spec, pair="", strategy=""):
    """Return (fixed_execute run, original-target-episode decisions).

    ``spec`` requires feature and state (integer 0..3); id is optional. Feature
    values are recomputed by entry_features at the original signal timestamp,
    including only already-earned BASE returns. Optional edges/bins metadata
    must exactly match the registry; arbitrary threshold overrides are rejected.
    Entry indices refer to the original scheduled entry; actual_entry_idx is
    populated only for an accepted entry executed inside the supplied sample.
    """
    if not isinstance(spec, dict) or spec.get("feature") not in _FIXED_BINS:
        raise ValueError("A registered state feature is required.")
    feature, rejected_state = spec["feature"], spec.get("state")
    if isinstance(rejected_state, (bool, np.bool_)) or not isinstance(rejected_state, (int, np.integer)) or not 0 <= rejected_state <= 3:
        raise ValueError("Rejected state must be an integer 0..3; UNKNOWN=-1 cannot be rejected.")
    edges = _FIXED_BINS[feature]
    for key in ("edges", "bins"):
        if key in spec and tuple(spec[key]) != edges:
            raise ValueError("State bin edges differ from the fixed registry.")
    if any(key in spec for key in ("threshold", "thresholds")):
        raise ValueError("Arbitrary threshold overrides are not supported.")
    prices = np.asarray(prices, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if prices.ndim != 1 or not len(prices) or len(prices) != len(dates) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must be a positive finite vector aligned with dates.")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or dates.tz is not None:
        raise ValueError("Dates must be unique increasing timezone-naive observations.")
    if not isinstance(features, pd.DataFrame) or len(features) != len(dates):
        raise ValueError("Features must have one row per supplied observation.")
    missing = set(_FEATURE_INPUTS).difference(features.columns)
    if missing:
        raise ValueError(f"Missing causal feature inputs: {sorted(missing)}")
    if not pd.DatetimeIndex(pd.to_datetime(features.date)).equals(dates):
        raise ValueError("Feature dates differ from the supplied price calendar.")
    if pair and "pair" in features and not features.pair.eq(pair).all():
        raise ValueError("Feature rows contain a different or missing currency.")
    original = np.asarray(base_run["target"])
    if original.shape != prices.shape or not np.isin(original, [-1, 0, 1]).all():
        raise ValueError("BASE targets must be aligned signed unit positions.")
    executed = np.r_[0, original[:-1]]
    held = np.r_[0, executed[:-1]]
    expected_return = held * np.r_[0., prices[1:] / prices[:-1] - 1.]
    for name, expected in (("executed", executed), ("held", held), ("gross_return", expected_return)):
        values = np.asarray(base_run[name])
        if values.shape != expected.shape or not np.allclose(values, expected, atol=1e-12, rtol=0):
            raise ValueError(f"BASE {name} does not match the supplied next-close price/target path.")
    if float(base_run.get("one_way_cost_bp", 0.)) != 0.:
        raise ValueError("State research requires the zero-cost BASE reference.")
    changes = np.flatnonzero(original != np.r_[0, original[:-1]])
    episodes = [(int(start), int(changes[i + 1]) if i + 1 < len(changes) else len(dates))
                for i, start in enumerate(changes) if original[start] != 0]
    starts = np.array([start for start, _ in episodes], dtype=int)
    synthetic = pd.DataFrame({"signal_idx": starts, "signal_date": dates[starts], "position": original[starts]})
    values = entry_features(features, base_run, synthetic, pair)
    if feature not in values:
        raise ValueError(f"Selected feature {feature!r} is missing from causal entry inputs.")
    numeric = pd.to_numeric(values[feature], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    codes = state_codes(numeric, edges)
    target = original.astype(np.int8, copy=True)
    identifier = str(spec.get("id", f"{feature}__REJECT_STATE_{rejected_state}"))
    rows = []
    for episode_number, ((start, end), value, code) in enumerate(zip(episodes, numeric, codes)):
        accepted = int(code) != int(rejected_state)
        if not accepted:
            target[start:end] = 0
        entry = start + 1
        pending = entry >= len(dates)
        rows.append({
            "pair": pair, "strategy": strategy, "candidate_id": identifier,
            "original_episode": episode_number, "feature": feature, "rejected_state": int(rejected_state),
            "signal_idx": start, "signal_date": dates[start],
            "entry_idx": entry if not pending else pd.NA,
            "entry_date": dates[entry] if not pending else pd.NaT,
            "actual_entry_idx": entry if accepted and not pending else pd.NA,
            "position": int(original[start]), "feature_value": float(value), "state": int(code),
            "accepted": bool(accepted), "status": "UNKNOWN" if code == -1 else "KEEP" if accepted else "REJECT",
            "entry_pending": bool(pending), "original_end_signal_idx": end if end < len(dates) else pd.NA,
            "original_end_signal_date": dates[end] if end < len(dates) else pd.NaT,
        })
    run = fixed_execute(prices, target, cost_bp=0., capital=float(base_run.get("initial_capital_usd", 100000.)))
    run["dates"] = dates
    run["candidate_id"] = identifier
    if "account_start_date" in base_run:
        run["account_start_date"] = base_run["account_start_date"]
    columns = ["pair", "strategy", "candidate_id", "original_episode", "feature", "rejected_state",
               "signal_idx", "signal_date", "entry_idx", "entry_date", "actual_entry_idx", "position",
               "feature_value", "state", "accepted", "status", "entry_pending",
               "original_end_signal_idx", "original_end_signal_date"]
    decisions = pd.DataFrame(rows, columns=columns)
    for name in ("entry_idx", "actual_entry_idx", "original_end_signal_idx"):
        decisions[name] = decisions[name].astype("Int64")
    for name in ("accepted", "entry_pending"):
        decisions[name] = decisions[name].astype(bool)
    return run, decisions
