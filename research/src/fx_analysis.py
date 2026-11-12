"""Recompute episode attribution and historical screens from reconstructed runs.

These are descriptive, zero-cost historical diagnostics. The original acceptance
protocol is unavailable; the screen below states its reconstructed conditions
explicitly and does not certify restoration of the missing frozen protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ACCOUNT_KEYS = ["pair", "base_strategy", "variant"]
EPISODE_KEYS = ACCOUNT_KEYS + ["original_episode"]
SCREEN_SPECIFICATION = {
    "win_rate_improvement_percentage_points": 2.0,
    "payoff_ratio_relative_improvement": 0.10,
    "return_improvement_percentage_points": 5.0,
    "maximum_drawdown_deterioration_percentage_points": 2.0,
    "minimum_retained_closed_fraction": 0.50,
    "minimum_closed_trades": 30,
    "minimum_positive_calendar_blocks": 3,
    "review_condition": "Win rate, payoff ratio and return must each be no worse than BASE in 2015 onward.",
    "source": "Delivered notebook cell 31; review and calendar conditions reconstructed explicitly.",
    "original_frozen_protocol_available": False,
    "new_untouched_out_of_sample": False,
}


def _validate_result(result):
    required = {"metrics", "trades", "runs", "phase", "calendar", "episode_decisions"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Missing reconstructed research results: {sorted(missing)}")
    if result["metrics"].duplicated(ACCOUNT_KEYS).any():
        raise ValueError("Duplicate account metrics")
    if result["episode_decisions"].duplicated(EPISODE_KEYS).any():
        raise ValueError("Duplicate original episode decisions")


def episode_effects(result, include_base=False):
    """Attribute log terminal wealth changes to complete original holding episodes.

    An episode entered at close i first earns the return ending at i+1; its
    original exit-close return belongs to that episode. Rejected entries remain
    zero-return for the entire episode, and risk exits keep the original end for
    comparison. Each effect is additive in log wealth, not in dollar profit.
    """
    _validate_result(result)
    decisions = result["episode_decisions"]
    baseline_ledger = result["trades"].loc[result["trades"].variant.eq("BASE")]
    if baseline_ledger.duplicated(["pair", "base_strategy", "original_episode"]).any():
        raise ValueError("Duplicate baseline episode ledger rows")
    baseline_groups = {
        key: group.set_index("original_episode")
        for key, group in baseline_ledger.groupby(["pair", "base_strategy"], sort=False)
    }
    records = []
    for key, account_decisions in decisions.groupby(ACCOUNT_KEYS, sort=False):
        pair, strategy, variant = key
        if variant == "BASE" and not include_base:
            continue
        baseline = result["runs"][(pair, strategy, "BASE")]
        run = result["runs"][key]
        dates = pd.DatetimeIndex(run["dates"])
        if not dates.equals(pd.DatetimeIndex(baseline["dates"])):
            raise ValueError(f"Calendar mismatch for {key}")
        baseline_returns = np.asarray(baseline["net_return"], dtype=float)
        variant_returns = np.asarray(run["net_return"], dtype=float)
        if (
            len(baseline_returns) != len(dates) or len(variant_returns) != len(dates)
            or not np.isfinite(baseline_returns).all() or not np.isfinite(variant_returns).all()
            or (baseline_returns <= -1).any() or (variant_returns <= -1).any()
        ):
            raise ValueError(f"Invalid returns for log-wealth attribution: {key}")
        baseline_logs = np.log1p(baseline_returns)
        variant_logs = np.log1p(variant_returns)
        ledger = baseline_groups[(pair, strategy)]
        if set(account_decisions.original_episode) != set(ledger.index):
            raise ValueError(f"Original episode coverage mismatch for {key}")
        for decision in account_decisions.to_dict("records"):
            episode = ledger.loc[decision["original_episode"]]
            entry_index = int(episode["entry_idx"])
            exit_index = int(episode["exit_idx"]) if bool(episode["closed"]) else len(dates) - 1
            if not 0 <= entry_index <= exit_index < len(dates):
                raise ValueError(f"Invalid original holding interval for {key}")
            baseline_log_return = float(baseline_logs[entry_index + 1:exit_index + 1].sum())
            effective_log_return = float(variant_logs[entry_index + 1:exit_index + 1].sum())
            effect = effective_log_return - baseline_log_return
            changed = not bool(decision["accepted"]) or bool(decision["early_risk_exit"])
            records.append({
                **decision, "base_entry_idx": entry_index, "base_exit_idx": exit_index,
                "base_closed": bool(episode["closed"]), "changed": changed,
                "base_trade_return_pct": float(np.expm1(baseline_log_return) * 100.0),
                "effective_trade_return_pct": float(np.expm1(effective_log_return) * 100.0),
                "base_log_return": baseline_log_return,
                "effective_log_return": effective_log_return,
                "log_wealth_effect": effect,
                "numerically_flat_effect": abs(effect) <= 1e-12,
                "evidence_status": "recomputed_from_daily_returns",
            })
    return pd.DataFrame(records)


def episode_concentration(result, effects=None):
    """Measure dependence on the largest beneficial original trading episodes.

    Counts treat absolute log effects at most 1e-12 as numerical zero; the raw
    signs are also retained because historical counts can classify floating
    roundoff as a gain or loss. Shares and counterfactual wealth use unmodified
    effects, so this display convention does not change their calculations.
    """
    effects = episode_effects(result) if effects is None else effects
    metrics = result["metrics"].set_index(ACCOUNT_KEYS)
    records = []
    for key, episodes in effects.groupby(ACCOUNT_KEYS, sort=False):
        changed = episodes.loc[episodes.changed]
        values = changed.log_wealth_effect.to_numpy(dtype=float)
        positive = np.sort(values[values > 0])[::-1]
        negative = values[values < 0]
        gross_positive = float(positive.sum())
        largest = float(positive[0]) if len(positive) else 0.0
        pair, strategy, variant = key
        variant_nav = float(result["runs"][key]["equity"][-1])
        baseline_nav = float(result["runs"][(pair, strategy, "BASE")]["equity"][-1])
        effect_sum = float(episodes.log_wealth_effect.sum())
        total_effect = float(np.log(variant_nav / baseline_nav))
        error = abs(effect_sum - total_effect)
        if error > 1e-9:
            raise ValueError(f"Episode effects do not reconcile to terminal wealth for {key}: {error}")
        records.append({
            "pair": pair, "base_strategy": strategy, "variant": variant,
            "changed_episodes": len(changed),
            "beneficial_n": int((values > 1e-12).sum()),
            "harmful_n": int((values < -1e-12).sum()),
            "raw_positive_n": len(positive), "raw_negative_n": len(negative),
            "exactly_flat_n": int((values == 0).sum()),
            "numerically_flat_n": int((np.abs(values) <= 1e-12).sum()),
            "materiality_tolerance_log": 1e-12,
            "gross_positive_log_effect": gross_positive,
            "gross_negative_log_effect": float(negative.sum()),
            "largest_positive_share_of_gross_positive": largest / gross_positive if gross_positive > 0 else np.nan,
            "top3_positive_share_of_gross_positive": float(positive[:3].sum() / gross_positive) if gross_positive > 0 else np.nan,
            "return_without_largest_positive_effect_pct": float((variant_nav * np.exp(-largest) - 1.0) * 100.0),
            "total_return_pct": float(metrics.loc[key, "total_return_pct"]),
            "episode_log_effect_sum": effect_sum,
            "terminal_log_wealth_effect": total_effect,
            "reconciliation_abs_error": error,
            "evidence_status": "recomputed_from_daily_returns",
        })
    return pd.DataFrame(records)


def historical_screen(result):
    """Apply explicit reconstructed conditions to every account, retaining failures.

    The six full-history thresholds are stated in the delivered notebook.
    Later-history consistency is implemented as no deterioration in all three
    reported review measures, plus at least three positive calendar blocks.
    These choices remain labelled reconstructed even when saved rows reconcile.
    """
    _validate_result(result)
    measures = ["win_rate", "payoff_ratio", "total_return_pct", "max_drawdown_pct", "eligible_closed_n"]
    metrics = result["metrics"].copy()
    baseline = metrics.loc[metrics.variant.eq("BASE"), ["pair", "base_strategy"] + measures]
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in measures})
    if "baseline_return_pct" in metrics:
        metrics = metrics.drop(columns="baseline_return_pct")
    screen = metrics.merge(baseline, on=["pair", "base_strategy"], how="left", validate="many_to_one")
    screen["win_rate_difference_pp"] = (screen.win_rate - screen.baseline_win_rate) * 100.0
    screen["payoff_ratio_relative_improvement"] = screen.payoff_ratio / screen.baseline_payoff_ratio.where(screen.baseline_payoff_ratio.gt(0)) - 1.0
    screen["return_difference_pp"] = screen.total_return_pct - screen.baseline_total_return_pct
    screen["drawdown_deterioration_pp"] = screen.baseline_max_drawdown_pct - screen.max_drawdown_pct
    screen["retained_closed_fraction"] = screen.eligible_closed_n / screen.baseline_eligible_closed_n.where(screen.baseline_eligible_closed_n.gt(0))
    review = result["phase"].loc[result["phase"].phase.eq("RETROSPECTIVE_REVIEW"), ACCOUNT_KEYS + measures].copy()
    if review.duplicated(ACCOUNT_KEYS).any():
        raise ValueError("Duplicate retrospective review accounts")
    review_baseline = review.loc[review.variant.eq("BASE"), ["pair", "base_strategy"] + measures]
    review_baseline = review_baseline.rename(columns={column: f"baseline_{column}" for column in measures})
    review = review.merge(review_baseline, on=["pair", "base_strategy"], how="left", validate="many_to_one")
    review_differences = review[ACCOUNT_KEYS].copy()
    review_differences["review_win_rate_difference_pp"] = (review.win_rate - review.baseline_win_rate) * 100.0
    review_differences["review_payoff_ratio_relative_improvement"] = review.payoff_ratio / review.baseline_payoff_ratio.where(review.baseline_payoff_ratio.gt(0)) - 1.0
    review_differences["review_return_difference_pp"] = review.total_return_pct - review.baseline_total_return_pct
    screen = screen.merge(review_differences, on=ACCOUNT_KEYS, how="left", validate="one_to_one")
    calendar = result["calendar"]
    if calendar.duplicated(ACCOUNT_KEYS + ["period"]).any():
        raise ValueError("Duplicate calendar blocks")
    blocks = calendar.groupby(ACCOUNT_KEYS).agg(
        available_calendar_blocks=("period", "nunique"),
        positive_calendar_blocks=("return_difference_pp", lambda values: int(values.gt(1e-10).sum())),
    ).reset_index()
    screen = screen.merge(blocks, on=ACCOUNT_KEYS, how="left", validate="one_to_one")
    screen["win_rate_threshold_pass"] = screen.win_rate_difference_pp.ge(2.0 - 1e-10)
    screen["payoff_threshold_pass"] = screen.payoff_ratio_relative_improvement.ge(0.10 - 1e-10)
    screen["return_threshold_pass"] = screen.return_difference_pp.ge(5.0 - 1e-10)
    screen["drawdown_threshold_pass"] = screen.drawdown_deterioration_pp.le(2.0 + 1e-10)
    screen["retention_threshold_pass"] = screen.retained_closed_fraction.ge(0.50 - 1e-10)
    screen["closed_count_threshold_pass"] = screen.eligible_closed_n.ge(30)
    criteria = [
        "win_rate_threshold_pass", "payoff_threshold_pass", "return_threshold_pass",
        "drawdown_threshold_pass", "retention_threshold_pass", "closed_count_threshold_pass",
    ]
    screen["full_history_practical_pass"] = screen[criteria].all(axis=1)
    screen["review_quality_and_pnl_pass"] = (
        screen.review_win_rate_difference_pp.ge(-1e-10)
        & screen.review_payoff_ratio_relative_improvement.ge(-1e-10)
        & screen.review_return_difference_pp.ge(-1e-10)
    )
    screen["calendar_consistency_pass"] = screen.available_calendar_blocks.eq(5) & screen.positive_calendar_blocks.ge(3)
    applicable = screen.applicable if "applicable" in screen else pd.Series(True, index=screen.index)
    screen["final_practical_pass"] = (
        screen.full_history_practical_pass & screen.review_quality_and_pnl_pass
        & screen.calendar_consistency_pass & applicable & screen.variant.ne("BASE")
    )
    screen["screen_specification_status"] = "reconstructed_conditions_original_protocol_unavailable"
    screen["new_untouched_out_of_sample"] = False
    screen.attrs["screen_specification"] = dict(SCREEN_SPECIFICATION)
    return screen


def case_evidence(result, known, effects=None):
    """Build a complete entry/exit evidence ledger instead of selecting anecdotes."""
    effects = episode_effects(result) if effects is None else effects.copy()
    required = {"pair", "date"}
    if not required.issubset(known.columns) or known.duplicated(["pair", "date"]).any():
        raise ValueError("Known inputs need unique pair/date observations")
    known = known.copy()
    known["date"] = pd.to_datetime(known["date"])
    fields = [column for column in known if column.startswith("K_")]
    entry = known[["pair", "date"] + fields].rename(
        columns={"date": "signal_date", **{column: f"entry_{column}" for column in fields}},
    )
    evidence = effects.merge(entry, on=["pair", "signal_date"], how="left", validate="many_to_one")
    risk = known[["pair", "date"] + fields].rename(
        columns={"date": "risk_signal_date", **{column: f"risk_signal_{column}" for column in fields}},
    )
    evidence["risk_signal_date"] = pd.to_datetime(evidence["risk_signal_date"])
    evidence = evidence.merge(risk, on=["pair", "risk_signal_date"], how="left", validate="many_to_one")
    evidence["case_selection_status"] = "Complete original episode census; retrospective explanatory evidence"
    for prefix, signal_field in (("entry_", "signal_date"), ("risk_signal_", "risk_signal_date")):
        observation_fields = [prefix + column for column in fields if column.endswith("_obs_date")]
        for field in observation_fields:
            observations = pd.to_datetime(evidence[field])
            age = evidence[signal_field] - observations
            valid = observations.isna() | (age.gt(pd.Timedelta(0)) & age.le(pd.Timedelta(days=7)))
            if not valid.all():
                raise ValueError(f"Invalid known-observation timing in {field}")
    return evidence
