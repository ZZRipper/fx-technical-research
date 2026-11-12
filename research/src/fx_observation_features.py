"""Describe entry conditions of unfiltered BASE trades, without fitting a filter.

Price features use the signal day's closing price and earlier observations. This
matches the existing execution convention: signal at close t, entry at close
t+1. Economic K inputs must have observation dates strictly before the signal
date and no more than seven calendar days old. The checks below enforce this
on the supplied inputs; they do not establish original publication timestamps.

WIN/LOSS/FLAT are determined by fixed-capital simple trade returns. Comparisons
are retrospective descriptions, not independent out-of-sample evidence. No
threshold, p-value, ranking, selected rule, or filtered PnL is produced. Features
never use trade outcomes, exit prices, excursions, or future price observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ACCOUNT_KEYS = ["pair", "strategy"]
TRADE_KEYS = ACCOUNT_KEYS + ["trade_id"]
FLAT_TOLERANCE_PCT = 1e-10
LOW_SAMPLE_MIN_PER_OUTCOME = 20
COHORTS = (
    ("FULL", None, None),
    ("2005_2014", "2005-01-01", "2015-01-01"),
    ("2015_2019", "2015-01-01", "2020-01-01"),
    ("2020_END", "2020-01-01", None),
)
MACRO_OBSERVATION_FIELDS = {
    "K_vix_obs_date": ("K_vix", "K_vix_d5"),
    "K_rate_obs_date": ("K_spread_bp", "K_spread_d20_bp"),
    "K_dxy_obs_date": ("K_dxy_d20_pct",),
    "K_brent_obs_date": ("K_brent_d20_pct",),
    "K_copper_obs_date": ("K_copper_d20_pct",),
}
PRICE_FEATURES = (
    "spot_return_20_pct", "spot_return_63_pct",
    "realized_vol_20_pct", "realized_vol_63_pct",
    "efficiency_20", "efficiency_63",
    "distance_ma_20_pct", "distance_ma_60_pct", "vol_ratio_20_63",
)
DIRECTION_PRICE_FEATURES = (
    "direction_spot_return_20_pct", "direction_spot_return_63_pct",
    "direction_distance_ma_20_pct", "direction_distance_ma_60_pct",
)
FEATURE_COLUMNS = PRICE_FEATURES + DIRECTION_PRICE_FEATURES + (
    "K_vix", "K_vix_d5", "K_spread_bp", "K_spread_d20_bp", "K_dxy_d20_pct",
    "direction_spread_bp", "direction_spread_d20_bp", "direction_dxy_d20_pct",
    "commodity_d20_pct", "direction_commodity_d20_pct",
)
COMMODITY_FEATURES = {"commodity_d20_pct", "direction_commodity_d20_pct"}
CONTRAST_COLUMNS = ACCOUNT_KEYS + [
    "cohort", "feature", "n_eligible_closed", "n_win", "n_loss", "n_flat",
    "n_applicable", "feature_n", "coverage", "n_feature_win", "n_feature_loss",
    "n_feature_flat", "win_mean", "loss_mean", "flat_mean", "win_median",
    "loss_median", "flat_median", "standardized_mean_difference",
    "low_sample_flag", "descriptive_only",
]
COVERAGE_COLUMNS = ACCOUNT_KEYS + [
    "cohort", "cohort_start", "cohort_end_inclusive", "n_entered", "n_closed",
    "n_eligible_closed", "n_cross_period", "n_open", "n_ineligible_closed",
    "n_win", "n_loss", "n_flat", "low_sample_flag",
    "macro_timing_checked", "descriptive_only",
] + [name + suffix for name in FEATURE_COLUMNS
     for suffix in ("_applicable_n", "_nonmissing_n", "_coverage")]


def _require_columns(frame, columns, label):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _dates(frame, columns, label):
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].dt.tz is not None:
            raise ValueError(f"{label}.{column} must use timezone-naive trading dates")


def _price_features(prices):
    _require_columns(prices, ["pair", "date", "price_USD_per_foreign"], "prices")
    frame = prices[["pair", "date", "price_USD_per_foreign"]].copy()
    _dates(frame, ["date"], "prices")
    if frame[["pair", "date"]].isna().any().any() or frame.duplicated(["pair", "date"]).any():
        raise ValueError("prices requires nonmissing, unique pair/date keys")
    frame["price_USD_per_foreign"] = pd.to_numeric(frame.price_USD_per_foreign, errors="raise")
    if not np.isfinite(frame.price_USD_per_foreign).all() or frame.price_USD_per_foreign.le(0).any():
        raise ValueError("prices must be finite and strictly positive")
    frames = []
    for _, group in frame.groupby("pair", sort=False):
        group = group.sort_values("date").copy()
        price = group.price_USD_per_foreign
        returns = price.div(price.shift(1)).sub(1.0)
        absolute_changes = price.diff().abs()
        for window in (20, 63):
            group[f"spot_return_{window}_pct"] = price.div(price.shift(window)).sub(1.0) * 100.0
            group[f"realized_vol_{window}_pct"] = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0) * 100.0
            path_length = absolute_changes.rolling(window, min_periods=window).sum()
            # A completely flat window has no directional efficiency.
            efficiency = price.diff(window).abs().div(path_length.where(path_length.ne(0)))
            group[f"efficiency_{window}"] = efficiency.mask(path_length.eq(0), 0.0)
        for window in (20, 60):
            average = price.rolling(window, min_periods=window).mean()
            group[f"distance_ma_{window}_pct"] = price.div(average).sub(1.0) * 100.0
        group["vol_ratio_20_63"] = group.realized_vol_20_pct.div(group.realized_vol_63_pct.where(group.realized_vol_63_pct.gt(0)))
        frames.append(group)
    if not frames:
        return pd.DataFrame(columns=["pair", "date", "price_USD_per_foreign", *PRICE_FEATURES])
    return pd.concat(frames, ignore_index=True)


def _known_inputs(known):
    columns = ["pair", "date"] + [column for date, values in MACRO_OBSERVATION_FIELDS.items() for column in (date, *values)]
    _require_columns(known, columns, "known")
    frame = known[columns].copy()
    _dates(frame, ["date", *MACRO_OBSERVATION_FIELDS], "known")
    if frame[["pair", "date"]].isna().any().any() or frame.duplicated(["pair", "date"]).any():
        raise ValueError("known requires nonmissing, unique pair/date keys")
    for observation_field, values in MACRO_OBSERVATION_FIELDS.items():
        age = frame.date - frame[observation_field]
        present = frame[observation_field].notna()
        if not (age.loc[present].gt(pd.Timedelta(0)) & age.loc[present].le(pd.Timedelta(days=7))).all():
            raise ValueError(f"Known input timing violation: {observation_field} must be prior and at most seven days old")
        for value in values:
            frame[value] = pd.to_numeric(frame[value], errors="raise")
            available = frame[value].notna()
            if not np.isfinite(frame.loc[available, value]).all():
                raise ValueError(f"Nonfinite known input: {value}")
            if (available & ~present).any():
                raise ValueError(f"Known value {value} has no supporting {observation_field}")
    return frame


def _pooled_standardized_difference(wins, losses):
    if len(wins) < 2 or len(losses) < 2:
        return np.nan
    variance = ((len(wins) - 1) * wins.var(ddof=1) + (len(losses) - 1) * losses.var(ddof=1)) / (len(wins) + len(losses) - 2)
    if not np.isfinite(variance) or variance <= 0:
        return np.nan
    return float((wins.mean() - losses.mean()) / np.sqrt(variance))


def _summaries(enriched, last_dates):
    contrast_rows, coverage_rows = [], []
    for (pair, strategy), account in enriched.groupby(ACCOUNT_KEYS, sort=False):
        for cohort, start, end in COHORTS:
            starts = pd.Timestamp(start) if start is not None else account.entry_date.min()
            end_exclusive = pd.Timestamp(end) if end is not None else last_dates[pair] + pd.Timedelta(days=1)
            entered = account.loc[account.entry_date.ge(starts) & account.entry_date.lt(end_exclusive)]
            cross_period = entered.closed & entered.exit_date.ge(end_exclusive)
            eligible = entered.loc[entered.eligible_closed & entered.exit_date.lt(end_exclusive)]
            counts = {outcome: int(eligible.outcome.eq(outcome).sum()) for outcome in ("WIN", "LOSS", "FLAT")}
            summary = {
                "pair": pair, "strategy": strategy, "cohort": cohort,
                "cohort_start": starts, "cohort_end_inclusive": end_exclusive - pd.Timedelta(days=1),
                "n_entered": len(entered), "n_closed": int(entered.closed.sum()),
                "n_eligible_closed": len(eligible), "n_cross_period": int(cross_period.sum()),
                "n_open": int((~entered.closed).sum()),
                "n_ineligible_closed": int((entered.closed & ~entered.eligible_closed).sum()),
                "n_win": counts["WIN"], "n_loss": counts["LOSS"], "n_flat": counts["FLAT"],
                "low_sample_flag": min(counts["WIN"], counts["LOSS"]) < LOW_SAMPLE_MIN_PER_OUTCOME,
                "macro_timing_checked": True, "descriptive_only": True,
            }
            for feature in FEATURE_COLUMNS:
                applicable = eligible.commodity_applicable if feature in COMMODITY_FEATURES else pd.Series(True, index=eligible.index)
                values = eligible.loc[applicable, feature]
                available = values.dropna()
                group_values = {outcome: eligible.loc[applicable & eligible.outcome.eq(outcome), feature].dropna() for outcome in counts}
                n_applicable = int(applicable.sum())
                ratio = len(available) / n_applicable if n_applicable else np.nan
                summary[feature + "_applicable_n"] = n_applicable
                summary[feature + "_nonmissing_n"] = len(available)
                summary[feature + "_coverage"] = ratio
                row = {key: summary[key] for key in ACCOUNT_KEYS + ["cohort", "n_eligible_closed", "n_win", "n_loss", "n_flat"]}
                row.update(feature=feature, n_applicable=n_applicable, feature_n=len(available), coverage=ratio)
                for outcome, values in group_values.items():
                    name = outcome.lower()
                    row[f"n_feature_{name}"] = len(values)
                    row[f"{name}_mean"] = float(values.mean()) if len(values) else np.nan
                    row[f"{name}_median"] = float(values.median()) if len(values) else np.nan
                row["standardized_mean_difference"] = _pooled_standardized_difference(group_values["WIN"], group_values["LOSS"])
                row["low_sample_flag"] = min(len(group_values["WIN"]), len(group_values["LOSS"])) < LOW_SAMPLE_MIN_PER_OUTCOME
                row["descriptive_only"] = True
                contrast_rows.append(row)
            coverage_rows.append(summary)
    return pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS), pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)


def build_entry_features(prices, trades, known):
    """Return (enriched_trades, contrasts, coverage) without mutating inputs.

    Contrasts have one row per pair/strategy/cohort/feature. Coverage has one row
    per pair/strategy/cohort. Period cohorts use entry_date, and only eligible
    trades exiting before the next period boundary contribute outcomes. Closed
    trades crossing that boundary are counted in n_cross_period. Open trades
    and ineligible closed trades remain in the enriched ledger and counts.

    Coverage denominators are eligible closed trades for which the feature is
    applicable. Commodity changes are copper for AUD and Brent for CAD/NOK;
    they are structurally inapplicable elsewhere, with NaN coverage. Raw K
    commodity inputs remain available for provenance but are not compared
    independently across unrelated currency pairs.

    Direction features multiply by position (+1 long foreign, -1 short).
    direction_dxy_d20_pct = position * K_dxy_d20_pct: a positive value indicates
    a dollar move conflicting with the foreign-currency position, consistent
    with the existing economic filter convention. Other direction features
    retain their raw signed transformation without assigning a fitted meaning.

    SMD is (WIN mean - LOSS mean) / pooled within-group sample standard
    deviation, requires two finite observations in each group, and is NaN for
    zero pooled variance. low_sample_flag means fewer than 20 observations in
    either outcome group. It is a display flag, not an acceptance criterion.
    """
    required = TRADE_KEYS + ["signal_date", "entry_date", "exit_date", "position", "eligible_closed", "closed", "simple_return_pct"]
    _require_columns(trades, required, "trades")
    price_features = _price_features(prices)
    macro = _known_inputs(known)
    enriched = trades.copy()
    _dates(enriched, ["signal_date", "entry_date", "exit_date"], "trades")
    if enriched[TRADE_KEYS + ["signal_date", "entry_date"]].isna().any().any() or enriched.duplicated(TRADE_KEYS).any():
        raise ValueError("Trades require nonmissing unique account/trade keys and signal/entry dates")
    for field in ("eligible_closed", "closed"):
        if enriched[field].isna().any() or not enriched[field].isin([True, False]).all():
            raise ValueError(f"trades.{field} must contain boolean values")
        enriched[field] = enriched[field].astype(bool)
    if not enriched.position.isin([-1, 1]).all():
        raise ValueError("Trades require position equal to +1 or -1")
    if (enriched.eligible_closed & ~enriched.closed).any():
        raise ValueError("Eligible closed trades must be closed")
    if not enriched.signal_date.lt(enriched.entry_date).all():
        raise ValueError("Each trade signal must precede entry")
    if (enriched.closed & (enriched.exit_date.isna() | enriched.exit_date.lt(enriched.entry_date))).any():
        raise ValueError("Closed trades require an exit date at or after entry")
    enriched["simple_return_pct"] = pd.to_numeric(enriched.simple_return_pct, errors="raise")
    if not np.isfinite(enriched.loc[enriched.eligible_closed, "simple_return_pct"]).all():
        raise ValueError("Eligible outcomes require finite simple returns")
    derived_columns = set(FEATURE_COLUMNS) | set(macro.columns).difference({"pair", "date"}) | {
        "signal_price_USD_per_foreign", "price_feature_date", "commodity_source", "commodity_applicable", "outcome",
    }
    collisions = derived_columns.intersection(enriched.columns)
    if collisions:
        raise ValueError(f"Trades already contain feature output columns: {sorted(collisions)}")
    price_features = price_features.rename(columns={"date": "signal_date", "price_USD_per_foreign": "signal_price_USD_per_foreign"})
    enriched = enriched.merge(price_features, on=["pair", "signal_date"], how="left", validate="many_to_one", sort=False)
    if enriched.signal_price_USD_per_foreign.isna().any():
        raise ValueError("Every trade signal requires an exact matching price observation")
    enriched["price_feature_date"] = enriched.signal_date
    enriched = enriched.merge(macro.rename(columns={"date": "signal_date"}), on=["pair", "signal_date"], how="left", validate="many_to_one", sort=False)
    # Enforce the information boundary again on the actual joined trade rows.
    for observation_field in MACRO_OBSERVATION_FIELDS:
        age = enriched.signal_date - enriched[observation_field]
        valid = enriched[observation_field].isna() | (age.gt(pd.Timedelta(0)) & age.le(pd.Timedelta(days=7)))
        if not valid.all():
            raise ValueError(f"Entry feature timing violation: {observation_field}")
    for feature in ("spot_return_20_pct", "spot_return_63_pct", "distance_ma_20_pct", "distance_ma_60_pct"):
        enriched["direction_" + feature] = enriched.position * enriched[feature]
    for feature in ("spread_bp", "spread_d20_bp", "dxy_d20_pct"):
        enriched["direction_" + feature] = enriched.position * enriched["K_" + feature]
    enriched["commodity_source"] = enriched.pair.map({"AUD": "COPPER_LME", "CAD": "BRENT", "NOK": "BRENT"}).fillna("NOT_APPLICABLE")
    enriched["commodity_applicable"] = enriched.commodity_source.ne("NOT_APPLICABLE")
    enriched["commodity_d20_pct"] = np.nan
    enriched.loc[enriched.pair.eq("AUD"), "commodity_d20_pct"] = enriched.loc[enriched.pair.eq("AUD"), "K_copper_d20_pct"]
    oil_pairs = enriched.pair.isin(["CAD", "NOK"])
    enriched.loc[oil_pairs, "commodity_d20_pct"] = enriched.loc[oil_pairs, "K_brent_d20_pct"]
    enriched["direction_commodity_d20_pct"] = enriched.position * enriched.commodity_d20_pct
    enriched["outcome"] = np.where(enriched.closed, "INELIGIBLE", "OPEN")
    eligible = enriched.eligible_closed
    enriched.loc[eligible, "outcome"] = np.select(
        [enriched.loc[eligible, "simple_return_pct"].gt(FLAT_TOLERANCE_PCT), enriched.loc[eligible, "simple_return_pct"].lt(-FLAT_TOLERANCE_PCT)],
        ["WIN", "LOSS"], default="FLAT",
    )
    if not np.isfinite(enriched[list(FEATURE_COLUMNS)].to_numpy(dtype=float)[enriched[list(FEATURE_COLUMNS)].notna().to_numpy()]).all():
        raise ValueError("Feature calculations produced nonfinite values")
    last_dates = price_features.groupby("pair").signal_date.max().to_dict()
    contrasts, coverage = _summaries(enriched, last_dates)
    for frame in (enriched, contrasts, coverage):
        frame.attrs.update(
            descriptive_only=True, outcome_basis="fixed_initial_capital_simple_return",
            new_untouched_out_of_sample=False, macro_timing="strictly_prior_at_most_7_calendar_days",
            low_sample_min_per_outcome=LOW_SAMPLE_MIN_PER_OUTCOME,
            flat_tolerance_pct=FLAT_TOLERANCE_PCT,
        )
    return enriched, contrasts, coverage
