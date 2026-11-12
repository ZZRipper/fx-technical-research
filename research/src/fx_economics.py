"""Reconstructed economic inputs and episode filters from the delivered research.

The original frozen engine is absent. This implementation derives results from
prices and native observations; archived outputs are used only for validation.
"""

from pathlib import Path
import gzip
import re

import numpy as np
import pandas as pd

try:
    from .fx_baselines import account_metrics, fixed_execute, read_xlsx_sheet, trade_ledger
except ImportError:
    from fx_baselines import account_metrics, fixed_execute, read_xlsx_sheet, trade_ledger


PAIR_COUNTRIES = {
    "EUR": "DE", "GBP": "UK", "AUD": "AU", "NZD": "NZ", "JPY": "JP",
    "CHF": "CH", "CAD": "CA", "NOK": "NO", "SEK": "SE",
}
CORE_RULES = ("TSMOM_63", "MA_20_60", "BB_MR_20", "RSI_MR_14")
VARIANTS = (
    "BASE", "RATE_CHANGE_CONFLICT", "DUAL_RATE_CONFLICT", "POLICY_USD",
    "RISK_YIELD", "COMMODITY_USD", "ECO_COMPOSITE", "RISK_YIELD_EXIT",
)
KNOWN_FIELDS = (
    "K_vix", "K_vix_d5", "K_spread_bp", "K_spread_d20_bp",
    "K_dxy_d20_pct", "K_brent_d20_pct", "K_copper_d20_pct",
)
THRESHOLD_TOLERANCE = 1e-10
PERIODS = (
    ("2005-2009", "2005-01-01", "2010-01-01"),
    ("2010-2014", "2010-01-01", "2015-01-01"),
    ("2015-2019", "2015-01-01", "2020-01-01"),
    ("2020-2024", "2020-01-01", "2025-01-01"),
    ("2025-2026YTD", "2025-01-01", "2027-01-01"),
)


def _data_directory(data_root=None):
    candidate = Path(data_root) if data_root is not None else Path(__file__).resolve().parents[1]
    for directory in (candidate, candidate / "data", candidate / "FX_Research_Datasets" / "data"):
        if (directory / "raw").is_dir():
            return directory
    raise FileNotFoundError(f"No raw dataset directory under {candidate}")


def read_native_features_from_raw(data_root=None):
    """Read each feature sheet without filling missing native observations."""
    import zipfile
    import xml.etree.ElementTree as element_tree

    directory = _data_directory(data_root)
    paths = sorted((directory / "raw").glob("FX_Features_Book*.xlsx"))
    if len(paths) != 1:
        raise ValueError(f"Expected one native feature workbook, found {len(paths)}")
    with zipfile.ZipFile(paths[0]) as archive:
        workbook = element_tree.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"sheet": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names = [sheet.attrib["name"] for sheet in workbook.findall("sheet:sheets/sheet:sheet", namespace)]
    native = {}
    for sheet_name in names:
        if not re.match(r"\d+_", sheet_name):
            continue
        frame = read_xlsx_sheet(paths[0], sheet_name)
        observation_dates = frame.iloc[:, 0]
        if pd.api.types.is_numeric_dtype(observation_dates):
            observation_dates = pd.to_datetime(observation_dates, unit="D", origin="1899-12-30")
        else:
            observation_dates = pd.to_datetime(observation_dates, errors="coerce")
        frame = frame.iloc[:, 1:].copy()
        frame.index = pd.DatetimeIndex(observation_dates, name="date")
        frame = frame.loc[frame.index.notna()].sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate observation dates in {sheet_name}")
        native[sheet_name] = frame.apply(pd.to_numeric, errors="coerce")
    return native


def _align_native(series, dates, observation_field, change_field, steps, max_span, percent, level_field=None):
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if series.index.has_duplicates:
        raise ValueError("Native series has duplicate dates")
    changes = (series / series.shift(steps) - 1.0) * 100.0 if percent else series.diff(steps)
    valid_span = series.index.to_series().diff(steps).dt.days.le(max_span)
    changes = changes.where(valid_span).replace([np.inf, -np.inf], np.nan)
    observations = pd.DataFrame({observation_field: series.index, change_field: changes.to_numpy()})
    if level_field is not None:
        observations[level_field] = series.to_numpy()
    return pd.merge_asof(
        pd.DataFrame({"date": dates}), observations,
        left_on="date", right_on=observation_field,
        direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=7),
    )


def rebuild_economic_known(native, dates):
    """Build prior-only K inputs with native-step and calendar-age limits."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    common = _align_native(native["5_Equities"]["VIX"], dates, "K_vix_obs_date", "K_vix_d5", 5, 12, False, "K_vix")
    for sheet_name, source_field, prefix in (
        ("1_FX_Spot", "DXY", "dxy"),
        ("8_Commodities", "BRENT", "brent"),
        ("8_Commodities", "COPPER_LME", "copper"),
    ):
        aligned = _align_native(native[sheet_name][source_field], dates, f"K_{prefix}_obs_date", f"K_{prefix}_d20_pct", 20, 45, True)
        common = common.merge(aligned, on="date", validate="one_to_one")
    frames = []
    for pair, country in PAIR_COUNTRIES.items():
        spread = (native["2_Rates"][f"{country}_2Y"] - native["2_Rates"]["US_2Y"]) * 100.0
        aligned = _align_native(spread, dates, "K_rate_obs_date", "K_spread_d20_bp", 20, 45, False, "K_spread_bp")
        frame = common.merge(aligned, on="date", validate="one_to_one")
        frame["pair"] = pair
        frames.append(frame)
    columns = [
        "date", "K_vix_obs_date", "K_vix", "K_vix_d5", "K_dxy_obs_date", "K_dxy_d20_pct",
        "K_brent_obs_date", "K_brent_d20_pct", "K_copper_obs_date", "K_copper_d20_pct",
        "K_rate_obs_date", "K_spread_bp", "K_spread_d20_bp", "pair",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def verify_known_features(known, data_root=None, tolerance=1e-8):
    """Compare rebuilt values and timestamps with the delivered clean reference."""
    reference = pd.read_csv(_data_directory(data_root) / "clean" / "economic_known_inputs_rebuilt.csv.gz", parse_dates=["date"])
    merged = known.merge(reference, on=["pair", "date"], suffixes=("_new", "_reference"), validate="one_to_one", how="outer", indicator=True)
    complete = len(merged) == len(known) == len(reference) and merged["_merge"].eq("both").all()
    checks = []
    for field in KNOWN_FIELDS:
        actual = merged[field + "_new"]
        expected = merged[field + "_reference"]
        mismatches = int(actual.isna().ne(expected.isna()).sum())
        error = (actual - expected).abs().max()
        error = 0.0 if pd.isna(error) else float(error)
        checks.append({"feature": field, "rows": len(merged), "missing_pattern_mismatches": mismatches,
                       "max_abs_error": error, "pass": bool(complete and not mismatches and error <= tolerance)})
    for field in [column for column in known if column.endswith("_obs_date")]:
        actual = pd.to_datetime(merged[field + "_new"])
        expected = pd.to_datetime(merged[field + "_reference"])
        mismatch = ~(actual.eq(expected) | (actual.isna() & expected.isna()))
        age = merged["date"] - actual
        valid_timing = actual.isna() | (age.gt(pd.Timedelta(0)) & age.le(pd.Timedelta(days=7)))
        checks.append({"feature": field, "rows": len(merged), "missing_pattern_mismatches": int(mismatch.sum()),
                       "max_abs_error": 0.0, "pass": bool(complete and not mismatch.any() and valid_timing.all())})
    return pd.DataFrame(checks)


def _at_most(value, threshold):
    """Preserve inclusive decimal thresholds after binary yield subtraction."""
    return value <= threshold or abs(value - threshold) <= THRESHOLD_TOLERANCE


def _at_least(value, threshold):
    return value >= threshold or abs(value - threshold) <= THRESHOLD_TOLERANCE


def _predicate_state(inputs, pair, direction, variant):
    if variant == "BASE":
        return "KEEP"
    if variant == "ECO_COMPOSITE":
        states = [_predicate_state(inputs, pair, direction, component) for component in ("POLICY_USD", "RISK_YIELD", "COMMODITY_USD")]
        if "REJECT" in states:
            return "REJECT"
        return "UNKNOWN" if "UNKNOWN" in states else "KEEP"
    if variant == "RATE_CHANGE_CONFLICT":
        fields = ("K_spread_d20_bp",)
    elif variant == "DUAL_RATE_CONFLICT":
        fields = ("K_spread_d20_bp", "K_spread_bp")
    elif variant == "POLICY_USD":
        fields = ("K_spread_d20_bp", "K_dxy_d20_pct")
    elif variant in ("RISK_YIELD", "RISK_YIELD_EXIT"):
        fields = ("K_vix", "K_vix_d5", "K_spread_bp")
    elif variant == "COMMODITY_USD":
        if pair not in ("AUD", "CAD", "NOK"):
            return "NOT_APPLICABLE"
        fields = ("K_copper_d20_pct" if pair == "AUD" else "K_brent_d20_pct", "K_dxy_d20_pct")
    else:
        raise ValueError(f"Unknown economic variant {variant}")
    values = [inputs.get(field, np.nan) for field in fields]
    if not np.isfinite(values).all():
        return "UNKNOWN"
    if variant == "RATE_CHANGE_CONFLICT":
        reject = _at_most(direction * values[0], -10)
    elif variant == "DUAL_RATE_CONFLICT":
        reject = _at_most(direction * values[0], -10) and _at_most(direction * values[1], -50)
    elif variant == "POLICY_USD":
        reject = _at_most(direction * values[0], -10) and _at_least(direction * values[1], 1)
    elif variant in ("RISK_YIELD", "RISK_YIELD_EXIT"):
        reject = _at_least(values[0], 20) and _at_least(values[1], 3) and _at_least(direction * values[2], 50)
    else:
        reject = direction * values[0] < 0 and _at_least(direction * values[1], 1)
    return "REJECT" if reject else "KEEP"


def _quality_statistics(trades):
    eligible = trades.loc[trades["eligible_closed"].astype(bool)] if len(trades) else trades
    returns = eligible["trade_return"].to_numpy(dtype=float) if len(eligible) else np.array([], dtype=float)
    wins = returns[returns > 1e-12]
    losses = returns[returns < -1e-12]
    average_win = float(wins.mean()) if len(wins) else np.nan
    average_loss = float(-losses.mean()) if len(losses) else np.nan
    return {
        "eligible_closed_n": len(returns), "win_n": len(wins), "loss_n": len(losses),
        "flat_n": int(np.sum(np.abs(returns) <= 1e-12)),
        "win_rate": len(wins) / len(returns) if len(returns) else np.nan,
        "avg_win_return": average_win, "avg_loss_abs_return": average_loss,
        "payoff_ratio": average_win / average_loss if average_loss > 0 else np.nan,
        "expectancy_return": float(returns.mean()) if len(returns) else np.nan,
    }


def _period_statistics(run, dates, start, end):
    selected = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    returns = np.asarray(run["net_return"])[selected]
    nav = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    return {"total_return_pct": float((nav[-1] - 1.0) * 100.0),
            "max_drawdown_pct": float(np.min(nav / np.maximum.accumulate(nav) - 1.0) * 100.0)}


def run_economic_filters(prices, known, baseline_runs, price_dates, baseline_trades):
    """Recompute all 288 accounts, locking rejected or exited original episodes.

    Each returned run uses full warmup dates and unit NAV. Trade rows contain
    accepted trades; episode_decisions records rejected entries and early exits.
    Phase quality uses entry cohorts closed within their phase boundaries.
    """
    metrics_rows, ledger_frames, decision_rows, phase_rows, calendar_rows = [], [], [], [], []
    runs = {}
    for pair in PAIR_COUNTRIES:
        dates = pd.DatetimeIndex(price_dates[pair] if isinstance(price_dates, dict) else price_dates)
        pair_prices = prices.loc[prices["pair"].eq(pair)].set_index("date")["price_USD_per_foreign"].reindex(dates).to_numpy(dtype=float)
        pair_known = known.loc[known["pair"].eq(pair)].set_index("date").reindex(dates)
        known_records = pair_known.to_dict("records")
        for strategy in CORE_RULES:
            original = baseline_runs[(pair, strategy)]
            episodes = baseline_trades.loc[baseline_trades["pair"].eq(pair) & baseline_trades["strategy"].eq(strategy)]
            original_metrics = account_metrics(original, dates, episodes)
            for variant in VARIANTS:
                executed = np.zeros(len(dates), dtype=np.int8)
                blocked, early_exits, unknown = 0, 0, 0
                account_decisions = {}
                for episode_number, episode in enumerate(episodes.itertuples(index=False)):
                    entry_index = int(episode.entry_idx)
                    original_end = int(episode.exit_idx) if episode.closed else len(dates)
                    direction = int(episode.position)
                    signal_index = int(episode.signal_idx)
                    state = _predicate_state(known_records[signal_index], pair, direction, variant)
                    final_end = original_end
                    risk_signal = None
                    if state == "REJECT":
                        blocked += 1
                    else:
                        unknown += int(state == "UNKNOWN")
                        if variant == "RISK_YIELD_EXIT":
                            for monitor_index in range(entry_index, min(original_end - 1, len(dates) - 1)):
                                if _predicate_state(known_records[monitor_index], pair, direction, "RISK_YIELD") == "REJECT":
                                    final_end = monitor_index + 1
                                    risk_signal = dates[monitor_index]
                                    early_exits += 1
                                    break
                        executed[entry_index:final_end] = direction
                    decision = {
                        "pair": pair, "base_strategy": strategy, "variant": variant,
                        "original_episode": episode_number, "signal_date": dates[signal_index],
                        "entry_date": dates[entry_index], "position": direction,
                        "entry_state": state, "accepted": state != "REJECT",
                        "early_risk_exit": risk_signal is not None, "risk_signal_date": risk_signal,
                        "execution_exit_date": dates[final_end] if final_end < len(dates) else pd.NaT,
                        "original_exit_date": dates[original_end] if original_end < len(dates) else pd.NaT,
                    }
                    decision_rows.append(decision)
                    account_decisions[entry_index] = decision
                targets = np.concatenate((executed[1:], executed[-1:]))
                run = fixed_execute(pair_prices, targets)
                run["dates"] = dates
                run["account_start_date"] = "2005-01-01"
                runs[(pair, strategy, variant)] = run
                ledger = trade_ledger(run, dates, pair, strategy)
                ledger["base_strategy"] = strategy
                ledger["variant"] = variant
                if len(ledger):
                    ledger["original_episode"] = ledger["entry_idx"].map(lambda index: account_decisions[index]["original_episode"])
                    ledger["early_risk_exit"] = ledger["entry_idx"].map(lambda index: account_decisions[index]["early_risk_exit"])
                ledger_frames.append(ledger)
                stats = account_metrics(run, dates, ledger)
                identity = {"pair": pair, "base_strategy": strategy, "variant": variant}
                stats.update(identity)
                stats.update({"blocked_base_trades": blocked, "early_risk_exits": early_exits, "unknown_entry_n": unknown,
                              "applicable": variant != "COMMODITY_USD" or pair in ("AUD", "CAD", "NOK"),
                              "baseline_return_pct": original_metrics["total_return_pct"],
                              "return_difference_pp": stats["total_return_pct"] - original_metrics["total_return_pct"]})
                metrics_rows.append(stats)
                for phase, start, end in (
                    ("FULL_HISTORY", "2005-01-01", "2027-01-01"),
                    ("DEVELOPMENT", "2005-01-01", "2015-01-01"),
                    ("RETROSPECTIVE_REVIEW", "2015-01-01", "2027-01-01"),
                ):
                    cohort = ledger.loc[(ledger["entry_date"] >= pd.Timestamp(start)) & (ledger["entry_date"] < pd.Timestamp(end))].copy()
                    cohort["eligible_closed"] = cohort["eligible_closed"] & cohort["exit_date"].lt(pd.Timestamp(end))
                    phase_rows.append({**identity, "phase": phase, **_quality_statistics(cohort),
                                       **_period_statistics(run, dates, start, end), "phase_entry_n": len(cohort),
                                       "phase_open_or_censored_n": int((~cohort["eligible_closed"]).sum())})
                for label, start, end in PERIODS:
                    actual = _period_statistics(run, dates, start, end)
                    baseline = _period_statistics(original, dates, start, end)
                    calendar_rows.append({**identity, "period": label, "variant_return_pct": actual["total_return_pct"],
                                          "baseline_return_pct": baseline["total_return_pct"],
                                          "return_difference_pp": actual["total_return_pct"] - baseline["total_return_pct"]})
    return {"metrics": pd.DataFrame(metrics_rows), "trades": pd.concat(ledger_frames, ignore_index=True),
            "runs": runs, "phase": pd.DataFrame(phase_rows), "calendar": pd.DataFrame(calendar_rows),
            "episode_decisions": pd.DataFrame(decision_rows)}


def write_daily_accounts(result, output_path):
    """Stream exact account observations without materializing a 288-account table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        first = True
        for (pair, strategy, variant), run in result["runs"].items():
            dates = pd.DatetimeIndex(run["dates"])
            selection = dates >= pd.Timestamp(run.get("account_start_date", "2005-01-01"))
            frame = pd.DataFrame({
                "pair": pair, "base_strategy": strategy, "variant": variant, "date": dates[selection],
                "cumulative_pnl_usd": (np.asarray(run["equity"])[selection] - 1.0) * 100000.0,
                "held_position": np.asarray(run["held"])[selection],
                "daily_return": np.asarray(run["net_return"])[selection],
            })
            frame.to_csv(handle, index=False, header=first)
            first = False
    return output_path
