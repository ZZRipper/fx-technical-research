"""Prepare technical-rule parameter diagnostics for the existing offline dashboard.

Called by build_simple_dashboard.py. Indicators use the canonical price source
and the original rule functions. Account returns and trade cohorts remain those
in the main explorer payload; a reconciliation failure aborts the build.
"""

import numpy as np
import pandas as pd

from .fx_baselines import (
    CURRENCIES, canonical_prices_from_raw, complete_signals, fixed_execute,
    rsi_wilder, sma, rolling_std,
)


def build_parameter_data(payload):
    prices, _ = canonical_prices_from_raw()
    result = {"pairs": list(CURRENCIES), "families": {
                  "bb": {"windows": [10, 20], "series": {}},
                  "bb_bo": {"windows": [10, 20], "series": {}},
                  "rsi": {"windows": [7, 14, 20], "series": {}},
                  "rsi_mom": {"windows": [7, 14, 20], "series": {}},
                  "ma": {"windows": ["10_30", "20_60"], "series": {}},
                  "tsmom": {"windows": [21, 63], "series": {}},
                  "donchian": {"windows": [10, 20], "series": {}}},
              "source": "Canonical USD-per-foreign daily prices; original main-history BASE accounts",
              "start": payload["datasets"]["main"]["start"],
              "end": payload["datasets"]["main"]["end"]}
    max_error = 0.0
    reconciled = 0
    common_calendar = None
    for pair in CURRENCIES:
        panel = prices.loc[prices.pair.eq(pair)].sort_values("date")
        values = panel.price_USD_per_foreign.to_numpy(float)
        dates = pd.DatetimeIndex(panel.date)
        days = (dates - pd.Timestamp("1970-01-01")).days.to_numpy()
        signals, _ = complete_signals(values)
        first_execution = int(dates.searchsorted(pd.Timestamp("2005-01-01")))
        for family, spec in result["families"].items():
            spec["series"][pair] = {}
            for window in spec["windows"]:
                rule = {"bb": "BB_MR", "bb_bo": "BB_BO", "rsi": "RSI_MR", "rsi_mom": "RSI_MOM", "ma": "MA", "tsmom": "TSMOM", "donchian": "DONCHIAN_BO"}[family] + f"_{window}"
                if family in ("bb", "bb_bo"):
                    sd = rolling_std(values, window)
                    scores = np.divide(values - sma(values, window), sd,
                                       out=np.full(len(values), np.nan),
                                       where=np.isfinite(sd) & (sd > 1e-12))
                elif family == "ma":
                    fast, slow = map(int, window.split("_"))
                    # Display percent units only; execution uses the original signal.
                    scores = 100 * (sma(values, fast) / sma(values, slow) - 1)
                elif family == "tsmom":
                    scores = np.full(len(values), np.nan)
                    scores[window:] = 100 * (np.log(values[window:]) - np.log(values[:-window]))
                elif family == "donchian":
                    previous = pd.Series(values).shift(1)
                    high = previous.rolling(window).max().to_numpy()
                    low = previous.rolling(window).min().to_numpy()
                    width = high - low
                    # Diagnostic coordinate: -1 lower band, 0 midpoint, +1 upper band.
                    # No clipping; breakout overshoots remain in the distribution.
                    scores = np.divide(2 * (values - low), width,
                                       out=np.full(len(values), np.nan), where=width > 0) - 1
                else:
                    scores = rsi_wilder(values, window)
                target = signals[rule].copy()
                target[:max(0, first_execution - 1)] = 0
                run = fixed_execute(values, target)
                account = next(a for a in payload["accounts"] if a["id"] == f"main|{pair}|{rule}|BASE")
                calendar = np.array(payload["calendars"][account["cal"]])
                indices = np.searchsorted(days, calendar)
                if not np.array_equal(days[indices], calendar):
                    raise ValueError(f"Calendar mismatch: {account['id']}")
                if common_calendar is None:
                    common_calendar = calendar
                if not np.array_equal(calendar, common_calendar):
                    raise ValueError("Parameter comparisons require a shared calendar.")
                error = float(np.max(np.abs(run["gross_return"][indices] - account["daily_returns"])))
                max_error = max(error, max_error)
                if error > 1e-12:
                    raise ValueError(f"Return reconciliation failed: {account['id']}: {error}")
                if not np.all(np.isfinite(scores[indices])):
                    raise ValueError(f"Incomplete indicator coverage: {account['id']}")
                spec["series"][pair][str(window)] = {
                    "account": account["id"], "values": scores[indices].tolist(),
                    "held": run["held"][indices].tolist(),
                }
                reconciled += 1
    result["validation"] = {"accounts_reconciled": reconciled, "maximum_daily_return_error": max_error}
    return result
