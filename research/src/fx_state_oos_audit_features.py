"""Independent, cutoff-first feature reconstruction for state-filter audits.

This module imports no production feature, economic, strategy, or filter code.
It does not open result files or choose states. ``load_native_inputs`` decodes a
row's date before any requested observation value and skips dates beyond the
explicit cutoff. Native changes count finite observations, not FX quote rows.

``entry_feature_arrays`` returns the 13 registered features on the entire quote
calendar. All ``direction_*`` arrays are UNSIGNED: the caller must multiply by
the original episode's position at its signal index. Other arrays, including
shadow BASE PnL, must not be multiplied. The supplied ``base_returns`` must be
the original unfiltered BASE's already-earned daily simple returns in decimal
units; this module cannot infer that provenance from a numeric array alone.

Signal-close prices and BASE returns through that close are known. Macro
observations must be strictly earlier and at most seven calendar days old.
Observation dates enforce this data convention, not publication-vintage proof.
"""

from pathlib import Path
import hashlib
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd


COUNTRIES = {
    "EUR": "DE", "GBP": "UK", "AUD": "AU", "NZD": "NZ", "JPY": "JP",
    "CHF": "CH", "CAD": "CA", "NOK": "NO", "SEK": "SE",
}
REQUIRED_SHEETS = {
    "1_FX_Spot": ("DXY",),
    "2_Rates": ("US_2Y", "DE_2Y", "UK_2Y", "AU_2Y", "NZ_2Y", "JP_2Y",
                "CH_2Y", "CA_2Y", "NO_2Y", "SE_2Y"),
    "5_Equities": ("VIX",),
    "8_Commodities": ("BRENT", "COPPER_LME"),
}
FEATURE_NAMES = (
    "vol_ratio_20_63", "realized_vol_20_pct", "efficiency_63",
    "direction_return_20_vol", "direction_distance_ma_60_vol", "K_vix", "K_vix_d5",
    "direction_spread_bp", "direction_spread_d20_bp", "direction_dxy_d20_pct",
    "direction_commodity_d20_pct", "base_pnl_63_pct", "base_pnl_252_pct",
)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_DAY = np.timedelta64(1, "D")


def _calendar(values, label, allow_empty=False):
    dates = pd.DatetimeIndex(values)
    if ((not allow_empty and not len(dates)) or dates.hasnans or dates.has_duplicates
            or not dates.is_monotonic_increasing or dates.tz is not None):
        raise ValueError(label + " must be a unique, increasing, timezone-naive date calendar")
    return dates


def load_native_inputs(path, cutoff):
    """Return 14 named finite native Series plus ``_validation`` metadata.

    Names: VIX, DXY, BRENT, COPPER_LME, and the ten ``<country>_2Y`` yields.
    Missing/non-numeric cells remain absent observations, without forward fill.
    The metadata contains source SHA256 and per-sheet decoded/skipped counts.
    No requested value belonging to a later row is converted to a number.
    """
    path, cutoff = Path(path), pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tz is not None:
        raise ValueError("An explicit timezone-naive cutoff is required")
    records = {name: [] for names in REQUIRED_SHEETS.values() for name in names}
    stats = {}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        props = workbook.find(_NS + "workbookPr")
        epoch = pd.Timestamp("1904-01-01" if props is not None and props.get("date1904") in ("1", "true") else "1899-12-30")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {node.get("Id"): node.get("Target") for node in relations}
        sheets = {}
        for node in workbook.findall(_NS + "sheets/" + _NS + "sheet"):
            target = targets[node.get(_REL + "id")]
            sheets[node.get("name")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        missing = set(REQUIRED_SHEETS).difference(sheets)
        if missing:
            raise ValueError("Missing native worksheets: " + repr(sorted(missing)))
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(n.text or "" for n in node.iter(_NS + "t"))
                      for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))]

        def text(cell):
            if cell is None:
                return None
            kind = cell.get("t", "")
            if kind == "inlineStr":
                return "".join(node.text or "" for node in cell.iter(_NS + "t"))
            value = cell.find(_NS + "v")
            raw = None if value is None else value.text
            if raw is None:
                return None
            return shared[int(raw)] if kind == "s" else raw

        for sheet, names in REQUIRED_SHEETS.items():
            columns, seen = None, set()
            count = {"rows_at_or_before_cutoff": 0, "rows_after_cutoff_skipped": 0,
                     "invalid_date_rows_skipped": 0, "value_cells_decoded": 0,
                     "post_cutoff_value_cells_decoded": 0}
            with archive.open(sheets[sheet]) as stream:
                for _, row in ET.iterparse(stream, events=("end",)):
                    if row.tag != _NS + "row":
                        continue
                    cells = {"".join(c for c in cell.get("r", "") if c.isalpha()): cell
                             for cell in row.findall(_NS + "c")}
                    if columns is None:
                        headers = {col: text(cell) for col, cell in cells.items()}
                        columns = {name: next((col for col, value in headers.items() if value == name), None) for name in names}
                        if any(col is None for col in columns.values()):
                            raise ValueError("Missing/ambiguous header in " + sheet)
                        if any(sum(value == name for value in headers.values()) != 1 for name in names):
                            raise ValueError("Duplicate native series header in " + sheet)
                        row.clear()
                        continue
                    raw_date = text(cells.get("A"))
                    try:
                        if raw_date is None:
                            raise ValueError("missing date")
                        try:
                            serial = float(raw_date)
                        except (ValueError, TypeError):
                            day = pd.Timestamp(raw_date)
                        else:
                            day = epoch + pd.Timedelta(days=serial)
                        if pd.isna(day) or day.tz is not None:
                            raise ValueError("invalid date")
                    except (ValueError, TypeError, OverflowError):
                        count["invalid_date_rows_skipped"] += 1
                        row.clear()
                        continue
                    if day > cutoff:
                        count["rows_after_cutoff_skipped"] += 1
                        row.clear()
                        continue
                    if day in seen:
                        raise ValueError("Duplicate native date in " + sheet + ": " + str(day))
                    seen.add(day)
                    count["rows_at_or_before_cutoff"] += 1
                    for name, col in columns.items():
                        raw = text(cells.get(col))
                        if raw is None:
                            continue
                        count["value_cells_decoded"] += 1
                        try:
                            value = float(raw)
                        except (ValueError, TypeError):
                            continue
                        if np.isfinite(value):
                            records[name].append((day, value))
                    row.clear()
            if columns is None:
                raise ValueError("Empty native worksheet: " + sheet)
            stats[sheet] = count
    result = {}
    for name, rows in records.items():
        rows.sort(key=lambda item: item[0])
        dates = _calendar([day for day, _ in rows], name, allow_empty=True)
        result[name] = pd.Series([value for _, value in rows], index=dates, dtype=float, name=name)
    result["_validation"] = {
        "cutoff": cutoff.isoformat(), "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sheets": stats, "post_cutoff_value_cells_decoded": 0,
        "native_finite_observations": {name: len(result[name]) for name in records},
    }
    return result


def _align(series, dates, steps, max_span, percent):
    """Select last strictly prior native row, independently of change validity."""
    index = _calendar(series.index, "native observations", allow_empty=True)
    numeric = np.asarray(series, dtype=float)
    finite = np.isfinite(numeric)
    observed, values = index.to_numpy()[finite], numeric[finite]
    n = len(dates)
    level, change = np.full(n, np.nan), np.full(n, np.nan)
    provenance = np.full(n, np.datetime64("NaT", "ns"))
    native_changes = np.full(len(values), np.nan)
    valid_span = np.zeros(len(values), dtype=bool)
    if len(values) > steps:
        span = (observed[steps:] - observed[:-steps]) / _DAY
        valid_span[steps:] = span <= max_span
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (values[steps:] / values[:-steps] - 1.) * 100 if percent else values[steps:] - values[:-steps]
        native_changes[steps:] = np.where((span <= max_span) & np.isfinite(raw), raw, np.nan)
    positions = np.searchsorted(observed, dates.to_numpy(), side="left") - 1
    usable = positions >= 0
    if len(observed):
        age = (dates.to_numpy() - observed[np.maximum(positions, 0)]) / _DAY
        usable &= (age > 0) & (age <= 7)
        slots = np.flatnonzero(usable)
        native_slots = positions[slots]
        provenance[slots], level[slots], change[slots] = observed[native_slots], values[native_slots], native_changes[native_slots]
    age = (dates.to_numpy() - provenance) / _DAY
    present = ~np.isnat(provenance)
    timing = bool(np.all((age[present] > 0) & (age[present] <= 7)))
    if not timing or not np.all(present[np.isfinite(level) | np.isfinite(change)]):
        raise AssertionError("Known macro values lack strict-prior, fresh provenance")
    stats = {"native_observations": len(values), "native_change_steps": steps,
             "max_native_change_span_days": max_span,
             "native_changes_valid": int(np.isfinite(native_changes).sum()),
             "native_changes_span_rejected": int((~valid_span[steps:]).sum()),
             "aligned_observations": int(present.sum()), "aligned_finite_changes": int(np.isfinite(change).sum()),
             "strict_prior_fresh_pass": timing}
    return level, change, provenance, stats


def _rolling(values, window, operation):
    """Direct window reductions, with complete-window warmup and no filling."""
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        if operation == "std":
            result[window - 1:] = np.std(windows, axis=1, ddof=1)
        elif operation == "mean":
            result[window - 1:] = np.mean(windows, axis=1)
        else:
            result[window - 1:] = np.sum(windows, axis=1)
    return result


def entry_feature_arrays(prices, dates, base_returns, pair, native):
    """Return 13 unsigned features plus provenance, known inputs and statistics.

    Reserved dictionary entries are ``_provenance`` (five date arrays),
    ``_known_inputs`` (seven raw K arrays), and ``_validation`` (serializable
    scope/timing statistics). Dates must remain inside native's decode cutoff.
    All numerical feature arrays have the same length as ``dates``.
    """
    dates = _calendar(dates, "quote dates")
    prices, base_returns = np.asarray(prices, float), np.asarray(base_returns, float)
    if (prices.ndim != 1 or base_returns.ndim != 1 or len(prices) != len(dates)
            or len(base_returns) != len(dates) or not np.isfinite(prices).all()
            or np.any(prices <= 0) or not np.isfinite(base_returns).all()):
        raise ValueError("Positive finite prices and finite original BASE returns must align with dates")
    if pair not in COUNTRIES:
        raise ValueError("Unknown currency: " + str(pair))
    metadata = native.get("_validation", {})
    cutoff = pd.Timestamp(metadata.get("cutoff", dates[-1]))
    if dates[-1] > cutoff:
        raise ValueError("Quote calendar exceeds the explicitly decoded native cutoff")
    for name in (name for names in REQUIRED_SHEETS.values() for name in names):
        if name not in native or not isinstance(native[name], pd.Series):
            raise ValueError("Missing native Series: " + name)
        source_dates = _calendar(native[name].index, name, allow_empty=True)
        if len(source_dates) and source_dates[-1] > cutoff:
            raise ValueError("Native series exceeds its decode cutoff: " + name)

    changes = np.r_[np.nan, prices[1:] / prices[:-1] - 1.]
    vol20 = _rolling(changes, 20, "std") * np.sqrt(252.) * 100
    vol63 = _rolling(changes, 63, "std") * np.sqrt(252.) * 100
    length63 = _rolling(np.r_[np.nan, np.abs(np.diff(prices))], 63, "sum")
    move63, return20 = np.full(len(prices), np.nan), np.full(len(prices), np.nan)
    move63[63:] = np.abs(prices[63:] - prices[:-63])
    return20[20:] = (prices[20:] / prices[:-20] - 1.) * 100
    scale = vol20 * np.sqrt(20. / 252.)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = {
            "vol_ratio_20_63": vol20 / np.where(vol63 > 0, vol63, np.nan),
            "realized_vol_20_pct": vol20,
            "efficiency_63": np.where(length63 == 0, 0., move63 / length63),
            "direction_return_20_vol": return20 / np.where(scale > 0, scale, np.nan),
            "direction_distance_ma_60_vol": (prices / _rolling(prices, 60, "mean") - 1.) * 100 / np.where(scale > 0, scale, np.nan),
        }
    # Rate differences use only dates on which BOTH native yields are finite.
    country = native[COUNTRIES[pair] + "_2Y"]
    us = native["US_2Y"]
    shared = country.index.intersection(us.index).sort_values()
    spread = pd.Series((country.reindex(shared).to_numpy() - us.reindex(shared).to_numpy()) * 100, index=shared)
    raw, provenance, checks = {}, {}, {}
    for source, prefix, steps, span, percent, level_name, change_name in (
        (native["VIX"], "vix", 5, 12, False, "K_vix", "K_vix_d5"),
        (native["DXY"], "dxy", 20, 45, True, None, "K_dxy_d20_pct"),
        (native["BRENT"], "brent", 20, 45, True, None, "K_brent_d20_pct"),
        (native["COPPER_LME"], "copper", 20, 45, True, None, "K_copper_d20_pct"),
        (spread, "rate", 20, 45, False, "K_spread_bp", "K_spread_d20_bp"),
    ):
        level, change, observed, check = _align(source, dates, steps, span, percent)
        if level_name is not None:
            raw[level_name] = level
        raw[change_name] = change
        provenance["K_" + prefix + "_obs_date"] = observed
        checks[prefix] = check
    values.update(K_vix=raw["K_vix"], K_vix_d5=raw["K_vix_d5"],
                  direction_spread_bp=raw["K_spread_bp"], direction_spread_d20_bp=raw["K_spread_d20_bp"],
                  direction_dxy_d20_pct=raw["K_dxy_d20_pct"],
                  direction_commodity_d20_pct=raw["K_copper_d20_pct"] if pair == "AUD" else
                  raw["K_brent_d20_pct"] if pair in ("CAD", "NOK") else np.full(len(dates), np.nan))
    for window in (63, 252):
        values[f"base_pnl_{window}_pct"] = _rolling(base_returns, window, "sum") * 100
    return {**{name: values[name] for name in FEATURE_NAMES}, "_provenance": provenance, "_known_inputs": raw,
            "_validation": {"pair": pair, "observations": len(dates), "native_decode_cutoff": cutoff.isoformat(),
                            "date_start": dates[0].isoformat(), "date_end": dates[-1].isoformat(),
                            "direction_applied": False, "shadow_returns_source": "caller_supplied_original_BASE",
                            "native_alignment": checks,
                            "finite_feature_observations": {name: int(np.isfinite(values[name]).sum()) for name in FEATURE_NAMES}}}
