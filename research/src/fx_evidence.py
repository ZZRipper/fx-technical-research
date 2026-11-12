"""Read archived evidence and compare independently recomputed research results.

The Explorer curves and saved notebook outputs are references, never backtests
executed by this module. Original frozen files and their provenance are not
reconstructed from these partial exports.
"""

from __future__ import annotations

import base64
from functools import lru_cache
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


CORE_RULES = ("RSI_MR_14", "BB_MR_20", "TSMOM_63", "MA_20_60")
FILTER_VARIANTS = (
    "BASE", "RATE_CHANGE_CONFLICT", "DUAL_RATE_CONFLICT", "POLICY_USD",
    "RISK_YIELD", "COMMODITY_USD", "ECO_COMPOSITE", "RISK_YIELD_EXIT",
)
COMMODITY_PAIRS = ("AUD", "CAD", "NOK")
ARCHIVED_TRADE_COLUMNS = (
    "entry_epoch_day", "exit_epoch_day", "trade_return_pct",
    "holding_observations", "closed", "direction",
)


def _root(root=None):
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]


def _data_path(filename, root=None):
    project = _root(root)
    candidates = (
        project / "FX_Research_Datasets" / "data" / "clean" / filename,
        project / "data" / "clean" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Archived clean reference is missing: {filename}")


@lru_cache(maxsize=2)
def _read_explorer(path_string, modified_ns, size_bytes):
    html = Path(path_string).read_text(encoding="utf-8")
    match = re.search(
        r'<script\b[^>]*\bid=[\"\x27]research-data[\"\x27][^>]*>(.*?)</script>',
        html, flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Explorer does not contain its research-data archive")
    payload = json.loads(gzip.decompress(base64.b64decode(match.group(1).strip(), validate=True)))
    required = {"version", "capital", "calendars", "accounts", "rules", "variants", "sources", "roundingUsd"}
    if not required.issubset(payload):
        raise ValueError(f"Incomplete Explorer schema: {sorted(required.difference(payload))}")
    identifiers = set()
    for account in payload["accounts"]:
        if account["id"] in identifiers:
            raise ValueError(f"Duplicate archived account: {account['id']}")
        identifiers.add(account["id"])
        calendar = payload["calendars"][account["cal"]]
        if len(calendar) != len(account["equity"]) or not calendar:
            raise ValueError(f"Incomplete archived path: {account['id']}")
        if any(later <= earlier for earlier, later in zip(calendar, calendar[1:])):
            raise ValueError(f"Unordered archived calendar: {account['id']}")
        if not np.isfinite(np.asarray(account["equity"], dtype=float)).all():
            raise ValueError(f"Nonfinite archived equity: {account['id']}")
    return payload


def load_explorer(root=None):
    """Return the cached, unmodified archive dictionary; callers must not mutate it."""
    path = _root(root) / "FX_PnL_Explorer.html"
    metadata = path.stat()
    return _read_explorer(str(path), metadata.st_mtime_ns, metadata.st_size)


def explorer_account_dates(account, root=None):
    """Decode a reference account's epoch-day calendar without resampling."""
    archive = load_explorer(root)
    return pd.DatetimeIndex(pd.to_datetime(archive["calendars"][account["cal"]], unit="D", origin="unix"))


def baseline_reference(root=None):
    """Read the delivered baseline metrics as an independent comparison reference."""
    path = _data_path("fixed_baseline_metrics_rebuilt.csv", root)
    frame = pd.read_csv(path)
    frame.attrs.update(evidence_status="archived_not_recomputed", source_file=str(path))
    return frame


def archived_account_metrics(dataset="main", root=None):
    """Expose available Explorer metadata, explicitly marked as archived evidence."""
    archive = load_explorer(root)
    rows = []
    for account in archive["accounts"]:
        if account["ds"] != dataset:
            continue
        dates = explorer_account_dates(account, root)
        quality = account.get("quality_full") or {}
        rows.append({
            "pair": account["pair"], "strategy": account["rule"],
            "base_strategy": account["rule"], "variant": account["variant"],
            "dataset": dataset, "start_date": dates[0], "end_date": dates[-1],
            "observations": len(dates), **account.get("source_metrics", {}),
            "eligible_closed_n": quality.get("closed", np.nan),
            "win_rate": quality.get("win", np.nan),
            "payoff_ratio": quality.get("payoff", np.nan),
            "source_account_id": account["id"],
            "evidence_status": "archived_not_recomputed",
        })
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        evidence_status="archived_not_recomputed",
        source_file=str(_root(root) / "FX_PnL_Explorer.html"),
    )
    return frame


def archived_trade_ledger(account, root=None):
    """Decode only the six exported trade fields; missing original fields stay absent."""
    if account.get("trades") is None:
        raise ValueError(f"Trade ledger was not archived for {account['id']}")
    trades = pd.DataFrame(account["trades"], columns=ARCHIVED_TRADE_COLUMNS)
    trades["entry_date"] = pd.to_datetime(trades.entry_epoch_day, unit="D", origin="unix")
    trades["exit_date"] = pd.to_datetime(trades.exit_epoch_day.where(trades.exit_epoch_day.ge(0)), unit="D", origin="unix")
    trades["source_account_id"] = account["id"]
    trades["evidence_status"] = "archived_not_recomputed"
    trades.attrs.update(
        evidence_status="archived_not_recomputed",
        source_file=str(_root(root) / "FX_PnL_Explorer.html"),
    )
    return trades


def _check(name, expected_rows, compared_rows, passed, source, error=np.nan, mismatches=0, tolerance=np.nan):
    return {
        "check": name, "expected_rows": int(expected_rows),
        "compared_rows": int(compared_rows), "max_abs_error": float(error),
        "mismatch_count": int(mismatches), "absolute_tolerance": float(tolerance),
        "pass": bool(passed), "reference_source": source,
    }


def _numeric_check(name, actual, reference, expected_rows, source, tolerance=1e-7, relative_tolerance=1e-9):
    actual_values = np.asarray(actual, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    matches = np.isclose(actual_values, reference_values, atol=tolerance, rtol=relative_tolerance, equal_nan=True)
    finite = np.isfinite(actual_values) & np.isfinite(reference_values)
    error = float(np.max(np.abs(actual_values[finite] - reference_values[finite]))) if finite.any() else 0.0
    mismatches = int((~matches).sum())
    if mismatches and not finite.all():
        error = float("inf")
    return _check(
        name, expected_rows, len(actual_values),
        len(actual_values) == expected_rows and expected_rows > 0 and mismatches == 0,
        source, error=error, mismatches=mismatches, tolerance=tolerance,
    )


def _compare_frames(actual, reference, keys, source, prefix, columns=None):
    missing_keys = set(keys).difference(actual.columns)
    if missing_keys:
        return [_check(f"{prefix}: account/date keys missing", len(reference), 0, False, source)]
    actual_keys = set(actual[keys].itertuples(index=False, name=None))
    reference_keys = set(reference[keys].itertuples(index=False, name=None))
    keys_valid = (
        not actual.duplicated(keys).any() and not reference.duplicated(keys).any()
        and actual_keys == reference_keys and len(reference) > 0
    )
    checks = [_check(
        f"{prefix}: exact key coverage", len(reference), len(actual_keys.intersection(reference_keys)),
        keys_valid, source, mismatches=len(actual_keys.symmetric_difference(reference_keys)),
    )]
    joined = actual.drop_duplicates(keys).merge(reference.drop_duplicates(keys), on=keys, suffixes=("_new", "_reference"))
    selected = columns if columns is not None else [column for column in reference.columns if column not in keys]
    for column in selected:
        if column not in actual.columns:
            checks.append(_check(f"{prefix}: {column} missing", len(reference), 0, False, source))
            continue
        actual_values = joined[f"{column}_new"]
        reference_values = joined[f"{column}_reference"]
        if pd.api.types.is_numeric_dtype(reference[column]) and not pd.api.types.is_bool_dtype(reference[column]):
            checks.append(_numeric_check(f"{prefix}: {column}", actual_values, reference_values, len(reference), source))
            continue
        if column == "date" or column.endswith("_date"):
            actual_values = pd.to_datetime(actual_values)
            reference_values = pd.to_datetime(reference_values)
        matches = actual_values.eq(reference_values) | (actual_values.isna() & reference_values.isna())
        checks.append(_check(
            f"{prefix}: {column}", len(reference), len(joined),
            len(joined) == len(reference) and len(reference) > 0 and bool(matches.all()),
            source, mismatches=int((~matches).sum()),
        ))
    return checks


def _path_check(run, run_dates, account, archive, root, name):
    reference_dates = explorer_account_dates(account, root)
    expected_rows = len(reference_dates)
    source = f"FX_PnL_Explorer.html#{account['id']}"
    if run is None or run_dates is None:
        return _check(name, expected_rows, 0, False, source), 0
    dates = pd.DatetimeIndex(pd.to_datetime(run_dates))
    if "equity_usd" in run:
        equity = np.asarray(run["equity_usd"], dtype=float)
    elif "cumulative_pnl_usd" in run:
        equity = np.asarray(run["cumulative_pnl_usd"], dtype=float) + archive["capital"]
    else:
        equity = np.asarray(run["equity"], dtype=float) * archive["capital"]
    if len(equity) != len(dates) or dates.has_duplicates or not dates.is_monotonic_increasing:
        return _check(name, expected_rows, 0, False, source), 0
    actual_path = pd.Series(equity, index=dates)
    overlapping = reference_dates.intersection(dates)
    reference_path = pd.Series(account["equity"], index=reference_dates, dtype=float)
    check = _numeric_check(
        name, actual_path.loc[overlapping], reference_path.loc[overlapping],
        expected_rows, source, tolerance=float(archive["roundingUsd"]) + 1e-7,
        relative_tolerance=0.0,
    )
    return check, len(overlapping)


def verify_baselines(metrics, runs, dates, root=None):
    """Compare all baseline metrics and 207 daily paths against archived exports."""
    reference = baseline_reference(root)
    checks = _compare_frames(metrics, reference, ["pair", "strategy"], reference.attrs["source_file"], "baseline metrics")
    archive = load_explorer(root)
    accounts = [account for account in archive["accounts"] if account["ds"] == "main" and account["variant"] == "BASE"]
    reference_keys = {(account["pair"], account["rule"]) for account in accounts}
    metric_keys = set(reference[["pair", "strategy"]].itertuples(index=False, name=None))
    checks.append(_check(
        "baseline reference account coverage", 207, len(reference_keys),
        len(reference_keys) == 207 and len(reference) == 207 and reference_keys == metric_keys,
        "FX_PnL_Explorer.html + fixed_baseline_metrics_rebuilt.csv",
    ))
    compared_daily_rows = 0
    for account in accounts:
        key = (account["pair"], account["rule"])
        run = runs.get(key)
        run_dates = run.get("dates") if run is not None else None
        if run_dates is None:
            run_dates = dates.get(account["pair"])
        check, compared_rows = _path_check(run, run_dates, account, archive, root, f"baseline daily: {key[0]} {key[1]}")
        checks.append(check)
        compared_daily_rows += compared_rows
    return pd.DataFrame(checks), compared_daily_rows


def verify_known_features(known, root=None):
    """Check values, missingness, observation dates, and the strict K-time limit."""
    path = _data_path("economic_known_inputs_rebuilt.csv.gz", root)
    reference = pd.read_csv(path)
    actual = known.copy()
    for frame in (reference, actual):
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"])
    checks = _compare_frames(actual, reference, ["pair", "date"], str(path), "known economic inputs")
    for column in reference.columns:
        if not column.endswith("_obs_date") or column not in actual or "date" not in actual:
            continue
        observation_dates = pd.to_datetime(actual[column])
        present = observation_dates.notna()
        age = (actual.loc[present, "date"] - observation_dates[present]).dt.total_seconds() / 86400
        valid = age.gt(0) & age.le(7)
        checks.append(_check(
            f"known timing: {column} strictly earlier and at most 7 days old",
            int(present.sum()), len(age), len(age) > 0 and bool(valid.all()),
            "recomputed observation dates", mismatches=int((~valid).sum()),
        ))
    return pd.DataFrame(checks)


def verify_filter_metrics_and_paths(metrics, runs, dates=None, root=None):
    """Validate 288 recomputed filter accounts, including explicit nonapplicable fallbacks.

    The Explorer omits 24 COMMODITY_USD accounts outside AUD/CAD/NOK; their
    expected result is the corresponding BASE account because that gate does
    not apply. These comparisons name their BASE reference explicitly.
    """
    archive = load_explorer(root)
    account_lookup = {
        (account["pair"], account["rule"], account["variant"]): account
        for account in archive["accounts"] if account["ds"] == "main"
    }
    expected_keys = {
        (pair, rule, variant)
        for pair in archive["pairs"] for rule in CORE_RULES for variant in FILTER_VARIANTS
    }
    keys = ["pair", "base_strategy", "variant"]
    if not set(keys).issubset(metrics.columns):
        return pd.DataFrame([_check("filter metric keys missing", 288, 0, False, "Explorer filter schema")])
    actual_keys = set(metrics[keys].itertuples(index=False, name=None))
    checks = [_check(
        "filter exact account coverage", 288, len(actual_keys.intersection(expected_keys)),
        len(expected_keys) == 288 and actual_keys == expected_keys and not metrics.duplicated(keys).any(),
        "Explorer filter schema", mismatches=len(actual_keys.symmetric_difference(expected_keys)),
    )]
    if isinstance(runs, pd.DataFrame):
        path_lookup = {key: frame.sort_values("date") for key, frame in runs.groupby(keys, sort=False)}
    else:
        path_lookup = runs
    indexed_metrics = metrics.drop_duplicates(keys).set_index(keys)
    for key in sorted(expected_keys):
        pair, rule, variant = key
        fallback = variant == "COMMODITY_USD" and pair not in COMMODITY_PAIRS
        reference_key = (pair, rule, "BASE") if fallback else key
        account = account_lookup.get(reference_key)
        label = " ".join(key)
        if account is None:
            checks.append(_check(f"filter reference missing: {label}", 1, 0, False, "FX_PnL_Explorer.html"))
            continue
        source = f"FX_PnL_Explorer.html#{account['id']}"
        if fallback:
            source += " (COMMODITY_USD does not apply; BASE reference)"
        expected_metrics = dict(account.get("source_metrics", {}))
        quality = account.get("quality_full")
        if quality is not None:
            expected_metrics.update(
                eligible_closed_n=quality["closed"], win_rate=quality["win"], payoff_ratio=quality["payoff"],
            )
        for column in ("total_return_pct", "max_drawdown_pct", "eligible_closed_n", "win_rate", "payoff_ratio"):
            if key not in indexed_metrics.index or column not in metrics or column not in expected_metrics:
                checks.append(_check(f"filter {column} missing: {label}", 1, 0, False, source))
                continue
            expected = expected_metrics[column]
            checks.append(_numeric_check(
                f"filter {column}: {label}", [indexed_metrics.loc[key, column]],
                [np.nan if expected is None else expected], 1, source,
            ))
        run = path_lookup.get(key)
        if isinstance(run, pd.DataFrame):
            run_dates = run["date"]
        else:
            run_dates = run.get("dates") if run is not None else None
        if run_dates is None and dates is not None:
            run_dates = dates.get(pair) if isinstance(dates, dict) else dates
        check, compared_rows = _path_check(run, run_dates, account, archive, root, f"filter daily: {label}")
        check["reference_source"] = source
        checks.append(check)
    return pd.DataFrame(checks)


@lru_cache(maxsize=2)
def _read_notebook_evidence(path_string, modified_ns, size_bytes):
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


def _saved_notebook(root=None):
    project = _root(root)
    path = project / "reference" / "notebook_saved_outputs.json"
    if not path.is_file():
        path = project / "FX_Technical_Research_Complete.ipynb"
    metadata = path.stat()
    return path, _read_notebook_evidence(str(path), metadata.st_mtime_ns, metadata.st_size)


class _SavedTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.rows = None
        self.row = None
        self.cell = None
        self.header = False

    def handle_starttag(self, tag, attributes):
        if tag == "table":
            self.rows = []
        elif tag == "thead" and self.rows is not None:
            self.header = True
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in ("th", "td") and self.row is not None:
            properties = dict(attributes)
            self.cell = {
                "parts": [], "rowspan": int(properties.get("rowspan", "1")),
                "colspan": int(properties.get("colspan", "1")),
            }

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["parts"].append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.cell["text"] = "".join(self.cell.pop("parts")).strip()
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append((self.header, self.row))
            self.row = None
        elif tag == "thead":
            self.header = False
        elif tag == "table" and self.rows is not None:
            self.tables.append(self.rows)
            self.rows = None


def _saved_html_tables(html):
    parser = _SavedTableParser()
    parser.feed(html)
    frames = []
    for table in parser.tables:
        expanded = []
        pending_spans = {}
        for is_header, cells in table:
            values = {}
            remaining_spans = {}
            for column_index, (remaining_rows, value) in pending_spans.items():
                values[column_index] = value
                if remaining_rows > 1:
                    remaining_spans[column_index] = (remaining_rows - 1, value)
            column_index = 0
            for cell in cells:
                while column_index in values:
                    column_index += 1
                for offset in range(cell["colspan"]):
                    position = column_index + offset
                    values[position] = cell["text"]
                    if cell["rowspan"] > 1:
                        remaining_spans[position] = (cell["rowspan"] - 1, cell["text"])
                column_index += cell["colspan"]
            pending_spans = remaining_spans
            expanded.append((is_header, values))
        width = max((max(values, default=-1) + 1 for _, values in expanded), default=0)
        headers = [[values.get(position, "") for position in range(width)] for is_header, values in expanded if is_header]
        body = [[values.get(position, "") for position in range(width)] for is_header, values in expanded if not is_header]
        if len(headers) > 1:
            columns = pd.MultiIndex.from_arrays(headers)
        else:
            columns = headers[0] if headers else list(range(width))
        frame = pd.DataFrame(body, columns=columns)
        if not frame.columns.has_duplicates:
            for column in frame.columns:
                values = frame[column].replace({"NaN": np.nan, "nan": np.nan, "NaT": np.nan})
                if values.dropna().isin(["True", "False"]).all() and values.notna().any():
                    frame[column] = values.map({"True": True, "False": False})
                else:
                    try:
                        frame[column] = pd.to_numeric(values)
                    except (ValueError, TypeError):
                        frame[column] = values
        frames.append(frame)
    return frames


def saved_notebook_table(cell_index, table_index=0, root=None):
    """Read a saved display table, which may be rounded or truncated, as history only."""
    path, notebook = _saved_notebook(root)
    cell = notebook["cells"][cell_index]
    tables = []
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html:
            html = "".join(html) if isinstance(html, list) else html
            if "<table" in html.lower():
                tables.extend(_saved_html_tables(html))
    if table_index < 0 or table_index >= len(tables):
        raise IndexError(f"No saved table {table_index} in notebook cell {cell_index}; found {len(tables)}")
    frame = tables[table_index]
    frame.attrs.update(
        evidence_status="archived_display_not_recomputed",
        source_file=str(path), original_cell_index=cell_index,
        original_table_index=table_index,
        source_sha256=notebook.get("source_sha256"),
        precision_note="Saved notebook display precision; may omit rows or columns.",
    )
    return frame


def print_saved_output(cell_index, root=None):
    """Print archived textual output with an explicit provenance banner."""
    path, notebook = _saved_notebook(root)
    print(f"ARCHIVED DISPLAY, NOT RECOMPUTED: {path.name}, original cell {cell_index}")
    for output in notebook["cells"][cell_index].get("outputs", []):
        value = output.get("data", {}).get("text/plain", output.get("text"))
        if value:
            print("".join(value) if isinstance(value, list) else value)
