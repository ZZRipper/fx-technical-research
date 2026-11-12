"""Reconstructed FX rules and accounting from the two delivered notebooks.

The original exploration preserves its original RSI transition order. The fixed
baseline uses the corrected RSI and execution functions printed in the complete
notebook. Prices are read from the raw workbook; archived results are never used
as inputs to these calculations.

The low-sample cutoff follows the delivered Explorer's 20-completed-trade
display threshold and reproduces every delivered baseline flag.
"""

from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree
import zipfile

import numpy as np
import pandas as pd


DATA_ROOT = Path(__file__).resolve().parents[1] / 'FX_Research_Datasets' / 'data'
CURRENCIES = ('EUR', 'GBP', 'AUD', 'NZD', 'JPY', 'CHF', 'CAD', 'NOK', 'SEK')
CORE_RULES = ['RSI_MR_14', 'BB_MR_20', 'MA_20_60', 'TSMOM_63']
MODES = ('Daily Continuous', 'Daily Weekflat', 'Weekly Flat', 'Fixed 5D')
RULES = {
    'RSI_MR_7': ('RSI', 'Mean reversion', '7d Wilder RSI re-enters 30..50 / 50..70; exit at 50'),
    'RSI_MR_14': ('RSI', 'Mean reversion', '14d Wilder RSI re-enters 30..50 / 50..70; exit at 50'),
    'RSI_MR_20': ('RSI', 'Mean reversion', '20d Wilder RSI re-enters 30..50 / 50..70; exit at 50'),
    'RSI_MOM_7': ('RSI', 'Momentum', '7d RSI >60 long FX, <40 short FX, otherwise flat'),
    'RSI_MOM_14': ('RSI', 'Momentum', '14d RSI >60 long FX, <40 short FX, otherwise flat'),
    'RSI_MOM_20': ('RSI', 'Momentum', '20d RSI >60 long FX, <40 short FX, otherwise flat'),
    'MA_10_30': ('Moving average', 'Trend following', 'Sign of SMA10 minus SMA30'),
    'MA_20_60': ('Moving average', 'Trend following', 'Sign of SMA20 minus SMA60'),
    'BB_MR_10': ('Bollinger', 'Mean reversion', '10d sample-SD z-score: fade +/-2, exit at zero'),
    'BB_MR_20': ('Bollinger', 'Mean reversion', '20d sample-SD z-score: fade +/-2, exit at zero'),
    'BB_BO_10': ('Bollinger', 'Breakout', '10d sample-SD z-score: follow +/-2, exit at zero'),
    'BB_BO_20': ('Bollinger', 'Breakout', '20d sample-SD z-score: follow +/-2, exit at zero'),
    'DONCHIAN_BO_10': ('Donchian', 'Breakout', 'Break prior 10 closes high/low; exit through midpoint'),
    'DONCHIAN_BO_20': ('Donchian', 'Breakout', 'Break prior 20 closes high/low; exit through midpoint'),
    'MACD_12_26_9': ('MACD', 'Trend following', 'Sign of EMA12 minus EMA26 minus signal EMA9'),
    'TSMOM_21': ('Time-series momentum', 'Momentum', 'Sign of 21d log price change'),
    'TSMOM_63': ('Time-series momentum', 'Momentum', 'Sign of 63d log price change'),
    'VOTE_EQUAL_4': ('Signal voting', 'Ensemble', 'Equal RSI MOM14, MA20/60, Donchian20, MACD votes; ties flat'),
    'VOTE_STRENGTH_4': ('Signal voting', 'Ensemble', 'Deterministic conviction-weighted votes; cutoff +/-0.2'),
    'REGIME_RULE_FIXED': ('Regime switch', 'Fixed regime filter', '63d efficiency >0.40: MA20/60; <0.20: RSI MR14; otherwise flat'),
    'BUY_HOLD_LONG_FX': ('Benchmark', 'Benchmark', 'Always long foreign currency'),
    'BUY_HOLD_LONG_USD': ('Benchmark', 'Benchmark', 'Always short foreign currency'),
}
STRATEGIES = list(RULES)[:20]
FIXED_BENCHMARKS = ('ALWAYS_LONG_FX', 'ALWAYS_SHORT_FX', 'CASH')
FLAT_TOLERANCE = 1e-12
_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def _workbook_rows(archive, sheet_path, columns=None):
    shared = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        root = ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))
        shared = [''.join(node.itertext()) for node in root]
    with archive.open(sheet_path) as stream:
        for event, element in ElementTree.iterparse(stream, events=('end',)):
            if element.tag != _NS + 'row':
                continue
            values = {}
            for cell in element.findall(_NS + 'c'):
                column = re.match(r'[A-Z]+', cell.get('r', '')).group()
                if columns is not None and column not in columns:
                    continue
                kind = cell.get('t', '')
                value = cell.find(_NS + 'v')
                if kind == 'inlineStr':
                    parsed = ''.join(node.text or '' for node in cell.iter(_NS + 't'))
                elif value is None or not value.text:
                    parsed = None
                elif kind == 's':
                    parsed = shared[int(value.text)]
                elif kind in ('str', 'e'):
                    parsed = value.text
                else:
                    parsed = float(value.text)
                values[column] = parsed
            yield values
            element.clear()


def read_xlsx_sheet(path, sheet_name):
    """Read a worksheet's cached values without an optional Excel dependency."""
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read('xl/workbook.xml'))
        relationships = ElementTree.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {element.get('Id'): element.get('Target') for element in relationships}
        relation_ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
        matching = [sheet for sheet in workbook.find(_NS + 'sheets') if sheet.get('name') == sheet_name]
        if not matching:
            raise ValueError(f'Worksheet {sheet_name!r} not found in {path}.')
        target = targets[matching[0].get(relation_ns + 'id')]
        sheet_path = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        rows = _workbook_rows(archive, sheet_path)
        header = next(rows)
        physical_rows = list(rows)
        columns = set(header).union(*(row.keys() for row in physical_rows))
        def column_number(column):
            number = 0
            for character in column:
                number = number * 26 + ord(character) - ord('A') + 1
            return number
        ordered = sorted(columns, key=column_number)
        names = {column: header.get(column) if header.get(column) is not None else f'Unnamed: {index}'
                 for index, column in enumerate(ordered)}
        return pd.DataFrame([{names[column]: row.get(column) for column in ordered} for row in physical_rows])


def canonical_prices_from_raw(data_root=None):
    """Rebuild the USD-per-foreign panel and original monthly indicators."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    candidates = sorted((root / 'raw').glob('FX_Technical_PIT_Dataset*.xlsx'))
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one PIT source workbook under {root / "raw"}.')
    records = []
    monthend = {pair: [] for pair in CURRENCIES}
    with zipfile.ZipFile(candidates[0]) as archive:
        rows = _workbook_rows(archive, 'xl/worksheets/sheet6.xml', {'B', 'D', 'E', 'F', 'N'})
        next(rows)
        for row in rows:
            if row.get('D') not in CURRENCIES or row.get('E') is None:
                continue
            records.append({
                'date': pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(row['B'])),
                'pair': row['D'], 'price_USD_per_foreign': float(row['E']),
                'source_log_return_1d': row.get('F'), 'core_ready': int(row.get('N') or 0),
            })
        mapping = {'J': 'spot', 'P': 'rsi14', 'Q': 'sma50gap', 'R': 'sma200gap',
                   'S': 'trend50_200', 'T': 'macdhist', 'U': 'bbz20', 'V': 'rv21', 'W': 'rv63'}
        rows = _workbook_rows(archive, 'xl/worksheets/sheet4.xml', {'B', 'E', *mapping})
        next(rows)
        for row in rows:
            if row.get('E') in CURRENCIES:
                record = {name: row.get(column) for column, name in mapping.items()}
                record['date'] = (pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(row['B']))).strftime('%Y-%m-%d')
                monthend[row['E']].append(record)
    prices = pd.DataFrame(records).sort_values(['pair', 'date']).reset_index(drop=True)
    if prices.duplicated(['pair', 'date']).any() or not np.all(np.isfinite(prices.price_USD_per_foreign) & (prices.price_USD_per_foreign > 0)):
        raise ValueError('Duplicate dates or invalid source prices.')
    return prices, monthend


def sma(values, window):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    cumulative = np.cumsum(np.where(valid, values, 0.0))
    counts = np.cumsum(valid.astype(int))
    if len(values) >= window:
        totals = cumulative[window - 1:] - np.r_[0.0, cumulative[:-window]]
        count = counts[window - 1:] - np.r_[0, counts[:-window]]
        result[window - 1:] = np.where(count == window, totals / window, np.nan)
    return result


def rolling_std(values, window, ddof=1):
    values = np.asarray(values, float)
    mean = sma(values, window)
    return np.sqrt(np.maximum((sma(values * values, window) - mean * mean) * window / (window - ddof), 0.0))


def ema(values, window):
    result = np.full(len(values), np.nan)
    weight = 2.0 / (window + 1)
    seeded = False
    for index, value in enumerate(values):
        if np.isfinite(value):
            result[index] = value if not seeded else weight * value + (1 - weight) * result[index - 1]
            seeded = True
    return result


def rsi_wilder(prices, window):
    """Wilder RSI with the raw workbook's first-observation seed."""
    changes = np.diff(prices)
    gains, losses = np.maximum(changes, 0.0), np.maximum(-changes, 0.0)
    result = np.full(len(prices), np.nan)
    if len(prices) <= window:
        return result
    average_gain, average_loss = float(gains[0]), float(losses[0])
    for index in range(2, len(prices)):
        average_gain = ((window - 1) * average_gain + gains[index - 1]) / window
        average_loss = ((window - 1) * average_loss + losses[index - 1]) / window
        if index >= window:
            result[index] = (50.0 if average_gain == average_loss == 0 else
                             100.0 if average_loss == 0 else
                             100.0 - 100.0 / (1.0 + average_gain / average_loss))
    return result


def sign_strict(values):
    return np.where(np.isfinite(values), np.sign(values), 0).astype(np.int8)


def event_osc_reversion(values, low=30.0, high=70.0, center=50.0):
    """Re-enter before the exit level, preserving state across missing values."""
    signals = np.zeros(len(values), np.int8)
    position = 0
    for index in range(1, len(values)):
        value, previous = values[index], values[index - 1]
        if not np.isfinite(value) or not np.isfinite(previous):
            signals[index] = position
            continue
        if position == 1 and value >= center:
            position = 0
        elif position == -1 and value <= center:
            position = 0
        if previous < low <= value < center:
            position = 1
        elif previous > high >= value > center:
            position = -1
        signals[index] = position
    return signals


def _original_osc_reversion(values, low=30.0, high=70.0, center=50.0):
    signals = np.zeros(len(values), np.int8)
    position = 0
    for index in range(1, len(values)):
        value, previous = values[index], values[index - 1]
        if not np.isfinite(value) or not np.isfinite(previous):
            continue
        if previous < low <= value:
            position = 1
        elif previous > high >= value:
            position = -1
        elif position == 1 and value >= center:
            position = 0
        elif position == -1 and value <= center:
            position = 0
        signals[index] = position
    return signals


def bb_strategy(scores, reversion):
    signals = np.zeros(len(scores), np.int8)
    position = 0
    for index, value in enumerate(scores):
        if not np.isfinite(value):
            continue
        if value >= 2:
            position = -1 if reversion else 1
        elif value <= -2:
            position = 1 if reversion else -1
        elif position == 1 and (value >= 0 if reversion else value <= 0):
            position = 0
        elif position == -1 and (value <= 0 if reversion else value >= 0):
            position = 0
        signals[index] = position
    return signals


def donchian(prices, window):
    signals = np.zeros(len(prices), np.int8)
    midpoint = np.full(len(prices), np.nan)
    position = 0
    for index in range(window, len(prices)):
        previous = prices[index - window:index]
        highest, lowest = previous.max(), previous.min()
        midpoint[index] = (highest + lowest) / 2
        if prices[index] > highest:
            position = 1
        elif prices[index] < lowest:
            position = -1
        elif position == 1 and prices[index] < midpoint[index]:
            position = 0
        elif position == -1 and prices[index] > midpoint[index]:
            position = 0
        signals[index] = position
    return signals, midpoint


def complete_signals(prices, original=False):
    prices = np.asarray(prices, float)
    length = len(prices)
    log_prices = np.log(prices)
    log_returns = np.r_[np.nan, np.diff(log_prices)]
    volatility21 = rolling_std(log_returns, 21)
    path_length = sma(np.abs(log_returns), 63) * 63
    displacement = np.full(length, np.nan)
    displacement[63:] = np.abs(log_prices[63:] - log_prices[:-63])
    efficiency = np.divide(displacement, path_length, out=np.full(length, np.nan), where=np.isfinite(path_length) & (path_length > 0))
    signals, cache = {}, {}
    oscillator = _original_osc_reversion if original else event_osc_reversion
    for window in (7, 14, 20):
        scores = rsi_wilder(prices, window)
        cache[f'rsi{window}'] = scores
        signals[f'RSI_MR_{window}'] = oscillator(scores)
        signals[f'RSI_MOM_{window}'] = np.where(scores > 60, 1, np.where(scores < 40, -1, 0)).astype(np.int8)
    for fast, slow in ((10, 30), (20, 60)):
        gap = sma(prices, fast) / sma(prices, slow) - 1
        signals[f'MA_{fast}_{slow}'] = sign_strict(gap)
        cache[f'magap_{fast}_{slow}'] = gap
    for window in (10, 20):
        deviation = rolling_std(prices, window)
        scores = np.divide(prices - sma(prices, window), deviation, out=np.full(length, np.nan), where=np.isfinite(deviation) & (deviation > 1e-12))
        cache[f'bbz{window}'] = scores
        signals[f'BB_MR_{window}'] = bb_strategy(scores, True)
        signals[f'BB_BO_{window}'] = bb_strategy(scores, False)
        signals[f'DONCHIAN_BO_{window}'], cache[f'donmid{window}'] = donchian(prices, window)
    macd = ema(prices, 12) - ema(prices, 26)
    histogram = (macd - ema(macd, 9)) / prices
    signals['MACD_12_26_9'] = sign_strict(histogram)
    for window in (21, 63):
        momentum = np.full(length, np.nan)
        momentum[window:] = log_prices[window:] - log_prices[:-window]
        signals[f'TSMOM_{window}'] = sign_strict(momentum)
    voters = ['RSI_MOM_14', 'MA_20_60', 'DONCHIAN_BO_20', 'MACD_12_26_9']
    votes = np.stack([signals[name].astype(float) for name in voters])
    signals['VOTE_EQUAL_4'] = sign_strict(votes.sum(axis=0))
    scale20 = np.maximum(volatility21 * np.sqrt(20), 1e-7)
    weights = np.nan_to_num(np.stack([
        np.clip(np.abs(cache['rsi14'] - 50) / 25, 0, 1),
        np.clip(np.abs(cache['magap_20_60']) / scale20, 0, 1),
        np.clip(np.abs(log_prices - np.log(np.maximum(cache['donmid20'], 1e-10))) / (scale20 * 2), 0, 1),
        np.clip(np.abs(histogram) / (np.maximum(volatility21, 1e-7) * 1.5), 0, 1),
    ]), nan=0.0)
    weighted_vote = (votes * weights).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-10)
    signals['VOTE_STRENGTH_4'] = np.where(weighted_vote > .2, 1, np.where(weighted_vote < -.2, -1, 0)).astype(np.int8)
    signals['REGIME_RULE_FIXED'] = np.where(efficiency > .4, signals['MA_20_60'], np.where(efficiency < .2, signals['RSI_MR_14'], 0)).astype(np.int8)
    signals['BUY_HOLD_LONG_FX'] = np.ones(length, np.int8)
    signals['BUY_HOLD_LONG_USD'] = -np.ones(length, np.int8)
    features = {'rsi14': cache['rsi14'], 'ma50gap': prices / sma(prices, 50) - 1,
                'ma200gap': prices / sma(prices, 200) - 1, 'bbz20': cache['bbz20'],
                'macdhist': histogram, 'eff63': efficiency, 'vol21': volatility21 * np.sqrt(252),
                'vol63': rolling_std(log_returns, 63) * np.sqrt(252), 'logr': log_returns}
    return signals, features


def fixed_execute(prices, targets, cost_bp=0.0, capital=100000.0):
    """Signal t executes at close t+1, earning its first return at t+2.

    Exposure is a daily 1x target. Direction changes incur the optional one-way
    cost after that day's price PnL. No synthetic final liquidation is added.
    """
    prices = np.asarray(prices, float)
    raw_target = np.asarray(targets)
    if prices.ndim != 1 or raw_target.ndim != 1 or len(prices) != len(raw_target) or not len(prices) or not np.all(np.isfinite(prices) & (prices > 0)):
        raise ValueError('Prices must be positive finite values aligned with targets.')
    if not np.isin(raw_target, [-1, 0, 1]).all():
        raise ValueError('Target must be -1, 0 or +1.')
    if not 0 <= cost_bp < 5000 or not np.isfinite(capital) or capital <= 0:
        raise ValueError('Unsupported one-way cost or initial capital.')
    target = raw_target.astype(np.int8)
    executed = np.r_[np.int8(0), target[:-1]].astype(np.int8)
    held = np.r_[np.int8(0), executed[:-1]].astype(np.int8)
    spot = np.r_[0.0, prices[1:] / prices[:-1] - 1.0]
    turnover = np.abs(executed.astype(int) - held.astype(int)).astype(float)
    gross = held * spot
    if np.any(1.0 + gross <= 0):
        raise ValueError('The daily account exhausted its equity; an insolvency rule is required.')
    charge = cost_bp / 10000.0 * turnover
    net = gross.copy() if cost_bp == 0.0 else (1.0 + gross) * (1.0 - charge) - 1.0
    equity = np.cumprod(1.0 + net)
    before = capital * np.r_[1.0, equity[:-1]]
    price_pnl = before * gross
    after_price = before + price_pnl
    actual_cost = after_price * charge
    return {'target': target, 'sig': target, 'executed': executed, 'held': held,
            'spot_return': spot, 'base_ret': spot, 'gross_return': gross,
            'net_return': net, 'daily_ret': net, 'turnover': turnover,
            'cost_fraction_post_pnl': charge, 'equity': equity,
            'equity_before_usd': before, 'equity_after_price_usd': after_price,
            'price_pnl_usd': price_pnl, 'transaction_cost_usd': actual_cost,
            'net_pnl_usd': price_pnl - actual_cost, 'one_way_cost_bp': float(cost_bp),
            'initial_capital_usd': float(capital)}


def week_groups(dates):
    dates = pd.DatetimeIndex(dates)
    mondays = dates - pd.to_timedelta(dates.weekday, unit='D')
    starts = np.r_[0, np.flatnonzero(mondays[1:] != mondays[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(dates) - 1]
    return list(zip(starts, ends))


def execute_mode(prices, dates, signals, mode):
    length = len(prices)
    executed = np.zeros(length, np.int8)
    decision = np.full(length, -1, np.int32)
    weeks = week_groups(dates)
    if mode in MODES[:2]:
        executed[1:] = signals[:-1]
        decision[2:] = np.arange(length - 2)
        if mode == 'Daily Weekflat':
            executed[pd.DatetimeIndex(dates).weekday == 4] = 0
    elif mode == 'Weekly Flat':
        for index in range(1, len(weeks)):
            start, end = weeks[index]
            previous_end = weeks[index - 1][1]
            executed[start:end] = signals[previous_end]
            decision[start:end + 1] = previous_end
    elif mode == 'Fixed 5D':
        for index in range(0, length - 1, 5):
            executed[index + 1:min(index + 6, length)] = signals[index]
            decision[index + 2:min(index + 7, length)] = index
    else:
        raise ValueError(f'Unrecognized mode: {mode}')
    held = np.r_[np.int8(0), executed[:-1]]
    base_return = np.r_[0.0, prices[1:] / prices[:-1] - 1]
    returns = held * base_return
    return {'sig': signals, 'executed': executed, 'held': held, 'daily_ret': returns,
            'base_ret': base_return, 'equity': np.cumprod(1 + returns),
            'turnover': np.abs(executed.astype(int) - held.astype(int)), 'decision_idx': decision}


def original_exploration(prices):
    records = []
    for pair in CURRENCIES:
        panel = prices.loc[prices.pair.eq(pair)].sort_values('date')
        values = panel.price_USD_per_foreign.to_numpy(float)
        signals, features = complete_signals(values, original=True)
        for strategy in RULES:
            for mode in MODES:
                run = execute_mode(values, panel.date, signals[strategy], mode)
                returns, held = run['daily_ret'], run['held']
                equity = np.r_[1., run['equity']]
                deviation = np.std(returns, ddof=1)
                records.append({'pair': pair, 'strategy': strategy, 'mode': mode,
                    'family': RULES[strategy][0],
                    'hypothesis': 'Adaptive rule' if strategy == 'REGIME_RULE_FIXED' else RULES[strategy][1],
                    'benchmark': strategy not in STRATEGIES,
                    'total_return_pct': 100 * (equity[-1] - 1),
                    'annualized_return_pct': 100 * (equity[-1] ** (252 / len(returns)) - 1),
                    'sharpe': np.mean(returns) / deviation * np.sqrt(252) if deviation > 1e-12 else np.nan,
                    'annualized_vol_pct': deviation * np.sqrt(252) * 100,
                    'max_drawdown_pct': (equity / np.maximum.accumulate(equity) - 1).min() * 100,
                    'average_daily_bp': np.mean(returns) * 1e4, 'active_pct': 100 * np.mean(held != 0),
                    'long_foreign_pct': 100 * np.mean(held == 1), 'long_usd_pct': 100 * np.mean(held == -1),
                    'turnover_units': float(run['turnover'].sum()), 'rebalances': int(np.sum(run['turnover'] > 0)),
                    'active_day_win_rate_pct': 100 * np.mean(returns[held != 0] > 0) if np.any(held != 0) else np.nan})
    return pd.DataFrame(records)


def trade_ledger(run, dates, pair='', strategy=''):
    """Nonzero execution episodes; a final open position is only marked to market."""
    dates = pd.DatetimeIndex(dates)
    executed = run['executed']
    changes = np.flatnonzero(executed != np.r_[0, executed[:-1]])
    records = []
    for offset, entry in enumerate(changes):
        if executed[entry] == 0:
            continue
        closed = offset + 1 < len(changes)
        exit_index = int(changes[offset + 1]) if closed else len(dates) - 1
        returns = run['gross_return'][entry + 1:exit_index + 1]
        trade_return = float(np.prod(1 + returns) - 1)
        pnl = float(run['price_pnl_usd'][entry + 1:exit_index + 1].sum())
        entry_cost = float(run['equity_after_price_usd'][entry] * run['one_way_cost_bp'] / 1e4)
        exit_cost = float(run['equity_after_price_usd'][exit_index] * run['one_way_cost_bp'] / 1e4) if closed else 0.0
        records.append({'pair': pair, 'strategy': strategy, 'trade_id': len(records) + 1,
                        'signal_idx': int(entry - 1), 'entry_idx': int(entry), 'exit_idx': exit_index,
                        'signal_date': dates[entry - 1], 'entry_date': dates[entry], 'exit_date': dates[exit_index],
                        'position': int(executed[entry]), 'holding_observations': int(exit_index - entry),
                        'closed': bool(closed), 'eligible_closed': bool(closed and exit_index > entry),
                        'trade_return': trade_return, 'return_pct': trade_return * 100,
                        'pnl_usd': pnl - entry_cost - exit_cost,
                        'price_pnl_usd': pnl, 'transaction_cost_usd': entry_cost + exit_cost})
    columns = ['pair', 'strategy', 'trade_id', 'signal_idx', 'entry_idx', 'exit_idx', 'signal_date',
               'entry_date', 'exit_date', 'position', 'holding_observations', 'closed', 'eligible_closed',
               'trade_return', 'return_pct', 'pnl_usd', 'price_pnl_usd', 'transaction_cost_usd']
    return pd.DataFrame(records, columns=columns)


def account_metrics(run, dates, trades, start_date='2005-01-01'):
    dates = pd.DatetimeIndex(dates)
    mask = dates >= pd.Timestamp(start_date)
    selected_dates = dates[mask]
    returns, held = run['daily_ret'][mask], run['held'][mask]
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    deviation = np.std(returns, ddof=1)
    first_index = int(np.flatnonzero(mask)[0])
    capital_anchor = dates[max(0, first_index - 1)]
    years = (selected_dates[-1] - capital_anchor).days / 365.25
    capital = run['initial_capital_usd']
    result = {'start_date': selected_dates[0].strftime('%Y-%m-%d'),
              'end_date': selected_dates[-1].strftime('%Y-%m-%d'), 'observations': len(returns),
              'total_return_pct': (equity[-1] - 1) * 100,
              'cagr_calendar_pct': (equity[-1] ** (1 / years) - 1) * 100 if years > 0 else np.nan,
              'sharpe_252': returns.mean() / deviation * np.sqrt(252) if deviation > 1e-12 else np.nan,
              'volatility_252_pct': deviation * np.sqrt(252) * 100,
              'max_drawdown_pct': (equity / np.maximum.accumulate(equity) - 1).min() * 100,
              'active_pct': 100 * np.mean(held != 0), 'long_foreign_pct': 100 * np.mean(held == 1),
              'short_foreign_pct': 100 * np.mean(held == -1),
              'turnover_one_way_units': float(run['turnover'][mask].sum()),
              'position_changes': int((run['turnover'][mask] > 0).sum()),
              'pnl_on_100k_usd': (equity[-1] - 1) * 100000, 'initial_capital_usd': capital,
              'account_pnl_usd': (equity[-1] - 1) * capital,
              'terminal_held_for_last_return': int(held[-1])}
    if trades.empty:
        result.update({name: np.nan for name in ('all_entry_n', 'win_n', 'loss_n', 'flat_n', 'win_rate',
            'win_rate_nonflat', 'avg_win_return', 'avg_loss_abs_return', 'payoff_ratio',
            'breakeven_win_rate_nonflat', 'expectancy_return', 'median_trade_return', 'profit_factor_return',
            'positive_return_sum', 'negative_return_abs_sum', 'eligible_return_sum', 'median_holding_observations',
            'open_n', 'open_zero_observation_n', 'open_mtm_pnl_usd', 'closed_pnl_usd', 'all_pnl_usd', 'low_sample_flag')})
        result['eligible_closed_n'] = 0
        return result
    eligible = trades.loc[trades.eligible_closed.astype(bool)]
    trade_returns = eligible.trade_return.to_numpy(float)
    wins = trade_returns[trade_returns > FLAT_TOLERANCE]
    losses = trade_returns[trade_returns < -FLAT_TOLERANCE]
    average_win = wins.mean() if len(wins) else np.nan
    average_loss = -losses.mean() if len(losses) else np.nan
    payoff = average_win / average_loss if average_loss > 0 else np.nan
    open_trades = trades.loc[~trades.closed.astype(bool)]
    result.update({'all_entry_n': len(trades), 'eligible_closed_n': len(eligible), 'win_n': len(wins),
                   'loss_n': len(losses), 'flat_n': len(eligible) - len(wins) - len(losses),
                   'win_rate': len(wins) / len(eligible) if len(eligible) else np.nan,
                   'win_rate_nonflat': len(wins) / (len(wins) + len(losses)) if len(wins) + len(losses) else np.nan,
                   'avg_win_return': average_win, 'avg_loss_abs_return': average_loss, 'payoff_ratio': payoff,
                   'breakeven_win_rate_nonflat': 1 / (1 + payoff) if np.isfinite(payoff) else np.nan,
                   'expectancy_return': trade_returns.mean() if len(eligible) else np.nan,
                   'median_trade_return': np.median(trade_returns) if len(eligible) else np.nan,
                   'profit_factor_return': wins.sum() / -losses.sum() if len(losses) else np.nan,
                   'positive_return_sum': wins.sum(), 'negative_return_abs_sum': -losses.sum(),
                   'eligible_return_sum': trade_returns.sum(),
                   'median_holding_observations': eligible.holding_observations.median(),
                   'open_n': len(open_trades), 'open_zero_observation_n': int(open_trades.holding_observations.eq(0).sum()),
                   'open_mtm_pnl_usd': open_trades.pnl_usd.sum(), 'closed_pnl_usd': eligible.pnl_usd.sum(),
                   'all_pnl_usd': trades.pnl_usd.sum(), 'low_sample_flag': len(eligible) < 20})
    return result


def reconstruct_fixed_baselines(prices):
    metrics, ledgers, runs, indicators = [], [], {}, {}
    common_dates = None
    for pair in CURRENCIES:
        panel = prices.loc[prices.pair.eq(pair)].sort_values('date')
        dates = pd.DatetimeIndex(panel.date)
        if common_dates is not None and not dates.equals(common_dates):
            raise ValueError('Fixed baselines require a shared currency calendar.')
        common_dates = dates
        values = panel.price_USD_per_foreign.to_numpy(float)
        signals, indicators[pair] = complete_signals(values)
        signals.update({'ALWAYS_LONG_FX': np.ones(len(dates), np.int8),
                        'ALWAYS_SHORT_FX': -np.ones(len(dates), np.int8), 'CASH': np.zeros(len(dates), np.int8)})
        first_execution = int(dates.searchsorted(pd.Timestamp('2005-01-01')))
        for strategy in STRATEGIES + list(FIXED_BENCHMARKS):
            targets = signals[strategy].copy()
            targets[:max(0, first_execution - 1)] = 0
            run = fixed_execute(values, targets)
            runs[(pair, strategy)] = run
            trades = trade_ledger(run, dates, pair, strategy)
            if not trades.empty:
                ledgers.append(trades)
            specification = RULES.get(strategy, ('Cash' if strategy == 'CASH' else 'Benchmark', 'Benchmark'))
            metrics.append({'pair': pair, 'strategy': strategy, 'indicator': specification[0],
                            'hypothesis': specification[1], 'technical': strategy in STRATEGIES,
                            **account_metrics(run, dates, trades)})
    price_dates = {pair: common_dates for pair in CURRENCIES}
    return pd.DataFrame(metrics), pd.concat(ledgers, ignore_index=True), runs, price_dates, indicators


def annual_table(runs, price_dates):
    dates = pd.DatetimeIndex(next(iter(price_dates.values())) if isinstance(price_dates, dict) else price_dates)
    records = []
    for (pair, strategy), run in runs.items():
        for year in sorted(set(dates[dates >= '2005-01-01'].year)):
            mask = dates.year == year
            records.append({'pair': pair, 'strategy': strategy, 'year': int(year),
                            'return_pct': float((np.prod(1 + run['daily_ret'][mask]) - 1) * 100),
                            'pnl_usd': float(run['net_pnl_usd'][mask].sum()),
                            'active_days': int(np.count_nonzero(run['held'][mask]))})
    return pd.DataFrame(records)


def period_table(runs, price_dates):
    dates = pd.DatetimeIndex(next(iter(price_dates.values())) if isinstance(price_dates, dict) else price_dates)
    periods = [('2005-2009', '2005-01-01', '2009-12-31'), ('2010-2014', '2010-01-01', '2014-12-31'),
               ('2015-2019', '2015-01-01', '2019-12-31'), ('2020-2024', '2020-01-01', '2024-12-31'),
               ('2025-May2026', '2025-01-01', '2026-05-31')]
    records = []
    for (pair, strategy), run in runs.items():
        if strategy not in CORE_RULES:
            continue
        for label, start, end in periods:
            mask = (dates >= start) & (dates <= end)
            records.append({'pair': pair, 'strategy': strategy, 'period': label,
                            'return_pct': float((np.prod(1 + run['daily_ret'][mask]) - 1) * 100),
                            'pnl_usd': float(run['net_pnl_usd'][mask].sum()),
                            'active_days': int(np.count_nonzero(run['held'][mask]))})
    return pd.DataFrame(records)
