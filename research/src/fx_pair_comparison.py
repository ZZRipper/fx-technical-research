"""Retrospective pair/category evidence from unchanged, continuous BASE accounts.

No fitting, parameter selection, portfolio construction or hypothesis tests.
All grouping and comparison windows are fixed in this file, not fitted to PnL.
"""
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from statistics import mean, median, stdev
import math

EPOCH = date(1970, 1, 1)
FAMILIES = {
    'RSI MR': ('MR', ['RSI_MR_7', 'RSI_MR_14', 'RSI_MR_20']),
    'Bollinger MR': ('MR', ['BB_MR_10', 'BB_MR_20']),
    'RSI Momentum': ('Trend', ['RSI_MOM_7', 'RSI_MOM_14', 'RSI_MOM_20']),
    'Moving Average': ('Trend', ['MA_10_30', 'MA_20_60']),
    'Bollinger Breakout': ('Trend', ['BB_BO_10', 'BB_BO_20']),
    'Donchian Breakout': ('Trend', ['DONCHIAN_BO_10', 'DONCHIAN_BO_20']),
    'MACD': ('Trend', ['MACD_12_26_9']),
    'Time-series Momentum': ('Trend', ['TSMOM_21', 'TSMOM_63']),
}

def day(value):
    return (date.fromisoformat(value) - EPOCH).days


def held_from_ledger(account, dates):
    held = [0] * len(dates)
    for t in account['trades']:
        left = bisect_right(dates, t[0])
        right = bisect_right(dates, t[1]) if t[4] else len(dates)
        for i in range(left, right):
            if held[i]:
                raise ValueError('Overlapping position intervals')
            held[i] = t[5]
    return held


def metrics(account, dates, held, start, end):
    lo, hi = bisect_left(dates, start), bisect_right(dates, end)
    values = account['daily_returns'][lo:hi]
    if len(values) < 2:
        raise ValueError('Insufficient common-window observations')
    total = math.fsum(values)
    cumulative = peak = drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    sd = stdev(values)
    # Calendar-window PnL includes existing positions. Closed count uses exit dates.
    closed = sum(bool(t[4] and start <= t[1] <= end and t[3] > 0) for t in account['trades'])
    return {'ret': total * 100, 'sharpe': mean(values) / sd * math.sqrt(252) if sd > 1e-15 else None,
            'dd': drawdown * 100, 'vol': sd * math.sqrt(252) * 100,
            'closed': closed, 'exposure': sum(x != 0 for x in held[lo:hi]) / len(values) * 100,
            'n': len(values)}


def category_summary(rows, category, excluded=()):
    grouped = []
    for family, (cat, rules) in FAMILIES.items():
        if cat != category or family in excluded:
            continue
        items = [rows[r] for r in rules]
        s = [x['sharpe'] for x in items if x['sharpe'] is not None]
        grouped.append({'family': family, 'sharpe': median(s) if s else None,
                        'ret': median(x['ret'] for x in items)})
    values = [x['sharpe'] for x in grouped if x['sharpe'] is not None]
    return {'sharpe': median(values) if values else None,
            'positive': sum(x['ret'] > 0 for x in grouped), 'families': len(grouped),
            'groups': grouped}


def build_pair_comparison(payload):
    start, end = payload['datasets']['main']['start'], payload['datasets']['main']['end']
    windows = {'recent': ['2020-01-01', end], 'history': [start, '2019-12-31'],
               'early': [start, '2009-12-31'], 'middle': ['2010-01-01', '2014-12-31'],
               'late': ['2015-01-01', '2019-12-31']}
    last = date.fromisoformat(end)
    years = list(range(2005, last.year + 1))
    for year in years:
        windows[str(year)] = [max(start, f'{year}-01-01'), min(end, f'{year}-12-31')]
    complete_recent = [str(y) for y in years if y >= 2020 and (y < last.year or end.endswith('12-31'))]
    index = {(a['pair'], a['rule']): a for a in payload['accounts'] if a['ds'] == 'main' and a['variant'] == 'BASE'}
    rules = [r for _, rs in FAMILIES.values() for r in rs]
    pairs = payload['pairs']
    output = {'start': start, 'end': end, 'windows': windows, 'years': years,
              'completeRecent': complete_recent, 'families': FAMILIES, 'pairs': pairs,
              'ruleOrder': rules, 'results': {}, 'summaries': {}, 'sensitivity': {}}
    checked = 0
    max_daily_error = max_sum_error = 0.0
    common_dates = None
    for pair in pairs:
        output['results'][pair] = {}
        for rule in rules + ['ALWAYS_LONG_FX', 'ALWAYS_SHORT_FX']:
            account = index[pair, rule]
            dates = payload['calendars'][int(account['cal'])]
            if common_dates is None:
                common_dates = dates
            if dates != common_dates or len(dates) != len(account['daily_returns']):
                raise ValueError('Calendars do not match')
            held = held_from_ledger(account, dates)
            underlying = index[pair, 'ALWAYS_LONG_FX']['daily_returns']
            err = max(abs(r - h * u) for r, h, u in zip(account['daily_returns'], held, underlying))
            max_daily_error = max(max_daily_error, err)
            if err > 1e-12:
                raise ValueError(f'Ledger reconciliation failed: {pair}/{rule}: {err}')
            results = {k: metrics(account, dates, held, day(v[0]), day(v[1])) for k, v in windows.items()}
            for key, (begin, finish) in windows.items():
                lo, hi = bisect_left(dates, day(begin)), bisect_right(dates, day(finish))
                existing = (account['equity'][hi-1] - (account['equity'][lo-1] if lo else 100000)) / 1000
                max_sum_error = max(max_sum_error, abs(existing - results[key]['ret']))
            output['results'][pair][rule] = results
            checked += 1
        output['summaries'][pair] = {}
        for key in windows:
            rows = {r: output['results'][pair][r][key] for r in rules}
            output['summaries'][pair][key] = {c: category_summary(rows, c) for c in ['MR', 'Trend']}
        rows = {r: output['results'][pair][r]['recent'] for r in rules}
        tests = [('All families', ())] + [(f'Without {f}', (f,)) for f in FAMILIES]
        tests += [('Without both Bollinger families', ('Bollinger MR', 'Bollinger Breakout'))]
        output['sensitivity'][pair] = [
            {'label': label, **{c: category_summary(rows, c, omitted)['sharpe'] for c in ['MR', 'Trend']}}
            for label, omitted in tests]
    if max_sum_error > 1e-8:
        raise ValueError('Existing equity / period return mismatch')
    output['validation'] = {'accounts': checked, 'rules': len(rules), 'pairs': len(pairs),
                            'daily_return_max_error': max_daily_error, 'period_return_max_error_pp': max_sum_error,
                            'same_calendar': True, 'observations': len(common_dates)}
    return output
