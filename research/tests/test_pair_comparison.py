"""Accounting boundaries and family weighting for the comparison module."""
import unittest
from src.fx_pair_comparison import held_from_ledger, metrics, category_summary

class PairComparisonTests(unittest.TestCase):
    def test_entry_and_reversal_timing(self):
        a = {'trades': [[1, 3, 0, 2, 1, 1], [3, -1, 0, 2, 0, -1]]}
        self.assertEqual(held_from_ledger(a, [1, 2, 3, 4, 5]), [0, 1, 1, -1, -1])

    def test_window_carry_in_and_drawdown(self):
        a = {'daily_returns': [0, .1, -.2, .05, 0], 'trades': [[1, 4, -5, 3, 1, 1]]}
        x = metrics(a, [1, 2, 3, 4, 5], [0, 1, 1, 1, 0], 3, 5)
        self.assertAlmostEqual(x['ret'], -15)
        self.assertAlmostEqual(x['dd'], -20)
        self.assertEqual(x['closed'], 1)
        self.assertAlmostEqual(x['exposure'], 200 / 3)

    def test_zero_returns_and_exposure(self):
        a = {'daily_returns': [0, 0, 0], 'trades': []}
        x = metrics(a, [1, 2, 3], [1, 1, 0], 1, 3)
        self.assertIsNone(x['sharpe'])
        self.assertAlmostEqual(x['exposure'], 200 / 3)
        self.assertEqual(x['dd'], 0)

    def test_family_weighting(self):
        rows = {r: {'ret': 1, 'sharpe': 1} for r in ['RSI_MR_7', 'RSI_MR_14', 'RSI_MR_20']}
        rows.update({r: {'ret': -1, 'sharpe': -1} for r in ['BB_MR_10', 'BB_MR_20']})
        x = category_summary(rows, 'MR')
        self.assertEqual(x['sharpe'], 0)
        self.assertEqual(x['positive'], 1)
        self.assertEqual(category_summary(rows, 'MR', ('Bollinger MR',))['sharpe'], 1)

class SavedComparisonReconciliationTests(unittest.TestCase):
    def test_saved_output_against_independent_numpy_calculations(self):
        import gzip, json, re
        from pathlib import Path
        import numpy as np
        from src.fx_pair_comparison import day
        root = Path(__file__).resolve().parents[1]
        html = (root / 'FX_PnL_Explorer_Simple.html').read_text()
        output = json.loads(re.search(r'<script id="pair-comparison-data" type="application/json">(.*?)</script>', html).group(1))
        with gzip.open(root / 'reproduced/simple_interest/explorer_payload.json.gz') as f:
            payload = json.load(f)
        index = {(a['pair'], a['rule']): a for a in payload['accounts'] if a['ds'] == 'main' and a['variant'] == 'BASE'}
        for pair, rules in output['results'].items():
            for rule, windows in rules.items():
                account = index[pair, rule]
                dates = np.array(payload['calendars'][int(account['cal'])])
                returns = np.array(account['daily_returns'])
                for key, result in windows.items():
                    lo, hi = map(day, output['windows'][key])
                    values = returns[(dates >= lo) & (dates <= hi)]
                    curve = np.r_[0, np.cumsum(values)]
                    self.assertAlmostEqual(float(np.min(curve - np.maximum.accumulate(curve))) * 100, result['dd'], places=9)
                    self.assertAlmostEqual(float(values.sum()) * 100, result['ret'], places=9)
                    if values.std(ddof=1) > 1e-15:
                        self.assertAlmostEqual(float(values.mean() / values.std(ddof=1)) * 252 ** .5, result['sharpe'], places=9)
                self.assertAlmostEqual(sum(windows[str(y)]['ret'] for y in range(2020, 2027)), windows['recent']['ret'], places=9)
                self.assertAlmostEqual(sum(windows[str(y)]['ret'] for y in range(2005, 2020)), windows['history']['ret'], places=9)
        self.assertEqual(output['completeRecent'], [str(y) for y in range(2020, 2026)])

if __name__ == '__main__':
    unittest.main()
