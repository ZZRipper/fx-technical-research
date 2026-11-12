"""Synthetic guards for frozen D-only interaction observation code."""
import csv
import gzip
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_entry_interaction_observations as subject


def episode(number, entry='2011-01-05', exit='2011-01-06', values=None, outcome=1.):
    signal = (pd.Timestamp(entry) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    observed = (pd.Timestamp(signal) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    row = {'pair': 'AUD', 'strategy': 'TSMOM_21', 'family': 'Trend', 'trade_id': str(number),
           'signal_date': signal, 'entry_date': entry, 'exit_date': exit,
           'signal_idx': '100', 'entry_idx': '101', 'exit_idx': '102', 'position': '1',
           'holding_observations': '1', 'gross_simple_return_pct': str(outcome),
           'transaction_cost_usd': '0', 'closed': 'True', 'eligible_closed': 'True'}
    for observation, fields in subject.MACRO_PROVENANCE.items():
        row[observation] = observed
        for field in fields:
            row[field] = '0'
    for feature, edges in subject.STATE_BINS.items():
        value = (values or {}).get(feature, 0.)
        row[feature] = str(value)
        row[feature + '_state'] = str(int(np.searchsorted(edges, value, side='right')) if np.isfinite(value) else -1)
    return row


class ObservationGuardTests(unittest.TestCase):
    def source(self, path, rows):
        with gzip.open(path / 'is_entry_state_ledger.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.source(source, rows)
            with patch.object(subject, 'SOURCE', source), patch.object(subject, 'CURRENCIES', ['AUD']), \
                    patch.object(subject, 'STRATEGIES', ['TSMOM_21']):
                return subject.load_discovery()

    def summarize(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory); output = source / 'result'
            self.source(source, rows)
            with patch.object(subject, 'SOURCE', source), patch.object(subject, 'CURRENCIES', ['AUD']), \
                    patch.object(subject, 'STRATEGIES', ['TSMOM_21']), \
                    patch.object(subject, 'source_chain', return_value=({}, {})), \
                    patch.object(sys, 'argv', ['builder', '--output', str(output)]), redirect_stdout(io.StringIO()):
                subject.main()
            return (pd.read_csv(output / 'interaction_state_summary.csv'),
                    pd.read_csv(output / 'interaction_comparison_summary.csv'),
                    pd.read_csv(output / 'interaction_account_coverage.csv'))

    def test_late_cross_D_and_open_bad_numbers_are_skipped_before_conversion(self):
        late = episode(2, entry='2020-01-06', exit='2020-01-07')
        crossing = episode(3, entry='2014-12-30', exit='2015-01-05')
        unclosed = episode(4, entry='2012-01-05', exit='2012-01-06')
        unclosed['eligible_closed'] = unclosed['closed'] = 'False'
        for row in (late, crossing, unclosed):
            row['gross_simple_return_pct'] = 'MUST_NOT_CONVERT_TO_FLOAT'
            row['efficiency_63_state'] = 'MUST_NOT_CONVERT_TO_INT'
            row['K_rate_obs_date'] = 'MUST_NOT_PARSE_AS_DATE'
        data, counts = self.load([episode(1), late, crossing, unclosed])
        self.assertEqual(data.trade_id.tolist(), ['1'])
        self.assertEqual(counts['late_entry_rows_skipped_before_return_parse'], 1)
        self.assertEqual(counts['nonclosed_or_cross_D_rows_skipped_before_return_parse'], 2)
        self.assertEqual(counts['decoded_D_closed_rows'], 1)

    def test_trade_crossing_D1_D2_is_counted_only_in_combined_D(self):
        _, _, coverage = self.summarize([episode(1, entry='2009-12-30', exit='2010-01-05', outcome=-2.5)])
        counts = coverage.set_index('block').closed_n.to_dict()
        self.assertEqual(counts, {'D1': 0, 'D2': 0, 'D': 1})
        self.assertEqual(coverage.set_index('block').loc['D', 'return_sum_pct'], -2.5)

    def test_UNKNOWN_partition_preserves_A_only_B_only_and_single_state_controls(self):
        values = [(0.05, .9), (.05, 1.1), (.15, .9), (.15, 1.1), (np.nan, .9), (.05, np.nan)]
        outcomes = [1., -2., 3., -4., 5., -6.]
        rows = [episode(i + 1, values={'efficiency_63': a, 'vol_ratio_20_63': b}, outcome=y)
                for i, ((a, b), y) in enumerate(zip(values, outcomes))]
        summary, controls, _ = self.summarize(rows)
        def cell(frame):
            return frame.loc[frame.interaction_id.eq('efficiency_volatility') & frame.state_a.eq(0)
                             & frame.state_b.eq(1) & frame.block.eq('D')].set_index('group')
        partition, comparison = cell(summary), cell(controls)
        self.assertEqual(partition.closed_n.to_dict(), {'joint': 1, 'A_only': 1, 'B_only': 1, 'neither': 1, 'UNKNOWN': 2})
        self.assertEqual(partition.return_sum_pct.to_dict(), {'joint': 1., 'A_only': -2., 'B_only': 3., 'neither': -4., 'UNKNOWN': -1.})
        self.assertEqual(int(partition.closed_n.sum()), 6)
        self.assertEqual(float(partition.return_sum_pct.sum()), sum(outcomes))
        self.assertEqual(comparison.closed_n.to_dict(), {'A_all': 3, 'B_all': 3, 'retained': 5})
        self.assertEqual(comparison.return_sum_pct.to_dict(), {'A_all': -7., 'B_all': 9., 'retained': -4.})

    def test_fixed_bin_edges_are_left_inclusive_with_UNKNOWN_separate(self):
        points = [(-1.0001, 0), (-1., 1), (-.0001, 1), (0., 2), (.9999, 2), (1., 3), (np.nan, -1)]
        rows = []
        for number, (value, expected) in enumerate(points, 1):
            row = episode(number, values={'direction_return_20_vol': value})
            row['direction_return_20_vol_state'] = str(expected)
            rows.append(row)
        loaded, _ = self.load(rows)
        self.assertEqual(loaded.direction_return_20_vol_state.tolist(), [0, 1, 1, 2, 2, 3, -1])
        wrong = episode(100, values={'direction_return_20_vol': 0.})
        wrong['direction_return_20_vol_state'] = '1'
        with self.assertRaisesRegex(AssertionError, 'Fixed state does not match'):
            self.load([wrong])


if __name__ == '__main__':
    unittest.main()
