"""Direction nomination uses only fixed discovery outcomes, not later results."""
import csv
from pathlib import Path
import tempfile
import unittest
import run_direction_filter_research as campaign


class DirectionDiscoveryTests(unittest.TestCase):
    def rows(self):
        return [dict(pair='AUD', strategy='TSMOM_63', position=1, block=block,
                     closed_n=24 if block == 'D' else 12, return_sum_pct=-2,
                     return_without_worst_trade_pct=3)
                for block in sorted(campaign.DISCOVERY_BLOCKS)]

    def parse(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'direction.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            return campaign.proposals(path)

    def test_future_numeric_outcomes_are_not_parsed(self):
        rows = self.rows()
        bad = {k: 'not-an-outcome' for k in rows[0]}
        for block in ('V', 'C', 'IS', 'OOS'):
            self.assertEqual(self.parse(rows), self.parse([*rows, {**bad, 'block': block}]))

    def test_direction_and_concentration_diagnostic(self):
        for position, identifier in [(-1, 'REJECT_SHORT'), (1, 'REJECT_LONG')]:
            rows = [{**r, 'position': position} for r in self.rows()]
            proposed = self.parse(rows)
            self.assertEqual(len(proposed), 1)
            self.assertEqual(proposed[0]['spec'], {'id': identifier, 'reject_position': position})
            self.assertIn(proposed[0]['spec'], campaign.registered_specs())

    def test_cross_half_trade_can_reverse_combined_evidence(self):
        rows = self.rows()
        combined = next(r for r in rows if r['block'] == 'D')
        combined.update(closed_n=25, return_sum_pct=4)
        self.assertEqual(self.parse(rows), [])
        combined['return_sum_pct'] = -1
        self.assertEqual(len(self.parse(rows)), 1)
        next(r for r in rows if r['block'] == 'D1')['closed_n'] = 9
        self.assertEqual(self.parse(rows), [])

    def test_invalid_and_ambiguous_evidence_rejected(self):
        rows = self.rows()
        for invalid in (rows + [rows[0]], rows[:-1],
                        [{**r, 'position': 0} for r in rows],
                        [{**r, 'closed_n': 12.5} for r in rows],
                        [{**r, 'return_sum_pct': float('nan')} for r in rows]):
            with self.subTest(rows=invalid):
                with self.assertRaises(ValueError):
                    self.parse(invalid)

    def test_empty_direction_group_is_not_a_candidate(self):
        rows = self.rows()
        empty = next(r for r in rows if r['block'] == 'D2')
        empty.update(closed_n=0, return_sum_pct=0, return_without_worst_trade_pct='')
        self.assertEqual(self.parse(rows), [])
        empty['return_sum_pct'] = 1
        with self.assertRaises(ValueError):
            self.parse(rows)


if __name__ == '__main__':
    unittest.main()
