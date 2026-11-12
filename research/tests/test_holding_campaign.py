"""The discovery boundary must exclude later outcome information before parsing."""
import csv
from pathlib import Path
import tempfile
import unittest
import run_holding_filter_research as campaign


class DiscoveryIsolationTests(unittest.TestCase):
    def rows(self):
        return [dict(pair='JPY', strategy='BB_MR_20', checkpoint_age=10,
                     condition='pnl_positive&recent5_negative', block=b, closed_n=24 if b == 'D' else 12,
                     remaining_return_sum_pct=-2, remaining_without_worst1_pct=1,
                     remaining_without_worst3_pct=3) for b in sorted(campaign.DISCOVERY_BLOCKS)]

    def parse(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observations.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            return campaign.proposals(path)

    def test_later_outcomes_are_never_parsed(self):
        baseline = self.rows()
        poison = {k: 'not-an-outcome' for k in baseline[0]}
        for block in ('V', 'C', 'IS', 'OOS'):
            self.assertEqual(self.parse(baseline), self.parse([*baseline, {**poison, 'block': block}]))

    def test_concentration_is_diagnostic_not_an_extra_gate(self):
        result = self.parse(self.rows())
        self.assertEqual(len(result), 1)
        self.assertIn(result[0]['spec'], campaign.registered_specs())
        self.assertGreater(result[0]['discovery_evidence']['D']['remaining_without_worst3_pct'], 0)

    def test_duplicate_or_incomplete_evidence_is_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            self.parse(rows + [rows[0]])
        with self.assertRaises(ValueError):
            self.parse(rows[:-1])

    def test_unsupported_discovery_does_not_create_a_candidate(self):
        rows = self.rows()
        next(row for row in rows if row['block'] == 'D1')['closed_n'] = 9
        self.assertEqual(self.parse(rows), [])


if __name__ == '__main__':
    unittest.main()
