"""Discovery isolation and fixed two-state nomination contract."""
import csv
from pathlib import Path
import tempfile
import unittest
import run_entry_interaction_research as campaign


class InteractionDiscoveryTests(unittest.TestCase):
    def rows(self):
        return [dict(pair='JPY', strategy='BB_MR_20',
                     interaction_id='efficiency_directional_stretch',
                     feature_a='efficiency_63', state_a=3,
                     feature_b='direction_return_20_vol', state_b=0,
                     block=b, group='joint', closed_n=24 if b == 'D' else 12,
                     return_sum_pct=-2, return_without_worst1_pct=1,
                     return_without_worst3_pct=3)
                for b in sorted(campaign.DISCOVERY_BLOCKS)]

    def parse(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observations.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            return campaign.proposals(path)

    def test_later_outcomes_and_other_cohorts_are_not_parsed(self):
        rows = self.rows()
        poison = {k: 'not-an-outcome' for k in rows[0]}
        for block in ('V', 'C', 'IS', 'OOS'):
            self.assertEqual(self.parse(rows), self.parse([*rows, {**poison, 'block': block}]))
        for group in ('A_only', 'B_only', 'neither', 'UNKNOWN', 'retained'):
            self.assertEqual(self.parse(rows), self.parse([*rows, {**poison, 'block': 'D', 'group': group}]))

    def test_canonical_spec_and_concentration_is_not_a_gate(self):
        result = self.parse(self.rows())
        self.assertEqual(len(result), 1)
        spec = result[0]['spec']
        self.assertIn(spec, campaign.registered_specs())
        self.assertEqual([c['feature'] for c in spec['conditions']],
                         ['direction_return_20_vol', 'efficiency_63'])
        self.assertEqual(len(campaign.registered_specs()), 48)
        self.assertGreater(result[0]['discovery_evidence']['D']['return_without_worst3_pct'], 0)

    def test_invalid_or_duplicate_evidence_rejected(self):
        rows = self.rows()
        for modified in (rows + [rows[0]], rows[:-1],
                         [{**r, 'state_a': -1} for r in rows],
                         [{**r, 'feature_a': 'K_vix'} for r in rows],
                         [{**r, 'closed_n': 12.5} for r in rows]):
            with self.subTest(rows=modified):
                with self.assertRaises(ValueError):
                    self.parse(modified)

    def test_support_and_combined_period_counterexample(self):
        rows = self.rows()
        next(r for r in rows if r['block'] == 'D1')['closed_n'] = 9
        self.assertEqual(self.parse(rows), [])
        rows = self.rows()
        next(r for r in rows if r['block'] == 'D')['return_sum_pct'] = 2
        self.assertEqual(self.parse(rows), [])
        # A crossing closed trade can be included in D but neither half.
        rows = self.rows()
        next(r for r in rows if r['block'] == 'D')['closed_n'] = 25
        self.assertEqual(len(self.parse(rows)), 1)


if __name__ == '__main__':
    unittest.main()
