"""Preserve the already fixed hypothesis identities and original D evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import run_state_original_screen as campaign


class OriginalHypothesisTests(unittest.TestCase):
    def item(self, state=1):
        spec = {'id': f'STATE_efficiency_63_{state}', 'feature': 'efficiency_63', 'state': state}
        return {**campaign.identity('JPY', 'BB_MR_20', spec), 'spec': spec,
                'discovery_evidence': {b: {'closed_n': 24 if b == 'DEVELOPMENT' else 12,
                    'return_sum_pct': -2., 'return_without_worst_trade_pct': -1.}
                    for b in ('DISCOVERY_1', 'DISCOVERY_2', 'DEVELOPMENT')}}

    def parse(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.json'
            path.write_text(json.dumps({'candidates': rows}), encoding='utf-8')
            return campaign.proposals(path)

    def test_list_order_specs_and_D_evidence_are_preserved(self):
        rows = [self.item(2), self.item(1)]
        before = deepcopy(rows)
        self.assertEqual(self.parse(rows), rows)
        self.assertEqual(rows, before)
        self.assertEqual(len(campaign.registered_specs()), 52)

    def test_duplicate_identity_threshold_override_or_bool_rejected(self):
        for operation in ('duplicate', 'threshold', 'boolean', 'identity'):
            rows = [self.item()]
            if operation == 'duplicate': rows.append(deepcopy(rows[0]))
            elif operation == 'threshold': rows[0]['spec']['threshold'] = 0.15
            elif operation == 'boolean': rows[0]['spec']['state'] = True
            else: rows[0]['candidate_id'] = 'JPY|BB_MR_10|STATE_efficiency_63_1'
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_original_worst_trade_condition_is_not_replaced_by_negative_sum(self):
        row = self.item()
        row['discovery_evidence']['DEVELOPMENT']['return_without_worst_trade_pct'] = 1
        with self.assertRaises(ValueError):
            self.parse([row])
        row = self.item()
        row['discovery_evidence']['OOS'] = {'return_sum_pct': 999}
        with self.assertRaises(ValueError):
            self.parse([row])


if __name__ == '__main__':
    unittest.main()
