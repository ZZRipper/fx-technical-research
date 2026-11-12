"""Rebuild IS-only entry-state evidence for all 180 unfiltered accounts."""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as prior
from fx_baselines import CURRENCIES, STRATEGIES, RULES, fixed_execute
from fx_filter_evaluation import simple_ledger, quality
from fx_is_states import STATE_BINS, entry_features, state_codes

ROOT = Path(__file__).resolve().parent
BLOCKS = (
    ('DISCOVERY_1', '2005-01-03', '2009-12-31'),
    ('DISCOVERY_2', '2010-01-01', '2014-12-31'),
    ('DEVELOPMENT', '2005-01-03', '2014-12-31'),
    ('VALIDATION', '2015-01-01', '2019-12-31'),
    ('CONFIRMATION', '2020-01-01', '2023-05-29'),
    ('IS', '2005-01-03', '2023-05-29'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced/filter_is_diagnostics_v2')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output/'entry_state_protocol.json').exists():
        raise ValueError('Existing observation campaign must not be silently overwritten.')
    protocol = {'created_utc': prior.now(), 'compute_cutoff': prior.IS_END, 'cost_bp': 0,
                'state_bins': STATE_BINS, 'blocks': BLOCKS, 'accounts': 180,
                'outcome_cohort': 'Entry and eligible true close must both lie inside a block. Open and crossing trades are excluded from block quality, retained in full ledger.',
                'signal_clock': 'Signal-date price and already-earned unfiltered BASE PnL; macro strictly earlier and at most seven days old.',
                'bin_closure': 'Left inclusive, right exclusive; UNKNOWN=-1 retained separately.',
                'purpose': 'Descriptive IS-only state evidence, not a fitted filter or independent untouched validation.',
                'oos_computed': False,
                'source_sha256': {**prior.source_hashes(), 'build_is_filter_observations.py':prior.sha(__file__), 'src/fx_is_states.py':prior.sha(ROOT/'src/fx_is_states.py')},
                'input_sha256': prior.input_hashes()}
    prior.save_json(output/'entry_state_protocol.json', protocol)
    prepared = prior.prepare(prior.IS_END)
    ledgers=[]; summaries=[]; coverage=[]
    for pair in CURRENCIES:
        print('IS entry states:',pair,flush=True)
        prices, dates, features, targets = prepared[pair]
        for strategy in STRATEGIES:
            run=fixed_execute(prices,targets[strategy]); ledger=simple_ledger(run,dates)
            ledger['pair']=pair; ledger['strategy']=strategy; ledger['family']=RULES[strategy][1]
            entry=entry_features(features,run,ledger,pair)
            for column in entry:
                if column not in ('pair','date'):
                    ledger[column]=entry[column].to_numpy()
            for column,edges in STATE_BINS.items():
                ledger[column+'_state']=state_codes(ledger[column],edges)
            ledgers.append(ledger)
            for block,start,end in BLOCKS:
                entered=ledger.loc[ledger.entry_date.between(start,end)]
                closed=entered.loc[entered.eligible_closed & entered.exit_date.le(pd.Timestamp(end))]
                coverage.append({'pair':pair,'strategy':strategy,'family':RULES[strategy][1], 'block':block,
                                 'entered_n':len(entered),'closed_n':len(closed), 'censored_n':len(entered)-len(closed),
                                 **quality(closed.gross_simple_return_pct)})
                for feature,edges in STATE_BINS.items():
                    for code in (-1,0,1,2,3):
                        sub=closed.loc[closed[feature+'_state'].eq(code)]
                        values=sub.gross_simple_return_pct.to_numpy()
                        total=float(values.sum())
                        summaries.append({'pair':pair,'strategy':strategy,'family':RULES[strategy][1],
                            'block':block,'feature':feature,'state':code,'cohort_n':len(closed),
                            **quality(values),'return_sum_pct':total,
                            'largest_win_pct':float(values.max()) if len(values) else np.nan,
                            'largest_loss_pct':float(values.min()) if len(values) else np.nan,
                            'return_without_best_trade_pct':total-max(0.,float(values.max())) if len(values) else 0.,
                            'return_without_worst_trade_pct':total-min(0.,float(values.min())) if len(values) else 0.})
    ledger=pd.concat(ledgers,ignore_index=True)
    assert ledger.entry_date.max()<=pd.Timestamp(prior.IS_END)
    assert ledger.exit_date.max()<=pd.Timestamp(prior.IS_END)
    assert set(zip(ledger.pair,ledger.strategy))==set((p,s) for p in CURRENCIES for s in STRATEGIES)
    ledger.to_csv(output/'is_entry_state_ledger.csv.gz',index=False)
    pd.DataFrame(summaries).to_csv(output/'is_entry_state_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'is_entry_cohort_coverage.csv',index=False)
    prior.assert_hashes(protocol['source_sha256']); prior.assert_hashes(protocol['input_sha256'])
    prior.save_json(output/'entry_state_validation.json', {'status':'passed','finished_utc':prior.now(),
        'compute_cutoff':prior.IS_END,'oos_computed':False,'accounts':180,'trade_rows':len(ledger),
        'summary_rows':len(summaries),'source_integrity_pass':True,
        'output_sha256':{path.name:prior.sha(path) for path in output.glob('is_entry*')}})
    print('Complete:',len(ledger),'IS episodes;',len(summaries),'state observations.',flush=True)

if __name__=='__main__':
    main()
