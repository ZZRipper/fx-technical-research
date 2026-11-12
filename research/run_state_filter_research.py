"""A second, IS-only-developed campaign: discovery, validation, locked confirmation.

The three-year OOS is reused historical evidence, never a new untouched sample.
All v1 artifacts and decisions remain immutable. Costs remain zero.
"""
from pathlib import Path
import argparse
import csv
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_filter_research as core
from fx_baselines import CURRENCIES, STRATEGIES, fixed_execute
from fx_filter_evaluation import SCREEN, OOS_SCREEN, IS_BLOCKS, simple_ledger, screen_candidates, fresh_start
from fx_is_states import STATE_BINS
from fx_state_filters import run_state_filter

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'reproduced/state_filter_v2'
DISCOVERY=('DISCOVERY','2005-01-03','2014-12-31')
VALIDATION=('VALIDATION','2015-01-01','2019-12-31')
CONFIRMATION=('CONFIRMATION','2020-01-01','2023-05-29')
RECENT=('RECENT','2015-01-01','2023-05-29')
SOURCES=['run_state_filter_research.py','src/fx_is_states.py','src/fx_state_filters.py']
TEMPORAL={
 'proposal':'One fixed feature bin per candidate; UNKNOWN is never rejected. Each 2005-09/2010-14 discovery cohort has >=10 closed trades and negative sum; combined 2005-14 sum remains negative after removing its worst trade.',
 'discovery':'FILTER annualized>=1%; improvement>=5pp; win rate +2pp; payoff +10%; DD deterioration<=2pp; return/DD improves; >=30 closed and >=50% retention; >=20 changed closed episodes.',
 'validation':'2015-19 FILTER annualized>=1%, positive improvement and return/DD improvement; win rate/payoff no worse; >=20 closed, >=5 wins/losses, >=50% retention, >=5 changed closed episodes.',
 'locking':'One candidate per pair/strategy, ranking VALIDATION annualized DESC, through-2019 return improvement DESC, candidate_id ASC. Lock before computing CONFIRMATION filtered paths; no fallback after failure.',
 'confirmation':'2020-2023-05 FILTER annualized>=1%, positive improvement and return/DD improvement; quality no worse; >=20 closed with >=5 wins/losses, >=50% retention, >=5 changed closed episodes.',
 'final':'All original full-IS SCREEN criteria, plus RECENT annualized>=1 and positive improvement/return-DD improvement. Full IS and RECENT delta remain positive after removing the three most helpful original episodes. >=20 full-IS changed closed episodes.',
 'oos':'Only final IS-passing candidates enter the same prewritten OOS_SCREEN, with all outcomes retained. Reused historical OOS is disclosed. No OOS-based rule change or fallback.',
}


def key(pair,strategy,spec):
 return {'candidate_id':f"{pair}|{strategy}|{spec['id']}",'pair':pair,'strategy':strategy,
         'filter_id':spec['id'],'filter_name':f"Reject {spec['feature']} state {spec['state']}",'is_new':True}


def integrity(protocol):
 core.assert_hashes(protocol['source_sha256']); core.assert_hashes(protocol['input_sha256'])
 core.assert_hashes(protocol.get('observation_source_sha256',{}))


def proposal_rows(path):
 """Only discovery block outcomes are parsed, even though the CSV also has later blocks."""
 allowed={'DISCOVERY_1','DISCOVERY_2','DEVELOPMENT'}; data={}
 with path.open(encoding='utf-8',newline='') as stream:
  for row in csv.DictReader(stream):
   if row['block'] not in allowed or int(row['state'])<0: continue
   identity=(row['pair'],row['strategy'],row['feature'],int(row['state']))
   data.setdefault(identity,{})[row['block']]={k:float(row[k]) for k in ['closed_n','return_sum_pct','return_without_worst_trade_pct']}
 selected=[]
 for (pair,strategy,feature,state),blocks in sorted(data.items()):
  if set(blocks)!=allowed: raise AssertionError('Incomplete discovery states.')
  if all(blocks[b]['closed_n']>=10 and blocks[b]['return_sum_pct']<0 for b in ('DISCOVERY_1','DISCOVERY_2')) and blocks['DEVELOPMENT']['return_without_worst_trade_pct']<0:
   spec={'id':f'STATE_{feature}_{state}','feature':feature,'state':state}
   selected.append({**key(pair,strategy,spec),'spec':spec,'discovery_evidence':blocks})
 return selected


def affected(base,filtered,dates,start,end,ledger=None):
 """Actual-day delta per original episode, including carry-in and open tails."""
 if ledger is None: ledger=simple_ledger(base,dates)
 dates=pd.DatetimeIndex(dates); mask=(dates>=start)&(dates<=end)
 delta=(np.asarray(filtered['gross_return'])-np.asarray(base['gross_return']))*100
 delta=np.where(mask,delta,0.)
 changed=mask & (np.asarray(filtered['held'])!=np.asarray(base['held']))
 prefix=np.r_[0.,np.cumsum(delta)]; cp=np.r_[0,np.cumsum(changed)]
 starts=ledger.entry_idx.to_numpy(int)+1; stops=ledger.exit_idx.to_numpy(int)+1
 contributions=prefix[stops]-prefix[starts]; used=(cp[stops]-cp[starts])>0
 closed=ledger.eligible_closed & ledger.entry_date.ge(start)&ledger.exit_date.le(end)
 total=float(delta.sum())
 if not np.isclose(contributions.sum(),total,atol=1e-8,rtol=0): raise AssertionError('Episode delta does not reconcile.')
 top=np.sort(contributions[contributions>0])[-3:]
 return {'delta_pp':total,'changed_closed_n':int((used & closed).sum()),
         'changed_episode_n':int(used.sum()),'delta_without_best3_pp':float(total-top.sum())}


def ratio_improves(b,f):
 return bool(f.gross_simple_return_pct*(-b.gross_max_drawdown_pp)>b.gross_simple_return_pct*(-f.gross_max_drawdown_pp)+1e-10)


def phase_flags(metrics,phase,evidence,discovery=False):
 rows=metrics.loc[metrics.split.eq(phase)].set_index('variant'); b,f=rows.loc['BASE'],rows.loc['FILTER']
 delta=f.gross_simple_return_pct-b.gross_simple_return_pct
 finite=np.isfinite([b.win_rate,f.win_rate,b.payoff_ratio,f.payoff_ratio]).all()
 flags={
  'annualized':bool(f.gross_annualized_simple_pct>=1.-1e-10),
  'return_improvement':bool(delta>=5.-1e-10 if discovery else delta>1e-10),
  'return_drawdown':ratio_improves(b,f),
  'quality':bool(finite and f.win_rate>=b.win_rate+(.02 if discovery else 0)-1e-10 and f.payoff_ratio>=b.payoff_ratio*(1.1 if discovery else 1)-1e-10),
  'drawdown':bool(f.gross_max_drawdown_pp>=b.gross_max_drawdown_pp-2.-1e-10),
  'sample':bool(f.closed_n>=(30 if discovery else 20) and f.closed_n>=b.closed_n*.5-1e-10),
  'outcome_sample':True if discovery else bool(f.win_n>=5 and f.loss_n>=5),
  'affected_sample':bool(evidence['changed_closed_n']>=(20 if discovery else 5)),
 }
 return flags


def fit(output):
 if (output/'validation_locked.json').exists(): raise ValueError('Already locked; do not replace candidates.')
 source=ROOT/'reproduced/filter_is_diagnostics_v2/is_entry_state_summary.csv'
 if (output/'protocol.json').exists(): raise ValueError('Protocol already exists; preserve this campaign record.')
 cache_validation=core.read_json(source.parent/'entry_state_validation.json')
 amendment=core.read_json(source.parent/'technical_validation_amendment.json')
 if cache_validation['status']!='passed' or amendment['status']!='passed': raise ValueError('IS cache audit failed.')
 for name,digest in cache_validation['output_sha256'].items():
  if core.sha(source.parent/name)!=digest: raise ValueError('IS cache changed: '+name)
 if core.sha(ROOT/'src/fx_is_states.py')!=amendment['amended_source_sha256']: raise ValueError('State guards differ from audited amendment.')
 protocol=core.protocol_document()
 protocol.update(version='state_filter_v2',created_utc=core.now(),candidate_specs=[],state_bins=STATE_BINS,
   temporal_protocol=TEMPORAL,discovery=DISCOVERY,validation=VALIDATION,confirmation=CONFIRMATION,
   prior_holdout_exposure='The three-year historical OOS was observed before v1 and evaluated by v1. V2 is a disclosed repeated historical evaluation, not independent untouched OOS.',
   source_sha256={**core.source_hashes(),**{n:core.sha(ROOT/n) for n in SOURCES}},
   input_sha256={**core.input_hashes(),source.relative_to(ROOT).as_posix():core.sha(source)},
   observation_source_sha256={source.relative_to(ROOT).as_posix():core.sha(source)})
 core.save_json(output/'protocol.json',protocol)
 proposals=proposal_rows(source); core.save_json(output/'discovery_proposals.json',{'created_utc':core.now(),'candidates':proposals})
 print('Discovery-stable proposals:',len(proposals),flush=True)
 prepared=core.prepare(VALIDATION[2]); rows=[]; selection=[]; cache={}
 for offset,item in enumerate(proposals):
  pair,strategy,spec=item['pair'],item['strategy'],item['spec']; identity=key(pair,strategy,spec)
  values,dates,features,targets=prepared[pair]
  if (pair,strategy) not in cache:
   base=fixed_execute(values,targets[strategy]); cache[pair,strategy]=(base,simple_ledger(base,dates))
  base,ledger=cache[pair,strategy]
  filtered,_=run_state_filter(values,dates,base,features,spec,pair,strategy)
  periods=[DISCOVERY,VALIDATION,('FIT','2005-01-03','2019-12-31')]
  m=pd.DataFrame(core.metric_rows(identity,base,dates,periods,'BASE')+core.metric_rows(identity,filtered,dates,periods,'FILTER'))
  rows.extend(m.to_dict('records'))
  d=affected(base,filtered,dates,*DISCOVERY[1:],ledger); v=affected(base,filtered,dates,*VALIDATION[1:],ledger)
  df=phase_flags(m,'DISCOVERY',d,True); vf=phase_flags(m,'VALIDATION',v)
  f=m.set_index(['split','variant'])
  selection.append({**identity,'discovery_pass':all(df.values()),'validation_pass':all(vf.values()),
   'fit_pass':all(df.values()) and all(vf.values()),'discovery_failed':'|'.join(k for k,v in df.items() if not v),
   'validation_failed':'|'.join(k for k,v in vf.items() if not v),'discovery_changed_n':d['changed_closed_n'],
   'validation_changed_n':v['changed_closed_n'],
   'validation_annualized_pct':float(f.loc[('VALIDATION','FILTER'),'gross_annualized_simple_pct']),
   'fit_delta_pp':float(f.loc[('FIT','FILTER'),'gross_simple_return_pct']-f.loc[('FIT','BASE'),'gross_simple_return_pct'])})
  if (offset+1)%100==0: print('D/V replay',offset+1,'/',len(proposals),flush=True)
 columns=['candidate_id','pair','strategy','filter_id','filter_name','is_new','discovery_pass','validation_pass','fit_pass','discovery_failed','validation_failed','discovery_changed_n','validation_changed_n','validation_annualized_pct','fit_delta_pp']
 frame=pd.DataFrame(selection,columns=columns).sort_values(['validation_annualized_pct','fit_delta_pp','candidate_id'],ascending=[False,False,True],kind='stable')
 frame['preconfirmation_rank']=np.nan
 chosen=frame.loc[frame.fit_pass.eq(True)].drop_duplicates(['pair','strategy'])
 frame.loc[chosen.index,'preconfirmation_rank']=np.arange(1,len(chosen)+1)
 frame.to_csv(output/'fit_selection.csv',index=False); core.add_retention(pd.DataFrame(rows)).to_csv(output/'fit_metrics.csv',index=False)
 lookup={r['candidate_id']:r for r in proposals}
 locked=[{**r,'spec':lookup[r['candidate_id']]['spec']} for r in frame.loc[frame.preconfirmation_rank.notna()].sort_values('preconfirmation_rank').to_dict('records')]
 frozen_daily=[]
 for item in locked:
  values,dates,features,targets=prepared[item['pair']]; base=cache[item['pair'],item['strategy']][0]
  filtered,_=run_state_filter(values,dates,base,features,item['spec'],item['pair'],item['strategy'])
  for variant,run in [('BASE',base),('FILTER',filtered)]:
   frozen_daily.append(core.daily_rows(key(item['pair'],item['strategy'],item['spec']),run,dates,[('FIT',core.IS_START,VALIDATION[2])],variant))
 if frozen_daily: pd.concat(frozen_daily,ignore_index=True).to_csv(output/'fit_locked_daily.csv.gz',index=False)
 integrity(protocol)
 core.save_json(output/'validation_locked.json',{'created_utc':core.now(),'protocol_sha256':core.sha(output/'protocol.json'),
   'candidates':locked,'confirmation_computed':False,'oos_computed':False,
   'source_sha256':protocol['source_sha256'],'input_sha256':protocol['input_sha256'],
   'fit_output_sha256':{n:core.sha(output/n) for n in ['fit_selection.csv','fit_metrics.csv','discovery_proposals.json',*(['fit_locked_daily.csv.gz'] if frozen_daily else [])]}})
 core.save_json(output/'fit_status.json',{'status':'passed','finished_utc':core.now(),'locked_candidates':len(locked),
   'proposals':len(proposals),'cutoff':VALIDATION[2],'lock_sha256':core.sha(output/'validation_locked.json')})
 print('Pre-confirmation locked:',len(locked),flush=True)


def load_locked(output):
 protocol=core.read_json(output/'protocol.json'); locked=core.read_json(output/'validation_locked.json')
 integrity(protocol)
 if core.sha(output/'validation_locked.json')!=core.read_json(output/'fit_status.json')['lock_sha256']: raise ValueError('Validation lock changed.')
 if core.sha(output/'protocol.json')!=locked['protocol_sha256']: raise ValueError('Protocol changed.')
 for name,digest in locked['fit_output_sha256'].items():
  if core.sha(output/name)!=digest: raise ValueError('Fit evidence changed: '+name)
 return protocol,locked


def confirm(output):
 protocol,locked=load_locked(output)
 if (output/'frozen_selection.json').exists(): raise ValueError('Confirmation already finished; no replacement.')
 prepared=core.prepare(core.IS_END) if locked['candidates'] else {}
 fit_daily=pd.read_csv(output/'fit_locked_daily.csv.gz',parse_dates=['date']) if locked['candidates'] else pd.DataFrame()
 rows=[]; extra=[]; decisions=[]; daily=[]; ledgers=[]; evidence=[]
 for item in locked['candidates']:
  pair,strategy,spec=item['pair'],item['strategy'],item['spec']; identity=key(pair,strategy,spec)
  values,dates,features,targets=prepared[pair]; base=fixed_execute(values,targets[strategy])
  filtered,decision=run_state_filter(values,dates,base,features,spec,pair,strategy)
  for variant,run in [('BASE',base),('FILTER',filtered)]:
   reference=fit_daily.loc[fit_daily.candidate_id.eq(identity['candidate_id'])&fit_daily.variant.eq(variant)].sort_values('date')
   mask=(dates>=core.IS_START)&(dates<=VALIDATION[2])
   if not pd.DatetimeIndex(reference.date).equals(dates[mask]): raise AssertionError('Fit calendar changed during confirmation.')
   for field in ('executed','held'):
    if not np.array_equal(reference[field],np.asarray(run[field])[mask]): raise AssertionError('Fit positions changed during confirmation.')
   if not np.allclose(reference.gross_daily_return,np.asarray(run['gross_return'])[mask],atol=1e-14,rtol=0): raise AssertionError('Fit returns changed during confirmation.')
   rows.extend(core.metric_rows(identity,run,dates,[('IS',core.IS_START,core.IS_END),*IS_BLOCKS],variant))
   extra.extend(core.metric_rows(identity,run,dates,[CONFIRMATION,RECENT],variant))
   daily.append(core.daily_rows(identity,run,dates,[('IS',core.IS_START,core.IS_END)],variant))
   ledgers.append(simple_ledger(run,dates).assign(**identity,variant=variant))
  decisions.append(decision.assign(**identity))
  evidence.append({'candidate_id':identity['candidate_id'],**{label:affected(base,filtered,dates,start,end) for label,start,end in [('IS',core.IS_START,core.IS_END),CONFIRMATION,RECENT]}})
 if rows:
  metrics=core.add_retention(pd.DataFrame(rows)); aux=core.add_retention(pd.DataFrame(extra)); selection=screen_candidates(metrics)
  selection['original_is_pass']=selection['pass']; selection['preconfirmation_rank']=selection.candidate_id.map({r['candidate_id']:r['preconfirmation_rank'] for r in locked['candidates']})
  ev={r['candidate_id']:r for r in evidence}
  for i,row in selection.iterrows():
   sub=aux.loc[aux.candidate_id.eq(row.candidate_id)]; cf=phase_flags(sub,'CONFIRMATION',ev[row.candidate_id]['CONFIRMATION'])
   recent=sub.loc[sub.split.eq('RECENT')].set_index('variant'); b,f=recent.loc['BASE'],recent.loc['FILTER']
   rf={'recent_absolute':f.gross_annualized_simple_pct>=1.-1e-10,'recent_delta':f.gross_simple_return_pct>b.gross_simple_return_pct+1e-10,'recent_ratio':ratio_improves(b,f),
       'full_effect_sample':ev[row.candidate_id]['IS']['changed_closed_n']>=20,
       'full_top3_robust':ev[row.candidate_id]['IS']['delta_without_best3_pp']>1e-10,
       'recent_top3_robust':ev[row.candidate_id]['RECENT']['delta_without_best3_pp']>1e-10}
   selection.loc[i,'confirmation_pass']=all(cf.values()); selection.loc[i,'robustness_pass']=all(rf.values())
   selection.loc[i,'confirmation_failed']='|'.join(k for k,v in cf.items() if not v)
   selection.loc[i,'robustness_failed']='|'.join(k for k,v in rf.items() if not v)
   selection.loc[i,'pass']=bool(row['pass'] and all(cf.values()) and all(rf.values()))
  selection['frozen']=selection['pass']; selection['freeze_rank']=np.nan
  admitted=selection.loc[selection.frozen].sort_values('preconfirmation_rank')
  selection.loc[admitted.index,'freeze_rank']=np.arange(1,len(admitted)+1)
  selection['filter_name']=selection.candidate_id.map({r['candidate_id']:r['filter_name'] for r in locked['candidates']})
  selection=selection.sort_values('preconfirmation_rank'); metrics.to_csv(output/'confirmation_full_metrics.csv',index=False); aux.to_csv(output/'confirmation_recent_metrics.csv',index=False)
  path=pd.concat(daily,ignore_index=True); path.to_csv(output/'locked_is_daily.csv.gz',index=False)
  pd.concat(ledgers,ignore_index=True).to_csv(output/'locked_is_trade_ledger.csv.gz',index=False)
  pd.concat(decisions,ignore_index=True).to_csv(output/'locked_is_decisions.csv.gz',index=False)
 else:
  selection=pd.DataFrame(columns=['candidate_id','pair','strategy','filter_id','filter_name','is_new','pass','frozen','freeze_rank','preconfirmation_rank'])
  metrics=pd.DataFrame(); path=pd.DataFrame(columns=['candidate_id','pair','strategy','filter_id','filter_name','is_new','variant','split','date','gross_daily_return','net_daily_return','executed','held'])
 selection.to_csv(output/'confirmation_selection.csv',index=False); core.save_json(output/'confirmation_effect_evidence.json',evidence)
 survivors=set(selection.loc[selection.frozen,'candidate_id'])
 if survivors:
  main=metrics.loc[metrics.candidate_id.isin(survivors)&metrics.split.eq('IS')]; main.to_csv(output/'is_frozen_metrics.csv',index=False)
  path.loc[path.candidate_id.isin(survivors)].to_csv(output/'is_frozen_daily.csv.gz',index=False)
 else:
  pd.DataFrame(columns=['candidate_id','pair','strategy','filter_id','filter_name','variant','split']).to_csv(output/'is_frozen_metrics.csv',index=False)
  path.iloc[:0].to_csv(output/'is_frozen_daily.csv.gz',index=False)
 lookup={r['candidate_id']:r for r in locked['candidates']}
 frozen=[{**r,'spec':lookup[r['candidate_id']]['spec']} for r in selection.loc[selection.frozen].to_dict('records')]
 integrity(protocol)
 core.save_json(output/'frozen_selection.json',{'created_utc':core.now(),'protocol_sha256':core.sha(output/'protocol.json'),
  'preconfirmation_lock_sha256':core.sha(output/'validation_locked.json'),'source_sha256':protocol['source_sha256'],'input_sha256':protocol['input_sha256'],
  'oos_evaluated_at_freeze':False,'candidates':frozen,'frozen_candidates':len(frozen),
  'is_output_sha256':{p.name:core.sha(p) for p in output.glob('*.csv*')}})
 core.save_json(output/'development_status.json',{'status':'passed','finished_utc':core.now(),'is_screen_completed':True,
  'frozen_candidates':len(frozen),'preconfirmation_locked':len(locked['candidates']),'oos_evaluated':False,
  'freeze_sha256':core.sha(output/'frozen_selection.json'),'compute_cutoff':core.IS_END})
 print(selection.to_string(index=False),flush=True); print('Confirmed IS survivors:',len(frozen),flush=True)


def oos(output):
 # Reuse the audited v1 evaluation orchestration, swapping only the candidate
 # replay function. Source hashes explicitly include both modules and this file.
 if core.sha(output/'frozen_selection.json')!=core.read_json(output/'development_status.json')['freeze_sha256']: raise ValueError('Final freeze changed.')
 core.run_candidate=run_state_filter
 core.identity=key
 core.evaluate_oos(output)


def report(output):
 from fx_filter_report import render_filter_report
 protocol=core.read_json(output/'protocol.json'); freeze=core.read_json(output/'frozen_selection.json')
 validation=core.read_json(output/'development_status.json')
 selection=pd.read_csv(output/'confirmation_selection.csv'); daily=pd.read_csv(output/'is_frozen_daily.csv.gz',parse_dates=['date']); metrics=pd.read_csv(output/'is_frozen_metrics.csv')
 if (output/'oos_status.json').exists() and core.read_json(output/'oos_status.json')['status']=='passed':
  validation.update(core.read_json(output/'oos_status.json')); daily=pd.concat([daily,pd.read_csv(output/'oos_daily.csv.gz',parse_dates=['date'])],ignore_index=True); metrics=pd.concat([metrics,pd.read_csv(output/'oos_metrics.csv')],ignore_index=True)
 validation.update(protocol_sha256=core.sha(output/'protocol.json'),freeze_sha256=core.sha(output/'frozen_selection.json'))
 built=render_filter_report(output,daily,metrics,selection,protocol,validation); core.save_json(output/'report_build.json',built)
 print('Report:',output/built['report'])


def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=['fit','confirm','oos','report'],required=True); parser.add_argument('--output',type=Path,default=DEFAULT)
 args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
 {'fit':fit,'confirm':confirm,'oos':oos,'report':report}[args.stage](output)

if __name__=='__main__': main()
