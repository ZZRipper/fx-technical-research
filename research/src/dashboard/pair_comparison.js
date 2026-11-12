/* Descriptive evidence only. No strategy selection or modification of account paths. */
(function(){
 'use strict';
 const A=window.FXResearch;if(!A||!A.ready)return;
 const P=JSON.parse(document.getElementById('pair-comparison-data').textContent),D=A.data,$=id=>document.getElementById(id);
 const e=A.escapeHTML,n=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'—';
 const table=(h,rows)=>'<table><thead><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
 // Reviewed conclusions for the fixed workbook window, 2020-01-01–2026-05-29.
 // These are qualitative readings of displayed evidence, not fitted classifications.
 const conclusions={
  EUR:{title:'Mixed evidence; no clear style preference',
   basis:'MR has a small median-Sharpe lead, but annual leadership is evenly split. Five of six Trend families have positive median returns.',
   limit:'RSI MR supports the MR result; Bollinger MR is close to flat on median Sharpe. The pre-2020 category ranking is reversed, with both medians negative.'},
  GBP:{title:'Mean Reversion has broader support',
   basis:'Both MR families have positive median returns. MR leads in five of six complete years and remains ahead when either MR family is removed.',
   limit:'This is not an all-period property: Trend leads in the 2005–09 and 2010–14 blocks. Costs and drawdowns still matter.'},
  AUD:{title:'Bollinger Mean Reversion, not MR in general',
   basis:'Both Bollinger MR lookbacks are profitable; the RSI MR family has a negative median return. MR leads in five of six complete years.',
   limit:'Removing both Bollinger families reverses the category ranking slightly, with both remaining category medians negative. Do not generalize the Bollinger result to all MR rules.'},
  NZD:{title:'Mixed evidence; MR lead depends on Bollinger',
   basis:'The MR median is only slightly positive, annual leadership is split three–three, and RSI MR has a negative median return.',
   limit:'Removing both Bollinger families reverses the ranking, with both remaining category medians negative. MACD is profitable, while most other Trend families are not.'},
  JPY:{title:'Trend / Momentum leads in the recent period',
   basis:'Four of six Trend families have positive median returns; neither MR family does. Trend leads in four of six complete years and survives family-removal checks.',
   limit:'Pre-2020 aggregate ranking favors MR. The RSI Momentum family median and MACD return are negative in the recent window; the result does not cover every Trend rule.'},
  CHF:{title:'Mean Reversion has a modest relative lead',
   basis:'Both MR families have positive median returns. MR leads in four of six complete years and remains ahead without either MR family.',
   limit:'The recent MR median Sharpe is modest. Trend leads in 2010–14; positive MACD and Moving Average results show that the category comparison has exceptions.'},
  CAD:{title:'Mean Reversion leads at category level, with exceptions',
   basis:'Both MR families have positive median returns, versus one of six Trend families. MR leads in four of six complete years.',
   limit:'The recent MR median Sharpe is low and Bollinger MR has little cumulative gain. MACD is a profitable Trend exception; the 2015–19 block favors Trend.'},
  NOK:{title:'No clear style advantage; category medians near zero',
   basis:'MR ranks above Trend and leads in five of six complete years, but the full-window category median Sharpes are both close to zero.',
   limit:'The recent ranking alone is insufficient to establish a useful edge. Both pre-2020 category medians are negative; costs could materially change these small gross results.'},
  SEK:{title:'Mixed: RSI MR and MACD provide different evidence',
   basis:'MR leads at category level and in four of six complete years. RSI MR has a positive family median; Bollinger MR median Sharpe is near zero.',
   limit:'MACD has a higher recent Sharpe than any MR rule, while the Trend category median is near zero. A blanket MR-over-Trend conclusion would hide this exception.'}
 };
 let pair='AUD';
 function showDetail(view){const timeline=view==='timeline';$('pc-evidence-view').hidden=timeline;$('pc-timeline-view').hidden=!timeline;$('pc-view-evidence').setAttribute('aria-pressed',String(!timeline));$('pc-view-timeline').setAttribute('aria-pressed',String(timeline));}
 $('pc-view-evidence').addEventListener('click',()=>showDetail('evidence'));
 $('pc-view-timeline').addEventListener('click',()=>showDetail('timeline'));
 $('pc-rules').addEventListener('click',event=>{const button=event.target.closest('[data-timeline-rule]');if(button){window.FXPairTimeline.setRule(button.dataset.timelineRule);showDetail('timeline');}});
 function delta(s){return Number.isFinite(s.MR.sharpe)&&Number.isFinite(s.Trend.sharpe)?s.MR.sharpe-s.Trend.sharpe:null;}
 function leader(d){return d===null?'Undefined':Math.abs(d)<1e-12?'Tie':d>0?'MR':'Trend';}
 function annualCounts(p){const ds=P.completeRecent.map(y=>delta(P.summaries[p][y]));return [ds.filter(x=>x!==null&&x>1e-12).length,ds.filter(x=>x!==null&&x< -1e-12).length,ds.filter(x=>x!==null&&Math.abs(x)<=1e-12).length];}
 function sensitivity(p){const base=leader(delta(P.summaries[p].recent));const tests=P.sensitivity[p].slice(1,9);return {same:tests.filter(x=>leader(x.MR-x.Trend)===base).length,n:tests.length};}
 function overview(){
  $('pc-overview').innerHTML=table(['Pair','Conclusion · 2020 onward','Supporting evidence','Qualification'],P.pairs.map(p=>{const x=conclusions[p];return [`<button data-pc-pair="${e(p)}" aria-pressed="${p===pair}">${e(p)} / USD</button>`,`<strong>${e(x.title)}</strong>`,e(x.basis),e(x.limit)];}));
  $('pc-statistics').innerHTML=table(['Pair','MR median Sharpe','Trend median Sharpe','Positive families · MR / Trend','Complete years led · MR / Trend','2020+ median ranking','Evidence qualification'],P.pairs.map(p=>{
   const s=P.summaries[p].recent,h=P.summaries[p].history,c=annualCounts(p),z=sensitivity(p),lead=leader(delta(s));
   const mirrored=P.sensitivity[p].slice(-1)[0],mirrorLead=leader(mirrored.MR-mirrored.Trend);
   const sign=s.MR.sharpe<=0&&s.Trend.sharpe<=0?'Both category medians are non-positive. ':'';
   return [e(p)+' / USD',n(s.MR.sharpe,3),n(s.Trend.sharpe,3),`${s.MR.positive}/2 · ${s.Trend.positive}/6`,`${c[0]} / ${c[1]} of ${P.completeRecent.length}${c[2]?' · '+c[2]+' ties':''}`,lead==='Tie'?'Tie':lead+' higher',sign+`Pre-2020: ${leader(delta(h))}. Family-removal agreement: ${z.same}/${z.n}. Without both Bollinger families: ${mirrorLead}.`];
  }));
 }
 function render(){
  overview();const R=P.results[pair],S=P.summaries[pair],c=annualCounts(pair),z=sensitivity(pair),lead=leader(delta(S.recent));
  $('pc-title').textContent=pair+' / USD · evidence detail';
  const conclusion=conclusions[pair];
  $('pc-conclusion').innerHTML=`<h3>${e(conclusion.title)}</h3><p><b>Supporting evidence:</b> ${e(conclusion.basis)}</p><p><b>Qualification:</b> ${e(conclusion.limit)}</p>`;
  $('pc-reading').textContent=`2020–${P.end}: ${lead} has the higher category-median Sharpe (MR ${n(S.recent.MR.sharpe,3)}, Trend ${n(S.recent.Trend.sharpe,3)}). MR leads in ${c[0]} and Trend in ${c[1]} of ${P.completeRecent.length} complete years. The ranking is unchanged in ${z.same} of ${z.n} single-family removals. Without both Bollinger families the ranking is ${leader(P.sensitivity[pair].slice(-1)[0].MR-P.sensitivity[pair].slice(-1)[0].Trend)}. Ranking can persist even when both sides lose. This is not a test of statistical significance or future profitability.`;
  const periodKeys=['early','middle','late','history','recent'];
  $('pc-periods').innerHTML=table(['Period','MR median Sharpe','Trend median Sharpe','Difference · MR − Trend','Positive median-return families · MR / Trend'],periodKeys.map(k=>{const s=S[k];return [P.windows[k].join(' → '),n(s.MR.sharpe,3),n(s.Trend.sharpe,3),n(delta(s),3),`${s.MR.positive}/2 · ${s.Trend.positive}/6`];}));
  const familyOf=r=>Object.entries(P.families).find(([,v])=>v[1].includes(r));
  $('pc-family').innerHTML=table(['Family','Category','Median Sharpe','Rule Sharpe range','Profitable rules','Median rule return (%)'],Object.entries(P.families).map(([f,[cat,rs]])=>{const g=S.recent[cat].groups.find(x=>x.family===f),v=rs.map(r=>R[r].recent);return [e(f),cat,n(g.sharpe,3),n(Math.min(...v.map(x=>x.sharpe)))+' to '+n(Math.max(...v.map(x=>x.sharpe))),v.filter(x=>x.ret>0).length+' / '+v.length,n(g.ret)];}));
  const rules=P.ruleOrder.concat(['ALWAYS_LONG_FX','ALWAYS_SHORT_FX']);
  $('pc-rules').innerHTML=table(['Rule / category','Simple return (%)','Sharpe','Max drawdown (pp)','Annualized volatility (%)','Closed trades','Exposure days (%)'],rules.map(r=>{const x=R[r].recent,f=familyOf(r);return [`${f?`<button type="button" data-timeline-rule="${r}" title="Inspect performance timeline">${e(D.rules[r].label)} ↗</button>`:`<span class="pc-label">${e(D.rules[r].label)}</span>`}<small>${f?f[1][0]:'Benchmark'}${f&&x.closed<20?' · low trade count':''}</small>`,n(x.ret),n(x.sharpe,3),n(x.dd),n(x.vol),x.closed,n(x.exposure,1)];}));
  $('pc-heatmap').innerHTML=table(['Rule',...displayYears().map(y=>String(y)+(P.windows[String(y)][1]!==`${y}-12-31`?'*':''))],rules.map(r=>[e(D.rules[r].label),...displayYears().map(y=>{const v=R[r][y].ret,alpha=.06+.48*Math.min(Math.abs(v)/20,1);return `<span style="display:block;padding:5px;background:rgba(${v>=0?'0,126,128':'183,92,39'},${alpha})" title="${e(D.rules[r].label)} · ${y}: ${n(v,4)}%">${n(v,1)}</span>`;})]));
  $('pc-sensitivity').innerHTML=table(['Included families','MR median Sharpe','Trend median Sharpe','Higher category median','Same ranking as full set?'],P.sensitivity[pair].map(x=>{const l=leader(x.MR-x.Trend);return [e(x.label),n(x.MR,3),n(x.Trend,3),l,l===lead?'Yes':'No'];}));
  window.FXPairTimeline.setPair(pair);
  $('pc-definitions').innerHTML=table(['Rule','Original definition'],P.ruleOrder.map(r=>[e(D.rules[r].label),e(D.rules[r].rule)]));
 }
 function displayYears(){const view=$('pc-years').value;return P.years.filter(y=>view==='all'||(view==='recent'?y>=2020:y<2020));}
 $('pc-years').addEventListener('change',render);
 $('pc-scope').textContent=`Main workbook: ${P.start}–${P.end}. Primary evaluation: 2020-01-01–${P.end}; historical reference: before 2020. All periods have been examined. Not an untouched OOS test. Gross spot returns: costs and carry are not included. No H.10 continuation is spliced into this sample.`;
 $('pc-validation').textContent=`Build checks: ${P.validation.accounts} accounts (${P.validation.rules} eligible rules + 2 benchmarks per pair), matching daily calendars; reconstructed positions × FX returns reconcile to existing daily PnL. Maximum daily-return error: ${P.validation.daily_return_max_error}; maximum period-return discrepancy: ${P.validation.period_return_max_error_pp.toExponential(2)} pp.`;
 $('pc-pair').innerHTML=P.pairs.map(p=>`<option value="${e(p)}">${e(p)} / USD</option>`).join('');$('pc-pair').value=pair;
 $('pc-pair').addEventListener('change',ev=>{pair=ev.target.value;render();});
 $('pc-overview').addEventListener('click',ev=>{const b=ev.target.closest('[data-pc-pair]');if(b){pair=b.dataset.pcPair;$('pc-pair').value=pair;render();}});
 $('module-comparison').addEventListener('click',()=>{location.hash='comparison';});
 $('pc-export').addEventListener('click',()=>{
  const rows=[['pair','rule','category','family','window','from','through','simple_return_pct','sharpe_zero_reference','max_drawdown_initial_capital_pp','annualized_volatility_pct','closed_exit_cohort','exposure_pct','observations']];
  for(const p of P.pairs)for(const r of rulesForExport())for(const [k,v] of Object.entries(P.results[p][r])){const f=Object.entries(P.families).find(([,v])=>v[1].includes(r));rows.push([p,r,f?f[1][0]:'Benchmark',f?f[0]:'Benchmark',k,...P.windows[k],v.ret,v.sharpe,v.dd,v.vol,v.closed,v.exposure,v.n]);}
  const blob=new Blob([rows.map(row=>row.map(x=>'"'+String(x??'').replace(/"/g,'""')+'"').join(',')).join('\n')],{type:'text/csv'});const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='FX_Pair_Strategy_Comparison.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 function rulesForExport(){return P.ruleOrder.concat(['ALWAYS_LONG_FX','ALWAYS_SHORT_FX']);}
 render();
 showDetail(new URLSearchParams(location.search).get('view')==='timeline'?'timeline':'evidence');
})();
