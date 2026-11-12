/* Exit-date trade cohorts. Reuse original BASE ledgers without changing execution. */
(function(){
 'use strict';
 const CAPITAL=100000;
 function summarize(trades,start,end,tolerance){
  const closed=trades.filter(t=>t[4]&&t[1]>=start&&t[1]<=end&&t[3]>0);
  const winners=closed.filter(t=>t[2]>tolerance).sort((a,b)=>b[2]-a[2]);
  const losers=closed.filter(t=>t[2]<-tolerance);
  const flat=closed.filter(t=>Math.abs(t[2])<=tolerance);
  const dollars=t=>t[2]*CAPITAL/100;
  const sum=ts=>ts.reduce((v,t)=>v+dollars(t),0);
  const grossProfit=sum(winners),grossLoss=-sum(losers),net=sum(closed);
  const avgWin=winners.length?grossProfit/winners.length:null,avgLoss=losers.length?grossLoss/losers.length:null;
  const top=[1,5,10].map(k=>{
   const n=Math.min(k,winners.length),profit=sum(winners.slice(0,n));
   return {k,n,profit,share:grossProfit>0?profit/grossProfit*100:null,remaining:net-profit};
  });
  return {closed,winners,losers,flat,grossProfit,grossLoss,flatPnl:sum(flat),net,avgWin,avgLoss,top,
   winRate:closed.length?winners.length/closed.length*100:null,
   payoff:avgLoss>0&&avgWin!==null?avgWin/avgLoss:null,
   holding:closed.length?closed.reduce((v,t)=>v+t[3],0)/closed.length:null,
   cross:closed.filter(t=>t[0]<start).length,
   open:trades.filter(t=>t[0]<=end&&(!t[4]||t[1]>end)).length,
   zero:trades.filter(t=>t[4]&&t[1]>=start&&t[1]<=end&&t[3]<=0).length};
 }
 function histogram(values){
  if(!values.length)return {bins:[],limit:1};
  const limit=Math.max(1,...values.map(Math.abs))*1.001,step=2*limit/30;
  const bins=Array.from({length:30},(_,i)=>({lo:-limit+i*step,hi:-limit+(i+1)*step,n:0}));
  values.forEach(v=>bins[Math.min(29,Math.max(0,Math.floor((v+limit)/step)))].n++);
  return {bins,limit};
 }
 window.FXTradeDistributionMath={summarize,histogram};
 const A=window.FXResearch;if(!A||!A.ready)return;
 const D=A.data,$=id=>document.getElementById(id),e=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const accounts=D.accounts.filter(a=>a.ds==='main'&&a.variant==='BASE');
 const pairs=[...new Set(accounts.map(a=>a.pair))],rules=Object.keys(D.rules).filter(r=>accounts.some(a=>a.rule===r));
 const state={pair:'AUD',rule:'BB_MR_20',from:D.datasets.main.start,to:D.datasets.main.end,page:0,sort:'recent'};
 let result=null;
 const num=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n}):'—';
 const usd=v=>Number.isFinite(v)?(v<0?'−':'')+'$'+num(Math.abs(v),2):'—';
 const pct=v=>Number.isFinite(v)?num(v)+'%':'—';
 const table=(head,rows)=>'<table><thead><tr>'+head.map(v=>'<th>'+v+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
 const empty=s=>'<div class="pd-empty">'+s+'</div>';
 const dims={W:900,H:350,L:70,R:25,T:30,B:65};
 function axes(xmin,xmax,ymax,xlabel,ylabel){
  const {W,H,L,R,T,B}=dims,X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),Y=y=>H-B-y/ymax*(H-T-B);
  let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${xlabel}"><rect width="100%" height="100%" fill="white"/>`;
  for(let i=0;i<=4;i++){let v=ymax*i/4;out+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e3eaf0"/><text x="${L-10}" y="${Y(v)+4}" text-anchor="end" font-size="12" fill="#607487">${num(v,Number.isInteger(v)?0:1)}</text>`;}
  for(let i=0;i<=4;i++){let v=xmin+(xmax-xmin)*i/4;out+=`<text x="${X(v)}" y="${H-B+24}" text-anchor="middle" font-size="12" fill="#607487">${num(v,0)}</text>`;}
  out+=`<text x="${L}" y="16" font-size="12" fill="#607487">${ylabel}</text><text x="${W/2}" y="${H-10}" text-anchor="middle" font-size="13" fill="#607487">${xlabel}</text>`;
  return {out,X,Y};
 }
 function drawHistogram(q){
  if(!q.closed.length)return empty('No eligible completed trades in this exit-date window.');
  const {bins,limit}=histogram(q.closed.map(t=>t[2]*1000));
  const ymax=Math.max(4,Math.ceil(Math.max(...bins.map(b=>b.n))/4)*4);
  let {out,X,Y}=axes(-limit,limit,ymax,'Trade PnL (USD)','Number of trades');
  bins.forEach(b=>{out+=`<rect x="${X(b.lo)+.5}" y="${Y(b.n)}" width="${Math.max(0,X(b.hi)-X(b.lo)-1)}" height="${Y(0)-Y(b.n)}" fill="${b.hi<=0?'#b75c27':'#007e80'}"><title>${usd(b.lo)} to ${usd(b.hi)}: ${b.n} trades</title></rect>`;});
  out+=`<line x1="${X(0)}" x2="${X(0)}" y1="30" y2="285" stroke="#526879" stroke-dasharray="4 3"/></svg>`;return out;
 }
 function drawConcentration(q){
  if(!q.winners.length)return empty('No winning trades in this cohort; gross-profit concentration is undefined.');
  let {out,X,Y}=axes(0,100,100,'Winning trades included (%) · largest first','Cumulative gross profit (%)'),cum=0;
  const points=[[0,0],...q.winners.map((t,i)=>{cum+=t[2]*1000;return [(i+1)/q.winners.length*100,cum/q.grossProfit*100];})];
  out+=`<path d="M${X(0)},${Y(0)} L${X(100)},${Y(100)}" stroke="#9aaaba" stroke-dasharray="5 5" fill="none"/><path d="${points.map((p,i)=>(i?'L':'M')+X(p[0])+','+Y(p[1])).join(' ')}" fill="none" stroke="#007e80" stroke-width="3"/>`;
  q.top.forEach(t=>{out+=`<circle cx="${X(t.n/q.winners.length*100)}" cy="${Y(t.share)}" r="4" fill="#b75c27"><title>Top ${t.n} winners: ${pct(t.share)} of gross profit</title></circle>`;});return out+'</svg>';
 }
 function ledger(){
  const rows=result.closed.slice().sort((a,b)=>state.sort==='winner'?b[2]-a[2]:state.sort==='loser'?a[2]-b[2]:state.sort==='absolute'?Math.abs(b[2])-Math.abs(a[2]):b[1]-a[1]);
  const pages=Math.max(1,Math.ceil(rows.length/50));state.page=Math.min(state.page,pages-1);
  $('td-ledger').innerHTML=table(['Entry','Exit','Direction','Holding days','Trade PnL','Cross-window entry'],rows.slice(state.page*50,(state.page+1)*50).map(t=>[A.dateString(t[0]),A.dateString(t[1]),t[5]>0?'Long FX':'Short FX',num(t[3],0),usd(t[2]*1000),t[0]<A.toDay(state.from)?'Yes':'No']));
  $('td-page').textContent=`Page ${state.page+1} of ${pages} · ${rows.length} trades`;$('td-prev').disabled=!state.page;$('td-next').disabled=state.page>=pages-1;
 }
 function render(){
  const start=A.toDay(state.from),end=A.toDay(state.to),a=accounts.find(a=>a.pair===state.pair&&a.rule===state.rule);
  let error='';
  if(!state.from||!state.to||!Number.isFinite(start)||!Number.isFinite(end)||start>end)error='Choose a valid date range.';
  else if(state.from<D.datasets.main.start||state.to>D.datasets.main.end)error=`Use dates within ${D.datasets.main.start}–${D.datasets.main.end}.`;
  else if(!a)error='No original Baseline account is available for this pair and strategy.';
  $('td-error').textContent=error;$('td-error').hidden=!error;$('td-content').hidden=!!error;if(error){result=null;return;}
  result=summarize(a.trades,start,end,D.flatTolerancePct);const q=result;
  $('td-title').textContent=state.pair+' / USD · '+D.rules[state.rule].label;
  $('td-rule-note').textContent=D.rules[state.rule].rule;
  const cards=[['Closed trades',num(q.closed.length,0)],['Win rate',pct(q.winRate)],['Average win',usd(q.avgWin)],['Average loss · absolute',usd(q.avgLoss)],['Payoff',num(q.payoff,3)],['Average holding days',num(q.holding,1)],['Closed-trade net PnL',q.closed.length?usd(q.net):'—'],['Open at selected end',num(q.open,0)]];
  $('td-cards').innerHTML=cards.map(([k,v])=>`<div class="td-card"><small>${k}</small><strong>${v}</strong></div>`).join('');
  $('td-cohort-note').textContent=`Exit dates: ${state.from}–${state.to} · ${q.cross} cross-window entries included · ${q.zero} zero-return-interval trades excluded · ${q.flat.length} flat trades included in win-rate denominator. Open positions are excluded from the distribution.`;
  $('td-small').textContent=q.closed.length<20?'Fewer than 20 completed trades: interpret this small cohort cautiously.':'';
  $('td-histogram').innerHTML=drawHistogram(q);
  $('td-reconcile').innerHTML=table(['Winning trades','Losing trades','Flat trades','Gross profit','Gross loss · absolute','Flat-trade PnL','Net PnL'],[[q.winners.length,q.losers.length,q.flat.length,usd(q.grossProfit),usd(q.grossLoss),usd(q.flatPnl),usd(q.net)]]);
  $('td-concentration').innerHTML=drawConcentration(q);
  $('td-top').innerHTML=table(['Winners removed','Profit removed','Share of gross profit','Net PnL excluding top winners'],q.top.map(t=>[t.n===t.k?`Top ${t.k}`:`Top ${t.k} · only ${t.n} available`,q.winners.length?usd(t.profit):'—',pct(t.share),q.winners.length?usd(t.remaining):'—']));
  ledger();
  document.querySelectorAll('#td-presets button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.from===state.from&&b.dataset.to===state.to)));
 }
 $('td-pair').innerHTML=pairs.map(p=>`<option value="${e(p)}">${e(p)} / USD</option>`).join('');
 $('td-rule').innerHTML=rules.map(r=>`<option value="${e(r)}">${e(D.rules[r].label)}</option>`).join('');
 for(const k of ['pair','rule','from','to']){const el=$('td-'+k);el.value=state[k];if(k==='from'||k==='to'){el.min=D.datasets.main.start;el.max=D.datasets.main.end;}el.addEventListener('change',()=>{state[k]=el.value;state.page=0;render();});}
 const periods=[['Full',D.datasets.main.start,D.datasets.main.end],['2005–09','2005-01-03','2009-12-31'],['2010–14','2010-01-01','2014-12-31'],['2015–19','2015-01-01','2019-12-31'],['2020–24','2020-01-01','2024-12-31'],['2025–26','2025-01-01',D.datasets.main.end]];
 $('td-presets').innerHTML=periods.map(([n,f,t])=>`<button type="button" data-from="${f}" data-to="${t}">${n}</button>`).join('');
 $('td-presets').addEventListener('click',ev=>{const b=ev.target.closest('[data-from]');if(!b)return;state.from=b.dataset.from;state.to=b.dataset.to;$('td-from').value=state.from;$('td-to').value=state.to;state.page=0;render();});
 $('td-sort').addEventListener('change',ev=>{state.sort=ev.target.value;state.page=0;if(result)ledger();});
 $('td-prev').addEventListener('click',()=>{state.page--;ledger();});$('td-next').addEventListener('click',()=>{state.page++;ledger();});
 $('module-trades').addEventListener('click',()=>{location.hash='trades';});
 render();
})();
