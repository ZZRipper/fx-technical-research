/* Parameter diagnostics uses the same accountWindow accounting as the explorer. */
(function () {
 'use strict';
 const A=window.FXResearch, $=id=>document.getElementById(id);
 if(!A || !A.ready) return;
 const P=JSON.parse(pako.ungzip(Uint8Array.from(atob($('parameter-data').textContent),c=>c.charCodeAt(0)),{to:'string'}));
 const D=A.data, colors={7:'#007e80',10:'#007e80',14:'#5563ac',20:'#b75c27',21:'#007e80',63:'#b75c27','10_30':'#007e80','20_60':'#b75c27'};
 const families={
  bb:{label:'Bollinger Mean Reversion',prefix:'BBMR',indicator:'Z-score',bandwidth:.15,
   tailLabel:'Days outside ±2 (%)',low:v=>v<=-2,high:v=>v>=2,
   lowLabel:'Z ≤ −2 · long condition',highLabel:'Z ≥ +2 · short condition',
   lines:[[-2,'−2 · long threshold'],[2,'+2 · short threshold']],
   axis:'Z-score · price deviation in sample standard deviations'},
  rsi:{label:'RSI Mean Reversion',prefix:'RSI MR ',indicator:'RSI',bandwidth:2.5,
   tailLabel:'Days outside 30–70 (%)',low:v=>v<30,high:v=>v>70,
   lowLabel:'RSI < 30 · oversold days',highLabel:'RSI > 70 · overbought days',
   lines:[[30,'30 · re-entry level'],[50,'50 · exit'],[70,'70 · re-entry level']],
   axis:'Wilder RSI · 0–100'}
 };
 families.bb_bo={...families.bb,label:'Bollinger Breakout',prefix:'BBBO',
  lowLabel:'Z ≤ −2 · short condition',highLabel:'Z ≥ +2 · long condition',
  lines:[[-2,'−2 · short threshold'],[2,'+2 · long threshold']]};
 families.rsi_mom={...families.rsi,label:'RSI Momentum',prefix:'RSI MOM ',
  tailLabel:'Directional target days (%)',low:v=>v<40,high:v=>v>60,
  lowLabel:'RSI < 40 · short target days',highLabel:'RSI > 60 · long target days',
  lines:[[40,'40 · short / flat'],[60,'60 · flat / long']]};
 families.ma={label:'Moving Average',prefix:'MA ',indicator:'MA Gap (%)',bandwidth:.15,
  tailLabel:'Long target days (%)',low:v=>v<0,high:v=>v>0,
  lowLabel:'Gap < 0 · short target days',highLabel:'Gap > 0 · long target days',
  lines:[[0,'0 · direction boundary']],axis:'MA Gap (%) · 100 × (short SMA / long SMA − 1)'};
 families.tsmom={label:'Time-series Momentum',prefix:'TSMOM ',indicator:'Trailing log return (%)',bandwidth:.5,
  tailLabel:'Long target days (%)',low:v=>v<0,high:v=>v>0,
  lowLabel:'Return < 0 · short target days',highLabel:'Return > 0 · long target days',
  lines:[[0,'0 · direction boundary']],axis:'Trailing log return (%) · 100 × ln(price / price n days ago)'};
 families.donchian={label:'Donchian Breakout',prefix:'Donchian ',indicator:'Channel position',bandwidth:.12,
  tailLabel:'Breakout condition days (%)',low:v=>v< -1,high:v=>v>1,
  lowLabel:'Position < −1 · short breakout',highLabel:'Position > +1 · long breakout',
  lines:[[-1,'−1 · lower band'],[0,'0 · midpoint'],[1,'+1 · upper band']],
  axis:'Channel position · (price − prior midpoint) / prior half-width'};
 const parameterName=n=>String(n).replace('_','/');
 const family=()=>families[state.family],spec=()=>P.families[state.family];
 const label=n=>family().prefix+parameterName(n);
 const accounts=new Map(D.accounts.map(a=>[a.id,a]));
 const requestedFamily=new URLSearchParams(location.search).get('family');
 const initialFamily=Object.hasOwn(families,requestedFamily)?requestedFamily:'bb';
 const state={family:initialFamily,pair:'AUD',from:P.start,to:P.end,windows:P.families[initialFamily].windows.slice(),metric:'return',tab:'distribution',scope:'pair',range:'focus'};
 let cache={},currentView=null;
 const number=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n}):'—';
 const percent=v=>Number.isFinite(v)?number(v)+'%':'—';
 const money=v=>Number.isFinite(v)?(v<0?'−':'')+'$'+number(Math.abs(v),0):'—';
 const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
 const quantile=(sorted,p)=>{if(!sorted.length)return null;const i=(sorted.length-1)*p,j=Math.floor(i);return sorted[j]+(sorted[Math.ceil(i)]-sorted[j])*(i-j);};
 const table=(head,rows)=>'<table><thead><tr>'+head.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
 const versions=()=>state.windows.map(n=>`<span style="color:${colors[n]}">${label(n)}</span>`);

 function calculate(pair,n,start,end){
  const s=spec().series[pair][n],a=accounts.get(s.account),r=A.accountWindow(a,D,start,end);
  if(!r)return null;
  const z=s.values.slice(r.first,r.last+1),held=s.held.slice(r.first,r.last+1);
  const closed=a.trades.filter(t=>t[0]>=start&&t[0]<=end&&t[4]&&t[1]<=end&&t[3]>0);
  return {r,z,return:r.ret,drawdown:r.dd,entries:r.quality.entries,
   tails:100*z.filter(v=>['ma','tsmom'].includes(state.family)?family().high(v):(family().low(v)||family().high(v))).length/z.length,
   exposure:100*held.filter(v=>v!==0).length/held.length,
   avgHolding:mean(closed.map(t=>t[3]))};
 }

 function overview(){
  const labels={return:'Simple return (%)',drawdown:'Max drawdown (pp)',tails:family().tailLabel,entries:'Entries',exposure:'Exposure days (%)'};
  const values=P.pairs.flatMap(p=>state.windows.map(n=>cache[p][n][state.metric]));
  const max=Math.max(1,...values.map(Math.abs)),decimals=state.metric==='entries'?0:2;
  const heads=['Currency pair',...state.windows.map(n=>`<span style="color:${colors[n]}">${label(n)}</span><small>${labels[state.metric]}</small>`),...(state.windows.length===2?[`Δ ${parameterName(state.windows[1])} − ${parameterName(state.windows[0])}`]:[])];
  const rows=P.pairs.map(p=>{
   const cells=state.windows.map(n=>{const v=cache[p][n][state.metric];return `<td class="pd-cellbar ${v<0?'neg':''}" style="--bar:${Math.abs(v)/max*85}%">${number(v,decimals)}</td>`;}).join('');
   const diff=state.windows.length===2?`<td>${number(cache[p][state.windows[1]][state.metric]-cache[p][state.windows[0]][state.metric],decimals)}</td>`:'';
   return `<tr class="${p===state.pair?'pd-selected':''}"><td><button class="pd-pair-button" data-pair="${p}" aria-pressed="${p===state.pair}">${p} / USD${p===state.pair?' · selected':''}</button></td>${cells}${diff}</tr>`;
  });
  $('pd-overview').innerHTML='<table><thead><tr>'+heads.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.join('')+'</tbody></table>';
  const sample=cache[P.pairs[0]][spec().windows[0]].r;
  $('pd-overview-note').textContent='Click a pair to update the detail below. '+(state.windows.length===2?`Δ = ${label(state.windows[1])} − ${label(state.windows[0])} in the displayed unit. `:'Versions are compared side by side. ')+'Accounts are separate, not a portfolio.';
  $('pd-coverage').textContent=`9 currency pairs · ${state.windows.length} parameter sets · ${number(sample.observations,0)} common daily observations per pair · ${A.dateString(sample.firstDate)}–${A.dateString(sample.lastDate)}`;
 }

 // Fine-bin Gaussian KDE: deterministic bandwidth, identical across variants.
 // RSI uses reflected kernels at 0 and 100 to respect its bounded support.
 function kde(values,grid,h,bounded){
  const lo=bounded?0:values.reduce((m,v)=>Math.min(m,v-6*h),grid[0]),hi=bounded?100:values.reduce((m,v)=>Math.max(m,v+6*h),grid[grid.length-1]),bins=1024,dx=(hi-lo)/bins;
  const counts=Array(bins).fill(0);
  values.forEach(v=>counts[Math.min(bins-1,Math.max(0,Math.floor((v-lo)/dx)))]++);
  const points=counts.flatMap((count,i)=>count?[[lo+(i+.5)*dx,count]]:[]);
  const kernel=d=>Math.abs(d)>6?0:Math.exp(-.5*d*d);
  return grid.map(x=>{
   let sum=0;
   for(const [v,count] of points){
    sum+=count*(kernel((x-v)/h)+(bounded?kernel((x+v)/h)+kernel((x-(200-v))/h):0));
   }
   return sum/(values.length*h*Math.sqrt(2*Math.PI));
  });
 }

 function densitySVG(series){
  const W=720,H=340,L=56,R=20,T=42,B=48,f=family(),bounded=family().indicator==='RSI';
  const all=series.flatMap(s=>s.z),extent=Math.max(3,Math.ceil(all.reduce((m,v)=>Math.max(m,Math.abs(v)),0)+6*f.bandwidth));
  const focus=state.family==='donchian'&&state.range==='focus';
  const lo=bounded?0:focus?-3:-extent,hi=bounded?100:focus?3:extent;
  const grid=Array.from({length:401},(_,i)=>lo+(hi-lo)*i/400);
  const lines=series.map(s=>({...s,y:kde(s.z,grid,f.bandwidth,bounded)}));
  const ymax=Math.max(...lines.flatMap(s=>s.y))*1.12||1;
  const X=v=>L+(v-lo)/(hi-lo)*(W-L-R),Y=v=>H-B-v/ymax*(H-T-B);
  const out=[`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${f.indicator} KDE density for ${state.scope==='pooled'?'all nine pairs':state.pair}"><rect width="100%" height="100%" fill="white"/>`];
  for(let i=0;i<=4;i++){const v=ymax*i/4;out.push(`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e5ecf1"/><text x="${L-9}" y="${Y(v)+4}" text-anchor="end" fill="#607487" font-size="11">${v.toFixed(bounded?3:2)}</text>`);}
  for(let x=lo;x<=hi;x+=bounded?10:focus?1:Math.max(1,Math.ceil(extent/5))){out.push(`<text x="${X(x)}" y="${H-B+22}" text-anchor="middle" fill="#607487" font-size="12">${x}</text>`);}
  f.lines.forEach(([x,text])=>out.push(`<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H-B}" stroke="#8a6870" stroke-dasharray="5 4"/><text x="${X(x)}" y="${T-(state.family==='donchian'&&x===0?27:10)}" text-anchor="middle" font-size="11" fill="#8a6870">${text}</text>`));
  lines.forEach(s=>out.push(`<path d="${s.y.map((y,i)=>(i?'L':'M')+X(grid[i]).toFixed(2)+','+Y(y).toFixed(2)).join('')}" fill="none" stroke="${colors[s.n]}" stroke-width="2.6"/>`));
  out.push(`<text x="${L}" y="14" fill="#607487" font-size="11">Density</text><text x="${(W+L-R)/2}" y="${H-6}" text-anchor="middle" fill="#607487" font-size="12">${f.axis}</text></svg>`);
  return out.join('');
 }

 function distribution(){
  const pairs=state.scope==='pooled'?P.pairs:[state.pair];
  const samples=state.windows.map(n=>({n,z:pairs.flatMap(p=>cache[p][n].z)}));
  $('pd-density-title').textContent=family().indicator+' distribution · '+(state.scope==='pooled'?'all nine pairs pooled':`${state.pair} / USD`);
  $('pd-density-note').textContent=`Gaussian KDE · bandwidth ${family().bandwidth}${family().indicator==='RSI'?' RSI points; reflected at 0 and 100':['ma','tsmom'].includes(state.family)?' percentage points; unbounded support':'; unbounded support'}. Same smoothing for every version and date window. Percentiles and threshold counts use raw observations.`;
  if(state.family==='donchian'&&state.range==='focus')$('pd-density-note').textContent+=' Chart shows −3 to +3; all observations remain in the KDE and tables. Outside this view: '+samples.map(s=>label(s.n)+' '+percent(100*s.z.filter(v=>Math.abs(v)>3).length/s.z.length)).join('; ')+'. Choose Full observed range to inspect the tails.';
  $('pd-frequency-note').textContent=state.family==='donchian'?'Breakout days compare today’s price with the prior n daily closes. Below −1 is a short breakout; above +1 is a long breakout; 0 is the moving channel midpoint. Breakout-day counts are not new trade counts: an existing position can persist inside the channel. The normalized coordinate is for diagnostics; the original price comparisons determine trades.':state.family==='tsmom'?'Positive trailing log return targets long foreign currency; negative targets short; exactly zero targets flat. The 21/63 days are lookbacks, not holding periods. Consecutive same-sign days maintain a position; actual entries count executed openings / reversals.':state.family==='ma'?'Positive MA Gap targets long foreign currency; negative targets short; exactly zero targets flat. Consecutive same-sign days maintain one position. Actual entries count executed openings / reversals. Percentiles are in percent units.':state.family==='rsi_mom'?'Target days count RSI < 40 (short) or RSI > 60 (long). RSI from 40 to 60, inclusive, targets flat. Consecutive target days can belong to one position; actual entries are counted at execution dates.':state.family==='rsi'?'Outside days count RSI < 30 or RSI > 70; they are not entry signals. The strategy enters on a return across 30 / 70 toward 50. Entries count actual openings / reversals at execution dates.':'Outside days count Z ≤ −2 or Z ≥ +2. Consecutive outside days can belong to one position. Entries count actual openings / reversals at execution dates; they are not threshold-day counts.';
  $('pd-density').innerHTML=densitySVG(samples);
  const sorted=samples.map(s=>s.z.slice().sort((a,b)=>a-b));
  $('pd-percentiles').innerHTML=table(['Percentile',...versions()],[.01,.05,.1,.5,.9,.95,.99].map(p=>['P'+Math.round(p*100),...sorted.map(s=>number(quantile(s,p)))]));
  const rows=[
   ['Indicator observations',...samples.map(s=>number(s.z.length,0))],
   [family().lowLabel,...samples.map(s=>percent(100*s.z.filter(family().low).length/s.z.length))],
   [family().highLabel,...samples.map(s=>percent(100*s.z.filter(family().high).length/s.z.length))],
   [['rsi_mom','ma','tsmom'].includes(state.family)?'Either directional target':state.family==='donchian'?'Either breakout condition':'Either extreme',...samples.map(s=>percent(100*s.z.filter(z=>family().low(z)||family().high(z)).length/s.z.length))],
   ['Actual entries in window',...state.windows.map(n=>number(pairs.reduce((a,p)=>a+cache[p][n].entries,0),0))],
   ['Completed entry-cohort trades',...state.windows.map(n=>number(pairs.reduce((a,p)=>a+cache[p][n].r.quality.closed,0),0))]
  ];
  $('pd-frequency').innerHTML=table([state.scope==='pooled'?'All pairs pooled / summed':'Selected pair',...versions()],rows);
 }

 function results(){
  const selected=state.windows.map(n=>cache[state.pair][n]);
  const records=selected.map((s,i)=>({...s.r,color:colors[state.windows[i]],label:label(state.windows[i])}));
  currentView={records,title:state.pair+' / USD · '+family().label+' parameter comparison',D,
   state:{dataset:'parameters',display:'pnl',normalization:'window',from:state.from,to:state.to}};
  $('pd-pnl').innerHTML=A.renderSVG(currentView).replaceAll('fxplotclip','pdplotclip').replaceAll('fxddclip','pdddclip').replaceAll('id="crosshair"','id="pd-crosshair"');
  $('pd-tooltip').hidden=true;
  $('pd-result-caption').textContent=`${state.pair} / USD · original BASE accounts · PnL reset to zero at the window start · hover for dated PnL / drawdown.`;
  const rows=[
   ['Window PnL',...selected.map(s=>money(s.r.pnl))],
   ['Simple return',...selected.map(s=>percent(s.r.ret))],
   ['Max drawdown / initial capital',...selected.map(s=>number(s.r.dd)+' pp')],
   ['Max drawdown / peak equity',...selected.map(s=>percent(s.r.ddEquity))],
   ['Entries in window',...selected.map(s=>number(s.entries,0))],
   ['Completed entry-cohort trades',...selected.map(s=>number(s.r.quality.closed,0)+(s.r.quality.closed<20?' · small cohort':''))],
   ['Win rate',...selected.map(s=>percent(s.r.quality.win))],
   ['Average winning trade',...selected.map(s=>percent(s.r.quality.avgWin))],
   ['Average losing trade · absolute',...selected.map(s=>percent(s.r.quality.avgLoss))],
   ['Payoff',...selected.map(s=>number(s.r.quality.payoff,3))],
   ['Average holding days · completed trades',...selected.map(s=>number(s.avgHolding,1))],
   ['Exposure days',...selected.map(s=>percent(s.exposure))],
   ['Open / censored entries',...selected.map(s=>number(s.r.quality.censored,0))],
   ['Carry-in positions',...selected.map(s=>number(s.r.quality.carry,0))]
  ];
  $('pd-trading-metrics').innerHTML=table(['Metric',...versions()],rows);
  const annual=records.map(r=>A.annualReturns(r.a,D,A.toDay(state.from),A.toDay(state.to)));
  const years=[...new Set(annual.flatMap(x=>Object.keys(x)))].sort();
  $('pd-annual').innerHTML=table(['Year',...versions()],years.map(y=>[y,...annual.map(a=>percent(a[y]))]));
 }

 function render(recompute=true){
  const start=A.toDay(state.from),end=A.toDay(state.to);
  let error='';
  if(!state.from||!state.to||!Number.isFinite(start)||!Number.isFinite(end)||start>end)error='Choose a valid date range: From must not be later than Through.';
  else if(state.from<P.start||state.to>P.end)error=`Use dates within main-history coverage: ${P.start}–${P.end}.`;
  else if(!state.windows.length)error='Select at least one parameter set.';
  $('pd-error').textContent=error;$('pd-error').hidden=!error;$('pd-content').hidden=!!error;
  if(error){currentView=null;return;}
  if(recompute){cache={};for(const p of P.pairs){cache[p]={};for(const n of spec().windows)cache[p][n]=calculate(p,n,start,end);}}
  if(!cache[P.pairs[0]][spec().windows[0]]){$('pd-error').textContent='No FX observations fall within these dates.';$('pd-error').hidden=false;$('pd-content').hidden=true;currentView=null;return;}
  overview();
  $('pd-detail-title').textContent=state.pair+' / USD';
  $('pd-legend').innerHTML=state.windows.map(n=>`<span class="legend-item"><span class="swatch" style="background:${colors[n]}"></span>${label(n)} · ${state.family==='ma'?parameterName(n)+'-day SMAs':n+'-day lookback'}</span>`).join('');
  distribution();results();
  $('pd-distribution').hidden=state.tab!=='distribution';$('pd-results').hidden=state.tab!=='results';
  ['distribution','results'].forEach(t=>$('pd-tab-'+t).setAttribute('aria-pressed',String(t===state.tab)));
  document.querySelectorAll('#pd-presets button').forEach(b=>b.setAttribute('aria-pressed',String(state.from===b.dataset.from&&state.to===b.dataset.to)));
 }

 function showModule(){
  const diagnostic=location.hash==='#parameters',trades=location.hash==='#trades',comparison=location.hash==='#comparison';
  $('parameter-diagnostics').hidden=!diagnostic;$('explorer-module').hidden=diagnostic||trades||comparison;
  $('trade-distribution').hidden=!trades;
  $('pair-comparison').hidden=!comparison;
  $('module-comparison').setAttribute('aria-pressed',String(comparison));
  $('module-trades').setAttribute('aria-pressed',String(trades));
  $('module-parameters').setAttribute('aria-pressed',String(diagnostic));$('module-explorer').setAttribute('aria-pressed',String(!diagnostic&&!trades&&!comparison));
 }
 $('module-parameters').addEventListener('click',()=>{location.hash='parameters';});
 $('module-explorer').addEventListener('click',()=>{location.hash='explorer';});
 window.addEventListener('hashchange',showModule);
 $('pd-pair').innerHTML=P.pairs.map(p=>`<option value="${p}">${p} / USD</option>`).join('');$('pd-pair').value=state.pair;
 $('pd-pair').addEventListener('change',e=>{state.pair=e.target.value;render(false);});
 ['from','to'].forEach(k=>{const e=$('pd-'+k);e.min=P.start;e.max=P.end;e.value=state[k];e.addEventListener('change',()=>{state[k]=e.value;render();});});
 $('pd-metric').addEventListener('change',e=>{state.metric=e.target.value;render(false);});
 $('pd-range').addEventListener('change',e=>{state.range=e.target.value;render(false);});
 $('pd-scope').addEventListener('change',e=>{state.scope=e.target.value;render(false);});
 $('pd-versions').addEventListener('change',()=>{state.windows=[...document.querySelectorAll('#pd-versions input:checked')].map(x=>x.value);render(false);});
 $('pd-overview').addEventListener('click',e=>{const b=e.target.closest('[data-pair]');if(!b)return;state.pair=b.dataset.pair;$('pd-pair').value=state.pair;render(false);});
 ['distribution','results'].forEach(t=>$('pd-tab-'+t).addEventListener('click',()=>{state.tab=t;render(false);}));
 const presets=[['Full',P.start,P.end],['2005–09','2005-01-03','2009-12-31'],['2010–14','2010-01-01','2014-12-31'],['2015–19','2015-01-01','2019-12-31'],['2020–24','2020-01-01','2024-12-31'],['2025–26','2025-01-01',P.end]];
 $('pd-presets').innerHTML=presets.map(([s,f,t])=>`<button type="button" data-from="${f}" data-to="${t}" aria-pressed="false">${s}</button>`).join('');
 $('pd-presets').addEventListener('click',e=>{const b=e.target.closest('[data-from]');if(!b)return;state.from=b.dataset.from;state.to=b.dataset.to;$('pd-from').value=state.from;$('pd-to').value=state.to;render();});
 $('pd-pnl-wrap').addEventListener('pointermove',e=>{
  const svg=$('pd-pnl').querySelector('svg');if(!svg||!currentView)return;
  const box=svg.getBoundingClientRect(),x=(e.clientX-box.left)/box.width*1200;
  const lo=Number(svg.dataset.xmin),hi=Number(svg.dataset.xmax),left=Number(svg.dataset.left),right=Number(svg.dataset.right);
  if(x<left||x>right){$('pd-tooltip').hidden=true;return;}
  const day=lo+(hi-lo)*(x-left)/(right-left),dates=currentView.records[0].points.map(p=>p[0]);
  const idx=Math.min(dates.length-1,A.lowerBound(dates,day)),d=dates[idx];
  $('pd-tooltip').innerHTML='<div class="tooltip-date">'+A.dateString(d)+'</div>'+currentView.records.map(r=>{const p=r.points[idx];return `<div class="tooltip-row"><span><i class="dot" style="background:${r.color}"></i>${r.label}</span><span>${money(p[1]-100000)} · ${number(p[2])} pp DD</span></div>`;}).join('');
  const parent=$('pd-pnl-wrap').getBoundingClientRect();$('pd-tooltip').style.left=Math.max(0,Math.min(e.clientX-parent.left+14,parent.width-300))+'px';$('pd-tooltip').style.top=Math.max(0,e.clientY-parent.top-85)+'px';$('pd-tooltip').hidden=false;
 });
 $('pd-pnl-wrap').addEventListener('pointerleave',()=>{$('pd-tooltip').hidden=true;});
 // Expose pure computations for independent numerical checks without UI mutation.
 window.FXParameterDiagnostics={calculate,quantile,kde,state,data:P,get view(){return currentView;}};
 function updateFamilyControls(){
  $('pd-family').value=state.family;
  $('pd-family-badge').textContent=family().label;
  $('pd-range-control').hidden=state.family!=='donchian';
  $('pd-versions').innerHTML=spec().windows.map(n=>`<label class="chip"><input type="checkbox" value="${n}" checked><span class="pd-dot" style="background:${colors[n]}"></span>${label(n)}</label>`).join('');
  $('pd-metric').querySelector('option[value="tails"]').textContent=family().tailLabel;
  $('pd-rule-method').innerHTML=state.family==='donchian'
   ? '<p><b>Prior channel:</b> H = highest and L = lowest of the previous n daily closes, excluding today; n = 10 or 20. These are close-based bands, not intraday high/low bands. The midpoint M = (H + L) / 2 changes each day.</p><p><b>Chart coordinate:</b> (P − M) / ((H − L) / 2). −1 is the lower band; +1 is the upper band; 0 is the midpoint. Values beyond ±1 are retained. This coordinate describes the original rule; it does not introduce a new signal.</p><p><b>Entry:</b> P &gt; H targets long foreign currency; P &lt; L targets short. Equal to a band does not trigger a breakout. <b>Exit:</b> a long closes when P &lt; M; a short closes when P &gt; M. Equality at the midpoint maintains the position. An opposite breakout takes priority and reverses it. Inside the channel, positions otherwise persist; a flat account waits for a breakout. No fixed holding limit or separate stop-loss. Execution follows the delayed-close convention below.</p>'
   : state.family==='tsmom'
   ? '<p><b>Momentum:</b> ln(P today / P n trading days ago), with n = 21 or 63. The distribution and percentiles show 100 × this log return. Prices are USD per unit of foreign currency. The lookback is not a holding period.</p><p><b>Target position:</b> positive momentum targets long foreign currency; negative targets short; exactly zero targets flat. The position stays open while the sign persists, and reverses when the sign changes. No fixed holding limit or separate stop-loss. Signals use only prices available at that close; execution follows the delayed-close convention below.</p>'
   : state.family==='ma'
   ? '<p><b>MA Gap:</b> short SMA / long SMA − 1; the chart and percentiles show this value × 100 in percent units. Compare 10/30-day and 20/60-day SMAs. Each SMA includes today’s price and uses available daily quote observations; pre-2005 prices provide warm-up.</p><p><b>Target position:</b> positive gap targets long foreign currency against USD; negative gap targets short; exactly zero targets flat. An opposite sign reverses the position; the same sign maintains it. No fixed holding limit or separate stop-loss. Execution follows the delayed-close convention below.</p>'
   : state.family==='bb_bo'
   ? '<p><b>Z-score:</b> (current price − n-day SMA) / n-day sample standard deviation, including today’s price; n = 10 or 20. Identical indicator values and smoothing to Bollinger Mean Reversion for the same pair and dates.</p><p><b>Entry:</b> Z ≥ +2 targets long foreign currency; Z ≤ −2 targets short foreign currency. No first-crossing requirement. Remaining beyond the same threshold maintains the position.</p><p><b>Exit / reversal:</b> a long closes at Z ≤ 0; a short closes at Z ≥ 0. An opposite outer-band signal takes priority and reverses the position. No fixed holding limit or separate stop-loss. Execution follows the delayed-close convention below.</p>'
   : state.family==='rsi_mom'
   ? '<p><b>Wilder RSI:</b> the same daily indicator and warm-up as RSI Mean Reversion, with lookbacks 7, 14 or 20. Distribution curves and percentiles are identical for the same pair, lookback and dates; the trading thresholds and account paths differ.</p><p><b>Target position:</b> RSI &gt; 60 targets long foreign currency; RSI &lt; 40 targets short foreign currency; 40 ≤ RSI ≤ 60 targets flat. The target is evaluated each day; no threshold crossing is required. Remaining in the same zone maintains the position, rather than opening another trade.</p><p><b>Exit / reversal:</b> returning to 40–60 targets an exit. Moving directly into the opposite directional zone targets a reversal. Execution follows the same delayed-close convention below. No fixed holding limit or separate stop-loss.</p>'
   : state.family==='rsi'
   ? '<p><b>Wilder RSI:</b> 100 − 100 / (1 + smoothed gains / smoothed losses), using daily price changes and recursive smoothing weight 1/n. Lookbacks: 7, 14, 20 trading days. The supplied implementation seeds gains/losses from the first price change; all variants are warmed on pre-2005 data.</p><p><b>Entry:</b> long when yesterday’s RSI &lt; 30 and today’s RSI is ≥ 30 but &lt; 50; short when yesterday’s RSI &gt; 70 and today’s RSI is ≤ 70 but &gt; 50. Merely remaining below 30 or above 70 does not open a trade.</p><p><b>Exit:</b> close a long at RSI ≥ 50; close a short at RSI ≤ 50. Exits are processed before new entries; an opposite qualifying re-entry may reverse the position. No fixed holding limit or separate stop-loss. Only lookback changes; thresholds remain 30 / 70 / 50.</p>'
   : '<p><b>Z-score:</b> (current price − n-day SMA) / n-day sample standard deviation. The n-day window includes today’s price; n = 10 or 20. Prices are USD per unit of foreign currency.</p><p><b>Entry:</b> Z ≤ −2 targets long foreign currency; Z ≥ +2 targets short foreign currency. <b>Exit:</b> a long closes when Z ≥ 0; a short closes when Z ≤ 0. An opposite outer-band signal takes priority and can reverse the position. No fixed holding limit or separate stop-loss.</p>';
 }
 $('pd-family').addEventListener('change',e=>{
  state.family=e.target.value;state.windows=spec().windows.slice();
  updateFamilyControls();render();
 });
 updateFamilyControls();render();showModule();
})();
