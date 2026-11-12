/* Historical period attribution inside Pair Comparison; unchanged daily accounts. */
(function(){
 'use strict';
 const DAY=86400000,CAPITAL=100000,EPS=1e-7;
 const iso=d=>new Date(d*DAY).toISOString().slice(0,10);
 function lower(a,x){let l=0,r=a.length;while(l<r){let m=(l+r)>>1;if(a[m]<x)l=m+1;else r=m;}return l;}
 function upper(a,x){let l=0,r=a.length;while(l<r){let m=(l+r)>>1;if(a[m]<=x)l=m+1;else r=m;}return l;}
 function heldFromTrades(a,dates){
  const held=Array(dates.length).fill(0);
  for(const t of a.trades){const l=upper(dates,t[0]),r=t[4]?upper(dates,t[1]):dates.length;
   for(let i=l;i<r;i++){if(held[i]!==0)throw Error('Overlapping ledger intervals');held[i]=t[5];}
  }return held;
 }
 function periodBounds(d,group){const dt=new Date(d*DAY),y=dt.getUTCFullYear(),m=dt.getUTCMonth(),step=group==='quarter'?3:1,base=Math.floor(m/step)*step;
  return {key:group==='quarter'?`${y}-Q${base/3+1}`:`${y}-${String(base+1).padStart(2,'0')}`,year:y,slot:base/step,
   start:Date.UTC(y,base,1)/DAY,end:Date.UTC(y,base+step,1)/DAY-1};
 }
 function classification(net,active){return active===0?'No exposure':net>EPS?'Gain':net< -EPS?'Loss':'Active flat';}
 function interval(a,dates,held,start,end){
  const lo=lower(dates,start),hi=upper(dates,end);let net=0,gains=0,losses=0,active=0,long=0,short=0,run=0,longest=0,streakStart=null,longestStart=null,longestEnd=null;
  for(let i=lo;i<hi;i++){
   const pnl=CAPITAL*a.daily_returns[i];net+=pnl;if(pnl>0)gains+=pnl;else losses-=pnl;
   if(held[i]!==0){active++;if(held[i]>0)long++;else short++;run=0;}else{if(!run)streakStart=dates[i];run++;if(run>longest){longest=run;longestStart=streakStart;longestEnd=dates[i];}}
  }
  const movement=gains+losses;
  return {start,end,lo,hi,n:hi-lo,first:dates[lo],last:dates[hi-1],net,gains,losses,active,long,short,flat:hi-lo-active,
   exposure:hi>lo?100*active/(hi-lo):null,
   longExposure:hi>lo?100*long/(hi-lo):null,shortExposure:hi>lo?100*short/(hi-lo):null,flatExposure:hi>lo?100*(hi-lo-active)/(hi-lo):null,state:classification(net,active),
   offset:movement>EPS?100*2*Math.min(gains,losses)/movement:null,
   entries:a.trades.filter(t=>t[0]>=start&&t[0]<=end).length,
   exits:a.trades.filter(t=>t[4]&&t[1]>=start&&t[1]<=end&&t[3]>0).length,
   carry:a.trades.filter(t=>t[0]<start&&(!t[4]||t[1]>=start)).length,
   longest,longestStart,longestEnd};
 }
 function periods(a,dates,held,start,end,group){
  const lo=lower(dates,start),hi=upper(dates,end),bounds=new Map();
  for(let i=lo;i<hi;i++){const b=periodBounds(dates[i],group);if(!bounds.has(b.key))bounds.set(b.key,b);}
  return [...bounds.values()].map(b=>{
   const begin=Math.max(start,b.start),finish=Math.min(end,b.end),v=interval(a,dates,held,begin,finish);
   // Only weekends are adjusted. Missing weekday boundary quotes are treated
   // conservatively as incomplete, rather than inventing holiday observations.
   const weekday=(d,direction)=>{while([0,6].includes(new Date(d*DAY).getUTCDay()))d+=direction;return d;};
   const requiredFirst=weekday(b.start,1),requiredLast=weekday(b.end,-1);
   const partial=Math.max(begin,dates[0])>requiredFirst||Math.min(finish,dates[dates.length-1])<requiredLast;
   return {...b,...v,calendarStart:b.start,calendarEnd:b.end,partial};
  });
 }
 function concentration(ps){
  const complete=ps.filter(p=>!p.partial),winners=complete.filter(p=>p.net>EPS).sort((a,b)=>b.net-a.net);
  const gross=winners.reduce((s,p)=>s+p.net,0),net=complete.reduce((s,p)=>s+p.net,0);
  return {complete:complete.length,partial:ps.length-complete.length,gross,net,
   top:[1,3].map(k=>{const chosen=winners.slice(0,k),removed=chosen.reduce((s,p)=>s+p.net,0);return {k,n:chosen.length,keys:chosen.map(p=>p.key),removed,share:gross>EPS?removed/gross*100:null,remaining:net-removed};})};
 }
 window.FXPeriodMath={heldFromTrades,periodBounds,interval,periods,concentration,classification,EPS};
 const A=window.FXResearch;if(!A||!A.ready||!document.getElementById('pt-rule'))return;
 const $=id=>document.getElementById(id),P=JSON.parse($('pair-comparison-data').textContent),D=A.data,e=A.escapeHTML;
 const accounts=new Map(D.accounts.filter(a=>a.ds==='main'&&a.variant==='BASE').map(a=>[a.pair+'|'+a.rule,a]));
 const state={pair:'AUD',rule:'BB_MR_20',scope:'recent',group:'month',first:null,last:null};
 const heldCache=new Map();let current=null;
 const num=(x,d=1)=>Number.isFinite(x)?x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
 const usd=x=>(x<0?'−':'')+'$'+num(Math.abs(x),0);
 const table=(h,rows)=>'<table><thead><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
 const cssClass=s=>({'Gain':'gain','Loss':'loss','No exposure':'idle','Active flat':'flat'}[s]);
 function exposureDetail(v){return `<div class="pt-direction"><span><b>Long</b> ${num(v.longExposure)}%</span><span><b>Short</b> ${num(v.shortExposure)}%</span><span><b>Flat</b> ${num(v.flatExposure)}%</span></div>`;}
 function source(pair=state.pair,rule=state.rule){const a=accounts.get(pair+'|'+rule),dates=D.calendars[a.cal];if(!heldCache.has(a.id))heldCache.set(a.id,heldFromTrades(a,dates));return {a,dates,held:heldCache.get(a.id)};}
 function windowBounds(){return state.scope==='all'?[A.toDay(P.start),A.toDay(P.end)]:P.windows[state.scope].map(A.toDay);}
 function curve(){
  const {a,dates,ps,start,end}=current,W=1000,H=255,L=70,R=15,T=22,B=35,width=W-L-R,height=H-T-B;
  const lo=lower(dates,start),hi=upper(dates,end),anchor=lo?dates[lo-1]:a.anchor;let cumulative=0;
  const points=[[anchor,0]];for(let i=lo;i<hi;i++){cumulative+=CAPITAL*a.daily_returns[i];points.push([dates[i],cumulative]);}
  let min=Math.min(0,...points.map(p=>p[1])),max=Math.max(0,...points.map(p=>p[1]));if(max-min<1){min-=1;max+=1;}const padding=(max-min)*.08;min-=padding;max+=padding;
  const X=x=>L+(x-anchor)/(dates[hi-1]-anchor||1)*width,Y=y=>T+(max-y)/(max-min)*height;
  let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cumulative PnL with historical period states"><rect width="100%" height="100%" fill="white"/>`;
  for(const p of ps){let left=p.lo?dates[p.lo-1]:anchor;left=Math.max(anchor,left);out+=`<rect x="${X(left)}" y="${T}" width="${Math.max(1,X(p.last)-X(left))}" height="${height}" class="pt-band ${cssClass(p.state)}"/>`;}
  const selected=ps.filter(p=>p.key>=state.first&&p.key<=state.last);if(selected.length){const p=selected[0],q=selected[selected.length-1],x=X(p.lo?dates[p.lo-1]:anchor);out+=`<rect x="${x}" y="${T}" width="${Math.max(1,X(q.last)-x)}" height="${height}" fill="none" stroke="#274c6b" stroke-width="2"/>`;}
  for(let k=0;k<4;k++){const v=min+(max-min)*k/3;out+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#dce5ec"/><text x="${L-7}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="#607487">${usd(v)}</text>`;}
  out+=`<path d="${points.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(2)+','+Y(p[1]).toFixed(2)).join(' ')}" fill="none" stroke="#244b6b" stroke-width="2"/>`;
  const startY=new Date(start*DAY).getUTCFullYear(),endY=new Date(end*DAY).getUTCFullYear(),step=endY-startY>10?3:1;
  for(let year=startY;year<=endY;year+=step){const d=Math.max(start,Date.UTC(year,0,1)/DAY);out+=`<text x="${X(d)}" y="${H-10}" font-size="11" fill="#607487">${year}</text>`;}
  return out+'</svg>';
 }
 function calendar(){
  const slots=state.group==='month'?12:4,labels=slots===12?['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']:['Q1','Q2','Q3','Q4'];
  const rows=[...new Set(current.ps.map(p=>p.year))].map(y=>[y,...Array.from({length:slots},(_,slot)=>{const p=current.ps.find(p=>p.year===y&&p.slot===slot);if(!p)return '<span class="pt-missing">—</span>';
   const selected=p.key>=state.first&&p.key<=state.last;return `<button type="button" class="pt-cell ${cssClass(p.state)} ${selected?'pt-selected':''}" data-period="${p.key}" aria-pressed="${selected}" title="${p.key}${p.partial?' (partial)':''}: ${p.state}; PnL ${usd(p.net)}; total exposure ${num(p.exposure)}%; long ${num(p.longExposure)}%; short ${num(p.shortExposure)}%; flat ${num(p.flatExposure)}%; ${p.entries} entries"><b>${usd(p.net)}${p.partial?'*':''}</b><small>${p.state==='No exposure'?'No exposure':num(p.exposure,0)+'% total'}</small><small>L ${num(p.longExposure,0)}% · S ${num(p.shortExposure,0)}%</small></button>`;})]);
  return table(['Year',...labels],rows);
 }
 function inspect(){
  const ps=current.ps,a=ps.find(p=>p.key===state.first),b=ps.find(p=>p.key===state.last);
  if(!a||!b||a.start>b.end){$('pt-inspection').innerHTML='<p>Choose a start period no later than the end period.</p>';$('pt-calendar').innerHTML=calendar();$('pt-curve').innerHTML=curve();return;}
  const v=interval(current.a,current.dates,current.held,a.start,b.end);
  const message=v.active===0?'No exposure: the strategy held no position over any return interval in this range. This does not prove that no raw indicator signal occurred.':`Positions were held on ${v.active} of ${v.n} quote intervals: ${v.long} long and ${v.short} short; ${v.flat} were flat. Positive daily PnL totaled ${usd(v.gains)} and negative daily PnL totaled ${usd(v.losses)}; the difference is ${usd(v.net)}.`;
  $('pt-inspection').innerHTML=`<div class="pt-inspection-title"><h4>${e(state.first)}${state.last!==state.first?' → '+e(state.last):''}</h4><span class="pt-state ${cssClass(v.state)}">${v.state}</span></div><p>${message}</p>`+
   table(['Net PnL','Total exposure · direction split','Daily PnL offset','Entries / closed exits','Carry-in trades','Longest no-exposure run'],[[usd(v.net),`${v.active}/${v.n} (${num(v.exposure)}%)${exposureDetail(v)}`,v.offset===null?'—':num(v.offset)+'%',`${v.entries} / ${v.exits}`,v.carry,v.longest?`${v.longest} quote days<small>${iso(v.longestStart)} → ${iso(v.longestEnd)}</small>`:'0']])+`<p class="pd-note">Actual return dates: ${iso(v.first)}–${iso(v.last)}. Daily PnL offset = 2 × min(positive daily PnL, absolute negative daily PnL) / their sum. Near 100% means gains and losses nearly cancel; it is not a flat-market classification. No exposure is established from positions, not zero price returns.</p>`;
  $('pt-calendar').innerHTML=calendar();$('pt-curve').innerHTML=curve();
 }
 function render(){
  const src=source(),[start,end]=windowBounds(),ps=periods(src.a,src.dates,src.held,start,end,state.group);current={...src,ps,start,end};
  if(!ps.some(p=>p.key===state.first)||!ps.some(p=>p.key===state.last)){state.first=ps[0].key;state.last=ps[ps.length-1].key;}
  for(const [id,key] of [['pt-first','first'],['pt-last','last']]){$(id).innerHTML=ps.map(p=>`<option value="${p.key}">${p.key}${p.partial?' *':''}</option>`).join('');$(id).value=state[key];}
  const total=interval(src.a,src.dates,src.held,start,end),c=concentration(ps),counts=['Gain','Loss','No exposure','Active flat'].map(s=>`${ps.filter(p=>p.state===s).length} ${s.toLowerCase()}`),unit=state.group==='month'?'months':'quarters';
  $('pt-title').textContent=state.pair+' / USD · '+D.rules[state.rule].label;
  $('pt-rule-text').textContent=D.rules[state.rule].rule;
  $('pt-summary').innerHTML=`<div><small>Window PnL · gross spot</small><strong>${usd(total.net)}</strong></div><div><small>Total exposure · held return intervals</small><strong>${num(total.exposure)}%</strong>${exposureDetail(total)}</div><div><small>${ps.length} ${unit} · including partial</small><span>${counts.join(' · ')}</span></div>`;
  $('pt-concentration').innerHTML=table(['Best complete '+unit+' removed','Dates','Positive-period PnL share','PnL removed','Remaining complete-period PnL'],c.top.map(x=>[x.n?`Top ${x.n}${x.n<x.k?' (only '+x.n+' positive available)':''}`:'No positive periods',x.keys.join(', ')||'—',x.share===null?'—':num(x.share)+'%',usd(x.removed),usd(x.remaining)]));
  $('pt-concentration-note').textContent=`${c.complete} complete ${unit}; ${c.partial} partial periods excluded from this concentration check. Complete-period PnL: ${usd(c.net)}. Denominator = sum of positive period net PnL (${usd(c.gross)}), not winning-trade profit or total net PnL. Removing periods is an arithmetic sensitivity check; no strategy is rerun and losing periods remain included.`;
  inspect();
 }
 $('pt-rule').innerHTML=Object.entries(P.families).map(([family,[category,rules]])=>`<optgroup label="${category} · ${e(family)}">${rules.map(r=>`<option value="${r}">${e(D.rules[r].label)}</option>`).join('')}</optgroup>`).join('');$('pt-rule').value=state.rule;
 for(const key of ['rule','scope','group'])$('pt-'+key).addEventListener('change',ev=>{state[key]=ev.target.value;if(key!=='rule')state.first=state.last=null;render();});
 for(const key of ['first','last'])$('pt-'+key).addEventListener('change',ev=>{state[key]=ev.target.value;inspect();});
 $('pt-calendar').addEventListener('click',ev=>{const b=ev.target.closest('[data-period]');if(!b)return;state.first=state.last=b.dataset.period;$('pt-first').value=state.first;$('pt-last').value=state.last;inspect();});
 $('pt-reset-range').addEventListener('click',()=>{state.first=current.ps[0].key;state.last=current.ps.slice(-1)[0].key;$('pt-first').value=state.first;$('pt-last').value=state.last;inspect();});
 $('pt-export').addEventListener('click',()=>{
  const rows=[['pair','rule','category','group','period','first_return_date','last_return_date','partial','state_known_after_last_return','net_pnl_usd','positive_daily_pnl_usd','negative_daily_pnl_abs_usd','held_intervals','quote_intervals','exposure_pct','entries','closed_exits','carry_in','daily_pnl_offset_pct','long_held_intervals','short_held_intervals','flat_intervals','long_exposure_pct','short_exposure_pct','flat_pct']],[start,end]=windowBounds();
  for(const p of P.pairs)for(const [,[category,rules]] of Object.entries(P.families))for(const rule of rules){const {a,dates,held}=source(p,rule);for(const b of periods(a,dates,held,start,end,state.group))rows.push([p,rule,category,state.group,b.key,iso(b.first),iso(b.last),b.partial,b.state,b.net,b.gains,b.losses,b.active,b.n,b.exposure,b.entries,b.exits,b.carry,b.offset,b.long,b.short,b.flat,b.longExposure,b.shortExposure,b.flatExposure]);}
  const csv=rows.map(r=>r.map(x=>'"'+String(x??'').replace(/"/g,'""')+'"').join(',')).join('\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),link=document.createElement('a');link.href=url;link.download=`FX_Historical_Periods_${state.scope}_${state.group}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 window.FXPairTimeline={setPair(p){state.pair=p;render();},setRule(rule){state.rule=rule;$('pt-rule').value=rule;render();}};
})();
