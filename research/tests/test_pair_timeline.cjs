/* Run with Node after rebuilding the dashboard. No browser automation. */
const fs=require('fs'),path=require('path'),vm=require('vm'),zlib=require('zlib'),assert=require('assert');
const root=path.resolve(__dirname,'..'),context={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'src/dashboard/pair_timeline.js'),'utf8'),context);
const M=context.window.FXPeriodMath,day=s=>Date.parse(s+'T00:00:00Z')/86400000,close=(a,b,t=1e-7)=>assert(Math.abs(a-b)<t,`${a} differs from ${b}`);
// A trade crosses a month boundary. Assign PnL by return date, never exit month.
let dates=['2024-01-30','2024-01-31','2024-02-01','2024-02-02'].map(day);
let a={daily_returns:[0,.01,-.01,0],trades:[[dates[0],dates[3],0,3,1,1]]};
let held=M.heldFromTrades(a,dates);assert.deepStrictEqual(Array.from(held),[0,1,1,1]);
let ps=M.periods(a,dates,held,dates[0],dates[3],'month');close(ps[0].net,1000);close(ps[1].net,-1000);assert.equal(ps[1].carry,1);
let whole=M.interval(a,dates,held,dates[0],dates[3]);assert.equal(whole.state,'Active flat');close(whole.offset,100);assert.equal(whole.active,3);close(whole.exposure,75);
// Same-close entry earns nothing. Reversal closes one return interval, then flips.
a={trades:[[1,3,0,2,1,1],[3,-1,0,2,0,-1]],daily_returns:[0,0,0,0,0]};
assert.deepStrictEqual(Array.from(M.heldFromTrades(a,[1,2,3,4,5])),[0,1,1,-1,-1]);
assert.equal(M.interval(a,[1,2,3,4,5],[0,1,1,-1,-1],1,5).state,'Active flat');
assert.equal(M.interval(a,[1,2,3,4,5],[0,0,0,0,0],1,5).state,'No exposure');
// Direction split uses all return intervals, including unchanged-price days.
let split=M.interval(a,[1,2,3,4,5],[0,1,1,-1,-1],1,5);
assert.equal(split.long,2);assert.equal(split.short,2);assert.equal(split.flat,1);
close(split.longExposure,40);close(split.shortExposure,40);close(split.flatExposure,20);
let exitOnly=M.interval(a,[1,2,3,4,5],M.heldFromTrades(a,[1,2,3,4,5]),3,3);
assert.equal(exitOnly.long,1);assert.equal(exitOnly.short,0);
let nextDay=M.interval(a,[1,2,3,4,5],M.heldFromTrades(a,[1,2,3,4,5]),4,4);
assert.equal(nextDay.long,0);assert.equal(nextDay.short,1);
// Month ends on a weekend: May 29 2026 completes May, but not Q2.
dates=['2026-05-01','2026-05-29'].map(day);a={trades:[],daily_returns:[0,0]};
assert.equal(M.periods(a,dates,[0,0],dates[0],dates[1],'month')[0].partial,false);
assert.equal(M.periods(a,dates,[0,0],dates[0],dates[1],'quarter')[0].partial,true);
// Do not include a partial high-profit block in complete-period concentration.
let c=M.concentration([{key:'a',net:100,partial:false},{key:'b',net:-200,partial:false},{key:'c',net:10000,partial:true}]);
close(c.net,-100);close(c.top[0].share,100);close(c.top[0].remaining,-200);assert.equal(c.partial,1);
assert.equal(M.concentration([{key:'a',net:-1,partial:false}]).top[0].share,null);
const html=fs.readFileSync(path.join(root,'FX_PnL_Explorer_Simple.html'),'utf8');
const P=JSON.parse(html.match(/<script id="pair-comparison-data" type="application\/json">(.*?)<\/script>/s)[1]);
const D=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(root,'reproduced/simple_interest/explorer_payload.json.gz'))));
let accounts=0,buckets=0,maxError=0;
const out=[];
for(const pair of P.pairs)for(const rule of P.ruleOrder){
 a=D.accounts.find(a=>a.ds==='main'&&a.variant==='BASE'&&a.pair===pair&&a.rule===rule);
 dates=D.calendars[a.cal];held=M.heldFromTrades(a,dates);const underlying=D.accounts.find(a=>a.ds==='main'&&a.variant==='BASE'&&a.pair===pair&&a.rule==='ALWAYS_LONG_FX');
 a.daily_returns.forEach((r,i)=>close(r,held[i]*underlying.daily_returns[i],1e-12));
 for(const key of ['recent','history','all'])for(const group of ['month','quarter']){
  const [start,end]=(key==='all'?[P.start,P.end]:P.windows[key]).map(day);
  ps=M.periods(a,dates,held,start,end,group);whole=M.interval(a,dates,held,start,end);
  close(ps.reduce((s,p)=>s+p.net,0),whole.net);assert.equal(ps.reduce((s,p)=>s+p.n,0),whole.n);assert.equal(ps.reduce((s,p)=>s+p.active,0),whole.active);
  const lo=dates.findIndex(d=>d>=start);let hi=dates.length-1;while(dates[hi]>end)hi--;
  const existing=a.equity[hi]-(lo?a.equity[lo-1]:100000),error=Math.abs(whole.net-existing);maxError=Math.max(maxError,error);close(whole.net,existing);
  for(const direction of ['long','short','flat'])assert.equal(ps.reduce((s,p)=>s+p[direction],0),whole[direction]);
  for(const p of ps){
   const actual=held.slice(p.lo,p.hi);
   assert.equal(p.long,actual.filter(x=>x>0).length);assert.equal(p.short,actual.filter(x=>x<0).length);
   assert.equal(p.long+p.short,p.active);assert.equal(p.long+p.short+p.flat,p.n);
   close(p.longExposure+p.shortExposure,p.exposure);close(p.exposure+p.flatExposure,100);
   close(p.gains-p.losses,p.net);if(!p.active){assert.equal(p.state,'No exposure');close(p.net,0);}assert(p.offset===null||(p.offset>=0&&p.offset<=100+1e-9));buckets++;}
  c=M.concentration(ps);for(const t of c.top)close(t.remaining+t.removed,c.net);
  if(key==='recent'&&group==='month'&&['AUD','JPY'].includes(pair)&&['BB_MR_20','TSMOM_63'].includes(rule))out.push({pair,rule,pnl:whole.net,complete:c.complete,partial:c.partial,top3:c.top[1],idle:ps.filter(p=>p.state==='No exposure').length});
 }
 accounts++;
}
const report={accounts,buckets,max_window_pnl_error_usd:maxError,boundary_and_reversal_fixtures:'passed',period_sum_and_state_checks:'passed',directional_exposure_checks:'passed: long + short = total; total + flat = 100%; held direction verified in all buckets',examples:out};
fs.writeFileSync(path.join(root,'reproduced/simple_interest/period_attribution_validation.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
