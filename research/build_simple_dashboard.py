"""Build the offline fixed-initial-capital explorer, preserving the original HTML.

Run build_simple_data.py first to reconstruct the unrounded daily returns.
This renderer reuses the original explorer's local plotting and decompression
code; all accounting functions below explicitly use additive PnL.
"""

from pathlib import Path
import base64
import gzip
import hashlib
import json
import re

from src.fx_parameter_diagnostics import build_parameter_data
from src.fx_pair_comparison import build_pair_comparison


ROOT = Path(__file__).resolve().parent


ACCOUNTING_JS = r'''
function accountWindow(a,D,start,end,normalization='window'){
 const dates=D.calendars[a.cal],first=lowerBound(dates,start),last=upperBound(dates,end)-1;
 if(first>last||first>=dates.length||last<0)return null;
 const preceding=first?dates[first-1]:a.anchor,base=first?a.equity[first-1]:CAPITAL;
 const points=[[preceding,normalization==='window'?CAPITAL:base,0,0,null]];
 let peak=CAPITAL,dd=0,ddEquity=0;
 for(let i=first;i<=last;i++){
  const windowEquity=CAPITAL+a.equity[i]-base;
  peak=Math.max(peak,windowEquity);
  const draw=(windowEquity-peak)/CAPITAL*100,drawEquity=(windowEquity/peak-1)*100;
  dd=Math.min(dd,draw);ddEquity=Math.min(ddEquity,drawEquity);
  const dailyPnl=Array.isArray(a.daily_returns)?CAPITAL*a.daily_returns[i]:a.equity[i]-(i?a.equity[i-1]:CAPITAL);
  points.push([dates[i],normalization==='window'?windowEquity:a.equity[i],draw,drawEquity,dailyPnl]);
 }
 let quality=null;
 if(Array.isArray(a.trades)){
  const entries=a.trades.filter(t=>t[0]>=start&&t[0]<=end),closed=entries.filter(t=>t[4]&&t[1]<=end&&t[3]>0);
  const w=closed.filter(t=>t[2]>D.flatTolerancePct),l=closed.filter(t=>t[2]<-D.flatTolerancePct);
  const aw=w.length?w.reduce((s,t)=>s+t[2],0)/w.length:null,al=l.length?-l.reduce((s,t)=>s+t[2],0)/l.length:null;
  quality={entries:entries.length,closed:closed.length,wins:w.length,losses:l.length,flats:closed.length-w.length-l.length,
   win:closed.length?w.length/closed.length*100:null,payoff:w.length&&l.length?aw/al:null,avgWin:aw,avgLoss:al,
   censored:entries.filter(t=>!t[4]||t[1]>end).length,zero:entries.filter(t=>t[4]&&t[1]<=end&&t[3]<=0).length,
   carry:a.trades.filter(t=>t[0]<start&&(!t[4]||t[1]>=start)).length};
 }
 const pnl=a.equity[last]-base,ret=pnl/CAPITAL*100,years=(dates[last]-preceding)/365.25;
 return {a,first,last,points,preceding,base,ret,pnl,dd,ddEquity,annualized:years>0?ret/years:null,
  quality,firstDate:dates[first],lastDate:dates[last],observations:last-first+1};
}
function annualReturns(a,D,start,end){
 const dates=D.calendars[a.cal],out={};
 for(let y=new Date(start*DAY).getUTCFullYear();y<=new Date(end*DAY).getUTCFullYear();y++){
  const lo=Math.max(start,toDay(y+'-01-01')),hi=Math.min(end,toDay(y+'-12-31'));
  const first=lowerBound(dates,lo),last=upperBound(dates,hi)-1;
  if(first<=last&&first<dates.length&&last>=0){const b=first?a.equity[first-1]:CAPITAL;out[y]=(a.equity[last]-b)/CAPITAL*100;}
 }return out;
}
function metricsHTML(view){
 const head=['Independent account / version','Window PnL','Simple return','Annualized simple','Max DD / initial capital','Max DD / peak equity','Closed trades','Win rate','Payoff','Open / censored','Carry-in'];
 const rows=view.records.map(r=>{const q=r.quality,col=r.ret>=0?'positive':'negative';return `<tr><td class="label"><span class="row-dot" style="background:${r.color}"></span>${escapeHTML(r.label)}<small>${r.a.pair} · ${escapeHTML(view.D.rules[r.a.rule].label)} · ${dateString(r.firstDate)}–${dateString(r.lastDate)}</small></td><td class="${col}">${r.pnl<0?'−':''}$${fmt(Math.abs(r.pnl),0)}</td><td class="${col}">${signed(r.ret)}</td><td>${signed(r.annualized)}</td><td>${fmt(r.dd)} pp</td><td>${fmt(r.ddEquity)}%</td><td>${q?q.closed:'—'}${q&&q.closed<20?'<span class="mini-flag">small cohort</span>':''}</td><td>${q&&Number.isFinite(q.win)?fmt(q.win)+'%':'—'}</td><td>${q?fmt(q.payoff,3):'—'}</td><td>${q?q.censored:'—'}</td><td>${q?q.carry:'—'}</td></tr>`;}).join('');
 return `<table><thead><tr>${head.map(x=>'<th>'+x+'</th>').join('')}</tr></thead><tbody>${rows}</tbody></table>`;
}
function csvForView(view){
 const rows=view.records,dates=[...new Set(rows.flatMap(r=>r.points.slice(1).map(p=>p[0])))].sort((a,b)=>a-b);
 const maps=rows.map(r=>new Map(r.points.slice(1).map(p=>[p[0],p])));
 const fields=['display_pnl_usd','display_equity_usd','window_drawdown_initial_capital_pp','window_drawdown_peak_equity_pct','daily_pnl_usd','daily_simple_return_pct'];
 const header=['date',...rows.flatMap(r=>fields.map(f=>r.a.id+'__'+f))];
 return header.join(',')+'\n'+dates.map(d=>[dateString(d),...maps.flatMap(m=>{const p=m.get(d);return p?[(p[1]-CAPITAL).toFixed(6),p[1].toFixed(6),p[2].toFixed(8),p[3].toFixed(8),p[4].toFixed(6),(p[4]/CAPITAL*100).toFixed(10)]:fields.map(()=>'');})].join(',')).join('\n');
}
'''


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Expected one template match, got {count}: {old[:100]}')
    return text.replace(old, new, 1)


def replace_function(text, name, body):
    pattern = rf'^function {name}\([^\n]*\)\{{.*?^\}}'
    result, count = re.subn(pattern, lambda _: body, text, flags=re.M | re.S)
    if count != 1:
        raise ValueError(f'Function replacement failed: {name}: {count}')
    return result


def main():
    original = ROOT / 'FX_PnL_Explorer.html'
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    text = original.read_text(encoding='utf-8')
    payload_path = ROOT / 'reproduced/simple_interest/explorer_payload.json.gz'
    payload_bytes = payload_path.read_bytes()
    payload = json.loads(gzip.decompress(payload_bytes))
    if len(payload['accounts']) != 912:
        raise ValueError('Expected the same 912 independent accounts')
    encoded = base64.b64encode(payload_bytes).decode('ascii')
    text, count = re.subn(r'(<script id="research-data"[^>]*>).*?(</script>)',
                         lambda m: m[1] + encoded + m[2], text, flags=re.S)
    if count != 1:
        raise ValueError('Embedded payload marker missing')
    for name in ['accountWindow', 'annualReturns', 'metricsHTML', 'csvForView']:
        body = re.search(rf'^function {name}\([^\n]*\)\{{.*?^\}}', ACCOUNTING_JS, flags=re.M | re.S)
        text = replace_function(text, name, body[0])

    replacements = [
        ('<title>FX Technical Timing | PnL Research Explorer</title>', '<title>FX 单利版 | Fixed Initial Capital PnL Explorer</title>'),
        ('<h1>FX Technical Timing</h1>', '<h1>FX Technical Timing · 单利版</h1>'),
        ('An interactive record of signals, economic filters and the paths they produced.', 'Fixed initial USD 100,000 exposure · daily PnL added · no reinvestment.'),
        ('Final research edition · 22 October 2026', 'Fixed initial capital · simple PnL · 22 October 2026'),
        ('Gross spot PnL · independent accounts', '单利 / Simple PnL · independent accounts'),
        ('FX Technical Timing · Final English research archive', 'FX Technical Timing · Fixed initial capital / simple PnL'),
        ('The companion PDF contains static results.', 'The companion PDF contains the original compound research results; this simple-interest edition requires JavaScript.'),
        ('the statistic is undefined or was not archived.', 'the statistic is undefined (for example, no closed trades or no losing trades).'),
        ('<main>', '''<main>
<section class="panel accounting-note"><nav aria-label="Companion files"><a href="FX_PnL_Explorer_README.html">详细中文说明：策略、filter 与持仓规则 ↗</a><a href="FX_PnL_Explorer.html">原复利版 ↗</a></nav><h2>固定本金单利：每天按最初的 $100,000 计算盈亏</h2><p><b>Daily PnL = $100,000 × held direction × FX daily return</b>；累计盈亏逐日相加，盈利不加仓、亏损不缩仓。技术信号、过滤条件、成交日和退出日沿用原版本。交易胜率和盈亏比也按单利交易收益重算。</p><p>主图回撤以初始本金的百分点（pp）表示；表格另列窗口重置权益的峰值回撤百分比。旧版历史筛选结论不自动适用于这些单利结果。</p></section>'''),
        ('Cumulative return · %', 'Simple return · % of initial capital'),
        ('Rebase selected window', 'Reset window PnL to zero (additive)'),
        ('<option value="original">Original account path</option>', '<option value="original">Original simple account path</option>'),
        ('Selected-window results', 'Selected-window simple-interest results'),
        ('Account return and drawdown use returns ending within the selected dates, including carried-in positions.', 'PnL is the sum of daily PnL ending within the selected dates, including carried-in positions; return divides that PnL by the fixed initial $100,000. Both drawdown columns start from window-reset equity of $100,000.'),
        ('Payoff = average winning trade return / absolute average losing trade return.', 'Each trade return is the sum of its daily strategy returns. Payoff = average winning simple trade return / absolute average losing simple trade return. Annualized simple return = window simple return / elapsed calendar years (365.25 days); it is a mechanical scale, not CAGR or a forecast.'),
        ('Each calendar year is normalized to the immediately preceding available close.', 'Each calendar year sums daily simple PnL and divides by the same initial $100,000. Years add up to the full-window simple return.'),
        ('Every path starts with USD 100,000 and daily signed 1x spot exposure.', 'Every path starts with USD 100,000. Signed daily spot exposure is fixed at 1x INITIAL capital, rather than tracking current equity; a flat signal has zero exposure.'),
        ('its four filter modes', 'its four execution modes'),
        ('These are exported, validated research outputs. Changing controls never changes a trading rule or refits a parameter. The companion notebook and complete data package document the original workbooks, transformations, frozen protocols, evidence and limitations.', 'These paths are rebuilt from unrounded daily strategy returns. The same reconstructed runs reproduce the archived compound paths before the simple-accounting conversion. V3 trade cohorts are newly reconstructed. Changing controls does not refit rules. The original frozen engine and complete historical protocols were not supplied; the companion guide explains provenance and remaining limits.'),
        ("/* Offline FX research explorer. No network, fitting, or strategy mutation. */", "/* Offline fixed-initial-capital simple PnL explorer. No network or strategy mutation. */"),
        ("${state.from} to ${state.to} · USD 100,000 independent accounts · Zero costs / no carry", "${state.from} to ${state.to} · SIMPLE PnL / fixed initial USD 100,000 · Zero costs / no carry"),
        ("${state.normalization==='window'?'Window rebased':'Original path'}", "${state.normalization==='window'?'Additive window reset':'Original simple path'}"),
        ("'CUMULATIVE RETURN · %':'CUMULATIVE PNL · USD'", "'SIMPLE RETURN · % INITIAL CAPITAL':'SIMPLE CUMULATIVE PNL · USD'"),
        ('DRAWDOWN FROM THE SELECTED WINDOW HIGH · %', 'SELECTED WINDOW DRAWDOWN · pp OF INITIAL CAPITAL'),
        ('${fmt(v,0)}%</text>', '${fmt(v,0)} pp</text>'),
        ("'Rebased to USD 100,000 at the preceding available close.'", "'Additive reset at the preceding close: displayed equity = $100,000 + window PnL.'"),
        ("'Original account path shown; metrics below still describe the selected window.'", "'Original simple path shown; both drawdown and all table metrics still describe the selected window.'"),
        ("'FX_PnL_'+state.dataset", "'FX_Simple_PnL_'+state.dataset"),
        ("'FX_Daily_'+state.dataset", "'FX_Simple_Daily_'+state.dataset"),
        ('Daily account equity is stored to six USD decimal places (at most half a millionth of a dollar rounding).', 'All paths and simple trade returns are recomputed from unrounded daily strategy returns. V3 trade records are reconstructed and are now available. Displayed dollar precision does not alter the trading rules.'),
    ]
    for old, new in replacements:
        text = replace_once(text, old, new)
    text = replace_once(text, '</style>', '''.accounting-note{padding:18px 22px;border-top:4px solid var(--teal)}.accounting-note nav{display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap;font-size:13px;font-weight:600}.accounting-note h2{font-size:19px;margin:17px 0 8px}.accounting-note p{font-size:12px;color:var(--muted);margin:7px 0}.accounting-note b{color:var(--ink)}
</style>''')
    # Empty and invalid views must clear every dependent panel and exported view.
    text = replace_once(text, "$('metrics').innerHTML='';$('legend').innerHTML='';",
                        "$('metrics').innerHTML='';$('legend').innerHTML='';$('annual').innerHTML='';$('definitions').innerHTML='';view=null;API.currentView=null;")
    # Extend the same self-contained dashboard; retain the original explorer.
    module_root = ROOT / 'src/dashboard'
    parameter_data = build_parameter_data(payload)
    comparison_data = build_pair_comparison(payload)
    parameter_bytes = gzip.compress(json.dumps(parameter_data, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    module_nav = '''<nav class="module-nav" aria-label="Dashboard modules"><button id="module-explorer" type="button" aria-pressed="true">PnL Explorer</button><button id="module-parameters" type="button" aria-pressed="false">Parameter Diagnostics</button><button id="module-trades" type="button" aria-pressed="false">Trade Profit Distribution</button><button id="module-comparison" type="button" aria-pressed="false">Pair × Strategy Comparison</button></nav>'''
    text = replace_once(text, '<main>', '<main>' + module_nav + module_root.joinpath('parameter_diagnostics.html').read_text() + module_root.joinpath('trade_distribution.html').read_text() + module_root.joinpath('pair_comparison.html').read_text() + '<div id="explorer-module">')
    text = replace_once(text, '</main>', '</div></main>')
    text = replace_once(text, '</style>', module_root.joinpath('parameter_diagnostics.css').read_text() + module_root.joinpath('trade_distribution.css').read_text() + module_root.joinpath('pair_comparison.css').read_text() + '\n</style>')
    extension = '<script id="parameter-data" type="application/octet-stream">' + base64.b64encode(parameter_bytes).decode() + '</script>\n<script>\n' + module_root.joinpath('parameter_diagnostics.js').read_text() + '\n</script>\n<script>\n' + module_root.joinpath('trade_distribution.js').read_text() + '\n</script>\n'
    extension += '<script id="pair-comparison-data" type="application/json">' + json.dumps(comparison_data, separators=(',', ':'), allow_nan=False) + '</script>\n<script>\n' + module_root.joinpath('pair_timeline.js').read_text() + '\n</script>\n<script>\n' + module_root.joinpath('pair_comparison.js').read_text() + '\n</script>\n'
    text = replace_once(text, '</body>', extension + '</body>')
    destination = ROOT / 'FX_PnL_Explorer_Simple.html'
    destination.write_text(text, encoding='utf-8')
    if hashlib.sha256(original.read_bytes()).hexdigest() != source_hash:
        raise ValueError('Original explorer changed during build')
    manifest = {'original_explorer_sha256': source_hash,
                'simple_explorer_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'payload_sha256': hashlib.sha256(payload_bytes).hexdigest(),
                'accounts': len(payload['accounts']), 'original_preserved': True,
                'accounting': 'fixed_initial_capital_simple',
                'window_reset': 'additive', 'primary_drawdown_unit': 'percentage_points_of_initial_capital',
                'parameter_diagnostics': parameter_data['validation'],
                'pair_comparison': comparison_data['validation']}
    (payload_path.parent / 'dashboard_build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Built {destination.name}: {len(payload["accounts"])} accounts; original HTML preserved.')


if __name__ == '__main__':
    main()
