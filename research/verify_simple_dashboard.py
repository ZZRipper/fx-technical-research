"""Exercise both offline HTML deliverables in an isolated, headless Chrome.

Uses the already-installed Chrome and websocket-client; no website, account,
browser profile or network data is involved. Saves screenshots and QA evidence
under reproduced/simple_interest/. Run after both HTML files have been built.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

import websocket


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduced/simple_interest'


class Browser:
    def __init__(self, profile):
        candidates = [Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'Google/Chrome/Application/chrome.exe',
                      Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')]
        executable = next((p for p in candidates if p.exists()), None)
        if executable is None:
            raise RuntimeError('Installed Chrome or Edge is required for the offline browser check')
        self.process = subprocess.Popen([
            str(executable), '--headless=new', '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1',
            '--remote-allow-origins=*', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
            '--disable-component-update', '--disable-sync', f'--user-data-dir={profile}', 'about:blank',
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.socket = None
        self.next_id = 0
        self.errors = []
        self.http_requests = []
        port_file = Path(profile) / 'DevToolsActivePort'
        deadline = time.monotonic() + 25
        while not port_file.exists():
            if time.monotonic() > deadline or self.process.poll() is not None:
                self.close()
                raise RuntimeError('Isolated browser did not start')
            time.sleep(.1)
        port = port_file.read_text().splitlines()[0]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/json/list', timeout=5) as response:
            pages = json.load(response)
        url = next(p['webSocketDebuggerUrl'] for p in pages if p['type'] == 'page')
        self.socket = websocket.create_connection(url, timeout=30, http_proxy_host=None)
        self.call('Page.enable')
        self.call('Runtime.enable')
        self.call('Network.enable')

    def call(self, method, params=None):
        self.next_id += 1
        ident = self.next_id
        self.socket.send(json.dumps({'id': ident, 'method': method, 'params': params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(message['params']['exceptionDetails'])
            if message.get('method') == 'Network.requestWillBeSent':
                url = message['params']['request']['url']
                if url.startswith(('http:', 'https:')):
                    self.http_requests.append(url)
            if message.get('id') == ident:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
        if 'exceptionDetails' in result:
            raise AssertionError(result['exceptionDetails'])
        return result['result'].get('value')

    def navigate(self, file, ready):
        self.call('Page.navigate', {'url': file.as_uri()})
        deadline = time.monotonic() + 45
        while not self.evaluate(ready):
            if time.monotonic() > deadline:
                raise RuntimeError(f'Page did not become ready: {file.name}')
            time.sleep(.1)

    def screenshot(self, name, width=1440, height=1050):
        self.call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': width < 600})
        result = self.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
        (OUT / name).write_bytes(base64.b64decode(result['data']))

    def close(self):
        if self.socket:
            try:
                self.call('Browser.close')
            except (OSError, RuntimeError, websocket.WebSocketException, json.JSONDecodeError):
                pass
            self.socket.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


MATHEMATICAL_CHECKS = r'''(() => {
 const A=FXResearch,check=(ok,message)=>{if(!ok)throw Error(message);},near=(a,b,t=1e-8)=>Math.abs(a-b)<t;
 const d=A.toDay('2024-01-02');
 const account={id:'test',cal:0,anchor:d-1,equity:[110000,100000,100000,105000],daily_returns:[.1,-.1,0,.05],
  trades:[[d,d+1,0,1,true],[d+2,d+3,5,1,true],[d+3,d+3,0,0,false]]};
 const data={calendars:[[d,d+1,d+2,d+3]],flatTolerancePct:1e-10};
 const full=A.accountWindow(account,data,d,d+3),tail=A.accountWindow(account,data,d+1,d+3),original=A.accountWindow(account,data,d+1,d+3,'original');
 check(near(full.ret,5)&&near(full.pnl,5000),'fixed capital +10%-10%+0%+5% gives +5%');
 check(near(full.dd,-10)&&near(full.ddEquity,-100/11),'initial-capital and peak-equity drawdown differ');
 check(near(tail.ret,-5)&&near(tail.points[0][1],100000)&&near(tail.points[1][1],90000),'additive window reset');
 check(near(tail.dd,-10)&&near(tail.ddEquity,-10),'window drawdown restarts from $100000');
 check(near(original.ret,tail.ret)&&near(original.dd,tail.dd)&&near(original.ddEquity,tail.ddEquity),'normalization does not alter metrics');
 check(near(original.points[0][1],110000)&&near(original.points[1][1],100000),'original chart keeps original wealth');
 check(full.quality.closed===2&&full.quality.flats===1&&near(full.quality.win,50),'flat completed trade remains in denominator');
 check(tail.quality.carry===1&&tail.quality.closed===1&&tail.quality.censored===1,'carry-in and censored cohorts');
 check(near(A.accountWindow(account,data,d+1,d+1).ret,-10),'single date includes return ending that date');
 check(A.accountWindow(account,data,d+10,d+11)===null,'no-quote interval');
 const dates=[A.toDay('2023-12-29'),A.toDay('2024-01-02'),A.toDay('2024-01-03')];
 const yearAccount={...account,anchor:dates[0]-1,equity:[110000,100000,105000],cal:0,trades:[],daily_returns:[.1,-.1,.05]};
 const years=A.annualReturns(yearAccount,{...data,calendars:[dates]},dates[0],dates[2]);
 check(near(years[2023],10)&&near(years[2024],-5),'calendar years use fixed initial capital');
 const cash={...account,equity:[100000,100000,100000,100000],daily_returns:[0,0,0,0],trades:[]};
 const cashView=A.accountWindow(cash,data,d,d+3);
 check(cashView.ret===0&&cashView.dd===0&&cashView.quality.win===null,'cash and undefined quality');
 const belowZero={...account,equity:[100000,-10000,5000,10000],trades:[]};
 const negative=A.accountWindow(belowZero,data,d,d+3);
 check(near(negative.dd,-110)&&near(negative.ret,-90)&&negative.points.every(p=>p.slice(0,4).every(Number.isFinite)),'fixed exposure may cross zero without invalid arithmetic');
 return {hand_calculated_accounting:true,window_and_anchor:true,cohort_and_flat_trades:true,calendar_additivity:true,negative_equity:true};
})()'''


REAL_DATA_CHECKS = r'''(() => {
 const A=FXResearch,D=A.data,check=(ok,message)=>{if(!ok)throw Error(message);};
 let maxEndpoint=0,maxAnnual=0,totalRows=0;
 for(const a of D.accounts){
  const dates=D.calendars[a.cal],sum=a.daily_returns.reduce((x,y)=>x+y,0),r=A.accountWindow(a,D,dates[0],dates[dates.length-1]);
  const err=Math.abs(r.pnl-100000*sum);maxEndpoint=Math.max(maxEndpoint,err);check(err<1e-6,'endpoint '+a.id);
  check(a.equity.length===dates.length&&a.daily_returns.length===dates.length,'coverage '+a.id);
  check(r.points.every(p=>p.slice(0,4).every(Number.isFinite)),'finite '+a.id);
  const y=A.annualReturns(a,D,dates[0],dates[dates.length-1]),ye=Math.abs(Object.values(y).reduce((s,v)=>s+v,0)-r.ret);
  maxAnnual=Math.max(maxAnnual,ye);check(ye<1e-8,'annual '+a.id);
  if(a.rule==='CASH')check(Math.abs(r.pnl)<1e-9&&r.quality.closed===0,'real cash '+a.id);
  totalRows+=dates.length;
 }
 const text=A.csvForView(A.currentView),lines=text.split('\n'),columns=lines[0].split(',');
 check(columns.some(x=>x.endsWith('__daily_simple_return_pct')),'simple CSV return column');
 check(lines.length===A.currentView.records[0].observations+1,'CSV excludes anchor');
 const svg=A.renderSVG(A.currentView,true);check(svg.includes('SIMPLE PnL')&&svg.includes('pp OF INITIAL CAPITAL'),'SVG labels');
 check(!/NaN|Infinity/.test(svg),'finite SVG');
 return {accounts:D.accounts.length,total_daily_rows:totalRows,max_endpoint_error_usd:maxEndpoint,max_annual_sum_error_pp:maxAnnual,csv_rows:lines.length-1,svg_labeled:true};
})()'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fx-simple-browser-', ignore_cleanup_errors=True) as profile:
        browser = Browser(profile)
        try:
            browser.navigate(ROOT / 'FX_PnL_Explorer_Simple.html', 'Boolean(window.FXResearch && FXResearch.ready)')
            result = {'math': browser.evaluate(MATHEMATICAL_CHECKS), 'data': browser.evaluate(REAL_DATA_CHECKS)}
            browser.screenshot('dashboard_desktop.png')
            browser.evaluate("document.querySelector('.chart-panel').scrollIntoView({block:'start'})")
            browser.screenshot('dashboard_chart.png', 1440, 1100)
            browser.evaluate('window.scrollTo(0,0)')
            downloads = Path(profile) / 'downloads'
            downloads.mkdir()
            browser.call('Browser.setDownloadBehavior', {'behavior': 'allow', 'downloadPath': str(downloads)})
            expected_csv = browser.evaluate('FXResearch.csvForView(FXResearch.currentView)')
            expected_svg = browser.evaluate('FXResearch.renderSVG(FXResearch.currentView,true)')
            browser.evaluate("document.getElementById('download-csv').click();document.getElementById('download-svg').click()")
            deadline = time.monotonic() + 20
            while len(list(downloads.glob('*.csv'))) != 1 or len(list(downloads.glob('*.svg'))) != 1:
                if time.monotonic() > deadline:
                    raise AssertionError('CSV/SVG buttons did not finish downloading')
                time.sleep(.1)
            csv_path = next(downloads.glob('*.csv'))
            svg_path = next(downloads.glob('*.svg'))
            if csv_path.read_text(encoding='utf-8') != expected_csv or svg_path.read_text(encoding='utf-8') != expected_svg:
                raise AssertionError('Downloaded files differ from the displayed view export')
            result['downloads'] = {'csv_matches_view': True, 'svg_matches_view': True,
                                   'csv_bytes': csv_path.stat().st_size, 'svg_bytes': svg_path.stat().st_size}
            # Use actual input events so the tests cover selector and redraw wiring.
            result['interactions'] = browser.evaluate(r'''(() => {
             const check=(ok,m)=>{if(!ok)throw Error(m);},set=(id,v)=>{const e=document.getElementById(id);e.value=v;e.dispatchEvent(new Event('change',{bubbles:true}));};
             document.getElementById('select-none').click();check(FXResearch.currentView.records.length===0&&document.getElementById('download-csv').disabled,'empty selection');
             document.getElementById('select-core').click();check(FXResearch.currentView.records.length>0,'restore selection');
             set('normalization','original');check(FXResearch.currentState.normalization==='original','original view');set('normalization','window');
             for(const ds of ['technical','h10','main']){set('dataset',ds);check(FXResearch.currentView.records.length>0,'dataset '+ds);}
             for(const mode of ['strategies','currencies','filters']){document.querySelector('[data-mode="'+mode+'"]').click();check(FXResearch.currentView.records.length>0,'mode '+mode);}
             set('from','2026-05-29');set('to','2005-01-03');check(FXResearch.currentView===null&&document.getElementById('download-svg').disabled&&document.getElementById('annual').innerHTML==='','invalid dates');
             document.querySelector('[data-from="2005-01-03"]').click();check(FXResearch.currentView.records.length>0,'date preset recovery');
             return {dataset_switches:3,comparison_modes:3,empty_selection:true,invalid_dates:true,normalization_toggle:true};
            })()''')
            browser.screenshot('dashboard_mobile.png', 390, 844)
            browser.navigate(ROOT / 'FX_PnL_Explorer_README.html', "document.readyState==='complete' && document.title.length>0")
            result['guide'] = browser.evaluate(r'''(() => {
             const check=(ok,m)=>{if(!ok)throw Error(m);};
             const links=[...document.querySelectorAll('a[href^="#"]')],missing=links.map(a=>a.getAttribute('href').slice(1)).filter(id=>id&&!document.getElementById(id));
             if(missing.length)throw Error('Broken guide anchors: '+missing.join(','));
             const search=document.getElementById('search'),sections=[...document.querySelectorAll('main .section')],details=[...document.querySelectorAll('main details')];
             search.value='COMMODITY_USD';search.dispatchEvent(new Event('input'));check(sections.some(s=>s.hidden)&&sections.some(s=>!s.hidden),'guide search filters sections');
             search.value='qwerty_missing_phrase_9182';search.dispatchEvent(new Event('input'));check(!document.getElementById('no-results').hidden,'guide empty search state');
             search.value='';search.dispatchEvent(new Event('input'));check(sections.every(s=>!s.hidden),'clear search restores sections');
             document.getElementById('expand-all').click();check(details.every(d=>d.open),'expand all');
             document.getElementById('collapse-all').click();check(details.every(d=>!d.open),'collapse all');
             window.dispatchEvent(new Event('beforeprint'));check(details.every(d=>d.open)&&sections.every(s=>!s.hidden),'print expands');
             window.dispatchEvent(new Event('afterprint'));check(details.every(d=>!d.open),'print restores state');
             return {title:document.title,internal_links:links.length,broken_anchors:missing.length,details:details.length,text_characters:document.body.textContent.length,search:true,expand_collapse:true,print_expansion:true};
            })()''')
            browser.screenshot('guide_desktop.png')
            browser.screenshot('guide_mobile.png', 390, 844)
            if browser.errors:
                raise AssertionError(f'Browser errors: {browser.errors}')
            if browser.http_requests:
                raise AssertionError(f'Unexpected external requests: {browser.http_requests}')
            result.update(browser_errors=0,external_http_requests=0,status='passed',
                          checked_file_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                               for name in ['FX_PnL_Explorer_Simple.html', 'FX_PnL_Explorer_README.html',
                                                            'verify_simple_dashboard.py']})
            (OUT / 'browser_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
