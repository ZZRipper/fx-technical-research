# FX Technical Research

G10 FX technical strategy research dashboard. Updated 11 November 2026.

## 1. Open the dashboard — no installation needed

Clone this repository or download and extract its ZIP, then open **OPEN_DASHBOARD.html** in Chrome, Edge, Safari or Firefox. Do not open the HTML inside the ZIP preview. Charts and data are embedded; no Bloomberg connection, API key, Python package or internet connection is required for viewing.

If local-file restrictions prevent opening it, use **START_MAC.command** on macOS or **START_WINDOWS.cmd** on Windows. These launch a local server and open the browser using an available port. Python 3.9+ is required for these launchers, but no additional packages are needed. Keep the terminal open while viewing. If your OS blocks a downloaded launcher, open a terminal in this folder and run `python3 run_dashboard.py` (macOS/Linux) or `py -3 run_dashboard.py` (Windows). Do not change OS security settings.

## 2. What changed

The original PnL Explorer is retained. All additions stay in the same dashboard:

- **Parameter Diagnostics:** Bollinger MR 10/20; RSI MR 7/14/20; RSI Momentum 7/14/20; Bollinger Breakout 10/20; MA 10/30 and 20/60; TSMOM 21/63; Donchian Breakout 10/20. All nine USD pairs. Distribution, raw percentiles, signal frequency, trade statistics, PnL and drawdown. Smoothing affects chart display only.
- **Trade Profit Distribution:** every main-history BASE account (20 technical rules and 3 benchmarks, across nine pairs). Closed-trade histogram, gross-profit concentration, top 1/5/10 winners, complete trade ledger, date filters and accounting definitions.

- **Pair × Strategy Comparison — new this week:** conclusions for all nine pairs, backed by 17 fixed rules in eight strategy families. The main review is 2020 onward, with earlier-history, annual and family-removal checks. Mixed evidence is reported explicitly; there is no automatic best-strategy selection.
- **Performance timeline — new this week:** inside the same pair detail, next to Style comparison. Choose any of the 17 rules, then inspect monthly or quarterly PnL and exposure. Click a calendar cell or choose a range. Compare results after removing the best 1 or 3 complete periods. The export includes all nine pairs and all 17 rules for the selected history and period size.

**Directional exposure:** the timeline summary, calendar, interval detail and CSV now split actual held exposure into Long and Short. Both percentages use all return intervals as the denominator; Long + Short = Total, and Total + Flat = 100%. Signals and account PnL are unchanged.

The dashboard opens the timeline view. Use **Style comparison** in the same pair detail to review the conclusions and their evidence. The original three dashboard tabs remain available.

Calendar-period PnL uses daily return-ending dates, including open and carried-in positions. Cross-month trades are split by daily PnL. This differs from the trade-level module: calendar concentration measures positive-period net PnL, while trade concentration measures winning-trade profit. Partial periods remain in the curve but are excluded from the best-period removal comparison. Gain/loss labels are retrospective; no exposure is determined from positions, not zero price changes. These labels are not macro regimes or forward trading signals.

The trade-distribution cohort uses exit dates and entire trade PnL, including cross-window entries. It excludes open trades and trades with no earned return interval. This differs deliberately from the main Explorer's daily-PnL window. Read the definitions inside the module.

All results retain the original execution and fixed $100,000 exposure conventions. They exclude costs and carry. Examined historical results are not fresh OOS evidence.

## 3. Review or modify code

Open **this whole repository** in VS Code. Complete research code, notebooks, input data and saved outputs are under `research/`; original relative paths are preserved.

| File | Purpose |
| --- | --- |
| `research/build_simple_dashboard.py` | Builds the self-contained dashboard |
| `research/src/fx_parameter_diagnostics.py` | Indicator series and return reconciliation |
| `research/src/dashboard/parameter_diagnostics.*` | Parameter module UI |
| `research/src/dashboard/trade_distribution.*` | Trade-distribution UI and statistics |
| `research/src/fx_pair_comparison.py` | Comparison data and account reconciliation |
| `research/src/dashboard/pair_comparison.*` | Pair conclusions and supporting evidence |
| `research/src/dashboard/pair_timeline.js` | Calendar attribution, exposure and period concentration |
| `research/src/fx_baselines.py` | Original trading rules and execution |
| `research/src/fx_simple.py` | Original simple-accounting reconstruction |
| `research/build_simple_data.py` | Reconstructs the 912-account dashboard payload |
| `research/FX_Research_Datasets/` | Included input data |
| `research/FX_Technical_Research_Complete.ipynb` | Teammate's original research notebook |
| `research/README.md` | Full original research workflow and provenance |

The original compound dashboard `research/FX_PnL_Explorer.html` is also the builder's template: keep it. The archived research files are retained for reproducibility, not additional steps required to view the dashboard.

## 4. Rebuild after edits

Recommended: Python **3.12**. Run commands from this extracted package folder. Viewing does not require these steps. Dependency installation requires internet access; subsequent rebuilds use local data only.

macOS / Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python research/build_simple_dashboard.py
python run_dashboard.py
```

Windows (Command Prompt):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python research/build_simple_dashboard.py
python run_dashboard.py
```

To rebuild account data from supplied inputs as well:

```sh
python research/build_simple_data.py
python research/build_simple_dashboard.py
```

The dashboard rebuild uses the included account payload and recalculates diagnostic indicators from the raw workbook. The data rebuild also reconstructs the legacy filter accounts. A reconciliation failure aborts rather than silently changing results. Outputs are written inside `research/`.

For the original notebooks, plotting and optional browser-verification scripts, install `requirements-research.txt` in the same environment and follow `research/README.md`. Those tools are not needed to rebuild the dashboard modules.

## 5. Check this week's additions

Run from the extracted package folder:

```sh
python run_dashboard.py --check
cd research
python -m unittest discover -s tests -p test_pair_comparison.py
node tests/test_pair_timeline.cjs
```

Node.js is only needed for the timeline validation command, not to view or rebuild the dashboard. The Python check independently verifies comparison metrics and accounting boundaries. The JavaScript check verifies all 153 rule/pair timelines, monthly/quarterly reconciliation, position timing, cross-month trades, zero-return exposure and partial periods. Validation records are in `research/reproduced/simple_interest/`.

## 6. Verification and scope

- Parameter diagnostics: 144 strategy/pair accounts reconcile to saved daily returns with zero error.
- Trade distribution: 54,280 records across 207 BASE accounts checked against daily-return segments; 621 account/window checks and boundary/empty-cohort checks passed.
- No computer-specific paths or external data connections are needed for the dashboard build.
- Local launchers are provided for macOS and Windows; actual execution is verified on macOS, not a Windows host.
