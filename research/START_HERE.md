# Vanguard FX Technical Research

This workspace uses the complete teammate delivery received in October 2026.

## Open these first

- [Interactive dashboard](FX_PnL_Explorer_Simple.html): existing fixed-capital results. Open in a web browser.
- [Research notebook](FX_Technical_Research_Complete.ipynb): the main research entry point.
- [Project README](README.md): reproduction instructions and subsequent research rounds.
- [Dashboard definitions](FX_PnL_Explorer_README.md): strategy, filter, execution and metric definitions.

## Project layout

| Location | Contents |
| --- | --- |
| `src/` | Strategy, execution, filtering and evaluation code |
| `FX_Research_Datasets/` | Supplied input data |
| `reproduced/` | Saved results, ledgers, reports and validation records |
| `tests/` | Automated checks |
| `reference/` | Retained reference outputs |
| `notebook_figures/` | Notebook figures |

In the VS Code Explorer, expand the research notebook to see its `run_`, `build_`, `verify_` and `analyze_` scripts. This grouping changes only the display; all scripts remain at their original paths. Expand README for the companion instructions. Expand the simple dashboard for the original compound dashboard.

## Reproduce the dashboard

From this folder, in a Python environment with the packages in `requirements.txt`:

```sh
python build_simple_data.py
python build_simple_dashboard.py
```

These commands regenerate saved simple-interest outputs. The dashboard builder also adds the Parameter Diagnostics module (Bollinger MR and Breakout 10/20, RSI MR 7/14/20 and RSI Momentum 7/14/20, Moving Average 10/30 vs 20/60, Time-series Momentum 21/63, and Donchian Breakout 10/20, all nine pairs). Its code is in `src/fx_parameter_diagnostics.py` and `src/dashboard/`; every account is reconciled against the existing main-history payload. To rebuild only the dashboard from the existing payload, run `python build_simple_dashboard.py`.

The original compound explorer and strategy engines remain unchanged. Historical results, costs and validation limitations are described in the supplied README.

Trade Profit Distribution is embedded in the same dashboard (`#trades`). It uses existing main-history BASE trade ledgers for all nine pairs. The cohort includes complete trades by exit date, with full PnL for cross-window entries; open and zero-return-interval trades are reported separately. The module code is in `src/dashboard/trade_distribution.*`. Rebuild with `python build_simple_dashboard.py`.

Pair × Strategy Comparison (`#comparison`) compares 17 unchanged rules and two directional benchmarks per pair. Main workbook only, 2005–May 2026; 2020 onward is the primary retrospective evaluation, not untouched OOS. Earlier subperiods, annual returns, family coverage and family-removal sensitivity are shown. Metrics use existing daily returns and ledgers, validated against original account equity. Build with the same dashboard command; run `python -m unittest discover -s tests -p test_pair_comparison.py` for accounting-boundary checks. Source: `src/fx_pair_comparison.py` and `src/dashboard/pair_comparison.*`. No transaction costs or carry are added.

The same pair detail contains **Performance timeline**: monthly or quarterly daily-PnL attribution, exposure, gain/loss cancellation, and concentration in the best complete periods. All nine pairs and 17 rules are available. Calendar labels are retrospective outcomes, not entry signals or macro regimes. Export covers all pairs and rules for the selected history and period size. Source: `src/dashboard/pair_timeline.js`; validate with `node tests/test_pair_timeline.cjs`. This checks position timing, cross-month trades, partial periods and reconciliation to original account equity. No new data build is required; use the same dashboard builder above.

Performance timeline exposure includes Long, Short and Flat percentages using all quote intervals as the denominator. Summary, calendar, selected-period detail and CSV export share the same position-ledger calculation. This changes reporting only, not signals or PnL.
