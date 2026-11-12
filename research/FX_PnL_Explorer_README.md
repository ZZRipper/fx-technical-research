# FX PnL Explorer: Strategies, Filters, Holdings, and Accounting

**English reference guide — 22 October 2026**

This guide explains every strategy and filter available in the FX PnL Explorer, the inputs used to construct them, and the rules governing entries, holdings, exits, and performance measurement. It covers both dashboard editions:

- [Fixed-initial-capital simple-interest dashboard](FX_PnL_Explorer_Simple.html): daily dollar PnL is calculated on the original USD 100,000 and added over time.
- [Original compound-return dashboard](FX_PnL_Explorer.html): daily strategy returns compound on changing account equity. This file is preserved as research evidence.

Both dashboards contain **912 independent account paths**, covering **9 currencies**, **20 technical strategies**, **3 benchmark definitions**, and **14 execution/filter variants** across three research samples. Availability depends on the sample, currency, and strategy; the dashboard does not run every possible combination.

The simple-interest edition uses the same signals, filters, positions, and execution dates as the reconstructed compound research. Trade returns, win rates, payoff ratios, account returns, and drawdowns are recalculated under the new accounting convention.

## Contents

1. [Quick start](#1-quick-start)
2. [Simple-interest and compound accounting](#2-simple-interest-and-compound-accounting)
3. [Datasets and account boundaries](#3-datasets-and-account-boundaries)
4. [Currency quotes and trading direction](#4-currency-quotes-and-trading-direction)
5. [Signals, execution, holdings, and trade episodes](#5-signals-execution-holdings-and-trade-episodes)
6. [All 20 technical strategies and 3 benchmarks](#6-all-20-technical-strategies-and-3-benchmarks)
7. [How the filters were developed and how they work](#7-how-the-filters-were-developed-and-how-they-work)
8. [Economic inputs and information timing](#8-economic-inputs-and-information-timing)
9. [Dashboard controls and exports](#9-dashboard-controls-and-exports)
10. [Metrics, trade cohorts, and interpretation](#10-metrics-trade-cohorts-and-interpretation)
11. [Historical screening results and evidence limits](#11-historical-screening-results-and-evidence-limits)
12. [Files, reproduction, and validation](#12-files-reproduction-and-validation)

## 1. Quick start

Open either dashboard in a modern browser. Its scripts, data, and plotting code are embedded in the HTML file; no server or network connection is required.

1. Choose **Main history** as the research sample.
2. Select **Filters**, then choose a currency and technical strategy. For example, select `CAD` and `BB_MR_20`.
3. Compare `BASE` with `COMMODITY_USD`. The first follows the original technical rule; the second can reject complete original trades using commodity and dollar information.
4. Inspect the full history, then individual calendar windows. Compare PnL, drawdown, closed-trade count, win rate, and payoff together.
5. Expand the annual-return table to see whether improvements persist across years. Move over the chart to inspect dated values.
6. Download the current chart as SVG or its daily values as CSV.

The three comparison modes answer different questions:

| Mode | Fixed dimensions | Compared dimension |
| --- | --- | --- |
| Filters | Currency and technical strategy | Execution/filter variants |
| Strategies / windows | Currency and variant | Technical strategies and their fixed lookbacks |
| Currencies | Strategy and variant | Available currencies, each with its own account |

The controls select existing paths. Changing dates does not refit parameters, select recent winners, reset indicator states, or close a position at the start of the selected window. Comparing currencies does not create a portfolio or allocate capital between accounts.

## 2. Simple-interest and compound accounting

### 2.1 Shared daily strategy return

Let:

- `C0 = USD 100,000`: initial capital for each independent account.
- `P_t`: canonical FX price, measured in USD per unit of foreign currency.
- `h_t ∈ {-1, 0, +1}`: position earning the price return ending on quote date `t`.
- `r_t`: the simple close-to-close return of the canonical FX price.
- `u_t`: the strategy return for that interval.

```text
r_t = P_t / P_(t-1) - 1
u_t = h_t * r_t
```

All displayed results currently assume zero transaction costs, carry, financing costs, and cash interest. A flat position has zero PnL.

### 2.2 Fixed-initial-capital simple interest

The requested simple-interest convention is **constant dollar exposure relative to initial capital**:

```text
Daily PnL_t       = C0 * u_t
Cumulative PnL_t  = C0 * sum(u_j, j <= t)
Equity_t          = C0 + cumulative PnL_t
Total return (%)  = 100 * sum(u_j)
```

Profits do not increase subsequent dollar exposure, and losses do not decrease it. This convention also differs from buying a fixed number of foreign-currency units at each entry and holding that quantity unchanged. The daily notional is fixed in USD terms whenever the position is active.

Under this convention, negative account equity is mathematically possible without the formula ceasing to work. The research does not add a margin call, liquidation rule, or leverage constraint when equity falls. A fixed USD 100,000 exposure can therefore represent more than 1x current equity after losses.

### 2.3 Original compound accounting

The original edition applies each daily return to the previous day's equity:

```text
Equity_t          = C0 * product(1 + u_j, j <= t)
Daily PnL_t       = Equity_(t-1) * u_t
Total return (%)  = 100 * [product(1 + u_j) - 1]
```

For a sequence of `+10%` and `-10%`, simple-interest equity returns to USD 100,000, while compound equity becomes USD 99,000.

### 2.4 Individual trades and annual returns

For a trade entered at close `i` and exited at close `j`, the relevant daily returns end at `i+1, ..., j`:

```text
Simple trade return (%)   = 100 * sum(u_t, t = i+1, ..., j)
Simple trade PnL          = C0 * sum(u_t, t = i+1, ..., j)
Compound trade return (%) = 100 * [product(1 + u_t, t = i+1, ..., j) - 1]
```

The simple-interest dashboard recalculates trade win/loss classifications from simple trade returns. Even a win rate can change when the same sequence of daily returns is added instead of compounded.

Simple calendar-year returns sum the daily strategy returns ending in that year and inside the selected window. They use the same initial-capital denominator every year. Consequently, annual simple returns add to the corresponding full-window return. Compound annual returns must be compounded rather than added.

### 2.5 Window normalization

Let `pre` be the last available quote before the first quote selected by the date window. If the first account quote is selected, use the account's capital anchor and `E_pre = C0`.

```text
Window PnL                  = E_last - E_pre
Window simple return (%)    = 100 * (E_last - E_pre) / C0
Window-reset equity W_t     = C0 + E_t - E_pre
Original-path equity        = E_t
```

The simple-interest window reset uses subtraction, not division. Dollar PnL is the dollar difference itself; only the percentage return divides by `C0`.

The original compound chart instead uses proportional rebasing:

```text
Compound window equity      = C0 * E_t / E_pre
Compound window return (%)  = 100 * (E_last / E_pre - 1)
```

Selecting **Original account path** changes the displayed equity/PnL level. The results table, annual returns, and drawdown calculation still describe the selected window.

### 2.6 Two drawdown conventions in the simple-interest edition

Define the running peak of window-reset equity, including the initial capital anchor:

```text
W_pre = C0
H_t   = max(C0, W_first, ..., W_t)

Drawdown / initial capital (pp) = 100 * (W_t - H_t) / C0
Drawdown / peak equity (%)     = 100 * (W_t / H_t - 1)
Maximum drawdown               = minimum of the corresponding drawdown series
```

The chart uses **percentage points of initial capital (`pp`)**. The table also shows the conventional percentage drawdown from peak window equity. Both series start at the selected-window anchor and remain unchanged when switching between window-reset and original-path displays.

Example: equity rises from USD 100,000 to USD 120,000, then falls to USD 108,000. The USD 12,000 loss from the peak is `-12 pp` of initial capital and `-10%` of peak equity.

### 2.7 Annualization

```text
Y = (last quote date - window capital-anchor date) / 365.25 calendar days
Annualized simple return (%) = window simple return (%) / Y
```

This is linear annualization, not CAGR. CAGR in the original research is the compound annual growth rate. Annualizing a short H.10 window is a mechanical rescaling, not a forecast.

Implementation: [fx_baselines.py](src/fx_baselines.py), [fx_simple.py](src/fx_simple.py), and [build_simple_dashboard.py](build_simple_dashboard.py).

## 3. Datasets and account boundaries

| Dashboard dataset | Displayed evaluation period | Paths | Contents and account origin |
| --- | --- | ---: | --- |
| `main` | 2005-01-03 to 2026-05-29 | 471 | 207 fixed baselines, 36 V5 variants, and 228 additional economic-filter paths. Indicators use earlier history for warmup; accounts begin from cash in 2005 and continue thereafter. |
| `technical` | 2020-01-01 to 2026-05-29 | 234 | V3 T-stage accounts: 9 currencies × (20 strategies + 2 directional benchmarks + 4 gates). All accounts reset to cash in 2020, with indicators warmed using earlier history. No CASH benchmark is included. |
| `h10` | 2026-06-01 to 2026-10-16 | 207 | 9 currencies × (20 strategies + 3 benchmarks), replayed on a separate H.10 quote source. Accounts begin independently in June; economic-filter variants are absent. |

The V7/V8 experiment contains `9 currencies × 4 core rules × 8 variants = 288` accounts. The Explorer omits 36 duplicated BASE records and 24 commodity-filter copies for currencies to which that filter does not apply. The remaining 228 economic-filter paths are included in `main`.

The underlying workbook price panel contains **62,010 records: 6,890 dates × 9 currencies**, from 2000-01-03 through 2026-05-29. The feature workbook has 11 sheets and 226 fields, ending on the same date. Additional variables do not constitute additional out-of-sample dates.

The complete V3 reconstruction has three separately funded stages:

- D: 2005–2014.
- V: 2015–2019.
- T: 2020–2026-05-29.

Each stage contains 234 accounts, giving 702 reconstructed V3 accounts. The dashboard displays only T. A date selector does not switch the dashboard to D or V, and the independently reset V3 curves must not be spliced onto continuous main-history curves.

H.10 includes older source history beginning in June 2023. The standard replay warms indicators from 2024-01-02, using 604 valid pre-evaluation quotes per currency. Each evaluation path has 96 valid quotes. The recorded nonquote dates are omitted, with no manufactured trades or interpolated returns. An additional warmup check using earlier available history reproduces the same evaluation-period technical targets.

H.10 has a different quote source and fixing convention from the workbooks. Its price levels are not appended to the main panel. It has already been examined and lacks matching later economic inputs, so it is not fresh validation for V5–V8 filters.

The original V3 HTML archive did not contain trade ledgers and shows unavailable trade-quality statistics. The simple-interest edition supplies newly reconstructed V3 ledgers. These are new reconstruction outputs, not recovered original archival ledgers.

Sources: [data dictionary](FX_Research_Datasets/data/README_Data_Dictionary.txt), [H.10 coverage](reproduced/h10_coverage.csv), and [fx_legacy.py](src/fx_legacy.py). Where old data-package documentation refers to unavailable frozen files, the current [root README](README.md) describes the actual reconstruction boundary.

## 4. Currency quotes and trading direction

All indicators and PnL use **USD per one unit of foreign currency**.

| Currency | Name | Common native market quote | Conversion to USD per foreign unit | Local 2Y yield proxy |
| --- | --- | --- | --- | --- |
| EUR | Euro | EURUSD | Use directly | Germany (`DE`) |
| GBP | Pound sterling | GBPUSD | Use directly | United Kingdom (`UK`) |
| AUD | Australian dollar | AUDUSD | Use directly | Australia (`AU`) |
| NZD | New Zealand dollar | NZDUSD | Use directly | New Zealand (`NZ`) |
| JPY | Japanese yen | USDJPY | Take reciprocal | Japan (`JP`) |
| CHF | Swiss franc | USDCHF | Take reciprocal | Switzerland (`CH`) |
| CAD | Canadian dollar | USDCAD | Take reciprocal | Canada (`CA`) |
| NOK | Norwegian krone | USDNOK | Take reciprocal | Norway (`NO`) |
| SEK | Swedish krona | USDSEK | Take reciprocal | Sweden (`SE`) |

The supplied PIT workbook and `price_USD_per_foreign` field already contain canonical prices. Do not invert those values again. H.10 retains both its native quote and its canonical `usd_per_foreign` price.

- `q = +1`: long the foreign currency, short USD. A rise in canonical `P` produces a positive return.
- `q = -1`: short the foreign currency, long USD. The strategy earns `-r_t`.
- `q = 0`: no spot exposure.

For example, a positive JPY position means long yen: its favorable direction is a decline in the conventional USDJPY quote.

Earning `-r_t` is not identical to buying and holding the reciprocal price. The reciprocal price has daily return `-r_t / (1+r_t)`. The research consistently uses signed exposure to the canonical price return.

## 5. Signals, execution, holdings, and trade episodes

### 5.1 Signal-to-return clock

The model separates the signal, execution, and earned-return dates:

| Time | Action |
| --- | --- |
| Close `t` | Compute technical target `s_t` from prices through `t`; evaluate any applicable economic gate using `K_t`. |
| Close `t+1` | Execute the target: `e_(t+1) = s_t`. |
| Return ending at `t+2` | The new position first earns a return: `h_(t+2) = e_(t+1)`. |

Here `t+1` and `t+2` mean successive available quotes, not necessarily successive calendar dates. A new position cannot earn a return from an interval that ended before its execution close.

### 5.2 Holding, closing, reversing, and sample-end positions

- **Hold:** retain the current direction while the target remains unchanged. Stateful strategies retain an internal target until an explicit transition; daily direction rules recompute the target each day.
- **Close:** a zero target executes at the next quote close. The old position still earns the return ending at that exit close.
- **Reverse:** a target changing directly from `+1` to `-1`, or vice versa, closes the old trade and starts the opposite trade at the same next execution close. The old trade receives the exit-close return; the new trade starts earning at the following quote.
- **End of sample:** an open trade is marked to market at the last available quote. There is no artificial terminal liquidation or invented extra return. An entry at the final close can have zero earned-return intervals.

“Daily” describes observation and execution frequency. A trade can remain open across weekends and for many days or months.

### 5.3 Warmup and initial funding

Indicators and state machines run on earlier prices before evaluation begins. Execution is suppressed before the permitted account start. At the first eligible execution close, a strategy can adopt a target already established using the preceding quote and earlier warmup data.

The indicators do not restart from scratch at the funding date. The account nevertheless starts from cash and does not inherit earlier PnL.

### 5.4 What an original episode means

A BASE episode is a continuous interval of nonzero executed position in one direction. A reversal ends one episode and starts another. Consecutive daily signals in the same direction are not separate trades.

V5 and the economic filters operate on these original episodes:

1. Identify the original BASE entry signal and execution close.
2. Evaluate the entry filter once, using information known at that original signal date.
3. If rejected, remain flat for the entire original episode.
4. If accepted under an entry-only variant, keep the original technical exit.
5. If accepted under an entry-plus-exit variant, monitor its specified risk condition. An early exit locks out the remainder of that original episode.
6. Evaluate entry again only when BASE starts a new episode.

There is no delayed entry when a rejected episode's economic conditions improve, and no re-entry after an early risk exit within the same original episode.

V3 uses a different rule: it gates the warmed technical target **every day**. It can flatten and later re-enter during an unchanged underlying technical episode when its price gate permits the target again.

### 5.5 Earlier four execution modes

The original exploratory notebook tested 792 combinations: 9 currencies × (20 strategies + 2 directional benchmarks) × 4 execution modes.

| Earlier mode | Implementation |
| --- | --- |
| Daily Continuous | Apply the previous quote's signal at each close and retain positions until targets change. |
| Daily Weekflat | Same daily execution, but force the post-close position to zero on recorded Fridays. If a Friday quote is absent, this literal weekday rule does not itself guarantee flattening at Thursday's close. |
| Weekly Flat | Use the prior week's last observed signal; enter at the current week's first quote close and be flat after its final quote close. |
| Fixed 5D | Sample a signal every five price observations, execute at the next close, and use that target for five return intervals, subject to sample-end truncation. |

These are historical exploration conventions. They are not the three dashboard comparison modes and are not all overlaid as final BASE accounts. Original exploration also uses an earlier evaluation start and older RSI entry-edge behavior.

Fixed 5D updates the target on a five-observation schedule; it does not guarantee that each position episode lasts exactly five intervals. An unchanged nonzero target can continue across consecutive blocks.

Implementation: [fx_baselines.py](src/fx_baselines.py), [fx_economics.py](src/fx_economics.py), and [fx_risk.py](src/fx_risk.py).

## 6. All 20 technical strategies and 3 benchmarks

Every strategy below produces a daily target before the shared execution delay. Lookbacks count valid price observations. No rule is selected dynamically according to past profitability.

### 6.1 Common indicators and initialization

```text
SMA_n(t) = arithmetic mean of the latest n prices, including today's price
SD_n(t)  = sample standard deviation over the latest n observations (ddof = 1)
Z_n(t)   = [P_t - SMA_n(t)] / SD_n(t)

EMA_n(t) = alpha * input_t + (1-alpha) * EMA_n(t-1)
alpha    = 2 / (n+1)

ell_t    = log(P_t / P_(t-1))
ER63_t   = abs(log(P_t / P_(t-63))) / sum(abs(ell_j), j = t-62, ..., t)

sign(x)  = +1 if x > 0; -1 if x < 0; 0 if x = 0 or x is nonfinite
```

SMA requires a fully valid window. EMA is seeded with its first finite input, not with an initial n-observation SMA. The same first-input initialization applies to MACD's signal EMA. The main price panel is required to be finite and positive; H.10 nonquotes are removed before indicator calculation.

Wilder RSI uses price changes rather than returns:

```text
gain_t = max(P_t - P_(t-1), 0)
loss_t = max(P_(t-1) - P_t, 0)

G_t = [(n-1) * G_(t-1) + gain_t] / n
L_t = [(n-1) * L_(t-1) + loss_t] / n
RSI_t = 100 - 100 / (1 + G_t / L_t)
```

The average gain/loss states are seeded with the **first price change**, then updated recursively. RSI is first emitted after at least `n` changes. When both averages are zero, RSI is 50; if average loss is zero and average gain is positive, RSI is 100. This differs from the alternative convention that seeds RSI with the average of the first `n` changes.

### 6.2 Complete strategy list

| # | Strategy ID | Signal, holding, and exit rule |
| ---: | --- | --- |
| 1 | `RSI_MR_7` | Event-based RSI mean reversion with lookback 7; use the threshold re-entry and center-exit state machine below. |
| 2 | `RSI_MR_14` | Same mean-reversion state machine with lookback 14; one of the four economic-filter core rules. |
| 3 | `RSI_MR_20` | Same mean-reversion state machine with lookback 20. |
| 4 | `RSI_MOM_7` | `RSI7 > 60`: target `+1`; `RSI7 < 40`: target `-1`; otherwise target `0`. Recalculate daily. |
| 5 | `RSI_MOM_14` | Same 60/40 direction rule using RSI14; also the RSI voter in the four-rule ensembles. |
| 6 | `RSI_MOM_20` | Same 60/40 direction rule using RSI20. |
| 7 | `MA_10_30` | `sign(SMA10 / SMA30 - 1)`. Hold the indicated direction; equality or insufficient history gives zero. A sign change can reverse the position. |
| 8 | `MA_20_60` | `sign(SMA20 / SMA60 - 1)`; a core rule, ensemble voter, and trend branch of the fixed-regime strategy. |
| 9 | `BB_MR_10` | Bollinger mean reversion using Z10: an upper excursion targets short, a lower excursion targets long, with center exits as specified below. |
| 10 | `BB_MR_20` | Same Bollinger mean-reversion state machine with lookback 20; a core rule. |
| 11 | `BB_BO_10` | Bollinger breakout using Z10: an upper excursion targets long, a lower excursion targets short, exiting on an adverse center crossing. |
| 12 | `BB_BO_20` | Same Bollinger breakout state machine with lookback 20. |
| 13 | `DONCHIAN_BO_10` | Compare today's close with the preceding 10 closes, excluding today. Strictly above the previous maximum: `+1`; strictly below the minimum: `-1`. Exit long below the channel midpoint or short above it; otherwise retain the state. |
| 14 | `DONCHIAN_BO_20` | Same Donchian rule using 20 preceding closes; also an ensemble voter. |
| 15 | `MACD_12_26_9` | `M = EMA12(P) - EMA26(P)`; `H_MACD = [M - EMA9(M)] / P`; target `sign(H_MACD)`. Recalculate daily. |
| 16 | `TSMOM_21` | Target `sign(log(P_t / P_(t-21)))`. Positive past change means long foreign currency; negative means short. |
| 17 | `TSMOM_63` | Same time-series momentum rule with lookback 63; a core rule. |
| 18 | `VOTE_EQUAL_4` | Take the sign of the sum of the four votes: RSI_MOM_14, MA_20_60, DONCHIAN_BO_20, and MACD_12_26_9. Ties are flat. |
| 19 | `VOTE_STRENGTH_4` | Use the same four voters with the fixed price-strength weights below. Score strictly above 0.2: long; strictly below -0.2: short; otherwise flat. |
| 20 | `REGIME_RULE_FIXED` | If `ER63 > 0.40`, use the warmed MA_20_60 target. If `ER63 < 0.20`, use the warmed RSI_MR_14 target. Otherwise, including both boundaries or missing ER, target zero. |

RSI momentum is flat at exactly 40 or 60. Donchian uses closes, not intraday highs/lows; equality with a channel boundary is not a breakout, and equality with the midpoint does not trigger an exit. Its breakout tests precede midpoint exit tests, so a direct reversal is possible.

### 6.3 RSI mean-reversion state machine

For lookbacks 7, 14, and 20, apply the following sequence on each valid RSI observation pair:

1. Process an existing target's exit: a long exits when current RSI is at least 50; a short exits when current RSI is at most 50.
2. Process a new entry: set `+1` if `previous RSI < 30 <= current RSI < 50`; set `-1` if `previous RSI > 70 >= current RSI > 50`.
3. Otherwise retain the resulting target.

This requires a threshold-crossing event. Remaining below 30 does not continually generate fresh long entries. The center guard prevents entry after the intended exit has already been passed: an RSI jump from 25 to 55 does not open a long; a fall from 75 to 45 does not open a short.

If either the current or previous RSI is nonfinite, preserve the existing RSI mean-reversion target. The final state machine differs from the original exploratory RSI edge behavior; those two numerical histories are kept separate.

### 6.4 Bollinger state machine

Apply these tests in order:

| Test | Mean reversion: BB_MR | Breakout: BB_BO |
| --- | --- | --- |
| `Z >= +2` | Set target `-1` | Set target `+1` |
| `Z <= -2` | Set target `+1` | Set target `-1` |
| Neither outer test applies; current target is `+1` | Exit if `Z >= 0`; otherwise hold | Exit if `Z <= 0`; otherwise hold |
| Neither outer test applies; current target is `-1` | Exit if `Z <= 0`; otherwise hold | Exit if `Z >= 0`; otherwise hold |
| Otherwise | Retain the target, including zero | Retain the target, including zero |

These are level tests, not first-crossing-only tests. The outer thresholds take precedence over center exits, allowing a direct reversal following a sufficiently large price move.

When price standard deviation is missing or no greater than `1e-12`, Z is nonfinite. The BB implementation outputs zero for that observation but retains its internal state for the next valid observation. This differs from RSI mean reversion's missing-value handling.

### 6.5 Strength-weighted voting

Let `sigma21` be the sample standard deviation of the last 21 daily log returns, and let `m20` be the midpoint of the preceding 20 closes' Donchian channel:

```text
S       = max(sigma21 * sqrt(20), 1e-7)
clip(x) = min(1, max(0, x))

w_RSI   = clip(abs(RSI14 - 50) / 25)
w_MA    = clip(abs(SMA20 / SMA60 - 1) / S)
w_DON   = clip(abs(log(P) - log(max(m20, 1e-10))) / (2*S))
w_MACD  = clip(abs(H_MACD) / [1.5 * max(sigma21, 1e-7)])

score   = sum(vote_i * weight_i) / max(sum(weight_i), 1e-10)
target  = +1 if score > 0.2
          -1 if score < -0.2
           0 otherwise
```

Missing weights become zero. A nonzero weight still contributes to the denominator when its corresponding vote is zero. These deterministic weights measure current indicator strength; they are not fitted probabilities or allocations based on historical strategy PnL.

### 6.6 Fixed-regime strategy and invalid values

`REGIME_RULE_FIXED` recomputes its branch each day using ER63. It adopts the already warmed target of the selected branch. It does not restart RSI or MA when the regime changes and can re-enter an existing branch's technical episode.

An invalid or zero ER denominator yields missing ER and a flat regime/gate target. Daily directional strategies without a valid directional indicator also produce zero. The strategy rules and economic filters have separate missing-data semantics.

### 6.7 Benchmarks

| Benchmark | Definition |
| --- | --- |
| `ALWAYS_LONG_FX` | Target `+1` throughout the permitted execution period, subject to the same execution delay and cash start. |
| `ALWAYS_SHORT_FX` | Target `-1`, earning the negative of the canonical daily simple price return. |
| `CASH` | Target `0`, no trades, no spot PnL, and no cash interest. Available in main and H.10, not in V3 technical. |

Under fixed-initial-capital accounting, always-long and always-short simple returns are exact opposites over matching dates. Under compounding, their terminal percentage returns need not be exact opposites.

Source of exact strategy definitions: [fx_baselines.py](src/fx_baselines.py), particularly `complete_signals`, `event_osc_reversion`, `bb_strategy`, `donchian`, and `fixed_execute`.

## 7. How the filters were developed and how they work

### 7.1 The documented research progression

The filters were developed as successive hypotheses about when a technical signal might conflict with its surrounding price or economic environment:

| Stage | Research question | Implemented response |
| --- | --- | --- |
| V3: price-only gates | Does mean reversion work better in low-efficiency price paths, and momentum in more directional paths? Does MA agreement help momentum? | Gate technical targets using ER63 or agreement between TSMOM and MA. |
| V5: uniform VIX filters | Is high and rising equity-market volatility an unfavorable setting for entering or retaining a mean-reversion position? | Apply the same VIX 30/3 condition to both trade directions, with entry-only and entry-plus-exit variants. |
| V7: relative-yield conflicts | Are relative government yields moving against the proposed direction? Does an adverse yield level strengthen that conflict? | Test a yield-change gate and a stricter joint yield-level/change gate. |
| V8: economic mechanisms | Can dollar confirmation, directional yield exposure during stress, or commodity-dollar conditions identify conflicting entries? | Test three separate entry conditions, their fixed OR combination, and a directional-risk exit addendum. |

All variants, including unsuccessful ones, remain in the research. The specified thresholds are carried over from the documented research; the current reconstruction and simple-interest conversion do not optimize them.

An economic explanation motivates a hypothesis. It does not establish that a threshold is optimal, a result is statistically significant, or a causal mechanism has been identified. The full original protocol archive is unavailable, so the reconstruction cannot independently certify every historical precommitment or freeze timestamp.

The four core rules for the economic-filter experiment are:

- `TSMOM_63`.
- `MA_20_60`.
- `BB_MR_20`.
- `RSI_MR_14`.

V5 covers only `BB_MR_20` and `RSI_MR_14`. V3 gates each apply to their specified price strategy. The 20-strategy comparison remains available independently of these narrower experiments.

### 7.2 Filter notation

All economic quantities below are **K values known before the signal date**, using the timing construction in Section 8.

| Symbol | Meaning | Unit |
| --- | --- | --- |
| `q` | Proposed or held trade direction: `+1` long foreign currency, `-1` long USD | Direction |
| `d` | Local 2Y government yield minus US 2Y yield | Basis points (`bp`) |
| `delta20d` | Change in that spread over 20 nonmissing native observations | bp |
| `VIX` | Known VIX level | Index points |
| `delta5VIX` | VIX change over five nonmissing native observations | Index points |
| `DXY20` | DXY price change over 20 nonmissing native observations | Percent; value `1` means `+1%` |
| `C20` | Applicable commodity price change over 20 nonmissing native observations | Percent |

### 7.3 All 14 execution/filter variants

| # | Variant ID | Applicable sample/rules | Exact behavior |
| ---: | --- | --- | --- |
| 1 | `BASE` | All displayed strategy/benchmark accounts | Original technical entry, holding, and exit; no additional gate. |
| 2 | `BB10_RANGE_GATE` | `technical`, BB_MR_10 | Use the warmed BB_MR_10 target only when `ER63 < 0.20`; otherwise target zero. |
| 3 | `BB20_RANGE_GATE` | `technical`, BB_MR_20 | Use the warmed BB_MR_20 target only when `ER63 < 0.20`; otherwise target zero. |
| 4 | `MOM63_TREND_GATE` | `technical`, TSMOM_63 | Use TSMOM_63 only when `ER63 > 0.40`; otherwise target zero. |
| 5 | `MOM_MA_CONFIRM` | `technical`, TSMOM_63 | Use TSMOM_63 only when its target equals the MA_20_60 target. Disagreement gives zero; agreement at zero remains flat. |
| 6 | `V5_ENTRY_ONLY` | `main`, BB_MR_20 and RSI_MR_14, all 9 currencies | Reject a new original episode if `VIX >= 30 AND delta5VIX >= 3`. Otherwise retain the original technical exit. |
| 7 | `V5_RISK_EXIT` | Same V5 coverage | Apply the same entry gate; for an accepted episode, exit at the next FX close after the first subsequent known occurrence of the same condition. Lock out the remainder of the episode. |
| 8 | `RATE_CHANGE_CONFLICT` | `main`, all 4 core rules, all 9 currencies | Reject if `q * delta20d <= -10 bp`. |
| 9 | `DUAL_RATE_CONFLICT` | Same core coverage | Reject if `q * delta20d <= -10 bp AND q * d <= -50 bp`. |
| 10 | `POLICY_USD` | Same core coverage | Reject if `q * delta20d <= -10 bp AND q * DXY20 >= 1%`. |
| 11 | `RISK_YIELD` | Same core coverage | Reject if `VIX >= 20 AND delta5VIX >= 3 AND q * d >= 50 bp`. |
| 12 | `COMMODITY_USD` | `main`, all 4 core rules; AUD, CAD, and NOK only | Reject if `q * C20 < 0 AND q * DXY20 >= 1%`. Use copper for AUD and Brent for CAD/NOK. |
| 13 | `ECO_COMPOSITE` | `main`, all 4 core rules, all 9 currencies | Reject if any confirmed POLICY_USD, RISK_YIELD, or applicable COMMODITY_USD component rejects. This is OR, not majority voting. |
| 14 | `RISK_YIELD_EXIT` | Same core coverage | Use RISK_YIELD at entry and monitor that same condition during an accepted episode. Exit at the next close after its first trigger; no re-entry within the episode. |

The V8 experiment's eight versions are BASE plus the seven economic variants in rows 8–14. Neither V5 nor the four V3 gates is silently added to that experiment.

### 7.4 Required inputs, economic motivation, and direction

#### RATE_CHANGE_CONFLICT

**Input:** `K_spread_d20_bp`.

**Hypothesis:** avoid a trade when relative market yields are changing against its proposed currency direction.

- Long foreign currency: reject if `delta20d <= -10 bp`.
- Long USD: reject if `delta20d >= +10 bp`.

A rising local-minus-US spread weakens the relative-yield argument for being long USD, even if the spread remains negative.

#### DUAL_RATE_CONFLICT

**Inputs:** `K_spread_d20_bp` and `K_spread_bp`.

**Hypothesis:** require both an adverse relative-yield change and an adverse relative-yield level before rejecting the trade.

- Long foreign currency: reject if `delta20d <= -10 bp AND d <= -50 bp`.
- Long USD: reject if `delta20d >= +10 bp AND d >= +50 bp`.

This is stricter than the change-only gate. A trade can fail RATE_CHANGE_CONFLICT but pass DUAL_RATE_CONFLICT when the yield level does not confirm the conflict.

#### POLICY_USD

**Inputs:** `K_spread_d20_bp` and `K_dxy_d20_pct`.

**Hypothesis:** require both a relative-yield change against the position and confirmation from the broad dollar's recent direction.

- Long foreign currency: reject if `delta20d <= -10 bp AND DXY20 >= +1%`.
- Long USD: reject if `delta20d >= +10 bp AND DXY20 <= -1%`.

The name does not mean the rule observes a central-bank policy surprise or identifies a causal policy shock. It uses government-yield and dollar-price proxies.

#### RISK_YIELD

**Inputs:** `K_vix`, `K_vix_d5`, and `K_spread_bp`.

**Hypothesis:** when risk rises, holding the higher-yield side of the currency pair may expose the position to an unfavorable adjustment.

Both directions require `VIX >= 20 AND delta5VIX >= 3`, plus:

- Long foreign currency: `d >= +50 bp`.
- Long USD: `d <= -50 bp`.

The USD can be the higher-yield side. This condition therefore does not assume that market stress always favors USD or always favors the foreign currency. The yield measure is not realized carry and does not observe investors' actual carry positions.

#### COMMODITY_USD

**Inputs:** `K_dxy_d20_pct`, plus `K_copper_d20_pct` for AUD or `K_brent_d20_pct` for CAD/NOK.

**Hypothesis:** a commodity-linked technical position may face a persistent conflict when both its commodity proxy and the broad dollar move against the proposed direction.

- Long foreign currency: reject if `C20 < 0 AND DXY20 >= +1%`.
- Long USD: reject if `C20 > 0 AND DXY20 <= -1%`.

The commodity sign test is strict: an exactly zero commodity change does not reject. The rule is not applied to EUR, GBP, NZD, JPY, CHF, or SEK. In particular, no dairy proxy is invented for NZD.

Copper and Brent are limited proxies. They are not export-weighted national commodity baskets, and the rule does not distinguish supply shocks from demand shocks or determine whether a past move is already fully priced.

#### ECO_COMPOSITE

**Inputs:** the union of the required fields for POLICY_USD, RISK_YIELD, and the applicable COMMODITY_USD component.

**Hypothesis:** allow any confirmed economic conflict to veto the original entry, then evaluate the trade-off between avoiding losses and deleting profitable trades.

```text
ECO_COMPOSITE = POLICY_USD OR RISK_YIELD OR applicable COMMODITY_USD
```

It does not include RATE_CHANGE_CONFLICT, DUAL_RATE_CONFLICT, or V5 as additional components. It has no early-exit addendum. An accepted composite-filter trade follows its original technical exit.

#### RISK_YIELD_EXIT

**Inputs:** the same three fields as RISK_YIELD, read at the original entry signal and again at each eligible monitoring date.

**Hypothesis:** a conflict may develop after an initially acceptable entry. The exit addendum tests whether reacting to that later information reduces subsequent losses.

Entry is identical to RISK_YIELD. Following acceptance, monitor from the actual entry close. If the condition first triggers on quote date `t`, exit at close `t+1`, including the price return earned up to that close. The remaining original episode stays flat.

If the next execution close would coincide with the original technical exit, it is not counted as a separate early exit. No information from subsequent news or the eventual outcome is inserted into the earlier decision.

#### How V5 differs

V5 uses only `K_vix` and `K_vix_d5`. Its threshold is **30/3**, not the later RISK_YIELD rule's **20/3**. It does not inspect trade direction or yield spreads. Both long and short entries use the same condition.

### 7.5 KEEP, REJECT, UNKNOWN, and NOT_APPLICABLE

Each individual economic component first checks that **all of its own required inputs are finite**:

| State | Meaning | Trading consequence |
| --- | --- | --- |
| `REJECT` | All required inputs are known and the rejection condition holds. | Reject the entry or, for a risk-exit monitor, schedule the specified next-close exit. |
| `KEEP` | All required inputs are known and the rejection condition does not hold. | Follow the original technical trade. |
| `UNKNOWN` | At least one required input is missing, stale, or invalid. | Preserve the original entry; during monitoring, do not trigger a risk exit from this component. |
| `NOT_APPLICABLE` | Commodity filtering is not defined for this currency. | Do not use that commodity component. |

For an individual AND condition, the implementation checks input completeness before evaluating the predicate. If one known term would be false but another required input is missing, the recorded state is still UNKNOWN rather than KEEP. UNKNOWN is not evidence of calm conditions.

The OR combination applies this priority:

```text
Any component REJECT  -> REJECT
Else any UNKNOWN     -> UNKNOWN
Else                 -> KEEP
```

A known component can therefore reject even when another component is UNKNOWN. A nonapplicable commodity component does not prevent the other components from making a decision.

Inclusive economic thresholds use an absolute floating-point tolerance of `1e-10` in the input's numerical units. This preserves intended equality cases after yield subtraction; it is not parameter tuning. The commodity sign test remains strict. Technical thresholds follow their stated strict or inclusive inequalities.

### 7.6 Three worked examples

These are hypothetical known-at-signal inputs illustrating the formulas, not additional historical observations.

**Example A — CAD long during an oil/dollar conflict.** Set `q=+1`, Brent20=`-8%`, and DXY20=`+1.2%`. The signed commodity change is negative and signed DXY change exceeds 1%, so COMMODITY_USD rejects the entire original CAD-long episode. For `q=-1`, the signed values become `+8%` and `-1.2%`, so the same observations do not reject that short-CAD trade.

**Example B — SEK short when USD is the higher-yield side.** Set `q=-1`, `d=-120 bp`, VIX=`23`, and delta5VIX=`+4`. Then `q*d=+120 bp`, and all RISK_YIELD conditions hold. RISK_YIELD rejects entry. If these inputs first become known after an accepted entry, RISK_YIELD_EXIT exits at the next quote close. V5 does not trigger because VIX is below 30.

**Example C — GBP yield change and dollar confirmation without a sufficiently adverse yield level.** Set `q=+1`, delta20d=`-12 bp`, `d=-40 bp`, and DXY20=`+1.3%`. RATE_CHANGE_CONFLICT rejects. DUAL_RATE_CONFLICT keeps the trade because the spread has not reached `-50 bp`. POLICY_USD rejects because both of its terms hold. ECO_COMPOSITE therefore rejects even if missing VIX inputs leave its RISK_YIELD component UNKNOWN.

Implementation: [fx_legacy.py](src/fx_legacy.py), [fx_risk.py](src/fx_risk.py), and [fx_economics.py](src/fx_economics.py).

## 8. Economic inputs and information timing

### 8.1 E, K, and A clocks

| Clock | Definition | Role |
| --- | --- | --- |
| E: contemporaneous end-date | Conditions associated with the return-ending date; potentially includes that date's observations. | Retrospective explanation, not an earlier trade's known information. |
| K: known before signal | A source observation strictly earlier than the FX signal date and no more than seven calendar days old. | Actual entry gates and risk-exit monitoring. |
| A: applied interval | K shifted forward by two FX quote observations, corresponding to the signal-to-earned-return delay. | Conditional daily-return attribution for the original technical positions. |

The current reconstruction directly builds K inputs and their observation dates. The daily VIX-state attribution uses `shift(2)` to construct A. E is a descriptive concept here; the complete old frozen E/K/A archive was not supplied and is not claimed to have been recovered.

A daily attribution state must not be confused with the K observation fixed at an episode's original entry. Entry-only filters do not reevaluate that entry every day.

### 8.2 All seven final numerical inputs

| Field | Definition and unit | Native source | Additional change-span limit |
| --- | --- | --- | --- |
| `K_vix` | VIX level, in index points | `5_Equities.VIX` | None beyond the common source-age limit |
| `K_vix_d5` | Current VIX minus its value five previous valid native observations earlier; index points | `5_Equities.VIX` | Five steps must span at most 12 calendar days |
| `K_spread_bp` | `(local 2Y - US 2Y) * 100`, in bp; source yield `1` means `1%` | `2_Rates`; Germany proxies EUR | None beyond the common source-age limit |
| `K_spread_d20_bp` | Current spread minus its value 20 previous valid spread observations earlier; bp | The native local-minus-US spread series | 20 steps must span at most 45 calendar days |
| `K_dxy_d20_pct` | `100 * (DXY / DXY_20_observations_earlier - 1)` | `1_FX_Spot.DXY` | At most 45 calendar days |
| `K_brent_d20_pct` | Brent percentage change over 20 valid native observations | `8_Commodities.BRENT` | At most 45 calendar days |
| `K_copper_d20_pct` | Copper percentage change over 20 valid native observations | `8_Commodities.COPPER_LME` | At most 45 calendar days |

All seven fields share the strictly-prior-date and maximum-seven-day age requirements. The 12/45-day change-span limits are separate from this seven-day source-age limit.

The spread is computed from same-date local and US yield observations when both are valid. Its 20-step change then uses the resulting nonmissing native spread sequence.

### 8.3 Construction order

1. Convert invalid native observations to missing values, retain each series' valid observations, and sort by date.
2. Compute five- or 20-observation changes on that native sequence, before mapping it onto the FX quote calendar.
3. Reject a change whose two endpoints span more than 12 or 45 calendar days, respectively. A level can remain valid when its change is invalid.
4. For each FX signal date, use a backward as-of match with same-date matching prohibited and an age tolerance of seven calendar days.
5. Preserve missing values. If the latest matched source observation has an invalid change, do not skip it and search for an older observation with a conveniently complete change.

Different fields may refer to different source dates. The retained date fields are:

```text
K_vix_obs_date
K_rate_obs_date
K_dxy_obs_date
K_brent_obs_date
K_copper_obs_date
```

The source data do not contain complete release timestamps, closing timezones, or revision vintages. Strictly prior observation-date matching is a conservative research convention, not certification that every value was available in real time at the stated trading close.

### 8.4 VIX diagnostic states

The daily attribution analysis combines three VIX levels with three five-observation changes:

| Dimension | States |
| --- | --- |
| Level | L: VIX `<20`; M: `20 <= VIX < 30`; H: VIX `>=30` |
| Change | U: change `>=+3`; D: change `<=-3`; F: strictly between `-3` and `+3` |

This produces nine labels such as `L_U` and `H_D`; missing inputs produce UNKNOWN. Attribution keeps the original account's returns and identifies the applicable state. It does not open a new funded account every time the state changes.

VIX is an equity-option volatility index, not an FX volatility measure or a panic probability. Government-yield differences are not policy rates or realized carry. DXY includes some of the currencies being studied, so it is a dollar-confirmation variable rather than an independent causal instrument.

Extra one-/two-quote delays and stricter stale-input sensitivity were suggested in earlier research discussions but were not executed as V8 tests. They should not be represented as completed checks.

Sources: [fx_economics.py](src/fx_economics.py), especially `_align_native` and `rebuild_economic_known`, and [fx_risk.py](src/fx_risk.py), especially `conditional_daily`.

## 9. Dashboard controls and exports

### 9.1 Controls

| Control | Meaning |
| --- | --- |
| Research sample | Select main, technical, or H.10. Changing the sample resets its date range and suggested selection. |
| Currency against USD | Choose the currency for filter or strategy comparisons. |
| Technical strategy | Choose the underlying strategy for filter or currency comparisons. |
| Execution / filter version | Choose the common variant for strategy or currency comparisons. |
| Suggested comparison | Load a fixed display preset. It does not choose the best-performing accounts in the selected window. |
| All available / Clear | Select all applicable paths or remove the current selection. |
| Selection checkboxes | Add or remove individual paths. Greyed choices are untested or inapplicable combinations, not zero-return accounts. |
| From / Through | Include actual return-ending quote dates within the chosen bounds. Nonquote boundary dates use the quotes that actually fall inside the interval. |
| Date presets | Select full or partial calendar periods. These are display windows, not new training or validation splits. |
| Chart measure | Display PnL in USD, cumulative return as a percentage of initial capital, or account wealth in USD. |
| Rebase selected window / Reset window PnL to zero | Start the displayed window at USD 100,000 using multiplication in the compound edition or addition/subtraction in the simple edition. |
| Original account path / Original simple account path | Preserve the account's original equity/PnL level while plotting only the selected dates. The metrics still describe the selected window. |
| Annual return comparison | Expand the annual table. Partial first/last years follow the selected dates. |
| What the selected rules do | Read the definitions for the currently selected rules and variants. |

The legend identifies colors; selection is controlled by the checkboxes. Hovering shows dated values. Where an account lacks a quote on another account's date, any last observed value is labeled with its actual date; no new quote or trade is manufactured.

The chart includes the preceding available quote as its window anchor. A line joining quotes does not imply that additional observations were inserted.

### 9.2 SVG exports

**Download chart SVG** saves the current PnL/equity chart and drawdown chart as vector graphics, including the selection and date labels. The simple-interest SVG identifies fixed initial capital and the drawdown unit in percentage points.

### 9.3 Daily CSV exports

The CSV reflects the current account selection, date window, and display normalization. It includes actual quote dates, excluding the extra chart anchor. Dates are the union across selected accounts; an account without a quote on a date has blank cells.

Each account-specific column is prefixed by its account ID followed by `__`. The compound edition exports:

```text
display_pnl_usd
display_equity_usd
window_drawdown_pct
```

The simple-interest edition exports:

| Column suffix | Meaning |
| --- | --- |
| `display_pnl_usd` | Displayed equity minus USD 100,000; affected by the normalization choice. |
| `display_equity_usd` | Window-reset or original-path equity, as selected. |
| `window_drawdown_initial_capital_pp` | Window drawdown divided by fixed initial capital, in percentage points. |
| `window_drawdown_peak_equity_pct` | Window-reset equity drawdown as a percentage of its running peak. |
| `daily_pnl_usd` | Fixed initial capital times the day's strategy return. |
| `daily_simple_return_pct` | That strategy return times 100. |

Only the two display-level columns change when switching between original and window-reset paths. Daily PnL, daily simple return, and the window drawdown definitions do not change.

To export the entire original account path, choose **both** the full date range and **Original account path**. Choosing the original-path setting alone still exports only the selected date window.

The CSV is a daily chart export. It is not a complete signal table or trade ledger. Trade and episode-level evidence is available in the files listed in Section 12.

### 9.4 Empty results and undefined values

No paths appear when the selection is empty, the date bounds are invalid, or the window contains no quotes. A dash in a metric means it is undefined; in the original V3 archive it can also mean the trade ledger was not supplied. It does not mean zero win rate.

H.10 is short and some strategies have no completed trades. Both the sample length and trade count matter when reading its statistics.

## 10. Metrics, trade cohorts, and interpretation

Account metrics and trade-quality metrics use different observation sets:

- **Account/window PnL** uses all strategy returns ending inside the window, including returns from positions opened before it and positions still open at its end.
- **Trade quality** uses trades entered inside the window that are actually closed by the window end and have earned at least one return observation.

| Metric | Definition |
| --- | --- |
| Window PnL | Simple edition: `C0 * sum(window u_t)`. Includes marked-to-market returns from open and carried-in positions. |
| Simple return | Window PnL divided by `C0`, multiplied by 100. |
| Annualized simple | Window simple return divided by the actual calendar years between its anchor and final quote. |
| Max DD / initial capital | Minimum window peak-to-trough difference divided by `C0`, in pp. |
| Max DD / peak equity | Minimum percentage decline from the running peak of window-reset equity. |
| Closed trades | Entry inside the date window, genuinely closed by its end, with at least one earned-return observation. |
| Win rate | Number of profitable eligible closed trades divided by all eligible closed trades. Flat trades remain in the denominator. |
| Payoff | Average winning trade return divided by the absolute average losing trade return. Undefined if either group is absent. |
| Open / censored | Entries inside the window that remain open at sample end or actually exit after the selected window ends. |
| Carry-in | Trades entered before the window and not exited before its start. Their window returns count in PnL, but the trades are excluded from that window's entry cohort. |
| Small cohort | Display flag when fewer than 20 eligible closed trades are available. It does not alter the strategy. |

A trade is classified as flat when its absolute return is at most `1e-12` in decimal-return units, equivalent to `1e-10` percentage points. An unclosed final trade is not included in the closed-trade win-rate denominator even though its earned PnL remains in account equity.

Example: a position opened in December and closed in February contributes its January daily returns to January PnL. It is a January carry-in and does not enter January's new-entry trade-quality cohort. In a December-only window, it is an open/censored entry. Its entire trade return must not be assigned to December's annual return.

For a full simple-interest account, the sum of all closed-trade PnL plus the earned PnL of any final open trade reconciles to total account PnL. Summing only a partial window's eligible completed entry-cohort trades generally does not reconcile to window PnL, because that selection omits carry-in and censored positions.

### Other research fields

The original research tables also contain daily-return and exposure diagnostics:

- `sharpe_252`: mean daily strategy return divided by its sample standard deviation, multiplied by `sqrt(252)`, including zero-return flat days and using a zero risk-free rate. Undefined when the denominator is effectively zero.
- Annualized volatility: daily strategy-return sample standard deviation times `sqrt(252) * 100`.
- `active_pct`: fraction of dates with a nonzero position earning the return.
- Long/short fractions: based on the position actually earning each daily return, not merely that day's newly observed signal.
- Turnover: absolute executed-position change; `0 -> +1` is one unit and `+1 -> -1` is two.

These fields are not all displayed in the simple-interest dashboard. Their definitions do not imply that compounding, costs, or capital allocation have been added.

A filter can raise win rate by deleting many trades while removing a few large winners. An early exit can avoid a loss or miss a rebound. Read trade count, retention, payoff, PnL, and the path through time together.

## 11. Historical screening results and evidence limits

### 11.1 Original compound-account screen

The reconstruction applies the following explicit historical improvement thresholds relative to the corresponding BASE account:

| Criterion | Requirement |
| --- | --- |
| Win rate | Improve by at least 2 percentage points |
| Payoff ratio | Improve by at least 10% relative |
| Cumulative return | Improve by at least 5 percentage points |
| Maximum drawdown | Deteriorate by no more than 2 percentage points |
| Trade retention | Retain at least 50% of eligible closed BASE trades |
| Closed-trade count | At least 30 eligible closed trades |
| Later-history consistency | From 2015 onward, win rate, payoff, and return must each be no worse than BASE |
| Calendar consistency | Return must improve in at least 3 of the 5 specified calendar blocks |

The calendar blocks are 2005–2009, 2010–2014, 2015–2019, 2020–2024, and 2025–May 2026.

These are **evaluation conditions applied to results**, not live trading triggers. They do not change an entry, exit, target weight, or filter threshold. The six full-history requirements are documented in the delivered notebook; later-history and calendar conditions are explicitly reconstructed because the original frozen protocol is unavailable.

Five compound-account identities pass the reconstructed historical screen:

| Currency | Base strategy | Filter | Original compound cumulative return |
| --- | --- | --- | ---: |
| GBP | RSI_MR_14 | POLICY_USD | 29.78% |
| GBP | RSI_MR_14 | ECO_COMPOSITE | 29.78% |
| CAD | BB_MR_20 | COMMODITY_USD | 42.97% |
| CAD | BB_MR_20 | ECO_COMPOSITE | 40.56% |
| SEK | RSI_MR_14 | RISK_YIELD_EXIT | 16.73% |

The two GBP variants produce the same result and are not independent discoveries. These five rows are historical screening outcomes, not a proof of statistical significance or future performance.

### 11.2 What changes under simple interest

The five historical labels have **not** been transferred to the simple-interest accounts, and the compound screen has not been represented as a new simple-interest screen.

Changing from products to sums can change total returns, individual trade signs, win rates, payoff, drawdown, and concentration. Original compound log-wealth contribution shares also cannot be reused as simple dollar-PnL contribution shares.

The new accounting convention supplies another consistent way to inspect the same historical positions. It does not supply new market observations.

### 11.3 What the evidence establishes

The current repository contains reconstructed source code. It rebuilds prices, signals, economic features, positions, trade ledgers, and account results, then compares them with independently retained workbook exports, notebook outputs, and the original HTML archive.

The supplied package did not include the complete original source tree, frozen engines, original protocol files and freeze timestamps, or the rejected rolling-selection snapshot. Numerical agreement does not recover those missing files or prove the original historical decision process was followed exactly.

The main history, its 2015-onward review segment, and the H.10 extension have already been examined. Reproduction does not create an untouched out-of-sample test. Transaction costs, carry, financing, and cash interest remain omitted. Economic observation dates do not provide complete publication/vintage evidence.

Old data-package prose referring to a complete frozen archive describes the earlier package design. For the current workspace, use the [root README](README.md), source code, and validation records to determine what is actually present.

Sources: [screen specification and implementation](src/fx_analysis.py), [original account acceptance table](reproduced/v8/tables/account_acceptance.csv), and [original validation summary](reproduced/notebook_validation_summary.json).

## 12. Files, reproduction, and validation

### 12.1 Main files

| File | Purpose |
| --- | --- |
| [FX_PnL_Explorer_README.md](FX_PnL_Explorer_README.md) | This English Markdown reference. |
| [FX_PnL_Explorer_Simple.html](FX_PnL_Explorer_Simple.html) | Offline dashboard using fixed-initial-capital additive PnL. |
| [FX_PnL_Explorer.html](FX_PnL_Explorer.html) | Preserved original compound dashboard and independent numerical reference. |
| [FX_Technical_Research_Complete.ipynb](FX_Technical_Research_Complete.ipynb) | Main reconstructed research notebook. |
| [README.md](README.md) | Repository execution instructions and reconstruction boundary. |
| [data_manifest.json](FX_Research_Datasets/data/catalog/data_manifest.json) | Data provenance, coverage, and recorded file hashes. |

### 12.2 Source modules

| File | Responsibility |
| --- | --- |
| [src/fx_baselines.py](src/fx_baselines.py) | Raw Excel reading, all indicators and technical rules, execution, compound-account metrics, and trade ledgers. |
| [src/fx_economics.py](src/fx_economics.py) | Native economic changes, K alignment, economic filters, episode locking, and phase/calendar analysis. |
| [src/fx_legacy.py](src/fx_legacy.py) | V3 cash-reset stages and separate H.10 replay. |
| [src/fx_risk.py](src/fx_risk.py) | V5 filters and applied-clock VIX-state attribution. |
| [src/fx_analysis.py](src/fx_analysis.py) | Original compound episode effects, concentration, explicit historical screening, and complete case evidence. |
| [src/fx_evidence.py](src/fx_evidence.py) | Decode retained HTML and compare paths, metrics, features, and original notebook evidence. |
| [src/fx_research.py](src/fx_research.py) | Notebook-facing functions and integrated original research outputs. |
| [src/fx_simple.py](src/fx_simple.py) | Rebuild unrounded daily returns, simple-interest paths, simple trade returns, and numerical validation for all 912 displayed accounts. |
| [build_simple_dashboard.py](build_simple_dashboard.py) | Package the simple dataset with additive chart, window, annual, metric, and export calculations. |

### 12.3 Rebuild the simple-interest edition

Use Python 3.10 or later with the repository requirements installed:

```powershell
python -m pip install -r requirements.txt
python build_simple_data.py
python build_simple_dashboard.py
```

The first script rebuilds data from the research engines and saves the simple-interest dataset. The second packages that dataset into the self-contained HTML dashboard. The original compound HTML is not overwritten.

Daily returns are taken from the reconstructed engines at floating-point precision. They are not inferred from the original archive's rounded equity values. The build first confirms that the reconstructed positions reproduce the original compound paths, then applies fixed-capital additive accounting.

### 12.4 Simple-interest outputs

All numerical outputs below are under `reproduced/simple_interest/`:

| File | Contents |
| --- | --- |
| [account_metrics.csv](reproduced/simple_interest/account_metrics.csv) | All 912 accounts' simple returns, PnL, linear annualization, two drawdown definitions, and trade-quality metrics. |
| [explorer_payload.json.gz](reproduced/simple_interest/explorer_payload.json.gz) | Full calendars, daily returns, equity paths, per-account simple trade records, rule metadata, and source hashes. |
| [reconciliation.csv](reproduced/simple_interest/reconciliation.csv) | Per-account compound-reference, simple daily-increment, terminal-PnL, and trade checks. |
| [validation_summary.json](reproduced/simple_interest/validation_summary.json) | Data-build validation status, coverage, maximum errors, and provenance. |
| [dashboard_build.json](reproduced/simple_interest/dashboard_build.json) | Original/dashboard/payload/builder hashes and the display-accounting convention. |
| [browser_validation.json](reproduced/simple_interest/browser_validation.json) | Browser checks of accounting, controls, annual additivity, actual CSV/SVG downloads, and the existing companion HTML guide. |

In the compressed payload, each account's `trades` is an array of rows:

```text
[entry_epoch_day, exit_epoch_day, simple_return_pct,
 holding_observations, closed_flag, position]
```

Epoch days count from 1970-01-01. An open trade has `closed_flag=0` and `exit_epoch_day=-1`; its recorded simple return includes earned PnL through the end of the available sample. When defined, `quality_full.win` is a fraction between 0 and 1, whereas the dashboard displays win rate as a percentage. It is `null` when there are no eligible closed trades.

The simple-interest reconstruction covers **3,041,889 daily observations** across the 912 account paths. It reconciles the boundaries of 678 retained original trade ledgers and supplies 234 newly reconstructed V3 ledgers. Those V3 records are explicitly labeled as new reconstruction evidence.

### 12.5 Optional browser verification

```powershell
python verify_simple_dashboard.py
```

This check uses an installed Chrome/Edge browser with an isolated temporary profile and the Python `websocket-client` package. These are verification dependencies; they are not required merely to open the dashboards or rebuild the numerical data.

The browser checks include hand-calculated accounting cases, window anchors, trade cohorts, annual-return additivity, all 912 account endpoints, dataset and comparison controls, invalid-date handling, and actual CSV/SVG downloads. Its saved hashes identify the exact HTML files that were tested. It also checks the pre-existing HTML guide's search/print interactions; that check is separate from this Markdown document.

### 12.6 Original research reproduction and evidence

```powershell
python run_research.py
```

This executes the main research notebook and writes its executed copy and original compound-research outputs under `reproduced/`. It is not the simple-interest build command.

Existing files such as `fixed_baseline_trade_ledger.csv.gz`, `v3_trade_ledger.csv.gz`, `v5_trade_ledger.csv.gz`, and `v8/tables/trade_ledger.csv.gz` belong to the original compound research. Do not substitute their trade-return values for the simple returns stored in the new payload.

The V8 episode decisions and [case evidence](reproduced/v8/tables/case_evidence.csv.gz) retain accepted, rejected, helpful, and harmful outcomes. [Saved notebook outputs](reference/notebook_saved_outputs.json) retain historical textual/table evidence and the original notebook hash; they do not replace missing original protocols.

When sharing a chart, table, or trade result, identify its dataset, date window, strategy, variant, and accounting convention. These choices determine what the displayed number means.
