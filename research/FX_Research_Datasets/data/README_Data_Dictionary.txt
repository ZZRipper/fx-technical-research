FX technical timing research - supplied and derived data
======================================================

raw/ preserves the two supplied Excel files byte-for-byte, the earliest supplied
notebook snapshot available in this workspace, and the rejected rolling-selection
snapshot. The latter is historical provenance only; it is not run as a strategy.

clean/fx_spot_daily.csv contains 62,010 rows: 6,890 dates for nine foreign currencies,
2000-01-03 to 2026-05-29. price_USD_per_foreign is the canonical signal/return price.
EUR/GBP/AUD/NZD native quotes are already USD per foreign; JPY/CHF/CAD/NOK/SEK
native market quotes require inversion. The PIT workbook already stores the
canonical prices; do not invert them a second time. Source log returns are audited
but are not used as simple account returns. No cross-sectional rank or fwd_excess
labels are used. All final20rules are recomputed from canonical prices.

clean/features_*.csv are lossless numeric/date exports of all 11feature workbook
sheets. Missing source cells remain missing; they are not indiscriminately filled.
The feature workbook has 226fields. Its date range ends 2026-05-29; it adds useful
variables but no new time sample. Its hash matches the upstream source recorded
in the PIT workbook. A later upload name does not create out-of-sample data.

Features2_Rates stores market government yields in percent (1 means 1%); the 2Y
spread is local minus US, multiplied by 100 to basis points. Germany proxies EUR.
These are not central-bank policy rates or realized carry. VIX is an equity
volatility index in points, not FX volatility. Copper/Brent changes are imperfect
commodity proxies. Stored native-quote technical indicators are references only:
research rules use canonical-quote Wilder RSI and other explicitly coded formulas.
Some named inflation/commodity series had unresolved unit/identity issues; they
are archived but not inputs to final filters. Source vendor definitions, release
timestamps, closing timezones and revision vintages are not fully supplied.

clean/economic_known_inputs_rebuilt.csv.gz reconstructs the seven final filter
inputs. Native 5/20 observation changes are computed before backward as-of alignment.
Each K observation is strictly earlier than its FX signal date, at most 7 calendar
days old. Missing required inputs remain UNKNOWN. Full E/K/A data and diagnostics
are retained in frozen/analysis_v5/. K alignment is not strict vintage certification.

clean/h10_official_daily.csv is a separate-source historical extension and warmup
from Federal Reserve H.10 pages. The source data record includes native quotes,
canonical prices, ND (nonquote) flags, URL and line references. Quote gaps are not
filled to manufacture trades. Evaluation starts 2026-06-01 and most series end
2026-10-16; inspect external_coverage.csv for pair-specific coverage. The extension
was a prior frozen-technical-rule replay and is now observed. It does not provide
new OOS evidence for later economic filters without matching macro data.

catalog/dataset_inventory.csv and data_dictionary.csv provide datasets, variables,
units, roles, coverage and missingness. data_manifest.json contains raw/clean/frozen
file hashes. Dataset date counts are at their own frequency; repeated dates across
currencies are expected and not duplicate(pair,date)records.

frozen/ preserves the exact input/result paths from prior stages. V3 controlled
variants/H.10, V4 time-currency, V5 daily risk/currency traits, V6 trade quality,
V7 rate gates andV8 economic filters are available. Different stage starts and
account resets are explicitly recorded; do not splice their curves.

The notebook reruns raw-price baselines and raw economic features, reconciles
stored histories, then executes the unchanged frozen filter engine into a separate
reproduced/ folder. Frozen files remain read-only. All values are zero-cost spot
research PnL; carry, financing and cash interest are omitted.
