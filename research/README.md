# FX 技术指标研究：源码重建版

本目录现在包含可运行的研究源码。计算从交付的原始 Excel 开始；现有 CSV、离线 HTML 和 notebook 的旧输出只作为独立核对证据。

## Dashboard 说明与固定本金单利版

说明与交互页面入口如下；HTML dashboard 可在现代浏览器中离线打开，无需启动服务器：

- [FX_PnL_Explorer_README.md](FX_PnL_Explorer_README.md)：完整英文 Markdown 说明，覆盖全部策略、filter 建立逻辑与输入阈值、持仓/退出、信息时序、单利/复利统计和复现方法。
- [FX_PnL_Explorer_README.html](FX_PnL_Explorer_README.html)：详细中文使用与方法说明，逐项解释 20 条技术策略、3 个基准、全部过滤版本、阈值、数据时序、入场/持仓/退出、交易段锁定、统计口径与操作方式。
- [FX_PnL_Explorer.html](FX_PnL_Explorer.html)：保留的原复利 dashboard，也是源码重建的独立核对证据。
- [FX_PnL_Explorer_Simple.html](FX_PnL_Explorer_Simple.html)：新增固定初始本金单利 dashboard，包含相同的 912 个账户。

单利口径经确认采用固定初始本金敞口：`daily_pnl = 100000 × held_position × spot_simple_return`，`equity = 100000 + sum(daily_pnl)`。盈利不增加后续名义敞口，亏损不减少后续名义敞口；这也不是每笔交易固定外币数量。信号、过滤、成交和退出时序不变。每日收益直接从重建策略取得，不从归档中已经舍入的净值反推。

单利交易收益为该交易持仓区间的日策略收益之和，胜率和盈亏比随之重算。年度收益可以相加；所选日期窗口以减去前一可用报价日净值的方式归零。简单年化是累计单利收益除以日历年数，不是 CAGR。主回撤为峰谷损失占初始本金的百分点（pp），表格同时提供窗口重置权益的峰值回撤百分比。V3 单利版还补充了原 HTML 未提供的重建逐笔交易统计。原复利的历史筛选结果不自动移用于单利账户。

重新生成单利数据和页面：

```powershell
python build_simple_data.py
python build_simple_dashboard.py
```

产物保存在 `reproduced/simple_interest/`：`account_metrics.csv` 为全部单利指标，`reconciliation.csv` 为逐账户核对，`validation_summary.json` 为数据验证，`explorer_payload.json.gz` 为页面数据，`dashboard_build.json` 记录页面与原档哈希。构建先验证相同持仓产生的复利路径与原 HTML 的全部 912 条曲线一致，再检查单利日增量、终值与交易段盈亏。原复利 HTML 和原有研究结果不被覆盖。

`verify_simple_dashboard.py` 提供额外的离线浏览器检查，覆盖手算例子、日期窗口、年度加总、交易统计、912 条路径、筛选控件及 CSV/SVG 导出；需要本机 Chrome/Edge 和 Python `websocket-client`。这不是打开 HTML 或重建数据的依赖。

## 运行

在此目录打开终端，使用 Python 3.10 或更高版本：

```powershell
python -m pip install -r requirements.txt
python run_research.py
```

已安装 NumPy、pandas、matplotlib 时可直接运行第二条命令。Excel 读取使用 Python 标准库，不需要 Excel、openpyxl、Jupyter 或联网。也可以用 Jupyter 打开 `FX_Technical_Research_Complete.ipynb`，依次执行所有单元。

运行会产生：

- `reproduced/FX_Technical_Research_Complete.executed.ipynb`：含新执行输出的副本；源 notebook 不被运行器覆盖。
- `reproduced/notebook_validation_summary.json`：账户数量、核对状态与源码哈希。
- `reproduced/fixed_baseline_metrics.csv`、`fixed_baseline_trade_ledger.csv.gz`：207 个固定账户及逐笔账本。
- `reproduced/v8/tables/`：288 个经济过滤账户的指标、交易账本、逐日曲线、阶段与日历分段结果。
- `reproduced/v3_*`、`v5_*`、`h10_*`：早期技术门控、VIX 过滤与独立 H.10 回放的重算结果及核对记录。
- `reproduced/v8/tables/episode_concentration.csv`、`account_acceptance.csv`、`case_evidence.csv.gz`：集中度、明确条件的历史筛选及完整逐段证据。
- `notebook_figures/`：重新生成的研究图。

任何核对失败都会中止执行并返回非零退出码；失败单元保留在执行副本中。运行会覆盖上述生成产物，不修改原始 Excel、参考 CSV、PDF 或 HTML。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| `src/fx_baselines.py` | Excel 解析、20 条技术规则、四种原始执行方式、固定账户、交易账本与指标 |
| `src/fx_economics.py` | 七个经济输入的严格前日对齐、八种过滤版本、交易段锁仓及风险退出 |
| `src/fx_evidence.py` | 解码离线 HTML，独立核对每日净值、指标、时间对齐与保留证据 |
| `src/fx_legacy.py` | V3 三阶段技术门控和独立 H.10 日历回放 |
| `src/fx_risk.py` | V5 VIX 入场/退出过滤与风险状态收益归因 |
| `src/fx_analysis.py` | 交易段效应、集中度、历史筛选与完整经济输入证据 |
| `src/fx_research.py` | notebook 公共接口、诊断表与输出整合 |
| `run_research.py` | 无 Jupyter 的逐单元执行器，记录输出并保存图表 |

所有回测使用简单收益复利，初始资金 100,000 美元，每日 1 倍有符号外币敞口。信号、成交与首次持仓收益分属相邻三个报价日；未平仓交易按期末市值计量。交易成本、carry、融资和现金利息仍为零。

经济输入先在原生非缺失序列计算变化，再严格向后匹配；来源日必须早于信号日且最多相差七天。缺失保持 UNKNOWN。涉及等号的十进制阈值使用 `1e-10` 绝对浮点容差，以免恰好 10 个基点的变化被二进制舍入到阈值外；这不是参数调优。

## 原始证据与重建边界

交付包缺少原 `src/`、完整 `data/frozen/` 和被拒绝的 rolling-selection 历史快照。因此新增实现是根据保留代码、规则说明和结果重建的源码，不声称恢复原始文件字节、版本历史或冻结时刻。

原主 notebook 的文本与 HTML 表格输出保存在 `reference/notebook_saved_outputs.json`，同时记录原 notebook 的 SHA-256。这份证据不能替代原来未提供的完整表格或历史协议。页面中的静态报告图仍可从原 PDF 查看。

V3、V5、H.10、交易段集中度和经济输入证据均重新计算。历史筛选使用 notebook 已明示的六项门槛，并明确采用后期胜率、盈亏比、收益不恶化及至少三个日历区间改善的条件；其定义保存在 `reconstructed_screen_specification.json`。它会与旧筛选结果比较，但不声称恢复缺失协议的原始决策过程。新增完整交易段证据表保留全部成功、失败和被拒绝交易，原案例叙述仅作为历史解释。

集中度的有利/有害交易计数将绝对对数财富效应不超过 `1e-12` 的修改单列为近零，同时保留未经容差处理的符号计数。因此 GBP 的一笔浮点近零交易不会被误解为实质收益贡献；收益曲线、贡献份额和去掉最大有利修改后的收益不受影响。

数据清单中的 `FX_Features_Book(8).xlsx` 对应实际文件 `FX_Features_Book.xlsx`；加载器核对内容哈希，并兼容这次重命名。缺失的历史快照会在核对表和总结中明确列出，而不会被记作验证通过。

主样本已经用于研究；新的实现和数值复现不产生新的样本外证据。H.10 仍作为独立报价来源处理，不与主样本拼接。

## 从 PnL 观察出发的过滤研究

`run_observation_research.py` 对全部 20 条技术规则、9 个币种的 **180 个未加经济过滤的固定账户**建立观察层。固定初始本金单利是主要口径，同一持仓的复利路径作为参照；旧策略、过滤规则和 dashboard 不受影响。

```powershell
python -B run_observation_research.py
python -B -m unittest discover -s tests -v
```

结果单独保存在 `reproduced/observation_v1/`：

- `observation_report.html`：中文离线观察报告，包含全部规则的曲线、年度收益、有效/失效窗口、回撤和交易证据。
- `pnl_atlas.pdf`、`figures/`：20 页图册和 20 张原图，每条规则一页、每页九个币种。
- `account_summary.csv`、`calendar_periods.csv`、`rolling_windows.csv`、`drawdown_episodes.csv`：全账户摘要、年度/半年度表现、不重叠的 252 报价日诊断窗口和完整回撤段。
- `enriched_trade_ledger.csv.gz`：逐笔单利/复利收益、收盘观测的最大浮盈/浮亏、回吐，以及信号时可知的价格与经济特征。
- `window_trade_attribution.csv.gz`：把原交易按窗口实际收益日截取的贡献，包括跨窗口持仓；贡献之和与窗口 PnL 核对。
- `entry_feature_contrasts.csv`、`entry_feature_coverage.csv`：23 个入场已知特征的赢亏对比、四个入场 cohort、样本数量与缺失率。
- `observation_protocol.json`、`validation_summary.json`：观察定义、数据对账、来源与产物哈希。`hypotheses.json` 记录由观察提出的候选解释及反例，若存在则显示在报告中。

修改候选解释后，运行 `python -B run_observation_research.py --report-only` 可从已保存表格重画报告。最大浮盈/浮亏只使用日收盘观测，不代表日内极值；2026 年为不完整年度。日历收益按实际发生日期归属，期间胜率只统计期间入场且在期末前平仓的交易，跨期/未平仓数量另列。

这一步用于发现失效模式并提出假设，不拟合过滤阈值，也不生成新的过滤收益。事后识别的盈利区间、回撤谷底、最终浮盈回吐均是结果标签，不能直接作为入场特征。价格特征截至信号日收盘，经济输入严格早于信号日且最多陈旧七天；已有历史被重复观察不会变成新的样本外证据。

## 本轮 filter 研究：IS 筛选与最后三年历史留出

入口为 [run_filter_research.py](run_filter_research.py)，本轮结果单独写入 `reproduced/filter_oos_v1/`。研究范围包含全部 20 条技术规则 × 9 个币种的 180 个 BASE 账户；每种预写 filter 按策略家族及币种适用性评估，不根据表现缩小账户范围。旧 dashboard、观察报告与原研究产物不被覆盖。

- **IS 开发区间**：2005-01-03 至 2023-05-29。
- **OOS 历史留出区间**：2023-05-30 至 2026-05-29，为数据终点前最后三个日历年。
- **账户口径**：每个独立账户固定初始本金 100,000 美元，日 PnL 为固定本金乘实际持仓方向及现货简单日收益，累计逐日相加。本轮成本为零，不加入交易成本敏感性；carry、融资及现金利息同样为零。简单年化是累计单利收益除以实际日历年数，不是 CAGR。

流程为 `protocol prewritten → IS screen → freeze SHA → OOS one-shot`。`develop` 先保存协议、候选条件及源码/输入哈希；工作簿解析后，在技术指标、特征和策略收益计算前截去 IS 终点之后的价格与原生经济数据，不使用整段收益产物作开发输入。IS 综合门槛包括收益与简单年化、相对 BASE 的收益/胜率/盈亏比改善、回撤约束、收益与最大回撤绝对值之比改善、交易数量和保留比例，以及预设 IS 分段的正收益与改善要求；具体数值在 `protocol.json` 和 `src/fx_filter_evaluation.py` 中明确记录。

每个币种 / 技术规则最多冻结一个 IS 排名最高的新增达标 filter。排名依次比较 IS 累计收益改善、盈亏比绝对改善、胜率改善，最后以候选编号确定顺序。其他 IS 通过版本、对照版本和未通过版本全部保留在 IS 筛选表中；已有对照不作为本轮新增冻结候选。冻结记录保存候选参数、IS 顺序、源码/输入/IS 产物哈希以及尚未评估 OOS 的状态。

按阶段运行：

```powershell
# 1. 只开发 IS、应用预写综合标准并保存冻结清单。
python -B run_filter_research.py --stage develop

# 2. 只有冻结清单非空才评估 OOS；无新增达标候选时不会计算候选 OOS 结果。
python -B verify_filter_research.py --stage is --audit-json reproduced/filter_oos_v1/pre_oos_independent_audit.json
python -B run_filter_research.py --stage oos

# 3. 独立核验已完成的阶段；有 OOS 产物时生成固定候选的后见归因。
python -B verify_filter_research.py --stage auto --audit-json reproduced/filter_oos_v1/independent_integrity_audit.json
python -B analyze_filter_outcomes.py

# 4. 从已保存结果生成离线报告；也可在 OOS 前查看 IS 状态。
python -B run_filter_research.py --stage report
```

这些命令默认使用同一目录；运行器和独立核验可传入相同的 `--output <目录>`，归因脚本使用 `--directory <目录>`。归因脚本针对本轮已保存的六个冻结候选，不是新的选参入口。已经冻结的目录拒绝再次 `develop`；OOS 成功完成后再次调用 `oos` 不重复评估。独立核验补充检查开发状态记录中的冻结文件哈希，OOS 开始前同时核对协议、决策源码、输入和 IS 证据哈希；完整回放时再次确认 IS 持仓、成交及收益与冻结记录一致。全部冻结候选都按原条件评估，OOS 亏损、劣化或不达标结果同样保留；不依据 OOS 重排、删掉失败者、换候选或修改阈值。没有可冻结的新候选时，跳过后见归因，只保留 IS 筛选与流程停止报告，不补造 OOS 结果。

执行与跨界统计沿用明确的时间约定：信号在报价日 t 收盘形成，下一报价日 t+1 收盘成交，首笔持仓收益结束于 t+2。主评估保持 IS/OOS 边界处持仓连续，图形仅将各区间累计 PnL 独立归零；跨界持仓在 OOS 内赚取的日收益仍计入 OOS PnL。交易质量只统计区间内开仓且在区间末前实际平仓、至少获得一个收益观测的交易，跨界持仓和期末未平仓交易另列；保留比例为 FILTER 与同区间 BASE 合格已平仓数之比。

`fresh_start` 另作边界敏感性诊断：OOS 起点以现金开始，只允许原 BASE 交易段在 OOS 起点及以后实际开仓的交易进入，禁止在已有交易段中途重新入场。主结果仍是连续持仓路径，该诊断不用于 OOS 重选。IS 还保留额外延迟一个报价日的固定目标序列诊断；它不重新求解延迟成交下的退出决策，不是实际延迟执行模拟，也不进入筛选门槛。

主要产物按阶段生成：

| 产物 | 内容 |
| --- | --- |
| `protocol.json`、`development_status.json` | 预写规则、区间、零成本口径、IS 标准、OOS 评价标准与开发状态 |
| `is_account_coverage.csv`、`is_all_metrics.csv`、`is_selection.csv` | 180 个账户覆盖、全部适用候选 IS/分段指标，以及通过/失败/冻结记录 |
| `frozen_selection.json`、`is_frozen_daily.csv.gz`、`is_frozen_metrics.csv` | 冻结身份与哈希、冻结候选的 BASE/FILTER IS 日路径及指标 |
| `oos_status.json`、`oos_metrics.csv`、`oos_daily.csv.gz` | 冻结后一次 OOS 评估状态、完整指标和日路径；有冻结候选才启动，状态先记录运行中，指标在计算后写出 |
| `candidate_assessment.csv`、`oos_anniversary_metrics.csv` | 每个冻结候选的预写 OOS 曲线门槛及额外交易质量确认、三个周年区间表现 |
| `oos_fresh_start_metrics.csv` | OOS 现金起步、仅接纳新交易段的边界敏感性 |
| `frozen_full_trade_ledger.csv.gz`、`frozen_full_episode_decisions.csv.gz` | 冻结候选的完整逐笔账本和过滤决策证据 |
| `independent_integrity_audit.json` | 独立复算收益、回撤、交易质量、冻结排序、文件哈希和 OOS 完整性 |
| `failure_attribution.csv`、`failure_episode_attribution.csv`、`failure_diagnostics.json` | 拒绝入场、提前退出、保留收益的分段归因；避免亏损与放弃盈利按实际收益日核算，不用后见结果重新选参 |
| `filter_research_report.html`、`pnl_comparison.pdf`、`figures/`、`report_build.json` | 中文离线报告、按 IS 冻结顺序展示的比较图册和 PNG，以及报告文件哈希 |

这是一轮**开发过程隔离的历史留出测试**。2023–2026 年历史在前期工作中已经被绘图和观察，本轮隔离与冻结不能将其变成从未看过的样本或新采集的样本外证据。OOS 曲线标准与交易质量确认在协议中预先写明，结果用于检验冻结假设，不作为当轮继续调整候选的依据。

## 第二轮：由入场状态观察提出 filter，并按时间确认

[run_state_filter_research.py](run_state_filter_research.py) 将 180 个 BASE 账户的样本内交易与信号日已知状态对齐。观察缓存位于 `reproduced/filter_is_diagnostics_v2/`，第二轮结果位于 `reproduced/state_filter_v2/`。13 个特征使用预先固定的四档区间；缺失状态保留交易。每个候选仅拒绝一个入场状态下的完整原策略交易段，不修改退出，也不在被拒交易中途重新入场。成本仍为零。

开发使用 2005–2014 年（另核查 2005–2009、2010–2014 两段），验证使用 2015–2019 年。每个账户至多锁定一个候选后，才回放 2020-01-01 至 2023-05-29 的确认期；确认失败不换备选。开发与验证阶段仅计算截至 2019 年的数据。已保存的 2019 年前持仓与收益在确认回放中逐项保持一致。

第二轮在原综合标准之外增加了更严格的检查：确认期至少 5 笔盈利和 5 笔亏损交易，以及全 IS / 2015 年以来的改善在扣除三笔最大有利交易贡献后仍为正。这些是新增稳健性要求，应与用户确认的原综合标准区分。结果为 **1,201 个提案、84 个开发单独通过、103 个验证单独通过、5 个同时通过、4 个账户锁定，严格最终筛选 0 个通过**。第二轮未启动 OOS；全部 4 个结果和曲线保留。

其中 JPY / BB_MR_20 在信号日 `efficiency_63 ∈ [0.1, 0.2)` 时拒绝入场，满足原综合标准：完整 IS 单利收益为 +31.50%，相对 BASE 改善 32.75 个百分点；确认期收益从 −1.94% 改善至 +6.35%。其确认期只有 4 笔亏损交易，且 2015 年以来改善在扣除前三笔有利贡献后为负，因此没有通过新增要求。此事实不等于原综合标准失败；任何按原标准开展的补充验证必须另存协议与结果，不能回写第二轮严格筛选记录。

```powershell
# 只在新目录首次运行观察 / 开发 / 确认；已有冻结记录拒绝覆盖。
python -B build_is_filter_observations.py
python -B run_state_filter_research.py --stage fit
python -B run_state_filter_research.py --stage confirm

# 已完成结果可直接核验、重画，无需重新计算策略。
python -B verify_state_filter_research.py
python -B build_state_filter_report.py
```

`state_research_report.html` 与 `state_figures/` 按确认前排名展示全部 4 个锁定候选的完整 IS 和确认期曲线。`confirmation_selection.csv` 分别记录原标准、确认期和稳健性结果；`confirmation_effect_evidence.json` 记录影响的交易数与剔除前三笔贡献后的改善。`independent_validation.json` 已通过 310 项独立检查，覆盖冻结顺序、哈希、筛选、72 组区间指标与 12 组交易贡献复算；全套 76 项测试通过。这里的检查通过只说明计算与流程一致，不表示候选策略通过经济表现标准。

## 按原综合标准补充验证 JPY 候选

[run_state_filter_supplement.py](run_state_filter_supplement.py) 将上面的唯一原综合标准达标候选保存到 `reproduced/state_filter_supplement_v1/`。该规则已在 C 之前锁定；**单独补充验证的晋级决定发生在看到 C 结果之后**，以用户此前确认的原综合标准为准，并将新增集中度与确认期亏损笔数要求作为披露诊断。严格 v2 的 0 个通过结论保持不变。补充分析完整保留原候选 ID、状态区间、原 SCREEN 与 OOS_SCREEN，不调参、不选择替补。

补充冻结前核对原始研究与独立审计；OOS 前再次独立核验协议、源码、输入、IS 数据和原参数。成本为零，原本金单利与连续持仓口径不变。2023-05-30 至 2026-05-29 是已被前序研究观察的历史，不能称为全新、未见的独立 OOS。

| 区间 / 版本 | 累计单利收益 | 最大回撤 | 胜率 | 盈亏比 | 期内闭合交易 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IS / BASE | −1.25% | −28.51 pp | 66.45% | 0.505 | 155 |
| IS / FILTER | +31.50% | −17.65 pp | 68.57% | 0.706 | 105 |
| 三年历史 OOS / BASE | −0.16% | −14.64 pp | 64.00% | 0.572 | 25 |
| 三年历史 OOS / FILTER | −1.25% | −13.19 pp | 68.42% | 0.446 | 19 |

OOS 未通过预写曲线标准：简单年化为 −0.42%，累计收益低于 BASE。胜率改善，但盈亏比下降，且 19 笔闭合交易不足质量确认的 20 笔门槛。三个周年的 FILTER 收益分别为 +0.03%、−9.02%、+7.74%；现金起步的边界诊断收益为 −0.68%，仍低于 BASE 的 +0.41%。全部结果保留。IS 净改善中约 93.8% 来自避开三笔大亏损，不能将样本内提升外推为稳定有效的尾部保护。

```powershell
# 首次按原参数建立补充冻结；已有目录拒绝覆盖。
python -B run_state_filter_supplement.py --stage freeze
python -B verify_state_filter_supplement.py --stage pre
python -B run_state_filter_supplement.py --stage oos

# 已有结果的审计和报告。
python -B verify_state_filter_supplement.py --stage post
python -B run_state_filter_supplement.py --stage report
```

补充协议、冻结、逐日曲线、完整交易账本和分年结果均保存在该独立目录。`jpy_observation_explanation.json` 提供被拒状态的分期观察及三笔主要交易证据，只作事后解释，不参与重选。研究目标仍未达到：本轮没有确认出 IS 与三年历史 OOS 同时满足综合要求的新 filter。

查看 `state_filter_supplement_report.html` 或单页图册 `jpy_is_oos_comparison.pdf`。补充 OOS 前的 247 项核验、完成后的 378 项独立检查和 32 项图表产物检查均通过；完整执行前缀、日收益、交易、分年与现金起步结果得到复核。`final_validation_summary.json` 明确区分计算验证通过与研究目标未达到。

## 第三轮：从持仓内已知状态观察退出时点

[build_holding_observations.py](build_holding_observations.py) 在 IS 内重建全部 180 个 BASE 账户的持仓过程，保存于 `reproduced/holding_observations_v3/`：46,545 个原交易段、54,082 个固定年龄 checkpoint、67,500 行分组观察。固定年龄为已赚取 5、10、20、40、63 个报价日收益；已知状态包括当前累计单利 PnL、截至当日的峰值与回吐、最近 5 日有向收益及当前 20 日价格方向。最终盈亏与未来收益均为标签，不参与触发。

假设 checkpoint 在收盘 m 发出退出指令，实际在 m+1 收盘退出，因此 m+1 收益仍归旧仓；可以避免的剩余收益从 m+2 开始。原技术退出/反转已在 checkpoint 当天发出时优先处理。每笔原交易只在指定年龄检查一次，未触发时不在后续日期重试；触发后不在原交易段内重新入场。开放、跨期、无法执行和技术退出优先的情况单列。

`run_holding_filter_research.py` 只从 2005–2009、2010–2014 和合计 2005–2014 年的观察提案：两个五年段各有至少 10 个完整、可执行且条件成立的原交易，两个子段及合计段的剩余收益均为负。分组条件为盈利、亏损、回吐至少一半、最近五日亏损、当前 20 日方向相反，以及最多两个条件的 AND；阈值固定。提案确定后才回放完整 IS，仍使用原 SCREEN、原排序和最后三年历史 OOS 规则，集中度仅作诊断。

结果为 **391 个登记版本，涉及 106 个账户；原 IS 综合标准 0 个通过，因此没有开启 OOS**。其余 74 个账户仍保留在完整覆盖表中。391 个版本对应 280 类同账户实际退出路径，不能当作 391 个独立试验；例如在第 5 个收益日，累计亏损和最近 5 日亏损是同一条件。所有版本保留，等价归组不改变筛选。

这轮的重要观察是胜率与盈亏比的取舍。例如 AUD / VOTE_EQUAL_4 在第 10 个收益日已有盈利时退出，完整 IS 累计收益 +68.83%、胜率改善 9.42 个百分点，但盈亏比下降 17.28%，因此没有通过原标准，也没有单独补测其 OOS。

```powershell
python -B build_holding_observations.py
python -B run_holding_filter_research.py --stage develop
python -B run_holding_filter_research.py --stage report
```

已有观察或研究目录拒绝覆盖。`reproduced/holding_filter_v3/holding_filter_report.html` 展示提案、原标准失败分布及等价路径；没有冻结候选时不生成候选曲线，配对质量与贡献表只保留空表头。`is_all_decisions.csv.gz` 保存全部候选的逐原交易段决策，供独立核验。新的条件退出引擎通过 693 次 prefix 因果检查，全套测试当时为 87 项通过。

## 第四轮：不依赖盈亏状态的最长持仓时间

[run_timecap_filter_research.py](run_timecap_filter_research.py) 单独使用同一观察缓存的 `ALL` 对照组，检验剩余收益是否整体随持仓过长而恶化。提案条件仍为 D1、D2 各至少 10 笔完整可执行交易、两个子段及合计 D 剩余收益均为负。固定年龄仍为 5、10、20、40、63；到龄时只要原技术仓位仍在，就在下一收盘退出，包含当时盈利、亏损或恰好为零的情况。

这项机制在第三轮 IS 条件退出结果之后提出，属于重复使用 IS 的研究；单独保存协议、参数、源码与结果，不改写第三轮。`src/fx_timecap_filters.py` 复用已验证的持仓 checkpoint 诊断，以独立的逐日锁定逻辑实现无条件退出。新增 6 项合成测试通过，含 75 次 prefix 对照及 `ALL` 观察对齐。

第四轮结果保存于 `reproduced/timecap_filter_v4/`：**101 个候选、69 个有提案账户、全 180 账户覆盖，原 SCREEN 0 个通过，未启动 OOS**。`holding_filter_report.html` 展示完整失败原因。独立审计已通过 1,666 项检查，覆盖全部候选决策、808 个窗口指标、提案、排序、覆盖及空冻结状态。

例如 AUD / DONCHIAN_BO_20 的 10 日限制，IS 累计单利收益由 +22.59% 升至 +75.76%，最大回撤由 38.25 降至 16.26 个百分点，胜率由 37.72% 升至 48.68%；平均盈利 / 平均亏损由 1.863 降至 1.612，因此仍未通过原标准。该候选每笔平均收益由 0.1014% 升至 0.3346%，总盈利 / 总亏损比由 1.128 升至 1.529。后两项与平均盈亏比是不同指标，不能替换原门槛后把结果追认成原标准通过。

```powershell
python -B run_timecap_filter_research.py --stage develop
python -B run_timecap_filter_research.py --stage report
```

第三、四轮完成时，是否接受胜率与平均盈亏比的取舍仍待用户选择，因此这些候选当时没有进入 OOS。用户后续批准改用每笔期望收益与总盈利/总亏损比同时改善；该变更在第八轮单独执行，第三、四轮及讨论材料仍保留当时的原标准与状态。

`reproduced/holding_quality_tradeoff_review/` 是只读的 IS 讨论材料。`review.csv` 保留第三、四轮全部 492 个版本及其原始失败状态，另列每笔期望收益和总盈利 / 总亏损比。若仅以这两项改善替换原胜率 / 平均盈亏比门槛、保留其他原标准，会有 60 个版本满足讨论条件；沿原排序跨两轮每账户至多一个为 21 个建议。这不是已批准标准或冻结清单，未用于 OOS。

独立核验入口可复用于两轮，不运行策略或读取 OOS：

```powershell
python -B verify_holding_filter_research.py --output reproduced/holding_filter_v3
python -B verify_holding_filter_research.py --output reproduced/timecap_filter_v4
```

第三、四轮最终独立数值核验分别为 4,565 / 4,565 与 1,666 / 1,666 项通过；另有 117、119 项产物完整性检查通过。各目录的 `final_validation_summary.json` 明确记录：计算核验通过，原标准通过数为零，尚未达到 IS 与 OOS 同时有效的研究目标。

[AUD Donchian 20 的样本内讨论曲线](reproduced/holding_quality_tradeoff_review/aud_donchian20_timecap10_is_review.pdf) 展示固定 10 日持仓检查点规则的累计单利收益与回撤。它从已保存决策重建，2,108 个账本与指标字段匹配；没有重新筛选或读取 OOS。2015—2019 年过滤后的收益仍为 −6.67%（BASE 为 −28.81%），不能仅凭完整 IS 曲线改善判断跨时期有效。图中保留原盈亏比门槛失败说明，讨论材料不改变冻结清单。

## 第五轮：两个入场状态的交互观察

[build_entry_interaction_observations.py](build_entry_interaction_observations.py) 只从原 IS 缓存中解析 2005—2014 年内开仓、真实平仓的 25,133 笔交易。特征对固定为：63 日趋势效率与 20/63 日波动比、63 日趋势效率与标准化有向 20 日价格变化、有向美元指数变化与有向利差。每对沿用既有四档边界，共 16 格，覆盖全部 180 账户；不根据结果扩张特征集合或调整阈值。

每格保存同时满足、仅满足 A、仅满足 B、均不满足、任一 UNKNOWN 五个互斥分组，另列单条件与保留交易对照。D1（2005—2009）、D2（2010—2014）各至少 10 笔且收益合计为负，合计 D 也为负，才形成候选。跨越两个子区间的交易可以只进入 D；三个区间不是独立证据。集中度、胜率及盈亏比差异只用于解释，没有新增硬门槛。267 个候选涉及 110 个账户，其他 70 个保留在覆盖表中。

[run_entry_interaction_research.py](run_entry_interaction_research.py) 先保存观察假设，再计算完整 IS。新引擎 `src/fx_entry_interaction_filters.py` 在同一原 BASE 入场信号上读取两个条件；仅在两者都已知且匹配时拒绝整个原交易段，UNKNOWN 放行，不在被拒段中途重新入场。两个条件均读取未过滤的原始历史，不能将两个单条件 filter 串联。成本为零，原 SCREEN、排名、每账户一个冻结候选及 OOS_SCREEN 均保持不变。

两个候选通过原 IS 标准，参数冻结和 4,252 项独立核验之后才开启三年历史 OOS：

| 候选 | 固定拒绝入场条件 | IS 原策略 → FILTER | OOS 原策略 → FILTER |
| --- | --- | ---: | ---: |
| NZD / VOTE_STRENGTH_4 | efficiency_63 < 0.1 且 direction_return_20_vol ∈ [0, 1) | +2.60% → +45.00% | −23.97% → −13.06% |
| GBP / BB_MR_20 | efficiency_63 < 0.1 且 direction_return_20_vol ∈ [−1, 0) | +24.25% → +34.19% | +12.24% → +8.49% |

`direction_return_20_vol` 为持仓方向乘 20 日价格收益，再除以用信号日 20 日波动估计的相应尺度；所有价格特征截至信号日收盘可知。累计收益均为固定本金单利。三年 OOS 是此前已观察过的历史复验，不能称为全新独立证据。

**两个候选均未通过原 OOS 曲线标准，也未确认交易质量。** NZD 的三个周年区间都亏损，累计回撤为 15.04 个百分点，胜率和盈亏比均低于 BASE。GBP 的三个周年区间分别为 +2.22%、+2.99%、+3.28%，最大回撤 3.94 个百分点，但收益低于 BASE，胜率从 67.65% 降至 65.38%；盈亏比从 1.061 升至 1.117。不能仅因其正收益或回撤较小而认定 filter 有效。

逐原交易段的 IS 归因保存在 `reproduced/entry_interaction_probe_v5/`。NZD 在早期、中期、后期的收益改善分别为 +41.75、−8.05、+8.70 个百分点；GBP 为 +11.79、−1.94、+0.09 个百分点，且剔除三笔最大有利贡献后全 IS 改善为负。这些诊断保留原筛选结论，不用于 OOS 重选或修改规则。方向分组观察另存 `reproduced/direction_probe_v5/`，尚未据此冻结或测试方向过滤器。

```powershell
# 开发命令只用于首次建立新目录，已有研究拒绝覆盖。
python -B build_entry_interaction_observations.py
python -B run_entry_interaction_research.py --stage develop
# 只在首次 OOS 开启前运行；保留此次 pre 审计记录，不在 OOS 后覆盖它。
python -B verify_entry_interaction_research.py
python -B run_entry_interaction_research.py --stage oos

# 已完成研究只运行下面的 post 核验与报告命令。
python -B verify_entry_interaction_oos.py
python -B build_entry_interaction_report.py
```

新引擎 6 项测试（含 90 次前缀因果检查）、提案隔离 4 项测试、观察读入与分组 4 项测试全部通过。本轮 OOS 的 127 项独立核验覆盖原始报价、固定条件、连续持仓、逐笔账本、三个周年窗口及现金起步对照。计算验证通过与策略有效是两个结论；目前仍未得到同时满足原 IS 和 OOS 要求的新 filter。

## 第六轮：既有多空方向观察的完整检验

[run_direction_filter_research.py](run_direction_filter_research.py) 使用第五轮 OOS 开启之前已保存的 D-only 方向观察。全 180 账户、360 个账户方向分别核对 D1、D2 与 D；两子期各至少 10 笔闭合交易且收益为负、合计 D 也为负，共 97 个提案。方向固定为 USD-per-foreign 报价的 +1（多外币、空美元）或 −1（空外币、多美元），仅拒绝指定方向的完整原交易段。没有依据后期收益重新选择方向。

方向观察独立核验 1,816 项通过。AUD / TSMOM_63 多头是跨期反例：D1、D2 各亏 −17.40%、−1.86%，但一笔跨 2009—2010 年的 +26.09% 交易令合计 D 为 +6.83%，所以不进入候选。所有子期与完整期均保留，不能将分段亏损直接相加推断方向无效。

97 个候选中，只有 GBP / RSI_MR_7 拒绝多头通过原 IS 综合标准。固定本金单利结果如下：

| 区间 / 版本 | 累计收益 | 最大回撤（初始本金百分点） | 胜率 | 平均盈亏比 | 闭合交易 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IS / BASE | +14.52% | 17.55 | 62.61% | 0.675 | 222 |
| IS / FILTER | +26.61% | 9.46 | 66.09% | 0.850 | 115 |
| 三年历史 OOS / BASE | +4.05% | 4.52 | 65.79% | 0.661 | 38 |
| 三年历史 OOS / FILTER | +3.63% | 2.67 | 65.00% | 1.055 | 20 |

FILTER 的三个 OOS 周年均盈利，但累计收益比 BASE 少 0.415 个百分点，胜率亦略降，**原 OOS 曲线与质量要求仍未通过**。不会因为差异较小而放宽门槛。完整结果保存在 `reproduced/direction_filter_v6/`，查看 [方向研究报告](reproduced/direction_filter_v6/direction_report.html) 与 [IS/OOS 图册](reproduced/direction_filter_v6/pnl_comparison.pdf)。

新引擎 6 项测试（含 180 次前缀检查）及提案隔离 5 项测试通过；IS 的 2,028 项独立核验与 OOS 的 137 项核验通过。`final_validation_summary.json` 明确标记计算通过、研究目标未达到。审计脚本分别为 `verify_direction_filter_research.py`（只在首次 OOS 之前运行，保留 pre 证据）与 `verify_direction_filter_oos.py`（已完成研究的复核）；报告重绘使用 `python -B build_direction_filter_report.py`。旧研究和所有失败版本保持不变。

## 第七轮：补齐既有单状态候选的原标准评估

[run_state_original_screen.py](run_state_original_screen.py) 补齐了第二轮的评估范围：原先保存的 1,201 个 D-only 候选都计算过 D/V/FIT，但只有通过额外分阶段要求的 4 个计算了完整 IS。本轮对**同一份 1,201 个候选**统一应用用户原先接受的综合标准。没有新增特征、调整分箱、扩大候选名单或改写第二轮结果；这些是既有假设，不能称为 1,201 个全新独立试验。

提案来自 2005—2014 年的原始策略交易观察：固定状态在 2005—2009、2010—2014 两段各至少有 10 笔完整交易、两段收益均为负，合计期去掉最差一笔交易后仍为负。跨期交易单独按合计窗口处理。IS 为 2005-01-03—2023-05-29，三年历史 OOS 为 2023-05-30—2026-05-29，仍覆盖全部 20 条技术规则、9 个币种、180 个账户。交易成本、融资、利息均为 0，以固定 100,000 美元本金的单利累计收益评价。

完整 IS 中 23 个版本通过原标准，按原排序每个账户至多保留一个，冻结 21 个。17,973 项独立 IS 核验通过之后才统一开启本轮 OOS，所有冻结者均保留结果。3 个同时通过原 OOS 曲线和质量要求，另外 3 个仅通过质量要求。下表按原冻结顺序列出同时通过者；箭头为原策略 → FILTER，收益为各完整区间累计值。

| 策略 | 固定拒绝入场条件 | IS 收益 | 三年 OOS 收益 | OOS 胜率 | OOS 平均盈亏比 | OOS 闭合交易 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GBP / MACD_12_26_9 | `position × DXY_20步变化% ≥ 1` | +3.31% → +44.87% | +9.22% → +11.80% | 51.06% → 57.14% | 1.331 → 2.559 | 47 → 21 |
| CAD / TSMOM_63 | `−1 ≤ position × DXY_20步变化% < 0` | −15.83% → +18.62% | +5.12% → +7.56% | 41.18% → 46.15% | 2.108 → 2.489 | 34 → 26 |
| EUR / BB_MR_20 | `position × 标准化60日均线距离 < −1` | +4.29% → +30.67% | +4.71% → +7.87% | 58.62% → 65.00% | 1.003 → 1.913 | 29 → 20 |

这里 `position=+1` 为多外币、空美元，`−1` 为反方向。DXY 使用源序列中 20 个原生观察步的百分比变化；宏观数据观察日严格早于信号日、距信号日最多 7 个自然日，20 步跨度最多 45 天。这个时点约束不能证明源数据原始公布时间或无修订。均线距离为 `(价格/60日均线−1)×100`，除以 `20日年化波动率% × sqrt(20/252)`。信号日收盘决定是否跳过整个原始交易段，下一收盘成交，保留原技术退出；未知状态放行。

观察与解释需要分开。GBP 的被拒状态在两个早期子段分别亏损 19.18%、1.30%，可能对应技术入场方向与较强美元趋势冲突。EUR 的对应状态分别亏损 14.57%、2.26%，可能对应在价格远离较长均线时过早逆势；但第二段移除最差交易后已转正，不能称为每个子段均不受极端交易影响。CAD 的对应状态分别亏损 11.00%、6.41%，是“美元变化略有利于持仓”的局部分箱现象，尚不能给出可信的单调经济解释。以上解释不改变冻结条件，也不将关联视为因果。

三个 FILTER 的 OOS 最大回撤分别为 3.30、4.31、3.18 个初始本金百分点，三个周年区间均盈利。GBP 周年收益为 +2.65%、+8.85%、+0.30%；CAD 为 +0.16%、+5.99%、+1.41%；EUR 为 +2.61%、+3.06%、+2.20%。GBP、CAD 的改善并非每个 IS 子期都盈利：2015—2019 年分别仍亏 5.97%、9.60%，完整 IS 最大回撤仍达 27.39、29.20 个百分点。它们满足预设标准，不代表整个历史都平稳。

完整结果在 [第七轮报告](reproduced/state_original_screen_v7/state_original_screen_report.html)、[21 个冻结候选图册](reproduced/state_original_screen_v7/pnl_comparison.pdf) 与 [3 个同时通过候选的曲线](reproduced/state_original_screen_v7/passed_candidates.pdf)。`candidate_assessment.csv` 保留全部 21 个冻结者及失败原因；`is_selection.csv` 保留全部 1,201 个版本。JPY / BB_MR_20 的效率状态候选再次冻结，复现其既有失败结果，属于旧证据复核，没有依据已知结果剔除或替换。

本轮三年 OOS 此前已在研究中被观察过，属于**重复使用的历史验证**，不能称为全新独立样本。通过者的 OOS 闭合交易只有 20—26 笔，多轮筛选也有选择偏差；当前结论限定为在这些数据、零成本与原预设标准下的历史通过，不代表已确认未来有效。没有根据 OOS 回调阈值、重排候选或组建投资组合。

独立 OOS 核验 **520 / 520 项通过**：从原始价格、独立读取的原生宏观序列和原 BASE 已赚收益重建 6,240 个交易段决定、210 个连续 OOS / 周年 / 现金起步指标窗口，并复现全部 21 个候选的判断。旧 JPY 候选的 1,568 行日数据及 2 行 OOS 指标与先前补充研究逐项数值完全相同。该核验使用已保存的原 BASE 账本提供交易段，不声称独立重写了 20 条技术策略；本轮冻结者没有 IS 末日 pending 行。核验记录为 `independent_validation.json` 与 `post_oos_independent_validation.json`，原 pre 记录及源码、输入、冻结结果哈希均保持不变。新增提案守卫 3 项测试通过，既有单状态执行引擎未修改。

```powershell
# 下列开发及 pre 核验命令只用于首次研究；已有目录拒绝覆盖。
python -B run_state_original_screen.py --stage develop
python -B verify_state_original_screen.py
python -B run_state_original_screen.py --stage oos

# 已完成研究只运行 post 复核与报告重绘，不覆盖首次 pre 证据。
python -B verify_state_original_screen_oos.py
python -B build_state_original_screen_report.py
```

## 第八轮：用户批准的期望收益与总盈亏比标准

用户明确选择：“允许取舍，但每笔期望收益与总盈利/总亏损比都须改善”。[run_holding_tradeoff_research.py](run_holding_tradeoff_research.py) 据此对原第三、四轮的 492 个固定候选开展单独研究，覆盖 180 个原账户。391 个条件退出候选与 101 个持仓上限候选的参数、D-only 观察证据和交易时序均沿用原记录；本轮先重放并核对全部 IS 指标，再按新标准冻结候选。

[fx_tradeoff_evaluation.py](src/fx_tradeoff_evaluation.py) 仅替换两项质量门槛：完整区间内开仓并真实平仓的交易，其每笔期望收益及总盈利/总亏损比都必须有限且严格高于 BASE。期望收益为该闭合交易集合的平均单利收益；总盈利/总亏损比为 `win_n × mean_win_pct / (loss_n × mean_loss_abs_pct)`，与平均盈利/平均亏损不同。没有赢单时总盈利为零；没有亏单时分母为零，指标未定义，不以无穷大判为通过。平手交易仍计入平均收益的分母。改善要求为严格大于，未拟合最小改善幅度。

收益、回撤、收益/回撤、交易数量、分段表现及原跨候选排名均保留。旧胜率和平均盈亏比继续作为诊断列展示。新标准下 60 个 IS 版本通过，按原排名跨两族每个账户保留一个，共冻结 21 个；名单与用户作选择前已保存的讨论建议完全一致。旧 492 个版本原本全部不通过的结论仍在 `is_original_selection.csv` 和原研究中保留。

IS 为 2005-01-03—2023-05-29；OOS 为 2023-05-30—2026-05-29，固定初始本金 100,000 美元、单利、交易成本/融资/利息均为 0。OOS 曲线要求继续沿用原规则，交易质量要求为至少 20 笔闭合交易且上述两项严格改善。所有冻结候选统一验证，完整保留失败者。这三年历史此前已经被观察过，本轮属于重复使用的历史验证。

本轮 21 个冻结候选中，曲线通过 4 个、质量通过 8 个、两者同时通过 4 个。下表按原 IS 冻结顺序列出同时通过者，箭头为 BASE → FILTER；每笔期望收益为闭合交易平均收益，占初始本金的百分比。

| 策略 / 固定退出条件 | IS 累计收益 | OOS 累计收益 | OOS 每笔期望收益 | OOS 总盈利/总亏损比 | OOS 闭合笔数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GBP / MA_10_30：第10个收益日若最近5日持仓收益合计为负 | −8.31% → +35.37% | +4.82% → +7.03% | 0.3267% → 0.4229% | 1.487 → 1.935 | 23 |
| JPY / BB_MR_20：第5个收益日无条件退出 | −1.25% → +30.19% | −0.16% → +3.01% | 0.0163% → 0.1204% | 1.016 → 1.242 | 25 |
| JPY / DONCHIAN_BO_20：第5个收益日若本笔累计收益为负 | +4.08% → +20.55% | +8.14% → +10.16% | 0.2424% → 0.3018% | 1.377 → 1.581 | 34 |
| JPY / VOTE_EQUAL_4：第5个收益日若本笔累计收益为负 | +21.58% → +37.82% | +4.93% → +7.99% | 0.1260% → 0.2024% | 1.190 → 1.376 | 40 |

上述天数均为已经赚取的报价收益日，不是自然日；条件成立时在检查日收盘发出退出信号，下一收盘执行，执行日的持仓收益仍计入。每个原交易段只检查一次，提前退出后不在该原交易段内重新入场，技术退出优先。JPY / BB_MR_20 的 OOS 胜率由 64% 降至 60%，但每笔期望收益及总盈利/总亏损比同时提高，是本次获准取舍的实际例子。

四条 FILTER 的 OOS 最大回撤分别为 7.33、9.93、11.04、14.17 个初始本金百分点，都仅有两个周年盈利。GBP 均线、JPY Donchian、JPY 等权投票在最后一个周年分别亏损 4.05%、6.71%、8.69%；JPY 布林均值回归在第二周年亏损 5.02%，且三年简单年化 1.0028% 仅略高于 1% 门槛。JPY Donchian 与等权投票的完整 IS 回撤仍为 42.85、35.95 个百分点。满足门槛不代表每段时期都稳定，也不能把这四条与第七轮按另一质量标准得到的三条直接合并成已验证组合。

用户讨论的 **AUD / DONCHIAN_BO_20 / TIME_CAP_10 在 OOS 失败**：虽然 IS 累计收益 +22.59% → +75.76%，OOS 却从 −17.09% 降至 −23.50%，每笔期望收益 −0.3617% → −0.5180%，总盈利/总亏损比 0.574 → 0.368，最大回撤 19.72 → 24.71 个百分点，三个周年全部亏损。该结果完整保留，没有因为它是最初示例而单独更换参数。

新评价模块 9 项测试通过，覆盖原 492 版本名单/排名对照、严格改善、空/无赢/无亏交易组、非有限值及 IS/OOS 隔离。独立 pre 审计 7,624 项通过，重建 134,056 条原交易段决定及 3,936 个指标窗口。492 个版本对应 362 条账户内不同的 IS 执行路径，不能按 492 次独立试验解释；此诊断未改变排名。原 BASE 信号继承已验证账本，5 条期末未成交信号无法单靠执行账本独立恢复，结构和旧证据一致且不影响已观测 IS 收益。

独立 post 审计 **501 / 501 项通过**，复核全部 21 个候选、6,915 条原交易段决定、210 个 OOS / 周年 / 现金起步窗口及 42 个 IS 前缀窗口，确认曲线通过 4、质量通过 8、同时通过 4。核验没有调用生产过滤器或评价器，也没有读取旧 OOS 结果；原技术信号仍继承 BASE 账本。全期有 6 条末端未执行信号只能核对冻结决定和结构，不能由成交账本独立恢复，不影响已观测收益。pre 与 post 的原始记录分别保存在 `independent_validation.json` 和 `post_oos_independent_validation.json`。

查看 [4个同时通过候选的曲线](reproduced/holding_tradeoff_v8/holding_tradeoff_passed_summary.pdf)、[完整第八轮报告](reproduced/holding_tradeoff_v8/holding_tradeoff_report.html) 与 [全部21个冻结候选图册](reproduced/holding_tradeoff_v8/holding_tradeoff_atlas.pdf)。`is_selection.csv` 保留全部 492 个版本的新旧门槛；`candidate_assessment.csv` 分别列出 OOS 曲线失败原因与新质量失败原因。此前仅供讨论的 `holding_quality_tradeoff_review/` 仍保留当时等待批准的状态，本轮授权和实际冻结记录在新目录中保存。

```powershell
# 开发及 pre 审计只用于首次建立研究，不覆盖已有结果。
python -B run_holding_tradeoff_research.py --stage develop
python -B verify_holding_tradeoff_research.py
python -B run_holding_tradeoff_research.py --stage oos

# 已完成研究的独立 post 复核与报告重绘。
python -B verify_holding_tradeoff_oos.py
python -B build_holding_tradeoff_report.py
```
