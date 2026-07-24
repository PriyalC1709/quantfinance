# Oil-Conditioned Cross-Sectional FX Strategy

**Course:** Applied Quantitative Macro Strategies (AQMS), Imperial College Business School
**Project type:** Group coursework submission
**Note on source material:** This project was completed as coursework for Imperial College Business School and the underlying Jupyter notebook cannot be made public. This document is a detailed written record of the methodology, results, and conclusions, prepared as a public-facing substitute for the notebook. Key output tables/charts are stored alongside this file.

---

## 1. Hypothesis

Oil price moves are hypothesised to move currencies in predictable, economically grounded ways: commodity exporters benefit when oil rallies (improved terms of trade, stronger fiscal position), while importers are hurt (higher import bill, inflationary pressure). The strategy trades this relationship **cross-sectionally**: long exporter currencies and short importer currencies when oil rises, scaled by each currency's rolling sensitivity to oil.

Two conditioning filters were layered on top of the core oil-beta signal:
1. A filter to isolate **demand-driven** oil moves (using the Kilian/IGREA global real economic activity index), since supply-driven oil shocks do not carry the same FX transmission logic.
2. A **VIX kill-switch** so the book goes flat when risk sentiment is abnormal (broad de-risking dominates commodity-FX logic in stress).

A risk-reversal (FX options skew) "conviction" overlay was also specified and tested, but was **dropped from the final model** because it reduced risk-adjusted performance rather than improving it — this negative result is documented rather than hidden.

## 2. Universe (12 currencies)

- **Exporters** (long when oil rises): CAD, NOK, MXN, COP
- **Importer** (short when oil rises): JPY
- **Other** (direction set by rolling oil beta, sign not fixed a priori): EUR, GBP, CHF, AUD, NZD, SEK, ZAR

INR and BRL were excluded due to missing 1-month forward data in the provided datasets; DKK was excluded as a EUR-pegged currency. All USDXXX quote conventions were inverted to a common USD-per-foreign-currency convention prior to signal construction and PnL calculation.

## 3. Data

Bloomberg-sourced daily data (via provided CSVs), spanning the currency panel above:
- **Spot FX** rates for all 12 currencies
- **1-month forward points**, converted to full forward outrights
- **25-delta risk reversals** (1M options skew), used for the (ultimately dropped) conviction overlay
- **Brent crude (CO1)** as the oil benchmark, with WTI (CL1) pulled in for comparison only (Brent preferred as the more globally representative benchmark, avoiding WTI's regional/storage-driven distortions such as the April 2020 negative-price event)
- **VIX index**, for the risk kill-switch
- **IGREA** (Kilian global real economic activity index), monthly, used to separate demand- from supply-driven oil moves

### Exploratory data analysis
Summary statistics (mean, std, skew, kurtosis), a full 12×12 FX return correlation heatmap, per-currency return histograms and Q-Q plots, and an outlier scan (rolling 252-day z-score, flagging |z| > 5) were run on the full panel before any signal construction, per course requirements. Brent was also benchmarked against WTI to justify the Brent choice, given WTI's well-known 2020 negative-price dislocation.

## 4. Return construction

FX log returns were used to estimate rolling oil betas. For PnL, a one-day **excess return** was used that captures both the spot move and the carry embedded in the prior day's forward:

$$ r_t = \log\left(\frac{S_t}{F_{t-1}}\right) $$

where $S_t$ is the spot rate and $F_{t-1}$ is the forward outright quoted the previous day for settlement at $t$. This avoids attributing pure interest-rate carry to the oil signal.

## 5. Signal construction

The signal was built in four layers, each tested for incremental value:

| Layer | Description | Retained in final model? |
|---|---|---|
| **S1 — Core oil-beta signal** | Rolling 90-day covariance/variance regression of each currency's return on Brent's return (i.e. rolling oil beta), multiplied by a 10-day Brent return kernel (to smooth daily oil noise), then cross-sectionally demeaned so the book is long/short balanced | **Yes** |
| **S2 — Risk-reversal conviction overlay** | Scaled the S1 signal up/down based on whether 1M 25-delta risk-reversal skew (option-implied FX skew, z-scored) aligned with the direction of the S1 signal | **No — tested and dropped** |
| **S3 — IGREA demand/supply filter** | Point-in-time **expanding** OLS regression of monthly Brent returns on IGREA (re-estimated at each month using only data available up to that point, avoiding look-ahead), with the residual used to identify and downweight/haircut supply-driven oil moves | **Yes** |
| **S4 — VIX kill-switch** | Book scaled to zero when an EWMA-smoothed VIX exceeds a threshold of 35 (i.e. flat during acute risk-off/crisis periods) | **Yes** |

The final reporting model is the path **S1 → S3 (IGREA) → S4 (VIX)**, explicitly skipping S2.

All rolling-beta and filter calculations are lagged appropriately (e.g. betas shifted one day, IGREA regression re-estimated only on data available at each point in time, minimum 60-month IGREA warm-up) to avoid look-ahead bias — this was independently confirmed in a full code audit (see Section 9).

## 6. Portfolio construction

For each signal layer:
1. Per-currency signals are scaled by EWMA (60-day half-life) volatility so each name contributes comparable risk.
2. Weights are L1-normalised (sum of absolute weights = 1) before vol targeting.
3. The book is scaled to a **10% annualised volatility target**, using a stacked EWMA covariance matrix (0.5-year half-life, 1-year seed window), with gross exposure capped at **200%**.
4. Rebalancing is **weekly (Friday)**, with a one-day lag applied for realistic execution.
5. Transaction costs are applied on turnover: **2 bps for G10 currencies, 5 bps for EM currencies**.

## 7. Headline performance results

### 7.1 Layered comparison (spec stack, including RR)
- S1 alone: Sharpe ≈ **0.41** — the only economically coherent standalone layer.
- S2 (adding RR conviction): Sharpe ≈ **0.29** (**worse** than S1) — RR alignment does not add exploitable information in this form.
- Further layers on top of the RR path underperform the no-RR path.

### 7.2 Final model (S1 → IGREA → VIX, RR excluded)
Each retained layer **improves** performance monotonically:
- S1: Sharpe ≈ **0.41**
- + IGREA (S3): Sharpe ≈ **0.43**
- + VIX (S4, final model): Sharpe ≈ **0.63**

### 7.3 Train/test split (80/20)
- Split date: **25 January 2021**
- **Train Sharpe: +0.81**
- **Test (out-of-sample) Sharpe: −0.17**
- **Full-sample Sharpe: +0.63**

The full-sample Sharpe materially overstates recent performance — the gap between train and test Sharpe is treated as the central, honest result of the project rather than something to be smoothed over.

### 7.4 Risk profile
- Maximum drawdown on the final model: **≈ −42%**, clustering around the 2008–09 GFC, the 2014–16 oil price collapse, the 2020 COVID shock, and the 2022 energy crisis (periods when the conditioning filters are most active).
- Rolling 1-year Sharpe is frequently positive from 2004–2014 and in parts of 2016–2019, but turns negative or near-zero after ~2020 — consistent with a **regime-dependent** strategy that worked when oil–FX co-movement was demand-driven, and struggled once supply shocks and USD safe-haven flows began to dominate.

## 8. Forecasting-power diagnostics

- **Rolling 1-year information coefficient (IC)** of the signal against 5-day forward excess returns fluctuates around zero with episodic positive spells — consistent with a **conditional**, not a stable, oil–FX relationship.
- **Cross-sectional OLS t-statistics** (signal vs 5-day and 21-day forward returns, by currency) are mixed: commodity exporters (COP, MXN) show more consistent positive association at the 5-day horizon; several G10 currencies are insignificant or carry the "wrong" sign.
- **Conclusion:** average raw predictive power is weak. The claim made is not that the raw signal is a strong forecaster, but that portfolio construction and the conditioning filters convert a weak, regime-specific signal into a tradable process — one that nonetheless still fails out-of-sample in the test window.
- **Lead/lag information ratio** analysis (signal lagged/led by up to ±100 days) shows peak IR at lag 0, with flat wings either side — confirming the relationship is contemporaneous (as expected for liquid, efficiently priced FX) rather than driven by a spurious timing offset.

## 9. Risk-reversal (S2) diagnostics

A dedicated diagnostic section tested whether RR-signal alignment predicts forward returns:
- Cross-sectional table of mean forward returns conditional on RR/signal alignment shows **no reliable sorting** — differences and correlations are mixed across currencies.
- Combined with S2's Sharpe reduction in the layered comparison, this supports dropping the RR overlay from the final model rather than forcing a narrative around it.

## 10. Regime attribution

Four historical stress episodes were annotated and analysed against the final model's cumulative return path: the 2008–09 Global Financial Crisis, the 2014–16 oil price collapse, the 2020 COVID shock, and the 2022 energy crisis.

- **VIX kill-switch bind frequency:** ~5.3% of train days vs ~0.4% of test days.
- **IGREA haircut bind frequency:** ~14.0% of train days vs ~9.3% of test days.
- Both filters bind **less often** in the out-of-sample window, yet the strategy still fails out-of-sample — the conclusion drawn is that the failure is not "the filters don't work," but that the filters plus signal together do not generalise to the post-2021 regime, where oil and FX increasingly decouple.

## 11. Jackknife (leave-one-currency-out) analysis

Each currency was removed in turn, with the portfolio's covariance matrix, weights, and volatility target re-computed on the reduced universe each time, and the resulting Sharpe drop measured.

- The largest Sharpe contributions come from **MXN, ZAR, COP, CAD, and NOK** — the commodity-sensitive names carrying the core oil-beta story.
- **JPY** (the designated importer) also matters materially, consistent with its role in hedging the book.
- **Interpretation:** the strategy is not a pure oil factor — EM high-yield currencies also carry carry/risk-premium exposure, meaning removing EM names changes both risk and return meaningfully. This is flagged as a genuine confound rather than treated as unambiguous "oil alpha."

## 12. Parameter sensitivity and stability (Lecture 4 requirement)

Multiple parameter grids were tested and reported in full, rather than only reporting the chosen configuration, in order to check for overfitting:

| Sensitivity axis | Range tested | Sharpe range | Stability |
|---|---|---|---|
| Momentum window × rebalance frequency | 5/10/20/40-day momentum × Daily/Weekly/Monthly rebalance | **−0.14 to +0.63** | **Most unstable** — monthly rebalance at 40-day momentum is strongly negative, flagged as a clear overfitting trap |
| Beta window × vol half-life | 60/90/120-day beta window × 30/60/90-day vol half-life | ≈0.44 to 0.67 | Moderately stable; chosen defaults (90, 60) sit near the centre |
| VIX threshold | 25 / 30 / 35 / 40 | ≈0.53 to 0.88 | Lower threshold protects more but sacrifices upside; 35 chosen as a compromise |
| IGREA \|z\| threshold | 0.5 / 1.0 / 1.5 / 2.0 | ≈0.61 to 0.67 | Not knife-edge around the chosen default of 1.0 |
| Signal EWMA smoothing half-life | 2 / 5 / 10 / 20 days | −0.45 to 0.93 | Highly sensitive; 20-day smoothing inflates in-sample Sharpe to 0.93, which is explicitly **not adopted** to avoid overfitting to the test data |

The explicit conclusion drawn from this grid analysis is that the timing/rebalance parameters are the least stable, and the project deliberately avoids "cherry-picking" the best cell in any grid — supporting an overall narrative of a plausible economic idea with weak out-of-sample generalisation.

## 13. Bootstrap analysis

A block bootstrap (21-day blocks, 1,000 resamples) was run on the **training** period returns to build a distribution of achievable Sharpe ratios:
- Observed full-sample Sharpe: **+0.628**
- Test-period Sharpe: **−0.170**
- Bootstrap probability of Sharpe > 0 (from train resamples): **99.5%**
- Bootstrap probability of Sharpe > 0.5 (from train resamples): **85.0%**

The actual out-of-sample test Sharpe falls in the **left tail** of the distribution implied by resampling the training data — i.e., the in-sample uncertainty band does not comfortably encompass the realised OOS outcome, supporting the interpretation that the post-2021 period reflects a genuine regime shift rather than ordinary sampling variation.

## 14. Macro correlation overlay

The final model's net returns were compared against an internally constructed DXY proxy (equal-weighted basket of G10 currencies in the universe, sign-flipped), an EM carry basket (AUD/NZD/MXN/ZAR), and Brent returns directly:
- Correlation vs Brent returns: **≈ −0.05**
- Correlation vs EM carry basket: **≈ +0.04**

Both are low, supporting the claim that the strategy is not simply a disguised long-Brent or long-carry trade. The DXY proxy correlation is also low but is explicitly flagged as **not reliable**, since it is built from currencies that overlap with the trading universe itself (a genuine external DXY series was not available in the provided data).

## 15. Turnover and transaction cost validation

- Mean daily (L1) turnover: **≈0.179**
- Annualised turnover (sum of |Δw|): **≈45×** notional per year
- Mean daily transaction cost: **≈0.0051%** of notional
- Annualised cost drag: **≈1.27%**

This cost drag is material but secondary to the primary driver of underperformance (signal/regime failure out-of-sample) — annualised gross returns are of the order of a few percent, so a ~1.3% drag is non-trivial but not dominant. Costs were also flagged as conservative for G10 pairs but potentially understating true EM liquidity stress costs.

## 16. Self-critique — structural weaknesses

The project deliberately documents its own weaknesses rather than presenting only favourable results:

1. **Regime dependence:** strong in-sample, negative out-of-sample performance suggests the strategy may be fit to the demand-driven oil cycles of the 2000s–2010s rather than the post-2020 supply-shock/USD-cycle regime.
2. **Signal weakness:** average raw forecasting power is modest; portfolio engineering amplifies a genuinely thin underlying edge.
3. **EM/carry confound:** high-yield EM currencies load on general risk appetite, making it difficult to isolate a "pure oil" effect (confirmed by the jackknife results).
4. **Frequency mismatch:** daily trading against a monthly, forward-filled IGREA filter is inherently coarse.
5. **Implementation shortcuts:** the 1-month forward is derived from daily forward points; the covariance matrix zero-fills missing returns for risk estimation only (not for PnL).
6. **S2 failure:** the risk-reversal overlay may be mis-specified (sign, horizon, or RR data quality issues) — this is acknowledged as an open question rather than forced into a positive narrative.
7. **Grading philosophy:** the project explicitly treats a well-explained negative out-of-sample Sharpe as a *stronger* piece of work than a full-sample Sharpe presented without the OOS caveat.

## 17. IGREA data revision risk

IGREA is a monthly index subject to **vintage revisions** as more underlying activity data become available. The project uses the fully revised (final-vintage) series, lagged by two months as a conservative but still imperfect proxy for real-time availability. In live trading, the first-print IGREA value could differ materially from the backtested value, meaning the demand/supply classification could be wrong in real time — this risk is flagged as most acute exactly during periods like the 2022 energy crisis, where correct classification matters most.

## 18. Concrete next steps identified

| Priority | Action | Addresses |
|---|---|---|
| 1 | Use point-in-time IGREA vintages | Data revision risk |
| 2 | Source an external DXY / commodity basket | Circular proxy issue in the correlation overlay |
| 3 | Add a carry-neutral portfolio overlay | EM confound identified in the jackknife |
| 4 | Re-test S2 with a per-currency risk-reversal sign audit | Diagnosing the S2 failure |
| 5 | Build explicit regime-conditional models (demand vs supply state) | Out-of-sample failure post-2021 |
| 6 | Test a 3-month Brent / broader energy basket vs front-month | Oil benchmark robustness |
| 7 | Add higher-fidelity forwards, and INR/BRL if/when data becomes available | Universe completeness |
| 8 | Walk-forward re-estimation instead of static full-sample parameter choices | Overfitting / parameter instability |

## 19. Final assessment

The project demonstrates a coherent, economically grounded hypothesis (oil terms-of-trade transmission into FX), careful and audited data handling (no look-ahead bias), and transparent reporting of failure modes — including a negative-Sharpe overlay that was tested and correctly discarded, and an honestly reported out-of-sample Sharpe decline. It does **not** demonstrate a robust, deployable trading edge in the 2021–2026 out-of-sample window. This is presented as an intentional and defensible research conclusion for the course, prioritising honest diagnosis over headline performance.

---

### Methods reference summary

- **Signal:** rolling 90-day oil beta (cov/var regression) × 10-day Brent return kernel, cross-sectionally demeaned
- **Filters:** point-in-time expanding-OLS IGREA demand/supply residual filter (60-month warm-up); EWMA-smoothed VIX kill-switch (threshold 35)
- **Portfolio:** vol-scaled (EWMA vol), L1-normalised, 10% annualised vol target via stacked EWMA covariance (0.5y half-life / 1y seed), 200% gross cap, weekly rebalance, 1-day execution lag
- **Costs:** 2 bps (G10) / 5 bps (EM) on turnover
- **Validation:** 80/20 train/test split, block bootstrap (21-day blocks, n=1,000), leave-one-currency-out jackknife, multi-axis parameter sensitivity grids, lead/lag IR, rolling IC/t-stats, regime-window attribution, macro correlation overlay, turnover/cost analysis

*Prepared as a written companion to a group coursework Jupyter notebook (AQMS, Imperial College Business School) that cannot be shared publicly due to academic integrity restrictions. Output tables and charts referenced above are provided separately alongside this document.*