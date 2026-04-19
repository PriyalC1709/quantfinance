# Project 2: GARCH Dynamic Risk Engine

A dynamic volatility forecasting engine using GARCH(1,1) integrated with
Black-Scholes delta hedging, demonstrating a 22% reduction in hedging error
variance versus a rolling window benchmark across 20 years of SPY data.

Part 2 of 3 in a quantitative derivatives research series:
**Greeks Engine → GARCH Risk Engine → P&L Attribution Engine**

---

## Motivation

Project 1 assumed volatility was a fixed input — one sigma, constant across
time. In reality volatility is dynamic: large moves cluster together, calm
periods persist, and vol spikes during stress regimes can be an order of
magnitude above normal levels. A static vol assumption produces systematically
wrong Greeks, and wrong Greeks produce hedging errors that accumulate over time.

GARCH(1,1) addresses this by modelling volatility as a time-varying process
that adapts to new information. The key question this project answers is:
**does a better volatility forecast actually reduce hedging error, and by how much?**

---

## The GARCH(1,1) Model

Volatility is updated each period according to:

$$\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

| Parameter | Description | Estimated Value |
|-----------|-------------|-----------------|
| $\omega$ | Long-run baseline variance | 0.0283 |
| $\alpha$ | Weight on yesterday's shock | 0.1318 |
| $\beta$ | Persistence of prior variance | 0.8435 |

**Stationarity condition:** $\alpha + \beta < 1$

Our result: $0.1318 + 0.8435 = 0.9753$ — stationary but highly persistent,
consistent with the well-documented long memory of equity volatility.

**Long-run annualised volatility:**

$$\sigma_{LR} = \sqrt{\frac{\omega}{1 - \alpha - \beta} \times 252} = 16.99\%$$

Consistent with long-run SPY realised volatility.

---

## What This Project Builds

**Data Pipeline**
20 years of SPY daily closing prices (2004–2024), log return computation,
and 21-day rolling volatility as the naive benchmark.

**GARCH Estimation**
Maximum likelihood estimation of GARCH(1,1) parameters using the `arch`
library. Conditional volatility extracted and annualised for comparison.

**Volatility Comparison**
GARCH conditional vol plotted against rolling vol across the full sample,
with stress regimes shaded — GFC (2008), COVID (2020), Rate Shock (2022).

**Delta Hedging Simulation**
Daily delta hedging of a synthetic ATM call option simulated across the
full 20-year sample using both vol estimates. Hedging error computed as
the difference between theoretical hedge P&L and actual option P&L.

**Regime Analysis**
VIX-based regime classification (low: VIX < 15, normal: 15–25, stress: > 25)
used to decompose hedging error reduction across market conditions.

---

## Key Results

### Overall Hedging Error Reduction

| Method | Error Variance | Reduction |
|--------|---------------|-----------|
| Rolling Vol (21-day) | 0.019288 | — |
| GARCH(1,1) | 0.015054 | **22.0%** |

### Reduction by Volatility Regime

| Regime | VIX Level | Rolling Var | GARCH Var | Reduction | Days |
|--------|-----------|-------------|-----------|-----------|------|
| Low | < 15 | 0.005128 | 0.003338 | **34.9%** | 2062 |
| Normal | 15–25 | 0.021060 | 0.016432 | **22.0%** | 2353 |
| Stress | > 25 | 0.048438 | 0.038945 | **19.6%** | 846 |

---

## Key Finding

GARCH's largest advantage is not during stress itself but during **regime
transitions** — specifically the recovery out of stress back into low vol.
Rolling vol remains elevated for weeks after conditions normalise because
bad days persist in its 21-day window. GARCH mean-reverts toward true vol
faster, producing more accurate deltas precisely when the hedge matters most.

During extreme stress, both models are wrong — vol moves too fast for either
to fully capture. The 19.6% reduction in stress confirms GARCH is still
superior, but the margin narrows because no backward-looking model can fully
anticipate a vol spike of COVID or GFC magnitude.

---

## Key Outputs

| Output | Description |
|--------|-------------|
| `returns_rolling_vol.png` | SPY log returns and 21-day rolling vol 2004–2024 |
| `garch_vs_rolling.png` | GARCH conditional vol vs rolling vol with regime shading |
| `hedging_errors.png` | Hedging error time series and distributions |
| `regime_errors.png` | Error distributions decomposed by volatility regime |

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample | 2004–2024 | 5,283 trading days |
| Asset | SPY | S&P 500 ETF |
| Option | ATM Call | Strike = spot, rolled monthly |
| Tenor | 3 months | Initial time to expiry |
| $r$ | 0.05 | Risk-free rate |
| $q$ | 0.013 | Continuous dividend yield |
| Rolling window | 21 days | ~1 trading month |

---

## Dependencies

```python
numpy
pandas
matplotlib
yfinance
arch
scipy
```

---

## Connection to Project 3

GARCH improves vol estimation and reduces hedging error — but even with a
perfect vol forecast, BSM-based delta hedging cannot fully replicate an
option's P&L. The remaining error comes from gamma convexity, vega exposure,
theta decay, and higher-order terms that the first-order delta hedge ignores.

Project 3 decomposes that residual explicitly — attributing realised options
P&L to each Greek component across volatility regimes, and quantifying where
the Black-Scholes framework systematically breaks down.

---

*Author: Priyal Chhawchharia — MSc Risk Management & Financial Engineering,
Imperial College London*
