# Kalman Filter Pairs Trading — META/QQQ

A three-stage research project implementing and extending a statistical arbitrage strategy on the META/QQQ pair using a Kalman Filter for dynamic hedge ratio estimation.

---

## Project Structure

```
kalman-pairs-trading/
│
├── Stage1/          # Kalman Filter — core strategy
├── Stage2/          # Robustness & realism (cointegration, MLE, transaction costs)
├── Stage3/          # Particle filter comparison & regime analysis
└── README.md
```

---

## Stage 1 — Kalman Filter Strategy (Complete)

### Overview
Implements a pairs trading strategy where the hedge ratio between META and QQQ is modelled as a latent state variable tracked by a Kalman Filter. Rather than using a fixed OLS regression coefficient, the hedge ratio updates bar-by-bar via Bayesian inference, adapting to structural shifts in the relationship between the two assets.

### Methodology

**State-Space Model**

The observation equation models META as a linear function of QQQ:

$$\text{META}_t = \beta_0 + \beta_1 \cdot \text{QQQ}_t + \varepsilon_t$$

The state vector $[\beta_0, \beta_1]$ evolves as a random walk:

$$\beta_t = \beta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, V_w)$$

**Kalman Update Equations**

At each bar $t$, given observation vector $F_t = [1, \text{QQQ}_t]$:

- **Predict:** $\hat{\beta}_{t|t-1} = \beta_{t-1}$, $P_{t|t-1} = P_{t-1} + V_w$
- **Innovation:** $e_t = \text{META}_t - F_t^\top \hat{\beta}_{t|t-1}$
- **Innovation variance:** $Q_t = F_t^\top P_{t|t-1} F_t + V_e$
- **Kalman gain:** $K_t = P_{t|t-1} F_t / Q_t$
- **Update:** $\beta_t = \hat{\beta}_{t|t-1} + K_t e_t$, $P_t = P_{t|t-1} - K_t F_t^\top P_{t|t-1}$

Noise parameters: $\delta = 10^{-4}$, $V_w = \frac{\delta}{1-\delta} I_2$, $V_e = 0.001$

**Z-Score Normalisation — MAD**

Standard rolling std was systematically compressed by fat-tailed spread distributions driven by the April 2026 tariff shock. Replaced with Median Absolute Deviation (MAD), scaled by 1.4826 to match std under Gaussian assumptions:

$$z_t = \frac{e_t - \text{median}(e_{t-w:t})}{\text{MAD}(e_{t-w:t}) \times 1.4826}$$

Rolling window: 78 bars (3 trading days of 15-min bars).

**Signal Logic**
| Condition | Action |
|-----------|--------|
| $z_t < -2$ | Enter long spread (buy META, sell QQQ) |
| $z_t > +2$ | Enter short spread (sell META, buy QQQ) |
| $z_t$ crosses 0 | Exit position |

**Filters Applied**
- Market open bar removed (13:30 UTC / 09:30 ET) — bid-ask spreads widest, price discovery unreliable
- Tariff shock regime excluded (Apr 7–10 2026) — correlation structure broke down as VIX spiked; pairs assumptions invalid

### Results

| Metric | Value |
|--------|-------|
| Total Return | 4.04% |
| Sharpe Ratio | 3.00 |
| Max Drawdown | -2.13% |
| Win Rate | 53.64% |
| Bars in Market | 110 / 1,091 |
| Data Window | Feb 25 – Apr 24 2026 (59 days) |
| Bar Frequency | 15-minute |

### Key Observations
- Hedge ratio $\beta_1$ drifted smoothly from ~1.08 (Feb) → 0.93 (tariff shock, Mar–Apr) → 1.03 (Apr recovery), demonstrating the filter correctly adapted to regime change without manual intervention
- MAD normalisation produced a z-score distribution with std 1.14 versus 0.63–0.75 under rolling std, resolving the signal compression issue
- Strategy is selective: only 10% of bars in market, with balanced long/short exposure (52 long, 58 short bars)

### Limitations
- 59-day window is short; daily bar robustness check across 2–3 years planned in Stage 2
- No transaction costs modelled in Stage 1
- Tariff shock exclusion flatters the Sharpe; regime-conditional performance is a Stage 3 deliverable
- Noise parameters ($\delta$, $V_e$) hand-tuned; MLE calibration planned for Stage 2

### Dependencies
```
yfinance
numpy
pandas
scipy
matplotlib
```

---

## Stage 2 — Robustness & Realism (Planned)

- Engle-Granger and Johansen cointegration tests with half-life calculation
- MLE calibration of Kalman noise covariance ($\delta$, $V_e$)
- Transaction costs — bid-ask spread and slippage model
- Bootstrapped Sharpe confidence intervals (10,000 samples)
- Extended daily bar backtest for multi-year robustness

---

## Stage 3 — Particle Filter Comparison (Planned)

- Particle filter (Sequential Monte Carlo) with Student-t transition noise, $N = 500$ particles
- Identical backtest framework applied to both Kalman and particle filter
- Diebold-Mariano test on hedge ratio forecast errors
- Regime-conditional performance: earnings windows, FOMC days, high-VIX periods, tariff shock
- Computational cost benchmarking — whether SMC overhead is justified by alpha improvement

---


*Part of a broader quant research portfolio. See also: BS Greeks Engine (Stage 1–2 complete), fBm/Hurst Exponent & Kolmogorov Scaling (in progress).*