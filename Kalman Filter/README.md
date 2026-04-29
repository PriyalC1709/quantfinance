# Kalman Filter Pairs Trading — META/QQQ
### A Three-Stage Research Project

A progressively rigorous implementation of a statistical arbitrage strategy on the META/QQQ pair, evolving from a baseline Kalman Filter pairs trade to a full Sequential Monte Carlo comparison with formal statistical validation. Each stage builds directly on the previous, with the research arc moving from in-sample proof-of-concept to realistic out-of-sample deployment to principled filter comparison.

---

## Repository Structure

```
kalman-pairs-trading/
│
├── Stage1/
│     ├── kalman_pairs_stage1.ipynb
│     ├── kalman_pairs.png
│
├── Stage2/
│     ├── kalman_pairs_stage2.ipynb
│     ├── kalman_pairs_stage2_oos.png
│
├── Stage3/
│     ├── kalman_pairs_stage3.ipynb
│     ├── kalman_vs_particle_stage3.png
│
└── README.md                ← this file
```

---

## Research Arc

```
Stage 1 — Does the Kalman Filter produce a tradeable spread?
    ↓
Stage 2 — Does the strategy survive realistic conditions?
    ↓
Stage 3 — Does relaxing Gaussian assumptions improve the filter?
```

The progression is deliberate. Stage 1 establishes the mechanism. Stage 2 stress-tests it with proper statistical validation, out-of-sample testing, and transaction costs. Stage 3 asks whether the core Gaussian assumption is a binding constraint — and finds that it is not, at daily frequency.

---

## The Pair — META/QQQ

META (Meta Platforms) and QQQ (Nasdaq-100 ETF) share substantial common factor exposure — large-cap technology, growth, and sentiment — making them candidates for a dynamic pairs trade. However, formal cointegration testing across multiple time windows and test statistics consistently fails to confirm a stationary long-run relationship.

This is an intentional and honest finding. The strategy does not rely on cointegration. It exploits short-run conditional mean reversion in a dynamically hedged spread, where the hedge ratio is estimated bar-by-bar rather than fixed at a static OLS coefficient. The Kalman filter produces a stationary spread (ADF p=0.000) even when no static linear combination of the two prices is stationary — this is the mathematical justification for the dynamic approach.

---

## Data

| Split | Period | Frequency | Bars | Purpose |
|-------|--------|-----------|------|---------|
| Train | 2012-05-18 → 2022-12-31 | Daily | 2,673 | MLE calibration, cointegration testing only |
| Test | 2023-01-03 → 2026-04-28 | Daily | 832 | Backtest only — never touched during training |

**Train/test discipline:** Parameters are estimated exclusively on training data and frozen before any test data is examined. The test period is genuinely out-of-sample. META's IPO date (May 18, 2012) defines the earliest possible start date.

---

## Stage 1 — Kalman Filter Baseline

### What it does

Implements a pairs trading strategy where the hedge ratio between META and QQQ is modelled as a latent state variable tracked by a Kalman Filter. Rather than using a fixed OLS regression coefficient, the hedge ratio updates bar-by-bar via Bayesian inference, adapting to structural shifts in the relationship between the two assets.

### State-Space Model

Observation equation:

$$\text{META}_t = \beta_{0,t} + \beta_{1,t} \cdot \text{QQQ}_t + \varepsilon_t$$

State evolution (random walk):

$$\beta_t = \beta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, V_w), \quad V_w = \frac{\delta}{1-\delta} I_2$$

Kalman update equations at each bar $t$, given $F_t = [1, \text{QQQ}_t]$:

$$\hat{\beta}_{t|t-1} = \beta_{t-1}$$
$$P_{t|t-1} = P_{t-1} + V_w$$
$$e_t = \text{META}_t - F_t^\top \hat{\beta}_{t|t-1}$$
$$Q_t = F_t^\top P_{t|t-1} F_t + V_e$$
$$K_t = P_{t|t-1} F_t / Q_t$$
$$\beta_t = \hat{\beta}_{t|t-1} + K_t e_t$$
$$P_t = P_{t|t-1} - K_t F_t^\top P_{t|t-1}$$

### Z-Score Normalisation

Standard rolling std was systematically compressed by fat-tailed spread distributions. Replaced with Median Absolute Deviation (MAD), scaled by 1.4826 to match std under Gaussian assumptions:

$$z_t = \frac{e_t - \text{median}(e_{t-w:t})}{\text{MAD}(e_{t-w:t}) \times 1.4826}$$

Rolling window: 78 bars (3 trading days of 15-min bars).

### Signal Logic

| Condition | Action |
|-----------|--------|
| $z_t < -2\sigma$ | Enter long spread (buy META, sell QQQ) |
| $z_t > +2\sigma$ | Enter short spread (sell META, buy QQQ) |
| $z_t$ crosses 0 | Exit position |

### Stage 1 Results

| Metric | Value |
|--------|-------|
| Data | 15-min bars, 59-day window (Feb–Apr 2026) |
| Total Return (gross) | 4.04% |
| Sharpe Ratio (gross) | 3.00 |
| Max Drawdown | -2.13% |
| Win Rate | 53.64% |
| Bars in Market | 110 / 1,091 |

**Note on Stage 1 Sharpe:** The gross Sharpe of 3.00 is an in-sample result on a 59-day window with no transaction costs and no out-of-sample validation. It is not claimed as a realistic performance figure — Stage 2 addresses all three limitations.

---

## Stage 2 — Robustness & Realism

### What it adds

Four substantive improvements over Stage 1:
1. Formal cointegration testing suite on 10.6 years of training data
2. MLE calibration of noise parameters on training data — frozen before test period
3. Transaction cost model with breakeven analysis
4. Bootstrapped Sharpe confidence interval

### Cointegration Testing Suite

All tests run on training data (2012-2022) only.

**ADF on individual series:** Both META (p=0.496) and QQQ (p=0.827) confirm non-stationarity.

**Engle-Granger:** P-value = 0.877 — cointegration not detected. The pair does not share a stationary long-run relationship, consistent with structural breaks from the Fed rate cycle and AI valuation re-rating.

**Johansen:** Trace statistic 12.25 vs 95% critical value 15.49 — fails to reject no cointegration.

**OLS Spread:** Static hedge ratio of 0.832 produces non-stationary spread (ADF p=0.707). Half-life via Ornstein-Uhlenbeck AR(1): 398 days — too slow for systematic pairs trading.

**Kalman Spread:** Dynamic hedge ratio produces strongly stationary spread (ADF p=0.000). The filter is justified from first principles — it extracts stationarity that no static combination can achieve.

**Note:** Kalman filter innovations are white noise by construction. The half-life concept applies to the raw OLS spread, not to Kalman innovations which have already had autocorrelation extracted.

### MLE Calibration

Maximises the log-likelihood of observing the price sequence under the filter model:

$$\mathcal{L}(\delta, V_e) = -\frac{1}{2} \sum_{t=1}^{n} \left( \log Q_t + \frac{e_t^2}{Q_t} \right)$$

L-BFGS-B bounded optimisation. Ve bounded below at 0.01 — observation noise cannot be zero for a liquid equity. Parameters frozen after training.

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| δ | 2.1246e-04 | Hedge ratio adaptation speed |
| Ve | 1.00e-02 | Observation noise floor |

### Vol-Scaled Entry Threshold

$$\text{threshold}_t = k \times \sigma_{z,t}, \quad k = 1.5$$

Where $\sigma_{z,t}$ is the 20-day rolling std of the z-score. Threshold rises during high-vol periods (harder to enter) and falls during low-vol periods (easier to enter).

### Execution

**Execution lag:** Signal at bar t-1 close, position entered at bar t close — approximates next-day open, eliminates lookahead bias.

**Transaction costs:** 3 bps realistic (META ~1.5bps + QQQ ~0.5bps + slippage ~1bps), 6 bps conservative. Applied once per complete round-trip trade.

### Stage 2 Results

| Metric | Value |
|--------|-------|
| Test period | 2023-01-03 → 2026-04-28 (3.3 years, OOS) |
| Total Return (net 3bps) | 9.53% |
| Sharpe Ratio (gross) | 0.36 |
| Sharpe Ratio (net 3bps) | 0.30 |
| Max Drawdown | -16.83% |
| Win Rate | 50.68% |
| Complete Trades | 71 |
| Mean Holding Period | 2.1 days |
| Breakeven Cost | ~19 bps per round trip |
| Bootstrapped 95% CI | [-0.87, 1.24] |
| Positive Sharpe Probability | 69.8% |

**On the Sharpe:** The 0.30 net Sharpe over 3.3 years is a defensible, realistic figure for a daily pairs strategy on a non-cointegrated pair. The wide bootstrapped CI reflects both the short sample and the modest signal — a known limitation. A strategy with Sharpe 0.30 requires approximately 7-8 years of data to statistically confirm the edge at 95% confidence.

**On the drawdown:** The -16.83% maximum drawdown spans mid-2024 through mid-2025, coinciding with META's AI valuation re-rating. This reflects the cost of trading a non-cointegrated pair — there is no structural guarantee of spread reversion, only empirical short-run mean reversion that can fail during regime shifts.

---

## Stage 3 — Particle Filter Comparison

### What it asks

Whether relaxing the Gaussian assumption in the state-space model generates economically meaningful differences in hedge ratio estimation or trading performance.

### Filter Specification

Both filters estimate a 1D state space — hedge ratio β₁ only, intercept fixed at zero:

$$\text{META}_t = \beta_{1,t} \cdot \text{QQQ}_t + \varepsilon_t$$

The intercept is fixed because it absorbs a persistent price level difference that is not tradeable. Estimating it dynamically introduces unnecessary degrees of freedom that destabilise cross-filter comparison at high price levels.

**Kalman Filter:** Gaussian state evolution and observation likelihood — closed-form Bayesian update.

**Particle Filter (SMC):** N=1000 particles, Student-t state evolution and likelihood with ν=5 degrees of freedom, systematic resampling at each bar.

Three-step SMC cycle:
1. Propagate: each particle evolves with independent Student-t noise
2. Weight: particles weighted by Student-t likelihood of observation (log-space for numerical stability)
3. Resample: systematic resampling concentrates mass on well-performing particles

**Why Student-t (ν=5):** Finite variance (ν>2), fat tails meaningful for financial data (ν<10). During earnings announcements, Gaussian likelihood assigns near-zero weight to all particles after a 5σ move — weight collapse. Student-t assigns non-negligible weight to particles closest to the observation — filter survives and adapts.

**Identical parameters for fair comparison:**

| Parameter | Value | Source |
|-----------|-------|--------|
| δ | 2.1246e-04 | MLE on training data |
| Ve | 1.00e-02 | MLE on training data |
| N (particles) | 1,000 | SMC literature standard |
| ν | 5 | SMC literature standard |
| Warmup | 120 bars | Both filters, identical training tail |

**Warmup:** Both filters warm up on the last 120 bars of training data. After warmup: Kalman β₁ = 0.4571, Particle β₁ = 0.4570 — difference of 0.0001. Starting points are economically equivalent.

### Stage 3 Results

**Hedge Ratio Comparison:**

| Metric | Kalman | Particle | Difference |
|--------|--------|---------|------------|
| β₁ mean | 1.0268 | 1.0265 | -0.0002 |
| β₁ std | 0.2130 | 0.2129 | 0.0001 |
| Mean abs difference | — | — | 0.000601 |
| β₁ correlation | — | — | 0.9993 |

**Regime-Specific Divergence:**

| Regime | Mean \|KF - PF\| | Relative to Normal |
|--------|-----------------|-------------------|
| Normal | 0.000601 | 1.0x |
| Tariff shock (Apr 2026) | 0.000045 | 0.07x |
| Earnings windows | 0.005053 | 8.4x |

The particle filter diverges most during earnings windows — discrete single-day jump events. During the tariff shock — a sustained multi-day macro trend — both filters produce near-identical estimates. Fat-tail benefits are event-specific, not regime-specific.

**Diebold-Mariano Test:**

| Statistic | Value |
|-----------|-------|
| P-value (two-sided) | 0.319 |
| Significant at 5%? | No |

No statistically detectable difference in hedge ratio forecast accuracy.

**Computational Benchmarking:**

| Filter | Runtime | Overhead |
|--------|---------|---------|
| Kalman | 0.01 seconds | 1x |
| Particle (N=1000) | 0.91 seconds | ~90x |

**Note on backtest comparability:** Direct PnL comparison is not feasible at daily frequency. Despite β₁ correlation of 0.9993, z-score correlation between filter spreads is only 0.14. Tiny β₁ differences (0.0006) are amplified by high QQQ price levels (~$500) into spread differences exceeding the rolling MAD window. The Diebold-Mariano test is the appropriate comparison metric.

### Central Research Question — Answer

> Does relaxing the Gaussian assumption generate economically meaningful alpha during fat-tail events, and is the 90x computational overhead of SMC justified?

**No, at daily frequency with MLE-calibrated parameters.**

Hedge ratio estimates are 99.93% correlated. Diebold-Mariano finds no significant forecast accuracy difference (p=0.319). The 8x divergence during earnings windows is consistent with theory but economically negligible. The Gaussian assumption is not a binding constraint. The Kalman filter is the recommended production implementation.

**Where the particle filter may add value:** At tick or 1-minute frequency, where earnings announcements produce extreme single-bar innovations that stress Gaussian likelihood more severely. Left for future work.

---

## Summary Across All Stages

| Stage | Focus | Sharpe | Key Finding |
|-------|-------|--------|-------------|
| Stage 1 | Kalman Filter, 15-min, in-sample | 3.00 gross | Dynamic hedge produces stationary spread |
| Stage 2 | MLE calibration, OOS test, costs | 0.30 net (3bps) | Strategy survives realistic conditions |
| Stage 3 | Particle Filter comparison | — | Gaussian assumption not binding at daily frequency |

---

## Intellectual Framework

The core insight — that a dynamic hedge ratio can produce a stationary spread even when no static combination of the prices is stationary — connects to the broader question of what makes a pair tradeable. The Kalman filter is optimal under Gaussian assumptions. The particle filter relaxes those assumptions at the cost of computational tractability.

The empirical finding that both produce equivalent results at daily frequency is informative: the distributional assumption is not the binding constraint for this strategy. The binding constraint is the absence of formal cointegration. The strategy trades correlation mean reversion rather than a structurally guaranteed relationship, and no filter assumption changes that fundamental limitation.

The Sharpe progression from 3.00 (Stage 1) to 0.30 (Stage 2) reflects the cost of honesty — moving from in-sample, no-cost, short-window estimation to genuine out-of-sample validation with realistic execution costs. The 0.30 net Sharpe is a defensible figure for this pair and frequency.

---

## Dependencies

```
yfinance
numpy
pandas
scipy
statsmodels
matplotlib
```

---

## References

- Kalman, R.E. (1960). A new approach to linear filtering and prediction problems
- Gordon, Salmond & Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation
- Engle & Granger (1987). Co-integration and error correction
- Johansen (1991). Estimation and hypothesis testing of cointegration vectors
- Diebold & Mariano (1995). Comparing predictive accuracy
- Ghashghaie et al. (1996). Turbulent cascades in foreign exchange markets