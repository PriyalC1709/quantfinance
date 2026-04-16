# Project 1: Option Greeks Sensitivity Engine

A vectorised Black-Scholes-Merton Greeks engine with second-order cross-Greeks,
3D surface visualisation, and Gamma-pin risk identification.

Part 1 of 3 in a quantitative derivatives research series:
**Greeks Engine → GARCH Risk Engine → P&L Attribution Engine**

---

## Motivation

Black-Scholes-Merton provides closed-form sensitivity measures that decompose
an option's risk into interpretable components. Most implementations stop at
first-order Greeks and assume a flat volatility surface. This engine extends
the standard framework to include second-order cross-Greeks — Vanna, Volga,
and Charm — and explicitly maps where the flat-vol assumption breaks down.
That breakdown is what the residual in the attribution engine (Project 3)
will quantify empirically.

---

## What This Project Builds

**BSM Pricing**
Generalised Black-Scholes-Merton with continuous dividend yield. Includes
an edge case handler for expiry day (T=0) returning intrinsic value.

**First-Order Greeks**
Delta, Vega, Theta, Rho — analytical closed-form implementations,
validated against put-call parity and known boundary conditions.

**Second-Order Greeks**
Gamma, Vanna, Volga, Charm — including cross-Greeks that capture
how delta and vega behave when both spot and volatility move simultaneously.

**Implied Volatility Solver**
Newton-Raphson inversion from market price to implied volatility.
Converges to machine precision in under 10 iterations for typical inputs.

**Surface Generation**
Vectorised meshgrid computation across spot × expiry and spot × strike
dimensions. Produces 3D Gamma and Vanna surfaces.

**Vol Smile and Skew Analysis**
Parametric smile construction showing where BSM's flat-vol assumption
systematically misprices OTM puts and calls relative to market prices.

**Gamma-Pin Risk Map**
Overlay of open interest distribution and Gamma concentration at short
DTE, illustrating the mechanical pinning of spot near high-OI strikes
on expiry dates.

---

## Key Outputs

| Output | Description |
|--------|-------------|
| `gamma_vanna_surfaces.png` | 3D Gamma and Vanna surfaces across spot and expiry |
| `greek_profile_panel.png` | Six-panel Greek profiles for a long straddle |
| `vol_smile.png` | Implied vol smile across three expiries |
| `flat_vs_smile.png` | BSM flat vol vs market smile — mispricing region |
| `gamma_pin.png` | Gamma-pin risk map at 5 DTE |

---

## Interesting Findings

**Gamma explodes near expiry ATM.** The 3D Gamma surface shows a sharp spike
at short expiry and ATM spot — the "gamma trap" that short-dated options
sellers are exposed to near expiry dates.

**Vanna changes sign across ATM.** Positive for OTM calls, negative for OTM
puts. This means delta hedges behave asymmetrically when vol and spot move
together — a key driver of the residual in the attribution engine.

**BSM systematically misprices the wings.** The flat-vol assumption
underprices OTM puts and overprices OTM calls relative to what equity
markets actually charge. This is the volatility skew — a direct consequence
of crash risk that BSM's log-normal assumption cannot capture.

**Gamma pinning concentrates at round-number strikes.** At 5 DTE, Gamma
peaks precisely where open interest is highest. Market maker delta-hedging
flows at these strikes create a gravitational pull on spot price near expiry.

---

## Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| S | 500 | Spot price (SPY-like scale) |
| K | 500 | Strike price (ATM) |
| T | 0.25 | Time to expiry (3 months) |
| r | 0.05 | Risk-free rate |
| q | 0.013 | Continuous dividend yield |
| σ | 0.20 | Implied volatility |

---

## Dependencies

```python
numpy
scipy
matplotlib
```

---

## Connection to Project 2

The Greeks engine assumes volatility is constant — one sigma input,
flat across time and strikes. In reality volatility clusters: large moves
follow large moves, and calm periods persist. GARCH RisK Engine replaces the static
vol input with a GARCH(1,1) forecast, and measures how much that dynamic
adjustment reduces hedging error versus a rolling window benchmark.

