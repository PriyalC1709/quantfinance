# Physics × Finance

Quantitative finance models viewed through the lens of physics — diffusion theory, thermodynamics, and fluid mechanics. Built alongside MSc Risk Management & Financial Engineering coursework at Imperial College London.

The central thesis: financial markets are diffusive, non-equilibrium systems. The mathematics of heat flow, turbulence, and entropy are not analogies — they are the same governing equations, applied to a different domain.

---

## Contents

### `week1_diffusion_regimes.py`
**Diffusion regimes in asset price dynamics**

Simulates 10,000 price paths under three increasingly realistic diffusion models and compares their return distributions using Shannon entropy.

| Model | Physical analogue | Key assumption |
|---|---|---|
| Black-Scholes (Gaussian) | Pure Fickian diffusion | Constant diffusivity, no memory |
| Student-t | Anomalous diffusion | Heavy tails from turbulent intermittency |
| Merton Jump Diffusion | Diffusion with shocks | Lévy-type discontinuities in the medium |

**What the entropy output tells you:** Shannon entropy quantifies the spread of each return distribution. The ordering — Gaussian < Student-t < Jump Diffusion — reflects increasing departure from pure diffusive behaviour. In physical terms: jump diffusion has the highest entropy because it explores the most state space, analogous to a turbulent rather than laminar flow regime.

**Output:** return distribution histograms and QQ plots against the Gaussian benchmark, making fat tails and jump discontinuities directly visible.

---

## The physics connection

The Black-Scholes PDE:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

is the heat equation under the substitution $x = \log(S/K)$, $\tau = T - t$:

$$\frac{\partial u}{\partial \tau} = \frac{1}{2}\sigma^2 \frac{\partial^2 u}{\partial x^2}$$

The diffusivity is $\frac{1}{2}\sigma^2$ — volatility is the thermal diffusivity of the market. The models in this repository explore what happens when that diffusivity is not constant: fat tails arise from intermittent diffusivity (Student-t), and jump diffusion introduces discontinuous transport analogous to shock waves.

---

## Related projects

| Project | Description |
|---|---|
| `fBm_Hurst` | Fractional Brownian motion and Hurst exponent estimation — rough diffusion, connecting to turbulence energy spectra |
| `GARCH_Risk_Engine` | Volatility clustering as discrete energy cascade — GARCH(1,1) as a Kolmogorov-type model |
| `BS_Greeks_Engine` | Vectorised Black-Scholes Greeks — sensitivity analysis across vol surfaces |
| `Kalman_Pairs_Trading` | State-space filtering applied to statistical arbitrage |

---

## Stack

Python 3.11 · NumPy · SciPy · Matplotlib · Pandas

---

*MSc Risk Management & Financial Engineering, Imperial College London (2025–26)*  
*BSc Aerospace Engineering, Purdue University*
