# QRT Data Challenge

**Predicting Next-Day Portfolio Return Direction**

> Priyal Chhawchharia · MSc Risk Management & Financial Engineering · Imperial College London

| | |
|---|---|
| **Leaderboard Score** | 0.5099 |
| **Benchmark Score** | 0.5079 |
| **Result** | Exceeded benchmark |
| **Participants** | ~1,000 |

---

## 1. Challenge Overview

The QRT Data Challenge asks competitors to predict the **sign** of a portfolio allocation's next-day return using 20 days of historical data. Each row in the dataset represents a unique (date, allocation) snapshot.

**Inputs per row:**
- `RET_1` … `RET_20` — 20 days of allocation returns (RET_1 = most recent)
- `SIGNED_VOLUME_1` … `SIGNED_VOLUME_20` — signed weighted volume per day
- `MEDIAN_DAILY_TURNOVER` — median rebalancing intensity over the window
- `GROUP` — anonymised allocation style group (1–4)

**Target:** Predict `1` (positive next-day return) or `0` (negative). Evaluated on accuracy.

**Key constraints:**
- Dates are anonymised and shuffled — no temporal ordering can be assumed across rows
- 527,073 training rows · 31,870 test rows · 278 allocations · 2,522 dates · 4 groups
- Near-perfect class balance (50.7% positive) — true 50% baseline

---

## 2. Exploratory Data Analysis

EDA was conducted before any modelling. All modelling decisions flow from what the data showed.

### 2.1 Missing Values

| Column | % Missing | Action |
|---|---|---|
| `SIGNED_VOLUME_1` | 73% train, 75% test | **Dropped — unusable** |
| `SIGNED_VOLUME_18–20` | < 2% | Filled with 0 (boundary artifact) |
| `RET_18–20` | < 0.01% | Filled with 0 (boundary artifact) |
| `MEDIAN_DAILY_TURNOVER` | 0.7% | Filled with median |

### 2.2 The Dominant Signal: RET_1

The single most important EDA finding:

| Feature | Correlation with TARGET |
|---|---|
| `RET_1` | **+0.085** |
| `RET_2` | −0.009 |
| `RET_3` | −0.012 |
| `RET_5–RET_20` | < ±0.003 |
| All volume lags | < ±0.008 |

RET_1 is the only feature with meaningful predictive signal. This was confirmed by a decile analysis — binning rows by RET_1 showed a perfectly monotonic relationship with next-day positive return probability (46.9% bottom decile → 54.8% top decile).

**Momentum accuracy by group:**

| Group | Accuracy (sign RET_1 → predict label) |
|---|---|
| G1 | 52.3% |
| G2 | 52.2% |
| G3 | 51.5% |
| G4 | 51.8% |

All groups exhibit momentum — none show mean reversion. Group 1 has the strongest signal.

### 2.3 Validation Strategy

**Critical finding:** A random 80/20 train-validation split produces severely inflated accuracy (~57%) that has no relationship to leaderboard scores (~50.5%). Because dates are shuffled, a random split shares the same date distribution between train and val, masking temporal overfitting.

**Fix:** Time-aware split — train on first 80% of anonymised timestamps, validate on last 20%. This produced estimates within ~1.3pp of actual leaderboard scores and was used for all model comparison.

---

## 3. Feature Engineering

Features were built in three layers.

### Layer 1 — Allocation-Level Features

| Feature | Description |
|---|---|
| `abs_ret1` | \|RET_1\| — larger moves carry stronger momentum signal |
| `ret1_signed_mag` | sign(RET_1) × log(1 + \|RET_1\| × 1000) — log-scaled directional magnitude |
| `ret1_ret2_agree` | 1 if RET_2 confirms RET_1 direction, else 0 |
| `ret_vol_5/10/20` | Return volatility over 5, 10, 20-day windows |
| `ret_mean_5/10/20` | Mean return over multiple windows |
| `log_turnover` | log(1 + MEDIAN_DAILY_TURNOVER) |

### Layer 2 — Cross-Sectional Features (Key Innovation)

On any given date, ~276 allocations are observed simultaneously. Computing statistics *across* allocations on the same date captures market-wide regime information unavailable from any single allocation's history.

| Feature | Description |
|---|---|
| `xs_mean_ret1/5/10/20` | Average return across all allocations on the same date |
| `xs_std_ret1/5/20` | Dispersion of returns across allocations — **#1 ranked feature by LightGBM** |
| `xs_mean_vol5/20` | Average volatility across allocations |
| `rel_ret1` | RET_1 minus xs_mean_ret1 — performance relative to all peers |
| `rel_ret_mean` | Window mean minus cross-sectional average |

### Layer 3 — Rank Features

| Feature | Description |
|---|---|
| `rank_ret1/5/20` | Percentile rank within all allocations on same date — scale-invariant relative performance |
| `group_rank_ret1/20` | Percentile rank within same GROUP on same date |

**Top 9 LightGBM features were all cross-sectional**, confirming that an allocation's relative standing among peers is more predictive than its absolute return history.

---

## 4. Models & Results

All comparisons use time-aware validation.

| Model | Time-aware Val | Leaderboard |
|---|---|---|
| Logistic Regression — RET_1 only | 52.24% | — |
| Logistic Regression — all lags | ~52% | 0.5042 |
| LightGBM — default, all features | 52.25% | 0.5056 |
| **LightGBM — cross-sectional features** | **52.25%** | **0.5099 ✓** |
| LightGBM — regularised | 52.76% | 0.5084 |
| LightGBM — group XS + rank features | 52.44% | 0.5046 |

**Best submission:** LightGBM with cross-sectional features, trained on full dataset.

```
n_estimators      = 200
learning_rate     = 0.05
num_leaves        = 31
min_child_samples = 50
subsample         = 0.8
colsample_bytree  = 0.8
```

### Key Lessons

- **Random CV lies here.** Shuffled dates mean random splits share date distribution with training, inflating val accuracy by 5–6pp. Time-aware splits are the only reliable guide.
- **More features ≠ better generalisation.** Group-specific and interaction features improved val accuracy but hurt leaderboard scores — temporal overfitting in disguise.
- **RET_1 importance is underestimated by LightGBM.** Its linear signal requires few splits to capture, so split-based importance ranks it low despite it being the core signal.
- **Cross-sectional features generalise well.** Unlike group-specific features, market-wide statistics transferred cleanly to the test set.

---

## 5. Repository Structure

```
quantfinance/
├── README.md                           ← this file
├── QRT Data Challenge/
│   ├── QRTdatachallenge.ipynb          ← full modelling notebook
│   ├── outputs/                        ← figures (gitignored)
│   └── dataset/                        ← raw CSVs (gitignored)
└── .gitignore
```

---

## 6. Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `LightGBM` · `matplotlib` · `seaborn`

No external data sources. All features derived from the provided dataset.