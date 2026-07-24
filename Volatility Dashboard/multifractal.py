"""
signals/multifractal.py

Rolling multifractal / Hurst regime classifier.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _rs_hurst(returns: np.ndarray, min_chunk: int = 8) -> float:
    n = len(returns)
    if n < min_chunk * 2:
        return np.nan

    chunk_sizes = np.unique(np.logspace(np.log10(min_chunk), np.log10(n // 2), num=8).astype(int))
    chunk_sizes = chunk_sizes[chunk_sizes >= min_chunk]

    log_rs, log_n = [], []
    for size in chunk_sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_vals = []
        for i in range(n_chunks):
            chunk = returns[i * size:(i + 1) * size]
            mean_adj = chunk - chunk.mean()
            cum = np.cumsum(mean_adj)
            r = cum.max() - cum.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            log_rs.append(np.log(np.mean(rs_vals)))
            log_n.append(np.log(size))

    if len(log_n) < 3:
        return np.nan

    slope, _ = np.polyfit(log_n, log_rs, 1)
    return slope

def _generalized_hurst_q(returns: np.ndarray, q: float, min_lag: int = 2, max_lag: int = 20) -> float:
    n = len(returns)
    lags = np.arange(min_lag, min(max_lag, n // 4))
    if len(lags) < 3:
        return np.nan

    log_lag, log_sq = [], []
    cum_ret = np.cumsum(returns)

    for lag in lags:
        diffs = cum_ret[lag:] - cum_ret[:-lag]
        sq = np.mean(np.abs(diffs) ** q)
        if sq > 0:
            log_lag.append(np.log(lag))
            log_sq.append(np.log(sq) / q)

    if len(log_lag) < 3:
        return np.nan

    slope, _ = np.polyfit(log_lag, log_sq, 1)
    return slope

class MultifractalRegimeSignal:
    def __init__(self, window: int = 250, step: int = 5, zscore_lookback: int = 100):
        self.window = window
        self.step = step
        self.zscore_lookback = zscore_lookback

    def compute(self, returns: pd.Series) -> pd.DataFrame:
        idx = returns.index
        vals = returns.values
        n = len(vals)

        h_local = np.full(n, np.nan)
        h2 = np.full(n, np.nan)
        h4 = np.full(n, np.nan)
        
        for t in range(self.window, n, self.step):
            x = vals[t - self.window : t]
            h_local[t] = _rs_hurst(x)
            h2[t] = _generalized_hurst_q(x, q=2.0)
            h4[t] = _generalized_hurst_q(x, q=4.0)
        
        

        df = pd.DataFrame({"H_local": h_local, "H2": h2, "H4": h4}, index=idx)
        df = pd.DataFrame({"H_local": h_local, "H2": h2, "H4": h4}, index=idx)
        df = df.ffill()

        # core dissertation metric: deviation from monofractal scaling
        df["zeta_dev"] = df["H4"] - df["H2"]

        # parametric thresholding: z-score zeta_dev against its own trailing history
        roll_mean = df["zeta_dev"].rolling(self.zscore_lookback, min_periods=30).mean()
        roll_std = df["zeta_dev"].rolling(self.zscore_lookback, min_periods=30).std()
        df["zeta_z"] = (df["zeta_dev"] - roll_mean) / roll_std.replace(0, np.nan)

        
        df = df.ffill()
        return df
    @staticmethod
    def classify_regime(zeta_z: pd.Series, stress_threshold: float = -1.5,
                         transition_threshold: float = -0.75) -> pd.Series:
        regime = pd.Series(0, index=zeta_z.index)
        regime[zeta_z < transition_threshold] = 1
        regime[zeta_z < stress_threshold] = 2
        return regime


import numpy as np
rng = np.random.default_rng(0)
white_noise = rng.standard_normal(2000)          # should give H ≈ 0.5
print(_rs_hurst(white_noise))

trending = np.cumsum(rng.standard_normal(2000)) # a random walk's *increments* used differently...

print(_generalized_hurst_q(white_noise, q=2.0))
print(_generalized_hurst_q(white_noise, q=4.0))


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from loader import SyntheticRegimeLoader

    loader = SyntheticRegimeLoader(n_equities=20, n_futures=3)
    prices, true_regime = loader.load_prices(n_days=1500)

    index_proxy = prices[loader.equity_tickers].pct_change().mean(axis=1).dropna()

    sig = MultifractalRegimeSignal(window=250, step=5)   # whatever window you're currently testing
    out = sig.compute(index_proxy)
    out["classified_regime"] = sig.classify_regime(out["zeta_z"])
    out["true_regime"] = true_regime

    valid = out.dropna(subset=["zeta_z"])

    n_classified_stress = (valid["classified_regime"] == 2).sum()
    n_true_stress = (valid["true_regime"] == 2).sum()
    overlap = ((valid["classified_regime"] == 2) & (valid["true_regime"] == 2)).sum()
    print(f"Days classified as stress: {n_classified_stress}")
    print(f"True stress days in sample: {n_true_stress}")
    print(f"Overlap: {overlap}")

    # --- new diagnostic: event-time view around true stress onsets ---
    stress_starts = valid.index[(valid["true_regime"] == 2) & (valid["true_regime"].shift(1) != 2)]
    for d in stress_starts[:3]:
        loc = valid.index.get_loc(d)
        window = valid.iloc[max(0, loc - 10): loc + 15][["zeta_z", "true_regime"]]
        print(f"\n--- stress onset {d.date()} ---")
        print(window)