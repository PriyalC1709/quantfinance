"""
loader.py
Synthetic regime-switching multi-asset price generator.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class SyntheticRegimeLoader:
    def __init__(self, n_equities: int = 20, n_futures: int = 3, seed: int = 42):
        self.n_equities = n_equities
        self.n_futures = n_futures
        self.n_assets = n_equities + n_futures
        self.rng = np.random.default_rng(seed)

        self.equity_tickers = [f"EQ{i:02d}" for i in range(n_equities)]
        self.futures_tickers = ["FUT_EQIDX", "FUT_RATES", "FUT_VOL"][:n_futures]
        self.tickers = self.equity_tickers + self.futures_tickers

        # regime transition matrix: rows sum to 1
        # P[i, j] = probability of moving from regime i today to regime j tomorrow
        self.P = np.array([
            [0.985, 0.014, 0.001],   # from calm
            [0.05,  0.90,  0.05],    # from transition
            [0.02,  0.28,  0.70],    # from stress
        ])

        # per-regime annualized drift/vol per asset class, fat-tail dof, base correlation
        self.regime_params = {
            0: dict(eq_mu=0.08,  eq_sigma=0.14, fut_mu=0.02, fut_sigma=0.08, dof=8, base_corr=0.25),
            1: dict(eq_mu=0.00,  eq_sigma=0.24, fut_mu=0.02, fut_sigma=0.14, dof=5, base_corr=0.45),
            2: dict(eq_mu=-0.35, eq_sigma=0.45, fut_mu=0.05, fut_sigma=0.30, dof=3, base_corr=0.75),
        }
        print(self.P.sum(axis=1))
        

    def load_prices(self, n_days: int = 1500, start: str = "2019-01-02") -> tuple[pd.DataFrame, pd.Series]:
        dates = pd.bdate_range(start=start, periods=n_days)

        regimes = np.zeros(n_days, dtype=int)
        for t in range(1, n_days):
            regimes[t] = self.rng.choice(3, p=self.P[regimes[t - 1]])

        rets = np.zeros((n_days, self.n_assets))
        for t in range(n_days):
            p = self.regime_params[regimes[t]]

            mu = np.array([p["eq_mu"]] * self.n_equities + [p["fut_mu"]] * self.n_futures) / 252
            sigma = np.array([p["eq_sigma"]] * self.n_equities + [p["fut_sigma"]] * self.n_futures) / np.sqrt(252)

            corr = self._corr_matrix(regimes[t])
            cov = np.outer(sigma, sigma) * corr
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(self.n_assets))

            z = self.rng.standard_t(p["dof"], size=self.n_assets)
            z = z / np.sqrt(p["dof"] / (p["dof"] - 2))

            rets[t] = mu + L @ z

        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        prices_df = pd.DataFrame(prices, index=dates, columns=self.tickers)
        regime_series = pd.Series(regimes, index=dates, name="true_regime")

        return prices_df, regime_series

    def _corr_matrix(self, regime: int) -> np.ndarray:
        base_corr = self.regime_params[regime]["base_corr"]
        C = np.full((self.n_assets, self.n_assets), base_corr)
        np.fill_diagonal(C, 1.0)

        # futures decorrelate from equities, more so under stress
        fut_eq_corr = -0.15 if regime == 2 else (-0.05 if regime == 1 else 0.05)
        C[self.n_equities:, :self.n_equities] = fut_eq_corr
        C[:self.n_equities, self.n_equities:] = fut_eq_corr
        np.fill_diagonal(C, 1.0)  # the block-fill above overwrote part of the diagonal, so reset it

        return C

        
    
    
    

    

if __name__ == "__main__":
    loader = SyntheticRegimeLoader()
    prices, regimes = loader.load_prices(n_days=1500)
    print(prices.tail())
    print(regimes.value_counts().sort_index())