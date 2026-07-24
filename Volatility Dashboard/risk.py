"""
risk/covariance.py

Covariance estimation: sample, Ledoit-Wolf shrinkage, PCA factor model.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import statsmodels.api as sm

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values



def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_, lw.shrinkage_



def pca_factor_covariance(returns: pd.DataFrame, n_factors: int = 5) -> np.ndarray:
    X = returns.values
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=n_factors)
    factor_scores = pca.fit_transform(X_centered)
    loadings = pca.components_.T

    factor_cov = np.atleast_2d(np.cov(factor_scores.T))   # <-- fix: guard against the n_factors=1 collapse
    systematic = loadings @ factor_cov @ loadings.T

    residuals = X_centered - factor_scores @ pca.components_
    idio_var = residuals.var(axis=0)

    return systematic + np.diag(idio_var)



def out_of_sample_frobenius_error(returns: pd.DataFrame, split: float = 0.7, n_factors: int = 5) -> pd.Series:
    n = len(returns)
    cut = int(n * split)
    train, test = returns.iloc[:cut], returns.iloc[cut:]
    realized_test_cov = test.cov().values

    sample_cov = sample_covariance(train)
    lw_cov, shrinkage = ledoit_wolf_covariance(train)
    pca_cov = pca_factor_covariance(train, n_factors=n_factors)

    errors = {
        "sample": np.linalg.norm(sample_cov - realized_test_cov, ord="fro"),
        "ledoit_wolf": np.linalg.norm(lw_cov - realized_test_cov, ord="fro"),
        f"pca_{n_factors}factor": np.linalg.norm(pca_cov - realized_test_cov, ord="fro"),
    }

    for k in [1, 2, 3, 5, 10]:
        pca_cov_k = pca_factor_covariance(train, n_factors=k)
        err = np.linalg.norm(pca_cov_k - realized_test_cov, ord="fro")
        print(f"n_factors={k}: OOS error = {err:.6f}")
    return pd.Series(errors)






def factor_exposures(returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(factor_returns.values)
    betas = {}
    for col in returns.columns:
        y = returns[col].values
        model = sm.OLS(y, X, missing="drop").fit()
        betas[col] = model.params[1:]  # drop the intercept (alpha), keep factor betas
    return pd.DataFrame(betas, index=factor_returns.columns).T





if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from loader import SyntheticRegimeLoader

    loader = SyntheticRegimeLoader(n_equities=20, n_futures=3)
    prices, true_regime = loader.load_prices(n_days=1500)
    returns = prices.pct_change().dropna()

    cov = sample_covariance(returns)
    print(cov.shape)
    print(f"Diagonal (variances) range: {np.diag(cov).min():.6f} to {np.diag(cov).max():.6f}")

    lw_cov, shrinkage = ledoit_wolf_covariance(returns)
    print(f"Shrinkage intensity: {shrinkage:.4f}")
    print(f"Sample cov diagonal range:      {np.diag(cov).min():.6f} to {np.diag(cov).max():.6f}")
    print(f"Ledoit-Wolf cov diagonal range: {np.diag(lw_cov).min():.6f} to {np.diag(lw_cov).max():.6f}")

    pca_cov = pca_factor_covariance(returns, n_factors=5)
    print(f"PCA cov diagonal range: {np.diag(pca_cov).min():.6f} to {np.diag(pca_cov).max():.6f}")
    oos_errors = out_of_sample_frobenius_error(returns, split=0.7, n_factors=5)
    print(oos_errors)
    prices = pd.read_csv("clean_prices.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()

    equity_tickers = [c for c in returns.columns if c not in ["ES=F", "ZN=F", "GC=F"]]

    factors = pd.DataFrame({
        "market": returns[equity_tickers].mean(axis=1),
        "rates": returns["ZN=F"],
        "gold": returns["GC=F"],
    })

    exposures = factor_exposures(returns, factors)
    print(exposures.round(3))
    