"""
portfolio/optimizer.py

Portfolio weight optimization: minimum-variance baseline, extended to
downside-deviation minimization, under parametric constraints.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    return weights @ cov @ weights



def portfolio_variance_scaled(weights: np.ndarray, cov: np.ndarray, scale: float = 1e6) -> float:
    return scale * (weights @ cov @ weights)


def optimize_min_variance(cov: np.ndarray, max_weight: float = 0.15, verbose: bool = False) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.full(n, 1 / n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight) for _ in range(n)]

    result = minimize(
        portfolio_variance_scaled, x0, args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if verbose:
        print(f"Converged: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Final objective value: {result.fun:.10f}")

    return result.x

def portfolio_downside_deviation(weights: np.ndarray, returns_matrix: np.ndarray, target: float = 0.0) -> float:
    portfolio_returns = returns_matrix @ weights
    downside = np.minimum(0, portfolio_returns - target)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    return downside_dev
def portfolio_downside_deviation_scaled(weights: np.ndarray, returns_matrix: np.ndarray, target: float = 0.0, scale: float = 1e4) -> float:
    return scale * portfolio_downside_deviation(weights, returns_matrix, target)


def optimize_min_downside_deviation(returns_matrix: np.ndarray, max_weight: float = 0.15,
                                      target: float = 0.0, verbose: bool = False) -> np.ndarray:
    n = returns_matrix.shape[1]
    x0 = np.full(n, 1 / n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight) for _ in range(n)]

    result = minimize(
        portfolio_downside_deviation_scaled, x0, args=(returns_matrix, target),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if verbose:
        print(f"Converged: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Final objective value: {result.fun:.10f}")

    return result.x

def portfolio_factor_exposure(weights: np.ndarray, exposures: pd.DataFrame) -> pd.Series:
    return exposures.T @ weights

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from loader import SyntheticRegimeLoader
    from risk import sample_covariance

    loader = SyntheticRegimeLoader(n_equities=20, n_futures=3)
    prices, _ = loader.load_prices(n_days=1500)
    returns = prices.pct_change().dropna()
    cov = sample_covariance(returns)

    n = cov.shape[0]
    equal_weights = np.full(n, 1 / n)
    concentrated = np.zeros(n); concentrated[0] = 1.0

    print(f"Equal-weight variance: {portfolio_variance(equal_weights, cov):.8f}")
    print(f"Single-asset variance: {portfolio_variance(concentrated, cov):.8f}")

    weights = optimize_min_variance(cov, max_weight=0.15)
    print(f"Sum of weights: {weights.sum():.6f}")
    print(f"Min weight: {weights.min():.6f}, Max weight: {weights.max():.6f}")
    print(f"Optimized portfolio variance: {portfolio_variance(weights, cov):.8f}")
    print(f"vs. equal-weight variance:    {portfolio_variance(equal_weights, cov):.8f}")

    # sanity check: inject one high-variance asset, confirm optimizer avoids it
    test_cov = cov.copy()
    test_cov[0, 0] *= 20  # blow up asset 0's variance massively
    test_cov[0, 1:] = 0   # and make it uncorrelated with everything else
    test_cov[1:, 0] = 0

    test_weights = optimize_min_variance(test_cov, max_weight=0.15)
    print(f"Weight on the artificially bad asset (index 0): {test_weights[0]:.6f}")
    print(f"Weight on everything else, mean: {test_weights[1:].mean():.6f}")

    returns_matrix = returns.values

    equal_weights = np.full(returns_matrix.shape[1], 1 / returns_matrix.shape[1])
    dd = portfolio_downside_deviation(equal_weights, returns_matrix)
    print(f"Equal-weight downside deviation: {dd:.6f}")

    dd_weights = optimize_min_downside_deviation(returns_matrix, max_weight=0.15, verbose=True)

    print(f"\nMin-variance portfolio downside deviation: {portfolio_downside_deviation(weights, returns_matrix):.6f}")
    print(f"Min-downside-dev portfolio downside deviation: {portfolio_downside_deviation(dd_weights, returns_matrix):.6f}")

    print(f"\nMin-variance portfolio variance: {portfolio_variance(weights, cov):.8f}")
    print(f"Min-downside-dev portfolio variance: {portfolio_variance(dd_weights, cov):.8f}")

    print(f"\nMax weight difference between the two portfolios: {np.abs(weights - dd_weights).max():.6f}")

