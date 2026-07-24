import pandas as pd
import numpy as np
import yfinance as yf

from loader import SyntheticRegimeLoader  # not used here, but keep for reference
from garch_voltiming import (
    walk_forward_vol_forecast, walk_forward_target_vol,
    vol_target_multiplier_dynamic, backtest_vol_timing, performance_summary
)
from risk import sample_covariance, ledoit_wolf_covariance, pca_factor_covariance, out_of_sample_frobenius_error, factor_exposures
from optimizer import (optimize_min_variance, optimize_min_downside_deviation,
                        portfolio_variance, portfolio_downside_deviation, portfolio_factor_exposure)

# --- 1. Load real data ---
prices = pd.read_csv("clean_prices.csv", index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

benchmark = yf.download("^GSPC", start=prices.index[0], end=prices.index[-1], auto_adjust=True)["Close"]
benchmark = benchmark.reindex(prices.index).ffill()
index_proxy = benchmark.pct_change().dropna()
index_proxy = index_proxy.squeeze()  # ensure it's a Series, not a 1-col DataFrame

print(f"Returns shape: {returns.shape}")
print(f"Benchmark returns shape: {index_proxy.shape}")

# --- 2. GARCH walk-forward vol-timing on the real benchmark ---
wf_vol = walk_forward_vol_forecast(index_proxy, min_train=250, refit_every=5)
target_series = walk_forward_target_vol(wf_vol, min_calib=250)
multiplier = vol_target_multiplier_dynamic(wf_vol, target_series, max_leverage=2.0)

bt = backtest_vol_timing(index_proxy, multiplier)
strategy_perf = performance_summary(bt["strategy_return"])
benchmark_perf = performance_summary(bt["raw_return"])

print("\n--- GARCH vol-timing vs S&P 500 buy-and-hold (full sample) ---")
print(pd.DataFrame({"vol_timed": strategy_perf, "buy_and_hold": benchmark_perf}))

# --- 3. Time-based dev/holdout split ---
split_date = "2019-01-01"
dev_returns = index_proxy[index_proxy.index < split_date]
test_returns = index_proxy[index_proxy.index >= split_date]
print(f"\nDev period: {dev_returns.index[0].date()} to {dev_returns.index[-1].date()} ({len(dev_returns)} days)")
print(f"Test period: {test_returns.index[0].date()} to {test_returns.index[-1].date()} ({len(test_returns)} days)")

# --- 4. Covariance estimation, OOS comparison ---
oos_errors = out_of_sample_frobenius_error(returns, split=0.7, n_factors=5)
print("\n--- Covariance estimator OOS comparison (real data) ---")
print(oos_errors)

# --- 5. Portfolio optimization on full real return history ---
cov = sample_covariance(returns)
returns_matrix = returns.values

mv_weights = optimize_min_variance(cov, max_weight=0.15, verbose=True)
dd_weights = optimize_min_downside_deviation(returns_matrix, max_weight=0.15, verbose=True)

print("\n--- Portfolio comparison (real data) ---")
print(f"Min-variance downside dev: {portfolio_downside_deviation(mv_weights, returns_matrix):.6f}")
print(f"Min-downside-dev downside dev: {portfolio_downside_deviation(dd_weights, returns_matrix):.6f}")
print(f"Min-variance variance: {portfolio_variance(mv_weights, cov):.8f}")
print(f"Min-downside-dev variance: {portfolio_variance(dd_weights, cov):.8f}")

top5_mv = pd.Series(mv_weights, index=returns.columns).sort_values(ascending=False).head(5)
top5_dd = pd.Series(dd_weights, index=returns.columns).sort_values(ascending=False).head(5)
print(f"\nTop 5 min-variance holdings:\n{top5_mv}")
print(f"\nTop 5 min-downside-dev holdings:\n{top5_dd}")

split_date = "2019-01-01"
bt_dev = bt[bt.index < split_date]
bt_test = bt[bt.index >= split_date]

print(f"\n--- Dev period ({bt_dev.index[0].date()} to {bt_dev.index[-1].date()}, {len(bt_dev)} days) ---")
dev_strategy_perf = performance_summary(bt_dev["strategy_return"])
dev_benchmark_perf = performance_summary(bt_dev["raw_return"])
print(pd.DataFrame({"vol_timed": dev_strategy_perf, "buy_and_hold": dev_benchmark_perf}))

print(f"\n--- Holdout test period ({bt_test.index[0].date()} to {bt_test.index[-1].date()}, {len(bt_test)} days) ---")
test_strategy_perf = performance_summary(bt_test["strategy_return"])
test_benchmark_perf = performance_summary(bt_test["raw_return"])
print(pd.DataFrame({"vol_timed": test_strategy_perf, "buy_and_hold": test_benchmark_perf}))


equity_tickers = [c for c in returns.columns if c not in ["ES=F", "ZN=F", "GC=F"]]
factors = pd.DataFrame({
    "market": returns[equity_tickers].mean(axis=1),
    "rates": returns["ZN=F"],
    "gold": returns["GC=F"],
})
exposures = factor_exposures(returns, factors)

mv_exposure = portfolio_factor_exposure(mv_weights, exposures)
dd_exposure = portfolio_factor_exposure(dd_weights, exposures)
print("\nPortfolio-level factor exposure:")
print(pd.DataFrame({"min_variance": mv_exposure, "min_downside_dev": dd_exposure}))


def historical_var_cvar(portfolio_returns: pd.Series, alpha: float = 0.05) -> dict:
    losses = -portfolio_returns.dropna()
    var = losses.quantile(1 - alpha)
    cvar = losses[losses >= var].mean()
    return {"VaR": var, "CVaR": cvar, "alpha": alpha}

mv_portfolio_returns = returns_matrix @ mv_weights
dd_portfolio_returns = returns_matrix @ dd_weights

mv_var_cvar = historical_var_cvar(pd.Series(mv_portfolio_returns), alpha=0.05)
dd_var_cvar = historical_var_cvar(pd.Series(dd_portfolio_returns), alpha=0.05)
benchmark_var_cvar = historical_var_cvar(index_proxy, alpha=0.05)

print("\n--- VaR / CVaR (95% confidence, daily) ---")
print(pd.DataFrame({
    "min_variance": mv_var_cvar,
    "min_downside_dev": dd_var_cvar,
    "benchmark": benchmark_var_cvar,
}))

def rolling_var(returns: pd.Series, window: int = 250, alpha: float = 0.05, step: int = 5) -> pd.Series:
    n = len(returns)
    var_series = pd.Series(np.nan, index=returns.index)

    for t in range(window, n, step):
        trailing = returns.iloc[t - window:t]
        result = historical_var_cvar(trailing, alpha=alpha)
        var_series.iloc[t] = result["VaR"]

    return var_series.ffill()


def hedge_trigger(rolling_var_series: pd.Series, lookback: int = 250, z_threshold: float = 1.0) -> pd.Series:
    roll_mean = rolling_var_series.rolling(lookback, min_periods=60).mean()
    roll_std = rolling_var_series.rolling(lookback, min_periods=60).std()
    z = (rolling_var_series - roll_mean) / roll_std.replace(0, np.nan)

    triggered = (z > z_threshold).astype(int)
    return triggered


def apply_hedge_overlay(weights: np.ndarray, tickers: list[str], futures_tickers: list[str],
                         triggered: bool, shift_amount: float = 0.10) -> np.ndarray:
    w = weights.copy()
    if not triggered:
        return w

    equity_idx = [i for i, t in enumerate(tickers) if t not in futures_tickers]
    futures_idx = [i for i, t in enumerate(tickers) if t in futures_tickers]

    equity_weight_total = w[equity_idx].sum()
    shift = min(shift_amount, equity_weight_total)  # never shift more than what's actually held in equities

    w[equity_idx] *= (equity_weight_total - shift) / equity_weight_total
    w[futures_idx] += shift * (w[futures_idx] / w[futures_idx].sum())  # distribute shift proportionally across futures

    return w

def backtest_hedge_overlay(base_weights: np.ndarray, returns_matrix: np.ndarray, tickers: list[str],
                            futures_tickers: list[str], triggered_series: pd.Series, shift_amount: float = 0.10):
    triggered_shifted = triggered_series.shift(1)  # yesterday's trigger, applied to today's return

    n_days = returns_matrix.shape[0]
    unhedged_returns = np.zeros(n_days)
    hedged_returns = np.zeros(n_days)

    for t in range(n_days):
        day_return = returns_matrix[t]
        unhedged_returns[t] = base_weights @ day_return

        trig_val = triggered_shifted.iloc[t]
        trig = bool(trig_val) if not pd.isna(trig_val) else False
        w_t = apply_hedge_overlay(base_weights, tickers, futures_tickers, triggered=trig, shift_amount=shift_amount)
        hedged_returns[t] = w_t @ day_return

    return pd.Series(unhedged_returns, index=returns.index[:n_days]), pd.Series(hedged_returns, index=returns.index[:n_days])

if __name__ == "__main__":
    prices = pd.read_csv("clean_prices.csv", index_col=0, parse_dates=True)
    returns_full = prices.pct_change().dropna()
    # ... (however you're loading index_proxy in this file) ...

    rolling_var_series = rolling_var(index_proxy, window=250, alpha=0.05, step=5)
    print(rolling_var_series.dropna().describe())

    tickers = list(returns.columns)
    futures_tickers = ["ES=F", "ZN=F", "GC=F"]

    triggered_series = hedge_trigger(rolling_var_series, lookback=250, z_threshold=1.0)
    print(f"Days triggered: {triggered_series.sum()} / {triggered_series.notna().sum()}")

    stress_day_idx = rolling_var_series.idxmax()
    print(f"Highest-VaR day: {stress_day_idx}, VaR={rolling_var_series[stress_day_idx]:.4f}, triggered={triggered_series[stress_day_idx]}")

    hedged_weights = apply_hedge_overlay(mv_weights, tickers, futures_tickers,
                                          triggered=bool(triggered_series[stress_day_idx]), shift_amount=0.10)
    print(f"\nOriginal equity total: {mv_weights[[i for i,t in enumerate(tickers) if t not in futures_tickers]].sum():.4f}")
    print(f"Hedged equity total:   {hedged_weights[[i for i,t in enumerate(tickers) if t not in futures_tickers]].sum():.4f}")
    print(f"Original futures total: {mv_weights[[i for i,t in enumerate(tickers) if t in futures_tickers]].sum():.4f}")
    print(f"Hedged futures total:   {hedged_weights[[i for i,t in enumerate(tickers) if t in futures_tickers]].sum():.4f}")

    unhedged_ret, hedged_ret = backtest_hedge_overlay(mv_weights, returns_matrix, tickers, futures_tickers,
                                                        triggered_series, shift_amount=0.10)

    unhedged_perf = performance_summary(unhedged_ret)
    hedged_perf = performance_summary(hedged_ret)
    print(pd.DataFrame({"unhedged": unhedged_perf, "hedged": hedged_perf}))