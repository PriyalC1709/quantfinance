"""
garch_vol_timing.py
GARCH(1,1)-based volatility forecasting for position sizing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns: pd.Series):
    scaled_returns = returns.dropna() * 100
    model = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="t")
    result = model.fit(disp="off")
    return result

def conditional_volatility(fitted_result) -> pd.Series:
    cond_vol_pct = fitted_result.conditional_volatility
    cond_vol = cond_vol_pct / 100
    return cond_vol

def one_step_forecast(fitted_result) -> float:
    forecast = fitted_result.forecast(horizon=1, reindex=False)
    variance_pct2 = forecast.variance.values[-1, 0]
    vol = np.sqrt(variance_pct2) / 100
    return vol

def walk_forward_vol_forecast(returns: pd.Series, min_train: int = 250, refit_every: int = 5) -> pd.Series:
    n = len(returns)
    forecasts = pd.Series(np.nan, index=returns.index)

    for t in range(min_train, n, refit_every):
        train = returns.iloc[:t]
        result = fit_garch(train)
        fcst = one_step_forecast(result)
        forecasts.iloc[t] = fcst

        


    forecasts = forecasts.ffill()
    return forecasts

def vol_target_multiplier(forecast_vol: pd.Series, target_vol: float = 0.01, max_leverage: float = 2.0) -> pd.Series:
    raw_multiplier = target_vol / forecast_vol
    capped = raw_multiplier.clip(upper=max_leverage)
    return capped


def backtest_vol_timing(raw_returns: pd.Series, multiplier: pd.Series) -> pd.DataFrame:
    aligned = pd.DataFrame({
        "raw_return": raw_returns,
        "multiplier": multiplier.shift(1),
    }).dropna()

    aligned["strategy_return"] = aligned["multiplier"] * aligned["raw_return"]

    return aligned

def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict:
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = ann_return / downside_dev if downside_dev > 0 else np.nan

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }
import os
output_path = os.path.expanduser("~/multi_seed_results_clean.csv")




def walk_forward_target_vol(wf_vol: pd.Series, min_calib: int = 250) -> pd.Series:
    expanding_median = wf_vol.expanding(min_periods=min_calib).median()
    return expanding_median.shift(1)


def vol_target_multiplier_dynamic(forecast_vol: pd.Series, target_vol_series: pd.Series, max_leverage: float = 2.0) -> pd.Series:
    raw_multiplier = target_vol_series / forecast_vol
    capped = raw_multiplier.clip(upper=max_leverage)
    return capped

def run_multi_seed_backtest(n_seeds: int = 30, n_days: int = 1500, n_equities: int = 20,
                             n_futures: int = 3, refit_every: int = 5,
                             output_path: str = None):
    results = []
    output_path = output_path or os.path.expanduser("~/multi_seed_results_clean.csv")

    for seed in range(n_seeds):
        loader = SyntheticRegimeLoader(n_equities=n_equities, n_futures=n_futures, seed=seed)
        prices, true_regime = loader.load_prices(n_days=n_days)
        index_proxy = prices[loader.equity_tickers].pct_change().mean(axis=1).dropna()

        wf_vol = walk_forward_vol_forecast(index_proxy, min_train=250, refit_every=refit_every)
        target_series = walk_forward_target_vol(wf_vol, min_calib=250)
        multiplier = vol_target_multiplier_dynamic(wf_vol, target_series, max_leverage=2.0)

        bt = backtest_vol_timing(index_proxy, multiplier)

        strategy_perf = performance_summary(bt["strategy_return"])
        benchmark_perf = performance_summary(bt["raw_return"])

        row = {"seed": seed}
        row.update({f"strategy_{k}": v for k, v in strategy_perf.items()})
        row.update({f"benchmark_{k}": v for k, v in benchmark_perf.items()})
        results.append(row)

        print(f"seed {seed}: strategy sharpe={strategy_perf['sharpe']:.3f}, "
              f"benchmark sharpe={benchmark_perf['sharpe']:.3f}")

        pd.DataFrame(results).to_csv(output_path, index=False)

    return pd.DataFrame(results)


DEV_SEEDS = range(0, 30)        # everything you've used so far — seeds 0-29
TEST_SEEDS = range(1000, 1020)  # 20 fresh seeds, never touched, never inspected

FROZEN_PARAMS = dict(
    n_days=1500,
    n_equities=20,
    n_futures=3,
    refit_every=5,
)

def run_on_seed_set(seeds, output_path):
    results = []
    for seed in seeds:
        loader = SyntheticRegimeLoader(n_equities=FROZEN_PARAMS["n_equities"],
                                        n_futures=FROZEN_PARAMS["n_futures"], seed=seed)
        prices, true_regime = loader.load_prices(n_days=FROZEN_PARAMS["n_days"])
        index_proxy = prices[loader.equity_tickers].pct_change().mean(axis=1).dropna()

        wf_vol = walk_forward_vol_forecast(index_proxy, min_train=250, refit_every=FROZEN_PARAMS["refit_every"])
        target_series = walk_forward_target_vol(wf_vol, min_calib=250)
        multiplier = vol_target_multiplier_dynamic(wf_vol, target_series, max_leverage=2.0)

        bt = backtest_vol_timing(index_proxy, multiplier)

        strategy_perf = performance_summary(bt["strategy_return"])
        benchmark_perf = performance_summary(bt["raw_return"])

        row = {"seed": seed}
        row.update({f"strategy_{k}": v for k, v in strategy_perf.items()})
        row.update({f"benchmark_{k}": v for k, v in benchmark_perf.items()})
        results.append(row)

        print(f"seed {seed}: strategy sharpe={strategy_perf['sharpe']:.3f}, benchmark sharpe={benchmark_perf['sharpe']:.3f}")
        pd.DataFrame(results).to_csv(output_path, index=False)

    return pd.DataFrame(results)

if __name__ == "__main__":
    import sys, os
    sys.path.append("..")
    from loader import SyntheticRegimeLoader

    # dev set already exists from the earlier clean run — reuse it, don't rerun
    dev_results = pd.read_csv(os.path.expanduser("~/Desktop/multi_seed_results_clean.csv"))  # adjust path to wherever it actually lives

    # only run the fresh, never-before-seen holdout set
    test_results = run_on_seed_set(TEST_SEEDS, os.path.expanduser("~/test_results.csv"))