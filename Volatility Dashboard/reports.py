"""
reports.py
D/W/M portfolio monitoring reports.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from garch_voltiming import performance_summary

RISK_LIMITS = {
    "max_single_weight": 0.15,
    "max_daily_var": 0.025,
}


def daily_report(date: pd.Timestamp, weights: np.ndarray, tickers: list[str],
                  rolling_var_series: pd.Series, triggered_series: pd.Series,
                  limits: dict = RISK_LIMITS) -> dict:
    weight_breach = weights.max() > limits["max_single_weight"]
    breached_names = [t for t, w in zip(tickers, weights) if w > limits["max_single_weight"]]

    today_var = rolling_var_series.get(date, np.nan)
    var_breach = today_var > limits["max_daily_var"] if not pd.isna(today_var) else False

    hedge_on = bool(triggered_series.get(date, 0))
    hedge_yesterday = bool(triggered_series.shift(1).get(date, 0))
    hedge_flipped = hedge_on != hedge_yesterday

    return {
        "date": date,
        "weight_breach": weight_breach,
        "breached_names": breached_names,
        "today_var": today_var,
        "var_breach": var_breach,
        "hedge_on": hedge_on,
        "hedge_flipped_today": hedge_flipped,
    }

def weekly_report(start_date: pd.Timestamp, end_date: pd.Timestamp,
                   weights: np.ndarray, tickers: list[str], futures_tickers: list[str],
                   exposures: pd.DataFrame, rolling_var_series: pd.Series,
                   triggered_series: pd.Series) -> dict:
    week_var = rolling_var_series.loc[start_date:end_date]
    week_triggered = triggered_series.loc[start_date:end_date]

    current_exposure = exposures.T @ weights

    return {
        "start_date": start_date,
        "end_date": end_date,
        "avg_var": week_var.mean(),
        "max_var": week_var.max(),
        "days_hedge_active": int(week_triggered.sum()),
        "hedge_active_pct": week_triggered.mean(),
        "market_beta": current_exposure["market"],
        "rates_beta": current_exposure["rates"],
        "gold_beta": current_exposure["gold"],
    }

def monthly_report(month_start: pd.Timestamp, month_end: pd.Timestamp,
                    unhedged_returns: pd.Series, hedged_returns: pd.Series,
                    benchmark_returns: pd.Series) -> dict:
    unhedged_m = unhedged_returns.loc[month_start:month_end]
    hedged_m = hedged_returns.loc[month_start:month_end]
    benchmark_m = benchmark_returns.loc[month_start:month_end]

    unhedged_total = (1 + unhedged_m).prod() - 1
    hedged_total = (1 + hedged_m).prod() - 1
    benchmark_total = (1 + benchmark_m).prod() - 1

    hedge_contribution = hedged_total - unhedged_total
    construction_contribution = unhedged_total - benchmark_total

    return {
        "month_start": month_start,
        "month_end": month_end,
        "benchmark_return": benchmark_total,
        "base_portfolio_return": unhedged_total,
        "hedged_portfolio_return": hedged_total,
        "construction_contribution": construction_contribution,
        "hedge_contribution": hedge_contribution,
        "sharpe_hedged": performance_summary(hedged_m)["sharpe"],
        "max_dd_hedged": performance_summary(hedged_m)["max_drawdown"],
    }

if __name__ == "__main__":
    import pandas as pd
    from risk import sample_covariance, factor_exposures
    from optimizer import optimize_min_variance, portfolio_factor_exposure
    from real_data_run import rolling_var, hedge_trigger, backtest_hedge_overlay

    prices = pd.read_csv("clean_prices.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    tickers = list(returns.columns)
    futures_tickers = ["ES=F", "ZN=F", "GC=F"]

    cov = sample_covariance(returns)
    mv_weights = optimize_min_variance(cov, max_weight=0.15)

    equity_tickers = [c for c in tickers if c not in futures_tickers]
    index_proxy = returns[equity_tickers].mean(axis=1)

    rolling_var_series = rolling_var(index_proxy, window=250, alpha=0.05, step=5)
    triggered_series = hedge_trigger(rolling_var_series, lookback=250, z_threshold=1.0)

    # needed for weekly_report
    factors = pd.DataFrame({
        "market": returns[equity_tickers].mean(axis=1),
        "rates": returns["ZN=F"],
        "gold": returns["GC=F"],
    })
    exposures = factor_exposures(returns, factors)

    stress_day = rolling_var_series.idxmax()
    report = daily_report(stress_day, mv_weights, tickers, rolling_var_series, triggered_series)
    for k, v in report.items():
        print(f"{k}: {v}")

    start = pd.Timestamp("2009-02-16")
    end = pd.Timestamp("2009-02-20")
    wr = weekly_report(start, end, mv_weights, tickers, futures_tickers, exposures, rolling_var_series, triggered_series)
    for k, v in wr.items():
        print(f"{k}: {v}")

    cov = sample_covariance(returns)
    returns_matrix = returns.values
    unhedged_ret, hedged_ret = backtest_hedge_overlay(mv_weights, returns_matrix, tickers, futures_tickers,
                                                        triggered_series, shift_amount=0.10)
    month_start = pd.Timestamp("2009-02-01")
    month_end = pd.Timestamp("2009-02-28")
    mr = monthly_report(month_start, month_end, unhedged_ret, hedged_ret, index_proxy)
    for k, v in mr.items():
        print(f"{k}: {v}")

    # consistency check
    reconstructed = mr["benchmark_return"] + mr["construction_contribution"] + mr["hedge_contribution"]
    print(f"\nReconstructed hedged return: {reconstructed:.6f} (should equal {mr['hedged_portfolio_return']:.6f})")