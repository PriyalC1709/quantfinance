"""
data/yfinance_loader.py

Real market data loader — matches the same output interface as
SyntheticRegimeLoader.load_prices() so nothing downstream needs to change.
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf


class YFinanceLoader:
    def __init__(self, equity_tickers: list[str], futures_tickers: list[str]):
        self.equity_tickers = equity_tickers
        self.futures_tickers = futures_tickers
        self.tickers = equity_tickers + futures_tickers

    def load_prices(self, start: str, end: str) -> pd.DataFrame:
        data = yf.download(self.tickers, start=start, end=end, auto_adjust=True)["Close"]
        data = data.dropna(how="all")
        return data[self.tickers]  # preserve consistent column order
EQUITY_TICKERS = [
    "AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM", "WMT", "KO", "DIS", "HD",
    "UNH", "CSCO", "PFE", "INTC", "VZ", "T", "CVX", "IBM",
]  # dropped V, MA — IPO'd too late (2008, 2006)
FUTURES_TICKERS = ["ES=F", "ZN=F", "GC=F"]

if __name__ == "__main__":
    loader = YFinanceLoader(EQUITY_TICKERS, FUTURES_TICKERS)
    prices = loader.load_prices(start="1995-01-01", end="2025-12-31")

    common_start = prices.apply(lambda col: col.first_valid_index()).max()
    print(f"Common start date across all tickers: {common_start}")

    prices_trimmed = prices.loc[common_start:]
    print(f"Trimmed shape: {prices_trimmed.shape}")
    print(f"Any remaining NaNs: {prices_trimmed.isna().sum().sum()}")

    nan_by_column = prices_trimmed.isna().sum()
    print("NaNs per ticker:")
    print(nan_by_column[nan_by_column > 0])

    nan_rows = prices_trimmed[prices_trimmed.isna().any(axis=1)]
    print(f"\nNumber of rows with at least one NaN: {len(nan_rows)}")
    print(f"Date range of first 5 NaN rows: {nan_rows.index[:5].tolist()}")
    print(f"Date range of last 5 NaN rows: {nan_rows.index[-5:].tolist()}")

    prices_clean = prices_trimmed.ffill()
    remaining_nans = prices_clean.isna().sum().sum()
    print(f"Remaining NaNs after forward-fill: {remaining_nans}")

    prices_clean.to_csv("clean_prices.csv")
    print(f"Final shape: {prices_clean.shape}")