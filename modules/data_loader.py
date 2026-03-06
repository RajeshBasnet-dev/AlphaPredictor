

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


class DataLoaderError(Exception):
    """Raised when stock data cannot be loaded."""


def fetch_stock_data(ticker: str, period_years: int = 5) -> pd.DataFrame:
    """Fetch historical stock price data for a ticker.

    Args:
        ticker: Stock symbol such as ``AAPL`` or ``TSLA``.
        period_years: Number of years of data to fetch (default: 5).

    Returns:
        DataFrame indexed by Date with OHLCV columns.

    Raises:
        DataLoaderError: If ticker is invalid or data download fails.
    """
    clean_ticker = (ticker or "").strip().upper()
    if not clean_ticker:
        raise DataLoaderError("Please enter a stock ticker symbol.")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * period_years)

    try:
        data = yf.download(
            clean_ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network exceptions vary
        raise DataLoaderError(
            "Could not download data. Please check your internet connection and ticker."
        ) from exc

    if data.empty:
        raise DataLoaderError(
            f"No data found for '{clean_ticker}'. Please verify the ticker symbol."
        )

    # Newer yfinance versions return a MultiIndex with (column, ticker).
    # Flatten to plain column names so downstream code works uniformly.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Ensure dataframe is clean for modeling.
    data = data.dropna().copy()
    data.index.name = "Date"
    return data
