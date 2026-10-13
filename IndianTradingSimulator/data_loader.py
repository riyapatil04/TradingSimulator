"""Download and validate historical Indian stock market data.

Yahoo Finance data is historical daily data, not real-time market data. This
module uses adjusted prices (`auto_adjust=True`) so the returned ``Close``
column reflects stock splits and dividends consistently with the other OHLC
prices returned by yfinance.
"""

from collections.abc import Iterable

import pandas as pd
import yfinance as yf

SUPPORTED_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y"}
SAMPLE_TICKERS = {
    "TCS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "SBIN.NS",
    "ITC.NS",
    "HDFCBANK.NS",
}
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def get_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Return clean daily adjusted historical data for an NSE ticker.

    Args:
        ticker: Yahoo Finance symbol, such as ``TCS.NS``. Symbols must use
            the ``.NS`` suffix for NSE-listed stocks.
        period: One of ``1mo``, ``3mo``, ``6mo``, ``1y`` or ``2y``.

    Returns:
        A DataFrame with ``Date``, ``Open``, ``High``, ``Low``, ``Close`` and
        ``Volume`` columns sorted from oldest to newest.

    Raises:
        ValueError: If the ticker, period, or downloaded data is invalid.
        RuntimeError: If yfinance cannot complete the network request.
    """
    clean_ticker = _validate_ticker(ticker)
    if not isinstance(period, str) or period not in SUPPORTED_PERIODS:
        valid_periods = ", ".join(sorted(SUPPORTED_PERIODS))
        raise ValueError(
            f"Unsupported period '{period}'. Choose one of: {valid_periods}."
        )

    try:
        raw_data = yf.download(
            clean_ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not download data for {clean_ticker}. "
            "Check your internet connection and try again."
        ) from error

    if raw_data.empty:
        raise ValueError(
            f"No historical data was found for {clean_ticker}. "
            "Check that the ticker is valid and listed on NSE."
        )

    return _clean_stock_data(raw_data, clean_ticker)


def _validate_ticker(ticker: str) -> str:
    """Validate and normalize an NSE Yahoo Finance ticker."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string, such as 'TCS.NS'.")

    clean_ticker = ticker.strip().upper()
    if not clean_ticker.endswith(".NS"):
        raise ValueError(
            f"Invalid NSE ticker '{ticker}'. Use a Yahoo Finance symbol ending in '.NS'."
        )
    return clean_ticker


def _clean_stock_data(raw_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance columns and return only complete required fields."""
    price_data = pd.DataFrame(index=raw_data.index)
    for column_name in PRICE_COLUMNS:
        source_column = _find_column(raw_data.columns, column_name)
        if source_column is None:
            raise ValueError(
                f"Downloaded data for {ticker} is missing the '{column_name}' column."
            )
        price_data[column_name] = raw_data[source_column]

    price_data.index.name = "Date"
    price_data = price_data.reset_index()
    date_column = "Date" if "Date" in price_data.columns else "Datetime"
    if date_column not in price_data.columns:
        raise ValueError(f"Downloaded data for {ticker} has no date column.")
    price_data = price_data.rename(columns={date_column: "Date"})
    price_data["Date"] = pd.to_datetime(price_data["Date"], errors="coerce")

    for column_name in PRICE_COLUMNS:
        price_data[column_name] = pd.to_numeric(
            price_data[column_name], errors="coerce"
        )

    cleaned_data = (
        price_data.dropna(subset=REQUIRED_COLUMNS)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )
    if cleaned_data.empty:
        raise ValueError(
            f"Downloaded data for {ticker} contains no complete daily price rows."
        )

    return cleaned_data[REQUIRED_COLUMNS]


def _find_column(columns: Iterable[object], field_name: str) -> object | None:
    """Find a field in flat columns or either orientation of a MultiIndex."""
    for column in columns:
        if column == field_name:
            return column
        if isinstance(column, tuple) and field_name in column:
            return column

    return None
