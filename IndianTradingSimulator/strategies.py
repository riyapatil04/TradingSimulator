"""Rule-based trading strategies for historical OHLCV data.

These strategies are educational examples. A signal is not a guarantee of
profit, and historical backtests do not predict future market performance.
"""

import pandas as pd

SIGNALS = ("BUY", "SELL", "HOLD")


def sma_trend_strategy(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Generate signals when Close crosses a configurable simple moving average.

    A BUY is generated when Close moves from at or below the SMA to above it.
    A SELL is generated when Close moves from at or above the SMA to below it.
    Rows without enough history for the SMA receive ``HOLD``.
    """
    _validate_window(window, "window")
    result = _prepare_data(data)
    close = pd.to_numeric(result["Close"], errors="coerce")
    result[f"SMA_{window}"] = close.rolling(window).mean()
    result["Signal"] = _crossing_signals(
        close, result[f"SMA_{window}"]
    )
    return result


def moving_average_crossover_strategy(
    data: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.DataFrame:
    """Generate signals when the 20-day SMA crosses the 50-day SMA.

    The default windows are 20 and 50 trading sessions. The short SMA crossing
    above the long SMA creates BUY; crossing below creates SELL.
    """
    _validate_window(short_window, "short_window")
    _validate_window(long_window, "long_window")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    result = _prepare_data(data)
    close = pd.to_numeric(result["Close"], errors="coerce")
    result[f"SMA_{short_window}"] = close.rolling(short_window).mean()
    result[f"SMA_{long_window}"] = close.rolling(long_window).mean()
    result["Signal"] = _crossing_signals(
        result[f"SMA_{short_window}"], result[f"SMA_{long_window}"]
    )
    return result


def donchian_breakout_strategy(
    data: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """Generate signals when Close breaks out of the previous high/low range.

    The upper and lower bands use the previous ``window`` trading sessions,
    excluding the current row. A BUY occurs on an upward breakout and a SELL
    occurs on a downward breakout. The function requires High and Low columns.
    """
    _validate_window(window, "window")
    _validate_columns(data, ("Close", "High", "Low"))

    result = data.copy()
    close = pd.to_numeric(result["Close"], errors="coerce")
    high = pd.to_numeric(result["High"], errors="coerce")
    low = pd.to_numeric(result["Low"], errors="coerce")
    result[f"Donchian_Upper_{window}"] = high.rolling(window).max().shift(1)
    result[f"Donchian_Lower_{window}"] = low.rolling(window).min().shift(1)

    previous_close = close.shift(1)
    previous_upper = result[f"Donchian_Upper_{window}"].shift(1)
    previous_lower = result[f"Donchian_Lower_{window}"].shift(1)
    valid_upper_breakout = (
        close.notna()
        & result[f"Donchian_Upper_{window}"].notna()
    )
    valid_lower_breakout = (
        close.notna()
        & result[f"Donchian_Lower_{window}"].notna()
    )
    was_above_upper = (
        previous_close.notna()
        & previous_upper.notna()
        & (previous_close > previous_upper)
    )
    was_below_lower = (
        previous_close.notna()
        & previous_lower.notna()
        & (previous_close < previous_lower)
    )

    result["Signal"] = "HOLD"
    result.loc[
        valid_upper_breakout
        & (close > result[f"Donchian_Upper_{window}"])
        & ~was_above_upper,
        "Signal",
    ] = "BUY"
    result.loc[
        valid_lower_breakout
        & (close < result[f"Donchian_Lower_{window}"])
        & ~was_below_lower,
        "Signal",
    ] = "SELL"
    return result


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Copy input data after checking that it contains a usable Close field."""
    _validate_columns(data, ("Close",))
    return data.copy()


def _validate_columns(data: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    """Raise a clear error when required market columns are missing."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Market data is missing required column(s): {missing}.")


def _validate_window(window: int, name: str) -> None:
    """Ensure a rolling window is a positive integer."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _crossing_signals(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Return event signals only when valid values cross a reference series."""
    previous_values = values.shift(1)
    previous_reference = reference.shift(1)
    valid_rows = (
        values.notna()
        & reference.notna()
        & previous_values.notna()
        & previous_reference.notna()
    )

    signals = pd.Series("HOLD", index=values.index, dtype="object")
    signals.loc[
        valid_rows
        & (values > reference)
        & (previous_values <= previous_reference)
    ] = "BUY"
    signals.loc[
        valid_rows
        & (values < reference)
        & (previous_values >= previous_reference)
    ] = "SELL"
    return signals
