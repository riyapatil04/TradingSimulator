"""Long-only backtesting for historical strategy signals.

The engine assumes that a signal calculated at the close of one trading
session is executed at the next available session's opening price. It uses
whole shares, does not short-sell, and values open holdings at the closing
price of each session. This is an educational backtest, not a prediction of
future profitability.
"""

import math

import pandas as pd

TRADE_COLUMNS = [
    "Signal_Date",
    "Trade_Date",
    "Execution_Price",
    "Quantity",
    "Transaction_Type",
    "Transaction_Cost",
    "Cash_After",
    "Shares_Held",
]
EQUITY_COLUMNS = [
    "Date",
    "Close",
    "Cash",
    "Shares_Held",
    "Holdings_Value",
    "Portfolio_Value",
    "Signal",
]


def backtest_strategy(
    data: pd.DataFrame,
    initial_capital: float = 100_000,
    transaction_cost_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Backtest BUY/SELL/HOLD signals using next-session open execution.

    Args:
        data: Historical data containing ``Open``, ``Close`` and ``Signal``.
            A ``Date`` column is preferred; otherwise the DataFrame index is
            used as the date.
        initial_capital: Starting virtual cash, in rupees. Defaults to
            ``100_000``.
        transaction_cost_rate: Proportional cost applied to every buy and
            sell, expressed as a decimal. For example, ``0.001`` means 0.1%.

    Returns:
        A tuple containing ``equity_curve``, ``trade_history`` and
        ``summary_metrics``. The equity curve has one row per usable trading
        session. The trade history contains only successfully executed trades.

    Assumptions:
        Signals on the final row cannot execute because no next session is
        available. BUY orders purchase the maximum affordable whole-share
        quantity. SELL orders liquidate all held shares. Rows with missing
        Open or Close values are ignored because no reliable price exists.
    """
    _validate_money_inputs(initial_capital, transaction_cost_rate)
    empty_result = _empty_result(initial_capital)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        return empty_result

    _validate_columns(data, ("Open", "Close", "Signal"))
    prepared = _prepare_data(data)
    if prepared.empty:
        return empty_result

    cash = float(initial_capital)
    shares_held = 0
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []

    for row_number, row in prepared.iterrows():
        if row_number > 0:
            previous_row = prepared.iloc[row_number - 1]
            signal = str(previous_row["Signal"]).strip().upper()
            execution_price = row["Open"]
            if pd.notna(execution_price):
                if signal == "BUY" and shares_held == 0:
                    quantity = math.floor(
                        cash / (execution_price * (1 + transaction_cost_rate))
                    )
                    if quantity > 0:
                        transaction_cost = (
                            quantity * execution_price * transaction_cost_rate
                        )
                        cash -= quantity * execution_price + transaction_cost
                        shares_held += quantity
                        trade_records.append(
                            _trade_record(
                                signal_date=previous_row["Date"],
                                trade_date=row["Date"],
                                execution_price=execution_price,
                                quantity=quantity,
                                transaction_type="BUY",
                                transaction_cost=transaction_cost,
                                cash_after=cash,
                                shares_held=shares_held,
                            )
                        )
                elif signal == "SELL" and shares_held > 0:
                    quantity = shares_held
                    transaction_cost = (
                        quantity * execution_price * transaction_cost_rate
                    )
                    cash += quantity * execution_price - transaction_cost
                    shares_held = 0
                    trade_records.append(
                        _trade_record(
                            signal_date=previous_row["Date"],
                            trade_date=row["Date"],
                            execution_price=execution_price,
                            quantity=quantity,
                            transaction_type="SELL",
                            transaction_cost=transaction_cost,
                            cash_after=cash,
                            shares_held=shares_held,
                        )
                    )

        close_price = row["Close"]
        holdings_value = shares_held * close_price
        equity_records.append(
            {
                "Date": row["Date"],
                "Close": close_price,
                "Cash": cash,
                "Shares_Held": shares_held,
                "Holdings_Value": holdings_value,
                "Portfolio_Value": cash + holdings_value,
                "Signal": str(row["Signal"]).strip().upper(),
            }
        )

    equity_curve = pd.DataFrame(equity_records, columns=EQUITY_COLUMNS)
    trade_history = pd.DataFrame(trade_records, columns=TRADE_COLUMNS)
    final_value = float(equity_curve.iloc[-1]["Portfolio_Value"])
    total_costs = float(trade_history["Transaction_Cost"].sum()) if not trade_history.empty else 0.0
    summary_metrics = {
        "initial_capital": float(initial_capital),
        "final_portfolio_value": final_value,
        "net_profit_loss": final_value - initial_capital,
        "return_percentage": (final_value - initial_capital) / initial_capital * 100,
        "total_trades": int(len(trade_history)),
        "buy_trades": int((trade_history["Transaction_Type"] == "BUY").sum()) if not trade_history.empty else 0,
        "sell_trades": int((trade_history["Transaction_Type"] == "SELL").sum()) if not trade_history.empty else 0,
        "transaction_costs": total_costs,
        "final_cash": cash,
        "final_shares_held": shares_held,
    }
    return equity_curve, trade_history, summary_metrics


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create a dated, numeric, ascending copy of usable market rows."""
    prepared = data.copy()
    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    else:
        prepared.insert(0, "Date", pd.to_datetime(prepared.index, errors="coerce"))

    prepared["Open"] = pd.to_numeric(prepared["Open"], errors="coerce")
    prepared["Close"] = pd.to_numeric(prepared["Close"], errors="coerce")
    prepared = (
        prepared.dropna(subset=["Date", "Open", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return prepared


def _trade_record(
    signal_date: object,
    trade_date: object,
    execution_price: float,
    quantity: int,
    transaction_type: str,
    transaction_cost: float,
    cash_after: float,
    shares_held: int,
) -> dict[str, object]:
    """Build one consistently shaped trade-history record."""
    return {
        "Signal_Date": signal_date,
        "Trade_Date": trade_date,
        "Execution_Price": execution_price,
        "Quantity": quantity,
        "Transaction_Type": transaction_type,
        "Transaction_Cost": transaction_cost,
        "Cash_After": cash_after,
        "Shares_Held": shares_held,
    }


def _empty_result(initial_capital: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Return stable empty outputs for empty or unusable input data."""
    return (
        pd.DataFrame(columns=EQUITY_COLUMNS),
        pd.DataFrame(columns=TRADE_COLUMNS),
        {
            "initial_capital": float(initial_capital),
            "final_portfolio_value": float(initial_capital),
            "net_profit_loss": 0.0,
            "return_percentage": 0.0,
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "transaction_costs": 0.0,
            "final_cash": float(initial_capital),
            "final_shares_held": 0,
        },
    )


def _validate_columns(data: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    """Raise a clear error when required backtest columns are missing."""
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Backtest data is missing required column(s): {missing}.")


def _validate_money_inputs(initial_capital: float, transaction_cost_rate: float) -> None:
    """Validate capital and transaction-cost settings before simulation."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than zero.")
    if transaction_cost_rate < 0 or transaction_cost_rate >= 1:
        raise ValueError("transaction_cost_rate must be between 0 and 1.")
