"""Streamlit dashboard for the Indian Stock Market Trading Simulator."""

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from backtester import backtest_strategy
from data_loader import SAMPLE_TICKERS, get_stock_data
from strategies import (
    donchian_breakout_strategy,
    moving_average_crossover_strategy,
    sma_trend_strategy,
)

STRATEGY_NAMES = {
    "SMA Trend": sma_trend_strategy,
    "Moving Average Crossover": moving_average_crossover_strategy,
    "Donchian Breakout": donchian_breakout_strategy,
}


st.set_page_config(
    page_title="Indian Stock Market Trading Simulator",
    page_icon="📈",
    layout="wide",
)


def format_inr(value: float) -> str:
    """Format a numeric value as Indian rupees."""
    return f"₹{value:,.2f}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, period: str):
    """Cache historical downloads for one hour during the session."""
    return get_stock_data(ticker, period)


def build_price_chart(strategy_data, strategy_name: str) -> go.Figure:
    """Build the price chart with strategy indicators and signal markers."""
    chart = go.Figure()
    chart.add_trace(
        go.Candlestick(
            x=strategy_data["Date"],
            open=strategy_data["Open"],
            high=strategy_data["High"],
            low=strategy_data["Low"],
            close=strategy_data["Close"],
            name="Price",
        )
    )

    indicator_columns = {
        "SMA Trend": ["SMA_20"],
        "Moving Average Crossover": ["SMA_20", "SMA_50"],
        "Donchian Breakout": ["Donchian_Upper_20", "Donchian_Lower_20"],
    }
    indicator_colors = ["#2563eb", "#f97316"]
    for index, column in enumerate(indicator_columns[strategy_name]):
        if column in strategy_data.columns:
            chart.add_trace(
                go.Scatter(
                    x=strategy_data["Date"],
                    y=strategy_data[column],
                    mode="lines",
                    name=column.replace("_", " "),
                    line={"width": 1.5, "color": indicator_colors[index]},
                )
            )

    for signal, color, symbol in [
        ("BUY", "#16a34a", "triangle-up"),
        ("SELL", "#dc2626", "triangle-down"),
    ]:
        signal_rows = strategy_data[strategy_data["Signal"] == signal]
        if not signal_rows.empty:
            chart.add_trace(
                go.Scatter(
                    x=signal_rows["Date"],
                    y=signal_rows["Close"],
                    mode="markers",
                    name=signal,
                    marker={"color": color, "size": 11, "symbol": symbol},
                )
            )

    chart.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (INR)",
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return chart


def build_equity_chart(equity_curve) -> go.Figure:
    """Build the portfolio-value chart."""
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=equity_curve["Date"],
            y=equity_curve["Portfolio_Value"],
            mode="lines",
            name="Portfolio value",
            line={"color": "#0f766e", "width": 2.5},
        )
    )
    chart.update_layout(
        height=350,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        hovermode="x unified",
        yaxis_title="Value (INR)",
        xaxis_title="Date",
    )
    return chart


def format_trade_history(trade_history):
    """Prepare trade history for a readable dashboard table."""
    display_data = trade_history.copy()
    for column in ("Signal_Date", "Trade_Date"):
        if column in display_data.columns:
            display_data[column] = pd.to_datetime(display_data[column]).dt.strftime(
                "%Y-%m-%d"
            )
    for column in ("Execution_Price", "Transaction_Cost", "Cash_After"):
        if column in display_data.columns:
            display_data[column] = display_data[column].map(format_inr)
    return display_data


def main() -> None:
    """Render the dashboard and run the selected historical simulation."""
    st.title("Indian Stock Market Algorithmic Trading Simulator")
    st.caption("Historical paper-trading simulation for NSE-listed stocks")

    with st.sidebar:
        st.header("Simulation settings")
        ticker = st.selectbox("Indian stock", sorted(SAMPLE_TICKERS))
        period = st.selectbox("Historical period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        strategy_name = st.selectbox("Strategy", list(STRATEGY_NAMES))
        initial_capital = st.number_input(
            "Initial capital (INR)", min_value=1.0, value=100_000.0, step=10_000.0
        )
        transaction_cost_percent = st.number_input(
            "Transaction cost (%)", min_value=0.0, value=0.0, step=0.01, format="%.2f"
        )
        st.caption("Transaction cost is applied to both buys and sells.")

    st.info(
        "This is a historical paper-trading simulation, not a live trading system "
        "or an investment recommendation. Historical signals do not guarantee profits."
    )

    try:
        with st.spinner(f"Loading historical data for {ticker}..."):
            market_data = load_data(ticker, period)
        strategy_data = STRATEGY_NAMES[strategy_name](market_data)
        equity_curve, trade_history, summary = backtest_strategy(
            strategy_data,
            initial_capital=initial_capital,
            transaction_cost_rate=transaction_cost_percent / 100,
        )
    except Exception as error:
        st.error(f"Unable to run the simulation: {error}")
        st.stop()

    if strategy_data.empty or equity_curve.empty:
        st.warning("No usable historical data is available for this selection.")
        st.stop()

    latest_close = float(strategy_data.iloc[-1]["Close"])
    st.subheader(f"{ticker} overview")
    overview_columns = st.columns(3)
    overview_columns[0].metric("Latest adjusted close", format_inr(latest_close))
    overview_columns[1].metric("Historical observations", f"{len(strategy_data):,}")
    overview_columns[2].metric("Selected strategy", strategy_name)

    st.plotly_chart(
        build_price_chart(strategy_data, strategy_name),
        use_container_width=True,
    )

    st.subheader("Backtest results")
    metrics = st.columns(4)
    metrics[0].metric("Initial capital", format_inr(summary["initial_capital"]))
    metrics[1].metric("Final portfolio value", format_inr(summary["final_portfolio_value"]))
    metrics[2].metric("Net P&L", format_inr(summary["net_profit_loss"]))
    metrics[3].metric("Return", f"{summary['return_percentage']:.2f}%")

    st.plotly_chart(build_equity_chart(equity_curve), use_container_width=True)

    st.subheader("Trade history")
    if trade_history.empty:
        st.info(
            "No trades were executed. This can happen when the strategy produced "
            "no signals or when the capital was insufficient for one whole share."
        )
    else:
        st.dataframe(
            format_trade_history(trade_history),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"Executed trades: {summary['total_trades']} | "
        f"Transaction costs: {format_inr(summary['transaction_costs'])} | "
        "Signals are calculated from historical data and executed at the next session open."
    )


if __name__ == "__main__":
    main()
