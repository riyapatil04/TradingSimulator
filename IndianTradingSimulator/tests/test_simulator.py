"""Offline unit tests for the trading simulator's core behavior."""

import unittest
from unittest.mock import patch

import pandas as pd

from backtester import backtest_strategy
from data_loader import _clean_stock_data, get_stock_data
from strategies import (
    donchian_breakout_strategy,
    moving_average_crossover_strategy,
    sma_trend_strategy,
)


def market_data(close, opening=None, high=None, low=None, signals=None):
    """Build a small deterministic OHLCV DataFrame for tests."""
    opening = opening or close
    high = high or [price + 1 if price is not None else None for price in close]
    low = low or [price - 1 if price is not None else None for price in close]
    signals = signals or ["HOLD"] * len(close)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=len(close), freq="D"),
            "Open": opening,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": [1000] * len(close),
            "Signal": signals,
        }
    )


class DataLoaderTests(unittest.TestCase):
    def test_clean_stock_data_handles_multiindex_and_missing_rows(self):
        dates = pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"])
        raw = pd.DataFrame(
            {
                ("Open", "TCS.NS"): [103, 101, 102],
                ("High", "TCS.NS"): [104, 102, 103],
                ("Low", "TCS.NS"): [102, 100, 101],
                ("Close", "TCS.NS"): [103, 101, None],
                ("Volume", "TCS.NS"): [1000, 900, 800],
            },
            index=dates,
        )
        cleaned = _clean_stock_data(raw, "TCS.NS")

        self.assertEqual(
            list(cleaned.columns), ["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        self.assertTrue(cleaned["Date"].is_monotonic_increasing)
        self.assertEqual(len(cleaned), 2)

    def test_invalid_ticker_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "ending in '.NS'"):
            get_stock_data("TCS", "1mo")

    @patch("data_loader.yf.download", return_value=pd.DataFrame())
    def test_empty_download_is_reported(self, _download):
        with self.assertRaisesRegex(ValueError, "No historical data"):
            get_stock_data("TCS.NS", "1mo")

    @patch("data_loader.yf.download", side_effect=OSError("network unavailable"))
    def test_network_error_is_reported(self, _download):
        with self.assertRaisesRegex(RuntimeError, "Could not download data"):
            get_stock_data("TCS.NS", "1mo")


class StrategyTests(unittest.TestCase):
    def test_sma_crossings_and_warmup(self):
        data = market_data([10, 10, 10, 12, 10])
        result = sma_trend_strategy(data, window=3)

        self.assertTrue(pd.isna(result.iloc[0]["SMA_3"]))
        self.assertEqual(result.iloc[0]["Signal"], "HOLD")
        self.assertEqual(result.iloc[3]["Signal"], "BUY")
        self.assertEqual(result.iloc[4]["Signal"], "SELL")

    def test_moving_average_crossover_calculates_both_averages(self):
        prices = [10, 10, 10, 10, 10, 12, 14, 16, 18, 20]
        result = moving_average_crossover_strategy(
            market_data(prices), short_window=3, long_window=5
        )

        self.assertAlmostEqual(result.iloc[4]["SMA_3"], 10)
        self.assertAlmostEqual(result.iloc[4]["SMA_5"], 10)
        self.assertEqual(result.iloc[5]["Signal"], "BUY")
        self.assertEqual(result.iloc[0]["Signal"], "HOLD")

    def test_donchian_band_uses_only_prior_sessions(self):
        result = donchian_breakout_strategy(
            market_data(
                close=[10, 10, 10, 12, 12],
                high=[10, 10, 10, 12, 12],
                low=[9, 9, 9, 11, 11],
            ),
            window=3,
        )

        self.assertEqual(result.iloc[3]["Donchian_Upper_3"], 10)
        self.assertEqual(result.iloc[3]["Signal"], "BUY")
        self.assertNotEqual(result.iloc[3]["Donchian_Upper_3"], result.iloc[3]["High"])


class BacktesterTests(unittest.TestCase):
    def test_next_open_execution_costs_and_final_valuation(self):
        data = market_data(
            close=[10, 11, 12, 13, 14],
            opening=[10, 10, 12, 13, 14],
            signals=["HOLD", "BUY", "HOLD", "SELL", "BUY"],
        )
        equity, trades, summary = backtest_strategy(
            data, initial_capital=100, transaction_cost_rate=0.01
        )

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades.iloc[0]["Trade_Date"], pd.Timestamp("2026-01-03"))
        self.assertEqual(trades.iloc[0]["Execution_Price"], 12)
        self.assertEqual(trades.iloc[0]["Quantity"], 8)
        self.assertAlmostEqual(trades.iloc[0]["Transaction_Cost"], 0.96)
        self.assertEqual(trades.iloc[1]["Transaction_Type"], "SELL")
        self.assertEqual(summary["final_shares_held"], 0)
        self.assertAlmostEqual(summary["final_portfolio_value"], 100.0 + 8 * 14 - 8 * 12 - 0.96 - 1.12)
        self.assertEqual(len(equity), 5)

    def test_empty_missing_and_no_trade_inputs(self):
        _, trades, summary = backtest_strategy(pd.DataFrame(), 100)
        self.assertTrue(trades.empty)
        self.assertEqual(summary["return_percentage"], 0.0)

        with self.assertRaisesRegex(ValueError, "missing required column"):
            backtest_strategy(pd.DataFrame({"Close": [10]}), 100)

        no_trade_data = market_data([10, 11, 12])
        _, no_trades, no_trade_summary = backtest_strategy(no_trade_data, 100)
        self.assertTrue(no_trades.empty)
        self.assertEqual(no_trade_summary["final_portfolio_value"], 100.0)

    def test_missing_prices_are_skipped_without_inventing_values(self):
        data = market_data(
            close=[10, None, 12, 13],
            opening=[10, None, 12, 13],
            signals=["HOLD", "BUY", "HOLD", "HOLD"],
        )
        equity, trades, _ = backtest_strategy(data, 100)

        self.assertEqual(len(equity), 3)
        self.assertTrue(trades.empty)


if __name__ == "__main__":
    unittest.main()