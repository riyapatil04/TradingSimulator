# Indian Stock Market Algorithmic Trading Simulator

A beginner-friendly Python prototype for exploring algorithmic trading ideas on historical Indian stock market data.

## Project structure

- `app.py` - Streamlit user interface.
- `data_loader.py` - Downloads and validates historical data from Yahoo Finance.
- `strategies.py` - Calculates indicators and generates trading signals.
- `backtester.py` - Simulates trades and calculates portfolio performance.
- `requirements.txt` - Lists the required Python packages.

## Setup on Windows

Open a terminal in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.13 is recommended.

## Run the app

```powershell
streamlit run app.py
```

Streamlit will open the simulator in your browser.

Run the command from inside the `IndianTradingSimulator` folder. From the
workspace root, use:

```powershell
cd IndianTradingSimulator
..\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run offline tests

The unit tests use small manually constructed datasets and do not contact
Yahoo Finance:

```powershell
python -m unittest discover -s tests -v
```

The tests cover data cleaning and error handling, moving-average signals,
prior-session Donchian bands, next-session-open execution, whole-share
quantities, transaction costs, valuation, missing values, and no-trade cases.

## Scope and assumptions

The dashboard uses historical daily data from Yahoo Finance. Prices are
adjusted by yfinance, signals are calculated at the session close, and trades
are executed at the next available session's opening price. This is a
paper-trading simulation, not a live trading system or investment advice.

Yahoo Finance ticker symbols for Indian stocks commonly end in `.NS`, for example `RELIANCE.NS` or `TCS.NS`.
