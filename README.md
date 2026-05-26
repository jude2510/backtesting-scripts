# backtesting-scripts

Two Python scripts for backtesting stock trading strategies using historical data from Yahoo Finance.

## Scripts

### `backtest.py` — Buy-and-hold baseline
Calculates Sharpe ratio and maximum drawdown for a simple buy-and-hold strategy over any ticker and timeframe.

### `backtest_realistic.py` — SMA crossover with execution simulation
Runs an SMA-crossover strategy with configurable execution variables — slippage, latency, and commission — and compares the results against an ideal frictionless run side-by-side.

---

## Setup

```bash
pip install -r requirements.txt
```

> **Note (macOS + Anaconda):** If you have a NumPy version conflict, use `/usr/bin/python3` instead of `python3`.

---

## Usage

### `backtest.py`

```bash
# 5-year buy-and-hold (default)
python backtest.py AAPL

# Explicit date range
python backtest.py TSLA --start 2020-01-01 --end 2024-01-01

# Custom risk-free rate
python backtest.py MSFT --period 2y --risk-free-rate 0.05
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `ticker` | *(required)* | Stock ticker symbol, e.g. `AAPL` |
| `--period` | `5y` | Relative lookback: `1mo` `3mo` `6mo` `1y` `2y` `5y` `10y` `ytd` `max` |
| `--start` | — | Start date `YYYY-MM-DD` (use with `--end`) |
| `--end` | today | End date `YYYY-MM-DD` |
| `--risk-free-rate` | `0.04` | Annual risk-free rate as a decimal |

**Sample output**

```
==================================================
  Backtest: AAPL  |  2021-05-26 → 2026-05-26
==================================================
  Total return        : +150.40%
  Annualised return   : +20.26%
  Annualised vol      : 27.44%
  Sharpe ratio        : 0.663  (Rf=4.0%)
  Maximum drawdown    : -33.36%
    Peak              : 2024-12-26
    Trough            : 2025-04-08
==================================================
```

---

### `backtest_realistic.py`

```bash
# Defaults: 50/200-day SMA, 5 bps slippage, 1-day latency, no commission
python backtest_realistic.py AAPL

# Explicit date range with heavy friction
python backtest_realistic.py TSLA --start 2020-01-01 --end 2024-01-01 \
    --slippage 20 --latency 2 --commission 2.50

# Faster SMA windows
python backtest_realistic.py SPY --short-sma 20 --long-sma 50
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `ticker` | *(required)* | Stock ticker symbol |
| `--period` | `5y` | Relative lookback period |
| `--start` / `--end` | — | Explicit date range |
| `--short-sma` | `50` | Short moving average window (days) |
| `--long-sma` | `200` | Long moving average window (days) |
| `--capital` | `100000` | Starting capital in USD |
| `--slippage` | `5.0` | Slippage per trade in basis points (1 bps = 0.01%) |
| `--latency` | `1` | Days between signal and order fill (1 = next-day open) |
| `--commission` | `0.0` | Flat commission per trade in USD |
| `--risk-free-rate` | `0.04` | Annual risk-free rate as a decimal |

**Execution variables explained**

- **Slippage** — When you place a market order, the actual fill price is slightly worse than the quoted price due to the bid-ask spread and market impact. Expressed in basis points (5 bps = 0.05%).
- **Latency** — The delay between when a signal is generated (at market close) and when the order is actually filled. A latency of 1 means the order fills at the next day's open, which is the most realistic assumption.
- **Commission** — A flat fee charged per trade by a broker.

**Sample output**

```
==============================================================
  IDEAL EXECUTION  (zero slippage · zero latency · zero commission)
  AAPL  |  2021-05-26 → 2026-05-26
==============================================================
  Total return        : +3.27%
  Annualised return   : +0.65%
  Sharpe ratio        : -0.088
  Maximum drawdown    : -31.60%

  --- Trade summary ---
  Completed trades    : 4
  Win rate            : 25.0%
  Time in market      : 55.1%
==============================================================

==============================================================
  REALISTIC EXECUTION
  Slippage: 5.0 bps  |  Latency: 1d  |  Commission: $0.00/trade
==============================================================
  ...

==============================================================
  Execution cost: Ideal → Realistic
==============================================================
  Total return          : +3.271%  →  +5.089%  (Δ +1.818%)
  Sharpe ratio          : -0.088  →  -0.069   (Δ +0.019)
==============================================================
```

---

## Metrics reference

| Metric | Description |
|---|---|
| **Sharpe ratio** | Annualised risk-adjusted return: `(mean_excess_return / std) × √252`. Higher is better; >1 is generally considered good. |
| **Maximum drawdown** | Worst peak-to-trough decline over the period, expressed as a percentage. |
| **CAGR** | Compound annual growth rate — the equivalent yearly return if compounded. |
| **Annualised vol** | Standard deviation of daily returns scaled by √252. Measures how much the strategy fluctuates. |
| **Win rate** | Percentage of completed trades that were profitable. |
| **Time in market** | Fraction of trading days the strategy held a position. |
