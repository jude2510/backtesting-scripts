#!/usr/bin/env python3
# Tells the OS to run this file with Python 3 when executed directly from the terminal

"""
Realistic SMA-crossover backtester that models execution variables:
  - Slippage    : fill-price degradation on every buy and sell
  - Latency     : delay in days between when the signal fires and when the order fills
  - Commission  : flat fee charged per trade

Also runs a frictionless "ideal" simulation so you can see the execution cost.

Usage:
    python backtest_realistic.py AAPL --period 5y
    python backtest_realistic.py TSLA --start 2020-01-01 --end 2024-01-01
    python backtest_realistic.py MSFT --slippage 10 --latency 2 --commission 1.00
    python backtest_realistic.py SPY  --short-sma 20 --long-sma 50
"""

import argparse          # standard library module for building the command-line interface
import sys               # standard library module used to exit with an error message
from typing import Optional, Tuple  # type annotations that allow None values and tuples
import numpy as np       # numerical library used for square-root and array operations
import pandas as pd      # data-analysis library used to manage the price time series

try:
    import yfinance as yf                                   # third-party library that fetches historical OHLCV data from Yahoo Finance
except ImportError:
    sys.exit("yfinance is required: pip install yfinance")  # exit cleanly if the library is missing


TRADING_DAYS = 252  # standard number of trading days per year; used to annualise daily statistics


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
) -> pd.DataFrame:
    # Downloads Open, High, Low, Close, Volume data for the given ticker
    t = yf.Ticker(ticker)                           # create a yfinance Ticker object for the requested stock
    if period:
        raw = t.history(period=period)              # fetch data using a relative window like "5y" or "1mo"
    else:
        raw = t.history(start=start, end=end)       # fetch data between two explicit calendar dates

    if raw.empty:
        sys.exit(f"No data for '{ticker}'. Check the ticker symbol and date range.")  # stop if Yahoo returned nothing

    return raw[["Open", "High", "Low", "Close", "Volume"]]  # keep only the price/volume columns we need


# ---------------------------------------------------------------------------
# Strategy + execution simulation
# ---------------------------------------------------------------------------

def simulate(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    slippage_bps: float,
    latency_days: int,
    commission: float,
    initial_capital: float,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Run an SMA-crossover strategy with configurable execution assumptions.

    Strategy logic:
      - Go long  when the short SMA crosses above the long SMA
      - Go flat  when the short SMA crosses below the long SMA

    Execution variables:
      - slippage_bps  : basis points added to buys and subtracted from sells
      - latency_days  : days between signal and fill; 1 = next-day open (most realistic)
      - commission    : flat dollar fee deducted on every trade

    Returns:
      portfolio  : daily dollar value of the account
      trades     : one row per completed round-trip trade
      enriched   : the original DataFrame with SMA and signal columns added
    """

    slippage = slippage_bps / 10_000  # convert basis points (e.g. 5 bps) to a decimal multiplier (0.0005)

    df = df.copy()  # work on a copy so we do not mutate the caller's DataFrame

    # --- Compute the two moving averages ---
    df["short_sma"] = df["Close"].rolling(short_window).mean()  # rolling average of the last short_window closing prices
    df["long_sma"]  = df["Close"].rolling(long_window).mean()   # rolling average of the last long_window closing prices

    # --- Generate the raw signal ---
    # 1 means "we want to be long today", 0 means "we want to be flat today"
    # Before either SMA has enough data (NaN), we stay flat (0)
    df["raw_signal"] = np.where(
        df["short_sma"].notna() & df["long_sma"].notna(),       # only after both SMAs have valid values
        (df["short_sma"] > df["long_sma"]).astype(int),         # 1 if short SMA is above long SMA, else 0
        0,                                                        # default to 0 (flat) while SMAs are warming up
    )

    # --- Apply execution latency ---
    # Shifting the signal forward by latency_days means we only act on information that was
    # available that many days ago; with latency=1 we see yesterday's signal and fill at today's open
    df["signal"] = df["raw_signal"].shift(latency_days).fillna(0)  # delayed signal; NaN at the start becomes 0

    # --- Detect crossovers ---
    # diff() gives the change in the signal: +1 when we transition from flat→long (buy),
    #                                        -1 when we transition from long→flat (sell)
    df["trade_signal"] = df["signal"].diff().fillna(0)  # fillna handles the very first row where diff() is undefined

    # --- Portfolio simulation ---
    cash   = float(initial_capital)  # start with all money in cash
    shares = 0.0                     # no shares held at the start
    entry_price = None               # price at which the current long position was entered
    entry_date  = None               # date the current position was opened
    portfolio_values = []            # daily total account value (cash + market value of shares)
    trades_log       = []            # record of every completed round-trip trade

    for date, row in df.iterrows():  # iterate over every trading day in chronological order
        signal = row["trade_signal"]  # +1, -1, or 0 for today

        # BUY: signal flipped from 0 → 1 and we are currently flat
        if signal > 0.5 and shares == 0:
            # We pay a slightly higher price than the open due to slippage (market impact)
            fill = row["Open"] * (1 + slippage)          # effective buy price after slippage
            affordable = max(0.0, cash - commission)      # deduct commission upfront from available cash
            shares = affordable / fill                    # number of shares we can buy with remaining cash
            cash   = 0.0                                 # all cash is now deployed into shares
            entry_price = fill                           # remember the entry price for later P&L calculation
            entry_date  = date                           # remember the entry date for the trade log

        # SELL: signal flipped from 1 → 0 and we currently hold shares
        elif signal < -0.5 and shares > 0:
            # We receive a slightly lower price than the open due to slippage
            fill     = row["Open"] * (1 - slippage)      # effective sell price after slippage
            proceeds = shares * fill - commission         # total cash received minus commission
            trade_return_pct = (fill / entry_price - 1) * 100  # percentage gain or loss on this trade

            trades_log.append({
                "entry_date":   entry_date,
                "exit_date":    date,
                "entry_price":  round(entry_price, 4),        # price we paid when we entered
                "exit_price":   round(fill, 4),               # price we received when we exited
                "return_%":     round(trade_return_pct, 2),   # profit or loss as a percentage
            })

            cash   = max(0.0, proceeds)  # update cash balance (floor at 0 to handle edge cases)
            shares = 0.0                 # we are now flat
            entry_price = None
            entry_date  = None

        # Mark the portfolio to market: cash + current value of any open position
        portfolio_values.append(cash + shares * row["Close"])  # use today's close as the fair value of held shares

    portfolio = pd.Series(portfolio_values, index=df.index, name="portfolio")  # daily account value as a time series
    trades    = pd.DataFrame(trades_log)                                        # one row per completed round trip

    return portfolio, trades, df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(portfolio: pd.Series, risk_free_rate: float) -> dict:
    # Computes all summary statistics from the daily portfolio value series
    returns = portfolio.pct_change().dropna()  # day-over-day percentage change in portfolio value

    total_ret = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100              # overall % gain over the full period
    ann_ret   = ((1 + total_ret / 100) ** (TRADING_DAYS / len(returns)) - 1) * 100  # CAGR: annualised compound return
    ann_vol   = returns.std() * np.sqrt(TRADING_DAYS) * 100                     # annualised volatility (risk)

    daily_rf = risk_free_rate / TRADING_DAYS                                     # daily equivalent of the annual risk-free rate
    excess   = returns - daily_rf                                                 # daily return above the risk-free rate
    sharpe   = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS) if excess.std() > 0 else float("nan")  # Sharpe: reward per unit of risk

    cumulative   = portfolio / portfolio.iloc[0]                  # normalise portfolio to start at 1.0
    rolling_peak = cumulative.cummax()                            # highest portfolio value seen up to each date
    drawdown     = (cumulative - rolling_peak) / rolling_peak     # fractional decline from the running peak

    mdd          = drawdown.min() * 100                           # worst drawdown as a percentage
    trough_date  = drawdown.idxmin()                              # date the portfolio hit its lowest relative point
    peak_date    = rolling_peak[:trough_date].idxmax()            # date of the highest point before that trough

    return {
        "total_return": total_ret,
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": mdd,
        "peak_date":    peak_date,
        "trough_date":  trough_date,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(
    ticker: str,
    df: pd.DataFrame,
    portfolio: pd.Series,
    trades: pd.DataFrame,
    metrics: dict,
    exec_params: Optional[dict],
    label: str,
) -> None:
    # Prints a formatted block of backtest results for one simulation run
    start_date = portfolio.index[0].date()  # first date in the price series
    end_date   = portfolio.index[-1].date() # last date in the price series

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  {ticker}  |  {start_date} → {end_date}")
    if exec_params:
        # Show the three execution-cost parameters so the user can see what was modelled
        print(
            f"  Slippage: {exec_params['slippage_bps']} bps  |  "
            f"Latency: {exec_params['latency']}d  |  "
            f"Commission: ${exec_params['commission']:.2f}/trade"
        )
    print(f"{'='*62}")

    # Performance metrics
    print(f"  Total return        : {metrics['total_return']:+.2f}%")
    print(f"  Annualised return   : {metrics['ann_return']:+.2f}%")
    print(f"  Annualised vol      : {metrics['ann_vol']:.2f}%")
    print(f"  Sharpe ratio        : {metrics['sharpe']:.3f}")
    print(f"  Maximum drawdown    : {metrics['max_drawdown']:.2f}%")
    print(f"    Peak              : {metrics['peak_date'].date()}")
    print(f"    Trough            : {metrics['trough_date'].date()}")

    # Per-trade statistics (only if at least one round trip completed)
    if not trades.empty:
        win_rate       = (trades["return_%"] > 0).mean() * 100  # percentage of trades that were profitable
        avg_ret        = trades["return_%"].mean()               # average return per trade
        best_trade     = trades["return_%"].max()                # single best trade
        worst_trade    = trades["return_%"].min()                # single worst trade
        time_in_market = (df["signal"] > 0).mean() * 100        # fraction of days holding a position

        print(f"\n  --- Trade summary ---")
        print(f"  Completed trades    : {len(trades)}")
        print(f"  Win rate            : {win_rate:.1f}%")
        print(f"  Avg trade return    : {avg_ret:+.2f}%")
        print(f"  Best trade          : {best_trade:+.2f}%")
        print(f"  Worst trade         : {worst_trade:+.2f}%")
        print(f"  Time in market      : {time_in_market:.1f}%")
    else:
        print(f"\n  No completed trades — try a shorter SMA window or a longer lookback period.")

    print(f"{'='*62}")


def print_comparison(ideal: dict, real: dict) -> None:
    # Prints a side-by-side delta table so the user can see the cost of realistic execution
    print(f"\n{'='*62}")
    print(f"  Execution cost: Ideal → Realistic")
    print(f"{'='*62}")

    rows = [
        ("Total return",      "total_return", "%"),  # did friction eat into total profit?
        ("Annualised return",  "ann_return",   "%"),  # same on a per-year basis
        ("Sharpe ratio",       "sharpe",       ""),   # did risk-adjusted performance change?
        ("Max drawdown",       "max_drawdown", "%"),  # did friction make drawdowns worse?
    ]

    for label, key, unit in rows:
        iv    = ideal[key]                                        # value under ideal conditions
        rv    = real[key]                                         # value under realistic conditions
        delta = rv - iv                                           # difference (negative = friction hurt)
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<22}: {iv:+.3f}{unit}  →  {rv:+.3f}{unit}  (Δ {sign}{delta:.3f}{unit})")

    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realistic SMA-crossover backtester with slippage, latency, and commission modelling"
    )

    # Required positional argument
    parser.add_argument("ticker", type=str.upper, help="Stock ticker symbol, e.g. AAPL")

    # Date range: either a relative period OR explicit start/end dates
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--period", default="5y",
        help="Relative lookback: 1mo 3mo 6mo 1y 2y 5y 10y ytd max (default: 5y)",
    )
    date_group.add_argument("--start", help="Start date in YYYY-MM-DD format (use with --end)")
    parser.add_argument("--end", help="End date in YYYY-MM-DD format (defaults to today if --start is provided)")

    # Strategy parameters
    parser.add_argument("--short-sma",  type=int,   default=50,      help="Short SMA window in days (default: 50)")
    parser.add_argument("--long-sma",   type=int,   default=200,     help="Long SMA window in days (default: 200)")
    parser.add_argument("--capital",    type=float, default=100_000, help="Starting capital in USD (default: 100000)")

    # Execution variables (the new parameters this script adds over the basic backtester)
    parser.add_argument(
        "--slippage", type=float, default=5.0,
        help="Slippage per trade in basis points; 1 bps = 0.01%%. Default: 5 bps (0.05%%)",
    )
    parser.add_argument(
        "--latency", type=int, default=1,
        help="Days between signal and order fill; 0 = same-day (theoretical), 1 = next-day open (default)",
    )
    parser.add_argument(
        "--commission", type=float, default=0.0,
        help="Flat commission per trade in USD (default: 0)",
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.04,
        help="Annual risk-free rate as a decimal used in Sharpe calculation (default: 0.04 = 4%%)",
    )

    args = parser.parse_args()  # parse everything the user typed

    if args.short_sma >= args.long_sma:
        sys.exit("--short-sma must be strictly less than --long-sma")  # guard against nonsensical SMA configuration

    # Resolve date range based on which flags were provided
    start, end, period = None, None, None
    if args.start:
        start, end = args.start, args.end  # use explicit calendar dates (end=None means today)
    else:
        period = args.period               # use a relative period like "5y"

    df = fetch_ohlcv(args.ticker, start, end, period)  # download historical OHLCV data

    # --- Ideal simulation: zero friction, zero latency ---
    # This is the theoretical best case — no slippage, no delay, no fees
    ideal_portfolio, ideal_trades, ideal_df = simulate(
        df,
        short_window=args.short_sma,
        long_window=args.long_sma,
        slippage_bps=0.0,   # no price degradation
        latency_days=0,     # instant execution
        commission=0.0,     # no fees
        initial_capital=args.capital,
    )

    # --- Realistic simulation: with user-specified execution costs ---
    real_portfolio, real_trades, real_df = simulate(
        df,
        short_window=args.short_sma,
        long_window=args.long_sma,
        slippage_bps=args.slippage,
        latency_days=args.latency,
        commission=args.commission,
        initial_capital=args.capital,
    )

    ideal_metrics = compute_metrics(ideal_portfolio, args.risk_free_rate)  # summary stats for the ideal run
    real_metrics  = compute_metrics(real_portfolio,  args.risk_free_rate)  # summary stats for the realistic run

    # Print ideal results first as the benchmark
    print_results(
        args.ticker, ideal_df, ideal_portfolio, ideal_trades, ideal_metrics,
        exec_params=None,
        label="IDEAL EXECUTION  (zero slippage · zero latency · zero commission)",
    )

    # Print realistic results showing the impact of execution costs
    print_results(
        args.ticker, real_df, real_portfolio, real_trades, real_metrics,
        exec_params={
            "slippage_bps": args.slippage,
            "latency":      args.latency,
            "commission":   args.commission,
        },
        label="REALISTIC EXECUTION",
    )

    # Print the delta table comparing both runs
    print_comparison(ideal_metrics, real_metrics)


if __name__ == "__main__":
    main()  # only execute when run directly, not when imported as a module
