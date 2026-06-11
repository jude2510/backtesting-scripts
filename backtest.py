#!/usr/bin/env python3
# Tells the OS to run this file with Python 3 when executed directly from the terminal

"""
Buy-and-hold backtester: calculates Sharpe ratio and maximum drawdown.

Usage:
    python backtest.py AAPL --start 2020-01-01 --end 2024-01-01
    python backtest.py TSLA --period 5y
    python backtest.py MSFT --period 1y --risk-free-rate 0.05
"""

import argparse          # standard library module for parsing command-line arguments
import sys               # standard library module for exiting the program with an error message
from typing import Optional  # lets us annotate function parameters that can be None
import numpy as np       # numerical computing library; used for sqrt and math operations
import pandas as pd      # data analysis library; used to work with time-series price data

try:
    import yfinance as yf                               # third-party library that downloads stock price data from Yahoo Finance
except ImportError:
    sys.exit("yfinance is required: pip install yfinance")  # stop immediately with a helpful message if yfinance isn't installed


TRADING_DAYS = 252  # number of trading days in a year; used to annualise daily statistics


def fetch_prices(ticker: str, start: Optional[str], end: Optional[str], period: Optional[str]) -> pd.Series:
    # Downloads historical closing prices for the given ticker from Yahoo Finance
    t = yf.Ticker(ticker)       # create a Ticker object that represents the requested stock
    if period:
        hist = t.history(period=period)         # download data for a relative period like "5y" or "1mo"
    else:
        hist = t.history(start=start, end=end)  # download data between two specific calendar dates

    if hist.empty:
        sys.exit(f"No data returned for '{ticker}'. Check the ticker and date range.")  # stop if Yahoo returned nothing (bad ticker or out-of-range dates)

    return hist["Close"].rename(ticker)  # keep only the daily closing price column and label it with the ticker symbol


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    # Measures risk-adjusted return: how much excess return you earn per unit of volatility
    daily_rf = risk_free_rate / TRADING_DAYS    # convert the annual risk-free rate to a daily equivalent
    excess = returns - daily_rf                 # subtract the daily risk-free rate from each day's return to get "excess" return
    if excess.std() == 0:
        return float("nan")                     # avoid division by zero if the stock never moved
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)  # annualise: divide average excess return by its standard deviation, then scale up by sqrt(252)


def max_drawdown(prices: pd.Series) -> "tuple[float, pd.Timestamp, pd.Timestamp]":
    # Finds the largest peak-to-trough decline over the entire period
    cumulative = prices / prices.iloc[0]        # normalise prices so day 1 = 1.0, making growth easy to compare
    rolling_peak = cumulative.cummax()          # at each date, record the highest price seen so far
    drawdown = (cumulative - rolling_peak) / rolling_peak  # how far the current price has fallen from that running peak (always <= 0)

    mdd = drawdown.min()                        # the single worst (most negative) drawdown value across the whole series
    trough_date = drawdown.idxmin()             # the date when the price hit its lowest point relative to the prior peak
    peak_date = rolling_peak[:trough_date].idxmax()  # the date of the highest price that occurred before the trough

    return mdd, peak_date, trough_date  # return the magnitude and the two boundary dates


def run_backtest(ticker: str, start: Optional[str], end: Optional[str], period: Optional[str], risk_free_rate: float) -> None:
    # Orchestrates the full backtest and prints a results summary
    prices = fetch_prices(ticker, start, end, period)  # download the closing price series
    returns = prices.pct_change().dropna()             # compute day-over-day percentage change; drop the first NaN (no prior day on day 1)

    sr = sharpe_ratio(returns, risk_free_rate)         # compute the annualised Sharpe ratio
    mdd, peak_date, trough_date = max_drawdown(prices)   # compute max drawdown and its peak/trough dates

    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100           # overall % gain from first to last price
    ann_return = ((1 + total_return / 100) ** (TRADING_DAYS / len(returns)) - 1) * 100  # compound annual growth rate (CAGR)
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS) * 100                  # annualise daily volatility by scaling by sqrt(252)

    print(f"\n{'='*50}")
    print(f"  Backtest: {ticker}  |  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"{'='*50}")
    print(f"  Total return        : {total_return:+.2f}%")                        # total % return over the period (+ or - sign always shown)
    print(f"  Annualised return   : {ann_return:+.2f}%")                          # CAGR: what the return looks like on a per-year basis
    print(f"  Annualised vol      : {ann_vol:.2f}%")                              # annualised standard deviation of daily returns (a measure of risk)
    print(f"  Sharpe ratio        : {sr:.3f}  (Rf={risk_free_rate*100:.1f}%)")    # risk-adjusted return; higher is better (>1 is generally good)
    print(f"  Maximum drawdown    : {mdd*100:.2f}%")                              # worst peak-to-trough loss expressed as a percentage
    print(f"    Peak              : {peak_date.date()}")                           # date the price reached its high before the worst drop
    print(f"    Trough            : {trough_date.date()}")                         # date the price hit its lowest point during the worst drop
    print(f"{'='*50}\n")


def main() -> None:
    # Entry point: defines all CLI arguments and kicks off the backtest
    parser = argparse.ArgumentParser(description="Buy-and-hold backtester")  # create an argument parser with a description shown in --help
    parser.add_argument("ticker", type=str.upper, help="Stock ticker symbol (e.g. AAPL)")  # positional argument: ticker is required and auto-uppercased

    group = parser.add_mutually_exclusive_group()  # ensures the user can't supply both --period and --start at the same time
    group.add_argument("--period", default="5y",
                       help="Lookback period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Default: 5y")  # relative lookback, e.g. "5y" means last 5 years
    group.add_argument("--start", help="Start date YYYY-MM-DD (use with --end)")  # explicit start date for a fixed date range

    parser.add_argument("--end", help="End date YYYY-MM-DD (defaults to today if --start is given)")  # optional end date; omit to use today
    parser.add_argument("--risk-free-rate", type=float, default=0.04,
                        help="Annual risk-free rate as a decimal (default: 0.04 = 4%%)")  # the return of a "safe" asset (e.g. T-bills) used in Sharpe calculation

    args = parser.parse_args()  # parse whatever the user typed on the command line into the args object

    start, end, period = None, None, None   # initialise all three date variables to None before deciding which path to take
    if args.start:
        start = args.start   # user provided an explicit start date, so use the fixed date range mode
        end = args.end       # end date may also be None, which yfinance interprets as today
    else:
        period = args.period  # no explicit start date, so use the relative period (e.g. "5y")

    run_backtest(args.ticker, start, end, period, args.risk_free_rate)  # run the backtest with all parsed parameters


if __name__ == "__main__":
    main()  # only run main() when this file is executed directly, not when imported as a module
