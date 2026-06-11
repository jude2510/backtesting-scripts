#!/usr/bin/env python3
"""
Buy-and-hold portfolio backtester: backtests a weighted basket of stocks.

Each ticker is normalised to 1.0 on the first common trading day, scaled by
its weight, and summed into a single portfolio value series. Weights are fixed
at inception (no rebalancing).

Usage:
    python backtest_portfolio.py AAPL MSFT GOOGL
    python backtest_portfolio.py AAPL MSFT --weights 0.6 0.4 --period 5y
    python backtest_portfolio.py AAPL TSLA MSFT --start 2020-01-01 --end 2024-01-01
    python backtest_portfolio.py SPY BND --weights 0.6 0.4 --risk-free-rate 0.05
"""

import argparse
import sys
from typing import Optional, cast
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required: pip install yfinance")


TRADING_DAYS = 252  # number of trading days in a year; used to annualise daily stats


def fetch_prices(ticker: str, start: Optional[str], end: Optional[str], period: Optional[str]) -> pd.Series:
    # Downloads historical closing prices for one ticker from Yahoo Finance
    t = yf.Ticker(ticker)
    if period:
        hist = t.history(period=period)          # relative lookback, e.g. "5y"
    else:
        hist = t.history(start=start, end=end)   # explicit calendar date range
    if hist.empty:
        sys.exit(f"No data returned for '{ticker}'. Check the ticker and date range.")
    return hist["Close"].rename(ticker)  # type: ignore[arg-type]  # label the series with the ticker name


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    # Risk-adjusted return: how much excess return earned per unit of volatility
    daily_rf = risk_free_rate / TRADING_DAYS     # convert annual rate to a daily equivalent
    excess = returns - daily_rf                  # daily return above the risk-free rate
    if excess.std() == 0:
        return float("nan")                      # avoid division by zero if the series never moved
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)  # annualise by scaling with sqrt(252)


def max_drawdown(prices: pd.Series) -> "tuple[float, pd.Timestamp, pd.Timestamp]":
    # Finds the worst peak-to-trough decline over the full period
    cumulative = prices / prices.iloc[0]         # normalise so day 1 = 1.0
    rolling_peak = cumulative.cummax()           # highest value seen up to each date
    drawdown = (cumulative - rolling_peak) / rolling_peak  # fractional drop from the running peak (always <= 0)
    mdd = drawdown.min()                         # single worst drawdown value
    trough_date = drawdown.idxmin()              # date the portfolio hit its lowest relative point
    peak_date = rolling_peak[:trough_date].idxmax()  # date of the highest point before that trough
    return mdd, peak_date, trough_date


def compute_and_print_metrics(label: str, prices: pd.Series, risk_free_rate: float) -> None:
    # Computes and prints the standard performance metrics for any price series
    returns = prices.pct_change().dropna()       # day-over-day % change; first NaN row dropped
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100           # overall % gain over the period
    ann_return = ((1 + total_return / 100) ** (TRADING_DAYS / len(returns)) - 1) * 100  # CAGR
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS) * 100                  # annualised volatility (risk)
    sr = sharpe_ratio(returns, risk_free_rate)
    mdd, peak_date, trough_date = max_drawdown(prices)

    print(f"  {label}")
    print(f"  Total return        : {total_return:+.2f}%")
    print(f"  Annualised return   : {ann_return:+.2f}%")
    print(f"  Annualised vol      : {ann_vol:.2f}%")
    print(f"  Sharpe ratio        : {sr:.3f}  (Rf={risk_free_rate*100:.1f}%)")
    print(f"  Maximum drawdown    : {mdd*100:.2f}%")
    print(f"    Peak              : {peak_date.date()}")
    print(f"    Trough            : {trough_date.date()}")


def run_backtest(
    tickers: list,
    weights: list,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
    risk_free_rate: float,
    initial_capital: float,
) -> None:
    # Fetch closing prices for every ticker
    price_series = [fetch_prices(t, start, end, period) for t in tickers]

    # Combine into a DataFrame and drop any day where at least one ticker has no data
    # (inner join — ensures all series share exactly the same trading days)
    prices_df = pd.concat(price_series, axis=1).dropna()

    if prices_df.empty:
        sys.exit("No overlapping trading days found across all tickers.")

    # Convert weights list into a Series indexed by ticker so the multiplication aligns by name
    weights_series = pd.Series(weights, index=tickers)

    # Normalise each stock to start at 1.0, then scale by its weight and sum across columns
    # Result: a single daily dollar-value series representing the whole portfolio
    normalised = prices_df / prices_df.iloc[0]                         # each stock starts at 1.0
    portfolio = (normalised * weights_series).sum(axis=1) * initial_capital  # weighted sum × starting capital

    start_date = cast(pd.Timestamp, prices_df.index[0]).date()
    end_date = cast(pd.Timestamp, prices_df.index[-1]).date()
    weight_str = "  ".join(f"{t}: {w*100:.1f}%" for t, w in zip(tickers, weights))

    print(f"\n{'='*58}")
    print(f"  Portfolio Backtest  |  {start_date} → {end_date}")
    print(f"  Weights: {weight_str}")
    print(f"{'='*58}")

    # Print portfolio-level metrics (the combined weighted series)
    compute_and_print_metrics("PORTFOLIO (combined)", portfolio, risk_free_rate)

    # Print individual metrics for each stock so you can compare contributions
    print(f"\n  {'─'*52}")
    print("  Per-ticker breakdown")
    print(f"  {'─'*52}")
    for ticker in tickers:
        print()
        compute_and_print_metrics(ticker, cast(pd.Series, prices_df[ticker]), risk_free_rate)

    print(f"\n{'='*58}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Buy-and-hold portfolio backtester")
    # nargs="+" means one or more tickers are required
    parser.add_argument("tickers", nargs="+", type=str.upper, help="Stock ticker symbols, e.g. AAPL MSFT GOOGL")

    # --period and --start are mutually exclusive: use one or the other, not both
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--period", default="5y",
        help="Lookback period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Default: 5y",
    )
    group.add_argument("--start", help="Start date YYYY-MM-DD (use with --end)")

    parser.add_argument("--end", help="End date YYYY-MM-DD (defaults to today if --start is given)")
    parser.add_argument(
        "--weights", nargs="+", type=float,
        help="Portfolio weights as decimals, must sum to 1.0 (default: equal weight)",
    )
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital in USD (default: 100000)")
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.04,
        help="Annual risk-free rate as a decimal (default: 0.04 = 4%%)",
    )

    args = parser.parse_args()

    if args.weights is None:
        # No weights provided — split evenly across all tickers
        n = len(args.tickers)
        weights = [1.0 / n] * n
    else:
        weights = args.weights
        if len(weights) != len(args.tickers):
            sys.exit(f"Number of weights ({len(weights)}) must match number of tickers ({len(args.tickers)}).")
        if abs(sum(weights) - 1.0) > 1e-6:  # allow for tiny floating-point rounding errors
            sys.exit(f"Weights must sum to 1.0 (got {sum(weights):.6f}).")

    # Resolve which date mode to use
    start, end, period = None, None, None
    if args.start:
        start = args.start
        end = args.end
    else:
        period = args.period

    run_backtest(args.tickers, weights, start, end, period, args.risk_free_rate, args.capital)


if __name__ == "__main__":
    main()
