#!/usr/bin/env python3
"""
Realistic SMA-crossover portfolio backtester with slippage, latency, and commission.

Each ticker runs its own independent SMA-crossover strategy. The resulting portfolio
value series are normalised and combined by weight into a single portfolio. Both an
ideal (zero-cost) and realistic (with execution costs) simulation are shown, along
with a per-ticker breakdown comparing the two runs side by side.

Usage:
    python backtest_realistic_portfolio.py AAPL MSFT GOOGL
    python backtest_realistic_portfolio.py AAPL MSFT --weights 0.6 0.4 --period 5y
    python backtest_realistic_portfolio.py SPY QQQ --weights 0.5 0.5 --slippage 10 --latency 2 --commission 1.00
    python backtest_realistic_portfolio.py AAPL TSLA MSFT --start 2020-01-01 --end 2024-01-01
"""

import argparse
import sys
from typing import Optional, Tuple, cast
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required: pip install yfinance")


TRADING_DAYS = 252  # trading days per year; used to annualise daily stats


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
) -> pd.DataFrame:
    # Downloads OHLCV price data for one ticker from Yahoo Finance
    t = yf.Ticker(ticker)
    if period:
        raw = t.history(period=period)         # relative window, e.g. "5y"
    else:
        raw = t.history(start=start, end=end)  # explicit calendar date range

    if raw.empty:
        sys.exit(f"No data for '{ticker}'. Check the ticker symbol and date range.")

    return pd.DataFrame(raw[["Open", "High", "Low", "Close", "Volume"]])


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
    SMA-crossover strategy with configurable execution assumptions.

    Goes long when short SMA crosses above long SMA; goes flat on the reverse.
    Slippage degrades the fill price on every trade.
    Latency delays the signal so we only act on past information.
    Commission is a flat dollar fee deducted per trade.

    Returns: (portfolio daily value, completed trades log, enriched df with signals).
    """
    slippage = slippage_bps / 10_000  # convert basis points to a decimal, e.g. 5 bps → 0.0005

    df = df.copy()  # work on a copy so we don't mutate the caller's DataFrame

    # Compute both moving averages over the Close price
    df["short_sma"] = df["Close"].rolling(short_window).mean()
    df["long_sma"] = df["Close"].rolling(long_window).mean()

    # Raw signal: 1 = want to be long, 0 = want to be flat
    # Before enough data exists for both SMAs the signal stays 0
    df["raw_signal"] = np.where(
        df["short_sma"].notna() & df["long_sma"].notna(),
        (df["short_sma"] > df["long_sma"]).astype(int),
        0,
    )

    # Apply latency: shift signal forward so we only react to information from latency_days ago
    df["signal"] = df["raw_signal"].shift(latency_days).fillna(0)

    # Trade signal: +1 = buy (flat → long), -1 = sell (long → flat), 0 = no change
    df["trade_signal"] = df["signal"].diff().fillna(0)

    # Portfolio simulation — iterate day by day and apply fills
    cash = float(initial_capital)
    shares = 0.0
    entry_price: Optional[float] = None
    entry_date = None
    portfolio_values = []
    trades_log = []

    for date, row in df.iterrows():
        signal = row["trade_signal"]

        # BUY: crossover fired and we are currently flat
        if signal > 0.5 and shares == 0:
            fill = float(row["Open"]) * (1 + slippage)  # type: ignore[arg-type]  # pay slightly above open
            affordable = max(0.0, cash - commission)     # subtract commission from available cash
            shares = affordable / fill                   # buy as many shares as we can afford
            cash = 0.0                                   # all capital is now in shares
            entry_price = fill
            entry_date = date

        # SELL: crossover reversed and we are currently holding shares
        elif signal < -0.5 and shares > 0:
            fill = float(row["Open"]) * (1 - slippage)  # type: ignore[arg-type]  # receive slightly below open
            proceeds = shares * fill - commission        # total cash received after commission
            assert entry_price is not None
            trade_return_pct = (fill / entry_price - 1) * 100  # % gain or loss on this trade

            trades_log.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": round(entry_price, 4),
                "exit_price": round(fill, 4),
                "return_%": round(trade_return_pct, 2),
            })

            cash = max(0.0, proceeds)
            shares = 0.0
            entry_price = None
            entry_date = None

        # Mark portfolio to market: cash + current value of any open position
        portfolio_values.append(cash + shares * row["Close"])

    portfolio = pd.Series(portfolio_values, index=df.index, name="portfolio")
    trades = pd.DataFrame(trades_log)

    return portfolio, trades, df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(portfolio: pd.Series, risk_free_rate: float) -> dict:
    # Computes all summary statistics from the daily portfolio value series
    returns = portfolio.pct_change().dropna()

    total_ret = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100
    ann_ret = ((1 + total_ret / 100) ** (TRADING_DAYS / len(returns)) - 1) * 100
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS) * 100

    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS) if excess.std() > 0 else float("nan")

    cumulative = portfolio / portfolio.iloc[0]
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / rolling_peak

    mdd = drawdown.min() * 100
    trough_date = drawdown.idxmin()
    peak_date = rolling_peak[:trough_date].idxmax()

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "peak_date": peak_date,
        "trough_date": trough_date,
    }


# ---------------------------------------------------------------------------
# Portfolio combination
# ---------------------------------------------------------------------------

def combine_portfolios(portfolios: list, tickers: list, weights: list, initial_capital: float) -> pd.Series:
    # Align all per-ticker portfolio series to common dates
    df = pd.concat(portfolios, axis=1)
    df.columns = pd.Index(tickers)  # label columns by ticker so weight alignment is by name
    df = df.dropna()

    # Normalise each series to start at 1.0 so we can combine growth rates
    # (each portfolio starts at initial_capital, so dividing by initial_capital normalises it)
    normalised = df / df.iloc[0]

    # Weighted sum of normalised series, then scale back to dollars
    weights_s = pd.Series(weights, index=tickers)
    combined = (normalised * weights_s).sum(axis=1) * initial_capital

    return pd.Series(combined, name="portfolio")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(
    tickers: list,
    weights: list,
    start_date,
    end_date,
    trades_list: list,
    metrics: dict,
    exec_params: Optional[dict],
    label: str,
) -> None:
    # Prints a formatted block of backtest results for one simulation run (ideal or realistic)
    weight_str = "  ".join(f"{t}: {w*100:.1f}%" for t, w in zip(tickers, weights))

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  {start_date} → {end_date}  |  Weights: {weight_str}")
    if exec_params:
        print(
            f"  Slippage: {exec_params['slippage_bps']} bps  |  "
            f"Latency: {exec_params['latency']}d  |  "
            f"Commission: ${exec_params['commission']:.2f}/trade"
        )
    print(f"{'='*62}")

    print(f"  Total return        : {metrics['total_return']:+.2f}%")
    print(f"  Annualised return   : {metrics['ann_return']:+.2f}%")
    print(f"  Annualised vol      : {metrics['ann_vol']:.2f}%")
    print(f"  Sharpe ratio        : {metrics['sharpe']:.3f}")
    print(f"  Maximum drawdown    : {metrics['max_drawdown']:.2f}%")
    print(f"    Peak              : {metrics['peak_date'].date()}")
    print(f"    Trough            : {metrics['trough_date'].date()}")

    # Sum completed trades across all tickers
    total_trades = sum(len(t) for t in trades_list)
    if total_trades > 0:
        print(f"  Total trades (all tickers) : {total_trades}")
    else:
        print("\n  No completed trades — try a shorter SMA window or a longer lookback period.")

    print(f"{'='*62}")


def print_comparison(ideal: dict, real: dict) -> None:
    # Delta table: shows how much execution costs hurt the portfolio vs the ideal run
    print(f"\n{'='*62}")
    print("  Execution cost: Ideal → Realistic (portfolio)")
    print(f"{'='*62}")

    rows = [
        ("Total return",      "total_return", "%"),
        ("Annualised return", "ann_return",   "%"),
        ("Sharpe ratio",      "sharpe",       ""),
        ("Max drawdown",      "max_drawdown", "%"),
    ]

    for label, key, unit in rows:
        iv = ideal[key]
        rv = real[key]
        delta = rv - iv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<22}: {iv:+.3f}{unit}  →  {rv:+.3f}{unit}  (Δ {sign}{delta:.3f}{unit})")

    print(f"{'='*62}\n")


def print_ticker_breakdown(
    tickers: list,
    weights: list,
    ideal_results: list,
    real_results: list,
    risk_free_rate: float,
) -> None:
    # Side-by-side ideal vs realistic metrics for each individual ticker
    print(f"{'='*62}")
    print("  Per-ticker breakdown")
    print(f"{'='*62}")

    for i, ticker in enumerate(tickers):
        ideal_portfolio, ideal_trades, _ = ideal_results[i]
        real_portfolio, real_trades, real_df = real_results[i]

        ideal_m = compute_metrics(ideal_portfolio, risk_free_rate)
        real_m = compute_metrics(real_portfolio, risk_free_rate)

        # Fraction of days the strategy held a position (same for ideal and realistic since signal is shared)
        time_in_market = (real_df["signal"] > 0).mean() * 100

        print(f"\n  {ticker}  (weight: {weights[i]*100:.1f}%)")
        print(f"  {'─'*54}")
        print(f"  {'':22}  {'Ideal':>10}  {'Realistic':>10}")
        print(f"  {'Total return':22}  {ideal_m['total_return']:>+9.2f}%  {real_m['total_return']:>+9.2f}%")
        print(f"  {'Ann. return':22}  {ideal_m['ann_return']:>+9.2f}%  {real_m['ann_return']:>+9.2f}%")
        print(f"  {'Sharpe ratio':22}  {ideal_m['sharpe']:>10.3f}  {real_m['sharpe']:>10.3f}")
        print(f"  {'Max drawdown':22}  {ideal_m['max_drawdown']:>9.2f}%  {real_m['max_drawdown']:>9.2f}%")
        print(f"  {'Completed trades':22}  {len(ideal_trades):>10}  {len(real_trades):>10}")
        print(f"  {'Time in market':22}  {time_in_market:>9.1f}%")

    print(f"\n{'='*62}\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_backtest(
    tickers: list,
    weights: list,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
    short_window: int,
    long_window: int,
    slippage_bps: float,
    latency_days: int,
    commission: float,
    risk_free_rate: float,
    initial_capital: float,
) -> None:
    # Download OHLCV data for every ticker
    ohlcv_list = [fetch_ohlcv(t, start, end, period) for t in tickers]

    # Find the common trading days across all tickers (inner join on the Close column)
    close_df = pd.concat(
        [df["Close"].rename(t) for df, t in zip(ohlcv_list, tickers)],  # type: ignore[arg-type]
        axis=1,
    ).dropna()

    if close_df.empty:
        sys.exit("No overlapping trading days found across all tickers.")

    # Slice every OHLCV DataFrame down to the shared date range so all simulations are aligned
    aligned_dfs = [ohlcv_list[i].loc[close_df.index] for i in range(len(tickers))]

    start_date = cast(pd.Timestamp, close_df.index[0]).date()
    end_date = cast(pd.Timestamp, close_df.index[-1]).date()

    # Run both simulations for each ticker independently
    ideal_results = [
        simulate(df, short_window, long_window, 0.0, 0, 0.0, initial_capital)
        for df in aligned_dfs
    ]
    real_results = [
        simulate(df, short_window, long_window, slippage_bps, latency_days, commission, initial_capital)
        for df in aligned_dfs
    ]

    # Combine per-ticker portfolio series into one weighted portfolio value series
    ideal_portfolio = combine_portfolios([r[0] for r in ideal_results], tickers, weights, initial_capital)
    real_portfolio = combine_portfolios([r[0] for r in real_results], tickers, weights, initial_capital)

    ideal_metrics = compute_metrics(ideal_portfolio, risk_free_rate)
    real_metrics = compute_metrics(real_portfolio, risk_free_rate)

    exec_params = {"slippage_bps": slippage_bps, "latency": latency_days, "commission": commission}

    print_results(
        tickers, weights, start_date, end_date,
        [r[1] for r in ideal_results], ideal_metrics, None,
        "IDEAL EXECUTION  (zero slippage · zero latency · zero commission)",
    )
    print_results(
        tickers, weights, start_date, end_date,
        [r[1] for r in real_results], real_metrics, exec_params,
        "REALISTIC EXECUTION",
    )

    print_comparison(ideal_metrics, real_metrics)
    print_ticker_breakdown(tickers, weights, ideal_results, real_results, risk_free_rate)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realistic SMA-crossover portfolio backtester with slippage, latency, and commission"
    )

    # nargs="+" requires at least one ticker; type=str.upper auto-uppercases input
    parser.add_argument("tickers", nargs="+", type=str.upper, help="Stock ticker symbols, e.g. AAPL MSFT GOOGL")

    # --period and --start/--end are mutually exclusive date range options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--period", default="5y",
        help="Relative lookback: 1mo 3mo 6mo 1y 2y 5y 10y ytd max (default: 5y)",
    )
    date_group.add_argument("--start", help="Start date YYYY-MM-DD (use with --end)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (defaults to today if --start is given)")

    parser.add_argument(
        "--weights", nargs="+", type=float,
        help="Portfolio weights as decimals, must sum to 1.0 (default: equal weight)",
    )
    parser.add_argument("--short-sma", type=int, default=50, help="Short SMA window in days (default: 50)")
    parser.add_argument("--long-sma", type=int, default=200, help="Long SMA window in days (default: 200)")
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital in USD (default: 100000)")

    parser.add_argument(
        "--slippage", type=float, default=5.0,
        help="Slippage per trade in basis points; 1 bps = 0.01%%. Default: 5 bps",
    )
    parser.add_argument(
        "--latency", type=int, default=1,
        help="Days between signal and fill; 1 = next-day open (default)",
    )
    parser.add_argument(
        "--commission", type=float, default=0.0,
        help="Flat commission per trade in USD (default: 0)",
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.04,
        help="Annual risk-free rate as a decimal (default: 0.04 = 4%%)",
    )

    args = parser.parse_args()

    if args.short_sma >= args.long_sma:
        sys.exit("--short-sma must be strictly less than --long-sma")

    # Resolve weights — default to equal allocation across all tickers
    if args.weights is None:
        n = len(args.tickers)
        weights = [1.0 / n] * n
    else:
        weights = args.weights
        if len(weights) != len(args.tickers):
            sys.exit(f"Number of weights ({len(weights)}) must match number of tickers ({len(args.tickers)}).")
        if abs(sum(weights) - 1.0) > 1e-6:  # allow tiny floating-point rounding errors
            sys.exit(f"Weights must sum to 1.0 (got {sum(weights):.6f}).")

    start, end, period = None, None, None
    if args.start:
        start, end = args.start, args.end
    else:
        period = args.period

    run_backtest(
        args.tickers, weights, start, end, period,
        args.short_sma, args.long_sma,
        args.slippage, args.latency, args.commission,
        args.risk_free_rate, args.capital,
    )


if __name__ == "__main__":
    main()
