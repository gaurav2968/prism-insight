#!/usr/bin/env python3
"""
PRISM India — Backtest Engine

Replays historical trigger results and measures forward returns.
Fetches actual prices from yfinance to compute real P&L.

Metrics computed:
  - Per-trade: return at 3/7/14/30 days, MAE (max drawdown), MFE (max run-up)
  - Per-trigger-type: win rate, avg win, avg loss, expectancy, profit factor
  - Overall: CAGR, Sharpe, Sortino, max drawdown, sector concentration

Usage:
    python prism-in/backtest_engine.py
    python prism-in/backtest_engine.py --min-date 20260501
    python prism-in/backtest_engine.py --output backtest_results.json
"""

import json
import glob
import logging
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PRISM_IN_DIR = Path(__file__).parent

# Forward return windows (trading days)
RETURN_WINDOWS = [3, 7, 14, 30]


def load_trigger_files(min_date: str = None, max_date: str = None, input_dir: str = None) -> list:
    """Load all trigger result JSON files from the given or default directory."""
    search_dir = Path(input_dir) if input_dir else PROJECT_ROOT
    pattern = str(search_dir / "trigger_results_in_morning_*.json")
    files = sorted(glob.glob(pattern))

    results = []
    for f in files:
        fname = os.path.basename(f)
        # Extract date from filename: trigger_results_in_morning_YYYYMMDD.json
        date_str = fname.replace("trigger_results_in_morning_", "").replace(".json", "")
        if len(date_str) != 8 or not date_str.isdigit():
            continue
        if min_date and date_str < min_date:
            continue
        if max_date and date_str > max_date:
            continue

        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            results.append({"date": date_str, "path": f, "data": data})
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Loaded {len(results)} trigger result files")
    return results


def extract_picks(trigger_files: list) -> pd.DataFrame:
    """Extract all stock picks from trigger files into a flat DataFrame."""
    rows = []
    for tf in trigger_files:
        trade_date = tf["date"]
        data = tf["data"]
        metadata = data.get("metadata", {})

        for trigger_type, stocks in data.items():
            if trigger_type in ("metadata", "screening_summary"):
                continue
            if not isinstance(stocks, list):
                continue

            for stock in stocks:
                ticker = stock.get("ticker", "")
                if not ticker:
                    continue

                rows.append({
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "name": stock.get("name", ticker),
                    "trigger_type": trigger_type,
                    "entry_price": stock.get("current_price", 0),
                    "change_on_trigger": stock.get("change_rate", 0),
                    "quality_score": stock.get("quality_score", 50),
                    "quality_signal": stock.get("quality_signal", ""),
                    "final_score": stock.get("final_score", 0),
                    "agent_fit_score": stock.get("agent_fit_score", 0),
                    "risk_reward_ratio": stock.get("risk_reward_ratio", 0),
                    "stop_loss_price": stock.get("stop_loss_price", 0),
                    "target_price": stock.get("target_price", 0),
                    "sector": stock.get("metrics", {}).get("sector", ""),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["trade_date_dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    logger.info(f"Extracted {len(df)} total picks across {df['trade_date'].nunique()} dates")
    return df


def fetch_forward_prices(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch forward prices for all picks using yfinance batch download."""
    if picks_df.empty:
        return picks_df

    result = picks_df.copy()

    # Initialize forward return columns
    for d in RETURN_WINDOWS:
        result[f"ret_{d}d"] = np.nan
        result[f"price_{d}d"] = np.nan
    result["mae_pct"] = np.nan  # Max Adverse Excursion (worst drawdown)
    result["mfe_pct"] = np.nan  # Max Favorable Excursion (best run-up)
    result["hit_stop_loss"] = False
    result["hit_target"] = False
    result["days_to_stop"] = np.nan
    result["days_to_target"] = np.nan

    # Simulated exit columns — real ATR-based SL/TP computed from historical data
    # For surge/momentum stocks: wide SL (give room), tight TP (capture the pop)
    MAX_HOLD_DAYS = 30   # max holding period
    ATR_PERIOD = 14      # ATR lookback
    ATR_SL_MULT = 3.0    # stop-loss = entry - 3×ATR₁₄ (wide, ~10%)
    ATR_TP_MULT = 2.0    # take-profit = entry + 2×ATR₁₄ (tight, ~7%)
    FALLBACK_SL_PCT = 0.10    # fallback if ATR can't be computed
    FALLBACK_TP_PCT = 0.08    # fallback if ATR can't be computed
    result["sim_exit_price"] = np.nan
    result["sim_exit_day"] = np.nan
    result["sim_exit_reason"] = ""  # "stop_loss", "take_profit", "time_exit"
    result["sim_return_pct"] = np.nan

    # Group by ticker to minimize API calls
    unique_tickers = result["ticker"].unique()
    earliest = result["trade_date_dt"].min() - timedelta(days=30)  # extra lookback for ATR₁₄
    latest = result["trade_date_dt"].max() + timedelta(days=45)  # extra buffer for 30d forward

    logger.info(f"Fetching prices for {len(unique_tickers)} unique tickers "
                f"from {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")

    # Batch download all tickers at once
    yf_tickers = [f"{t}.NS" for t in unique_tickers]

    # Download in chunks to avoid timeouts
    chunk_size = 50
    all_price_data = {}

    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i + chunk_size]
        logger.info(f"  Downloading chunk {i // chunk_size + 1}/{(len(yf_tickers) + chunk_size - 1) // chunk_size} "
                     f"({len(chunk)} tickers)")
        try:
            data = yf.download(
                chunk,
                start=earliest.strftime("%Y-%m-%d"),
                end=latest.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if not data.empty:
                for yf_t in chunk:
                    nse_ticker = yf_t.replace(".NS", "")
                    try:
                        if len(chunk) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data.xs(yf_t, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
                        if not ticker_data.empty and "Close" in ticker_data.columns:
                            all_price_data[nse_ticker] = ticker_data
                    except (KeyError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"  Chunk download failed: {e}")

    logger.info(f"Got price data for {len(all_price_data)}/{len(unique_tickers)} tickers")

    # Compute forward returns for each pick
    for idx, row in result.iterrows():
        ticker = row["ticker"]
        entry_price = row["entry_price"]
        entry_date = row["trade_date_dt"]
        sl_price = row["stop_loss_price"]
        tp_price = row["target_price"]

        if ticker not in all_price_data or entry_price <= 0:
            continue

        hist = all_price_data[ticker]
        # Get trading days after entry
        future = hist[hist.index > entry_date]
        if future.empty:
            continue

        closes = future["Close"].values
        highs = future["High"].values if "High" in future.columns else closes
        lows = future["Low"].values if "Low" in future.columns else closes

        # Forward returns at each window
        for d in RETURN_WINDOWS:
            if len(closes) >= d:
                fwd_price = closes[d - 1]
                result.at[idx, f"price_{d}d"] = fwd_price
                result.at[idx, f"ret_{d}d"] = (fwd_price / entry_price - 1) * 100

        # MAE and MFE over 30 days (or available data)
        lookforward = min(30, len(closes))
        if lookforward > 0:
            min_low = np.min(lows[:lookforward])
            max_high = np.max(highs[:lookforward])
            result.at[idx, "mae_pct"] = (min_low / entry_price - 1) * 100
            result.at[idx, "mfe_pct"] = (max_high / entry_price - 1) * 100

            # Did it hit stop loss or target?
            if sl_price > 0:
                sl_hits = np.where(lows[:lookforward] <= sl_price)[0]
                if len(sl_hits) > 0:
                    result.at[idx, "hit_stop_loss"] = True
                    result.at[idx, "days_to_stop"] = sl_hits[0] + 1

            if tp_price > 0:
                tp_hits = np.where(highs[:lookforward] >= tp_price)[0]
                if len(tp_hits) > 0:
                    result.at[idx, "hit_target"] = True
                    result.at[idx, "days_to_target"] = tp_hits[0] + 1

            # ── Simulated exit: ATR-based SL/TP → Time exit ──
            # Compute ATR₁₄ from historical data preceding entry
            pre_entry = hist[hist.index <= entry_date].tail(ATR_PERIOD + 1)
            if len(pre_entry) >= 2 and "High" in pre_entry.columns and "Low" in pre_entry.columns:
                hi = pre_entry["High"].values
                lo = pre_entry["Low"].values
                cl = pre_entry["Close"].values
                tr_vals = []
                for ti in range(1, len(pre_entry)):
                    tr_vals.append(max(hi[ti] - lo[ti], abs(hi[ti] - cl[ti-1]), abs(lo[ti] - cl[ti-1])))
                atr = np.mean(tr_vals[-ATR_PERIOD:]) if len(tr_vals) >= ATR_PERIOD else np.mean(tr_vals)
                sim_sl = entry_price - ATR_SL_MULT * atr
                sim_tp = entry_price + ATR_TP_MULT * atr
            else:
                sim_sl = entry_price * (1 - FALLBACK_SL_PCT)
                sim_tp = entry_price * (1 + FALLBACK_TP_PCT)
            sim_days = min(MAX_HOLD_DAYS, lookforward)

            exited = False
            for day_i in range(sim_days):
                # Check stop-loss first (worst case in the day)
                if lows[day_i] <= sim_sl:
                    result.at[idx, "sim_exit_price"] = sim_sl
                    result.at[idx, "sim_exit_day"] = day_i + 1
                    result.at[idx, "sim_exit_reason"] = "stop_loss"
                    result.at[idx, "sim_return_pct"] = (sim_sl / entry_price - 1) * 100
                    exited = True
                    break
                # Check take-profit (best case in the day)
                if highs[day_i] >= sim_tp:
                    result.at[idx, "sim_exit_price"] = sim_tp
                    result.at[idx, "sim_exit_day"] = day_i + 1
                    result.at[idx, "sim_exit_reason"] = "take_profit"
                    result.at[idx, "sim_return_pct"] = (sim_tp / entry_price - 1) * 100
                    exited = True
                    break

            if not exited and sim_days > 0:
                # Time exit: close at end of max hold period
                exit_price = closes[sim_days - 1]
                result.at[idx, "sim_exit_price"] = exit_price
                result.at[idx, "sim_exit_day"] = sim_days
                result.at[idx, "sim_exit_reason"] = "time_exit"
                result.at[idx, "sim_return_pct"] = (exit_price / entry_price - 1) * 100

    filled = result["ret_7d"].notna().sum()
    logger.info(f"Forward returns computed for {filled}/{len(result)} picks")
    return result


def compute_stats(df: pd.DataFrame, label: str = "Overall") -> dict:
    """Compute performance statistics for a set of trades."""
    if df.empty:
        return {"label": label, "n_trades": 0}

    # Use 7-day return as primary metric
    rets = df["ret_7d"].dropna()
    if rets.empty:
        return {"label": label, "n_trades": len(df), "n_with_data": 0}

    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    win_rate = len(wins) / len(rets) * 100 if len(rets) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    expectancy = (len(wins) / len(rets) * avg_win) - (len(losses) / len(rets) * avg_loss) if len(rets) > 0 else 0

    # Profit factor = gross wins / gross losses
    gross_wins = wins.sum() if len(wins) > 0 else 0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # MAE / MFE
    mae = df["mae_pct"].dropna()
    mfe = df["mfe_pct"].dropna()

    stats = {
        "label": label,
        "n_trades": len(df),
        "n_with_data": len(rets),
        "win_rate_7d": round(win_rate, 1),
        "avg_return_3d": round(df["ret_3d"].dropna().mean(), 2) if df["ret_3d"].notna().any() else None,
        "avg_return_7d": round(rets.mean(), 2),
        "avg_return_14d": round(df["ret_14d"].dropna().mean(), 2) if df["ret_14d"].notna().any() else None,
        "avg_return_30d": round(df["ret_30d"].dropna().mean(), 2) if df["ret_30d"].notna().any() else None,
        "median_return_7d": round(rets.median(), 2),
        "avg_win_7d": round(avg_win, 2),
        "avg_loss_7d": round(avg_loss, 2),
        "expectancy_7d": round(expectancy, 2),
        "profit_factor_7d": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "best_trade_7d": round(rets.max(), 2),
        "worst_trade_7d": round(rets.min(), 2),
        "std_return_7d": round(rets.std(), 2),
        # MAE / MFE
        "avg_mae": round(mae.mean(), 2) if not mae.empty else None,
        "avg_mfe": round(mfe.mean(), 2) if not mfe.empty else None,
        "worst_mae": round(mae.min(), 2) if not mae.empty else None,
        "best_mfe": round(mfe.max(), 2) if not mfe.empty else None,
        # Stop loss / target hit rates
        "stop_loss_hit_rate": round(df["hit_stop_loss"].sum() / len(df) * 100, 1),
        "target_hit_rate": round(df["hit_target"].sum() / len(df) * 100, 1),
        "avg_days_to_stop": round(df.loc[df["hit_stop_loss"], "days_to_stop"].mean(), 1) if df["hit_stop_loss"].any() else None,
        "avg_days_to_target": round(df.loc[df["hit_target"], "days_to_target"].mean(), 1) if df["hit_target"].any() else None,
    }

    # Sharpe-like ratio (using 7d returns, annualized)
    if rets.std() > 0:
        # ~52 weekly periods per year (7d ≈ 1 week)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(52)
        stats["sharpe_annualized"] = round(sharpe, 2)

        # Sortino (only downside deviation)
        downside = rets[rets < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (rets.mean() / downside.std()) * np.sqrt(52)
            stats["sortino_annualized"] = round(sortino, 2)

    # Simulated exit stats
    sim_rets = df["sim_return_pct"].dropna()
    if not sim_rets.empty:
        sim_wins = sim_rets[sim_rets > 0]
        sim_losses = sim_rets[sim_rets <= 0]
        sim_wr = len(sim_wins) / len(sim_rets) * 100
        sim_avg = sim_rets.mean()
        sim_gross_win = sim_wins.sum() if len(sim_wins) > 0 else 0
        sim_gross_loss = abs(sim_losses.sum()) if len(sim_losses) > 0 else 0
        sim_pf = sim_gross_win / sim_gross_loss if sim_gross_loss > 0 else float("inf")

        # Exit reason breakdown
        reasons = df["sim_exit_reason"].value_counts()

        stats["sim_n_trades"] = len(sim_rets)
        stats["sim_win_rate"] = round(sim_wr, 1)
        stats["sim_avg_return"] = round(sim_avg, 2)
        stats["sim_profit_factor"] = round(sim_pf, 2) if sim_pf != float("inf") else "inf"
        stats["sim_stopped_out"] = int(reasons.get("stop_loss", 0))
        stats["sim_target_hit"] = int(reasons.get("take_profit", 0))
        stats["sim_time_exit"] = int(reasons.get("time_exit", 0))
        stats["sim_avg_hold_days"] = round(df["sim_exit_day"].dropna().mean(), 1)

    return stats


def print_report(picks_df: pd.DataFrame):
    """Print a comprehensive backtest report."""
    print("\n" + "=" * 70)
    print("PRISM INDIA — BACKTEST REPORT")
    print(f"Period: {picks_df['trade_date'].min()} to {picks_df['trade_date'].max()}")
    print(f"Total picks: {len(picks_df)} across {picks_df['trade_date'].nunique()} trading days")
    print("=" * 70)

    # Overall stats
    overall = compute_stats(picks_df, "OVERALL")
    print(f"\n{'─' * 50}")
    print(f"OVERALL PERFORMANCE (7-day hold)")
    print(f"{'─' * 50}")
    print(f"  Trades:          {overall['n_trades']} ({overall.get('n_with_data', 0)} with price data)")
    print(f"  Win Rate:        {overall.get('win_rate_7d', 'N/A')}%")
    print(f"  Avg Return:      3d={overall.get('avg_return_3d', 'N/A')}% | 7d={overall.get('avg_return_7d', 'N/A')}% | 14d={overall.get('avg_return_14d', 'N/A')}% | 30d={overall.get('avg_return_30d', 'N/A')}%")
    print(f"  Avg Win:         +{overall.get('avg_win_7d', 'N/A')}%")
    print(f"  Avg Loss:        -{overall.get('avg_loss_7d', 'N/A')}%")
    print(f"  Expectancy:      {overall.get('expectancy_7d', 'N/A')}% per trade")
    print(f"  Profit Factor:   {overall.get('profit_factor_7d', 'N/A')}")
    print(f"  Best/Worst:      +{overall.get('best_trade_7d', 'N/A')}% / {overall.get('worst_trade_7d', 'N/A')}%")
    print(f"  Sharpe (ann):    {overall.get('sharpe_annualized', 'N/A')}")
    print(f"  Sortino (ann):   {overall.get('sortino_annualized', 'N/A')}")
    print(f"  Avg MAE:         {overall.get('avg_mae', 'N/A')}% (worst: {overall.get('worst_mae', 'N/A')}%)")
    print(f"  Avg MFE:         +{overall.get('avg_mfe', 'N/A')}% (best: +{overall.get('best_mfe', 'N/A')}%)")
    print(f"  SL Hit Rate:     {overall.get('stop_loss_hit_rate', 'N/A')}% (avg {overall.get('avg_days_to_stop', 'N/A')} days)")
    print(f"  Target Hit Rate: {overall.get('target_hit_rate', 'N/A')}% (avg {overall.get('avg_days_to_target', 'N/A')} days)")

    # Simulated exit performance
    if overall.get("sim_n_trades"):
        print(f"\n{'─' * 50}")
        print(f"SIMULATED TRADING (ATR-based SL/TP, 21-day max hold)")
        print(f"{'─' * 50}")
        print(f"  Trades:          {overall['sim_n_trades']}")
        print(f"  Win Rate:        {overall['sim_win_rate']}%")
        print(f"  Avg Return:      {overall['sim_avg_return']:+.2f}% per trade")
        print(f"  Profit Factor:   {overall['sim_profit_factor']}")
        print(f"  Avg Hold:        {overall['sim_avg_hold_days']} days")
        print(f"  Exit Breakdown:")
        print(f"    Stop-loss:     {overall['sim_stopped_out']} ({overall['sim_stopped_out']/overall['sim_n_trades']*100:.0f}%)")
        print(f"    Take-profit:   {overall['sim_target_hit']} ({overall['sim_target_hit']/overall['sim_n_trades']*100:.0f}%)")
        print(f"    Time exit:     {overall['sim_time_exit']} ({overall['sim_time_exit']/overall['sim_n_trades']*100:.0f}%)")

        # Estimated P&L on ₹10L capital, quality-weighted allocation
        # Top-heavy: 40% to highest quality pick, rest split equally
        capital = 1_000_000
        per_trade_equal = capital / 5  # fallback equal weight

        # Compute per-trade allocation weighted by quality
        def compute_quality_weights(group):
            scores = group["quality_score"].fillna(50).clip(lower=10).values
            n = len(scores)
            if n <= 1:
                return pd.Series([1.0] * n, index=group.index)
            # Top-heavy: best quality gets 40%, rest split equally
            sorted_idx = np.argsort(-scores)  # descending
            weights = np.full(n, 0.6 / (n - 1)) if n > 1 else np.array([1.0])
            weights = np.full(n, (1.0 - 0.40) / (n - 1))
            weights[sorted_idx[0]] = 0.40
            return pd.Series(weights, index=group.index)

        picks_df["alloc_weight"] = picks_df.groupby("trade_date", group_keys=False).apply(compute_quality_weights)
        picks_df["alloc_capital"] = picks_df["alloc_weight"] * capital

        total_pnl_weighted = sum(
            (picks_df["sim_return_pct"].dropna() / 100 * picks_df.loc[picks_df["sim_return_pct"].notna(), "alloc_capital"])
        )
        total_pnl_equal = sum(picks_df["sim_return_pct"].dropna() / 100 * per_trade_equal)

        print(f"\n  Estimated P&L (₹10L capital):")
        print(f"    Equal allocation (₹2L each):      ₹{total_pnl_equal:+,.0f} ({total_pnl_equal/capital*100:+.1f}%)")
        print(f"    Quality-weighted (top-heavy 40%):  ₹{total_pnl_weighted:+,.0f} ({total_pnl_weighted/capital*100:+.1f}%)")
        print(f"    Improvement:                       ₹{total_pnl_weighted - total_pnl_equal:+,.0f} ({(total_pnl_weighted/total_pnl_equal - 1)*100:+.1f}%)" if total_pnl_equal != 0 else "")

    # ── Equity curve, drawdown, streaks ──
    sim_rets = picks_df[["trade_date", "sim_return_pct", "sector", "alloc_capital"]].dropna(subset=["sim_return_pct"]).copy()
    if not sim_rets.empty:
        print(f"\n{'─' * 50}")
        print(f"EQUITY CURVE & DRAWDOWN")
        print(f"{'─' * 50}")

        # Build daily P&L using quality-weighted allocation
        capital = 1_000_000
        daily_pnl = sim_rets.groupby("trade_date").apply(
            lambda x: (x["sim_return_pct"] / 100 * x["alloc_capital"]).sum()
        ).sort_index()

        # Cumulative equity
        equity = capital + daily_pnl.cumsum()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        peak_at_dd = peak.loc[max_dd_date]
        equity_at_dd = equity.loc[max_dd_date]

        # Recovery: first date equity exceeds previous peak after max drawdown
        post_dd = equity.loc[max_dd_date:]
        recovery_dates = post_dd[post_dd >= peak_at_dd]
        if len(recovery_dates) > 1:
            recovery_date = recovery_dates.index[1]
            recovery_days = len(daily_pnl.loc[max_dd_date:recovery_date])
        else:
            recovery_date = "Not recovered"
            recovery_days = "N/A"

        # CAGR
        n_days = (pd.Timestamp(sim_rets["trade_date"].max()) - pd.Timestamp(sim_rets["trade_date"].min())).days
        if n_days > 0:
            final_equity = equity.iloc[-1]
            cagr = ((final_equity / capital) ** (365 / n_days) - 1) * 100
        else:
            cagr = 0

        # Calmar ratio = CAGR / |Max Drawdown|
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        print(f"  Starting Capital: ₹{capital:,.0f}")
        print(f"  Final Equity:     ₹{equity.iloc[-1]:,.0f}")
        print(f"  Total Return:     {(equity.iloc[-1]/capital - 1)*100:+.1f}%")
        print(f"  CAGR:             {cagr:+.1f}%")
        print(f"  Max Drawdown:     {max_dd:.1f}% (on {max_dd_date})")
        print(f"  Drawdown Depth:   ₹{peak_at_dd:,.0f} → ₹{equity_at_dd:,.0f}")
        print(f"  Recovery:         {recovery_days} trading days")
        print(f"  Calmar Ratio:     {calmar:.2f} (CAGR/MaxDD, target: >1.0)")

        # Consecutive wins/losses
        print(f"\n{'─' * 50}")
        print(f"STREAK ANALYSIS")
        print(f"{'─' * 50}")

        # Group by date, check if day was net positive
        daily_results = daily_pnl.apply(lambda x: "W" if x > 0 else "L")

        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        current_type = None

        for result in daily_results:
            if result == current_type:
                current_streak += 1
            else:
                current_type = result
                current_streak = 1
            if result == "W":
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)

        winning_days = (daily_pnl > 0).sum()
        losing_days = (daily_pnl <= 0).sum()
        total_days = len(daily_pnl)

        print(f"  Winning days:    {winning_days}/{total_days} ({winning_days/total_days*100:.0f}%)")
        print(f"  Losing days:     {losing_days}/{total_days} ({losing_days/total_days*100:.0f}%)")
        print(f"  Max win streak:  {max_win_streak} consecutive days")
        print(f"  Max loss streak: {max_loss_streak} consecutive days")
        print(f"  Best day P&L:    ₹{daily_pnl.max():+,.0f}")
        print(f"  Worst day P&L:   ₹{daily_pnl.min():+,.0f}")

        # Sector concentration risk
        print(f"\n{'─' * 50}")
        print(f"SECTOR RISK (simulated trades)")
        print(f"{'─' * 50}")
        if "sector" in sim_rets.columns and sim_rets["sector"].notna().any():
            sector_stats = sim_rets.groupby("sector")["sim_return_pct"].agg(
                count="count", avg_ret="mean", total_ret="sum", win_rate=lambda x: (x > 0).mean() * 100
            ).sort_values("count", ascending=False)
            for sector, row in sector_stats.head(8).iterrows():
                if not sector:
                    continue
                emoji = "🟢" if row["avg_ret"] > 0 else "🔴"
                print(f"  {emoji} {sector:25s}  n={int(row['count']):3d}  "
                      f"WR={row['win_rate']:.0f}%  Avg={row['avg_ret']:+.2f}%  "
                      f"Total={row['total_ret']:+.1f}%")

    # Per trigger type
    print(f"\n{'─' * 50}")
    print(f"BY TRIGGER TYPE")
    print(f"{'─' * 50}")
    for trigger_type in sorted(picks_df["trigger_type"].unique()):
        subset = picks_df[picks_df["trigger_type"] == trigger_type]
        s = compute_stats(subset, trigger_type)
        wr = s.get("win_rate_7d", 0)
        ar = s.get("avg_return_7d", 0)
        exp = s.get("expectancy_7d", 0)
        pf = s.get("profit_factor_7d", 0)
        n = s.get("n_with_data", 0)
        print(f"  {trigger_type:30s}  n={n:3d}  WR={wr:5.1f}%  Avg={ar:+6.2f}%  E={exp:+6.2f}%  PF={pf}")

    # Per date
    print(f"\n{'─' * 50}")
    print(f"BY DATE (7-day return)")
    print(f"{'─' * 50}")
    for date in sorted(picks_df["trade_date"].unique()):
        subset = picks_df[picks_df["trade_date"] == date]
        rets = subset["ret_7d"].dropna()
        if rets.empty:
            print(f"  {date}  n={len(subset):2d}  (no forward data yet)")
            continue
        wins = (rets > 0).sum()
        avg = rets.mean()
        tickers = ", ".join(subset["ticker"].tolist())
        emoji = "🟢" if avg > 0 else "🔴"
        print(f"  {date}  n={len(rets):2d}  {emoji} avg={avg:+6.2f}%  W={wins}/{len(rets)}  [{tickers}]")

    # Sector concentration
    if "sector" in picks_df.columns and picks_df["sector"].notna().any():
        print(f"\n{'─' * 50}")
        print(f"SECTOR CONCENTRATION")
        print(f"{'─' * 50}")
        sector_counts = picks_df["sector"].value_counts().head(10)
        for sector, count in sector_counts.items():
            pct = count / len(picks_df) * 100
            bar = "█" * int(pct / 2)
            print(f"  {sector:30s}  {count:3d} ({pct:4.1f}%) {bar}")

    # Individual trade detail
    print(f"\n{'─' * 50}")
    print(f"TOP 10 BEST TRADES (7-day)")
    print(f"{'─' * 50}")
    best = picks_df.nlargest(10, "ret_7d", "first")
    for _, row in best.iterrows():
        if pd.notna(row["ret_7d"]):
            print(f"  {row['trade_date']} {row['ticker']:12s} +{row['ret_7d']:5.1f}%  "
                  f"(entry ₹{row['entry_price']:,.1f}, trigger: {row['change_on_trigger']:+.1f}%)  "
                  f"via {row['trigger_type']}")

    print(f"\n{'─' * 50}")
    print(f"TOP 10 WORST TRADES (7-day)")
    print(f"{'─' * 50}")
    worst = picks_df.nsmallest(10, "ret_7d", "first")
    for _, row in worst.iterrows():
        if pd.notna(row["ret_7d"]):
            print(f"  {row['trade_date']} {row['ticker']:12s} {row['ret_7d']:+5.1f}%  "
                  f"(entry ₹{row['entry_price']:,.1f}, trigger: {row['change_on_trigger']:+.1f}%)  "
                  f"via {row['trigger_type']}")

    # MAE/MFE insight
    mae = picks_df["mae_pct"].dropna()
    mfe = picks_df["mfe_pct"].dropna()
    if not mae.empty and not mfe.empty:
        print(f"\n{'─' * 50}")
        print(f"MAE/MFE ANALYSIS (30-day window)")
        print(f"{'─' * 50}")
        print(f"  If you used a 5% stop-loss:")
        sl5_hit = (mae <= -5).sum()
        print(f"    Stopped out: {sl5_hit}/{len(mae)} trades ({sl5_hit/len(mae)*100:.0f}%)")
        print(f"  If you used a 3% stop-loss:")
        sl3_hit = (mae <= -3).sum()
        print(f"    Stopped out: {sl3_hit}/{len(mae)} trades ({sl3_hit/len(mae)*100:.0f}%)")
        print(f"  Typical MFE (how far stocks run in your favor):")
        for pctl in [25, 50, 75, 90]:
            print(f"    P{pctl}: +{np.percentile(mfe, pctl):.1f}%")
        print(f"  → If your stocks typically run +{np.percentile(mfe, 50):.0f}% before reversing,")
        print(f"    taking profit at +{np.percentile(mfe, 50) * 0.7:.0f}% captures ~70% of the move.")


def main():
    parser = argparse.ArgumentParser(description="PRISM India Backtest Engine")
    parser.add_argument("--min-date", type=str, default=None, help="Minimum date (YYYYMMDD)")
    parser.add_argument("--max-date", type=str, default=None, help="Maximum date (YYYYMMDD)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory to read trigger JSONs from (default: project root)")
    args = parser.parse_args()

    # Load trigger files
    trigger_files = load_trigger_files(min_date=args.min_date, max_date=args.max_date, input_dir=args.input_dir)
    if not trigger_files:
        print("No trigger result files found.")
        return

    # Extract picks
    picks_df = extract_picks(trigger_files)
    if picks_df.empty:
        print("No picks extracted from trigger files.")
        return

    # Fetch forward prices
    picks_df = fetch_forward_prices(picks_df)

    # Print report
    print_report(picks_df)

    # Save results
    if args.output:
        # Compute all stats
        output = {
            "run_time": datetime.now().isoformat(),
            "period": f"{picks_df['trade_date'].min()} to {picks_df['trade_date'].max()}",
            "total_picks": len(picks_df),
            "unique_dates": int(picks_df["trade_date"].nunique()),
            "overall": compute_stats(picks_df, "overall"),
            "by_trigger": {},
            "by_date": {},
            "trades": [],
        }
        for tt in picks_df["trigger_type"].unique():
            output["by_trigger"][tt] = compute_stats(picks_df[picks_df["trigger_type"] == tt], tt)
        for date in sorted(picks_df["trade_date"].unique()):
            output["by_date"][date] = compute_stats(picks_df[picks_df["trade_date"] == date], date)

        # Add equity curve stats to output
        sim_rets = picks_df[["trade_date", "sim_return_pct", "alloc_capital"]].dropna(subset=["sim_return_pct"]).copy()
        if not sim_rets.empty:
            eq_capital = 1_000_000
            daily_pnl = sim_rets.groupby("trade_date").apply(
                lambda x: (x["sim_return_pct"] / 100 * x["alloc_capital"]).sum(),
                include_groups=False,
            ).sort_index()
            equity = eq_capital + daily_pnl.cumsum()
            peak = equity.cummax()
            drawdown = (equity - peak) / peak * 100
            max_dd = drawdown.min()
            final_eq = equity.iloc[-1]
            n_days = (pd.Timestamp(sim_rets["trade_date"].max()) - pd.Timestamp(sim_rets["trade_date"].min())).days
            cagr = ((final_eq / eq_capital) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            output["overall"]["sim_final_equity"] = round(final_eq, 0)
            output["overall"]["sim_total_return_pct"] = round((final_eq / eq_capital - 1) * 100, 1)
            output["overall"]["sim_cagr_pct"] = round(cagr, 1)
            output["overall"]["sim_max_drawdown_pct"] = round(max_dd, 1)
            output["overall"]["sim_calmar"] = round(calmar, 2)

        for _, row in picks_df.iterrows():
            trade = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                     for k, v in row.items() if k != "trade_date_dt"}
            if isinstance(trade.get("hit_stop_loss"), (np.bool_,)):
                trade["hit_stop_loss"] = bool(trade["hit_stop_loss"])
            if isinstance(trade.get("hit_target"), (np.bool_,)):
                trade["hit_target"] = bool(trade["hit_target"])
            output["trades"].append(trade)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
