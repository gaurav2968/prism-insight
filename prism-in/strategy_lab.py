#!/usr/bin/env python3
"""
Strategy Lab — Compare exit strategies, position sizing, and regime filters.

Reuses data loading from backtest_engine but runs multiple strategy configs
through the same price data to find what works best.

Usage:
    python prism-in/strategy_lab.py --min-date 20240101 --max-date 20240925
    python prism-in/strategy_lab.py --input-dir trigger_results_v2 --min-date 20240101 --max-date 20240925
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine import load_trigger_files, extract_picks, RETURN_WINDOWS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ══════════════════════════════════════════════════════════════
# TRANSACTION COSTS (Indian delivery trades, Zerodha-style)
# ══════════════════════════════════════════════════════════════

@dataclass
class TxnCosts:
    """Indian equity delivery transaction costs."""
    brokerage_pct: float = 0.0003    # 0.03% or ₹20 flat (we use %)
    stt_sell_pct: float = 0.001      # 0.1% STT on sell side
    stamp_buy_pct: float = 0.00015   # 0.015% stamp duty on buy
    exchange_pct: float = 0.0000345  # exchange txn charges
    sebi_pct: float = 0.000001       # SEBI turnover fee
    gst_on_brokerage: float = 0.18   # 18% GST on brokerage

    def round_trip_pct(self) -> float:
        """Total cost for buy + sell as % of trade value."""
        buy_cost = self.brokerage_pct + self.stamp_buy_pct + self.exchange_pct + self.sebi_pct
        sell_cost = self.brokerage_pct + self.stt_sell_pct + self.exchange_pct + self.sebi_pct
        gst = (self.brokerage_pct * 2) * self.gst_on_brokerage
        return buy_cost + sell_cost + gst

    def __str__(self):
        return f"{self.round_trip_pct()*100:.3f}% round-trip"


DEFAULT_COSTS = TxnCosts()


# ══════════════════════════════════════════════════════════════
# EXIT STRATEGIES
# ══════════════════════════════════════════════════════════════

@dataclass
class ExitStrategy:
    """Configuration for a simulated exit strategy."""
    name: str
    # ATR-based exits
    atr_period: int = 14
    atr_sl_mult: float = 3.0       # SL = entry - N×ATR
    atr_tp_mult: float = 2.0       # TP = entry + N×ATR
    # Fallback if ATR unavailable
    fallback_sl_pct: float = 0.10
    fallback_tp_pct: float = 0.08
    # Max hold
    max_hold_days: int = 30
    # Trailing stop (0 = disabled)
    trail_activate_atr: float = 0   # activate trailing after price moves N×ATR above entry
    trail_offset_atr: float = 0     # trail stop follows at N×ATR below highest price
    # Transaction costs
    include_costs: bool = True


# Pre-defined strategies to compare
STRATEGIES = {
    "baseline_fixed_10_8": ExitStrategy(
        name="Fixed 10%SL / 8%TP / 30d",
        atr_sl_mult=0, atr_tp_mult=0,  # disable ATR
        fallback_sl_pct=0.10, fallback_tp_pct=0.08,
        max_hold_days=30,
    ),
    "atr_3sl_2tp": ExitStrategy(
        name="ATR 3×SL / 2×TP / 30d",
        atr_sl_mult=3.0, atr_tp_mult=2.0,
        max_hold_days=30,
    ),
    "atr_2sl_2tp": ExitStrategy(
        name="ATR 2×SL / 2×TP / 30d",
        atr_sl_mult=2.0, atr_tp_mult=2.0,
        max_hold_days=30,
    ),
    "atr_3sl_3tp": ExitStrategy(
        name="ATR 3×SL / 3×TP / 30d",
        atr_sl_mult=3.0, atr_tp_mult=3.0,
        max_hold_days=30,
    ),
    "atr_3sl_2tp_trail": ExitStrategy(
        name="ATR 3×SL / 2×TP + Trail / 30d",
        atr_sl_mult=3.0, atr_tp_mult=0,  # no fixed TP — let trail do the work
        trail_activate_atr=1.0,            # activate after +1×ATR
        trail_offset_atr=2.0,              # trail at 2×ATR below high
        max_hold_days=30,
    ),
    "atr_3sl_trail_only": ExitStrategy(
        name="ATR 3×SL / Trail only / 30d",
        atr_sl_mult=3.0, atr_tp_mult=0,  # no fixed TP
        trail_activate_atr=1.5,
        trail_offset_atr=2.0,
        max_hold_days=30,
    ),
    "atr_3sl_2tp_21d": ExitStrategy(
        name="ATR 3×SL / 2×TP / 21d",
        atr_sl_mult=3.0, atr_tp_mult=2.0,
        max_hold_days=21,
    ),
    "tight_atr_2sl_1.5tp": ExitStrategy(
        name="ATR 2×SL / 1.5×TP / 21d",
        atr_sl_mult=2.0, atr_tp_mult=1.5,
        max_hold_days=21,
    ),
}


# ══════════════════════════════════════════════════════════════
# REGIME-AWARE SIZING
# ══════════════════════════════════════════════════════════════

@dataclass
class RegimeSizing:
    """Position sizing based on market regime."""
    name: str
    bull_alloc: float = 1.0       # fraction of capital to deploy in bull
    sideways_alloc: float = 1.0   # fraction in sideways
    correction_alloc: float = 1.0  # fraction in correction

    def get_alloc(self, regime: str) -> float:
        if regime == "bull":
            return self.bull_alloc
        elif regime == "correction":
            return self.correction_alloc
        return self.sideways_alloc


SIZING_MODES = {
    "equal": RegimeSizing(name="Equal (always 100%)"),
    "regime_conservative": RegimeSizing(
        name="Regime Conservative",
        bull_alloc=1.0, sideways_alloc=0.6, correction_alloc=0.2,
    ),
    "regime_moderate": RegimeSizing(
        name="Regime Moderate",
        bull_alloc=1.0, sideways_alloc=0.8, correction_alloc=0.5,
    ),
    "skip_correction": RegimeSizing(
        name="Skip Corrections",
        bull_alloc=1.0, sideways_alloc=1.0, correction_alloc=0.0,
    ),
}


# ══════════════════════════════════════════════════════════════
# PRICE DATA FETCHER (shared across strategies)
# ══════════════════════════════════════════════════════════════

def fetch_price_data(picks_df: pd.DataFrame) -> dict:
    """Download price data for all tickers. Returns {ticker: DataFrame}."""
    unique_tickers = picks_df["ticker"].unique()
    earliest = picks_df["trade_date_dt"].min() - timedelta(days=30)  # ATR lookback
    latest = picks_df["trade_date_dt"].max() + timedelta(days=45)

    yf_tickers = [f"{t}.NS" for t in unique_tickers]
    all_data = {}
    chunk_size = 50

    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i + chunk_size]
        logger.info(f"  Downloading chunk {i // chunk_size + 1}/{(len(yf_tickers) + chunk_size - 1) // chunk_size}")
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
                            all_data[nse_ticker] = ticker_data
                    except (KeyError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"  Download failed: {e}")

    logger.info(f"Got price data for {len(all_data)}/{len(unique_tickers)} tickers")
    return all_data


def compute_atr(hist: pd.DataFrame, entry_date, atr_period: int = 14) -> float:
    """Compute ATR from historical data preceding entry date."""
    pre_entry = hist[hist.index <= entry_date].tail(atr_period + 1)
    if len(pre_entry) < 2 or "High" not in pre_entry.columns:
        return 0
    hi = pre_entry["High"].values
    lo = pre_entry["Low"].values
    cl = pre_entry["Close"].values
    tr_vals = []
    for ti in range(1, len(pre_entry)):
        tr_vals.append(max(hi[ti] - lo[ti], abs(hi[ti] - cl[ti - 1]), abs(lo[ti] - cl[ti - 1])))
    if not tr_vals:
        return 0
    return float(np.mean(tr_vals[-atr_period:]) if len(tr_vals) >= atr_period else np.mean(tr_vals))


# ══════════════════════════════════════════════════════════════
# STRATEGY SIMULATOR
# ══════════════════════════════════════════════════════════════

def simulate_strategy(
    picks_df: pd.DataFrame,
    price_data: dict,
    strategy: ExitStrategy,
    sizing: RegimeSizing,
    costs: TxnCosts = DEFAULT_COSTS,
    capital: float = 1_000_000,
) -> dict:
    """
    Run a single strategy through all picks and return performance metrics.
    """
    cost_pct = costs.round_trip_pct() if strategy.include_costs else 0

    trades = []

    for _, row in picks_df.iterrows():
        ticker = row["ticker"]
        entry_price = row["entry_price"]
        entry_date = row["trade_date_dt"]
        regime = row.get("regime", "unknown")

        if ticker not in price_data or entry_price <= 0:
            continue

        hist = price_data[ticker]
        future = hist[hist.index > entry_date]
        if future.empty:
            continue

        closes = future["Close"].values
        highs = future["High"].values if "High" in future.columns else closes
        lows = future["Low"].values if "Low" in future.columns else closes
        lookforward = min(strategy.max_hold_days, len(closes))
        if lookforward == 0:
            continue

        # Compute ATR
        atr = compute_atr(hist, entry_date, strategy.atr_period)

        # Determine SL/TP levels
        if atr > 0 and strategy.atr_sl_mult > 0:
            sim_sl = entry_price - strategy.atr_sl_mult * atr
        else:
            sim_sl = entry_price * (1 - strategy.fallback_sl_pct)

        if atr > 0 and strategy.atr_tp_mult > 0:
            sim_tp = entry_price + strategy.atr_tp_mult * atr
        else:
            sim_tp = 0  # no fixed TP (trail-only or disabled)

        # Trailing stop state
        use_trail = strategy.trail_activate_atr > 0 and atr > 0
        trail_active = False
        trail_stop = 0
        trail_activate_price = entry_price + strategy.trail_activate_atr * atr if use_trail else 0
        highest_price = entry_price

        # Regime-based allocation
        alloc_fraction = sizing.get_alloc(regime)
        if alloc_fraction <= 0:
            continue  # skip this trade entirely

        # Simulate day by day
        exit_price = None
        exit_day = 0
        exit_reason = ""

        for day_i in range(lookforward):
            day_high = highs[day_i]
            day_low = lows[day_i]
            day_close = closes[day_i]

            # Update highest price for trailing stop
            if day_high > highest_price:
                highest_price = day_high

            # Activate trailing stop if price reached activation threshold
            if use_trail and not trail_active and highest_price >= trail_activate_price:
                trail_active = True
                trail_stop = highest_price - strategy.trail_offset_atr * atr

            # Update trailing stop level
            if trail_active:
                new_trail = highest_price - strategy.trail_offset_atr * atr
                if new_trail > trail_stop:
                    trail_stop = new_trail

            # Check stop-loss (fixed ATR SL)
            if day_low <= sim_sl:
                exit_price = sim_sl
                exit_day = day_i + 1
                exit_reason = "stop_loss"
                break

            # Check trailing stop (if active, takes priority over fixed TP)
            if trail_active and day_low <= trail_stop:
                exit_price = trail_stop
                exit_day = day_i + 1
                exit_reason = "trail_stop"
                break

            # Check fixed take-profit (only if set)
            if sim_tp > 0 and day_high >= sim_tp:
                exit_price = sim_tp
                exit_day = day_i + 1
                exit_reason = "take_profit"
                break

        # Time exit
        if exit_price is None and lookforward > 0:
            exit_price = closes[lookforward - 1]
            exit_day = lookforward
            exit_reason = "time_exit"

        if exit_price is None:
            continue

        gross_return_pct = (exit_price / entry_price - 1) * 100
        net_return_pct = gross_return_pct - (cost_pct * 100)

        trades.append({
            "ticker": ticker,
            "trade_date": row["trade_date"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_day": exit_day,
            "exit_reason": exit_reason,
            "gross_return_pct": gross_return_pct,
            "net_return_pct": net_return_pct,
            "regime": regime,
            "alloc_fraction": alloc_fraction,
            "trigger_type": row.get("trigger_type", ""),
            "atr": atr,
            "sl_pct": (entry_price - sim_sl) / entry_price * 100 if sim_sl > 0 else 0,
            "tp_pct": (sim_tp - entry_price) / entry_price * 100 if sim_tp > 0 else 0,
        })

    if not trades:
        return {"strategy": strategy.name, "sizing": sizing.name, "n_trades": 0}

    df = pd.DataFrame(trades)

    # ── Equity curve (quality-weighted with regime sizing) ──
    # Per-day: allocate capital equally across that day's picks, scaled by regime
    daily_groups = df.groupby("trade_date")
    daily_pnls = []
    for date, group in daily_groups:
        n = len(group)
        per_stock = capital / max(n, 5)  # max 5 slots, same as live
        day_pnl = 0
        for _, t in group.iterrows():
            alloc = per_stock * t["alloc_fraction"]
            day_pnl += alloc * t["net_return_pct"] / 100
        daily_pnls.append({"date": date, "pnl": day_pnl})

    pnl_df = pd.DataFrame(daily_pnls).sort_values("date")
    equity = capital + pnl_df["pnl"].cumsum()
    final_equity = equity.iloc[-1]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()

    n_days = (pd.Timestamp(pnl_df["date"].max()) - pd.Timestamp(pnl_df["date"].min())).days
    if n_days > 0 and final_equity > 0:
        cagr = ((final_equity / capital) ** (365 / n_days) - 1) * 100
    else:
        cagr = -100.0  # total wipeout
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade-level stats
    wins = df[df["net_return_pct"] > 0]
    losses = df[df["net_return_pct"] <= 0]
    win_rate = len(wins) / len(df) * 100
    avg_return = df["net_return_pct"].mean()
    gross_wins = wins["net_return_pct"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["net_return_pct"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Exit breakdown
    reasons = df["exit_reason"].value_counts().to_dict()

    # Per trigger type
    by_trigger = {}
    for tt in df["trigger_type"].unique():
        tdf = df[df["trigger_type"] == tt]
        tw = tdf[tdf["net_return_pct"] > 0]
        by_trigger[tt] = {
            "n": len(tdf),
            "win_rate": round(len(tw) / len(tdf) * 100, 1) if len(tdf) > 0 else 0,
            "avg_return": round(tdf["net_return_pct"].mean(), 2),
            "profit_factor": round(
                tw["net_return_pct"].sum() / abs(tdf[tdf["net_return_pct"] <= 0]["net_return_pct"].sum())
                if tdf[tdf["net_return_pct"] <= 0]["net_return_pct"].sum() != 0 else float("inf"),
                2
            ),
        }

    # Winning/losing days
    winning_days = (pnl_df["pnl"] > 0).sum()

    return {
        "strategy": strategy.name,
        "sizing": sizing.name,
        "costs": f"{cost_pct*100:.3f}%",
        "n_trades": len(df),
        "win_rate": round(win_rate, 1),
        "avg_return_gross": round(df["gross_return_pct"].mean(), 2),
        "avg_return_net": round(avg_return, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "avg_hold_days": round(df["exit_day"].mean(), 1),
        "exit_breakdown": reasons,
        "final_equity": round(final_equity, 0),
        "total_return_pct": round((final_equity / capital - 1) * 100, 1),
        "cagr_pct": round(cagr, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "calmar": round(calmar, 2),
        "winning_days_pct": round(winning_days / len(pnl_df) * 100, 1),
        "avg_sl_pct": round(df["sl_pct"].mean(), 1),
        "avg_tp_pct": round(df[df["tp_pct"] > 0]["tp_pct"].mean(), 1) if (df["tp_pct"] > 0).any() else 0,
        "by_trigger": by_trigger,
    }


# ══════════════════════════════════════════════════════════════
# NIFTY REGIME COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_nifty_regimes(start_date, end_date) -> dict:
    """
    Compute daily market regime from Nifty 50 price data.
    Returns {date_str: regime} mapping.
    Matches the logic in in_trigger_batch.py.
    """
    nifty_start = (start_date - timedelta(days=400)).strftime("%Y-%m-%d")
    nifty_end = (end_date + timedelta(days=5)).strftime("%Y-%m-%d")

    nifty = yf.Ticker("^NSEI")
    hist = nifty.history(start=nifty_start, end=nifty_end)
    if hist.empty or len(hist) < 200:
        logger.warning("Insufficient Nifty data for regime computation")
        return {}

    closes = hist["Close"]
    regimes = {}

    for i in range(200, len(closes)):
        date = closes.index[i]
        date_str = date.strftime("%Y%m%d")

        price = float(closes.iloc[i])
        sma50 = float(closes.iloc[i - 50:i].mean())
        sma200 = float(closes.iloc[i - 200:i].mean())

        # 50MA slope
        if i >= 70:
            sma50_20ago = float(closes.iloc[i - 70:i - 20].mean())
            slope_pct = (sma50 / sma50_20ago - 1) * 100
        else:
            slope_pct = 1.0

        # Realized volatility
        if i >= 21:
            rets = np.diff(np.log(closes.iloc[i - 21:i + 1].values))
            rvol = float(np.std(rets) * np.sqrt(252) * 100)
        else:
            rvol = 15.0

        # RSI
        deltas = np.diff(closes.iloc[:i + 1].values)
        gains = np.where(deltas > 0, deltas, 0)
        losses_arr = np.where(deltas < 0, -deltas, 0)
        if len(gains) >= 14:
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(losses_arr[:14])
            for ii in range(14, len(gains)):
                avg_gain = (avg_gain * 13 + gains[ii]) / 14
                avg_loss = (avg_loss * 13 + losses_arr[ii]) / 14
            rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0
        else:
            rsi = 50.0

        below_50 = price < sma50
        below_200 = price < sma200

        if below_50 and below_200:
            if rsi <= 25:
                regime = "bear_bottom"
            elif rsi <= 40:
                regime = "bear_bottom"
            else:
                regime = "bear"
        elif below_50 or below_200:
            regime = "correction"
        else:
            strong_slope = slope_pct >= 1.0
            low_vol = rvol <= 13.0
            if strong_slope and low_vol:
                regime = "bull"
            elif strong_slope or low_vol:
                regime = "bull"
            else:
                regime = "correction"  # weak bull ≈ correction for sizing

        regimes[date_str] = regime

    logger.info(f"Computed regimes for {len(regimes)} dates")
    regime_counts = {}
    for r in regimes.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    logger.info(f"  Regime distribution: {regime_counts}")

    return regimes


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

def print_comparison(results: list[dict]):
    """Print a side-by-side comparison table of all strategy results."""

    print("\n" + "=" * 120)
    print("STRATEGY LAB — COMPARISON")
    print("=" * 120)

    # Header
    headers = ["Metric"] + [r["strategy"][:25] for r in results]
    col_w = 18
    hdr_line = f"  {'Metric':<30}" + "".join(f"{h:>{col_w}}" for h in [r["strategy"][:16] for r in results])
    print(hdr_line)
    print("  " + "─" * (30 + col_w * len(results)))

    rows = [
        ("Sizing", "sizing"),
        ("Txn Costs", "costs"),
        ("Trades", "n_trades"),
        ("Win Rate (%)", "win_rate"),
        ("Avg Return Gross (%)", "avg_return_gross"),
        ("Avg Return Net (%)", "avg_return_net"),
        ("Profit Factor", "profit_factor"),
        ("Avg Hold (days)", "avg_hold_days"),
        ("Final Equity (₹)", "final_equity"),
        ("Total Return (%)", "total_return_pct"),
        ("CAGR (%)", "cagr_pct"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Calmar Ratio", "calmar"),
        ("Winning Days (%)", "winning_days_pct"),
        ("Avg SL Distance (%)", "avg_sl_pct"),
        ("Avg TP Distance (%)", "avg_tp_pct"),
    ]

    # Find best for each metric
    higher_better = {"win_rate", "avg_return_net", "avg_return_gross", "profit_factor",
                     "final_equity", "total_return_pct", "cagr_pct", "calmar", "winning_days_pct"}
    lower_better = {"max_drawdown_pct"}

    for label, key in rows:
        vals = [r.get(key, "—") for r in results]
        line = f"  {label:<30}"
        # Find best value
        numeric_vals = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
        best_idx = -1
        if numeric_vals:
            if key in higher_better:
                best_idx = max(numeric_vals, key=lambda x: x[1])[0]
            elif key in lower_better:
                best_idx = max(numeric_vals, key=lambda x: x[1])[0]  # least negative = max

        for i, v in enumerate(vals):
            if isinstance(v, float):
                s = f"{v:,.1f}" if abs(v) >= 100 else f"{v}"
            else:
                s = str(v)
            if i == best_idx:
                s = f"*{s}*"
            line += f"{s:>{col_w}}"
        print(line)

    # Exit breakdown
    print(f"\n  {'Exit Breakdown':<30}")
    print("  " + "─" * (30 + col_w * len(results)))
    for reason in ["stop_loss", "take_profit", "trail_stop", "time_exit"]:
        vals = []
        for r in results:
            eb = r.get("exit_breakdown", {})
            n = eb.get(reason, 0)
            total = r.get("n_trades", 1)
            vals.append(f"{n} ({n/total*100:.0f}%)" if n > 0 else "—")
        line = f"  {reason:<30}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(line)

    # By trigger type
    print(f"\n  {'By Trigger Type':<30}")
    print("  " + "─" * (30 + col_w * len(results)))
    all_triggers = set()
    for r in results:
        all_triggers.update(r.get("by_trigger", {}).keys())
    for tt in sorted(all_triggers):
        print(f"  {tt}:")
        for metric, label in [("win_rate", "  WR"), ("avg_return", "  Avg"), ("profit_factor", "  PF")]:
            vals = []
            for r in results:
                bt = r.get("by_trigger", {}).get(tt, {})
                v = bt.get(metric, "—")
                vals.append(f"{v}" if v != "—" else "—")
            line = f"  {label:<30}" + "".join(f"{v:>{col_w}}" for v in vals)
            print(line)

    print("\n" + "=" * 120)
    print("  * = best in row")
    print("  Costs: STT 0.1% sell, stamp 0.015% buy, brokerage 0.03%, GST 18%")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="PRISM Strategy Lab")
    parser.add_argument("--min-date", type=str, default=None)
    parser.add_argument("--max-date", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy names (default: all)")
    parser.add_argument("--sizing", type=str, default="equal",
                        help="Sizing mode: equal, regime_conservative, regime_moderate, skip_correction (default: equal)")
    parser.add_argument("--no-costs", action="store_true", help="Disable transaction costs")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    # Load data
    trigger_files = load_trigger_files(min_date=args.min_date, max_date=args.max_date, input_dir=args.input_dir)
    if not trigger_files:
        print("No trigger files found.")
        return

    picks_df = extract_picks(trigger_files)
    if picks_df.empty:
        print("No picks extracted.")
        return

    # Add regime info from trigger file metadata, or compute from Nifty
    picks_df["regime"] = "unknown"
    regime_from_meta = 0
    for f in trigger_files:
        meta = f.get("data", {}).get("metadata", {})
        regime = meta.get("regime", "")
        date = f.get("date", "")
        if date and regime:
            picks_df.loc[picks_df["trade_date"] == date, "regime"] = regime
            regime_from_meta += 1

    unknown_frac = (picks_df["regime"] == "unknown").mean()
    logger.info(f"Regimes from metadata: {regime_from_meta}, unknown fraction: {unknown_frac:.2f}")

    # If regimes are mostly unknown, compute from Nifty 50 data
    if unknown_frac > 0.5:
        logger.info("Computing regimes from Nifty 50 data...")
        nifty_regimes = compute_nifty_regimes(picks_df["trade_date_dt"].min(), picks_df["trade_date_dt"].max())
        matched = 0
        for date_str, regime in nifty_regimes.items():
            mask = picks_df["trade_date"] == date_str
            if mask.any():
                picks_df.loc[mask, "regime"] = regime
                matched += 1
        logger.info(f"Matched {matched} trade dates with Nifty regimes")

    logger.info(f"Loaded {len(picks_df)} picks across {picks_df['trade_date'].nunique()} dates")
    logger.info(f"Regimes: {picks_df['regime'].value_counts().to_dict()}")

    # Fetch price data (once, shared)
    logger.info("Fetching price data...")
    price_data = fetch_price_data(picks_df)

    # Select strategies
    if args.strategies:
        strat_names = [s.strip() for s in args.strategies.split(",")]
        strategies = {k: v for k, v in STRATEGIES.items() if k in strat_names}
    else:
        strategies = STRATEGIES

    sizing = SIZING_MODES.get(args.sizing, SIZING_MODES["equal"])
    costs = TxnCosts() if not args.no_costs else TxnCosts(
        brokerage_pct=0, stt_sell_pct=0, stamp_buy_pct=0,
        exchange_pct=0, sebi_pct=0, gst_on_brokerage=0,
    )

    logger.info(f"Running {len(strategies)} strategies with sizing={sizing.name}, costs={costs}")

    # Run all strategies
    results = []
    for key, strat in strategies.items():
        strat.include_costs = not args.no_costs
        logger.info(f"  Simulating: {strat.name}")
        result = simulate_strategy(picks_df, price_data, strat, sizing, costs)
        results.append(result)

    # Also run with regime sizing if comparing
    if args.sizing != "equal":
        # Add equal sizing baseline for comparison
        for key, strat in list(strategies.items())[:1]:  # just first strategy
            strat_copy = ExitStrategy(**{k: getattr(strat, k) for k in strat.__dataclass_fields__})
            strat_copy.name = f"{strat.name} (equal sizing)"
            result = simulate_strategy(picks_df, price_data, strat_copy, SIZING_MODES["equal"], costs)
            results.insert(0, result)

    # Print comparison
    print_comparison(results)

    # Save
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
