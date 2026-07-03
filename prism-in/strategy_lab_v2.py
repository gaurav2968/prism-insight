#!/usr/bin/env python3
"""
Strategy Lab V2 — Day-by-day portfolio simulation with independent feature testing.

Unlike V1 which computed entry→exit in one shot, this simulates each trading day:
- Recomputes ATR daily for every open position
- Adjusts stops dynamically based on current volatility
- Tests features independently (A/B style) then combines winners

Features tested independently:
  A: daily_atr_recompute — update ATR each day, adjust SL/TP dynamically
  B: ratchet_stop — after stock runs up, ratchet SL higher using current ATR
  C: time_decay — tighten SL multiplier as holding period increases
  D: winner_sizing — allocate more capital to recently winning trigger types
  E: vol_sizing — risk parity: lower-ATR% stocks get more capital
  F: breakeven_stop — move SL to entry after stock hits +1×ATR
  G: r_multiple_trail — trail stop in R-steps (+1R→BE, +2R→+1R, +3R→+2R)
  H: parabolic_sar — accelerating stop that speeds up on new highs
  I: time_expectancy_cut — exit early if not up +0.5R after N days
  J: partial_profit — close 50% at +1R, rest rides with BE stop
  K: kelly_sizing — position size from rolling win-rate & payoff ratio

Usage:
    python prism-in/strategy_lab_v2.py --input-dir trigger_results_v2 --min-date 20240101 --max-date 20240925
    python prism-in/strategy_lab_v2.py --features A,B,F  # test specific combo
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine import load_trigger_files, extract_picks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ══════════════════════════════════════════════════════════════
# TRANSACTION COSTS
# ══════════════════════════════════════════════════════════════

ROUND_TRIP_COST_PCT = 0.00193  # 0.193% (STT + brokerage + stamp + GST)

# Regime → numeric score for Feature N
REGIME_SCORE_MAP = {
    "bull_strong": 5,
    "bull_medium": 3,
    "bull_weak": 0,
    "slope_declining": 0,
    "correction": -3,
    "bear": -5,
    "bear_bottom": -5,
    "bear_bottom_extreme": -5,
}


# ══════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ══════════════════════════════════════════════════════════════

@dataclass
class Features:
    """Toggle individual features for A/B testing."""
    name: str = "baseline"

    # Base params
    atr_period: int = 14
    initial_sl_mult: float = 3.0   # entry SL = entry - 3×ATR
    initial_tp_mult: float = 2.0   # entry TP = entry + 2×ATR
    max_hold_days: int = 30

    # Feature A: Daily ATR recompute
    daily_atr_recompute: bool = False  # recompute ATR each day

    # Feature B: Ratchet stop (ATR-based trailing using daily ATR)
    ratchet_stop: bool = False         # ratchet SL up as stock rises
    ratchet_atr_mult: float = 2.0     # SL = highest_close - N×current_ATR

    # Feature C: Time decay — tighten SL as holding period grows
    time_decay: bool = False
    decay_start_day: int = 10          # start tightening after N days
    decay_end_mult: float = 1.5        # SL mult decays to this by max_hold

    # Feature D: Winner sizing — allocate more to winning trigger types
    winner_sizing: bool = False
    winner_lookback: int = 20          # look at last N completed trades per type

    # Feature E: Volatility-adjusted sizing (risk parity)
    vol_sizing: bool = False           # inverse ATR% weighting

    # Feature F: Breakeven stop — move SL to entry after +1×ATR
    breakeven_stop: bool = False
    breakeven_atr_mult: float = 1.0    # activate after stock is up N×ATR

    # Feature G: R-multiple trailing — trail in R-steps
    r_multiple_trail: bool = False
    # R = entry_atr * initial_sl_mult (initial risk)
    # At +1R profit → SL to entry (BE)
    # At +2R profit → SL to +1R
    # At +3R profit → SL to +2R

    # Feature H: Parabolic SAR exit
    parabolic_sar: bool = False
    sar_af_start: float = 0.02         # starting acceleration factor
    sar_af_step: float = 0.02          # AF increment on each new high
    sar_af_max: float = 0.20           # max acceleration factor

    # Feature I: Time expectancy cut — exit if not moving
    time_expectancy_cut: bool = False
    time_cut_days: int = 7             # check after N trading days
    time_cut_r_threshold: float = 0.5  # must be up at least this many R

    # Feature J: Partial profit taking
    partial_profit: bool = False
    partial_close_pct: float = 0.50    # close 50% of position
    partial_trigger_r: float = 1.0     # trigger at +1R profit

    # Feature K: Kelly criterion sizing
    kelly_sizing: bool = False
    kelly_lookback: int = 30           # rolling window of completed trades
    kelly_fraction: float = 0.5        # half-Kelly for safety

    # Capital management (L: realistic portfolio constraints)
    max_positions: int = 0             # max concurrent positions (0=unlimited/old behavior)
    quality_weighted: bool = False     # 40% to best quality pick, 60% split equally
    quality_top_pct: float = 0.40      # fraction allocated to top quality pick

    # Feature M: Regime emergency exit — close all positions when regime turns bad
    regime_exit: bool = False
    regime_exit_states: tuple = ("correction", "bear", "bear_bottom", "bear_bottom_extreme")

    # Feature N: Dynamic regime score — adjust SL/TP based on rolling market regime
    regime_score: bool = False
    regime_score_lookback: int = 5     # rolling window of days for avg score
    # When avg score < 0: tighten SL (3x→1.5x), disable TP (exit at SL or time)
    # When avg score > 0: normal SL, widen TP (2x→3x at score +5)
    regime_sl_tight: float = 1.5       # SL multiplier when avg score = -5
    regime_sl_normal: float = 3.0      # SL multiplier when avg score >= 0
    regime_tp_normal: float = 2.0      # TP multiplier when avg score = 0
    regime_tp_wide: float = 3.0        # TP multiplier when avg score = +5

    # Feature O: Regime-based position limits — fewer concurrent positions in weak regimes
    regime_position_limits: dict = None  # {regime_type: max_positions}, None = disabled

    # Feature P: Regime-adaptive entry SL/TP — adjust SL/TP multipliers based on entry-day regime
    regime_entry_sltp: dict = None  # {regime_type: (sl_mult, tp_mult)}, None = use global initial_sl/tp_mult

    # Feature Q: Market-adaptive dynamic SL/TP using smoothed market indicators
    # Uses EMA/SMA of regime scores + optionally NIFTY vol to continuously adjust exits
    market_adaptive: bool = False
    market_score_key: str = "ema5"       # which smoothed score: raw/ema3/ema5/ema10/ema20/sma5/sma10/momentum
    # Score-to-multiplier mapping (linear interpolation over score range -5 to +5)
    market_sl_bear: float = 2.0          # SL mult when score = -5 (most bearish)
    market_sl_bull: float = 3.0          # SL mult when score = +5 (most bullish)
    market_tp_bear: float = 1.5          # TP mult when score = -5
    market_tp_bull: float = 2.5          # TP mult when score = +5
    # During-hold SL/TP adjustment
    market_hold_adjust: bool = False     # re-evaluate SL/TP each day during holding
    market_ratchet_only: bool = True     # only tighten (SL up, TP down), never loosen
    # Vol scaling: multiply SL/TP by NIFTY vol ratio (current_vol / median_vol)
    market_vol_scale: bool = False       # enable vol-based scaling on top of score
    market_vol_sl_mode: str = "widen"    # "widen" = wider SL in high vol, "tighten" = tighter
    market_vol_tp_mode: str = "tighten"  # "tighten" = tighter TP in high vol, "widen" = wider


def _score_to_mult(score: float, at_neg5: float, at_pos5: float) -> float:
    """Linear interpolation: score in [-5, +5] → multiplier."""
    t = (score + 5.0) / 10.0  # normalize to [0, 1]
    t = max(0.0, min(1.0, t))
    return at_neg5 + t * (at_pos5 - at_neg5)


def make_feature_set(name: str, **overrides) -> Features:
    f = Features(name=name)
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


def extract_regime_by_date(trigger_files: list) -> dict:
    """Build a {date_str: regime_type} lookup from trigger files metadata."""
    regime_map = {}
    for tf in trigger_files:
        date_str = tf["date"]
        data = tf["data"]
        meta = data.get("metadata", {})
        regime = meta.get("regime", {})
        if isinstance(regime, dict):
            rtype = regime.get("type", "")
        elif isinstance(regime, str):
            rtype = regime
        else:
            rtype = ""
        if rtype:
            regime_map[date_str] = rtype
    return regime_map


def build_market_indicators(regime_by_date: dict, nifty_hist: pd.DataFrame = None) -> dict:
    """
    Pre-compute smoothed market indicators from regime scores and optional NIFTY data.
    
    Known quantitative concepts used:
    - EMA (Exponential Moving Average): weights recent data more — reacts faster to regime changes
    - SMA (Simple Moving Average): equal weights — more stable, filters impulse noise
    - Momentum: rate-of-change of smoothed score — detects trend acceleration/deceleration
    - Realized Volatility: annualized std of log returns — measures market uncertainty
    - Vol Percentile: current vol vs trailing distribution — identifies vol regime (calm/stressed)
    - ADX (Average Directional Index): trend strength regardless of direction
    
    Returns: {date_str: {
        'raw': float,           # raw regime score (-5 to +5)
        'ema3/5/10/20': float,  # EMA-smoothed scores
        'sma5/10': float,       # SMA-smoothed scores
        'momentum': float,      # 5-day change in EMA(5)
        'nifty_rvol': float,    # NIFTY 20-day realized vol (if nifty_hist provided)
        'nifty_vol_ratio': float, # current_vol / 252-day median vol
        'nifty_adx': float,     # ADX(14) trend strength
    }}
    """
    dates = sorted(regime_by_date.keys())
    if not dates:
        return {}

    # Convert regimes to numeric scores
    raw_scores = [REGIME_SCORE_MAP.get(regime_by_date[d], 0) for d in dates]

    # ── EMA computation ──
    def _ema(values, span):
        alpha = 2.0 / (span + 1)
        result = [float(values[0])]
        for i in range(1, len(values)):
            result.append(alpha * values[i] + (1 - alpha) * result[-1])
        return result

    # ── SMA computation ──
    def _sma(values, window):
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start:i + 1]) / (i - start + 1))
        return result

    ema3 = _ema(raw_scores, 3)
    ema5 = _ema(raw_scores, 5)
    ema10 = _ema(raw_scores, 10)
    ema20 = _ema(raw_scores, 20)
    sma5 = _sma(raw_scores, 5)
    sma10 = _sma(raw_scores, 10)

    # ── NIFTY volatility & ADX (if data provided) ──
    nifty_by_date = {}
    if nifty_hist is not None and not nifty_hist.empty:
        nifty_closes = nifty_hist["Close"].values
        nifty_highs = nifty_hist["High"].values
        nifty_lows = nifty_hist["Low"].values
        nifty_dates = [d.strftime("%Y%m%d") if hasattr(d, 'strftime') else str(d)[:10].replace("-", "")
                       for d in nifty_hist.index]

        # Realized vol (20-day trailing, annualized)
        log_rets = np.diff(np.log(nifty_closes))
        rvol_20 = []
        for i in range(len(log_rets)):
            window = log_rets[max(0, i - 19):i + 1]
            rvol_20.append(float(np.std(window) * np.sqrt(252) * 100) if len(window) >= 5 else 15.0)

        # Vol percentile: current vol vs trailing 252 days
        vol_pctile = []
        for i in range(len(rvol_20)):
            lookback = rvol_20[max(0, i - 251):i + 1]
            current = rvol_20[i]
            pctile = sum(1 for v in lookback if v <= current) / len(lookback)
            vol_pctile.append(pctile)

        # Median vol for ratio computation
        vol_medians = []
        for i in range(len(rvol_20)):
            lookback = rvol_20[max(0, i - 251):i + 1]
            vol_medians.append(float(np.median(lookback)))

        # ADX(14) computation
        def _compute_adx(highs, lows, closes, period=14):
            """Average Directional Index — measures trend strength (0-100)."""
            n = len(closes)
            if n < period + 1:
                return [20.0] * n

            # True Range, +DM, -DM
            tr = np.zeros(n)
            plus_dm = np.zeros(n)
            minus_dm = np.zeros(n)
            for i in range(1, n):
                h_l = highs[i] - lows[i]
                h_pc = abs(highs[i] - closes[i - 1])
                l_pc = abs(lows[i] - closes[i - 1])
                tr[i] = max(h_l, h_pc, l_pc)
                up = highs[i] - highs[i - 1]
                down = lows[i - 1] - lows[i]
                plus_dm[i] = up if (up > down and up > 0) else 0
                minus_dm[i] = down if (down > up and down > 0) else 0

            # Wilder smoothing
            atr14 = np.zeros(n)
            pdm14 = np.zeros(n)
            mdm14 = np.zeros(n)
            atr14[period] = np.mean(tr[1:period + 1])
            pdm14[period] = np.mean(plus_dm[1:period + 1])
            mdm14[period] = np.mean(minus_dm[1:period + 1])
            for i in range(period + 1, n):
                atr14[i] = (atr14[i - 1] * (period - 1) + tr[i]) / period
                pdm14[i] = (pdm14[i - 1] * (period - 1) + plus_dm[i]) / period
                mdm14[i] = (mdm14[i - 1] * (period - 1) + minus_dm[i]) / period

            # +DI, -DI, DX, ADX
            adx = [20.0] * n  # default
            dx_vals = []
            for i in range(period, n):
                if atr14[i] > 0:
                    pdi = 100 * pdm14[i] / atr14[i]
                    mdi = 100 * mdm14[i] / atr14[i]
                    dx = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0
                else:
                    dx = 0
                dx_vals.append(dx)
                if len(dx_vals) >= period:
                    adx[i] = np.mean(dx_vals[-period:])
            return adx

        adx_vals = _compute_adx(nifty_highs, nifty_lows, nifty_closes)

        # Build nifty lookup (offset by 1 for log returns alignment)
        for i in range(1, len(nifty_dates)):
            d = nifty_dates[i]
            vol_idx = i - 1  # log_rets is 1 shorter
            if vol_idx < len(rvol_20):
                median_vol = vol_medians[vol_idx] if vol_medians[vol_idx] > 0 else 15.0
                nifty_by_date[d] = {
                    'nifty_rvol': round(rvol_20[vol_idx], 2),
                    'nifty_vol_pctile': round(vol_pctile[vol_idx], 2),
                    'nifty_vol_ratio': round(rvol_20[vol_idx] / median_vol, 3),
                    'nifty_adx': round(adx_vals[i], 1),
                }

    # ── Build final indicators dict ──
    indicators = {}
    for i, d in enumerate(dates):
        ind = {
            'raw': raw_scores[i],
            'ema3': round(ema3[i], 3),
            'ema5': round(ema5[i], 3),
            'ema10': round(ema10[i], 3),
            'ema20': round(ema20[i], 3),
            'sma5': round(sma5[i], 3),
            'sma10': round(sma10[i], 3),
            'momentum': round(ema5[i] - (ema5[i - 5] if i >= 5 else ema5[0]), 3),
        }
        # Merge NIFTY data if available
        if d in nifty_by_date:
            ind.update(nifty_by_date[d])
        indicators[d] = ind

    return indicators


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def fetch_all_price_data(picks_df: pd.DataFrame) -> dict:
    """Download OHLCV for all tickers. Returns {ticker: DataFrame}."""
    unique_tickers = picks_df["ticker"].unique()
    earliest = picks_df["trade_date_dt"].min() - timedelta(days=30)
    latest = picks_df["trade_date_dt"].max() + timedelta(days=60)

    yf_tickers = [f"{t}.NS" for t in unique_tickers]
    all_data = {}
    chunk_size = 50

    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i + chunk_size]
        logger.info(f"  Downloading chunk {i // chunk_size + 1}/{(len(yf_tickers) + chunk_size - 1) // chunk_size}")
        try:
            data = yf.download(chunk, start=earliest.strftime("%Y-%m-%d"),
                               end=latest.strftime("%Y-%m-%d"),
                               auto_adjust=True, progress=False)
            if not data.empty:
                for yf_t in chunk:
                    nse = yf_t.replace(".NS", "")
                    try:
                        if len(chunk) == 1:
                            td = data
                        else:
                            td = data.xs(yf_t, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
                        if not td.empty and "Close" in td.columns:
                            all_data[nse] = td
                    except (KeyError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"  Download failed: {e}")

    logger.info(f"Got price data for {len(all_data)}/{len(unique_tickers)} tickers")
    return all_data


def compute_atr_at_date(hist: pd.DataFrame, date, period: int = 14) -> float:
    """Compute ATR using data up to (and including) the given date."""
    available = hist[hist.index <= date]
    if len(available) < period + 1:
        return 0
    recent = available.tail(period + 1)
    hi = recent["High"].values
    lo = recent["Low"].values
    cl = recent["Close"].values
    tr_vals = []
    for i in range(1, len(recent)):
        tr_vals.append(max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1])))
    return float(np.mean(tr_vals[-period:])) if len(tr_vals) >= period else float(np.mean(tr_vals))


# ══════════════════════════════════════════════════════════════
# DAY-BY-DAY PORTFOLIO SIMULATOR
# ══════════════════════════════════════════════════════════════

@dataclass
class Position:
    """An open position being tracked day by day."""
    ticker: str
    entry_price: float
    entry_date: pd.Timestamp
    entry_atr: float
    trigger_type: str
    stop_loss: float
    take_profit: float     # 0 if disabled
    max_exit_date: pd.Timestamp
    highest_close: float   # track for ratchet
    lowest_close: float = 0   # track MAE
    breakeven_activated: bool = False
    day_count: int = 0
    alloc_capital: float = 0
    # R-multiple tracking
    initial_risk_r: float = 0          # 1R = entry_price * sl_mult * atr at entry
    r_trail_level: int = 0             # highest R-level reached (1, 2, 3)
    # Parabolic SAR tracking
    sar_value: float = 0               # current SAR value
    sar_af: float = 0.02               # current acceleration factor
    sar_ep: float = 0                  # extreme point (highest high)
    # Partial profit tracking
    partial_closed: bool = False       # already took partial?
    remaining_alloc: float = 0         # alloc after partial close


def simulate_portfolio(
    picks_df: pd.DataFrame,
    price_data: dict,
    features: Features,
    capital: float = 1_000_000,
    regime_by_date: dict = None,
    market_indicators: dict = None,
) -> dict:
    """
    Day-by-day portfolio simulation.
    Each trading day:
      1. Check exits for open positions (SL, TP, time)
      2. Recompute ATR and adjust stops (if features enabled)
      3. Open new positions from today's picks
      4. Record P&L
    """
    # Build a calendar of all trading days from price data
    all_dates = set()
    for hist in price_data.values():
        all_dates.update(hist.index.tolist())
    trading_days = sorted(all_dates)

    # Index picks by trade date
    picks_by_date = {}
    for _, row in picks_df.iterrows():
        d = row["trade_date"]
        if d not in picks_by_date:
            picks_by_date[d] = []
        picks_by_date[d].append(row)

    # Track strategy-level win rates per trigger type (for winner sizing)
    trigger_history = {}  # {trigger_type: list of (date, return_pct)}

    open_positions: list[Position] = []
    completed_trades = []
    daily_equity = []
    cumulative_pnl = 0.0  # track total realized P&L (non-compounding)
    available_cash = capital  # cash not deployed in positions
    use_cash_tracking = features.max_positions > 0  # realistic mode
    max_pos = features.max_positions if features.max_positions > 0 else 9999
    trades_skipped = 0
    regime_score_history = []  # for Feature N

    for day in trading_days:
        day_str = day.strftime("%Y%m%d")

        # ── Feature M: Regime emergency exit — close all if regime is bad ──
        if features.regime_exit and regime_by_date and open_positions:
            day_regime = regime_by_date.get(day_str, "")
            if day_regime in features.regime_exit_states:
                for pos in open_positions:
                    if pos.ticker not in price_data:
                        continue
                    hist = price_data[pos.ticker]
                    day_data = hist[hist.index == day]
                    if day_data.empty:
                        continue
                    exit_price = float(day_data["Close"].iloc[0])
                    gross_ret = (exit_price / pos.entry_price - 1) * 100
                    net_ret = gross_ret - ROUND_TRIP_COST_PCT * 100
                    pnl = pos.alloc_capital * net_ret / 100
                    completed_trades.append({
                        "ticker": pos.ticker,
                        "entry_date": pos.entry_date.strftime("%Y%m%d"),
                        "exit_date": day_str,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "exit_reason": "regime_exit",
                        "days_held": pos.day_count,
                        "gross_return_pct": round(gross_ret, 2),
                        "net_return_pct": round(net_ret, 2),
                        "pnl": round(pnl, 0),
                        "trigger_type": pos.trigger_type,
                        "entry_atr": round(pos.entry_atr, 2),
                        "alloc_capital": round(pos.alloc_capital, 0),
                        "mfe_pct": round((pos.highest_close / pos.entry_price - 1) * 100, 2) if pos.highest_close > 0 else 0,
                        "mae_pct": round((pos.lowest_close / pos.entry_price - 1) * 100, 2) if pos.lowest_close > 0 else 0,
                    })
                    cumulative_pnl += pnl
                    if use_cash_tracking:
                        available_cash += pos.alloc_capital + pnl
                open_positions = []
                # Skip to equity recording — no new entries on bad regime days
                unrealized = 0
                deployed = 0
                daily_equity.append({
                    "date": day_str,
                    "equity": (available_cash) if use_cash_tracking else (capital + cumulative_pnl),
                    "open_positions": 0,
                    "available_cash": round(available_cash, 0) if use_cash_tracking else 0,
                    "deployed": 0,
                    "capital": capital + cumulative_pnl,
                })
                continue

        # ── Feature N: Dynamic regime score — adjust SL/TP based on rolling avg ──
        regime_sl_mult_override = None
        regime_tp_mult_override = None
        if features.regime_score and regime_by_date:
            day_regime = regime_by_date.get(day_str, "")
            score = REGIME_SCORE_MAP.get(day_regime, 0)
            regime_score_history.append(score)
            window = regime_score_history[-features.regime_score_lookback:]
            avg_score = sum(window) / len(window)

            if avg_score < 0:
                # Negative regime: tighten SL proportionally (-5 = tightest, 0 = normal)
                # Linear interpolation: avg=-5 → regime_sl_tight, avg=0 → regime_sl_normal
                t = min(abs(avg_score) / 5.0, 1.0)
                regime_sl_mult_override = features.regime_sl_normal - t * (features.regime_sl_normal - features.regime_sl_tight)
                regime_tp_mult_override = features.regime_tp_normal  # keep TP normal in bad regime
            else:
                # Positive regime: widen TP proportionally (0 = normal, +5 = widest)
                t = min(avg_score / 5.0, 1.0)
                regime_sl_mult_override = features.regime_sl_normal  # keep SL normal in good regime
                regime_tp_mult_override = features.regime_tp_normal + t * (features.regime_tp_wide - features.regime_tp_normal)

        # ── Phase 1: Check exits for open positions ──
        closed_today = []
        for pos in open_positions:
            pos.day_count += 1

            if pos.ticker not in price_data:
                continue
            hist = price_data[pos.ticker]
            day_data = hist[hist.index == day]
            if day_data.empty:
                continue

            price_high = float(day_data["High"].iloc[0])
            price_low = float(day_data["Low"].iloc[0])
            price_close = float(day_data["Close"].iloc[0])

            # ── Track MFE/MAE (highest high, lowest low during hold) ──
            if price_high > pos.highest_close:
                pos.highest_close = price_high
            if pos.lowest_close == 0 or price_low < pos.lowest_close:
                pos.lowest_close = price_low

            # ── Feature A: Daily ATR recompute ──
            if features.daily_atr_recompute:
                current_atr = compute_atr_at_date(hist, day, features.atr_period)
            else:
                current_atr = pos.entry_atr

            # ── Feature C: Time decay — tighten SL multiplier ──
            if features.time_decay and pos.day_count >= features.decay_start_day:
                progress = min((pos.day_count - features.decay_start_day) /
                               (features.max_hold_days - features.decay_start_day), 1.0)
                current_sl_mult = features.initial_sl_mult - progress * (features.initial_sl_mult - features.decay_end_mult)
            else:
                current_sl_mult = features.initial_sl_mult

            # ── Feature F: Breakeven stop ──
            if features.breakeven_stop and not pos.breakeven_activated:
                if current_atr > 0 and price_close >= pos.entry_price + features.breakeven_atr_mult * current_atr:
                    pos.breakeven_activated = True
                    # Move SL to entry price (breakeven)
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)

            # ── Feature G: R-multiple trailing ──
            if features.r_multiple_trail and pos.initial_risk_r > 0:
                r_unit = pos.initial_risk_r
                current_profit = price_close - pos.entry_price
                current_r = current_profit / r_unit if r_unit > 0 else 0

                if current_r >= 3.0 and pos.r_trail_level < 3:
                    pos.r_trail_level = 3
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price + 2.0 * r_unit)
                elif current_r >= 2.0 and pos.r_trail_level < 2:
                    pos.r_trail_level = 2
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price + 1.0 * r_unit)
                elif current_r >= 1.0 and pos.r_trail_level < 1:
                    pos.r_trail_level = 1
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)  # breakeven

            # ── Feature H: Parabolic SAR ──
            if features.parabolic_sar:
                if price_high > pos.sar_ep:
                    # New extreme point — accelerate
                    pos.sar_ep = price_high
                    pos.sar_af = min(pos.sar_af + features.sar_af_step, features.sar_af_max)
                # SAR moves toward EP
                new_sar = pos.sar_value + pos.sar_af * (pos.sar_ep - pos.sar_value)
                # SAR can only move up (never down for long positions)
                if new_sar > pos.sar_value:
                    pos.sar_value = new_sar
                    pos.stop_loss = max(pos.stop_loss, pos.sar_value)

            # ── Feature B: Ratchet stop (trailing based on daily ATR) ──
            if features.ratchet_stop and current_atr > 0:
                if price_close > pos.highest_close:
                    pos.highest_close = price_close
                ratchet_sl = pos.highest_close - features.ratchet_atr_mult * current_atr
                pos.stop_loss = max(pos.stop_loss, ratchet_sl)
            elif features.daily_atr_recompute and current_atr > 0:
                # Even without ratchet, update SL based on new ATR from entry
                new_sl = pos.entry_price - current_sl_mult * current_atr
                # Only tighten, never widen
                pos.stop_loss = max(pos.stop_loss, new_sl)

            # ── Feature A: Update TP with daily ATR ──
            if features.daily_atr_recompute and current_atr > 0 and pos.take_profit > 0:
                new_tp = pos.entry_price + features.initial_tp_mult * current_atr
                # Only move TP closer (tighter), not further away
                if new_tp < pos.take_profit:
                    pos.take_profit = new_tp

            # ── Check exit conditions ──
            exit_price = None
            exit_reason = ""

            # ── Feature N: Apply regime-adjusted SL/TP ──
            if features.regime_score and regime_sl_mult_override is not None:
                atr_for_regime = current_atr if features.daily_atr_recompute else pos.entry_atr
                if atr_for_regime > 0:
                    # Regime-adjusted SL: tighter in bad regimes
                    regime_sl = pos.entry_price - regime_sl_mult_override * atr_for_regime
                    # Only tighten SL, never widen it beyond initial
                    pos.stop_loss = max(pos.stop_loss, regime_sl)
                    # Regime-adjusted TP: wider in good regimes
                    if pos.take_profit > 0 and regime_tp_mult_override is not None:
                        regime_tp = pos.entry_price + regime_tp_mult_override * atr_for_regime
                        pos.take_profit = regime_tp  # allow TP to widen in bull

            # ── Feature Q: Market-adaptive hold adjustment ──
            if features.market_adaptive and features.market_hold_adjust and market_indicators:
                mi = market_indicators.get(day_str)
                if mi:
                    score = mi.get(features.market_score_key, 0)
                    new_sl_mult = _score_to_mult(score, features.market_sl_bear, features.market_sl_bull)
                    new_tp_mult = _score_to_mult(score, features.market_tp_bear, features.market_tp_bull)
                    hold_atr = current_atr if features.daily_atr_recompute else pos.entry_atr
                    if hold_atr > 0:
                        new_sl = pos.entry_price - new_sl_mult * hold_atr
                        new_tp = pos.entry_price + new_tp_mult * hold_atr if new_tp_mult > 0 else 0
                        if features.market_ratchet_only:
                            # Only tighten: SL can only go up, TP can only go down
                            pos.stop_loss = max(pos.stop_loss, new_sl)
                            if pos.take_profit > 0 and new_tp > 0:
                                pos.take_profit = min(pos.take_profit, new_tp)
                        else:
                            # Full dynamic: allow both tightening and loosening
                            pos.stop_loss = new_sl
                            if new_tp > 0:
                                pos.take_profit = new_tp

            # ── Feature I: Time expectancy cut ──
            if features.time_expectancy_cut and pos.day_count >= features.time_cut_days:
                r_unit = pos.initial_risk_r if pos.initial_risk_r > 0 else pos.entry_atr * features.initial_sl_mult
                current_profit = price_close - pos.entry_price
                current_r = current_profit / r_unit if r_unit > 0 else 0
                if current_r < features.time_cut_r_threshold:
                    exit_price = price_close
                    exit_reason = "time_cut"

            # Stop loss
            if exit_price is None and price_low <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "stop_loss"
            # Take profit
            elif exit_price is None and pos.take_profit > 0 and price_high >= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "take_profit"
            # Time exit
            elif exit_price is None and day >= pos.max_exit_date:
                exit_price = price_close
                exit_reason = "time_exit"

            # ── Feature J: Partial profit (before full exit) ──
            if features.partial_profit and not pos.partial_closed and exit_price is None:
                r_unit = pos.initial_risk_r if pos.initial_risk_r > 0 else pos.entry_atr * features.initial_sl_mult
                current_profit = price_close - pos.entry_price
                current_r = current_profit / r_unit if r_unit > 0 else 0
                if current_r >= features.partial_trigger_r:
                    # Close partial_close_pct of position
                    close_alloc = pos.alloc_capital * features.partial_close_pct
                    gross_ret = (price_close / pos.entry_price - 1) * 100
                    net_ret = gross_ret - ROUND_TRIP_COST_PCT * 100
                    partial_pnl = close_alloc * net_ret / 100
                    cumulative_pnl += partial_pnl
                    # Return partial capital + P&L to cash
                    if use_cash_tracking:
                        available_cash += close_alloc + partial_pnl
                    completed_trades.append({
                        "ticker": pos.ticker,
                        "entry_date": pos.entry_date.strftime("%Y%m%d"),
                        "exit_date": day_str,
                        "entry_price": pos.entry_price,
                        "exit_price": price_close,
                        "exit_reason": "partial_profit",
                        "days_held": pos.day_count,
                        "gross_return_pct": round(gross_ret, 2),
                        "net_return_pct": round(net_ret, 2),
                        "pnl": round(partial_pnl, 0),
                        "trigger_type": pos.trigger_type,
                        "entry_atr": round(pos.entry_atr, 2),
                        "alloc_capital": round(close_alloc, 0),
                        "mfe_pct": round((pos.highest_close / pos.entry_price - 1) * 100, 2) if pos.highest_close > 0 else 0,
                        "mae_pct": round((pos.lowest_close / pos.entry_price - 1) * 100, 2) if pos.lowest_close > 0 else 0,
                    })
                    pos.alloc_capital -= close_alloc
                    pos.remaining_alloc = pos.alloc_capital
                    pos.partial_closed = True
                    # Move SL to breakeven for the remaining position
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)

            if exit_price is not None:
                gross_ret = (exit_price / pos.entry_price - 1) * 100
                net_ret = gross_ret - ROUND_TRIP_COST_PCT * 100
                pnl = pos.alloc_capital * net_ret / 100

                completed_trades.append({
                    "ticker": pos.ticker,
                    "entry_date": pos.entry_date.strftime("%Y%m%d"),
                    "exit_date": day_str,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "days_held": pos.day_count,
                    "gross_return_pct": round(gross_ret, 2),
                    "net_return_pct": round(net_ret, 2),
                    "pnl": round(pnl, 0),
                    "trigger_type": pos.trigger_type,
                    "entry_atr": round(pos.entry_atr, 2),
                    "alloc_capital": round(pos.alloc_capital, 0),
                    "mfe_pct": round((pos.highest_close / pos.entry_price - 1) * 100, 2) if pos.highest_close > 0 else 0,
                    "mae_pct": round((pos.lowest_close / pos.entry_price - 1) * 100, 2) if pos.lowest_close > 0 else 0,
                })
                cumulative_pnl += pnl
                # Return capital + P&L to available cash
                if use_cash_tracking:
                    available_cash += pos.alloc_capital + pnl
                closed_today.append(pos)

                # Track for winner sizing
                if pos.trigger_type not in trigger_history:
                    trigger_history[pos.trigger_type] = []
                trigger_history[pos.trigger_type].append((day_str, net_ret))

        # Remove closed positions
        for pos in closed_today:
            open_positions.remove(pos)

        # ── Phase 2: Open new positions ──
        new_picks = picks_by_date.get(day_str, [])
        if new_picks:
            # Filter out tickers already held
            held_tickers = {p.ticker for p in open_positions}
            new_picks = [p for p in new_picks if p["ticker"] not in held_tickers
                         and p["ticker"] in price_data and p["entry_price"] > 0]

            # Limit by available slots
            # Feature O: Regime-based position limit override
            effective_max = max_pos
            if features.regime_position_limits and regime_by_date:
                day_regime = regime_by_date.get(day_str, "")
                if day_regime in features.regime_position_limits:
                    effective_max = features.regime_position_limits[day_regime]
            
            slots_available = effective_max - len(open_positions)
            if use_cash_tracking and slots_available <= 0:
                trades_skipped += len(new_picks)
                new_picks = []
            elif use_cash_tracking and available_cash < 10000:  # minimum ₹10k to open
                trades_skipped += len(new_picks)
                new_picks = []

            # Sort by quality score (best first) for slot prioritization
            if new_picks and use_cash_tracking:
                new_picks = sorted(new_picks, key=lambda p: p.get("quality_score", 0) or 0, reverse=True)
                if len(new_picks) > slots_available:
                    trades_skipped += len(new_picks) - slots_available
                    new_picks = new_picks[:slots_available]

            if new_picks:
                # ── Feature D: Winner sizing ──
                if features.winner_sizing:
                    type_weights = {}
                    for pick in new_picks:
                        tt = pick["trigger_type"]
                        history = trigger_history.get(tt, [])
                        recent = [r for d, r in history[-features.winner_lookback:]]
                        if len(recent) >= 5:
                            wr = sum(1 for r in recent if r > 0) / len(recent)
                            type_weights[tt] = max(0.2, wr)  # min 20% allocation
                        else:
                            type_weights[tt] = 0.5  # neutral until enough data
                else:
                    type_weights = {p["trigger_type"]: 1.0 for p in new_picks}

                # ── Feature E: Volatility sizing ──
                if features.vol_sizing:
                    atr_pcts = {}
                    for pick in new_picks:
                        ticker = pick["ticker"]
                        hist = price_data[ticker]
                        atr = compute_atr_at_date(hist, day, features.atr_period)
                        ep = pick["entry_price"]
                        atr_pcts[ticker] = atr / ep if ep > 0 and atr > 0 else 0.05
                    # Inverse vol: lower ATR% → more capital
                    max_atr_pct = max(atr_pcts.values()) if atr_pcts else 0.05
                    vol_weights = {t: max_atr_pct / max(v, 0.001) for t, v in atr_pcts.items()}
                    total_vw = sum(vol_weights.values())
                    vol_weights = {t: v / total_vw * len(new_picks) for t, v in vol_weights.items()}
                else:
                    vol_weights = {p["ticker"]: 1.0 for p in new_picks}

                # ── Feature K: Kelly criterion sizing ──
                kelly_weight = 1.0
                if features.kelly_sizing:
                    all_completed = [t["net_return_pct"] for t in completed_trades[-features.kelly_lookback:]]
                    if len(all_completed) >= 10:
                        wins_k = [r for r in all_completed if r > 0]
                        losses_k = [r for r in all_completed if r <= 0]
                        p = len(wins_k) / len(all_completed)
                        avg_win = np.mean(wins_k) if wins_k else 1
                        avg_loss = abs(np.mean(losses_k)) if losses_k else 1
                        b = avg_win / avg_loss if avg_loss > 0 else 1
                        q = 1 - p
                        kelly_f = (p * b - q) / b if b > 0 else 0
                        kelly_f = max(0.05, min(kelly_f * features.kelly_fraction, 0.25))
                        kelly_weight = kelly_f / 0.10  # normalize: 10% = neutral

                # Compute per-stock allocation
                if use_cash_tracking:
                    # Realistic: allocate from available cash
                    n_to_open = len(new_picks)
                    if features.quality_weighted and n_to_open > 1:
                        # Top pick gets quality_top_pct, rest split equally
                        top_alloc = available_cash * features.quality_top_pct
                        rest_alloc = available_cash * (1 - features.quality_top_pct) / (n_to_open - 1)
                        alloc_list = [top_alloc] + [rest_alloc] * (n_to_open - 1)
                    else:
                        per_stock = available_cash / max(n_to_open, 1)
                        alloc_list = [per_stock] * n_to_open
                else:
                    # Legacy: fixed capital / n_slots (ignores cash tracking)
                    n_slots = max(len(new_picks), 5)
                    base_per_stock = capital / n_slots
                    alloc_list = [base_per_stock] * len(new_picks)

                for idx, pick in enumerate(new_picks):
                    ticker = pick["ticker"]
                    hist = price_data[ticker]
                    entry_price = pick["entry_price"]
                    entry_date = day

                    atr = compute_atr_at_date(hist, day, features.atr_period)
                    if atr <= 0:
                        atr = entry_price * 0.03  # 3% fallback

                    # Feature P: regime-adaptive entry SL/TP
                    sl_mult = features.initial_sl_mult
                    tp_mult = features.initial_tp_mult
                    if features.regime_entry_sltp and regime_by_date:
                        day_regime = regime_by_date.get(day_str, "")
                        if day_regime in features.regime_entry_sltp:
                            sl_mult, tp_mult = features.regime_entry_sltp[day_regime]

                    # Feature Q: Market-adaptive entry SL/TP from smoothed indicators
                    if features.market_adaptive and market_indicators:
                        mi = market_indicators.get(day_str)
                        if mi:
                            score = mi.get(features.market_score_key, 0)
                            sl_mult = _score_to_mult(score, features.market_sl_bear, features.market_sl_bull)
                            tp_mult = _score_to_mult(score, features.market_tp_bear, features.market_tp_bull)
                            # Vol scaling on top
                            if features.market_vol_scale and 'nifty_vol_ratio' in mi:
                                vr = mi['nifty_vol_ratio']
                                vr = max(0.5, min(2.0, vr))  # clamp
                                if features.market_vol_sl_mode == "widen":
                                    sl_mult *= vr  # wider SL in high vol
                                else:
                                    sl_mult /= vr  # tighter SL in high vol
                                if features.market_vol_tp_mode == "tighten":
                                    tp_mult /= vr  # tighter TP in high vol
                                else:
                                    tp_mult *= vr  # wider TP in high vol

                    sl = entry_price - sl_mult * atr
                    tp = entry_price + tp_mult * atr if tp_mult > 0 else 0
                    initial_risk = sl_mult * atr  # 1R

                    # Apply sizing weights
                    type_w = type_weights.get(pick["trigger_type"], 1.0)
                    vol_w = vol_weights.get(ticker, 1.0)
                    base_alloc = alloc_list[idx]
                    if use_cash_tracking:
                        # In realistic mode, sizing features scale within available cash
                        alloc = base_alloc * type_w * vol_w * kelly_weight
                        # Cap to available cash
                        alloc = min(alloc, available_cash)
                        if alloc < 10000:  # skip if too small
                            trades_skipped += 1
                            continue
                    else:
                        alloc = base_alloc * type_w * vol_w * kelly_weight

                    max_exit = day + pd.Timedelta(days=int(features.max_hold_days * 1.5))
                    # Approximate trading days
                    max_exit_approx = day
                    td_count = 0
                    while td_count < features.max_hold_days:
                        max_exit_approx += pd.Timedelta(days=1)
                        if max_exit_approx.weekday() < 5:
                            td_count += 1

                    pos = Position(
                        ticker=ticker,
                        entry_price=entry_price,
                        entry_date=day,
                        entry_atr=atr,
                        trigger_type=pick["trigger_type"],
                        stop_loss=sl,
                        take_profit=tp,
                        max_exit_date=max_exit_approx,
                        highest_close=entry_price,
                        lowest_close=entry_price,
                        alloc_capital=alloc,
                        initial_risk_r=initial_risk,
                        sar_value=sl,          # SAR starts at initial SL
                        sar_af=features.sar_af_start,
                        sar_ep=entry_price,    # EP starts at entry
                        remaining_alloc=alloc,
                    )
                    open_positions.append(pos)
                    if use_cash_tracking:
                        available_cash -= alloc

        # Record daily equity (capital + unrealized P&L)
        unrealized = 0
        for pos in open_positions:
            if pos.ticker in price_data:
                hist = price_data[pos.ticker]
                day_data = hist[hist.index == day]
                if not day_data.empty:
                    curr_price = float(day_data["Close"].iloc[0])
                    unrealized += pos.alloc_capital * ((curr_price / pos.entry_price - 1) - ROUND_TRIP_COST_PCT)

        deployed = sum(p.alloc_capital for p in open_positions)
        daily_equity.append({
            "date": day_str,
            "equity": (available_cash + deployed + unrealized) if use_cash_tracking else (capital + cumulative_pnl + unrealized),
            "open_positions": len(open_positions),
            "available_cash": round(available_cash, 0) if use_cash_tracking else 0,
            "deployed": round(deployed, 0),
            "capital": capital + cumulative_pnl,
        })

    # ── Compute results ──
    if not completed_trades:
        return {"features": features.name, "n_trades": 0}

    df = pd.DataFrame(completed_trades)
    eq_df = pd.DataFrame(daily_equity)

    # Filter to only dates where we had trades
    eq_series = eq_df.set_index("date")["equity"]
    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / peak * 100
    max_dd = drawdown.min()
    final_eq = eq_series.iloc[-1]

    n_days = (pd.Timestamp(eq_df["date"].iloc[-1]) - pd.Timestamp(eq_df["date"].iloc[0])).days
    cagr = ((final_eq / capital) ** (365 / n_days) - 1) * 100 if n_days > 0 and final_eq > 0 else -100

    wins = df[df["net_return_pct"] > 0]
    losses = df[df["net_return_pct"] <= 0]
    win_rate = len(wins) / len(df) * 100
    avg_return = df["net_return_pct"].mean()
    gross_wins = wins["net_return_pct"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["net_return_pct"].sum()) if len(losses) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # ── Advanced metrics: Sharpe, Sortino, Expectancy, MFE/MAE ──
    daily_returns = eq_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0
    downside = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0
    expectancy = df["net_return_pct"].mean() if len(df) > 0 else 0
    avg_win = wins["net_return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["net_return_pct"].mean() if len(losses) > 0 else 0
    # MFE/MAE stats
    mfe_col = df["mfe_pct"] if "mfe_pct" in df.columns else pd.Series(dtype=float)
    mae_col = df["mae_pct"] if "mae_pct" in df.columns else pd.Series(dtype=float)
    # Consecutive wins/losses
    streaks_w, streaks_l, cur_w, cur_l = [], [], 0, 0
    for _, row in df.iterrows():
        if row["net_return_pct"] > 0:
            cur_w += 1
            if cur_l > 0: streaks_l.append(cur_l); cur_l = 0
        else:
            cur_l += 1
            if cur_w > 0: streaks_w.append(cur_w); cur_w = 0
    if cur_w > 0: streaks_w.append(cur_w)
    if cur_l > 0: streaks_l.append(cur_l)

    reasons = df["exit_reason"].value_counts().to_dict()

    by_trigger = {}
    for tt in df["trigger_type"].unique():
        tdf = df[df["trigger_type"] == tt]
        tw = tdf[tdf["net_return_pct"] > 0]
        tl = tdf[tdf["net_return_pct"] <= 0]
        gw = tw["net_return_pct"].sum() if len(tw) > 0 else 0
        gl = abs(tl["net_return_pct"].sum()) if len(tl) > 0 else 0
        by_trigger[tt] = {
            "n": len(tdf),
            "win_rate": round(len(tw) / len(tdf) * 100, 1),
            "avg_return": round(tdf["net_return_pct"].mean(), 2),
            "pf": round(gw / gl, 2) if gl > 0 else "inf",
        }

    result = {
        "features": features.name,
        "n_trades": len(df),
        "win_rate": round(win_rate, 1),
        "avg_return_net": round(avg_return, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "avg_hold_days": round(df["days_held"].mean(), 1),
        "final_equity": round(final_eq, 0),
        "total_return_pct": round((final_eq / capital - 1) * 100, 1),
        "cagr_pct": round(cagr, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "calmar": round(cagr / abs(max_dd), 2) if max_dd != 0 else 0,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "expectancy_pct": round(expectancy, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "best_trade_pct": round(df["net_return_pct"].max(), 2),
        "worst_trade_pct": round(df["net_return_pct"].min(), 2),
        "median_return_pct": round(df["net_return_pct"].median(), 2),
        "std_return_pct": round(df["net_return_pct"].std(), 2),
        "avg_mfe_pct": round(mfe_col.mean(), 2) if len(mfe_col) > 0 else 0,
        "avg_mae_pct": round(mae_col.mean(), 2) if len(mae_col) > 0 else 0,
        "max_mfe_pct": round(mfe_col.max(), 2) if len(mfe_col) > 0 else 0,
        "max_mae_pct": round(mae_col.min(), 2) if len(mfe_col) > 0 else 0,
        "mfe_winners": round(mfe_col[df["net_return_pct"] > 0].mean(), 2) if len(wins) > 0 and len(mfe_col) > 0 else 0,
        "mae_losers": round(mae_col[df["net_return_pct"] <= 0].mean(), 2) if len(losses) > 0 and len(mae_col) > 0 else 0,
        "max_consec_wins": max(streaks_w) if streaks_w else 0,
        "max_consec_losses": max(streaks_l) if streaks_l else 0,
        "total_pnl": round(df["pnl"].sum(), 0),
        "exit_breakdown": reasons,
        "by_trigger": by_trigger,
    }
    if use_cash_tracking:
        result["trades_skipped"] = trades_skipped
        result["max_positions"] = features.max_positions
    result["_trades"] = completed_trades
    result["_equity_curve"] = daily_equity
    return result


# ══════════════════════════════════════════════════════════════
# FEATURE TEST CONFIGS
# ══════════════════════════════════════════════════════════════

def get_independent_tests() -> dict:
    """Return configs that test each feature independently against baseline."""
    return {
        "0_baseline": make_feature_set("Baseline (3×SL/2×TP static)"),

        "A_daily_atr": make_feature_set("A: Daily ATR recompute",
            daily_atr_recompute=True),

        "B_ratchet": make_feature_set("B: Ratchet stop (2×ATR trail)",
            ratchet_stop=True, ratchet_atr_mult=2.0),

        "C_time_decay": make_feature_set("C: Time decay (3×→1.5× over 30d)",
            time_decay=True, decay_start_day=10, decay_end_mult=1.5),

        "F_breakeven": make_feature_set("F: Breakeven stop after +1×ATR",
            breakeven_stop=True, breakeven_atr_mult=1.0),

        "D_winner_sizing": make_feature_set("D: Winner sizing (lookback 20)",
            winner_sizing=True, winner_lookback=20),

        "E_vol_sizing": make_feature_set("E: Vol-adjusted sizing",
            vol_sizing=True),

        "AB_atr_ratchet": make_feature_set("A+B: Daily ATR + Ratchet",
            daily_atr_recompute=True, ratchet_stop=True, ratchet_atr_mult=2.0),

        "ABF_atr_ratchet_be": make_feature_set("A+B+F: ATR + Ratchet + Breakeven",
            daily_atr_recompute=True, ratchet_stop=True, ratchet_atr_mult=2.0,
            breakeven_stop=True, breakeven_atr_mult=1.0),

        "ABCF_full_dynamic": make_feature_set("A+B+C+F: Full dynamic exits",
            daily_atr_recompute=True, ratchet_stop=True, ratchet_atr_mult=2.0,
            time_decay=True, decay_start_day=10, decay_end_mult=1.5,
            breakeven_stop=True, breakeven_atr_mult=1.0),

        "ALL_features": make_feature_set("ALL(A-F): Full dynamic + sizing",
            daily_atr_recompute=True, ratchet_stop=True, ratchet_atr_mult=2.0,
            time_decay=True, decay_start_day=10, decay_end_mult=1.5,
            breakeven_stop=True, breakeven_atr_mult=1.0,
            winner_sizing=True, winner_lookback=20,
            vol_sizing=True),

        # ── New features G-K (tested independently) ──

        "G_r_multiple": make_feature_set("G: R-multiple trail (+1R→BE→+1R→+2R)",
            r_multiple_trail=True),

        "H_parabolic_sar": make_feature_set("H: Parabolic SAR exit",
            parabolic_sar=True),

        "I_time_cut": make_feature_set("I: Time cut (exit if <0.5R after 7d)",
            time_expectancy_cut=True, time_cut_days=7, time_cut_r_threshold=0.5),

        "I2_time_cut_lax": make_feature_set("I2: Time cut lax (<0R after 10d)",
            time_expectancy_cut=True, time_cut_days=10, time_cut_r_threshold=0.0),

        "J_partial_profit": make_feature_set("J: Partial profit (50% at +1R)",
            partial_profit=True, partial_close_pct=0.50, partial_trigger_r=1.0),

        "K_kelly_sizing": make_feature_set("K: Kelly criterion sizing",
            kelly_sizing=True, kelly_lookback=30, kelly_fraction=0.5),

        # ── Promising combos with vol sizing (E was best from round 1) ──

        "EG_vol_rmult": make_feature_set("E+G: Vol sizing + R-multiple trail",
            vol_sizing=True, r_multiple_trail=True),

        "EI_vol_timecut": make_feature_set("E+I: Vol sizing + Time cut 7d",
            vol_sizing=True, time_expectancy_cut=True, time_cut_days=7, time_cut_r_threshold=0.5),

        "EI2_vol_timecut_lax": make_feature_set("E+I2: Vol sizing + Time cut lax",
            vol_sizing=True, time_expectancy_cut=True, time_cut_days=10, time_cut_r_threshold=0.0),

        "EJ_vol_partial": make_feature_set("E+J: Vol sizing + Partial profit",
            vol_sizing=True, partial_profit=True, partial_close_pct=0.50, partial_trigger_r=1.0),

        "EGI_vol_rmult_timecut": make_feature_set("E+G+I: Vol + R-trail + Time cut",
            vol_sizing=True, r_multiple_trail=True,
            time_expectancy_cut=True, time_cut_days=7, time_cut_r_threshold=0.5),

        "EGIJ_best_combo": make_feature_set("E+G+I+J: Vol+R-trail+TimeCut+Partial",
            vol_sizing=True, r_multiple_trail=True,
            time_expectancy_cut=True, time_cut_days=7, time_cut_r_threshold=0.5,
            partial_profit=True, partial_close_pct=0.50, partial_trigger_r=1.0),
    }


def get_realistic_tests() -> dict:
    """Return configs with proper capital management (max concurrent positions, cash tracking)."""
    return {
        # ── Realistic baselines with different slot counts ──
        "R5_equal": make_feature_set("R: 5 slots equal",
            max_positions=5),

        "R8_equal": make_feature_set("R: 8 slots equal",
            max_positions=8),

        "R10_equal": make_feature_set("R: 10 slots equal",
            max_positions=10),

        # ── Quality-weighted (top pick gets 40%) ──
        "R5_qual": make_feature_set("R: 5 slots quality-weighted",
            max_positions=5, quality_weighted=True),

        "R8_qual": make_feature_set("R: 8 slots quality-weighted",
            max_positions=8, quality_weighted=True),

        "R10_qual": make_feature_set("R: 10 slots quality-weighted",
            max_positions=10, quality_weighted=True),

        # ── Best features (E+I2) with realistic capital ──
        "R5_ei2_equal": make_feature_set("R: 5 slots E+I2 equal",
            max_positions=5,
            vol_sizing=True, time_expectancy_cut=True,
            time_cut_days=10, time_cut_r_threshold=0.0),

        "R8_ei2_equal": make_feature_set("R: 8 slots E+I2 equal",
            max_positions=8,
            vol_sizing=True, time_expectancy_cut=True,
            time_cut_days=10, time_cut_r_threshold=0.0),

        "R5_ei2_qual": make_feature_set("R: 5 slots E+I2 quality",
            max_positions=5, quality_weighted=True,
            vol_sizing=True, time_expectancy_cut=True,
            time_cut_days=10, time_cut_r_threshold=0.0),

        "R8_ei2_qual": make_feature_set("R: 8 slots E+I2 quality",
            max_positions=8, quality_weighted=True,
            vol_sizing=True, time_expectancy_cut=True,
            time_cut_days=10, time_cut_r_threshold=0.0),

        "R10_ei2_qual": make_feature_set("R: 10 slots E+I2 quality",
            max_positions=10, quality_weighted=True,
            vol_sizing=True, time_expectancy_cut=True,
            time_cut_days=10, time_cut_r_threshold=0.0),
    }


# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════

def print_results(results: list[dict]):
    """Print comparison table."""
    print("\n" + "=" * 130)
    print("STRATEGY LAB V2 — INDEPENDENT FEATURE TESTING (day-by-day simulation)")
    print("=" * 130)

    hdr = f"  {'Feature Config':<40} {'Trades':>6} {'WR%':>6} {'NetAvg':>7} {'PF':>6} {'Hold':>5} {'Equity':>12} {'Ret%':>7} {'CAGR%':>7} {'MaxDD%':>7} {'Calmar':>7}"
    print(hdr)
    print("  " + "─" * 125)

    # Find best for each numeric metric
    metrics = ["win_rate", "avg_return_net", "profit_factor", "final_equity",
               "total_return_pct", "cagr_pct", "calmar"]
    bests = {}
    for m in metrics:
        vals = [(i, r.get(m, -999)) for i, r in enumerate(results)
                if isinstance(r.get(m), (int, float))]
        if vals:
            bests[m] = max(vals, key=lambda x: x[1])[0]
    # MaxDD: higher (less negative) is better
    dd_vals = [(i, r.get("max_drawdown_pct", -999)) for i, r in enumerate(results)
               if isinstance(r.get("max_drawdown_pct"), (int, float))]
    if dd_vals:
        bests["max_drawdown_pct"] = max(dd_vals, key=lambda x: x[1])[0]

    for i, r in enumerate(results):
        name = r["features"][:39]
        eq = r.get("final_equity", 0)
        eq_str = f"₹{eq/100000:,.1f}L" if eq > 0 else f"₹{eq:,.0f}"

        markers = []
        for m in metrics + ["max_drawdown_pct"]:
            if bests.get(m) == i:
                markers.append(m.split("_")[0][:3].upper())

        marker = " ◀" + ",".join(markers) if markers else ""

        print(f"  {name:<40} {r.get('n_trades', 0):>6} "
              f"{r.get('win_rate', 0):>6.1f} {r.get('avg_return_net', 0):>+7.2f} "
              f"{r.get('profit_factor', 0):>6} {r.get('avg_hold_days', 0):>5.1f} "
              f"{eq_str:>12} {r.get('total_return_pct', 0):>+7.1f} "
              f"{r.get('cagr_pct', 0):>+7.1f} {r.get('max_drawdown_pct', 0):>7.1f} "
              f"{r.get('calmar', 0):>7.2f}{marker}")
        if r.get("trades_skipped"):
            print(f"  {'':>40} (skipped {r['trades_skipped']} picks, max {r.get('max_positions', '?')} positions)")

    # Exit breakdown
    print(f"\n  {'Exit Breakdown':<40}", end="")
    for reason in ["stop_loss", "take_profit", "time_exit", "time_cut", "partial_profit"]:
        print(f"  {reason}:", end="")
    print()
    print("  " + "─" * 125)
    for r in results:
        name = r["features"][:39]
        eb = r.get("exit_breakdown", {})
        n = r.get("n_trades", 1)
        sl = eb.get("stop_loss", 0)
        tp = eb.get("take_profit", 0)
        te = eb.get("time_exit", 0)
        tc = eb.get("time_cut", 0)
        pp = eb.get("partial_profit", 0)
        parts = f"  {sl:>3} ({sl/n*100:>4.0f}%)      {tp:>3} ({tp/n*100:>4.0f}%)       {te:>3} ({te/n*100:>4.0f}%)"
        if tc > 0 or pp > 0:
            parts += f"      {tc:>3} ({tc/n*100:>4.0f}%)           {pp:>3} ({pp/n*100:>4.0f}%)"
        print(f"  {name:<40}{parts}")

    # By trigger type
    print(f"\n  {'By Trigger Type':<40}")
    print("  " + "─" * 125)
    all_tt = set()
    for r in results:
        all_tt.update(r.get("by_trigger", {}).keys())
    for tt in sorted(all_tt):
        print(f"  {tt}:")
        for r in results:
            bt = r.get("by_trigger", {}).get(tt, {})
            if bt:
                name = r["features"][:36]
                print(f"    {name:<36} n={bt['n']:>4} WR={bt['win_rate']:>5.1f}% "
                      f"Avg={bt['avg_return']:>+6.2f}% PF={bt['pf']:>5}")

    print("\n" + "=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Strategy Lab V2 — Feature Testing")
    parser.add_argument("--min-date", type=str, default=None)
    parser.add_argument("--max-date", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature keys (e.g. A_daily_atr,B_ratchet)")
    parser.add_argument("--realistic", action="store_true",
                        help="Run realistic capital-constrained tests (max positions, cash tracking)")
    parser.add_argument("--output", type=str, default=None)
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

    logger.info(f"Loaded {len(picks_df)} picks across {picks_df['trade_date'].nunique()} dates")

    # Fetch price data once
    logger.info("Fetching price data...")
    price_data = fetch_all_price_data(picks_df)

    # Select tests
    if args.realistic:
        all_tests = get_realistic_tests()
    else:
        all_tests = get_independent_tests()

    if args.features:
        keys = [k.strip() for k in args.features.split(",")]
        tests = {k: v for k, v in all_tests.items() if k in keys}
        # Always include baseline for comparison (only for non-realistic)
        if not args.realistic and "0_baseline" not in tests:
            tests = {"0_baseline": all_tests["0_baseline"], **tests}
    else:
        tests = all_tests

    # Run all tests
    results = []
    for key, features in tests.items():
        logger.info(f"  Testing: {features.name}")
        result = simulate_portfolio(picks_df, price_data, features)
        results.append(result)

    # Print
    print_results(results)

    # Save
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
