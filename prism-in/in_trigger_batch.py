#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
India Trigger Batch System

Surge stock detection system for Indian market (NSE/BSE).
Adapted from US trigger_batch.py for Indian market characteristics.

Key Differences from US Version:
- Data source: yfinance (.NS suffix) + nsetools bulk API
- Market cap filter: ₹10,000 Cr INR (~$1.2B USD) for large + upper mid-cap
- Trading value filter: ₹100 Cr INR (~$12M USD)
- Change rate filter: 20% max (NSE circuit limits: 5%/10%/20%)
- Market hours: 09:15-15:30 IST
- Unique: Delivery volume ratio (NSE-specific)

Usage:
    python prism-in/in_trigger_batch.py morning INFO --output trigger_results_in.json
    python prism-in/in_trigger_batch.py afternoon INFO --output trigger_results_in.json
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
import datetime
import logging
import json
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cores.in_surge_detector import (
    get_snapshot,
    get_previous_snapshot,
    get_volume_baseline,
    get_multi_day_ohlcv,
    get_major_tickers,
    get_ticker_name,
    get_nearest_business_day,
    apply_absolute_filters,
    normalize_and_score,
    enhance_dataframe,
)
from cores.in_stock_evaluator import (
    evaluate_stock,
    evaluate_batch,
    QUALITY_GATE_MIN,
    EvalResult,
)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


# Trigger-type specific criteria (rr_target used for agent fit scoring)
# sl_max retained as floor — ATR-based SL is used instead (3×ATR₁₄)
TRIGGER_CRITERIA = {
    "Volume Surge Top": {"rr_target": 1.2, "sl_max": 0.05},
    "Gap Up Momentum Top": {"rr_target": 1.2, "sl_max": 0.05},
    "Intraday Rise Top": {"rr_target": 1.2, "sl_max": 0.05},
    "Closing Strength Top": {"rr_target": 1.3, "sl_max": 0.05},
    "Value-to-Cap Ratio Top": {"rr_target": 1.3, "sl_max": 0.05},
    "Volume Surge Sideways": {"rr_target": 1.5, "sl_max": 0.07},
    "default": {"rr_target": 1.5, "sl_max": 0.07}
}

# Market cap filter: ₹10,000 Cr INR (large + upper mid-cap)
MIN_MARKET_CAP = 100_000_000_000  # ₹10,000 Cr = 100 billion

# Trading value filter: ₹100 Cr INR
MIN_TRADING_VALUE = 1_000_000_000  # ₹100 Cr = 1 billion

# Maximum number of stocks to select for report generation
MAX_FINAL_SELECTIONS = 5

# ── Structural filters (backtest-validated on Oct-Dec 2025, 224 trades) ──
# These are regime-independent: they work in bull, bear, and sideways markets.
MAX_TRIGGER_CHANGE = 6.0   # Don't chase: reject stocks already up > 6% today
MAX_ENTRY_RSI = 60.0       # Don't buy overbought: reject RSI > 60


def calculate_agent_fit_metrics(ticker: str, current_price: float, trade_date: str,
                                lookback_days: int = 10, trigger_type: str = None) -> dict:
    """
    Calculate metrics for buy/sell agent criteria.

    ATR-based exit method (backtest-validated: 3×ATR₁₄ SL, 2×ATR₁₄ TP):
    - Wide SL gives room for normal volatility (avg ~10%)
    - Tight TP captures the initial surge pop (avg ~7%)
    - Adaptive per-stock: volatile stocks get wider levels
    - TP/SL recalculated dynamically by position monitor using live ATR

    Args:
        ticker: NSE stock ticker
        current_price: Current price in INR
        trade_date: Reference trading date
        lookback_days: Number of past trading days
        trigger_type: Trigger type for differentiated criteria

    Returns:
        dict with agent fit metrics
    """
    ATR_PERIOD = 14
    ATR_SL_MULT = 3.0    # stop-loss = entry - 3×ATR₁₄
    ATR_TP_MULT = 2.0    # take-profit = entry + 2×ATR₁₄
    FALLBACK_SL_PCT = 0.10
    FALLBACK_TP_PCT = 0.08

    result = {
        "stop_loss_price": 0,
        "target_price": 0,
        "stop_loss_pct": 1.0,
        "risk_reward_ratio": 0,
        "agent_fit_score": 0,
        "atr_14": 0,
    }

    if current_price <= 0:
        return result

    criteria = TRIGGER_CRITERIA.get(trigger_type, TRIGGER_CRITERIA["default"])
    rr_target = criteria["rr_target"]

    # Compute ATR₁₄ from historical OHLCV data
    multi_day_df = get_multi_day_ohlcv(ticker, trade_date, lookback_days)
    atr = 0
    if not multi_day_df.empty and len(multi_day_df) >= 3 and "High" in multi_day_df.columns:
        hi = multi_day_df["High"].values
        lo = multi_day_df["Low"].values
        cl = multi_day_df["Close"].values
        tr_vals = []
        for ti in range(1, len(multi_day_df)):
            tr_vals.append(max(hi[ti] - lo[ti], abs(hi[ti] - cl[ti-1]), abs(lo[ti] - cl[ti-1])))
        if tr_vals:
            atr = float(np.mean(tr_vals[-ATR_PERIOD:]))

    # ATR-based stop-loss and take-profit
    if atr > 0:
        stop_loss_price = current_price - ATR_SL_MULT * atr
        target_price = current_price + ATR_TP_MULT * atr
        stop_loss_pct = (current_price - stop_loss_price) / current_price
    else:
        # Fallback to fixed percentages if ATR can't be computed
        stop_loss_price = current_price * (1 - FALLBACK_SL_PCT)
        target_price = current_price * (1 + FALLBACK_TP_PCT)
        stop_loss_pct = FALLBACK_SL_PCT
        logger.debug(f"{ticker}: No ATR data, using fallback {FALLBACK_SL_PCT*100:.0f}% SL / {FALLBACK_TP_PCT*100:.0f}% TP")

    # Risk/Reward
    potential_gain = target_price - current_price
    potential_loss = current_price - stop_loss_price

    if potential_loss > 0 and potential_gain > 0:
        risk_reward_ratio = potential_gain / potential_loss
    else:
        risk_reward_ratio = 0

    # Agent fit score
    rr_score = min(risk_reward_ratio / rr_target, 1.0) if risk_reward_ratio > 0 else 0
    sl_score = 1.0  # Fixed stop-loss always scores full
    agent_fit_score = rr_score * 0.6 + sl_score * 0.4

    result = {
        "stop_loss_price": stop_loss_price,
        "target_price": target_price,
        "stop_loss_pct": stop_loss_pct,
        "risk_reward_ratio": risk_reward_ratio,
        "agent_fit_score": agent_fit_score,
        "atr_14": round(atr, 2),
    }

    logger.debug(f"{ticker}: ATR₁₄=₹{atr:.2f}, SL=₹{stop_loss_price:.2f} ({stop_loss_pct*100:.1f}%), "
                f"TP=₹{target_price:.2f} ({(target_price/current_price-1)*100:.1f}%), "
                f"R/R={risk_reward_ratio:.2f}, AgentScore={agent_fit_score:.3f}")

    return result


def score_candidates_by_agent_criteria(candidates_df: pd.DataFrame, trade_date: str,
                                       lookback_days: int = 10, trigger_type: str = None) -> pd.DataFrame:
    """Calculate agent criteria scores for all candidate stocks."""
    if candidates_df.empty:
        return candidates_df

    result_df = candidates_df.copy()
    result_df["StopLossPrice"] = 0.0
    result_df["TargetPrice"] = 0.0
    result_df["StopLossPct"] = 0.0
    result_df["RiskRewardRatio"] = 0.0
    result_df["AgentFitScore"] = 0.0

    total = len(result_df.index)
    for idx, ticker in enumerate(result_df.index):
        if idx % 5 == 0:
            logger.info(f"  Agent scoring progress: {idx}/{total} ({ticker})")
        current_price = result_df.loc[ticker, "Close"]
        if hasattr(current_price, 'item'):
            current_price = current_price.item()
        elif hasattr(current_price, 'iloc'):
            current_price = current_price.iloc[0]
        metrics = calculate_agent_fit_metrics(ticker, float(current_price), trade_date,
                                             lookback_days, trigger_type)
        result_df.loc[ticker, "StopLossPrice"] = metrics["stop_loss_price"]
        result_df.loc[ticker, "TargetPrice"] = metrics["target_price"]
        result_df.loc[ticker, "StopLossPct"] = metrics["stop_loss_pct"]
        result_df.loc[ticker, "RiskRewardRatio"] = metrics["risk_reward_ratio"]
        result_df.loc[ticker, "AgentFitScore"] = metrics["agent_fit_score"]

    return result_df


# === Morning Triggers (Market Open Snapshot) ===

def trigger_morning_volume_surge(trade_date: str, snapshot: pd.DataFrame,
                                 prev_snapshot: pd.DataFrame, cap_df: pd.DataFrame = None,
                                 baseline_df: pd.DataFrame = None,
                                 top_n: int = 10) -> pd.DataFrame:
    """
    [Morning Trigger 1] Volume Surge Top
    - Uses 20-day average volume as baseline (not just yesterday)
    - VolumeVsAvg: today's volume / 20-day avg (>= 1.5x required)
    - VolumeZScore: statistical significance of today's volume
    - Composite: VolumeVsAvg (40%) + VolumeZScore (30%) + AbsoluteVolume (30%)
    - Secondary: Only rising stocks (close > open)
    - Market cap filter: >= ₹10,000 Cr
    - Falls back to prev-day comparison if baseline unavailable
    """
    logger.debug("trigger_morning_volume_surge started")

    common = snapshot.index.intersection(prev_snapshot.index)
    snap = snapshot.loc[common].copy()
    prev = prev_snapshot.loc[common].copy()

    if cap_df is not None and not cap_df.empty:
        snap = snap.drop(columns=["MarketCap"], errors="ignore")
        snap = snap.merge(cap_df[["MarketCap"]], left_index=True, right_index=True, how="inner")
        snap = snap[snap["MarketCap"] >= MIN_MARKET_CAP]
        if snap.empty:
            return pd.DataFrame()

    snap = apply_absolute_filters(snap, min_trading_value=MIN_TRADING_VALUE)

    # Price metrics (always need prev_snapshot for these)
    snap["IntradayChange"] = (snap["Close"] / snap["Open"] - 1) * 100
    snap["DailyChange"] = ((snap["Close"] - prev.reindex(snap.index)["Close"]) / prev.reindex(snap.index)["Close"]) * 100
    snap = snap[snap["DailyChange"] <= 20.0]  # NSE circuit limit
    snap["IsRising"] = snap["Close"] > snap["Open"]

    # Volume surge metrics: prefer 20-day baseline, fall back to prev-day
    if baseline_df is not None and not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid = bl["AvgVolume"].notna() & (bl["AvgVolume"] > 0)
        snap["VolumeSurgeRatio"] = np.nan
        snap["VolumeZScore"] = 0.0
        if valid.any():
            snap.loc[valid, "VolumeSurgeRatio"] = snap.loc[valid, "Volume"] / bl.loc[valid, "AvgVolume"]
            std = bl.loc[valid, "StdVolume"].replace(0, np.nan)
            snap.loc[valid, "VolumeZScore"] = (
                (snap.loc[valid, "Volume"] - bl.loc[valid, "AvgVolume"]) / std
            ).fillna(0)
        # Fill tickers without baseline using prev-day ratio
        no_bl = snap["VolumeSurgeRatio"].isna()
        if no_bl.any():
            prev_vol = prev.reindex(snap.index).loc[no_bl, "Volume"].replace(0, np.nan)
            snap.loc[no_bl, "VolumeSurgeRatio"] = snap.loc[no_bl, "Volume"] / prev_vol
        logger.info(f"Volume surge: 20-day baseline available for {valid.sum()}/{len(snap)} tickers")
    else:
        # Fallback: prev-day only
        snap["VolumeSurgeRatio"] = snap["Volume"] / prev.reindex(snap.index)["Volume"].replace(0, np.nan)
        snap["VolumeZScore"] = 0.0
        logger.info("Volume surge: no baseline available, using prev-day ratio")

    snap["VolumeIncreaseRate"] = (snap["VolumeSurgeRatio"] - 1) * 100  # For display

    # Filter: >= 1.5x baseline (or >= 1.3x prev if no baseline)
    threshold = 1.5 if (baseline_df is not None and not baseline_df.empty) else 1.3
    snap = snap[snap["VolumeSurgeRatio"] >= threshold]

    if snap.empty:
        return pd.DataFrame()

    # Composite score: VolumeVsAvg (40%) + ZScore (30%) + AbsVolume (30%)
    for col in ["VolumeSurgeRatio", "VolumeZScore", "Volume"]:
        cmax = snap[col].max()
        cmin = snap[col].min()
        crange = cmax - cmin if cmax > cmin else 1
        snap[f"{col}_norm"] = (snap[col] - cmin) / crange

    snap["CompositeScore"] = (
        snap["VolumeSurgeRatio_norm"] * 0.4 +
        snap["VolumeZScore_norm"] * 0.3 +
        snap["Volume_norm"] * 0.3
    )

    candidates = snap.sort_values("CompositeScore", ascending=False).head(top_n)
    result = candidates[candidates["IsRising"] == True].copy()

    if result.empty:
        return pd.DataFrame()

    logger.debug(f"Volume surge detected: {len(result)} stocks")
    return enhance_dataframe(result.sort_values("CompositeScore", ascending=False).head(15))


def trigger_morning_gap_up_momentum(trade_date: str, snapshot: pd.DataFrame,
                                    prev_snapshot: pd.DataFrame, cap_df: pd.DataFrame = None,
                                    baseline_df: pd.DataFrame = None,
                                    top_n: int = 15) -> pd.DataFrame:
    """
    [Morning Trigger 2] Gap Up Momentum Top
    - GapVsATR: gap size relative to 20-day average daily range (ATR-like)
      A 2% gap on a stock that normally moves 1% is 2x ATR (significant).
      A 2% gap on a stock that normally moves 3% is 0.67x ATR (noise).
    - Composite: GapVsATR (35%) + IntradayChange (30%) + TradingValue (20%) + GapUpRate (15%)
    - Secondary: Only momentum continuing (close > open)
    - Market cap: >= ₹10,000 Cr
    """
    logger.debug("trigger_morning_gap_up_momentum started")

    common = snapshot.index.intersection(prev_snapshot.index)
    snap = snapshot.loc[common].copy()
    prev = prev_snapshot.loc[common].copy()

    if cap_df is not None and not cap_df.empty:
        snap = snap.drop(columns=["MarketCap"], errors="ignore")
        snap = snap.merge(cap_df[["MarketCap"]], left_index=True, right_index=True, how="inner")
        snap = snap[snap["MarketCap"] >= MIN_MARKET_CAP]
        if snap.empty:
            return pd.DataFrame()

    snap = apply_absolute_filters(snap, min_trading_value=MIN_TRADING_VALUE)

    snap["GapUpRate"] = (snap["Open"] / prev.reindex(snap.index)["Close"] - 1) * 100
    snap["IntradayChange"] = (snap["Close"] / snap["Open"] - 1) * 100
    snap["DailyChange"] = ((snap["Close"] - prev.reindex(snap.index)["Close"]) / prev.reindex(snap.index)["Close"]) * 100
    snap["MomentumContinuing"] = snap["Close"] > snap["Open"]

    snap = snap[(snap["GapUpRate"] >= 1.0) & (snap["DailyChange"] <= 15.0)]

    if snap.empty:
        return pd.DataFrame()

    # GapVsATR: gap significance relative to normal price movement
    if baseline_df is not None and not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid = bl["AvgDailyRange"].notna() & (bl["AvgDailyRange"] > 0)
        snap["GapVsATR"] = snap["GapUpRate"]  # default: use raw gap
        if valid.any():
            snap.loc[valid, "GapVsATR"] = snap.loc[valid, "GapUpRate"].abs() / bl.loc[valid, "AvgDailyRange"]
        logger.info(f"Gap-up: ATR baseline for {valid.sum()}/{len(snap)} tickers")
    else:
        snap["GapVsATR"] = snap["GapUpRate"]  # fallback: raw gap rate

    # Volume confirmation: use as score boost, not standalone trigger
    # Stocks with above-average volume get up to +15% score bonus
    snap["VolumeBoost"] = 0.0
    if baseline_df is not None and not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        vol_valid = bl["AvgVolume"].notna() & (bl["AvgVolume"] > 0)
        if vol_valid.any():
            vol_ratio = snap.loc[vol_valid, "Volume"] / bl.loc[vol_valid, "AvgVolume"]
            # Boost: 0 if vol <= avg, up to 0.15 if vol >= 3x avg
            snap.loc[vol_valid, "VolumeBoost"] = ((vol_ratio - 1).clip(lower=0) / 2).clip(upper=0.15)

    for col in ["GapVsATR", "IntradayChange", "Amount", "GapUpRate"]:
        col_max = snap[col].max()
        col_min = snap[col].min()
        col_range = col_max - col_min if col_max > col_min else 1
        snap[f"{col}_norm"] = (snap[col] - col_min) / col_range

    snap["CompositeScore"] = (
        snap["GapVsATR_norm"] * 0.35 +
        snap["IntradayChange_norm"] * 0.30 +
        snap["Amount_norm"] * 0.20 +
        snap["GapUpRate_norm"] * 0.15 +
        snap["VolumeBoost"]  # volume confirmation bonus (0 to +0.15)
    )

    candidates = snap.sort_values("CompositeScore", ascending=False).head(top_n)
    result = candidates[candidates["MomentumContinuing"] == True].copy()

    if result.empty:
        return pd.DataFrame()

    result["TotalMomentum"] = result["GapUpRate"] + result["IntradayChange"]
    logger.debug(f"Gap up momentum detected: {len(result)} stocks")
    return enhance_dataframe(result.sort_values("CompositeScore", ascending=False).head(15))


def trigger_morning_value_to_cap_ratio(trade_date: str, snapshot: pd.DataFrame,
                                       prev_snapshot: pd.DataFrame, cap_df: pd.DataFrame = None,
                                       baseline_df: pd.DataFrame = None,
                                       top_n: int = 10) -> pd.DataFrame:
    """
    [Morning Trigger 3] Value-to-Cap Ratio Top (Concentrated Capital Inflow)
    - AmountVsAvg: today's turnover vs. 20-day average (unusual capital inflow)
    - Composite: ValueCapRatio (30%) + AmountVsAvg (30%) + AbsValue (20%) + IntradayChange (20%)
    - Secondary: Only rising stocks
    """
    logger.info("Value-to-Cap ratio analysis started")

    if cap_df is None or cap_df.empty or 'MarketCap' not in cap_df.columns:
        logger.warning("Value-to-Cap ratio skipped: Market cap data not available")
        return pd.DataFrame()

    if snapshot.empty or prev_snapshot.empty:
        return pd.DataFrame()

    try:
        snap_clean = snapshot.drop(columns=["MarketCap"], errors="ignore")
        merged = snap_clean.merge(cap_df[["MarketCap"]], left_index=True, right_index=True, how="inner").copy()
        common = merged.index.intersection(prev_snapshot.index)
        if len(common) == 0:
            return pd.DataFrame()

        merged = merged.loc[common].copy()
        prev = prev_snapshot.loc[common].copy()

        merged = apply_absolute_filters(merged, min_trading_value=MIN_TRADING_VALUE)
        if merged.empty:
            return pd.DataFrame()

        merged["ValueCapRatio"] = (merged["Amount"] / merged["MarketCap"]) * 100
        merged["IntradayChange"] = (merged["Close"] / merged["Open"] - 1) * 100
        merged["DailyChange"] = ((merged["Close"] - prev["Close"]) / prev["Close"]) * 100
        merged["IsRising"] = merged["Close"] > merged["Open"]
        merged = merged[merged["DailyChange"] <= 20.0]
        merged = merged[merged["MarketCap"] >= MIN_MARKET_CAP]

        if merged.empty:
            return pd.DataFrame()

        # AmountVsAvg: today's turnover relative to 20-day average
        if baseline_df is not None and not baseline_df.empty:
            bl = baseline_df.reindex(merged.index)
            valid = bl["AvgAmount"].notna() & (bl["AvgAmount"] > 0)
            merged["AmountVsAvg"] = 1.0
            if valid.any():
                merged.loc[valid, "AmountVsAvg"] = merged.loc[valid, "Amount"] / bl.loc[valid, "AvgAmount"]
            logger.info(f"Value-to-cap: amount baseline for {valid.sum()}/{len(merged)} tickers")
        else:
            merged["AmountVsAvg"] = 1.0  # neutral when no baseline

        score_cols = ["ValueCapRatio", "AmountVsAvg", "Amount", "IntradayChange"]
        for col in score_cols:
            col_max = merged[col].max()
            col_min = merged[col].min()
            col_range = col_max - col_min if col_max > col_min else 1
            merged[f"{col}_norm"] = (merged[col] - col_min) / col_range

        # Volume confirmation boost (same as Gap Up)
        merged["VolumeBoost"] = 0.0
        if baseline_df is not None and not baseline_df.empty:
            bl_v = baseline_df.reindex(merged.index)
            vol_valid = bl_v["AvgVolume"].notna() & (bl_v["AvgVolume"] > 0)
            if vol_valid.any():
                vol_ratio = merged.loc[vol_valid, "Volume"] / bl_v.loc[vol_valid, "AvgVolume"]
                merged.loc[vol_valid, "VolumeBoost"] = ((vol_ratio - 1).clip(lower=0) / 2).clip(upper=0.15)

        merged["CompositeScore"] = (
            merged["ValueCapRatio_norm"] * 0.30 +
            merged["AmountVsAvg_norm"] * 0.30 +
            merged["Amount_norm"] * 0.20 +
            merged["IntradayChange_norm"] * 0.20 +
            merged["VolumeBoost"]  # volume confirmation bonus (0 to +0.15)
        )

        candidates = merged.sort_values("CompositeScore", ascending=False).head(top_n)
        result = candidates[candidates["IsRising"] == True].copy()

        if result.empty:
            return pd.DataFrame()

        logger.info(f"Value-to-cap analysis complete: {len(result)} stocks selected")
        return enhance_dataframe(result.sort_values("CompositeScore", ascending=False).head(15))

    except Exception as e:
        logger.error(f"Error in value-to-cap analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


# === Afternoon Triggers (Market Close Snapshot) ===

def trigger_afternoon_daily_rise_top(trade_date: str, snapshot: pd.DataFrame,
                                     prev_snapshot: pd.DataFrame, cap_df: pd.DataFrame = None,
                                     baseline_df: pd.DataFrame = None,
                                     top_n: int = 15) -> pd.DataFrame:
    """
    [Afternoon Trigger 1] Intraday Rise Top
    - VolumeVsAvg: volume confirmation against 20-day average
    - Composite: IntradayChange (40%) + VolumeVsAvg (30%) + TradingValue (30%)
    - Filter: 3% <= change <= 15%
    - Market cap: >= ₹10,000 Cr
    """
    logger.debug("trigger_afternoon_daily_rise_top started")

    common = snapshot.index.intersection(prev_snapshot.index)
    snap = snapshot.loc[common].copy()
    prev = prev_snapshot.loc[common].copy()

    if cap_df is not None and not cap_df.empty:
        snap = snap.drop(columns=["MarketCap"], errors="ignore")
        snap = snap.merge(cap_df[["MarketCap"]], left_index=True, right_index=True, how="inner")
        snap = snap[snap["MarketCap"] >= MIN_MARKET_CAP]
        if snap.empty:
            return pd.DataFrame()

    snap = apply_absolute_filters(snap.copy(), min_trading_value=MIN_TRADING_VALUE)

    snap["IntradayChange"] = (snap["Close"] / snap["Open"] - 1) * 100
    snap["DailyChange"] = ((snap["Close"] - prev.reindex(snap.index)["Close"]) / prev.reindex(snap.index)["Close"]) * 100

    snap = snap[(snap["DailyChange"] >= 3.0) & (snap["DailyChange"] <= 15.0)]

    if snap.empty:
        return pd.DataFrame()

    # Volume confirmation: rise backed by above-average volume is more reliable
    if baseline_df is not None and not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid = bl["AvgVolume"].notna() & (bl["AvgVolume"] > 0)
        snap["VolumeVsAvg"] = 1.0
        if valid.any():
            snap.loc[valid, "VolumeVsAvg"] = snap.loc[valid, "Volume"] / bl.loc[valid, "AvgVolume"]
    else:
        snap["VolumeVsAvg"] = snap["Volume"] / prev.reindex(snap.index)["Volume"].replace(0, np.nan)
        snap["VolumeVsAvg"] = snap["VolumeVsAvg"].fillna(1.0)

    for col in ["IntradayChange", "VolumeVsAvg", "Amount"]:
        col_max = snap[col].max()
        col_min = snap[col].min()
        col_range = col_max - col_min if col_max > col_min else 1
        snap[f"{col}_norm"] = (snap[col] - col_min) / col_range

    snap["CompositeScore"] = (
        snap["IntradayChange_norm"] * 0.40 +
        snap["VolumeVsAvg_norm"] * 0.30 +
        snap["Amount_norm"] * 0.30
    )

    result = snap.sort_values("CompositeScore", ascending=False).head(top_n).copy()

    logger.debug(f"Intraday rise top detected: {len(result)} stocks")
    return enhance_dataframe(result.head(15))


def trigger_afternoon_closing_strength(trade_date: str, snapshot: pd.DataFrame,
                                       prev_snapshot: pd.DataFrame, cap_df: pd.DataFrame = None,
                                       baseline_df: pd.DataFrame = None,
                                       top_n: int = 15) -> pd.DataFrame:
    """
    [Afternoon Trigger 2] Closing Strength Top
    - VolumeVsAvg: volume above 20-day average (replaces prev-day comparison)
    - Composite: ClosingStrength (40%) + VolumeVsAvg (30%) + TradingValue (30%)
    - Secondary: Only rising stocks (close > open)
    - Market cap: >= ₹10,000 Cr
    """
    logger.debug("trigger_afternoon_closing_strength started")

    common = snapshot.index.intersection(prev_snapshot.index)
    snap = snapshot.loc[common].copy()
    prev = prev_snapshot.loc[common].copy()

    if cap_df is not None and not cap_df.empty:
        snap = snap.drop(columns=["MarketCap"], errors="ignore")
        snap = snap.merge(cap_df[["MarketCap"]], left_index=True, right_index=True, how="inner")
        snap = snap[snap["MarketCap"] >= MIN_MARKET_CAP]
        if snap.empty:
            return pd.DataFrame()

    snap = apply_absolute_filters(snap, min_trading_value=MIN_TRADING_VALUE)

    snap["ClosingStrength"] = 0.0
    valid_range = (snap["High"] != snap["Low"])
    snap.loc[valid_range, "ClosingStrength"] = (
        (snap.loc[valid_range, "Close"] - snap.loc[valid_range, "Low"]) /
        (snap.loc[valid_range, "High"] - snap.loc[valid_range, "Low"])
    )

    snap["IntradayChange"] = (snap["Close"] / snap["Open"] - 1) * 100
    snap["DailyChange"] = ((snap["Close"] - prev.reindex(snap.index)["Close"]) / prev.reindex(snap.index)["Close"]) * 100
    snap["IsRising"] = snap["Close"] > snap["Open"]

    # Volume confirmation: use 20-day avg baseline instead of prev-day boolean
    if baseline_df is not None and not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid_bl = bl["AvgVolume"].notna() & (bl["AvgVolume"] > 0)
        snap["VolumeVsAvg"] = 1.0
        if valid_bl.any():
            snap.loc[valid_bl, "VolumeVsAvg"] = snap.loc[valid_bl, "Volume"] / bl.loc[valid_bl, "AvgVolume"]
        # Filter: volume must be at least 20% above 20-day average
        candidates = snap[snap["VolumeVsAvg"] >= 1.2].copy()
    else:
        # Fallback: prev-day comparison
        snap["VolumeVsAvg"] = snap["Volume"] / prev.reindex(snap.index)["Volume"].replace(0, np.nan)
        snap["VolumeVsAvg"] = snap["VolumeVsAvg"].fillna(1.0)
        candidates = snap[snap["VolumeVsAvg"] > 1.0].copy()

    if candidates.empty:
        return pd.DataFrame()

    for col in ["ClosingStrength", "VolumeVsAvg", "Amount"]:
        col_max = candidates[col].max()
        col_min = candidates[col].min()
        col_range = col_max - col_min if col_max > col_min else 1
        candidates[f"{col}_norm"] = (candidates[col] - col_min) / col_range

    candidates["CompositeScore"] = (
        candidates["ClosingStrength_norm"] * 0.40 +
        candidates["VolumeVsAvg_norm"] * 0.30 +
        candidates["Amount_norm"] * 0.30
    )

    candidates = candidates.sort_values("CompositeScore", ascending=False).head(top_n)
    result = candidates[candidates["IsRising"] == True].copy()

    if result.empty:
        return pd.DataFrame()

    logger.debug(f"Closing strength top detected: {len(result)} stocks")
    return enhance_dataframe(result.sort_values("CompositeScore", ascending=False).head(15))


def trigger_afternoon_volume_surge_flat(trade_date: str, snapshot: pd.DataFrame,
                                        prev_snapshot: pd.DataFrame, cap_df: pd.DataFrame = None,
                                        baseline_df: pd.DataFrame = None,
                                        top_n: int = 20) -> pd.DataFrame:
    """
    [Afternoon Trigger 3] Volume Surge Sideways (Consolidation Stocks)
    - VolumeSurgeRatio: today's volume / 20-day avg (>= 2x required)
    - Composite: VolumeSurgeRatio (40%) + VolumeZScore (30%) + TradingValue (30%)
    - Secondary: Sideways only (change within ±5%)
    - Market cap: >= ₹10,000 Cr
    """
    logger.debug("trigger_afternoon_volume_surge_flat started")

    common = snapshot.index.intersection(prev_snapshot.index)
    snap = snapshot.loc[common].copy()
    prev = prev_snapshot.loc[common].copy()

    if cap_df is not None and not cap_df.empty:
        snap = snap.drop(columns=["MarketCap"], errors="ignore")
        snap = snap.merge(cap_df[["MarketCap"]], left_index=True, right_index=True, how="inner")
        snap = snap[snap["MarketCap"] >= MIN_MARKET_CAP]
        if snap.empty:
            return pd.DataFrame()

    snap = apply_absolute_filters(snap, min_trading_value=MIN_TRADING_VALUE)

    snap["IntradayChange"] = (snap["Close"] / snap["Open"] - 1) * 100
    snap["DailyChange"] = ((snap["Close"] - prev.reindex(snap.index)["Close"]) / prev.reindex(snap.index)["Close"]) * 100
    snap["IsSideways"] = (snap["DailyChange"].abs() <= 5)

    # Volume surge: prefer 20-day baseline, fall back to prev-day
    if baseline_df is not None and not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid = bl["AvgVolume"].notna() & (bl["AvgVolume"] > 0)
        snap["VolumeSurgeRatio"] = np.nan
        snap["VolumeZScore"] = 0.0
        if valid.any():
            snap.loc[valid, "VolumeSurgeRatio"] = snap.loc[valid, "Volume"] / bl.loc[valid, "AvgVolume"]
            std = bl.loc[valid, "StdVolume"].replace(0, np.nan)
            snap.loc[valid, "VolumeZScore"] = (
                (snap.loc[valid, "Volume"] - bl.loc[valid, "AvgVolume"]) / std
            ).fillna(0)
        no_bl = snap["VolumeSurgeRatio"].isna()
        if no_bl.any():
            snap.loc[no_bl, "VolumeSurgeRatio"] = snap.loc[no_bl, "Volume"] / prev.reindex(snap.index).loc[no_bl, "Volume"].replace(0, np.nan)
    else:
        snap["VolumeSurgeRatio"] = snap["Volume"] / prev.reindex(snap.index)["Volume"].replace(0, np.nan)
        snap["VolumeZScore"] = 0.0

    snap["VolumeIncreaseRate"] = (snap["VolumeSurgeRatio"] - 1) * 100  # For display

    # Filter: sideways consolidation with significant volume (>= 2x avg or 1.5x prev)
    threshold = 2.0 if (baseline_df is not None and not baseline_df.empty) else 1.5
    snap = snap[snap["VolumeSurgeRatio"] >= threshold]

    if snap.empty:
        return pd.DataFrame()

    for col in ["VolumeSurgeRatio", "VolumeZScore", "Amount"]:
        cmax = snap[col].max()
        cmin = snap[col].min()
        crange = cmax - cmin if cmax > cmin else 1
        snap[f"{col}_norm"] = (snap[col] - cmin) / crange

    snap["CompositeScore"] = (
        snap["VolumeSurgeRatio_norm"] * 0.40 +
        snap["VolumeZScore_norm"] * 0.30 +
        snap["Amount_norm"] * 0.30
    )

    candidates = snap.sort_values("CompositeScore", ascending=False).head(top_n)
    result = candidates[candidates["IsSideways"] == True].copy()

    if result.empty:
        return pd.DataFrame()

    logger.debug(f"Volume surge sideways detected: {len(result)} stocks")
    return enhance_dataframe(result.sort_values("CompositeScore", ascending=False).head(15))


# === Final Selection ===

def select_final_tickers(triggers: dict, trade_date: str = None, use_hybrid: bool = True,
                         lookback_days: int = 10, max_selections: int = None,
                         regime_min_quality: float = None,
                         regime_max_change: float = None,
                         regime_max_rsi: float = None) -> dict:
    """
    Aggregate and select final stocks from all triggers using 3-factor scoring.

    Pipeline:
        1. Compute CompositeScore (momentum signal) — from trigger algorithms
        2. Compute AgentFitScore (R/R viability) — from calculate_agent_fit_metrics()
        3. Compute QualityScore (fundamental + technical) — from stock evaluator
        4. Hard filter: QualityScore < QUALITY_GATE_MIN (40) → REJECT
        5. Three-factor FinalScore:
             FinalScore = CompositeScore_norm × 0.20
                        + AgentFitScore       × 0.40
                        + QualityScore_norm   × 0.40
        6. Select top MAX_FINAL_SELECTIONS (7) unique tickers

    Scoring weights rationale:
        - Momentum (20%): Timing signal — why we noticed the stock today
        - R/R viability (40%): Trade math — stop-loss, target, risk/reward
        - Quality (40%): Is the stock fundamentally and technically sound?
          Prevents garbage stocks (negative margins, death cross, etc.) from
          consuming expensive LLM report generation.
    """
    max_picks = max_selections if max_selections is not None else MAX_FINAL_SELECTIONS
    # Default regime filters to module constants if not overridden
    if regime_min_quality is None:
        regime_min_quality = QUALITY_GATE_MIN
    if regime_max_change is None:
        regime_max_change = MAX_TRIGGER_CHANGE
    if regime_max_rsi is None:
        regime_max_rsi = MAX_ENTRY_RSI

    final_result = {}
    trigger_candidates = {}
    all_tickers = set()

    for name, df in triggers.items():
        if not df.empty:
            trigger_candidates[name] = df.copy()
            all_tickers.update(df.index.tolist())

    if not trigger_candidates:
        logger.warning("No candidates from any trigger")
        return final_result, {}

    if use_hybrid and trade_date:
        logger.info(f"Hybrid selection mode - calculating agent scores + quality evaluation")

        # ── Step 1-2: Agent fit scoring (stop-loss, target, R/R) ──
        for name, candidates_df in trigger_candidates.items():
            scored_df = score_candidates_by_agent_criteria(candidates_df, trade_date,
                                                         lookback_days, trigger_type=name)
            trigger_candidates[name] = scored_df

        # ── Step 3: Quality evaluation (fundamentals + technicals) ──
        # Collect unique tickers across all triggers (top 10 per trigger max)
        tickers_to_evaluate = set()
        for name, df in trigger_candidates.items():
            if "AgentFitScore" in df.columns:
                # Prefer sorting by preliminary score to evaluate most promising first
                sorted_idx = df.sort_values("AgentFitScore", ascending=False).index[:10]
            else:
                sorted_idx = df.index[:10]
            tickers_to_evaluate.update(sorted_idx.tolist())

        logger.info(f"Quality evaluation: {len(tickers_to_evaluate)} unique candidates")
        eval_results = {}
        import time as _eval_time
        for eval_idx, ticker in enumerate(tickers_to_evaluate):
            # Rate limit: pause between yfinance calls to avoid 429 errors
            if eval_idx > 0:
                _eval_time.sleep(1.5)

            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    ev = evaluate_stock(ticker)
                    # Check if it's a rate-limit error disguised as success
                    if ev.signal == "ERROR" and "rate" in str(ev.summary_reasons).lower():
                        raise Exception("Rate limited")
                    eval_results[ticker] = ev
                    gate_status = "PASS" if ev.composite_score >= QUALITY_GATE_MIN else "REJECT"
                    logger.info(f"  Quality [{gate_status}] {ticker}: "
                               f"{ev.signal} {ev.composite_score:.0f}/100 "
                               f"(V:{ev.valuation_score:.0f} G:{ev.growth_score:.0f} "
                               f"P:{ev.profitability_score:.0f} T:{ev.technical_score:.0f} "
                               f"A:{ev.analyst_score:.0f} R:{ev.risk_score:.0f}) "
                               f"{'— ' + ', '.join(ev.summary_reasons[:3]) if ev.summary_reasons else ''}")
                    break  # success
                except Exception as e:
                    if attempt < max_retries and ("rate" in str(e).lower() or "429" in str(e) or "too many" in str(e).lower()):
                        wait = 5 * (attempt + 1)
                        logger.warning(f"  Rate limited on {ticker}, waiting {wait}s (retry {attempt + 1}/{max_retries})")
                        _eval_time.sleep(wait)
                    else:
                        logger.warning(f"  Quality evaluation failed for {ticker}: {e}")
                        # On error, assign neutral score (don't block the pipeline)
                        eval_results[ticker] = EvalResult(
                            ticker=ticker, composite_score=50.0, signal="HOLD",
                            summary_reasons=[f"Eval error: {e}"]
                        )
                        break

        # ── Step 4-5: Quality gate + 3-factor scoring ──
        for name, scored_df in trigger_candidates.items():
            # Add quality columns
            scored_df["QualityScore"] = 50.0  # Default neutral
            scored_df["QualitySignal"] = "HOLD"
            scored_df["QualityReasons"] = ""

            for ticker in scored_df.index:
                if ticker in eval_results:
                    ev = eval_results[ticker]
                    scored_df.loc[ticker, "QualityScore"] = ev.composite_score
                    scored_df.loc[ticker, "QualitySignal"] = ev.signal
                    scored_df.loc[ticker, "QualityReasons"] = "; ".join(ev.summary_reasons[:3])

            # Hard filter: reject fundamentally broken stocks (regime-adjusted)
            before_count = len(scored_df)
            scored_df = scored_df[scored_df["QualityScore"] >= regime_min_quality]
            rejected = before_count - len(scored_df)
            if rejected > 0:
                logger.info(f"  [{name}] Quality gate rejected {rejected}/{before_count} candidates "
                           f"(score < {regime_min_quality})")

            # ── Structural filters (regime-adjusted) ──
            # Filter 1: Don't chase — reject stocks that already moved too much today
            if "DailyChange" in scored_df.columns:
                before_chase = len(scored_df)
                scored_df = scored_df[scored_df["DailyChange"] <= regime_max_change]
                chase_rejected = before_chase - len(scored_df)
                if chase_rejected > 0:
                    logger.info(f"  [{name}] Chase filter rejected {chase_rejected}/{before_chase} "
                               f"(daily change > {regime_max_change}%)")

            # Filter 2: Don't buy overbought — reject RSI above regime threshold
            for ticker in scored_df.index:
                if ticker in eval_results:
                    ev = eval_results[ticker]
                    rsi = getattr(ev, 'rsi_14', None)
                    if rsi is not None and rsi > regime_max_rsi:
                        scored_df = scored_df.drop(ticker, errors='ignore')
                        logger.info(f"  [{name}] RSI filter rejected {ticker} (RSI={rsi:.0f} > {MAX_ENTRY_RSI})")

            if scored_df.empty:
                trigger_candidates[name] = scored_df
                continue

            # Three-factor scoring
            if "CompositeScore" in scored_df.columns and "AgentFitScore" in scored_df.columns:
                # Use absolute CompositeScore (already 0-1 from trigger algorithms)
                # Don't normalize within trigger — single-candidate triggers get zeroed out
                scored_df["CompositeScore_norm"] = scored_df["CompositeScore"].clip(0, 1.5)

                # Normalize QualityScore to 0-1 (absolute scale: divide by 100)
                scored_df["QualityScore_norm"] = scored_df["QualityScore"] / 100.0

                # FinalScore: momentum-heavy, quality as tiebreaker
                scored_df["FinalScore"] = (
                    scored_df["CompositeScore_norm"] * 0.75    # Momentum signal (primary)
                    + scored_df["QualityScore_norm"] * 0.25    # Quality tiebreaker (secondary)
                )
                scored_df = scored_df.sort_values("FinalScore", ascending=False)

                for ticker in scored_df.index[:max_picks]:
                    company = scored_df.loc[ticker, "CompanyName"] if "CompanyName" in scored_df.columns else ""
                    qs = scored_df.loc[ticker, "QualityScore"]
                    q_sig = scored_df.loc[ticker, "QualitySignal"]
                    logger.info(f"  [{name}] {ticker} ({company}): "
                               f"Final={scored_df.loc[ticker, 'FinalScore']:.3f}, "
                               f"R/R={scored_df.loc[ticker, 'RiskRewardRatio']:.2f}, "
                               f"Quality={qs:.0f} ({q_sig})")

            trigger_candidates[name] = scored_df

    selected_tickers = set()
    score_column = "FinalScore" if use_hybrid and trade_date else "CompositeScore"

    # Single-pool ranking: merge ALL candidates from ALL triggers, pick top N by score.
    # No diversity rule — if one trigger dominates, that's where the edge is.
    # 2024 data showed Gap Up PF=1.71 vs V2C PF=0.95. Forcing V2C picks hurt returns.
    all_candidates = []
    for name, df in trigger_candidates.items():
        for ticker in df.index:
            if ticker not in selected_tickers:
                score = df.loc[ticker, score_column] if score_column in df.columns else 0
                all_candidates.append((name, ticker, score, df.loc[[ticker]]))
    all_candidates.sort(key=lambda x: x[2], reverse=True)

    for trigger_name, ticker, score, ticker_df in all_candidates:
        if ticker in selected_tickers or len(selected_tickers) >= max_picks:
            continue
        if score < 0.2:
            logger.info(f"[{trigger_name}] Skipping {ticker} — FinalScore {score:.3f} too low")
            continue
        if trigger_name in final_result:
            final_result[trigger_name] = pd.concat([final_result[trigger_name], ticker_df])
        else:
            final_result[trigger_name] = ticker_df
        selected_tickers.add(ticker)
        logger.info(f"[{trigger_name}] Selected: {ticker} (score={score:.3f})")

    logger.info(f"Final selections: {len(selected_tickers)}/{max_picks} stocks")
    return final_result, (eval_results if use_hybrid and trade_date else {})


# === Batch Execution ===

def run_batch(trigger_time: str, log_level: str = "INFO", output_file: str = None, reference_date: str = None):
    """
    Execute trigger batch for Indian market.

    Args:
        trigger_time: "morning" or "afternoon"
        log_level: Logging level
        output_file: Path to save results as JSON
        reference_date: Override date in YYYYMMDD format (default: today IST)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    ch.setLevel(numeric_level)

    # Use IST for date calculation
    ist = ZoneInfo("Asia/Kolkata")
    if reference_date:
        today_str = reference_date
    else:
        today_str = datetime.datetime.now(tz=ist).strftime("%Y%m%d")
    trade_date = get_nearest_business_day(today_str, prev=True)
    logger.info(f"Batch reference date: {trade_date} (IST)")

    # Get NIFTY 500 tickers
    import time as _time
    t0 = _time.time()
    tickers = get_major_tickers()
    logger.info(f"Ticker list: {len(tickers)} tickers loaded in {_time.time() - t0:.1f}s")

    t0 = _time.time()
    try:
        snapshot = get_snapshot(trade_date, tickers)
    except ValueError as e:
        logger.error(f"Snapshot retrieval failed: {e}")
        trade_date = get_nearest_business_day(
            (datetime.datetime.strptime(trade_date, '%Y%m%d') - datetime.timedelta(days=1)).strftime('%Y%m%d'),
            prev=True
        )
        logger.info(f"Retry with date: {trade_date}")
        snapshot = get_snapshot(trade_date, tickers)
    logger.info(f"Current snapshot: {len(snapshot)} stocks in {_time.time() - t0:.1f}s")

    t0 = _time.time()
    prev_snapshot, prev_date = get_previous_snapshot(trade_date, tickers)
    logger.info(f"Previous snapshot ({prev_date}): {len(prev_snapshot)} stocks in {_time.time() - t0:.1f}s")

    # Fix: Correct prev_snapshot Close using snapshot's PreviousClose from nsetools.
    # When nsetools provides the current snapshot, PreviousClose is the official
    # exchange previous close. The yfinance-based prev_snapshot often returns the
    # same day's data (DailyChange=0) due to timezone/availability issues.
    if "PreviousClose" in snapshot.columns:
        common_idx = snapshot.index.intersection(prev_snapshot.index)
        if len(common_idx) > 0:
            prev_close_vals = snapshot.loc[common_idx, "PreviousClose"]
            current_close_vals = snapshot.loc[common_idx, "Close"]
            # Only correct where PreviousClose exists, is positive, and differs from Close
            valid_mask = (prev_close_vals > 0) & (prev_close_vals != current_close_vals)
            if valid_mask.any():
                valid_idx = valid_mask[valid_mask].index
                prev_snapshot.loc[valid_idx, "Close"] = snapshot.loc[valid_idx, "PreviousClose"]
                logger.info(f"Corrected prev_snapshot Close for {len(valid_idx)}/{len(common_idx)} "
                           f"stocks using nsetools PreviousClose")
            else:
                logger.warning("PreviousClose == Close for all stocks — "
                              "snapshot may be stale or market hasn't moved")

    # 20-day volume/price baseline for robust surge detection
    t0 = _time.time()
    baseline_df = pd.DataFrame()
    try:
        baseline_df = get_volume_baseline(tickers, trade_date, window=20)
        logger.info(f"Volume baseline (20-day): {len(baseline_df)} stocks in {_time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"Volume baseline failed (will use prev-day fallback): {e}")

    # Market cap filtering for NIFTY 500 (unlike S&P 500, not all are large-cap)
    # Strategy: Use NSE snapshot MarketCap if available, then fill gaps via yfinance batch
    cap_df = None
    try:
        # First: check if snapshot already has MarketCap from nsetools bulk API
        if "MarketCap" in snapshot.columns and (snapshot["MarketCap"] > 0).sum() > 50:
            cap_df = snapshot[snapshot["MarketCap"] > 0][["MarketCap"]].copy()
            logger.info(f"Market cap from NSE snapshot: {len(cap_df)} stocks")
        else:
            # Fallback: yfinance batch — only top 50 by volume to limit time
            import yfinance as yf
            # Sort by volume desc, take top 50 (most liquid = most likely large-cap)
            if "Volume" in snapshot.columns:
                snapshot_tickers = list(snapshot.sort_values("Volume", ascending=False).index[:50])
            else:
                snapshot_tickers = list(snapshot.index[:50])
            cap_data = {}
            logger.info(f"Fetching market cap via yfinance for {len(snapshot_tickers)} tickers (fallback)...")
            for i, t in enumerate(snapshot_tickers):
                if i % 10 == 0:
                    logger.info(f"  Market cap progress: {i}/{len(snapshot_tickers)}")
                try:
                    ticker_obj = yf.Ticker(f"{t}.NS")
                    # Use fast_info when available (faster than .info)
                    try:
                        mcap = ticker_obj.fast_info.get("marketCap", 0)
                    except Exception:
                        mcap = ticker_obj.info.get("marketCap", 0)
                    if mcap:
                        cap_data[t] = {"MarketCap": mcap}
                except Exception:
                    continue
            logger.info(f"  Market cap progress: {len(snapshot_tickers)}/{len(snapshot_tickers)} done")
            if cap_data:
                cap_df = pd.DataFrame.from_dict(cap_data, orient="index")
                logger.info(f"Market cap from yfinance: {len(cap_df)} stocks")
    except Exception as e:
        logger.warning(f"Market cap loading failed: {e}")

    # ── Market Regime Check ──
    # Scale picks based on NIFTY 50 trend. Don't blindly block — adjust aggressiveness.
    # Bear market bottom (RSI < 30) is actually the BEST time to buy quality stocks.
    # Uses trade_date (not today) so historical simulations get correct regime.
    import yfinance as yf_regime
    regime = "bull"  # default
    nifty_rsi = 50.0  # default
    try:
        nifty = yf_regime.Ticker("^NSEI")
        # Fetch 1 year of data ending at trade_date (not today)
        td = datetime.datetime.strptime(trade_date, "%Y%m%d")
        nifty_start = (td - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
        nifty_end = (td + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        nifty_hist = nifty.history(start=nifty_start, end=nifty_end)
        if not nifty_hist.empty and len(nifty_hist) >= 50:
            close_series = nifty_hist["Close"].dropna()

            # Use completed daily candles only. During morning runs, today's bar can be partial.
            completed = close_series[close_series.index.date < td.date()]
            if len(completed) >= 50:
                calc_series = completed
            else:
                calc_series = close_series

            nifty_closes = calc_series.values
            nifty_price = float(nifty_closes[-1])
            nifty_sma50 = float(np.mean(nifty_closes[-50:]))
            nifty_sma200 = float(np.mean(nifty_closes[-200:])) if len(nifty_closes) >= 200 else nifty_sma50

            # 50MA slope: is the medium-term trend rising or falling?
            # Compare current 50MA vs 50MA from 20 trading days ago
            if len(nifty_closes) >= 70:
                nifty_sma50_20ago = float(np.mean(nifty_closes[-70:-20]))
                nifty_slope_rising = nifty_sma50 > nifty_sma50_20ago
                nifty_slope_pct = (nifty_sma50 / nifty_sma50_20ago - 1) * 100
            else:
                nifty_slope_rising = True
                nifty_slope_pct = 1.0

            # Realized volatility (20-day annualized)
            if len(nifty_closes) >= 21:
                nifty_rets = np.diff(np.log(nifty_closes[-21:]))
                nifty_rvol = float(np.std(nifty_rets) * np.sqrt(252) * 100)
            else:
                nifty_rvol = 15.0

            # Compute NIFTY RSI
            deltas = np.diff(nifty_closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            if len(gains) >= 14:
                avg_gain = np.mean(gains[:14])
                avg_loss = np.mean(losses[:14])
                for ii in range(14, len(gains)):
                    avg_gain = (avg_gain * 13 + gains[ii]) / 14
                    avg_loss = (avg_loss * 13 + losses[ii]) / 14
                nifty_rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0

            below_50ma = nifty_price < nifty_sma50
            below_200ma = nifty_price < nifty_sma200

            if below_50ma and below_200ma:
                if nifty_rsi <= 25:
                    regime = "bear_bottom_extreme"
                    logger.warning(f"🟢 REGIME: BEAR BOTTOM (EXTREME OVERSOLD) — NIFTY RSI={nifty_rsi:.0f} ≤ 25. "
                                  f"Rare opportunity! 5 picks, buy quality aggressively.")
                elif nifty_rsi <= 40:
                    regime = "bear_bottom"
                    logger.warning(f"🟡 REGIME: BEAR BOTTOM (OVERSOLD) — NIFTY RSI={nifty_rsi:.0f} ≤ 40. "
                                  f"Bounce likely. 3 picks, quality only.")
                else:
                    regime = "bear"
                    logger.warning(f"⛔ REGIME: BEAR — NIFTY ({nifty_price:,.0f}) below both MAs, RSI={nifty_rsi:.0f}. "
                                  f"Still falling. 0 picks.")
            elif below_50ma or below_200ma:
                regime = "correction"
                logger.warning(f"⚠️ REGIME: CORRECTION — NIFTY ({nifty_price:,.0f}) below {'50MA' if below_50ma else '200MA'}. "
                              f"RSI={nifty_rsi:.0f}. 0 picks.")
            else:
                # Above BOTH 50MA and 200MA — determine signal strength
                # Hybrid tiered system optimized across 2008-2026 (19 years, 1584 configs)
                strong_slope = nifty_slope_pct >= 1.0
                low_vol = nifty_rvol <= 13.0

                if strong_slope and low_vol:
                    regime = "bull_strong"
                    logger.info(f"✅ REGIME: BULL (STRONG) — NIFTY ({nifty_price:,.0f}) above both MAs, "
                               f"slope={nifty_slope_pct:+.1f}% ≥1%, RVol={nifty_rvol:.1f}% ≤13%. "
                               f"RSI={nifty_rsi:.0f} — 5 picks")
                elif strong_slope or low_vol:
                    regime = "bull_medium"
                    reason = f"slope={nifty_slope_pct:+.1f}%" if strong_slope else f"RVol={nifty_rvol:.1f}%"
                    logger.info(f"✅ REGIME: BULL (MEDIUM) — NIFTY ({nifty_price:,.0f}) above both MAs, "
                               f"{reason}. RSI={nifty_rsi:.0f} — 4 picks")
                elif nifty_slope_rising:
                    regime = "bull_weak"
                    logger.warning(f"⚠️ REGIME: BULL (WEAK) — NIFTY ({nifty_price:,.0f}) above both MAs, "
                                  f"but slope={nifty_slope_pct:+.1f}% <1% and RVol={nifty_rvol:.1f}% >13%. "
                                  f"RSI={nifty_rsi:.0f} — 0 picks, conditions too weak")
                else:
                    regime = "slope_declining"
                    logger.warning(f"⚠️ REGIME: SLOPE DECLINING — NIFTY ({nifty_price:,.0f}) above both MAs, "
                                  f"but slope falling ({nifty_slope_pct:+.1f}%). "
                                  f"RSI={nifty_rsi:.0f}. 0 picks.")
        else:
            logger.warning("Regime check: insufficient NIFTY data, assuming bull")
    except Exception as e:
        logger.warning(f"Regime check failed: {e} — assuming bull")

    # Adjust parameters based on regime
    # Hybrid tiered system optimized across 2008-2026 (19 years, 1584 configs)
    #
    # OPTIMAL CONFIG (rank #1 balanced score, 15/19 years positive, 10.8x G/L ratio):
    #   Bull strong (slope ≥1% + RVol ≤13%):   5 picks — best conditions
    #   Bull medium (slope ≥1% OR RVol ≤13%):  4 picks — one condition met
    #   Bull weak (slope <1% + RVol >13%):      0 picks — conditions too weak
    #   Slope declining:                         0 picks — trend exhaustion
    #   Correction (below 50MA or 200MA):        0 picks
    #   Bear (below both MAs, RSI>40):           0 picks
    #   Bear bottom (RSI ≤ 40):                  3 picks, quality ≥ 60
    #   Bear extreme (RSI ≤ 25):                 3 picks, quality ≥ 55
    #
    # Results: +1,733L total over 19 years, only -176L total losses

    regime_max_picks = MAX_FINAL_SELECTIONS  # default 5
    regime_max_change = MAX_TRIGGER_CHANGE   # default 6%
    regime_max_rsi = MAX_ENTRY_RSI           # default 60
    regime_min_quality = QUALITY_GATE_MIN    # default 40

    if regime == "bull_strong":
        regime_max_picks = 5
    elif regime == "bull_medium":
        regime_max_picks = 4
    elif regime in ("bull_weak", "slope_declining", "correction"):
        regime_max_picks = 0
        logger.info(f"  {regime}: 0 picks — conditions insufficient for trading.")
    elif regime == "bear":
        regime_max_picks = 0
        logger.info(f"  Bear (RSI={nifty_rsi:.0f} > 40): NO PICKS — still falling, preserve capital")
    elif regime == "bear_bottom":
        # Data-driven: quality≥60 selects 73% WR, +2.71% avg, 1.08× payoff (30 trades).
        # Rejected stocks: 52% WR, -0.12% avg — quality gate is doing real work.
        regime_max_picks = 3
        regime_max_change = 4.0
        regime_max_rsi = 55.0
        regime_min_quality = 60
        logger.info(f"  Bear bottom (RSI={nifty_rsi:.0f}): {regime_max_picks} picks, 30-day recovery play, "
                    f"quality>={regime_min_quality}")
    elif regime == "bear_bottom_extreme":
        # Data-driven: 17 trades at 88% WR, +6.75% avg. Even rejected stocks 86% WR.
        # In extreme oversold (RSI≤25), everything bounces — lower quality gate.
        regime_max_picks = 3
        regime_max_change = 6.0
        regime_max_rsi = 60.0
        regime_min_quality = QUALITY_GATE_MIN  # 40 — at RSI≤25 everything recovers
        logger.info(f"  EXTREME OVERSOLD (RSI={nifty_rsi:.0f}): {regime_max_picks} picks, "
                    f"30-day hold for recovery capture")

    if regime in ("bear", "slope_declining", "correction", "bull_weak"):
        label = regime.replace("_", " ").title()
        logger.warning(f"{label} — skipping all triggers. Returning empty.")
        if output_file:
            output_data = {
                "metadata": {
                    "run_time": datetime.datetime.now().isoformat(),
                    "trigger_mode": trigger_time,
                    "trade_date": trade_date,
                    "regime": regime,
                    "nifty_rsi": round(nifty_rsi, 1),
                    "nifty_slope_pct": round(nifty_slope_pct, 2) if 'nifty_slope_pct' in dir() else None,
                    "nifty_rvol": round(nifty_rvol, 1) if 'nifty_rvol' in dir() else None,
                    "message": f"No picks — {regime}. Capital preservation mode.",
                },
                "screening_summary": {"total_tickers_scanned": 0, "triggers": {}},
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        return {}

    if trigger_time == "morning" or trigger_time == "midday":
        label = "Midday" if trigger_time == "midday" else "Morning"
        logger.info(f"=== {label} Batch Execution (NSE) ===")
        # Volume Surge demoted from standalone trigger to confirmation filter.
        # Backtest showed PF=0.51 (worst trigger) — buying after volume spike = chasing.
        # Volume data is still used inside Gap Up and Value-to-Cap via baseline_df.
        res2 = trigger_morning_gap_up_momentum(trade_date, snapshot, prev_snapshot, cap_df, baseline_df)
        res3 = trigger_morning_value_to_cap_ratio(trade_date, snapshot, prev_snapshot, cap_df, baseline_df)
        triggers = {
            "Gap Up Momentum Top": res2,
            "Value-to-Cap Ratio Top": res3
        }
    elif trigger_time == "afternoon":
        logger.info("=== Afternoon Batch Execution (NSE) ===")
        res1 = trigger_afternoon_daily_rise_top(trade_date, snapshot, prev_snapshot, cap_df, baseline_df)
        res2 = trigger_afternoon_closing_strength(trade_date, snapshot, prev_snapshot, cap_df, baseline_df)
        res3 = trigger_afternoon_volume_surge_flat(trade_date, snapshot, prev_snapshot, cap_df, baseline_df)
        triggers = {
            "Intraday Rise Top": res1,
            "Closing Strength Top": res2,
            "Volume Surge Sideways": res3
        }
    else:
        logger.error("Invalid trigger_time. Use 'morning', 'midday', or 'afternoon'.")
        return

    active_triggers = sum(1 for df in triggers.values() if not df.empty)
    logger.info(f"Active triggers: {active_triggers}/{len(triggers)}")

    # Build screening summary for intermediate reporting
    screening_summary = {
        "total_tickers_scanned": len(tickers),
        "snapshot_count": len(snapshot),
        "prev_snapshot_count": len(prev_snapshot),
        "baseline_count": len(baseline_df) if baseline_df is not None and not baseline_df.empty else 0,
        "market_cap_count": len(cap_df) if cap_df is not None else 0,
        "triggers": {},
    }

    for name, df in triggers.items():
        if df.empty:
            logger.info(f"{name}: No qualifying stocks")
            screening_summary["triggers"][name] = {
                "candidate_count": 0,
                "top_candidates": [],
            }
        else:
            logger.info(f"{name} detected ({len(df)} stocks):")
            top_candidates = []
            for ticker in df.index:
                company = df.loc[ticker, "CompanyName"] if "CompanyName" in df.columns else ""
                logger.info(f"  - {ticker} ({company})")
                price = float(df.loc[ticker, "Close"]) if "Close" in df.columns else 0
                pct = float(df.loc[ticker, "DailyChange"]) if "DailyChange" in df.columns else 0
                if pct == 0 and "PctChange" in df.columns:
                    pct = float(df.loc[ticker, "PctChange"])
                score = float(df.loc[ticker, "CompositeScore"]) if "CompositeScore" in df.columns else 0
                vol_ratio = float(df.loc[ticker, "VolumeSurgeRatio"]) if "VolumeSurgeRatio" in df.columns else 0
                top_candidates.append({
                    "ticker": ticker,
                    "name": company,
                    "price": price,
                    "change_pct": pct,
                    "composite_score": score,
                    "volume_surge_ratio": vol_ratio,
                })
            screening_summary["triggers"][name] = {
                "candidate_count": len(df),
                "top_candidates": top_candidates,
            }

    final_results, eval_results = select_final_tickers(triggers, trade_date=trade_date,
                                                        max_selections=regime_max_picks,
                                                        regime_min_quality=regime_min_quality,
                                                        regime_max_change=regime_max_change,
                                                        regime_max_rsi=regime_max_rsi)

    if output_file:
        output_data = {}
        for trigger_type, stocks_df in final_results.items():
            if not stocks_df.empty:
                if trigger_type not in output_data:
                    output_data[trigger_type] = []
                for ticker in stocks_df.index:
                    stock_info = {
                        "ticker": ticker,
                        "name": stocks_df.loc[ticker, "CompanyName"] if "CompanyName" in stocks_df.columns else "",
                        "current_price": float(stocks_df.loc[ticker, "Close"]) if "Close" in stocks_df.columns else 0,
                        "change_rate": float(
                        stocks_df.loc[ticker, "DailyChange"]
                        if "DailyChange" in stocks_df.columns and float(stocks_df.loc[ticker, "DailyChange"]) != 0
                        else stocks_df.loc[ticker, "PctChange"]
                        if "PctChange" in stocks_df.columns
                        else 0
                    ),
                        "volume": int(stocks_df.loc[ticker, "Volume"]) if "Volume" in stocks_df.columns else 0,
                        "trade_value": float(stocks_df.loc[ticker, "Amount"]) if "Amount" in stocks_df.columns else 0,
                    }
                    if "AgentFitScore" in stocks_df.columns:
                        stock_info["agent_fit_score"] = float(stocks_df.loc[ticker, "AgentFitScore"])
                        stock_info["risk_reward_ratio"] = float(stocks_df.loc[ticker, "RiskRewardRatio"])
                        stock_info["stop_loss_pct"] = float(stocks_df.loc[ticker, "StopLossPct"]) * 100
                        stock_info["stop_loss_price"] = float(stocks_df.loc[ticker, "StopLossPrice"])
                        stock_info["target_price"] = float(stocks_df.loc[ticker, "TargetPrice"])
                    if "QualityScore" in stocks_df.columns:
                        stock_info["quality_score"] = float(stocks_df.loc[ticker, "QualityScore"])
                        stock_info["quality_signal"] = str(stocks_df.loc[ticker, "QualitySignal"])
                        stock_info["quality_reasons"] = str(stocks_df.loc[ticker, "QualityReasons"])
                    if "FinalScore" in stocks_df.columns:
                        stock_info["final_score"] = float(stocks_df.loc[ticker, "FinalScore"])
                    # Add detailed metrics from evaluator
                    if ticker in eval_results:
                        ev = eval_results[ticker]
                        stock_info["metrics"] = {
                            "sector": ev.sector or "",
                            "industry": ev.industry or "",
                            "market_cap_cr": round(ev.market_cap / 1e7, 0) if ev.market_cap else None,
                            # Valuation
                            "trailing_pe": round(ev.trailing_pe, 1) if ev.trailing_pe else None,
                            "forward_pe": round(ev.forward_pe, 1) if ev.forward_pe else None,
                            "price_to_book": round(ev.price_to_book, 2) if ev.price_to_book else None,
                            "dividend_yield_pct": round(ev.dividend_yield * 100, 2) if ev.dividend_yield else None,
                            # Growth
                            "revenue_growth_pct": round(ev.revenue_growth * 100, 1) if ev.revenue_growth else None,
                            "earnings_growth_pct": round(ev.earnings_growth * 100, 1) if ev.earnings_growth else None,
                            # Profitability
                            "operating_margin_pct": round(ev.operating_margin * 100, 1) if ev.operating_margin else None,
                            "profit_margin_pct": round(ev.profit_margin * 100, 1) if ev.profit_margin else None,
                            "roe_pct": round(ev.return_on_equity * 100, 1) if ev.return_on_equity else None,
                            # Technicals
                            "rsi_14": round(ev.rsi_14, 1) if ev.rsi_14 else None,
                            "sma_50": round(ev.sma_50, 2) if ev.sma_50 else None,
                            "sma_200": round(ev.sma_200, 2) if ev.sma_200 else None,
                            "pct_from_52w_high": round(ev.pct_from_52w_high, 1) if ev.pct_from_52w_high is not None else None,
                            "beta": round(ev.beta, 2) if ev.beta else None,
                            # Analyst
                            "analyst_recommendation": ev.recommendation or "",
                            "target_mean": round(ev.target_mean, 2) if ev.target_mean else None,
                            "target_upside_pct": round((ev.target_mean / ev.price - 1) * 100, 1) if ev.target_mean and ev.price else None,
                            "analyst_count": ev.analyst_count,
                            # Risk
                            "debt_to_equity": round(ev.debt_to_equity, 1) if ev.debt_to_equity is not None else None,
                            # Sub-scores
                            "valuation_score": round(ev.valuation_score, 0),
                            "growth_score": round(ev.growth_score, 0),
                            "profitability_score": round(ev.profitability_score, 0),
                            "technical_score": round(ev.technical_score, 0),
                            "analyst_score": round(ev.analyst_score, 0),
                            "risk_score": round(ev.risk_score, 0),
                        }
                    output_data[trigger_type].append(stock_info)

        output_data["screening_summary"] = screening_summary

        output_data["metadata"] = {
            "run_time": datetime.datetime.now().isoformat(),
            "trigger_mode": trigger_time,
            "trade_date": trade_date,
            "selection_mode": "momentum_first",
            "scoring_weights": {
                "momentum_signal": 0.75,
                "quality_tiebreaker": 0.25,
            },
            "quality_gate_min": QUALITY_GATE_MIN,
            "lookback_days": 10,
            "market": "IN",
            "min_market_cap_inr": MIN_MARKET_CAP,
            "min_trading_value_inr": MIN_TRADING_VALUE,
            "filters": {
                "max_trigger_change_pct": regime_max_change,
                "max_entry_rsi": regime_max_rsi,
                "min_quality": regime_min_quality,
            },
            "regime": {
                "type": regime,
                "nifty_rsi": round(nifty_rsi, 1),
                "nifty_slope_pct": round(nifty_slope_pct, 2) if 'nifty_slope_pct' in dir() else None,
                "nifty_rvol": round(nifty_rvol, 1) if 'nifty_rvol' in dir() else None,
                "max_picks": regime_max_picks,
            },
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="India (NSE) Trigger Batch Execution")
    parser.add_argument("mode", help="Execution mode (morning or afternoon)")
    parser.add_argument("log_level", nargs="?", default="INFO", help="Logging level")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()
    run_batch(args.mode, args.log_level, args.output)
