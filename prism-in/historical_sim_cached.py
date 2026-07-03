#!/usr/bin/env python3
"""
PRISM India — Cached Historical Trigger Simulation

Runs trigger detection on historical dates using the local price cache.
Zero yfinance API calls during simulation (except quality evaluator).

Steps:
  1. Load price cache (parquet) — one-time bulk download
  2. For each trading day: extract snapshot/baseline from cache
  3. Run trigger detection pipeline
  4. Skip quality evaluation (uses yfinance) — use CompositeScore only
  5. Save results JSON

Usage:
    # First: build the cache (one-time, ~2 min)
    python prism-in/price_cache.py --start 20250901 --end 20260601

    # Then: run simulation (no API calls, ~5 sec/date)
    python prism-in/historical_sim_cached.py --start 20251001 --end 20260301

    # Finally: backtest
    python prism-in/backtest_engine.py --min-date 20251001 --output backtest_full.json
"""

import sys
import os
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PRISM_IN_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PRISM_IN_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_trading_days(start: str, end: str, cache_df: pd.DataFrame) -> list:
    """Get trading days that exist in the cache data."""
    start_dt = pd.Timestamp(datetime.strptime(start, "%Y%m%d"))
    end_dt = pd.Timestamp(datetime.strptime(end, "%Y%m%d"))
    
    # Get unique dates from cache
    all_dates = cache_df["Date"].dt.normalize().unique()
    trading_days = sorted([
        d for d in all_dates 
        if start_dt <= d <= end_dt and d.weekday() < 5
    ])
    
    return [d.strftime("%Y%m%d") for d in trading_days]


def run_triggers_from_cache(cache_df: pd.DataFrame, trade_date: str) -> dict:
    """Run all morning triggers using cached data (no API calls)."""
    from price_cache import get_snapshot_from_cache, get_previous_snapshot_from_cache, get_baseline_from_cache
    from cores.in_surge_detector import apply_absolute_filters, normalize_and_score, enhance_dataframe
    
    snapshot = get_snapshot_from_cache(cache_df, trade_date)
    prev_snapshot, prev_date = get_previous_snapshot_from_cache(cache_df, trade_date)
    baseline_df = get_baseline_from_cache(cache_df, trade_date)
    
    if snapshot.empty or prev_snapshot.empty:
        return {}, {}, 0
    
    # Use inline trigger logic (avoids the full pipeline which calls yfinance for market cap)
    # Simplified: no market cap filter (we don't have it in cache), use trading value filter only
    MIN_TRADING_VALUE = 1_000_000_000  # ₹100 Cr
    
    results = {}
    
    common = snapshot.index.intersection(prev_snapshot.index)
    snap = snapshot.loc[common].copy()
    prev = prev_snapshot.loc[common].copy()
    
    # Basic filters
    snap = snap[snap["Amount"] >= MIN_TRADING_VALUE]
    if snap.empty:
        return results, {"snapshot": len(snapshot), "filtered": 0}, 0
    
    snap["IntradayChange"] = (snap["Close"] / snap["Open"] - 1) * 100
    snap["DailyChange"] = ((snap["Close"] - prev.reindex(snap.index)["Close"]) / prev.reindex(snap.index)["Close"]) * 100
    snap = snap[snap["DailyChange"].notna() & (snap["DailyChange"] <= 20.0) & (snap["DailyChange"] >= -20.0)]
    snap["IsRising"] = snap["Close"] > snap["Open"]
    
    if snap.empty:
        return results, {"snapshot": len(snapshot), "filtered": 0}, 0
    
    # Volume metrics from baseline
    if not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid = bl["AvgVolume"].notna() & (bl["AvgVolume"] > 0)
        snap["VolumeSurgeRatio"] = np.nan
        snap["VolumeZScore"] = 0.0
        if valid.any():
            snap.loc[valid, "VolumeSurgeRatio"] = snap.loc[valid, "Volume"] / bl.loc[valid, "AvgVolume"]
            std = bl.loc[valid, "StdVolume"].replace(0, np.nan)
            snap.loc[valid, "VolumeZScore"] = ((snap.loc[valid, "Volume"] - bl.loc[valid, "AvgVolume"]) / std).fillna(0)
        no_bl = snap["VolumeSurgeRatio"].isna()
        if no_bl.any():
            prev_vol = prev.reindex(snap.index).loc[no_bl, "Volume"].replace(0, np.nan)
            snap.loc[no_bl, "VolumeSurgeRatio"] = snap.loc[no_bl, "Volume"] / prev_vol
    else:
        snap["VolumeSurgeRatio"] = snap["Volume"] / prev.reindex(snap.index)["Volume"].replace(0, np.nan)
        snap["VolumeZScore"] = 0.0
    
    snap["GapUpRate"] = (snap["Open"] / prev.reindex(snap.index)["Close"] - 1) * 100
    
    # Amount vs average
    if not baseline_df.empty:
        bl = baseline_df.reindex(snap.index)
        valid = bl["AvgAmount"].notna() & (bl["AvgAmount"] > 0)
        snap["AmountVsAvg"] = 1.0
        if valid.any():
            snap.loc[valid, "AmountVsAvg"] = snap.loc[valid, "Amount"] / bl.loc[valid, "AvgAmount"]
    else:
        snap["AmountVsAvg"] = 1.0
    
    def minmax_norm(series):
        mn, mx = series.min(), series.max()
        rng = mx - mn if mx > mn else 1
        return (series - mn) / rng
    
    # ── Trigger 1: Volume Surge Top ──
    vol_thresh = 1.5 if not baseline_df.empty else 1.3
    vs = snap[(snap["VolumeSurgeRatio"] >= vol_thresh) & snap["IsRising"]].copy()
    if not vs.empty and len(vs) >= 1:
        vs["CompositeScore"] = (
            minmax_norm(vs["VolumeSurgeRatio"]) * 0.4 +
            minmax_norm(vs["VolumeZScore"]) * 0.3 +
            minmax_norm(vs["Volume"]) * 0.3
        )
        results["Volume Surge Top"] = vs.nlargest(10, "CompositeScore")
    
    # ── Trigger 2: Gap Up Momentum ──
    gap = snap[(snap["GapUpRate"] >= 1.0) & (snap["DailyChange"] <= 15.0) & (snap["Close"] > snap["Open"])].copy()
    if not gap.empty and len(gap) >= 1:
        if not baseline_df.empty:
            bl = baseline_df.reindex(gap.index)
            valid = bl["AvgDailyRange"].notna() & (bl["AvgDailyRange"] > 0)
            gap["GapVsATR"] = gap["GapUpRate"]
            if valid.any():
                gap.loc[valid, "GapVsATR"] = gap.loc[valid, "GapUpRate"].abs() / bl.loc[valid, "AvgDailyRange"]
        else:
            gap["GapVsATR"] = gap["GapUpRate"]
        
        gap["CompositeScore"] = (
            minmax_norm(gap["GapVsATR"]) * 0.35 +
            minmax_norm(gap["IntradayChange"]) * 0.30 +
            minmax_norm(gap["Amount"]) * 0.20 +
            minmax_norm(gap["GapUpRate"]) * 0.15
        )
        results["Gap Up Momentum Top"] = gap.nlargest(10, "CompositeScore")
    
    # ── Trigger 3: Value-to-Cap Ratio ──
    # Use AmountVsAvg as proxy (no market cap in cache)
    val = snap[snap["IsRising"] & (snap["AmountVsAvg"] > 1.5)].copy()
    if not val.empty and len(val) >= 1:
        val["CompositeScore"] = (
            minmax_norm(val["AmountVsAvg"]) * 0.30 +
            minmax_norm(val["Amount"]) * 0.30 +
            minmax_norm(val["IntradayChange"]) * 0.20 +
            minmax_norm(val["Volume"]) * 0.20
        )
        results["Value-to-Cap Ratio Top"] = val.nlargest(10, "CompositeScore")
    
    screening = {
        "snapshot": len(snapshot),
        "prev_snapshot": len(prev_snapshot),
        "baseline": len(baseline_df),
        "filtered": len(snap),
    }
    
    total_picks = sum(len(df) for df in results.values())
    return results, screening, total_picks


def select_top_picks(triggers: dict, max_picks: int = 7) -> list:
    """Select top picks from triggers (simplified — no quality eval, just CompositeScore)."""
    selected = []
    selected_tickers = set()
    
    # Phase 1: top 1 per trigger
    for trigger_type, df in triggers.items():
        if df.empty:
            continue
        for ticker in df.index:
            if ticker not in selected_tickers:
                row = df.loc[ticker]
                selected.append({
                    "ticker": ticker,
                    "trigger_type": trigger_type,
                    "current_price": float(row["Close"]),
                    "change_rate": float(row["DailyChange"]),
                    "composite_score": float(row["CompositeScore"]),
                    "volume": int(row["Volume"]),
                    "trade_value": float(row["Amount"]),
                })
                selected_tickers.add(ticker)
                break
    
    # Phase 2: fill by score
    all_candidates = []
    for trigger_type, df in triggers.items():
        for ticker in df.index:
            if ticker not in selected_tickers:
                all_candidates.append((trigger_type, ticker, float(df.loc[ticker, "CompositeScore"]), df.loc[ticker]))
    
    all_candidates.sort(key=lambda x: x[2], reverse=True)
    for trigger_type, ticker, score, row in all_candidates:
        if len(selected) >= max_picks:
            break
        if ticker not in selected_tickers:
            selected.append({
                "ticker": ticker,
                "trigger_type": trigger_type,
                "current_price": float(row["Close"]),
                "change_rate": float(row["DailyChange"]),
                "composite_score": score,
                "volume": int(row["Volume"]),
                "trade_value": float(row["Amount"]),
            })
            selected_tickers.add(ticker)
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="Cached Historical Trigger Simulation")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--max-days", type=int, default=None, help="Max trading days to process")
    args = parser.parse_args()
    
    # Load cache
    from price_cache import load_cache, OHLCV_CACHE_FILE
    
    logger.info(f"Loading price cache from {OHLCV_CACHE_FILE}...")
    try:
        cache_df = load_cache()
    except FileNotFoundError:
        logger.error("Price cache not found! Run: python price_cache.py --start 20250901 --end 20260601")
        sys.exit(1)
    
    logger.info(f"Cache loaded: {cache_df['Ticker'].nunique()} tickers, "
                f"{cache_df['Date'].min().strftime('%Y-%m-%d')} to {cache_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Get trading days
    trading_days = get_trading_days(args.start, args.end, cache_df)
    if args.max_days:
        trading_days = trading_days[:args.max_days]
    
    logger.info(f"Processing {len(trading_days)} trading days ({args.start} to {args.end})")
    
    # Check existing results
    existing = 0
    for d in trading_days:
        f = PROJECT_ROOT / f"trigger_results_in_morning_{d}.json"
        if f.exists():
            existing += 1
    logger.info(f"Already have results for {existing}/{len(trading_days)} days")
    
    # Run simulation
    t_start = time.time()
    success = 0
    skipped = 0
    empty = 0
    
    for i, trade_date in enumerate(trading_days):
        output_file = PROJECT_ROOT / f"trigger_results_in_morning_{trade_date}.json"
        
        if output_file.exists():
            skipped += 1
            continue
        
        t0 = time.time()
        triggers, screening, n_picks = run_triggers_from_cache(cache_df, trade_date)
        
        if not triggers:
            logger.info(f"  [{i+1}/{len(trading_days)}] {trade_date}: no triggers "
                       f"(snap={screening.get('snapshot', 0)}, filtered={screening.get('filtered', 0)})")
            empty += 1
            continue
        
        # Select top picks
        picks = select_top_picks(triggers)
        
        # Build output JSON (compatible with backtest_engine)
        output_data = {}
        for pick in picks:
            tt = pick["trigger_type"]
            if tt not in output_data:
                output_data[tt] = []
            output_data[tt].append({
                "ticker": pick["ticker"],
                "name": pick["ticker"],
                "current_price": pick["current_price"],
                "change_rate": pick["change_rate"],
                "volume": pick["volume"],
                "trade_value": pick["trade_value"],
                "final_score": pick["composite_score"],
                "quality_score": 50.0,
                "quality_signal": "N/A",
            })
        
        output_data["screening_summary"] = {
            "total_tickers_scanned": screening.get("snapshot", 0),
            "snapshot_count": screening.get("snapshot", 0),
            "baseline_count": screening.get("baseline", 0),
            "triggers": {name: {"candidate_count": len(df)} for name, df in triggers.items()},
        }
        output_data["metadata"] = {
            "run_time": datetime.now().isoformat(),
            "trigger_mode": "morning",
            "trade_date": trade_date,
            "selection_mode": "cached_composite",
            "market": "IN",
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - t0
        tickers_str = ", ".join(p["ticker"] for p in picks[:5])
        logger.info(f"  [{i+1}/{len(trading_days)}] {trade_date}: {len(picks)} picks in {elapsed:.1f}s [{tickers_str}]")
        success += 1
    
    total_time = time.time() - t_start
    logger.info(f"\nSimulation complete in {total_time:.0f}s")
    logger.info(f"  Success: {success}, Skipped: {skipped}, Empty: {empty}")
    logger.info(f"\nNow run backtest:")
    logger.info(f"  python backtest_engine.py --min-date {args.start} --max-date {args.end} --output backtest_full.json")


if __name__ == "__main__":
    main()
