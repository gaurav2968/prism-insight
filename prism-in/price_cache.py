#!/usr/bin/env python3
"""
PRISM India — Bulk Price Cache

Downloads ALL historical OHLCV data for the ticker universe in one shot,
saves to a local parquet file. Subsequent runs skip the download.

This replaces ~5,500 yfinance API calls with ~6 bulk downloads.

Usage:
    python prism-in/price_cache.py --start 20250901 --end 20260601
    python prism-in/price_cache.py --start 20250901 --end 20260601 --force

Data is saved to: prism-in/cache/ohlcv_cache.parquet
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
PRISM_IN_DIR = Path(__file__).parent
sys.path.insert(0, str(PRISM_IN_DIR))

CACHE_DIR = PRISM_IN_DIR / "cache"
OHLCV_CACHE_FILE = CACHE_DIR / "ohlcv_cache.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_ticker_universe():
    """Get the full ticker list from surge detector."""
    from cores.in_surge_detector import NIFTY500_TICKERS
    return NIFTY500_TICKERS


def download_bulk_ohlcv(tickers: list, start: str, end: str, chunk_size: int = 80) -> pd.DataFrame:
    """
    Download OHLCV data for all tickers in bulk chunks.
    
    Returns a MultiIndex DataFrame: (Date, Ticker) → OHLCV columns.
    """
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    
    # Add buffer: 30 days before start (for baselines) and 45 days after end (for forward returns)
    fetch_start = (start_dt - timedelta(days=45)).strftime("%Y-%m-%d")
    fetch_end = (end_dt + timedelta(days=60)).strftime("%Y-%m-%d")
    
    yf_tickers = [f"{t}.NS" for t in tickers]
    
    all_data = {}
    n_chunks = (len(yf_tickers) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        logger.info(f"  Chunk {chunk_num}/{n_chunks}: downloading {len(chunk)} tickers...")
        
        if i > 0:
            time.sleep(3)  # pause between chunks
        
        retry_count = 0
        max_retries = 2
        
        while retry_count <= max_retries:
            try:
                raw = yf.download(
                    chunk,
                    start=fetch_start,
                    end=fetch_end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
                
                if raw.empty:
                    logger.warning(f"  Chunk {chunk_num}: empty response")
                    break
                
                # Parse into per-ticker DataFrames
                for yf_t in chunk:
                    nse_sym = yf_t.replace(".NS", "")
                    try:
                        if len(chunk) == 1:
                            df_sym = raw
                        else:
                            if isinstance(raw.columns, pd.MultiIndex):
                                df_sym = raw.xs(yf_t, axis=1, level=1) if yf_t in raw.columns.get_level_values(1) else pd.DataFrame()
                            else:
                                df_sym = raw
                        
                        if df_sym.empty:
                            continue
                        
                        df_sym = df_sym.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                        if df_sym.empty:
                            continue
                        
                        df_sym = df_sym.copy()
                        df_sym.index = pd.to_datetime(df_sym.index).tz_localize(None)
                        df_sym["Ticker"] = nse_sym
                        df_sym["Amount"] = df_sym["Close"] * df_sym["Volume"]
                        all_data[nse_sym] = df_sym
                    except Exception:
                        continue
                
                logger.info(f"  Chunk {chunk_num}: got {sum(1 for t in chunk if t.replace('.NS','') in all_data)} tickers")
                break  # success
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait = 10 * retry_count
                    logger.warning(f"  Chunk {chunk_num} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"  Chunk {chunk_num} failed after {max_retries} retries: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine into single DataFrame with Ticker column
    combined = pd.concat(all_data.values(), axis=0)
    combined = combined.reset_index()
    combined = combined.rename(columns={"index": "Date", "date": "Date", "Date": "Date"})
    
    # Ensure Date column exists
    if "Date" not in combined.columns:
        for col in combined.columns:
            if combined[col].dtype == "datetime64[ns]":
                combined = combined.rename(columns={col: "Date"})
                break
    
    logger.info(f"Total: {len(all_data)} tickers, {len(combined)} rows")
    return combined


def build_cache(start: str, end: str, force: bool = False):
    """Download and cache all OHLCV data."""
    CACHE_DIR.mkdir(exist_ok=True)
    
    if OHLCV_CACHE_FILE.exists() and not force:
        logger.info(f"Cache already exists: {OHLCV_CACHE_FILE}")
        logger.info(f"Use --force to rebuild")
        
        # Show cache info
        df = pd.read_parquet(OHLCV_CACHE_FILE)
        logger.info(f"  Tickers: {df['Ticker'].nunique()}")
        logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"  Rows: {len(df):,}")
        return df
    
    tickers = get_ticker_universe()
    logger.info(f"Downloading OHLCV for {len(tickers)} tickers ({start} to {end})...")
    
    t0 = time.time()
    df = download_bulk_ohlcv(tickers, start, end)
    elapsed = time.time() - t0
    
    if df.empty:
        logger.error("No data downloaded!")
        return df
    
    # Save to parquet (fast, compact, columnar)
    df.to_parquet(OHLCV_CACHE_FILE, index=False)
    file_size_mb = OHLCV_CACHE_FILE.stat().st_size / 1e6
    
    logger.info(f"Cache saved: {OHLCV_CACHE_FILE}")
    logger.info(f"  Tickers: {df['Ticker'].nunique()}")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"  Rows: {len(df):,}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Download time: {elapsed:.0f}s")
    
    return df


def load_cache() -> pd.DataFrame:
    """Load cached OHLCV data."""
    if not OHLCV_CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache not found: {OHLCV_CACHE_FILE}. Run price_cache.py first.")
    return pd.read_parquet(OHLCV_CACHE_FILE)


def get_snapshot_from_cache(cache_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Extract a single-day snapshot from the cache (same schema as get_snapshot)."""
    dt = pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
    
    # Get data on or before trade_date
    available = cache_df[cache_df["Date"].dt.normalize() <= dt]
    if available.empty:
        return pd.DataFrame()
    
    # Get the last available date
    last_date = available["Date"].dt.normalize().max()
    day_data = available[available["Date"].dt.normalize() == last_date]
    
    if day_data.empty:
        return pd.DataFrame()
    
    # Build snapshot indexed by ticker
    records = {}
    for _, row in day_data.iterrows():
        ticker = row["Ticker"]
        records[ticker] = {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
            "Amount": float(row["Amount"]),
            "PreviousClose": 0.0,
            "MarketCap": 0.0,
        }
    
    df = pd.DataFrame.from_dict(records, orient="index")
    return df[df["Close"] > 0]


def get_previous_snapshot_from_cache(cache_df: pd.DataFrame, trade_date: str) -> tuple:
    """Get previous day's snapshot from cache."""
    dt = pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
    
    # Get data strictly before trade_date
    available = cache_df[cache_df["Date"].dt.normalize() < dt]
    if available.empty:
        return pd.DataFrame(), ""
    
    last_date = available["Date"].dt.normalize().max()
    prev_date_str = last_date.strftime("%Y%m%d")
    
    return get_snapshot_from_cache(cache_df, prev_date_str), prev_date_str


def get_baseline_from_cache(cache_df: pd.DataFrame, trade_date: str, window: int = 20) -> pd.DataFrame:
    """Compute 20-day volume/price baseline from cache (same schema as get_volume_baseline)."""
    import numpy as np
    
    dt = pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
    
    # Get data strictly before trade_date
    available = cache_df[cache_df["Date"].dt.normalize() < dt].copy()
    if available.empty:
        return pd.DataFrame()
    
    records = {}
    for ticker, group in available.groupby("Ticker"):
        group = group.sort_values("Date").tail(window)
        if len(group) < 5:
            continue
        
        avg_vol = group["Volume"].mean()
        std_vol = group["Volume"].std()
        if pd.isna(std_vol) or std_vol == 0:
            std_vol = avg_vol * 0.3
        
        avg_amount = group["Amount"].mean()
        
        # AvgDailyRange = mean((High - Low) / Close * 100)
        daily_range = ((group["High"] - group["Low"]) / group["Close"] * 100)
        avg_daily_range = daily_range.mean()
        if pd.isna(avg_daily_range):
            avg_daily_range = 1.0
        
        records[ticker] = {
            "AvgVolume": avg_vol,
            "StdVolume": std_vol,
            "AvgAmount": avg_amount,
            "AvgDailyRange": avg_daily_range,
        }
    
    return pd.DataFrame.from_dict(records, orient="index")


def main():
    parser = argparse.ArgumentParser(description="PRISM India Price Cache")
    parser.add_argument("--start", default="20250901", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20260601", help="End date YYYYMMDD")
    parser.add_argument("--force", action="store_true", help="Force rebuild cache")
    args = parser.parse_args()
    
    build_cache(args.start, args.end, args.force)


if __name__ == "__main__":
    main()
