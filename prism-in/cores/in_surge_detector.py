#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
India Surge Detector - NSE/BSE market data fetching and surge detection utilities.

Provides snapshot, baseline, and filtering functions used by in_trigger_batch.py.
Data source: Yahoo Finance (.NS suffix) via yfinance.
"""

import logging
import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── NIFTY 500 universe (NSE symbols, no suffix) ──────────────────────────────
# Core NIFTY 100 + selected NIFTY 500 liquid names.
# yfinance appends ".NS" automatically where needed.
# Last cleaned: 2026-05-19 — removed delisted, renamed tickers updated
_NIFTY500_TICKERS = [
    # NIFTY 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "BAJFINANCE",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO",
    "ONGC", "NTPC", "POWERGRID", "TECHM", "M&M", "ADANIENT", "ADANIPORTS",
    "BAJAJFINSV", "JSWSTEEL", "TATAMOTORS", "TATASTEEL", "HINDALCO", "COALINDIA",
    "DIVISLAB", "BRITANNIA", "CIPLA", "DRREDDY", "EICHERMOT", "GRASIM", "HEROMOTOCO",
    "INDUSINDBK", "SBILIFE", "HDFCLIFE", "BPCL", "IOC", "APOLLOHOSP",
    "BAJAJ-AUTO", "TATACONSUM", "UPL", "LTIM",
    # Additional NIFTY 100 (renamed tickers updated)
    "ADANIGREEN", "ADANIENSOL", "AMBUJACEM", "AUROPHARMA", "BANDHANBNK",
    "BERGEPAINT", "BIOCON", "BOSCHLTD", "ZYDUSLIFE", "CHOLAFIN",
    "COLPAL", "DABUR", "DLF", "ESCORTS", "GAIL", "GODREJCP", "GODREJPROP",
    "HAVELLS", "IDFCFIRSTB", "IGL", "INDHOTEL", "INDUSTOWER",
    "ICICIGI", "ICICIPRULI", "IRCTC", "JINDALSTEL", "JUBLFOOD",
    "LICHSGFIN", "LUPIN", "MARICO", "MFSL", "MPHASIS",
    "MRF", "NAUKRI", "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND",
    "PETRONET", "PIDILITIND", "PIIND", "PNB", "RBLBANK",
    "RECLTD", "SAIL", "SHREECEM", "SIEMENS", "SRF", "SUNTV",
    "TATAPOWER", "TORNTPHARM", "TORNTPOWER", "TRENT", "UNIONBANK",
    "VEDL", "VOLTAS", "WHIRLPOOL", "ZEEL", "ETERNAL",
    # NIFTY 200 / 500 additions (cleaned: removed delisted, fixed renames)
    "ABCAPITAL", "ABFRL", "ACC", "AFFLE", "AJANTPHARM",
    "ALKEM", "ALKYLAMINE", "AMBER", "APOLLOTYRE", "APTUS",
    "ASTRAL", "ATUL", "AWHCL", "BAJAJHFL", "BALKRISIND",
    "BATAINDIA", "BAYERCROP", "BEML", "BHARATFORG", "BSOFT",
    "CAMS", "CANFINHOME", "CANBK", "CARBORUNIV", "CASTROLIND",
    "CEATLTD", "CENTRALBK", "CESC", "CGPOWER",
    "CHAMBLFERT", "CLEAN", "CMSINFO", "CONCOR", "COROMANDEL",
    "CREDITACC", "CROMPTON", "CSBBANK", "CYIENT", "DCMSHRIRAM",
    "DEEPAKNTR", "DELTACORP", "DMART", "DALBHARAT", "ELGIEQUIP",
    "EMAMILTD", "ENDURANCE", "ENGINERSIN", "EPL", "EQUITASBNK",
    "EXIDEIND", "FINCABLES", "FINPIPE", "FLUOROCHEM", "FORTIS",
    "GLAND", "GLAXO", "GMRAIRPORT", "GNFC",
    "GODFRYPHLP", "GRANULES", "GREENPLY", "GRINDWELL", "GSFC",
    "GUJGASLTD", "HAPPSTMNDS", "HATSUN", "HEG", "HEIDELBERG",
    "HFCL", "HINDCOPPER", "HINDPETRO", "HONAUT",
    "HUDCO", "IIFL", "IPCALAB", "IRB", "IRFC",
    "ITC", "J&KBANK", "JBCHEPHARM", "JKCEMENT",
    "JKLAKSHMI", "JKPAPER", "JKTYRE", "JSL", "JSWENERGY",
    "JYOTHYLAB", "KAJARIACER", "KPIL", "KANSAINER", "KPITTECH",
    "KRBL", "KSB", "KSCL", "LTF", "LATENTVIEW",
    "LINDEINDIA", "LUXIND", "CIEINDIA",
    "MASFIN", "MAXHEALTH", "MEDANTA", "METROPOLIS",
    "MIRCELECTR", "MMTC", "MOTHERSON", "MUTHOOTFIN", "NATCOPHARM",
    "NAVINFLUOR", "NBCC", "NCC", "NIACL", "NILKAMAL",
    "NLCINDIA", "NOCIL", "NUVOCO", "OLECTRA", "ORIENTCEM",
    "ORIENTELEC", "PGHH", "PHOENIXLTD", "POLYCAB", "POLYMED",
    "PRAJIND", "PRINCEPIPE", "PRSMJOHNSN", "PVRINOX",
    "QUESS", "RADICO", "RAJESHEXPO", "RAMCOCEM", "RATNAMANI",
    "RAYMOND", "RPOWER", "RVNL", "SAFARI", "SAREGAMA",
    "SHYAMMETL", "SIGNATURE", "SOBHA", "SPARC",
    "STAR", "STYRENIX", "SUMICHEM", "SUNDARMFIN",
    "SUNDRMBRAK", "SUPRAJIT", "SUPREMEIND", "SURYAROSNI",
    "SWSOLAR", "SYMPHONY", "TANLA", "TATACHEM", "TATACOMM",
    "TATAINVEST", "TEAMLEASE", "TIINDIA", "TIMKEN",
    "TITAGARH", "TTKPRESTIG", "TVSHLTD",
    "UCOBANK", "UJJIVANSFB", "UNITDSPR", "UTIAMC",
    "VAIBHAVGBL", "VGUARD", "VHL", "VINATIORGA", "VIPIND",
    "VSTIND", "WELCORP", "WELSPUNLIV",
    "WESTLIFE", "WONDERLA", "XTGLOBAL", "YESBANK", "ZENSARTECH",
    # Added: stocks that were missing but are in NIFTY 500
    "HSCL",  # Himadri Speciality Chemical
]

# Deduplicate while preserving order
_seen = set()
NIFTY500_TICKERS = []
for t in _NIFTY500_TICKERS:
    if t not in _seen:
        _seen.add(t)
        NIFTY500_TICKERS.append(t)

# Company name cache
_name_cache: dict = {}


def get_major_tickers() -> List[str]:
    """Return the NIFTY 500 universe as a list of plain NSE symbols."""
    return list(NIFTY500_TICKERS)


def get_ticker_name(ticker: str) -> str:
    """Return a human-readable company name for an NSE ticker."""
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(f"{ticker}.NS").fast_info
        name = getattr(info, "company_name", None) or ticker
    except Exception:
        name = ticker
    _name_cache[ticker] = name
    return name


def get_nearest_business_day(date_str: str, prev: bool = True) -> str:
    """
    Return the nearest NSE business day (Mon–Fri, non-holiday) as YYYYMMDD.

    Args:
        date_str: Date in YYYYMMDD format.
        prev: If True, look backward; if False, look forward.
    """
    try:
        import holidays as hol
        india_holidays = hol.India()
    except Exception:
        india_holidays = {}

    dt = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    delta = datetime.timedelta(days=-1 if prev else 1)

    for _ in range(14):
        if dt.weekday() < 5 and dt not in india_holidays:
            return dt.strftime("%Y%m%d")
        dt += delta

    # Fallback: just return the input stripped to weekday
    return date_str


def _yf_ticker(symbol: str) -> str:
    """Convert plain NSE symbol to yfinance ticker (append .NS)."""
    return f"{symbol}.NS"


def get_snapshot(trade_date: str, tickers: List[str]) -> pd.DataFrame:
    """
    Fetch current-day OHLCV snapshot for all tickers via yfinance.

    Returns DataFrame indexed by NSE symbol with columns:
        Open, High, Low, Close, Volume, Amount, PreviousClose, MarketCap
    Amount = Close × Volume (proxy for INR turnover).
    """
    dt = datetime.datetime.strptime(trade_date, "%Y%m%d").date()
    # Fetch 2 days of data to ensure we get the target date even across weekends
    start = (dt - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    end = (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    yf_tickers = [_yf_ticker(t) for t in tickers]

    logger.info(f"Downloading snapshot for {len(yf_tickers)} tickers ({trade_date})…")

    # Download in chunks to avoid yfinance rate limiting on historical dates
    import time as _dl_time
    chunk_size = 100
    raw_frames = []
    for ci in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[ci:ci + chunk_size]
        if ci > 0:
            _dl_time.sleep(2)  # pause between chunks
        try:
            chunk_raw = yf.download(
                chunk,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if not chunk_raw.empty:
                raw_frames.append((chunk, chunk_raw))
        except Exception as e:
            logger.warning(f"Chunk download failed ({len(chunk)} tickers): {e}")

    if not raw_frames:
        logger.warning(f"All downloads failed for {trade_date}")
        return pd.DataFrame()

    records = {}
    for chunk_tickers, raw in raw_frames:
      chunk_syms = [t.replace(".NS", "") for t in chunk_tickers]
      for sym in chunk_syms:
        yt = _yf_ticker(sym)
        try:
            if len(chunk_tickers) == 1:
                df_sym = raw
            else:
                df_sym = raw[yt] if yt in raw.columns.get_level_values(0) else pd.DataFrame()

            if df_sym.empty:
                continue

            # Drop rows with all-NaN OHLCV
            df_sym = df_sym.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if df_sym.empty:
                continue

            # Pick the row closest to trade_date
            df_sym.index = pd.to_datetime(df_sym.index).tz_localize(None)
            target = pd.Timestamp(dt)
            # Get the last available row on or before target date
            available = df_sym[df_sym.index.normalize() <= target]
            if available.empty:
                available = df_sym

            row = available.iloc[-1]
            prev_rows = available.iloc[:-1]
            prev_close = float(prev_rows.iloc[-1]["Close"]) if len(prev_rows) > 0 else float(row["Close"])

            close = float(row["Close"])
            volume = float(row["Volume"])
            records[sym] = {
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": close,
                "Volume": volume,
                "Amount": close * volume,
                "PreviousClose": prev_close,
                "MarketCap": 0.0,
            }
        except Exception as e:
            logger.debug(f"Snapshot skip {sym}: {e}")
            continue

    df = pd.DataFrame.from_dict(records, orient="index")
    if df.empty:
        logger.warning(f"Snapshot returned empty for {trade_date}")
        return df

    df = df[df["Close"] > 0].copy()
    logger.info(f"Snapshot complete: {len(df)} stocks")
    return df


def get_previous_snapshot(trade_date: str, tickers: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    Fetch the previous business day's snapshot.

    Returns:
        (DataFrame, prev_date_str) — same schema as get_snapshot.
    """
    dt = datetime.datetime.strptime(trade_date, "%Y%m%d").date()
    prev_date = get_nearest_business_day(
        (dt - datetime.timedelta(days=1)).strftime("%Y%m%d"), prev=True
    )
    logger.info(f"Fetching previous snapshot for {prev_date}…")
    df = get_snapshot(prev_date, tickers)
    return df, prev_date


def get_volume_baseline(tickers: List[str], trade_date: str, window: int = 20) -> pd.DataFrame:
    """
    Compute 20-day volume/amount/range baseline ending on trade_date (exclusive).

    Returns DataFrame with columns:
        AvgVolume, StdVolume, AvgAmount, AvgDailyRange
    """
    dt = datetime.datetime.strptime(trade_date, "%Y%m%d").date()
    end = dt.strftime("%Y-%m-%d")
    start = (dt - datetime.timedelta(days=window * 2)).strftime("%Y-%m-%d")

    yf_tickers = [_yf_ticker(t) for t in tickers]
    logger.info(f"Downloading {window}-day baseline for {len(yf_tickers)} tickers…")

    try:
        raw = yf.download(
            yf_tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.warning(f"Baseline download failed: {e}")
        return pd.DataFrame()

    records = {}
    for sym in tickers:
        yt = _yf_ticker(sym)
        try:
            if len(yf_tickers) == 1:
                df_sym = raw
            else:
                df_sym = raw[yt] if yt in raw.columns.get_level_values(0) else pd.DataFrame()

            if df_sym is None or df_sym.empty:
                continue

            df_sym = df_sym.dropna(subset=["Volume", "Close"])
            if len(df_sym) < 5:
                continue

            tail = df_sym.tail(window)
            avg_vol = float(tail["Volume"].mean())
            std_vol = float(tail["Volume"].std())
            avg_amount = float((tail["Close"] * tail["Volume"]).mean())
            avg_range = float(((tail["High"] - tail["Low"]) / tail["Close"].replace(0, np.nan) * 100).mean())

            records[sym] = {
                "AvgVolume": avg_vol,
                "StdVolume": std_vol if not np.isnan(std_vol) else avg_vol * 0.3,
                "AvgAmount": avg_amount,
                "AvgDailyRange": avg_range if not np.isnan(avg_range) else 1.0,
            }
        except Exception as e:
            logger.debug(f"Baseline skip {sym}: {e}")
            continue

    df = pd.DataFrame.from_dict(records, orient="index")
    logger.info(f"Baseline complete: {len(df)} stocks")
    return df


def get_multi_day_ohlcv(ticker: str, trade_date: str, lookback_days: int = 10) -> pd.DataFrame:
    """
    Fetch multi-day OHLCV for a single ticker ending on trade_date (inclusive).

    Returns DataFrame with standard OHLCV columns.
    """
    dt = datetime.datetime.strptime(trade_date, "%Y%m%d").date()
    end = (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start = (dt - datetime.timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

    try:
        df = yf.download(
            _yf_ticker(ticker),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index).tz_localize(None)
        target = pd.Timestamp(dt)
        df = df[df.index.normalize() <= target].tail(lookback_days)
        return df
    except Exception as e:
        logger.debug(f"multi_day_ohlcv failed for {ticker}: {e}")
        return pd.DataFrame()


def apply_absolute_filters(df: pd.DataFrame, min_trading_value: float = 1_000_000_000) -> pd.DataFrame:
    """
    Filter out stocks below minimum daily trading value (Amount in INR).

    Args:
        df: Snapshot DataFrame with an 'Amount' column.
        min_trading_value: Minimum INR turnover (default ₹100 Cr).
    """
    if df.empty:
        return df

    if "Amount" not in df.columns:
        if "Close" in df.columns and "Volume" in df.columns:
            df = df.copy()
            df["Amount"] = df["Close"] * df["Volume"]
        else:
            return df

    result = df[df["Amount"] >= min_trading_value].copy()
    logger.debug(f"apply_absolute_filters: {len(df)} → {len(result)} stocks (min_value={min_trading_value:,.0f})")
    return result


def normalize_and_score(
    df: pd.DataFrame,
    columns: List[str],
    weights: List[float],
    score_col: str = "CompositeScore",
) -> pd.DataFrame:
    """
    Min-max normalize columns and compute a weighted composite score.

    Args:
        df: Input DataFrame.
        columns: Column names to normalize.
        weights: Weights for each column (must sum to ~1).
        score_col: Name of the output score column.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        col_max = df[col].max()
        col_min = df[col].min()
        col_range = col_max - col_min if col_max > col_min else 1
        df[f"{col}_norm"] = (df[col] - col_min) / col_range

    df[score_col] = sum(
        df[f"{col}_norm"] * w for col, w in zip(columns, weights)
    )
    return df


def enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CompanyName column to a DataFrame indexed by NSE ticker symbols.
    Uses a lightweight lookup to avoid too many API calls.
    """
    if df.empty:
        return df

    df = df.copy()
    if "CompanyName" not in df.columns:
        df["CompanyName"] = ""

    for ticker in df.index:
        if not df.loc[ticker, "CompanyName"]:
            df.loc[ticker, "CompanyName"] = get_ticker_name(ticker)

    return df
