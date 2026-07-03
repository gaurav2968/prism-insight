#!/usr/bin/env python3
"""
Position Monitor Service for PRISM-INSIGHT (India / NSE)

Runs as a persistent background process (Windows Service via NSSM).
Monitors open positions every minute during market hours,
checks SL/TP/time exits, and sends Telegram + Firebase alerts.

Usage:
    python prism-in/position_monitor.py              # Run as foreground process
    python prism-in/position_monitor.py --dry-run     # No Telegram, just log
    python prism-in/position_monitor.py --check-once  # Single check then exit
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pytz
import yfinance as yf

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "prism-in"))

from check_market_day import is_nse_market_day, is_market_open_now, get_next_trading_day
from cores.in_surge_detector import get_multi_day_ohlcv

IST = pytz.timezone("Asia/Kolkata")
DB_PATH = PROJECT_ROOT / "stock_tracking_db.sqlite"
TRIGGER_DIR = PROJECT_ROOT  # trigger_results_in_morning_*.json live here

# ── Config ────────────────────────────────────────────────────
CHECK_INTERVAL_SECONDS = 60       # Price check interval during market hours
IDLE_INTERVAL_SECONDS = 300       # Sleep interval when market is closed
PRE_MARKET_MINUTES = 5            # Start checking 5 min before market open (9:10 IST)
MAX_HOLD_TRADING_DAYS = 21        # Exit after 21 trading days (King 2 strategy)
PRICE_FETCH_TIMEOUT = 10          # yfinance timeout per ticker
TRIGGER_RUN_HOUR = 9              # Hour (IST) to run trigger batch
TRIGGER_RUN_MINUTE = 30           # Minute (IST) to run trigger batch

# ATR-based dynamic TP/SL (must match in_trigger_batch.py constants)
ATR_PERIOD = 14
ATR_SL_MULT = 3.0                # stop-loss  = entry - 3×ATR₁₄
ATR_TP_MULT = 2.0                # take-profit = entry + 2×ATR₁₄
ATR_RECALC_INTERVAL = 3600       # Recalculate ATR at most once per hour per ticker

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "position_monitor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("position_monitor")

# ── Graceful shutdown ─────────────────────────────────────────
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ══════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════

def init_db():
    """Create monitored_positions table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitored_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            company_name TEXT,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            max_exit_date TEXT NOT NULL,
            trigger_type TEXT,
            sector TEXT,
            quality_score REAL,
            final_score REAL,
            change_rate REAL,
            status TEXT DEFAULT 'OPEN',
            current_price REAL,
            last_checked TEXT,
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            return_pct REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_mon_pos_status
        ON monitored_positions(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_mon_pos_ticker
        ON monitored_positions(ticker)
    """)

    # Track which trigger files have been ingested
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitor_ingested_files (
            filename TEXT PRIMARY KEY,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Persist service state across restarts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS service_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def get_service_state(conn, key: str) -> str | None:
    """Get a persisted service state value."""
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM service_state WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row["value"] if row else None


def set_service_state(conn, key: str, value: str):
    """Set a persisted service state value."""
    conn.execute(
        "INSERT OR REPLACE INTO service_state (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (key, value),
    )
    conn.commit()


def get_open_positions(conn) -> list:
    """Get all OPEN monitored positions."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM monitored_positions WHERE status = 'OPEN'")
    return [dict(row) for row in cursor.fetchall()]


def close_position(conn, position_id: int, exit_price: float,
                   exit_reason: str, return_pct: float):
    """Close a position with exit details."""
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE monitored_positions
        SET status = ?, exit_price = ?, exit_date = ?,
            exit_reason = ?, return_pct = ?, last_checked = ?
        WHERE id = ?
    """, (exit_reason, exit_price, now, exit_reason, return_pct, now, position_id))

    # Also record in in_trading_history for consistency
    pos = cursor.execute(
        "SELECT * FROM monitored_positions WHERE id = ?", (position_id,)
    ).fetchone()
    if pos:
        pos = dict(pos)
        holding_days = (datetime.now(IST) - datetime.strptime(
            pos["entry_date"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=IST)).days
        cursor.execute("""
            INSERT INTO in_trading_history
            (ticker, company_name, buy_price, buy_date, sell_price, sell_date,
             profit_rate, holding_days, scenario, trigger_type, trigger_mode, sector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos["ticker"], pos["company_name"], pos["entry_price"],
            pos["entry_date"], exit_price, now, return_pct, holding_days,
            json.dumps({"exit_reason": exit_reason, "quality_score": pos.get("quality_score")}),
            pos.get("trigger_type", ""), "morning", pos.get("sector", "")
        ))

    conn.commit()


def update_current_price(conn, position_id: int, price: float):
    """Update the live price for a position."""
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE monitored_positions
        SET current_price = ?, last_checked = ?
        WHERE id = ?
    """, (price, now, position_id))
    conn.commit()


CLEANUP_RETENTION_DAYS = 21  # Keep closed positions for 21 days, then purge


def cleanup_old_data(conn):
    """
    Remove closed positions older than CLEANUP_RETENTION_DAYS.
    Also remove old ingested file records.
    Trading history (in_trading_history) is kept permanently.
    """
    cursor = conn.cursor()
    cutoff = (datetime.now(IST) - timedelta(days=CLEANUP_RETENTION_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    # Delete closed positions older than retention period
    cursor.execute("""
        DELETE FROM monitored_positions
        WHERE status != 'OPEN' AND exit_date < ?
    """, (cutoff,))
    deleted_positions = cursor.rowcount

    # Delete old ingested file records (keep last 30 days)
    cutoff_files = (datetime.now(IST) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        DELETE FROM monitor_ingested_files
        WHERE ingested_at < ?
    """, (cutoff_files,))
    deleted_files = cursor.rowcount

    conn.commit()

    if deleted_positions > 0 or deleted_files > 0:
        logger.info(
            f"Cleanup: removed {deleted_positions} old positions, "
            f"{deleted_files} old file records"
        )


# ══════════════════════════════════════════════════════════════
# TRIGGER FILE INGESTION
# ══════════════════════════════════════════════════════════════

def _compute_max_exit_date(entry_date_str: str) -> str:
    """Compute exit date as entry_date + MAX_HOLD_TRADING_DAYS trading days."""
    entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d %H:%M:%S").date()
    current = entry_dt
    trading_days_counted = 0
    while trading_days_counted < MAX_HOLD_TRADING_DAYS:
        current = get_next_trading_day(current)
        trading_days_counted += 1
    return current.strftime("%Y-%m-%d")


def ingest_trigger_file(conn, filepath: Path) -> int:
    """
    Read a trigger_results JSON and insert new positions.
    Returns number of positions added.
    """
    cursor = conn.cursor()
    filename = filepath.name

    # Skip if already ingested
    existing = cursor.execute(
        "SELECT 1 FROM monitor_ingested_files WHERE filename = ?", (filename,)
    ).fetchone()
    if existing:
        return 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return 0

    metadata = data.get("metadata", {})
    regime = metadata.get("regime", {})

    # regime can be a string ("bear") or a dict ({"type": "bear", "max_picks": 0})
    if isinstance(regime, str):
        regime = {"type": regime}

    # Skip if regime explicitly sets 0 picks
    max_picks = regime.get("max_picks")
    if max_picks == 0:
        logger.info(f"Regime allows 0 picks in {filename}, skipping")
        cursor.execute(
            "INSERT INTO monitor_ingested_files (filename) VALUES (?)", (filename,)
        )
        conn.commit()
        return 0

    # Default: take top 5 if regime not specified
    if max_picks is None:
        max_picks = 5

    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    # Collect all stocks from all trigger categories
    skip_keys = {"screening_summary", "metadata"}
    all_picks = []
    for trigger_name, stocks in data.items():
        if trigger_name in skip_keys or not isinstance(stocks, list):
            continue
        for stock in stocks:
            stock["_trigger_type"] = trigger_name
            all_picks.append(stock)

    # Sort by final_score descending, take top max_picks
    all_picks.sort(key=lambda s: s.get("final_score", 0), reverse=True)
    selected = all_picks[:max_picks]

    for stock in selected:
        ticker = stock.get("ticker", "")
        if not ticker:
            continue

        # Skip if already monitoring this ticker
        existing_pos = cursor.execute(
            "SELECT 1 FROM monitored_positions WHERE ticker = ? AND status = 'OPEN'",
            (ticker,)
        ).fetchone()
        if existing_pos:
            logger.info(f"Already monitoring {ticker}, skipping")
            continue

        entry_price = stock.get("current_price", 0)
        stop_loss = stock.get("stop_loss_price", 0)
        target_price = stock.get("target_price", 0)

        if not all([entry_price, stop_loss, target_price]):
            logger.warning(f"Missing price data for {ticker}, skipping")
            continue

        max_exit_date = _compute_max_exit_date(now)

        cursor.execute("""
            INSERT INTO monitored_positions
            (ticker, company_name, entry_price, entry_date, stop_loss, take_profit,
             max_exit_date, trigger_type, sector, quality_score, final_score,
             change_rate, status, current_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (
            ticker,
            stock.get("name", ticker),
            entry_price,
            now,
            stop_loss,
            target_price,
            max_exit_date,
            stock.get("_trigger_type", ""),
            stock.get("metrics", {}).get("sector", "Unknown"),
            stock.get("quality_score", 0),
            stock.get("final_score", 0),
            stock.get("change_rate", 0),
            entry_price,
        ))
        added += 1
        logger.info(
            f"Added position: {ticker} @ ₹{entry_price:,.2f} "
            f"SL=₹{stop_loss:,.2f} TP=₹{target_price:,.2f} "
            f"Exit by {max_exit_date}"
        )

    cursor.execute(
        "INSERT INTO monitor_ingested_files (filename) VALUES (?)", (filename,)
    )
    conn.commit()
    logger.info(f"Ingested {filename}: {added} new positions")
    return added


def check_for_new_triggers(conn):
    """Scan trigger directory for un-ingested trigger files."""
    today_str = datetime.now(IST).strftime("%Y%m%d")
    pattern = f"trigger_results_in_morning_{today_str}.json"
    filepath = TRIGGER_DIR / pattern

    if filepath.exists():
        added = ingest_trigger_file(conn, filepath)
        if added > 0:
            logger.info(f"Ingested {added} positions from today's trigger file")


# ══════════════════════════════════════════════════════════════
# PRICE FETCHING
# ══════════════════════════════════════════════════════════════

def fetch_live_price(ticker: str) -> Optional[float]:
    """Fetch current price for an NSE ticker via yfinance."""
    try:
        yf_ticker = f"{ticker}.NS"
        data = yf.download(
            yf_ticker, period="1d", interval="1m",
            progress=False, timeout=PRICE_FETCH_TIMEOUT
        )
        if data is not None and not data.empty:
            price = float(data["Close"].iloc[-1])
            if hasattr(price, 'item'):
                price = price.item()
            return price

        # Fallback: use daily data
        data = yf.download(
            yf_ticker, period="2d", interval="1d",
            progress=False, timeout=PRICE_FETCH_TIMEOUT
        )
        if data is not None and not data.empty:
            price = float(data["Close"].iloc[-1])
            if hasattr(price, 'item'):
                price = price.item()
            return price

        logger.warning(f"No price data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        return None


def fetch_prices_batch(tickers: list) -> dict:
    """Fetch prices for multiple tickers. Returns {ticker: price}."""
    prices = {}
    if not tickers:
        return prices

    # yfinance batch download
    yf_tickers = " ".join(f"{t}.NS" for t in tickers)
    try:
        data = yf.download(
            yf_tickers, period="1d", interval="1m",
            progress=False, timeout=PRICE_FETCH_TIMEOUT, group_by="ticker"
        )
        if data is not None and not data.empty:
            for ticker in tickers:
                try:
                    yf_t = f"{ticker}.NS"
                    if len(tickers) == 1:
                        col = data["Close"]
                    else:
                        col = data[(yf_t, "Close")]
                    if not col.empty:
                        price = float(col.dropna().iloc[-1])
                        if hasattr(price, 'item'):
                            price = price.item()
                        prices[ticker] = price
                except (KeyError, IndexError):
                    pass
    except Exception as e:
        logger.warning(f"Batch download failed: {e}")

    # Fallback: individual fetch for missing tickers
    for ticker in tickers:
        if ticker not in prices:
            price = fetch_live_price(ticker)
            if price is not None:
                prices[ticker] = price

    return prices


# ══════════════════════════════════════════════════════════════
# TELEGRAM ALERTS
# ══════════════════════════════════════════════════════════════

_telegram_bot = None


def _get_telegram_bot():
    """Lazy-init Telegram bot."""
    global _telegram_bot
    if _telegram_bot is None:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        from telegram_bot_agent import TelegramBotAgent
        _telegram_bot = TelegramBotAgent()
    return _telegram_bot


async def send_alert(message: str, dry_run: bool = False):
    """Send alert via Telegram + Firebase."""
    logger.info(f"ALERT: {message}")

    if dry_run:
        logger.info("[DRY RUN] Would send Telegram message")
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        chat_id = os.environ.get("TELEGRAM_CHANNEL_ID") or os.environ.get("TELEGRAM_CHAT_ID")
        if not chat_id:
            logger.error("No TELEGRAM_CHANNEL_ID or TELEGRAM_CHAT_ID in .env")
            return

        bot = _get_telegram_bot()
        await bot.send_message(chat_id, message)
        logger.info("Telegram alert sent")
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")


async def send_exit_alert(position: dict, exit_price: float,
                          exit_reason: str, return_pct: float, dry_run: bool = False):
    """Send formatted exit alert."""
    ticker = position["ticker"]
    entry = position["entry_price"]

    if return_pct > 0:
        emoji = "🟢"
        arrow = "⬆️"
    elif return_pct < 0:
        emoji = "🔴"
        arrow = "⬇️"
    else:
        emoji = "⚪"
        arrow = "➖"

    reason_map = {
        "SL_HIT": "⛔ Stop Loss Hit",
        "TP_HIT": "🎯 Target Price Hit",
        "TIME_EXIT": "⏰ Max Hold Period (21 days)",
        "MANUAL": "🖐️ Manual Exit",
    }

    message = (
        f"{emoji} **EXIT: {position['company_name']}** (`{ticker}.NS`)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Reason: {reason_map.get(exit_reason, exit_reason)}\n"
        f"Entry: ₹{entry:,.2f}\n"
        f"Exit: ₹{exit_price:,.2f}\n"
        f"Return: {arrow} {abs(return_pct):.2f}%\n"
        f"Sector: {position.get('sector', 'Unknown')}\n"
        f"Trigger: {position.get('trigger_type', '')}\n"
    )
    await send_alert(message, dry_run=dry_run)


async def send_entry_alert(position: dict, dry_run: bool = False):
    """Send formatted entry alert."""
    message = (
        f"📈 **NEW POSITION: {position['company_name']}** (`{position['ticker']}.NS`)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: ₹{position['entry_price']:,.2f}\n"
        f"Stop Loss: ₹{position['stop_loss']:,.2f} "
        f"({((position['stop_loss'] - position['entry_price']) / position['entry_price'] * 100):+.1f}%)\n"
        f"Target: ₹{position['take_profit']:,.2f} "
        f"({((position['take_profit'] - position['entry_price']) / position['entry_price'] * 100):+.1f}%)\n"
        f"Max Exit: {position['max_exit_date']}\n"
        f"Quality: {position.get('quality_score', 0):.0f} | "
        f"Score: {position.get('final_score', 0):.3f}\n"
        f"Sector: {position.get('sector', 'Unknown')}\n"
    )
    await send_alert(message, dry_run=dry_run)


async def send_daily_summary(positions: list, dry_run: bool = False):
    """Send end-of-day summary of all open positions."""
    if not positions:
        return

    total_return = sum(
        ((p.get("current_price", p["entry_price"]) - p["entry_price"]) / p["entry_price"] * 100)
        for p in positions
    )
    avg_return = total_return / len(positions) if positions else 0

    lines = [
        f"📊 **Daily Position Summary** ({datetime.now(IST).strftime('%Y-%m-%d')})",
        f"━━━━━━━━━━━━━━━━━━━",
        f"Open Positions: {len(positions)} | Avg Return: {avg_return:+.2f}%",
        "",
    ]
    for p in positions:
        price = p.get("current_price", p["entry_price"])
        ret = (price - p["entry_price"]) / p["entry_price"] * 100
        arrow = "🟢" if ret > 0 else "🔴" if ret < 0 else "⚪"
        entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d %H:%M:%S")
        days_held = (datetime.now(IST).replace(tzinfo=None) - entry_dt).days
        lines.append(
            f"{arrow} `{p['ticker']}` ₹{price:,.2f} ({ret:+.1f}%) "
            f"Day {days_held}/{MAX_HOLD_TRADING_DAYS}"
        )

    message = "\n".join(lines)
    await send_alert(message, dry_run=dry_run)


# ══════════════════════════════════════════════════════════════
# DAILY TRIGGER SCHEDULING
# ══════════════════════════════════════════════════════════════


def get_trigger_filename(trade_date: str = None) -> str:
    """Get the trigger results filename for a given date."""
    if trade_date is None:
        trade_date = datetime.now(IST).strftime("%Y%m%d")
    return f"trigger_results_in_morning_{trade_date}.json"


def trigger_already_ran_today() -> bool:
    """Check if trigger batch already has results for today."""
    filepath = TRIGGER_DIR / get_trigger_filename()
    return filepath.exists()


def get_cached_trigger_results(trade_date: str = None) -> Optional[dict]:
    """Load cached trigger results for a date. Returns None if not found."""
    filepath = TRIGGER_DIR / get_trigger_filename(trade_date)
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def run_trigger_batch(trade_date: str = None) -> bool:
    """
    Run the morning trigger batch as a subprocess.
    Returns True if successful, False otherwise.
    Skips if results already exist for today (cached).
    """
    if trade_date is None:
        trade_date = datetime.now(IST).strftime("%Y%m%d")

    output_file = TRIGGER_DIR / get_trigger_filename(trade_date)

    # Skip if already ran today
    if output_file.exists():
        logger.info(f"Trigger batch already ran for {trade_date}, using cached results")
        return True

    logger.info(f"Running trigger batch for {trade_date}...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "prism-in" / "in_trigger_batch.py"),
                "morning", "INFO",
                "--output", str(output_file),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )
        if result.returncode == 0:
            logger.info(f"Trigger batch completed: {output_file.name}")
            return True
        else:
            logger.error(f"Trigger batch failed (exit {result.returncode}): {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Trigger batch timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Trigger batch error: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# DYNAMIC ATR RECALCULATION
# ══════════════════════════════════════════════════════════════

# Cache: ticker -> (timestamp, atr_value) to avoid redundant yfinance calls
_atr_cache: dict[str, tuple[float, float]] = {}


def compute_current_atr(ticker: str, trade_date_str: str) -> float:
    """
    Compute ATR₁₄ for a ticker using recent OHLCV data.
    Results are cached for ATR_RECALC_INTERVAL seconds.

    Returns ATR value, or 0 if data is unavailable.
    """
    now = time.time()
    cached = _atr_cache.get(ticker)
    if cached and (now - cached[0]) < ATR_RECALC_INTERVAL:
        return cached[1]

    try:
        df = get_multi_day_ohlcv(ticker, trade_date_str, lookback_days=ATR_PERIOD + 5)
        if df.empty or len(df) < 3 or "High" not in df.columns:
            return 0

        hi = df["High"].values
        lo = df["Low"].values
        cl = df["Close"].values
        tr_vals = []
        for i in range(1, len(df)):
            tr_vals.append(max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1])))

        atr = float(np.mean(tr_vals[-ATR_PERIOD:])) if tr_vals else 0
        _atr_cache[ticker] = (now, atr)
        return atr
    except Exception as e:
        logger.debug(f"ATR compute failed for {ticker}: {e}")
        return 0


def recalc_tp_sl(entry_price: float, atr: float) -> tuple[float, float]:
    """
    Recalculate take-profit and stop-loss from entry price and current ATR.
    Returns (new_tp, new_sl).
    """
    tp = entry_price + ATR_TP_MULT * atr
    sl = entry_price - ATR_SL_MULT * atr
    return tp, sl


# ══════════════════════════════════════════════════════════════
# CORE MONITORING LOOP
# ══════════════════════════════════════════════════════════════

async def check_positions(conn, dry_run: bool = False) -> dict:
    """
    Check all open positions against SL/TP/time limits.
    Returns summary dict.
    """
    positions = get_open_positions(conn)
    if not positions:
        return {"checked": 0, "exits": 0}

    tickers = [p["ticker"] for p in positions]
    logger.info(f"Checking {len(tickers)} positions: {', '.join(tickers)}")

    # Fetch all prices at once
    prices = fetch_prices_batch(tickers)

    exits = 0
    today = datetime.now(IST).date()
    today_str = today.strftime("%Y%m%d")

    for pos in positions:
        ticker = pos["ticker"]
        price = prices.get(ticker)

        if price is None:
            logger.warning(f"Could not get price for {ticker}, skipping")
            continue

        update_current_price(conn, pos["id"], price)
        entry_price = pos["entry_price"]
        target_price = pos["take_profit"]
        current_sl = pos["stop_loss"]

        # ── Dynamic ATR recalculation ──────────────────────────────
        # Recompute TP/SL daily using fresh ATR₁₄ anchored to entry price.
        # If volatility expanded since entry, TP moves up (more room to run);
        # if volatility contracted, TP tightens (lock in gains sooner).
        atr = compute_current_atr(ticker, today_str)
        if atr > 0:
            new_tp, new_sl = recalc_tp_sl(entry_price, atr)
            if new_tp != target_price or new_sl != current_sl:
                # Only widen SL from original — never tighten below ratcheted level
                if new_sl < current_sl:
                    new_sl = current_sl
                if new_tp != target_price or new_sl != current_sl:
                    logger.info(
                        f"ATR RECALC: {ticker} ATR₁₄=₹{atr:.2f} — "
                        f"TP ₹{target_price:,.2f}→₹{new_tp:,.2f}, "
                        f"SL ₹{current_sl:,.2f}→₹{new_sl:,.2f}"
                    )
                    conn.execute(
                        "UPDATE monitored_positions SET take_profit = ?, stop_loss = ? WHERE id = ?",
                        (new_tp, new_sl, pos["id"])
                    )
                    conn.commit()
                    target_price = new_tp
                    current_sl = new_sl

        return_pct = (price - entry_price) / entry_price * 100

        # King 2: Ratchet SL to breakeven when profit reaches 50% of target
        max_profit = target_price - entry_price
        current_profit = price - entry_price
        profit_ratio = current_profit / max_profit if max_profit > 0 else 0
        
        if profit_ratio >= 0.5 and current_sl < entry_price:
            # Ratchet SL to breakeven (entry price)
            logger.info(
                f"RATCHET: {ticker} profit at {profit_ratio*100:.0f}% of target, "
                f"moving SL from ₹{current_sl:,.2f} to breakeven ₹{entry_price:,.2f}"
            )
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE monitored_positions SET stop_loss = ? WHERE id = ?",
                (entry_price, pos["id"])
            )
            conn.commit()
            current_sl = entry_price

        # Check Stop Loss (may have been ratcheted)
        if price <= current_sl:
            logger.info(f"SL HIT: {ticker} @ ₹{price:,.2f} (SL=₹{current_sl:,.2f})")
            close_position(conn, pos["id"], price, "SL_HIT", return_pct)
            await send_exit_alert(pos, price, "SL_HIT", return_pct, dry_run)
            exits += 1
            continue

        # Check Take Profit
        if price >= target_price:
            logger.info(f"TP HIT: {ticker} @ ₹{price:,.2f} (TP=₹{target_price:,.2f})")
            close_position(conn, pos["id"], price, "TP_HIT", return_pct)
            await send_exit_alert(pos, price, "TP_HIT", return_pct, dry_run)
            exits += 1
            continue

        # Check max hold period
        max_exit = datetime.strptime(pos["max_exit_date"], "%Y-%m-%d").date()
        if today >= max_exit:
            logger.info(f"TIME EXIT: {ticker} @ ₹{price:,.2f} (max date={pos['max_exit_date']})")
            close_position(conn, pos["id"], price, "TIME_EXIT", return_pct)
            await send_exit_alert(pos, price, "TIME_EXIT", return_pct, dry_run)
            exits += 1
            continue

        # Log status
        logger.debug(
            f"  {ticker}: ₹{price:,.2f} ({return_pct:+.2f}%) "
            f"SL=₹{current_sl:,.2f} TP=₹{target_price:,.2f}"
        )

    return {"checked": len(positions), "exits": exits}


async def run_monitor(dry_run: bool = False, check_once: bool = False):
    """Main monitoring loop."""
    conn = init_db()
    logger.info("=" * 60)
    logger.info("Position Monitor started")
    logger.info(f"  DB: {DB_PATH}")
    logger.info(f"  Trigger dir: {TRIGGER_DIR}")
    logger.info(f"  Check interval: {CHECK_INTERVAL_SECONDS}s")
    logger.info(f"  Max hold days: {MAX_HOLD_TRADING_DAYS}")
    logger.info(f"  Dry run: {dry_run}")
    logger.info("=" * 60)

    last_daily_summary = None
    last_trigger_run = None  # Track which date we ran triggers for

    while not _shutdown:
        now = datetime.now(IST)
        today = now.date()

        # Check if market day
        if not is_nse_market_day(today):
            if check_once:
                logger.info("Not a market day. Exiting.")
                break
            logger.info(f"Not a market day. Sleeping {IDLE_INTERVAL_SECONDS}s...")
            await asyncio.sleep(IDLE_INTERVAL_SECONDS)
            continue

        # Market hours: 9:15 - 15:30 IST, start checking at 9:10
        market_start = now.replace(hour=9, minute=15 - PRE_MARKET_MINUTES, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        trigger_time = now.replace(hour=TRIGGER_RUN_HOUR, minute=TRIGGER_RUN_MINUTE, second=0)

        # Run trigger batch once per day at scheduled time
        if last_trigger_run != today and now >= trigger_time:
            logger.info("Running scheduled daily trigger batch...")
            success = run_trigger_batch()
            if success:
                last_trigger_run = today
            else:
                logger.error("Trigger batch failed, will retry next cycle")

        if now < market_start:
            if check_once:
                logger.info("Market not open yet. Exiting.")
                break
            wait_seconds = (market_start - now).total_seconds()
            logger.info(f"Market opens in {wait_seconds/60:.0f} min. Sleeping...")
            await asyncio.sleep(min(wait_seconds, IDLE_INTERVAL_SECONDS))
            continue

        if now > market_close:
            # After market close — send daily summary + cleanup once
            if last_daily_summary != today:
                positions = get_open_positions(conn)
                if positions:
                    # Final price check
                    await check_positions(conn, dry_run)
                    positions = get_open_positions(conn)  # Refresh after exits
                    await send_daily_summary(positions, dry_run)

                # Daily cleanup — purge closed positions older than 21 days
                cleanup_old_data(conn)
                last_daily_summary = today

            if check_once:
                break
            logger.info(f"Market closed. Sleeping {IDLE_INTERVAL_SECONDS}s...")
            await asyncio.sleep(IDLE_INTERVAL_SECONDS)
            continue

        # ── Market is open ────────────────────────────────────

        # 1. Check for new trigger files (morning picks)
        check_for_new_triggers(conn)

        # Send entry alerts for any newly added positions
        new_positions = get_open_positions(conn)
        for pos in new_positions:
            if pos.get("last_checked") is None:
                await send_entry_alert(pos, dry_run)

        # 2. Check prices against SL/TP/time
        result = await check_positions(conn, dry_run)
        logger.info(
            f"Check complete: {result['checked']} positions, "
            f"{result['exits']} exits"
        )

        if check_once:
            break

        # 3. Sleep until next check
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    conn.close()
    logger.info("Position Monitor stopped")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PRISM-INSIGHT Position Monitor")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without sending Telegram messages"
    )
    parser.add_argument(
        "--check-once", action="store_true",
        help="Run a single check and exit"
    )
    parser.add_argument(
        "--ingest", type=str, default=None,
        help="Manually ingest a trigger file (path)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current open positions and exit"
    )
    parser.add_argument(
        "--close", type=str, default=None,
        help="Manually close a position by ticker"
    )
    args = parser.parse_args()

    # Status command
    if args.status:
        conn = init_db()
        positions = get_open_positions(conn)
        if not positions:
            print("No open positions.")
        else:
            print(f"\n{'Ticker':<12} {'Entry':>10} {'Current':>10} {'SL':>10} {'TP':>10} {'Return':>8} {'Days':>5} {'Exit By'}")
            print("─" * 85)
            for p in positions:
                cur = p.get("current_price") or p["entry_price"]
                ret = (cur - p["entry_price"]) / p["entry_price"] * 100
                entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d %H:%M:%S")
                days = (datetime.now() - entry_dt).days
                print(
                    f"{p['ticker']:<12} "
                    f"₹{p['entry_price']:>9,.2f} "
                    f"₹{cur:>9,.2f} "
                    f"₹{p['stop_loss']:>9,.2f} "
                    f"₹{p['take_profit']:>9,.2f} "
                    f"{ret:>+7.2f}% "
                    f"{days:>5} "
                    f"{p['max_exit_date']}"
                )
        conn.close()
        return

    # Manual ingest
    if args.ingest:
        conn = init_db()
        path = Path(args.ingest)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        added = ingest_trigger_file(conn, path)
        print(f"Ingested {added} positions from {path.name}")
        conn.close()
        return

    # Manual close
    if args.close:
        conn = init_db()
        ticker = args.close.upper()
        cursor = conn.cursor()
        pos = cursor.execute(
            "SELECT * FROM monitored_positions WHERE ticker = ? AND status = 'OPEN'",
            (ticker,)
        ).fetchone()
        if not pos:
            print(f"No open position for {ticker}")
            conn.close()
            sys.exit(1)
        pos = dict(pos)
        price = fetch_live_price(ticker)
        if price is None:
            print(f"Could not fetch price for {ticker}. Enter manually:")
            price = float(input("Exit price: ₹"))
        return_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        close_position(conn, pos["id"], price, "MANUAL", return_pct)
        asyncio.run(send_exit_alert(pos, price, "MANUAL", return_pct, dry_run=args.dry_run))
        print(f"Closed {ticker} @ ₹{price:,.2f} ({return_pct:+.2f}%)")
        conn.close()
        return

    # Main monitor loop
    asyncio.run(run_monitor(dry_run=args.dry_run, check_once=args.check_once))


if __name__ == "__main__":
    main()
