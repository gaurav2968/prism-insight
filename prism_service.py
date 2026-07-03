#!/usr/bin/env python3
"""
PRISM Service — Unified background service for PRISM-INSIGHT

Runs as a single persistent process (Windows Service / Task Scheduler):
  1. Telegram Command Bot — handles /analyze, /morning, /positions, /close
  2. Position Monitor — tracks open positions, checks SL/TP every minute
  3. Daily Trigger — auto-runs morning trigger batch at 9:30 AM IST

Why one process?
  - One thing to install, monitor, restart
  - Shared asyncio event loop — no IPC overhead
  - Shared Telegram bot — no duplicate connections
  - Shared DB connection — no lock contention
  - Half the failure modes of two separate services

Usage:
    python prism_service.py                # Run everything
    python prism_service.py --no-bot       # Monitor only, no Telegram bot
    python prism_service.py --no-monitor   # Bot only, no position monitor
    python prism_service.py --dry-run      # No Telegram alerts from monitor
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
PRISM_IN_DIR = PROJECT_ROOT / "prism-in"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PRISM_IN_DIR))

from check_market_day import is_nse_market_day, is_market_open_now, get_next_trading_day

IST = pytz.timezone("Asia/Kolkata")

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "prism_service.log", encoding="utf-8"),
    ],
    force=True,  # Override any prior basicConfig from imported modules
)
logger = logging.getLogger("prism_service")

# ── Config ─────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

DB_PATH = PROJECT_ROOT / "stock_tracking_db.sqlite"
TRIGGER_DIR = PROJECT_ROOT

CHECK_INTERVAL_SECONDS = 60
IDLE_INTERVAL_SECONDS = 300
MAX_HOLD_TRADING_DAYS = 21
TRIGGER_RUN_HOUR = 9
TRIGGER_RUN_MINUTE = 30
PRICE_FETCH_TIMEOUT = 10

# Concurrency
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
MAX_JOBS_PER_USER = int(os.getenv("MAX_JOBS_PER_USER", "1"))

# ── Rate Limiting & Abuse Protection ──────────────────────────────────────
# Per-user message rate limit
RATE_LIMIT_MESSAGES = int(os.getenv("RATE_LIMIT_MESSAGES", "20"))       # max messages
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))           # per N seconds
# Daily caps (reset at midnight IST)
DAILY_NLP_CALLS_MAX = int(os.getenv("DAILY_NLP_CALLS_MAX", "50"))       # GPT calls/day/user
DAILY_ANALYSIS_MAX = int(os.getenv("DAILY_ANALYSIS_MAX", "10"))          # /analyze jobs/day/user
DAILY_MORNING_MAX = int(os.getenv("DAILY_MORNING_MAX", "5"))             # /morning runs/day/user
# Global daily caps (all users combined)
GLOBAL_NLP_DAILY_MAX = int(os.getenv("GLOBAL_NLP_DAILY_MAX", "200"))
GLOBAL_ANALYSIS_DAILY_MAX = int(os.getenv("GLOBAL_ANALYSIS_DAILY_MAX", "30"))

_raw_ids = os.getenv("ALLOWED_TELEGRAM_USER_IDS", "")
ALLOWED_USER_IDS: set[int] = (
    {int(x.strip()) for x in _raw_ids.split(",") if x.strip().isdigit()}
    if _raw_ids.strip()
    else set()
)

# ── Shared state ───────────────────────────────────────────────────────────
_shutdown = False
_running_jobs: dict[str, bool] = {}
_job_semaphore: asyncio.Semaphore = None  # initialized in main
_user_active_jobs: dict[int, int] = {}

# ── Rate limit tracking ───────────────────────────────────────────────────
from collections import defaultdict
import time as _time

# Per-user sliding window: {user_id: [timestamp, timestamp, ...]}
_user_msg_timestamps: dict[int, list] = defaultdict(list)
# Daily counters: {(user_id, "nlp"): count, (user_id, "analysis"): count}
_daily_counters: dict[tuple, int] = defaultdict(int)
_daily_counter_date: str = ""  # reset when date changes
# Global daily counters
_global_counters: dict[str, int] = defaultdict(int)
# Blocked users (auto-ban after repeated violations)
_violation_counts: dict[int, int] = defaultdict(int)
_blocked_users: set[int] = set()
BLOCK_AFTER_VIOLATIONS = 10  # auto-block after 10 violations in a session


def _reset_daily_counters_if_needed():
    """Reset all daily counters at midnight IST."""
    global _daily_counter_date
    today = datetime.now(IST).strftime("%Y%m%d")
    if _daily_counter_date != today:
        _daily_counters.clear()
        _global_counters.clear()
        _daily_counter_date = today


def _check_rate_limit(user_id: int) -> Optional[str]:
    """
    Check if user is rate-limited. Returns error message if blocked, None if OK.
    """
    # Auto-blocked users
    if user_id in _blocked_users:
        return "⛔ You have been temporarily blocked due to excessive requests."

    _reset_daily_counters_if_needed()
    now = _time.time()

    # Sliding window rate limit
    timestamps = _user_msg_timestamps[user_id]
    # Remove old timestamps outside window
    cutoff = now - RATE_LIMIT_WINDOW
    _user_msg_timestamps[user_id] = [t for t in timestamps if t > cutoff]
    timestamps = _user_msg_timestamps[user_id]

    if len(timestamps) >= RATE_LIMIT_MESSAGES:
        _violation_counts[user_id] += 1
        if _violation_counts[user_id] >= BLOCK_AFTER_VIOLATIONS:
            _blocked_users.add(user_id)
            logger.warning(f"Auto-blocked user {user_id} after {BLOCK_AFTER_VIOLATIONS} rate limit violations")
            return "⛔ You have been temporarily blocked due to excessive requests."
        wait = int(RATE_LIMIT_WINDOW - (now - timestamps[0]))
        return f"⏳ Rate limit: {RATE_LIMIT_MESSAGES} messages per {RATE_LIMIT_WINDOW}s. Wait {wait}s."

    # Record this message
    _user_msg_timestamps[user_id].append(now)
    return None


def _check_daily_limit(user_id: int, action: str, limit: int) -> Optional[str]:
    """Check per-user daily limit for an action."""
    _reset_daily_counters_if_needed()
    key = (user_id, action)
    if _daily_counters[key] >= limit:
        return f"⏳ Daily limit reached: {limit} {action} requests per day."
    return None


def _check_global_limit(action: str, limit: int) -> Optional[str]:
    """Check global daily limit across all users."""
    _reset_daily_counters_if_needed()
    if _global_counters[action] >= limit:
        return f"⏳ System daily limit reached for {action}. Try again tomorrow."
    return None


def _increment_counter(user_id: int, action: str):
    """Increment both per-user and global counters."""
    _reset_daily_counters_if_needed()
    _daily_counters[(user_id, action)] += 1
    _global_counters[action] += 1


def _signal_handler(signum, frame):
    global _shutdown
    logger.info(f"Signal {signum} received, shutting down...")
    _shutdown = True


# ══════════════════════════════════════════════════════════════════════════
# DATABASE (shared connection)
# ══════════════════════════════════════════════════════════════════════════

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        CREATE INDEX IF NOT EXISTS idx_mon_pos_status ON monitored_positions(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_mon_pos_ticker ON monitored_positions(ticker)
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitor_ingested_files (
            filename TEXT PRIMARY KEY,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS service_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════
# POSITION MONITOR LOGIC (imported from position_monitor.py)
# ══════════════════════════════════════════════════════════════════════════

# Re-use the existing module — don't duplicate code
from position_monitor import (
    get_open_positions,
    close_position,
    update_current_price,
    cleanup_old_data,
    check_for_new_triggers,
    fetch_prices_batch,
    trigger_already_ran_today,
    get_cached_trigger_results,
    run_trigger_batch,
    get_trigger_filename,
    get_service_state,
    set_service_state,
    compute_current_atr,
    recalc_tp_sl,
)


async def check_positions(conn, bot=None, chat_id=None, dry_run=False) -> dict:
    """Check all open positions. Send alerts via shared bot if available."""
    positions = get_open_positions(conn)
    if not positions:
        return {"checked": 0, "exits": 0}

    tickers = [p["ticker"] for p in positions]
    logger.info(f"[Monitor] Checking {len(tickers)} positions: {', '.join(tickers)}")

    prices = fetch_prices_batch(tickers)
    exits = 0
    today = datetime.now(IST).date()
    today_str = today.strftime("%Y%m%d")

    for pos in positions:
        ticker = pos["ticker"]
        price = prices.get(ticker)
        if price is None:
            continue

        update_current_price(conn, pos["id"], price)
        entry_price = pos["entry_price"]
        target_price = pos["take_profit"]
        current_sl = pos["stop_loss"]

        # ── Dynamic ATR recalculation ──────────────────────────────
        atr = compute_current_atr(ticker, today_str)
        if atr > 0:
            new_tp, new_sl = recalc_tp_sl(entry_price, atr)
            if new_sl < current_sl:
                new_sl = current_sl  # Never tighten SL below ratcheted level
            if new_tp != target_price or new_sl != current_sl:
                logger.info(
                    f"[Monitor] ATR RECALC: {ticker} ATR₁₄=₹{atr:.2f} — "
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

        exit_reason = None
        if price <= current_sl:
            exit_reason = "SL_HIT"
        elif price >= target_price:
            exit_reason = "TP_HIT"
        else:
            max_exit = datetime.strptime(pos["max_exit_date"], "%Y-%m-%d").date()
            if today >= max_exit:
                exit_reason = "TIME_EXIT"

        if exit_reason:
            close_position(conn, pos["id"], price, exit_reason, return_pct)
            exits += 1
            msg = _format_exit_message(pos, price, exit_reason, return_pct)
            logger.info(f"[Monitor] {exit_reason}: {ticker} @ ₹{price:,.2f} ({return_pct:+.2f}%)")
            if bot and chat_id and not dry_run:
                try:
                    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                except Exception as e:
                    logger.error(f"Alert send failed: {e}")

    return {"checked": len(positions), "exits": exits}


def _format_exit_message(pos, exit_price, exit_reason, return_pct):
    emoji = "🟢" if return_pct > 0 else "🔴" if return_pct < 0 else "⚪"
    arrow = "⬆️" if return_pct > 0 else "⬇️" if return_pct < 0 else "➖"
    reason_map = {
        "SL_HIT": "⛔ Stop Loss Hit",
        "TP_HIT": "🎯 Target Price Hit",
        "TIME_EXIT": "⏰ Max Hold Period (21 days)",
    }
    return (
        f"{emoji} *EXIT: {pos['company_name']}* (`{pos['ticker']}.NS`)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Reason: {reason_map.get(exit_reason, exit_reason)}\n"
        f"Entry: ₹{pos['entry_price']:,.2f}\n"
        f"Exit: ₹{exit_price:,.2f}\n"
        f"Return: {arrow} {abs(return_pct):.2f}%\n"
        f"Sector: {pos.get('sector', 'Unknown')}\n"
    )


def _format_entry_message(pos):
    return (
        f"📈 *NEW POSITION: {pos['company_name']}* (`{pos['ticker']}.NS`)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: ₹{pos['entry_price']:,.2f}\n"
        f"Stop Loss: ₹{pos['stop_loss']:,.2f} "
        f"({((pos['stop_loss'] - pos['entry_price']) / pos['entry_price'] * 100):+.1f}%)\n"
        f"Target: ₹{pos['take_profit']:,.2f} "
        f"({((pos['take_profit'] - pos['entry_price']) / pos['entry_price'] * 100):+.1f}%)\n"
        f"Max Exit: {pos['max_exit_date']}\n"
        f"Quality: {pos.get('quality_score', 0):.0f} | "
        f"Score: {pos.get('final_score', 0):.3f}\n"
    )


def _format_daily_summary(positions):
    if not positions:
        return None
    total_return = sum(
        ((p.get("current_price", p["entry_price"]) - p["entry_price"]) / p["entry_price"] * 100)
        for p in positions
    )
    avg_return = total_return / len(positions)

    lines = [
        f"📊 *Daily Summary* ({datetime.now(IST).strftime('%Y-%m-%d')})",
        f"━━━━━━━━━━━━━━━━━━━",
        f"Open: {len(positions)} | Avg Return: {avg_return:+.2f}%",
        "",
    ]
    for p in positions:
        price = p.get("current_price", p["entry_price"])
        ret = (price - p["entry_price"]) / p["entry_price"] * 100
        arrow = "🟢" if ret > 0 else "🔴" if ret < 0 else "⚪"
        entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d %H:%M:%S")
        days = (datetime.now(IST).replace(tzinfo=None) - entry_dt).days
        lines.append(f"{arrow} `{p['ticker']}` ₹{price:,.2f} ({ret:+.1f}%) Day {days}/{MAX_HOLD_TRADING_DAYS}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MONITOR BACKGROUND TASK
# ══════════════════════════════════════════════════════════════════════════

async def monitor_loop(conn, bot=None, dry_run=False):
    """Background task: position monitoring + daily trigger scheduling."""
    logger.info("[Monitor] Background monitor started")

    # Restore state from DB so restarts don't re-trigger
    _saved_summary = get_service_state(conn, "last_daily_summary")
    last_daily_summary = datetime.strptime(_saved_summary, "%Y-%m-%d").date() if _saved_summary else None
    _saved_trigger = get_service_state(conn, "last_trigger_run")
    last_trigger_run = datetime.strptime(_saved_trigger, "%Y-%m-%d").date() if _saved_trigger else None
    logger.info(f"[Monitor] Restored state: last_summary={last_daily_summary}, last_trigger={last_trigger_run}")
    chat_id = CHANNEL_ID

    while not _shutdown:
        try:
            now = datetime.now(IST)
            today = now.date()

            if not is_nse_market_day(today):
                await asyncio.sleep(IDLE_INTERVAL_SECONDS)
                continue

            market_start = now.replace(hour=9, minute=10, second=0)
            market_close = now.replace(hour=15, minute=30, second=0)
            trigger_time = now.replace(hour=TRIGGER_RUN_HOUR, minute=TRIGGER_RUN_MINUTE, second=0)

            # Daily trigger batch
            if last_trigger_run != today and now >= trigger_time:
                logger.info("[Monitor] Running scheduled trigger batch...")
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(None, run_trigger_batch)
                if success:
                    last_trigger_run = today
                    set_service_state(conn, "last_trigger_run", str(today))
                    # Ingest new positions
                    check_for_new_triggers(conn)
                    # Alert for new entries
                    new_entries = 0
                    for pos in get_open_positions(conn):
                        if pos.get("last_checked") is None and bot and chat_id and not dry_run:
                            new_entries += 1
                            try:
                                await bot.send_message(
                                    chat_id=chat_id,
                                    text=_format_entry_message(pos),
                                    parse_mode="Markdown",
                                )
                            except Exception as e:
                                logger.error(f"Entry alert failed: {e}")
                    # Notify when trigger ran but no new picks
                    if new_entries == 0 and bot and chat_id and not dry_run:
                        trigger_data = get_cached_trigger_results()
                        meta = trigger_data.get("metadata", {}) if trigger_data else {}
                        regime = meta.get("regime", "unknown")
                        msg = meta.get("message", "")
                        nifty_rsi = meta.get("nifty_rsi", "?")
                        text = (
                            f"📋 *Morning Scan Complete* ({today.strftime('%d %b %Y')})\n\n"
                            f"No new picks today.\n"
                            f"Regime: *{regime}* (Nifty RSI: {nifty_rsi})\n"
                        )
                        if msg:
                            text += f"_{msg}_"
                        try:
                            await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                        except Exception as e:
                            logger.error(f"No-picks notification failed: {e}")

            if now < market_start:
                await asyncio.sleep(min((market_start - now).total_seconds(), IDLE_INTERVAL_SECONDS))
                continue

            if now > market_close:
                if last_daily_summary != today:
                    # Final check + daily summary
                    await check_positions(conn, bot, chat_id, dry_run)
                    positions = get_open_positions(conn)
                    summary = _format_daily_summary(positions)
                    if summary and bot and chat_id and not dry_run:
                        try:
                            await bot.send_message(chat_id=chat_id, text=summary, parse_mode="Markdown")
                        except Exception as e:
                            logger.error(f"Summary send failed: {e}")
                    cleanup_old_data(conn)
                    last_daily_summary = today
                    set_service_state(conn, "last_daily_summary", str(today))

                await asyncio.sleep(IDLE_INTERVAL_SECONDS)
                continue

            # Market is open — check positions
            check_for_new_triggers(conn)
            await check_positions(conn, bot, chat_id, dry_run)
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

        except Exception as e:
            logger.error(f"[Monitor] Error in loop: {e}", exc_info=True)
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)


# ══════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT COMMANDS
# ══════════════════════════════════════════════════════════════════════════

from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest

_db_conn: sqlite3.Connection = None  # Set in main


def _is_authorised(user_id: int) -> bool:
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def _deny(update: Update):
    await update.message.reply_text("⛔ You are not authorised. Contact the bot owner.")


def _user_can_start_job(user_id: int) -> bool:
    return _user_active_jobs.get(user_id, 0) < MAX_JOBS_PER_USER


def _user_job_start(user_id: int):
    _user_active_jobs[user_id] = _user_active_jobs.get(user_id, 0) + 1


def _user_job_end(user_id: int):
    _user_active_jobs[user_id] = max(0, _user_active_jobs.get(user_id, 0) - 1)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🇮🇳 *PRISM India — Stock Analysis Bot*\n\n"
        "*Commands:*\n"
        "/morning `[YYYYMMDD]` — Run or view morning analysis\n"
        "/analyze `TICKER [...]` — Full AI analysis + PDF\n"
        "/positions — View open monitored positions\n"
        "/close `TICKER` — Manually close a position\n"
        "/status — Show running jobs\n"
        "/help — This message\n\n"
        "_You can also just type naturally:_\n"
        "• _analyze Reliance_\n"
        "• _how's TCS doing?_\n"
        "• _run morning pipeline_\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorised(update.effective_user.id):
        await _deny(update)
        return
    rate_err = _check_rate_limit(update.effective_user.id)
    if rate_err:
        await update.message.reply_text(rate_err)
        return

    active = [k for k, v in _running_jobs.items() if v]
    positions = get_open_positions(_db_conn) if _db_conn else []
    today = datetime.now(IST).strftime("%Y%m%d")
    has_trigger = (TRIGGER_DIR / f"trigger_results_in_morning_{today}.json").exists()

    lines = [
        "📊 *PRISM Service Status*",
        f"━━━━━━━━━━━━━━━━━━━",
        f"Market Open: {'✅ Yes' if is_market_open_now() else '❌ No'}",
        f"Today's Trigger: {'✅ Done' if has_trigger else '⏳ Pending'}",
        f"Open Positions: {len(positions)}",
        f"Active Jobs: {len(active)}",
    ]
    if active:
        lines.append(f"Running: {', '.join(active)}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all open monitored positions with live P&L."""
    if not _is_authorised(update.effective_user.id):
        await _deny(update)
        return
    rate_err = _check_rate_limit(update.effective_user.id)
    if rate_err:
        await update.message.reply_text(rate_err)
        return

    positions = get_open_positions(_db_conn) if _db_conn else []
    if not positions:
        await update.message.reply_text("No open positions.")
        return

    # Fetch live prices
    tickers = [p["ticker"] for p in positions]
    prices = fetch_prices_batch(tickers)

    lines = ["📊 *Open Positions*", "━━━━━━━━━━━━━━━━━━━"]
    total_return = 0
    for p in positions:
        price = prices.get(p["ticker"]) or p.get("current_price") or p["entry_price"]
        ret = (price - p["entry_price"]) / p["entry_price"] * 100
        total_return += ret
        entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d %H:%M:%S")
        days = (datetime.now(IST).replace(tzinfo=None) - entry_dt).days
        arrow = "🟢" if ret > 0 else "🔴" if ret < 0 else "⚪"

        lines.append(
            f"\n{arrow} *{p['ticker']}* ({p.get('sector', '')})\n"
            f"  Entry: ₹{p['entry_price']:,.2f} → Now: ₹{price:,.2f} ({ret:+.1f}%)\n"
            f"  SL: ₹{p['stop_loss']:,.2f} | TP: ₹{p['take_profit']:,.2f}\n"
            f"  Day {days}/{MAX_HOLD_TRADING_DAYS} | Exit by {p['max_exit_date']}"
        )

    avg = total_return / len(positions) if positions else 0
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Avg Return: {avg:+.2f}% across {len(positions)} positions")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually close a position: /close TICKER"""
    if not _is_authorised(update.effective_user.id):
        await _deny(update)
        return
    rate_err = _check_rate_limit(update.effective_user.id)
    if rate_err:
        await update.message.reply_text(rate_err)
        return

    if not context.args:
        await update.message.reply_text("Usage: /close TICKER\nExample: /close RELIANCE")
        return

    ticker = context.args[0].strip().upper().replace(".NS", "")

    cursor = _db_conn.cursor()
    pos = cursor.execute(
        "SELECT * FROM monitored_positions WHERE ticker = ? AND status = 'OPEN'",
        (ticker,)
    ).fetchone()
    if not pos:
        await update.message.reply_text(f"No open position for `{ticker}`")
        return

    pos = dict(pos)
    # Fetch live price
    prices = fetch_prices_batch([ticker])
    price = prices.get(ticker)
    if price is None:
        await update.message.reply_text(f"Cannot fetch price for `{ticker}`. Try again.")
        return

    return_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
    close_position(_db_conn, pos["id"], price, "MANUAL", return_pct)

    msg = _format_exit_message(pos, price, "MANUAL", return_pct)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    # Also notify channel
    if CHANNEL_ID:
        try:
            await context.bot.send_message(chat_id=CHANNEL_ID, text=msg, parse_mode="Markdown")
        except Exception:
            pass


async def cmd_morning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /morning [YYYYMMDD]
    Show cached results or run pipeline if not yet done.
    Always replies in the user's chat.
    """
    if not _is_authorised(update.effective_user.id):
        await _deny(update)
        return

    user_id = update.effective_user.id

    # Rate limit
    rate_err = _check_rate_limit(user_id)
    if rate_err:
        await update.message.reply_text(rate_err)
        return

    date_arg = None
    if context.args:
        raw = context.args[0].strip()
        if re.fullmatch(r"\d{8}", raw):
            date_arg = raw

    reference_date = date_arg or datetime.now(IST).strftime("%Y%m%d")

    # Check cache first
    cached = get_cached_trigger_results(reference_date)
    if cached:
        msg = _format_trigger_summary(cached, reference_date)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        return

    # No cache — run trigger batch (not full pipeline, just triggers)
    # Daily limit check (only counts when we actually run, not for cached)
    morning_err = _check_daily_limit(user_id, "morning", DAILY_MORNING_MAX)
    if morning_err:
        await update.message.reply_text(morning_err)
        return

    _increment_counter(user_id, "morning")
    job_key = f"morning_{reference_date}"

    if _running_jobs.get(job_key):
        await update.message.reply_text(f"⏳ Already running for {reference_date}.")
        return

    if not _user_can_start_job(user_id):
        await update.message.reply_text("⏳ You have a job running. Please wait.")
        return

    async with _job_semaphore:
        _running_jobs[job_key] = True
        _user_job_start(user_id)
        await update.message.reply_text(
            f"🌅 Running trigger batch for *{reference_date}*...\n"
            f"_This takes ~2-5 minutes._",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, lambda: run_trigger_batch(reference_date))
            if success:
                cached = get_cached_trigger_results(reference_date)
                if cached:
                    msg = _format_trigger_summary(cached, reference_date)
                    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

                    # Ingest positions
                    check_for_new_triggers(_db_conn)
                else:
                    await update.message.reply_text("✅ Trigger batch done but no results file found.")
            else:
                await update.message.reply_text("❌ Trigger batch failed. Check logs.")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: `{str(e)[:300]}`", parse_mode=ParseMode.MARKDOWN)
        finally:
            _running_jobs[job_key] = False
            _user_job_end(user_id)


def _format_trigger_summary(data: dict, trade_date: str) -> str:
    """Format cached trigger JSON into a readable Telegram message."""
    metadata = data.get("metadata", {})
    regime = metadata.get("regime", {})
    skip_keys = {"screening_summary", "metadata"}

    formatted_date = f"{trade_date[:4]}.{trade_date[4:6]}.{trade_date[6:8]}"
    lines = [f"🔔 *Morning Signal — {formatted_date}*", "━━━━━━━━━━━━━━━━━━━"]

    if regime:
        regime_type = regime.get("type", "unknown")
        emoji = "🟢" if "bull" in regime_type else "🔴" if "bear" in regime_type else "🟡"
        lines.append(
            f"{emoji} Regime: *{regime_type}* | "
            f"NIFTY RSI: {regime.get('nifty_rsi', 'N/A'):.1f} | "
            f"Picks: {regime.get('max_picks', 'N/A')}"
        )
        lines.append("")

    total_picks = 0
    for trigger_name, stocks in data.items():
        if trigger_name in skip_keys or not isinstance(stocks, list) or not stocks:
            continue
        lines.append(f"*{trigger_name}*")
        for s in stocks[:10]:
            ticker = s.get("ticker", "?")
            price = s.get("current_price", 0)
            change = s.get("change_rate", 0)
            quality = s.get("quality_score", 0)
            score = s.get("final_score", 0)
            sl = s.get("stop_loss_price", 0)
            tp = s.get("target_price", 0)
            sl_pct = ((sl - price) / price * 100) if price else 0
            tp_pct = ((tp - price) / price * 100) if price else 0
            lines.append(
                f"  `{ticker}` ₹{price:,.2f} ({change:+.1f}%)\n"
                f"    SL ₹{sl:,.0f} ({sl_pct:+.1f}%) → TP ₹{tp:,.0f} ({tp_pct:+.1f}%)\n"
                f"    Q:{quality:.0f} Score:{score:.3f}"
            )
            total_picks += 1
        lines.append("")

    if total_picks == 0:
        lines.append("_No stocks triggered (regime blocks picks)._")

    return "\n".join(lines)


# NLP intent classification — uses GitHub AI (same as telegram_command_bot.py)
_ai_client = None

def _get_ai_client():
    global _ai_client
    if _ai_client is None:
        from openai import AsyncOpenAI
        _ai_client = AsyncOpenAI(
            api_key=os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY", ""),
            base_url="https://models.github.ai/inference",
        )
    return _ai_client


async def classify_intent(text: str) -> dict:
    """Use GPT-4.1 to classify user intent."""
    _INTENT_SYSTEM_PROMPT = """
You are the intent classifier for PRISM India, an AI-powered NSE stock analysis bot.
A user has sent a message. Extract the intent and any relevant parameters.

Return ONLY valid JSON — no markdown, no explanation — with this exact schema:
{
  "intent": "<analyze|morning_pipeline|positions|close|question|help|unknown>",
  "tickers": ["TICKER1", "TICKER2"],   // NSE symbols, uppercase, no .NS suffix. Empty list if not applicable.
  "date": "YYYYMMDD or null",           // specific date if mentioned, otherwise null
  "reply": "<short friendly reply to show the user before acting, or a direct answer for 'question' intent>"
}

Intent meanings:
- analyze       : user wants an analysis/report for one or more specific stocks
- morning_pipeline : user wants to run today's (or a dated) morning scan
- positions     : user asks about current portfolio/holdings/open positions
- close         : user wants to close/sell a specific stock position
- question      : user is asking a general question you can answer directly (e.g. "what is P/E ratio?", or just casual chat like "hey", "how are you")
- help          : user wants to know what the bot can do
- unknown       : message is truly unclear or unrelated

For NSE tickers: normalise common names (e.g. "Reliance" → "RELIANCE", "TCS" → "TCS",
"Infosys" → "INFY", "HDFC Bank" → "HDFCBANK", "ITC" → "ITC").
Only include tickers you are confident about.

IMPORTANT: For casual greetings like "hey", "hello", "hi", "what's up" — use intent=question and reply with a friendly greeting.
"""
    try:
        client = _get_ai_client()
        response = await client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=300,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if the model wraps output
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        return {"intent": "unknown", "tickers": [], "date": None, "reply": "Sorry, I couldn't understand that. Try /help to see what I can do."}


def _normalise_ticker(raw: str) -> str:
    return raw.strip().upper().replace(".NS", "").replace(".BO", "")


async def handle_natural_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """NLP handler — understand plain text and route to commands."""
    if not update.message or not update.message.text:
        return
    if not _is_authorised(update.effective_user.id):
        await _deny(update)
        return

    user_id = update.effective_user.id
    user_text = update.message.text.strip()
    if not user_text:
        return

    # Rate limit check
    rate_err = _check_rate_limit(user_id)
    if rate_err:
        await update.message.reply_text(rate_err)
        return

    # Daily NLP call limit (GPT tokens cost money)
    nlp_err = _check_daily_limit(user_id, "nlp", DAILY_NLP_CALLS_MAX)
    if nlp_err:
        await update.message.reply_text(nlp_err)
        return
    global_err = _check_global_limit("nlp", GLOBAL_NLP_DAILY_MAX)
    if global_err:
        await update.message.reply_text(global_err)
        return

    _increment_counter(user_id, "nlp")

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    intent = await classify_intent(user_text)
    action = intent.get("intent", "unknown")
    tickers = [_normalise_ticker(t) for t in intent.get("tickers", []) if t.strip()]
    date_arg = intent.get("date")
    reply_text = intent.get("reply", "")

    logger.info(f"NLP: '{user_text}' → {action}, tickers={tickers}")

    if action == "analyze" and tickers:
        if reply_text:
            await update.message.reply_text(reply_text)
        context.args = tickers
        await cmd_analyze(update, context)
    elif action == "morning_pipeline":
        if reply_text:
            await update.message.reply_text(reply_text)
        context.args = [date_arg] if date_arg else []
        await cmd_morning(update, context)
    elif action == "positions":
        await cmd_positions(update, context)
    elif action == "close" and tickers:
        context.args = tickers[:1]
        await cmd_close(update, context)
    elif action == "help":
        await cmd_help(update, context)
    elif action == "question":
        await update.message.reply_text(reply_text or "I'm not sure. Try /help")
    else:
        await update.message.reply_text(
            reply_text or "Not sure what you need. Try:\n• _analyze Reliance_\n• _show positions_\n• _run morning pipeline_",
            parse_mode=ParseMode.MARKDOWN,
        )


# Import analyze command from existing bot (heavy analysis — keep as subprocess)
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run full AI analysis for tickers. Delegates to existing orchestrator."""
    if not _is_authorised(update.effective_user.id):
        await _deny(update)
        return

    user_id = update.effective_user.id

    # Rate limit
    rate_err = _check_rate_limit(user_id)
    if rate_err:
        await update.message.reply_text(rate_err)
        return

    tickers = [_normalise_ticker(t) for t in (context.args or [])]
    if not tickers:
        await update.message.reply_text("Usage: /analyze TICKER1 [TICKER2 ...]\nExample: /analyze RELIANCE TCS")
        return

    # Cap ticker count per request
    if len(tickers) > 5:
        await update.message.reply_text("⚠️ Max 5 tickers per request.")
        tickers = tickers[:5]

    # Daily analysis limit
    analysis_err = _check_daily_limit(user_id, "analysis", DAILY_ANALYSIS_MAX)
    if analysis_err:
        await update.message.reply_text(analysis_err)
        return
    global_err = _check_global_limit("analysis", GLOBAL_ANALYSIS_DAILY_MAX)
    if global_err:
        await update.message.reply_text(global_err)
        return

    _increment_counter(user_id, "analysis")

    user_id = update.effective_user.id
    plural = "s" if len(tickers) > 1 else ""
    job_key = f"analyze_{'_'.join(tickers)}"

    if _running_jobs.get(job_key):
        await update.message.reply_text(f"⏳ Already analyzing {', '.join(tickers)}")
        return
    if not _user_can_start_job(user_id):
        await update.message.reply_text("⏳ You have a job running. Please wait.")
        return

    async with _job_semaphore:
        _running_jobs[job_key] = True
        _user_job_start(user_id)

        await update.message.reply_text(
            f"🔍 Analyzing {len(tickers)} stock{plural}: *{', '.join(tickers)}*\n"
            f"_~5-10 min per stock_",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:
            for i, ticker in enumerate(tickers, 1):
                progress = f"[{i}/{len(tickers)}]"
                try:
                    python = sys.executable
                    script = str(PRISM_IN_DIR / "in_stock_analysis_orchestrator.py")
                    cmd = [python, script, "--mode", "morning", "--no-telegram",
                           "--analyze-only", ticker]

                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=str(PROJECT_ROOT),
                    )
                    stdout, _ = await proc.communicate()

                    if proc.returncode == 0:
                        # Find and send PDF
                        pdf_dir = PRISM_IN_DIR / "pdf_reports"
                        today_str = datetime.now().strftime("%Y%m%d")
                        pdfs = sorted(pdf_dir.glob(f"*{ticker}*{today_str}*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if pdfs:
                            await update.message.reply_document(document=open(pdfs[0], "rb"))
                        else:
                            await update.message.reply_text(f"✅ {progress} Analysis done for *{ticker}* (no PDF found)", parse_mode=ParseMode.MARKDOWN)
                    else:
                        output = stdout.decode("utf-8", errors="replace")[-500:]
                        await update.message.reply_text(
                            f"❌ {progress} *{ticker}* failed:\n```\n{output}\n```",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                except Exception as e:
                    await update.message.reply_text(
                        f"❌ {progress} Error for *{ticker}*: `{str(e)[:200]}`",
                        parse_mode=ParseMode.MARKDOWN,
                    )

        finally:
            _running_jobs[job_key] = False
            _user_job_end(user_id)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — unified entry point
# ══════════════════════════════════════════════════════════════════════════

async def post_init(app: Application):
    """Called after bot starts. Launch monitor as background task."""
    global _db_conn
    _db_conn = init_db()

    dry_run = getattr(app, '_prism_dry_run', False)
    no_monitor = getattr(app, '_prism_no_monitor', False)

    if not no_monitor:
        bot = app.bot
        asyncio.create_task(monitor_loop(_db_conn, bot, dry_run))
        logger.info("[Service] Monitor background task launched")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRISM Unified Service")
    parser.add_argument("--no-bot", action="store_true", help="Run monitor only")
    parser.add_argument("--no-monitor", action="store_true", help="Run bot only")
    parser.add_argument("--dry-run", action="store_true", help="No Telegram alerts from monitor")
    args = parser.parse_args()

    global _job_semaphore
    _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.no_bot:
        # Monitor only mode
        logger.info("[Service] Starting in monitor-only mode")
        conn = init_db()
        asyncio.run(monitor_loop(conn, dry_run=args.dry_run))
        return

    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("[Service] PRISM Unified Service starting")
    logger.info(f"  Bot: {'enabled' if not args.no_bot else 'disabled'}")
    logger.info(f"  Monitor: {'enabled' if not args.no_monitor else 'disabled'}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Authorized users: {ALLOWED_USER_IDS or 'all'}")
    logger.info(f"  Max concurrent jobs: {MAX_CONCURRENT_JOBS}")
    logger.info("=" * 60)

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app._prism_dry_run = args.dry_run
    app._prism_no_monitor = args.no_monitor

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("analyse", cmd_analyze))
    app.add_handler(CommandHandler("morning", cmd_morning))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("close", cmd_close))
    app.add_handler(MessageHandler(filters.COMMAND, lambda u, c: u.message.reply_text("Unknown command. /help")))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_natural_language))

    # Error handler — prevents unhandled Telegram errors from crashing the service
    async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f"[Bot] Telegram error: {context.error}", exc_info=context.error)
    app.add_error_handler(_error_handler)

    logger.info("[Service] Starting bot polling + monitor loop...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Crash log for pythonw.exe (no stderr)
        crash_log = Path(__file__).parent / "logs" / "prism_service_crash.log"
        crash_log.parent.mkdir(exist_ok=True)
        with open(crash_log, "a", encoding="utf-8") as f:
            import traceback
            f.write(f"\n{'='*60}\n")
            f.write(f"CRASH at {datetime.now()}\n")
            f.write(traceback.format_exc())
        raise
