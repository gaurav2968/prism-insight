#!/usr/bin/env python3
"""
India (NSE) Stock Market Business Day Checker

Uses the holidays library for accurate NSE holiday detection.
Returns exit code 0 if today is a market day, 1 otherwise.

NSE Market Holidays include:
- Republic Day (January 26)
- Holi
- Good Friday
- Dr. Ambedkar Jayanti (April 14)
- May Day (May 1)
- Independence Day (August 15)
- Mahatma Gandhi Jayanti (October 2)
- Diwali (Lakshmi Puja)
- Dussehra
- Christmas (December 25)
- Multiple regional / religious holidays

Trading Hours: 09:15-15:30 IST (India Standard Time)
Pre-open:     09:00-09:08 IST
"""

import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pytz

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Logging setup
logging.basicConfig(
    filename=PROJECT_ROOT / 'in_stock_scheduler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timezone
IST = pytz.timezone('Asia/Kolkata')


def _get_india_holidays(year: int):
    """Get Indian market holidays for a given year."""
    try:
        from holidays.countries import IN
        return IN(years=year)
    except ImportError:
        logger.warning("holidays package not installed. Weekend-only check.")
        return {}


def is_nse_market_day(check_date: date = None) -> bool:
    """
    Check if the given date is an NSE trading day.

    Args:
        check_date: Date to check (defaults to today in IST)

    Returns:
        bool: True if it's a market day, False otherwise
    """
    if check_date is None:
        check_date = datetime.now(IST).date()

    # Weekend check (5: Saturday, 6: Sunday)
    if check_date.weekday() >= 5:
        logger.debug(f"{check_date} is a weekend.")
        return False

    # Holiday check
    holidays = _get_india_holidays(check_date.year)
    if check_date in holidays:
        holiday_name = holidays.get(check_date, "Unknown holiday")
        logger.debug(f"{check_date} is an NSE holiday: {holiday_name}")
        return False

    return True


def get_holiday_name(check_date: date = None) -> str:
    """
    Get the name of the holiday for a given date.

    Args:
        check_date: Date to check (defaults to today in IST)

    Returns:
        str: Holiday name or empty string if not a holiday
    """
    if check_date is None:
        check_date = datetime.now(IST).date()

    holidays = _get_india_holidays(check_date.year)
    return holidays.get(check_date, "")


def get_last_trading_day(check_date: date = None) -> date:
    """
    Get the most recent NSE trading day on or before the given date.

    Args:
        check_date: Reference date (defaults to today in IST)

    Returns:
        date: Most recent trading day
    """
    if check_date is None:
        check_date = datetime.now(IST).date()

    for i in range(30):  # Look back up to 30 days
        candidate = check_date - timedelta(days=i)
        if is_nse_market_day(candidate):
            return candidate

    return check_date  # fallback


def get_next_trading_day(check_date: date = None) -> date:
    """
    Get the next NSE trading day after the given date.

    Args:
        check_date: Reference date (defaults to today in IST)

    Returns:
        date: Next trading day
    """
    if check_date is None:
        check_date = datetime.now(IST).date()

    for i in range(1, 30):  # Look forward up to 30 days
        candidate = check_date + timedelta(days=i)
        if is_nse_market_day(candidate):
            return candidate

    return check_date + timedelta(days=1)  # fallback


def is_market_open_now() -> bool:
    """
    Check if the NSE market is currently open.

    Market hours: 09:15 - 15:30 IST

    Returns:
        bool: True if market is open right now
    """
    now = datetime.now(IST)

    if not is_nse_market_day(now.date()):
        return False

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


def main():
    """Main entry point — check if today is a market day."""
    today = datetime.now(IST).date()
    is_market = is_nse_market_day(today)

    if is_market:
        logger.info(f"{today} is an NSE trading day.")
        print(f"✅ {today} is an NSE trading day.")
        sys.exit(0)
    else:
        holiday_name = get_holiday_name(today)
        reason = f" ({holiday_name})" if holiday_name else ""
        if today.weekday() >= 5:
            reason = " (Weekend)"
        logger.info(f"{today} is NOT a trading day{reason}.")
        print(f"❌ {today} is NOT a trading day{reason}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
