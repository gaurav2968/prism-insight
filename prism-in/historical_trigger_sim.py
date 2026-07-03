#!/usr/bin/env python3
"""
PRISM India — Historical Trigger Simulation

Runs the trigger batch pipeline retroactively on historical dates.
Generates trigger_results JSON files as if the system was running daily.

Usage:
    python prism-in/historical_trigger_sim.py --start 20251001 --end 20260301
    python prism-in/historical_trigger_sim.py --start 20251001 --end 20260301 --skip-quality
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
PRISM_IN_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PRISM_IN_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_trading_days(start: str, end: str) -> list:
    """Generate NSE trading days between start and end (YYYYMMDD)."""
    from check_market_day import is_nse_market_day
    
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    
    days = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        try:
            if is_nse_market_day(current.date()):
                days.append(date_str)
        except Exception:
            # Fallback: skip weekends at minimum
            if current.weekday() < 5:
                days.append(date_str)
        current += timedelta(days=1)
    
    return days


def run_single_date(trade_date: str, output_dir: str, skip_quality: bool = False):
    """Run trigger batch for a single historical date."""
    output_file = os.path.join(output_dir, f"trigger_results_in_morning_{trade_date}.json")
    
    # Skip if already exists
    if os.path.exists(output_file):
        logger.info(f"  [{trade_date}] Already exists, skipping")
        return True
    
    try:
        from in_trigger_batch import run_batch
        
        logger.info(f"  [{trade_date}] Running trigger batch...")
        t0 = time.time()
        
        results = run_batch(
            trigger_time="morning",
            log_level="WARNING",  # Quiet mode
            output_file=output_file,
            reference_date=trade_date,
        )
        
        elapsed = time.time() - t0
        
        if os.path.exists(output_file):
            import json
            with open(output_file) as f:
                data = json.load(f)
            n_picks = sum(
                len(v) for k, v in data.items() 
                if isinstance(v, list) and k not in ("metadata", "screening_summary")
            )
            logger.info(f"  [{trade_date}] Done: {n_picks} picks in {elapsed:.1f}s")
            return True
        else:
            logger.warning(f"  [{trade_date}] No output generated in {elapsed:.1f}s")
            return False
            
    except Exception as e:
        logger.error(f"  [{trade_date}] Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Historical trigger simulation")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT), help="Output directory for JSON files")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality evaluation (faster)")
    parser.add_argument("--max-days", type=int, default=None, help="Max number of days to process")
    args = parser.parse_args()

    logger.info(f"Historical trigger simulation: {args.start} to {args.end}")
    
    # Get trading days
    trading_days = get_trading_days(args.start, args.end)
    logger.info(f"Found {len(trading_days)} trading days")
    
    if args.max_days:
        trading_days = trading_days[:args.max_days]
        logger.info(f"Limited to {args.max_days} days")
    
    # Check which dates already have results
    existing = 0
    for d in trading_days:
        f = os.path.join(args.output_dir, f"trigger_results_in_morning_{d}.json")
        if os.path.exists(f):
            existing += 1
    logger.info(f"Already have results for {existing}/{len(trading_days)} days")
    
    # Run simulation
    success = 0
    failed = 0
    skipped = 0
    total = len(trading_days)
    
    t_start = time.time()
    
    for i, trade_date in enumerate(trading_days):
        logger.info(f"[{i+1}/{total}] Processing {trade_date}...")
        
        output_file = os.path.join(args.output_dir, f"trigger_results_in_morning_{trade_date}.json")
        if os.path.exists(output_file):
            skipped += 1
            continue
        
        ok = run_single_date(trade_date, args.output_dir, args.skip_quality)
        if ok:
            success += 1
        else:
            failed += 1
        
        # Rate limit: yfinance needs breathing room between full pipeline runs
        # Each date makes ~600+ HTTP requests (3 snapshots × 288 tickers + quality evals)
        # 15s cooldown — aggressive but workable for batch runs
        if i < total - 1:
            time.sleep(15)
    
    elapsed = time.time() - t_start
    logger.info(f"\nSimulation complete in {elapsed:.0f}s")
    logger.info(f"  Success: {success}, Failed: {failed}, Skipped: {skipped}")
    logger.info(f"\nNow run backtest:")
    logger.info(f"  python backtest_engine.py --min-date {args.start} --max-date {args.end} --output backtest_results_historical.json")


if __name__ == "__main__":
    main()
