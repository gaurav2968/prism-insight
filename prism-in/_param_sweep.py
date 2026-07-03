#!/usr/bin/env python3
"""
Parameter sweep: find optimal SL, TP, and hold period from backtest data.
Tests all combinations and ranks by Profit Factor and total P&L.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, glob
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def load_picks():
    pattern = str(PROJECT_ROOT / "trigger_results_in_morning_2025*.json")
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        date = data.get("metadata", {}).get("trade_date", "")
        for ttype, stocks in data.items():
            if not isinstance(stocks, list): continue
            for s in stocks:
                if s.get("ticker"):
                    rows.append({
                        "trade_date": date,
                        "ticker": s["ticker"],
                        "entry_price": s.get("current_price", 0),
                        "trigger_type": ttype,
                    })
    return pd.DataFrame(rows)

def fetch_prices(picks):
    tickers = picks["ticker"].unique()
    earliest = pd.Timestamp(datetime.strptime(picks["trade_date"].min(), "%Y%m%d"))
    latest = pd.Timestamp(datetime.strptime(picks["trade_date"].max(), "%Y%m%d")) + timedelta(days=45)
    
    yf_tickers = [f"{t}.NS" for t in tickers]
    all_data = {}
    
    for i in range(0, len(yf_tickers), 50):
        chunk = yf_tickers[i:i+50]
        raw = yf.download(chunk, start=earliest.strftime("%Y-%m-%d"), 
                         end=latest.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        if not raw.empty:
            for yt in chunk:
                sym = yt.replace(".NS", "")
                try:
                    if len(chunk) == 1:
                        df = raw
                    else:
                        df = raw.xs(yt, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw
                    if not df.empty and "Close" in df.columns:
                        all_data[sym] = df
                except: continue
    return all_data

def simulate(picks, prices, sl_pct, tp_pct, max_days):
    results = []
    for _, row in picks.iterrows():
        ticker = row["ticker"]
        entry = row["entry_price"]
        entry_date = pd.Timestamp(datetime.strptime(row["trade_date"], "%Y%m%d"))
        
        if ticker not in prices or entry <= 0: continue
        hist = prices[ticker]
        future = hist[hist.index > entry_date]
        if future.empty: continue
        
        closes = future["Close"].values
        highs = future["High"].values if "High" in future.columns else closes
        lows = future["Low"].values if "Low" in future.columns else closes
        
        sim_sl = entry * (1 - sl_pct)
        sim_tp = entry * (1 + tp_pct)
        n = min(max_days, len(closes))
        
        exit_ret = None
        exit_reason = None
        exit_day = None
        
        for d in range(n):
            if lows[d] <= sim_sl:
                exit_ret = -sl_pct * 100
                exit_reason = "SL"
                exit_day = d + 1
                break
            if highs[d] >= sim_tp:
                exit_ret = tp_pct * 100
                exit_reason = "TP"
                exit_day = d + 1
                break
        
        if exit_ret is None and n > 0:
            exit_ret = (closes[n-1] / entry - 1) * 100
            exit_reason = "TIME"
            exit_day = n
        
        if exit_ret is not None:
            results.append({"ret": exit_ret, "reason": exit_reason, "day": exit_day})
    
    return results

def main():
    print("Loading picks...")
    picks = load_picks()
    print(f"Loaded {len(picks)} picks across {picks['trade_date'].nunique()} dates")
    
    print("Fetching prices...")
    prices = fetch_prices(picks)
    print(f"Got prices for {len(prices)} tickers\n")
    
    # Parameter grid
    sl_options = [0.03, 0.05, 0.07, 0.10]
    tp_options = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    hold_options = [5, 7, 10, 14, 21, 30]
    
    print(f"{'SL':>5} {'TP':>5} {'Days':>5} | {'WR':>6} {'AvgRet':>7} {'PF':>6} {'Trades':>7} "
          f"{'SL%':>5} {'TP%':>5} {'Time%':>6} | {'P&L(₹2L/trade)':>15}")
    print("─" * 95)
    
    results_table = []
    
    for max_days in hold_options:
        for sl in sl_options:
            for tp in tp_options:
                res = simulate(picks, prices, sl, tp, max_days)
                if not res:
                    continue
                
                rets = [r["ret"] for r in res]
                wins = [r for r in rets if r > 0]
                losses = [r for r in rets if r <= 0]
                
                wr = len(wins) / len(rets) * 100
                avg_ret = np.mean(rets)
                gross_win = sum(wins) if wins else 0
                gross_loss = abs(sum(losses)) if losses else 0.01
                pf = gross_win / gross_loss if gross_loss > 0 else 99
                
                sl_count = sum(1 for r in res if r["reason"] == "SL")
                tp_count = sum(1 for r in res if r["reason"] == "TP")
                time_count = sum(1 for r in res if r["reason"] == "TIME")
                n = len(res)
                
                per_trade = 200000  # ₹2L per trade
                total_pnl = sum(r / 100 * per_trade for r in rets)
                
                results_table.append({
                    "sl": sl, "tp": tp, "days": max_days,
                    "wr": wr, "avg_ret": avg_ret, "pf": pf,
                    "n": n, "pnl": total_pnl,
                    "sl_pct": sl_count/n*100, "tp_pct": tp_count/n*100,
                    "time_pct": time_count/n*100,
                })
                
                print(f"{sl*100:>4.0f}% {tp*100:>4.0f}% {max_days:>5d} | "
                      f"{wr:>5.1f}% {avg_ret:>+6.2f}% {pf:>5.2f} {n:>7d} "
                      f"{sl_count/n*100:>4.0f}% {tp_count/n*100:>4.0f}% {time_count/n*100:>5.0f}% | "
                      f"₹{total_pnl:>+12,.0f}")
    
    # Top 10 by Profit Factor
    results_table.sort(key=lambda x: x["pf"], reverse=True)
    print(f"\n{'=' * 95}")
    print(f"TOP 10 BY PROFIT FACTOR")
    print(f"{'=' * 95}")
    for i, r in enumerate(results_table[:10]):
        print(f"  #{i+1}: SL={r['sl']*100:.0f}% TP={r['tp']*100:.0f}% Hold={r['days']}d → "
              f"PF={r['pf']:.2f} WR={r['wr']:.1f}% Avg={r['avg_ret']:+.2f}% P&L=₹{r['pnl']:+,.0f}")
    
    # Top 10 by Total P&L
    results_table.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n{'=' * 95}")
    print(f"TOP 10 BY TOTAL P&L")
    print(f"{'=' * 95}")
    for i, r in enumerate(results_table[:10]):
        print(f"  #{i+1}: SL={r['sl']*100:.0f}% TP={r['tp']*100:.0f}% Hold={r['days']}d → "
              f"PF={r['pf']:.2f} WR={r['wr']:.1f}% Avg={r['avg_ret']:+.2f}% P&L=₹{r['pnl']:+,.0f}")
    
    # Best balanced (PF > 1.1 AND highest P&L)
    balanced = [r for r in results_table if r["pf"] > 1.1]
    if balanced:
        balanced.sort(key=lambda x: x["pnl"], reverse=True)
        print(f"\n{'=' * 95}")
        print(f"BEST BALANCED (PF > 1.1, ranked by P&L)")
        print(f"{'=' * 95}")
        for i, r in enumerate(balanced[:5]):
            print(f"  #{i+1}: SL={r['sl']*100:.0f}% TP={r['tp']*100:.0f}% Hold={r['days']}d → "
                  f"PF={r['pf']:.2f} WR={r['wr']:.1f}% Avg={r['avg_ret']:+.2f}% P&L=₹{r['pnl']:+,.0f}")


if __name__ == "__main__":
    main()
