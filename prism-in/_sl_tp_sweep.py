"""
SL/TP/Hold parameter sweep across 2024 and Jan-Jun 2025.
Tests different combinations and shows PnL for each.
"""
import json, glob, os, numpy as np, datetime
import yfinance as yf
import time

PROJECT_ROOT = "C:/Users/kumagaura/source/repos/prism-insight"

# Load all picks from trigger JSONs
def load_picks(min_date, max_date):
    files = sorted(glob.glob(f"{PROJECT_ROOT}/trigger_results_in_morning_*.json"))
    picks = []
    for f in files:
        date = os.path.basename(f).replace("trigger_results_in_morning_", "").replace(".json", "")
        if date < min_date or date > max_date: continue
        with open(f) as fh:
            d = json.load(fh)
        for k, v in d.items():
            if k in ("metadata", "screening_summary"): continue
            if isinstance(v, list):
                for s in v:
                    if isinstance(s, dict) and "ticker" in s:
                        picks.append({
                            "date": date, "ticker": s["ticker"],
                            "quality": s.get("quality_score", 50),
                        })
    return picks

# Fetch OHLC data
def fetch_ohlc(tickers, start, end):
    ohlc = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i+50]
        yf_tickers = [f"{t}.NS" for t in chunk]
        data = yf.download(yf_tickers, start=start, end=end, progress=False)
        if len(chunk) == 1:
            ohlc[chunk[0]] = {
                "Close": data["Close"].dropna(),
                "High": data["High"].dropna(),
                "Low": data["Low"].dropna(),
            }
        else:
            for t in chunk:
                col = f"{t}.NS"
                if col in data["Close"].columns:
                    ohlc[t] = {
                        "Close": data["Close"][col].dropna(),
                        "High": data["High"][col].dropna(),
                        "Low": data["Low"][col].dropna(),
                    }
    return ohlc

def simulate(picks, ohlc, sl_pct, tp_pct, max_hold):
    """Simulate with given SL/TP/hold params. Quality-weighted allocation."""
    capital = 10_00_000
    results = []
    
    # Group by date for allocation
    by_date = {}
    for p in picks:
        by_date.setdefault(p["date"], []).append(p)
    
    for date, day_picks in sorted(by_date.items()):
        n = len(day_picks)
        # Quality top-heavy allocation
        scores = [max(p["quality"], 10) for p in day_picks]
        sorted_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
        weights = [(1.0 - 0.40) / (n - 1) if n > 1 else 1.0] * n
        if n > 1:
            for j in range(n):
                weights[j] = (1.0 - 0.40) / (n - 1)
            weights[sorted_idx[0]] = 0.40
        else:
            weights = [1.0]
        
        for i, p in enumerate(day_picks):
            t = p["ticker"]
            if t not in ohlc: continue
            
            try:
                td = datetime.datetime.strptime(date, "%Y%m%d")
                close_s = ohlc[t]["Close"]
                high_s = ohlc[t]["High"]
                low_s = ohlc[t]["Low"]
                
                mask = close_s.index >= td.strftime("%Y-%m-%d")
                if mask.sum() == 0: continue
                entry_idx = close_s.index[mask][0]
                pos = close_s.index.get_loc(entry_idx)
                entry = float(close_s.iloc[pos])
                
                sl_price = entry * (1 - sl_pct)
                tp_price = entry * (1 + tp_pct)
                
                exit_ret = None
                exit_type = None
                exit_day = 0
                
                for d in range(1, min(max_hold + 1, len(close_s) - pos)):
                    low = float(low_s.iloc[pos + d])
                    high = float(high_s.iloc[pos + d])
                    
                    if low <= sl_price:
                        exit_ret = -sl_pct * 100
                        exit_type = "SL"
                        exit_day = d
                        break
                    if high >= tp_price:
                        exit_ret = tp_pct * 100
                        exit_type = "TP"
                        exit_day = d
                        break
                
                if exit_ret is None:
                    if pos + max_hold < len(close_s):
                        exit_ret = (float(close_s.iloc[pos + max_hold]) / entry - 1) * 100
                        exit_type = "TIME"
                        exit_day = max_hold
                    else:
                        continue
                
                alloc = capital * weights[i]
                pnl = alloc * exit_ret / 100
                results.append({"pnl": pnl, "ret": exit_ret, "type": exit_type, "day": exit_day})
            except:
                continue
    
    if not results:
        return None
    
    total_pnl = sum(r["pnl"] for r in results)
    wins = sum(1 for r in results if r["pnl"] > 0)
    wr = wins / len(results) * 100
    tp_count = sum(1 for r in results if r["type"] == "TP")
    sl_count = sum(1 for r in results if r["type"] == "SL")
    avg_hold = np.mean([r["day"] for r in results])
    
    return {
        "trades": len(results), "pnl": total_pnl, "wr": wr,
        "tp": tp_count, "sl": sl_count,
        "tp_pct": tp_count/len(results)*100,
        "sl_pct_rate": sl_count/len(results)*100,
        "avg_hold": avg_hold,
    }

# Run for both periods
for period_name, min_d, max_d, price_start, price_end in [
    ("2024", "20240101", "20241231", "2024-01-01", "2025-03-01"),
    ("Jan-Jun 2025", "20250101", "20250630", "2025-01-01", "2025-09-01"),
]:
    print(f"\n{'='*120}")
    print(f"  {period_name} — SL/TP/HOLD PARAMETER SWEEP (Quality-Weighted)")
    print(f"{'='*120}")
    
    picks = load_picks(min_d, max_d)
    tickers = sorted(set(p["ticker"] for p in picks))
    print(f"  Picks: {len(picks)}, Tickers: {len(tickers)}")
    print(f"  Fetching OHLC...")
    ohlc = fetch_ohlc(tickers, price_start, price_end)
    print(f"  OHLC for {len(ohlc)} tickers")
    
    print(f"\n  {'SL':>4} {'TP':>4} {'Hold':>5} {'Trades':>7} {'PnL':>10} {'Return':>8} {'WR':>6} "
          f"{'TP%':>5} {'SL%':>5} {'AvgHold':>7}")
    print(f"  {'-'*75}")
    
    results_all = []
    for sl in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        for tp in [0.06, 0.08, 0.10, 0.12, 0.15]:
            for hold in [7, 10, 14, 21, 30]:
                r = simulate(picks, ohlc, sl, tp, hold)
                if r is None: continue
                ret = r["pnl"] / 10_00_000 * 100
                results_all.append({"sl": sl, "tp": tp, "hold": hold, **r, "ret_pct": ret})
    
    # Sort by PnL
    results_all.sort(key=lambda x: x["pnl"], reverse=True)
    
    for r in results_all[:20]:
        status = "🟢" if r["pnl"] > 0 else "🔴"
        print(f"  {r['sl']*100:>3.0f}% {r['tp']*100:>3.0f}% {r['hold']:>4}d {r['trades']:>7} "
              f"{status}₹{r['pnl']/100000:>+7.1f}L {r['ret_pct']:>+6.1f}% {r['wr']:>5.1f}% "
              f"{r['tp_pct']:>4.0f}% {r['sl_pct_rate']:>4.0f}% {r['avg_hold']:>6.1f}d")
    
    print(f"\n  WORST 5:")
    for r in results_all[-5:]:
        print(f"  {r['sl']*100:>3.0f}% {r['tp']*100:>3.0f}% {r['hold']:>4}d {r['trades']:>7} "
              f"🔴₹{r['pnl']/100000:>+7.1f}L {r['ret_pct']:>+6.1f}% {r['wr']:>5.1f}%")
