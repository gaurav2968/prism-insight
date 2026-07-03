"""Monthly breakdown of backtest results for 2024 and 2025."""
import json, glob, os, numpy as np, datetime
import yfinance as yf

PROJECT_ROOT = "C:/Users/kumagaura/source/repos/prism-insight"

# Load all trigger files
files = sorted(glob.glob(f"{PROJECT_ROOT}/trigger_results_in_morning_*.json"))
print(f"Total files: {len(files)}")

# Extract picks
picks = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    date = os.path.basename(f).replace("trigger_results_in_morning_", "").replace(".json", "")
    regime = d.get("metadata", {}).get("regime", {})
    rtype = regime.get("type", "?") if isinstance(regime, dict) else str(regime)
    
    for k, v in d.items():
        if k in ("metadata", "screening_summary"): continue
        if isinstance(v, list):
            for s in v:
                if isinstance(s, dict) and "ticker" in s:
                    picks.append({
                        "date": date, "ticker": s["ticker"],
                        "price": s.get("current_price", 0),
                        "trigger": k, "regime": rtype,
                    })

print(f"Total picks: {len(picks)}")

# Get unique tickers and fetch prices
tickers = sorted(set(p["ticker"] for p in picks))
print(f"Unique tickers: {len(tickers)}")
print("Fetching prices...")

price_data = {}
for i in range(0, len(tickers), 50):
    chunk = tickers[i:i+50]
    yf_tickers = [f"{t}.NS" for t in chunk]
    data = yf.download(yf_tickers, start="2024-01-01", end="2026-06-01", progress=False)
    if "Close" in data.columns or len(chunk) == 1:
        close = data["Close"] if len(chunk) > 1 else data[["Close"]]
        if len(chunk) == 1:
            close.columns = [yf_tickers[0]]
        for t in chunk:
            col = f"{t}.NS"
            if col in close.columns:
                series = close[col].dropna()
                price_data[t] = series

print(f"Price data for {len(price_data)} tickers")

# Compute forward returns
results = []
for p in picks:
    t = p["ticker"]
    if t not in price_data:
        continue
    series = price_data[t]
    try:
        td = datetime.datetime.strptime(p["date"], "%Y%m%d")
        # Find entry price (on or after trade date)
        mask = series.index >= td.strftime("%Y-%m-%d")
        if mask.sum() == 0:
            continue
        entry_idx = series.index[mask][0]
        entry_pos = series.index.get_loc(entry_idx)
        entry_price = float(series.iloc[entry_pos])
        
        # Forward returns
        ret_7d = (float(series.iloc[entry_pos + 5]) / entry_price - 1) * 100 if entry_pos + 5 < len(series) else None
        ret_14d = (float(series.iloc[entry_pos + 10]) / entry_price - 1) * 100 if entry_pos + 10 < len(series) else None
        
        # Simulated SL/TP (10% SL, 8% TP, 30d max)
        sim_ret = None
        sim_exit = None
        for j in range(1, min(22, len(series) - entry_pos)):
            fwd_price = float(series.iloc[entry_pos + j])
            ret = (fwd_price / entry_price - 1) * 100
            if ret <= -10:
                sim_ret = -10.0; sim_exit = "SL"; break
            elif ret >= 8:
                sim_ret = 8.0; sim_exit = "TP"; break
        if sim_ret is None and entry_pos + 21 < len(series):
            sim_ret = (float(series.iloc[entry_pos + 21]) / entry_price - 1) * 100
            sim_exit = "TIME"
        
        results.append({
            **p, "entry": entry_price, "ret_7d": ret_7d, "ret_14d": ret_14d,
            "sim_ret": sim_ret, "sim_exit": sim_exit,
            "month": p["date"][:6],
        })
    except Exception:
        continue

print(f"Results: {len(results)}")

# Monthly breakdown
months = sorted(set(r["month"] for r in results))

print(f"\n{'='*110}")
print(f"MONTHLY BREAKDOWN")
print(f"{'='*110}")
print(f"{'Month':<8} {'Trades':>6} {'WR_7d':>6} {'Avg7d':>7} {'Avg14d':>7} {'SimWR':>6} {'SimAvg':>7} {'PnL(2L)':>8} {'TP%':>5} {'SL%':>5} {'Regime':<20}")
print("-" * 110)

cumulative_pnl = 0
for m in months:
    subset = [r for r in results if r["month"] == m]
    
    rets_7 = [r["ret_7d"] for r in subset if r["ret_7d"] is not None]
    rets_14 = [r["ret_14d"] for r in subset if r["ret_14d"] is not None]
    sim_rets = [r["sim_ret"] for r in subset if r["sim_ret"] is not None]
    
    wr7 = np.mean([r > 0 for r in rets_7]) * 100 if rets_7 else 0
    avg7 = np.mean(rets_7) if rets_7 else 0
    avg14 = np.mean(rets_14) if rets_14 else 0
    
    sim_wr = np.mean([r > 0 for r in sim_rets]) * 100 if sim_rets else 0
    sim_avg = np.mean(sim_rets) if sim_rets else 0
    pnl = sum(2 * r / 100 for r in sim_rets)  # ₹2L per trade
    cumulative_pnl += pnl
    
    tp_count = sum(1 for r in subset if r["sim_exit"] == "TP")
    sl_count = sum(1 for r in subset if r["sim_exit"] == "SL")
    tp_pct = tp_count / len(subset) * 100 if subset else 0
    sl_pct = sl_count / len(subset) * 100 if subset else 0
    
    # Dominant regime
    regimes = [r["regime"] for r in subset]
    regime_counts = {}
    for rg in regimes:
        regime_counts[rg] = regime_counts.get(rg, 0) + 1
    top_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "?"
    
    status = "🟢" if pnl > 0 else "🔴"
    
    print(f"{m:<8} {len(subset):>6} {wr7:>5.0f}% {avg7:>+6.2f}% {avg14:>+6.2f}% {sim_wr:>5.0f}% {sim_avg:>+6.2f}% "
          f"{status}{pnl:>+7.2f}L {tp_pct:>4.0f}% {sl_pct:>4.0f}% {top_regime:<20}")

print(f"\n{'CUMULATIVE':>62} {cumulative_pnl:>+8.2f}L")

# Quarterly summary
print(f"\n{'='*80}")
print(f"QUARTERLY SUMMARY")
print(f"{'='*80}")
print(f"{'Quarter':<12} {'Trades':>6} {'WR':>6} {'Avg':>7} {'PnL':>8}")
print("-" * 45)

quarters = {
    "2024 Q1": ["202401", "202402", "202403"],
    "2024 Q2": ["202404", "202405", "202406"],
    "2024 Q3": ["202407", "202408", "202409"],
    "2024 Q4": ["202410", "202411", "202412"],
    "2025 Q1": ["202501", "202502", "202503"],
    "2025 Q2": ["202504", "202505", "202506"],
    "2025 Q3": ["202507", "202508", "202509"],
}

for qname, qmonths in quarters.items():
    subset = [r for r in results if r["month"] in qmonths]
    if not subset: continue
    sim_rets = [r["sim_ret"] for r in subset if r["sim_ret"] is not None]
    if not sim_rets: continue
    wr = np.mean([r > 0 for r in sim_rets]) * 100
    avg = np.mean(sim_rets)
    pnl = sum(2 * r / 100 for r in sim_rets)
    status = "🟢" if pnl > 0 else "🔴"
    print(f"{qname:<12} {len(subset):>6} {wr:>5.0f}% {avg:>+6.2f}% {status}{pnl:>+7.2f}L")
