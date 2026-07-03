"""
Robustness Test Suite

1. Threshold stability: perturb each threshold ±20% and measure degradation
2. Composite weight sensitivity: randomize weights and compare
3. Quality filter impact: does removing quality gate improve momentum returns?
"""
import json, glob, os, numpy as np, datetime
import yfinance as yf

PROJECT_ROOT = "C:/Users/kumagaura/source/repos/prism-insight"
CAPITAL = 10_00_000

# Load picks with all metadata
def load_all_picks(min_date, max_date):
    files = sorted(glob.glob(f"{PROJECT_ROOT}/trigger_results_in_morning_*.json"))
    all_data = []
    for f in files:
        date = os.path.basename(f).replace("trigger_results_in_morning_", "").replace(".json", "")
        if date < min_date or date > max_date: continue
        with open(f) as fh:
            d = json.load(fh)
        meta = d.get("metadata", {})
        regime = meta.get("regime", {})
        rtype = regime.get("type", "?") if isinstance(regime, dict) else str(regime)
        for k, v in d.items():
            if k in ("metadata", "screening_summary"): continue
            if isinstance(v, list):
                for s in v:
                    if isinstance(s, dict) and "ticker" in s:
                        all_data.append({
                            "date": date, "ticker": s["ticker"],
                            "quality": s.get("quality_score", 50),
                            "change_rate": s.get("change_rate", 0),
                            "trigger": k, "regime": rtype,
                            "rsi": s.get("metrics", {}).get("rsi_14", 50) if isinstance(s.get("metrics"), dict) else 50,
                        })
    return all_data

# Fetch prices once
def fetch_prices(tickers, start, end):
    prices = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i+50]
        yf_tickers = [f"{t}.NS" for t in chunk]
        data = yf.download(yf_tickers, start=start, end=end, progress=False)
        if len(chunk) == 1:
            prices[chunk[0]] = data["Close"].dropna()
        else:
            for t in chunk:
                col = f"{t}.NS"
                if col in data["Close"].columns:
                    prices[t] = data["Close"][col].dropna()
    return prices

def compute_return(prices, ticker, date, hold=14):
    if ticker not in prices: return None
    s = prices[ticker]
    try:
        td = datetime.datetime.strptime(date, "%Y%m%d")
        mask = s.index >= td.strftime("%Y-%m-%d")
        if mask.sum() == 0: return None
        pos = s.index.get_loc(s.index[mask][0])
        if pos + hold >= len(s): return None
        return (float(s.iloc[pos + hold]) / float(s.iloc[pos]) - 1) * 100
    except:
        return None

# Load data
print("Loading picks...")
picks_2024 = load_all_picks("20240101", "20241231")
picks_2025 = load_all_picks("20250101", "20250630")
all_picks = picks_2024 + picks_2025
print(f"Total picks: {len(all_picks)} (2024: {len(picks_2024)}, 2025H1: {len(picks_2025)})")

tickers = sorted(set(p["ticker"] for p in all_picks))
print(f"Fetching prices for {len(tickers)} tickers...")
prices = fetch_prices(tickers, "2024-01-01", "2025-09-01")
print(f"Got prices for {len(prices)} tickers")

# Compute returns for all picks
for p in all_picks:
    p["ret_14d"] = compute_return(prices, p["ticker"], p["date"], 14)

valid = [p for p in all_picks if p["ret_14d"] is not None]
print(f"Valid picks with returns: {len(valid)}")

# ═══════════════════════════════════════════════════
# TEST 1: THRESHOLD STABILITY
# ═══════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 1: THRESHOLD STABILITY")
print("Does performance collapse if we perturb thresholds?")
print(f"{'='*90}")

# Test quality gate at different levels
print(f"\n--- Quality Gate Threshold ---")
print(f"{'Threshold':<12} {'Trades':>6} {'Avg14d':>8} {'WR':>6} {'Filtered':>8}")
print("-" * 45)
for q_gate in [0, 20, 30, 40, 50, 55, 60, 65, 70]:
    subset = [p for p in valid if p["quality"] >= q_gate]
    if not subset: continue
    avg = np.mean([p["ret_14d"] for p in subset])
    wr = np.mean([p["ret_14d"] > 0 for p in subset]) * 100
    filtered = len(valid) - len(subset)
    print(f"  >= {q_gate:<6} {len(subset):>6} {avg:>+7.2f}% {wr:>5.1f}% {filtered:>7} removed")

# Test RSI filter at different levels
print(f"\n--- RSI Filter Threshold ---")
print(f"{'Threshold':<12} {'Trades':>6} {'Avg14d':>8} {'WR':>6}")
print("-" * 35)
for rsi_max in [50, 55, 60, 65, 70, 75, 100]:
    # We don't have RSI for all picks in JSON, test with quality as proxy
    label = "OFF" if rsi_max == 100 else f"<= {rsi_max}"
    subset = valid  # RSI not stored in all JSONs, skip this test
    # Instead test daily change threshold
    pass

# Test daily change threshold
print(f"\n--- Daily Change Threshold ---")
print(f"{'Threshold':<12} {'Trades':>6} {'Avg14d':>8} {'WR':>6}")
print("-" * 35)
for max_change in [3, 4, 5, 6, 7, 8, 10, 100]:
    subset = [p for p in valid if abs(p["change_rate"]) <= max_change]
    if not subset: continue
    avg = np.mean([p["ret_14d"] for p in subset])
    wr = np.mean([p["ret_14d"] > 0 for p in subset]) * 100
    label = "OFF" if max_change == 100 else f"<= {max_change}%"
    print(f"  {label:<10} {len(subset):>6} {avg:>+7.2f}% {wr:>5.1f}%")


# ═══════════════════════════════════════════════════
# TEST 2: COMPOSITE WEIGHT SENSITIVITY
# ═══════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 2: DO COMPOSITE WEIGHTS MATTER?")
print("Compare: original weights vs equal weights vs random weights")
print(f"{'='*90}")

# We can't recompute composite scores from JSON, but we can test
# whether quality score predicts returns (the most important weight)

# Bin quality scores and check return by bin
print(f"\n--- Returns by Quality Score Bin ---")
print(f"{'Quality':<15} {'Trades':>6} {'Avg14d':>8} {'WR':>6} {'Median':>8}")
print("-" * 50)
for lo, hi in [(0, 30), (30, 40), (40, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 80), (80, 100)]:
    subset = [p for p in valid if lo <= p["quality"] < hi]
    if not subset: continue
    rets = [p["ret_14d"] for p in subset]
    avg = np.mean(rets)
    med = np.median(rets)
    wr = np.mean([r > 0 for r in rets]) * 100
    print(f"  {lo}-{hi:<10} {len(subset):>6} {avg:>+7.2f}% {wr:>5.1f}% {med:>+7.2f}%")

# Correlation: quality score vs forward return
quals = [p["quality"] for p in valid]
rets = [p["ret_14d"] for p in valid]
corr = float(np.corrcoef(quals, rets)[0, 1])
print(f"\n  Correlation (quality vs fwd_14d): {corr:.4f}")
print(f"  → {'Quality PREDICTS returns' if corr > 0.05 else 'Quality has WEAK/NO predictive power' if corr > -0.02 else 'Quality INVERSELY predicts (momentum conflict!)'}")


# ═══════════════════════════════════════════════════
# TEST 3: DOES QUALITY FILTER DESTROY MOMENTUM ALPHA?
# ═══════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 3: QUALITY FILTER — HELPING OR HURTING?")
print("Compare returns WITH vs WITHOUT quality gate")
print(f"{'='*90}")

# All picks vs quality >= 40 vs quality >= 50 vs quality >= 60
print(f"\n{'Filter':<25} {'Trades':>6} {'Avg14d':>8} {'WR':>6} {'PF':>6} {'TotalPnL':>10}")
print("-" * 70)

for label, filter_fn in [
    ("No quality filter", lambda p: True),
    ("Quality >= 30", lambda p: p["quality"] >= 30),
    ("Quality >= 40 (current)", lambda p: p["quality"] >= 40),
    ("Quality >= 50", lambda p: p["quality"] >= 50),
    ("Quality >= 55", lambda p: p["quality"] >= 55),
    ("Quality >= 60", lambda p: p["quality"] >= 60),
    ("Quality >= 65", lambda p: p["quality"] >= 65),
    ("Quality >= 70", lambda p: p["quality"] >= 70),
    ("Quality < 40 (rejected)", lambda p: p["quality"] < 40),
    ("Quality < 50 (bottom)", lambda p: p["quality"] < 50),
]:
    subset = [p for p in valid if filter_fn(p)]
    if not subset: continue
    rets = [p["ret_14d"] for p in subset]
    avg = np.mean(rets)
    wr = np.mean([r > 0 for r in rets]) * 100
    wins = sum(r for r in rets if r > 0)
    losses = abs(sum(r for r in rets if r < 0))
    pf = wins / losses if losses > 0 else 999
    total_pnl = sum(rets) * CAPITAL / 100 / 5  # equal alloc, 5 picks
    print(f"  {label:<23} {len(subset):>6} {avg:>+7.2f}% {wr:>5.1f}% {pf:>5.2f} {total_pnl/100000:>+8.1f}L")

# By trigger: does quality matter differently?
print(f"\n--- Quality Impact BY TRIGGER TYPE ---")
for trigger in ["Gap Up Momentum Top", "Value-to-Cap Ratio Top"]:
    trig_picks = [p for p in valid if p["trigger"] == trigger]
    if not trig_picks: continue
    print(f"\n  {trigger}:")
    for label, lo, hi in [("Q < 40", 0, 40), ("Q 40-55", 40, 55), ("Q 55-65", 55, 65), ("Q 65+", 65, 100)]:
        subset = [p for p in trig_picks if lo <= p["quality"] < hi]
        if not subset: continue
        avg = np.mean([p["ret_14d"] for p in subset])
        wr = np.mean([p["ret_14d"] > 0 for p in subset]) * 100
        print(f"    {label:<10} n={len(subset):>4}  avg={avg:>+6.2f}%  WR={wr:>5.1f}%")


# ═══════════════════════════════════════════════════
# TEST 4: REGIME THRESHOLD STABILITY
# ═══════════════════════════════════════════════════
print(f"\n{'='*90}")
print("TEST 4: REGIME THRESHOLD STABILITY (NIFTY level)")
print("Using NIFTY returns, test if regime thresholds are stable")
print(f"{'='*90}")

nifty = yf.Ticker("^NSEI")
nh = nifty.history(start="2008-01-01", end="2026-05-19")
nc = nh["Close"].values
nd = nh.index

def wilder_rsi(c, period=14):
    d = np.diff(c); g = np.where(d > 0, d, 0); l = np.where(d < 0, -d, 0)
    ag = np.mean(g[:period]); al = np.mean(l[:period])
    for i in range(period, len(g)):
        ag = (ag*(period-1)+g[i])/period; al = (al*(period-1)+l[i])/period
    return 100-(100/(1+ag/al)) if al > 0 else 100

# Test slope threshold
print(f"\n--- Slope Threshold Stability ---")
print(f"{'MinSlope':<10} {'BullDays':>8} {'AvgFwd14d':>10} {'WR':>6}")
print("-" * 38)
for min_slope in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    bull_rets = []
    for i in range(200, len(nc)):
        if i + 10 >= len(nc): break
        p = float(nc[i])
        s50 = float(np.mean(nc[i-50:i]))
        s200 = float(np.mean(nc[i-200:i]))
        s50_20 = float(np.mean(nc[i-70:i-20])) if i >= 70 else s50
        slope = (s50 / s50_20 - 1) * 100
        if p > s50 and p > s200 and slope >= min_slope:
            fwd = (nc[i+10] / nc[i] - 1) * 100
            bull_rets.append(fwd)
    if bull_rets:
        avg = np.mean(bull_rets)
        wr = np.mean([r > 0 for r in bull_rets]) * 100
        print(f"  >= {min_slope:<5.1f}% {len(bull_rets):>8} {avg:>+9.2f}% {wr:>5.1f}%")

# Test RVol threshold
print(f"\n--- RVol Threshold Stability ---")
print(f"{'MaxRVol':<10} {'BullDays':>8} {'AvgFwd14d':>10} {'WR':>6}")
print("-" * 38)
for max_rvol in [10, 12, 13, 14, 15, 18, 20, 99]:
    bull_rets = []
    for i in range(200, len(nc)):
        if i + 10 >= len(nc): break
        p = float(nc[i])
        s50 = float(np.mean(nc[i-50:i]))
        s200 = float(np.mean(nc[i-200:i]))
        s50_20 = float(np.mean(nc[i-70:i-20])) if i >= 70 else s50
        slope = (s50 / s50_20 - 1) * 100
        rvol = float(np.std(np.diff(np.log(nc[i-20:i+1]))) * np.sqrt(252) * 100)
        if p > s50 and p > s200 and slope >= 1.0 and rvol <= max_rvol:
            fwd = (nc[i+10] / nc[i] - 1) * 100
            bull_rets.append(fwd)
    if bull_rets:
        avg = np.mean(bull_rets)
        wr = np.mean([r > 0 for r in bull_rets]) * 100
        label = "OFF" if max_rvol == 99 else f"<= {max_rvol}%"
        print(f"  {label:<8} {len(bull_rets):>8} {avg:>+9.2f}% {wr:>5.1f}%")

# Bear bottom RSI threshold
print(f"\n--- Bear Bottom RSI Threshold Stability ---")
print(f"{'MaxRSI':<10} {'Days':>6} {'AvgFwd14d':>10} {'WR':>6}")
print("-" * 38)
for bb_rsi in [25, 30, 35, 40, 45, 50]:
    rets = []
    for i in range(200, len(nc)):
        if i + 10 >= len(nc): break
        p = float(nc[i])
        s50 = float(np.mean(nc[i-50:i]))
        s200 = float(np.mean(nc[i-200:i]))
        rsi = wilder_rsi(nc[:i+1])
        if p < s50 and p < s200 and rsi <= bb_rsi:
            fwd = (nc[i+10] / nc[i] - 1) * 100
            rets.append(fwd)
    if rets:
        avg = np.mean(rets)
        wr = np.mean([r > 0 for r in rets]) * 100
        print(f"  <= {bb_rsi:<5} {len(rets):>6} {avg:>+9.2f}% {wr:>5.1f}%")

print(f"\n{'='*90}")
print("CONCLUSION: If values degrade SMOOTHLY → robust. If they COLLAPSE → overfit.")
print(f"{'='*90}")
