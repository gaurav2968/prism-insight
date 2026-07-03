"""
Professional Exit Strategy Backtest
ATR-based stops + trailing stop + time decay

Compares:
1. Fixed SL/TP (current: 5% SL, 15% TP, 14d)
2. ATR-based SL/TP (2×ATR stop, 3×ATR target)
3. ATR + Trailing stop (trail by 1.5×ATR after 1×ATR profit)
4. ATR + Trailing + Time decay (tighten after day 10)
"""
import json, glob, os, numpy as np, datetime
import yfinance as yf

PROJECT_ROOT = "C:/Users/kumagaura/source/repos/prism-insight"
CAPITAL = 10_00_000  # ₹10L

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
                        picks.append({"date": date, "ticker": s["ticker"],
                                      "quality": s.get("quality_score", 50)})
    return picks

def fetch_ohlc(tickers, start, end):
    ohlc = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i+50]
        yf_tickers = [f"{t}.NS" for t in chunk]
        data = yf.download(yf_tickers, start=start, end=end, progress=False)
        if len(chunk) == 1:
            ohlc[chunk[0]] = {"Close": data["Close"].dropna(), "High": data["High"].dropna(),
                              "Low": data["Low"].dropna()}
        else:
            for t in chunk:
                col = f"{t}.NS"
                if col in data["Close"].columns:
                    ohlc[t] = {"Close": data["Close"][col].dropna(), "High": data["High"][col].dropna(),
                               "Low": data["Low"][col].dropna()}
    return ohlc

def compute_atr(ohlc, ticker, pos, period=14):
    """Compute ATR at entry position."""
    close = ohlc[ticker]["Close"]
    high = ohlc[ticker]["High"]
    low = ohlc[ticker]["Low"]
    if pos < period + 1:
        return None
    trs = []
    for i in range(pos - period, pos):
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        pc = float(close.iloc[i - 1]) if i > 0 else h
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return np.mean(trs)

def quality_weights(picks_for_day):
    n = len(picks_for_day)
    if n <= 1: return [1.0]
    scores = [max(p["quality"], 10) for p in picks_for_day]
    sorted_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
    weights = [(1.0 - 0.40) / (n - 1)] * n
    weights[sorted_idx[0]] = 0.40
    return weights

def simulate_strategy(picks, ohlc, strategy="fixed", max_hold=14):
    """
    Strategies:
    - fixed: 5% SL, 15% TP
    - atr: 2×ATR SL, 3×ATR TP
    - atr_trail: ATR entry + trail by 1.5×ATR after 1×ATR profit
    - atr_trail_decay: above + tighten trail after day 10
    """
    by_date = {}
    for p in picks:
        by_date.setdefault(p["date"], []).append(p)
    
    results = []
    for date, day_picks in sorted(by_date.items()):
        weights = quality_weights(day_picks)
        
        for i, p in enumerate(day_picks):
            t = p["ticker"]
            if t not in ohlc: continue
            close_s = ohlc[t]["Close"]
            high_s = ohlc[t]["High"]
            low_s = ohlc[t]["Low"]
            
            try:
                td = datetime.datetime.strptime(date, "%Y%m%d")
                mask = close_s.index >= td.strftime("%Y-%m-%d")
                if mask.sum() == 0: continue
                entry_idx = close_s.index[mask][0]
                pos = close_s.index.get_loc(entry_idx)
                entry = float(close_s.iloc[pos])
                
                atr = compute_atr(ohlc, t, pos)
                if atr is None or atr <= 0:
                    atr = entry * 0.02  # fallback 2%
                
                atr_pct = atr / entry * 100
                
                # Set initial SL/TP based on strategy
                if strategy == "fixed":
                    sl_price = entry * 0.95  # 5%
                    tp_price = entry * 1.15  # 15%
                    trail_active = False
                elif strategy == "atr":
                    sl_price = entry - 2 * atr
                    tp_price = entry + 3 * atr
                    trail_active = False
                elif strategy in ("atr_trail", "atr_trail_decay"):
                    sl_price = entry - 2 * atr
                    tp_price = entry + 4 * atr  # wider TP since trailing captures
                    trail_active = True
                    trail_atr_mult = 1.5
                    trail_trigger = entry + 1 * atr  # start trailing after 1×ATR profit
                    trailing_on = False
                    highest_since_entry = entry
                
                exit_ret = None
                exit_type = None
                exit_day = 0
                
                for d in range(1, min(max_hold + 1, len(close_s) - pos)):
                    low = float(low_s.iloc[pos + d])
                    high = float(high_s.iloc[pos + d])
                    close = float(close_s.iloc[pos + d])
                    
                    # Time decay: tighten trail after day 10
                    if strategy == "atr_trail_decay" and d > 10 and trail_active:
                        trail_atr_mult = 1.0  # tighter trail
                        if not trailing_on and close > entry:
                            # Force trailing on after day 10 if in profit
                            trailing_on = True
                            highest_since_entry = max(highest_since_entry, high)
                            sl_price = max(sl_price, highest_since_entry - trail_atr_mult * atr)
                    
                    # Update trailing stop
                    if trail_active and trailing_on:
                        highest_since_entry = max(highest_since_entry, high)
                        new_trail_sl = highest_since_entry - trail_atr_mult * atr
                        sl_price = max(sl_price, new_trail_sl)  # only move up
                    
                    # Check SL
                    if low <= sl_price:
                        exit_ret = (sl_price / entry - 1) * 100
                        exit_type = "SL"
                        exit_day = d
                        break
                    
                    # Check TP
                    if high >= tp_price:
                        exit_ret = (tp_price / entry - 1) * 100
                        exit_type = "TP"
                        exit_day = d
                        break
                    
                    # Activate trailing after trigger
                    if trail_active and not trailing_on and high >= trail_trigger:
                        trailing_on = True
                        highest_since_entry = high
                        sl_price = max(sl_price, high - trail_atr_mult * atr)
                
                if exit_ret is None:
                    if pos + max_hold < len(close_s):
                        exit_ret = (float(close_s.iloc[pos + max_hold]) / entry - 1) * 100
                        exit_type = "TIME"
                        exit_day = max_hold
                    else:
                        continue
                
                alloc = CAPITAL * weights[i]
                pnl = alloc * exit_ret / 100
                
                results.append({
                    "pnl": pnl, "ret": exit_ret, "type": exit_type, "day": exit_day,
                    "atr_pct": atr_pct, "ticker": t, "date": date,
                })
            except:
                continue
    
    return results

def summarize(results, label):
    if not results:
        print(f"  {label:<30s}: No trades")
        return
    total_pnl = sum(r["pnl"] for r in results)
    wins = sum(1 for r in results if r["pnl"] > 0)
    wr = wins / len(results) * 100
    
    gross_win = sum(r["pnl"] for r in results if r["pnl"] > 0)
    gross_loss = abs(sum(r["pnl"] for r in results if r["pnl"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    
    avg_win = np.mean([r["ret"] for r in results if r["ret"] > 0]) if wins > 0 else 0
    avg_loss = np.mean([r["ret"] for r in results if r["ret"] <= 0]) if wins < len(results) else 0
    avg_hold = np.mean([r["day"] for r in results])
    
    tp = sum(1 for r in results if r["type"] == "TP")
    sl = sum(1 for r in results if r["type"] == "SL")
    trail = sum(1 for r in results if r["type"] == "TRAIL")
    time_exit = sum(1 for r in results if r["type"] == "TIME")
    
    # Drawdown
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0
    daily = {}
    for r in results:
        daily.setdefault(r["date"], 0)
        daily[r["date"]] += r["pnl"]
    for d in sorted(daily):
        equity += daily[d]
        peak = max(peak, equity)
        dd = (equity - peak) / peak * 100
        max_dd = min(max_dd, dd)
    
    ret_pct = total_pnl / CAPITAL * 100
    
    print(f"  {label:<30s}: PnL=₹{total_pnl/100000:>+7.1f}L ({ret_pct:>+6.1f}%)  "
          f"WR={wr:>5.1f}%  PF={pf:>5.2f}  AvgW={avg_win:>+5.1f}%  AvgL={avg_loss:>+5.1f}%  "
          f"Hold={avg_hold:>4.1f}d  DD={max_dd:>+5.1f}%  "
          f"TP={tp} SL={sl} Time={time_exit}")

# Run
for period_name, min_d, max_d, ps, pe in [
    ("2024 Full Year", "20240101", "20241231", "2024-01-01", "2025-03-01"),
    ("Jan-Jun 2025", "20250101", "20250630", "2025-01-01", "2025-09-01"),
]:
    print(f"\n{'='*130}")
    print(f"  {period_name}")
    print(f"{'='*130}")
    
    picks = load_picks(min_d, max_d)
    tickers = sorted(set(p["ticker"] for p in picks))
    print(f"  Picks: {len(picks)}, Tickers: {len(tickers)}")
    ohlc = fetch_ohlc(tickers, ps, pe)
    print(f"  OHLC: {len(ohlc)} tickers\n")
    
    for hold in [14, 21]:
        print(f"  --- Max Hold: {hold} days ---")
        for strat in ["fixed", "atr", "atr_trail", "atr_trail_decay"]:
            res = simulate_strategy(picks, ohlc, strat, hold)
            labels = {
                "fixed": f"Fixed 5%SL/15%TP",
                "atr": f"ATR 2x SL / 3x TP",
                "atr_trail": f"ATR + Trail 1.5x",
                "atr_trail_decay": f"ATR + Trail + TimeDecay",
            }
            summarize(res, labels[strat])
        print()

# Also show ATR distribution
print(f"\n{'='*80}")
print(f"ATR DISTRIBUTION (what stocks' volatility looks like)")
print(f"{'='*80}")
picks = load_picks("20240101", "20250630")
tickers = sorted(set(p["ticker"] for p in picks))
ohlc = fetch_ohlc(tickers, "2024-01-01", "2025-09-01")
atrs = []
for p in picks[:200]:
    t = p["ticker"]
    if t not in ohlc: continue
    close_s = ohlc[t]["Close"]
    try:
        td = datetime.datetime.strptime(p["date"], "%Y%m%d")
        mask = close_s.index >= td.strftime("%Y-%m-%d")
        if mask.sum() == 0: continue
        pos = close_s.index.get_loc(close_s.index[mask][0])
        atr = compute_atr(ohlc, t, pos)
        if atr:
            entry = float(close_s.iloc[pos])
            atrs.append(atr / entry * 100)
    except:
        continue

if atrs:
    print(f"  Sample: {len(atrs)} trades")
    print(f"  ATR% distribution:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    P{p}: {np.percentile(atrs, p):.2f}%")
    print(f"  Mean: {np.mean(atrs):.2f}%")
    print(f"  → 2×ATR SL = ~{np.median(atrs)*2:.1f}% (vs fixed 5%)")
    print(f"  → 3×ATR TP = ~{np.median(atrs)*3:.1f}% (vs fixed 15%)")
