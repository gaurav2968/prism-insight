"""
Analyze most profitable vs most losing trades and find optimal Quality vs Momentum weight.
Uses V1 backtest trades (Jan 2024 - Jul 2025).
"""
import json
import numpy as np
from collections import defaultdict

def load_trades(path):
    with open(path) as f:
        return json.load(f)['trades']

def analyze():
    trades = load_trades('prism-in/bt_v1_samep.json')
    
    # Filter trades with valid sim_return
    valid = [t for t in trades if t['sim_return_pct'] is not None]
    print(f"Total trades: {len(valid)}")
    print()
    
    # === TOP 20 MOST PROFITABLE ===
    by_profit = sorted(valid, key=lambda t: t['sim_return_pct'], reverse=True)
    
    print("=" * 90)
    print("TOP 20 MOST PROFITABLE TRADES")
    print("=" * 90)
    print(f"{'Date':<12} {'Ticker':<15} {'Type':<10} {'Quality':>8} {'Momentum':>10} {'Return%':>10}")
    print("-" * 90)
    for t in by_profit[:20]:
        tt = 'GapUp' if 'Gap' in t['trigger_type'] else 'ValCap'
        print(f"{t['trade_date']:<12} {t['ticker']:<15} {tt:<10} {t['quality_score']:>8.1f} {t['change_on_trigger']:>10.2f}% {t['sim_return_pct']:>9.2f}%")
    
    print()
    print("=" * 90)
    print("TOP 20 MOST LOSING TRADES")
    print("=" * 90)
    print(f"{'Date':<12} {'Ticker':<15} {'Type':<10} {'Quality':>8} {'Momentum':>10} {'Return%':>10}")
    print("-" * 90)
    for t in by_profit[-20:]:
        tt = 'GapUp' if 'Gap' in t['trigger_type'] else 'ValCap'
        print(f"{t['trade_date']:<12} {t['ticker']:<15} {tt:<10} {t['quality_score']:>8.1f} {t['change_on_trigger']:>10.2f}% {t['sim_return_pct']:>9.2f}%")
    
    # === SCORE COMPONENT STATS: WINNERS vs LOSERS ===
    print()
    print("=" * 90)
    print("SCORE COMPONENTS: WINNERS vs LOSERS")
    print("=" * 90)
    
    winners = [t for t in valid if t['sim_return_pct'] > 0]
    losers = [t for t in valid if t['sim_return_pct'] <= 0]
    
    w_qual = np.mean([t['quality_score'] for t in winners])
    l_qual = np.mean([t['quality_score'] for t in losers])
    w_mom = np.mean([t['change_on_trigger'] for t in winners])
    l_mom = np.mean([t['change_on_trigger'] for t in losers])
    
    print(f"Winners ({len(winners)}): avg quality={w_qual:.1f}, avg momentum={w_mom:.2f}%")
    print(f"Losers  ({len(losers)}):  avg quality={l_qual:.1f}, avg momentum={l_mom:.2f}%")
    print(f"Quality gap:  {w_qual - l_qual:+.1f} (higher = better predictor)")
    print(f"Momentum gap: {w_mom - l_mom:+.2f}% (higher = better predictor)")
    
    # === BY QUALITY QUARTILE ===
    print()
    print("=" * 90)
    print("PERFORMANCE BY QUALITY QUARTILE")
    print("=" * 90)
    
    quals = sorted(valid, key=lambda t: t['quality_score'])
    n = len(quals)
    for i, label in enumerate(['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']):
        chunk = quals[i * n // 4 : (i + 1) * n // 4]
        rets = [t['sim_return_pct'] for t in chunk]
        wr = len([r for r in rets if r > 0]) / len(rets) * 100
        avg = np.mean(rets)
        med = np.median(rets)
        q_range = f"{chunk[0]['quality_score']:.0f}-{chunk[-1]['quality_score']:.0f}"
        print(f"  {label:<15} quality={q_range:<10} trades={len(chunk):<5} WR={wr:.1f}%  avg={avg:+.2f}%  med={med:+.2f}%")
    
    # === BY MOMENTUM QUARTILE ===
    print()
    print("=" * 90)
    print("PERFORMANCE BY MOMENTUM QUARTILE")
    print("=" * 90)
    
    moms = sorted(valid, key=lambda t: t['change_on_trigger'])
    for i, label in enumerate(['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']):
        chunk = moms[i * n // 4 : (i + 1) * n // 4]
        rets = [t['sim_return_pct'] for t in chunk]
        wr = len([r for r in rets if r > 0]) / len(rets) * 100
        avg = np.mean(rets)
        med = np.median(rets)
        m_range = f"{chunk[0]['change_on_trigger']:.1f}-{chunk[-1]['change_on_trigger']:.1f}"
        print(f"  {label:<15} momentum={m_range:<12} trades={len(chunk):<5} WR={wr:.1f}%  avg={avg:+.2f}%  med={med:+.2f}%")
    
    # === WEIGHT OPTIMIZATION ===
    # Normalize quality and momentum to 0-1 for fair comparison
    print()
    print("=" * 90)
    print("WEIGHT OPTIMIZATION: Quality vs Momentum")
    print("=" * 90)
    
    all_qual = np.array([t['quality_score'] for t in valid])
    all_mom = np.array([t['change_on_trigger'] for t in valid])
    all_ret = np.array([t['sim_return_pct'] for t in valid])
    
    # Normalize to 0-1
    q_norm = (all_qual - all_qual.min()) / (all_qual.max() - all_qual.min())
    m_norm = (all_mom - all_mom.min()) / (all_mom.max() - all_mom.min())
    
    print(f"\n{'Q_weight':>9} {'M_weight':>9} | {'Top25% WR':>10} {'Top25% Avg':>11} {'Top25% PF':>10} | {'Bot25% WR':>10} {'Bot25% Avg':>11} | {'Spread':>8}")
    print("-" * 105)
    
    best_pf = 0
    best_w = 0
    
    for q_w in range(0, 105, 5):
        m_w = 100 - q_w
        q_frac = q_w / 100
        m_frac = m_w / 100
        
        # Combined score
        combined = q_frac * q_norm + m_frac * m_norm
        
        # Sort by combined score
        idx = np.argsort(combined)[::-1]  # highest first
        top_n = len(idx) // 4
        bot_n = len(idx) // 4
        
        top_rets = all_ret[idx[:top_n]]
        bot_rets = all_ret[idx[-bot_n:]]
        
        top_wr = np.mean(top_rets > 0) * 100
        top_avg = np.mean(top_rets)
        bot_wr = np.mean(bot_rets > 0) * 100
        bot_avg = np.mean(bot_rets)
        
        # Profit factor for top quartile
        top_wins = top_rets[top_rets > 0].sum()
        top_losses = abs(top_rets[top_rets <= 0].sum())
        top_pf = top_wins / top_losses if top_losses > 0 else 99
        
        spread = top_avg - bot_avg
        
        marker = ""
        if top_pf > best_pf:
            best_pf = top_pf
            best_w = q_w
            marker = " <-- best PF"
        
        print(f"  Q={q_w:>3}%  M={m_w:>3}% | {top_wr:>9.1f}% {top_avg:>+10.2f}% {top_pf:>10.2f} | {bot_wr:>9.1f}% {bot_avg:>+10.2f}% | {spread:>+7.2f}%{marker}")
    
    print(f"\nBest top-quartile PF: Q={best_w}% M={100-best_w}% → PF={best_pf:.2f}")
    
    # === ALSO TEST: WHAT IF WE ONLY TAKE HIGH QUALITY? ===
    print()
    print("=" * 90)
    print("QUALITY THRESHOLD: What if we require minimum quality?")
    print("=" * 90)
    
    for min_q in [40, 50, 55, 60, 65, 70, 75, 80]:
        filtered = [t for t in valid if t['quality_score'] >= min_q]
        if not filtered:
            continue
        rets = [t['sim_return_pct'] for t in filtered]
        wr = len([r for r in rets if r > 0]) / len(rets) * 100
        avg = np.mean(rets)
        wins = sum(r for r in rets if r > 0)
        losses = abs(sum(r for r in rets if r <= 0))
        pf = wins / losses if losses > 0 else 99
        print(f"  quality >= {min_q}: trades={len(filtered):<5} WR={wr:.1f}%  avg={avg:+.2f}%  PF={pf:.2f}")
    
    # === TRIGGER TYPE SPLIT ===
    print()
    print("=" * 90)
    print("SAME ANALYSIS BY TRIGGER TYPE")
    print("=" * 90)
    
    for tt_name, tt_filter in [('Gap Up Momentum', 'Gap'), ('Value-to-Cap', 'Value')]:
        subset = [t for t in valid if tt_filter in t['trigger_type']]
        if not subset:
            continue
        
        print(f"\n--- {tt_name} ({len(subset)} trades) ---")
        
        sub_qual = np.array([t['quality_score'] for t in subset])
        sub_mom = np.array([t['change_on_trigger'] for t in subset])
        sub_ret = np.array([t['sim_return_pct'] for t in subset])
        
        if sub_qual.max() == sub_qual.min() or sub_mom.max() == sub_mom.min():
            print("  (insufficient variance for optimization)")
            continue
            
        sq_norm = (sub_qual - sub_qual.min()) / (sub_qual.max() - sub_qual.min())
        sm_norm = (sub_mom - sub_mom.min()) / (sub_mom.max() - sub_mom.min())
        
        best_pf2 = 0
        best_w2 = 0
        
        for q_w in [0, 25, 50, 75, 100]:
            m_w = 100 - q_w
            combined = (q_w/100) * sq_norm + (m_w/100) * sm_norm
            idx = np.argsort(combined)[::-1]
            top_n = max(len(idx) // 4, 1)
            
            top_rets = sub_ret[idx[:top_n]]
            top_wr = np.mean(top_rets > 0) * 100
            top_avg = np.mean(top_rets)
            top_wins = top_rets[top_rets > 0].sum()
            top_losses = abs(top_rets[top_rets <= 0].sum())
            top_pf = top_wins / top_losses if top_losses > 0 else 99
            
            marker = ""
            if top_pf > best_pf2:
                best_pf2 = top_pf
                best_w2 = q_w
                marker = " <--"
            
            print(f"    Q={q_w:>3}% M={m_w:>3}%: top25% WR={top_wr:.1f}% avg={top_avg:+.2f}% PF={top_pf:.2f}{marker}")
        
        # Quality threshold for this type
        print(f"  Quality thresholds:")
        for min_q in [50, 60, 70, 80]:
            filtered = [t for t in subset if t['quality_score'] >= min_q]
            if not filtered:
                continue
            rets = [t['sim_return_pct'] for t in filtered]
            wr = len([r for r in rets if r > 0]) / len(rets) * 100
            avg = np.mean(rets)
            wins = sum(r for r in rets if r > 0)
            losses = abs(sum(r for r in rets if r <= 0))
            pf = wins / losses if losses > 0 else 99
            print(f"    quality >= {min_q}: trades={len(filtered):<5} WR={wr:.1f}%  avg={avg:+.2f}%  PF={pf:.2f}")

if __name__ == '__main__':
    analyze()
