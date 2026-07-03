"""Compare backtest_engine vs strategy_lab_v2 P&L calculation differences."""
import json

bt = json.load(open('prism-in/bt_v1_full.json'))
trades = bt['trades']
capital = 1_000_000

# ── 1. CAPITAL DEPLOYMENT PER DAY ──
from collections import defaultdict
daily_alloc = defaultdict(float)
daily_trades = defaultdict(int)
daily_alloc_max = defaultdict(float)

for t in trades:
    d = t['trade_date']
    alloc = t.get('alloc_capital', 0) or 0
    daily_alloc[d] += alloc
    daily_trades[d] += 1
    daily_alloc_max[d] = max(daily_alloc_max[d], alloc)

print("═" * 70)
print("1. CAPITAL DEPLOYMENT PER DAY (backtest_engine)")
print("═" * 70)
allocs = list(daily_alloc.values())
print(f"  Trading days: {len(allocs)}")
print(f"  Avg daily capital deployed: Rs {sum(allocs)/len(allocs):,.0f}")
print(f"  Min daily capital deployed: Rs {min(allocs):,.0f}")
print(f"  Max daily capital deployed: Rs {max(allocs):,.0f}")
print(f"  TOTAL capital deployed over period: Rs {sum(allocs):,.0f}")
print(f"  → That's {sum(allocs)/capital:.0f}x the initial capital!")
print()

# ── 2. QUALITY-WEIGHTED vs EQUAL P&L ──
print("═" * 70)
print("2. P&L COMPARISON: Quality-weighted vs Equal")
print("═" * 70)

pnl_qual = 0
pnl_equal = 0
pnl_qual_with_costs = 0
COST = 0.00193  # 0.193% round-trip

for t in trades:
    ret = t.get('sim_return_pct', 0) or 0
    alloc = t.get('alloc_capital', 0) or 0
    n_picks = daily_trades[t['trade_date']]
    
    pnl_qual += alloc * ret / 100
    pnl_qual_with_costs += alloc * (ret / 100 - COST)
    pnl_equal += (capital / max(n_picks, 5)) * ret / 100

print(f"  Quality-weighted (no costs):    Rs {capital + pnl_qual:>12,.0f}  ({pnl_qual:>+12,.0f})")
print(f"  Quality-weighted (with costs):  Rs {capital + pnl_qual_with_costs:>12,.0f}  ({pnl_qual_with_costs:>+12,.0f})")
print(f"  Equal alloc (no costs):         Rs {capital + pnl_equal:>12,.0f}  ({pnl_equal:>+12,.0f})")
print()

# ── 3. HOW MUCH DOES TOP PICK CONTRIBUTE? ──
print("═" * 70)
print("3. TOP PICK vs REST (quality-weighted)")
print("═" * 70)

daily_picks = defaultdict(list)
for t in trades:
    daily_picks[t['trade_date']].append(t)

pnl_top = 0
pnl_rest = 0
top_wins = 0
rest_wins = 0
n_top = 0
n_rest = 0

for d, picks in daily_picks.items():
    # Sort by alloc_capital descending (top pick has highest alloc)
    picks_sorted = sorted(picks, key=lambda x: x.get('alloc_capital', 0), reverse=True)
    for i, p in enumerate(picks_sorted):
        ret = p.get('sim_return_pct', 0) or 0
        alloc = p.get('alloc_capital', 0) or 0
        pnl = alloc * ret / 100
        if i == 0:
            pnl_top += pnl
            n_top += 1
            if ret > 0: top_wins += 1
        else:
            pnl_rest += pnl
            n_rest += 1
            if ret > 0: rest_wins += 1

print(f"  Top pick (40% alloc):")
print(f"    Trades: {n_top}, WR: {top_wins/n_top*100:.1f}%")
print(f"    Total P&L: Rs {pnl_top:>+12,.0f}")
print(f"    Avg P&L/trade: Rs {pnl_top/n_top:>+10,.0f}")
print()
print(f"  Rest (60% split):")
print(f"    Trades: {n_rest}, WR: {rest_wins/n_rest*100:.1f}%")
print(f"    Total P&L: Rs {pnl_rest:>+12,.0f}")
print(f"    Avg P&L/trade: Rs {pnl_rest/n_rest:>+8,.0f}")
print()
print(f"  TOP PICK CONTRIBUTION: {pnl_top/(pnl_top+pnl_rest)*100:.1f}% of total profit")

# ── 4. P&L ATTRIBUTION TIMING ──
print()
print("═" * 70)
print("4. P&L ATTRIBUTION: Entry date vs realistic")
print("═" * 70)
print("  backtest_engine: ALL P&L booked on ENTRY date")
print("  strategy_lab_v2: P&L booked on EXIT date")
print()

hold_days = [t.get('sim_exit_day', 0) or 0 for t in trades]
import statistics
print(f"  Avg hold days: {statistics.mean(hold_days):.1f}")
print(f"  Median hold days: {statistics.median(hold_days):.1f}")
print(f"  Max hold days: {max(hold_days):.0f}")
print()

# Count overlapping positions
# For each trade, it occupies capital from entry to exit
# How much capital is actually needed?
import datetime
all_days = sorted(set(t['trade_date'] for t in trades))

# Find max concurrent capital needed
concurrent_capital = defaultdict(float)
for t in trades:
    entry = t['trade_date']
    hold = int(t.get('sim_exit_day', 21) or 21)
    alloc = t.get('alloc_capital', 0) or 0
    # Mark capital as deployed for each day the position is open
    entry_idx = all_days.index(entry) if entry in all_days else 0
    for i in range(entry_idx, min(entry_idx + hold, len(all_days))):
        concurrent_capital[all_days[i]] += alloc

concurrent_vals = list(concurrent_capital.values())
print(f"  Max concurrent capital needed: Rs {max(concurrent_vals):>12,.0f}")
print(f"  Avg concurrent capital needed: Rs {sum(concurrent_vals)/len(concurrent_vals):>12,.0f}")
print(f"  → Need {max(concurrent_vals)/capital:.1f}x initial capital for all positions simultaneously!")
print()
print("  ⚠ backtest_engine assumes you have UNLIMITED capital")
print("  ⚠ strategy_lab_v2 also deploys capital/n_slots per day (no constraint on overlap)")
print("  Both engines IGNORE the fact that open positions tie up capital!")

# ── 5. WHAT IF WE LIMIT TO 5 CONCURRENT POSITIONS? ──
print()
print("═" * 70)
print("5. REALISTIC SCENARIO: What if we could only hold 5 positions max?")
print("═" * 70)

# Simulate: open max 5 concurrent positions, skip new ones if full
# Pick by quality score (highest first)
open_pos = []  # list of (exit_day_idx, alloc, pnl)
pnl_constrained = 0
trades_taken = 0
trades_skipped = 0
MAX_CONCURRENT = 5

for day_idx, day in enumerate(all_days):
    # Close expired positions
    open_pos = [(ed, a, p) for ed, a, p in open_pos if ed > day_idx]
    
    # Get today's picks, sorted by quality
    today_picks = sorted(
        [t for t in trades if t['trade_date'] == day],
        key=lambda x: x.get('quality_score', 0) or 0,
        reverse=True
    )
    
    slots_free = MAX_CONCURRENT - len(open_pos)
    for p in today_picks:
        if slots_free <= 0:
            trades_skipped += 1
            continue
        
        ret = p.get('sim_return_pct', 0) or 0
        hold = int(p.get('sim_exit_day', 21) or 21)
        alloc = capital / MAX_CONCURRENT  # Rs 2L per slot
        pnl = alloc * ret / 100
        pnl_constrained += pnl
        open_pos.append((day_idx + hold, alloc, pnl))
        trades_taken += 1
        slots_free -= 1

print(f"  Trades taken: {trades_taken} (skipped: {trades_skipped})")
print(f"  Final equity: Rs {capital + pnl_constrained:>12,.0f}")
print(f"  Total P&L:    Rs {pnl_constrained:>+12,.0f}")
