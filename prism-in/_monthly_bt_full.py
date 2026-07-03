"""Quick monthly breakdown of backtest engine results."""
import json, sys

bt = json.load(open('prism-in/bt_v1_full.json'))
trades = bt['trades']
capital = 1000000

monthly = {}
for t in trades:
    m = t['trade_date'][:6]
    if m not in monthly:
        monthly[m] = {'n': 0, 'wins': 0, 'pnl_qual': 0, 'pnl_equal': 0, 'total_ret': 0}
    ret = t.get('sim_return_pct', 0) or 0
    alloc = t.get('alloc_capital', 0) or 0
    monthly[m]['n'] += 1
    if ret > 0:
        monthly[m]['wins'] += 1
    monthly[m]['total_ret'] += ret
    monthly[m]['pnl_qual'] += alloc * ret / 100
    monthly[m]['pnl_equal'] += (capital / 5) * ret / 100

print(f"{'Month':>8} {'Trades':>6} {'WR%':>6} {'AvgRet':>8} {'Qual PnL':>11} {'Eq PnL':>11}")
print("=" * 65)
cum_q = 0
cum_e = 0
for m in sorted(monthly):
    d = monthly[m]
    n = d['n']
    wr = d['wins'] / n * 100 if n > 0 else 0
    avg = d['total_ret'] / n if n > 0 else 0
    cum_q += d['pnl_qual']
    cum_e += d['pnl_equal']
    marker = " <<<" if avg < -1 else (" ***" if avg > 1 else "")
    pq = d['pnl_qual']
    pe = d['pnl_equal']
    print(f"  {m}  {n:>5}  {wr:>5.1f}  {avg:>+7.2f}%  {pq:>+11,.0f}  {pe:>+11,.0f}{marker}")

print("=" * 65)
total_n = sum(d['n'] for d in monthly.values())
print(f"  {'TOTAL':>6}  {total_n:>5}                 {cum_q:>+11,.0f}  {cum_e:>+11,.0f}")
print(f"  Final equity (Qual): Rs {capital + cum_q:,.0f}")
print(f"  Final equity (Eq):   Rs {capital + cum_e:,.0f}")
print()
print("  Qual = quality-weighted allocation (40% to top pick)")
print("  Eq   = equal allocation (Rs 2L per pick)")
