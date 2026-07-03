"""Compare V1 vs V2 trigger results with realistic capital management."""
import json

v1 = json.load(open('prism-in/lab_v2_realistic_v1_samep.json'))
v2 = json.load(open('prism-in/lab_v2_realistic_v2data.json'))

print("=" * 100)
print("REALISTIC COMPARISON: V1 vs V2 triggers (Jan 2024 - Jul 2025)")
print("=" * 100)

# Get total picks
v1_picks = v1[0].get('n_trades', 0) + v1[0].get('trades_skipped', 0)
v2_picks = v2[0].get('n_trades', 0) + v2[0].get('trades_skipped', 0)
print(f"\nV1 total picks: {v1_picks}")
print(f"V2 total picks: {v2_picks}")
print()

hdr = f"{'Config':<38} {'Tr':>4} {'WR%':>5} {'Avg':>6} {'PF':>5} {'Hold':>5} {'Equity':>10} {'Ret%':>7} {'CAGR':>6} {'MDD':>6} {'Calmar':>7}"
print(hdr)
print("-" * 100)

# Map V2 by short key
v2_map = {}
for r in v2:
    v2_map[r['features']] = r

# For each V1 config, find matching V2
for r in v1:
    name = r['features']
    eq = r.get('final_equity', 0)
    eq_s = f"Rs{eq/100000:.1f}L"
    sk = r.get('trades_skipped', 0)
    print(f"  V1 {name:<33} {r.get('n_trades',0):>4} {r.get('win_rate',0):>5.1f} "
          f"{r.get('avg_return_net',0):>+5.2f}% {r.get('profit_factor',0):>5} "
          f"{r.get('avg_hold_days',0):>5.1f} {eq_s:>10} {r.get('total_return_pct',0):>+6.1f}% "
          f"{r.get('cagr_pct',0):>+5.1f}% {r.get('max_drawdown_pct',0):>5.1f}% "
          f"{r.get('calmar',0):>7.2f}  (skip={sk})")

    if name in v2_map:
        r2 = v2_map[name]
        eq2 = r2.get('final_equity', 0)
        eq_s2 = f"Rs{eq2/100000:.1f}L"
        sk2 = r2.get('trades_skipped', 0)
        print(f"  V2 {name:<33} {r2.get('n_trades',0):>4} {r2.get('win_rate',0):>5.1f} "
              f"{r2.get('avg_return_net',0):>+5.2f}% {r2.get('profit_factor',0):>5} "
              f"{r2.get('avg_hold_days',0):>5.1f} {eq_s2:>10} {r2.get('total_return_pct',0):>+6.1f}% "
              f"{r2.get('cagr_pct',0):>+5.1f}% {r2.get('max_drawdown_pct',0):>5.1f}% "
              f"{r2.get('calmar',0):>7.2f}  (skip={sk2})")
        # Delta
        d_eq = eq2 - eq
        d_ret = r2.get('total_return_pct',0) - r.get('total_return_pct',0)
        d_dd = r2.get('max_drawdown_pct',0) - r.get('max_drawdown_pct',0)
        better = "V2 BETTER" if d_eq > 0 else "V1 BETTER"
        print(f"  >> {better}: equity diff Rs{d_eq/100000:+.1f}L, return {d_ret:+.1f}%, MDD {d_dd:+.1f}%")
    print()

# Also show all V2 configs
print("\n" + "=" * 100)
print("ALL V2 REALISTIC RESULTS")
print("=" * 100)
print(hdr)
print("-" * 100)
for r in v2:
    name = r['features']
    eq = r.get('final_equity', 0)
    eq_s = f"Rs{eq/100000:.1f}L"
    sk = r.get('trades_skipped', 0)
    print(f"  {name:<38} {r.get('n_trades',0):>4} {r.get('win_rate',0):>5.1f} "
          f"{r.get('avg_return_net',0):>+5.2f}% {r.get('profit_factor',0):>5} "
          f"{r.get('avg_hold_days',0):>5.1f} {eq_s:>10} {r.get('total_return_pct',0):>+6.1f}% "
          f"{r.get('cagr_pct',0):>+5.1f}% {r.get('max_drawdown_pct',0):>5.1f}% "
          f"{r.get('calmar',0):>7.2f}  (skip={sk})")
