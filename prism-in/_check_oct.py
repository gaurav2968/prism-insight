"""Check Oct 2024 trades with different exit strategies."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from strategy_lab_v2 import (simulate_portfolio, Features, make_feature_set,
                              load_trigger_files, extract_picks, fetch_all_price_data,
                              extract_regime_by_date)
import logging
logging.basicConfig(level=logging.WARNING)

# Load V1 triggers — need wider window so open positions from Oct 1 see regime change on Oct 4+
trigger_files = load_trigger_files(min_date='20240920', max_date='20241031')
picks_df = extract_picks(trigger_files)
price_data = fetch_all_price_data(picks_df)
regime_by_date = extract_regime_by_date(trigger_files)

print(f"Picks: {len(picks_df)}")
print(f"Regime data: {len(regime_by_date)} days")
for d in sorted(regime_by_date.keys()):
    if d >= '20241001':
        print(f"  {d}: {regime_by_date[d]}")
print()

configs = [
    ("Baseline (static 3xATR SL)", Features(name="Baseline", max_positions=8), False),
    ("B: Ratchet stop", make_feature_set("B: Ratchet 2xATR", ratchet_stop=True, ratchet_atr_mult=2.0), False),
    ("M: Regime exit only", Features(name="M: Regime exit", max_positions=8, regime_exit=True), True),
    ("B+M: Ratchet + Regime exit", make_feature_set("B+M", ratchet_stop=True, ratchet_atr_mult=2.0, regime_exit=True), True),
]

# Set max_positions on all
for label, f, _ in configs:
    f.max_positions = 8

for label, features, use_regime in configs:
    r = simulate_portfolio(picks_df, price_data, features, regime_by_date=regime_by_date if use_regime else None)
    trades = r.get("_trades", [])
    # Filter to Oct entries only
    oct_trades = [t for t in trades if t["entry_date"] >= "20241001"]
    total_pnl = sum(t["pnl"] for t in oct_trades)
    
    print(f"{'='*95}")
    print(f"  {label}  |  Oct Trades: {len(oct_trades)}  |  Oct PnL: {total_pnl:+,.0f}")
    print(f"{'='*95}")
    
    for t in oct_trades:
        ticker = t["ticker"]
        entry = t["entry_date"]
        exit_d = t["exit_date"]
        reason = t["exit_reason"]
        days = t["days_held"]
        ret = t["net_return_pct"]
        pnl = t["pnl"]
        alloc = t["alloc_capital"]
        print(f"  {entry} -> {exit_d}  {ticker:<15} {reason:<15} {days:>3}d  ret={ret:>+7.2f}%  pnl=Rs{pnl:>+9,.0f}  (Rs{alloc:,.0f})")
    print()
