import yfinance as yf
import pandas as pd
import numpy as np

nifty = yf.download("^NSEI", start="2024-01-01", end="2025-07-31", progress=False)
# Flatten multi-index columns if present
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)
monthly = nifty["Close"].resample("ME").last()

start_price = float(nifty["Close"].iloc[0])
capital = 1000000

print("NIFTY 50 BUY-AND-HOLD (10L invested Jan 2024)")
print("=" * 75)
print(f"Entry: {nifty.index[0].strftime('%Y-%m-%d')} at {start_price:,.0f}")
print()

# King 2 monthly P&L from apples-to-apples run
k2_v1 = {
    "202401": 94327, "202402": 77089, "202403": -49710, "202404": 242600,
    "202405": 2641, "202406": -59304, "202407": 78167, "202408": 21367,
    "202409": 65137, "202410": -133828, "202412": -6383,
    "202501": 4784, "202502": -32290, "202503": 29023,
    "202504": 94362, "202505": 82616, "202506": 92995, "202507": -26148,
}
k2_v2 = {
    "202401": 93211, "202402": 43203, "202403": -76849, "202404": 114046,
    "202405": 7153, "202406": 7408, "202407": 46557, "202408": 20778,
    "202409": 51165, "202410": -77017, "202412": 25364,
    "202501": 25952, "202502": -38210, "202503": 67143,
    "202505": 110143, "202506": 102212, "202507": -14950,
}

print(f"{'Month':<8} {'NIFTY':>10} {'NIFTY%':>8} {'NIFTY PnL':>12} {'K2-V1 PnL':>12} {'K2-V2 PnL':>12} {'K2V1 cum':>12} {'K2V2 cum':>12} {'NIFTY cum':>12}")
print("-" * 110)

prev = start_price
nifty_cum = 0
k2v1_cum = 0
k2v2_cum = 0

for date, close in monthly.items():
    c = float(close)
    mo_ret = (c / prev - 1) * 100
    cum_ret = (c / start_price - 1) * 100
    nifty_pnl = capital * mo_ret / 100
    nifty_cum = capital * cum_ret / 100
    mo_str = date.strftime("%Y%m")

    v1_pnl = k2_v1.get(mo_str, 0)
    v2_pnl = k2_v2.get(mo_str, 0)
    k2v1_cum += v1_pnl
    k2v2_cum += v2_pnl

    print(f"{mo_str:<8} {c:>10,.0f} {mo_ret:>+7.1f}% {nifty_pnl:>+12,.0f} {v1_pnl:>+12,.0f} {v2_pnl:>+12,.0f} {k2v1_cum:>+12,.0f} {k2v2_cum:>+12,.0f} {nifty_cum:>+12,.0f}")
    prev = c

# Summary
final = float(monthly.iloc[-1])
total_ret = (final / start_price - 1) * 100
nifty_final_pnl = capital * total_ret / 100

eq = nifty["Close"] / start_price * capital
peak = eq.cummax()
dd = (eq - peak) / peak * 100
mdd = float(dd.min())
n_days = (monthly.index[-1] - nifty.index[0]).days
cagr = ((final / start_price) ** (365 / n_days) - 1) * 100

print()
print("=" * 110)
print(f"{'':>50} {'NIFTY B&H':>14} {'King2 V1':>14} {'King2 V2':>14}")
print("-" * 110)
print(f"{'Total P&L':>50} {nifty_final_pnl:>+14,.0f} {k2v1_cum:>+14,.0f} {k2v2_cum:>+14,.0f}")
print(f"{'Total Return %':>50} {total_ret:>+13.1f}% {k2v1_cum/capital*100:>+13.1f}% {k2v2_cum/capital*100:>+13.1f}%")
print(f"{'Max Drawdown %':>50} {mdd:>13.1f}% {-15.0:>13.1f}% {-8.4:>13.1f}%")
print(f"{'CAGR %':>50} {cagr:>+13.1f}% {30.1:>+13.1f}% {26.7:>+13.1f}%")
print(f"{'Calmar':>50} {cagr/abs(mdd):>14.2f} {30.1/15.0:>14.2f} {26.7/8.4:>14.2f}")
print(f"{'Sharpe':>50} {'--':>14} {1.38:>14.2f} {1.48:>14.2f}")
print(f"{'Sortino':>50} {'--':>14} {1.72:>14.2f} {1.73:>14.2f}")

# Alpha
print()
print(f"  King2 V1 alpha over NIFTY: {k2v1_cum - nifty_final_pnl:>+12,.0f} ({k2v1_cum/capital*100 - total_ret:>+.1f}pp)")
print(f"  King2 V2 alpha over NIFTY: {k2v2_cum - nifty_final_pnl:>+12,.0f} ({k2v2_cum/capital*100 - total_ret:>+.1f}pp)")

# Monthly win rate vs NIFTY
v1_beat = 0
v2_beat = 0
total_mo = 0
prev = start_price
for date, close in monthly.items():
    c = float(close)
    nifty_mo = (c / prev - 1) * 100
    nifty_mo_pnl = capital * nifty_mo / 100
    mo_str = date.strftime("%Y%m")
    v1p = k2_v1.get(mo_str, 0)
    v2p = k2_v2.get(mo_str, 0)
    total_mo += 1
    if v1p > nifty_mo_pnl:
        v1_beat += 1
    if v2p > nifty_mo_pnl:
        v2_beat += 1
    prev = c

print(f"  King2 V1 beat NIFTY: {v1_beat}/{total_mo} months ({v1_beat/total_mo*100:.0f}%)")
print(f"  King2 V2 beat NIFTY: {v2_beat}/{total_mo} months ({v2_beat/total_mo*100:.0f}%)")
