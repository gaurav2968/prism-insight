"""
King 2 — Apples-to-apples comparison on common date range only.
Both V1 and V2 have complete data Jan 2024 – Jun 2025.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict
from strategy_lab_v2 import (
    simulate_portfolio, Features, _score_to_mult,
    load_trigger_files, extract_picks, fetch_all_price_data,
    extract_regime_by_date, build_market_indicators,
)
import yfinance as yf
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

RPL = {
    "bull_strong": 8, "bull_medium": 6, "bull_weak": 0,
    "slope_declining": 0, "correction": 0,
    "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
}

def king2():
    return Features(
        name="KING 2: Ratch SL(2-3) pos 8/6",
        max_positions=8, max_hold_days=20, regime_position_limits=RPL,
        market_adaptive=True, market_score_key="ema5",
        market_hold_adjust=True, market_ratchet_only=True,
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=2.0, market_tp_bull=2.0,
    )

def baseline():
    return Features(name="Baseline", max_positions=8)

MIN_DATE = "20240101"
MAX_DATE = "20250630"

def filter_picks(picks_df, min_date, max_date):
    """Keep only picks within date range."""
    if hasattr(picks_df["trade_date"].iloc[0], "strftime"):
        dates = picks_df["trade_date"].dt.strftime("%Y%m%d")
    else:
        dates = picks_df["trade_date"].astype(str).str.replace("-", "")
    mask = (dates >= min_date) & (dates <= max_date)
    out = picks_df[mask].copy()
    print(f"  Filtered: {len(picks_df)} -> {len(out)} picks ({min_date}-{max_date})")
    return out

def filter_regime(regime_by_date, min_date, max_date):
    return {d: r for d, r in regime_by_date.items() if min_date <= d <= max_date}

def print_stats(label, res):
    trades = res.get("_trades", [])
    total_pnl = sum(t["pnl"] for t in trades) if trades else 0
    monthly = defaultdict(float)
    for t in trades:
        monthly[t["exit_date"][:6]] += t["pnl"]

    print(f"\n{'='*75}")
    print(f"  {label}")
    print(f"{'='*75}")
    print(f"  Total P&L:          ₹{total_pnl:>+12,.0f}")
    print(f"  CAGR:               {res.get('cagr_pct', 0):>+8.1f}%")
    print(f"  Max Drawdown:       {res.get('max_drawdown_pct', 0):>8.1f}%")
    print(f"  Calmar:             {res.get('calmar', 0):>8.2f}")
    print(f"  Sharpe:             {res.get('sharpe', 0):>8.2f}")
    print(f"  Sortino:            {res.get('sortino', 0):>8.2f}")
    print(f"  Trades:             {res.get('n_trades', 0):>8}  (skipped: {res.get('trades_skipped', 0)})")
    print(f"  Win Rate:           {res.get('win_rate', 0):>7.1f}%")
    print(f"  Profit Factor:      {res.get('profit_factor', 0):>8}")
    print(f"  Expectancy:         {res.get('expectancy_pct', 0):>+7.2f}%")
    print(f"  Avg Win:            {res.get('avg_win_pct', 0):>+7.2f}%")
    print(f"  Avg Loss:           {res.get('avg_loss_pct', 0):>+7.2f}%")
    print(f"  Payoff Ratio:       {abs(res.get('avg_win_pct',0)/res.get('avg_loss_pct',-1)):>8.2f}" if res.get('avg_loss_pct', 0) != 0 else "")
    print(f"  Avg Hold Days:      {res.get('avg_hold_days', 0):>8.1f}")
    print(f"  Best Trade:         {res.get('best_trade_pct', 0):>+7.2f}%")
    print(f"  Worst Trade:        {res.get('worst_trade_pct', 0):>+7.2f}%")
    print(f"  Avg MFE:            {res.get('avg_mfe_pct', 0):>+7.2f}%")
    print(f"  Avg MAE:            {res.get('avg_mae_pct', 0):>+7.2f}%")
    print(f"  MFE Winners:        {res.get('mfe_winners', 0):>+7.2f}%")
    print(f"  MAE Losers:         {res.get('mae_losers', 0):>+7.2f}%")
    print(f"  Max Consec Wins:    {res.get('max_consec_wins', 0):>8}")
    print(f"  Max Consec Losses:  {res.get('max_consec_losses', 0):>8}")

    # Exit breakdown
    print(f"\n  Exits:")
    for reason, count in sorted(res.get("exit_breakdown", {}).items(), key=lambda x: -x[1]):
        pct = count / res["n_trades"] * 100
        print(f"    {reason:<22s} {count:>4}  ({pct:>5.1f}%)")

    # By trigger
    print(f"  Triggers:")
    for tt, s in sorted(res.get("by_trigger", {}).items()):
        print(f"    {tt:<32s} N={s['n']:>3} WR={s['win_rate']:>5.1f}% Avg={s['avg_return']:>+5.2f}% PF={s['pf']}")

    # Monthly
    if monthly:
        print(f"\n  Monthly P&L:")
        pos_m = sum(1 for v in monthly.values() if v > 0)
        print(f"    Profitable: {pos_m}/{len(monthly)} ({pos_m/len(monthly)*100:.0f}%)")
        print(f"    Best:  ₹{max(monthly.values()):>+11,.0f}  Worst: ₹{min(monthly.values()):>+11,.0f}  Avg: ₹{np.mean(list(monthly.values())):>+11,.0f}")
        cum = 0
        for mo in sorted(monthly):
            cum += monthly[mo]
            print(f"    {mo}  ₹{monthly[mo]:>+10,.0f}  cum ₹{cum:>+10,.0f}")

    return total_pnl


def run():
    print(f"KING 2 — APPLES-TO-APPLES COMPARISON")
    print(f"Date range: {MIN_DATE} to {MAX_DATE} (both datasets complete)")
    print(f"{'='*75}")

    nifty = yf.download("^NSEI", period="2y", progress=False)

    configs = [baseline(), king2()]

    results = {}

    for ds_label, ds_dir in [("V1", None), ("V2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trigger_results_v2"))]:
        print(f"\n{'▓'*75}")
        print(f"  DATASET: {ds_label}")
        print(f"{'▓'*75}")

        trigger_files = load_trigger_files(input_dir=ds_dir)
        picks_df = extract_picks(trigger_files)
        picks_filtered = filter_picks(picks_df, MIN_DATE, MAX_DATE)
        price_data = fetch_all_price_data(picks_filtered)
        regime_by_date = extract_regime_by_date(trigger_files)
        regime_filtered = filter_regime(regime_by_date, MIN_DATE, MAX_DATE)
        mi = build_market_indicators(regime_filtered, nifty)
        print(f"  Regime days: {len(regime_filtered)}, Indicator days: {len(mi)}")

        for feat in configs:
            res = simulate_portfolio(picks_filtered, price_data, feat,
                                     regime_by_date=regime_filtered, market_indicators=mi)
            key = f"{ds_label}_{feat.name}"
            pnl = print_stats(f"{ds_label} — {feat.name}", res)
            results[key] = {"pnl": pnl, "res": res}

    # ── CROSS COMPARISON TABLE ──
    print(f"\n\n{'='*100}")
    print(f"  SIDE-BY-SIDE COMPARISON (Jan 2024 – Jun 2025)")
    print(f"{'='*100}")

    metrics = [
        ("Total P&L", "total_pnl", True),
        ("CAGR %", "cagr_pct", False),
        ("Max Drawdown %", "max_drawdown_pct", False),
        ("Calmar", "calmar", False),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Trades", "n_trades", False),
        ("Win Rate %", "win_rate", False),
        ("Profit Factor", "profit_factor", False),
        ("Expectancy %", "expectancy_pct", False),
        ("Avg Win %", "avg_win_pct", False),
        ("Avg Loss %", "avg_loss_pct", False),
        ("Avg MFE %", "avg_mfe_pct", False),
        ("Avg MAE %", "avg_mae_pct", False),
        ("MFE Winners %", "mfe_winners", False),
        ("MAE Losers %", "mae_losers", False),
        ("Avg Hold Days", "avg_hold_days", False),
        ("Skipped", "trades_skipped", False),
    ]

    v1b = results.get("V1_Baseline", {}).get("res", {})
    v1k = results.get("V1_KING 2: Ratch SL(2-3) pos 8/6", {}).get("res", {})
    v2b = results.get("V2_Baseline", {}).get("res", {})
    v2k = results.get("V2_KING 2: Ratch SL(2-3) pos 8/6", {}).get("res", {})

    print(f"\n  {'Metric':<20} {'V1 Base':>12} {'V1 King2':>12} {'V2 Base':>12} {'V2 King2':>12}")
    print(f"  {'-'*72}")
    for name, key, is_rupee in metrics:
        vals = [v1b.get(key, 0), v1k.get(key, 0), v2b.get(key, 0), v2k.get(key, 0)]
        if is_rupee:
            # For P&L, compute from trades
            v1b_pnl = results.get("V1_Baseline", {}).get("pnl", 0)
            v1k_pnl = results.get("V1_KING 2: Ratch SL(2-3) pos 8/6", {}).get("pnl", 0)
            v2b_pnl = results.get("V2_Baseline", {}).get("pnl", 0)
            v2k_pnl = results.get("V2_KING 2: Ratch SL(2-3) pos 8/6", {}).get("pnl", 0)
            print(f"  {name:<20} {v1b_pnl:>+12,.0f} {v1k_pnl:>+12,.0f} {v2b_pnl:>+12,.0f} {v2k_pnl:>+12,.0f}")
        else:
            strs = []
            for v in vals:
                if isinstance(v, float):
                    strs.append(f"{v:>12.2f}")
                else:
                    strs.append(f"{v:>12}")
            print(f"  {name:<20} {''.join(strs)}")

    # King 2 improvement over baseline
    print(f"\n  KING 2 vs BASELINE IMPROVEMENT:")
    for ds, bl, k2 in [("V1", v1b, v1k), ("V2", v2b, v2k)]:
        bl_pnl = results.get(f"{ds}_Baseline", {}).get("pnl", 0)
        k2_pnl = results.get(f"{ds}_KING 2: Ratch SL(2-3) pos 8/6", {}).get("pnl", 0)
        pnl_imp = k2_pnl - bl_pnl
        mdd_imp = k2.get("max_drawdown_pct", 0) - bl.get("max_drawdown_pct", 0)
        cal_imp = k2.get("calmar", 0) - bl.get("calmar", 0)
        print(f"    {ds}: P&L {pnl_imp:>+12,.0f} | MDD {mdd_imp:>+6.1f}pp | Calmar {cal_imp:>+6.2f} | Sharpe {k2.get('sharpe',0)-bl.get('sharpe',0):>+5.2f}")


if __name__ == "__main__":
    run()
