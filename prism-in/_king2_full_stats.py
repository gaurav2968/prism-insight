"""
King 2 Strategy — Full Performance Metrics
Runs King 2 (Ratchet SL 2-3, pos 8/6, EMA5, 20d hold) on both V1 and V2 datasets
and prints comprehensive stats: P&L, MFE, MAE, Sharpe, Sortino, Calmar, etc.
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

RPL_KING2 = {
    "bull_strong": 8, "bull_medium": 6, "bull_weak": 0,
    "slope_declining": 0, "correction": 0,
    "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
}

def king2_config():
    return Features(
        name="KING 2: Ratch SL(2-3) pos 8/6",
        max_positions=8,
        max_hold_days=20,
        regime_position_limits=RPL_KING2,
        market_adaptive=True,
        market_score_key="ema5",
        market_hold_adjust=True,
        market_ratchet_only=True,
        market_sl_bear=2.0,
        market_sl_bull=3.0,
        market_tp_bear=2.0,
        market_tp_bull=2.0,
    )


def print_full_stats(label, res):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Config: {res['features']}")
    print(f"{'='*80}")

    print(f"\n── RETURNS ──")
    print(f"  Total P&L:            ₹{res.get('total_pnl', 0):>+12,.0f}")
    print(f"  Total Return:         {res.get('total_return_pct', 0):>+8.1f}%")
    print(f"  CAGR:                 {res.get('cagr_pct', 0):>+8.1f}%")
    print(f"  Final Equity:         ₹{res.get('final_equity', 0):>12,.0f}")

    print(f"\n── RISK ──")
    print(f"  Max Drawdown:         {res.get('max_drawdown_pct', 0):>8.1f}%")
    print(f"  Calmar Ratio:         {res.get('calmar', 0):>8.2f}")
    print(f"  Sharpe Ratio:         {res.get('sharpe', 0):>8.2f}")
    print(f"  Sortino Ratio:        {res.get('sortino', 0):>8.2f}")

    print(f"\n── TRADE STATS ──")
    print(f"  Total Trades:         {res.get('n_trades', 0):>8}")
    print(f"  Trades Skipped:       {res.get('trades_skipped', 0):>8}")
    print(f"  Win Rate:             {res.get('win_rate', 0):>7.1f}%")
    print(f"  Profit Factor:        {res.get('profit_factor', 0):>8}")
    print(f"  Avg Hold Days:        {res.get('avg_hold_days', 0):>8.1f}")
    print(f"  Expectancy:           {res.get('expectancy_pct', 0):>+7.2f}% per trade")

    print(f"\n── RETURN DISTRIBUTION ──")
    print(f"  Avg Return (net):     {res.get('avg_return_net', 0):>+7.2f}%")
    print(f"  Median Return:        {res.get('median_return_pct', 0):>+7.2f}%")
    print(f"  Std Dev:              {res.get('std_return_pct', 0):>7.2f}%")
    print(f"  Best Trade:           {res.get('best_trade_pct', 0):>+7.2f}%")
    print(f"  Worst Trade:          {res.get('worst_trade_pct', 0):>+7.2f}%")
    print(f"  Avg Winner:           {res.get('avg_win_pct', 0):>+7.2f}%")
    print(f"  Avg Loser:            {res.get('avg_loss_pct', 0):>+7.2f}%")
    print(f"  Payoff Ratio:         {abs(res.get('avg_win_pct',0)/res.get('avg_loss_pct',-1)):>8.2f}" if res.get('avg_loss_pct', 0) != 0 else "  Payoff Ratio:              N/A")

    print(f"\n── MFE / MAE (Max Favorable / Adverse Excursion) ──")
    print(f"  Avg MFE (all):        {res.get('avg_mfe_pct', 0):>+7.2f}%   (avg peak unrealized gain)")
    print(f"  Avg MAE (all):        {res.get('avg_mae_pct', 0):>+7.2f}%   (avg peak unrealized loss)")
    print(f"  Max MFE:              {res.get('max_mfe_pct', 0):>+7.2f}%   (single best intra-trade peak)")
    print(f"  Max MAE:              {res.get('max_mae_pct', 0):>+7.2f}%   (single worst intra-trade dip)")
    print(f"  Avg MFE (winners):    {res.get('mfe_winners', 0):>+7.2f}%   (how far winners ran before exit)")
    print(f"  Avg MAE (losers):     {res.get('mae_losers', 0):>+7.2f}%   (how deep losers dipped before exit)")

    # MFE efficiency: how much of peak gain was captured?
    trades = res.get("_trades", [])
    if trades:
        winners = [t for t in trades if t["net_return_pct"] > 0]
        losers = [t for t in trades if t["net_return_pct"] <= 0]
        if winners:
            mfe_eff = [t["net_return_pct"] / t["mfe_pct"] * 100 if t.get("mfe_pct", 0) > 0 else 0 for t in winners]
            avg_mfe_eff = np.mean([e for e in mfe_eff if e > 0])
            print(f"  MFE Efficiency:       {avg_mfe_eff:>7.1f}%   (% of peak gain captured at exit)")
        if losers:
            mae_ratios = [t["net_return_pct"] / t["mae_pct"] * 100 if t.get("mae_pct", 0) < 0 else 0 for t in losers]
            avg_mae_ratio = np.mean([r for r in mae_ratios if r > 0])
            print(f"  MAE Capture:          {avg_mae_ratio:>7.1f}%   (% of worst dip realized as loss)")

    print(f"\n── STREAKS ──")
    print(f"  Max Consecutive Wins:  {res.get('max_consec_wins', 0):>6}")
    print(f"  Max Consecutive Losses:{res.get('max_consec_losses', 0):>6}")

    print(f"\n── EXIT BREAKDOWN ──")
    for reason, count in sorted(res.get("exit_breakdown", {}).items(), key=lambda x: -x[1]):
        pct = count / res["n_trades"] * 100
        print(f"  {reason:<25s} {count:>5}  ({pct:>5.1f}%)")

    print(f"\n── BY TRIGGER TYPE ──")
    for tt, stats in sorted(res.get("by_trigger", {}).items()):
        print(f"  {tt:<35s}  N={stats['n']:>3}  WR={stats['win_rate']:>5.1f}%  AvgRet={stats['avg_return']:>+6.2f}%  PF={stats['pf']}")

    # Monthly P&L
    if trades:
        monthly = defaultdict(float)
        for t in trades:
            mo = t["exit_date"][:6]
            monthly[mo] += t["pnl"]
        print(f"\n── MONTHLY P&L ──")
        months_sorted = sorted(monthly.keys())
        pos_months = sum(1 for v in monthly.values() if v > 0)
        neg_months = sum(1 for v in monthly.values() if v <= 0)
        print(f"  Profitable months: {pos_months}/{len(monthly)} ({pos_months/len(monthly)*100:.0f}%)")
        print(f"  Best month:    ₹{max(monthly.values()):>+12,.0f}")
        print(f"  Worst month:   ₹{min(monthly.values()):>+12,.0f}")
        print(f"  Avg month:     ₹{np.mean(list(monthly.values())):>+12,.0f}")
        print(f"  Median month:  ₹{np.median(list(monthly.values())):>+12,.0f}")
        print(f"  {'Month':<8} {'P&L':>12} {'Cumulative':>12}")
        print(f"  {'-'*36}")
        cum = 0
        for mo in months_sorted:
            cum += monthly[mo]
            bar = "█" * max(0, int(monthly[mo] / 10000)) if monthly[mo] > 0 else "▓" * max(0, int(-monthly[mo] / 10000))
            print(f"  {mo:<8} ₹{monthly[mo]:>+11,.0f} ₹{cum:>+11,.0f}  {bar}")

    # Drawdown analysis
    eq_curve = res.get("_equity_curve", [])
    if eq_curve:
        eq_df = pd.DataFrame(eq_curve)
        eq_series = eq_df.set_index("date")["equity"]
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak * 100
        # Top 3 drawdowns
        print(f"\n── TOP DRAWDOWNS ──")
        in_dd = False
        dd_start = None
        dds = []
        for i, (date, val) in enumerate(dd.items()):
            if val < 0 and not in_dd:
                in_dd = True
                dd_start = date
            elif val == 0 and in_dd:
                in_dd = False
                dd_slice = dd[dd_start:date]
                worst = dd_slice.min()
                worst_date = dd_slice.idxmin()
                dds.append((worst, dd_start, worst_date, date))
        if in_dd:
            dd_slice = dd[dd_start:]
            worst = dd_slice.min()
            worst_date = dd_slice.idxmin()
            dds.append((worst, dd_start, worst_date, "(ongoing)"))
        dds.sort(key=lambda x: x[0])
        for i, (depth, start, trough, end) in enumerate(dds[:5]):
            print(f"  #{i+1}: {depth:>6.1f}%  from {start} to {trough} (recovered: {end})")


def run():
    feat = king2_config()

    # ── V1 Dataset ──
    print("\n" + "▓" * 80)
    print("  LOADING V1 DATASET (trigger_results_in_morning_*)")
    print("▓" * 80)
    v1_trigger = load_trigger_files(input_dir=None)  # default: root trigger_results_in_morning_*
    v1_picks = extract_picks(v1_trigger)
    v1_prices = fetch_all_price_data(v1_picks)
    v1_regime = extract_regime_by_date(v1_trigger)
    nifty = yf.download("^NSEI", period="2y", progress=False)
    v1_mi = build_market_indicators(v1_regime, nifty)
    print(f"  V1: {len(v1_picks)} picks, {len(v1_regime)} regime days, {len(v1_mi)} indicator days")

    v1_res = simulate_portfolio(v1_picks, v1_prices, feat,
                                regime_by_date=v1_regime, market_indicators=v1_mi)
    print_full_stats("KING 2 — V1 (hybrid_3factor scoring)", v1_res)

    # ── V2 Dataset ──
    v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trigger_results_v2")
    if os.path.isdir(v2_dir):
        print("\n" + "▓" * 80)
        print("  LOADING V2 DATASET (trigger_results_v2/)")
        print("▓" * 80)
        v2_trigger = load_trigger_files(input_dir=v2_dir)
        v2_picks = extract_picks(v2_trigger)
        v2_prices = fetch_all_price_data(v2_picks)
        v2_regime = extract_regime_by_date(v2_trigger)
        v2_mi = build_market_indicators(v2_regime, nifty)
        print(f"  V2: {len(v2_picks)} picks, {len(v2_regime)} regime days, {len(v2_mi)} indicator days")

        v2_res = simulate_portfolio(v2_picks, v2_prices, feat,
                                    regime_by_date=v2_regime, market_indicators=v2_mi)
        print_full_stats("KING 2 — V2 (momentum_first scoring)", v2_res)

        # ── Cross comparison ──
        print(f"\n{'='*80}")
        print(f"  KING 2 — V1 vs V2 COMPARISON")
        print(f"{'='*80}")
        metrics = [
            ("Total P&L", "total_pnl", "₹{:>+12,.0f}"),
            ("CAGR", "cagr_pct", "{:>+8.1f}%"),
            ("Max Drawdown", "max_drawdown_pct", "{:>8.1f}%"),
            ("Calmar", "calmar", "{:>8.2f}"),
            ("Sharpe", "sharpe", "{:>8.2f}"),
            ("Sortino", "sortino", "{:>8.2f}"),
            ("Win Rate", "win_rate", "{:>7.1f}%"),
            ("Profit Factor", "profit_factor", "{:>8}"),
            ("Expectancy", "expectancy_pct", "{:>+7.2f}%"),
            ("Avg MFE", "avg_mfe_pct", "{:>+7.2f}%"),
            ("Avg MAE", "avg_mae_pct", "{:>+7.2f}%"),
            ("MFE Winners", "mfe_winners", "{:>+7.2f}%"),
            ("MAE Losers", "mae_losers", "{:>+7.2f}%"),
            ("N Trades", "n_trades", "{:>8}"),
            ("Trades Skipped", "trades_skipped", "{:>8}"),
        ]
        print(f"  {'Metric':<22} {'V1':>14} {'V2':>14} {'Diff':>14}")
        print(f"  {'-'*66}")
        for name, key, fmt in metrics:
            v1v = v1_res.get(key, 0)
            v2v = v2_res.get(key, 0)
            v1s = fmt.format(v1v) if isinstance(v1v, (int, float)) else str(v1v)
            v2s = fmt.format(v2v) if isinstance(v2v, (int, float)) else str(v2v)
            if isinstance(v1v, (int, float)) and isinstance(v2v, (int, float)):
                diff = v2v - v1v
                if "₹" in fmt:
                    ds = f"₹{diff:>+12,.0f}"
                elif "%" in fmt:
                    ds = f"{diff:>+7.2f}%"
                else:
                    ds = f"{diff:>+8.2f}"
            else:
                ds = ""
            print(f"  {name:<22} {v1s:>14} {v2s:>14} {ds:>14}")

        # Average
        avg_pnl = (v1_res.get("total_pnl", 0) + v2_res.get("total_pnl", 0)) / 2
        print(f"\n  ★ Average P&L (V1+V2): ₹{avg_pnl:>+12,.0f}")
    else:
        print(f"\n  [!] V2 dataset not found at {v2_dir} — skipping V2")


if __name__ == "__main__":
    run()
