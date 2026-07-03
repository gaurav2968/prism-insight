"""
Comprehensive regime experiments — 20+ variants on both V1 and V2 data.
Tests: position limits, hold period by regime, early exit in bear, 
extended hold in bull, SL/TP scaling, combo configs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_lab_v2 import (
    simulate_portfolio, Features, make_feature_set,
    extract_regime_by_date, REGIME_SCORE_MAP,
    fetch_all_price_data,
)
from backtest_engine import load_trigger_files, extract_picks
import logging
logging.basicConfig(level=logging.WARNING)


def run_experiment(picks_df, price_data, regime_by_date, features, label):
    """Run one experiment and return summary dict."""
    result = simulate_portfolio(picks_df, price_data, features, regime_by_date=regime_by_date)
    trades = result.get("_trades", [])
    total_pnl = sum(t["pnl"] for t in trades)

    # Monthly worst/best
    from collections import defaultdict
    monthly = defaultdict(float)
    for t in trades:
        monthly[t["exit_date"][:6]] += t["pnl"]
    worst_m = min(monthly.values()) if monthly else 0
    best_m = max(monthly.values()) if monthly else 0

    return {
        "label": label,
        "n_trades": result.get("n_trades", 0),
        "pnl": round(total_pnl, 0),
        "wr": result.get("win_rate", 0),
        "pf": result.get("profit_factor", 0),
        "cagr": result.get("cagr_pct", 0),
        "mdd": result.get("max_drawdown_pct", 0),
        "calmar": result.get("calmar", 0),
        "skipped": result.get("trades_skipped", 0),
        "worst_month": round(worst_m, 0),
        "best_month": round(best_m, 0),
        "final_eq": result.get("final_equity", 0),
    }


def build_experiments():
    """Return list of (label, Features) tuples to test."""
    configs = []

    # ── GROUP A: Position Limits ──
    configs.append(("A0: Baseline 8 slots", 
        Features(name="baseline", max_positions=8)))

    for strong, med in [(8, 5), (8, 4), (8, 6), (6, 4), (6, 3), (10, 6), (10, 5)]:
        f = Features(name=f"pos_{strong}_{med}", max_positions=strong,
                     regime_position_limits={
                         "bull_strong": strong, "bull_medium": med, "bull_weak": 0,
                         "slope_declining": 0, "correction": 0,
                         "bear": 0, "bear_bottom": 2, "bear_bottom_extreme": 2,
                     })
        configs.append((f"A: pos {strong}/{med}", f))

    # ── GROUP B: Hold Period by Regime ──
    # Shorter hold in bearish = exit faster
    for hold in [15, 20, 25]:
        f = Features(name=f"hold_{hold}", max_positions=8, max_hold_days=hold)
        configs.append((f"B: hold {hold}d (always)", f))

    # ── GROUP C: SL/TP Scaling ──
    # Tighter SL
    for sl_m, tp_m in [(2.0, 2.0), (2.5, 2.0), (2.0, 3.0), (3.0, 3.0), (2.5, 2.5), (2.0, 1.5)]:
        f = Features(name=f"sl{sl_m}_tp{tp_m}", max_positions=8,
                     initial_sl_mult=sl_m, initial_tp_mult=tp_m)
        configs.append((f"C: SL={sl_m}x TP={tp_m}x", f))

    # ── GROUP D: Ratchet Stop ──
    for ratchet_m in [1.5, 2.0, 2.5]:
        f = Features(name=f"ratchet_{ratchet_m}", max_positions=8,
                     ratchet_stop=True, ratchet_atr_mult=ratchet_m)
        configs.append((f"D: Ratchet {ratchet_m}x", f))

    # ── GROUP E: Time Cut ──
    for days, r_thresh in [(5, 0.0), (7, 0.0), (7, 0.5), (10, 0.5), (5, 0.5)]:
        f = Features(name=f"timecut_{days}d_{r_thresh}R", max_positions=8,
                     time_expectancy_cut=True, time_cut_days=days, time_cut_r_threshold=r_thresh)
        configs.append((f"E: TimeCut {days}d/{r_thresh}R", f))

    # ── GROUP F: Best Position Limit + Other Features ──
    # Combine winner (8/5) with other improvements
    base_limits = {
        "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
        "slope_declining": 0, "correction": 0,
        "bear": 0, "bear_bottom": 2, "bear_bottom_extreme": 2,
    }

    f = Features(name="pos85_ratchet2", max_positions=8,
                 regime_position_limits=base_limits,
                 ratchet_stop=True, ratchet_atr_mult=2.0)
    configs.append(("F: 8/5 + Ratchet 2x", f))

    f = Features(name="pos85_timecut7", max_positions=8,
                 regime_position_limits=base_limits,
                 time_expectancy_cut=True, time_cut_days=7, time_cut_r_threshold=0.5)
    configs.append(("F: 8/5 + TimeCut 7d/0.5R", f))

    f = Features(name="pos85_hold20", max_positions=8,
                 regime_position_limits=base_limits,
                 max_hold_days=20)
    configs.append(("F: 8/5 + Hold 20d", f))

    f = Features(name="pos85_sl2_tp3", max_positions=8,
                 regime_position_limits=base_limits,
                 initial_sl_mult=2.0, initial_tp_mult=3.0)
    configs.append(("F: 8/5 + SL2/TP3", f))

    f = Features(name="pos85_sl25_tp25", max_positions=8,
                 regime_position_limits=base_limits,
                 initial_sl_mult=2.5, initial_tp_mult=2.5)
    configs.append(("F: 8/5 + SL2.5/TP2.5", f))

    f = Features(name="pos85_ratchet_timecut", max_positions=8,
                 regime_position_limits=base_limits,
                 ratchet_stop=True, ratchet_atr_mult=2.0,
                 time_expectancy_cut=True, time_cut_days=7, time_cut_r_threshold=0.5)
    configs.append(("F: 8/5 + Ratchet + TimeCut", f))

    # ── GROUP G: Breakeven + R-trail combos ──
    f = Features(name="pos85_breakeven", max_positions=8,
                 regime_position_limits=base_limits,
                 breakeven_stop=True, breakeven_atr_mult=1.0)
    configs.append(("G: 8/5 + Breakeven 1xATR", f))

    f = Features(name="pos85_rtrail", max_positions=8,
                 regime_position_limits=base_limits,
                 r_multiple_trail=True)
    configs.append(("G: 8/5 + R-trail", f))

    # ── GROUP H: Extended hold in bull ──
    for hold in [40, 45]:
        f = Features(name=f"pos85_hold{hold}", max_positions=8,
                     regime_position_limits=base_limits,
                     max_hold_days=hold)
        configs.append((f"H: 8/5 + Hold {hold}d (bull run)", f))

    return configs


def print_results(results, dataset_label):
    print(f"\n{'='*140}")
    print(f"  {dataset_label}")
    print(f"{'='*140}")
    print(f"{'#':<3} {'Config':<35} {'Trades':>7} {'P&L':>12} {'WR%':>6} {'PF':>6} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Worst Mo':>10} {'Best Mo':>10} {'Skip':>6}")
    print(f"-"*140)

    # Sort by P&L
    for i, r in enumerate(sorted(results, key=lambda x: x["pnl"], reverse=True), 1):
        pf_str = f"{r['pf']:.2f}" if isinstance(r['pf'], float) else str(r['pf'])
        marker = " ***" if r["pnl"] == max(x["pnl"] for x in results) else ""
        print(f"{i:<3} {r['label']:<35} {r['n_trades']:>7} {r['pnl']:>+12,.0f} {r['wr']:>5.1f}% {pf_str:>6} {r['cagr']:>6.1f}% {r['mdd']:>6.1f}% {r['calmar']:>7.2f} {r['worst_month']:>+10,.0f} {r['best_month']:>+10,.0f} {r['skipped']:>6}{marker}")


def main():
    configs = build_experiments()
    print(f"Built {len(configs)} experiment configs")

    # ── V1 Data ──
    print("\n" + "="*60)
    print("  Loading V1 data...")
    print("="*60)
    tf_v1 = load_trigger_files()
    picks_v1 = extract_picks(tf_v1)
    price_v1 = fetch_all_price_data(picks_v1)
    regime_v1 = extract_regime_by_date(tf_v1)
    print(f"  V1: {len(picks_v1)} picks, {len(regime_v1)} regime days")

    results_v1 = []
    for label, features in configs:
        use_regime = features.regime_position_limits is not None
        r = run_experiment(picks_v1, price_v1,
                          regime_v1 if use_regime else None,
                          features, label)
        results_v1.append(r)
        print(f"  ✓ {label}: {r['pnl']:>+12,.0f}")

    print_results(results_v1, "V1 DATA (Jan 2024 - May 2026)")

    # ── V2 Data ──
    print("\n" + "="*60)
    print("  Loading V2 data...")
    print("="*60)
    v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trigger_results_v2")
    tf_v2 = load_trigger_files(input_dir=v2_dir)
    picks_v2 = extract_picks(tf_v2)
    price_v2 = fetch_all_price_data(picks_v2)
    regime_v2 = extract_regime_by_date(tf_v2)
    print(f"  V2: {len(picks_v2)} picks, {len(regime_v2)} regime days")

    results_v2 = []
    for label, features in configs:
        use_regime = features.regime_position_limits is not None
        r = run_experiment(picks_v2, price_v2,
                          regime_v2 if use_regime else None,
                          features, label)
        results_v2.append(r)
        print(f"  ✓ {label}: {r['pnl']:>+12,.0f}")

    print_results(results_v2, "V2 DATA (Jan 2024 - Jul 2025)")

    # ── Top 5 that work on BOTH ──
    print(f"\n{'='*100}")
    print(f"  CONFIGS THAT WORK ON BOTH V1 AND V2 (sorted by avg P&L)")
    print(f"{'='*100}")
    combined = []
    for rv1, rv2 in zip(results_v1, results_v2):
        avg_pnl = (rv1["pnl"] + rv2["pnl"]) / 2
        avg_calmar = (rv1["calmar"] + rv2["calmar"]) / 2
        combined.append({
            "label": rv1["label"],
            "v1_pnl": rv1["pnl"], "v2_pnl": rv2["pnl"], "avg_pnl": avg_pnl,
            "v1_mdd": rv1["mdd"], "v2_mdd": rv2["mdd"],
            "v1_calmar": rv1["calmar"], "v2_calmar": rv2["calmar"], "avg_calmar": avg_calmar,
            "v1_wr": rv1["wr"], "v2_wr": rv2["wr"],
        })

    print(f"{'Config':<35} {'V1 P&L':>12} {'V2 P&L':>12} {'Avg P&L':>12} {'V1 MDD':>8} {'V2 MDD':>8} {'V1 Cal':>7} {'V2 Cal':>7}")
    print("-"*100)
    for c in sorted(combined, key=lambda x: x["avg_pnl"], reverse=True)[:15]:
        print(f"{c['label']:<35} {c['v1_pnl']:>+12,.0f} {c['v2_pnl']:>+12,.0f} {c['avg_pnl']:>+12,.0f} {c['v1_mdd']:>7.1f}% {c['v2_mdd']:>7.1f}% {c['v1_calmar']:>7.2f} {c['v2_calmar']:>7.2f}")


if __name__ == "__main__":
    main()
