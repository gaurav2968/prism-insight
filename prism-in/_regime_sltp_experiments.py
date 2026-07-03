"""Regime-adaptive SL/TP experiments: should we widen TP in bull / tighten SL in bear?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_lab_v2 import (simulate_portfolio, Features,
                              load_trigger_files, extract_picks, fetch_all_price_data,
                              extract_regime_by_date)
import logging
logging.basicConfig(level=logging.INFO)

# Base regime position limits (best config from prior experiments)
BASE_RPL = {
    "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
    "slope_declining": 0, "correction": 0,
    "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
}

def build_configs():
    configs = []

    # ── 0: Baseline (8 slots, SL=3x, TP=2x, 30d hold) ──
    configs.append(Features(name="A0: Baseline", max_positions=8))

    # ── 1: Best config from prior run (8/5 + Hold 20d) ──
    configs.append(Features(name="A1: 8/5+Hold20d", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL))

    # ═══ GROUP B: Widen TP in bull only (SL stays 3x) ═══
    # B1: bull_strong TP=3x (wider), everything else TP=2x
    configs.append(Features(name="B1: Bull TP=3x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 3.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # B2: bull_strong TP=2.5x
    configs.append(Features(name="B2: Bull TP=2.5x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.5),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # B3: bull_strong AND bull_medium TP=2.5x
    configs.append(Features(name="B3: Both bull TP=2.5x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.5),
                                "bull_medium": (3.0, 2.5),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # B4: bull_strong TP=3x, bull_medium TP=2.5x (tiered)
    configs.append(Features(name="B4: Tiered TP 3x/2.5x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 3.0),
                                "bull_medium": (3.0, 2.5),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # ═══ GROUP C: Tighten SL in bear (TP stays 2x) ═══
    # C1: bear_bottom SL=2x (tighter)
    configs.append(Features(name="C1: Bear SL=2x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.0, 2.0),
                                "bear_bottom_extreme": (2.0, 2.0),
                            }))

    # C2: bear_bottom SL=2.5x
    configs.append(Features(name="C2: Bear SL=2.5x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.5, 2.0),
                                "bear_bottom_extreme": (2.5, 2.0),
                            }))

    # C3: bear_bottom SL=2x + wider TP=3x (tight risk, big reward in recovery)
    configs.append(Features(name="C3: Bear SL=2x TP=3x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.0, 3.0),
                                "bear_bottom_extreme": (2.0, 3.0),
                            }))

    # ═══ GROUP D: Bull wide TP + Bear tight SL (both directions) ═══
    # D1: bull TP=2.5x + bear SL=2x
    configs.append(Features(name="D1: Bull TP=2.5 + Bear SL=2", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.5),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.0, 2.0),
                                "bear_bottom_extreme": (2.0, 2.0),
                            }))

    # D2: bull TP=3x + bear SL=2x
    configs.append(Features(name="D2: Bull TP=3 + Bear SL=2", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 3.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.0, 2.0),
                                "bear_bottom_extreme": (2.0, 2.0),
                            }))

    # D3: bull_strong TP=2.5x + bull_medium TP=2.5x + bear SL=2x
    configs.append(Features(name="D3: All bull TP=2.5 + Bear SL=2", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.5),
                                "bull_medium": (3.0, 2.5),
                                "bear_bottom": (2.0, 2.0),
                                "bear_bottom_extreme": (2.0, 2.0),
                            }))

    # ═══ GROUP E: Asymmetric — tighter SL everywhere + wider TP in bull ═══
    # E1: SL=2.5x everywhere + bull_strong TP=2.5x
    configs.append(Features(name="E1: SL=2.5x all + Bull TP=2.5", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (2.5, 2.5),
                                "bull_medium": (2.5, 2.0),
                                "bear_bottom": (2.5, 2.0),
                                "bear_bottom_extreme": (2.5, 2.0),
                            }))

    # E2: SL=2.5x everywhere + bull TP=3x
    configs.append(Features(name="E2: SL=2.5x all + Bull TP=3x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (2.5, 3.0),
                                "bull_medium": (2.5, 2.0),
                                "bear_bottom": (2.5, 2.0),
                                "bear_bottom_extreme": (2.5, 2.0),
                            }))

    # ═══ GROUP F: Hold period variations with regime SL/TP ═══
    # F1: 30d hold + bull TP=2.5x (no hold reduction, just widen TP)
    configs.append(Features(name="F1: 30d + Bull TP=2.5x", max_positions=8, max_hold_days=30,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.5),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # F2: 30d hold + bear SL=2x
    configs.append(Features(name="F2: 30d + Bear SL=2x", max_positions=8, max_hold_days=30,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.0, 2.0),
                                "bear_bottom_extreme": (2.0, 2.0),
                            }))

    # F3: 15d hold (shorter) + bull TP=2.5x (faster turnover + wider target in bull)
    configs.append(Features(name="F3: 15d + Bull TP=2.5x", max_positions=8, max_hold_days=15,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.5),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # ═══ GROUP G: Extreme tests ═══
    # G1: bull_strong SL=3.5x TP=3x (very wide both — let it breathe in strong bull)
    configs.append(Features(name="G1: Bull SL=3.5/TP=3", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.5, 3.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (3.0, 2.0),
                                "bear_bottom_extreme": (3.0, 2.0),
                            }))

    # G2: bear SL=1.5x (very tight — quick cut)
    configs.append(Features(name="G2: Bear SL=1.5x", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (1.5, 2.0),
                                "bear_bottom_extreme": (1.5, 2.0),
                            }))

    # G3: bear SL=2x + bear hold=15d (shorter hold in bear recovery)
    configs.append(Features(name="G3: Bear SL=2x (bear_bottom only)", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL,
                            regime_entry_sltp={
                                "bull_strong": (3.0, 2.0),
                                "bull_medium": (3.0, 2.0),
                                "bear_bottom": (2.0, 2.5),
                                "bear_bottom_extreme": (2.0, 3.0),
                            }))

    print(f"Built {len(configs)} experiment configs")
    return configs


def extract_stats(name, res):
    """Extract consistent stats from simulate_portfolio result."""
    trades = res.get("_trades", [])
    total_pnl = sum(t["pnl"] for t in trades) if trades else 0
    n_trades = res.get("n_trades", 0)
    # Monthly P&L
    from collections import defaultdict
    monthly = defaultdict(float)
    for t in trades:
        monthly[t["exit_date"][:6]] += t["pnl"]
    worst_mo = min(monthly.values()) if monthly else 0
    best_mo = max(monthly.values()) if monthly else 0
    return {
        "name": name,
        "n_trades": n_trades,
        "pnl": round(total_pnl, 0),
        "wr": res.get("win_rate", 0),
        "pf": res.get("profit_factor", 0),
        "cagr": res.get("cagr_pct", 0),
        "mdd": res.get("max_drawdown_pct", 0),
        "calmar": res.get("calmar", 0),
        "worst_mo": round(worst_mo, 0),
        "best_mo": round(best_mo, 0),
        "skipped": res.get("trades_skipped", 0),
    }


def run_dataset(label, configs, input_dir=None, min_date=None, max_date=None):
    trigger_files = load_trigger_files(input_dir=input_dir, min_date=min_date, max_date=max_date)
    picks_df = extract_picks(trigger_files)
    price_data = fetch_all_price_data(picks_df)
    regime_by_date = extract_regime_by_date(trigger_files)
    print(f"  {label}: {len(picks_df)} picks, {len(regime_by_date)} regime days")

    results = []
    for feat in configs:
        res = simulate_portfolio(picks_df, price_data, feat, regime_by_date=regime_by_date)
        stats = extract_stats(feat.name, res)
        print(f"  ✓ {feat.name}: {stats['pnl']:>+12,.0f}")
        results.append(stats)
    return results


def print_results(label, results):
    results_sorted = sorted(results, key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'='*140}")
    print(f"  {label}")
    print(f"{'='*140}")
    print(f"{'#':<4} {'Config':<35} {'Trades':>7} {'P&L':>12} {'WR%':>6} {'PF':>6} "
          f"{'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Worst Mo':>11} {'Best Mo':>11} {'Skip':>6}")
    print("-" * 140)
    for i, r in enumerate(results_sorted, 1):
        marker = " ***" if i == 1 else ""
        pf = r["pf"] if isinstance(r["pf"], (int, float)) else 0
        print(f"{i:<4} {r['name']:<35} {r['n_trades']:>7} {r['pnl']:>+12,.0f} "
              f"{r['wr']:>5.1f}% {pf:>5.2f} "
              f"{r['cagr']:>6.1f}% {r['mdd']:>6.1f}% "
              f"{r['calmar']:>7.2f} "
              f"{r['worst_mo']:>+11,.0f} {r['best_mo']:>+11,.0f} "
              f"{r['skipped']:>6}{marker}")


def print_cross_compare(v1_results, v2_results):
    v1_map = {r["name"]: r for r in v1_results}
    v2_map = {r["name"]: r for r in v2_results}
    all_names = list(dict.fromkeys([r["name"] for r in v1_results] + [r["name"] for r in v2_results]))

    rows = []
    for name in all_names:
        r1 = v1_map.get(name, {})
        r2 = v2_map.get(name, {})
        p1 = r1.get("pnl", 0)
        p2 = r2.get("pnl", 0)
        avg = (p1 + p2) / 2
        rows.append((name, p1, p2, avg,
                      r1.get("mdd", 0), r2.get("mdd", 0),
                      r1.get("calmar", 0), r2.get("calmar", 0)))
    rows.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*130}")
    print(f"  CROSS-DATASET: CONFIGS SORTED BY AVG P&L (V1 + V2)")
    print(f"{'='*130}")
    print(f"{'Config':<35} {'V1 P&L':>12} {'V2 P&L':>12} {'Avg P&L':>12} {'V1 MDD':>8} {'V2 MDD':>8} {'V1 Cal':>8} {'V2 Cal':>8}")
    print("-" * 130)
    for name, p1, p2, avg, mdd1, mdd2, cal1, cal2 in rows:
        print(f"{name:<35} {p1:>+12,.0f} {p2:>+12,.0f} {avg:>+12,.0f} {mdd1:>7.1f}% {mdd2:>7.1f}% {cal1:>8.2f} {cal2:>8.2f}")


if __name__ == "__main__":
    configs = build_configs()

    print(f"\n{'='*60}")
    print(f"  Loading V1 data...")
    print(f"{'='*60}")
    v1_results = run_dataset("V1", configs)
    print_results("V1 DATA (Jan 2024 - May 2026)", v1_results)

    print(f"\n{'='*60}")
    print(f"  Loading V2 data...")
    print(f"{'='*60}")
    v2_results = run_dataset("V2", configs,
                              input_dir=os.path.join(os.path.dirname(__file__), "..", "trigger_results_v2"))
    print_results("V2 DATA (Jan 2024 - Jul 2025)", v2_results)

    print_cross_compare(v1_results, v2_results)
