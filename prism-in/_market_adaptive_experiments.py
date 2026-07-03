"""
Comprehensive Market-Adaptive SL/TP Experiments
================================================
Tests known quantitative methods for noise-filtered regime signals:

1. EMA-smoothed regime score → entry SL/TP (various windows)
2. SMA-smoothed regime score → entry SL/TP
3. Score momentum → adjust SL/TP based on trend acceleration
4. During-hold ratchet adjustment (only tighten as market weakens)
5. During-hold full dynamic (bidirectional adjustment)
6. NIFTY realized vol scaling (vol-targeted exits)
7. ADX trend strength filtering
8. Composite: smoothed score + vol + position limits

Philosophy behind each approach:
- EMA: recent data weighted more → faster reaction to regime change
- SMA: equal weights → filters single-day noise spikes
- Momentum: detects when trend is accelerating vs decelerating
- Ratchet: "you can tighten stops but never loosen" → asymmetric risk management
- Vol scaling: high vol → wider SL (avoid noise stopouts) + tighter TP (take profits in vol)
- ADX: strong trend → let winners run (wider TP), weak trend → take profits (tighter TP)
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
import logging
logging.basicConfig(level=logging.INFO)

# Base position limits (proven winner from prior experiments)
BASE_RPL = {
    "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
    "slope_declining": 0, "correction": 0,
    "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
}


def build_configs():
    configs = []

    # ═══ A: REFERENCES ═══
    configs.append(Features(name="A0: Baseline (no adaptive)", max_positions=8))
    configs.append(Features(name="A1: 8/5+Hold20d (prev best)", max_positions=8, max_hold_days=20,
                            regime_position_limits=BASE_RPL))

    # ═══ B: EMA SMOOTHING AT ENTRY — which window filters noise best? ═══
    # Score → SL/TP: bear(-5)=SL2/TP1.5, bull(+5)=SL3/TP2.5 (conservative range)
    for key, label in [("ema3", "EMA3"), ("ema5", "EMA5"), ("ema10", "EMA10"), ("ema20", "EMA20")]:
        configs.append(Features(
            name=f"B: {label} entry SL(2-3)/TP(1.5-2.5)",
            max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
            market_adaptive=True, market_score_key=key,
            market_sl_bear=2.0, market_sl_bull=3.0,
            market_tp_bear=1.5, market_tp_bull=2.5,
        ))

    # ═══ C: SMA SMOOTHING AT ENTRY ═══
    for key, label in [("sma5", "SMA5"), ("sma10", "SMA10")]:
        configs.append(Features(
            name=f"C: {label} entry SL(2-3)/TP(1.5-2.5)",
            max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
            market_adaptive=True, market_score_key=key,
            market_sl_bear=2.0, market_sl_bull=3.0,
            market_tp_bear=1.5, market_tp_bull=2.5,
        ))

    # ═══ D: DIFFERENT SL/TP MAPPING RANGES ═══
    # D1: Wide range — big difference between bull and bear
    configs.append(Features(
        name="D1: EMA5 wide SL(1.5-3.5)/TP(1-3)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=1.5, market_sl_bull=3.5,
        market_tp_bear=1.0, market_tp_bull=3.0,
    ))
    # D2: TP-only scaling (keep SL=3x always, only adjust TP)
    configs.append(Features(
        name="D2: EMA5 SL=3x fixed, TP(1.5-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=3.0, market_sl_bull=3.0,  # SL doesn't change
        market_tp_bear=1.5, market_tp_bull=2.5,
    ))
    # D3: SL-only scaling (keep TP=2x always, only adjust SL)
    configs.append(Features(
        name="D3: EMA5 SL(2-3), TP=2x fixed",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=2.0, market_tp_bull=2.0,  # TP doesn't change
    ))
    # D4: Asymmetric — tighten SL in bear MORE, widen TP in bull LESS
    configs.append(Features(
        name="D4: EMA5 asymm SL(1.5-3)/TP(2-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=1.5, market_sl_bull=3.0,
        market_tp_bear=2.0, market_tp_bull=2.5,
    ))

    # ═══ E: MOMENTUM-BASED — rate of change of score ═══
    # When score is rising (momentum > 0): bull signal → widen TP
    # When score is falling (momentum < 0): weakening → tighten SL
    configs.append(Features(
        name="E1: Momentum entry SL(2-3)/TP(1.5-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="momentum",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
    ))

    # ═══ F: DURING-HOLD RATCHET (only tighten as market weakens) ═══
    # Uses EMA5, adjusts every day but can only tighten SL/TP
    configs.append(Features(
        name="F1: EMA5 hold-ratchet SL(2-3)/TP(1.5-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_hold_adjust=True, market_ratchet_only=True,
    ))
    configs.append(Features(
        name="F2: EMA10 hold-ratchet SL(2-3)/TP(1.5-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema10",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_hold_adjust=True, market_ratchet_only=True,
    ))
    # F3: Ratchet with SL-only (don't touch TP during hold)
    configs.append(Features(
        name="F3: EMA5 hold-ratchet SL only(2-3)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=2.0, market_tp_bull=2.0,  # TP fixed
        market_hold_adjust=True, market_ratchet_only=True,
    ))

    # ═══ G: DURING-HOLD FULL DYNAMIC (bidirectional) ═══
    configs.append(Features(
        name="G1: EMA5 hold-dynamic SL(2-3)/TP(1.5-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_hold_adjust=True, market_ratchet_only=False,
    ))
    configs.append(Features(
        name="G2: EMA10 hold-dynamic SL(2-3)/TP(1.5-2.5)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema10",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_hold_adjust=True, market_ratchet_only=False,
    ))

    # ═══ H: NIFTY VOL SCALING ═══
    # H1: Vol widen SL + tighten TP (classic vol targeting)
    configs.append(Features(
        name="H1: EMA5+Vol(widen SL,tight TP)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_vol_scale=True, market_vol_sl_mode="widen", market_vol_tp_mode="tighten",
    ))
    # H2: Vol tighten SL + widen TP (contrarian vol approach)
    configs.append(Features(
        name="H2: EMA5+Vol(tight SL,widen TP)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_vol_scale=True, market_vol_sl_mode="tighten", market_vol_tp_mode="widen",
    ))
    # H3: Vol scaling only (no score-based — pure vol targeting)
    configs.append(Features(
        name="H3: Vol-only (no score) widen SL, tight TP",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=3.0, market_sl_bull=3.0,  # score doesn't change SL
        market_tp_bear=2.0, market_tp_bull=2.0,  # score doesn't change TP
        market_vol_scale=True, market_vol_sl_mode="widen", market_vol_tp_mode="tighten",
    ))

    # ═══ I: HOLD-RATCHET + VOL (combo) ═══
    configs.append(Features(
        name="I1: EMA5+Vol+Ratchet",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
        market_hold_adjust=True, market_ratchet_only=True,
        market_vol_scale=True, market_vol_sl_mode="widen", market_vol_tp_mode="tighten",
    ))

    # ═══ J: BEST COMBOS — 30d hold variants ═══
    configs.append(Features(
        name="J1: EMA5 entry 30d SL(2-3)/TP(1.5-2.5)",
        max_positions=8, max_hold_days=30, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=2.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
    ))
    configs.append(Features(
        name="J2: EMA5 entry 30d TP-only(1.5-2.5)",
        max_positions=8, max_hold_days=30, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema5",
        market_sl_bear=3.0, market_sl_bull=3.0,
        market_tp_bear=1.5, market_tp_bull=2.5,
    ))

    # ═══ K: CONSERVATIVE — mild adjustments ═══
    configs.append(Features(
        name="K1: EMA10 mild SL(2.5-3)/TP(1.8-2.2)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema10",
        market_sl_bear=2.5, market_sl_bull=3.0,
        market_tp_bear=1.8, market_tp_bull=2.2,
    ))
    configs.append(Features(
        name="K2: EMA20 mild SL(2.5-3)/TP(1.8-2.2)",
        max_positions=8, max_hold_days=20, regime_position_limits=BASE_RPL,
        market_adaptive=True, market_score_key="ema20",
        market_sl_bear=2.5, market_sl_bull=3.0,
        market_tp_bear=1.8, market_tp_bull=2.2,
    ))

    print(f"Built {len(configs)} experiment configs")
    return configs


def extract_stats(name, res):
    """Extract consistent stats from simulate_portfolio result."""
    trades = res.get("_trades", [])
    total_pnl = sum(t["pnl"] for t in trades) if trades else 0
    monthly = defaultdict(float)
    for t in trades:
        monthly[t["exit_date"][:6]] += t["pnl"]
    return {
        "name": name,
        "n_trades": res.get("n_trades", 0),
        "pnl": round(total_pnl, 0),
        "wr": res.get("win_rate", 0),
        "pf": res.get("profit_factor", 0),
        "cagr": res.get("cagr_pct", 0),
        "mdd": res.get("max_drawdown_pct", 0),
        "calmar": res.get("calmar", 0),
        "worst_mo": round(min(monthly.values()), 0) if monthly else 0,
        "best_mo": round(max(monthly.values()), 0) if monthly else 0,
        "skipped": res.get("trades_skipped", 0),
    }


def run_dataset(label, configs, market_indicators, input_dir=None, min_date=None, max_date=None):
    trigger_files = load_trigger_files(input_dir=input_dir, min_date=min_date, max_date=max_date)
    picks_df = extract_picks(trigger_files)
    price_data = fetch_all_price_data(picks_df)
    regime_by_date = extract_regime_by_date(trigger_files)
    print(f"  {label}: {len(picks_df)} picks, {len(regime_by_date)} regime days, "
          f"{len(market_indicators)} indicator days")

    results = []
    for feat in configs:
        res = simulate_portfolio(picks_df, price_data, feat,
                                 regime_by_date=regime_by_date,
                                 market_indicators=market_indicators)
        stats = extract_stats(feat.name, res)
        print(f"  ✓ {feat.name:<45s} {stats['pnl']:>+12,.0f}")
        results.append(stats)
    return results


def print_results(label, results):
    results_sorted = sorted(results, key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'='*150}")
    print(f"  {label}")
    print(f"{'='*150}")
    print(f"{'#':<4} {'Config':<45} {'Trades':>7} {'P&L':>12} {'WR%':>6} {'PF':>6} "
          f"{'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Worst Mo':>11} {'Best Mo':>11} {'Skip':>6}")
    print("-" * 150)
    for i, r in enumerate(results_sorted, 1):
        marker = " ***" if i == 1 else ""
        pf = r["pf"] if isinstance(r["pf"], (int, float)) else 0
        print(f"{i:<4} {r['name']:<45} {r['n_trades']:>7} {r['pnl']:>+12,.0f} "
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

    print(f"\n{'='*145}")
    print(f"  CROSS-DATASET: CONFIGS SORTED BY AVG P&L (V1 + V2)")
    print(f"{'='*145}")
    print(f"{'Config':<45} {'V1 P&L':>12} {'V2 P&L':>12} {'Avg P&L':>12} "
          f"{'V1 MDD':>8} {'V2 MDD':>8} {'V1 Cal':>8} {'V2 Cal':>8}")
    print("-" * 145)
    for name, p1, p2, avg, mdd1, mdd2, cal1, cal2 in rows:
        print(f"{name:<45} {p1:>+12,.0f} {p2:>+12,.0f} {avg:>+12,.0f} "
              f"{mdd1:>7.1f}% {mdd2:>7.1f}% {cal1:>8.2f} {cal2:>8.2f}")


if __name__ == "__main__":
    configs = build_configs()

    # Fetch NIFTY 50 data for vol/ADX computation
    print("Fetching NIFTY 50 data for vol/ADX indicators...")
    nifty = yf.Ticker("^NSEI")
    nifty_hist = nifty.history(start="2023-01-01", end="2026-06-01")
    print(f"  NIFTY data: {len(nifty_hist)} days ({nifty_hist.index[0].date()} to {nifty_hist.index[-1].date()})")

    # ── V1 ──
    print(f"\n{'='*60}")
    print(f"  Loading V1 data...")
    print(f"{'='*60}")
    v1_trigger = load_trigger_files()
    v1_regime = extract_regime_by_date(v1_trigger)
    v1_mi = build_market_indicators(v1_regime, nifty_hist)
    print(f"  V1 market indicators: {len(v1_mi)} days")
    # Show sample indicator
    sample_dates = sorted(v1_mi.keys())[:3]
    for d in sample_dates:
        mi = v1_mi[d]
        vol_info = f", vol_ratio={mi.get('nifty_vol_ratio','N/A')}, adx={mi.get('nifty_adx','N/A')}" if 'nifty_vol_ratio' in mi else ""
        print(f"    {d}: raw={mi['raw']:+.0f} ema5={mi['ema5']:+.2f} ema10={mi['ema10']:+.2f} mom={mi['momentum']:+.2f}{vol_info}")

    v1_results = run_dataset("V1", configs, v1_mi)
    print_results("V1 DATA (Jan 2024 - May 2026)", v1_results)

    # ── V2 ──
    print(f"\n{'='*60}")
    print(f"  Loading V2 data...")
    print(f"{'='*60}")
    v2_dir = os.path.join(os.path.dirname(__file__), "..", "trigger_results_v2")
    v2_trigger = load_trigger_files(input_dir=v2_dir)
    v2_regime = extract_regime_by_date(v2_trigger)
    v2_mi = build_market_indicators(v2_regime, nifty_hist)
    print(f"  V2 market indicators: {len(v2_mi)} days")

    v2_results = run_dataset("V2", configs, v2_mi, input_dir=v2_dir)
    print_results("V2 DATA (Jan 2024 - Jul 2025)", v2_results)

    print_cross_compare(v1_results, v2_results)
