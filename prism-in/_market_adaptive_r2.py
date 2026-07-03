"""
Round 2: Focused experiments on top performers from Round 1.
Winners to explore deeper:
  #1 F3: EMA5 hold-ratchet SL only — vary SL range, EMA window, hold period
  #2 F1: EMA5 hold-ratchet SL+TP — vary TP range
  #3 EMA5/SMA5 entry-only — vary SL/TP ranges
  + Combos: best entry scoring + best hold adjustment
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

RPL = {
    "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
    "slope_declining": 0, "correction": 0,
    "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
}

def F(name, **kw):
    """Shorthand to build Features with position limits + market_adaptive."""
    defaults = dict(max_positions=8, max_hold_days=20, regime_position_limits=RPL,
                    market_adaptive=True)
    defaults.update(kw)
    return Features(name=name, **defaults)

def build_configs():
    c = []

    # ═══ REF: baselines ═══
    c.append(Features(name="REF: Baseline", max_positions=8))
    c.append(Features(name="REF: 8/5+Hold20d", max_positions=8, max_hold_days=20,
                      regime_position_limits=RPL))

    # ═══ A: RATCHET SL ONLY — vary SL range (winner from R1) ═══
    # Original: SL bear=2.0, bull=3.0
    for sl_lo, sl_hi in [(1.5, 3.0), (2.0, 3.0), (2.0, 3.5), (2.5, 3.0), (2.5, 3.5), (1.5, 3.5)]:
        c.append(F(f"A: Ratch SL({sl_lo}-{sl_hi})",
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=sl_lo, market_sl_bull=sl_hi,
                   market_tp_bear=2.0, market_tp_bull=2.0))  # TP fixed at 2x

    # ═══ B: RATCHET SL — vary EMA window ═══
    for key in ["ema3", "ema5", "sma5", "ema10"]:
        c.append(F(f"B: Ratch {key} SL(2-3)",
                   market_score_key=key, market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=2.0, market_tp_bull=2.0))

    # ═══ C: RATCHET SL — vary hold period ═══
    for hold in [15, 20, 25, 30]:
        c.append(F(f"C: Ratch SL(2-3) hold={hold}d",
                   max_hold_days=hold,
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=2.0, market_tp_bull=2.0))

    # ═══ D: RATCHET SL + TP — vary TP range ═══
    for tp_lo, tp_hi in [(1.5, 2.0), (1.5, 2.5), (1.5, 3.0), (2.0, 2.5), (2.0, 3.0), (1.8, 2.2)]:
        c.append(F(f"D: Ratch SL(2-3)/TP({tp_lo}-{tp_hi})",
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=tp_lo, market_tp_bull=tp_hi))

    # ═══ E: ENTRY-ONLY (no hold adjust) — vary SL/TP ranges ═══
    for sl_lo, sl_hi, tp_lo, tp_hi in [
        (2.0, 3.0, 1.5, 2.5),   # original R1 winner
        (2.0, 3.0, 1.5, 2.0),   # tighter TP range
        (2.0, 3.0, 2.0, 2.5),   # TP only widens
        (2.0, 3.0, 1.8, 2.2),   # mild TP
        (2.5, 3.0, 1.5, 2.5),   # SL doesn't go below 2.5
        (2.0, 3.5, 1.5, 2.5),   # SL can go to 3.5 in bull
    ]:
        c.append(F(f"E: Entry EMA5 SL({sl_lo}-{sl_hi})/TP({tp_lo}-{tp_hi})",
                   market_score_key="ema5",
                   market_sl_bear=sl_lo, market_sl_bull=sl_hi,
                   market_tp_bear=tp_lo, market_tp_bull=tp_hi,
                   market_hold_adjust=False))

    # ═══ F: ENTRY SMA5 — vary ranges ═══
    for sl_lo, sl_hi, tp_lo, tp_hi in [
        (2.0, 3.0, 1.5, 2.5),
        (2.0, 3.0, 1.5, 2.0),
        (2.0, 3.0, 2.0, 2.5),
        (2.5, 3.0, 1.5, 2.5),
    ]:
        c.append(F(f"F: Entry SMA5 SL({sl_lo}-{sl_hi})/TP({tp_lo}-{tp_hi})",
                   market_score_key="sma5",
                   market_sl_bear=sl_lo, market_sl_bull=sl_hi,
                   market_tp_bear=tp_lo, market_tp_bull=tp_hi,
                   market_hold_adjust=False))

    # ═══ G: RATCHET SL(2-3) + different fixed TP ═══
    for tp_fixed in [1.5, 1.8, 2.0, 2.2, 2.5]:
        c.append(F(f"G: Ratch SL(2-3) TP={tp_fixed}x fixed",
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=tp_fixed, market_tp_bull=tp_fixed))

    # ═══ H: BEST COMBOS — ratchet SL + entry TP scaling ═══
    # Use ratchet for SL during hold, but set TP at entry based on score
    for tp_lo, tp_hi in [(1.5, 2.5), (1.5, 2.0), (2.0, 2.5), (1.8, 2.2)]:
        c.append(F(f"H: Ratch SL(2-3) + Entry TP({tp_lo}-{tp_hi})",
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=tp_lo, market_tp_bull=tp_hi))

    # ═══ I: POSITION LIMIT VARIANTS with best adaptive ═══
    for bs, bm, label in [(8, 5, "8/5"), (8, 4, "8/4"), (6, 3, "6/3"), (10, 5, "10/5"), (8, 6, "8/6")]:
        rpl = {"bull_strong": bs, "bull_medium": bm, "bull_weak": 0,
               "slope_declining": 0, "correction": 0,
               "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3}
        c.append(F(f"I: Ratch SL(2-3) pos {label}",
                   regime_position_limits=rpl,
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=2.0, market_tp_bull=2.0))

    # ═══ J: HOLD PERIOD + RATCHET combos ═══
    for hold, tp in [(15, 2.0), (20, 2.0), (20, 2.5), (25, 2.0), (30, 2.0), (30, 2.5)]:
        c.append(F(f"J: Ratch SL(2-3) hold={hold} TP={tp}",
                   max_hold_days=hold,
                   market_score_key="ema5", market_hold_adjust=True, market_ratchet_only=True,
                   market_sl_bear=2.0, market_sl_bull=3.0,
                   market_tp_bear=tp, market_tp_bull=tp))

    print(f"Built {len(c)} configs")
    return c


def extract_stats(name, res):
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


def run_dataset(label, configs, market_indicators, input_dir=None):
    trigger_files = load_trigger_files(input_dir=input_dir)
    picks_df = extract_picks(trigger_files)
    price_data = fetch_all_price_data(picks_df)
    regime_by_date = extract_regime_by_date(trigger_files)
    print(f"  {label}: {len(picks_df)} picks, {len(regime_by_date)} regime days")

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
    rs = sorted(results, key=lambda r: r["pnl"], reverse=True)
    print(f"\n{'='*155}")
    print(f"  {label}")
    print(f"{'='*155}")
    print(f"{'#':<4} {'Config':<48} {'Trades':>6} {'P&L':>12} {'WR%':>6} {'PF':>6} "
          f"{'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Worst Mo':>11} {'Best Mo':>11} {'Skip':>5}")
    print("-" * 155)
    for i, r in enumerate(rs, 1):
        pf = r["pf"] if isinstance(r["pf"], (int, float)) else 0
        m = " ***" if i <= 3 else ""
        print(f"{i:<4} {r['name']:<48} {r['n_trades']:>6} {r['pnl']:>+12,.0f} "
              f"{r['wr']:>5.1f}% {pf:>5.2f} {r['cagr']:>6.1f}% {r['mdd']:>6.1f}% "
              f"{r['calmar']:>7.2f} {r['worst_mo']:>+11,.0f} {r['best_mo']:>+11,.0f} {r['skipped']:>5}{m}")


def print_cross(v1r, v2r):
    v1m = {r["name"]: r for r in v1r}
    v2m = {r["name"]: r for r in v2r}
    names = list(dict.fromkeys([r["name"] for r in v1r]))
    rows = []
    for n in names:
        r1, r2 = v1m.get(n, {}), v2m.get(n, {})
        p1, p2 = r1.get("pnl", 0), r2.get("pnl", 0)
        rows.append((n, p1, p2, (p1+p2)/2, r1.get("mdd",0), r2.get("mdd",0),
                      r1.get("calmar",0), r2.get("calmar",0),
                      r1.get("wr",0), r2.get("wr",0)))
    rows.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*160}")
    print(f"  TOP 30 BY AVG P&L (V1 + V2)")
    print(f"{'='*160}")
    print(f"{'#':<3} {'Config':<48} {'V1 P&L':>11} {'V2 P&L':>11} {'Avg P&L':>11} "
          f"{'V1 MDD':>7} {'V2 MDD':>7} {'V1 Cal':>7} {'V2 Cal':>7} {'V1 WR':>6} {'V2 WR':>6}")
    print("-" * 160)
    for i, (n, p1, p2, avg, m1, m2, c1, c2, w1, w2) in enumerate(rows[:30], 1):
        mark = " <<<" if i <= 5 else ""
        print(f"{i:<3} {n:<48} {p1:>+11,.0f} {p2:>+11,.0f} {avg:>+11,.0f} "
              f"{m1:>6.1f}% {m2:>6.1f}% {c1:>7.2f} {c2:>7.2f} {w1:>5.1f}% {w2:>5.1f}%{mark}")


if __name__ == "__main__":
    configs = build_configs()

    print("Fetching NIFTY 50 data...")
    nifty_hist = yf.Ticker("^NSEI").history(start="2023-01-01", end="2026-06-01")
    print(f"  NIFTY: {len(nifty_hist)} days")

    # V1
    print(f"\n{'='*60}\n  V1 data...\n{'='*60}")
    v1_tf = load_trigger_files()
    v1_mi = build_market_indicators(extract_regime_by_date(v1_tf), nifty_hist)
    v1r = run_dataset("V1", configs, v1_mi)
    print_results("V1 (Jan 2024 - May 2026)", v1r)

    # V2
    print(f"\n{'='*60}\n  V2 data...\n{'='*60}")
    v2_dir = os.path.join(os.path.dirname(__file__), "..", "trigger_results_v2")
    v2_tf = load_trigger_files(input_dir=v2_dir)
    v2_mi = build_market_indicators(extract_regime_by_date(v2_tf), nifty_hist)
    v2r = run_dataset("V2", configs, v2_mi, input_dir=v2_dir)
    print_results("V2 (Jan 2024 - Jul 2025)", v2r)

    print_cross(v1r, v2r)
