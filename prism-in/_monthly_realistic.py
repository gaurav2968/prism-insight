"""Monthly PnL — V1 side-by-side: Baseline vs Best Config, with regime context."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict, Counter
from strategy_lab_v2 import (simulate_portfolio, Features, 
                              load_trigger_files, extract_picks, fetch_all_price_data,
                              extract_regime_by_date)
import logging
logging.basicConfig(level=logging.INFO)


def build_monthly(res, regime_by_date=None):
    """Build monthly stats dict from simulation result."""
    trades = res.get("_trades", [])
    monthly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        m = t["exit_date"][:6]
        monthly[m]["pnl"] += t["pnl"]
        monthly[m]["trades"] += 1
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1
    return monthly


def month_regime_summary(regime_by_date):
    """For each YYYYMM, count regime days and return dominant regime + day counts."""
    monthly_regimes = defaultdict(lambda: Counter())
    if not regime_by_date:
        return {}
    for date_str, rtype in regime_by_date.items():
        m = date_str[:6]
        monthly_regimes[m][rtype] += 1
    result = {}
    for m, counter in monthly_regimes.items():
        dominant = counter.most_common(1)[0][0] if counter else "?"
        total = sum(counter.values())
        # Build short summary: e.g. "bull_strong(15) correction(5)"
        parts = []
        for rtype, cnt in counter.most_common(3):
            short = rtype.replace("bull_", "b_").replace("bear_bottom", "bb").replace("slope_declining", "slope↓").replace("correction", "corr").replace("_extreme", "X")
            parts.append(f"{short}({cnt})")
        result[m] = {"dominant": dominant, "summary": " ".join(parts), "days": total}
    return result


def print_comparison(label_a, res_a, label_b, res_b, regime_by_date=None):
    """Print side-by-side monthly comparison of two configs with regime info."""
    ma = build_monthly(res_a)
    mb = build_monthly(res_b)
    regime_months = month_regime_summary(regime_by_date)

    all_months = sorted(set(list(ma.keys()) + list(mb.keys())))

    print(f"\n{'='*160}")
    print(f"  V1 MONTHLY COMPARISON: [{label_a}] vs [{label_b}]")
    print(f"{'='*160}")
    header = (f"{'Month':<8} {'Regime (dominant days)':<30} "
              f"{'':>2}{'--- '+label_a+' ---':^30}{'':>2}"
              f"{'--- '+label_b+' ---':^30}{'':>2}"
              f"{'Delta':>10} {'Cum Delta':>10}")
    print(header)
    sub = (f"{'':8} {'':30} "
           f"{'P&L':>10} {'Trades':>7} {'WR%':>6} {'CumPnL':>10}  "
           f"{'P&L':>10} {'Trades':>7} {'WR%':>6} {'CumPnL':>10}  "
           f"{'':>10} {'':>10}")
    print(sub)
    print("-" * 160)

    cum_a = cum_b = cum_delta = 0
    total_a_pnl = total_b_pnl = 0
    total_a_trades = total_b_trades = 0
    total_a_wins = total_b_wins = 0

    bull_months_a = bear_months_a = 0
    bull_months_b = bear_months_b = 0

    for m in all_months:
        da = ma.get(m, {"pnl": 0, "trades": 0, "wins": 0})
        db = mb.get(m, {"pnl": 0, "trades": 0, "wins": 0})
        
        cum_a += da["pnl"]
        cum_b += db["pnl"]
        delta = db["pnl"] - da["pnl"]
        cum_delta += delta

        total_a_pnl += da["pnl"]
        total_b_pnl += db["pnl"]
        total_a_trades += da["trades"]
        total_b_trades += db["trades"]
        total_a_wins += da["wins"]
        total_b_wins += db["wins"]

        wr_a = da["wins"] / da["trades"] * 100 if da["trades"] else 0
        wr_b = db["wins"] / db["trades"] * 100 if db["trades"] else 0

        rm = regime_months.get(m, {"summary": "no data", "dominant": "?"})
        regime_str = rm["summary"][:28]
        dom = rm["dominant"]

        # Classify month as bull/bear for summary
        if "bull" in dom:
            bull_months_a += da["pnl"]
            bull_months_b += db["pnl"]
        elif dom in ("correction", "bear", "bear_bottom", "bear_bottom_extreme", "slope_declining"):
            bear_months_a += da["pnl"]
            bear_months_b += db["pnl"]

        # Highlight: arrow shows if B improved vs A
        marker = "▲" if delta > 5000 else ("▼" if delta < -5000 else " ")

        print(f"{m:<8} {regime_str:<30} "
              f"{da['pnl']:>+10,.0f} {da['trades']:>7} {wr_a:>5.1f}% {cum_a:>+10,.0f}  "
              f"{db['pnl']:>+10,.0f} {db['trades']:>7} {wr_b:>5.1f}% {cum_b:>+10,.0f}  "
              f"{delta:>+10,.0f} {cum_delta:>+10,.0f} {marker}")

    print("-" * 160)
    wr_ta = total_a_wins / total_a_trades * 100 if total_a_trades else 0
    wr_tb = total_b_wins / total_b_trades * 100 if total_b_trades else 0
    total_delta = total_b_pnl - total_a_pnl
    print(f"{'TOTAL':<8} {'':30} "
          f"{total_a_pnl:>+10,.0f} {total_a_trades:>7} {wr_ta:>5.1f}% {'':>10}  "
          f"{total_b_pnl:>+10,.0f} {total_b_trades:>7} {wr_tb:>5.1f}% {'':>10}  "
          f"{total_delta:>+10,.0f}")

    print(f"\n  REGIME BREAKDOWN:")
    print(f"    Bull months P&L:  {label_a}: {bull_months_a:>+10,.0f}  |  {label_b}: {bull_months_b:>+10,.0f}  |  Delta: {bull_months_b - bull_months_a:>+10,.0f}")
    print(f"    Bear months P&L:  {label_a}: {bear_months_a:>+10,.0f}  |  {label_b}: {bear_months_b:>+10,.0f}  |  Delta: {bear_months_b - bear_months_a:>+10,.0f}")
    print(f"    (Bull = bull_strong/medium/weak dominant | Bear = correction/bear/slope_declining dominant)")

    # Summary stats
    for lbl, r in [(label_a, res_a), (label_b, res_b)]:
        print(f"\n  {lbl}: P&L={r.get('total_pnl',0):>+10,.0f} | MDD={r.get('max_drawdown_pct',0):.1f}% | Trades={r.get('total_trades',0)} | Skipped={r.get('trades_skipped',0)}")


def print_expanded_target_analysis(res_baseline, res_best, regime_by_date):
    """Analyze: should we expand TP in bull months? Show avg win/loss by regime."""
    print(f"\n{'='*120}")
    print(f"  BULL vs BEAR TRADE ANALYSIS — Should we expand targets in bull / tighten in bear?")
    print(f"{'='*120}")

    for lbl, res in [("Baseline", res_baseline), ("8/5+Hold20d", res_best)]:
        trades = res.get("_trades", [])
        if not trades:
            continue

        # Classify each trade by the regime on its entry date
        regime_trades = defaultdict(lambda: {"wins": [], "losses": [], "all": []})
        for t in trades:
            entry_date = t["entry_date"]
            regime = regime_by_date.get(entry_date, "unknown")
            # Simplify regime
            if "bull_strong" in regime:
                bucket = "BULL_STRONG"
            elif "bull_medium" in regime:
                bucket = "BULL_MEDIUM"
            elif "bull_weak" in regime:
                bucket = "BULL_WEAK"
            elif regime in ("correction", "slope_declining"):
                bucket = "CORRECTION"
            elif "bear" in regime:
                bucket = "BEAR"
            else:
                bucket = "OTHER"

            regime_trades[bucket]["all"].append(t["pnl"])
            if t["pnl"] > 0:
                regime_trades[bucket]["wins"].append(t["pnl"])
            else:
                regime_trades[bucket]["losses"].append(t["pnl"])

        print(f"\n  [{lbl}] — Per-regime trade stats (entry-day regime)")
        print(f"  {'Regime':<15} {'Trades':>7} {'WR%':>6} {'Avg Win':>10} {'Avg Loss':>10} {'Total P&L':>12} {'Avg P&L':>10} {'Max Win':>10} {'Max Loss':>10}")
        print(f"  {'-'*105}")

        for bucket in ["BULL_STRONG", "BULL_MEDIUM", "BULL_WEAK", "CORRECTION", "BEAR", "OTHER"]:
            d = regime_trades.get(bucket)
            if not d or not d["all"]:
                continue
            n = len(d["all"])
            wins = d["wins"]
            losses = d["losses"]
            wr = len(wins) / n * 100 if n else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            total = sum(d["all"])
            avg_pnl = total / n
            max_win = max(wins) if wins else 0
            max_loss = min(losses) if losses else 0
            print(f"  {bucket:<15} {n:>7} {wr:>5.1f}% {avg_win:>+10,.0f} {avg_loss:>+10,.0f} {total:>+12,.0f} {avg_pnl:>+10,.0f} {max_win:>+10,.0f} {max_loss:>+10,.0f}")

    # Analysis: how many trades hit TP vs SL vs time-exit, by regime
    print(f"\n  EXIT REASON BY REGIME (Baseline)")
    trades = res_baseline.get("_trades", [])
    regime_exits = defaultdict(lambda: Counter())
    for t in trades:
        entry_date = t["entry_date"]
        regime = regime_by_date.get(entry_date, "unknown")
        if "bull_strong" in regime:
            bucket = "BULL_STRONG"
        elif "bull_medium" in regime:
            bucket = "BULL_MEDIUM"
        else:
            bucket = "OTHER"
        regime_exits[bucket][t.get("exit_reason", "?")] += 1

    for bucket in ["BULL_STRONG", "BULL_MEDIUM", "OTHER"]:
        if bucket not in regime_exits:
            continue
        counts = regime_exits[bucket]
        total = sum(counts.values())
        print(f"  {bucket}: ", end="")
        for reason, cnt in counts.most_common():
            print(f"{reason}={cnt}({cnt/total*100:.0f}%) ", end="")
        print()


if __name__ == "__main__":
    print("Loading V1 data...")
    trigger_files = load_trigger_files()
    picks_df = extract_picks(trigger_files)
    price_data = fetch_all_price_data(picks_df)
    regime_by_date = extract_regime_by_date(trigger_files)
    print(f"V1: {len(picks_df)} picks, {len(regime_by_date)} regime days")

    # --- Config A: Baseline (8 slots, 30d hold, SL=3x, TP=2x) ---
    feat_baseline = Features(name="Baseline", max_positions=8)
    res_baseline = simulate_portfolio(picks_df, price_data, feat_baseline, regime_by_date=regime_by_date)

    # --- Config B: Winner = 8/5 + Hold 20d ---
    feat_best = Features(
        name="8/5+Hold20d", max_positions=8, max_hold_days=20,
        regime_position_limits={
            "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
            "slope_declining": 0, "correction": 0,
            "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
        }
    )
    res_best = simulate_portfolio(picks_df, price_data, feat_best, regime_by_date=regime_by_date)

    # --- Config C: Expand TP in bull = SL3x TP3x + 8/5 ---
    feat_wide_tp = Features(
        name="8/5+WideTP3x", max_positions=8,
        initial_tp_mult=3.0,
        regime_position_limits={
            "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
            "slope_declining": 0, "correction": 0,
            "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
        }
    )
    res_wide_tp = simulate_portfolio(picks_df, price_data, feat_wide_tp, regime_by_date=regime_by_date)

    # --- Config D: Hold 20d + Expand TP=3x in bull ---
    feat_hold20_tp3 = Features(
        name="8/5+Hold20d+TP3x", max_positions=8, max_hold_days=20,
        initial_tp_mult=3.0,
        regime_position_limits={
            "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
            "slope_declining": 0, "correction": 0,
            "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
        }
    )
    res_hold20_tp3 = simulate_portfolio(picks_df, price_data, feat_hold20_tp3, regime_by_date=regime_by_date)

    # --- Print comparisons ---
    print_comparison("Baseline (8/30d/SL3/TP2)", res_baseline,
                     "8/5+Hold20d", res_best,
                     regime_by_date)

    print_comparison("Baseline (8/30d/SL3/TP2)", res_baseline,
                     "8/5+WideTP3x (30d)", res_wide_tp,
                     regime_by_date)

    print_comparison("8/5+Hold20d", res_best,
                     "8/5+Hold20d+TP3x", res_hold20_tp3,
                     regime_by_date)

    # --- Bull/bear trade analysis ---
    print_expanded_target_analysis(res_baseline, res_best, regime_by_date)
