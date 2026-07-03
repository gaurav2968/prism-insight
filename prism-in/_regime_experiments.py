"""
Regime-based experiments — try many approaches to see what actually helps.
All use V1 data, 8-slot realistic sim, full costs.

Experiments:
1. Baseline (no regime adjustment)
2. Cut losers only on regime flip (keep winners)
3. Regime-based max positions (8 in bull, 4 in medium, 0 in bad)
4. Regime-based trigger filter (only Gap Up in weak regimes)
5. Half-size positions when regime weakening
6. Combo: cut losers + fewer positions + gap-only in weak
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from collections import defaultdict
from strategy_lab_v2 import (
    simulate_portfolio, Features, make_feature_set,
    load_trigger_files, extract_picks, fetch_all_price_data,
    extract_regime_by_date, compute_atr_at_date,
    ROUND_TRIP_COST_PCT, REGIME_SCORE_MAP, Position,
)
from backtest_engine import load_trigger_files as bt_load_trigger_files
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def regime_simulate(
    picks_df, price_data, regime_by_date, capital=1_000_000,
    max_positions=8, max_hold_days=30,
    # Experiment flags
    cut_losers_on_correction=False,     # Exp 2: close underwater positions on regime flip
    regime_position_limits=None,        # Exp 3: {regime: max_pos} override
    gap_only_in_weak=False,             # Exp 4: only Gap Up triggers when regime < bull_strong
    half_size_in_weak=False,            # Exp 5: 50% allocation when regime not bull_strong
    label="",
):
    """Custom simulation with regime-based adjustments."""
    
    all_dates = set()
    for hist in price_data.values():
        all_dates.update(hist.index.tolist())
    trading_days = sorted(all_dates)

    picks_by_date = {}
    for _, row in picks_df.iterrows():
        d = row["trade_date"]
        if d not in picks_by_date:
            picks_by_date[d] = []
        picks_by_date[d].append(row)

    open_positions = []
    completed_trades = []
    daily_equity = []
    cumulative_pnl = 0.0
    available_cash = capital
    trades_skipped = 0
    regime_exits = 0

    for day in trading_days:
        day_str = day.strftime("%Y%m%d")
        day_regime = regime_by_date.get(day_str, "bull_medium")
        regime_score = REGIME_SCORE_MAP.get(day_regime, 0)

        # ── Exp 2: Cut losers on regime correction ──
        if cut_losers_on_correction and regime_score < 0 and open_positions:
            surviving = []
            for pos in open_positions:
                if pos.ticker not in price_data:
                    surviving.append(pos)
                    continue
                hist = price_data[pos.ticker]
                day_data = hist[hist.index == day]
                if day_data.empty:
                    surviving.append(pos)
                    continue
                curr_price = float(day_data["Close"].iloc[0])
                # If position is underwater → close it
                if curr_price < pos.entry_price:
                    gross_ret = (curr_price / pos.entry_price - 1) * 100
                    net_ret = gross_ret - ROUND_TRIP_COST_PCT * 100
                    pnl = pos.alloc_capital * net_ret / 100
                    completed_trades.append({
                        "ticker": pos.ticker, "entry_date": pos.entry_date.strftime("%Y%m%d"),
                        "exit_date": day_str, "entry_price": pos.entry_price,
                        "exit_price": curr_price, "exit_reason": "regime_cut_loser",
                        "days_held": pos.day_count, "gross_return_pct": round(gross_ret, 2),
                        "net_return_pct": round(net_ret, 2), "pnl": round(pnl, 0),
                        "trigger_type": pos.trigger_type, "entry_atr": round(pos.entry_atr, 2),
                        "alloc_capital": round(pos.alloc_capital, 0),
                    })
                    cumulative_pnl += pnl
                    available_cash += pos.alloc_capital + pnl
                    regime_exits += 1
                else:
                    surviving.append(pos)  # keep winners
            open_positions = surviving

        # ── Phase 1: Check normal exits ──
        closed_today = []
        for pos in open_positions:
            pos.day_count += 1
            if pos.ticker not in price_data:
                continue
            hist = price_data[pos.ticker]
            day_data = hist[hist.index == day]
            if day_data.empty:
                continue

            price_high = float(day_data["High"].iloc[0])
            price_low = float(day_data["Low"].iloc[0])
            price_close = float(day_data["Close"].iloc[0])

            exit_price = None
            exit_reason = ""

            # Stop loss
            if price_low <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "stop_loss"
            # Take profit
            elif pos.take_profit > 0 and price_high >= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "take_profit"
            # Time exit
            elif day >= pos.max_exit_date:
                exit_price = price_close
                exit_reason = "time_exit"

            if exit_price is not None:
                gross_ret = (exit_price / pos.entry_price - 1) * 100
                net_ret = gross_ret - ROUND_TRIP_COST_PCT * 100
                pnl = pos.alloc_capital * net_ret / 100
                completed_trades.append({
                    "ticker": pos.ticker, "entry_date": pos.entry_date.strftime("%Y%m%d"),
                    "exit_date": day_str, "entry_price": pos.entry_price,
                    "exit_price": exit_price, "exit_reason": exit_reason,
                    "days_held": pos.day_count, "gross_return_pct": round(gross_ret, 2),
                    "net_return_pct": round(net_ret, 2), "pnl": round(pnl, 0),
                    "trigger_type": pos.trigger_type, "entry_atr": round(pos.entry_atr, 2),
                    "alloc_capital": round(pos.alloc_capital, 0),
                })
                cumulative_pnl += pnl
                available_cash += pos.alloc_capital + pnl
                closed_today.append(pos)

        for pos in closed_today:
            open_positions.remove(pos)

        # ── Phase 2: Open new positions ──
        day_picks = picks_by_date.get(day_str, [])
        if day_picks:
            # Exp 3: regime-based position limit
            if regime_position_limits:
                day_max = regime_position_limits.get(day_regime, max_positions)
            else:
                day_max = max_positions

            slots_available = day_max - len(open_positions)
            if slots_available <= 0:
                trades_skipped += len(day_picks)
            else:
                # Exp 4: filter to Gap Up only in weak regimes
                if gap_only_in_weak and regime_score < 5:  # anything below bull_strong
                    day_picks = [p for p in day_picks if "Gap" in p.get("trigger_type", "")]
                    if not day_picks:
                        continue

                # Sort by quality
                day_picks = sorted(day_picks, key=lambda p: p.get("quality_score", 0), reverse=True)
                day_picks = day_picks[:slots_available]

                if not day_picks:
                    continue

                # Exp 5: half size in non-bull_strong
                if half_size_in_weak and regime_score < 5:
                    size_mult = 0.5
                else:
                    size_mult = 1.0

                alloc_per = available_cash / len(day_picks) * size_mult
                if alloc_per < 10000:
                    trades_skipped += len(day_picks)
                    continue

                for pick in day_picks:
                    ticker = pick["ticker"]
                    entry_price = pick.get("entry_price", 0)
                    if entry_price <= 0 or ticker not in price_data:
                        continue

                    hist = price_data[ticker]
                    atr = compute_atr_at_date(hist, day, 14)
                    if atr <= 0:
                        atr = entry_price * 0.03

                    sl = entry_price - 3.0 * atr
                    tp = entry_price + 2.0 * atr

                    alloc = min(alloc_per, available_cash)
                    if alloc < 10000:
                        trades_skipped += 1
                        continue

                    max_exit_approx = day
                    td_count = 0
                    while td_count < max_hold_days:
                        max_exit_approx += pd.Timedelta(days=1)
                        if max_exit_approx.weekday() < 5:
                            td_count += 1

                    pos = Position(
                        ticker=ticker, entry_price=entry_price, entry_date=day,
                        entry_atr=atr, trigger_type=pick.get("trigger_type", ""),
                        stop_loss=sl, take_profit=tp, max_exit_date=max_exit_approx,
                        highest_close=entry_price, alloc_capital=alloc,
                        initial_risk_r=3.0 * atr,
                        sar_value=sl, sar_af=0.02, sar_ep=entry_price,
                        remaining_alloc=alloc,
                    )
                    open_positions.append(pos)
                    available_cash -= alloc

        # Record equity
        unrealized = 0
        for pos in open_positions:
            if pos.ticker in price_data:
                hist = price_data[pos.ticker]
                day_data = hist[hist.index == day]
                if not day_data.empty:
                    curr_price = float(day_data["Close"].iloc[0])
                    unrealized += pos.alloc_capital * ((curr_price / pos.entry_price - 1) - ROUND_TRIP_COST_PCT)
        deployed = sum(p.alloc_capital for p in open_positions)
        daily_equity.append({
            "date": day_str,
            "equity": available_cash + deployed + unrealized,
        })

    # Results
    if not completed_trades:
        return {"label": label, "n_trades": 0, "pnl": 0}

    df = pd.DataFrame(completed_trades)
    eq_df = pd.DataFrame(daily_equity)
    eq_series = eq_df.set_index("date")["equity"]
    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / peak * 100
    max_dd = drawdown.min()
    final_eq = eq_series.iloc[-1]
    n_days = (pd.Timestamp(eq_df["date"].iloc[-1]) - pd.Timestamp(eq_df["date"].iloc[0])).days
    cagr = ((final_eq / capital) ** (365 / n_days) - 1) * 100 if n_days > 0 and final_eq > 0 else -100

    wins = df[df["net_return_pct"] > 0]
    losses = df[df["net_return_pct"] <= 0]
    total_pnl = df["pnl"].sum()
    wr = len(wins) / len(df) * 100
    avg_ret = df["net_return_pct"].mean()
    gw = wins["net_return_pct"].sum() if len(wins) > 0 else 0
    gl = abs(losses["net_return_pct"].sum()) if len(losses) > 0 else 0
    pf = gw / gl if gl > 0 else 99

    return {
        "label": label,
        "n_trades": len(df),
        "pnl": round(total_pnl, 0),
        "final_equity": round(final_eq, 0),
        "wr": round(wr, 1),
        "avg_ret": round(avg_ret, 2),
        "pf": round(pf, 2),
        "cagr": round(cagr, 1),
        "max_dd": round(max_dd, 1),
        "calmar": round(cagr / abs(max_dd), 2) if max_dd != 0 else 0,
        "trades_skipped": trades_skipped,
        "regime_exits": regime_exits,
        "_trades": completed_trades,
    }


def print_summary(results):
    print(f"\n{'='*120}")
    print(f"{'Config':<45} {'Trades':>7} {'P&L':>12} {'WR%':>6} {'PF':>6} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Equity':>12} {'RegExit':>8}")
    print(f"-"*120)
    for r in results:
        print(f"{r['label']:<45} {r['n_trades']:>7} {r['pnl']:>+12,.0f} {r['wr']:>5.1f}% {r['pf']:>6.2f} {r['cagr']:>6.1f}% {r['max_dd']:>6.1f}% {r['calmar']:>7.2f} {r['final_equity']:>12,.0f} {r.get('regime_exits',0):>8}")
    print(f"{'='*120}")


def print_monthly(label, trades):
    monthly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        m = t["exit_date"][:6]
        monthly[m]["pnl"] += t["pnl"]
        monthly[m]["trades"] += 1
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1

    print(f"\n  {label} monthly:")
    cum = 0
    for m in sorted(monthly.keys()):
        d = monthly[m]
        cum += d["pnl"]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"    {m}: {d['pnl']:>+10,.0f}  ({d['trades']:>3} trades, {wr:.0f}% WR)  cum={cum:>+12,.0f}")


def main():
    trigger_files = load_trigger_files(input_dir=None)
    picks_df = extract_picks(trigger_files)
    price_data = fetch_all_price_data(picks_df)
    regime_by_date = extract_regime_by_date(trigger_files)
    
    print(f"Loaded {len(picks_df)} picks, {len(regime_by_date)} regime days")

    experiments = []

    # 1. Baseline
    r = regime_simulate(picks_df, price_data, regime_by_date, label="1. Baseline (no regime adj)")
    experiments.append(r)

    # 2. Cut losers on correction
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       cut_losers_on_correction=True,
                       label="2. Cut losers on correction")
    experiments.append(r)

    # 3. Regime-based position limits
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       regime_position_limits={
                           "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
                           "slope_declining": 0, "correction": 0,
                           "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
                       },
                       label="3. Regime position limits (8/5/0)")
    experiments.append(r)

    # 4. Gap Up only in non-bull_strong
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       gap_only_in_weak=True,
                       label="4. Gap Up only (non-bull_strong)")
    experiments.append(r)

    # 5. Half size in non-bull_strong
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       half_size_in_weak=True,
                       label="5. Half size in weak regime")
    experiments.append(r)

    # 6. Combo: cut losers + position limits
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       cut_losers_on_correction=True,
                       regime_position_limits={
                           "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
                           "slope_declining": 0, "correction": 0,
                           "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
                       },
                       label="6. Cut losers + position limits")
    experiments.append(r)

    # 7. Combo: cut losers + gap only
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       cut_losers_on_correction=True,
                       gap_only_in_weak=True,
                       label="7. Cut losers + Gap only in weak")
    experiments.append(r)

    # 8. Combo: position limits + gap only
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       regime_position_limits={
                           "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
                           "slope_declining": 0, "correction": 0,
                           "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
                       },
                       gap_only_in_weak=True,
                       label="8. Position limits + Gap only")
    experiments.append(r)

    # 9. Full combo: all three
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       cut_losers_on_correction=True,
                       regime_position_limits={
                           "bull_strong": 8, "bull_medium": 5, "bull_weak": 0,
                           "slope_declining": 0, "correction": 0,
                           "bear": 0, "bear_bottom": 3, "bear_bottom_extreme": 3,
                       },
                       gap_only_in_weak=True,
                       label="9. FULL: cut losers+limits+gap only")
    experiments.append(r)

    # 10. Cut losers + half size
    r = regime_simulate(picks_df, price_data, regime_by_date,
                       cut_losers_on_correction=True,
                       half_size_in_weak=True,
                       label="10. Cut losers + half size")
    experiments.append(r)

    print_summary(experiments)

    # Print monthly for top 3
    sorted_exp = sorted(experiments, key=lambda x: x.get("calmar", 0), reverse=True)
    for r in sorted_exp[:3]:
        print_monthly(r["label"], r["_trades"])


if __name__ == "__main__":
    from backtest_engine import load_trigger_files, extract_picks
    main()
