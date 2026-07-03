#!/usr/bin/env python3
"""
PRISM-INSIGHT Portfolio Dashboard v4
KR-style · Dark/Light toggle · Tabs: Dashboard | Trades | Watchlist
"""

import streamlit as st
import sqlite3
import json
import glob
from html import escape as _esc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "stock_tracking_db.sqlite"

st.set_page_config(page_title="PRISM-INSIGHT", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
# THEMES
# ══════════════════════════════════════════════════════════════

DARK = {
    "bg": "#0d1117", "surface": "#161b22", "surface2": "#1c2333",
    "border": "#30363d", "border_light": "#21262d",
    "text": "#e6edf3", "text2": "#8b949e", "text3": "#484f58",
    "green": "#3fb950", "green_bg": "rgba(63,185,80,0.12)",
    "red": "#f85149", "red_bg": "rgba(248,81,73,0.12)",
    "yellow": "#d29922", "yellow_bg": "rgba(210,153,34,0.12)",
    "blue": "#58a6ff", "blue_bg": "rgba(88,166,255,0.10)",
    "purple": "#bc8cff",
    "grid": "#21262d", "chart_text": "#8b949e",
}
LIGHT = {
    "bg": "#ffffff", "surface": "#f6f8fa", "surface2": "#eef1f5",
    "border": "#d0d7de", "border_light": "#e8ecf0",
    "text": "#1f2328", "text2": "#656d76", "text3": "#afb8c1",
    "green": "#1a7f37", "green_bg": "rgba(26,127,55,0.08)",
    "red": "#cf222e", "red_bg": "rgba(207,34,46,0.08)",
    "yellow": "#9a6700", "yellow_bg": "rgba(154,103,0,0.08)",
    "blue": "#0969da", "blue_bg": "rgba(9,105,218,0.08)",
    "purple": "#8250df",
    "grid": "#e8ecf0", "chart_text": "#656d76",
}


def inject_css(t):
    st.markdown(f"""<style>
    /* ── Page ── */
    .stApp, [data-testid="stAppViewContainer"], .main,
    [data-testid="stMain"], [data-testid="stAppViewBlockContainer"]
    {{ background-color: {t['bg']} !important; color: {t['text']} !important; }}
    [data-testid="stHeader"] {{ background-color: {t['bg']} !important; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {t['surface']} !important;
        border-right: 1px solid {t['border']} !important;
    }}
    section[data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {{
        background: {t['surface2']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}

    /* ── Text ── */
    h1,h2,h3,.stTitle,.stSubheader {{ color: {t['text']} !important; }}
    p,span,label,.stCaption,.stMarkdown {{ color: {t['text2']} !important; }}
    .stDivider {{ border-color: {t['border']} !important; }}

    /* ── Metrics ── */
    [data-testid="stMetric"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.72rem !important; font-weight: 600 !important;
        letter-spacing: 0.05em !important; color: {t['text2']} !important;
        text-transform: uppercase !important;
        white-space: nowrap !important; overflow: visible !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.3rem !important; font-weight: 700 !important;
        color: {t['text']} !important;
    }}
    [data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}

    /* ── Containers ── */
    [data-testid="stExpander"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
    }}
    .stDataFrame {{
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stAlert"] {{
        background: {t['surface']} !important;
        border: 1px solid {t['border']} !important;
    }}

    /* ── Spacing ── */
    .main .block-container {{ padding-top: 0.8rem !important; max-width: 1400px; }}
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def query_df(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# STOCK PRICE CHARTS
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_history_range(ticker, start_date, end_date):
    """Fetch OHLCV from start_date to end_date for a ticker from yfinance."""
    import yfinance as yf
    yf_ticker = ticker if ticker.endswith(".NS") or ticker.endswith(".BO") else f"{ticker}.NS"
    try:
        df = yf.download(yf_ticker, start=start_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"),
                         interval="1d", progress=False, timeout=10)
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


def mini_stock_chart(ticker, trigger_date, target_price, stop_loss, entry_price, t):
    """30-day forward chart from trigger date with entry price, TP, SL lines."""
    start = trigger_date - timedelta(days=5)   # few days before for context
    end = trigger_date + timedelta(days=35)     # 30 trading days forward
    today = datetime.now()
    if end > today:
        end = today + timedelta(days=1)

    df = fetch_stock_history_range(ticker, start, end)
    if df is None or df.empty or len(df) < 2:
        return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    # Split into before/after trigger for coloring
    before = close[close.index < pd.Timestamp(trigger_date)]
    after = close[close.index >= pd.Timestamp(trigger_date)]

    # Determine if price went up or down after trigger
    if len(after) >= 2:
        is_up = float(after.iloc[-1]) >= float(after.iloc[0])
    else:
        is_up = float(close.iloc[-1]) >= float(close.iloc[0])
    line_clr = t["green"] if is_up else t["red"]

    # Y-axis range: tight around the data + TP/SL
    all_vals = list(close.values)
    if target_price and target_price > 0:
        all_vals.append(target_price)
    if stop_loss and stop_loss > 0:
        all_vals.append(stop_loss)
    y_min = min(all_vals) * 0.97
    y_max = max(all_vals) * 1.03

    fig = go.Figure()

    # Pre-trigger line (dimmed)
    if len(before) >= 2:
        fig.add_trace(go.Scatter(
            x=before.index, y=before.values, mode="lines",
            line=dict(color=t["text3"], width=1, dash="dot"),
            hovertemplate="%{x|%b %d}: ₹%{y:,.2f}<extra></extra>",
            showlegend=False,
        ))

    # Post-trigger line (main)
    if len(after) >= 1:
        fig.add_trace(go.Scatter(
            x=after.index, y=after.values, mode="lines",
            line=dict(color=line_clr, width=2),
            hovertemplate="%{x|%b %d}: ₹%{y:,.2f}<extra></extra>",
            showlegend=False,
        ))

    # Entry price line
    if entry_price and entry_price > 0:
        fig.add_hline(y=entry_price, line_dash="solid", line_color=t["blue"], line_width=1,
                      opacity=0.5)
        fig.add_annotation(x=close.index[-1], y=entry_price, text="Entry",
                           showarrow=False, font=dict(size=8, color=t["blue"]),
                           xanchor="left", xshift=4)

    # Target line
    if target_price and target_price > 0:
        fig.add_hline(y=target_price, line_dash="dash", line_color=t["green"], line_width=1,
                      opacity=0.7)
        fig.add_annotation(x=close.index[-1], y=target_price, text="TP",
                           showarrow=False, font=dict(size=8, color=t["green"]),
                           xanchor="left", xshift=4)

    # Stop loss line
    if stop_loss and stop_loss > 0:
        fig.add_hline(y=stop_loss, line_dash="dash", line_color=t["red"], line_width=1,
                      opacity=0.7)
        fig.add_annotation(x=close.index[-1], y=stop_loss, text="SL",
                           showarrow=False, font=dict(size=8, color=t["red"]),
                           xanchor="left", xshift=4)

    # Trigger marker
    fig.add_vline(x=trigger_date, line_dash="dot", line_color=t["blue"], line_width=1,
                  opacity=0.4)

    fig.update_layout(
        height=150, margin=dict(l=50, r=30, t=4, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["chart_text"], size=9),
        xaxis=dict(
            showgrid=False, showticklabels=True, tickformat="%b %d",
            color=t["chart_text"], showline=True, linecolor=t["border"],
            linewidth=1,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=t["grid"], gridwidth=0.5,
            side="left", tickprefix="₹", color=t["chart_text"],
            range=[y_min, y_max],
            showline=False,
        ),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], font_color=t["text"], font_size=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# TRIGGER JSON LOADER
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_trigger_jsons(days_back=30):
    """Load trigger result JSONs from project root, returns list of day dicts sorted newest first."""
    pattern = str(PROJECT_ROOT / "trigger_results_in_morning_*.json")
    files = sorted(glob.glob(pattern), reverse=True)

    cutoff = datetime.now() - timedelta(days=days_back)
    days = []
    for fpath in files:
        fname = Path(fpath).stem  # trigger_results_in_morning_20260615
        date_str = fname.replace("trigger_results_in_morning_", "")
        if not date_str.isdigit() or len(date_str) != 8:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        if dt < cutoff:
            break

        with open(fpath, "r", encoding="utf-8") as f:
            try:
                raw = f.read()
                # Handle NaN/Infinity in JSON (not valid JSON but yfinance outputs them)
                raw = raw.replace(': NaN', ': null').replace(':NaN', ':null')
                raw = raw.replace(': Infinity', ': null').replace(':Infinity', ':null')
                raw = raw.replace(': -Infinity', ': null').replace(':-Infinity', ':null')
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

        meta = data.get("metadata", {})

        # Parse regime — can be string or dict
        regime_raw = meta.get("regime", "unknown")
        if isinstance(regime_raw, dict):
            regime_type = regime_raw.get("type", "unknown")
            nifty_rsi = regime_raw.get("nifty_rsi")
            nifty_slope = regime_raw.get("nifty_slope_pct")
            max_picks = regime_raw.get("max_picks")
        else:
            regime_type = str(regime_raw)
            nifty_rsi = meta.get("nifty_rsi")
            nifty_slope = meta.get("nifty_slope_pct")
            max_picks = None

        # Collect stocks — prefer top-level trigger arrays (detailed data)
        stocks = []
        skip_keys = {"metadata", "screening_summary"}
        for key in data:
            if key in skip_keys:
                continue
            val = data[key]
            if isinstance(val, list):
                for s in val:
                    if isinstance(s, dict) and "ticker" in s:
                        s["_trigger_type"] = key
                        stocks.append(s)

        # Fallback to screening_summary.triggers.top_candidates (basic data)
        if not stocks:
            triggers = data.get("screening_summary", {}).get("triggers", {})
            for trig_name, trig_data in triggers.items():
                if isinstance(trig_data, dict):
                    for c in trig_data.get("top_candidates", []):
                        if isinstance(c, dict) and "ticker" in c:
                            stocks.append({
                                "ticker": c.get("ticker", ""),
                                "name": c.get("name", ""),
                                "current_price": c.get("price", 0),
                                "change_rate": c.get("change_pct", 0),
                                "_trigger_type": trig_name,
                                "_basic": True,
                            })

        summary = data.get("screening_summary", {})
        total_scanned = summary.get("total_tickers_scanned", 0)

        # Extract filters from metadata
        filters = meta.get("filters", {})
        max_change = filters.get("max_trigger_change_pct", 6.0)
        max_rsi = filters.get("max_entry_rsi", 60.0)
        min_quality = filters.get("min_quality", 40)

        days.append({
            "date": dt,
            "date_str": date_str,
            "regime_type": regime_type,
            "nifty_rsi": nifty_rsi,
            "nifty_slope": nifty_slope,
            "max_picks": max_picks,
            "message": meta.get("message", ""),
            "total_scanned": total_scanned,
            "stocks": stocks,
            "filters": {
                "max_change": max_change,
                "max_rsi": max_rsi,
                "min_quality": min_quality,
                "max_picks": max_picks,
            },
        })

    return days


# ══════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ══════════════════════════════════════════════════════════════

def hero_pnl(pnl_pct, total_pnl, invested, current_val, n_pos, t):
    """Big hero P&L banner — PRIMARY visual element."""
    clr = t["green"] if pnl_pct >= 0 else t["red"]
    bg = t["green_bg"] if pnl_pct >= 0 else t["red_bg"]
    arrow = "▲" if pnl_pct >= 0 else "▼"
    sign = "+" if pnl_pct >= 0 else ""
    return f"""
    <div style="background:{t['surface']};border:1px solid {t['border']};border-radius:14px;
         padding:24px 28px;margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
        <div>
          <div style="font-size:0.72rem;color:{t['text3']};text-transform:uppercase;
               letter-spacing:0.06em;font-weight:600;margin-bottom:6px;">Total P&L</div>
          <div style="display:flex;align-items:baseline;gap:12px;">
            <span style="font-size:2.2rem;font-weight:800;color:{clr};">{sign}{pnl_pct:.2f}%</span>
            <span style="background:{bg};color:{clr};padding:4px 12px;border-radius:6px;
                 font-weight:700;font-size:0.85rem;">{arrow} {sign}₹{total_pnl:,.0f}</span>
          </div>
        </div>
        <div style="display:flex;gap:32px;">
          <div style="text-align:right;">
            <div style="font-size:0.68rem;color:{t['text3']};text-transform:uppercase;letter-spacing:0.05em;">Entry Total</div>
            <div style="font-size:1.1rem;font-weight:600;color:{t['text']};">₹{invested:,.0f}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.68rem;color:{t['text3']};text-transform:uppercase;letter-spacing:0.05em;">Current</div>
            <div style="font-size:1.1rem;font-weight:600;color:{t['text']};">₹{current_val:,.0f}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.68rem;color:{t['text3']};text-transform:uppercase;letter-spacing:0.05em;">Positions</div>
            <div style="font-size:1.1rem;font-weight:600;color:{t['text']};">{n_pos}</div>
          </div>
        </div>
      </div>
    </div>"""


def insights_bar(open_pos, t):
    """Quick insights strip — removes need to scan charts."""
    if open_pos.empty:
        return ""
    pnls = (open_pos["current_price"] - open_pos["entry_price"]) / open_pos["entry_price"] * 100
    best_idx = pnls.idxmax()
    worst_idx = pnls.idxmin()
    best_t = open_pos.loc[best_idx, "ticker"]
    worst_t = open_pos.loc[worst_idx, "ticker"]
    best_v = pnls.loc[best_idx]
    worst_v = pnls.loc[worst_idx]

    # Closest to target
    progress = (open_pos["current_price"] - open_pos["entry_price"]) / (open_pos["take_profit"] - open_pos["entry_price"]) * 100
    closest_idx = progress.idxmax()
    closest_t = open_pos.loc[closest_idx, "ticker"]
    closest_v = progress.loc[closest_idx]

    # Closest to SL
    sl_dist = (open_pos["current_price"] - open_pos["stop_loss"]) / open_pos["stop_loss"] * 100
    sl_idx = sl_dist.idxmin()
    sl_ticker = open_pos.loc[sl_idx, "ticker"]
    sl_val = sl_dist.loc[sl_idx]

    all_loss = (pnls < 0).all()
    all_win = (pnls > 0).all()
    status_msg = "⚠️ All positions in loss" if all_loss else ("🎯 All positions in profit" if all_win else f"📊 Mixed: {(pnls>0).sum()}W / {(pnls<0).sum()}L")

    items = [
        f'<span style="color:{t["green"]};">▲ Best:</span> {best_t} ({best_v:+.2f}%)',
        f'<span style="color:{t["red"]};">▼ Worst:</span> {worst_t} ({worst_v:+.2f}%)',
        f'<span style="color:{t["blue"]};">⊕ Closest to TP:</span> {closest_t} ({closest_v:.0f}%)',
        f'<span style="color:{t["yellow"]};">⊖ Closest to SL:</span> {sl_ticker} ({sl_val:.1f}% away)',
        status_msg,
    ]

    return f"""
    <div style="background:{t['surface']};border:1px solid {t['border']};border-radius:10px;
         padding:10px 18px;margin-bottom:20px;display:flex;flex-wrap:wrap;gap:6px 24px;
         font-size:0.8rem;color:{t['text2']};">
      {''.join(f'<span>{item}</span>' for item in items)}
    </div>"""


def position_card(row, t, is_best=False, is_worst=False):
    """Single position card with hierarchy: P&L first, prices second."""
    pnl = (row["current_price"] - row["entry_price"]) / row["entry_price"] * 100
    pnl_val = row["current_price"] - row["entry_price"]
    progress = max(0, min(100, (row["current_price"] - row["entry_price"]) / (row["take_profit"] - row["entry_price"]) * 100))
    days = (datetime.now() - pd.to_datetime(row["entry_date"])).days
    quality = float(row.get("quality_score", 0) or 0)
    sl_dist = (row["current_price"] - row["stop_loss"]) / row["stop_loss"] * 100

    is_profit = pnl >= 0
    clr = t["green"] if is_profit else t["red"]
    bg = t["green_bg"] if is_profit else t["red_bg"]
    arrow = "▲" if is_profit else "▼"
    sign = "+" if is_profit else ""

    # Highlight border for best/worst
    border_clr = t["green"] if is_best else (t["red"] if is_worst else t["border"])
    border_w = "2px" if (is_best or is_worst) else "1px"

    # SL warning
    sl_warn = ""
    if sl_dist < 2:
        sl_warn = f'<span style="background:{t["yellow_bg"]};color:{t["yellow"]};font-size:0.68rem;padding:2px 6px;border-radius:4px;font-weight:600;">⚠ {sl_dist:.1f}% to SL</span>'

    return f"""<div style="background:{t['surface']};border:{border_w} solid {border_clr};border-radius:12px;padding:18px 20px;margin-bottom:12px;"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;"><div style="display:flex;align-items:center;gap:8px;"><span style="font-size:1.1rem;font-weight:700;color:{t['text']};">{row['ticker']}</span>{sl_warn}</div><div style="background:{bg};color:{clr};padding:5px 12px;border-radius:8px;font-weight:700;font-size:0.95rem;">{arrow} {sign}{pnl:.2f}%</div></div><div style="background:{bg};border-radius:8px;padding:10px 14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;"><div><span style="font-size:0.68rem;color:{t['text3']};text-transform:uppercase;">P&L</span><div style="font-size:1.2rem;font-weight:700;color:{clr};">{sign}₹{pnl_val:,.2f}</div></div><div style="text-align:right;"><span style="font-size:0.68rem;color:{t['text3']};text-transform:uppercase;">Current</span><div style="font-size:1.2rem;font-weight:700;color:{clr};">₹{row['current_price']:,.2f}</div></div></div><div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:14px;"><div><div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">Entry</div><div style="font-size:0.85rem;font-weight:500;color:{t['text']};">₹{row['entry_price']:,.2f}</div></div><div><div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">↑ Target</div><div style="font-size:0.85rem;font-weight:500;color:{t['green']};">₹{row['take_profit']:,.2f}</div></div><div><div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">↓ Stop Loss</div><div style="font-size:0.85rem;font-weight:500;color:{t['red']};">₹{row['stop_loss']:,.2f}</div></div></div><div style="margin-bottom:6px;"><div style="display:flex;justify-content:space-between;font-size:0.68rem;color:{t['text3']};margin-bottom:4px;"><span>Target progress</span><span>{progress:.0f}%</span></div><div style="height:4px;background:{t['border']};border-radius:2px;overflow:hidden;"><div style="width:{progress:.0f}%;height:100%;background:linear-gradient(90deg,{t['blue']},{t['purple']});border-radius:2px;"></div></div></div><div style="display:flex;justify-content:space-between;font-size:0.68rem;color:{t['text3']};margin-top:8px;"><span>{days}d held · Exit by {str(row.get('max_exit_date',''))[:10]}</span><span>Quality: {quality:.0f}</span></div></div>"""


# ══════════════════════════════════════════════════════════════
# TRADE HISTORY & WATCHLIST COMPONENTS
# ══════════════════════════════════════════════════════════════

def trade_history_card(row, t):
    """Detailed trade history card — KR-style with target achievement."""
    ret = float(row.get("profit_rate", 0) or row.get("return_pct", 0) or 0)
    buy_price = float(row.get("buy_price", 0) or row.get("entry_price", 0))
    sell_price = float(row.get("sell_price", 0) or row.get("exit_price", 0))
    ticker = row.get("ticker", "")
    sector = row.get("sector", "") or ""
    trigger = row.get("trigger_type", "") or ""
    days = int(row.get("holding_days", 0) or 0)

    # Parse scenario JSON for extra data
    scenario = row.get("scenario", "")
    quality = 0
    target_price = 0
    exit_reason = ""
    if isinstance(scenario, str) and scenario:
        try:
            s = json.loads(scenario)
            quality = float(s.get("quality_score", 0) or 0)
            exit_reason = s.get("exit_reason", "")
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to get target from monitored_positions if available
    if "take_profit" in row:
        target_price = float(row.get("take_profit", 0) or 0)
    elif "target_price" in row:
        target_price = float(row.get("target_price", 0) or 0)

    # Target achievement
    if target_price > 0 and buy_price > 0:
        target_gain = target_price - buy_price
        actual_gain = sell_price - buy_price
        achievement = actual_gain / target_gain * 100 if target_gain > 0 else 0
    else:
        achievement = 0

    buy_date = str(row.get("buy_date", "") or row.get("entry_date", ""))[:10]
    sell_date = str(row.get("sell_date", "") or row.get("exit_date", ""))[:10]

    is_win = ret > 0
    clr = t["green"] if is_win else t["red"]
    sign = "+" if is_win else ""

    reason_map = {
        "TP_HIT": "🎯 Target Hit",
        "SL_HIT": "⛔ Stop Loss",
        "TIME_EXIT": "⏰ Time Exit",
    }
    reason_label = reason_map.get(exit_reason, exit_reason or row.get("exit_reason", ""))

    return f"""
    <div style="background:{t['surface']};border:1px solid {t['border']};border-radius:12px;
         padding:20px 24px;margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="font-size:1.15rem;font-weight:700;color:{t['text']};">{ticker}</span>
            {f'<span style="background:{t["surface2"]};color:{t["text2"]};font-size:0.68rem;padding:3px 8px;border-radius:4px;">{sector}</span>' if sector else ''}
            {f'<span style="background:{t["surface2"]};color:{t["text2"]};font-size:0.68rem;padding:3px 8px;border-radius:4px;">{trigger}</span>' if trigger else ''}
          </div>
          <div style="font-size:0.78rem;color:{t['text3']};">
            {buy_date} → {sell_date} ( {days} days held )
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.68rem;color:{t['text3']};text-transform:uppercase;">Return</div>
          <div style="font-size:1.5rem;font-weight:800;color:{clr};">{sign}{ret:.2f}%</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;
           background:{t['surface2']};border-radius:8px;padding:14px 16px;">
        <div>
          <div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;margin-bottom:4px;">Buy Price</div>
          <div style="font-size:1rem;font-weight:600;color:{t['text']};">₹{buy_price:,.2f}</div>
        </div>
        <div>
          <div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;margin-bottom:4px;">Sell Price</div>
          <div style="font-size:1rem;font-weight:600;color:{t['text']};">₹{sell_price:,.2f}</div>
        </div>
        <div>
          <div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;margin-bottom:4px;">AI Target</div>
          <div style="font-size:1rem;font-weight:600;color:{t['green']};">{'₹{:,.2f}'.format(target_price) if target_price > 0 else '—'}</div>
        </div>
        <div>
          <div style="font-size:0.65rem;color:{t['text3']};text-transform:uppercase;margin-bottom:4px;">Target Achievement</div>
          <div style="font-size:1rem;font-weight:700;color:{t['green'] if achievement >= 80 else t['yellow'] if achievement >= 50 else t['text2']};">{achievement:.0f}%</div>
        </div>
      </div>
      {f'<div style="margin-top:10px;font-size:0.78rem;color:{t["text2"]};">{reason_label}</div>' if reason_label else ''}
    </div>"""


def watchlist_card(stock, t, was_entered=False, filters=None):
    """Watchlist/analysis card built from trigger JSON data."""
    ticker = _esc(str(stock.get("ticker", "")))
    trigger_type = _esc(str(stock.get("_trigger_type", "")))
    is_basic = stock.get("_basic", False)
    current_price = float(stock.get("current_price", 0) or 0)
    change_rate = float(stock.get("change_rate", 0) or stock.get("change_pct", 0) or 0)
    target_price = float(stock.get("target_price", 0) or 0)
    stop_loss = float(stock.get("stop_loss_price", 0) or 0)
    rr = float(stock.get("risk_reward_ratio", 0) or 0)
    quality = float(stock.get("quality_score", 0) or 0)
    quality_signal = _esc(str(stock.get("quality_signal", "") or ""))
    quality_reasons = _esc(str(stock.get("quality_reasons", "") or ""))
    final_score = float(stock.get("final_score", 0) or 0)

    metrics = stock.get("metrics", {}) or {}
    sector = _esc(str(metrics.get("sector", "") or ""))
    rsi = metrics.get("rsi_14")

    val_score = float(metrics.get("valuation_score", 0) or 0)
    growth_score = float(metrics.get("growth_score", 0) or 0)
    profit_score = float(metrics.get("profitability_score", 0) or 0)
    tech_score = float(metrics.get("technical_score", 0) or 0)
    analyst_score = float(metrics.get("analyst_score", 0) or 0)
    risk_score = float(metrics.get("risk_score", 0) or 0)

    # ── Derive decision + rejection reason ──
    if filters is None:
        filters = {}
    max_change = filters.get("max_change", 6.0)
    max_rsi = filters.get("max_rsi", 60.0)
    min_quality = filters.get("min_quality", 40)
    max_picks = filters.get("max_picks")

    reject_reasons = []
    if not is_basic:
        if quality > 0 and quality < min_quality:
            reject_reasons.append(f"Quality {quality:.0f} &lt; {min_quality:.0f} min")
        if rsi is not None and rsi > max_rsi:
            reject_reasons.append(f"RSI {rsi:.0f} &gt; {max_rsi:.0f} limit")
        if abs(change_rate) > max_change:
            reject_reasons.append(f"Change {change_rate:+.1f}% exceeds ±{max_change:.0f}%")
        if final_score > 0 and final_score < 0.2:
            reject_reasons.append(f"Score {final_score:.2f} too low")

    if was_entered:
        dec_clr, dec_bg = t["green"], t["green_bg"]
        dec_label, dec_icon = "Entered", "✓"
        reason_text = ""
    elif reject_reasons:
        dec_clr, dec_bg = t["red"], t["red_bg"]
        dec_label, dec_icon = "Rejected", "✗"
        reason_text = " · ".join(reject_reasons)
    else:
        # Passed filters but not entered (max picks limit or not top-ranked)
        dec_clr, dec_bg = t["yellow"], t["yellow_bg"]
        dec_label, dec_icon = "Not Selected", "–"
        if max_picks is not None:
            reason_text = f"Ranked below top {max_picks} picks"
        else:
            reason_text = "Not top-ranked for entry"

    # Colors
    chg_clr = t["green"] if change_rate > 0 else t["red"] if change_rate < 0 else t["text2"]
    chg_sign = "+" if change_rate > 0 else ""
    q_clr = t["green"] if quality >= 70 else t["yellow"] if quality >= 55 else t["red"] if quality > 0 else t["text3"]
    rr_clr = t["green"] if rr >= 2 else t["yellow"] if rr >= 1 else t["red"] if rr > 0 else t["text3"]
    rr_bg = t["green_bg"] if rr >= 2 else t["yellow_bg"] if rr >= 1 else t["red_bg"] if rr > 0 else t["surface2"]

    exp_ret = (target_price / current_price - 1) * 100 if current_price > 0 and target_price > 0 else 0
    exp_loss = (stop_loss / current_price - 1) * 100 if current_price > 0 and stop_loss > 0 else 0

    # Build parts list
    parts = []
    parts.append(f'<div style="background:{t["surface"]};border:1px solid {t["border"]};border-radius:12px;padding:18px 22px;margin-bottom:14px;">')

    # Header row
    sector_badge = f'<span style="background:{t["surface2"]};color:{t["text2"]};font-size:0.65rem;padding:3px 8px;border-radius:4px;">{sector}</span>' if sector else ''
    trigger_badge = f'<span style="background:{t["surface2"]};color:{t["text2"]};font-size:0.65rem;padding:3px 8px;border-radius:4px;">{trigger_type}</span>' if trigger_type else ''
    quality_badge = f'<div style="background:{t["surface2"]};border:2px solid {q_clr};border-radius:8px;padding:3px 9px;font-size:0.85rem;font-weight:700;color:{q_clr};">{quality:.0f}</div>' if quality > 0 else ''

    parts.append(f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">')
    parts.append(f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">')
    parts.append(f'<span style="font-size:1.1rem;font-weight:700;color:{t["text"]};">{ticker}</span>')
    if sector_badge:
        parts.append(sector_badge)
    if trigger_badge:
        parts.append(trigger_badge)
    parts.append(f'<span style="font-size:0.72rem;color:{chg_clr};font-weight:600;">{chg_sign}{change_rate:.1f}%</span>')
    parts.append('</div>')
    if quality_badge:
        parts.append(quality_badge)
    parts.append('</div>')

    # Score bars
    if not is_basic and any([val_score, growth_score, profit_score, tech_score]):
        def _bar(label, val):
            c = t["green"] if val >= 70 else t["yellow"] if val >= 50 else t["red"]
            w = min(100, max(0, val))
            return (f'<div style="flex:1;min-width:60px;">'
                    f'<div style="font-size:0.58rem;color:{t["text3"]};margin-bottom:2px;">{label}</div>'
                    f'<div style="background:{t["surface2"]};border-radius:3px;height:4px;overflow:hidden;">'
                    f'<div style="background:{c};height:100%;width:{w}%;border-radius:3px;"></div></div>'
                    f'<div style="font-size:0.6rem;color:{c};margin-top:1px;">{val:.0f}</div></div>')
        parts.append(f'<div style="display:flex;gap:6px;margin-bottom:12px;">')
        parts.append(_bar("Val", val_score))
        parts.append(_bar("Grw", growth_score))
        parts.append(_bar("Prof", profit_score))
        parts.append(_bar("Tech", tech_score))
        parts.append(_bar("Anlst", analyst_score))
        parts.append(_bar("Risk", risk_score))
        parts.append('</div>')

    # Price grid
    if not is_basic and target_price > 0:
        parts.append(f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">')
        parts.append(f'<div><div style="font-size:0.65rem;color:{t["text3"]};text-transform:uppercase;margin-bottom:4px;">Current</div><div style="font-size:1rem;font-weight:600;color:{t["text"]};">&#8377;{current_price:,.2f}</div></div>')
        parts.append(f'<div><div style="font-size:0.65rem;color:{t["text3"]};text-transform:uppercase;margin-bottom:4px;">Target</div><div style="font-size:1rem;font-weight:600;color:{t["green"]};">&#8377;{target_price:,.2f}</div></div>')
        parts.append(f'<div><div style="font-size:0.65rem;color:{t["text3"]};text-transform:uppercase;margin-bottom:4px;">Stop Loss</div><div style="font-size:1rem;font-weight:600;color:{t["red"]};">&#8377;{stop_loss:,.2f}</div></div>')
        parts.append('</div>')
    else:
        parts.append(f'<div style="display:flex;gap:24px;margin-bottom:12px;">')
        parts.append(f'<div><div style="font-size:0.65rem;color:{t["text3"]};text-transform:uppercase;margin-bottom:4px;">Price</div><div style="font-size:1rem;font-weight:600;color:{t["text"]};">&#8377;{current_price:,.2f}</div></div>')
        parts.append(f'<div><div style="font-size:0.65rem;color:{t["text3"]};text-transform:uppercase;margin-bottom:4px;">Change</div><div style="font-size:1rem;font-weight:600;color:{chg_clr};">{chg_sign}{change_rate:.1f}%</div></div>')
        parts.append('</div>')

    # R/R box
    if not is_basic and rr > 0:
        parts.append(f'<div style="background:{rr_bg};border:1px solid {t["border"]};border-radius:8px;padding:10px 14px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">')
        parts.append(f'<div><div style="font-size:0.72rem;font-weight:600;color:{rr_clr};">Risk/Reward Ratio</div>')
        parts.append(f'<div style="display:flex;gap:16px;margin-top:4px;"><span style="font-size:0.75rem;color:{t["green"]};">Expected Return: +{exp_ret:.1f}%</span><span style="font-size:0.75rem;color:{t["red"]};">Expected Loss: {exp_loss:.1f}%</span></div></div>')
        parts.append(f'<div style="background:{rr_bg};border:2px solid {rr_clr};border-radius:8px;padding:4px 10px;font-size:1.1rem;font-weight:700;color:{rr_clr};">{rr:.1f}</div>')
        parts.append('</div>')

    # Decision bar with reason
    parts.append(f'<div style="background:{dec_bg};border-radius:8px;padding:9px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">')
    parts.append(f'<span style="font-size:0.85rem;font-weight:700;color:{dec_clr};">{dec_icon} Decision : {dec_label}</span>')
    if reason_text:
        parts.append(f'<span style="font-size:0.72rem;color:{t["text2"]};">{reason_text}</span>')
    parts.append('</div>')

    # AI analysis reasons
    if quality_reasons:
        parts.append(f'<div style="margin-top:8px;font-size:0.72rem;color:{t["text3"]};padding:8px;background:{t["surface2"]};border-radius:6px;">')
        parts.append(f'<span style="font-weight:600;color:{t["text2"]};">AI Analysis:</span> {quality_reasons}')
        parts.append('</div>')

    parts.append('</div>')
    return '\n'.join(parts)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"### 📊 PRISM-INSIGHT")
        st.caption("King 2 Strategy · NSE India")

        st.markdown("##### ⚙️ Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        t = DARK if dark_mode else LIGHT
        auto_refresh = st.toggle("🔄 Auto-refresh", value=False)
        if auto_refresh:
            refresh_sec = st.select_slider("Interval", options=[15, 30, 60, 120, 300], value=60,
                                           format_func=lambda x: f"{x}s")
            st.markdown(f'<meta http-equiv="refresh" content="{refresh_sec}">', unsafe_allow_html=True)

        st.markdown("---")
        now = datetime.now()
        st.caption(f"🕐 {now.strftime('%d %b %Y · %H:%M IST')}")

    inject_css(t)
    conn = get_db()

    # ── Navigation Tabs ──
    tab_dashboard, tab_trades, tab_watchlist = st.tabs(["💼 Dashboard", "📜 Trades", "🔍 Watchlist"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: DASHBOARD (Open Holdings)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_dashboard:
        open_pos = query_df(conn, "SELECT * FROM monitored_positions WHERE status='OPEN' ORDER BY entry_date DESC")

        if not open_pos.empty:
            open_pos["pnl_pct"] = (open_pos["current_price"] - open_pos["entry_price"]) / open_pos["entry_price"] * 100
            invested = open_pos["entry_price"].sum()
            current_val = open_pos["current_price"].sum()
            total_pnl = current_val - invested
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0
        else:
            invested = current_val = total_pnl = pnl_pct = 0

        # Hero P&L
        st.markdown(hero_pnl(pnl_pct, total_pnl, invested, current_val, len(open_pos), t),
                    unsafe_allow_html=True)

        # Quick insights
        st.markdown(insights_bar(open_pos, t), unsafe_allow_html=True)

        # Open positions as cards
        if not open_pos.empty:
            best_idx = open_pos["pnl_pct"].idxmax()
            worst_idx = open_pos["pnl_pct"].idxmin()
            n_cols = min(len(open_pos), 3)
            cols = st.columns(n_cols)
            for i, (idx, row) in enumerate(open_pos.iterrows()):
                with cols[i % n_cols]:
                    st.markdown(position_card(row, t, is_best=(idx == best_idx),
                                              is_worst=(idx == worst_idx)),
                                unsafe_allow_html=True)

            # P&L chart
            sorted_pos = open_pos.sort_values("pnl_pct", ascending=True)
            pnls = sorted_pos["pnl_pct"]

            def pnl_color(v):
                if v >= 0:
                    intensity = min(1, v / 5) * 0.6 + 0.4
                    return f"rgba(63,185,80,{intensity})" if dark_mode else f"rgba(26,127,55,{intensity})"
                else:
                    intensity = min(1, abs(v) / 5) * 0.6 + 0.4
                    return f"rgba(248,81,73,{intensity})" if dark_mode else f"rgba(207,34,46,{intensity})"

            colors = [pnl_color(v) for v in pnls]
            pnl_vals = sorted_pos["current_price"] - sorted_pos["entry_price"]

            fig = go.Figure(go.Bar(
                y=sorted_pos["ticker"], x=pnls, orientation="h",
                marker_color=colors,
                text=[f" {v:+.2f}% (₹{pv:+,.0f}) " for v, pv in zip(pnls, pnl_vals)],
                textposition="outside", textfont=dict(size=11, color=t["chart_text"]),
                hovertemplate="<b>%{y}</b><br>P&L: %{x:.2f}%<extra></extra>",
                customdata=pnl_vals,
            ))
            fig.add_vline(x=0, line_dash="solid", line_color=t["border"], line_width=1)
            fig.update_layout(
                title=dict(text="P&L by Position", font=dict(color=t["text"], size=14)),
                height=max(200, len(sorted_pos) * 55 + 60),
                margin=dict(l=10, r=80, t=45, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t["chart_text"]),
                xaxis=dict(gridcolor=t["grid"], zerolinecolor=t["border"], title="Return %"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full positions table
            with st.expander("📋 Full Positions Table"):
                open_pos["max_pnl_pct"] = (open_pos["take_profit"] - open_pos["entry_price"]) / open_pos["entry_price"] * 100
                open_pos["sl_pct"] = (open_pos["stop_loss"] - open_pos["entry_price"]) / open_pos["entry_price"] * 100
                open_pos["days_held"] = (datetime.now() - pd.to_datetime(open_pos["entry_date"])).dt.days
                tbl = open_pos[["ticker", "entry_price", "current_price", "pnl_pct",
                                 "take_profit", "max_pnl_pct", "stop_loss", "sl_pct",
                                 "days_held", "max_exit_date", "quality_score"]].copy()
                tbl.columns = ["Ticker", "Entry", "Current", "P&L %",
                                "Target", "Target %", "SL", "SL %",
                                "Days", "Exit By", "Quality"]
                st.dataframe(
                    tbl.style.format({
                        "Entry": "₹{:,.2f}", "Current": "₹{:,.2f}",
                        "P&L %": "{:+.2f}%", "Target": "₹{:,.2f}", "Target %": "+{:.2f}%",
                        "SL": "₹{:,.2f}", "SL %": "{:.2f}%", "Quality": "{:.0f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("No open positions — waiting for King 2 signals.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: TRADES (Detailed History)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_trades:
        st.markdown(f'<div style="font-size:1.2rem;font-weight:700;color:{t["text"]};margin-bottom:20px;">Detailed History</div>',
                    unsafe_allow_html=True)

        # Get closed positions with full data (has target/SL info)
        closed_pos = query_df(conn, """
            SELECT * FROM monitored_positions WHERE status != 'OPEN' ORDER BY exit_date DESC
        """)
        # Get trading history for extra data
        trade_hist = query_df(conn, "SELECT * FROM in_trading_history ORDER BY sell_date DESC")

        # Merge: use closed_pos as primary (has target/SL), enrich from trade_hist
        if not closed_pos.empty:
            # Performance summary
            closed_pos["return_pct"] = pd.to_numeric(closed_pos.get("return_pct", 0), errors="coerce").fillna(0)
            n = len(closed_pos)
            w = int((closed_pos["return_pct"] > 0).sum())
            l = int((closed_pos["return_pct"] < 0).sum())
            wr = w / n * 100 if n else 0
            avg_ret = closed_pos["return_pct"].mean()
            avg_w = closed_pos.loc[closed_pos["return_pct"] > 0, "return_pct"].mean() if w else 0
            avg_l = abs(closed_pos.loc[closed_pos["return_pct"] < 0, "return_pct"].mean()) if l else 0
            pf = avg_w / avg_l if avg_l > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Trades", n)
            c2.metric("Win Rate", f"{wr:.0f}%", delta=f"{w}W / {l}L")
            c3.metric("Avg Return", f"{avg_ret:+.2f}%")
            c4.metric("Profit Factor", f"{pf:.2f}")
            avg_days = (pd.to_datetime(closed_pos["exit_date"]) - pd.to_datetime(closed_pos["entry_date"])).dt.days.mean()
            c5.metric("Avg Hold", f"{avg_days:.0f} days")

            # Return distribution chart
            fig = go.Figure(go.Histogram(
                x=closed_pos["return_pct"], nbinsx=15,
                marker=dict(color=t["purple"], line=dict(color=t["blue"], width=1)),
            ))
            fig.add_vline(x=0, line_dash="dash", line_color=t["text3"], line_width=1)
            fig.update_layout(
                title=dict(text="Return Distribution", font=dict(color=t["text"], size=14)),
                height=280, margin=dict(l=50, r=20, t=45, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t["chart_text"]),
                yaxis=dict(gridcolor=t["grid"], title="# Trades"),
                xaxis=dict(gridcolor=t["grid"], title="Return %"),
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Individual trade cards
            for _, row in closed_pos.iterrows():
                rd = row.to_dict()
                # Enrich with trade history data
                rd["buy_price"] = rd.get("entry_price", 0)
                rd["sell_price"] = rd.get("exit_price", 0)
                rd["buy_date"] = rd.get("entry_date", "")
                rd["sell_date"] = rd.get("exit_date", "")
                rd["profit_rate"] = rd.get("return_pct", 0)
                rd["holding_days"] = (pd.to_datetime(rd.get("exit_date", "")) - pd.to_datetime(rd.get("entry_date", ""))).days if rd.get("exit_date") and rd.get("entry_date") else 0
                rd["scenario"] = json.dumps({"exit_reason": rd.get("exit_reason", ""), "quality_score": rd.get("quality_score", 0)})
                # Get sector from trade history if missing
                if not rd.get("sector") and not trade_hist.empty:
                    match = trade_hist[trade_hist["ticker"] == rd["ticker"]]
                    if not match.empty:
                        rd["sector"] = match.iloc[0].get("sector", "")
                st.markdown(trade_history_card(rd, t), unsafe_allow_html=True)
        elif not trade_hist.empty:
            # Fallback: use in_trading_history (less data but still useful)
            for _, row in trade_hist.iterrows():
                st.markdown(trade_history_card(row.to_dict(), t), unsafe_allow_html=True)
        else:
            st.info("No trade history yet.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: WATCHLIST (Daily Trigger Results from JSON)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_watchlist:
        # Sidebar-like controls inside the tab
        wl_col1, wl_col2 = st.columns([1, 3])
        with wl_col1:
            days_back = st.selectbox("Show last", [7, 14, 30, 60, 90, 180, 365], index=2,
                                     format_func=lambda d: f"{d} days")

        trigger_days = load_trigger_jsons(days_back=days_back)

        if not trigger_days:
            st.info("No trigger results found.")
        else:
            # Build set of entered tickers with their entry dates for cross-reference
            entered_positions = query_df(conn, "SELECT ticker, entry_date FROM monitored_positions")
            entered_set = set()
            if not entered_positions.empty:
                for _, erow in entered_positions.iterrows():
                    ed = str(erow.get("entry_date", ""))[:10].replace("-", "")
                    entered_set.add((erow["ticker"], ed))

            for day in trigger_days:
                dt = day["date"]
                regime = day["regime_type"]
                stocks = day["stocks"]
                n_stocks = len(stocks)
                nifty_rsi = day["nifty_rsi"]
                nifty_slope = day["nifty_slope"]
                max_picks = day["max_picks"]
                message = day["message"]

                # Regime badge
                regime_map = {
                    "bull": ("🟢", t["green"], t["green_bg"]),
                    "bull_transition": ("🟢", t["green"], t["green_bg"]),
                    "neutral": ("🟡", t["yellow"], t["yellow_bg"]),
                    "correction": ("🟡", t["yellow"], t["yellow_bg"]),
                    "bear": ("🔴", t["red"], t["red_bg"]),
                    "bear_bottom": ("🔴", t["red"], t["red_bg"]),
                }
                r_emoji, r_clr, r_bg = regime_map.get(regime, ("⚪", t["text2"], t["surface2"]))

                # Count entered stocks for this day
                n_entered = sum(1 for s in stocks if (s.get("ticker", ""), day["date_str"]) in entered_set)

                # Date header with regime
                date_label = dt.strftime("%B %d, %Y")
                picks_info = f"Max picks: {max_picks}" if max_picks else ""
                rsi_info = f"RSI {nifty_rsi:.1f}" if nifty_rsi is not None else ""
                slope_info = f"Slope {nifty_slope:+.1f}%" if nifty_slope is not None else ""
                meta_parts = [x for x in [rsi_info, slope_info, picks_info] if x]
                meta_str = " · ".join(meta_parts)

                if n_stocks > 0:
                    count_badge = f"{n_stocks} stocks"
                    if n_entered > 0:
                        count_badge += f" · {n_entered} entered"
                else:
                    count_badge = "No picks"

                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;margin:28px 0 14px;
                     padding:12px 18px;background:{t['surface']};border:1px solid {t['border']};border-radius:10px;">
                  <div>
                    <span style="font-size:1.1rem;font-weight:700;color:{t['text']};">{date_label}</span>
                    <span style="background:{r_bg};color:{r_clr};font-size:0.72rem;padding:3px 10px;
                         border-radius:5px;font-weight:600;margin-left:12px;">{r_emoji} {regime.replace('_',' ').title()}</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:14px;">
                    <span style="font-size:0.72rem;color:{t['text3']};">{meta_str}</span>
                    <span style="background:{t['surface2']};color:{t['text2']};font-size:0.75rem;padding:4px 12px;
                         border-radius:6px;font-weight:500;">{count_badge}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Show message for no-pick days
                if n_stocks == 0 and message:
                    st.markdown(f"""
                    <div style="background:{t['surface2']};border-radius:8px;padding:12px 18px;margin-bottom:12px;
                         font-size:0.82rem;color:{t['text2']};">
                      {message}
                    </div>""", unsafe_allow_html=True)

                # Stock cards in 2-column grid
                if n_stocks > 0:
                    day_filters = day.get("filters", {})
                    n_cols = min(n_stocks, 2)
                    cols = st.columns(n_cols)
                    for i, stock in enumerate(stocks):
                        was_entered = (stock.get("ticker", ""), day["date_str"]) in entered_set
                        with cols[i % n_cols]:
                            st.markdown(watchlist_card(stock, t, was_entered=was_entered, filters=day_filters), unsafe_allow_html=True)
                            # Mini price chart
                            s_ticker = stock.get("ticker", "")
                            s_tp = float(stock.get("target_price", 0) or 0)
                            s_sl = float(stock.get("stop_loss_price", 0) or 0)
                            s_entry = float(stock.get("current_price", 0) or 0)
                            chart = mini_stock_chart(s_ticker, dt, s_tp, s_sl, s_entry, t)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True, key=f"wl_{day['date_str']}_{s_ticker}_{i}")

    # Footer
    st.markdown(f"""<div style="text-align:center;padding:16px 0;margin-top:16px;border-top:1px solid {t['border']};
         font-size:0.72rem;color:{t['text3']};">
        PRISM-INSIGHT · King 2 Strategy · NSE India ·
        {datetime.now().strftime('%d %b %Y %H:%M:%S IST')}
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
