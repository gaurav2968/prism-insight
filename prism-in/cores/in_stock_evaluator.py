#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
India (NSE) Stock Quality Evaluator

Pure mathematical stock evaluation — no LLM required.
Uses yfinance (.NS suffix) for all data.

Produces a 7-dimension composite score (0-100):
  1. Valuation       (13%): Trailing PE, Forward PE, P/B
  2. Growth          (13%): Revenue growth, Earnings growth
  3. Profitability   (11%): Operating margin, Net margin, ROE
  4. Technicals      (35%): RSI-14, Price vs 50/200 MA, Golden cross, 52W position
  5. Analyst         (11%): Recommendation score, Target upside, Coverage count
  6. Risk             (7%): Beta, Debt/Equity
  7. Market Context  (10%): NIFTY 50 trend — bullish/bearish macro environment

Quality Gate (used in trigger_batch pipeline):
  - REJECT  (< 40): Fundamentally broken — skip LLM report generation
  - WEAK    (40-49): Below average — penalized in final scoring
  - NEUTRAL (50-64): Average — fair inclusion
  - GOOD    (65-79): Above average — scoring boost
  - STRONG  (80+):   Excellent — strong scoring boost

Signal Output:
  - Composite >= 70 → BUY
  - Composite >= 50 → HOLD
  - Composite <  50 → SELL

Adapted from archived prism-kr/stock_evaluator.py with India (NSE) tuning.

Usage:
    # Standalone
    python prism-in/cores/in_stock_evaluator.py RELIANCE
    python prism-in/cores/in_stock_evaluator.py TCS BEL INFY

    # As module
    from cores.in_stock_evaluator import evaluate_stock, quick_quality_check
    result = evaluate_stock("RELIANCE")
    score, signal, reasons = quick_quality_check("OLAELEC")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality Gate threshold — stocks below this score are rejected from the
# trigger batch pipeline before LLM report generation.
# ---------------------------------------------------------------------------
QUALITY_GATE_MIN = 40

# Dimension weights for composite score (must sum to 1.0)
DIMENSION_WEIGHTS = {
    "valuation":       0.13,
    "growth":          0.13,
    "profitability":   0.11,
    "technical":       0.35,
    "analyst":         0.11,
    "risk":            0.07,
    "market_context":  0.10,   # NIFTY 50 macro trend
}

# ---------------------------------------------------------------------------
# NIFTY 50 market context — fetched once per process, cached for 1 hour
# ---------------------------------------------------------------------------
_MARKET_CONTEXT_CACHE: Optional[Tuple[float, float]] = None  # (score, timestamp)
_MARKET_CONTEXT_TTL = 3600  # seconds


def _fetch_market_context_score() -> float:
    """
    Fetch NIFTY 50 (^NSEI) and return a market-trend score 0-100.

    Score interpretation:
        >= 65  → Bullish: broad market tailwind, all stocks benefit
        50-64  → Neutral: no strong macro bias
        <= 49  → Bearish: broad market headwind, reduces stock scores

    Uses the same _score_technicals logic on the index itself.
    Cached for 1 hour so only one yfinance call per batch run.
    """
    global _MARKET_CONTEXT_CACHE

    now = time.time()
    if _MARKET_CONTEXT_CACHE is not None:
        cached_score, cached_at = _MARKET_CONTEXT_CACHE
        if now - cached_at < _MARKET_CONTEXT_TTL:
            return cached_score

    try:
        t = yf.Ticker("^NSEI")
        hist = t.history(period="1y")
        if len(hist) < 50:
            return 50.0  # not enough data — neutral

        closes = hist["Close"].values
        price = float(closes[-1])
        sma50 = float(np.mean(closes[-50:]))
        sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else None
        rsi = _compute_rsi(closes, 14)
        week52_high = float(np.max(closes[-252:])) if len(closes) >= 252 else float(np.max(closes))
        pct_from_high = ((price - week52_high) / week52_high) * 100 if week52_high > 0 else None

        score = _score_technicals(rsi, price, sma50, sma200, pct_from_high)
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        logger.info(f"Market context (NIFTY 50): score={score:.1f} RSI={rsi_str}")
        _MARKET_CONTEXT_CACHE = (score, now)
        return score
    except Exception as e:
        logger.warning(f"Market context fetch failed: {e} — defaulting to neutral 50")
        return 50.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Container for stock evaluation output."""

    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    price: float = 0.0
    prev_close: float = 0.0
    day_change_pct: float = 0.0

    # Valuation
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    total_revenue: Optional[float] = None

    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    return_on_equity: Optional[float] = None

    # Balance sheet
    debt_to_equity: Optional[float] = None
    free_cashflow: Optional[float] = None

    # Technicals
    rsi_14: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    beta: Optional[float] = None
    avg_volume: Optional[int] = None

    # Analyst
    recommendation: str = ""
    recommendation_score: Optional[float] = None  # 1=strong buy … 5=strong sell
    target_mean: Optional[float] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    analyst_count: int = 0
    analyst_strong_buy: int = 0
    analyst_buy: int = 0
    analyst_hold: int = 0
    analyst_sell: int = 0

    # Sub-scores (0–100 each)
    valuation_score: float = 0.0
    growth_score: float = 0.0
    profitability_score: float = 0.0
    technical_score: float = 0.0
    analyst_score: float = 0.0
    risk_score: float = 0.0
    market_context_score: float = 50.0  # NIFTY 50 macro trend

    # Final
    composite_score: float = 0.0   # 0–100
    signal: str = ""               # BUY / HOLD / SELL
    confidence: str = ""           # HIGH / MEDIUM / LOW
    summary_reasons: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe(val, default=None):
    """Safely extract a numeric value, handling None / NaN / Inf."""
    if val is None or val == "N/A":
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _compute_rsi(prices, period: int = 14) -> Optional[float]:
    """
    Compute RSI (Relative Strength Index) from a price series.

    Formula:
        RSI = 100 − 100 / (1 + RS)
        RS  = EMA(gains, period) / EMA(losses, period)

    Uses Wilder's smoothing (exponential moving average with α = 1/period).
    """
    if len(prices) < period + 1:
        return None

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Seed with SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Scoring functions — each returns 0-100 (higher = better investment quality)
# ---------------------------------------------------------------------------

def _score_valuation(pe: Optional[float],
                     fwd_pe: Optional[float],
                     pb: Optional[float]) -> float:
    """
    Score valuation 0-100 (higher = cheaper = better).

    Indian market context: NIFTY 50 average PE ≈ 22-24.
    Thresholds are set slightly above global averages.

    Trailing PE scoring:
        PE < 0      → 10  (Negative earnings — loss-making)
        PE < 15     → 90  (Deep value)
        PE 15-22    → 75  (Fairly valued)
        PE 22-30    → 55  (Above average)
        PE 30-40    → 35  (Expensive)
        PE ≥ 40     → 15  (Very expensive)

    Forward PE vs Trailing PE:
        Forward < Trailing × 0.85 → 85  (Strong EPS growth expected)
        Forward < Trailing        → 70  (Moderate growth)
        Forward ≥ Trailing        → 40  (No growth / contraction)

    P/B scoring:
        P/B < 1     → 90  (Below book value — deep value)
        P/B 1-2     → 75
        P/B 2-4     → 55
        P/B 4-8     → 35
        P/B ≥ 8     → 15  (Very expensive relative to assets)
    """
    scores = []

    if pe is not None:
        if pe < 0:
            scores.append(10)
        elif pe < 15:
            scores.append(90)
        elif pe < 22:
            scores.append(75)
        elif pe < 30:
            scores.append(55)
        elif pe < 40:
            scores.append(35)
        else:
            scores.append(15)

    if fwd_pe is not None and pe is not None and fwd_pe > 0:
        if fwd_pe < pe * 0.85:
            scores.append(85)
        elif fwd_pe < pe:
            scores.append(70)
        else:
            scores.append(40)

    if pb is not None:
        if pb < 1:
            scores.append(90)
        elif pb < 2:
            scores.append(75)
        elif pb < 4:
            scores.append(55)
        elif pb < 8:
            scores.append(35)
        else:
            scores.append(15)

    return float(np.mean(scores)) if scores else 50.0


def _score_growth(rev_growth: Optional[float],
                  earn_growth: Optional[float]) -> float:
    """
    Score growth 0-100 (higher = faster growth).

    Revenue growth scoring:
        > 25%   → 90  (Hyper-growth)
        > 15%   → 75  (Strong)
        > 8%    → 60  (Healthy)
        > 0%    → 40  (Slow)
        ≤ 0%    → 15  (Declining)

    Earnings growth scoring:
        > 30%   → 90  (Hyper-growth)
        > 15%   → 75  (Strong)
        > 5%    → 60  (Steady)
        > 0%    → 40  (Slow)
        ≤ 0%    → 15  (Declining)
    """
    scores = []

    if rev_growth is not None:
        if rev_growth > 0.25:
            scores.append(90)
        elif rev_growth > 0.15:
            scores.append(75)
        elif rev_growth > 0.08:
            scores.append(60)
        elif rev_growth > 0:
            scores.append(40)
        else:
            scores.append(15)

    if earn_growth is not None:
        if earn_growth > 0.30:
            scores.append(90)
        elif earn_growth > 0.15:
            scores.append(75)
        elif earn_growth > 0.05:
            scores.append(60)
        elif earn_growth > 0:
            scores.append(40)
        else:
            scores.append(15)

    return float(np.mean(scores)) if scores else 50.0


def _score_profitability(gross_m: Optional[float],
                         op_m: Optional[float],
                         profit_m: Optional[float],
                         roe: Optional[float]) -> float:
    """
    Score profitability 0-100 (higher = more profitable).

    Operating margin scoring:
        > 25%   → 90  (Excellent — pricing power)
        > 15%   → 75  (Good)
        > 8%    → 55  (Average)
        > 0%    → 35  (Thin margins)
        ≤ 0%    → 10  (Unprofitable — RED FLAG)

    Net profit margin scoring:
        > 20%   → 90  (Very profitable)
        > 10%   → 70
        > 5%    → 50
        > 0%    → 30  (Barely profitable)
        ≤ 0%    → 10  (Loss-making — RED FLAG)

    ROE scoring:
        > 25%   → 90  (Exceptional capital efficiency)
        > 15%   → 70  (Good)
        > 8%    → 50  (Average)
        ≤ 8%    → 25  (Poor capital allocation)
    """
    scores = []

    if op_m is not None:
        if op_m > 0.25:
            scores.append(90)
        elif op_m > 0.15:
            scores.append(75)
        elif op_m > 0.08:
            scores.append(55)
        elif op_m > 0:
            scores.append(35)
        else:
            scores.append(10)

    if profit_m is not None:
        if profit_m > 0.20:
            scores.append(90)
        elif profit_m > 0.10:
            scores.append(70)
        elif profit_m > 0.05:
            scores.append(50)
        elif profit_m > 0:
            scores.append(30)
        else:
            scores.append(10)

    if roe is not None:
        if roe > 0.25:
            scores.append(90)
        elif roe > 0.15:
            scores.append(70)
        elif roe > 0.08:
            scores.append(50)
        else:
            scores.append(25)

    return float(np.mean(scores)) if scores else 50.0


def _score_technicals(rsi: Optional[float],
                      price: float,
                      sma50: Optional[float],
                      sma200: Optional[float],
                      pct_from_high: Optional[float]) -> float:
    """
    Score technicals 0-100 (higher = healthier trend).

    RSI-14 scoring (Wilder, momentum oscillator — trend-context-aware):
        RSI < 30 + death cross → 25  (Falling knife — do NOT reward)
        RSI < 30 + golden cross → 85  (Oversold bounce potential)
        RSI 30-40       → 75  (Approaching oversold — buying zone)
        RSI 40-60       → 60  (Neutral — healthy)
        RSI 60-70       → 45  (Getting warm — caution)
        RSI > 70        → 20  (Overbought — pullback risk)

    Price vs 50-day MA:
        Price > 1.05 × 50MA → 70  (Clear uptrend)
        Price > 0.98 × 50MA → 60  (Near MA — consolidating)
        Price < 0.98 × 50MA → 35  (Below MA — downtrend)

    Price vs 200-day MA:
        Price > 1.05 × 200MA → 75  (Long-term bull territory)
        Price > 0.95 × 200MA → 55  (Near long-term average)
        Price < 0.95 × 200MA → 25  (Long-term bear territory)

    Golden / Death Cross (50MA vs 200MA):
        50MA > 200MA → 75  (Golden cross — bullish)
        50MA ≤ 200MA → 30  (Death cross — bearish)

    Distance from 52-week high:
        < 5% from high  → 65  (Momentum / near highs)
        < 15%           → 55  (Moderate pullback)
        < 30%           → 45  (Significant correction)
        ≥ 30%           → 30  (Deep correction — weak trend)
    """
    scores = []

    # Determine trend context for RSI interpretation
    death_cross = (
        sma50 is not None and sma200 is not None
        and sma50 > 0 and sma200 > 0
        and sma50 <= sma200
    )

    # RSI — trend-context-aware
    if rsi is not None:
        if 40 <= rsi <= 60:
            scores.append(60)
        elif 30 <= rsi < 40:
            scores.append(75)
        elif rsi < 30:
            # Death cross + oversold = falling knife, not a bounce
            scores.append(25 if death_cross else 85)
        elif 60 < rsi <= 70:
            scores.append(45)
        else:
            scores.append(20)

    # Price vs 50-day MA
    if sma50 is not None and price > 0 and sma50 > 0:
        ratio = price / sma50
        if ratio > 1.05:
            scores.append(70)
        elif ratio > 0.98:
            scores.append(60)
        else:
            scores.append(35)

    # Price vs 200-day MA
    if sma200 is not None and price > 0 and sma200 > 0:
        ratio = price / sma200
        if ratio > 1.05:
            scores.append(75)
        elif ratio > 0.95:
            scores.append(55)
        else:
            scores.append(25)

    # Golden cross: 50MA > 200MA
    if sma50 is not None and sma200 is not None and sma50 > 0 and sma200 > 0:
        if sma50 > sma200:
            scores.append(75)
        else:
            scores.append(30)

    # Distance from 52-week high
    if pct_from_high is not None:
        if pct_from_high > -5:
            scores.append(65)
        elif pct_from_high > -15:
            scores.append(55)
        elif pct_from_high > -30:
            scores.append(45)
        else:
            scores.append(30)

    raw = float(np.mean(scores)) if scores else 50.0

    # Hard cap: death cross signals a downtrend — never let technical score
    # appear healthy even if other sub-scores (52W distance, etc.) are high
    if death_cross:
        return min(raw, 45.0)
    return raw


def _score_analyst(rec_score: Optional[float],
                   target_mean: Optional[float],
                   price: float,
                   count: int) -> float:
    """
    Score analyst consensus 0-100 (higher = more bullish consensus).

    Recommendation score (1=Strong Buy … 5=Strong Sell):
        ≤ 1.5  → 90  (Strong consensus buy)
        ≤ 2.0  → 80  (Consensus buy)
        ≤ 2.5  → 65  (Moderate buy)
        ≤ 3.0  → 50  (Hold / neutral)
        ≤ 3.5  → 35  (Moderate sell)
        > 3.5  → 15  (Consensus sell)

    Target price upside:
        > 30%   → 90
        > 15%   → 75
        > 5%    → 60
        > -5%   → 45  (Near/at fair value)
        ≤ -5%   → 20  (Overvalued per analysts)

    Analyst coverage count:
        ≥ 20    → 75  (Well followed — high confidence)
        ≥ 10    → 65
        ≥ 5     → 55
        ≥ 1     → 45
    """
    scores = []

    if rec_score is not None:
        if rec_score <= 1.5:
            scores.append(90)
        elif rec_score <= 2.0:
            scores.append(80)
        elif rec_score <= 2.5:
            scores.append(65)
        elif rec_score <= 3.0:
            scores.append(50)
        elif rec_score <= 3.5:
            scores.append(35)
        else:
            scores.append(15)

    if target_mean is not None and price > 0:
        upside = (target_mean - price) / price
        if upside > 0.30:
            scores.append(90)
        elif upside > 0.15:
            scores.append(75)
        elif upside > 0.05:
            scores.append(60)
        elif upside > -0.05:
            scores.append(45)
        else:
            scores.append(20)

    if count >= 20:
        scores.append(75)
    elif count >= 10:
        scores.append(65)
    elif count >= 5:
        scores.append(55)
    elif count >= 1:
        scores.append(45)

    return float(np.mean(scores)) if scores else 50.0


def _score_risk(beta: Optional[float],
                debt_equity: Optional[float]) -> float:
    """
    Score risk 0-100 (higher = LOWER risk = better).

    Beta scoring (volatility relative to NIFTY 50):
        β < 0.5  → 85  (Very low volatility)
        β < 0.8  → 75  (Low volatility)
        β < 1.2  → 60  (Market-like)
        β < 1.5  → 40  (Above market volatility)
        β ≥ 1.5  → 20  (High volatility — risky)

    Debt/Equity scoring:
        D/E < 20   → 85  (Minimal debt — strong balance sheet)
        D/E < 50   → 70  (Conservative leverage)
        D/E < 100  → 50  (Moderate leverage)
        D/E < 200  → 30  (High leverage)
        D/E ≥ 200  → 15  (Very high leverage — risky)
    """
    scores = []

    if beta is not None:
        if beta < 0.5:
            scores.append(85)
        elif beta < 0.8:
            scores.append(75)
        elif beta < 1.2:
            scores.append(60)
        elif beta < 1.5:
            scores.append(40)
        else:
            scores.append(20)

    if debt_equity is not None:
        if debt_equity < 20:
            scores.append(85)
        elif debt_equity < 50:
            scores.append(70)
        elif debt_equity < 100:
            scores.append(50)
        elif debt_equity < 200:
            scores.append(30)
        else:
            scores.append(15)

    return float(np.mean(scores)) if scores else 50.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_stock(ticker: str) -> EvalResult:
    """
    Run full 7-dimension evaluation on an NSE stock.

    Makes 2 yfinance API calls per stock:
      1. Ticker.info  — fundamentals, valuation, analyst consensus
      2. Ticker.history(3mo) — closing prices for RSI computation

    Args:
        ticker: NSE symbol (e.g. 'RELIANCE', 'TCS', 'INFY')

    Returns:
        EvalResult with all sub-scores, composite score, and signal

    Timing: ~2-3 seconds per stock (yfinance latency dependent).
    """
    result = EvalResult(ticker=ticker)

    # Append .NS for yfinance if not present
    yf_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"

    # ── Fetch info ──
    try:
        t = yf.Ticker(yf_ticker)
        info = t.info
    except Exception as e:
        logger.error(f"yfinance info failed for {yf_ticker}: {e}")
        result.signal = "ERROR"
        result.composite_score = 50.0  # Neutral on error
        result.summary_reasons = [f"Failed to fetch data: {e}"]
        return result

    # ── Basic info ──
    result.name = info.get("shortName", ticker)
    result.sector = info.get("sector", "")
    result.industry = info.get("industry", "")
    result.price = _safe(info.get("currentPrice"), 0)
    result.prev_close = _safe(info.get("previousClose"), 0)
    if result.price and result.prev_close:
        result.day_change_pct = ((result.price - result.prev_close) / result.prev_close) * 100

    # ── Valuation ──
    result.trailing_pe = _safe(info.get("trailingPE"))
    result.forward_pe = _safe(info.get("forwardPE"))
    result.price_to_book = _safe(info.get("priceToBook"))
    result.dividend_yield = _safe(info.get("dividendYield"))
    result.market_cap = _safe(info.get("marketCap"))

    # ── Growth ──
    result.revenue_growth = _safe(info.get("revenueGrowth"))
    result.earnings_growth = _safe(info.get("earningsGrowth"))
    result.total_revenue = _safe(info.get("totalRevenue"))

    # ── Profitability ──
    result.gross_margin = _safe(info.get("grossMargins"))
    result.operating_margin = _safe(info.get("operatingMargins"))
    result.profit_margin = _safe(info.get("profitMargins"))
    result.return_on_equity = _safe(info.get("returnOnEquity"))

    # ── Balance sheet / Risk ──
    result.debt_to_equity = _safe(info.get("debtToEquity"))
    result.free_cashflow = _safe(info.get("freeCashflow"))
    result.beta = _safe(info.get("beta"))

    # ── 52-week & Moving Averages (from yfinance info) ──
    result.week52_high = _safe(info.get("fiftyTwoWeekHigh"))
    result.week52_low = _safe(info.get("fiftyTwoWeekLow"))
    result.sma_50 = _safe(info.get("fiftyDayAverage"))
    result.sma_200 = _safe(info.get("twoHundredDayAverage"))
    result.avg_volume = int(_safe(info.get("averageVolume"), 0))

    if result.price and result.week52_high:
        result.pct_from_52w_high = (
            (result.price - result.week52_high) / result.week52_high
        ) * 100

    # ── Analyst consensus ──
    result.recommendation = info.get("recommendationKey", "")
    result.recommendation_score = _safe(info.get("recommendationMean"))
    result.target_mean = _safe(info.get("targetMeanPrice"))
    result.target_high = _safe(info.get("targetHighPrice"))
    result.target_low = _safe(info.get("targetLowPrice"))
    result.analyst_count = int(_safe(info.get("numberOfAnalystOpinions"), 0))

    # Analyst breakdown from recommendations table
    try:
        recs = t.recommendations
        if recs is not None and len(recs) > 0:
            latest = recs.iloc[0]
            result.analyst_strong_buy = int(latest.get("strongBuy", 0))
            result.analyst_buy = int(latest.get("buy", 0))
            result.analyst_hold = int(latest.get("hold", 0))
            result.analyst_sell = int(latest.get("sell", 0) + latest.get("strongSell", 0))
    except Exception:
        pass

    # ── RSI from 3-month historical prices ──
    try:
        hist = t.history(period="3mo")
        if len(hist) >= 15:
            closes = hist["Close"].values
            result.rsi_14 = _compute_rsi(closes, 14)
    except Exception as e:
        logger.warning(f"History fetch failed for {yf_ticker}: {e}")

    # ── Compute sub-scores ──
    result.valuation_score = _score_valuation(
        result.trailing_pe, result.forward_pe, result.price_to_book
    )
    result.growth_score = _score_growth(
        result.revenue_growth, result.earnings_growth
    )
    result.profitability_score = _score_profitability(
        result.gross_margin, result.operating_margin,
        result.profit_margin, result.return_on_equity
    )
    result.technical_score = _score_technicals(
        result.rsi_14, result.price, result.sma_50,
        result.sma_200, result.pct_from_52w_high
    )
    result.analyst_score = _score_analyst(
        result.recommendation_score, result.target_mean,
        result.price, result.analyst_count
    )
    result.risk_score = _score_risk(result.beta, result.debt_to_equity)
    result.market_context_score = _fetch_market_context_score()

    # ── Composite score (weighted average of 7 dimensions) ──
    result.composite_score = (
        result.valuation_score * DIMENSION_WEIGHTS["valuation"]
        + result.growth_score * DIMENSION_WEIGHTS["growth"]
        + result.profitability_score * DIMENSION_WEIGHTS["profitability"]
        + result.technical_score * DIMENSION_WEIGHTS["technical"]
        + result.analyst_score * DIMENSION_WEIGHTS["analyst"]
        + result.risk_score * DIMENSION_WEIGHTS["risk"]
        + result.market_context_score * DIMENSION_WEIGHTS["market_context"]
    )

    # ── Signal & Confidence ──
    if result.composite_score >= 70:
        result.signal = "BUY"
    elif result.composite_score >= 50:
        result.signal = "HOLD"
    else:
        result.signal = "SELL"

    # Confidence = how aligned are the sub-scores? Low divergence → HIGH
    spread = max(
        abs(result.valuation_score - result.composite_score),
        abs(result.technical_score - result.composite_score),
        abs(result.growth_score - result.composite_score),
    )
    if spread < 15:
        result.confidence = "HIGH"
    elif spread < 25:
        result.confidence = "MEDIUM"
    else:
        result.confidence = "LOW"

    # ── Human-readable reasons ──
    reasons = []

    # Valuation
    if result.trailing_pe is not None:
        if result.trailing_pe < 0:
            reasons.append(f"Loss-making (negative PE)")
        elif result.trailing_pe < 15:
            reasons.append(f"Attractively valued (PE {result.trailing_pe:.1f})")
        elif result.trailing_pe > 35:
            reasons.append(f"Expensive valuation (PE {result.trailing_pe:.1f})")

    # Growth
    if result.revenue_growth is not None:
        if result.revenue_growth > 0.15:
            reasons.append(f"Strong revenue growth ({result.revenue_growth * 100:.1f}%)")
        elif result.revenue_growth < 0:
            reasons.append(f"Revenue declining ({result.revenue_growth * 100:.1f}%)")

    if result.earnings_growth is not None:
        if result.earnings_growth < -0.20:
            reasons.append(f"Earnings collapsing ({result.earnings_growth * 100:.1f}%)")

    # Profitability
    if result.profit_margin is not None:
        if result.profit_margin < 0:
            reasons.append(f"Net loss (margin {result.profit_margin * 100:.1f}%)")
        elif result.profit_margin > 0.15:
            reasons.append(f"Highly profitable (margin {result.profit_margin * 100:.1f}%)")

    if result.return_on_equity is not None:
        if result.return_on_equity < 0:
            reasons.append(f"Negative ROE ({result.return_on_equity * 100:.1f}%)")
        elif result.return_on_equity > 0.20:
            reasons.append(f"Strong ROE ({result.return_on_equity * 100:.1f}%)")

    # Technicals
    if result.rsi_14 is not None:
        if result.rsi_14 < 30:
            reasons.append(f"Oversold (RSI {result.rsi_14:.0f})")
        elif result.rsi_14 > 70:
            reasons.append(f"Overbought (RSI {result.rsi_14:.0f})")

    if result.sma_50 and result.sma_200:
        if result.sma_50 > result.sma_200:
            reasons.append("Golden cross (50MA > 200MA)")
        else:
            reasons.append("Death cross (50MA < 200MA)")

    if result.pct_from_52w_high is not None:
        if result.pct_from_52w_high > -5:
            reasons.append("Trading near 52-week high")
        elif result.pct_from_52w_high < -25:
            reasons.append(f"Down {abs(result.pct_from_52w_high):.0f}% from 52W high")

    # Analyst
    if result.recommendation:
        reasons.append(f"Analyst: {result.recommendation.replace('_', ' ').title()}")
    if result.target_mean and result.price:
        upside = ((result.target_mean - result.price) / result.price) * 100
        if abs(upside) > 5:
            reasons.append(f"Target ₹{result.target_mean:,.0f} ({upside:+.0f}% upside)")

    # Risk
    if result.beta is not None and result.beta > 1.5:
        reasons.append(f"High volatility (β={result.beta:.2f})")
    if result.debt_to_equity is not None and result.debt_to_equity > 100:
        reasons.append(f"High debt (D/E={result.debt_to_equity:.0f})")

    # Market context
    mcs = result.market_context_score
    if mcs >= 65:
        reasons.append(f"Bullish market (NIFTY score {mcs:.0f}/100)")
    elif mcs <= 40:
        reasons.append(f"Bearish market — macro headwind (NIFTY score {mcs:.0f}/100)")

    result.summary_reasons = reasons
    return result


def evaluate_batch(tickers: List[str]) -> List[EvalResult]:
    """
    Evaluate multiple stocks sequentially.

    Args:
        tickers: List of NSE symbols

    Returns:
        List of EvalResult (one per ticker)

    Timing: ~2-3 seconds per stock.
    """
    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        logger.info(f"  Quality eval [{i + 1}/{total}]: {ticker}")
        t0 = time.time()
        try:
            result = evaluate_stock(ticker)
            elapsed = time.time() - t0
            logger.info(f"    → {result.signal} {result.composite_score:.0f}/100 "
                        f"(V:{result.valuation_score:.0f} G:{result.growth_score:.0f} "
                        f"P:{result.profitability_score:.0f} T:{result.technical_score:.0f} "
                        f"A:{result.analyst_score:.0f} R:{result.risk_score:.0f}) "
                        f"[{elapsed:.1f}s]")
            results.append(result)
        except Exception as e:
            logger.error(f"    → ERROR: {e}")
            fallback = EvalResult(ticker=ticker, signal="ERROR",
                                  composite_score=50.0,
                                  summary_reasons=[f"Evaluation failed: {e}"])
            results.append(fallback)
    return results


def quick_quality_check(ticker: str) -> Tuple[float, str, List[str]]:
    """
    Quick quality gate check for pipeline integration.

    Returns:
        Tuple of (composite_score, signal, summary_reasons)

    Usage in trigger_batch:
        score, signal, reasons = quick_quality_check("OLAELEC")
        if score < QUALITY_GATE_MIN:
            logger.info(f"REJECTED: {ticker} quality {score:.0f} < {QUALITY_GATE_MIN}")
    """
    result = evaluate_stock(ticker)
    return (result.composite_score, result.signal, result.summary_reasons)


def format_eval_summary(r: EvalResult) -> str:
    """Format a concise one-line summary for logging."""
    if r.signal == "ERROR":
        return f"❌ {r.ticker}: Error"

    sig_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(r.signal, "⚪")
    return (
        f"{sig_emoji} {r.ticker} ({r.name}) | {r.signal} {r.composite_score:.0f}/100 "
        f"[V:{r.valuation_score:.0f} G:{r.growth_score:.0f} P:{r.profitability_score:.0f} "
        f"T:{r.technical_score:.0f} A:{r.analyst_score:.0f} R:{r.risk_score:.0f} "
        f"M:{r.market_context_score:.0f}] "
        f"| {r.confidence} confidence"
    )


def format_telegram_message(r: EvalResult) -> str:
    """Format EvalResult as a Telegram-friendly message."""
    if r.signal == "ERROR":
        return (f"❌ Could not evaluate {r.ticker}: "
                f"{r.summary_reasons[0] if r.summary_reasons else 'Unknown error'}")

    sig_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(r.signal, "⚪")

    # Market cap formatting
    mcap_str = "N/A"
    if r.market_cap:
        if r.market_cap >= 1e12:
            mcap_str = f"₹{r.market_cap / 1e12:.2f}T"
        elif r.market_cap >= 1e9:
            mcap_str = f"₹{r.market_cap / 1e9:.1f}B"
        else:
            mcap_str = f"₹{r.market_cap / 1e6:.0f}M"

    chg_arrow = "📈" if r.day_change_pct >= 0 else "📉"

    lines = [
        f"{sig_emoji} *{r.ticker}* — {r.name}",
        f"Signal: *{r.signal}* | Score: *{r.composite_score:.0f}/100* | Confidence: {r.confidence}",
        f"Sector: {r.sector} | {r.industry}",
        "",
        f"{chg_arrow} *Price: ₹{r.price:,.2f}* ({r.day_change_pct:+.2f}%)",
    ]

    if r.week52_high:
        lines.append(
            f"52W Range: ₹{r.week52_low:,.0f} — ₹{r.week52_high:,.0f} "
            f"({r.pct_from_52w_high:+.1f}% from high)"
        )
    lines.append(f"Market Cap: {mcap_str}")

    # Score breakdown with bar chart
    lines += [
        "",
        "📊 *Score Breakdown*",
        f"  Valuation:     {'█' * int(r.valuation_score / 10)}{'░' * (10 - int(r.valuation_score / 10))} {r.valuation_score:.0f}",
        f"  Growth:        {'█' * int(r.growth_score / 10)}{'░' * (10 - int(r.growth_score / 10))} {r.growth_score:.0f}",
        f"  Profitability: {'█' * int(r.profitability_score / 10)}{'░' * (10 - int(r.profitability_score / 10))} {r.profitability_score:.0f}",
        f"  Technicals:    {'█' * int(r.technical_score / 10)}{'░' * (10 - int(r.technical_score / 10))} {r.technical_score:.0f}",
        f"  Analyst:       {'█' * int(r.analyst_score / 10)}{'░' * (10 - int(r.analyst_score / 10))} {r.analyst_score:.0f}",
        f"  Risk:          {'█' * int(r.risk_score / 10)}{'░' * (10 - int(r.risk_score / 10))} {r.risk_score:.0f}",
    ]

    # Key metrics
    lines.append("")
    lines.append("📈 *Key Metrics*")

    pe_str = f"PE: {r.trailing_pe:.1f}" if r.trailing_pe else "PE: N/A"
    fpe_str = f"Fwd PE: {r.forward_pe:.1f}" if r.forward_pe else ""
    pb_str = f"P/B: {r.price_to_book:.2f}" if r.price_to_book else ""
    lines.append(f"  {pe_str}  {fpe_str}  {pb_str}")

    rev_str = f"Rev Growth: {r.revenue_growth * 100:.1f}%" if r.revenue_growth is not None else ""
    earn_str = f"Earn Growth: {r.earnings_growth * 100:.1f}%" if r.earnings_growth is not None else ""
    lines.append(f"  {rev_str}  {earn_str}")

    om_str = f"Op Margin: {r.operating_margin * 100:.1f}%" if r.operating_margin is not None else ""
    pm_str = f"Net Margin: {r.profit_margin * 100:.1f}%" if r.profit_margin is not None else ""
    lines.append(f"  {om_str}  {pm_str}")

    rsi_str = f"RSI(14): {r.rsi_14:.0f}" if r.rsi_14 else "RSI: N/A"
    ma_str = ""
    if r.sma_50 and r.sma_200:
        ma_str = f"50MA: ₹{r.sma_50:,.0f} | 200MA: ₹{r.sma_200:,.0f}"
    lines.append(f"  {rsi_str}  {ma_str}")

    beta_str = f"Beta: {r.beta:.2f}" if r.beta else ""
    de_str = f"D/E: {r.debt_to_equity:.1f}" if r.debt_to_equity is not None else ""
    lines.append(f"  {beta_str}  {de_str}")

    # Analyst
    if r.analyst_count > 0:
        lines.append("")
        lines.append(f"🏦 *Analyst Consensus* ({r.analyst_count} analysts)")
        if any([r.analyst_strong_buy, r.analyst_buy, r.analyst_hold, r.analyst_sell]):
            lines.append(
                f"  Strong Buy: {r.analyst_strong_buy} | Buy: {r.analyst_buy} "
                f"| Hold: {r.analyst_hold} | Sell: {r.analyst_sell}"
            )
        if r.target_mean and r.price:
            upside = ((r.target_mean - r.price) / r.price) * 100
            tl = f"₹{r.target_low:,.0f}" if r.target_low else "—"
            th = f"₹{r.target_high:,.0f}" if r.target_high else "—"
            lines.append(f"  Target: {tl} — ₹{r.target_mean:,.0f} — {th} ({upside:+.1f}%)")

    # Key observations
    if r.summary_reasons:
        lines.append("")
        lines.append("💡 *Key Observations*")
        for reason in r.summary_reasons:
            lines.append(f"  • {reason}")

    return "\n".join([line for line in lines if line is not None])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["RELIANCE"]

    for ticker in tickers:
        print(f"\n{'=' * 60}")
        result = evaluate_stock(ticker)
        print(format_telegram_message(result))

        if result.composite_score < QUALITY_GATE_MIN:
            print(f"\n⛔ QUALITY GATE: REJECTED (score {result.composite_score:.0f} < {QUALITY_GATE_MIN})")
        print(f"{'=' * 60}")
