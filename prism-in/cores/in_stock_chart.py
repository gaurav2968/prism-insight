"""
India Stock Chart Generation Module

Generates charts for Indian stock (NSE/BSE) analysis reports using yfinance data.
Chart types:
1. Price Chart (Candlestick + MA) - Technical analysis
2. Institutional Holdings Chart - Promoter/FII/DII/Public breakdown
3. Technical Indicators Chart - RSI + MACD

All charts are returned as matplotlib figures or base64 HTML img tags.
Currency: INR (₹)
"""

import logging
from io import BytesIO
import base64
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# =============================================================================
# Color Palettes
# =============================================================================

PRIMARY_COLORS = ["#0066cc", "#ff9500", "#00cc99", "#cc3300", "#6600cc"]

# Chart-specific colors (Indian convention: green=up, red=down)
UP_COLOR = "#26a69a"       # Green for up
DOWN_COLOR = "#ef5350"     # Red for down

# Institutional chart colors (India-specific breakdown)
INST_COLORS = {
    "promoter": "#FF6F00",     # Deep Orange - Promoters
    "fii": "#1565C0",          # Blue - Foreign Institutional
    "dii": "#2E7D32",          # Green - Domestic Institutional
    "public": "#7B1FA2",       # Purple - Public
    "other": "#9E9E9E"         # Gray
}

RSI_COLOR = "#9C27B0"
MACD_COLOR = "#2196F3"
SIGNAL_COLOR = "#FF9800"
HIST_UP_COLOR = "#26a69a"
HIST_DOWN_COLOR = "#ef5350"


# =============================================================================
# Utility Functions
# =============================================================================

def figure_to_base64_html(
    fig,
    chart_name: str = "chart",
    width: int = 900,
    dpi: int = 80,
    image_format: str = 'jpg'
) -> Optional[str]:
    """Convert a matplotlib figure to base64 HTML img tag."""
    try:
        buffer = BytesIO()
        save_kwargs = {
            'format': image_format,
            'bbox_inches': 'tight',
            'dpi': dpi
        }
        if image_format.lower() == 'png':
            save_kwargs['transparent'] = False
            save_kwargs['facecolor'] = 'white'

        fig.savefig(buffer, **save_kwargs)
        plt.close(fig)
        buffer.seek(0)

        if image_format.lower() in ['jpg', 'jpeg']:
            try:
                from PIL import Image
                img = Image.open(buffer)
                new_buffer = BytesIO()
                img.save(new_buffer, format='JPEG', quality=85, optimize=True)
                buffer = new_buffer
                buffer.seek(0)
            except ImportError:
                pass

        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content_type = 'image/jpeg' if image_format.lower() in ['jpg', 'jpeg'] else f'image/{image_format.lower()}'
        return f'<img src="data:{content_type};base64,{img_str}" alt="{chart_name}" width="{width}" />'
    except Exception as e:
        logger.warning(f"Failed to convert figure to base64: {e}")
        return None


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def format_inr(value: float) -> str:
    """Format value in Indian numbering system (lakhs/crores)."""
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value / 1e5:,.2f} L"
    else:
        return f"₹{value:,.2f}"


# =============================================================================
# Chart Creation Functions
# =============================================================================

def create_in_price_chart(
    ticker: str,
    company_name: str,
    hist_df: pd.DataFrame
) -> Optional[plt.Figure]:
    """
    Create candlestick price chart with volume and moving averages.

    Args:
        ticker: NSE stock ticker (e.g., "RELIANCE")
        company_name: Company name
        hist_df: DataFrame with OHLCV data from yfinance (.NS suffix)

    Returns:
        matplotlib figure or None
    """
    try:
        import mplfinance as mpf

        if hist_df is None or hist_df.empty:
            return None

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in hist_df.columns:
                return None

        df = hist_df[required_cols].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()

        ohlc_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        mc = mpf.make_marketcolors(
            up=UP_COLOR, down=DOWN_COLOR,
            edge='inherit', wick='inherit',
            volume={'up': UP_COLOR, 'down': DOWN_COLOR}
        )
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#e0e0e0')

        additional_plots = []
        if not df['MA20'].isna().all():
            additional_plots.append(mpf.make_addplot(df['MA20'], color='#ff9500', width=1))
        if not df['MA60'].isna().all():
            additional_plots.append(mpf.make_addplot(df['MA60'], color='#0066cc', width=1.5))
        if not df['MA120'].isna().all():
            additional_plots.append(mpf.make_addplot(df['MA120'], color='#cc3300', width=1.5, linestyle='--'))

        fig, axes = mpf.plot(
            ohlc_df, type='candle', style=style,
            title=f"{company_name} ({ticker}.NS) - Price Chart",
            ylabel='Price (₹)',
            volume=True, figsize=(12, 8), tight_layout=True,
            addplot=additional_plots if additional_plots else None,
            panel_ratios=(4, 1), returnfig=True
        )

        # Annotations
        max_idx = df['Close'].idxmax()
        min_idx = df['Close'].idxmin()
        last_idx = df.index[-1]
        ax1 = axes[0]
        bbox_props = dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="none", alpha=0.9)

        ax1.annotate(
            f"High: ₹{df.loc[max_idx, 'Close']:,.2f}",
            xy=(max_idx, df.loc[max_idx, 'Close']),
            xytext=(0, 15), textcoords='offset points',
            ha='center', va='bottom', bbox=bbox_props, fontsize=9
        )
        ax1.annotate(
            f"Low: ₹{df.loc[min_idx, 'Close']:,.2f}",
            xy=(min_idx, df.loc[min_idx, 'Close']),
            xytext=(0, -15), textcoords='offset points',
            ha='center', va='top', bbox=bbox_props, fontsize=9
        )
        ax1.annotate(
            f"Current: ₹{df.loc[last_idx, 'Close']:,.2f}",
            xy=(last_idx, df.loc[last_idx, 'Close']),
            xytext=(15, 0), textcoords='offset points',
            ha='left', va='center', bbox=bbox_props, fontsize=9
        )

        if additional_plots:
            legend_labels = []
            if not df['MA20'].isna().all(): legend_labels.append('MA20')
            if not df['MA60'].isna().all(): legend_labels.append('MA60')
            if not df['MA120'].isna().all(): legend_labels.append('MA120')
            if legend_labels:
                ax1.legend(legend_labels, loc='upper left', fontsize=8)

        return fig

    except Exception as e:
        logger.warning(f"Failed to create India price chart: {e}")
        return None


def create_in_institutional_chart(
    ticker: str,
    company_name: str,
    major_holders: Optional[pd.DataFrame] = None,
    institutional_holders: Optional[pd.DataFrame] = None
) -> Optional[plt.Figure]:
    """
    Create institutional holdings chart for Indian stock.

    India-specific: Shows Promoter/FII/DII/Public breakdown.
    """
    try:
        if major_holders is None or institutional_holders is None:
            import yfinance as yf
            stock = yf.Ticker(f"{ticker}.NS")
            if major_holders is None:
                major_holders = stock.major_holders
            if institutional_holders is None:
                institutional_holders = stock.institutional_holders

        if major_holders is None or major_holders.empty:
            logger.warning(f"No major holders data for {ticker}.NS")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{company_name} ({ticker}.NS) - Shareholding Pattern", fontsize=14, fontweight='bold')

        # Parse major_holders
        ownership_data = {}
        for _, row in major_holders.iterrows():
            value = row.iloc[0]
            desc = row.iloc[1] if len(row) > 1 else ""

            if isinstance(value, str) and '%' in value:
                try:
                    pct = float(value.replace('%', '').strip())
                except ValueError:
                    continue
            elif isinstance(value, (int, float)):
                pct = float(value) * 100 if value <= 1 else float(value)
            else:
                continue

            desc_lower = str(desc).lower()
            if 'institution' in desc_lower:
                ownership_data['Institutional'] = pct
            elif 'insider' in desc_lower:
                ownership_data['Promoter & Insider'] = pct

        total = sum(ownership_data.values())
        if total < 100:
            ownership_data['Public & Other'] = 100 - total

        if not ownership_data or all(v == 0 for v in ownership_data.values()):
            return None

        # Pie chart
        labels = list(ownership_data.keys())
        sizes = list(ownership_data.values())
        color_map = {
            'Promoter & Insider': INST_COLORS['promoter'],
            'Institutional': INST_COLORS['fii'],
            'Public & Other': INST_COLORS['public'],
        }
        colors = [color_map.get(k, INST_COLORS['other']) for k in labels]

        def autopct_func(pct):
            return f'{pct:.1f}%' if pct >= 2 else ''

        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, colors=colors,
            autopct=autopct_func, startangle=90,
            explode=[0.02] * len(sizes), shadow=False
        )
        for t in autotexts:
            t.set_color('white')
            t.set_fontweight('bold')
            t.set_fontsize(10)
        ax1.set_title("Shareholding Pattern", fontsize=12, pad=10)

        # Bar chart - top institutional holders
        if institutional_holders is not None and not institutional_holders.empty:
            top = institutional_holders.head(10).copy()
            names = top['Holder'].tolist() if 'Holder' in top.columns else top.index.tolist()
            pcts = []
            for p in (top.get('% Out', top.get('pctHeld', pd.Series([0]*len(top))))):
                if isinstance(p, (int, float)):
                    pcts.append(float(p) * 100 if p <= 1 else float(p))
                else:
                    pcts.append(0)

            names = [n[:25] + '...' if len(str(n)) > 28 else n for n in names]
            y_pos = range(len(names))
            bars = ax2.barh(y_pos, pcts, color=INST_COLORS['fii'], alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel('% of Outstanding', fontsize=10)
            ax2.set_title('Top 10 Institutional Holders', fontsize=12, pad=10)
            for bar, pct in zip(bars, pcts):
                if pct > 0:
                    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                             f'{pct:.2f}%', va='center', fontsize=8)
            ax2.set_xlim(0, max(pcts) * 1.15 if pcts else 10)
            ax2.grid(axis='x', linestyle='--', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No institutional holder data available',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_axis_off()

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.warning(f"Failed to create India institutional chart: {e}")
        return None


def create_in_technical_indicators_chart(
    ticker: str,
    company_name: str,
    hist_df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9
) -> Optional[plt.Figure]:
    """Create technical indicators chart with RSI and MACD for Indian stock."""
    try:
        if hist_df is None or hist_df.empty or 'Close' not in hist_df.columns:
            return None

        df = hist_df.copy()
        if len(df) > 126:
            df = df.tail(126)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df['RSI'] = calculate_rsi(df['Close'], rsi_period)
        df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(
            df['Close'], macd_fast, macd_slow, macd_signal
        )

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10),
                                            gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(f"{company_name} ({ticker}.NS) - Technical Indicators", fontsize=14, fontweight='bold')

        # Price
        ax1.plot(df.index, df['Close'], color=PRIMARY_COLORS[0], linewidth=1.5, label='Close')
        ax1.fill_between(df.index, df['Close'], alpha=0.1, color=PRIMARY_COLORS[0])
        if len(df) >= 20:
            ma20 = df['Close'].rolling(window=20).mean()
            ax1.plot(df.index, ma20, color='#ff9500', linewidth=1, linestyle='--', label='MA20', alpha=0.7)
        ax1.set_ylabel('Price (₹)', fontsize=10)
        ax1.set_title('Price', fontsize=11, loc='left')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        last_price = df['Close'].iloc[-1]
        ax1.annotate(f'₹{last_price:,.2f}', xy=(df.index[-1], last_price),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=PRIMARY_COLORS[0])

        # RSI
        ax2.plot(df.index, df['RSI'], color=RSI_COLOR, linewidth=1.5)
        ax2.fill_between(df.index, df['RSI'], alpha=0.1, color=RSI_COLOR)
        ax2.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax2.fill_between(df.index, 70, 100, alpha=0.1, color='red')
        ax2.fill_between(df.index, 0, 30, alpha=0.1, color='green')
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.set_title(f'RSI ({rsi_period})', fontsize=11, loc='left')
        ax2.grid(True, linestyle='--', alpha=0.3)
        current_rsi = df['RSI'].iloc[-1]
        rsi_status = "Overbought" if current_rsi >= 70 else "Oversold" if current_rsi <= 30 else "Neutral"
        rsi_color = 'red' if current_rsi >= 70 else 'green' if current_rsi <= 30 else 'gray'
        ax2.text(0.98, 0.95, f'RSI: {current_rsi:.1f} ({rsi_status})',
                transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color=rsi_color, fontweight='bold')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # MACD
        ax3.plot(df.index, df['MACD'], color=MACD_COLOR, linewidth=1.5, label='MACD')
        ax3.plot(df.index, df['Signal'], color=SIGNAL_COLOR, linewidth=1.5, label='Signal')
        hist_colors = [HIST_UP_COLOR if h >= 0 else HIST_DOWN_COLOR for h in df['Histogram']]
        ax3.bar(df.index, df['Histogram'], color=hist_colors, alpha=0.6, width=0.8)
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('MACD', fontsize=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_title(f'MACD ({macd_fast}/{macd_slow}/{macd_signal})', fontsize=11, loc='left')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, linestyle='--', alpha=0.3)
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['Signal'].iloc[-1]
        macd_status = "Bullish" if current_macd > current_signal else "Bearish"
        macd_color = 'green' if current_macd > current_signal else 'red'
        ax3.text(0.98, 0.95, f'MACD: {macd_status}',
                transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color=macd_color, fontweight='bold')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.warning(f"Failed to create India technical indicators chart: {e}")
        return None


# =============================================================================
# Wrapper Functions
# =============================================================================

def get_in_price_chart_html(ticker, company_name, hist_df, width=900, dpi=80) -> Optional[str]:
    """Generate price chart and return as base64 HTML."""
    fig = create_in_price_chart(ticker, company_name, hist_df)
    if fig is None:
        return None
    return figure_to_base64_html(fig, f"{ticker}.NS Price Chart", width, dpi, 'jpg')


def get_in_institutional_chart_html(ticker, company_name, major_holders=None, institutional_holders=None, width=900, dpi=80) -> Optional[str]:
    """Generate institutional holdings chart and return as base64 HTML."""
    fig = create_in_institutional_chart(ticker, company_name, major_holders, institutional_holders)
    if fig is None:
        return None
    return figure_to_base64_html(fig, f"{ticker}.NS Shareholding", width, dpi, 'jpg')


def get_in_technical_chart_html(ticker, company_name, hist_df, width=900, dpi=80) -> Optional[str]:
    """Generate technical indicators chart and return as base64 HTML."""
    fig = create_in_technical_indicators_chart(ticker, company_name, hist_df)
    if fig is None:
        return None
    return figure_to_base64_html(fig, f"{ticker}.NS Technical Indicators", width, dpi, 'jpg')


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf

    ticker = "RELIANCE"
    company_name = "Reliance Industries Limited"

    print(f"Testing India stock charts for {ticker}.NS...")

    stock = yf.Ticker(f"{ticker}.NS")
    hist = stock.history(period="1y")

    if not hist.empty:
        html = get_in_price_chart_html(ticker, company_name, hist)
        print(f"Price chart: {'Generated' if html else 'Failed'}")

        html = get_in_institutional_chart_html(ticker, company_name)
        print(f"Institutional chart: {'Generated' if html else 'Failed'}")

        html = get_in_technical_chart_html(ticker, company_name, hist)
        print(f"Technical chart: {'Generated' if html else 'Failed'}")
    else:
        print("No historical data available")
