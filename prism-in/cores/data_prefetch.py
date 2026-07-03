"""
Data Prefetch Module for India Stock Analysis

Pre-fetches Indian stock data using INDataClient (yfinance + nsetools) to inject
into agent instructions, eliminating the need for MCP server tool calls.

This reduces token usage by avoiding MCP tool call round-trips for predictable,
parameterized data fetches (OHLCV, holder info, market indices).
"""

import logging
from pathlib import Path
import importlib.util

import pandas as pd

logger = logging.getLogger(__name__)


def _df_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convert DataFrame to markdown table string."""
    if df is None or df.empty:
        return f"### {title}\n\n_No data available_\n" if title else "_No data available_\n"

    result = ""
    if title:
        result += f"### {title}\n\n"

    result += df.to_markdown(index=True) + "\n"
    return result


def _fmt_inr(val, fmt_type="default"):
    """Smart format values with INR currency, percentages, or ratios.

    Args:
        val: Value to format
        fmt_type: One of 'currency', 'percent', 'percent_raw', 'ratio',
                  'number', 'default'
            - percent: Value is in decimal form (0.22 = 22%). Multiplies by 100.
            - percent_raw: Value is already in percentage form (0.96 = 0.96%).
              Used for yfinance fields like dividendYield that return percentage
              values directly (NOT decimal fractions).

    Returns:
        Formatted string
    """
    if val is None or (isinstance(val, (int, float)) and val == 0):
        return "N/A"
    if fmt_type == "currency":
        if abs(val) >= 1e12:
            return f"₹{val / 1e12:.2f} Trillion"
        elif abs(val) >= 1e9:
            return f"₹{val / 1e9:.2f} Billion"
        elif abs(val) >= 1e7:
            return f"₹{val / 1e7:.2f} Crore"
        elif abs(val) >= 1e5:
            return f"₹{val / 1e5:.2f} Lakh"
        return f"₹{val:,.2f}"
    elif fmt_type == "percent":
        # Value is a decimal fraction (e.g., 0.22 → 22.00%)
        return f"{val * 100:.2f}%" if abs(val) < 1 else f"{val:.2f}%"
    elif fmt_type == "percent_raw":
        # Value is already a percentage (e.g., 0.96 → 0.96%)
        # Used for yfinance dividendYield which returns percentage directly
        return f"{val:.2f}%"
    elif fmt_type == "ratio":
        return f"{val:.2f}"
    elif fmt_type == "number":
        if abs(val) >= 1e9:
            return f"{val / 1e9:.2f} Billion"
        elif abs(val) >= 1e7:
            return f"{val / 1e7:.2f} Crore"
        elif abs(val) >= 1e5:
            return f"{val / 1e5:.2f} Lakh"
        return f"{val:,.0f}"
    return str(val)


def _fmt_financial_value(val):
    """Format financial statement values (large INR numbers) into readable form.

    Converts raw values like 71538500000 into '₹7,153.85Cr' so LLMs
    don't misinterpret the scale.

    Args:
        val: Raw financial value (int or float)

    Returns:
        Human-readable INR string
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if isinstance(val, (int, float)):
        if val == 0:
            return "₹0"
        return _fmt_inr(val, "currency")
    return str(val)


def build_key_reference_data(ticker: str, info: dict) -> str:
    """Build a concise KEY REFERENCE DATA block with ground-truth numbers.

    This block is injected into ALL agent prompts as a single source of truth
    for the most critical data points, preventing LLM hallucination of key
    figures like current price, 52-week range, dividend yield, etc.

    Args:
        ticker: NSE stock ticker
        info: yfinance info dictionary

    Returns:
        Markdown formatted reference data block
    """
    if not info:
        return ""

    price = info.get('regularMarketPrice') or info.get('currentPrice')
    prev_close = info.get('previousClose')

    lines = [
        "## ⚠️ KEY REFERENCE DATA — USE THESE EXACT VALUES ⚠️",
        "",
        "**CRITICAL**: The values below are verified ground-truth data.",
        "You MUST use these exact numbers in your report. Do NOT round, approximate, or recalculate them.",
        "",
        "| Metric | Verified Value |",
        "|--------|---------------|",
        f"| **Current Price** | {_fmt_inr(price, 'currency')} |",
        f"| **Previous Close** | {_fmt_inr(prev_close, 'currency')} |",
        f"| **52-Week High** | {_fmt_inr(info.get('fiftyTwoWeekHigh'), 'currency')} |",
        f"| **52-Week Low** | {_fmt_inr(info.get('fiftyTwoWeekLow'), 'currency')} |",
        f"| **Market Cap** | {_fmt_inr(info.get('marketCap'), 'currency')} |",
        f"| **Trailing P/E** | {_fmt_inr(info.get('trailingPE'), 'ratio')} |",
        f"| **Forward P/E** | {_fmt_inr(info.get('forwardPE'), 'ratio')} |",
        f"| **P/B** | {_fmt_inr(info.get('priceToBook'), 'ratio')} |",
        f"| **P/S** | {_fmt_inr(info.get('priceToSalesTrailing12Months'), 'ratio')} |",
        f"| **Dividend Yield** | {_fmt_inr(info.get('dividendYield'), 'percent_raw')} |",
        f"| **Profit Margin** | {_fmt_inr(info.get('profitMargins'), 'percent')} |",
        f"| **ROE** | {_fmt_inr(info.get('returnOnEquity'), 'percent')} |",
        f"| **Revenue (TTM)** | {_fmt_inr(info.get('totalRevenue'), 'currency')} |",
        f"| **Net Income** | {_fmt_inr(info.get('netIncomeToCommon'), 'currency')} |",
        f"| **Beta** | {_fmt_inr(info.get('beta'), 'ratio')} |",
        "",
        "**NOTE**: 'Current Price' is the LIVE market price. 'Previous Close' is yesterday's close.",
        "**NOTE**: Dividend Yield is already in percentage form (e.g., 0.96% means 0.96%, NOT 96%).",
        "",
    ]

    return "\n".join(lines)


def _get_in_data_client():
    """Get INDataClient instance."""
    _current_dir = Path(__file__).parent
    _client_path = _current_dir / "in_data_client.py"
    spec = importlib.util.spec_from_file_location("in_data_client_local", _client_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.INDataClient()


def prefetch_in_stock_ohlcv(ticker: str, period: str = "1y") -> str:
    """Prefetch India stock OHLCV data using yfinance (.NS suffix).

    Args:
        ticker: NSE stock ticker (e.g., "RELIANCE") or index (e.g., "^NSEI")
        period: Data period (default: "1y")

    Returns:
        Markdown formatted OHLCV data string
    """
    try:
        client = _get_in_data_client()
        df = client.get_ohlcv(ticker, period=period, interval="1d")

        if df is None or df.empty:
            logger.warning(f"No OHLCV data for {ticker}")
            return ""

        df.columns = [col.title().replace("_", " ") for col in df.columns]
        df.index.name = "Date"

        return _df_to_markdown(df, f"OHLCV: {ticker} ({period})")
    except Exception as e:
        logger.error(f"Error prefetching OHLCV for {ticker}: {e}")
        return ""


def prefetch_in_holder_info(ticker: str) -> str:
    """Prefetch institutional holder data.

    Returns major holders summary, top institutional holders,
    and top mutual fund holders as separate markdown tables.

    Args:
        ticker: NSE stock ticker

    Returns:
        Markdown formatted holder data string
    """
    try:
        client = _get_in_data_client()
        holders = client.get_institutional_holders(ticker)

        if not holders:
            logger.warning(f"No holder data for {ticker}")
            return ""

        result = ""

        # Major holders
        major = holders.get("major_holders")
        if major is not None and not major.empty:
            result += _df_to_markdown(major, f"Major Holders: {ticker}")
            result += "\n"

        # Institutional holders
        institutional = holders.get("institutional_holders")
        if institutional is not None and not institutional.empty:
            result += _df_to_markdown(institutional, f"Top Institutional Holders: {ticker}")
            result += "\n"

        # Mutual fund holders
        mutualfund = holders.get("mutualfund_holders")
        if mutualfund is not None and not mutualfund.empty:
            result += _df_to_markdown(mutualfund, f"Top Mutual Fund Holders: {ticker}")
            result += "\n"

        return result if result else ""
    except Exception as e:
        logger.error(f"Error prefetching holder info for {ticker}: {e}")
        return ""


def prefetch_in_stock_info(ticker: str) -> str:
    """Prefetch comprehensive stock information with smart formatting.

    Includes valuation, financial highlights, trading info, dividends,
    and analyst price targets — all formatted with human-readable INR values.

    Args:
        ticker: NSE stock ticker

    Returns:
        Markdown formatted stock info string
    """
    try:
        client = _get_in_data_client()
        info = client.get_company_info(ticker)

        if not info:
            return ""

        result = f"### Company Info: {info.get('longName', ticker)} ({ticker}.NS)\n\n"

        # Valuation Measures
        result += "#### Valuation Measures\n\n"
        result += "| Metric | Value |\n|--------|-------|\n"
        result += f"| Market Cap | {_fmt_inr(info.get('marketCap'), 'currency')} |\n"
        result += f"| Enterprise Value | {_fmt_inr(info.get('enterpriseValue'), 'currency')} |\n"
        result += f"| Trailing P/E | {_fmt_inr(info.get('trailingPE'), 'ratio')} |\n"
        result += f"| Forward P/E | {_fmt_inr(info.get('forwardPE'), 'ratio')} |\n"
        result += f"| PEG Ratio | {_fmt_inr(info.get('pegRatio'), 'ratio')} |\n"
        result += f"| Price/Sales | {_fmt_inr(info.get('priceToSalesTrailing12Months'), 'ratio')} |\n"
        result += f"| Price/Book | {_fmt_inr(info.get('priceToBook'), 'ratio')} |\n"
        result += "\n"

        # Financial Highlights
        result += "#### Financial Highlights\n\n"
        result += "| Metric | Value |\n|--------|-------|\n"
        result += f"| Revenue (TTM) | {_fmt_inr(info.get('totalRevenue'), 'currency')} |\n"
        result += f"| Gross Profit | {_fmt_inr(info.get('grossProfits'), 'currency')} |\n"
        result += f"| EBITDA | {_fmt_inr(info.get('ebitda'), 'currency')} |\n"
        result += f"| Net Income | {_fmt_inr(info.get('netIncomeToCommon'), 'currency')} |\n"
        result += f"| Diluted EPS | {_fmt_inr(info.get('trailingEps'), 'ratio')} |\n"
        result += f"| Profit Margin | {_fmt_inr(info.get('profitMargins'), 'percent')} |\n"
        result += f"| Operating Margin | {_fmt_inr(info.get('operatingMargins'), 'percent')} |\n"
        result += f"| ROA | {_fmt_inr(info.get('returnOnAssets'), 'percent')} |\n"
        result += f"| ROE | {_fmt_inr(info.get('returnOnEquity'), 'percent')} |\n"
        result += f"| Debt/Equity | {_fmt_inr(info.get('debtToEquity'), 'ratio')} |\n"
        result += "\n"

        # Trading Information
        result += "#### Trading Information\n\n"
        result += "| Metric | Value |\n|--------|-------|\n"
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        result += f"| Current Price | {_fmt_inr(price, 'currency')} |\n"
        result += f"| Previous Close | {_fmt_inr(info.get('previousClose'), 'currency')} |\n"
        result += f"| Beta | {_fmt_inr(info.get('beta'), 'ratio')} |\n"
        result += f"| 52-Week High | {_fmt_inr(info.get('fiftyTwoWeekHigh'), 'currency')} |\n"
        result += f"| 52-Week Low | {_fmt_inr(info.get('fiftyTwoWeekLow'), 'currency')} |\n"
        result += f"| 50-Day Average | {_fmt_inr(info.get('fiftyDayAverage'), 'currency')} |\n"
        result += f"| 200-Day Average | {_fmt_inr(info.get('twoHundredDayAverage'), 'currency')} |\n"
        result += f"| Avg Volume (3mo) | {_fmt_inr(info.get('averageVolume'), 'number')} |\n"
        result += f"| Shares Outstanding | {_fmt_inr(info.get('sharesOutstanding'), 'number')} |\n"
        result += f"| Float Shares | {_fmt_inr(info.get('floatShares'), 'number')} |\n"
        result += f"| Short Ratio | {_fmt_inr(info.get('shortRatio'), 'ratio')} |\n"
        result += "\n"

        # Dividend Info
        result += "#### Dividend Info\n\n"
        result += "| Metric | Value |\n|--------|-------|\n"
        result += f"| Dividend Rate | {_fmt_inr(info.get('dividendRate'), 'currency')} |\n"
        # NOTE: yfinance dividendYield is already in percentage form (0.96 = 0.96%)
        result += f"| Dividend Yield | {_fmt_inr(info.get('dividendYield'), 'percent_raw')} |\n"
        result += f"| Payout Ratio | {_fmt_inr(info.get('payoutRatio'), 'percent')} |\n"
        result += "\n"

        # Analyst Targets
        result += "#### Analyst Targets\n\n"
        result += "| Metric | Value |\n|--------|-------|\n"
        result += f"| Target High | {_fmt_inr(info.get('targetHighPrice'), 'currency')} |\n"
        result += f"| Target Low | {_fmt_inr(info.get('targetLowPrice'), 'currency')} |\n"
        result += f"| Target Mean | {_fmt_inr(info.get('targetMeanPrice'), 'currency')} |\n"
        result += f"| Target Median | {_fmt_inr(info.get('targetMedianPrice'), 'currency')} |\n"
        result += f"| Recommendation | {info.get('recommendationKey', 'N/A')} |\n"
        result += f"| Number of Analysts | {info.get('numberOfAnalystOpinions', 'N/A')} |\n"
        result += "\n"

        return result
    except Exception as e:
        logger.error(f"Error prefetching stock info for {ticker}: {e}")
        return ""


def prefetch_in_recommendations(ticker: str) -> str:
    """Prefetch analyst recommendations.

    Args:
        ticker: NSE stock ticker

    Returns:
        Markdown formatted recommendations string
    """
    try:
        client = _get_in_data_client()
        recs = client.get_recommendations(ticker)

        if recs is None or recs.empty:
            return ""

        # Only last 20 recommendations
        recent = recs.tail(20)
        return _df_to_markdown(recent, f"Analyst Recommendations: {ticker}")
    except Exception as e:
        logger.error(f"Error prefetching recommendations for {ticker}: {e}")
        return ""


def prefetch_in_financial_statements(ticker: str) -> str:
    """Prefetch financial statements (annual + quarterly).

    Fetches income statement, balance sheet, and cash flow — both
    annual and quarterly variants (6 tables total).
    Values are formatted in human-readable INR (₹Cr/₹B/₹T) to prevent
    LLM misinterpretation of raw large numbers.

    Args:
        ticker: NSE stock ticker

    Returns:
        Markdown formatted financial data string
    """
    try:
        client = _get_in_data_client()
        financials = client.get_financials(ticker)

        if not financials:
            return ""

        # Ordered output for consistent presentation
        ordered_keys = [
            ("income_stmt", f"Annual Income Statement: {ticker}"),
            ("balance_sheet", f"Annual Balance Sheet: {ticker}"),
            ("cash_flow", f"Annual Cash Flow: {ticker}"),
            ("quarterly_income_stmt", f"Quarterly Income Statement: {ticker}"),
            ("quarterly_balance_sheet", f"Quarterly Balance Sheet: {ticker}"),
            ("quarterly_cash_flow", f"Quarterly Cash Flow: {ticker}"),
        ]

        # Row names that are NOT currency values (ratios, counts, per-share)
        _non_currency_rows = {
            "tax rate for calcs", "diluted average shares", "basic average shares",
            "diluted eps", "basic eps", "ordinary shares number",
            "share issued", "peg ratio", "enterprise value/ebitda",
            "net debt to ebitda", "debt to equity",
        }

        parts = [
            "**Note**: Financial values are formatted in INR (₹Crore/₹Billion/₹Trillion). "
            "Non-monetary fields like EPS, tax rates, and share counts are shown as plain numbers."
        ]
        for key, title in ordered_keys:
            df = financials.get(key)
            if df is not None and not df.empty:
                # Format numeric values to human-readable INR (skip non-currency rows)
                formatted_df = df.copy().astype(object)  # Convert to object to avoid dtype warnings
                for col in formatted_df.columns:
                    for idx in formatted_df.index:
                        val = formatted_df.at[idx, col]
                        row_name = str(idx).strip().lower()
                        if row_name in _non_currency_rows:
                            # Keep ratios/counts as plain numbers
                            if val is None or (isinstance(val, float) and pd.isna(val)):
                                formatted_df.at[idx, col] = "N/A"
                            elif isinstance(val, (int, float)):
                                formatted_df.at[idx, col] = f"{val:,.2f}" if abs(val) < 1e5 else _fmt_inr(val, "number")
                            else:
                                formatted_df.at[idx, col] = str(val)
                        else:
                            formatted_df.at[idx, col] = _fmt_financial_value(val)
                parts.append(_df_to_markdown(formatted_df, title))

        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.error(f"Error prefetching financials for {ticker}: {e}")
        return ""


def prefetch_in_market_indices() -> dict:
    """Prefetch major Indian market indices.

    Returns:
        Dictionary of index_name → markdown OHLCV string
    """
    indices = {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
        "BSE SENSEX": "^BSESN",
        "NIFTY IT": "^CNXIT",
    }

    results = {}
    client = _get_in_data_client()

    for name, yf_ticker in indices.items():
        try:
            df = client.get_ohlcv(yf_ticker, period="1y", interval="1d")
            if df is not None and not df.empty:
                df.columns = [col.title().replace("_", " ") for col in df.columns]
                df.index.name = "Date"
                results[name] = _df_to_markdown(df, f"Index: {name} (1y)")
                logger.info(f"Prefetched index {name}: {len(df)} records")
        except Exception as e:
            logger.warning(f"Error prefetching index {name}: {e}")

    return results


def prefetch_in_analysis_estimates(ticker: str) -> str:
    """Prefetch earnings/revenue estimates and analyst data via yfinance.

    Collects earnings estimates, revenue estimates, EPS trend, EPS revisions,
    growth estimates, analyst price targets, and recommendations summary.

    Args:
        ticker: NSE stock ticker

    Returns:
        Markdown formatted analysis estimates string
    """
    try:
        import yfinance as yf

        yf_sym = f"{ticker}.NS" if not ticker.startswith("^") and "." not in ticker else ticker
        stock = yf.Ticker(yf_sym)

        result = ""

        # 1. Earnings Estimates
        try:
            earnings_est = stock.earnings_estimate
            if earnings_est is not None and not earnings_est.empty:
                result += _df_to_markdown(earnings_est, f"Earnings Estimates: {ticker}")
                result += "\n"
        except Exception as e:
            logger.debug(f"No earnings estimates for {ticker}: {e}")

        # 2. Revenue Estimates
        try:
            revenue_est = stock.revenue_estimate
            if revenue_est is not None and not revenue_est.empty:
                result += _df_to_markdown(revenue_est, f"Revenue Estimates: {ticker}")
                result += "\n"
        except Exception as e:
            logger.debug(f"No revenue estimates for {ticker}: {e}")

        # 3. EPS Trend
        try:
            eps_trend = stock.eps_trend
            if eps_trend is not None and not eps_trend.empty:
                result += _df_to_markdown(eps_trend, f"EPS Trend: {ticker}")
                result += "\n"
        except Exception as e:
            logger.debug(f"No EPS trend for {ticker}: {e}")

        # 4. EPS Revisions
        try:
            eps_revisions = stock.eps_revisions
            if eps_revisions is not None and not eps_revisions.empty:
                result += _df_to_markdown(eps_revisions, f"EPS Revisions: {ticker}")
                result += "\n"
        except Exception as e:
            logger.debug(f"No EPS revisions for {ticker}: {e}")

        # 5. Growth Estimates
        try:
            growth_est = stock.growth_estimates
            if growth_est is not None and not growth_est.empty:
                result += _df_to_markdown(growth_est, f"Growth Estimates: {ticker}")
                result += "\n"
        except Exception as e:
            logger.debug(f"No growth estimates for {ticker}: {e}")

        # 6. Analyst Price Targets
        try:
            targets = stock.analyst_price_targets
            if targets and isinstance(targets, dict):
                result += f"### Analyst Price Targets: {ticker}\n\n"
                result += "| Metric | Value |\n|--------|-------|\n"
                result += f"| Current | {_fmt_inr(targets.get('current'), 'currency')} |\n"
                result += f"| High | {_fmt_inr(targets.get('high'), 'currency')} |\n"
                result += f"| Low | {_fmt_inr(targets.get('low'), 'currency')} |\n"
                result += f"| Mean | {_fmt_inr(targets.get('mean'), 'currency')} |\n"
                result += f"| Median | {_fmt_inr(targets.get('median'), 'currency')} |\n"
                result += "\n"
        except Exception as e:
            logger.debug(f"No analyst price targets for {ticker}: {e}")

        # 7. Recommendations Summary
        try:
            rec_summary = stock.recommendations_summary
            if rec_summary is not None and not rec_summary.empty:
                result += _df_to_markdown(rec_summary, f"Recommendations Summary: {ticker}")
                result += "\n"
        except Exception as e:
            logger.debug(f"No recommendations summary for {ticker}: {e}")

        if not result:
            logger.warning(f"No analysis estimates data for {ticker}")
            return ""

        return result
    except Exception as e:
        logger.error(f"Error prefetching analysis estimates for {ticker}: {e}")
        return ""


def prefetch_in_company_profile(ticker: str) -> str:
    """Prefetch company profile data via yfinance.

    Collects business description, headquarters, sector/industry,
    employee count, and key executives with compensation.

    Args:
        ticker: NSE stock ticker

    Returns:
        Markdown formatted company profile string
    """
    try:
        import yfinance as yf

        yf_sym = f"{ticker}.NS" if not ticker.startswith("^") and "." not in ticker else ticker
        stock = yf.Ticker(yf_sym)
        info = stock.info

        if not info:
            logger.warning(f"No profile info for {ticker}")
            return ""

        result = f"### Company Profile: {info.get('longName', ticker)}\n\n"

        result += "#### Basic Information\n\n"
        result += "| Field | Value |\n|-------|-------|\n"
        result += f"| Company Name | {info.get('longName', 'N/A')} |\n"
        result += f"| Sector | {info.get('sector', 'N/A')} |\n"
        result += f"| Industry | {info.get('industry', 'N/A')} |\n"
        result += f"| Website | {info.get('website', 'N/A')} |\n"
        employees = info.get('fullTimeEmployees')
        result += f"| Full-Time Employees | {employees:,} |\n" if employees else "| Full-Time Employees | N/A |\n"
        city = info.get('city', '')
        state = info.get('state', '')
        country = info.get('country', '')
        address = ", ".join(filter(None, [city, state, country]))
        result += f"| Headquarters | {address or 'N/A'} |\n"
        result += "\n"

        # Business description
        description = info.get('longBusinessSummary', '')
        if description:
            result += "#### Business Description\n\n"
            result += f"{description}\n\n"

        # Key executives
        officers = info.get('companyOfficers', [])
        if officers:
            result += "#### Key Executives\n\n"
            result += "| Name | Title | Total Pay |\n|------|-------|-----------| \n"
            for officer in officers[:10]:
                name = officer.get('name', 'N/A')
                title = officer.get('title', 'N/A')
                pay = officer.get('totalPay', 0)
                pay_str = _fmt_inr(pay, 'currency') if pay else "N/A"
                result += f"| {name} | {title} | {pay_str} |\n"
            result += "\n"

        return result
    except Exception as e:
        logger.error(f"Error prefetching company profile for {ticker}: {e}")
        return ""


def prefetch_in_analysis_data(ticker: str) -> dict:
    """
    Prefetch all data needed for India stock analysis.

    Combines all prefetch functions into a single call for convenience.

    Args:
        ticker: NSE stock ticker

    Returns:
        Dictionary with all prefetched data sections
    """
    result = {}

    # Fetch raw company info first — used by both stock_info and key_reference_data
    raw_info = {}
    try:
        client = _get_in_data_client()
        raw_info = client.get_company_info(ticker) or {}
    except Exception as e:
        logger.warning(f"Raw info fetch failed for {ticker}: {e}")

    # Build key reference data block (concise ground-truth for all agents)
    try:
        result["key_reference_data"] = build_key_reference_data(ticker, raw_info)
        logger.info(f"Built key reference data for {ticker}")
    except Exception as e:
        logger.warning(f"Key reference data failed for {ticker}: {e}")

    try:
        result["stock_ohlcv"] = prefetch_in_stock_ohlcv(ticker)
        logger.info(f"Prefetched OHLCV for {ticker}")
    except Exception as e:
        logger.warning(f"OHLCV prefetch failed for {ticker}: {e}")

    try:
        result["holder_info"] = prefetch_in_holder_info(ticker)
        logger.info(f"Prefetched holder info for {ticker}")
    except Exception as e:
        logger.warning(f"Holder info prefetch failed for {ticker}: {e}")

    try:
        result["stock_info"] = prefetch_in_stock_info(ticker)
        logger.info(f"Prefetched stock info for {ticker}")
    except Exception as e:
        logger.warning(f"Stock info prefetch failed for {ticker}: {e}")

    try:
        result["recommendations"] = prefetch_in_recommendations(ticker)
        logger.info(f"Prefetched recommendations for {ticker}")
    except Exception as e:
        logger.warning(f"Recommendations prefetch failed for {ticker}: {e}")

    try:
        result["company_profile"] = prefetch_in_company_profile(ticker)
        logger.info(f"Prefetched company profile for {ticker}")
    except Exception as e:
        logger.warning(f"Company profile prefetch failed for {ticker}: {e}")

    try:
        result["analysis_estimates"] = prefetch_in_analysis_estimates(ticker)
        logger.info(f"Prefetched analysis estimates for {ticker}")
    except Exception as e:
        logger.warning(f"Analysis estimates prefetch failed for {ticker}: {e}")

    try:
        result["financial_statements"] = prefetch_in_financial_statements(ticker)
        logger.info(f"Prefetched financials for {ticker}")
    except Exception as e:
        logger.warning(f"Financials prefetch failed for {ticker}: {e}")

    try:
        result["market_indices"] = prefetch_in_market_indices()
        logger.info(f"Prefetched market indices")
    except Exception as e:
        logger.warning(f"Market indices prefetch failed: {e}")

    if result:
        logger.info(f"Prefetched IN data for {ticker}: {list(result.keys())}")
    else:
        logger.warning(f"Failed to prefetch any IN data for {ticker}")

    return result
