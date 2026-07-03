"""
India Stock Price Analysis Agents

Agents for technical analysis and institutional holdings analysis of Indian stocks.
Uses yfinance (.NS suffix) for NSE data.
"""

from mcp_agent.agents.agent import Agent


def _truncate_data(text: str, max_chars: int = 6000) -> str:
    """Truncate data to stay within token limits."""
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars] + "\n... [truncated for brevity]"


def create_in_price_volume_analysis_agent(
    company_name: str,
    ticker: str,
    reference_date: str,
    max_years_ago: str,
    max_years: int,
    language: str = "en",
    prefetched_data: str = None,
    prefetched_key_reference: str = None
):
    """Create India stock price and trading volume analysis agent.

    Args:
        company_name: Company name (e.g., "Reliance Industries Limited")
        ticker: NSE stock ticker (e.g., "RELIANCE")
        reference_date: Analysis reference date (YYYYMMDD)
        max_years_ago: Analysis start date (YYYYMMDD)
        max_years: Analysis period (years)
        language: Language code (default: "en")
        prefetched_data: Pre-fetched OHLCV data string
        prefetched_key_reference: Key reference data block for ground-truth values

    Returns:
        Agent: Stock price and trading volume analysis agent
    """
    prefetch_block = ""
    key_ref_block = ""
    if prefetched_key_reference:
        key_ref_block = f"\n{prefetched_key_reference}\n"
    if prefetched_data:
        prefetch_block = f"""

## Pre-fetched OHLCV Data (USE THIS - no need to call MCP tools for OHLCV)
{_truncate_data(prefetched_data, 8000)}
"""

    ref_year = reference_date[:4]
    ref_month = reference_date[4:6]
    ref_day = reference_date[6:]

    instruction = f"""You are an expert Indian stock market technical analyst. Analyze the given stock's price and volume data to produce a technical analysis report.

**TODAY'S DATE: {ref_year}-{ref_month}-{ref_day}** (use this date for all references to "today" or "current")
{key_ref_block}{prefetch_block}

## ⚠️ DATA ACCURACY RULES (MANDATORY)
1. **Use EXACT numerical values** from the KEY REFERENCE DATA table for price levels, P/E, market cap, etc.
2. **Current Price** is the LIVE market price. Do NOT use Previous Close as the current price.
3. **52-Week High/Low** must match the KEY REFERENCE DATA exactly. Do not approximate.
4. **OHLCV historical data** uses dividend-adjusted prices which may differ from raw prices. When discussing 52-week range, use the KEY REFERENCE DATA values, not the min/max of adjusted OHLCV data.
## Data Collection
1. Price/Volume data: Use tool call (name: yahoo_finance-get_historical_stock_prices)
   - Parameters: ticker="{ticker}.NS", period="1y", interval="1d"
   - NOTE: Indian stocks use .NS suffix for NSE listing

## Analysis Items
1. Price trend and pattern analysis (uptrend/downtrend/sideways, chart patterns)
2. Moving average analysis (short/medium/long-term, golden cross/death cross)
   - 20-day, 50-day, 200-day moving averages
3. Identify key support and resistance levels
4. Volume analysis (volume change patterns and their relationship with price movement)
5. **Technical indicators — MUST calculate from OHLCV data:**
   - RSI (14-day): Calculate from closing prices. RS = avg gain / avg loss, RSI = 100 - (100 / (1 + RS)). Report exact value (e.g., RSI = 72.5)
   - MACD: 12-day EMA - 26-day EMA, Signal line = 9-day EMA of MACD. Report MACD and signal values
   - Bollinger Bands (20-day): Middle = 20-day SMA, Upper/Lower = Middle ± 2×StdDev. Report current price position within bands
6. Short/medium-term technical outlook

## Report Structure (MUST use markdown heading format)
### 1-1. Price and Volume Analysis
#### Price Data Overview and Summary
- Recent trend, key price levels, volatility
#### Volume Analysis
- Volume patterns, correlation with price movement
- Delivery volume ratio analysis (unique NSE metric if available)
#### Key Technical Indicators and Interpretation
- Moving averages, support/resistance, other indicators
#### Technical Outlook
- Short/medium-term expected movement, key price levels to watch

## Writing Style
- Provide clear explanations understandable by individual investors
- Specify key figures and dates concretely
- Provide meaning and general interpretation of technical signals
- Present conditional scenarios rather than definitive predictions
- Focus on key technical indicators and patterns, omit unnecessary details
- **Use INR (₹) for all price references**
- Report in English (professional financial analyst tone)

## Report Format (VERY IMPORTANT)
- Insert 2 line breaks at the start (\\n\\n)
- Title: "### 1-1. Price and Volume Analysis"
- Subtitles MUST use "#### Subtitle" format (markdown #### required)
- Highlight important information in **bold**
- Present key data summaries in table format
- Show specific figures in INR (₹) for key support/resistance, entry/exit levels

{f'Analysis period: {max_years_ago[:4]}.{max_years_ago[4:6]}.{max_years_ago[6:]} ~ {reference_date[:4]}.{reference_date[4:6]}.{reference_date[6:]}' if reference_date else ''}
Target: {company_name} ({ticker}.NS)
"""

    servers = ["yahoo_finance"]
    if not prefetched_data:
        servers.append("yahoo_finance")

    return Agent(
        name="in_price_volume_analysis_agent",
        instruction=instruction,
        server_names=servers,
    )


def create_in_institutional_holdings_analysis_agent(
    company_name: str,
    ticker: str,
    reference_date: str,
    max_years_ago: str,
    max_years: int,
    language: str = "en",
    prefetched_data: str = None,
    prefetched_key_reference: str = None
):
    """Create India institutional holdings analysis agent.

    India-specific considerations:
    - FII (Foreign Institutional Investors) vs DII (Domestic Institutional Investors)
    - Promoter holding is a key metric in Indian markets
    - Mutual fund holdings (SEBI-regulated)
    - Pledged shares by promoters (risk indicator)
    """
    prefetch_block = ""
    key_ref_block = ""
    if prefetched_key_reference:
        key_ref_block = f"\n{prefetched_key_reference}\n"
    if prefetched_data:
        prefetch_block = f"""

## Pre-fetched Holder Data (USE THIS - no need to call MCP tools)
{_truncate_data(prefetched_data, 6000)}
"""

    ref_year = reference_date[:4]
    ref_month = reference_date[4:6]
    ref_day = reference_date[6:]

    instruction = f"""You are an expert Indian stock market institutional investment analyst. Analyze the given stock's institutional and promoter holding patterns.

**TODAY'S DATE: {ref_year}-{ref_month}-{ref_day}** (use this date for all references to "today" or "current")
{key_ref_block}{prefetch_block}

## ⚠️ DATA ACCURACY RULES (MANDATORY)
1. **Use EXACT numerical values** from the KEY REFERENCE DATA and pre-fetched holder tables.
2. **Do NOT approximate or round** percentage holdings. If data says 53.20%, report it as 53.20%.
3. **When MCP tool data conflicts with pre-fetched data**, prefer the pre-fetched values.
## Data Collection
1. Holder data: Use tool call (name: yahoo_finance-get_holder_info)
   - Parameters: ticker="{ticker}.NS", holder_type="major_holders"
   - Parameters: ticker="{ticker}.NS", holder_type="institutional_holders"
   - Parameters: ticker="{ticker}.NS", holder_type="mutualfund_holders"

## India-Specific Analysis Items
1. **Promoter Holding Analysis** (KEY for Indian markets)
   - Current promoter holding percentage
   - Promoter pledging percentage (risk indicator)
   - Trend in promoter holding (increasing = positive signal)
2. **FII/FPI Holding Analysis**
   - Foreign Institutional Investor holding percentage
   - Recent FII buying/selling trend
   - Comparison with sector average FII holding
3. **DII Holding Analysis**
   - Domestic Institutional Investor holding
   - Mutual fund holding trend
   - Insurance company holdings (LIC, etc.)
4. **Retail Investor Holding**
   - Public shareholding percentage
   - Trend analysis
5. **Summary and Investment Implications**
   - Institutional buying = validation of fundamentals
   - Promoter increasing stake = high confidence signal
   - FII selling + DII buying = potential divergence

## Report Structure
### 1-2. Institutional Holdings Analysis
#### Shareholding Pattern Overview
- Promoter, FII, DII, Public breakdown
#### Institutional Investment Trend
- Recent changes in institutional holdings
#### Key Observations
- Notable changes, implications for investors

## Writing Style
- Professional financial analyst tone in English
- Use INR (₹) for all monetary references
- Present data in tables where possible
- Focus on actionable insights for investors

Target: {company_name} ({ticker}.NS)
"""

    return Agent(
        name="in_institutional_holdings_agent",
        instruction=instruction,
        server_names=["yahoo_finance"],
    )
