"""
India Company Information Analysis Agents

Agents for fundamental analysis of Indian companies.
Uses yfinance prefetched data and web scraping for financial analysis.

India-specific data sources:
- Yahoo Finance (.NS suffix)
- Screener.in (excellent for Indian company financials)
- MoneyControl (news + fundamentals)
"""

from mcp_agent.agents.agent import Agent
from typing import Dict


def _truncate_data(text: str, max_chars: int = 6000) -> str:
    """Truncate prefetched data to stay within token budget.

    Args:
        text: Raw data text
        max_chars: Maximum character count (default: 6000 ≈ ~1500 tokens)

    Returns:
        Truncated text with indicator if cut
    """
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars] + "\n... [truncated for brevity]"


def create_in_company_status_agent(
    company_name: str,
    ticker: str,
    reference_date: str,
    urls: Dict[str, str],
    language: str = "en",
    prefetched_data: dict = None
):
    """Create India company status analysis agent.

    Args:
        company_name: Company name
        ticker: NSE stock ticker
        reference_date: Analysis reference date (YYYYMMDD)
        urls: Dictionary of data source URLs
        language: Language code (default: "en")
        prefetched_data: Pre-fetched data dict

    Returns:
        Agent: Company status analysis agent
    """
    prefetch_block = ""
    if prefetched_data:
        parts = []
        if prefetched_data.get("key_reference_data"):
            parts.append(prefetched_data["key_reference_data"])
        if prefetched_data.get("stock_info"):
            parts.append(f"### Pre-fetched Stock Info\n{_truncate_data(prefetched_data['stock_info'])}")
        if prefetched_data.get("recommendations"):
            parts.append(f"### Pre-fetched Recommendations\n{_truncate_data(prefetched_data['recommendations'], 3000)}")
        if prefetched_data.get("financial_statements"):
            parts.append(f"### Pre-fetched Financial Statements\n{_truncate_data(prefetched_data['financial_statements'], 5000)}")
        if parts:
            prefetch_block = "\n\n## Pre-fetched Data (USE THIS FIRST)\n" + "\n\n".join(parts)

    ref_year = reference_date[:4]
    ref_month = reference_date[4:6]
    ref_day = reference_date[6:]

    instruction = f"""You are a fundamental analysis expert for Indian companies. Analyze Yahoo Finance data combined with Indian market-specific metrics to create a comprehensive company status report.

**TODAY'S DATE: {ref_year}-{ref_month}-{ref_day}** (use this date for all references to "today" or "current")
{prefetch_block}

## ⚠️ DATA ACCURACY RULES (MANDATORY)
1. **Use EXACT numerical values** from the KEY REFERENCE DATA and pre-fetched tables. Do NOT round, approximate, or recalculate values.
2. **Current Price vs Previous Close**: 'Current Price' is the LIVE market price. 'Previous Close' is yesterday's closing price. Never confuse these.
3. **Dividend Yield**: Already expressed as a percentage in the data (e.g., 0.96% means 0.96%). Do NOT re-multiply by 100.
4. **Financial Statements**: Values in financial tables are pre-formatted in INR (₹Cr/₹B/₹T). Use them as-is.
5. **When MCP tool data conflicts with pre-fetched data**, prefer the pre-fetched KEY REFERENCE DATA values.

## Data Collection

### 1. Yahoo Finance Key Statistics (URL: {urls['key_statistics']}):
   - Valuation metrics: Market Cap, Enterprise Value, Trailing P/E, Forward P/E, PEG Ratio, P/S, P/B, EV/Revenue, EV/EBITDA
   - Financial highlights: Profit margin, Operating margin, ROA, ROE, Revenue, Net Income, Diluted EPS
   - Trading info: Beta, 52W High/Low, 50-day/200-day MA, Avg Volume, Shares Outstanding, Float, Short ratio

### 2. Yahoo Finance Financials (URL: {urls['financials']}):
   - Income statement: Revenue, Operating expenses, Net income (annual and quarterly)
   - Balance sheet: Total assets, Total liabilities, Equity
   - Cash flow: Operating, Investing, Financing cash flows, Free cash flow

### 3. yahoo_finance MCP server:
   - Tool call (name: yahoo_finance-get_stock_info), ticker="{ticker}.NS"
   - Tool call (name: yahoo_finance-get_financial_statement), ticker="{ticker}.NS", statement_type="income_stmt"
   - Tool call (name: yahoo_finance-get_recommendations), ticker="{ticker}.NS"

## India-Specific Analysis
1. **Business Model and Competitive Position**
   - Market position within Indian industry landscape
   - Government policy impact (PLI schemes, Make in India, etc.)
   - Import/export dynamics

2. **Financial Performance Analysis**
   - Revenue/profit trend (last 4 fiscal years, Indian FY: Apr-Mar)
   - Profitability metrics (operating margin, net margin) trends
   - Quarterly performance seasonality
   - Earnings surprise/miss analysis

3. **Valuation Analysis**
   - Current P/E, P/B, P/S vs historical average and sector average
   - Forward P/E-based valuation assessment
   - Dividend yield and payout ratio (if applicable)

4. **Balance Sheet Strength**
   - Debt-to-equity ratio (critical for Indian companies)
   - Interest coverage ratio
   - Working capital management
   - Promoter pledging impact on balance sheet

## Report Structure
### 2-1. Company Status Analysis
#### Financial Summary
- Key valuation and financial metrics overview
#### Revenue and Profitability Analysis
- Growth trends, margin analysis
#### Balance Sheet Analysis
- Debt structure, capital efficiency
#### Investment Implications
- Strengths, risks, valuation assessment

## Writing Style
- Professional financial analyst tone in English
- Use INR (₹) for all monetary references
- Present key data in tables
- Reference Indian fiscal year (Apr-Mar) where relevant
- Compare with BSE/NSE sector averages where possible

Target: {company_name} ({ticker}.NS)
"""

    servers = ["yahoo_finance"]
    if not prefetched_data:
        servers.append("firecrawl")

    return Agent(
        name="in_company_status_agent",
        instruction=instruction,
        server_names=servers,
    )


def create_in_company_overview_agent(
    company_name: str,
    ticker: str,
    reference_date: str,
    urls: Dict[str, str],
    language: str = "en",
    prefetched_data: dict = None
):
    """Create India company overview analysis agent.

    Focuses on business model, competitive landscape, and growth drivers
    specific to the Indian market context.
    """
    prefetch_block = ""
    if prefetched_data:
        parts = []
        if prefetched_data.get("key_reference_data"):
            parts.append(prefetched_data["key_reference_data"])
        if prefetched_data.get("stock_info"):
            parts.append(f"### Pre-fetched Stock Info\n{_truncate_data(prefetched_data['stock_info'])}")
        if prefetched_data.get("holder_info"):
            parts.append(f"### Pre-fetched Holder Info\n{_truncate_data(prefetched_data['holder_info'], 5000)}")
        if parts:
            prefetch_block = "\n\n## Pre-fetched Data (USE THIS FIRST)\n" + "\n\n".join(parts)

    ref_year = reference_date[:4]
    ref_month = reference_date[4:6]
    ref_day = reference_date[6:]

    instruction = f"""You are an expert on Indian companies and their business models. Provide a comprehensive company overview report.

**TODAY'S DATE: {ref_year}-{ref_month}-{ref_day}** (use this date for all references to "today" or "current")
{prefetch_block}

## ⚠️ DATA ACCURACY RULES (MANDATORY)
1. **Use EXACT numerical values** from the KEY REFERENCE DATA and pre-fetched tables. Do NOT round, approximate, or recalculate values.
2. **Current Price vs Previous Close**: 'Current Price' is the LIVE market price. 'Previous Close' is yesterday's closing price. Never confuse these.
3. **Dividend Yield**: Already expressed as a percentage in the data (e.g., 0.96% means 0.96%). Do NOT re-multiply by 100.
4. **When MCP tool data conflicts with pre-fetched data**, prefer the pre-fetched KEY REFERENCE DATA values.

## Data Collection
1. Yahoo Finance Profile (URL: {urls['profile']}):
   - Business description, sector, industry, number of employees
   - Key executives and board members
2. yahoo_finance MCP server:
   - Tool call (name: yahoo_finance-get_stock_info), ticker="{ticker}.NS"

## India-Specific Analysis
1. **Business Model Overview**
   - Core business segments and revenue contribution
   - Market position within India
   - Geographic diversification (domestic vs export)

2. **Competitive Landscape**
   - Top competitors in India
   - Market share dynamics
   - Barriers to entry / competitive moat

3. **Growth Drivers**
   - Government policies (PLI, Digital India, Atmanirbhar Bharat)
   - Demographic tailwinds (young population, urbanization)
   - Technology adoption and digital transformation
   - Expansion plans (capex, new markets, M&A)

4. **Risk Factors**
   - Regulatory risks (SEBI, RBI, sector-specific regulations)
   - Currency risk (INR/USD for export-dependent companies)
   - Commodity price sensitivity
   - Promoter governance concerns (if any)

5. **ESG Considerations**
   - Environmental compliance
   - Social impact and CSR spending (Companies Act 2013 mandate)
   - Governance quality (board independence, related-party transactions)

## Report Structure
### 2-2. Company Overview Analysis
#### Business Model and Market Position
- Core operations, market position
#### Competitive Landscape
- Key competitors, competitive advantages
#### Growth Drivers and Catalysts
- Near-term and long-term growth triggers
#### Risk Assessment
- Key risks and mitigating factors

## Writing Style
- Professional financial analyst tone in English
- Reference Indian market-specific frameworks and regulations
- Present competitive comparison in tables
- Focus on actionable insights

Target: {company_name} ({ticker}.NS)
"""

    servers = ["yahoo_finance"]
    if not prefetched_data:
        servers.append("firecrawl")

    return Agent(
        name="in_company_overview_agent",
        instruction=instruction,
        server_names=servers,
    )
