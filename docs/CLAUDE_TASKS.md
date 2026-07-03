# Common Tasks - PRISM-INSIGHT

> **Note**: This is a detailed task reference. For quick overview, see main [CLAUDE.md](../CLAUDE.md).

---

## Task 1: Adding a New AI Agent

```python
# 1. Create agent file
# File: cores/agents/your_agent.py

from mcp_agent import Agent

def create_your_agent(company_name, company_code, reference_date, language="en"):
    if language == "en":
        instruction = """Your English instruction..."""
    else:
        instruction = """Translated instruction..."""

    return Agent(
        instruction=instruction,
        description=f"Your Agent for {company_name}",
        mcp_servers=["yahoo_finance"],  # Add required MCP servers
    )

# 2. Register in prism-in/cores/agents/
# Add your agent to the India analysis pipeline

# 3. Add to base_sections in prism-in/cores/in_analysis.py
base_sections = [
    "price_volume_analysis",
    # ... existing sections
    "your_section",  # Add your section
]

# 4. Add section template in cores/report_generation.py
section_templates = {
    # ... existing templates
    "your_section": """
## Your Section Title

{content}
""",
}
```

---

## Task 2: Modifying Surge Detection Criteria

```python
# File: prism-in/in_trigger_batch.py / prism-in/cores/in_surge_detector.py

def detect_surge_stocks(mode="morning"):
    # Modify thresholds
    VOLUME_THRESHOLD = 2.0  # Change: Volume surge ratio
    GAP_THRESHOLD = 3.0     # Change: Price gap percentage
    MIN_MARKET_CAP = 20000  # Change: Minimum market cap (Cr INR)

    # Add custom filters
    filtered_stocks = df[
        (df['volume_ratio'] >= VOLUME_THRESHOLD) &
        (df['gap_percent'] >= GAP_THRESHOLD) &
        (df['market_cap'] >= MIN_MARKET_CAP) &
        (df['your_custom_condition'])  # Add custom condition
    ]

    return filtered_stocks
```

---

## Task 3: Adding Multi-Language Support

```python
# 1. Add language to cores/language_config.py
class LanguageConfig:
    SUPPORTED_LANGUAGES = ["ko", "en", "ja", "zh", "es", "fr", "de", "your_lang"]

    TEMPLATES = {
        "your_lang": {
            "report_title": "Your Language Title",
            "sections": {
                "technical_analysis": "Technical Analysis",
                # ... add all sections
            }
        }
    }

# 2. Add Telegram channel to .env
TELEGRAM_CHANNEL_ID_YOUR_LANG="-1001234567899"

# 3. Use in broadcasting
python prism-in/in_stock_analysis_orchestrator.py --broadcast-languages ko,ja,zh,es
```

---

## Task 4: Modifying Trading Strategy

```python
# File: prism-in/cores/agents/trading_agents.py

def create_trading_scenario_agent(...):
    instruction = """
    Trading Scenario Generation Instructions:

    BUY SCORE CRITERIA (Modify these):
    - Valuation (PER, PBR vs peers): 0-3 points
    - Technical Momentum: 0-3 points
    - News Catalyst: 0-2 points
    - Market Environment: 0-2 points
    - TOTAL: 10 points (buy threshold: 6+)

    RISK MANAGEMENT (Modify these):
    - Stop Loss: -5% to -7% (change percentage)
    - Target Price: +10% to +30% (change percentage)
    - Risk/Reward Ratio: Min 2:1 (change ratio)

    PORTFOLIO CONSTRAINTS (Modify these):
    - Max positions: 10 (change number)
    - Max same sector: 3 (change number)
    - Sector concentration: 30% (change percentage)
    """
    return Agent(instruction=instruction, ...)

# Apply changes
# 1. Modify instruction text
# 2. Update prism-in/in_stock_tracking_agent.py if needed
# 3. Test with demo mode
```

---

## Task 5: Customizing Report Format

```python
# File: cores/report_generation.py

# 1. Modify report template
REPORT_TEMPLATE = """
# {company_name} ({company_code}) Investment Analysis Report

**Analysis Date**: {reference_date}
**Analyst**: PRISM-INSIGHT AI Agent System
**Language**: {language}

---

## Your Custom Section

{custom_content}

---

{sections}

---

## Investment Strategy

{investment_strategy}

---

**Disclaimer**: {disclaimer}
"""

# 2. Add custom sections
def generate_full_report(section_reports, investment_strategy, ...):
    custom_content = generate_custom_section(...)

    report = REPORT_TEMPLATE.format(
        company_name=company_name,
        custom_content=custom_content,
        sections=format_sections(section_reports),
        investment_strategy=investment_strategy,
        ...
    )
    return report
```

---

## Task 6: Adding New MCP Server

```bash
# 1. Install MCP server
npm install -g your-mcp-server
# or
pip install your-mcp-server
```

```yaml
# 2. Add to mcp_agent.config.yaml
mcp:
  servers:
    your_server: npx your-mcp-server
    # or
    your_server: python3 -m your_mcp_server
```

```yaml
# 3. Add credentials to mcp_agent.secrets.yaml (if needed)
YOUR_SERVER_API_KEY: "your-api-key"
```

```python
# 4. Use in agent
def create_your_agent(...):
    return Agent(
        instruction="...",
        mcp_servers=["your_server"],  # Add your server
    )
```

---

## Task 7: India NSE Data Integration

```python
# Yahoo Finance data for Indian stocks
import yfinance as yf

# Fetch NSE stock data (append .NS for NSE tickers)
ticker = yf.Ticker("RELIANCE.NS")
info = ticker.info
hist = ticker.history(period="1y")

# Key fields available:
# - currentPrice, marketCap, volume
# - trailingPE, priceToBook, dividendYield
# - sector, industry, longBusinessSummary
```

---

*See also: [CLAUDE.md](../CLAUDE.md) | [CLAUDE_AGENTS.md](CLAUDE_AGENTS.md) | [CLAUDE_TROUBLESHOOTING.md](CLAUDE_TROUBLESHOOTING.md)*
