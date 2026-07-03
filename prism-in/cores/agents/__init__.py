"""
PRISM-IN AI Agents Module

Specialized AI agents for Indian stock market (NSE/BSE) analysis:
- stock_price_agents: Technical analysis (yfinance .NS)
- company_info_agents: Fundamental analysis (yfinance + nsetools)
- news_strategy_agents: News and investment strategy (perplexity)
- market_index_agents: NIFTY 50, SENSEX, Bank NIFTY analysis
- trading_agents: Buy/Sell decision agents
"""

from typing import Dict, List
from pathlib import Path
import importlib.util

# Get the directory containing this file
_AGENTS_DIR = Path(__file__).parent


def _load_local_module(module_name: str):
    """Load a module from the local agents directory."""
    module_path = _AGENTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"in_{module_name}", module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load {module_name} from {module_path}")


# Pre-load all agent modules
_stock_price_agents = _load_local_module("stock_price_agents")
_company_info_agents = _load_local_module("company_info_agents")
_news_strategy_agents = _load_local_module("news_strategy_agents")
_market_index_agents = _load_local_module("market_index_agents")


def get_in_data_urls(ticker: str) -> Dict[str, str]:
    """
    Generate URLs for Indian stock data sources.

    Args:
        ticker: NSE stock ticker (e.g., "RELIANCE", "TCS")

    Returns:
        Dictionary of data source URLs
    """
    return {
        "profile": f"https://finance.yahoo.com/quote/{ticker}.NS/profile",
        "key_statistics": f"https://finance.yahoo.com/quote/{ticker}.NS/key-statistics",
        "financials": f"https://finance.yahoo.com/quote/{ticker}.NS/financials",
        "analysis": f"https://finance.yahoo.com/quote/{ticker}.NS/analysis",
        "holders": f"https://finance.yahoo.com/quote/{ticker}.NS/holders",
        "news": f"https://finance.yahoo.com/quote/{ticker}.NS/news",
        "moneycontrol": f"https://www.moneycontrol.com/india/stockpricequote/{ticker.lower()}",
        "screener": f"https://www.screener.in/company/{ticker}/",
    }


def get_in_agent_directory(
    company_name: str,
    ticker: str,
    reference_date: str,
    base_sections: List[str],
    language: str = "en",
    prefetched_data: dict = None
) -> Dict:
    """
    Returns a directory of agents for each section.

    Args:
        company_name: Company name (e.g., "Reliance Industries Limited")
        ticker: NSE stock ticker (e.g., "RELIANCE")
        reference_date: Analysis reference date (YYYYMMDD)
        base_sections: List of sections to generate agents for
        language: Language code (default: "en" for India)

    Returns:
        Dict[str, Agent]: Dictionary of agents keyed by section name
    """
    create_in_price_volume_analysis_agent = _stock_price_agents.create_in_price_volume_analysis_agent
    create_in_institutional_holdings_analysis_agent = _stock_price_agents.create_in_institutional_holdings_analysis_agent
    create_in_company_status_agent = _company_info_agents.create_in_company_status_agent
    create_in_company_overview_agent = _company_info_agents.create_in_company_overview_agent
    create_in_news_analysis_agent = _news_strategy_agents.create_in_news_analysis_agent
    create_in_market_index_analysis_agent = _market_index_agents.create_in_market_index_analysis_agent

    urls = get_in_data_urls(ticker)

    from datetime import datetime, timedelta
    ref_date = datetime.strptime(reference_date, "%Y%m%d")
    max_years = 1
    max_years_ago = (ref_date - timedelta(days=365 * max_years)).strftime("%Y%m%d")

    pf = prefetched_data or {}
    market_indices = pf.get("market_indices", {})
    combined_indices = "\n\n".join(market_indices.values()) if market_indices else None

    agent_creators = {
        "price_volume_analysis": lambda: create_in_price_volume_analysis_agent(
            company_name, ticker, reference_date, max_years_ago, max_years, language,
            prefetched_data=pf.get("stock_ohlcv"),
            prefetched_key_reference=pf.get("key_reference_data")
        ),
        "institutional_holdings_analysis": lambda: create_in_institutional_holdings_analysis_agent(
            company_name, ticker, reference_date, max_years_ago, max_years, language,
            prefetched_data=pf.get("holder_info"),
            prefetched_key_reference=pf.get("key_reference_data")
        ),
        "company_status": lambda: create_in_company_status_agent(
            company_name, ticker, reference_date, urls, language,
            prefetched_data={
                "stock_info": pf.get("stock_info", ""),
                "recommendations": pf.get("recommendations", ""),
                "financial_statements": pf.get("financial_statements", ""),
                "key_reference_data": pf.get("key_reference_data", ""),
            } if pf.get("stock_info") else None
        ),
        "company_overview": lambda: create_in_company_overview_agent(
            company_name, ticker, reference_date, urls, language,
            prefetched_data={
                "stock_info": pf.get("stock_info", ""),
                "holder_info": pf.get("holder_info", ""),
                "key_reference_data": pf.get("key_reference_data", ""),
            } if pf.get("stock_info") else None
        ),
        "news_analysis": lambda: create_in_news_analysis_agent(
            company_name, ticker, reference_date, language
        ),
        "market_index_analysis": lambda: create_in_market_index_analysis_agent(
            reference_date, max_years_ago, max_years, language,
            prefetched_indices=combined_indices
        )
    }

    agents = {}
    for section in base_sections:
        if section in agent_creators:
            agents[section] = agent_creators[section]()

    return agents
