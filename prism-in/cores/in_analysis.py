"""
India Stock Analysis Module

Generate comprehensive stock analysis reports for Indian stocks (NSE/BSE).
Uses yfinance MCP server for market data (.NS suffix) and India-specific agents.

Entry point: analyze_in_stock()
"""
import os
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from mcp_agent.app import MCPApp

# Set up import paths using direct file import (avoids namespace collision)
import sys
import importlib.util

_prism_in_dir = Path(__file__).parent.parent
_project_root = Path(__file__).parent.parent.parent


def _import_from_project_root(module_name: str, file_path: Path):
    """Import a module directly from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import report_generation from main project's cores
_report_gen_module = _import_from_project_root(
    "main_report_generation",
    _project_root / "cores" / "report_generation.py"
)
generate_report = _report_gen_module.generate_report
generate_summary = _report_gen_module.generate_summary
generate_investment_strategy = _report_gen_module.generate_investment_strategy
get_disclaimer = _report_gen_module.get_disclaimer
generate_market_report = _report_gen_module.generate_market_report

# Import utils from main project's cores
_utils_module = _import_from_project_root(
    "main_utils",
    _project_root / "cores" / "utils.py"
)
clean_markdown = _utils_module.clean_markdown

# Add prism-in directory for local imports
sys.path.insert(0, str(_prism_in_dir))

# Import from prism-in local cores.agents
_agents_path = _prism_in_dir / "cores" / "agents" / "__init__.py"
_spec = importlib.util.spec_from_file_location("in_agents", _agents_path)
_in_agents_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_in_agents_module)
get_in_agent_directory = _in_agents_module.get_in_agent_directory

# Market analysis cache (prevents re-running market analysis for each stock)
_in_market_analysis_cache = {}

# Import chart functions from in_stock_chart module
_chart_module = _import_from_project_root(
    "in_stock_chart",
    _prism_in_dir / "cores" / "in_stock_chart.py"
)
get_in_price_chart_html = _chart_module.get_in_price_chart_html
get_in_institutional_chart_html = _chart_module.get_in_institutional_chart_html
get_in_technical_chart_html = _chart_module.get_in_technical_chart_html


async def analyze_in_stock(
    ticker: str = "RELIANCE",
    company_name: str = "Reliance Industries Limited",
    reference_date: str = None,
    language: str = "en",
    include_news: bool = True
) -> str:
    """
    Generate comprehensive stock analysis report for Indian stock (NSE).

    Args:
        ticker: NSE stock ticker (e.g., "RELIANCE", "TCS", "INFY")
        company_name: Company name (e.g., "Reliance Industries Limited")
        reference_date: Analysis reference date (YYYYMMDD format)
        language: Language code (default: "en" for India)
        include_news: Whether to include news analysis (uses Yahoo Finance news)

    Returns:
        str: Generated final report markdown text
    """
    # 1. Initial setup
    app = MCPApp(name="in_stock_analysis")

    if reference_date is None:
        reference_date = datetime.now().strftime("%Y%m%d")

    async with app.run() as parallel_app:
        logger = parallel_app.logger
        logger.info(f"Starting: {company_name}({ticker}.NS) India analysis - reference date: {reference_date}")

        # 2. Section reports storage
        section_reports = {}

        # 3. Define sections to analyze
        # yfinance sections: run sequentially to avoid rate limits
        yfinance_sections = [
            "price_volume_analysis",           # Technical analysis (yfinance OHLCV)
            "institutional_holdings_analysis",  # yfinance holders (FII/DII/Promoter)
            "company_status",                  # yfinance financials
            "company_overview",                # yfinance info
            "market_index_analysis"            # yfinance indices (NIFTY/SENSEX)
        ]
        # Non-yfinance sections: can run in parallel
        parallel_sections = []
        if include_news:
            parallel_sections.append("news_analysis")
        else:
            section_reports["news_analysis"] = "_News analysis skipped. Technical and fundamental analysis are provided normally._"
            logger.info("Skipping news_analysis (disabled)")

        base_sections = yfinance_sections + ["news_analysis"]

        # 4. Prefetch data to reduce MCP tool call overhead
        try:
            _prefetch_path = Path(__file__).parent / "data_prefetch.py"
            _prefetch_spec = importlib.util.spec_from_file_location("in_data_prefetch", _prefetch_path)
            _prefetch_module = importlib.util.module_from_spec(_prefetch_spec)
            _prefetch_spec.loader.exec_module(_prefetch_module)
            prefetched = _prefetch_module.prefetch_in_analysis_data(ticker)
            logger.info(f"Prefetched India data for {ticker}: {list(prefetched.keys()) if prefetched else 'none'}")
        except Exception as e:
            logger.warning(f"India data prefetch failed, falling back to MCP: {e}")
            prefetched = {}

        # 5. Get India-specific agents (with prefetched data)
        agents = get_in_agent_directory(
            company_name, ticker, reference_date, base_sections, language,
            prefetched_data=prefetched
        )

        # 6. Execute analysis in HYBRID mode
        # - yfinance sections: sequential with 3 sec delay (rate limit friendly)
        # - news_analysis: parallel (uses yahoo_finance news tools)
        logger.info(f"Running India analysis in HYBRID mode for {company_name}...")
        logger.info(f"  - yfinance sections (sequential): {yfinance_sections}")
        logger.info(f"  - parallel sections: {parallel_sections}")

        async def process_yfinance_sections():
            """Process yfinance-dependent sections sequentially."""
            results = {}
            for section in yfinance_sections:
                if section in agents:
                    logger.info(f"Processing {section} for {company_name}...")
                    try:
                        agent = agents[section]
                        if section == "market_index_analysis":
                            if "report" in _in_market_analysis_cache:
                                logger.info("Using cached India market analysis")
                                report = _in_market_analysis_cache["report"]
                            else:
                                logger.info("Generating new India market analysis")
                                report = await generate_market_report(
                                    agent, section, reference_date, logger, language
                                )
                                _in_market_analysis_cache["report"] = report
                        else:
                            report = await generate_report(
                                agent, section, company_name, ticker, reference_date, logger, language
                            )
                        results[section] = report
                        # Delay between yfinance calls to avoid rate limits
                        await asyncio.sleep(3)
                    except Exception as e:
                        logger.error(f"Error processing {section}: {e}")
                        results[section] = f"Analysis failed: {section}"
            return results

        async def process_parallel_section(section):
            """Process a non-yfinance section with its own MCPApp context."""
            if section not in agents:
                return section, None

            section_app = MCPApp(name=f"in_stock_analysis_{section}")
            async with section_app.run() as section_context:
                section_logger = section_context.logger
                section_logger.info(f"Processing {section} for {company_name}...")
                try:
                    agent = agents[section]
                    report = await generate_report(
                        agent, section, company_name, ticker, reference_date, section_logger, language
                    )
                    return section, report
                except Exception as e:
                    section_logger.error(f"Error processing {section}: {e}")
                    return section, f"Analysis failed: {section}"

        # Execute hybrid: yfinance sequential + parallel sections concurrently
        parallel_tasks = [process_parallel_section(s) for s in parallel_sections]
        yfinance_task = process_yfinance_sections()
        all_results = await asyncio.gather(yfinance_task, *parallel_tasks)

        # Collect results
        yfinance_results = all_results[0]
        section_reports.update(yfinance_results)
        for result in all_results[1:]:
            if result and result[1] is not None:
                section_reports[result[0]] = result[1]

        # 7. Combine section reports
        combined_reports = ""
        for section in base_sections:
            if section in section_reports:
                combined_reports += f"\n\n--- {section.upper()} ---\n\n"
                combined_reports += section_reports[section]

        # 8. Generate investment strategy
        try:
            logger.info(f"Processing investment_strategy for {company_name}...")
            investment_strategy = await generate_investment_strategy(
                section_reports, combined_reports, company_name, ticker, reference_date, logger, language
            )
            section_reports["investment_strategy"] = investment_strategy.lstrip('\n')
            logger.info(f"Completed investment_strategy - {len(investment_strategy)} characters")
        except Exception as e:
            logger.error(f"Error processing investment_strategy: {e}")
            section_reports["investment_strategy"] = "Investment strategy analysis failed"

        # 9. Generate executive summary
        try:
            logger.info(f"Processing summary for {company_name}...")
            summary = await generate_summary(
                section_reports, company_name, ticker, reference_date, logger, language
            )
            import re
            summary = summary.lstrip('\n')
            summary = re.sub(
                r'^#\s*' + re.escape(company_name) + r'\s*\(' + re.escape(ticker) + r'\)[^\n]*\n+',
                '', summary, flags=re.IGNORECASE
            )
            summary = re.sub(
                r'^\*{0,2}(Publication Date|발행일)\*{0,2}\s*:\s*[^\n]+\n+',
                '', summary, flags=re.IGNORECASE
            )
            summary = re.sub(r'^-{3,}\s*\n+', '', summary)
            section_reports["summary"] = summary.lstrip('\n')
            logger.info(f"Completed summary - {len(summary)} characters")
        except Exception as e:
            logger.error(f"Error processing summary: {e}")
            section_reports["summary"] = "Summary generation failed"

        # 10. Generate charts
        price_chart_html = ""
        institutional_chart_html = ""
        technical_chart_html = ""

        try:
            import yfinance as yf

            stock = yf.Ticker(f"{ticker}.NS")
            hist = stock.history(period="1y")

            if not hist.empty:
                price_chart_html = get_in_price_chart_html(
                    ticker, company_name, hist, width=900, dpi=80
                )
                if price_chart_html:
                    logger.info(f"Generated price chart for {ticker}.NS")

                major_holders = stock.major_holders
                institutional_holders = stock.institutional_holders
                institutional_chart_html = get_in_institutional_chart_html(
                    ticker, company_name, major_holders, institutional_holders, width=900, dpi=80
                )
                if institutional_chart_html:
                    logger.info(f"Generated institutional chart for {ticker}.NS")

                technical_chart_html = get_in_technical_chart_html(
                    ticker, company_name, hist, width=900, dpi=80
                )
                if technical_chart_html:
                    logger.info(f"Generated technical indicators chart for {ticker}.NS")

        except Exception as e:
            logger.warning(f"Chart generation skipped: {e}")

        # 11. Compile final report
        formatted_date = f"{reference_date[:4]}.{reference_date[4:6]}.{reference_date[6:]}"

        price_chart_section = f"\n\n#### Price Chart\n\n{price_chart_html}\n" if price_chart_html else ""
        institutional_chart_section = f"\n\n#### Shareholding Pattern Chart\n\n{institutional_chart_html}\n" if institutional_chart_html else ""
        technical_chart_section = f"\n\n#### Technical Indicators (RSI & MACD)\n\n{technical_chart_html}\n" if technical_chart_html else ""

        # English headers (India module default)
        headers = {
            "title": f"# {company_name} ({ticker}.NS) Analysis Report",
            "pub_date": "Publication Date",
            "exec_summary": "## Executive Summary",
            "tech_analysis": "## 1. Technical Analysis",
            "fundamental": "## 2. Fundamental Analysis",
            "news": "## 3. Recent Major News Summary",
            "market": "## 4. Market Analysis",
            "strategy": "## 5. Investment Strategy and Opinion",
        }

        final_report = f"""{headers["title"]}

**{headers["pub_date"]}:** {formatted_date}

---

{section_reports.get("summary", headers["exec_summary"] + " - Summary not available")}

---

{headers["tech_analysis"]}

{section_reports.get("price_volume_analysis", "Analysis not available")}
{price_chart_section}
{section_reports.get("institutional_holdings_analysis", "Analysis not available")}
{institutional_chart_section}
---

{headers["fundamental"]}

{section_reports.get("company_status", "Analysis not available")}

{section_reports.get("company_overview", "Analysis not available")}

---

{headers["news"]}

{section_reports.get("news_analysis", "Analysis not available")}

---

{headers["market"]}

{section_reports.get("market_index_analysis", "Analysis not available")}
{technical_chart_section}
---

{headers["strategy"]}

{section_reports.get("investment_strategy", "Strategy not available")}

---

{get_disclaimer(language)}
"""

        final_report = clean_markdown(final_report)
        logger.info(f"Final report generated: {company_name}({ticker}.NS) - {len(final_report)} characters")
        return final_report


def clear_in_market_cache():
    """Clear the India market analysis cache."""
    global _in_market_analysis_cache
    _in_market_analysis_cache = {}


if __name__ == "__main__":
    import time
    import threading
    import signal

    def exit_after_timeout():
        time.sleep(3600)
        print("60-minute timeout reached: forcefully terminating process")
        os.kill(os.getpid(), signal.SIGTERM)

    timer_thread = threading.Thread(target=exit_after_timeout, daemon=True)
    timer_thread.start()

    start = time.time()

    result = asyncio.run(analyze_in_stock(
        ticker="RELIANCE",
        company_name="Reliance Industries Limited",
        reference_date=datetime.now().strftime("%Y%m%d"),
        language="en"
    ))

    output_path = f"RELIANCE_Reliance Industries_{datetime.now().strftime('%Y%m%d')}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")
    print(f"Final report length: {len(result):,} characters")
    print(f"Report saved to: {output_path}")
