from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


# Language name mapping for report generation
LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}


@retry(
    stop=stop_after_attempt(2),  # Maximum 2 attempts (initial + 1 retry)
    wait=wait_exponential(multiplier=1, min=10, max=30),  # Exponentially increasing wait time
    retry=retry_if_exception_type(Exception)  # Retry on all exceptions
)
async def generate_report(agent, section, company_name, company_code, reference_date, logger, language="en"):
    """
    Generate report using agent with retry logic

    Args:
        agent: Analysis agent
        section: Report section name
        company_name: Company name
        company_code: Stock code
        reference_date: Analysis reference date (YYYYMMDD)
        logger: Logger
        language: Report language code (default: "en")
    """
    language_name = LANGUAGE_NAMES.get(language, language.upper())

    llm = await agent.attach_llm(OpenAIAugmentedLLM)

    # Create language-specific message
    message = f"""Please write an analysis report for {section} of {company_name}({company_code}).
(Report language: {language_name})

## Analysis and Report Writing Guidelines:
1. Perform all processes from data collection to analysis.
2. Write detailed reports while focusing on key information.
3. Write at a level that is easy for general individual investors to understand.
4. Focus on practical content that directly helps investment decisions.
5. Analyze based only on actual collected data, and do not speculate on missing data.
6. **Always translate company names to {language_name}.**

## Format Requirements:
1. Always start the report with two newline characters (\\n\\n) before the title.
2. Follow the format specified in the agent's instructions for section titles and structure.
3. Divide paragraphs appropriately for readability and emphasize important content.

## Output Format Rules:
- Write sentences in natural prose style. Do not break lines in the middle of sentences.
- Do not use unnecessary bullet points. Use them only when listing is absolutely necessary.
- Each paragraph should consist of complete sentences.
- General explanations (not table data) must be written in sentence form.
- ⚠️ Do NOT use ## (h2 headers) arbitrarily in the middle of content. Use **bold text** or ### for sub-sections.

## ⚠️ CHARACTER LIMIT: Keep the report under 3000 characters. Be concise and focus on key insights!

##Analysis Date: {reference_date} (YYYYMMDD format)
"""

    report = await llm.generate_str(
        message=message,
        request_params=RequestParams(
            model="openai/gpt-4.1",
            maxTokens=16000,
            parallel_tool_calls=True,
            use_history=True
        )
    )
    logger.info(f"Completed {section} - {len(report)} characters")
    return report

async def generate_market_report(agent, section, reference_date, logger, language="en"):
    """
    Generate market analysis report using agent

    Args:
        agent: Analysis agent
        section: Report section name
        reference_date: Analysis reference date (YYYYMMDD)
        logger: Logger
        language: Report language code (default: "en")
    """
    language_name = LANGUAGE_NAMES.get(language, language.upper())

    llm = await agent.attach_llm(OpenAIAugmentedLLM)

    # Create message
    message = f"""Please write a market and macroeconomic analysis report.
(Report language: {language_name})

## Analysis and Report Writing Guidelines:
1. Perform all processes from data collection to analysis.
2. Write detailed reports while focusing on key information.
3. Write at a level that is easy for general individual investors to understand.
4. Focus on practical content that directly helps investment decisions.
5. Analyze based only on actual collected data, and do not speculate on missing data.
6. **Always translate company names to {language_name}.**

## Format Requirements:
1. Always start the report with two newline characters (\\n\\n) before the title.
2. Follow the format specified in the agent's instructions for section titles and structure.
3. Divide paragraphs appropriately for readability and emphasize important content.

## Output Format Rules:
- Write sentences in natural prose style. Do not break lines in the middle of sentences.
- Do not use unnecessary bullet points. Use them only when listing is absolutely necessary.
- Each paragraph should consist of complete sentences.
- General explanations (not table data) must be written in sentence form.
- ⚠️ Do NOT use ## (h2 headers) arbitrarily in the middle of content. Use **bold text** or ### for sub-sections.

## ⚠️ CHARACTER LIMIT: Keep the report under 3000 characters. Be concise and focus on key insights!

##Analysis Date: {reference_date} (YYYYMMDD format)
"""

    report = await llm.generate_str(
        message=message,
        request_params=RequestParams(
            model="openai/gpt-5",
            maxTokens=16000,
            max_iterations=3,
            parallel_tool_calls=True,
            use_history=True
        )
    )
    logger.info(f"Completed {section} - {len(report)} characters")
    return report


async def generate_summary(section_reports, company_name, company_code, reference_date, logger, language="en"):
    """
    Generate executive summary based on section reports

    Args:
        section_reports: Dictionary of reports by section
        company_name: Company name
        company_code: Stock code
        reference_date: Analysis reference date (YYYYMMDD)
        logger: Logger
        language: Report language code (default: "en")
    """
    try:
        from mcp_agent.agents.agent import Agent
        from mcp_agent.workflows.llm.augmented_llm import RequestParams
        from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

        language_name = LANGUAGE_NAMES.get(language, language.upper())

        # Generate comprehensive report including all sections
        all_reports = ""
        for section, report in section_reports.items():
            all_reports += f"\n\n--- {section.upper()} ---\n\n"
            all_reports += report

        logger.info(f"Generating executive summary for {company_name}...")

        # Create instruction and message
        instruction = f"""
You are an investment expert who writes executive summaries of the {company_name} ({company_code}) company analysis report.
Extract and concisely summarize the 3-5 most important key points from each section of the entire report.
Provide a summary that investors can quickly read and understand the key points.

**Always translate company names to {language_name}.**

##Analysis Date: {reference_date} (YYYYMMDD format)
"""
        message = f"""Based on the comprehensive analysis report of {company_name}({company_code}) below, please write a summary of key investment points.
(Report language: {language_name})

The summary should include the company's current situation, investment attraction points, major risk factors, suitable investor types, etc.
Write a concise yet insightful summary of about 500-800 characters.

## Format Guidelines:
- Title: "## Executive Summary" (markdown ## required)
- First paragraph: Overview of the company's current situation and investment perspective
- Bullet points: 3-5 key investment points
- Last paragraph: Suggested investor types and approaches

## Style Guidelines:
- Use concise and clear sentences
- Focus on practical content that directly helps investment decisions
- Use conditional/probabilistic expressions rather than definitive expressions
- All points are based on technical/fundamental analysis data
- **Always translate company names to {language_name}.**

Comprehensive Analysis Report:
{all_reports}
"""

        summary_agent = Agent(
            name="summary_agent",
            instruction=instruction
        )

        llm = await summary_agent.attach_llm(OpenAIAugmentedLLM)
        executive_summary = await llm.generate_str(
            message=message,
            request_params=RequestParams(
                model="openai/gpt-5",
                maxTokens=16000,
                max_iterations=2,
                parallel_tool_calls=True,
                use_history=True
            )
        )
        return executive_summary
    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        return "## Executive Summary\n\nA problem occurred while generating the analysis summary."


async def generate_investment_strategy(section_reports, combined_reports, company_name, company_code, reference_date, logger, language="en"):
    """
    Generate investment strategy report

    Args:
        section_reports: Dictionary of reports by section
        combined_reports: Combined report content
        company_name: Company name
        company_code: Stock code
        reference_date: Analysis reference date (YYYYMMDD)
        logger: Logger
        language: Report language code (default: "en")
    """
    from mcp_agent.agents.agent import Agent
    from mcp_agent.workflows.llm.augmented_llm import RequestParams
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

    language_name = LANGUAGE_NAMES.get(language, language.upper())

    try:
        logger.info(f"Processing investment_strategy for {company_name}...")

        # Create instruction and message
        instruction = f"""You are an investment strategy expert. Synthesize the previously analyzed technical analysis, company information, financial analysis, news trends, and market analysis to present investment strategies and opinions.

**Always translate company names to {language_name}.**

## Analysis Integration Elements
1. Stock Price/Volume Analysis Summary - Price trends, major support/resistance levels, volume patterns
2. Investor Trading Trends Analysis Summary - Institutional/foreign/retail trading patterns
3. Company Basic Information Summary - Core business model, competitiveness, growth drivers
4. News Analysis Summary - Major issues, market reactions, upcoming events
5. Market Analysis Summary - Market volatility factors, current status, trends, macroeconomic environment, technical analysis, market investment strategy

## Investment Strategy Components
1. Comprehensive Investment Perspective - Investment outlook combining technical/fundamental analysis
2. Strategies by Investor Type
   - Short-term trader perspective (within 1 month)
   - Swing trader perspective (1-3 months)
   - Mid-term investor perspective (3-12 months)
   - Long-term investor perspective (over 1 year)
   - Perspectives for new entrants and existing holders (explained using position sizing)
3. Key Trading Points
   - Buy consideration price range and conditions
   - Sell/stop-loss price range and conditions
   - Profit-taking strategy
4. Core Monitoring Elements
   - Technical signals to watch
   - Performance indicators to pay attention to
   - News and events to check
   - Market conditions to check
5. Risk Factors
   - Potential downside risks
   - Upside opportunity factors
   - Risk management measures

## Writing Style
- Present investment views based on objective data
- Present conditional scenarios rather than definitive predictions
- Provide differentiated strategies considering various investment preferences and timeframes
- Present specific price ranges and executable strategies
- Balanced risk-reward analysis

## Output Format Rules
- Write sentences in natural prose style. Do not break lines in the middle of sentences.
- Do not use unnecessary bullet points. Use them only when listing is absolutely necessary.
- Each paragraph should consist of complete sentences.
- General explanations (not table data) must be written in sentence form.
- ⚠️ Do NOT add arbitrary ## (h2 headers) in the middle of content. Use only the defined section structure.

## Report Format
- Insert 2 newline characters at the start of the report (\\n\\n)
- Title: "### 5-1. Investment Strategy and Opinion" (markdown ### required - main section header is added separately)
- Sub-sections MUST use "#### Sub-section Title" format (markdown #### required)
- Clearly distinguish strategies by investor type
- Express key trading points with specific price ranges and conditions
- Explain risk factors according to importance

## Cautions
- Provide as "investment reference information" not "investment solicitation"
- Avoid unilateral buy/sell solicitation, present conditional approaches
- Avoid excessive optimism or pessimism
- All investment strategies are based on actual data from technical/fundamental analysis
- Use expressions like "~possibility", "~expected" rather than definitive expressions like "certainly", "definitely"
- Clearly state that all investments involve risks

## Conclusion
- Provide a brief summary and 3-5 key investment points at the end
- Include the statement "This report is for investment reference only, and investment decisions and responsibilities lie with the investor."

Company: {company_name} ({company_code})
##Analysis Date: {reference_date} (YYYYMMDD format)
"""
        message = f"""Please write an investment strategy analysis report for {company_name}({company_code}).
(Report language: {language_name})

## Contents of Other Previously Analyzed Sections:
{combined_reports}

## Investment Strategy Writing Guidelines:
Based on all previously analyzed information, write a comprehensive investment strategy report.
Follow the guidelines set in the investment strategy agent, but pay particular attention to the following:

1. Reinterpret the various analyzed data (technical/fundamental/news) from an integrated perspective, not just a simple summary
2. Evaluate investment attractiveness at the current stock price level ({reference_date})
3. Present investment scenarios linking valuation and earnings outlook
4. Analyze relative investment attractiveness within the overall industry and market flow
5. **Always translate company names to {language_name}.**

Please present a consistent and executable investment strategy that investors can use for actual decision-making.

## Format and Style Requirements:
- Follow the previously set format (title, structure, style) as is
- Focus on presenting practical strategies that investors can act on

## ⚠️ CHARACTER LIMIT: Keep the report under 3000 characters. Be concise and focus on key insights!
"""

        investment_strategy_agent = Agent(
            name="investment_strategy_agent",
            instruction=instruction
        )

        llm = await investment_strategy_agent.attach_llm(OpenAIAugmentedLLM)
        investment_strategy = await llm.generate_str(
            message=message,
            request_params=RequestParams(
                model="openai/gpt-5",
                maxTokens=16000,
                max_iterations=3,
                parallel_tool_calls=True,
                use_history=True
            )
        )
        logger.info(f"Completed investment_strategy - {len(investment_strategy)} characters")
        return investment_strategy
    except Exception as e:
        logger.error(f"Error processing investment_strategy: {e}")
        return "Investment strategy analysis failed"


def get_disclaimer(language="en"):
    """
    Get disclaimer text

    Args:
        language: Disclaimer language code (default: "en")

    Returns:
        Disclaimer text in specified language
    """
    return """## Investment Disclaimer

This report is provided for informational purposes only and is not intended as investment advice.
The content in this report is AI-generated based on reliable sources as of the time of writing,
but its accuracy and completeness are not guaranteed.

Investments should be made carefully at your own judgment and risk,
and you are solely responsible for any investment results based on this report."""
