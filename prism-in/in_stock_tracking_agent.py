#!/usr/bin/env python3
"""
India (NSE) Stock Tracking and Trading Agent

This module performs buy/sell decisions using AI-based India stock analysis reports
and manages trading records.

Main Features:
1. Generate trading scenarios based on analysis reports
2. Manage stock purchases/sales (maximum 10 slots)
3. Track trading history and returns
4. Share results through Telegram channel

Key Differences from KR/US Versions:
- Uses NSE ticker symbols (RELIANCE, TCS, INFY)
- Uses yfinance with .NS suffix for price data
- Uses INR (₹) currency
- NSE market hours (09:15-15:30 IST)
- Uses in_* database tables
- India-specific: T+1 settlement, circuit limits, STT, FII/DII flows
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
PRISM_IN_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PRISM_IN_DIR))

from telegram import Bot
from telegram.error import TelegramError

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"in_stock_tracking_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# MCP imports
from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


# =============================================================================
# Helper function to import modules from main project cores/ (avoid namespace collision)
# =============================================================================
def _import_from_main_cores(module_name: str, relative_path: str):
    """Import module directly from main project cores/ directory."""
    import importlib.util
    file_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Pre-load telegram_translator_agent from main project
_translator_module = _import_from_main_cores(
    "telegram_translator_agent",
    "cores/agents/telegram_translator_agent.py"
)
translate_telegram_message = _translator_module.translate_telegram_message

try:
    from cores.agents.trading_agents import create_in_trading_scenario_agent
    from tracking.db_schema import (
        create_in_tables,
        create_in_indexes,
        add_sector_column_if_missing,
        add_market_column_to_shared_tables,
        migrate_in_performance_tracker_columns,
        is_in_ticker_in_holdings,
        get_in_holdings_count,
    )
    from tracking.journal import INJournalManager
    from tracking.compression import INCompressionManager
except ImportError as e:
    logger.warning(f"Direct import failed: {e}, trying fallback...")
    _prism_in_fallback = Path(__file__).parent
    if str(_prism_in_fallback) not in sys.path:
        sys.path.insert(0, str(_prism_in_fallback))
    from cores.agents.trading_agents import create_in_trading_scenario_agent
    from tracking.db_schema import (
        create_in_tables,
        create_in_indexes,
        add_sector_column_if_missing,
        add_market_column_to_shared_tables,
        migrate_in_performance_tracker_columns,
        is_in_ticker_in_holdings,
        get_in_holdings_count,
    )
    from tracking.journal import INJournalManager
    from tracking.compression import INCompressionManager

# Create MCPApp instance
app = MCPApp(name="in_stock_tracking")


# =============================================================================
# India-Specific Helper Functions
# =============================================================================

def extract_ticker_info(report_path: str) -> Tuple[str, str]:
    """Extract ticker and company name from report file path."""
    try:
        file_name = Path(report_path).stem
        pattern = r'^([A-Z]+)_([^_]+)'
        match = re.match(pattern, file_name)

        if match:
            return match.group(1), match.group(2)
        else:
            parts = file_name.split('_')
            if len(parts) >= 2:
                return parts[0], parts[1]

        logger.error(f"Cannot extract ticker info from filename: {file_name}")
        return "", ""
    except Exception as e:
        logger.error(f"Error extracting ticker info: {str(e)}")
        return "", ""


async def get_current_stock_price(cursor, ticker: str) -> float:
    """Get current India stock price using yfinance (.NS suffix)."""
    try:
        import yfinance as yf

        yf_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        current_price = info.get('regularMarketPrice', 0) or info.get('previousClose', 0)

        if current_price > 0:
            logger.info(f"{ticker} current price: ₹{current_price:.2f}")
            return float(current_price)
        else:
            logger.warning(f"Cannot get price for {ticker}")
            return _get_last_price_from_db(cursor, ticker)

    except Exception as e:
        logger.error(f"Error querying current price for {ticker}: {str(e)}")
        return _get_last_price_from_db(cursor, ticker)


def _get_last_price_from_db(cursor, ticker: str) -> float:
    """Get last saved price from DB as fallback."""
    try:
        cursor.execute(
            "SELECT current_price FROM in_stock_holdings WHERE ticker = ?",
            (ticker,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            last_price = float(row[0])
            logger.warning(f"{ticker} price query failed, using last price: ₹{last_price:.2f}")
            return last_price
    except:
        pass
    return 0.0


async def get_trading_value_rank_change(ticker: str) -> Tuple[float, str]:
    """Calculate trading value ranking change for an India stock."""
    try:
        import yfinance as yf

        yf_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period="5d")

        if hist.empty or len(hist) < 2:
            return 0, "Insufficient historical data"

        recent_volume = hist['Volume'].iloc[-1]
        previous_volume = hist['Volume'].iloc[-2]
        recent_price = hist['Close'].iloc[-1]
        previous_price = hist['Close'].iloc[-2]

        recent_value = recent_volume * recent_price
        previous_value = previous_volume * previous_price

        if previous_value > 0:
            value_change_percentage = ((recent_value - previous_value) / previous_value) * 100
        else:
            value_change_percentage = 0

        avg_volume = hist['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        result_msg = (
            f"Trading value: ₹{recent_value/1e7:.1f}Cr "
            f"(prev: ₹{previous_value/1e7:.1f}Cr, "
            f"change: {'▲' if value_change_percentage > 0 else '▼' if value_change_percentage < 0 else '='}"
            f"{abs(value_change_percentage):.1f}%), "
            f"Volume ratio: {volume_ratio:.2f}x"
        )

        logger.info(f"{ticker} {result_msg}")
        return value_change_percentage, result_msg

    except Exception as e:
        logger.error(f"Error analyzing trading value for {ticker}: {str(e)}")
        return 0, "Trading value analysis failed"


def check_sector_diversity(cursor, sector: str, max_same_sector: int, concentration_ratio: float) -> bool:
    """Check for over-concentration in same sector."""
    try:
        if not sector or sector.lower() == "unknown":
            return True

        cursor.execute("SELECT scenario FROM in_stock_holdings")
        holdings_scenarios = cursor.fetchall()

        sectors = []
        for row in holdings_scenarios:
            if row[0]:
                try:
                    scenario_data = json.loads(row[0])
                    if 'sector' in scenario_data:
                        sectors.append(scenario_data['sector'])
                except:
                    pass

        same_sector_count = sum(1 for s in sectors if s and s.lower() == sector.lower())

        if same_sector_count >= max_same_sector or \
           (sectors and same_sector_count / len(sectors) >= concentration_ratio):
            logger.warning(
                f"Sector '{sector}' over-concentration risk: "
                f"Currently holding {same_sector_count} stocks "
                f"(max {max_same_sector}, limit {concentration_ratio*100:.0f}%)"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking sector diversity: {str(e)}")
        return True


def parse_price_value(value: Any) -> float:
    """Parse price value and convert to number (handles ₹ and comma formatting)."""
    try:
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            value = value.replace(',', '').replace('₹', '').replace('$', '').strip()

            range_patterns = [r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)']
            for pattern in range_patterns:
                match = re.search(pattern, value)
                if match:
                    low = float(match.group(1))
                    high = float(match.group(2))
                    return (low + high) / 2

            number_match = re.search(r'(\d+(?:\.\d+)?)', value)
            if number_match:
                return float(number_match.group(1))

        return 0
    except Exception as e:
        logger.warning(f"Failed to parse price value: {value} - {str(e)}")
        return 0


def default_scenario() -> Dict[str, Any]:
    """Return default trading scenario for India stocks."""
    return {
        "portfolio_analysis": "Analysis failed",
        "buy_score": 0,
        "decision": "no_entry",
        "target_price": 0,
        "stop_loss": 0,
        "investment_period": "short",
        "rationale": "Analysis failed",
        "sector": "Unknown",
        "considerations": "Analysis failed"
    }


# =============================================================================
# India Stock Tracking Agent
# =============================================================================

class INStockTrackingAgent:
    """India (NSE) Stock Tracking and Trading Agent"""

    MAX_SLOTS = 10
    MAX_SAME_SECTOR = 3
    SECTOR_CONCENTRATION_RATIO = 0.3

    PERIOD_SHORT = "short"
    PERIOD_MEDIUM = "medium"
    PERIOD_LONG = "long"

    SCORE_STRONG_BUY = 8
    SCORE_CONSIDER = 7
    SCORE_UNSUITABLE = 6

    def __init__(
        self,
        db_path: str = "stock_tracking_db.sqlite",
        telegram_token: str = None,
        enable_journal: bool = False
    ):
        self.max_slots = self.MAX_SLOTS
        self.message_queue = []
        self._broadcast_task = None
        self.trading_agent = None
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.language = "en"  # Default to English for India
        self.enable_journal = enable_journal

        self.journal_manager = None
        self.compression_manager = None

        self.telegram_token = telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_bot = None
        if self.telegram_token:
            self.telegram_bot = Bot(token=self.telegram_token)

    async def initialize(self, language: str = "en"):
        """Create necessary tables and initialize."""
        logger.info("Starting India tracking agent initialization")

        self.language = language

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        self.trading_agent = create_in_trading_scenario_agent(language=language)

        await self._create_tables()

        self.journal_manager = INJournalManager(
            cursor=self.cursor,
            conn=self.conn,
            language=language,
            enable_journal=self.enable_journal
        )

        self.compression_manager = INCompressionManager(
            cursor=self.cursor,
            conn=self.conn
        )

        logger.info(f"India tracking agent initialization complete (journal: {self.enable_journal})")
        return True

    async def _create_tables(self):
        """Create necessary India database tables."""
        create_in_tables(self.cursor, self.conn)
        create_in_indexes(self.cursor, self.conn)
        add_sector_column_if_missing(self.cursor, self.conn)
        add_market_column_to_shared_tables(self.cursor, self.conn)
        migrate_in_performance_tracker_columns(self.cursor, self.conn)

    def _normalize_decision(self, decision: str) -> str:
        """Normalize decision string for comparison."""
        if not decision:
            return "no_entry"
        d = decision.lower().strip()
        if d in ("enter", "entry", "yes", "buy"):
            return "entry"
        elif d in ("no entry", "no_entry", "no-entry", "no", "skip", "pass"):
            return "no_entry"
        return d

    async def _extract_ticker_info(self, report_path: str) -> Tuple[str, str]:
        return extract_ticker_info(report_path)

    async def _get_current_stock_price(self, ticker: str) -> float:
        return await get_current_stock_price(self.cursor, ticker)

    async def _get_trading_value_rank_change(self, ticker: str) -> Tuple[float, str]:
        return await get_trading_value_rank_change(ticker)

    async def _is_ticker_in_holdings(self, ticker: str) -> bool:
        return is_in_ticker_in_holdings(self.cursor, ticker)

    async def _get_current_slots_count(self) -> int:
        return get_in_holdings_count(self.cursor)

    async def _check_sector_diversity(self, sector: str) -> bool:
        return check_sector_diversity(
            self.cursor, sector,
            self.MAX_SAME_SECTOR, self.SECTOR_CONCENTRATION_RATIO
        )

    async def _extract_trading_scenario(
        self,
        report_content: str,
        rank_change_msg: str = "",
        ticker: str = None,
        sector: str = None,
        trigger_type: str = "",
        trigger_mode: str = ""
    ) -> Dict[str, Any]:
        """Extract trading scenario from report."""
        try:
            current_slots = await self._get_current_slots_count()

            self.cursor.execute("""
                SELECT ticker, company_name, buy_price, current_price, scenario
                FROM in_stock_holdings
            """)
            holdings = [dict(row) for row in self.cursor.fetchall()]

            sector_distribution = {}
            investment_periods = {"short": 0, "medium": 0, "long": 0}

            for holding in holdings:
                scenario_str = holding.get('scenario', '{}')
                try:
                    if isinstance(scenario_str, str):
                        scenario_data = json.loads(scenario_str)
                        sector_name = scenario_data.get('sector', 'Unknown')
                        sector_distribution[sector_name] = sector_distribution.get(sector_name, 0) + 1
                        period = scenario_data.get('investment_period', 'medium')
                        investment_periods[period] = investment_periods.get(period, 0) + 1
                except:
                    pass

            portfolio_info = f"""
            Current holdings: {current_slots}/{self.max_slots}
            Sector distribution: {json.dumps(sector_distribution, ensure_ascii=False)}
            Investment period distribution: {json.dumps(investment_periods, ensure_ascii=False)}
            """

            journal_context = ""
            score_adjustment_info = ""
            if ticker:
                journal_context = self.get_journal_context(
                    ticker=ticker, sector=sector, trigger_type=trigger_type
                )
                adjustment, reasons = self.get_score_adjustment(ticker, sector, trigger_type)
                if adjustment != 0 or reasons:
                    score_adjustment_info = f"""
                ### Score Adjustment Suggestion (Experience-Based)
                - Recommended Adjustment: {'+' if adjustment > 0 else ''}{adjustment} points
                - Reason: {', '.join(reasons) if reasons else 'N/A'}
                """

            llm = await self.trading_agent.attach_llm(OpenAIAugmentedLLM)

            trigger_info_section = ""
            if trigger_type:
                trigger_info_section = f"""
                ### Trigger Info (Apply Trigger-Based Entry Criteria)
                - **Triggered By**: {trigger_type}
                - **Trigger Mode**: {trigger_mode or 'unknown'}
                """

            prompt_message = f"""
            This is an AI analysis report for an India (NSE) stock. Generate a trading scenario.

            ### Current Portfolio Status:
            {portfolio_info}
            {trigger_info_section}
            ### Trading Value Analysis:
            {rank_change_msg}
            {score_adjustment_info}
            {journal_context}

            ### India-Specific Considerations:
            - NSE T+1 settlement
            - Circuit limits: 5%/10%/20% depending on stock
            - STT (Securities Transaction Tax) on sell side
            - FII/DII flow data impacts price action
            - Promoter holding >50% is positive signal
            - Pledged shares >20% is a red flag

            ### Report Content:
            {report_content}
            """

            response = await llm.generate_str(
                message=prompt_message,
                request_params=RequestParams(model="openai/gpt-5", maxTokens=16000)
            )

            try:
                def fix_json_syntax(json_str):
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    json_str = re.sub(r'(\])\s*(\n\s*")', r'\1,\2', json_str)
                    json_str = re.sub(r'(})\s*(\n\s*")', r'\1,\2', json_str)
                    json_str = re.sub(r'([0-9]|")\s*(\n\s*")', r'\1,\2', json_str)
                    json_str = re.sub(r',\s*,', ',', json_str)
                    return json_str

                markdown_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response, re.DOTALL)
                if markdown_match:
                    json_str = fix_json_syntax(markdown_match.group(1))
                    return json.loads(json_str)

                json_match = re.search(r'({[\s\S]*?})(?:\s*$|\n\n)', response, re.DOTALL)
                if json_match:
                    json_str = fix_json_syntax(json_match.group(1))
                    return json.loads(json_str)

                clean_response = fix_json_syntax(response)
                return json.loads(clean_response)

            except Exception as json_err:
                logger.error(f"Trading scenario JSON parse error: {json_err}")
                try:
                    import json_repair
                    repaired = json_repair.repair_json(response)
                    return json.loads(repaired)
                except:
                    pass
                return default_scenario()

        except Exception as e:
            logger.error(f"Error extracting trading scenario: {str(e)}")
            logger.error(traceback.format_exc())
            return default_scenario()

    async def analyze_report(self, pdf_report_path: str) -> Dict[str, Any]:
        """Analyze India stock analysis report and make trading decision."""
        try:
            logger.info(f"Starting report analysis: {pdf_report_path}")

            ticker, company_name = await self._extract_ticker_info(pdf_report_path)
            if not ticker or not company_name:
                return {"success": False, "error": "Failed to extract ticker info"}

            is_holding = await self._is_ticker_in_holdings(ticker)
            if is_holding:
                logger.info(f"{ticker} ({company_name}) already in holdings")
                holding_current_price = await self._get_current_stock_price(ticker)
                return {
                    "success": True, "decision": "holding",
                    "ticker": ticker, "company_name": company_name,
                    "current_price": holding_current_price
                }

            current_price = await self._get_current_stock_price(ticker)
            if current_price <= 0:
                return {"success": False, "error": "Current price query failed"}

            rank_change_percentage, rank_change_msg = await self._get_trading_value_rank_change(ticker)

            from pdf_converter import pdf_to_markdown_text
            report_content = pdf_to_markdown_text(pdf_report_path)

            trigger_info = getattr(self, 'trigger_info_map', {}).get(ticker, {})
            trigger_type = trigger_info.get('trigger_type', '')
            trigger_mode = trigger_info.get('trigger_mode', '')

            scenario = await self._extract_trading_scenario(
                report_content, rank_change_msg,
                ticker=ticker, sector=None,
                trigger_type=trigger_type, trigger_mode=trigger_mode
            )

            sector = scenario.get("sector", "Unknown")
            is_sector_diverse = await self._check_sector_diversity(sector)

            raw_decision = scenario.get("decision", "no_entry")
            normalized_decision = self._normalize_decision(raw_decision)

            return {
                "success": True,
                "ticker": ticker, "company_name": company_name,
                "current_price": current_price, "scenario": scenario,
                "decision": normalized_decision, "raw_decision": raw_decision,
                "sector": sector, "sector_diverse": is_sector_diverse,
                "rank_change_percentage": rank_change_percentage,
                "rank_change_msg": rank_change_msg
            }

        except Exception as e:
            logger.error(f"Error analyzing report: {str(e)}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    async def buy_stock(self, ticker: str, company_name: str, current_price: float,
                        scenario: Dict[str, Any], rank_change_msg: str = "") -> bool:
        """Process stock purchase (simulation - no broker API for India yet)."""
        try:
            if await self._is_ticker_in_holdings(ticker):
                logger.warning(f"{ticker} ({company_name}) already in holdings")
                return False

            current_slots = await self._get_current_slots_count()
            if current_slots >= self.max_slots:
                logger.warning(f"Holdings already at maximum ({self.max_slots})")
                return False

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trigger_info = getattr(self, 'trigger_info_map', {}).get(ticker, {})
            trigger_type = trigger_info.get('trigger_type', 'AI_Analysis')
            trigger_mode = trigger_info.get('trigger_mode', getattr(self, 'trigger_mode', 'unknown'))

            self.cursor.execute(
                """INSERT INTO in_stock_holdings
                (ticker, company_name, buy_price, buy_date, current_price, last_updated,
                 scenario, target_price, stop_loss, trigger_type, trigger_mode, sector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker, company_name, current_price, now, current_price, now,
                    json.dumps(scenario, ensure_ascii=False),
                    scenario.get('target_price', 0),
                    scenario.get('stop_loss', 0),
                    trigger_type, trigger_mode,
                    scenario.get('sector', 'Unknown')
                )
            )
            self.conn.commit()

            target_price = scenario.get('target_price', 0)
            stop_loss = scenario.get('stop_loss', 0)

            message = f"📈 New Buy: {company_name}({ticker}.NS)\n" \
                      f"Buy Price: ₹{current_price:,.2f}\n" \
                      f"Target: ₹{target_price:,.2f}\n" \
                      f"Stop Loss: ₹{stop_loss:,.2f}\n" \
                      f"Period: {scenario.get('investment_period', 'short')}\n" \
                      f"Sector: {scenario.get('sector', 'Unknown')}\n"

            trigger_win_rate = self._get_trigger_win_rate(trigger_type)
            if trigger_win_rate:
                message += f"{trigger_win_rate}\n"

            if scenario.get('valuation_analysis'):
                message += f"Valuation: {scenario.get('valuation_analysis')}\n"
            if scenario.get('sector_outlook'):
                message += f"Sector Outlook: {scenario.get('sector_outlook')}\n"
            if rank_change_msg:
                message += f"Trading Value Analysis: {rank_change_msg}\n"

            message += f"Rationale: {scenario.get('rationale', 'No information')}\n"

            self.message_queue.append(message)
            logger.info(f"{ticker} ({company_name}) purchase complete")
            return True

        except Exception as e:
            logger.error(f"{ticker} Error during purchase: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def _save_watchlist_item(
        self, ticker: str, company_name: str, current_price: float,
        buy_score: int, min_score: int, decision: str, skip_reason: str,
        scenario: Dict[str, Any], sector: str, was_traded: bool = False
    ) -> bool:
        """Save stocks not purchased to in_watchlist_history and in_analysis_performance_tracker."""
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_price = scenario.get('target_price', 0)
            stop_loss = scenario.get('stop_loss', 0)
            investment_period = scenario.get('investment_period', 'short')
            rationale = scenario.get('rationale', '')
            market_condition = scenario.get('market_condition', '')

            trigger_info = getattr(self, 'trigger_info_map', {}).get(ticker, {})
            trigger_type = trigger_info.get('trigger_type', '')
            trigger_mode = trigger_info.get('trigger_mode', '')
            risk_reward_ratio = trigger_info.get('risk_reward_ratio', scenario.get('risk_reward_ratio', 0))

            self.cursor.execute(
                """INSERT INTO in_watchlist_history
                (ticker, company_name, current_price, analyzed_date, buy_score, min_score,
                 decision, skip_reason, target_price, stop_loss, investment_period, sector,
                 scenario, portfolio_analysis, valuation_analysis, sector_outlook,
                 market_condition, rationale, trigger_type, trigger_mode, risk_reward_ratio, was_traded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker, company_name, current_price, now, buy_score, min_score,
                    decision, skip_reason, target_price, stop_loss, investment_period, sector,
                    json.dumps(scenario, ensure_ascii=False),
                    scenario.get('portfolio_analysis', ''),
                    scenario.get('valuation_analysis', ''),
                    scenario.get('sector_outlook', ''),
                    market_condition, rationale,
                    trigger_type, trigger_mode, risk_reward_ratio,
                    1 if was_traded else 0
                )
            )

            self.cursor.execute(
                """INSERT INTO in_analysis_performance_tracker
                (ticker, company_name, analysis_date, analysis_price,
                 predicted_direction, target_price, stop_loss, buy_score,
                 decision, skip_reason, risk_reward_ratio,
                 trigger_type, trigger_mode, sector,
                 tracking_status, was_traded, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    ticker, company_name, now, current_price,
                    'UP' if target_price > current_price else 'DOWN' if target_price < current_price else 'NEUTRAL',
                    target_price, stop_loss, buy_score,
                    decision, skip_reason, risk_reward_ratio,
                    trigger_type, trigger_mode, sector,
                    1 if was_traded else 0, now
                )
            )

            self.conn.commit()

            skip_message = f"⚠️ Buy Skipped: {company_name}({ticker}.NS)\n" \
                           f"Current Price: ₹{current_price:,.2f}\n" \
                           f"Buy Score: {buy_score}/{min_score}\n" \
                           f"Decision: {decision}\n" \
                           f"Sector: {sector}\n" \
                           f"Skip Reason: {skip_reason}\n" \
                           f"Analysis: {rationale if rationale else 'No information'}"

            trigger_win_rate = self._get_trigger_win_rate(trigger_type)
            if trigger_win_rate:
                skip_message += f"\n{trigger_win_rate}"

            self.message_queue.append(skip_message)
            logger.info(f"{ticker}({company_name}) Watchlist save complete - Score: {buy_score}/{min_score}")
            return True

        except Exception as e:
            logger.error(f"{ticker} Error saving watchlist: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def _analyze_sell_decision(self, stock_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Sell decision analysis for India stocks."""
        try:
            ticker = stock_data.get('ticker', '')
            buy_price = stock_data.get('buy_price', 0)
            buy_date = stock_data.get('buy_date', '')
            current_price = stock_data.get('current_price', 0)
            target_price = stock_data.get('target_price', 0)
            stop_loss = stock_data.get('stop_loss', 0)

            profit_rate = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0

            buy_datetime = datetime.strptime(buy_date, "%Y-%m-%d %H:%M:%S")
            days_passed = (datetime.now() - buy_datetime).days

            scenario_str = stock_data.get('scenario', '{}')
            investment_period = "medium"
            try:
                if isinstance(scenario_str, str):
                    scenario_data = json.loads(scenario_str)
                    investment_period = scenario_data.get('investment_period', 'medium')
            except:
                pass

            # Stop-loss
            if stop_loss > 0 and current_price <= stop_loss:
                return True, f"Stop-loss reached (stop-loss: ₹{stop_loss:,.2f})"

            # Target price reached
            if target_price > 0 and current_price >= target_price:
                return True, f"Target price achieved (target: ₹{target_price:,.2f})"

            # Short-term sell conditions
            if investment_period == "short":
                if days_passed >= 15 and profit_rate >= 5:
                    return True, f"Short-term goal achieved (holding: {days_passed} days, return: {profit_rate:.2f}%)"
                if days_passed >= 10 and profit_rate <= -3:
                    return True, f"Short-term loss protection (holding: {days_passed} days, return: {profit_rate:.2f}%)"

            # General sell conditions
            if profit_rate >= 10:
                return True, f"Return exceeds 10% (current: {profit_rate:.2f}%)"
            if profit_rate <= -5:
                return True, f"Loss exceeds -5% (current: {profit_rate:.2f}%)"
            if days_passed >= 30 and profit_rate < 0:
                return True, f"Held 30+ days with loss (holding: {days_passed} days, return: {profit_rate:.2f}%)"
            if days_passed >= 60 and profit_rate >= 3:
                return True, f"Held 60+ days with 3%+ profit (holding: {days_passed} days, return: {profit_rate:.2f}%)"
            if investment_period == "long" and days_passed >= 90 and profit_rate < 0:
                return True, f"Long-term loss cleanup (holding: {days_passed} days, return: {profit_rate:.2f}%)"

            return False, "Continue holding"

        except Exception as e:
            logger.error(f"Error analyzing sell decision: {str(e)}")
            return False, "Analysis error"

    async def _save_holding_decision(self, ticker: str, current_price: float,
                                      should_sell: bool, sell_reason: str,
                                      stock_data: Dict[str, Any]) -> bool:
        """Save AI sell decision results for held stocks."""
        try:
            now = datetime.now()
            buy_price = stock_data.get('buy_price', 0)
            profit_rate = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0

            decision_json = {
                "should_sell": should_sell,
                "sell_reason": sell_reason,
                "confidence": 7 if should_sell else 5,
                "analysis_summary": {
                    "technical_trend": "Rule-based analysis",
                    "volume_analysis": "",
                    "market_condition_impact": "",
                    "time_factor": f"Holding days: {stock_data.get('holding_days', 0)}"
                },
                "portfolio_adjustment": {
                    "needed": False, "reason": "",
                    "new_target_price": stock_data.get('target_price'),
                    "new_stop_loss": stock_data.get('stop_loss'),
                    "urgency": "low"
                },
                "current_price": current_price,
                "buy_price": buy_price,
                "profit_rate": profit_rate
            }

            full_json_data = json.dumps(decision_json, ensure_ascii=False)

            self.cursor.execute("DELETE FROM in_holding_decisions WHERE ticker = ?", (ticker,))
            self.cursor.execute("""
                INSERT INTO in_holding_decisions (
                    ticker, decision_date, decision_time, current_price, should_sell,
                    sell_reason, confidence, technical_trend, volume_analysis,
                    market_condition_impact, time_factor, portfolio_adjustment_needed,
                    adjustment_reason, new_target_price, new_stop_loss, adjustment_urgency,
                    full_json_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                current_price, should_sell, sell_reason,
                decision_json.get("confidence", 5),
                decision_json["analysis_summary"]["technical_trend"],
                decision_json["analysis_summary"]["volume_analysis"],
                decision_json["analysis_summary"]["market_condition_impact"],
                decision_json["analysis_summary"]["time_factor"],
                decision_json["portfolio_adjustment"]["needed"],
                decision_json["portfolio_adjustment"]["reason"],
                decision_json["portfolio_adjustment"]["new_target_price"],
                decision_json["portfolio_adjustment"]["new_stop_loss"],
                decision_json["portfolio_adjustment"]["urgency"],
                full_json_data
            ))

            self.conn.commit()
            logger.info(f"{ticker} India holding decision saved - should_sell: {should_sell}")
            return True

        except Exception as e:
            logger.error(f"{ticker} India holding decision save failed: {str(e)}")
            return False

    async def sell_stock(self, stock_data: Dict[str, Any], sell_reason: str) -> bool:
        """Process stock sale (simulation)."""
        try:
            ticker = stock_data.get('ticker', '')
            company_name = stock_data.get('company_name', '')
            buy_price = stock_data.get('buy_price', 0)
            buy_date = stock_data.get('buy_date', '')
            current_price = stock_data.get('current_price', 0)
            scenario_json = stock_data.get('scenario', '{}')
            trigger_type = stock_data.get('trigger_type', 'AI_Analysis')
            trigger_mode = stock_data.get('trigger_mode', 'unknown')
            sector = stock_data.get('sector', 'Unknown')

            profit_rate = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            buy_datetime = datetime.strptime(buy_date, "%Y-%m-%d %H:%M:%S")
            holding_days = (datetime.now() - buy_datetime).days
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.cursor.execute(
                """INSERT INTO in_trading_history
                (ticker, company_name, buy_price, buy_date, sell_price, sell_date,
                 profit_rate, holding_days, scenario, trigger_type, trigger_mode, sector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker, company_name, buy_price, buy_date,
                    current_price, now, profit_rate, holding_days,
                    scenario_json, trigger_type, trigger_mode, sector
                )
            )

            self.cursor.execute("DELETE FROM in_stock_holdings WHERE ticker = ?", (ticker,))
            self.conn.commit()

            arrow = "⬆️" if profit_rate > 0 else "⬇️" if profit_rate < 0 else "➖"
            message = f"📉 Sell: {company_name}({ticker}.NS)\n" \
                      f"Buy Price: ₹{buy_price:,.2f}\n" \
                      f"Sell Price: ₹{current_price:,.2f}\n" \
                      f"Return: {arrow} {abs(profit_rate):.2f}%\n" \
                      f"Holding Period: {holding_days} days\n" \
                      f"Sell Reason: {sell_reason}"

            trigger_win_rate = self._get_trigger_win_rate(trigger_type)
            if trigger_win_rate:
                message += f"\n{trigger_win_rate}"

            self.message_queue.append(message)
            logger.info(f"{ticker} ({company_name}) sell complete (return: {profit_rate:.2f}%)")

            if self.enable_journal and self.journal_manager:
                try:
                    await self.journal_manager.create_entry(
                        stock_data=stock_data, sell_price=current_price,
                        profit_rate=profit_rate, holding_days=holding_days,
                        sell_reason=sell_reason
                    )
                except Exception as journal_err:
                    logger.warning(f"Failed to create India journal entry: {journal_err}")

            return True

        except Exception as e:
            logger.error(f"Error during sell: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def update_holdings(self) -> List[Dict[str, Any]]:
        """Update holdings information and make sell decisions."""
        try:
            logger.info("Starting India holdings update")

            self.cursor.execute(
                """SELECT ticker, company_name, buy_price, buy_date, current_price,
                   scenario, target_price, stop_loss, last_updated,
                   trigger_type, trigger_mode, sector
                   FROM in_stock_holdings"""
            )
            holdings = [dict(row) for row in self.cursor.fetchall()]

            if not holdings:
                logger.info("No India holdings")
                return []

            sold_stocks = []

            for stock in holdings:
                ticker = stock.get('ticker')
                current_price = await self._get_current_stock_price(ticker)

                if current_price <= 0:
                    current_price = stock.get('current_price', 0)

                stock['current_price'] = current_price
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                should_sell, sell_reason = await self._analyze_sell_decision(stock)

                if should_sell:
                    self.cursor.execute("DELETE FROM in_holding_decisions WHERE ticker = ?", (ticker,))
                    sell_success = await self.sell_stock(stock, sell_reason)

                    if sell_success:
                        sold_stocks.append({
                            "ticker": ticker,
                            "company_name": stock.get('company_name', ''),
                            "buy_price": stock.get('buy_price', 0),
                            "sell_price": current_price,
                            "profit_rate": ((current_price - stock.get('buy_price', 0)) / stock.get('buy_price', 0) * 100) if stock.get('buy_price', 0) > 0 else 0,
                            "reason": sell_reason
                        })
                else:
                    await self._save_holding_decision(ticker, current_price, should_sell, sell_reason, stock)
                    self.cursor.execute(
                        "UPDATE in_stock_holdings SET current_price = ?, last_updated = ? WHERE ticker = ?",
                        (current_price, now, ticker)
                    )
                    self.conn.commit()

            return sold_stocks

        except Exception as e:
            logger.error(f"Error updating holdings: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def generate_report_summary(self) -> str:
        """Generate holdings and profit statistics summary."""
        try:
            self.cursor.execute(
                """SELECT ticker, company_name, buy_price, current_price, buy_date,
                   scenario, target_price, stop_loss, sector
                   FROM in_stock_holdings"""
            )
            holdings = [dict(row) for row in self.cursor.fetchall()]

            self.cursor.execute("SELECT SUM(profit_rate) FROM in_trading_history")
            total_profit = self.cursor.fetchone()[0] or 0

            self.cursor.execute("SELECT COUNT(*) FROM in_trading_history")
            total_trades = self.cursor.fetchone()[0] or 0

            self.cursor.execute("SELECT COUNT(*) FROM in_trading_history WHERE profit_rate > 0")
            successful_trades = self.cursor.fetchone()[0] or 0

            message = f"📊 PRISM India Simulator | Portfolio ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
            message += f"🔸 Current Holdings: {len(holdings) if holdings else 0}/{self.max_slots}\n"

            if holdings:
                profit_rates = []
                for h in holdings:
                    buy_price = h.get('buy_price', 0)
                    current_price = h.get('current_price', 0)
                    if buy_price > 0:
                        profit_rate = ((current_price - buy_price) / buy_price) * 100
                        profit_rates.append((h.get('ticker'), h.get('company_name'), profit_rate))

                if profit_rates:
                    best = max(profit_rates, key=lambda x: x[2])
                    worst = min(profit_rates, key=lambda x: x[2])
                    message += f"✅ Best Return: {best[1]}({best[0]}.NS) {'+' if best[2] > 0 else ''}{best[2]:.2f}%\n"
                    message += f"⚠️ Worst Return: {worst[1]}({worst[0]}.NS) {'+' if worst[2] > 0 else ''}{worst[2]:.2f}%\n"

                message += "\n🔸 Holdings List:\n"
                sector_counts = {}

                for stock in holdings:
                    ticker = stock.get('ticker', '')
                    company_name = stock.get('company_name', '')
                    buy_price = stock.get('buy_price', 0)
                    current_price = stock.get('current_price', 0)
                    buy_date = stock.get('buy_date', '')
                    target_price = stock.get('target_price', 0)
                    stop_loss = stock.get('stop_loss', 0)

                    sector = "Unknown"
                    try:
                        scenario_str = stock.get('scenario', '{}')
                        if isinstance(scenario_str, str):
                            scenario_data = json.loads(scenario_str)
                            sector = scenario_data.get('sector', 'Unknown')
                    except:
                        sector = stock.get('sector', 'Unknown')

                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    profit_rate = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                    arrow = "⬆️" if profit_rate > 0 else "⬇️" if profit_rate < 0 else "➖"

                    buy_datetime = datetime.strptime(buy_date, "%Y-%m-%d %H:%M:%S") if buy_date else datetime.now()
                    days_passed = (datetime.now() - buy_datetime).days

                    message += f"- {company_name}({ticker}.NS) [{sector}]\n"
                    message += f"  Buy: ₹{buy_price:.2f} / Current: ₹{current_price:.2f}\n"
                    message += f"  Target: ₹{target_price:.2f} / Stop: ₹{stop_loss:.2f}\n"
                    message += f"  Return: {arrow} {profit_rate:.2f}% / Holding: {days_passed} days\n\n"

                message += "🔸 Sector Distribution:\n"
                for sector, count in sector_counts.items():
                    percentage = (count / len(holdings)) * 100
                    message += f"- {sector}: {count} ({percentage:.1f}%)\n"
                message += "\n"
            else:
                message += "No holdings.\n\n"

            message += "🔸 Trading History\n"
            message += f"- Total Trades: {total_trades}\n"
            message += f"- Profitable: {successful_trades}\n"
            message += f"- Loss: {total_trades - successful_trades}\n"
            message += f"- Win Rate: {(successful_trades / total_trades * 100):.2f}%\n" if total_trades > 0 else "- Win Rate: 0.00%\n"
            message += f"- Cumulative Return: {total_profit:.2f}%\n\n"

            message += "📝 Important Notice:\n"
            message += "- This is an AI-based simulation. Not related to actual trading.\n"
            message += "- For reference only. Investment decisions are your responsibility.\n"
            message += "- This channel does not recommend buying/selling specific stocks."

            return message

        except Exception as e:
            logger.error(f"Error generating report summary: {str(e)}")
            return f"Error generating report: {str(e)}"

    async def process_reports(self, pdf_report_paths: List[str]) -> Tuple[int, int]:
        """Process analysis reports and make buy/sell decisions."""
        try:
            logger.info(f"Processing {len(pdf_report_paths)} India reports")

            buy_count = 0
            sell_count = 0

            sold_stocks = await self.update_holdings()
            sell_count = len(sold_stocks)

            for pdf_report_path in pdf_report_paths:
                analysis_result = await self.analyze_report(pdf_report_path)

                if not analysis_result.get("success", False):
                    continue

                if analysis_result.get("decision") == "holding":
                    continue

                ticker = analysis_result.get("ticker")
                company_name = analysis_result.get("company_name")
                current_price = analysis_result.get("current_price", 0)
                scenario = analysis_result.get("scenario", {})
                sector_diverse = analysis_result.get("sector_diverse", True)
                rank_change_msg = analysis_result.get("rank_change_msg", "")

                if not sector_diverse:
                    logger.info(f"Purchase deferred: {company_name} ({ticker}) - Sector over-concentration")
                    continue

                buy_score = scenario.get("buy_score", 0)
                min_score = scenario.get("min_score", 0)
                sector = analysis_result.get("sector", "Unknown")

                score_adjustment = 0
                adjustment_reasons = []
                trigger_info = getattr(self, 'trigger_info_map', {}).get(ticker, {})
                trigger_type = trigger_info.get('trigger_type', '')
                if self.enable_journal and ticker:
                    score_adjustment, adjustment_reasons = self.get_score_adjustment(ticker, sector, trigger_type)

                adjusted_score = buy_score + score_adjustment

                raw_decision = analysis_result.get("raw_decision", "")
                normalized_decision = analysis_result.get("decision", "no_entry")

                # Score-decision consistency enforcement
                if adjusted_score >= min_score and normalized_decision != "entry":
                    logger.info(f"Score-decision override: {company_name}({ticker}) - Score {adjusted_score} >= {min_score}")
                    normalized_decision = "entry"

                if normalized_decision == "entry":
                    buy_success = await self.buy_stock(ticker, company_name, current_price, scenario, rank_change_msg)
                    if buy_success:
                        buy_count += 1

                        # Save traded item to watchlist for tracking
                        await self._save_watchlist_item(
                            ticker=ticker, company_name=company_name,
                            current_price=current_price, buy_score=buy_score,
                            min_score=min_score, decision="entry",
                            skip_reason="", scenario=scenario,
                            sector=sector, was_traded=True
                        )
                else:
                    reason = ""
                    if adjusted_score < min_score:
                        reason = f"Score insufficient ({adjusted_score} < {min_score})"
                    else:
                        reason = f"No entry decision (raw: '{raw_decision}', normalized: '{normalized_decision}')"

                    await self._save_watchlist_item(
                        ticker=ticker, company_name=company_name,
                        current_price=current_price, buy_score=buy_score,
                        min_score=min_score, decision=normalized_decision,
                        skip_reason=reason, scenario=scenario,
                        sector=sector, was_traded=False
                    )

            logger.info(f"Report processing complete - Bought: {buy_count}, Sold: {sell_count}")
            return buy_count, sell_count

        except Exception as e:
            logger.error(f"Error processing reports: {str(e)}")
            logger.error(traceback.format_exc())
            return 0, 0

    async def _notify_firebase(self, message: str, chat_id: str, message_id: int = None):
        """Send Firebase Bridge notification for Prism Mobile push."""
        try:
            from firebase_bridge import notify
            await notify(message=message, market="in", telegram_message_id=message_id, channel_id=chat_id)
        except Exception as e:
            logger.debug(f"Firebase bridge: {e}")

    def _schedule_firebase(self, message: str, chat_id: str, message_id: int = None):
        return asyncio.create_task(self._notify_firebase(message, chat_id, message_id))

    async def send_telegram_message(self, chat_id: str, language: str = "en") -> bool:
        """Send message via Telegram."""
        try:
            if not chat_id:
                logger.info("No Telegram channel ID. Skipping message send")
                for message in self.message_queue:
                    logger.info(f"[Message (not sent)] {message[:100]}...")
                self.message_queue = []
                return True

            if not self.telegram_bot:
                logger.warning("Telegram bot not initialized")
                self.message_queue = []
                return False

            summary = await self.generate_report_summary()
            self.message_queue.append(summary)

            success = True
            firebase_tasks = []
            for message in self.message_queue:
                try:
                    MAX_MESSAGE_LENGTH = 4096

                    if len(message) <= MAX_MESSAGE_LENGTH:
                        result = await self.telegram_bot.send_message(chat_id=chat_id, text=message)
                        firebase_tasks.append(self._schedule_firebase(message, chat_id, result.message_id))
                    else:
                        parts = []
                        current_part = ""
                        for line in message.split('\n'):
                            if len(current_part) + len(line) + 1 <= MAX_MESSAGE_LENGTH:
                                current_part += line + '\n'
                            else:
                                if current_part:
                                    parts.append(current_part.rstrip())
                                current_part = line + '\n'
                        if current_part:
                            parts.append(current_part.rstrip())

                        first_msg_id = None
                        for i, part in enumerate(parts, 1):
                            result = await self.telegram_bot.send_message(
                                chat_id=chat_id, text=f"[{i}/{len(parts)}]\n{part}"
                            )
                            if i == 1:
                                first_msg_id = result.message_id
                            await asyncio.sleep(0.5)

                        firebase_tasks.append(self._schedule_firebase(message, chat_id, first_msg_id))

                except TelegramError as e:
                    logger.error(f"India Telegram message send failed: {e}")
                    success = False

                await asyncio.sleep(1)

            if firebase_tasks:
                await asyncio.gather(*firebase_tasks, return_exceptions=True)

            if hasattr(self, 'telegram_config') and self.telegram_config and self.telegram_config.broadcast_languages:
                self._broadcast_task = asyncio.create_task(
                    self._send_to_translation_channels(self.message_queue.copy())
                )

            self.message_queue = []
            return success

        except Exception as e:
            logger.error(f"Error sending India Telegram message: {str(e)}")
            return False

    async def _send_to_translation_channels(self, messages: List[str]):
        """Send messages to translation channels."""
        try:
            for lang in self.telegram_config.broadcast_languages:
                try:
                    channel_id = self.telegram_config.get_broadcast_channel_id(lang)
                    if not channel_id:
                        continue

                    firebase_tasks = []
                    for message in messages:
                        try:
                            translated_message = await translate_telegram_message(
                                message, model="openai/gpt-4.1-nano",
                                from_lang="en", to_lang=lang
                            )

                            MAX_MESSAGE_LENGTH = 4096
                            if len(translated_message) <= MAX_MESSAGE_LENGTH:
                                result = await self.telegram_bot.send_message(
                                    chat_id=channel_id, text=translated_message
                                )
                                firebase_tasks.append(self._schedule_firebase(translated_message, channel_id, result.message_id))
                            else:
                                parts = []
                                current_part = ""
                                for line in translated_message.split('\n'):
                                    if len(current_part) + len(line) + 1 <= MAX_MESSAGE_LENGTH:
                                        current_part += line + '\n'
                                    else:
                                        if current_part:
                                            parts.append(current_part.rstrip())
                                        current_part = line + '\n'
                                if current_part:
                                    parts.append(current_part.rstrip())

                                first_msg_id = None
                                for i, part in enumerate(parts, 1):
                                    result = await self.telegram_bot.send_message(
                                        chat_id=channel_id, text=f"[{i}/{len(parts)}]\n{part}"
                                    )
                                    if i == 1:
                                        first_msg_id = result.message_id
                                    await asyncio.sleep(0.5)

                                firebase_tasks.append(self._schedule_firebase(translated_message, channel_id, first_msg_id))

                            await asyncio.sleep(1)

                        except Exception as e:
                            logger.error(f"Error translating/sending India message to {lang}: {str(e)}")

                    if firebase_tasks:
                        await asyncio.gather(*firebase_tasks, return_exceptions=True)

                except Exception as e:
                    logger.error(f"Error processing language {lang}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in _send_to_translation_channels: {str(e)}")

    def get_compression_stats(self) -> Dict[str, Any]:
        if self.compression_manager:
            return self.compression_manager.get_compression_stats()
        return {"error": "Compression manager not initialized"}

    async def compress_old_journal_entries(self, **kwargs) -> Dict[str, Any]:
        if self.compression_manager:
            return await self.compression_manager.compress_old_journal_entries(**kwargs)
        return {"error": "Compression manager not initialized"}

    def cleanup_stale_data(self, **kwargs) -> Dict[str, Any]:
        if self.compression_manager:
            return self.compression_manager.cleanup_stale_data(**kwargs)
        return {"error": "Compression manager not initialized"}

    def get_journal_context(self, ticker: str, sector: str = None, trigger_type: str = None) -> str:
        if self.journal_manager and self.enable_journal:
            return self.journal_manager.get_context_for_ticker(ticker, sector, trigger_type=trigger_type)
        return ""

    def get_score_adjustment(self, ticker: str, sector: str = None, trigger_type: str = None) -> Tuple[int, List[str]]:
        if self.journal_manager and self.enable_journal:
            return self.journal_manager.get_score_adjustment(ticker, sector, trigger_type=trigger_type)
        return 0, []

    def _get_trigger_win_rate(self, trigger_type: str) -> str:
        """Get trigger win rate string from in_analysis_performance_tracker."""
        if not trigger_type or not self.conn:
            return ""
        try:
            cursor = self.conn.cursor()
            table_check = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='in_analysis_performance_tracker'"
            ).fetchone()
            if not table_check:
                return ""
            row = cursor.execute("""
                SELECT COUNT(*) as completed,
                       SUM(CASE WHEN return_30d > 0 THEN 1 ELSE 0 END) as wins
                FROM in_analysis_performance_tracker
                WHERE trigger_type = ? AND return_30d IS NOT NULL
            """, (trigger_type,)).fetchone()
            if row and row[0] >= 3:
                win_rate = int(row[1] / row[0] * 100)
                return f"📡 Trigger Win Rate: {win_rate}% ({row[0]} trades)"
            return ""
        except Exception:
            return ""

    async def run(self, pdf_report_paths: List[str], chat_id: str = None,
                  language: str = "en", telegram_config=None, trigger_results_file: str = None) -> bool:
        """Main execution function for India stock tracking system."""
        try:
            logger.info("Starting India tracking system batch execution")

            self.telegram_config = telegram_config

            self.trigger_info_map = {}
            if trigger_results_file:
                try:
                    if os.path.exists(trigger_results_file):
                        with open(trigger_results_file, 'r', encoding='utf-8') as f:
                            trigger_data = json.load(f)
                        for trigger_type, stocks in trigger_data.items():
                            if trigger_type == 'metadata':
                                self.trigger_mode = trigger_data.get('metadata', {}).get('trigger_mode', '')
                                continue
                            if isinstance(stocks, list):
                                for stock in stocks:
                                    ticker = stock.get('ticker', stock.get('code', ''))
                                    if ticker:
                                        self.trigger_info_map[ticker] = {
                                            'trigger_type': trigger_type,
                                            'trigger_mode': trigger_data.get('metadata', {}).get('trigger_mode', ''),
                                            'risk_reward_ratio': stock.get('risk_reward_ratio', 0)
                                        }
                        logger.info(f"Loaded trigger info for {len(self.trigger_info_map)} stocks")
                except Exception as e:
                    logger.warning(f"Failed to load trigger results file: {e}")

            await self.initialize(language)

            try:
                buy_count, sell_count = await self.process_reports(pdf_report_paths)

                if chat_id:
                    message_sent = await self.send_telegram_message(chat_id, language)
                    if message_sent:
                        logger.info("India Telegram message sent successfully")
                else:
                    await self.send_telegram_message(None, language)

                logger.info("India tracking system batch execution complete")
                return True
            finally:
                if self._broadcast_task:
                    try:
                        logger.info("Waiting for India tracking broadcast translation...")
                        await self._broadcast_task
                    except Exception as e:
                        logger.error(f"India tracking broadcast translation failed: {e}")
                    self._broadcast_task = None

                if self.conn:
                    self.conn.close()
                    logger.info("Database connection closed")

        except Exception as e:
            logger.error(f"Error during India tracking system execution: {str(e)}")
            logger.error(traceback.format_exc())

            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                except:
                    pass

            return False


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="India (NSE) Stock tracking and trading agent")
    parser.add_argument("--reports", nargs="+", help="List of analysis report file paths")
    parser.add_argument("--chat-id", help="Telegram channel ID")
    parser.add_argument("--telegram-token", help="Telegram bot token")
    parser.add_argument("--language", default="en", help="Language (default: en)")
    parser.add_argument("--enable-journal", action="store_true", help="Enable trading journal")

    args = parser.parse_args()

    if not args.reports:
        logger.error("Report path not specified")
        return False

    async with app.run():
        agent = INStockTrackingAgent(
            telegram_token=args.telegram_token,
            enable_journal=args.enable_journal
        )
        success = await agent.run(args.reports, args.chat_id, args.language)
        return success


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error during program execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
