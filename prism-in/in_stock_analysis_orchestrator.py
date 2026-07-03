#!/usr/bin/env python3
"""
India Stock Analysis and Telegram Transmission Orchestrator

Overall Process:
1. Execute time-based (morning/afternoon) trigger batch jobs for NSE
2. Generate detailed analysis reports for selected stocks
3. Convert reports to PDF
4. Generate and send telegram channel summary messages
5. Send generated PDF attachments
6. Execute trading simulation / tracking

Key Differences from KR/US:
- Uses NSE tickers (RELIANCE, TCS) with .NS suffix for yfinance
- Uses yfinance + nsetools for market data
- NSE market hours (09:15-15:30 IST)
- English language default (en)
- INR (₹) currency
- NIFTY 50/SENSEX indices
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
PRISM_IN_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PRISM_IN_DIR))

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"in_orchestrator_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def _import_from_main_cores(module_name: str, relative_path: str):
    """Import module from main project cores/ directory (avoids namespace collision)."""
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

# Directory configuration
IN_REPORTS_DIR = PRISM_IN_DIR / "reports"
IN_TELEGRAM_MSGS_DIR = PRISM_IN_DIR / "telegram_messages"
IN_PDF_REPORTS_DIR = PRISM_IN_DIR / "pdf_reports"

# Create directories
IN_REPORTS_DIR.mkdir(exist_ok=True)
IN_TELEGRAM_MSGS_DIR.mkdir(exist_ok=True)
IN_PDF_REPORTS_DIR.mkdir(exist_ok=True)
(IN_TELEGRAM_MSGS_DIR / "sent").mkdir(exist_ok=True)


# Trigger type display (English)
TRIGGER_TYPE_DISPLAY = {
    "Volume Surge Top": "Volume Surge Top",
    "Gap Up Momentum Top": "Gap Up Momentum Top",
    "Value-to-Cap Ratio Top": "Value-to-Cap Ratio Top",
    "Intraday Rise Top": "Intraday Rise Top",
    "Closing Strength Top": "Closing Strength Top",
    "Volume Surge Sideways": "Volume Surge Sideways",
}


class INStockAnalysisOrchestrator:
    """India (NSE) Stock Analysis and Telegram Transmission Orchestrator"""

    def __init__(self, telegram_config=None):
        from telegram_config import TelegramConfig
        self.selected_tickers = {}
        self.telegram_config = telegram_config or TelegramConfig(use_telegram=True)
        self._broadcast_tasks = []

    @staticmethod
    def _extract_base64_images(markdown_text: str) -> tuple:
        """Extract base64 images and replace with placeholders."""
        images = {}
        counter = 0

        def replace_image(match):
            nonlocal counter
            placeholder = f"<<<__BASE64_IMAGE_{counter}__>>>"
            images[placeholder] = match.group(0)
            counter += 1
            return placeholder

        patterns = [
            r'<img\s+src="data:image/[^;]+;base64,[A-Za-z0-9+/=]+"\s+[^>]*>',
            r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
        ]

        text_without_images = markdown_text
        for pattern in patterns:
            text_without_images = re.sub(pattern, replace_image, text_without_images)

        return text_without_images, images

    @staticmethod
    def _restore_base64_images(translated_text: str, images: dict) -> str:
        """Restore base64 images to translated text."""
        restored_text = translated_text
        for placeholder, original_image in images.items():
            if placeholder in restored_text:
                restored_text = restored_text.replace(placeholder, original_image)
            else:
                simple_key = placeholder.replace("<<<", "").replace(">>>", "").replace("__", "_")
                if simple_key in restored_text:
                    restored_text = restored_text.replace(simple_key, original_image)
        return restored_text

    async def run_trigger_batch(self, mode: str, reference_date: str = None):
        """Execute India trigger batch and save results."""
        logger.info(f"Starting India trigger batch execution: {mode}")
        try:
            from in_trigger_batch import run_batch

            date_str = reference_date or datetime.now().strftime('%Y%m%d')
            results_file = f"trigger_results_in_{mode}_{date_str}.json"

            loop = asyncio.get_event_loop()
            # Timeout: 15 minutes for trigger batch (500 tickers takes time)
            try:
                results = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: run_batch(mode, "INFO", results_file, reference_date=reference_date)
                    ),
                    timeout=900  # 15 minutes
                )
            except asyncio.TimeoutError:
                logger.error("Trigger batch timed out after 15 minutes")
                return []

            if not results:
                logger.warning("India batch returned empty results")
                return []

            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    full_results = json.load(f)
                self.selected_tickers[mode] = full_results

            tickers = []
            ticker_set = set()

            for trigger_type, stocks_df in results.items():
                if hasattr(stocks_df, 'index'):
                    for ticker in stocks_df.index:
                        if ticker not in ticker_set:
                            ticker_set.add(ticker)
                            name = ""
                            if "CompanyName" in stocks_df.columns:
                                name = stocks_df.loc[ticker, "CompanyName"]
                            rr_ratio = 0
                            if "risk_reward_ratio" in stocks_df.columns:
                                rr_ratio = float(stocks_df.loc[ticker, "risk_reward_ratio"])
                            tickers.append({
                                'ticker': ticker,
                                'name': name or ticker,
                                'trigger_type': trigger_type,
                                'trigger_mode': mode,
                                'risk_reward_ratio': rr_ratio
                            })

            logger.info(f"Number of selected India stocks: {len(tickers)}")
            return tickers

        except Exception as e:
            logger.error(f"Error during India trigger batch: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def generate_reports(self, tickers: list, mode: str, timeout: int = None, language: str = "en", reference_date: str = None) -> list:
        """Generate reports serially for all India stocks."""
        logger.info(f"Starting India report generation for {len(tickers)} stocks")

        successful_reports = []

        for idx, ticker_info in enumerate(tickers, 1):
            if isinstance(ticker_info, dict):
                ticker = ticker_info.get('ticker')
                company_name = ticker_info.get('name', ticker)
            else:
                ticker = ticker_info
                company_name = ticker

            logger.info(f"[{idx}/{len(tickers)}] Starting India stock analysis: {company_name}({ticker})")

            reference_date = reference_date or datetime.now().strftime("%Y%m%d")
            output_file = str(IN_REPORTS_DIR / f"{ticker}_{company_name}_{reference_date}_{mode}_gpt4o.md")

            try:
                from cores.in_analysis import analyze_in_stock

                report = await analyze_in_stock(
                    ticker=ticker,
                    company_name=company_name,
                    reference_date=reference_date,
                    language=language
                )

                if report and len(report.strip()) > 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(report)
                    logger.info(f"[{idx}/{len(tickers)}] Report complete: {company_name}({ticker}) - {len(report)} chars")
                    successful_reports.append(output_file)
                else:
                    logger.error(f"[{idx}/{len(tickers)}] Report failed: {company_name}({ticker}) - empty")

            except Exception as e:
                logger.error(f"[{idx}/{len(tickers)}] Error: {company_name}({ticker}) - {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info(f"India report generation: {len(successful_reports)}/{len(tickers)} successful")
        return successful_reports

    async def convert_to_pdf(self, report_paths: list) -> list:
        """Convert markdown reports to PDF."""
        logger.info(f"Starting PDF conversion for {len(report_paths)} India reports")
        pdf_paths = []
        from pdf_converter import markdown_to_pdf

        for report_path in report_paths:
            try:
                report_file = Path(report_path)
                pdf_file = IN_PDF_REPORTS_DIR / f"{report_file.stem}.pdf"
                markdown_to_pdf(report_path, pdf_file, 'playwright', add_theme=True, enable_watermark=False)
                logger.info(f"PDF conversion complete: {pdf_file}")
                pdf_paths.append(pdf_file)
            except Exception as e:
                logger.error(f"Error during PDF conversion of {report_path}: {str(e)}")

        return pdf_paths

    async def generate_telegram_messages(self, report_pdf_paths: list, language: str = "en") -> list:
        """Generate telegram messages for India stocks."""
        logger.info(f"Starting India telegram message generation for {len(report_pdf_paths)} reports")

        # Use main project's telegram summary agent (adapted for India)
        from telegram_summary_agent import TelegramSummaryGenerator, app as summary_app

        generator = TelegramSummaryGenerator()
        message_paths = []

        async with summary_app.run():
            for report_pdf_path in report_pdf_paths:
                try:
                    await generator.process_report(str(report_pdf_path), str(IN_TELEGRAM_MSGS_DIR), from_lang=language, to_lang=language)
                    report_file = Path(report_pdf_path)
                    ticker = report_file.stem.split('_')[0]
                    company_name = report_file.stem.split('_')[1]
                    message_path = IN_TELEGRAM_MSGS_DIR / f"{ticker}_{company_name}_telegram.txt"
                    if message_path.exists():
                        message_paths.append(message_path)
                except Exception as e:
                    logger.error(f"Error generating telegram message for {report_pdf_path}: {str(e)}")

        return message_paths

    async def send_telegram_messages(self, message_paths: list, pdf_paths: list, report_paths: list = None):
        """Send telegram messages and PDF files."""
        if not self.telegram_config.use_telegram:
            logger.info("Telegram disabled - skipping India message transmission")
            return

        chat_id = self.telegram_config.channel_id
        if not chat_id:
            logger.error("Telegram channel ID not configured for India stocks.")
            return

        from telegram_bot_agent import TelegramBotAgent

        try:
            bot_agent = TelegramBotAgent()

            # Broadcast translations (non-blocking)
            if self.telegram_config.broadcast_languages:
                message_contents = []
                for mp in message_paths:
                    try:
                        with open(mp, 'r', encoding='utf-8') as f:
                            message_contents.append(f.read())
                    except Exception as e:
                        logger.error(f"Error reading message file {mp}: {str(e)}")
                if message_contents:
                    self._broadcast_tasks.append(
                        asyncio.create_task(self._send_translated_messages(bot_agent, message_contents))
                    )

            # Send to main channel
            await bot_agent.process_messages_directory(
                str(IN_TELEGRAM_MSGS_DIR), chat_id, str(IN_TELEGRAM_MSGS_DIR / "sent")
            )

            for pdf_path in pdf_paths:
                success = await bot_agent.send_document(chat_id, str(pdf_path))
                if success:
                    logger.info(f"PDF sent: {pdf_path}")
                await asyncio.sleep(1)

            if self.telegram_config.broadcast_languages and report_paths:
                self._broadcast_tasks.append(
                    asyncio.create_task(self._send_translated_pdfs(bot_agent, report_paths))
                )

        except Exception as e:
            logger.error(f"Error during telegram transmission: {str(e)}")

    async def _send_translated_messages(self, bot_agent, message_contents: list):
        """Send translated telegram messages to broadcast channels."""
        try:
            async def _translate_lang(lang, channel_id):
                for original in message_contents:
                    try:
                        translated = await translate_telegram_message(
                            original, model="openai/gpt-4.1-nano",
                            from_lang="en", to_lang=lang
                        )
                        await bot_agent.send_message(channel_id, translated)
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error translating India message to {lang}: {str(e)}")

            tasks = []
            for lang in self.telegram_config.broadcast_languages:
                channel_id = self.telegram_config.get_broadcast_channel_id(lang)
                if channel_id:
                    tasks.append(_translate_lang(lang, channel_id))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in _send_translated_messages: {str(e)}")

    async def _send_translated_pdfs(self, bot_agent, report_paths: list):
        """Send translated PDF reports to broadcast channels (sequential for memory)."""
        try:
            for lang in self.telegram_config.broadcast_languages:
                channel_id = self.telegram_config.get_broadcast_channel_id(lang)
                if not channel_id:
                    continue
                for report_path in report_paths:
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            original = f.read()
                        text, images = self._extract_base64_images(original)
                        translated = await translate_telegram_message(
                            text, model="openai/gpt-4.1-nano",
                            from_lang="en", to_lang=lang
                        )
                        translated = self._restore_base64_images(translated, images)
                        report_file = Path(report_path)
                        translated_path = report_file.parent / f"{report_file.stem}_{lang}.md"
                        with open(translated_path, 'w', encoding='utf-8') as f:
                            f.write(translated)
                        pdf_paths = await self.convert_to_pdf([str(translated_path)])
                        if pdf_paths:
                            await bot_agent.send_document(channel_id, str(pdf_paths[0]))
                            await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error processing {report_path} for {lang}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in _send_translated_pdfs: {str(e)}")

    async def send_trigger_alert(self, mode: str, trigger_results_file: str, language: str = "en"):
        """Send trigger results to telegram channel."""
        if not self.telegram_config.use_telegram:
            return False

        try:
            with open(trigger_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            metadata = results.get("metadata", {})
            trade_date = metadata.get("trade_date", datetime.now().strftime("%Y%m%d"))

            all_results = {k: v for k, v in results.items() if k != "metadata" and isinstance(v, list)}
            if not all_results:
                return False

            message = self._create_trigger_alert_message(mode, all_results, trade_date, language)

            chat_id = self.telegram_config.channel_id
            if not chat_id:
                return False

            from telegram_bot_agent import TelegramBotAgent
            bot_agent = TelegramBotAgent()
            success = await bot_agent.send_message(chat_id, message)

            if self.telegram_config.broadcast_languages:
                self._broadcast_tasks.append(
                    asyncio.create_task(self._send_translated_trigger_alert(bot_agent, message, mode))
                )

            return success

        except Exception as e:
            logger.error(f"Error during India trigger alert: {str(e)}")
            return False

    async def _send_translated_trigger_alert(self, bot_agent, original_message: str, mode: str):
        """Send translated trigger alerts to broadcast channels."""
        try:
            tasks = []
            for lang in self.telegram_config.broadcast_languages:
                channel_id = self.telegram_config.get_broadcast_channel_id(lang)
                if channel_id:
                    async def _send(l, cid):
                        translated = await translate_telegram_message(
                            original_message, model="openai/gpt-4.1-nano",
                            from_lang="en", to_lang=l
                        )
                        await bot_agent.send_message(cid, translated)
                    tasks.append(_send(lang, channel_id))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in _send_translated_trigger_alert: {str(e)}")

    def _create_trigger_alert_message(self, mode: str, results: dict, trade_date: str, language: str = "en") -> str:
        """Generate telegram alert message for India trigger results."""
        formatted_date = f"{trade_date[:4]}.{trade_date[4:6]}.{trade_date[6:8]}"

        if mode == "morning":
            title = "🔔 India Stock Morning Prism Signal Alert"
            time_desc = "10 minutes after NSE market open"
        elif mode == "midday":
            title = "🔔 India Stock Midday Prism Signal Alert"
            time_desc = "at 12:30 PM IST"
        else:
            title = "🔔 India Stock Afternoon Prism Signal Alert"
            time_desc = "after NSE market close"

        header = f"{title}\n📅 {formatted_date} Stocks detected {time_desc}\n\n"
        footer = "📋 Detailed analysis report will be available in 10-30 minutes\n※ This is for investment reference only. Investment decisions are your responsibility."

        message = header

        for trigger_type, stocks in results.items():
            emoji = self._get_trigger_emoji(trigger_type)
            message += f"{emoji} {trigger_type}\n"

            for stock in stocks:
                ticker = stock.get("ticker", "")
                name = stock.get("name", ticker)
                current_price = stock.get("current_price", 0)
                change_rate = stock.get("change_rate", 0)
                arrow = "⬆️" if change_rate > 0 else "⬇️" if change_rate < 0 else "➖"

                message += f"· {name} ({ticker}.NS)\n"
                message += f"  ₹{current_price:,.2f} {arrow} {abs(change_rate):.2f}%\n"

                if "volume_increase" in stock and "Volume" in trigger_type:
                    message += f"  Volume Increase: {stock['volume_increase']:.2f}%\n"
                elif "gap_rate" in stock and "Gap" in trigger_type:
                    message += f"  Gap Up: {stock['gap_rate']:.2f}%\n"

                message += "\n"

        message += footer
        return message

    async def _send_progress_message(self, message: str):
        """Send an intermediate progress message to Telegram."""
        if not self.telegram_config.use_telegram:
            logger.info(f"[Progress] {message[:100]}...")
            return
        chat_id = self.telegram_config.channel_id
        if not chat_id:
            return
        try:
            from telegram_bot_agent import TelegramBotAgent
            bot_agent = TelegramBotAgent()
            await bot_agent.send_message(chat_id, message, parse_mode=None)
        except Exception as e:
            logger.warning(f"Failed to send progress message: {e}")

    def _build_screening_results_message(self, mode: str, results: dict, trade_date: str) -> str:
        """Build a detailed screening results message from trigger results JSON with data-backed metrics."""
        formatted_date = f"{trade_date[:4]}.{trade_date[4:6]}.{trade_date[6:8]}"
        session = "Morning" if mode == "morning" else "Midday" if mode == "midday" else "Afternoon"

        screening = results.get("screening_summary", {})
        total_scanned = screening.get("total_tickers_scanned", 0)
        snapshot_count = screening.get("snapshot_count", 0)
        baseline_count = screening.get("baseline_count", 0)
        trigger_data = screening.get("triggers", {})

        lines = [
            f"📊 PRISM Screening Results",
            f"📅 {formatted_date} | {session} Session",
            f"",
            f"🏦 Universe: NIFTY 500 ({total_scanned} tickers)",
            f"📈 Valid data: {snapshot_count} | Baseline: {baseline_count}",
            f"",
            f"━━━ Trigger Screening ━━━",
        ]

        for trigger_name, data in trigger_data.items():
            emoji = self._get_trigger_emoji(trigger_name)
            count = data.get("candidate_count", 0)
            lines.append(f"")
            lines.append(f"{emoji} {trigger_name}: {count} candidates")
            candidates = data.get("top_candidates", [])
            for c in candidates[:5]:
                ticker = c.get("ticker", "")
                price = c.get("price", 0)
                change = c.get("change_pct", 0)
                score = c.get("composite_score", 0)
                arrow = "+" if change > 0 else ""
                lines.append(f"  · {ticker} ₹{price:,.1f} ({arrow}{change:.1f}%) S:{score:.2f}")

        # Final selections with detailed metrics
        all_results = {k: v for k, v in results.items()
                       if k not in ("metadata", "screening_summary") and isinstance(v, list)}
        if all_results:
            lines.append(f"")
            lines.append(f"━━━ TOP SELECTIONS ━━━")
            idx = 1
            for trigger_type, stocks in all_results.items():
                for stock in stocks:
                    ticker = stock.get("ticker", "")
                    name = stock.get("name", ticker)
                    price = stock.get("current_price", 0)
                    change = stock.get("change_rate", 0)
                    final_score = stock.get("final_score", 0)
                    rr = stock.get("risk_reward_ratio", 0)
                    sl_pct = stock.get("stop_loss_pct", 0)
                    tp = stock.get("target_price", 0)
                    sl_price = stock.get("stop_loss_price", 0)
                    qs = stock.get("quality_score", 0)
                    q_sig = stock.get("quality_signal", "")
                    q_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(q_sig, "⚪")
                    metrics = stock.get("metrics", {})

                    display = f"{name} ({ticker}.NS)" if name and name != ticker else f"{ticker}.NS"
                    arrow = "▲" if change > 0 else "▼" if change < 0 else "─"

                    lines.append(f"")
                    lines.append(f"{'─' * 30}")
                    lines.append(f"{idx}. {display}")
                    lines.append(f"   ₹{price:,.2f} {arrow} {abs(change):.1f}%")

                    # Sector & Market Cap
                    sector = metrics.get("sector", "")
                    mcap = metrics.get("market_cap_cr")
                    if sector or mcap:
                        mcap_str = f"₹{mcap:,.0f} Cr" if mcap else ""
                        sector_str = sector if sector else ""
                        sep = " | " if sector_str and mcap_str else ""
                        lines.append(f"   {sector_str}{sep}{mcap_str}")

                    # Quality + Signal
                    lines.append(f"   {q_emoji} Quality: {qs:.0f}/100 ({q_sig}) | Score: {final_score:.3f}")

                    # Trade Setup
                    lines.append(f"   📐 R/R: {rr:.1f}x | SL: ₹{sl_price:,.1f} ({sl_pct:.1f}%) | Target: ₹{tp:,.1f}")

                    # Key Fundamental Data
                    pe = metrics.get("trailing_pe")
                    fpe = metrics.get("forward_pe")
                    pb = metrics.get("price_to_book")
                    pe_parts = []
                    if pe: pe_parts.append(f"PE:{pe}")
                    if fpe: pe_parts.append(f"Fwd:{fpe}")
                    if pb: pe_parts.append(f"P/B:{pb}")
                    if pe_parts:
                        lines.append(f"   📊 Valuation: {' | '.join(pe_parts)}")

                    # Growth & Profitability
                    rev_g = metrics.get("revenue_growth_pct")
                    earn_g = metrics.get("earnings_growth_pct")
                    opm = metrics.get("operating_margin_pct")
                    roe = metrics.get("roe_pct")
                    gp_parts = []
                    if rev_g is not None: gp_parts.append(f"Rev:{'+' if rev_g > 0 else ''}{rev_g}%")
                    if earn_g is not None: gp_parts.append(f"EPS:{'+' if earn_g > 0 else ''}{earn_g}%")
                    if opm is not None: gp_parts.append(f"OPM:{opm}%")
                    if roe is not None: gp_parts.append(f"ROE:{roe}%")
                    if gp_parts:
                        lines.append(f"   📈 Growth: {' | '.join(gp_parts)}")

                    # Technical Indicators
                    rsi = metrics.get("rsi_14")
                    pct_52h = metrics.get("pct_from_52w_high")
                    beta = metrics.get("beta")
                    tech_parts = []
                    if rsi:
                        rsi_tag = "OB" if rsi > 70 else "OS" if rsi < 30 else ""
                        rsi_str = f"RSI:{rsi}" + (f"({rsi_tag})" if rsi_tag else "")
                        tech_parts.append(rsi_str)
                    if pct_52h is not None: tech_parts.append(f"52W:{pct_52h:+.1f}%")
                    if beta: tech_parts.append(f"β:{beta}")
                    if tech_parts:
                        lines.append(f"   📉 Technical: {' | '.join(tech_parts)}")

                    # Analyst Consensus
                    rec = metrics.get("analyst_recommendation", "")
                    upside = metrics.get("target_upside_pct")
                    a_count = metrics.get("analyst_count", 0)
                    de = metrics.get("debt_to_equity")
                    analyst_parts = []
                    if rec: analyst_parts.append(rec.upper())
                    if upside is not None: analyst_parts.append(f"Upside:{upside:+.1f}%")
                    if a_count: analyst_parts.append(f"{a_count} analysts")
                    if de is not None: analyst_parts.append(f"D/E:{de}")
                    if analyst_parts:
                        lines.append(f"   🎯 Analyst: {' | '.join(analyst_parts)}")

                    # Sub-scores radar
                    vs = metrics.get("valuation_score", 0)
                    gs = metrics.get("growth_score", 0)
                    ps = metrics.get("profitability_score", 0)
                    ts = metrics.get("technical_score", 0)
                    as_ = metrics.get("analyst_score", 0)
                    rs = metrics.get("risk_score", 0)
                    if any([vs, gs, ps, ts, as_, rs]):
                        lines.append(f"   🔬 V:{vs:.0f} G:{gs:.0f} P:{ps:.0f} T:{ts:.0f} A:{as_:.0f} R:{rs:.0f}")

                    # Trigger source
                    lines.append(f"   🏷️ Via: {trigger_type}")
                    idx += 1

        # Scoring explanation
        metadata = results.get("metadata", {})
        weights = metadata.get("scoring_weights", {})
        if weights:
            lines.append(f"")
            lines.append(f"━━━ Scoring Weights ━━━")
            lines.append(f"Momentum {int(weights.get('momentum_signal', 0.2)*100)}% + "
                        f"R/R {int(weights.get('agent_fit_rr', 0.4)*100)}% + "
                        f"Quality {int(weights.get('quality_fundamental_technical', 0.4)*100)}%")
            gate = metadata.get("quality_gate_min", 40)
            lines.append(f"Quality Gate: ≥{gate}/100 to pass")

        lines.append(f"")
        lines.append(f"📋 Generating detailed AI analysis reports...")
        return "\n".join(lines)

    def _get_trigger_emoji(self, trigger_type: str) -> str:
        """Return emoji matching trigger type."""
        if "Volume" in trigger_type and "Sideways" not in trigger_type:
            return "📊"
        elif "Gap" in trigger_type:
            return "📈"
        elif "Value" in trigger_type or "Cap" in trigger_type:
            return "💰"
        elif "Rise" in trigger_type or "Intraday" in trigger_type:
            return "🚀"
        elif "Closing" in trigger_type or "Strength" in trigger_type:
            return "🔨"
        elif "Sideways" in trigger_type:
            return "↔️"
        else:
            return "🔎"

    async def run_full_pipeline(self, mode: str, language: str = "en", reference_date: str = None):
        """
        Execute full India pipeline.

        Args:
            mode: 'morning' or 'afternoon'
            language: Analysis language (default: "en" for India)
            reference_date: Override date in YYYYMMDD format (default: today)
        """
        logger.info(f"Starting India full pipeline - mode: {mode}")

        try:
            date_str = reference_date or datetime.now().strftime("%Y%m%d")
            # 0. Send pipeline start notification
            trade_date_str = datetime.strptime(date_str, "%Y%m%d").strftime("%Y.%m.%d")
            session_label = "Morning" if mode == "morning" else "Midday" if mode == "midday" else "Afternoon"
            await self._send_progress_message(
                f"🔍 PRISM India Analysis Started\n"
                f"📅 {trade_date_str} | {session_label} Session\n"
                f"🏦 Scanning NIFTY 500 universe...\n"
                f"⏳ Running surge detection algorithms"
            )

            # 1. Execute trigger batch
            results_file = f"trigger_results_in_{mode}_{date_str}.json"
            tickers = await self.run_trigger_batch(mode, reference_date=reference_date)

            if not tickers:
                await self._send_progress_message(
                    f"⚠️ PRISM {session_label} Session\n"
                    f"No stocks met screening criteria today. Pipeline stopped."
                )
                logger.warning("No India stocks selected. Terminating.")
                return

            # 1-1. Send detailed screening results
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                screening_msg = self._build_screening_results_message(mode, full_data,
                    full_data.get("metadata", {}).get("trade_date", datetime.now().strftime("%Y%m%d")))
                await self._send_progress_message(screening_msg)

            # 1-2. Send trigger alert
            if os.path.exists(results_file):
                await self.send_trigger_alert(mode, results_file, language)

            # 2. Generate reports
            ticker_names = ", ".join(
                t.get("name", t.get("ticker", "")) or t.get("ticker", "")
                for t in tickers
            )
            await self._send_progress_message(
                f"🤖 Starting AI Analysis\n"
                f"Analyzing {len(tickers)} stocks: {ticker_names}\n"
                f"⏳ This may take 10-30 minutes..."
            )
            report_paths = await self.generate_reports(tickers, mode, timeout=600, language=language, reference_date=reference_date)
            if not report_paths:
                logger.warning("No India reports generated. Terminating.")
                return

            await self._send_progress_message(
                f"✅ Analysis Complete\n"
                f"{len(report_paths)}/{len(tickers)} reports generated.\n"
                f"📄 Converting to PDF and preparing Telegram messages..."
            )

            # 3. PDF conversion
            pdf_paths = await self.convert_to_pdf(report_paths)

            # 4-5. Generate and send telegram messages
            if self.telegram_config.use_telegram:
                message_paths = await self.generate_telegram_messages(pdf_paths, language)
                await self.send_telegram_messages(message_paths, pdf_paths, report_paths)

            # 6. Tracking system
            try:
                from in_stock_tracking_agent import INStockTrackingAgent

                tracker = INStockTrackingAgent(
                    db_path=str(PROJECT_ROOT / "stock_tracking_db.sqlite"),
                    telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
                    enable_journal=True
                )
                tracker.telegram_config = self.telegram_config
                tracker.trigger_mode = mode

                # Load trigger info from results file
                trigger_results_file = None
                trigger_files = list(PRISM_IN_DIR.glob(f"in_trigger_results_*.json"))
                if trigger_files:
                    trigger_results_file = str(sorted(trigger_files)[-1])

                tracking_chat_id = self.telegram_config.chat_id if self.telegram_config.use_telegram else None
                # Fall back to markdown report paths if PDF conversion failed
                tracking_report_paths = pdf_paths if pdf_paths else report_paths
                await tracker.run(
                    pdf_report_paths=tracking_report_paths,
                    chat_id=tracking_chat_id,
                    language=language,
                    telegram_config=self.telegram_config,
                    trigger_results_file=trigger_results_file
                )
                logger.info("India tracking system execution complete")
            except Exception as tracking_err:
                logger.warning(f"India tracking system error (non-fatal): {tracking_err}")
                import traceback as tb
                logger.debug(tb.format_exc())

            # 7. Generate dashboard JSON from live DB data
            try:
                from examples.generate_in_dashboard_json import IndiaDashboardDataGenerator
                generator = IndiaDashboardDataGenerator(
                    db_path=str(PROJECT_ROOT / "stock_tracking_db.sqlite"),
                    trading_mode="demo"
                )
                dashboard_data = generator.generate()
                generator.save(dashboard_data)
                logger.info("Dashboard JSON regenerated from live data")
            except Exception as dash_err:
                logger.warning(f"Dashboard JSON generation error (non-fatal): {dash_err}")
                import traceback as tb
                logger.debug(tb.format_exc())

            logger.info(f"India full pipeline complete - mode: {mode}")

        except Exception as e:
            logger.error(f"Error during India pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            if self._broadcast_tasks:
                logger.info(f"Waiting for {len(self._broadcast_tasks)} broadcast task(s)...")
                results = await asyncio.gather(*self._broadcast_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Broadcast task {i+1} failed: {result}")
                self._broadcast_tasks.clear()


async def main():
    """Main function - CLI interface."""
    parser = argparse.ArgumentParser(description="India stock analysis and telegram orchestrator")
    parser.add_argument("--mode", choices=["morning", "midday", "afternoon", "both"], default="both",
                        help="Execution mode")
    parser.add_argument("--language", choices=["en", "hi", "ko"], default="en",
                        help="Analysis language (en: English, hi: Hindi, ko: Korean)")
    parser.add_argument("--broadcast-languages", type=str, default="",
                        help="Additional broadcast languages (comma-separated, e.g., 'hi,ko,ja')")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Disable telegram transmission")
    parser.add_argument("--force", action="store_true",
                        help="Force execution on market holidays")
    parser.add_argument("--date", type=str, default=None,
                        help="Override analysis date in YYYYMMDD format (e.g. 20260507)")

    args = parser.parse_args()

    broadcast_languages = [lang.strip() for lang in args.broadcast_languages.split(",") if lang.strip()]

    from telegram_config import TelegramConfig
    telegram_config = TelegramConfig(use_telegram=not args.no_telegram, broadcast_languages=broadcast_languages)

    if telegram_config.use_telegram:
        try:
            telegram_config.validate_or_raise()
        except ValueError as e:
            logger.error(f"Telegram config error: {str(e)}")
            sys.exit(1)

    orchestrator = INStockAnalysisOrchestrator(telegram_config=telegram_config)

    if args.mode == "morning" or args.mode == "both":
        await orchestrator.run_full_pipeline("morning", language=args.language, reference_date=args.date)

    if args.mode == "midday":
        await orchestrator.run_full_pipeline("midday", language=args.language, reference_date=args.date)

    if args.mode == "afternoon" or args.mode == "both":
        await orchestrator.run_full_pipeline("afternoon", language=args.language, reference_date=args.date)


if __name__ == "__main__":
    force_execution = "--force" in sys.argv

    from check_market_day import is_nse_market_day

    if not force_execution and not is_nse_market_day():
        current_date = datetime.now().date()
        logger.info(f"Today ({current_date}) is an NSE market holiday. Not executing batch.")
        sys.exit(0)

    if force_execution:
        logger.warning("Force execution enabled - ignoring market holiday check")

    import threading

    def exit_after_timeout():
        import time
        import signal
        time.sleep(7200)
        logger.warning("120-minute timeout reached: forcefully terminating")
        os.kill(os.getpid(), signal.SIGTERM)

    timer_thread = threading.Thread(target=exit_after_timeout, daemon=True)
    timer_thread.start()

    asyncio.run(main())
