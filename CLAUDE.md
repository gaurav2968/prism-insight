# CLAUDE.md - AI Assistant Guide for PRISM-INSIGHT

> **Version**: 3.0.0 | **Updated**: 2026-03-01

## Quick Overview

**PRISM-INSIGHT** = AI-powered Indian (NSE) stock analysis & automated trading system

```yaml
Stack: Python 3.10+, mcp-agent, GPT-5/Claude 4.5, SQLite, Telegram, Yahoo Finance
Scale: ~30 active files, 5 AI agents, India (NSE/BSE) market support
```

## Project Structure

```
prism-insight/
├── cores/                    # Shared AI Infrastructure
│   ├── agents/              # Shared agents (translator)
│   ├── report_generation.py # Report templates (multilingual)
│   └── utils.py             # Common utilities
├── prism-in/                # India (NSE) Stock Module (ACTIVE)
│   ├── cores/               # India analysis engine
│   │   ├── agents/          # 5 specialized AI agents
│   │   ├── in_analysis.py   # Core orchestration
│   │   ├── in_data_client.py # NSE data fetcher
│   │   ├── in_stock_chart.py # Chart generation
│   │   ├── in_surge_detector.py # Surge detection
│   │   └── data_prefetch.py # Yahoo Finance prefetch
│   ├── in_stock_analysis_orchestrator.py  # Main entry point
│   ├── in_stock_tracking_agent.py         # Portfolio tracking
│   ├── in_trigger_batch.py                # Surge trigger batch
│   └── check_market_day.py               # NSE market calendar
├── firebase_bridge.py       # Firebase push notifications
├── pdf_converter.py         # HTML→PDF converter
├── telegram_config.py       # Telegram channel config
├── telegram_bot_agent.py    # Telegram bot agent
├── telegram_summary_agent.py # Summary generation
├── _archive/                # Archived KR & US modules (reference only)
│   ├── prism-kr/            # Korean market code
│   └── prism-us/            # US market code
└── sqlite/                  # Database files
```

## Key Entry Points

| Command | Purpose |
|---------|---------|
| `python prism-in/in_stock_analysis_orchestrator.py --mode morning` | India morning analysis |
| `python prism-in/in_stock_analysis_orchestrator.py --mode morning --no-telegram` | Local test (no Telegram) |
| `python prism-in/in_trigger_batch.py morning INFO` | India surge detection only |

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Telegram tokens, channel IDs, Firebase settings |
| `mcp_agent.secrets.yaml` | API keys (OpenAI, Anthropic, Firecrawl, etc.) |
| `mcp_agent.config.yaml` | MCP server configuration |

**Setup**: Copy `*.example` files and fill in credentials.

## Code Conventions

### Async Pattern (Required)
```python
# ✅ Correct
result = await agent.run(prompt)

# ❌ Wrong - blocks event loop
result = requests.get(url)  # Use aiohttp instead
```

### Sequential Agent Execution
```python
# ✅ Correct - respects rate limits
for section in sections:
    report = await generate_report(agent, section)

# ❌ Wrong - hits rate limits
reports = await asyncio.gather(*[generate_report(a, s) for s in sections])
```

### India Market Specifics
```python
# NSE ticker format: plain uppercase symbols (e.g., RELIANCE, TCS, INFY)
# Market hours: 09:15-15:30 IST
# Market cap filter: ₹20,000 Cr (configurable in in_surge_detector.py)
# Data source: Yahoo Finance (yfinance) for prices, volumes, fundamentals
```

## Database Tables (India)

| Table | Purpose |
|-------|---------|
| `in_stock_holdings` | Current portfolio |
| `in_trading_history` | Trade records |
| `in_watchlist_history` | Analyzed but not entered |
| `in_analysis_performance_tracker` | 7/14/30-day tracking |

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Playwright PDF fails | `python3 -m playwright install chromium` |
| NSE data unavailable | Check yfinance connectivity, NSE may block IPs |
| Telegram not sending | Verify `.env` has correct `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` |
| Firebase push fails | Check Firebase credentials in `.env` |

## i18n Strategy

- **Code comments/logs**: English
- **Telegram messages**: English (default for India channel)
- **Broadcast channels**: Translation agent converts to target language (ja/zh/es/ko)

```bash
# Default channel
python prism-in/in_stock_analysis_orchestrator.py --mode morning

# Broadcast to all language channels
python prism-in/in_stock_analysis_orchestrator.py --mode morning --broadcast-languages ja,zh,es,ko
```

## Branch & Commit Convention

### Branch Rule
- **Code changes** (`.py`, `.ts`, `.tsx`, `.js`, `.jsx`): Work on feature branches, create PR
- **Docs only** (`.md`): Direct commit to main allowed
- Branch naming: `feat/`, `fix/`, `refactor/`, `test/` + description (e.g., `feat/in-surge-detector`)

### Commit Message
```
feat: New feature
fix: Bug fix
docs: Documentation
refactor: Code refactoring
test: Tests
```

## Detailed Documentation

For comprehensive guides, see:
- `docs/CLAUDE_AGENTS.md` - AI agent system documentation
- `docs/CLAUDE_TASKS.md` - Common development tasks
- `docs/CLAUDE_TROUBLESHOOTING.md` - Full troubleshooting guide
- `_archive/prism-kr/` - Korean market code (archived reference)
- `_archive/prism-us/` - US market code (archived reference)

---

## Version History

| Ver | Date | Changes |
|-----|------|---------|
| 3.0.0 | 2026-03-01 | **India-focused restructure** - Archived all KR & US market code to `_archive/`, repo now India (NSE) only. Shared infrastructure (cores/, telegram, firebase_bridge, pdf_converter) retained and updated for India market patterns. prism-in module is the active entry point. |

For full KR/US history, see git log or `_archive/` directories.
