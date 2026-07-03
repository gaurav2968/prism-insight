# Troubleshooting - PRISM-INSIGHT

> **Note**: This is a detailed troubleshooting reference. For quick overview, see main [CLAUDE.md](../CLAUDE.md).

---

## Common Issues

### Issue 1: Playwright PDF Generation Fails

**Symptoms**:
```
Error: Browser executable not found
```

**Solution**:
```bash
# Install Chromium browser
python3 -m playwright install chromium

# Ubuntu: Install dependencies
python3 -m playwright install --with-deps chromium

# Or use setup script
cd utils && chmod +x setup_playwright.sh && ./setup_playwright.sh
```

---

### Issue 2: Korean Fonts Not Displaying in Charts

**Symptoms**: Korean text shows as squares in generated charts

**Solution**:
```bash
# Rocky Linux/CentOS
sudo dnf install google-nanum-fonts

# Ubuntu/Debian
python3 cores/ubuntu_font_installer.py

# Rebuild font cache
sudo fc-cache -fv
python3 -c "import matplotlib.font_manager as fm; fm.fontManager.rebuild()"
```

---

### Issue 3: Telegram Bot Not Responding

**Symptoms**: Bot doesn't reply to messages

**Checklist**:
1. Verify `.env` configuration:
   ```bash
   cat .env | grep TELEGRAM
   ```
2. Check bot token validity
3. Verify bot has access to channel
4. Check logs:
   ```bash
   tail -f log_*.log
   ```
5. Test configuration:
   ```python
   from telegram_config import TelegramConfig
   config = TelegramConfig()
   config.validate_or_raise()
   config.log_status()
   ```

---

### Issue 4: MCP Server Connection Failed

**Symptoms**:
```
Error: MCP server 'yahoo_finance' not responding
```

**Solution**:
```bash
# 1. Check Yahoo Finance MCP server
uvx --from yahoo-finance-mcp yahoo-finance-mcp

# 2. Verify mcp_agent.config.yaml
cat mcp_agent.config.yaml | grep yahoo_finance

# 3. Check API keys in mcp_agent.secrets.yaml
cat mcp_agent.secrets.yaml

# 4. Test Perplexity server
npx -y @perplexity-ai/mcp-server
```

---

### Issue 5: JSON Parsing Error in Trading Scenarios

**Symptoms**:
```
Error: Invalid JSON in trading scenario
```

**Solution**:
```python
# 1. Use json-repair for automatic fixing
from json_repair import repair_json
import ujson

try:
    data = ujson.loads(json_str)
except Exception:
    # Attempt repair
    repaired = repair_json(json_str)
    data = ujson.loads(repaired)

# 2. Test JSON validation
python tests/quick_json_test.py

# 3. Check agent output format
# Ensure trading agents return valid JSON structure
```

---

### Issue 6: GPT-5 Output Formatting Issues

**Symptoms**:
- Unexpected `##` headers appearing in output
- Tool call artifacts in generated text
- Markdown formatting inconsistencies

**Solution**:
```python
# cores/utils.py provides automatic cleanup
from cores.utils import clean_markdown

# Automatic fixes applied:
# - Remove GPT-5 tool call artifacts
# - Convert ## headers to bold text in body
# - Add missing newlines after headers
# - Clean up inconsistent markdown

cleaned_text = clean_markdown(raw_output)
```

**Note**: GPT-5 model output requires additional processing compared to GPT-4.1. The `cores/utils.py` file contains several fixes for GPT-5-specific formatting quirks.

---

### Issue 7: Out of Memory During Analysis

**Symptoms**: Process killed during large batch analysis

**Solution**:
```bash
# 1. Reduce batch size
# Modify prism-in/in_stock_analysis_orchestrator.py
MAX_CONCURRENT_ANALYSES = 3  # Reduce from 5

# 2. Use --no-telegram to skip summary generation
python prism-in/in_stock_analysis_orchestrator.py --mode morning --no-telegram

# 3. Increase system memory or swap

# 4. Process stocks individually
python prism-in/in_stock_analysis_orchestrator.py --mode morning --stock RELIANCE
```

---

## Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

# Set to DEBUG level
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
```

---

## Getting Help

1. **Check logs**: `tail -f log_*.log`
2. **GitHub Issues**: [Report issues](https://github.com/dragon1086/prism-insight/issues)
3. **Telegram**: @stock_ai_ko
4. **Documentation**:
   - [README.md](../README.md)
   - [CONTRIBUTING.md](../CONTRIBUTING.md)
   - [utils/CRONTAB_SETUP.md](../utils/CRONTAB_SETUP.md)
   - [utils/PLAYWRIGHT_SETUP.md](../utils/PLAYWRIGHT_SETUP.md)

---

*See also: [CLAUDE.md](../CLAUDE.md) | [CLAUDE_AGENTS.md](CLAUDE_AGENTS.md) | [CLAUDE_TASKS.md](CLAUDE_TASKS.md)*
