from mcp_agent.agents.agent import Agent


def create_telegram_translator_agent(from_lang: str = "en", to_lang: str = "ja"):
    """
    Create telegram message translation agent

    Translates telegram messages from source language to target language while preserving formatting,
    emojis, numbers, and technical terms.

    Args:
        from_lang: Source language code (default: "en" for English)
        to_lang: Target language code (default: "ja" for Japanese)

    Returns:
        Agent: Telegram message translation agent
    """

    # Language name mapping
    lang_names = {
        "ko": "Korean",
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German"
    }

    from_lang_name = lang_names.get(from_lang, from_lang.upper())
    to_lang_name = lang_names.get(to_lang, to_lang.upper())

    instruction = f"""You are a professional translator specializing in stock market and trading communications.

Your task is to translate {from_lang_name} telegram messages to {to_lang_name}.

## Translation Guidelines

### 1. Preserve Formatting
- Keep all line breaks and spacing
- Maintain bullet points and numbered lists
- Preserve all emojis exactly as they appear
- Keep markdown formatting (*, -, etc.)

### 2. Number and Currency Formatting
- Convert Indian rupee amounts appropriately: "₹1,000" → equivalent in target language
- Preserve all numeric values and percentages
- Keep date formats: "2025.01.10" → "2025.01.10"

### 3. Technical Terms
- Translate stock market terminology accurately:
  - "Buy" → {{translation in {to_lang_name}}}
  - "Sell" → {{translation in {to_lang_name}}}
  - "Return" or "Profit Rate" → {{translation in {to_lang_name}}}
  - "Holding Period" → {{translation in {to_lang_name}}}
  - "Stop Loss" → {{translation in {to_lang_name}}}
  - "Target Price" → {{translation in {to_lang_name}}}
  - "Market Cap" → {{translation in {to_lang_name}}}
  - "Volume" → {{translation in {to_lang_name}}}
  - "Trading Value" → {{translation in {to_lang_name}}}

### 4. Stock Names - CRITICAL
- **ALWAYS translate company names to {to_lang_name}**
- **DO NOT keep the original language company names**
- Always include ticker symbols if present
- Example (English to {to_lang_name}): "Reliance Industries (RELIANCE)" → translate company name, keep ticker
- Example (English to {to_lang_name}): "Tata Consultancy Services" → translate to official {to_lang_name} name
- Example (English to {to_lang_name}): "Infosys (INFY)" → translate company name, keep ticker
- For well-known companies, use their official {to_lang_name} names
- For lesser-known companies, provide a descriptive translation

### 5. Tone and Style
- Maintain professional but accessible tone
- Keep urgency and emphasis from original message
- Preserve any disclaimers or warnings

### 6. Emojis and Symbols
- Keep all emojis: 📈, 📊, 🔔, ✅, ⚠️, etc.
- Preserve arrows: ⬆️, ⬇️, ➖, ↔️
- Maintain visual hierarchy with emojis

## Instructions
Translate the following {from_lang_name} telegram message to {to_lang_name} following all guidelines above.
**CRITICAL**: Make sure to translate ALL company names to {to_lang_name}. Do not leave them in {from_lang_name}.
Only return the translated text without any explanations or metadata.
"""

    agent = Agent(
        name="telegram_translator",
        instruction=instruction,
        server_names=[]
    )

    return agent


async def translate_telegram_message(
    message: str,
    model: str = "openai/gpt-4.1-nano",
    from_lang: str = "en",
    to_lang: str = "ja"
) -> str:
    """
    Translate a telegram message from source language to target language

    Args:
        message: Telegram message to translate
        model: Model to use (default: openai/gpt-4.1-nano for cost efficiency)
        from_lang: Source language code (default: "en" for English)
        to_lang: Target language code (default: "ja" for Japanese)

    Returns:
        str: Translated message
    """
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
    from mcp_agent.workflows.llm.augmented_llm import RequestParams

    try:
        # Create translator agent
        translator = create_telegram_translator_agent(from_lang=from_lang, to_lang=to_lang)

        # Attach LLM to the agent
        llm = await translator.attach_llm(OpenAIAugmentedLLM)

        # Generate translation
        translated = await llm.generate_str(
            message=message,
            request_params=RequestParams(
                model=model,
                maxTokens=100000,
                temperature=0.3,  # Lower temperature for more consistent translations
                max_iterations=1  # Single pass translation, no complex reasoning needed
            )
        )

        return translated.strip()

    except Exception as e:
        # If translation fails, return original message with error note
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Translation failed: {str(e)}")
        return message  # Fallback to original message
