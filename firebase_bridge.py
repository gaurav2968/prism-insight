"""
Firebase Bridge for PRISM-Mobile

Saves message metadata to Firestore and sends FCM push notifications
when Telegram messages are sent. Failure never affects Telegram delivery.

IMPORTANT: This module is opt-in and disabled by default.
Set FIREBASE_BRIDGE_ENABLED=true in .env to activate.
This is a PRISM-Mobile specific feature, not required for core prism-insight usage.
"""

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy initialization - only import firebase when actually needed
_db = None
_messaging = None
_initialized = False
_checked_enabled = False
_enabled = False


def _is_enabled() -> bool:
    """Check if Firebase Bridge is enabled via environment variable."""
    global _checked_enabled, _enabled
    if _checked_enabled:
        return _enabled
    _checked_enabled = True
    _enabled = os.environ.get('FIREBASE_BRIDGE_ENABLED', '').lower() in ('true', '1', 'yes')
    if not _enabled:
        logger.debug("Firebase Bridge disabled (set FIREBASE_BRIDGE_ENABLED=true to enable)")
    return _enabled


def _initialize():
    """Lazy initialize Firebase Admin SDK."""
    global _db, _messaging, _initialized
    if _initialized:
        return _initialized

    if not _is_enabled():
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, messaging

        # Check if already initialized
        try:
            app = firebase_admin.get_app()
        except ValueError:
            # Not yet initialized
            cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if not cred_path:
                logger.warning("Firebase not configured: GOOGLE_APPLICATION_CREDENTIALS not set")
                _initialized = False
                return False

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        _messaging = messaging
        _initialized = True
        logger.info("Firebase Bridge initialized successfully")
        return True
    except ImportError:
        logger.warning("firebase-admin not installed. Firebase Bridge disabled.")
        _initialized = False
        return False
    except Exception as e:
        logger.warning(f"Firebase Bridge initialization failed: {e}")
        _initialized = False
        return False


# Channel username for constructing telegram links (configurable via env)
TELEGRAM_CHANNEL_USERNAME = os.environ.get('TELEGRAM_CHANNEL_USERNAME', 'stock_ai_agent')

# Multi-channel username mapping: JSON dict of chat_id -> public username
# e.g. TELEGRAM_CHANNEL_USERNAMES='{"-1001234567890":"stock_ai_agent","-1009876543210":"prism_insight_global_en"}'
# Also supports per-language env vars: TELEGRAM_CHANNEL_USERNAME_EN, TELEGRAM_CHANNEL_USERNAME_JA, etc.
_channel_usernames_cache = None


def _get_channel_lang(channel_id: Optional[str]) -> str:
    """Resolve Telegram chat_id to language code.

    Returns 'ko' for the default channel, or the language suffix (lowercased)
    for TELEGRAM_CHANNEL_ID_{LANG} matches.
    """
    if not channel_id:
        return 'ko'

    default_channel = os.environ.get('TELEGRAM_CHANNEL_ID', '')
    if channel_id == default_channel:
        return 'ko'

    for key, value in os.environ.items():
        if key.startswith('TELEGRAM_CHANNEL_ID_') and value == channel_id:
            return key[len('TELEGRAM_CHANNEL_ID_'):].lower()  # e.g. 'en', 'ja'

    return 'ko'


def _get_channel_username(channel_id: Optional[str]) -> Optional[str]:
    """Resolve Telegram chat_id to public channel username for deep links.

    Lookup order:
    1. TELEGRAM_CHANNEL_USERNAMES JSON mapping (chat_id -> username)
    2. TELEGRAM_CHANNEL_ID_{LANG} / TELEGRAM_CHANNEL_USERNAME_{LANG} pairs
    3. Fallback to TELEGRAM_CHANNEL_USERNAME (default)

    Returns None if no username is found (e.g. private channels).
    """
    if not channel_id:
        return TELEGRAM_CHANNEL_USERNAME

    global _channel_usernames_cache
    if _channel_usernames_cache is None:
        raw = os.environ.get('TELEGRAM_CHANNEL_USERNAMES', '{}')
        try:
            _channel_usernames_cache = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            _channel_usernames_cache = {}

    # 1. Direct chat_id -> username mapping
    if channel_id in _channel_usernames_cache:
        return _channel_usernames_cache[channel_id]

    # 2. Scan TELEGRAM_CHANNEL_ID_{LANG} env vars
    for key, value in os.environ.items():
        if key.startswith('TELEGRAM_CHANNEL_ID_') and value == channel_id:
            lang_suffix = key[len('TELEGRAM_CHANNEL_ID_'):]  # e.g. 'EN', 'JA'
            username = os.environ.get(f'TELEGRAM_CHANNEL_USERNAME_{lang_suffix}', '')
            if username:
                _channel_usernames_cache[channel_id] = username
                return username

    # 3. Fallback
    return TELEGRAM_CHANNEL_USERNAME


def _build_telegram_link(channel_id: Optional[str], message_id: int) -> str:
    """Build telegram deep link, supporting both public and private channels.

    Public channels:  https://t.me/{username}/{message_id}
    Private channels: https://t.me/c/{channel_id_stripped}/{message_id}
    """
    username = _get_channel_username(channel_id)
    if username:
        return f"https://t.me/{username}/{message_id}"

    # Private channel fallback: strip -100 prefix
    if channel_id and channel_id.startswith('-100'):
        stripped = channel_id[4:]
        return f"https://t.me/c/{stripped}/{message_id}"

    return ''


def detect_market(message: str) -> str:
    """Detect market from message content."""
    # Indian stock patterns: NSE/BSE tickers, INR amounts, Indian company names
    in_patterns = [
        r'\bNSE\b|\bBSE\b|\bNIFTY\b|\bSENSEX\b',
        r'₹|INR|\bRs\.?\s',
        r'\bRELIANCE\b|\bTCS\b|\bINFOSYS\b|\bHDFC\b|\bICICI\b|\bSBI\b',
        r'\bFII\b|\bDII\b',  # Foreign/Domestic Institutional Investors
    ]

    in_score = sum(1 for p in in_patterns if re.search(p, message))

    if in_score > 0:
        return 'in'
    return 'in'  # Default to IN


def detect_type(message: str) -> str:
    """Detect message type from content.

    Priority order matters: portfolio > pdf > analysis > trigger.
    'portfolio report' must match portfolio, not pdf.
    """
    msg_lower = message.lower()

    # Portfolio detection FIRST
    if any(kw in msg_lower for kw in ['portfolio', 'holdings', 'positions']):
        return 'portfolio'

    # PDF report detection
    if any(kw in msg_lower for kw in ['.pdf', 'pdf report']):
        return 'pdf'

    # Analysis detection
    if any(kw in msg_lower for kw in [
        'analysis', 'outlook', 'summary',
        'buy', 'sell', 'hold', 'review',
    ]):
        return 'analysis'

    # Broad report as fallback
    if 'report' in msg_lower:
        return 'pdf'

    # Default: trigger (trading signal)
    return 'trigger'


def _clean_filename(text: str) -> str:
    """Clean a raw filename into a human-readable title.

    'pdfreports/012450_Hanwha_Aerospace_20260210_afternoon.pdf'
    → '012450 Hanwha Aerospace 20260210 afternoon'
    """
    # Extract filename from path
    name = text.split('/')[-1]
    # Remove file extension
    name = re.sub(r'\.\w{2,4}$', '', name)
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    return name.strip()


def _looks_like_filepath(text: str) -> bool:
    """Check if text looks like a file path or raw filename."""
    return bool(
        '/' in text
        or text.endswith('.pdf')
        or re.match(r'^[\w._-]+\.\w{2,4}$', text)
    )


def extract_title(message: str, max_length: int = 80) -> str:
    """Extract title from message - first meaningful line."""
    lines = message.strip().split('\n')
    for line in lines:
        cleaned = line.strip()
        # Skip empty lines, emoji-only lines, separator lines
        if not cleaned:
            continue
        if cleaned.startswith('---') or cleaned.startswith('==='):
            continue
        if len(cleaned) < 3:
            continue
        # Remove markdown formatting
        cleaned = re.sub(r'[*_`#]', '', cleaned).strip()
        if cleaned:
            # Clean file paths used as titles
            if _looks_like_filepath(cleaned):
                cleaned = _clean_filename(cleaned)
            return cleaned[:max_length]
    return message[:max_length].strip()


def extract_preview(message: str, max_length: int = 100) -> str:
    """Extract preview text from message."""
    # Remove markdown formatting
    text = re.sub(r'[*_`#]', '', message)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Don't expose raw file paths as preview
    if _looks_like_filepath(text.split(' ')[0]):
        text = _clean_filename(text.split(' ')[0])
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + '...'


def extract_stock_info(message: str) -> tuple:
    """Extract stock code and name from message.

    Returns:
        tuple: (stock_code, stock_name) or (None, None)
    """
    # Indian stock: NSE ticker symbol (e.g., RELIANCE, TCS, INFY)
    nse_match = re.search(r'\b([A-Z][A-Z0-9&]{1,20})\b', message)

    stock_code = nse_match.group(1) if nse_match else None
    stock_name = None  # Name extraction from message is optional

    return stock_code, stock_name


async def notify(
    message: str,
    market: Optional[str] = None,
    msg_type: Optional[str] = None,
    telegram_message_id: Optional[int] = None,
    channel_id: Optional[str] = None,
    has_pdf: bool = False,
    pdf_telegram_link: Optional[str] = None,
):
    """
    Save message metadata to Firestore and send FCM push.

    This function NEVER raises exceptions - all errors are logged and ignored
    to ensure Telegram delivery is never affected.

    Args:
        message: The telegram message text
        market: Market identifier ('kr' or 'us'). Auto-detected if None.
        msg_type: Message type. Auto-detected if None.
        telegram_message_id: Telegram message ID for deep link
        channel_id: Telegram channel ID (for reference)
        has_pdf: Whether this message has an associated PDF
        pdf_telegram_link: Direct link to PDF in Telegram
    """
    try:
        if not _initialize():
            return

        # Auto-detect if not provided
        if not market:
            market = detect_market(message)
        if not msg_type:
            msg_type = detect_type(message)

        title = extract_title(message)
        preview = extract_preview(message)
        stock_code, stock_name = extract_stock_info(message)

        # Resolve language from channel
        lang = _get_channel_lang(channel_id)

        # Build telegram link (public or private channel)
        telegram_link = ''
        if telegram_message_id:
            telegram_link = _build_telegram_link(channel_id, telegram_message_id)

        # Save to Firestore
        from google.cloud.firestore import SERVER_TIMESTAMP

        doc_data = {
            'type': msg_type,
            'market': market,
            'lang': lang,
            'title': title,
            'preview': preview,
            'telegram_link': telegram_link,
            'channel_id': channel_id,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'has_pdf': has_pdf,
            'pdf_telegram_link': pdf_telegram_link,
            'created_at': SERVER_TIMESTAMP,
        }

        _db.collection('messages').add(doc_data)
        logger.info(f"Firebase: Saved {msg_type}/{market} message to Firestore")

        # Send FCM push notification
        await _send_push(title, preview, msg_type, market)

    except Exception as e:
        logger.warning(f"Firebase Bridge notify failed (ignored): {e}")


async def _send_push(title: str, body: str, msg_type: str, market: str):
    """Send FCM push notification to subscribed devices."""
    try:
        if not _messaging:
            return

        # Query devices matching preferences
        devices_ref = _db.collection('devices')
        docs = devices_ref.stream()

        tokens = []
        for doc in docs:
            device = doc.to_dict()
            prefs = device.get('preferences', {})

            # Check market preference
            pref_markets = prefs.get('markets', ['in'])
            if market not in pref_markets:
                continue

            # Check type preference
            pref_types = prefs.get('types', ['trigger', 'analysis', 'portfolio', 'pdf'])
            if msg_type not in pref_types:
                continue

            token = device.get('token')
            if token:
                tokens.append(token)

        if not tokens:
            logger.info("Firebase: No matching devices for push notification")
            return

        # Send in batches of 500 (FCM limit)
        for i in range(0, len(tokens), 500):
            batch_tokens = tokens[i:i + 500]
            message = _messaging.MulticastMessage(
                notification=_messaging.Notification(
                    title=title,
                    body=body,
                ),
                data={
                    'type': msg_type,
                    'market': market,
                },
                tokens=batch_tokens,
            )

            response = _messaging.send_each_for_multicast(message)
            logger.info(
                f"Firebase: FCM sent to {response.success_count}/{len(batch_tokens)} devices"
            )
    except Exception as e:
        logger.warning(f"Firebase: FCM push failed (ignored): {e}")
