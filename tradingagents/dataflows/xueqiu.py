"""雪球 (Xueqiu) stock-discussion fetcher.

⛔  DEPRECATED — Xueqiu is behind Alibaba Cloud WAF (Aliyun WAF) which
requires JavaScript challenge solving. Simple HTTP requests cannot bypass
it. This module is kept as a reference for when a browser-based proxy
(e.g. Playwright, Selenium) is wired in.

Xueqiu is China's most active long-form investor community, comparable to
a mix of Seeking Alpha and StockTwits. If you do wire in a browser-based
solution, the primary endpoint is:

    https://xueqiu.com/statuses/stock_timeline.json?symbol_id={symbol_id}&count=20

The current implementation attempts to use ``HTTPCookieProcessor`` for
session management, but the Aliyun WAF intercepts the API call with a
JS challenge page before the JSON response is served.

See ``taoguba.py`` for a working alternative that provides short-term
trader sentiment data for A-share stocks, or ``eastmoney.py`` for the
retail-investor board sentiment.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import time as _time
from http.cookiejar import CookieJar
from typing import Any
from urllib.request import HTTPCookieProcessor, HTTPErrorProcessor, Request, build_opener

import certifi

logger = logging.getLogger(__name__)

# ---------- constants ----------

# Real Chrome UA — generic/bot UAs trip Xueqiu's WAF.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# Session warmup — the homepage 302 redirects before setting the
# xq_a_token cookie that the API requires.
_XUEQIU_HOME = "https://xueqiu.com/"

# Timeline API endpoints (primary + mobile fallback).
_PRIMARY_API = (
    "https://xueqiu.com/statuses/stock_timeline.json"
    "?symbol_id={symbol_id}&count=20"
)
_MOBILE_API = (
    "https://api.xueqiu.com/statuses/stock_timeline.json"
    "?symbol_id={symbol_id}&count=20"
)

# A-share exchange suffixes (must match the benchmark_map in default_config.py).
_A_SHARE_SUFFIX = re.compile(r"\.(SS|SZ)$", re.IGNORECASE)

_PREFIX_MAP: dict[str, str] = {"SS": "SH", "SZ": "SZ"}

# Bullish/bearish keyword list (kept consistent with eastmoney.py).
_BULLISH_KEYWORDS = [
    "涨", "牛", "买", "入", "突破", "利好", "起飞", "加仓",
    "暴", "赚", "红", "升", "持有", "抄底", "放量", "拉升",
    "强势", "爆发", "看好", "做多", "持有待涨",
]
_BEARISH_KEYWORDS = [
    "跌", "崩", "空", "卖", "跑", "割肉", "止损", "利空",
    "垃圾", "逃", "亏", "惨", "危险", "熊", "出货", "暴跌",
    "跳水", "破位", "清仓", "离场", "做空", "唱空",
]


# ---------- helpers ----------


def _ticker_to_symbol_id(ticker: str) -> str | None:
    """Convert an A-share ticker to the Xueqiu ``symbol_id`` format.

    ``600519.SS``  →  ``SH600519``
    ``000001.SZ``  →  ``SZ000001``

    Returns ``None`` when the ticker is not an A-share stock.
    """
    match = _A_SHARE_SUFFIX.search(ticker)
    if not match:
        return None
    code = ticker[: match.start()]
    if not code.isdigit() or len(code) != 6:
        return None
    prefix = _PREFIX_MAP.get(match.group(1).upper())
    return None if prefix is None else f"{prefix}{code}"


def _truncate(text: str, max_len: int = 280) -> str:
    """Truncate text with ellipsis if it exceeds ``max_len``."""
    text = text.replace("\n", " ").strip()
    return text[: max_len - 1] + "…" if len(text) > max_len else text


def _parse_entities(status: dict[str, Any]) -> str:
    """Build a concise description of attached images/links from entities."""
    entities = status.get("entities") or {}
    parts = []
    images = entities.get("images") or []
    if images:
        parts.append(f"{len(images)} image(s)")
    links = entities.get("links") or []
    if links:
        parts.append(f"{len(links)} link(s)")
    return f" [{', '.join(parts)}]" if parts else ""


def _heuristic_sentiment(text: str) -> str:
    """Rough Bullish/Bearish/no-label hint from post content."""
    tl = text.lower()
    has_bull = any(kw in tl for kw in _BULLISH_KEYWORDS)
    has_bear = any(kw in tl for kw in _BEARISH_KEYWORDS)

    if has_bull and not has_bear:
        return "Bullish"
    if has_bear and not has_bull:
        return "Bearish"
    return "no-label"


# ---------- cookie-aware opener ----------


def _build_session_opener() -> tuple[Any, CookieJar]:
    """Create a ``build_opener`` with cookie processing.

    Uses a custom ``HTTPErrorProcessor`` so that error responses don't
    raise — the caller inspects the response directly.
    """

    class _SilentHTTPErrorProcessor(HTTPErrorProcessor):
        def http_response(self, request, response):
            return response

        def https_response(self, request, response):
            return response

    cj = CookieJar()
    opener = build_opener(_SilentHTTPErrorProcessor, HTTPCookieProcessor(cj))
    return opener, cj


def _warm_session(
    opener: Any,
    cj: CookieJar,
    timeout: float,
) -> bool:
    """Visit ``_XUEQIU_HOME`` to populate the ``CookieJar`` with session cookies.

    Xueqiu's homepage 302-redirects to ``/`` or a landing page, and the
    ``xq_a_token`` cookie is set during that redirect chain.  Using
    ``HTTPCookieProcessor`` means the cookies from every response in the
    chain are automatically accumulated in the jar.

    Returns ``True`` if the ``xq_a_token`` cookie was captured, ``False``
    otherwise (the caller may still try the API — some endpoints work with
    a partial session).
    """
    req = Request(
        _XUEQIU_HOME,
        headers={
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        opener.open(req, timeout=timeout)
    except Exception as exc:
        logger.debug("Xueqiu session warmup failed: %s", exc)
        return False

    # Check whether we got a useful token.
    for cookie in cj:
        if cookie.name in ("xq_a_token", "xq_r_token"):
            logger.debug(
                "Xueqiu session cookie %s=%s… (domain=%s)",
                cookie.name,
                cookie.value[:8] if cookie.value else "None",
                cookie.domain,
            )
    has_token = any(c.name == "xq_a_token" for c in cj)
    if not has_token:
        logger.debug("Xueqiu warmup completed but no xq_a_token cookie was set")
    return has_token


def _make_api_headers(symbol_id: str) -> dict[str, str]:
    """Build browser-like headers for the timeline API request."""
    return {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": f"https://xueqiu.com/S/{symbol_id}",
        "Origin": "https://xueqiu.com",
        "Accept-Encoding": "identity",
        "X-Requested-With": "XMLHttpRequest",
    }


def _try_timeline_api(
    symbol_id: str,
    url_tpl: str,
    opener: Any,
    timeout: float,
) -> dict | None:
    """Call a Xueqiu timeline API endpoint and return parsed JSON or None."""
    url = url_tpl.format(symbol_id=symbol_id)
    req = Request(url, headers=_make_api_headers(symbol_id))
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
    except Exception as exc:
        logger.debug("Xueqiu API %s … %s failed: %s", url[:55], symbol_id, exc)
        return None

    if status != 200:
        logger.debug("Xueqiu API returned HTTP %s for %s", status, symbol_id)
        return None

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug("Xueqiu JSON decode failed for %s: %s", symbol_id, exc)
        return None

    if not isinstance(data, dict):
        logger.debug("Xueqiu non-dict response for %s: %s", symbol_id, type(data).__name__)
        return None

    statuses = data.get("list") or []
    if not statuses:
        logger.debug("Xueqiu empty timeline for %s", symbol_id)
        return None

    return data


# ---------- public interface ----------


def fetch_xueqiu_posts(ticker: str, timeout: float = 15.0) -> str:
    """Fetch recent 雪球 posts for an A-share ``ticker``.

    The ticker must use an A-share exchange suffix:
        ``.SS``  — Shanghai Stock Exchange
        ``.SZ``  — Shenzhen Stock Exchange (incl. ChiNext 300xxx)

    Returns a formatted plaintext block ready for prompt injection.
    Returns a placeholder string when the ticker is not an A-share stock,
    session negotiation fails, the API is unreachable, or no posts are
    found — the caller never has to special-case ``None`` or exceptions.
    """
    symbol_id = _ticker_to_symbol_id(ticker)
    if symbol_id is None:
        return (
            f"<xueqiu unavailable: {ticker} does not appear to be "
            f"an A-share stock (expected .SS or .SZ suffix)>"
        )

    # Build a cookie-aware opener and warm the session.
    opener, cj = _build_session_opener()
    has_token = _warm_session(opener, cj, timeout)

    # Try the endpoints in order.
    endpoints = [
        ("primary", _PRIMARY_API),
        ("mobile", _MOBILE_API),
    ]

    data = None
    used_label = ""
    for label, url_tpl in endpoints:
        result = _try_timeline_api(symbol_id, url_tpl, opener, timeout)
        if result is not None:
            data = result
            used_label = label
            logger.info("Xueqiu %s succeeded via %s endpoint", symbol_id, label)
            break

    if data is None:
        reason = (
            "session cookie unavailable"
            if not has_token
            else "all endpoints returned no data"
        )
        return (
            f"<xueqiu unavailable for {symbol_id}: {reason}>"
        )

    statuses = data.get("list", [])
    lines = []
    lines.append(
        f"雪球 ({symbol_id}) — {len(statuses)} recent posts "
        f"(long-form investor community; fetched via {used_label} endpoint)"
    )

    bullish_cnt = bearish_cnt = no_label_cnt = 0
    for s in statuses[:30]:
        text = s.get("text") or s.get("description") or ""
        # Strip HTML tags from rich-text content.
        text_clean = re.sub(r"<[^>]+>", " ", text).strip()
        truncated = _truncate(text_clean, 400)

        user_obj = s.get("user") or {}
        author = user_obj.get("screen_name") or "?"
        followers = user_obj.get("followers_count") or 0
        follow_str = f" ({followers} followers)" if followers > 1000 else ""

        created = ""
        raw_ts = s.get("created_at")
        if raw_ts:
            created = _time.strftime(
                "%Y-%m-%d %H:%M",
                _time.localtime(raw_ts / 1000),
            )

        retweets = s.get("retweet_count", 0)
        replies = s.get("reply_count", 0)
        likes = s.get("like_count", 0)

        tag = _heuristic_sentiment(truncated)
        if tag == "Bullish":
            bullish_cnt += 1
        elif tag == "Bearish":
            bearish_cnt += 1
        else:
            no_label_cnt += 1

        entities_desc = _parse_entities(s)

        lines.append(
            f"  [{created} · {author}{follow_str} · {tag}] "
            f"↻{retweets} ♡{likes} 💬{replies}{entities_desc}"
            f"\n    {truncated}"
        )

    total = bullish_cnt + bearish_cnt + no_label_cnt
    summary = (
        f"Bullish (heuristic): {bullish_cnt} · "
        f"Bearish (heuristic): {bearish_cnt} · "
        f"Unlabeled: {no_label_cnt} · "
        f"Total: {total} most-recent posts"
    )
    return summary + "\n\n" + "\n".join(lines)
