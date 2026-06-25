"""淘股吧 (TaoGuBa) stock-discussion fetcher.

淘股吧 is China's leading short-term trader community, focused on
momentum trading, "hot stocks" (龙头股), and short-term sentiment.
It is the Chinese equivalent of /r/wallstreetbets meets StockTwits —
retail traders sharing trade ideas, daily P&L, and hot takes.

Unlike 雪球 (which is behind Alibaba Cloud WAF and cannot be scraped
programmatically), 淘股吧 embeds structured discussion data directly
in its stock quotes page (``taoguba.com.cn/quotes/sh{code}``) as a
JavaScript variable (``coolAttr``). No API key, no cookies, no WAF
bypass needed.

Design follows the same pattern as ``stocktwits.py``: self-contained,
short timeout, graceful degradation, string return.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
from urllib.request import Request, urlopen

import certifi

logger = logging.getLogger(__name__)

# ---------- constants ----------

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# Taoguba stock quotes page. The prefix is:
#   sh  - Shanghai (600xxx, 601xxx, 603xxx, 688xxx)
#   sz  - Shenzhen (000xxx, 001xxx, 002xxx, 300xxx)
_QUOTES_PAGE = "https://www.taoguba.com.cn/quotes/{exchange}{code}"

_A_SHARE_SUFFIX = re.compile(r"\.(SS|SZ)$", re.IGNORECASE)
_EXCHANGE_MAP = {"SS": "sh", "SZ": "sz"}

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

# Regex to extract the embedded ``coolAttr`` JSON variable from the page.
_RE_COOL_ATTR = re.compile(
    r'var coolAttr\s*=\s*(\[.*?\])\s*;\s*(?:var|\n)',
    re.DOTALL,
)


# ---------- helpers ----------


def _ticker_to_params(ticker: str) -> tuple[str, str] | None:
    """Convert an A-share ticker to (exchange_prefix, numeric_code).

    ``600519.SS``  →  ``("sh", "600519")``
    ``000001.SZ``  →  ``("sz", "000001")``

    Returns ``None`` when the ticker is not an A-share stock.
    """
    match = _A_SHARE_SUFFIX.search(ticker)
    if not match:
        return None
    code = ticker[: match.start()]
    if not code.isdigit() or len(code) != 6:
        return None
    exchange = _EXCHANGE_MAP.get(match.group(1).upper())
    if exchange is None:
        return None
    return exchange, code


def _truncate(text: str, max_len: int = 280) -> str:
    """Truncate text with ellipsis if it exceeds ``max_len``."""
    text = text.replace("\n", " ").strip()
    return text[: max_len - 1] + "…" if len(text) > max_len else text


def _heuristic_sentiment(text: str) -> str:
    """Rough Bullish/Bearish/no-label hint from post text."""
    tl = text.lower()
    has_bull = any(kw in tl for kw in _BULLISH_KEYWORDS)
    has_bear = any(kw in tl for kw in _BEARISH_KEYWORDS)

    if has_bull and not has_bear:
        return "Bullish"
    if has_bear and not has_bull:
        return "Bearish"
    return "no-label"


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&amp;", "&").replace("&quot;", '"')
    return " ".join(text.split())


# ---------- public interface ----------


def fetch_taoguba_posts(ticker: str, timeout: float = 15.0) -> str:
    """Fetch recent 淘股吧 posts for an A-share ``ticker``.

    The ticker must use an A-share exchange suffix:
        ``.SS``  — Shanghai Stock Exchange
        ``.SZ``  — Shenzhen Stock Exchange (incl. ChiNext 300xxx)

    Returns a formatted plaintext block ready for prompt injection.
    Returns a placeholder string when the ticker is not an A-share stock,
    the page is unreachable, or no posts are found — the caller never has
    to special-case ``None`` or exceptions.
    """
    params = _ticker_to_params(ticker)
    if params is None:
        return (
            f"<taoguba unavailable: {ticker} does not appear to be "
            f"an A-share stock (expected .SS or .SZ suffix)>"
        )

    exchange, code = params
    url = _QUOTES_PAGE.format(exchange=exchange, code=code)

    req = Request(
        url,
        headers={
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("TaoGuBa fetch failed for %s: %s", ticker, exc)
        return f"<taoguba unavailable for {ticker}: {type(exc).__name__}>"

    # Extract embedded JSON (coolAttr).
    match = _RE_COOL_ATTR.search(html)
    if not match:
        return (
            f"<taoguba unavailable for {ticker}: no discussion data "
            f"found on the quotes page.>"
        )

    try:
        posts = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        logger.warning("TaoGuBa JSON parse failed for %s: %s", ticker, exc)
        return f"<taoguba unavailable for {ticker}: JSON parse error>"

    if not posts or not isinstance(posts, list):
        return f"<taoguba unavailable for {ticker}: empty post list>"

    lines = []
    lines.append(
        f"淘股吧 ({code}) — {len(posts)} recent posts "
        f"(short-term trader community)"
    )

    bullish_cnt = bearish_cnt = no_label_cnt = 0
    for p in posts[:30]:
        subject = _truncate(_strip_html(p.get("subject") or ""), 120)
        body = _truncate(_strip_html(p.get("body") or ""), 300)
        author = p.get("userName") or "?"
        action_date = (p.get("actionDate") or "")[:19].replace("T", " ")
        reply_num = p.get("replyNum") or 0
        view_num = p.get("viewNum") or 0
        useful_num = p.get("usefulNum") or 0
        fans = p.get("totalFansNum") or 0
        rtype = p.get("rType", "")
        # rType: T=thread (original post), R=reply, W=weibo
        type_label = {"T": "thread", "R": "reply", "W": "weibo"}.get(rtype, rtype)

        tag = _heuristic_sentiment(subject)
        if tag == "Bullish":
            bullish_cnt += 1
        elif tag == "Bearish":
            bearish_cnt += 1
        else:
            no_label_cnt += 1

        line = (
            f"  [{action_date} · {author} · {tag} · {type_label} · "
            f"👁{view_num} 💬{reply_num} 👍{useful_num}"
        )
        if fans > 1000:
            line += f" ★{fans}fans"
        line += f"] {subject}"
        if body and body != subject:
            line += f"\n    body: {body}"
        lines.append(line)

    total = bullish_cnt + bearish_cnt + no_label_cnt
    summary = (
        f"Bullish (heuristic): {bullish_cnt} · "
        f"Bearish (heuristic): {bearish_cnt} · "
        f"Unlabeled: {no_label_cnt} · "
        f"Total: {total} most-recent posts"
    )
    return summary + "\n\n" + "\n".join(lines)
