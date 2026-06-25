"""东方财富股吧 (EastMoney Stock Bar) fetcher.

EastMoney's 股吧 (guba / stock discussion board) is the largest
retail-investor community in China. Every A-share listed stock has a
dedicated board where users post opinions, analysis, and rumours.

The JSON API endpoint that earlier versions of this module relied on
(``guba.eastmoney.com/ajax/guba/list``) was removed by EastMoney.
We now scrape the HTML listing page directly (``list,{code}.html``)
and extract post data via regex. The page is publicly accessible with
a browser User-Agent and short timeout.

Design follows the same pattern as ``stocktwits.py``: self-contained,
short timeout, graceful degradation, string return.
"""

from __future__ import annotations

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

_HTML_LIST = "https://guba.eastmoney.com/list,{code}.html"

_A_SHARE_SUFFIX = re.compile(r"\.(SS|SZ)$", re.IGNORECASE)

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

# Regex matching the current EastMoney 股吧 listing page structure.
#
# Each post is a <tr class="listitem"> containing:
#   <td><div class="read">CLICKS</div></td>
#   <td><div class="reply">REPLIES</div></td>
#   <td><div class="title"><a ...>TITLE</a></div></td>
#   <td><div class="author"><a ...>AUTHOR</a></div></td>
#   <td><div class="update">DATE</div></td>
_RE_POST_ROW = re.compile(
    r'<tr class="listitem">.*?'
    r'<div class="read">(\d+)</div>.*?'       # group(1) = clicks
    r'<div class="reply">(\d+)</div>.*?'       # group(2) = replies
    r'<div class="title"><a[^>]*>([^<]+)</a>.*?'    # group(3) = title
    r'<div class="author"><a[^>]*>([^<]+)</a>.*?'   # group(4) = author
    r'<div class="update">([^<]+)</div>',      # group(5) = update time
    re.DOTALL,
)


# ---------- helpers ----------


def _ticker_to_code(ticker: str) -> str | None:
    """Convert an A-share ticker to the 6-digit numeric stock code.

    ``600519.SS``  →  ``600519``
    ``000001.SZ``  →  ``000001``
    ``300750.SZ``  →  ``300750``

    Returns ``None`` when the ticker is not an A-share stock.
    """
    match = _A_SHARE_SUFFIX.search(ticker)
    if not match:
        return None
    code = ticker[: match.start()]
    return code if code.isdigit() and len(code) == 6 else None


def _format_user_name(raw: str | None) -> str:
    """Shorten user name for display, masking middle characters."""
    if not raw:
        return "?"
    name = raw.strip()
    return name if len(name) <= 4 else name[:2] + "**" + name[-1]


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis if it exceeds ``max_len``."""
    text = text.replace("\n", " ").strip()
    return text[: max_len - 1] + "…" if len(text) > max_len else text


def _heuristic_sentiment(title: str) -> str:
    """Rough Bullish/Bearish/no-label hint based on title keywords."""
    tl = title.lower()
    has_bull = any(kw in tl for kw in _BULLISH_KEYWORDS)
    has_bear = any(kw in tl for kw in _BEARISH_KEYWORDS)

    if has_bull and not has_bear:
        return "Bullish"
    if has_bear and not has_bull:
        return "Bearish"
    return "no-label"


# ---------- public interface ----------


def fetch_eastmoney_posts(ticker: str, timeout: float = 15.0) -> str:
    """Fetch recent 东方财富股吧 posts for an A-share ``ticker``.

    The ticker must use an A-share exchange suffix:
        ``.SS``  — Shanghai Stock Exchange
        ``.SZ``  — Shenzhen Stock Exchange (incl. ChiNext 300xxx)

    Returns a formatted plaintext block ready for prompt injection.
    Returns a placeholder string when the ticker is not an A-share stock,
    the listing page is unreachable, or no posts are found — the caller
    never has to special-case ``None`` or exceptions.
    """
    code = _ticker_to_code(ticker)
    if code is None:
        return (
            f"<eastmoney unavailable: {ticker} does not appear to be "
            f"an A-share stock (expected .SS or .SZ suffix)>"
        )

    # Fetch the HTML listing page.
    url = _HTML_LIST.format(code=code)
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
        logger.warning("EastMoney HTML fetch failed for %s: %s", code, exc)
        return f"<eastmoney unavailable for {code}: {type(exc).__name__}>"

    # Extract posts via regex.
    rows = _RE_POST_ROW.findall(html)
    if not rows:
        logger.warning("EastMoney HTML parse found 0 rows for %s", code)
        return (
            f"<eastmoney unavailable for {code}: no discussion posts found "
            f"on the listing page.>"
        )

    lines = []
    lines.append(
        f"东财股吧 ({code}) — {len(rows)} recent posts"
    )

    bullish_cnt = bearish_cnt = no_label_cnt = 0
    for row in rows[:30]:
        clicks_str, replies_str, title, author, update_time = row
        title_clean = _truncate(title.strip(), 120)
        author_fmt = _format_user_name(author.strip())

        tag = _heuristic_sentiment(title_clean)
        if tag == "Bullish":
            bullish_cnt += 1
        elif tag == "Bearish":
            bearish_cnt += 1
        else:
            no_label_cnt += 1

        lines.append(
            f"  [{update_time.strip()} · {author_fmt} · {tag} · "
            f"{clicks_str} clicks / {replies_str} replies] "
            f"{title_clean}"
        )

    total = bullish_cnt + bearish_cnt + no_label_cnt
    summary = (
        f"Bullish (heuristic): {bullish_cnt} · "
        f"Bearish (heuristic): {bearish_cnt} · "
        f"Unlabeled: {no_label_cnt} · "
        f"Total: {total} most-recent posts"
    )
    return summary + "\n\n" + "\n".join(lines)
