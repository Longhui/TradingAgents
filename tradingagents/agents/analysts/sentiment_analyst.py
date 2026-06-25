"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks.
Which sources are fetched depends on the market:

  **Non-A-share (US, HK, IN, JP, etc.):**
    1. News headlines     — Yahoo Finance (institutional framing)
    2. StockTwits messages — retail-trader posts with user-labeled sentiment
    3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

  **A-share (.SS / .SZ):**
    1. News headlines     — Yahoo Finance (institutional framing)
    2. 东方财富股吧         — China's largest retail-investor stock board
    3. 雪球                — China's leading long-form investor community

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.eastmoney import fetch_eastmoney_posts
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.taoguba import fetch_taoguba_posts

# A-share exchange suffixes (must match the benchmark_map in default_config.py).
_A_SHARE_SUFFIX = re.compile(r"\.(SS|SZ)$", re.IGNORECASE)


def _is_a_share(ticker: str) -> bool:
    """Return ``True`` when the ticker is an A-share stock (.SS or .SZ)."""
    return bool(_A_SHARE_SUFFIX.search(ticker))


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + market-appropriate social-media data, injects them
    into the prompt as structured blocks, and produces a deterministic
    sentiment report via structured output (with a free-text fallback for
    providers that do not support it).

    For A-share tickers (``.SS`` / ``.SZ``) the social-media sources are
    东方财富股吧 and 雪球; for all other markets StockTwits and Reddit are
    used instead.
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state: dict[str, Any]):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # News is fetched for all markets (yfinance covers A-share news too).
        news_block = get_news.func(ticker, start_date, end_date)

        if _is_a_share(ticker):
            # Chinese A-share data sources.
            eastmoney_block = fetch_eastmoney_posts(ticker)
            taoguba_block = fetch_taoguba_posts(ticker)
            system_message = _build_a_share_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                eastmoney_block=eastmoney_block,
                taoguba_block=taoguba_block,
            )
        else:
            # Default (US/international) data sources.
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)
            system_message = _build_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                stocktwits_block=stocktwits_block,
                reddit_block=reddit_block,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


# ---------------------------------------------------------------------------
# System message builders
# ---------------------------------------------------------------------------


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble the sentiment-analyst system message (non-A-share / default)."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


def _build_a_share_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    eastmoney_block: str,
    taoguba_block: str,
) -> str:
    """Assemble the sentiment-analyst system message for A-share (.SS/.SZ) stocks."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

{ticker} is an **A-share stock** listed on either the Shanghai Stock Exchange (.SS) or the Shenzhen Stock Exchange (.SZ). The Chinese A-share market has distinct characteristics from US markets: heavy retail participation, strong policy sensitivity, high short-term volatility, and a different news/regulatory environment.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal. Note that yfinance covers A-share stocks under their local ticker, but coverage may be less comprehensive than for US stocks.

<start_of_news>
{news_block}
<end_of_news>

### 东方财富股吧 (EastMoney Stock Board) — largest Chinese retail-investor community
Very high activity, very emotional. This is the Chinese equivalent of StockTwits × Reddit. Each post shows a sentiment tag:
- **Bullish / Bearish (heuristic)** — derived from the post title via keyword matching (e.g. "涨/牛/利好" → Bullish, "跌/崩/利空" → Bearish). These are ROUGH HINTS, NOT user-labeled tags. Treat them as directional indicators and read the post body for the real tone.
- **no-label** — the heuristic could not assign a direction; read the body to judge.

<start_of_eastmoney>
{eastmoney_block}
<end_of_eastmoney>

### 淘股吧 (TaoGuBa) — short-term trader community
Fast-moving, momentum-focused. The Chinese equivalent of /r/wallstreetbets. Users share trade ideas, daily P&L, sector rotation views, and hot stock picks. Posts are categorized as:
- **thread** — original post (most signal)
- **reply** — comment on another post
- **weibo** — short broadcast

<start_of_taoguba>
{taoguba_block}
<end_of_taoguba>

## How to analyze this data (best practices for A-share sentiment)

1. **Understand Chinese retail-investor language.** A-share retail investors use very direct emotional language. "牛" (bull), "涨" (up), "突破" (breakout), "利好" (good news), "起飞" (taking off) signal bullishness. "跌" (down), "崩" (collapse), "割肉" (cut losses), "利空" (bad news), "垃圾" (trash) signal bearishness. But the same posters who are exuberant one day may panic the next — treat extremes as contrarian signals.

2. **东方财富 股吧 is a sentiment amplifier, not a fundamental analysis source.** Posts are short, emotional, and trend-following. A flood of bullish posts often coincides with a local top (retail buying euphoria); a flood of bearish posts may mark a bottom (retail panic selling). This "reverse indicator" dynamic is well-known among A-share traders. Use it as a **sentiment thermometer** rather than a directional signal.

3. **淘股吧 reflects the short-term momentum crowd.** Pay attention to sector rotation discussions, hot stock picks (龙头股), and whether the tone is risk-on (chasing breakouts) or risk-off (cutting losses, going to cash). 淘股吧 sentiment is more leading/tactical than 股吧 — it shows what active short-term traders are thinking right now, not the broader retail base.

4. **Look for cross-source divergences.** If news is neutral but 股吧 is overwhelmingly bullish and 淘股吧 is cautious about a sector rotation, the divergence between retail euphoria and trader caution is itself the signal.

5. **Policy sensitivity.** A-share markets are heavily influenced by government policy, regulatory announcements, and PBOC/CSRC statements. Pay extra attention to news about interest rates, industry regulation, and government directives — these move markets more than earnings in many cases.

6. **Watch for pump-and-dump / coordinated sentiment.** Chinese retail investors sometimes coordinate on social platforms to hype a stock. Unusually uniform bullish language across many posts at once, especially from new/low-follower accounts, may signal coordinated activity rather than organic sentiment.

7. **Sample size and data honesty.** If one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly. The heuristic sentiment tags are directional, not definitive.

8. **Identify theme clusters.** What narrative keeps appearing across 股吧 and 淘股吧? Policy change? Earnings season? Sector rotation? A thematic consensus across both communities is the strongest signal.

9. **Catalysts and risks.** Flag upcoming events visible across sources — earnings, product launches, regulatory decisions, macro data releases, etc.

10. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## A-share market context (for your analysis)

- The A-share market has a **T+1 settlement** rule (shares bought today cannot be sold until tomorrow), which affects short-term trading dynamics.
- There is a **10% daily price limit** (±10% from previous close; ±20% for ChiNext/STAR boards) — extreme sentiment may manifest as limit-up/limit-down rather than moderate price moves.
- The market is **retail-dominated** (retail investors account for ~60-80% of daily turnover), making sentiment indicators particularly relevant.
- **State media** (Xinhua, CCTV, Securities Times) and official statements can rapidly shift market sentiment — news sources may carry policy signals.
- **North-bound flows** (foreign investor access via Stock Connect) are tracked closely as a "smart money" indicator among Chinese retail traders.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
