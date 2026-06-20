"""Futu (富途牛牛) data vendor via FutuOpenD gateway.

Provides stock price data and fundamentals through the local FutuOpenD
daemon (port 11111 by default).  Unlike yfinance/Alpha Vantage, Futu
requires a running FutuOpenD instance AND a Futu brokerage account.

Market prefix mapping (Futu convention):
  US.AAPL, HK.00700, SH.600036, SZ.399001, UK.III, JP.7203
"""
import logging
from datetime import datetime

import pandas as pd

from .errors import NoMarketDataError

logger = logging.getLogger(__name__)

# Default FutuOpenD connection — override via FUTU_OPEND_HOST / FUTU_OPEND_PORT
_FUTU_HOST = "127.0.0.1"
_FUTU_QUOTE_PORT = 11111

# ---------------------------------------------------------------------------
# Symbol conversion  (yfinance / plain ticker -> Futu market prefix)
# ---------------------------------------------------------------------------
_FUTU_PREFIX_MAP: dict[str, str] = {
    ".HK": "HK",
    ".T":  "JP",
    ".L":  "UK",
    ".TO": "CA",
    ".AX": "AU",
    ".SS": "SH",
    ".SZ": "SZ",
}
_FUTU_PREFIXES = {"HK", "US", "SH", "SZ", "JP", "UK", "CA", "AU"}


def _to_futu_code(symbol: str) -> str:
    """Convert a yfinance-style or bare symbol to Futu format.

    Examples:
        AAPL        ->  US.AAPL
        0700.HK     ->  HK.00700
        600036.SS   ->  SH.600036
        HK.00700    ->  HK.00700  (passthrough)
    """
    s = symbol.strip()

    for prefix in _FUTU_PREFIXES:
        if s.upper().startswith(prefix + "."):
            return s.upper()

    upper = s.upper()
    for suffix, market in _FUTU_PREFIX_MAP.items():
        if upper.endswith(suffix):
            code = s[: -len(suffix)]
            if market == "HK":
                try:
                    code = f"{int(code):05d}"
                except ValueError:
                    pass
            return f"{market}.{code}"

    return f"US.{upper}"


# ---------------------------------------------------------------------------
# Lazily import the futu SDK (bypass local namespace collision)
# ---------------------------------------------------------------------------
def _get_futu():
    """Return the ``futu`` package module (installed SDK, not this file)."""
    try:
        return __import__("futu")
    except ImportError:
        raise ImportError(
            "futu-api SDK not installed. Run: pip install futu-api"
        )


# ---------------------------------------------------------------------------
# Stock price data  (K-line)
# ---------------------------------------------------------------------------
def get_futu_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Fetch daily OHLCV data from Futu, returned as CSV (same shape as yfinance).

    Args:
        symbol: Ticker in yfinance (e.g. ``AAPL``, ``0700.HK``) or Futu format.
        start_date: Start date ``YYYY-MM-DD``.
        end_date: End date ``YYYY-MM-DD`` (inclusive).

    Returns:
        CSV string with columns ``Date,Open,High,Low,Close,Volume``.
    """
    futu_code = _to_futu_code(symbol)
    futu = _get_futu()

    ctx = futu.OpenQuoteContext(host=_FUTU_HOST, port=_FUTU_QUOTE_PORT)
    try:
        ret_code, data, page_key = ctx.request_history_kline(
            futu_code,
            start=start_date,
            end=end_date,
            ktype="K_DAY",
            fields=[""],
        )
        if ret_code != futu.RET_OK or data is None or data.empty:
            raise NoMarketDataError(
                symbol,
                futu_code,
                f"no rows between {start_date} and {end_date}",
            )

        # Rename columns to match yfinance convention
        mapping = {
            "time_key": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = data.rename(columns=mapping)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col].round(2)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.sort_values("Date")

        csv_string = df.to_csv(index=False)

        header = (
            f"# Stock data for {futu_code} from {start_date} to {end_date}\n"
            f"# Total records: {len(df)}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Company fundamentals (via market snapshot)
# ---------------------------------------------------------------------------
def get_futu_fundamentals(
    symbol: str,
    curr_date: str | None = None,
) -> str:
    """Fetch fundamental data snapshot from Futu.

    Returns:
        Formatted text report with key financial metrics.
    """
    futu_code = _to_futu_code(symbol)
    futu = _get_futu()

    ctx = futu.OpenQuoteContext(host=_FUTU_HOST, port=_FUTU_QUOTE_PORT)
    try:
        ret_code, data = ctx.get_market_snapshot([futu_code])
        if ret_code != futu.RET_OK or data is None or data.empty:
            raise NoMarketDataError(
                symbol, futu_code, "no snapshot data returned"
            )

        row = data.iloc[0]
        name = row.get("name", "")

        fields = [
            ("Name", name),
            ("Last Price", row.get("last_price")),
            ("Open", row.get("open_price")),
            ("High", row.get("high_price")),
            ("Low", row.get("low_price")),
            ("Prev Close", row.get("prev_close_price")),
            ("Volume", row.get("volume")),
            ("Turnover", row.get("turnover")),
            ("P/E Ratio (TTM)", row.get("pe_ttm_ratio")),
            ("P/E Ratio", row.get("pe_ratio")),
            ("P/B Ratio", row.get("pb_ratio")),
            ("EPS (TTM)", row.get("earning_per_share")),
            ("Net Asset Per Share", row.get("net_asset_per_share")),
            ("Dividend Yield (TTM)", row.get("dividend_ratio_ttm")),
            ("Dividend (LFY)", row.get("dividend_lfy")),
            ("Market Cap", row.get("total_market_val")),
            ("Outstanding Shares", row.get("outstanding_shares")),
            ("Issued Shares", row.get("issued_shares")),
            ("Turnover Rate", row.get("turnover_rate")),
            ("Amplitude", row.get("amplitude")),
            ("52-Week High", row.get("highest52weeks_price")),
            ("52-Week Low", row.get("lowest52weeks_price")),
            ("Net Asset", row.get("net_asset")),
            ("Net Profit", row.get("net_profit")),
        ]

        lines = [f"{label}: {value}" for label, value in fields if value is not None]
        if not lines:
            raise NoMarketDataError(symbol, futu_code, "no fundamental fields returned")

        header = (
            f"# Company Fundamentals for {futu_code} ({name})\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + "\n".join(lines)
    finally:
        ctx.close()
