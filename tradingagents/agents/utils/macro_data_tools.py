import logging
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor

logger = logging.getLogger(__name__)


def _safe_vendor_call(tool_name: str, *args, **kwargs) -> str:
    """Call route_to_vendor and return a helpful message on error instead of crashing."""
    try:
        return route_to_vendor(tool_name, *args, **kwargs)
    except ValueError as e:
        logger.warning("%s failed: %s", tool_name, e)
        return (
            f"⚠️ Unable to fetch data for {tool_name}: {e}\n"
            f"The tool was called with args: {args!r}, kwargs: {kwargs!r}\n"
        )


_VALID_ALIASES = [
    "cpi", "core_cpi", "pce", "core_pce", "inflation_expectations",
    "fed_funds_rate", "federal_funds_rate", "fed_funds",
    "2y_treasury", "10y_treasury", "30y_treasury",
    "10y_2y_spread", "yield_curve",
    "real_gdp", "gdp", "industrial_production",
    "unemployment_rate", "unemployment", "nonfarm_payrolls", "payrolls",
    "initial_claims",
    "m2", "money_supply", "vix", "dollar_index",
    "consumer_sentiment", "housing_starts", "retail_sales",
]


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "Macro indicator: a friendly alias such as 'cpi', 'core_pce', "
        "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', "
        "'real_gdp', 'vix', or a raw FRED series ID such as 'CPIAUCSL'.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 1-year window"
    ] = None,
) -> str:
    """
    Retrieve a macroeconomic indicator time series from FRED (Federal Reserve
    Economic Data): policy rates, Treasury yields, inflation, labor, and growth.
    Returns the series title, units, frequency, the latest value, the change
    over the window, and a recent observation table. Uses the configured
    macro_data vendor.

    Args:
        indicator (str): Friendly alias or raw FRED series ID
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 1-year window

    Returns:
        str: A formatted markdown report of the macro series
    """
    result = _safe_vendor_call("get_macro_indicators", indicator, curr_date, look_back_days)
    if result.startswith("⚠️"):
        result += (
            f"\nValid aliases: {', '.join(sorted(_VALID_ALIASES))}\n"
            f"Or pass a raw FRED series ID (e.g. 'CPIAUCSL', 'DGS10')."
        )
    return result
