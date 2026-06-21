# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests (quiet mode)
pytest -q

# Run a single test file
pytest tests/test_some_feature.py -q

# Run a single test
pytest tests/test_some_feature.py::test_name -q

# Lint
ruff check .

# Run the CLI interactively
tradingagents           # or: python -m cli.main

# Run a standalone analysis programmatically
python main.py

# Docker
docker compose run --rm tradingagents
docker compose --profile ollama run --rm tradingagents-ollama
```

## Architecture

### Pipeline flow

The trading pipeline is a **LangGraph state machine** that runs agents sequentially:

```
Analyst Team (selected: market → sentiment → news → fundamentals)
  → each analyst has a tool-calling loop (conditional edges back to tools)
  → Bull Researcher ↔ Bear Researcher (N debate rounds)
  → Research Manager (structured output → ResearchPlan)
  → Trader (structured output → TraderProposal)
  → Aggressive ↔ Conservative ↔ Neutral risk debators (N rounds)
  → Portfolio Manager (structured output → PortfolioDecision)
```

The graph is built in `tradingagents/graph/setup.py` (`GraphSetup.setup_graph()`), invoked via `tradingagents/graph/trading_graph.py` (`TradingAgentsGraph`), and streamed live to the CLI. The CLI (`cli/main.py`) uses Rich's `Live` display to show agent progress, tool calls, and accumulating reports.

### Directory layout

- **`tradingagents/graph/`** — LangGraph workflow orchestration (the "brain"). Entry in `trading_graph.py`, graph wiring in `setup.py`, routing logic in `conditional_logic.py`, checkpoint/resume in `checkpointer.py`, post-trade reflection in `reflection.py`, analyst execution plans in `analyst_execution.py`.

- **`tradingagents/agents/`** — Agent implementations (the "roles"). Each subdirectory has a `create_<role>(llm)` factory:
  - `analysts/` — Market, Sentiment, News, Fundamentals analysts
  - `researchers/` — Bull and Bear researchers (structured debate)
  - `risk_mgmt/` — Aggressive, Conservative, Neutral debators
  - `managers/` — Research Manager, Portfolio Manager
  - `trader/` — Trader agent
  - `utils/` — Shared tool modules, `agent_states.py` (TypedDict state), `memory.py` (decision log), `rating.py` (5-tier rating parser), `structured.py` (structured output with graceful fallback)

- **`tradingagents/agents/schemas.py`** — Pydantic models for structured output: `SentimentReport`, `ResearchPlan`, `TraderProposal`, `PortfolioDecision`. Each has a `render_*()` function converting back to the markdown shape downstream expects.

- **`tradingagents/dataflows/`** — Data vendor abstraction layer. `interface.py` is the dispatcher: `route_to_vendor(method, *args)` reads config, falls back through vendor chains, and surfaces `NoMarketDataError` / `VendorRateLimitError` / `VendorNotConfiguredError` from `errors.py`. Vendors: `y_finance.py`, `alpha_vantage*.py`, `fred.py`, `polymarket.py`, `reddit.py`, `stocktwits.py`.

- **`tradingagents/llm_clients/`** — LLM provider abstraction. `factory.py` dispatches to native clients (Anthropic, Google, Azure, Bedrock) or the OpenAI-compatible path. `capabilities.py` has a per-model table of which structured-output method and tool-choice quirks each model supports. `openai_client.py` extends `ChatOpenAI` with `NormalizedChatOpenAI` (content normalization + capability-aware `with_structured_output`) and `DeepSeekChatOpenAI` (thinking-mode roundtrip).

- **`cli/`** — Typer CLI. `main.py` has the interactive questionnaire, Rich `Live` layout, agent status tracking (`MessageBuffer`), report saving, and post-analysis display.

- **`tests/`** — Pytest suite (46 files). `conftest.py` auto-setup: dummy API keys for all providers, config isolation per test (prevents dataflows config leaks), a `mock_llm_client` fixture.

### Two-tier LLM model system

Every pipeline run uses two models:
- **`deep_think_llm`** — Used by Research Manager and Portfolio Manager (complex reasoning tasks). Default: `gpt-5.5`
- **`quick_think_llm`** — Used by all other agents (analysts, debaters, trader). Default: `gpt-5.4-mini`

Set via config keys or `TRADINGAGENTS_DEEP_THINK_LLM` / `TRADINGAGENTS_QUICK_THINK_LLM` env vars.

### Configuration

`tradingagents/default_config.py` is the single source of truth. Keys can be overridden via:
1. `TRADINGAGENTS_*` environment variables (automatically coerced to the type of the default)
2. `.env` file (loaded by python-dotenv)
3. Programmatic `DEFAULT_CONFIG.copy()` + assignment

Notable config groups: LLM provider/models, debate rounds, data vendor selection per category (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`, `macro_data`, `prediction_markets`), per-market benchmark index map, output language, checkpoint/resume toggle, sampling temperature.

### Persistence

- **Decision log** (`~/.tradingagents/memory/trading_memory.md`) — Append-only markdown. Each propagate() stores a pending entry; next same-ticker run fetches returns, generates reflection, updates the entry.
- **Checkpoints** (`~/.tradingagents/cache/checkpoints/`) — Per-ticker SQLite databases via LangGraph's `SqliteSaver`. Opt-in via `config["checkpoint_enabled"]` or `--checkpoint`.
- **Run results** (`~/.tradingagents/logs/`) — JSON state dumps per ticker/date, plus CLI report saving.

### Testing patterns

- Tests use `conftest.py` auto-fixtures: dummy API keys (prevents CI hangs), fresh config per test (prevents pollution from `dataflows.config` global state).
- Many tests patch or mock `create_llm_client` with `mock_llm_client` fixture or manual mocks.
- `dataflows/` tests often use monkeypatching for `yfinance`, `requests`, etc.
- `tests/test_memory_log.py` is the largest test file (40 KB) — exercises the decision log.
- Add `pytest-subtests` for multi-case tests of the same behavior.

### Key patterns to know

- **Structured output with fallback**: `structured.py` has `bind_structured()` (wraps LLM with `with_structured_output`, returns `None` if unsupported) and `invoke_structured_or_freetext()` (tries structured, falls back to free-text on failure). Used by Sentiment Analyst, Research Manager, Trader, Portfolio Manager.
- **Vendor routing**: `interface.py`'s `route_to_vendor()` chains vendors in configured order, catching typed errors: `VendorRateLimitError` → skip, `NoMarketDataError` → report "NO_DATA_AVAILABLE", unhandled → raise. The chain is the user's explicit config — no silent fallback to unconfigured vendors.
- **Capability dispatch**: `llm_clients/capabilities.py` maps model IDs to quirks via exact match + regex patterns. Controls which structured-output method is used and whether `tool_choice` is suppressed (DeepSeek, MiniMax reasoning models reject it).
- **Git hook**: Pre-commit linting via `ruff check` on the full repo. Run before pushing to avoid CI failure.
