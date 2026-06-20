# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Install**: `pip install -e ".[dev]"` (editable install with dev deps)
- **Run CLI**: `tradingagents` or `python -m cli.main`
- **Run Python API**: `python main.py`
- **Lint**: `ruff check .`
- **Test all**: `pytest`
- **Test file**: `pytest tests/<file>.py`
- **Test single**: `pytest tests/<file>.py::test_name -k "substring"` (use `-k` for selective runs)
- **Test markers**: `pytest -m unit`, `pytest -m integration`, `pytest -m smoke`
- **Docker**: `docker compose run --rm tradingagents`
- **Env setup**: `cp .env.example .env` then fill in API keys
- **Checkpoints**: `tradingagents analyze --checkpoint`, `tradingagents analyze --clear-checkpoints`

## Architecture Overview

Multi-agent LLM trading framework using **LangGraph** as the orchestration layer. The workflow is a directed graph: an `AgentState` TypedDict flows through sequential nodes, with **loop-back edges for debate rounds** and **tool-call routing** within each analyst.

### Pipeline Steps
1. **Analyst Team** (parallel-capable): Market, Sentiment, News, Fundamentals analysts each produce a report using yfinance / Alpha Vantage tools. Each analyst runs in a loop (agent → tools → agent) until no more tool calls are made, then a "Msg Clear" node strips tool messages before passing to the next analyst.
2. **Research Team**: Bull & Bear researchers debate — the graph alternates between them (routing controlled by `investment_debate_state.count`) up to `max_debate_rounds`. After debate, the Research Manager produces a structured `ResearchPlan`.
3. **Trader**: Converts the research plan + analyst reports into a concrete `TraderProposal`.
4. **Risk Management**: Aggressive, Neutral, and Conservative debaters discuss risk in a 3-way rotation (Aggressive → Conservative → Neutral → Aggressive) controlled by `risk_debate_state.count`.
5. **Portfolio Manager**: Makes final `PortfolioDecision` — rating is Buy/Overweight/Hold/Underweight/Sell.

### Graph Structure

```
START → Analyst 1 → (tools loop) → Msg Clear → Analyst 2 → ... → Bull Researcher
   ↕ (debate)                                                  ↕ (debate)
Bear Researcher                                            Research Manager
     ↓
  Trader → Aggressive → (3-way risk debate loop) → Portfolio Manager → END
```

### Key Files

| Path | Role |
|---|---|
| `tradingagents/graph/trading_graph.py` | Main entry point (`TradingAgentsGraph` class: init LLMs, build graph, propagate) |
| `tradingagents/graph/setup.py` | `GraphSetup` — builds the `StateGraph` topology, registers all nodes/edges |
| `tradingagents/graph/conditional_logic.py` | `ConditionalLogic` — edge routing (tool loops, debate rounds, risk rounds) |
| `tradingagents/graph/propagation.py` | `Propagator` — initial state creation, graph invocation args |
| `tradingagents/graph/analyst_execution.py` | `AnalystExecutionPlan`, `AnalystNodeSpec`, `AnalystWallTimeTracker` |
| `tradingagents/graph/reflection.py` | `Reflector` — deferred trade reflection (Phase B) |
| `tradingagents/graph/signal_processing.py` | Extracts 5-tier rating from PM decision |
| `tradingagents/graph/checkpointer.py` | Per-ticker SQLite checkpoint for resume |
| `tradingagents/default_config.py` | `DEFAULT_CONFIG` + `TRADINGAGENTS_*` env-var override system |
| `tradingagents/agents/schemas.py` | Pydantic: `ResearchPlan`, `TraderProposal`, `PortfolioDecision`, `SentimentReport` |
| `tradingagents/agents/utils/structured.py` | `bind_structured` + `invoke_structured_or_freetext` (fallback chain) |
| `tradingagents/agents/utils/agent_states.py` | `AgentState`, `InvestDebateState`, `RiskDebateState` TypedDicts |
| `tradingagents/agents/utils/agent_utils.py` | Tool functions bound to agents (`get_stock_data`, `get_news`, etc.) |
| `tradingagents/agents/utils/rating.py` | Shared 5-tier rating vocabulary + heuristic `parse_rating()` |
| `tradingagents/agents/utils/memory.py` | `TradingMemoryLog` — append-only markdown decision log with Phase B reflection |
| `tradingagents/dataflows/interface.py` | Vendor routing layer (`route_to_vendor` with ordered fallback chains) |
| `tradingagents/dataflows/errors.py` | `VendorError` hierarchy (`NoMarketDataError`, `VendorRateLimitError`, `VendorNotConfiguredError`) |
| `tradingagents/llm_clients/factory.py` | LLM client factory (lazy imports, provider routing) |
| `tradingagents/llm_clients/capabilities.py` | Per-model capability table (tool_choice, json_mode, reasoning split) |
| `tradingagents/llm_clients/model_catalog.py` | Shared model lists for CLI dropdowns |
| `cli/main.py` | Interactive CLI using Rich + Questionary + Typer |

### Key Patterns

- **Agent factory functions**: Every agent is a `create_*` function returning a LangGraph node function. Providers are pluggable — the factory binds tools and prompts but the `llm` parameter is provider-agnostic.
- **Structured output with fallback**: Research Manager, Trader, and Portfolio Manager use `with_structured_output` (Pydantic schema). The `invoke_structured_or_freetext` helper catches failures and retries as free-text, so no single malformed JSON blocks the pipeline. The Sentiment Analyst also uses structured output for its `SentimentReport`.
- **Debate loop mechanics**: The graph routes between Bull/Bear researchers (or Agressive/Neutral/Conservative risk analysts) by reading `state["investment_debate_state"]["count"]` / `state["risk_debate_state"]["count"]`. After `max_debate_rounds` alternations, the conditional edge routes to the manager node instead of the other debater.
- **Config-driven with env overrides**: `DEFAULT_CONFIG` is the single source of truth. `TRADINGAGENTS_*` env vars auto-override via `_apply_env_overrides()` with type coercion (bool, int, float based on the default value's type). Config keys control everything: provider, models, debate rounds, data vendors, news limits, language, temperature, checkpoints, benchmarks.
- **Data vendor routing**: Tools are dispatched through `route_to_vendor(method, *args)` which looks up `VENDOR_METHODS[method][vendor]`. Vendor selection is per-category (`data_vendors`) with per-tool overrides (`tool_vendors`). Falls back through ordered vendor chains on rate limits. The `VendorError` hierarchy ensures routing reacts by error *type* (rate-limit → skip, no data → try next, unconfigured → raise).
- **LLM capabilities table**: Model-specific quirks (DeepSeek no `tool_choice`, MiniMax `reasoning_split`) are declared declaratively in `capabilities.py` and matched by exact ID → regex pattern → default. Clients consult `get_capabilities(model_name)` instead of hardcoding model-name if-ladders.
- **LLM client factory**: Uses lazy imports per provider so importing the factory doesn't pull in heavy SDKs or fail on missing API keys. Native APIs (Anthropic, Google, Azure, Bedrock) matched first; everything else routes through `OpenAIClient` via a provider registry.
- **Memory log with Phase B reflection**: Append-only markdown at `~/.tradingagents/memory/trading_memory.md`. Phase A: stores pending entries with rating tag. On next same-ticker run, Phase B: fetches realized returns (raw + alpha vs regional benchmark), generates LLM reflection, writes outcome. Past decisions (same-ticker + cross-ticker lessons) are injected into the Portfolio Manager prompt.
- **Checkpoint/resume**: Opt-in LangGraph `SqliteSaver`. Recompiles the graph with `checkpointer=saver` on `propagate()` when `checkpoint_enabled=True`. Cleared on successful completion. Thread ID ties to ticker+date so same inputs resume, different dates start fresh.
- **Instrument identity resolution**: Ticker identity is resolved deterministically via `resolve_instrument_identity()` before any agent runs. The result is injected as `instrument_context` so every agent anchors to the real company instead of hallucinating one from price data.

### LLM Provider Support
Providers: `openai`, `anthropic`, `google`, `xai`, `deepseek`, `qwen`/`qwen-cn`, `glm`/`glm-cn`, `minimax`/`minimax-cn`, `openrouter`, `ollama`, `azure`, `bedrock`, `openai_compatible`. Anthropic, Google, Azure, and Bedrock each have their own client class; all others use `OpenAIClient`.

### Data Vendor System
Three vendor categories: `yfinance` (default), `alpha_vantage` (fallback), `fred` (macro), `polymarket` (prediction markets). Configured under `config["data_vendors"]` (category-level) or `config["tool_vendors"]` (tool-level). Falls back between vendors on `VendorRateLimitError`.

### Market Support
Works with any market Yahoo Finance covers: US (no suffix), Hong Kong (`.HK`), Tokyo (`.T`), London (`.L`), India (`.NS`/`.BO`), Canada (`.TO`), Australia (`.AX`), China A-shares (`.SS`/`.SZ`), crypto (`-USD`). Alpha benchmark auto-resolved per exchange via `benchmark_map`.

### Test Structure
~45 test files in `tests/` covering: provider compatibility, data vendor routing, symbol normalization, environment overrides, checkpoint resume, structured agent output, edge cases (stale data, no data, rate limits, lookahead bias), and per-provider API quirks. Run with `pytest -m unit` for fast isolated tests (the default).

### Project Structure
```
tradingagents/
  agents/
    analysts/          # Market, Sentiment, News, Fundamentals
    researchers/       # Bull, Bear
    risk_mgmt/         # Aggressive, Neutral, Conservative
    managers/          # Research Manager, Portfolio Manager
    trader/            # Trader
    utils/             # Agent states, tool bindings, memory, structured output, rating
  dataflows/           # Vendor implementations + routing layer (yfinance, Alpha Vantage, FRED, Polymarket)
  graph/               # LangGraph setup, execution plan, conditional routing, checkpoint, reflection
  llm_clients/         # Provider wrappers, factory, capabilities table, model catalog
cli/                   # Rich-based interactive CLI
tests/                 # Pytest suite (~45 files)
```
