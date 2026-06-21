#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
source ta-venv/bin/activate

echo "🚀 TradingAgents starting..."
tradingagents "$@"
