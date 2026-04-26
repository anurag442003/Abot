# Autonomous Trading Agents 📈🤖

Autonomous Trading Agents is a multi-agent AI system where four independent traders — each inspired by a legendary investor — autonomously research financial news, make portfolio decisions, and execute real stock trades on a loop, every hour. Built with the OpenAI Agents SDK, Model Context Protocol (MCP), and Gemini 2.5 Flash.

## 🎯 What It Does

Four AI agents run continuously and concurrently. Each one:

1. Reads its current portfolio (cash balance, holdings, past transactions)
2. Reads its investment strategy
3. Calls a **Researcher sub-agent** to search the web for relevant financial news
4. Analyzes market data and live stock prices
5. Makes buy/sell decisions aligned with its strategy
6. Executes trades against real Polygon.io stock prices
7. Sends a push notification summarizing activity
8. Logs everything to a SQLite database for real-time display in a Gradio dashboard

All four traders run **concurrently** using Python asyncio. The loop repeats every hour (configurable).

> **Note:** The trading is simulated but prices are real. Each trader starts with $10,000 in the simulation database — the money is not real.

## 👤 The Four Traders

Each trader starts with a strategy inspired by their namesake, and has autonomy to **rewrite their own strategy** over time using a tool.
The legendary investors and their strategies listed below:

* **Warren** : Value investor. Seeks undervalued companies with strong fundamentals. Patient, long-term. 
* **George**: Macro trader. Looks for large-scale mispricings driven by economic/geopolitical events. Bold, contrarian. 
* **Ray** : Systematic, principles-based. Risk parity across asset classes. Watches macro cycles. 
* **Cathie**: Disruptive innovation. Focused on crypto ETFs. High volatility tolerance. 

Each trader starts with **$10,000**.

## 🏗️ Architecture Overview

Two completely separate processes share state through a single SQLite file:

* **`trading_floor.py`** — the engine. Runs agents, executes trades, writes to DB.
* **`app.py`** — the Gradio dashboard. Reads and displays data in real time.
## 🔁 Step-by-Step Execution Flow (per trader, per run)
1. `asyncio.gather()` starts all 4 traders at the same time
2. MCP server subprocesses start: accounts, market, push, fetch, Tavily, memory
3. Researcher sub-agent created with web search + memory tools; wrapped as a callable tool
4. Trader agent created with: Researcher tool + account/market/push tools
5. Current portfolio and strategy are read from SQLite
6. `Runner.run(agent, message, max_turns=30)` starts the agent loop:
   - LLM reads strategy + account state
   - Calls Researcher sub-agent → searches Tavily, fetches web pages, stores findings in knowledge graph, returns summary
   - Checks live share prices via `lookup_share_price()`
   - Executes buy/sell trades via `buy_shares()` / `sell_shares()` — SQLite updated immediately
   - Sends push notification
   - Returns final 2–3 sentence appraisal
7. Loop repeats every N minutes (default: 60)
## 🔄 Trade / Rebalance Cycle
Traders alternate between two modes on successive runs:
* **Trade mode** — find new investment opportunities and execute new positions
* **Rebalance mode** — review existing holdings, trim or exit positions that no longer fit the strategy

This keeps each LLM run focused rather than trying to do everything at once.

## 🖥️ Live Dashboard (`app.py`)

A Gradio UI with four columns (one per trader), each showing:

* **Portfolio value** — total value (cash + holdings at current prices) with P&L in green/red
* **Portfolio chart** — Plotly line chart of portfolio value over time
* **Live log** — color-coded agent activity, refreshed every 500ms:
  * White — trace events | Cyan — agent activity | Green — tool calls | Yellow — LLM generation | Red — buy/sell actions
* **Holdings table** — current stock positions
* **Transactions table** — full trade history with rationale

The dashboard and engine are fully independent — you can restart either without affecting the other.


## ⚙️ Tech Stack

* **Agent Framework:** OpenAI Agents SDK
* **LLM:** Gemini 2.5 Flash (via OpenAI-compatible endpoint)
* **Tool Protocol:** MCP (Model Context Protocol) — stdio transport
* **Market Data:** Polygon.io API (free = end-of-day prices; paid = 15-min delayed; realtime = live)
* **Web Search:** Tavily MCP
* **Web Fetch:** mcp-server-fetch
* **Knowledge Graph:** mcp-memory-libsql (LibSQL / SQLite — persists across runs per trader)
* **Push Notifications:** Pushover API
* **Database:** SQLite (Python built-in sqlite3)
* **UI:** Gradio + Plotly
* **Data Modelling:** Pydantic
* **Package Manager:** uv
* **Runtime:** Python 3.12+

## 🔑 Required API Keys

| Service | Purpose | Free Tier? |
|---|---|---|
| Google AI Studio | Gemini 2.5 Flash (the LLM) | Yes |
| Polygon.io | Stock price data | Yes (end-of-day) |
| Tavily | Web search for the Researcher | Yes |
| Pushover | Push notifications (optional) | One-time $5 |

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `RUN_EVERY_N_MINUTES` | 60 | How often the trader loop runs |
| `RUN_EVEN_WHEN_MARKET_IS_CLOSED` | false | If false, skips runs when market is closed |
| `POLYGON_PLAN` | free | `free` / `paid` / `realtime` |

## 🚀 Running the Project

```bash
# Terminal 1 — start the dashboard
uv run app.py
# Terminal 2 — start the trading engine
uv run trading_floor.py
# Reset all traders to $10,000 starting state
uv run reset.py
```

> Start the dashboard first so you can watch activity appear from the beginning.