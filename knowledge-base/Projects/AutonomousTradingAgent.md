# Autonomous Trading Agents

A multi-agent AI system where four independent traders — each inspired by a legendary investor — autonomously research financial news, make portfolio decisions, and execute real stock trades on a loop, every hour. Built with the OpenAI Agents SDK, Model Context Protocol (MCP), and Gemini 2.5 Flash.

---

## What This Project Does

Four AI agents run continuously. Each one:

1. Reads its current portfolio (cash balance, holdings, past transactions)
2. Reads its investment strategy
3. Calls a Researcher sub-agent to search the web for relevant financial news
4. Analyzes market data and stock prices
5. Makes buy/sell decisions aligned with its strategy
6. Executes trades against live stock prices via the Polygon.io API
7. Sends a push notification summarizing activity
8. Logs everything to a database for real-time display in a Gradio dashboard

All four traders run **concurrently** using Python asyncio, and the whole loop repeats every hour (configurable).

---

## The Four Traders

Each trader starts with a strategy inspired by their namesake — but they have autonomy to **rewrite their own strategy** over time using a tool.

| Trader | Inspired By | Strategy |
|--------|-------------|----------|
| **Warren** | Warren Buffett | Value investor. Seeks undervalued companies with strong fundamentals. Patient, long-term. |
| **George** | George Soros | Macro trader. Looks for large-scale mispricings driven by economic/geopolitical events. Bold, contrarian. |
| **Ray** | Ray Dalio | Systematic, principles-based. Risk parity across asset classes. Watches macro cycles and central bank policy. |
| **Cathie** | Cathie Wood | Disruptive innovation. Focused on crypto ETFs. High volatility tolerance for exceptional return potential. |

Each trader starts with **$10,000** and trades from that pool.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              trading_floor.py (engine)               │
│   asyncio.gather → runs all 4 traders concurrently  │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  Trader Agent  │  × 4 (Warren, George, Ray, Cathie)
       │  Gemini 2.5    │  — decision maker, up to 30 turns
       └───┬───────┬────┘
           │       │
    ┌──────▼──┐  ┌─▼──────────────┐
    │Researcher│  │  MCP Servers   │
    │ Agent   │  │  (Trader tools)│
    │(sub-agent│  ├────────────────┤
    │as tool) │  │ accounts_server│ ← buy / sell / balance
    └──┬──────┘  │ market_server  │ ← share prices
       │         │ push_server    │ ← push notifications
  ┌────▼──────┐  └────────────────┘
  │Researcher │
  │MCP Servers│
  ├───────────┤
  │fetch      │ ← fetch web pages
  │tavily     │ ← web search
  │memory     │ ← knowledge graph (persists across runs)
  └─────┬─────┘
        │
        ▼
   accounts.db (SQLite)          memory/[name].db (LibSQL)
   ├── accounts table            └── knowledge graph per trader
   ├── logs table
   └── market table (price cache)
        ▲
        │ reads every 120s / 500ms
┌───────┴────────┐
│   app.py       │
│  Gradio UI     │ ← live dashboard, portfolio charts, logs
└────────────────┘
```

Two completely separate processes share state through a single SQLite file:
- **`trading_floor.py`** — the engine. Runs agents, writes trades and logs.
- **`app.py`** — the dashboard. Reads and displays data in real time.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent framework | OpenAI Agents SDK |
| LLM | Gemini 2.5 Flash (via OpenAI-compatible endpoint) |
| Tool protocol | MCP (Model Context Protocol) — stdio transport |
| Market data | Polygon.io API |
| Web search | Tavily MCP |
| Web fetch | mcp-server-fetch |
| Knowledge graph | mcp-memory-libsql (LibSQL / SQLite) |
| Push notifications | Pushover API |
| Database | SQLite (via Python's built-in sqlite3) |
| UI | Gradio + Plotly |
| Data modelling | Pydantic |
| Package manager | uv |
| Runtime | Python 3.12+ |

---

## Project Structure

```
autonomous-trading-agents/
│
├── trading_floor.py       # Engine — runs all traders on a loop
├── traders.py             # Trader agent class — full agent lifecycle
├── app.py                 # Gradio dashboard UI
│
├── accounts.py            # Account model — buy/sell/portfolio logic
├── accounts_server.py     # MCP server exposing account tools
├── accounts_client.py     # MCP client helper for reading account data
│
├── market.py              # Share price lookup via Polygon API
├── market_server.py       # MCP server exposing market price tool
│
├── mcp_params.py          # All MCP server configurations in one place
├── push_server.py         # MCP server for push notifications (Pushover)
│
├── database.py            # SQLite read/write for accounts, logs, market
├── templates.py           # All prompt templates (system + user messages)
├── tracers.py             # Custom SDK tracer → logs agent activity to DB
├── reset.py               # Reset all traders to starting state + strategy
├── util.py                # CSS, JS, color constants for the UI
│
├── trader_explaination.md # How the trader execution flow works
│
├── .env.example           # Template for required environment variables
├── .gitignore
├── pyproject.toml         # Project dependencies
├── uv.lock                # Locked dependency versions
└── README.md
```

---

## Prerequisites

You need the following installed before setup:

- **Python 3.12+**
- **uv** (Python package manager)
- **Node.js v22+** (required for MCP servers that use `npx`)
- **Git**

> **Windows users:** MCP servers only work on Windows under WSL (Windows Subsystem for Linux). See the [Windows / WSL Setup](#windows--wsl-setup) section below before proceeding.

---

## API Keys You Will Need

| Service | What For | Free Tier? |
|---------|----------|------------|
| [Google AI Studio](https://aistudio.google.com/) | Gemini 2.5 Flash (the LLM) | Yes |
| [Polygon.io](https://polygon.io/) | Stock price data | Yes (end-of-day prices) |
| [Tavily](https://tavily.com/) | Web search for the Researcher | Yes |
| [Pushover](https://pushover.net/) | Push notifications to your phone | One-time $5 |

Pushover is optional — if you skip it, push notifications simply won't fire but everything else works.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/autonomous-trading-agents.git
cd autonomous-trading-agents
```

### Step 2 — Install Python dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and installs everything in an isolated virtual environment. No manual pip installs needed.

### Step 3 — Install Node.js dependencies

The Researcher agent uses two npm-based MCP servers (Tavily and memory). These are installed on first run via `npx` automatically — no manual npm install needed. But you must have Node.js v22+ installed.

Check your version:

```bash
node --version   # should be v22.x.x or later
npx --version
```

If not installed, visit [nodejs.org](https://nodejs.org) and follow the instructions for your OS.

### Step 4 — Set up environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
GEMINI_API_KEY=your_gemini_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
POLYGON_PLAN=free
TAVILY_API_KEY=your_tavily_api_key_here
PUSHOVER_USER=your_pushover_user_key_here
PUSHOVER_TOKEN=your_pushover_app_token_here
RUN_EVERY_N_MINUTES=60
RUN_EVEN_WHEN_MARKET_IS_CLOSED=false
```

**POLYGON_PLAN options:**
- `free` — end-of-day prices only (previous close). Fine for simulation.
- `paid` — 15-minute delayed snapshot prices.
- `realtime` — live trade prices.

**If you skip Pushover**, just leave those two values blank — the push tool will fire but fail silently without breaking the agent.

### Step 5 — Reset traders to starting state

This creates all four trader accounts in the database with $10,000 each and loads their initial strategies:

```bash
uv run reset.py
```

Run this once before the first run, and any time you want to wipe the slate and start fresh.

---

## Running the Project

The project has two components that run **in separate terminals simultaneously**.

### Terminal 1 — Start the dashboard (UI)

```bash
uv run app.py
```

This starts the Gradio web server. Open your browser at `http://localhost:7860` (it opens automatically). You'll see four columns — one per trader — showing portfolio value, a chart, live logs, holdings, and transactions.

### Terminal 2 — Start the trading engine

```bash
uv run trading_floor.py
```

This starts the main loop. All four traders run immediately, then again every N minutes (default: 60). You'll see output in this terminal, and the dashboard will update in real time as trades happen.

> **Note:** Start the dashboard first so you can watch activity appear from the beginning.

---

## How It Works — Step by Step

When `trading_floor.py` runs, here is what happens for each trader:

```
1. trading_floor.py
   └── asyncio.gather() → runs all 4 traders at the same time

2. trader.run()
   └── Sets up tracing (logs all agent activity to DB)
   └── Starts MCP server subprocesses:
       - accounts_server.py  (buy/sell/balance tools)
       - market_server.py    (share price tool)
       - push_server.py      (push notification tool)
       - mcp-server-fetch    (web page fetching)
       - tavily-mcp          (web search)
       - mcp-memory-libsql   (knowledge graph)

3. create_agent()
   └── Researcher sub-agent created (web search + memory tools)
   └── Researcher wrapped as a single callable tool
   └── Trader agent created with: Researcher tool + account/market tools

4. get_account_report()   ← reads current portfolio from SQLite
   read_strategy_resource() ← reads current strategy from SQLite

5. Runner.run(agent, message, max_turns=30)
   └── Agent loop begins (up to 30 LLM turns):
       Turn 1: LLM reads strategy + account state
       Turn 2: Calls Researcher("find news on value stocks")
               └── Researcher runs its own loop:
                   - Searches Tavily for financial news
                   - Fetches relevant web pages
                   - Stores findings in knowledge graph
                   - Returns clean summary to Trader
       Turn 3: LLM reads summary, checks prices
       Turn 4: Calls lookup_share_price("AAPL")
       Turn 5: Calls buy_shares("Warren", "AAPL", 10, "rationale...")
               └── accounts_server.py executes buy
               └── SQLite updated immediately
               └── Log written to DB
               └── Result (updated account JSON) returned to LLM
       ...
       Turn N: Calls push() with brief summary
       Turn N+1: LLM returns final 2-3 sentence appraisal

6. do_trade toggles
   Next run: rebalance existing portfolio instead of finding new trades
   Run after: find new trades again
   (alternates every cycle)
```

---

## The Trade / Rebalance Cycle

Traders alternate between two modes on successive runs:

- **Trade mode** — find new investment opportunities, execute new positions
- **Rebalance mode** — review existing holdings, trim or exit positions that no longer fit the strategy

This mirrors how real fund managers divide their attention, and keeps each LLM run focused rather than trying to do everything at once.


## The Live Dashboard

`app.py` runs a Gradio UI with four columns (one per trader), each showing:

- **Portfolio value** — total value (cash + holdings at current prices) with P&L in green/red
- **Portfolio chart** — line chart of portfolio value over time (Plotly)
- **Live log** — color-coded agent activity, refreshed every 500ms:
  - 🔵 White — trace events
  - 🔵 Cyan — agent activity  
  - 🟢 Green — tool/function calls
  - 🟡 Yellow — LLM generation
  - 🟣 Magenta — responses
  - 🔴 Red — account actions (buy/sell)
- **Holdings table** — current stock positions
- **Transactions table** — full trade history with rationale

The dashboard and engine are **completely independent processes** — the UI just reads SQLite. You can restart either one without affecting the other.

---

## Configuration Options

All configurable via your `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_EVERY_N_MINUTES` | `60` | How often the trader loop runs |
| `RUN_EVEN_WHEN_MARKET_IS_CLOSED` | `false` | If false, skips runs when market is closed (checks Polygon API) |
| `POLYGON_PLAN` | `free` | `free` / `paid` / `realtime` — affects which price tool is used |

---

## Resetting Everything

To wipe all portfolio data and start fresh:

```bash
uv run reset.py
```

This resets all four traders to $10,000 balance with empty holdings and restores their original strategies. It does **not** clear the knowledge graph files in `memory/` — those persist intentionally. To also clear those:

```bash
rm -rf memory/
```

---

## Windows / WSL Setup

> MCP servers do not work natively on Windows. You must use WSL (Windows Subsystem for Linux).

### Part 1 — Install WSL

Open PowerShell and run:

```powershell
wsl --install -d Ubuntu
```

Allow elevated permissions when prompted. Once complete:

```powershell
wsl -d Ubuntu
cd ~
```

Your prompt should show `/home/yourusername` — you are now in the Linux environment.

### Part 2 — Install uv inside WSL

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After it completes, type `exit` to leave WSL, then re-enter with `wsl -d Ubuntu` so the PATH changes take effect. Then:

```bash
cd ~
mkdir projects
cd projects
git clone https://github.com/yourusername/autonomous-trading-agents.git
cd autonomous-trading-agents
uv sync
```

### Part 3 — Install Node.js inside WSL

Node must be installed on the WSL/Linux side, not Windows side:

```bash
# Visit https://nodejs.org for the latest Linux install instructions
# or use nvm (Node Version Manager):
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
# exit and re-enter WSL, then:
nvm install 22
nvm use 22
node --version   # should show v22.x.x
```

### Part 4 — Configure your editor for WSL

In Cursor or VS Code:
1. Install the **WSL** extension (by Anysphere / Microsoft)
2. Press `Ctrl+Shift+P` → search **Remote-WSL: New Window**
3. Open the project folder from within the WSL window

### Troubleshooting — npx MCP servers hanging on WSL

If node-based MCP servers (Tavily, memory) hang or don't start:

```bash
which node
# example output: /home/user/.nvm/versions/node/v22.18.0/bin/node
```

Then in `mcp_params.py`, use the full path for npx-based servers:

```python
{"command": "/home/user/.nvm/versions/node/v22.18.0/bin/npx",
 "args": ["-y", "tavily-mcp"], "env": tavily_env}
```

Or set the PATH explicitly in the env dict:

```python
node_path = "/home/user/.nvm/versions/node/v22.18.0/bin"
{"command": "npx", "args": ["-y", "tavily-mcp"],
 "env": {**tavily_env, "PATH": f"{node_path}:{os.environ['PATH']}"}}
```

---

## Important Notes


**The trading is simulated but prices are real.** Trades execute against real Polygon.io stock prices. The money is not real — each trader starts with $10,000 in the simulation database.

**Stop the engine when done.** Press `Ctrl+C` in the trading_floor.py terminal to stop the loop. The dashboard can keep running independently.


## License

MIT
