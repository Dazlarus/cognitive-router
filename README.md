# Cognitive Router

**Intelligent model routing for multi-provider LLM setups.**

Cognitive Router is a self-healing HTTP proxy that sits between your AI client (OpenClaw, LibreChat, or any OpenAI-compatible consumer) and your LLM providers. It classifies each request by intent, scores available models against the task, and transparently handles failover — so the caller never sees a failure unless every provider is down.

```
Client ──► POST /v1/chat/completions ──► Cognitive Router
                                              │
                                    1. Classify intent
                                    2. Score all models
                                    3. Build candidate list
                                    4. Try providers in order
                                    5. Return clean response
                                              │
                            ┌─────────────────┼─────────────────┐
                            ▼                 ▼                 ▼
                         ZAI             OpenRouter          Ollama
                      (GLM-5.2)        (free models)      (local GPU)
                            │                 │                 │
                            └────── failure? try next ─────────┘
```

## Why?

Most multi-model setups use a static fallback list — if provider A fails, try B, then C. This works, but it's dumb:

- **No intent awareness** — a coding task goes to the same model as casual chat
- **No health tracking** — a rate-limited provider gets retried on every request
- **No cost optimization** — you pay for premium models when free ones would suffice
- **No circuit breaking** — one bad provider can stall the entire pipeline

Cognitive Router fixes all of this by making routing decisions **per request**, not per session.

## Features

- **🧠 Intent classification** — Embedding-based classification across 10 categories (coding, research, creative, math, science, analysis, etc.)
- **⚖️ Multi-factor scoring** — Capability (50%), reliability (25%), cost (15%), latency (10%)
- **🔄 Self-healing failover** — If the top pick fails, automatically tries the next provider
- **🔌 Circuit breakers** — Per-provider AND per-model health tracking with exponential backoff
- **📡 OpenAI-compatible** — Drop-in replacement for any client that speaks OpenAI API
- **🌊 Streaming support** — Full SSE streaming passthrough
- **🔗 Embeddings endpoint** — `/v1/embeddings` routes to Ollama with Gemini fallback
- **📊 Stats dashboard** — Live provider health, model registry, and routing decisions at `/stats`
- **🪟 Windows Service** — Auto-starts with Windows via NSSM

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) 20+
- One or more LLM provider API keys (ZAI, OpenRouter, Gemini)
- [Optional] [Ollama](https://ollama.ai/) for local model support

### Build

```bash
git clone https://github.com/dazwritescode/cognitive-router.git
cd cognitive-router
npm install
npm run build
```

### Configure

Copy `.env.example` to `.env` and add your API keys:

```env
ZAI_API_KEY=your-zai-key
OPENROUTER_API_KEY=your-openrouter-key
GEMINI_API_KEY=your-gemini-key
# Optional: customize routing behavior
ROUTER_PORT=3456
ROUTER_PRIORITY=zai,openrouter,gemini,ollama
```

### Run

```bash
npm start
# or: npx tsx src/server.ts (dev mode)
```

### Test

```bash
curl http://127.0.0.1:3456/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "CognitiveRouter:latest",
    "messages": [{"role": "user", "content": "Say hello in 5 words"}]
  }'
```

Check health:

```bash
curl http://127.0.0.1:3456/health
curl http://127.0.0.1:3456/stats
```

## How Routing Works

### Step 1: Intent Classification

The router embeds the latest message and compares it against prototype embeddings for 10 intent categories:

| Category | Example |
|----------|---------|
| `coding` | "Write a function that handles database queries" |
| `research` | "Compare the performance of Rust vs Go" |
| `creative` | "Write a short story about a Mars colony" |
| `math` | "Solve this system of differential equations" |
| `science` | "Explain the mechanism of CRISPR gene editing" |
| `analysis` | "Evaluate the risks of this architecture decision" |
| `business` | "Create a go-to-market strategy" |
| `summary` | "Summarize this article" |
| `retrieval` | "What is the capital of Brazil" |
| `conversation` | "Hey, how are you doing today" |

If embeddings are unavailable, it falls back to keyword matching with lower confidence.

### Step 2: Model Scoring

Each available model is scored on four axes:

| Factor | Weight | What it measures |
|--------|--------|------------------|
| **Capability** | 50% | How well does this model match the detected intent? |
| **Reliability** | 25% | Circuit breaker state, recent failure rate |
| **Cost** | 15% | Free vs paid, usage multipliers |
| **Latency** | 10% | Historical response time |

The highest-scoring model is the top candidate. If two models are within 0.03 of each other, the cheaper one wins.

### Step 3: Candidate List & Retry Loop

The router builds an ordered fallback list (router's top pick → best model per provider in priority order → Ollama as last resort), then tries each one sequentially:

1. Check circuit breaker (skip if open)
2. Send **non-streaming** request to provider
3. On success → stream/return response to caller as `CognitiveRouter:latest`
4. On failure → record error type, increment strikes, try next candidate
5. If all candidates exhausted → return 503 `all_providers_exhausted`

The router manages SSE streaming to the client internally — providers are always called in non-streaming mode for simpler error handling.

### Tool Requests

Requests with `tools` or `functions` are handled differently:
- Only tool-capable providers are considered
- Provider priority is enforced strictly (no scoring override)
- Specific tool-capable fallback models are used per provider

## API Endpoints

### `POST /v1/chat/completions`

Standard OpenAI chat completions. Supports `stream: true` for SSE.

**Request:**
```json
{
  "model": "CognitiveRouter:latest",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "temperature": 0.7
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-108",
  "object": "chat.completion",
  "model": "CognitiveRouter:latest",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello there!"},
      "finish_reason": "stop"
    }
  ]
}
```

### `POST /v1/embeddings`

Embeddings endpoint. Routes to Ollama `nomic-embed-text` with Gemini fallback.

### `GET /v1/models`

Returns router aliases: `CognitiveRouter:latest`, `CogRouter:latest` (legacy), `Embeddings:latest`.

### `GET /health`

```json
{"status": "ok", "initialized": true}
```

### `GET /stats`

Live provider health, model registry, per-intent capability scores, circuit breaker state, and fallback configuration.

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_PORT` | `3456` | Proxy HTTP port |
| `ROUTER_LOG_LEVEL` | `info` | Log level (`debug`/`info`/`warn`/`error`) |
| `ROUTER_DB_PATH` | `data/cognitive-router.db` | SQLite database path |
| `ROUTER_PRIORITY` | `zai,openrouter,gemini,ollama` | Provider priority order |
| `ROUTER_VRAM_LIMIT` | `11` | Max VRAM (GB) for local models |
| `ROUTER_REQUEST_TIMEOUT_MS` | `55000` | Total deadline per request |
| `ROUTER_MAX_ATTEMPTS_PER_PROVIDER` | `1` | Models to try per provider before moving on |
| `ZAI_API_KEY` | — | Z.AI API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |

See [`.env.example`](.env.example) for the full list.

## Circuit Breakers

When a provider fails, the router tracks the failure and escalates backoff:

| Tier | Failures | Cooldown |
|------|----------|----------|
| 0 | 0 | Healthy |
| 1 | 1-2 | 30 seconds |
| 2 | 3-5 | 1 minute |
| 3 | 6-9 | 5 minutes |
| 4 | 10+ | 30 minutes |

ZAI circuit breaking is **per-model** — a rate limit on `glm-5.2` doesn't block `glm-4.7-flash`.

## Install as Windows Service

Requires [NSSM](https://nssm.cc/):

```powershell
.\install-service.ps1
sc start CognitiveRouter
```

Logs: `C:\Logs\CognitiveRouter\stdout.log`

## Integration Examples

### OpenClaw

Add as a provider in `openclaw.json`:

```json
{
  "providers": {
    "cognitive-router": {
      "baseUrl": "http://127.0.0.1:3456/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "models": [
        {
          "id": "CognitiveRouter:latest",
          "name": "Cognitive Router",
          "contextWindow": 200000,
          "input": ["text"],
          "reasoning": true
        }
      ]
    }
  }
}
```

### Any OpenAI-compatible client

Point your client at `http://127.0.0.1:3456/v1` with any API key — the router ignores it.

## Project Structure

```
cognitive-router/
├── src/
│   ├── server.ts          # Entry point — starts the proxy
│   ├── proxy-stream.ts    # HTTP server, request handling, retry loop
│   ├── classifier.ts      # Embedding-based intent classification
│   ├── router.ts          # Multi-factor scoring engine
│   ├── providers.ts       # Provider adapters (ZAI, OpenRouter, Gemini, Ollama)
│   ├── model_registry.ts  # Model discovery, capability scores
│   ├── cost_tracker.ts    # Circuit breakers, health tracking
│   ├── model_policy.ts    # Generation/embedding/tool classification
│   ├── db_service.ts      # SQLite persistence
│   ├── stats.ts           # /stats endpoint builder
│   ├── config.ts          # Config loading from env
│   ├── env.ts             # .env loading
│   └── logger.ts          # Leveled logger
├── scripts/
├── tests/
├── .env.example
└── package.json
```

## Supported Providers

| Provider | Type | Models |
|----------|------|--------|
| [Z.AI](https://z.ai/) | Remote | GLM-5.2, GLM-5.1, GLM-4.7-flash |
| [OpenRouter](https://openrouter.ai/) | Remote | Qwen3-coder (free), Nemotron (free), Owl-alpha |
| [Google Gemini](https://ai.google.dev/) | Remote | Gemini 3.5 Flash, Gemini 2.5 Flash |
| [Ollama](https://ollama.ai/) | Local | Any local model (VRAM-filtered) |

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
