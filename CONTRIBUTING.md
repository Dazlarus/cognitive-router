# Contributing to Cognitive Router

Thanks for your interest in improving Cognitive Router! This document covers the basics.

## Development Setup

```bash
git clone https://github.com/dazwritescode/cognitive-router.git
cd cognitive-router
npm install
npm run build
```

For development with auto-rebuild:

```bash
npm run dev   # tsc --watch
```

## Project Architecture

Cognitive Router is a standalone HTTP proxy (not a framework plugin). The core flow:

1. **`server.ts`** — Entry point, starts the HTTP server
2. **`proxy-stream.ts`** — Request handling, retry loop, SSE streaming, candidate list building
3. **`classifier.ts`** — Embeds incoming messages, compares against intent prototypes
4. **`router.ts`** — Scores models on capability/reliability/cost/latency, picks the best
5. **`providers.ts`** — Adapters for each upstream provider (ZAI, OpenRouter, Gemini, Ollama)
6. **`model_registry.ts`** — Model metadata, capability scores, discovery
7. **`cost_tracker.ts`** — Circuit breakers, health tracking, backoff tiers
8. **`db_service.ts`** — SQLite persistence for decisions, call outcomes, model state

## Code Style

- TypeScript strict mode
- ESM modules (`import/export`, not `require`)
- Prefer pure functions in scoring/classification logic
- No external runtime dependencies beyond `better-sqlite3` — stdlib first
- Keep provider adapters in `providers.ts` — don't leak provider-specific logic into the proxy layer

## Pull Requests

1. Fork the repo and create a feature branch (`git checkout -b feat/my-feature`)
2. Make your changes — keep diffs focused
3. Ensure it compiles: `npm run build`
4. If adding a new provider, add tests for it
5. Write a clear PR description explaining what and why

## Reporting Issues

Include:
- What you expected vs what happened
- Router logs (redact API keys)
- Output of `curl http://127.0.0.1:3456/stats`
- Your `.env` (redacted — just show which keys are set)

## Adding a New Provider

1. Add an adapter object to `src/providers.ts` following the existing pattern
2. Add the provider name to the `Provider` type
3. Add env var for the API key in `.env.example`
4. Add capability scores in `model_registry.ts` (or let auto-discovery handle it)
5. Add the provider to the default `ROUTER_PRIORITY` in `.env.example`

## License

By contributing, you agree your contributions are licensed under the MIT license.
