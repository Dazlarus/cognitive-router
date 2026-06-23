# Embedding Model Benchmarking - Usage Guide

## Overview

The Cognitive Router now includes an embedding model benchmarking feature that automatically tests all available Ollama embedding models and recommends the best one based on quality, latency, and resource usage.

## How to Run

```bash
# Trigger benchmark via HTTP
curl -X POST http://127.0.0.1:3456/v1/benchmark/embeddings

# Or via PowerShell
Invoke-WebRequest -Method POST -Uri "http://127.0.0.1:3456/v1/benchmark/embeddings" | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

## What It Tests

1. **Semantic Similarity** (quality): How well the model separates different intent categories (coding, research, creative, conversation, summary, retrieval, science, business, math, analysis)
2. **Latency**: Average embedding generation time (target: <100ms)
3. **VRAM Usage**: Estimated GPU memory consumption

## Expected Output

```json
{
  "timestamp": "2026-06-23T20:00:00.000Z",
  "results": [
    {
      "modelName": "qwen3-embedding:8b",
      "qualityScore": 0.92,
      "latencyMs": 85,
      "vramUsageGb": 4.0,
      "overallScore": 0.94,
      "recommendation": "NEW DEFAULT"
    },
    {
      "modelName": "nomic-embed-text-v2-moe:latest",
      "qualityScore": 0.88,
      "latencyMs": 75,
      "vramUsageGb": 1.8,
      "overallScore": 0.92,
      "recommendation": "GOOD ALTERNATIVE"
    },
    {
      "modelName": "qwen3-embedding:4b",
      "qualityScore": 0.80,
      "latencyMs": 65,
      "vramUsageGb": 2.0,
      "overallScore": 0.90,
      "recommendation": "GOOD ALTERNATIVE"
    },
    {
      "modelName": "nomic-embed-text:latest",
      "qualityScore": 0.75,
      "latencyMs": 70,
      "vramUsageGb": 1.5,
      "overallScore": 0.87,
      "recommendation": "GOOD ALTERNATIVE"
    },
    {
      "modelName": "embeddinggemma:latest",
      "qualityScore": 0.72,
      "latencyMs": 95,
      "vramUsageGb": 2.0,
      "overallScore": 0.84,
      "recommendation": "SKIP"
    }
  ],
  "winner": {
    "modelName": "qwen3-embedding:8b",
    "reason": "excellent semantic separation, fast latency (<100ms)"
  },
  "notes": [
    "Recommended default: qwen3-embedding:8b (score: 0.94)",
    "Good alternatives: nomic-embed-text-v2-moe:latest, qwen3-embedding:4b, nomic-embed-text:latest",
    "Average latency across all models: 78ms"
  ]
}
```

## Scoring Formula

```
overallScore = (qualityScore × 0.6) + (latencyScore × 0.4)

Where latencyScore = 
  1.0 if latency < 100ms
  Linear decay from 1.0 to 0.0 for 100ms–500ms
  0.0 if latency > 500ms
```

## Recommendations

- **NEW DEFAULT**: Score > 0.90 AND latency < 100ms
- **GOOD ALTERNATIVE**: Score > 0.80
- **SKIP**: Score ≤ 0.80

## After Benchmarking

1. Review the winner recommendation
2. Update `src/config.ts` to change the default embedding model:
   ```typescript
   const DEFAULT_EMBEDDING: EmbeddingConfig = {
     provider: "ollama",
     model: "<winner_model_name>",  // e.g., "qwen3-embedding:8b"
     fallback: "gemini",
     fallbackModel: "gemini-embedding-001",
     timeoutMs: 2000,
   };
   ```
3. Rebuild: `npm run build`
4. Restart service: `pwsh restart.ps1`

## Persistence

Benchmark results are persisted to SQLite (`data/cognitive-router.db`) in the `benchmark_results` table for historical comparison.

## Notes

- Benchmark takes ~1-2 minutes (10 prototypes × 8 models × ~8ms per embedding)
- Ollama must be running (`http://localhost:11434`)
- Uses 10 intent prototype texts for semantic similarity testing
- VRAM estimates are heuristics based on model name size