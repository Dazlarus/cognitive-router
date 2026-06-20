# Z.AI Usage Multiplier Research

**Last updated:** 2026-06-17
**Source:** https://docs.z.ai/devpack/faq.md

## Subscription Quota System

The GLM Coding Plan uses a **prompt-based quota** (not per-token):
- **Lite**: ~80 prompts / 5 hours
- **Pro**: ~400 prompts / 5 hours
- **Max**: ~1600 prompts / 5 hours
- **Weekly cap**: 7-day rolling cycle from order date
- Each prompt allows ~15-20 model calls internally

## Usage Multipliers (Quota Consumption Rate)

Models consume quota at different rates:

| Model | Peak Hours | Off-Peak | Notes |
|-------|-----------|----------|-------|
| GLM-5.2 | **3×** | **2×** (1× through Sep 2026) | Opus-tier |
| GLM-5-Turbo | **3×** | **2×** (1× through Sep 2026) | Opus-tier |
| GLM-5.1 | ~2×? | ~1×? | (estimated, not in FAQ) |
| GLM-4.7 | 1× | 1× | Sonnet-tier, standard |
| GLM-4.5-Air | 1× | 1× | Haiku-tier, lightweight |
| GLM-4.7-Flash | Free | Free | No quota cost |
| GLM-4.5-Flash | Free | Free | No quota cost |

## Rate Limit Windows

1. **5-hour rolling window**: Prompt count resets every 5 hours
2. **Weekly quota**: 7-day cycle, refreshes from subscription start date
3. **Dynamic adjustment**: Platform adjusts limits based on resource availability
4. **Off-peak** = higher concurrency allowed

## Routing Implications

- **Heavy tasks** (coding, reasoning, analysis): GLM-5.2 despite 2-3× cost
- **Medium tasks** (business, science): GLM-5.1 or GLM-4.7 at 1×
- **Light tasks** (chat, retrieval, summary): Flash models (FREE, no quota)
- **When rate-limited**: escalate backoff tier (5min → 30min → 5hr → weekly)

## TODO

- [ ] Monitor Z.AI announcements for multiplier changes
- [ ] Check if Sep 2026 promo ends (GLM-5.2 1× off-peak → 2×)
- [ ] Research GLM-5.1 multiplier (not documented in FAQ — assumed 1-2×)
- [ ] Detect peak vs off-peak hours automatically (likely China business hours)
