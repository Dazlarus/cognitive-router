// src/quota_probe.ts - Real Z.AI coding-plan quota telemetry
//
// GET https://api.z.ai/api/monitor/usage/quota/limit — auth via x-api-key
// (NOT Authorization: Bearer; that 401s — verified live 2026-09-05).
// Response shape:
//   { code:200, success:true, data:{ level:"pro", limits:[
//     { type:"TIME_LIMIT",  percentage:1, usage:1000, currentValue:2,
//       remaining:998, nextResetTime:<ms>,
//       usageDetails:[{modelCode:"zread",usage:2},...] },          // monthly tool quota
//     { type:"TOKENS_LIMIT", unit:3, number:5, percentage:10,
//       nextResetTime:<ms> }                                       // 5h token window
//   ]}}
// (TOKENS_LIMIT reports percentage only — no absolute tokens.)
//
// Strategy this feeds (2026-09-05): subscription quota costs the same used
// or unused — burn it. Ride ~90% of the 5h window: flagship models while
// headroom is large, taper to efficient tiers (5.1-flash/5.2) as it fills.
// "Cheap is always more expensive than free."

import { logger } from "./logger.js";

export interface ZaiQuotaSnapshot {
  plan: string | null;
  /** 5-hour token window usage, 0-100, or null before first fetch. */
  fiveHourPercent: number | null;
  /** Epoch ms when the 5h window resets. */
  fiveHourResetAt: number | null;
  monthlyPercent: number | null;
  monthlyUsed: number | null;
  monthlyLimit: number | null;
  monthlyResetAt: number | null;
  monthlyUsageDetails: Array<{ modelCode: string; usage: number }>;
  lastFetchAt: number | null;
  fetchError: string | null;
}

const EMPTY: ZaiQuotaSnapshot = {
  plan: null, fiveHourPercent: null, fiveHourResetAt: null,
  monthlyPercent: null, monthlyUsed: null, monthlyLimit: null,
  monthlyResetAt: null, monthlyUsageDetails: [], lastFetchAt: null, fetchError: null,
};

class ZaiQuotaProbe {
  private snapshot: ZaiQuotaSnapshot = { ...EMPTY };
  private timer: ReturnType<typeof setInterval> | null = null;
  private inflight = false;

  start(): void {
    if (this.timer) return;
    const pollMs = Math.max(30_000, parseInt(process.env.ROUTER_QUOTA_POLL_MS ?? "300000", 10) || 300_000);
    void this.fetch();
    this.timer = setInterval(() => void this.fetch(), pollMs);
    this.timer.unref?.();
    logger.info(`Zai quota probe active — polling every ${Math.round(pollMs / 1000)}s`);
  }

  current(): ZaiQuotaSnapshot {
    return { ...this.snapshot };
  }

  /** 5h window pressure as 0-1, or null when no data yet. */
  pressure(): number | null {
    const p = this.snapshot.fiveHourPercent;
    return p === null ? null : p / 100;
  }

  private async fetch(): Promise<void> {
    if (this.inflight) return;
    const key = process.env.ZAI_API_KEY;
    if (!key) {
      this.snapshot = { ...this.snapshot, fetchError: "ZAI_API_KEY not set" };
      return;
    }
    this.inflight = true;
    const ac = new AbortController();
    const timeout = setTimeout(() => ac.abort(), 10_000);
    try {
      const res = await fetch("https://api.z.ai/api/monitor/usage/quota/limit", {
        method: "GET",
        headers: { "x-api-key": key, Accept: "application/json" },
        signal: ac.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as {
        success?: boolean; code?: number; msg?: string;
        data?: { level?: unknown; limits?: unknown[] };
      };
      if (body.success !== true || body.code !== 200) throw new Error(body.msg ?? "API error");

      const next: ZaiQuotaSnapshot = { ...this.snapshot, lastFetchAt: Date.now(), fetchError: null };
      if (typeof body.data?.level === "string") next.plan = body.data.level;
      for (const lim of Array.isArray(body.data?.limits) ? body.data!.limits! : []) {
        const l = lim as Record<string, unknown>;
        if (l.type === "TOKENS_LIMIT") {
          next.fiveHourPercent = typeof l.percentage === "number" ? l.percentage : null;
          next.fiveHourResetAt = typeof l.nextResetTime === "number" ? l.nextResetTime : null;
        } else if (l.type === "TIME_LIMIT") {
          next.monthlyPercent = typeof l.percentage === "number" ? l.percentage : null;
          // Zai's naming is inverted: `usage` is the limit, `currentValue` is used.
          next.monthlyUsed = typeof l.currentValue === "number" ? l.currentValue : null;
          next.monthlyLimit = typeof l.usage === "number" ? l.usage : null;
          next.monthlyResetAt = typeof l.nextResetTime === "number" ? l.nextResetTime : null;
          next.monthlyUsageDetails = Array.isArray(l.usageDetails)
            ? (l.usageDetails as Array<Record<string, unknown>>)
                .filter((d) => d && typeof d.modelCode === "string")
                .map((d) => ({ modelCode: String(d.modelCode), usage: Number(d.usage) || 0 }))
            : [];
        }
      }
      this.snapshot = next;
      logger.info(
        `Zai quota [${next.plan ?? "?"}]: 5h=${next.fiveHourPercent ?? "?"}%` +
        ` (reset ${next.fiveHourResetAt ? new Date(next.fiveHourResetAt).toISOString() : "?"}),` +
        ` monthly=${next.monthlyPercent ?? "?"}%`,
      );
    } catch (err) {
      // Keep last good snapshot; record the error.
      this.snapshot = { ...this.snapshot, fetchError: err instanceof Error ? err.message : String(err) };
      logger.warn(`Zai quota probe failed: ${this.snapshot.fetchError}`);
    } finally {
      clearTimeout(timeout);
      this.inflight = false;
    }
  }
}

// Module singleton — imported by router, proxy-stream, and stats.
const probe = new ZaiQuotaProbe();
export function getZaiQuotaProbe(): ZaiQuotaProbe {
  return probe;
}

/** Quota-burn weight from DOCUMENTED zai economics (2026-09-05, replacing
 *  name heuristics). Sources: docs.z.ai/guides/overview/pricing (per-token
 *  API prices, output-price primary since reasoning tokens bill as output)
 *  and docs.z.ai/guides/llm/glm-5.3 (points-based plan rules).
 *
 *  Per-token price weight:
 *    GLM-5.3-Flash   $0.25/1M out → 0.06   (promo price, ends Sep 9)
 *    GLM-4.7/4.6/4.5 $2.2/1M out  → 0.5
 *    GLM-5/5.1/5.2/5.3 $4.4/1M out → 1.0   (same price — a heavier 5.3 burn
 *      comes from reasoning-token VOLUME, which the effort policy manages)
 *
 *  Time factors (points-based plan):
 *    off-peak + all-day weekends → 50% points. Off-peak taken as
 *    23:00–09:00 SGT, matching the campaign window definition.
 *    GLM-5.3-Flash campaign, Sep 3–20 2026, 23:00–09:00 SGT: available
 *    quota doubled for non-ZCode agents → ×0.5 (ZCode-only zero-quota
 *    rule doesn't apply to API callers). */
export function zaiQuotaCostWeight(model: string, now = Date.now()): number {
  const m = model.toLowerCase();
  let weight = 1.0;
  if (m.includes("flash")) weight = 0.06;
  else if (m.includes("glm-4.7") || m.includes("glm-4.6") || m.includes("glm-4.5")) weight = 0.5;

  // SGT clock (UTC+8, no DST)
  const sgt = new Date(now + 8 * 3600_000);
  const hour = sgt.getUTCHours();
  const weekend = sgt.getUTCDay() === 0 || sgt.getUTCDay() === 6;
  const offPeak = hour >= 23 || hour < 9;

  if (offPeak || weekend) weight *= 0.5; // documented: 50% points

  const campaignStart = Date.UTC(2026, 8, 3);  // Sep 3 2026
  const campaignEnd = Date.UTC(2026, 8, 20);   // through Sep 20
  if (
    m.includes("glm-5.3") && m.includes("flash") &&
    now >= campaignStart && now < campaignEnd && offPeak
  ) {
    weight *= 0.5; // documented: doubled available quota
  }
  return weight;
}
