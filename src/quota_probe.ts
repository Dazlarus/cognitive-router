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

/** Relative quota-burn weight of a zai model: heavier models drain the 5h
 *  window faster. Name heuristics for now — recalibrate against measured
 *  per-call percentage deltas once telemetry accumulates. */
export function zaiQuotaCostWeight(model: string): number {
  const m = model.toLowerCase();
  if (m.includes("flash")) return 0.5;
  if (m.includes("glm-5.3") || m.includes("glm-5.4") || m.includes("glm-6")) return 1.5;
  return 1.0;
}
