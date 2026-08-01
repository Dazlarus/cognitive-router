// src/budget_tracker.ts — Predictive budget tracking + burn rate + auto-downgrade
//
// Computes rolling average spend per hour from the provider_spend table,
// projects remaining budget vs remaining period time, and provides
// auto-downgrade signals when budget exhaustion is projected too early.
// Also provides per-model cost efficiency scoring and anomaly detection.

import { logger } from "./logger.js";
import type { DBService } from "./db_service.js";
import type { CostTracker } from "./cost_tracker.js";
import type { ModelRegistry, ModelCapability } from "./model_registry.js";

/** Result of a budget projection computation. */
export interface BudgetProjection {
  /** Total budget for the period (daily or monthly). */
  budgetUsd: number;
  /** Spend so far in the current period. */
  spentUsd: number;
  /** Remaining budget. */
  remainingUsd: number;
  /** Fraction of budget consumed (0–1+). */
  consumedFraction: number;
  /** Rolling average spend per hour (USD/hr) based on recent data. */
  burnRatePerHour: number;
  /** Estimated hours until budget exhaustion at current burn rate. */
  hoursToExhaustion: number | null;
  /** Estimated timestamp of exhaustion (ISO string) or null if infinite. */
  projectedExhaustionAt: string | null;
  /** Fraction of the period elapsed (0–1). */
  periodElapsedFraction: number;
  /** Whether auto-downgrade should be active. */
  autoDowngradeActive: boolean;
  /** Reason auto-downgrade is active (or why it's not). */
  downgradeReason: string;
  /** How much cost weight is boosted when auto-downgrade is active. */
  costWeightBoost: number;
}

/** Cost efficiency result for a single model. */
export interface CostEfficiency {
  provider: string;
  model: string;
  /** Capability score (0–1) for the given intent. */
  capabilityScore: number;
  /** Cost per 1K input tokens (USD). */
  costPer1kInput: number;
  /** Cost per 1K output tokens (USD). */
  costPer1kOutput: number;
  /** Combined cost index relative to the cheapest model (1.0 = cheapest). */
  costRatio: number;
  /** Efficiency score: capability / costRatio. Higher = better value. */
  efficiency: number;
  /** Whether this model is recommended based on cost efficiency. */
  recommended: boolean;
}

/** Anomaly detection result. */
export interface AnomalyDetection {
  /** Whether this request is anomalous. */
  isAnomalous: boolean;
  /** Estimated tokens for the current request. */
  currentTokens: number;
  /** Rolling average tokens per request. */
  averageTokens: number;
  /** Ratio of current to average (>2 = anomalous). */
  ratio: number;
  /** Human-readable reason. */
  reason: string;
}

/** Hours of spend history to use for burn rate calculation. */
const BURN_RATE_WINDOW_HOURS = 24;

/** When >40% of period remains and projected exhaustion hits, trigger downgrade. */
const DOWNGRADE_PERIOD_REMAINING_THRESHOLD = 0.40;

/** Multiplier applied to cost weight when auto-downgrade is active. */
const DOWNGRADE_COST_WEIGHT_MULTIPLIER = 2.0;

/** Paid provider penalty (subtracted from cost score) during downgrade. */
const DOWNGRADE_PAID_PENALTY = 0.20;

/** Minimum spend data points required for a reliable burn rate. */
const MIN_DATA_POINTS = 2;

/** Rolling window for average token count (number of recent requests). */
const TOKEN_AVERAGE_WINDOW = 50;

export class BudgetTracker {
  private recentTokenCounts: number[] = [];

  constructor(
    private db: DBService,
    private costTracker: CostTracker,
  ) {}

  /**
   * Compute a full budget projection for a given period.
   * @param period "daily" or "monthly"
   * @param budgetUsd The budget limit for the period.
   */
  projectBudget(period: "daily" | "monthly", budgetUsd: number): BudgetProjection {
    const spentUsd = this.db.getTotalSpend(period);
    const remainingUsd = Math.max(0, budgetUsd - spentUsd);
    const consumedFraction = budgetUsd > 0 ? spentUsd / budgetUsd : 0;

    // Compute rolling burn rate
    const burnRate = this.computeBurnRate();

    // Compute period elapsed fraction
    const periodElapsedFraction = this.computePeriodElapsedFraction(period);

    // Project exhaustion
    let hoursToExhaustion: number | null = null;
    let projectedExhaustionAt: string | null = null;

    if (burnRate > 0) {
      hoursToExhaustion = remainingUsd / burnRate;
      if (hoursToExhaustion > 0 && Number.isFinite(hoursToExhaustion)) {
        projectedExhaustionAt = new Date(
          Date.now() + hoursToExhaustion * 3_600_000,
        ).toISOString();
      }
    }

    // Determine if auto-downgrade should be active
    const periodRemainingFraction = 1 - periodElapsedFraction;
    let autoDowngradeActive = false;
    let downgradeReason = "Budget healthy";

    if (consumedFraction >= 1.0) {
      autoDowngradeActive = true;
      downgradeReason = `Budget exhausted (${(consumedFraction * 100).toFixed(1)}% consumed)`;
    } else if (
      hoursToExhaustion !== null &&
      periodRemainingFraction > DOWNGRADE_PERIOD_REMAINING_THRESHOLD &&
      burnRate > 0
    ) {
      // Projected to run out before 60% of period elapses (i.e., >40% remaining when it runs out)
      autoDowngradeActive = true;
      downgradeReason =
        `Projected exhaustion in ${hoursToExhaustion.toFixed(1)}h ` +
        `with ${(periodRemainingFraction * 100).toFixed(0)}% of ${period} period remaining`;
    } else if (consumedFraction > 0.75 && periodRemainingFraction > 0.3) {
      autoDowngradeActive = true;
      downgradeReason =
        `Budget ${(consumedFraction * 100).toFixed(0)}% consumed ` +
        `with ${(periodRemainingFraction * 100).toFixed(0)}% of period remaining`;
    }

    const costWeightBoost = autoDowngradeActive ? DOWNGRADE_COST_WEIGHT_MULTIPLIER : 1.0;

    return {
      budgetUsd,
      spentUsd,
      remainingUsd,
      consumedFraction,
      burnRatePerHour: burnRate,
      hoursToExhaustion,
      projectedExhaustionAt,
      periodElapsedFraction,
      autoDowngradeActive,
      downgradeReason,
      costWeightBoost,
    };
  }

  /**
   * Compute rolling average spend per hour from the provider_spend table.
   * Uses the last BURN_RATE_WINDOW_HOURS hours of data.
   */
  private computeBurnRate(): number {
    const hourlyData = this.db.getHourlySpend(BURN_RATE_WINDOW_HOURS);

    if (hourlyData.length < MIN_DATA_POINTS) {
      // Not enough data — compute from total daily spend
      const totalDaily = this.db.getTotalSpend("daily");
      if (totalDaily > 0) {
        // Assume spend happened over the elapsed portion of the day
        const hoursElapsed = this.computePeriodElapsedFraction("daily") * 24;
        if (hoursElapsed > 0) {
          return totalDaily / hoursElapsed;
        }
      }
      return 0;
    }

    // Group by hour bucket and sum across providers
    const hourlyTotals = new Map<string, number>();
    for (const row of hourlyData) {
      const current = hourlyTotals.get(row.hourBucket) ?? 0;
      hourlyTotals.set(row.hourBucket, row.spendUsd + current);
    }

    const hours = Array.from(hourlyTotals.values());

    // Simple average: total spend / number of active hours
    // This gives the average burn rate during active periods
    const totalSpend = hours.reduce((a, b) => a + b, 0);
    const activeHours = hours.length;

    if (activeHours === 0) return 0;

    // Weighted: more recent hours weigh more (exponential decay)
    // But for simplicity and stability, use simple average
    return totalSpend / activeHours;
  }

  /**
   * Compute the fraction of the current period that has elapsed (0–1).
   */
  private computePeriodElapsedFraction(period: "daily" | "monthly"): number {
    const now = new Date();

    if (period === "daily") {
      const startOfDay = new Date(now);
      startOfDay.setHours(0, 0, 0, 0);
      const elapsedMs = now.getTime() - startOfDay.getTime();
      return Math.min(1, elapsedMs / 86_400_000);
    }

    // Monthly
    const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const endOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 1);
    const totalMs = endOfMonth.getTime() - startOfMonth.getTime();
    const elapsedMs = now.getTime() - startOfMonth.getTime();
    return Math.min(1, elapsedMs / totalMs);
  }

  /**
   * Compute cost efficiency scores for a set of models relative to each other.
   * A model that costs 3x more but only scores 10% higher gets deprioritized.
   *
   * @param models List of candidate models.
   * @param intent The routing intent to score against.
   * @param registry Model registry for capability lookup.
   * @returns Array of cost efficiency results, sorted by efficiency descending.
   */
  computeCostEfficiency(
    models: ModelCapability[],
    intent: string,
    registry: ModelRegistry,
  ): CostEfficiency[] {
    if (models.length === 0) return [];

    // Gather capability + cost data
    const rawData = models.map((m) => {
      const capScore = registry.getCapabilityScore(m.provider, m.model, intent);
      const inputCost = m.costPer1kInput ?? 0;
      const outputCost = m.costPer1kOutput ?? 0;
      // Combined cost: weighted average of input+output (assume 3:1 input:output ratio)
      const combinedCost = inputCost * 0.75 + outputCost * 0.25;
      return {
        provider: m.provider,
        model: m.model,
        capabilityScore: capScore,
        costPer1kInput: inputCost,
        costPer1kOutput: outputCost,
        combinedCost,
      };
    });

    // Find the cheapest non-zero cost for normalization
    const costs = rawData
      .map((d) => d.combinedCost)
      .filter((c) => c > 0);
    const minCost = costs.length > 0 ? Math.min(...costs) : 1;

    // Compute efficiency for each model
    const results: CostEfficiency[] = rawData.map((d) => {
      // Free models (combinedCost = 0) get costRatio of 0 (cheapest possible)
      const costRatio = d.combinedCost > 0 ? d.combinedCost / minCost : 0;

      // Efficiency: capability per unit of relative cost
      // Free models (costRatio = 0) get a high efficiency boost
      let efficiency: number;
      if (costRatio === 0) {
        // Free models are always efficient
        efficiency = d.capabilityScore * 2.0; // Boost free models
      } else {
        // Penalize expensive models: if costRatio=3 but capability only 10% higher,
        // efficiency = 1.1 / 3 = 0.37 (much worse than cheapest model's 1.0/1.0 = 1.0)
        efficiency = d.capabilityScore / costRatio;
      }

      return {
        provider: d.provider,
        model: d.model,
        capabilityScore: d.capabilityScore,
        costPer1kInput: d.costPer1kInput,
        costPer1kOutput: d.costPer1kOutput,
        costRatio,
        efficiency,
        recommended: false, // Set below
      };
    });

    // Sort by efficiency descending
    results.sort((a, b) => b.efficiency - a.efficiency);

    // Mark top models as recommended (efficiency within 80% of best)
    if (results.length > 0) {
      const bestEfficiency = results[0].efficiency;
      for (const r of results) {
        r.recommended = r.efficiency >= bestEfficiency * 0.8;
      }
    }

    return results;
  }

  /**
   * Detect token count anomalies. If current request tokens > 2x rolling average.
   *
   * @param currentTokens Estimated tokens for the current request.
   * @returns Anomaly detection result.
   */
  detectAnomaly(currentTokens: number): AnomalyDetection {
    // Update rolling window
    this.recentTokenCounts.push(currentTokens);
    if (this.recentTokenCounts.length > TOKEN_AVERAGE_WINDOW) {
      this.recentTokenCounts.shift();
    }

    // Need at least a few data points for a meaningful average
    if (this.recentTokenCounts.length < 5) {
      return {
        isAnomalous: false,
        currentTokens,
        averageTokens: currentTokens,
        ratio: 1.0,
        reason: "Insufficient data for anomaly detection",
      };
    }

    // Compute average excluding the current request
    const previousCounts = this.recentTokenCounts.slice(0, -1);
    const averageTokens =
      previousCounts.reduce((a, b) => a + b, 0) / previousCounts.length;

    if (averageTokens === 0) {
      return {
        isAnomalous: false,
        currentTokens,
        averageTokens: 0,
        ratio: 1.0,
        reason: "Average is zero — no baseline",
      };
    }

    const ratio = currentTokens / averageTokens;
    const isAnomalous = ratio > 2.0;

    return {
      isAnomalous,
      currentTokens,
      averageTokens: Math.round(averageTokens),
      ratio,
      reason: isAnomalous
        ? `Token spike: ${currentTokens} tokens is ${ratio.toFixed(1)}x the rolling average (${Math.round(averageTokens)})`
        : "Normal",
    };
  }

  /**
   * Get the auto-downgrade cost weight multiplier for the current budget state.
   * When active, cost scoring weight is boosted to prefer cheaper providers.
   */
  getCostWeightMultiplier(): number {
    const projection = this.projectBudget("daily", this.costTracker.dailyBudget);
    return projection.costWeightBoost;
  }

  /**
   * Check if a provider should be penalized during auto-downgrade.
   * Paid providers (pay_per_token, credits) get penalized; free/subscription don't.
   */
  shouldPenalizeProvider(
    providerName: string,
    budgetType: string,
  ): boolean {
    const projection = this.projectBudget("daily", this.costTracker.dailyBudget);
    if (!projection.autoDowngradeActive) return false;

    // Penalize paid providers (pay_per_token and credits)
    // Free (openrouter/ollama) and subscription (zai) are not penalized
    return budgetType === "pay_per_token" || budgetType === "credits";
  }

  /**
   * Get the penalty amount for paid providers during auto-downgrade.
   */
  getPaidProviderPenalty(): number {
    return DOWNGRADE_PAID_PENALTY;
  }

  /**
   * Build a full budget status payload for the /v1/budget endpoint.
   */
  getBudgetStatus(): {
    daily: BudgetProjection;
    monthly: BudgetProjection;
    spendByProvider: Array<{
      provider: string;
      dailySpendUsd: number;
      monthlySpendUsd: number;
    }>;
    hourlySpend: Array<{
      hourBucket: string;
      provider: string;
      spendUsd: number;
    }>;
    costEfficiency: CostEfficiency[];
    timestamp: string;
  } {
    const daily = this.projectBudget("daily", this.costTracker.dailyBudget);
    const monthly = this.projectBudget("monthly", this.costTracker.monthlyBudget);

    const dailySpend = this.db.getAllSpend("daily");
    const monthlySpend = this.db.getAllSpend("monthly");

    // Merge daily + monthly into per-provider rows
    const providerSet = new Set<string>([
      ...dailySpend.map((s) => s.provider),
      ...monthlySpend.map((s) => s.provider),
    ]);
    const spendByProvider = Array.from(providerSet).map((provider) => ({
      provider,
      dailySpendUsd: dailySpend.find((s) => s.provider === provider)?.spendUsd ?? 0,
      monthlySpendUsd: monthlySpend.find((s) => s.provider === provider)?.spendUsd ?? 0,
    }));

    const hourlySpend = this.db.getHourlySpend(BURN_RATE_WINDOW_HOURS);

    return {
      daily,
      monthly,
      spendByProvider,
      hourlySpend,
      costEfficiency: [], // Populated by caller with model registry data
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Populate cost efficiency data using the model registry.
   */
  getCostEfficiencyForModels(
    models: ModelCapability[],
    intent: string,
    registry: ModelRegistry,
  ): CostEfficiency[] {
    return this.computeCostEfficiency(models, intent, registry);
  }
}
