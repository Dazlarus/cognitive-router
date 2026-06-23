// tests/budget.test.ts — Cost budget enforcement tests
// Run with: node --import tsx tests/budget.test.ts
// Or:       npx tsx --test tests/budget.test.ts

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { CostTracker } from "../src/cost_tracker.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig, type CognitiveRouterConfig } from "../src/config.ts";

// ─── Test Helpers ───────────────────────────────────────────

function makeConfig(overrides: Partial<CognitiveRouterConfig> = {}): CognitiveRouterConfig {
  const base = loadConfig({
    enabled: true,
    logLevel: "warn",
    providerPriority: ["zai", "openrouter", "gemini", "ollama"],
    providers: {
      zai: { budgetType: "subscription", priority: "high" },
      openrouter: { budgetType: "free", priority: "high" },
      gemini: { budgetType: "credits", priority: "medium" },
      ollama: { budgetType: "free", priority: "low" },
    },
    weights: { capability: 0.5, reliability: 0.25, cost: 0.15, latency: 0.1 },
  });
  return { ...base, ...overrides };
}

/** Mock DB with in-memory spend tracking for testing budget enforcement. */
function makeMockDBWithSpend(): DBService & { _spendStore: Map<string, number> } {
  const spendStore = new Map<string, number>(); // key: `${period}:${provider}`

  const db = {
    initializeSchema: async () => {},
    recordDecision: () => {},
    recordCallOutcome: () => {},
    recordRetry: () => {},
    getDecisionByRequestId: () => null,
    getRetryCount: () => 0,
    getRecentDecisions: () => [],
    getModelStats: () => [],
    getProviderHealth: () => [],
    getSpendByProvider: () => [],
    loadCapabilityOverrides: () => [],
    getCapabilityOverride: () => null,
    upsertCapabilityOverride: () => {},
    recordJudgeEvaluation: () => {},
    recordSpend: (provider: string, amountUsd: number, period: "daily" | "monthly") => {
      const key = `${period}:${provider}`;
      spendStore.set(key, (spendStore.get(key) ?? 0) + amountUsd);
    },
    getSpend: (provider: string, period: "daily" | "monthly") => {
      return spendStore.get(`${period}:${provider}`) ?? 0;
    },
    getAllSpend: (period: "daily" | "monthly") => {
      const result: Array<{ provider: string; spendUsd: number }> = [];
      for (const [key, value] of spendStore.entries()) {
        if (key.startsWith(`${period}:`)) {
          result.push({ provider: key.split(":")[1], spendUsd: value });
        }
      }
      return result;
    },
    close: () => {},
    _spendStore: spendStore,
  } as any;

  return db;
}

// ─── Tests ──────────────────────────────────────────────────

describe("CostTracker — Budget Enforcement", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let costTracker: CostTracker;
  const originalDailyBudget = process.env.ROUTER_DAILY_BUDGET_USD;
  const originalMonthlyBudget = process.env.ROUTER_MONTHLY_BUDGET_USD;

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDBWithSpend();
    costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();
  });

  afterEach(() => {
    if (originalDailyBudget === undefined) {
      delete process.env.ROUTER_DAILY_BUDGET_USD;
    } else {
      process.env.ROUTER_DAILY_BUDGET_USD = originalDailyBudget;
    }
    if (originalMonthlyBudget === undefined) {
      delete process.env.ROUTER_MONTHLY_BUDGET_USD;
    } else {
      process.env.ROUTER_MONTHLY_BUDGET_USD = originalMonthlyBudget;
    }
  });

  it("should default to $5 daily and $50 monthly budget", () => {
    const tracker = new CostTracker(db, config);
    assert.equal(tracker.dailyBudget, 5.0, "Default daily budget should be $5");
    assert.equal(tracker.monthlyBudget, 50.0, "Default monthly budget should be $50");
  });

  it("should respect ROUTER_DAILY_BUDGET_USD env var", () => {
    process.env.ROUTER_DAILY_BUDGET_USD = "2.50";
    const tracker = new CostTracker(db, config);
    assert.equal(tracker.dailyBudget, 2.50, "Daily budget should be $2.50 from env");
  });

  it("should respect ROUTER_MONTHLY_BUDGET_USD env var", () => {
    process.env.ROUTER_MONTHLY_BUDGET_USD = "25.00";
    const tracker = new CostTracker(db, config);
    assert.equal(tracker.monthlyBudget, 25.0, "Monthly budget should be $25 from env");
  });

  it("should not report budget exceeded when spend is below budget", () => {
    costTracker.recordSpend("openrouter", 1.0);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), false);
    assert.equal(costTracker.getDailySpend("openrouter"), 1.0);
    assert.equal(costTracker.getMonthlySpend("openrouter"), 1.0);
  });

  it("should report budget exceeded when daily spend exceeds daily budget", () => {
    // Default daily budget is $5
    costTracker.recordSpend("openrouter", 5.0);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), true);
  });

  it("should report budget exceeded when monthly spend exceeds monthly budget", () => {
    // Default monthly budget is $50
    costTracker.recordSpend("openrouter", 50.0);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), true);
  });

  it("should accumulate spend across multiple calls", () => {
    costTracker.recordSpend("openrouter", 1.5);
    costTracker.recordSpend("openrouter", 2.0);
    costTracker.recordSpend("openrouter", 1.5);

    assert.equal(costTracker.getDailySpend("openrouter"), 5.0);
    assert.equal(costTracker.getMonthlySpend("openrouter"), 5.0);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), true);
  });

  it("should track spend independently per provider", () => {
    costTracker.recordSpend("openrouter", 3.0);
    costTracker.recordSpend("gemini", 1.0);

    assert.equal(costTracker.getDailySpend("openrouter"), 3.0);
    assert.equal(costTracker.getDailySpend("gemini"), 1.0);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), false);
    assert.equal(costTracker.isBudgetExceeded("gemini"), false);

    // Push openrouter over budget
    costTracker.recordSpend("openrouter", 2.5);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), true);
    assert.equal(costTracker.isBudgetExceeded("gemini"), false);
  });

  it("should persist spend to the database", () => {
    costTracker.recordSpend("openrouter", 1.5);

    // Check that the mock DB recorded it
    const dbSpend = db.getSpend("openrouter", "daily");
    assert.equal(dbSpend, 1.5);

    const dbMonthly = db.getSpend("openrouter", "monthly");
    assert.equal(dbMonthly, 1.5);
  });

  it("should load persisted spend on refreshProviderStatus", async () => {
    // Record some spend
    costTracker.recordSpend("openrouter", 3.0);

    // Create a new tracker with the same DB
    const newTracker = new CostTracker(db, config);
    await newTracker.refreshProviderStatus();

    assert.equal(newTracker.getDailySpend("openrouter"), 3.0);
    assert.equal(newTracker.getMonthlySpend("openrouter"), 3.0);
  });

  it("should handle recordUsage with costUsd", async () => {
    await costTracker.recordUsage("openrouter", "deepseek/deepseek-v4-flash", {
      costUsd: 2.0,
    });

    assert.equal(costTracker.getDailySpend("openrouter"), 2.0);
    assert.equal(costTracker.getMonthlySpend("openrouter"), 2.0);
    assert.equal(costTracker.isBudgetExceeded("openrouter"), false);
  });

  it("should handle recordUsage without costUsd gracefully", async () => {
    await costTracker.recordUsage("openrouter", "some-model", {});
    assert.equal(costTracker.getDailySpend("openrouter"), 0);
  });

  it("should not crash for unknown provider", () => {
    costTracker.recordSpend("unknown-provider", 100.0);
    assert.equal(costTracker.isBudgetExceeded("unknown-provider"), false);
    assert.equal(costTracker.getDailySpend("unknown-provider"), 0);
  });
});

describe("CostTracker — Budget Daily/Monthly Reset", () => {
  let config: CognitiveRouterConfig;
  let db: DBService & { _spendStore: Map<string, number> };

  beforeEach(async () => {
    config = makeConfig();
    db = makeMockDBWithSpend();
  });

  it("should reset daily spend when date changes (new day = new date_key)", async () => {
    // The SQLite getSpend method queries by date_key (YYYY-MM-DD).
    // When a new day starts, the date_key changes, so yesterday's spend
    // is not matched. A fresh CostTracker loading from DB will see 0 spend.
    const costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();

    // Record spend
    costTracker.recordSpend("openrouter", 10.0);
    assert.equal(costTracker.getDailySpend("openrouter"), 10.0);

    // Simulate day rollover: clear the DB spend store (equivalent to date_key changing)
    db._spendStore.delete("daily:openrouter");

    // Fresh tracker loading from DB should see 0 spend for the new day
    const newTracker = new CostTracker(db, config);
    await newTracker.refreshProviderStatus();
    assert.equal(newTracker.getDailySpend("openrouter"), 0, "Daily spend should reset after day change");
  });

  it("should reset monthly spend when month changes", async () => {
    const costTracker = new CostTracker(db, config);
    await costTracker.refreshProviderStatus();

    // Record spend in current month
    costTracker.recordSpend("openrouter", 30.0);
    assert.equal(costTracker.getMonthlySpend("openrouter"), 30.0);

    // Simulate month rollover by clearing the store entry for current month
    // (in real SQLite, the date_key changes so old entries aren't matched)
    db._spendStore.delete("monthly:openrouter");

    // Create fresh tracker — should load 0 spend for new month
    const newTracker = new CostTracker(db, config);
    await newTracker.refreshProviderStatus();
    assert.equal(newTracker.getMonthlySpend("openrouter"), 0, "Monthly spend should reset after month change");
  });
});

describe("CostTracker — Budget Warning Logging", () => {
  let config: CognitiveRouterConfig;
  let db: DBService;
  let originalDaily: string | undefined;
  let originalMonthly: string | undefined;

  beforeEach(async () => {
    originalDaily = process.env.ROUTER_DAILY_BUDGET_USD;
    originalMonthly = process.env.ROUTER_MONTHLY_BUDGET_USD;
    process.env.ROUTER_DAILY_BUDGET_USD = "1.00";
    process.env.ROUTER_MONTHLY_BUDGET_USD = "10.00";

    config = makeConfig();
    db = makeMockDBWithSpend();
  });

  afterEach(() => {
    if (originalDaily === undefined) delete process.env.ROUTER_DAILY_BUDGET_USD;
    else process.env.ROUTER_DAILY_BUDGET_USD = originalDaily;
    if (originalMonthly === undefined) delete process.env.ROUTER_MONTHLY_BUDGET_USD;
    else process.env.ROUTER_MONTHLY_BUDGET_USD = originalMonthly;
  });

  it("should track budget state correctly with custom budgets", async () => {
    const tracker = new CostTracker(db, config);
    await tracker.refreshProviderStatus();

    assert.equal(tracker.dailyBudget, 1.0);
    assert.equal(tracker.monthlyBudget, 10.0);

    // Under budget
    tracker.recordSpend("openrouter", 0.5);
    assert.equal(tracker.isBudgetExceeded("openrouter"), false);

    // Over daily budget
    tracker.recordSpend("openrouter", 0.6);
    assert.equal(tracker.getDailySpend("openrouter"), 1.1);
    assert.equal(tracker.isBudgetExceeded("openrouter"), true);
  });
});
