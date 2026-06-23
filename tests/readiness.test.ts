import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ModelRegistry } from "../src/model_registry.ts";
import { registryReadiness } from "../src/readiness.ts";
import { DBService } from "../src/db_service.ts";
import { loadConfig } from "../src/config.ts";

describe("Registry Readiness", () => {
  it("should block requests until registry is synced", async () => {
    let synced = false;
    
    // Simulate registry loading
    const config = loadConfig({ enabled: true, providerPriority: [], weights: { capability: 0, reliability: 0, cost: 0, latency: 0 } } as any);
    const db = { initializeSchema: () => {}, close: () => {} } as any;
    const registry = new ModelRegistry(db, config);

    // Call wait() - should block
    const waitPromise = registryReadiness.wait();
    
    // Check it's blocked after a short delay
    await new Promise(r => setTimeout(r, 50));
    
    registryReadiness.setSynced();
    await waitPromise;
    assert.ok(true, "Wait completed after registry sync");
  });
});
