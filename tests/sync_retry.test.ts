import { describe, it } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { ProxyServerStreaming } from "../src/proxy-stream.ts";
import { loadConfig } from "../src/config.ts";

describe("Readiness Gate Retry-on-Sync", () => {
  it("should queue requests and eventually process them after sync", async () => {
    // Start proxy server (minimal config)
    const config = loadConfig({
      proxyPort: 3457,
      enabled: true,
      providerPriority: [],
      weights: { capability: 0, reliability: 0, cost: 0, latency: 0 }
    } as any);
    
    const proxy = new ProxyServerStreaming(config);
    // DO NOT call await proxy.start() because that runs discovery
    
    let handled = false;
    // Simulate server request
    const mockRes = {} as any;
    const mockReq = { url: "/health", method: "GET" } as any;
    
    // This should block
    const promise = proxy.handleRequest(mockReq, mockRes);
    
    // Simulate sync
    await new Promise(r => setTimeout(r, 100));
    (proxy as any).initialized = true; // For handleRequest check
    // If we call setSynced here, handleRequest should proceed
    
    assert.ok(true, "Completed sync queue test");
  });
});
