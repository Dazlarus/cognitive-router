// src/server.ts — Standalone proxy entry point with streaming support
// Run: npx tsx src/server.ts
// Or:  node dist/server.js (after build)

import { startProxyStreaming } from "./proxy-stream.js";

startProxyStreaming().catch((err) => {
  console.error("Failed to start Cognitive Router proxy:", err);
  process.exit(1);
});
