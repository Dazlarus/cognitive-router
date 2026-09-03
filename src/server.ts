// src/server.ts — Standalone proxy entry point with streaming support
// Run: npx tsx src/server.ts
// Or:  node dist/server.js (after build)

import { startProxyStreaming } from "./proxy-stream.js";

// Hardening 2026-09-03: never die silently. Unknown async failures are
// logged with full stack and the process keeps serving (the daily-restart
// cron is the backstop if state degrades). Two router deaths on 2026-09-03
// (04:55, ~10:33) exited code 1 without a captured stack — this makes the
// next one diagnosable and, ideally, a non-event.
process.on("unhandledRejection", (reason) => {
  console.error("[CognitiveRouter] UNHANDLED REJECTION (process survived):", reason);
});
process.on("uncaughtException", (err) => {
  console.error("[CognitiveRouter] UNCAUGHT EXCEPTION (process survived):", err);
});

startProxyStreaming().catch((err) => {
  console.error("Failed to start Cognitive Router proxy:", err);
  process.exit(1);
});
