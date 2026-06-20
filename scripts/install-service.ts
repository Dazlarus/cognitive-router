// scripts/install-service.ts — Install the Cognitive Router as a Windows Service
// Run: npx tsx scripts/install-service.ts
// Or:  node dist/scripts/install-service.js

import { Service } from "node-windows";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

const svc = new Service({
  name: "Cognitive Router",
  description: "Self-healing LLM proxy — routes requests to the best available model provider.",
  script: resolve(projectRoot, "dist", "server.js"),
  cwd: projectRoot,
  env: [
    { name: "NODE_ENV", value: "production" },
    { name: "ROUTER_PORT", value: process.env.ROUTER_PORT ?? "3456" },
    { name: "ROUTER_LOG_LEVEL", value: process.env.ROUTER_LOG_LEVEL ?? "info" },
    { name: "ROUTER_DB_PATH", value: resolve(projectRoot, "data", "cognitive-router.db") },
    { name: "ROUTER_PRIORITY", value: process.env.ROUTER_PRIORITY ?? "zai,openrouter,gemini,ollama" },
    { name: "ROUTER_VRAM_LIMIT", value: process.env.ROUTER_VRAM_LIMIT ?? "11" },
  ],
  // node-windows uses winsw.exe which handles stdio logging
  logPath: resolve(projectRoot, "logs"),
});

svc.on("install", () => {
  console.log("✅ Cognitive Router service installed.");
  svc.start();
});

svc.on("alreadyinstalled", () => {
  console.log("ℹ️  Service already installed. Use --uninstall first if you need to reinstall.");
});

svc.on("start", () => {
  console.log("🚀 Cognitive Router service started.");
  console.log("   Listening on http://127.0.0.1:3456");
});

svc.on("stop", () => {
  console.log("🛑 Cognitive Router service stopped.");
});

svc.on("uninstall", () => {
  console.log("🗑️ Cognitive Router service uninstalled.");
});

svc.on("error", (err) => {
  console.error("❌ Service error:", err);
});

// Parse command line
const arg = process.argv[2];

if (arg === "--uninstall") {
  console.log("Uninstalling Cognitive Router service...");
  svc.uninstall();
} else if (arg === "--status") {
  // node-windows doesn't have a status method, use sc.exe
  const { execSync } = require("node:child_process");
  try {
    const status = execSync('sc query "Cognitive Router"').toString();
    console.log(status);
  } catch {
    console.log("Service not found.");
  }
} else {
  console.log("Installing Cognitive Router service...");
  svc.install();
}
