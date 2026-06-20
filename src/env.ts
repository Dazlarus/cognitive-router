// src/env.ts - dependency-free .env loader for standalone proxy entry points.

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function loadEnv(envPath: string): void {
  if (!existsSync(envPath)) return;

  const content = readFileSync(envPath, "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const eqIdx = trimmed.indexOf("=");
    if (eqIdx === -1) continue;

    const key = trimmed.substring(0, eqIdx).trim();
    const value = unquoteEnvValue(trimmed.substring(eqIdx + 1).trim());
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

export function loadProjectEnv(importMetaUrl: string): void {
  const moduleDir = dirname(fileURLToPath(importMetaUrl));
  loadEnv(resolve(moduleDir, "..", ".env"));
}

function unquoteEnvValue(value: string): string {
  if (value.length < 2) return value;
  const quote = value[0];
  if ((quote !== `"` && quote !== "'") || value[value.length - 1] !== quote) {
    return value;
  }
  return value.substring(1, value.length - 1);
}
