// src/logger.ts — Minimal logger that respects config.logLevel
// Routes to console (visible in gateway logs) without polluting chat.

export type LogLevel = "debug" | "info" | "warn" | "error";

const PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

let currentLevel: LogLevel = "info";

export function setLevel(level: LogLevel): void {
  currentLevel = level;
}

function shouldLog(level: LogLevel): boolean {
  return PRIORITY[level] >= PRIORITY[currentLevel];
}

function log(level: LogLevel, msg: string): void {
  if (!shouldLog(level)) return;
  const prefix = "[CognitiveRouter]";
  switch (level) {
    case "debug":
      console.debug(`${prefix} ${msg}`);
      break;
    case "info":
      console.info(`${prefix} ${msg}`);
      break;
    case "warn":
      console.warn(`${prefix} ${msg}`);
      break;
    case "error":
      console.error(`${prefix} ${msg}`);
      break;
  }
}

export const logger = {
  debug: (msg: string) => log("debug", msg),
  info: (msg: string) => log("info", msg),
  warn: (msg: string) => log("warn", msg),
  error: (msg: string) => log("error", msg),
};
