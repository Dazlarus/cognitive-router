// src/readiness.ts — Readiness gate for provider registry synchronization
import { logger } from "./logger.js";

class RegistryReadiness {
  private isSynced = false;
  private queue: Array<() => void> = [];

  setSynced() {
    this.isSynced = true;
    logger.info("Provider registry sync complete.");
    for (const resolve of this.queue) {
      resolve();
    }
    this.queue = [];
  }

  async wait() {
    if (this.isSynced) return;
    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
    });
  }
}

export const registryReadiness = new RegistryReadiness();
