// src/classifier.ts — Intent classification via embeddings + tiebreaker

import { logger } from "./logger.js";

export interface Classification {
  intent: string;
  confidence: number;
}

// Intent categories and their prototype descriptions.
// These are embedded once and compared against incoming messages.
const INTENT_PROTOTYPES: Record<string, string[]> = {
  coding: [
    "write a function that handles database queries with proper error handling",
    "fix this bug in the authentication middleware",
    "refactor the class to use dependency injection",
    "implement a REST API endpoint for user management",
    "add unit tests for the payment processing module",
  ],
  research: [
    "what are the latest developments in quantum computing",
    "compare the performance of rust versus go for web servers",
    "find recent papers on transformer architecture improvements",
    "investigate the current state of solid state battery technology",
  ],
  creative: [
    "write a short story about a mars colony",
    "compose a poem about autumn leaves",
    "draft a screenplay scene set in a cyberpunk city",
    "create character backstories for a fantasy novel",
  ],
  conversation: [
    "hey how are you doing today",
    "what do you think about that",
    "that's interesting, tell me more",
    "thanks for the help earlier",
  ],
  summary: [
    "summarize this article for me",
    "give me the key points from this document",
    "tl;dr of this meeting notes",
    "condense this chapter into bullet points",
  ],
  retrieval: [
    "what is the capital of brazil",
    "when was python 3.12 released",
    "what does this error code mean",
    "find the documentation for this api method",
  ],
  science: [
    "explain the mechanism of CRISPR gene editing",
    "derive the equations of motion for a pendulum",
    "what causes protein folding and how does it relate to disease",
    "calculate the equilibrium constant for this reaction",
  ],
  business: [
    "analyze the competitive landscape for SaaS startups",
    "create a go-to-market strategy for a new mobile app",
    "what are the key metrics for a subscription business",
    "draft a business plan for a consulting practice",
  ],
  math: [
    "solve this system of differential equations",
    "prove that the sum of two odd numbers is even",
    "calculate the integral of x squared from 0 to 1",
    "find the eigenvalues of this matrix",
  ],
  analysis: [
    "compare these two approaches and recommend the best one",
    "what are the trade-offs between microservices and monoliths",
    "evaluate the risks of this architecture decision",
    "assess the feasibility of this project timeline",
  ],
};

export class IntentClassifier {
  private prototypes: Map<string, number[][]> = new Map();
  private initialized = false;

  constructor(private config: { tiebreakerThreshold: number }) {}

  async initialize(embedFn: (text: string) => Promise<number[]>): Promise<void> {
    logger.info("Pre-computing intent prototype embeddings...");

    for (const [intent, examples] of Object.entries(INTENT_PROTOTYPES)) {
      const embeddings: number[][] = [];
      for (const example of examples) {
        try {
          const vec = await embedFn(example);
          embeddings.push(vec);
        } catch (err) {
          logger.warn(`Failed to embed prototype for "${intent}": ${err}`);
        }
      }
      this.prototypes.set(intent, embeddings);
    }

    this.initialized = true;
    logger.info(
      `Classifier ready — ${this.prototypes.size} intent categories loaded.`,
    );
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  async classify(
    prompt: string,
    embedFn: (text: string) => Promise<number[]>,
  ): Promise<Classification> {
    if (!this.initialized) {
      await this.initialize(embedFn);
    }

    // Phase 1: Embedding similarity against prototypes
    // If the embedding call fails or times out, fall back to keyword matching
    let promptEmbedding: number[];
    try {
      promptEmbedding = await embedFn(prompt);
    } catch (err) {
      logger.warn(`Embedding failed — using keyword fallback: ${err instanceof Error ? err.message : err}`);
      return this.classifyByKeyword(prompt);
    }

    const scores = this.scoreAllIntents(promptEmbedding);

    const sorted = [...scores.entries()].sort((a, b) => b[1] - a[1]);
    const top = sorted[0];
    const second = sorted[1];

    const topIntent = top[0];
    const topScore = top[1];
    const margin = topScore - (second ? second[1] : 0);

    // Phase 2: Tiebreaker — if confidence is ambiguous, use LLM
    if (topScore < this.config.tiebreakerThreshold || margin < 0.1) {
      logger.debug(
        `Ambiguous classification (top=${topScore.toFixed(2)}, margin=${margin.toFixed(2)}) — tiebreaker needed`,
      );
      // TODO: Call Gemma 4 for disambiguation
      // For now, return the top embedding result with adjusted confidence
      return {
        intent: topIntent,
        confidence: Math.max(topScore, 0.5),
      };
    }

    return {
      intent: topIntent,
      confidence: topScore,
    };
  }

  /** Keyword-based fallback when embeddings are unavailable.
   *  Not as accurate but fast and never blocks. */
  classifyByKeyword(prompt: string): Classification {
    const lower = prompt.toLowerCase();
    const keywords: Record<string, string[]> = {
      coding: ["function", "code", "bug", "refactor", "test", "api", "class", "import", "error", "compile"],
      research: ["research", "compare", "latest", "find", "investigate", "papers", "developments"],
      creative: ["write a story", "poem", "screenplay", "character", "novel", "creative"],
      conversation: ["hey", "thanks", "how are you", "what do you think", "interesting", "cool", "nice"],
      summary: ["summarize", "tl;dr", "key points", "condense", "bullet points"],
      retrieval: ["what is", "when was", "who", "where", "capital of", "what does"],
      science: ["explain", "derive", "equation", "crispr", "protein", "chemical", "physics"],
      business: ["business", "market", "strategy", "metrics", "plan", "revenue", "saas"],
      math: ["solve", "integral", "matrix", "eigenvalue", "equation", "prove", "calculate"],
      analysis: ["analyze", "trade-off", "evaluate", "assess", "compare", "risk", "feasibility"],
    };

    let bestIntent = "conversation";
    let bestScore = 0;

    for (const [intent, words] of Object.entries(keywords)) {
      let hits = 0;
      for (const w of words) {
        if (lower.includes(w)) hits++;
      }
      const score = hits / words.length;
      if (score > bestScore) {
        bestScore = score;
        bestIntent = intent;
      }
    }

    // Low confidence for keyword match — router will weight accordingly
    return { intent: bestIntent, confidence: Math.max(bestScore, 0.4) };
  }

  private scoreAllIntents(embedding: number[]): Map<string, number> {
    const scores = new Map<string, number>();

    for (const [intent, prototypes] of this.prototypes) {
      let maxSim = 0;
      let sumSim = 0;

      for (const proto of prototypes) {
        const sim = cosineSimilarity(embedding, proto);
        maxSim = Math.max(maxSim, sim);
        sumSim += sim;
      }

      // Use max similarity with slight boost from average
      const avgSim = sumSim / prototypes.length;
      const combined = maxSim * 0.7 + avgSim * 0.3;
      scores.set(intent, combined);
    }

    return scores;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
