// scripts/test-gemini.mts — Debug test for Gemini adapter with real tool schemas
import { getProvider } from "../src/providers.ts";
import { readFileSync } from "node:fs";

const env = readFileSync(".env", "utf8");
const apiKey = env.match(/GEMINI_API_KEY=\s*(\S+)/)?.[1]?.trim() ?? "";

const adapter = getProvider("gemini");
if (!adapter) {
  console.log("No gemini adapter found");
  process.exit(1);
}

// Test 1: Simple tool with additionalProperties
console.log("=== Test 1: Simple tool with additionalProperties ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hello" }],
    tools: [{
      type: "function",
      function: {
        name: "test",
        description: "Test function",
        parameters: {
          type: "object",
          properties: { x: { type: "string" } },
          required: ["x"],
          additionalProperties: false,
        },
      },
    }],
    max_tokens: 50,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(empty)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 400));
}

// Test 2: Tool with $schema and $defs (should be stripped)
console.log("\n=== Test 2: Tool with \$schema, \$defs, \$ref ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hello" }],
    tools: [{
      type: "function",
      function: {
        name: "edit",
        description: "Edit a file",
        parameters: {
          $schema: "http://json-schema.org/draft-07/schema#",
          type: "object",
          properties: {
            path: { type: "string" },
            edits: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  oldText: { type: "string" },
                  newText: { type: "string" },
                },
              },
            },
          },
          $defs: {},
          additionalProperties: false,
        },
      },
    }],
    max_tokens: 50,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(empty)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 400));
}

// Test 3: Tool with nested anyOf/oneOf (common in OpenClaw schemas)
console.log("\n=== Test 3: Tool with anyOf/oneOf ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hello" }],
    tools: [{
      type: "function",
      function: {
        name: "search",
        description: "Search for things",
        parameters: {
          type: "object",
          properties: {
            query: {
              anyOf: [
                { type: "string" },
                { type: "array", items: { type: "string" } },
              ],
            },
          },
        },
      },
    }],
    max_tokens: 50,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(empty)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 400));
}

// Test 4: Tool with deeply nested objects containing additionalProperties
console.log("\n=== Test 4: Nested additionalProperties ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hello" }],
    tools: [{
      type: "function",
      function: {
        name: "exec",
        description: "Execute command",
        parameters: {
          type: "object",
          properties: {
            command: { type: "string" },
            env: {
              type: "object",
              description: "Environment variables",
              additionalProperties: { type: "string" },
            },
          },
          required: ["command"],
        },
      },
    }],
    max_tokens: 50,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(empty)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 400));
}

// Test 5: Check what openAIToolsToGeminiFunctionDeclarations produces
console.log("\n=== Test 5: Inspect sanitized output ===");
import { GeminiAdapter } from "../src/providers.ts";
const geminiAny = GeminiAdapter as any;
// Check if there's a build method we can inspect
const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(GeminiAdapter));
console.log("Adapter methods:", methods);
console.log("Adapter keys:", Object.keys(GeminiAdapter));
