// scripts/test-gemini-v2.mts — Test Gemini with real-world problematic schemas
import { getProvider } from "../src/providers.ts";
import { readFileSync } from "node:fs";

const env = readFileSync(".env", "utf8");
const apiKey = env.match(/GEMINI_API_KEY=\s*(\S+)/)?.[1]?.trim() ?? "";

const adapter = getProvider("gemini")!;

// Test: Tool with anyOf (THE bug from production)
console.log("=== Test: anyOf in schema ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hi" }],
    tools: [{
      type: "function",
      function: {
        name: "search",
        description: "Search",
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
    max_tokens: 20,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(tool call)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 300));
}

// Test: Tool with const
console.log("\n=== Test: const in schema ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hi" }],
    tools: [{
      type: "function",
      function: {
        name: "set_mode",
        description: "Set mode",
        parameters: {
          type: "object",
          properties: {
            mode: { type: "string", const: "advanced" },
          },
        },
      },
    }],
    max_tokens: 20,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(tool call)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 300));
}

// Test: Array content (the "text" proto error)
console.log("\n=== Test: Array content blocks ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "Say hi" },
      ],
    } as any],
    max_tokens: 20,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(empty)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 300));
}

// Test: Deeply nested anyOf with const (the actual production killer)
console.log("\n=== Test: Deeply nested anyOf + const ===");
try {
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hi" }],
    tools: [{
      type: "function",
      function: {
        name: "edit",
        description: "Edit file",
        parameters: {
          type: "object",
          properties: {
            edits: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  type: {
                    anyOf: [
                      { type: "string", const: "replace" },
                      { type: "string", const: "insert" },
                    ],
                  },
                  text: { type: "string" },
                },
              },
            },
          },
        },
      },
    }],
    max_tokens: 20,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(tool call)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 300));
}

// Test: Many tools (like OpenClaw's 30+ tool definitions)
console.log("\n=== Test: 30 tool definitions ===");
try {
  const manyTools = Array.from({ length: 30 }, (_, i) => ({
    type: "function",
    function: {
      name: `tool_${i}`,
      description: `Tool number ${i}`,
      parameters: {
        type: "object",
        properties: {
          arg1: { type: "string", description: `Arg ${i}` },
          arg2: {
            type: "string",
            enum: ["a", "b", "c"],
          },
          arg3: {
            anyOf: [
              { type: "string" },
              { type: "number" },
            ],
          },
        },
        required: ["arg1"],
        additionalProperties: false,
      },
    },
  }));
  const resp = await adapter.chatCompletion("gemini-2.5-flash", {
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "Say hi" }],
    tools: manyTools,
    max_tokens: 20,
    stream: false,
  }, apiKey);
  console.log("PASS:", resp.choices?.[0]?.message?.content ?? "(tool call)");
} catch (e: any) {
  console.log("FAIL:", e.message?.substring(0, 300));
}
