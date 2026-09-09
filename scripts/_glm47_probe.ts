// Probe: does z-ai/glm-4.7 serve when max_tokens fits under its 3072 output cap?
const key = process.env.OPENROUTER_API_KEY;
if (!key) { console.error("OPENROUTER_API_KEY not set"); process.exit(1); }

const PROMPT =
  "Write a TypeScript function `debounceAsync(fn, ms)` that debounces an async function. Include full type signatures.";

async function probe(maxTokens: number) {
  const t0 = Date.now();
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    signal: AbortSignal.timeout(90_000),
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: "z-ai/glm-4.7",
      messages: [{ role: "user", content: PROMPT }],
      max_tokens: maxTokens,
      temperature: 0.1,
    }),
  });
  const ms = Date.now() - t0;
  const body: any = await res.json().catch(() => ({}));
  const choice = body?.choices?.[0];
  console.log(
    `max_tokens=${maxTokens} -> HTTP ${res.status} ${ms}ms finish=${choice?.finish_reason ?? "n/a"} ` +
    `content_len=${choice?.message?.content?.length ?? 0} ` +
    `err=${body?.error?.message?.slice(0, 120) ?? "none"}`
  );
  if (choice?.message?.content) {
    console.log(`  head: ${choice.message.content.slice(0, 100).replace(/\n/g, " ")}`);
  }
}

(async () => {
  try { await probe(6144); } catch (e: any) { console.log(`max_tokens=6144 -> ABORT ${e.name}: ${e.message}`); }
  try { await probe(3072); } catch (e: any) { console.log(`max_tokens=3072 -> ABORT ${e.name}: ${e.message}`); }
  try { await probe(3000); } catch (e: any) { console.log(`max_tokens=3000 -> ABORT ${e.name}: ${e.message}`); }
})();
