// Card says glm-4.7 is a reasoning model. Probe: does controlling reasoning fix the empty?
const key = process.env.OPENROUTER_API_KEY!;

async function probe(label: string, extra: Record<string, unknown>, prompt = "Say hello.") {
  const t0 = Date.now();
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: "z-ai/glm-4.7",
      messages: [{ role: "user", content: prompt }],
      ...extra,
    }),
    signal: AbortSignal.timeout(120_000),
  });
  const ms = Date.now() - t0;
  const b: any = await res.json().catch(() => ({}));
  const c = b?.choices?.[0];
  console.log(
    `${label} -> HTTP ${res.status} ${ms}ms finish=${c?.finish_reason ?? "n/a"} len=${c?.message?.content?.length ?? 0} ` +
    `reasoning_len=${c?.message?.reasoning?.length ?? c?.reasoning?.length ?? 0} ` +
    `text=${JSON.stringify(c?.message?.content?.slice(0, 50) ?? b?.error?.message?.slice(0, 120) ?? "empty")}`
  );
}

(async () => {
  const P = /PROMPT_PLACEHOLDER/.test("") ? "" : "Write a TypeScript function `debounceAsync(fn, ms)` with full type signatures.";
  try { await probe("reasoning-excluded cap=1024", { max_tokens: 1024, temperature: 0.1, reasoning: { exclude: true } }, P); } catch (e: any) { console.log(`1 ABORT ${e.message}`); }
  try { await probe("effort-low cap=1024", { max_tokens: 1024, temperature: 0.1, reasoning: { effort: "low" } }, P); } catch (e: any) { console.log(`2 ABORT ${e.message}`); }
  try { await probe("no-cap plain", { temperature: 0.1 }, P); } catch (e: any) { console.log(`3 ABORT ${e.message}`); }
})();
