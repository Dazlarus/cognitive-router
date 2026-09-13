# Exec Sandbox Harness — Coding Benchmark Verification

**Status:** Draft for Daz review — 2026-09-10
**Scope:** Phase 3 coding-intent work. Companion to `benchmark_ladder.ts` (ordinal ranking) and the planned replacement of the `benchmark_chat.ts` coding probe.

## 1. Goal

Give coding probes an objective correctness signal the judge can't fake: run the model's code against tests and score pass@1. The ladder keeps ranking models pairwise; exec results become (a) a judge-free floor score per coding identity and (b) a calibration yardstick for the judge (how often does a judge 8/10 actually pass tests?).

## 2. Threat model — this runs LLM output

Non-negotiables before any probe runs:

- **No network.** Every supported path must deny socket egress. Localhost included (prevents exfil to a local service and SSRF-style probing).
- **No filesystem writes outside a per-run temp dir.** Fresh temp dir per execution, deleted after.
- **Wall-clock + CPU limits.** Per-run timeout (default 10s), kill process tree on expiry.
- **No spawning long-lived processes.** Deny obvious runners; rely on timeout as backstop, not the only defense.
- **Output caps.** stdout/stderr capped (default 64KB each) — a model that prints a gigabyte must not OOM the router host.
- **Process count cap** per run (default 32).

### Execution backends, in order of preference

| Backend | Isolation | Notes |
|---------|-----------|-------|
| Windows Sandbox / Docker container | Strong | Preferred. Router host already has Docker (Pagehawk). Image: slim node + python, no network. |
| Restricted child process (job object) | Weak-moderate | Fallback only. Job object caps CPU/memory/handles on Windows; no real fs/network denial. Gate behind an env flag, default OFF. |

If Docker isn't available and the flag is off, exec probes are skipped — models keep ladder-only scores. **Never silently fall back to unrestricted execution.**

## 3. Probe format

New file family: `probes/coding/*.json` (generation-tagged, same `PROMPT_GENERATION` discipline as the ladder).

```json
{
  "id": "coding-exec-001",
  "generation": "gen-2026-09-10",
  "subIntent": "quick",
  "prompt": "Write a Python function is_balanced(s: str) -> bool ...",
  "language": "python",
  "entry": "is_balanced",
  "harness": {
    "setup": "from solution import is_balanced",
    "cases": [
      { "call": "is_balanced('()[]{}')", "expect": "True" },
      { "call": "is_balanced('([)]')", "expect": "False" }
    ],
    "casesHidden": 3
  }
}
```

- **Model contract:** response must contain a fenced code block; harness extracts the last fenced block in the target language (falls back to first block). No block = automatic fail with reason `no_code_block`.
- **Hidden cases:** a slice of cases is NOT included in the probe prompt but IS run. Prevents overfitting-to-visible-tests and prompt leakage between ladder rounds. Prompts say "must satisfy edge cases" without enumerating them.
- **Sub-intent field:** `quick` | `repo` | `agent` — matches the ladder identity extension. Phase 1 of the harness ships `quick` only; `repo` (long-context, multi-file patches) and `agent` (tool-call sequences) come later on the same schema.

### Case evaluation

Default: eval-and-compare (`repr(eval(call)) == expect`). For richer checks, a case may carry `check`: a small assertion function body string run in the same sandbox (still counts against the run's budget). Keep `check` rare — every exec of model-adjacent code is attack surface, and setup/case strings are authored by us, so they're trusted, but simpler is safer.

## 4. Scoring

- `pass@1` = fraction of cases passing (0.0–1.0), plus fail reasons: `no_code_block`, `syntax_error`, `timeout`, `runtime_error`, `wrong_output`, `sandbox_unavailable`.
- Per identity (model×quant×effort) over N probes: mean pass@1 + Wilson interval. **A coding identity needs N≥3 probe results before its exec floor participates in any blend.**
- Ladder integration: exec floor is stored alongside BT strength; the routing blend (separate task) uses `capability = w_bt·strength_shrunk + w_exec·floor + w_live·judged_ewma`. Exact weights are the routing task's job, not this harness's.
- **Judge calibration:** whenever the judge also scores an exec-probe response (ladder rounds include coding probes), log `(judge_score, pass@1)` pairs to a `judge_calibration` table. Report agreement periodically. If judge and exec diverge persistently, the judge prompt for coding gets rebuilt around the rubric — data first.

## 5. Module shape

`src/exec_sandbox.ts`:

```ts
export interface ExecRunResult {
  casesPassed: number;
  casesTotal: number;
  passRate: number;         // 0..1
  failReason: FailReason | null;  // first fatal reason, if any
  durationMs: number;
  backend: "docker" | "win32-job";
}
export interface ExecSandbox {
  run(opts: { language: "python" | "typescript"; code: string; harness: ProbeHarness; timeoutMs?: number }): Promise<ExecRunResult>;
  available(): Promise<"docker" | "win32-job" | null>;
}
```

- `probes.ts`: load/validate probe JSON, generation filter, `probesFor(subIntent, generation)`.
- `benchmark_exec.ts`: orchestrates model-call → extraction → sandbox run → persist to `exec_benchmark_results` (SQLite: model identity key, probe id, generation, passRate, failReason, durationMs, timestamp). Version-hash caching mirrors `benchmark_chat` (skip if <7 days + unchanged), but probes are cheap to re-run — shorter cache (48h) is fine for remotes; locals gated on GPU idle windows like the existing auto-bench.
- Extraction helper (`extractCodeBlock`) unit-tested hard: fenced/unnfenced, language-tagged/untagged, prose-wrapped, multiple blocks.

## 6. TypeScript probes

Node is on the host; container image includes it. TS responses: run through `tsc --noEmit` (or strip-types via node ≥22 native) then execute with a tiny assert-harness. Module imports are **forbidden** in probe prompts ("single self-contained block") — keeps the sandbox surface tiny. Python probes: same rule, stdlib only, no pip.

## 7. Initial probe set

10–15 `quick` probes to start, drawn to cover: pure functions with edge cases, string/data manipulation, a small algorithm (DP or graph), a SQL answer evaluated via sqlite3 in-sandbox (setup creates the table, cases run the model's query), and one "explain-then-code" instruction-following check (code must still pass; prose ignored). Realistic-bug probes: mine 3–5 from Narrator/Karl Code closed issues, paraphrased to self-contained form. All probes Daz-reviewed before entering the pool — authored content is trusted, model output is not.

## 8. Acceptance criteria

1. Docker backend runs a probe end-to-end with network-disabled container; timeout kill verified (infinite-loop probe); fs-escape attempt verified contained.
2. `extractCodeBlock` test suite green on the pathological cases.
3. Two models benchmarked across the initial probe set; scores + fail reasons in `exec_benchmark_results`; `sandbox_unavailable` path verified with Docker down.
4. Judge calibration table populated from ≥20 (judge, exec) pairs and a first agreement report.
5. No probe run possible when both backends unavailable — exec floor simply absent, ladder unaffected.

## 9. Explicitly out of scope (this spec)

- Wiring ladder strengths into `RoutingEngine.decide()` (routing-blend task).
- `repo` and `agent` sub-intent probes (schema supports them; harness v2).
- Windows Sandbox backend beyond "investigate if Docker exits the stack".
