// src/exec_sandbox.ts - Exec Sandbox Harness
//
// Runs untrusted code (LLM output) against test harnesses in isolated
// execution environments. Primary backend: Docker (network-disabled,
// per-run temp dir, strict resource caps). Fallback: win32 job object
// behind env flag COGNITIVE_ROUTER_SANDBOX_WIN32_JOB=1 (default OFF).
// When neither backend is available, sandbox_unavailable is returned.
//
// Design per docs/EXEC_SANDBOX_SPEC.md (Phase 3 coding intent).

import { logger } from "./logger.js";

// ─── Types ───────────────────────────────────────────────────

export type FailReason =
  | "no_code_block"
  | "syntax_error"
  | "timeout"
  | "runtime_error"
  | "wrong_output"
  | "sandbox_unavailable";

export interface ExecRunResult {
  casesPassed: number;
  casesTotal: number;
  passRate: number;             // 0..1
  failReason: FailReason | null; // first fatal reason, if any
  durationMs: number;
  backend: "docker" | "win32-job" | null;
}

export interface TestCase {
  call: string;
  expect: string;
  check?: string;   // optional assertion function body (trusted, authored by us)
}

export interface ProbeHarness {
  setup: string;      // e.g. "from solution import is_balanced"
  cases: TestCase[];
  casesHidden?: number;
}

export interface ExecRunOptions {
  language: "python" | "typescript";
  code: string;
  harness: ProbeHarness;
  timeoutMs?: number;
}

export interface ExecSandbox {
  run(opts: ExecRunOptions): Promise<ExecRunResult>;
  available(): Promise<"docker" | "win32-job" | null>;
}

// ─── Constants ───────────────────────────────────────────────

const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_OUTPUT_BYTES = 65_536;  // 64KB each for stdout/stderr
const MAX_PROCESS_COUNT = 32;

const DOCKER_IMAGE = "cognitive-router-sandbox:latest";

// Purpose-built image (sandbox.Dockerfile): node 22 + python3, non-root
// sandbox user, no network. node:22-slim does NOT ship python3.
const PYTHON_INTERPRETER = "/usr/bin/python3";
// Debian slim installs node to /usr/local/bin (python3 stays /usr/bin).
const NODE_INTERPRETER = "/usr/local/bin/node";

// ─── Backend detection ───────────────────────────────────────

function isWin32JobEnabled(): boolean {
  return process.env.COGNITIVE_ROUTER_SANDBOX_WIN32_JOB === "1";
}

let _dockerAvailable: boolean | null = null;
let _dockerCheckInFlight: Promise<boolean> | null = null;

async function checkDockerAvailable(): Promise<boolean> {
  if (_dockerAvailable !== null) return _dockerAvailable;
  if (_dockerCheckInFlight) return _dockerCheckInFlight;

  _dockerCheckInFlight = (async (): Promise<boolean> => {
    try {
      const cp = await spawnProcess("docker", ["info", "--format", "{{.ServerVersion}}"]);
      const ok = cp.exitCode === 0 && cp.stdout.trim().length > 0;
      _dockerAvailable = ok;
      return ok;
    } catch {
      _dockerAvailable = false;
      return false;
    }
  })();

  return _dockerCheckInFlight;
}

function resetDockerCache(): void {
  _dockerAvailable = null;
  _dockerCheckInFlight = null;
}

// ─── Cross-platform process spawn ────────────────────────────

interface SpawnResult {
  exitCode: number | null;
  signal: string | null;
  stdout: string;
  stderr: string;
}

function spawnProcess(
  command: string,
  args: string[],
  opts: { timeoutMs?: number; cwd?: string; env?: Record<string, string> } = {},
): Promise<SpawnResult> {
  return new Promise((resolve, reject) => {
    // Dynamic import to avoid bundler issues with child_process
    import("node:child_process").then(({ spawn }) => {
      const cp = spawn(command, args, {
        stdio: ["pipe", "pipe", "pipe"],
        cwd: opts.cwd,
        env: opts.env ? { ...process.env, ...opts.env } : process.env,
        // On Windows, spawn resolves docker.exe via PATH directly; shell:true
        // routes through cmd.exe, which mangles multi-line sh -c scripts
        // (heredoc/newline corruption → instant runtime_error).
        shell: false,
        windowsHide: true,
      });

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      let stdoutBytes = 0;
      let stderrBytes = 0;
      let timedOut = false;

      const timeout = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
      const timer = setTimeout(() => {
        timedOut = true;
        cp.kill("SIGKILL");
        // Also kill process tree on Windows via taskkill
        if (process.platform === "win32" && cp.pid) {
          try {
            import("node:child_process").then(({ execSync }) => {
              execSync(`taskkill /F /T /PID ${cp.pid}`, { stdio: "ignore" });
            }).catch(() => {});
          } catch { /* best effort */ }
        }
      }, timeout);

      const capCollector = (chunk: Buffer, buffer: Buffer[], byteCount: number): { buffer: Buffer[]; byteCount: number; capped: boolean } => {
        const remaining = MAX_OUTPUT_BYTES - byteCount;
        if (remaining <= 0) return { buffer, byteCount, capped: true };
        if (chunk.length <= remaining) {
          buffer.push(chunk);
          return { buffer, byteCount: byteCount + chunk.length, capped: false };
        }
        buffer.push(chunk.subarray(0, remaining));
        return { buffer, byteCount: MAX_OUTPUT_BYTES, capped: true };
      };

      // NOTE: never pause() on cap. node's `close` fires only after stdio
      // streams END; a paused stream never ends, so the promise never
      // resolves and the caller hangs forever (deadlock found 2026-09-12:
      // output-cap test stalled both manual + detached runs). Instead keep
      // draining and DISCARD overflow — capCollector already drops bytes
      // past MAX_OUTPUT_BYTES, so memory stays bounded and the child's
      // writes never backpressure the pipe.
      cp.stdout?.on("data", (chunk: Buffer) => {
        const result = capCollector(chunk, stdoutChunks, stdoutBytes);
        stdoutChunks.splice(0, stdoutChunks.length, ...result.buffer);
        stdoutBytes = result.byteCount;
      });

      cp.stderr?.on("data", (chunk: Buffer) => {
        const result = capCollector(chunk, stderrChunks, stderrBytes);
        stderrChunks.splice(0, stderrChunks.length, ...result.buffer);
        stderrBytes = result.byteCount;
      });

      cp.on("close", (exitCode, signal) => {
        clearTimeout(timer);
        resolve({
          exitCode: timedOut ? null : exitCode,
          signal: timedOut ? "SIGKILL" : signal,
          stdout: Buffer.concat(stdoutChunks).toString("utf-8"),
          stderr: Buffer.concat(stderrChunks).toString("utf-8"),
        });
      });

      cp.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });
    }).catch(reject);
  });
}

// ─── Code extraction (moved inline; unit-tested separately) ──

export function extractCodeBlock(
  response: string,
  language: string,
): string | null {
  // Pattern: ```language\n...code...```  — global, LAST match wins per spec.
  // Content group allows empty (```python\n``` is a valid empty block).
  const fencedLang = new RegExp(
    "```" + language + "[ \\t]*\n([\\s\\S]*?)```",
    "gi",
  );
  const allLang = [...response.matchAll(fencedLang)];
  if (allLang.length > 0) {
    return allLang[allLang.length - 1][1].trim();
  }

  // Pattern: ```\n...code...```  (no language tag). Last match wins.
  // Unclosed fence (EOF instead of closing ```) also matches — content to EOF.
  const fencedNoLang = /```[ \t]*\n([\s\S]*?)(?:\n```|$(?!\n?```))/g;
  const allFenced = [...response.matchAll(fencedNoLang)];
  if (allFenced.length > 0) {
    return allFenced[allFenced.length - 1][1].trim();
  }

  // Inline code: single backtick or indented block (fallback)
  const inlineMatch = response.match(/`([^`]+)`/);
  if (inlineMatch) return inlineMatch[1].trim();

  return null;
}

// ─── Docker Sandbox ──────────────────────────────────────────

function makeDockerRunArgs(
  language: "python" | "typescript",
  code: string,
  harness: ProbeHarness,
  tempDir: string,
): { interpreter: string; scriptContent: string } {
  // Build a temporary test script that:
  // 1. Writes the user code to a file
  // 2. Executes the harness setup and runs all test cases
  // 3. Outputs JSON results: { results: [{ index, passed, actual, error }], combined: true/false }

  const entryFile = language === "python" ? "/tmp/solution.py" : "/tmp/solution.ts";
  const runnerFile = "/tmp/runner.js";
  const codeFile = language === "python" ? "/tmp/solution.py" : "/tmp/solution.mjs";

  let harnessScript: string;

  if (language === "python") {
    harnessScript = `
import sys, json, io

# Suppress import-time side effects
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Write solution code
with open('${entryFile}', 'w') as f:
    f.write("""${escapePythonString(code)}""")

# Execute setup
${harness.setup}

# Reset stdout after setup
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

results = []
for i, (call_str, expected) in enumerate([
${harness.cases.map((c, i) => `    (${JSON.stringify(c.call)}, ${JSON.stringify(c.expect)})`).join(",\n")}
]):
    try:
        # Redirect stdout during eval
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        result = repr(eval(call_str))
        sys.stdout = old_stdout
        passed = result == expected
        results.append({"index": i, "passed": passed, "actual": result, "error": None})
    except Exception as e:
        results.append({"index": i, "passed": False, "actual": None, "error": str(e)})

print("__SANDBOX_RESULT__" + json.dumps({"results": results}))
`;
  } else {
    // TypeScript: compile to JS then run
    harnessScript = `
const fs = require('fs');
const path = require('path');

// Write solution code
fs.writeFileSync('${codeFile}', ${JSON.stringify(code)});

// Execute setup inline
${harness.setup.replace(/^from solution import /g, "const ")}

const results = [];
const cases = [
${harness.cases.map((c, i) => `  { call: ${JSON.stringify(c.call)}, expect: ${JSON.stringify(c.expect)} }`).join(",\n")}
];

for (let i = 0; i < cases.length; i++) {
  const { call, expect } = cases[i];
  try {
    // For TS, eval the call expression in a sandbox with the imported names
    const actual = String(eval(call));
    const passed = actual === expect;
    results.push({ index: i, passed, actual, error: null });
  } catch (e) {
    results.push({ index: i, passed: false, actual: null, error: String(e) });
  }
}

console.log("__SANDBOX_RESULT__" + JSON.stringify({ results }));
`;
  }

  const finalScript = `node -e ${JSON.stringify(harnessScript)}`;

  return { interpreter: NODE_INTERPRETER, scriptContent: finalScript };
}

function escapePythonString(s: string): string {
  return s
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
}

function parseSandboxOutput(stdout: string): Array<{ index: number; passed: boolean; actual: string | null; error: string | null }> | null {
  const marker = "__SANDBOX_RESULT__";
  const idx = stdout.lastIndexOf(marker);
  if (idx === -1) return null;

  const jsonStr = stdout.slice(idx + marker.length);
  try {
    const parsed = JSON.parse(jsonStr);
    if (parsed && Array.isArray(parsed.results)) {
      return parsed.results;
    }
    return null;
  } catch {
    return null;
  }
}

async function runDocker(
  language: "python" | "typescript",
  code: string,
  harness: ProbeHarness,
  timeoutMs: number,
): Promise<ExecRunResult> {
  const startTime = Date.now();

  const entryFile = language === "python" ? "/tmp/solution.py" : "/tmp/solution";
  const runnerFile = "/tmp/runner.js";
  const outputLimit = MAX_OUTPUT_BYTES;

  // Self-contained docker run: bind-mount the code+runner in, execute, print JSON.
  let command: string[];

  // Unique container name so a timed-out run can be force-removed — `--rm`
  // never fires when the docker *client* is SIGKILLed and the container's
  // entrypoint keeps running detached.
  const containerName = `cogrouter-sbx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  if (language === "python") {
    command = [
      "run", "--rm",
      "--name", containerName,
      "--network", "none",
      "--cap-drop", "ALL",
      "--security-opt", "no-new-privileges",
      "--read-only",
      "--tmpfs", "/tmp:noexec,nosuid,size=64m",
      "--pids-limit", String(MAX_PROCESS_COUNT),
      DOCKER_IMAGE,
      "sh", "-c",
      [
        `cat > /tmp/solution.py << 'SANDBOXEOF'`,
        code,
        `SANDBOXEOF`,
        `cat > /tmp/runner.py << 'SANDBOXEOF'`,
        buildPythonRunner(harness),
        `SANDBOXEOF`,
        `${PYTHON_INTERPRETER} /tmp/runner.py`,
      ].join("\n"),
    ];
  } else {
    command = [
      "run", "--rm",
      "--name", containerName,
      "--network", "none",
      "--cap-drop", "ALL",
      "--security-opt", "no-new-privileges",
      "--read-only",
      "--tmpfs", "/tmp:noexec,nosuid,size=64m",
      "--pids-limit", String(MAX_PROCESS_COUNT),
      DOCKER_IMAGE,
      "sh", "-c",
      [
        // Type stripping only applies to .ts files — node refuses TS syntax
        // in .mjs even with --experimental-strip-types.
        `cat > /tmp/solution.ts << 'SANDBOXEOF'`,
        code,
        `SANDBOXEOF`,
        `cat > /tmp/runner.ts << 'SANDBOXEOF'`,
        buildTypeScriptRunner(harness),
        `SANDBOXEOF`,
        `${NODE_INTERPRETER} --experimental-strip-types /tmp/runner.ts`,
      ].join("\n"),
    ];
  }

  try {
    const result = await spawnProcess("docker", command, { timeoutMs });

    const durationMs = Date.now() - startTime;

    // Timeout detection: spawnProcess resolves with exitCode=null + signal=SIGKILL
    // when the timeout fired (no exception is thrown).
    if (result.exitCode === null && result.signal === "SIGKILL") {
      // The container outlived the killed client — force-remove it.
      spawnProcess("docker", ["rm", "-f", containerName], { timeoutMs: 10_000 }).catch(() => {});
      return {
        casesPassed: 0,
        casesTotal: harness.cases.length,
        passRate: 0,
        failReason: "timeout",
        durationMs,
        backend: "docker",
      };
    }

    // Parse results
    const parsed = parseSandboxOutput(result.stdout);
    if (!parsed) {
      if (process.env.COGNITIVE_ROUTER_SANDBOX_DEBUG) {
        // eslint-disable-next-line no-console
        console.error("[sandbox-debug] stdout:", result.stdout.slice(0, 2000), "\n[sandbox-debug] stderr:", (result.stderr || "").slice(0, 2000), "\n[sandbox-debug] exitCode:", result.exitCode);
      }
      // Check for obvious errors
      const stderr = (result.stderr || "").trim();
      if (stderr.length > 0 || result.exitCode !== 0) {
        return {
          casesPassed: 0,
          casesTotal: harness.cases.length,
          passRate: 0,
          failReason: "runtime_error",
          durationMs,
          backend: "docker",
        };
      }
      return {
        casesPassed: 0,
        casesTotal: harness.cases.length,
        passRate: 0,
        failReason: "runtime_error",
        durationMs,
        backend: "docker",
      };
    }

    const casesPassed = parsed.filter((r) => r.passed).length;
    const casesTotal = parsed.length;
    const failReason: FailReason | null =
      casesPassed === casesTotal ? null : "wrong_output";

    return {
      casesPassed,
      casesTotal,
      passRate: casesTotal > 0 ? casesPassed / casesTotal : 1,
      failReason,
      durationMs,
      backend: "docker",
    };
  } catch (err: any) {
    const durationMs = Date.now() - startTime;

    if (err.message?.includes("timed out") || err.killed) {
      return {
        casesPassed: 0,
        casesTotal: harness.cases.length,
        passRate: 0,
        failReason: "timeout",
        durationMs,
        backend: "docker",
      };
    }

    return {
      casesPassed: 0,
      casesTotal: harness.cases.length,
      passRate: 0,
      failReason: "sandbox_unavailable",
      durationMs,
      backend: "docker",
    };
  }
}

// ─── Win32 Job Object Fallback ───────────────────────────────

async function runWin32Job(
  language: "python" | "typescript",
  code: string,
  harness: ProbeHarness,
  timeoutMs: number,
): Promise<ExecRunResult> {
  const startTime = Date.now();
  const tempDirBase = process.env.TEMP || "C:\\Windows\\Temp";
  const tempDirName = `sandbox-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const tempDir = `${tempDirBase}\\${tempDirName}`;

  try {
    // Create temp dir
    await spawnProcess("cmd", ["/c", "mkdir", tempDir], { timeoutMs: 2000 });
  } catch {
    return {
      casesPassed: 0,
      casesTotal: harness.cases.length,
      passRate: 0,
      failReason: "runtime_error",
      durationMs: Date.now() - startTime,
      backend: "win32-job",
    };
  }

  try {
    const solutionFile = `${tempDir}\\solution.${language === "python" ? "py" : "mjs"}`;
    const runnerFile = `${tempDir}\\runner.${language === "python" ? "py" : "mjs"}`;

    // Write solution code
    await spawnProcess("cmd", ["/c", `type nul > "${solutionFile}" && echo.> "${solutionFile}"`], { timeoutMs: 2000 });
    await spawnProcess("cmd", ["/c", `echo ${escapeWin32Arg(code)} > "${solutionFile}"`], { timeoutMs: 2000 });

    // Write runner
    const runnerContent = language === "python"
      ? buildPythonRunner(harness)
      : buildTypeScriptRunner(harness);
    await spawnProcess("cmd", ["/c", `echo ${escapeWin32Arg(runnerContent)} > "${runnerFile}"`], { timeoutMs: 2000 });

    // Execute
    const interpreter = language === "python"
      ? (await findPython()) || "python"
      : "node";

    const result = await spawnProcess(interpreter, [runnerFile], {
      cwd: tempDir,
      timeoutMs,
    });

    // Cleanup
    spawnProcess("cmd", ["/c", `rmdir /s /q "${tempDir}"`], { timeoutMs: 2000 }).catch(() => {});

    const durationMs = Date.now() - startTime;

    const parsed = parseSandboxOutput(result.stdout);
    if (!parsed) {
      return {
        casesPassed: 0,
        casesTotal: harness.cases.length,
        passRate: 0,
        failReason: "runtime_error",
        durationMs,
        backend: "win32-job",
      };
    }

    const casesPassed = parsed.filter((r) => r.passed).length;
    return {
      casesPassed,
      casesTotal: parsed.length,
      passRate: parsed.length > 0 ? casesPassed / parsed.length : 1,
      failReason: casesPassed === parsed.length ? null : "wrong_output",
      durationMs,
      backend: "win32-job",
    };
  } catch (err: any) {
    const durationMs = Date.now() - startTime;

    // Cleanup on error
    spawnProcess("cmd", ["/c", `rmdir /s /q "${tempDir}"`], { timeoutMs: 2000 }).catch(() => {});

    if (err.message?.includes("timed out") || err.killed) {
      return {
        casesPassed: 0,
        casesTotal: harness.cases.length,
        passRate: 0,
        failReason: "timeout",
        durationMs,
        backend: "win32-job",
      };
    }

    return {
      casesPassed: 0,
      casesTotal: harness.cases.length,
      passRate: 0,
      failReason: "runtime_error",
      durationMs,
      backend: "win32-job",
    };
  }
}

function escapeWin32Arg(s: string): string {
  return s
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\r\n");
}

async function findPython(): Promise<string | null> {
  try {
    const r = await spawnProcess("python", ["--version"], { timeoutMs: 2000 });
    if (r.exitCode === 0) return "python";
  } catch { /* not found */ }
  try {
    const r = await spawnProcess("python3", ["--version"], { timeoutMs: 2000 });
    if (r.exitCode === 0) return "python3";
  } catch { /* not found */ }
  return null;
}

// ─── Runner builders ─────────────────────────────────────────

function buildPythonRunner(harness: ProbeHarness): string {
  const lines: string[] = [
    "import sys, json, io",
    "",
    "def run_tests():",
    indentPythonBlock(harness.setup, 4),
    "",
    "    results = []",
    "    cases = [",
  ];

  for (const c of harness.cases) {
    lines.push(`        (${JSON.stringify(c.call)}, ${JSON.stringify(c.expect)}),`);
  }

  lines.push("    ]", "");

  lines.push(
    "    for i, (call_str, expected) in enumerate(cases):",
    "        try:",
    "            result = repr(eval(call_str))",
    "            passed = result == expected",
    '            results.append({"index": i, "passed": passed, "actual": result, "error": None})',
    "        except Exception as e:",
    '            results.append({"index": i, "passed": False, "actual": None, "error": str(e)})',
    "",
    '    print("__SANDBOX_RESULT__" + json.dumps({"results": results}))',
    "",
    "run_tests()",
  );

  return lines.join("\n");
}

function indentPythonBlock(block: string, spaces: number): string {
  const indent = " ".repeat(spaces);
  return block
    .split("\n")
    .map((line) => (line.trim() ? indent + line : ""))
    .join("\n");
}

function buildTypeScriptRunner(harness: ProbeHarness): string {
  const lines: string[] = [
    "const results = [];",
    "const cases = [",
  ];

  for (const c of harness.cases) {
    lines.push(`  { call: ${JSON.stringify(c.call)}, expect: ${JSON.stringify(c.expect)} },`);
  }

  lines.push("];", "");

  if (harness.setup) {
    // Transform "from solution import is_balanced" into a dynamic import of
    // the solution file — static require() doesn't exist in ESM.
    const transformedSetup = harness.setup.replace(
      /from solution import (\w+)/g,
      "const $1 = (await import('./solution.ts')).$1;",
    );
    lines.push(transformedSetup);
  }

  // Always expose the solution's exports to eval scope (harness setup may be
  // empty — eval can only see module/global scope, not import bindings).
  lines.push("Object.assign(globalThis, await import('./solution.ts'));")

  lines.push(
    "",
    "for (let i = 0; i < cases.length; i++) {",
    "  const { call, expect } = cases[i];",
    "  try {",
    "    const actual = String(eval(call));",
    "    const passed = actual === expect;",
    "    results.push({ index: i, passed, actual, error: null });",
    "  } catch (e) {",
    "    results.push({ index: i, passed: false, actual: null, error: String(e) });",
    "  }",
    "}",
    "",
    'console.log("__SANDBOX_RESULT__" + JSON.stringify({ results }));',
  );

  return lines.join("\n");
}

// ─── Main Sandbox Implementation ─────────────────────────────

class ExecSandboxImpl implements ExecSandbox {
  async run(opts: ExecRunOptions): Promise<ExecRunResult> {
    const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;

    // Try Docker first
    const dockerOk = await checkDockerAvailable();
    if (dockerOk) {
      return await runDocker(opts.language, opts.code, opts.harness, timeoutMs);
    }

    // Try win32 job object fallback (gate behind env flag)
    if (isWin32JobEnabled()) {
      if (process.platform === "win32") {
        return await runWin32Job(opts.language, opts.code, opts.harness, timeoutMs);
      }
      // Win32 job object requires Windows
      return {
        casesPassed: 0,
        casesTotal: opts.harness.cases.length,
        passRate: 0,
        failReason: "sandbox_unavailable",
        durationMs: 0,
        backend: null,
      };
    }

    // Sandbox unavailable
    return {
      casesPassed: 0,
      casesTotal: opts.harness.cases.length,
      passRate: 0,
      failReason: "sandbox_unavailable",
      durationMs: 0,
      backend: null,
    };
  }

  async available(): Promise<"docker" | "win32-job" | null> {
    const dockerOk = await checkDockerAvailable();
    if (dockerOk) return "docker";

    if (isWin32JobEnabled() && process.platform === "win32") {
      return "win32-job";
    }

    return null;
  }
}

// ─── Factory ─────────────────────────────────────────────────

let _instance: ExecSandbox | null = null;

export function createExecSandbox(): ExecSandbox {
  if (!_instance) {
    _instance = new ExecSandboxImpl();
  }
  return _instance;
}

export function resetExecSandbox(): void {
  _instance = null;
  resetDockerCache();
}

// For testing: create a fresh instance without singleton caching
export function createFreshExecSandbox(): ExecSandbox {
  return new ExecSandboxImpl();
}

// ─── Default export ──────────────────────────────────────────

const defaultSandbox = createExecSandbox();
export default defaultSandbox;
