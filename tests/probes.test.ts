// tests/probes.test.ts - Tests for src/probes.ts (p3e-002 follow-up)
//
// Covers loadAllProbes / probesFor / probeSummary against the 12 committed
// probes in probes/coding/ (p3e-003), hidden-case slicing semantics, and
// malformed-probe rejection via validateProbe.
//
// Hidden-case contract (docs/EXEC_SANDBOX_SPEC.md §3): the last
// harness.casesHidden cases are NOT included in the probe prompt but ARE run
// by the sandbox. The loader sends probe.prompt verbatim and the sandbox runs
// the full harness, so "hidden" is a data-level property: hidden case
// call/expect strings must not leak into the prompt text.

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadAllProbes, probesFor, probeSummary, validateProbe } from "../src/probes.js";
import { PROMPT_GENERATION } from "../src/benchmark_ladder.js";

// ---------- helpers ----------

/** Effective visible/hidden split for a loaded probe per the spec's slicing
 *  rule: last casesHidden cases are hidden. Note coding-exec-011 declares
 *  casesHidden=2 with only 1 case — the slice still behaves (everything is
 *  hidden), so assertions below use the sliced result, not the raw field. */
/** An expect string is "trivial" when it carries no leakage signal: bare
 *  Python literals (True/False/None/numbers/empty collections) and
 *  single-character strings legitimately occur in prompt prose (e.g.
 *  "Return True if balanced"). */
function isTrivialExpect(expect: string): boolean {
  const unquoted = expect.replace(/^['"](.*)['"]$/, "$1");
  if (unquoted.length <= 1) return true;
  if (/^(?:True|False|None)$/.test(unquoted)) return true;
  if (/^-?\d+(?:\.\d+)?$/.test(unquoted)) return true;
  if (expect === "[]" || expect === "{}" || expect === "''" || expect === '""') return true;
  return false;
}

function sliceHidden(probe: ReturnType<typeof loadAllProbes>[number]): {
  visible: Array<{ call: string; expect: string }>;
  hidden: Array<{ call: string; expect: string }>;
} {
  const { cases, casesHidden = 0 } = probe.harness;
  return {
    visible: cases.slice(0, cases.length - casesHidden),
    hidden: cases.slice(cases.length - casesHidden),
  };
}

function validProbeFixture(): Record<string, unknown> {
  return {
    id: "test-probe-001",
    generation: "gen-test",
    subIntent: "quick",
    prompt: "Write a Python function add(a, b) that returns the sum.",
    language: "python",
    entry: "add",
    harness: {
      setup: "from solution import add",
      cases: [
        { call: "add(1, 2)", expect: "3" },
        { call: "add(-1, 1)", expect: "0" },
      ],
      casesHidden: 1,
      check: "assert add(1, 1) == 2",
    },
  };
}

// ---------- loadAllProbes ----------

describe("loadAllProbes (committed probe set)", () => {
  const probes = loadAllProbes();

  it("loads all 12 committed probes", () => {
    assert.equal(probes.length, 12);
  });

  it("returns probes with unique, non-empty ids", () => {
    const ids = probes.map((p) => p.id);
    assert.equal(new Set(ids).size, ids.length);
    for (const id of ids) assert.ok(id.length > 0);
  });

  it("every committed probe validates (loader applies validateProbe)", () => {
    // The loader already ran validateProbe on each file; re-run to pin the
    // exact committed contents against the current validator.
    for (const p of probes) {
      assert.deepEqual(validateProbe(p), p, `probe ${p.id} should validate as-is`);
    }
  });

  it("every probe is tagged with the current PROMPT_GENERATION", () => {
    for (const p of probes) assert.equal(p.generation, PROMPT_GENERATION);
  });

  it("every probe has a valid subIntent and language", () => {
    for (const p of probes) {
      assert.ok(["quick", "repo", "agent"].includes(p.subIntent));
      assert.ok(["python", "typescript"].includes(p.language));
    }
  });

  it("every probe has a non-empty prompt, entry, and at least one case", () => {
    for (const p of probes) {
      assert.ok(p.prompt.length > 0);
      assert.ok(p.entry.length > 0);
      assert.ok(p.harness.cases.length >= 1);
      assert.ok(typeof p.harness.setup === "string");
    }
  });
});

// ---------- probesFor ----------

describe("probesFor", () => {
  it("returns all 12 probes for subIntent=quick at current generation (default)", () => {
    const quick = probesFor("quick");
    assert.equal(quick.length, 12);
    for (const p of quick) {
      assert.equal(p.subIntent, "quick");
      assert.equal(p.generation, PROMPT_GENERATION);
    }
  });

  it("explicitly passing the current generation matches the default", () => {
    assert.deepEqual(probesFor("quick", PROMPT_GENERATION), probesFor("quick"));
  });

  it("returns [] for unshipped subIntents (repo, agent)", () => {
    assert.deepEqual(probesFor("repo"), []);
    assert.deepEqual(probesFor("agent"), []);
  });

  it("returns [] for a non-matching generation", () => {
    assert.deepEqual(probesFor("quick", "gen-1999-01-01"), []);
  });
});

// ---------- probeSummary ----------

describe("probeSummary", () => {
  it("aggregates the committed set into one (quick, current generation) entry of 12", () => {
    const summary = probeSummary();
    assert.deepEqual(summary, [{ subIntent: "quick", generation: PROMPT_GENERATION, count: 12 }]);
  });

  it("summary counts sum to loadAllProbes().length", () => {
    const total = probeSummary().reduce((acc, s) => acc + s.count, 0);
    assert.equal(total, loadAllProbes().length);
  });
});

// ---------- hidden-case slicing ----------

describe("hidden-case slicing (data-level hidden contract)", () => {
  const probes = loadAllProbes();

  it("every probe declares a non-negative integer casesHidden", () => {
    for (const p of probes) {
      const h = p.harness.casesHidden;
      assert.ok(h === undefined || (Number.isInteger(h) && (h as number) >= 0), p.id);
    }
  });

  it("slicing yields at least one visible case where casesHidden < cases.length", () => {
    for (const p of probes) {
      const { visible } = sliceHidden(p);
      if ((p.harness.casesHidden ?? 0) < p.harness.cases.length) {
        assert.ok(visible.length >= 1, p.id);
      }
    }
  });

  it("hidden slice + visible slice = full case list, order preserved", () => {
    for (const p of probes) {
      const { visible, hidden } = sliceHidden(p);
      assert.deepEqual([...visible, ...hidden], p.harness.cases, p.id);
    }
  });

  it("hidden case calls do not leak into the prompt", () => {
    // The strong mechanical invariant: the hidden verification expressions
    // (the call code the sandbox runs) must never appear in prompt text.
    for (const p of probes) {
      const { hidden } = sliceHidden(p);
      for (const c of hidden) {
        assert.ok(!p.prompt.includes(c.call), `${p.id}: hidden call "${c.call}" leaked into prompt`);
      }
    }
  });

  it("non-trivial hidden expects not in prompt except documented task-semantics disclosures", () => {
    // Some prompts must name an expected value as part of the task spec:
    // coding-exec-011 is a bug-fix probe whose prompt states
    // "transition('inactive') must set status='inactive'" — that value is
    // required semantics, not a leaked case. Everything else must stay out.
    const intentional: ReadonlyMap<string, string[]> = new Map([
      ["coding-exec-011", ["'inactive'"]],
    ]);
    for (const p of probes) {
      const { hidden } = sliceHidden(p);
      for (const c of hidden) {
        if (isTrivialExpect(c.expect)) continue;
        if (p.prompt.includes(c.expect)) {
          const allowed = intentional.get(p.id) ?? [];
          assert.ok(allowed.includes(c.expect), `${p.id}: hidden expect "${c.expect}" leaked into prompt`);
        }
      }
    }
  });

  it("probes with no hidden declaration expose all cases as visible", () => {
    const probe = validateProbe({ ...validProbeFixture(), harness: { setup: "from solution import add", cases: [{ call: "add(1, 2)", expect: "3" }] } });
    assert.ok(probe);
    const { visible, hidden } = sliceHidden(probe);
    assert.equal(visible.length, 1);
    assert.equal(hidden.length, 0);
  });
});

// ---------- validateProbe: malformed rejection ----------

describe("validateProbe (malformed rejection)", () => {
  it("accepts a fully-formed probe", () => {
    const v = validateProbe(validProbeFixture());
    assert.ok(v);
    assert.equal(v!.id, "test-probe-001");
    assert.equal(v!.harness.casesHidden, 1);
    assert.equal(v!.harness.check, "assert add(1, 1) == 2");
  });

  it("rejects non-object and null inputs", () => {
    assert.equal(validateProbe(null), null);
    assert.equal(validateProbe(undefined), null);
    assert.equal(validateProbe("a string"), null);
    assert.equal(validateProbe(42), null);
    assert.equal(validateProbe([]), null);
  });

  it("rejects missing/empty/non-string id", () => {
    assert.equal(validateProbe({ ...validProbeFixture(), id: "" }), null);
    assert.equal(validateProbe({ ...validProbeFixture(), id: 123 }), null);
    const { id: _omit, ...noId } = validProbeFixture() as Record<string, never>;
    assert.equal(validateProbe(noId), null);
  });

  it("rejects missing/empty generation", () => {
    assert.equal(validateProbe({ ...validProbeFixture(), generation: "" }), null);
    const { generation: _g, ...noGen } = validProbeFixture() as Record<string, never>;
    assert.equal(validateProbe(noGen), null);
  });

  it("rejects invalid subIntent", () => {
    assert.equal(validateProbe({ ...validProbeFixture(), subIntent: "debugging" }), null);
    assert.equal(validateProbe({ ...validProbeFixture(), subIntent: null }), null);
  });

  it("rejects missing/empty prompt", () => {
    assert.equal(validateProbe({ ...validProbeFixture(), prompt: "" }), null);
    const { prompt: _p, ...noPrompt } = validProbeFixture() as Record<string, never>;
    assert.equal(validateProbe(noPrompt), null);
  });

  it("rejects unsupported languages", () => {
    assert.equal(validateProbe({ ...validProbeFixture(), language: "javascript" }), null);
    assert.equal(validateProbe({ ...validProbeFixture(), language: "rust" }), null);
  });

  it("rejects missing/empty entry", () => {
    assert.equal(validateProbe({ ...validProbeFixture(), entry: "" }), null);
    const { entry: _e, ...noEntry } = validProbeFixture() as Record<string, never>;
    assert.equal(validateProbe(noEntry), null);
  });

  it("rejects missing/non-object harness", () => {
    const { harness: _h, ...noHarness } = validProbeFixture() as Record<string, never>;
    assert.equal(validateProbe(noHarness), null);
    assert.equal(validateProbe({ ...validProbeFixture(), harness: null }), null);
    assert.equal(validateProbe({ ...validProbeFixture(), harness: "from solution import add" }), null);
  });

  it("rejects non-string setup", () => {
    const fixture = validProbeFixture();
    (fixture.harness as Record<string, unknown>).setup = 42;
    assert.equal(validateProbe(fixture), null);
  });

  it("rejects non-array or empty cases", () => {
    const fixture = validProbeFixture();
    (fixture.harness as Record<string, unknown>).cases = "not-an-array";
    assert.equal(validateProbe(fixture), null);
    const empty = validProbeFixture();
    (empty.harness as Record<string, unknown>).cases = [];
    assert.equal(validateProbe(empty), null);
  });

  it("rejects cases with non-string call/expect or non-object entries", () => {
    const badCase = validProbeFixture();
    (badCase.harness as { cases: unknown[] }).cases = [{ call: "add(1, 2)", expect: 3 }];
    assert.equal(validateProbe(badCase), null);

    const nullCase = validProbeFixture();
    (nullCase.harness as { cases: unknown[] }).cases = [null];
    assert.equal(validateProbe(nullCase), null);
  });

  it("rejects invalid casesHidden (non-number, non-integer, negative)", () => {
    for (const bad of ["2", 1.5, -1]) {
      const fixture = validProbeFixture();
      (fixture.harness as Record<string, unknown>).casesHidden = bad;
      assert.equal(validateProbe(fixture), null, `casesHidden=${JSON.stringify(bad)}`);
    }
  });

  it("rejects non-string check", () => {
    const fixture = validProbeFixture();
    (fixture.harness as Record<string, unknown>).check = 42;
    assert.equal(validateProbe(fixture), null);
  });

  it("accepts casesHidden=0 and omitted optional fields (check omitted)", () => {
    const v = validateProbe({
      ...validProbeFixture(),
      harness: {
        setup: "from solution import add",
        cases: [{ call: "add(1, 2)", expect: "3" }],
        casesHidden: 0,
      },
    });
    assert.ok(v);
    assert.equal(v!.harness.casesHidden, 0);
    assert.equal(v!.harness.check, undefined);
  });
});
