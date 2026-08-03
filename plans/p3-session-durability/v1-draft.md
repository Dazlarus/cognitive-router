# Phase 3: Session-Level Durability for Cognitive Router

**Plan ID:** `a93baef9-edca-4ceb-88ad-6dc65bbef55c`  
**Status:** Draft (Cycle 1 — revised in Cycle 2)  
**CodeSpec Session:** `20431b97-6a1b-4775-903a-2e508e6addbf`

## Goal

Make the Cognitive Router durable enough that a single slow or hung model cannot kill an entire cron job or agent session, by adding abort feedback loops, session-level fallback chains, proactive stall detection, and cron-optimized routing profiles.

## Steps

### 1. Update database schema for abort events
**File:** `db_service.ts`  
**Dependencies:** None

Modify the SQLite database schema in `db_service.ts` to include a new table for storing abort events (provider, model, turnsCompleted, durationMs, sessionKey). Create a migration script if necessary or handle schema creation on startup.

**Outcome:** Database schema supports recording abort events with all required fields, and the application starts successfully with the updated schema.

---

### 2. Implement POST /v1/report/abort endpoint
**File:** `server.ts`  
**Dependencies:** Step 1

Add the `POST /v1/report/abort` endpoint. Validate the request body `{provider, model, turnsCompleted, durationMs, sessionKey}`. Call a new method in `cost_tracker.ts` to record the abort event and update reliability scores.

**Outcome:** The endpoint accepts valid abort reports, persists them to the database, and triggers the cost tracker update logic.

---

### 3. Update cost tracker for abort penalties
**File:** `cost_tracker.ts`  
**Dependencies:** Step 2

Modify `cost_tracker.ts` to handle abort events. Implement logic to decrement the reliability score for the specific provider/model by at least 0.2 per event. Ensure this state is persisted and affects subsequent routing decisions.

**Outcome:** Reported aborts result in a measurable reduction (>= 0.2) of the offending model's reliability score in the cost tracker.

---

### 4. Implement GET /v1/fallback endpoint
**File:** `server.ts`  
**Dependencies:** Step 3

Add the `GET /v1/fallback` endpoint. Accept query parameters provider, model, turn, and reason. Call `router.ts` to find the next-best candidate excluding the failed one.

**Outcome:** The endpoint returns a JSON response with the next best candidate and rationale, ensuring the failed provider/model is excluded.

---

### 5. Implement mid-stream stall detection
**File:** `proxy-stream.ts`  
**Dependencies:** None

Modify `proxy-stream.ts` to detect stalled connections using `ROUTER_STREAM_STALL_TIMEOUT_MS`. If no data is received within the timeout, abort the request and mark the connection as a zombie.

**Outcome:** Active streams that idle for longer than `ROUTER_STREAM_STALL_TIMEOUT_MS` are aborted and tagged as zombies.

---

### 6. Update cost tracker zombie pattern flags
**File:** `cost_tracker.ts`  
**Dependencies:** Step 5

Track zombie connection patterns per provider/model. If a model accumulates 3+ zombie stalls, automatically set the `unstable` or `contextLimited` flag.

**Outcome:** Models causing frequent stalls are flagged as unstable or contextLimited in the cost tracker.

---

### 7. Implement cron-specific routing profile
**File:** `router.ts`  
**Dependencies:** None

Update `router.ts` to accept an optional `routingProfile` parameter in `decide()`. If profile is "cron", adjust the scoring algorithm to weigh proven completion rates higher than raw capability. Update cost_tracker to track completion rates per model/intent.

**Outcome:** The `decide()` function supports a "cron" profile that prioritizes reliable models over theoretically capable ones.

---

### 8. Write integration tests for new features
**File:** `tests/`  
**Dependencies:** Steps 3, 4, 6, 7

Create tests covering: abort endpoint writing to DB, reliability score reduction, fallback endpoint selection, stall detection timeouts, zombie flagging, and cron profile selection.

**Outcome:** All new functionality is covered by automated tests that verify the specific success criteria.

---

### 9. Run full test suite and verify compatibility
**Dependencies:** Step 8

Run `npm test` to ensure all existing Phase 1+2 tests pass and the new tests pass. Verify that the application builds correctly with `tsc`.

**Outcome:** All tests pass, the build succeeds, and no existing functionality is broken.

## Assumptions

- Existing SQLite database setup in `db_service.ts` is accessible and writable
- `ROUTER_STREAM_STALL_TIMEOUT_MS` is defined in the configuration
- Node.js and npm installed in the project environment
- Existing cost_tracker metrics infrastructure can be extended without major refactoring

## Risks

- Modifying the database schema on a live system could cause downtime if not handled carefully
- Aggressive reliability score reduction (0.2) might unfairly deprioritize a generally good model that had a transient issue
- Stall detection logic might mistakenly abort long-running valid computations if the timeout is too aggressive
- Adding new parameters to `decide()` might break compatibility with existing clients not expecting them (though it is optional)

## Success Criteria

1. `POST /v1/report/abort` endpoint exists and writes abort events to SQLite with provider, model, turnsCompleted, durationMs, sessionKey fields
2. `GET /v1/fallback` endpoint returns next-best candidate excluding the failed provider/model, with rationale
3. Abort events reduce reliability score for the reported provider+model by at least 0.2 per event
4. Mid-stream stall detection aborts zombie connections after configurable idle period (`ROUTER_STREAM_STALL_TIMEOUT_MS` already exists at 45s default)
5. Zombie connection tracking feeds into cost_tracker pattern flags — 3+ zombie stalls set unstable or contextLimited flag
6. `decide()` accepts optional routingProfile parameter; "cron" profile biases scoring toward models with proven completion rates over raw capability
7. All existing tests pass (`npm test`)
8. New tests cover abort feedback, fallback chain, stall detection, and cron profile selection

## Scope

**In scope:** `proxy-stream.ts`, `router.ts`, `cost_tracker.ts`, `failure_classifier.ts`, `server.ts`, `config.ts`, `db_service.ts`, `tests/`

**Out of scope:** OpenClaw core (gateway session management), OpenClaw cron subsystem, OpenClaw agent runtime, `benchmark*`, `judge.ts`, `curator.ts`, `ollama_warmth.ts`
