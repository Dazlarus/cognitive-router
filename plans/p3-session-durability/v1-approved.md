# Phase 3: Session-Level Durability for Cognitive Router

**Plan ID:** `eb85d533-f6df-4170-a236-f42b87f943ef`  
**Status:** Approved (Cycle 2)  
**CodeSpec Session:** `20431b97-6a1b-4775-903a-2e508e6addbf`

## Goal

Make the Cognitive Router durable enough that a single slow or hung model cannot kill an entire cron job or agent session, by adding abort feedback loops, session-level fallback chains, proactive stall detection, and cron-optimized routing profiles.

## Steps

### 1. Create database migration with version tracking
**File:** `db_service.ts`  
**Dependencies:** None

Create a dedicated `migrate_v3()` method that contains all DDL changes wrapped in a transaction. Implement `PRAGMA user_version` checks to manage schema versions. Create the `abort_events` table (provider, model, turnsCompleted, durationMs, sessionKey) and `session_outcomes` table. Use parameterized queries via better-sqlite3 `.run()` with bound parameters — no string concatenation for SQL. Verify WAL mode is enabled. Create a reverse migration script (v3→v2) that drops new tables and columns for rollback.

**Outcome:** Database schema supports recording abort events with all required fields, migration versioning works via PRAGMA user_version, rollback strategy is documented and tested, and the application starts successfully with the updated schema.

---

### 2. Define reliability penalty configuration constant
**File:** `config.ts`  
**Dependencies:** None

Add `RELIABILITY_ABORT_PENALTY` to `config.ts` with a default value of 0.2. This allows tuning the reliability score decrement without code changes. Ensure the constant is exported and accessible to `cost_tracker.ts`.

**Outcome:** `RELIABILITY_ABORT_PENALTY` is defined in config.ts and can be imported by other modules.

---

### 3. Implement POST /v1/report/abort endpoint
**File:** `server.ts`  
**Dependencies:** Steps 1, 2

Add the `POST /v1/report/abort` endpoint. Implement strict schema validation: provider must be configured, model must be non-empty string, turnsCompleted must be non-negative integer, durationMs must be non-negative integer, sessionKey must be non-empty string. On database write failure: (1) log WARN with full context, (2) return HTTP 500 with `{error: 'abort_event_write_failed', details: <sanitized error>}`, (3) apply reliability penalty to in-memory state regardless of DB success.

**Outcome:** The endpoint accepts valid abort reports with strict validation, persists them to SQLite via synchronous writes, applies the reliability penalty to in-memory state, handles DB failures gracefully, and returns appropriate HTTP status codes.

---

### 4. Update cost tracker for abort penalties with bounds
**File:** `cost_tracker.ts`  
**Dependencies:** Step 3

Modify `cost_tracker.ts` to handle abort events using the `RELIABILITY_ABORT_PENALTY` constant. Decrement the reliability score for the specific provider/model by the configured amount per event. Enforce hard bounds (0.0 to 1.0) on reliability scores. Use synchronous write via better-sqlite3. If persistence fails, in-memory state remains the source of truth. Use atomic updates or a mutex/queue for concurrent state modifications.

**Outcome:** Reported aborts result in a measurable reduction of the offending model's reliability score, scores stay within 0.0-1.0 bounds, concurrent updates are handled safely, and state is persisted via synchronous writes.

---

### 5. Implement mid-stream stall detection with abort reporting
**File:** `proxy-stream.ts`  
**Dependencies:** Step 4

Modify `proxy-stream.ts` to detect stalled connections using `ROUTER_STREAM_STALL_TIMEOUT_MS`. If no data is received within the timeout, call `cost_tracker.recordAbort()` directly with provider, model, and context from the stream, then abort the request and mark the connection as a zombie. Explicitly close the underlying connection and clean up memory structures to prevent leaks. This closes the feedback loop: stall → abort event → reliability penalty.

**Outcome:** Active streams that idle longer than `ROUTER_STREAM_STALL_TIMEOUT_MS` are aborted, connections are closed, memory is cleaned up, and `cost_tracker.recordAbort()` is called to apply reliability penalties.

---

### 6. Update cost tracker zombie pattern flags
**File:** `cost_tracker.ts`  
**Dependencies:** Step 5

Track zombie connection patterns per provider/model using atomic updates or a mutex/queue for concurrency safety. If a model accumulates 3+ zombie stalls, automatically set the `unstable` or `contextLimited` flag. Ensure concurrent state modifications are handled safely when multiple streams stall simultaneously.

**Outcome:** Models causing frequent stalls are flagged as unstable or contextLimited in the cost tracker, and concurrent zombie reports are handled safely without race conditions.

---

### 7. Implement GET /v1/fallback endpoint with validation
**File:** `server.ts`  
**Dependencies:** Step 4

Add the `GET /v1/fallback` endpoint. Accept query parameters: provider (must be one of configured providerPriority values), model (non-empty string for exclusion), turn (non-negative integer ≤ 1000), reason (must be one of: abort, timeout, stall, context_overflow). Implement allow-listing validation for provider and model names. Invalid inputs return 400 with descriptive error. Call `router.ts` to find the next-best candidate excluding the failed one.

**Outcome:** The endpoint returns a JSON response with the next best candidate and rationale, excludes the failed provider/model, validates all inputs with allow-listing, and returns 400 for invalid requests.

---

### 8. Implement cron-specific routing profile
**File:** `router.ts`  
**Dependencies:** None

Update `router.ts` to accept an optional `routingProfile` parameter in `decide()`. If profile is "cron", adjust the scoring algorithm to weigh proven completion rates higher than raw capability. Update cost_tracker to track completion rates per model/intent. Ensure the profile parameter is optional and backward compatible.

**Outcome:** The `decide()` function supports a "cron" profile that prioritizes reliable models over theoretically capable ones, and the parameter is optional for backward compatibility.

---

### 9. Write integration tests for new features
**File:** `tests/`  
**Dependencies:** Steps 1, 3, 4, 6, 7, 8

Create tests covering: migration script forward/reverse with PRAGMA user_version, abort endpoint writing to DB with validation, reliability score reduction with 0.0-1.0 bounds enforcement, fallback endpoint selection with allow-list validation, stall detection timeouts with abort reporting, zombie flagging with concurrent updates, cron profile selection, and error handling paths (DB write failures, invalid inputs).

**Outcome:** All new functionality is covered by automated tests that verify the specific success criteria, error paths, and edge cases.

---

### 10. Run full test suite and verify compatibility
**Dependencies:** Step 9

Run `npm test` to ensure all existing Phase 1+2 tests pass and the new tests pass. Verify that the application builds correctly with `tsc`. Confirm WAL mode is enabled in the database setup.

**Outcome:** All tests pass, the build succeeds, no existing functionality is broken, and WAL mode is confirmed enabled.

## Assumptions

- Existing SQLite database setup in `db_service.ts` is accessible and writable
- `ROUTER_STREAM_STALL_TIMEOUT_MS` is defined in the configuration
- Node.js and npm installed in the project environment
- Existing cost_tracker metrics infrastructure can be extended without major refactoring
- `better-sqlite3` is available for parameterized queries and synchronous writes
- WAL mode is already enabled on the existing database
- Zod or similar validation library is available for schema validation

## Risks

- Modifying the database schema on a live system could cause downtime if not handled carefully, even with rollback scripts
- Aggressive reliability score reduction (0.2) might unfairly deprioritize a generally good model that had a transient issue
- Stall detection logic might mistakenly abort long-running valid computations if the timeout is too aggressive
- Adding new parameters to `decide()` might break compatibility with existing clients not expecting them (though it is optional)
- High-frequency abort reporting could cause SQLite write lock contention, though WAL mode should mitigate this
- Race conditions in concurrent stall detection could lead to inconsistent reliability scores if mutex/queue implementation is faulty

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
