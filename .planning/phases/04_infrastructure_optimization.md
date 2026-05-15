# Plan: Phase 4 — Infrastructure Optimization (Autonomous Engine Kill-Switch)

## Objective
Stabilize server startup and eliminate background resource saturation by disabling the autonomous insight engine by default. Background processing will be gated behind the `SYNAPSE_AUTONOMOUS_INSIGHTS` environment variable.

## Proposed Changes

### `src/synapse_mcp/server.py`
- Modify `lifespan_context` to check for `SYNAPSE_AUTONOMOUS_INSIGHTS` before calling `start_autonomous_processing()`.
- Default behavior: Do NOT start background tasks.
- Ensure logging clearly indicates if the engine is enabled or disabled.

### `.env.example`
- Add `SYNAPSE_AUTONOMOUS_INSIGHTS=off` to the example file.

## Tasks

### 04-01: Implement Toggle & Refactor Lifespan
- [x] Update `src/synapse_mcp/server.py` lifespan logic.
- [x] Add `SYNAPSE_AUTONOMOUS_INSIGHTS` to `.env.example`.
- [x] Verify server starts without the 120s delay/background loop unless the flag is set.

## Verification Plan

### Automated Tests
- [x] Create a mock server startup test to verify background tasks are not created by default.
- [x] Verify that setting `SYNAPSE_AUTONOMOUS_INSIGHTS=on` still triggers the task.

### Manual Verification
- [x] Run the server with `npm run dev` (or `mcp dev`).
- [x] Observe logs: should NOT see "Starting autonomous insight processing" by default.
- [x] Set `SYNAPSE_AUTONOMOUS_INSIGHTS=on` in `.env` and verify logs show the delay and startup.
