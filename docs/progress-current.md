# Cusco Phase 10 current evidence snapshot

This file tracks only the most recent, high-signal evidence for current work and avoids older historical snapshots.

- Branch: `feat/phase10-compaction`
- Last updated: 2026-08-08

## Canonical artifacts (current)

- `results/smoke-report.json` — deterministic smoke probes including cache/continuation telemetry:
  - `cusco_usage.input_tokens`
  - `cusco_usage.generated_tokens`
  - `cusco_usage.evaluated_tokens`
  - `cusco_usage.cached_tokens`
  - `cusco_usage.prefill` split (`cached_tokens`, `uncached_tokens`, `cached_ratio`, timing)
  - continuity scenarios: `context_reuse_seed`, `context_reuse_followup`, `compacted_successor_continuation`
- `results/phase6c-server.json` — integrated GPU/live server proof including mapped execution, continuation, restart/cancel semantics, and cache accounting

Latest confirmed run (2026-08-08):
- `docker compose -f compose.test.yaml run --rm --no-deps --build api-smoke`
- `results/smoke-report.json` recorded **13 scenarios, 13 passed, 0 failed**
- cache aggregate trend in stats: `cache_ratio = 0.428571` (`prefill_cached_tokens=96`, `prefill_uncached_tokens=128`)

Re-run these to refresh:
- `docker compose -f compose.test.yaml run --rm --no-deps --build api-smoke`
- `docker compose -f compose.test.yaml run --rm test cargo test -p cusco-server`

If you want, I can keep this file machine-updated by replacing the top summary values after each validated run.