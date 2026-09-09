# Failure recovery

Self-correction is **not** retry. After a classified failure the agent mutates
plan or configuration from observed evidence, then executes a **new**
experiment ID.

Naive baseline (`NaiveStrategy`): same planner/tools/goal/seed/budget/faults;
after failure it retries the failed action without recovery strategies.

## Shared pipeline

```text
Failure
 ↓
Detection signal (tool response + GoalChecker)
 ↓
Classification (FailureType)
 ↓
Recovery policy (self-correcting only)
 ↓
State mutation (memory, hypothesis, parameters)
 ↓
Retry with repaired args  or  replan a new configuration
 ↓
Validation (evaluator on the next observation)
```

## goal_drift

**Detection:** accuracy (or primary metric) meets target; a named constraint
fails. Real G01 `experiment_001` `evaluation.json`:

```json
"failure_type": "goal_drift",
"criterion_status": {
  "accuracy": {"observed": 0.9555, "target": 0.95, "status": "PASS"},
  "latency":  {"observed": 417.1,  "target": 100.0, "status": "FAIL"}
}
```

**Classification:** `FailureType.GOAL_DRIFT`

**Recovery:** `constraint_reject_and_rollback` — reject as `best_valid`, keep
as evidence / `best_observed` if the metric is higher.

**Mutation:** planner (or evidence fallback) reduces model complexity.

**Real evidence:** G01 path 256×3 → 128×2 → 128×1. Real 10-goal set:
21 goal-drift recoveries, success rate 1.0.

Naive G01 detected the same drift four times and retried 128×2 0.961/156.9 ms
until `no_progress`.

## Accuracy miss (not goal_drift)

**Detection:** constraints hold; primary metric misses. G01 `experiment_004`:

```text
0.932 / 76.1ms → decision=continue, failure_type=null
outcome=OBJECTIVE_FAILURE
```

**Recovery:** not a taxonomy recovery event. The planner is given
`blocking_criterion=accuracy` and must change that axis.

**Mutation:** G01 hypothesis.json on experiment_005:
`normalize: False -> True`, keep latency ≤ 100 ms.

**Result:** 0.9575 / 76.1 ms, independently verified.

## timeout / tool_crash

**Detection:** subprocess `TimeoutExpired` or non-zero exit. G07
`experiment_001/run.log`:

```text
Process timed out after 600s
```

**Classification:** `tool_crash` (timeout text is included in the error).

**Recovery:** `tool_crash_repair_and_retry`

- new experiment ID (failed dir not overwritten)
- `max_iter` reduced (G07: 100 → 50 on `experiment_002`)
- timeout adjusted (G07 `experiment_002` stored `timeout_seconds: 450`)

**Validation:** G07 continued, then legal `experiment_004` 0.9185 / 76.1 ms,
`independently_verified=true`.

## Schema error

**Detection:** Pydantic validation or argparse unrecognized arguments.

**Classification:** `tool_schema_error`

**Recovery:** `schema_validate_and_repair`

```text
invalid argument
        ↓
known alias?
   YES                 NO
    ↓                   ↓
learning_rate → lr    recovery fails
    ↓                   ↓
retry repaired args   terminate / replan
```

Covered by tests `test_learning_rate_alias_is_repaired_to_lr` and
`test_schema_recovery_exhausts_honestly_when_unrepairable`. Real 10-goal:
6 schema recoveries, success rate 1.0 (self-correcting). Naive G05/G09/G10
retried missing `paper_id` and never launched MNIST.

## fail_once / tool crash

**Detection:** first-call injected crash. G08 `experiment_001/run.log`:

```text
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB.
```

**Classification:** `tool_crash`

**Recovery:** same `tool_crash_repair_and_retry` — new ID, continue.

**Result:** G08 `experiment_004` 0.9575 / 76.1 ms, independently verified.

The aggregate counter `timeout_recoveries = 2 / success 0.5` counts G07
(timeout text in `action_taken`) and G08 (crash strategy without the word
“timeout”). Both recoveries succeeded in the artifacts.

## corrupt_metric

**Detection:** G09 first experiment `metrics.json`:

```json
"accuracy": 0.99,
"latency_ms": 156.9,
"_injected_fault": "corrupt_metric"
```

**Recovery in this run:** treated as an illegal high-accuracy / high-latency
result (`best_observed` 0.99, not `best_valid`). Search continued to legal
`experiment_003` 0.932 / 76.1 ms, independently verified.

`independent_metric_reaudit` exists for true train-vs-eval disagreement; the
real 10-goal summary recorded 0 uses of that named strategy.

## wrong_result

**Detection:** G10 `experiment_001/metrics.json`:

```json
"accuracy": 0.968,
"latency_ms": 143.7,
"_injected_fault": "wrong_result"
```

**Recovery:** constraint reject; not stored as `best_valid`.

**Result:** legal `experiment_003` 0.9575 / 76.1 ms.

## regression

**Detection:** GoalChecker vs current `best_experiment_id` when a legal result
is worse.

**Recovery:** `best_node_rollback_and_prune` — keep previous best, change
search direction.

**Real 10-goal named count:** 0. The strategy is implemented and unit-tested;
this campaign did not emit that recovery name.

## non_determinism

**Detection:** repeated samples / high variance.

**Recovery:** `multi_seed_variance_filter`.

**Real 10-goal named count:** 0. Implemented; not exercised by G01–G10.

## Resource failure

`FailureType.RESOURCE_FAILURE` uses `resource_budget_repair_and_retry`
(shares timeout/OOM repairs with tool crash). Additive to the original six
taxonomy values.

## Genuine strategy change vs naive retry

| | Self-correcting | Naive |
|---|---|---|
| After goal_drift | new config attacking latency | same config again |
| After timeout | new ID + lower `max_iter` | retry same action |
| After schema error | alias repair | retry same invalid args |
| Counts | `self_corrections`, `recovery_attempts` | `retries` only |
| Real 10-goal | 29 recoveries, 10/10 complete | 40 retries, 0/10 complete |
