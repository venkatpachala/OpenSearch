# Benchmark Report

## 1. Executive Summary

This report compares a **self-correcting** agent against a **naive retry** baseline
on the same 10 goals, budgets, seeds, and fault schedules.
Naive retries are not counted as self-corrections.
Do not treat a stub 100%/0% scorecard as ten real MNIST subprocess campaigns.

## 2. Experimental Setup

- Goals: `evaluations/goals.yaml` (G01–G10)
- Self-correcting: diagnose → recover → replan from evidence
- Naive: retry the failed action without recovery strategies
- Constraint metric: `latency_ms` computational-cost proxy (not serving latency)

## Controlled Benchmark — Stub Tools

- Execution mode: **stub**
- Generated at: 2026-09-09T06:02:03.119388+00:00
- Goals: 10
- Denominator: average_steps = total steps across all runs / total runs (failed runs included)

### Self-Correcting vs Naive Results

| | Self-Correcting | Naive |
|---|---|---|
| Completion | 10/10 (100.0%) | 0/10 (0.0%) |
| Average steps (all runs) | 5.7 | 6.3 |
| Average steps (completed) | 5.7 | None |
| Self-corrections (total / mean) | 21 / 2.1 | 0 / 0 |
| Retries (total / mean) | 0 / 0 | 0 / 0.0 |
| Recovery attempts | 21 | 0 |
| Recovery failures | 0 | 0 |
| Recovery failure rate | 0.0 | n/a |
| Runs with recovery failure | 0 | 0 |

Naive retries are **not** counted as self-corrections.

### Per-Goal Results

| Goal | Self-Correcting | Naive | Self Steps | Naive Steps | Self reason | Naive reason |
|---|---|---|---|---|---|---|
| G01 | PASS | FAIL | 3 | 4 | completed | no_progress |
| G02 | PASS | FAIL | 3 | 4 | completed | no_progress |
| G03 | PASS | FAIL | 7 | 7 | completed | no_progress |
| G04 | PASS | FAIL | 3 | 4 | completed | no_progress |
| G05 | PASS | FAIL | 6 | 7 | completed | no_progress |
| G06 | PASS | FAIL | 6 | 6 | completed | no_progress |
| G07 | PASS | FAIL | 7 | 7 | completed | no_progress |
| G08 | PASS | FAIL | 6 | 7 | completed | incomplete |
| G09 | PASS | FAIL | 10 | 10 | completed | no_progress |
| G10 | PASS | FAIL | 6 | 7 | completed | no_progress |

### Failure Recovery Analysis

- Goal-drift recoveries: 20 (success rate 1.0)
- Timeout recoveries: 0 (success rate None)
- Schema recoveries: 1 (success rate 1.0)
- Result-inconsistency recoveries: 0 (success rate None)
- Regression recoveries: 0 (success rate None)

### Correction traces from this execution

#### G01
- Initial: `baseline_experiment` accuracy=0.961 latency=156.9
- Failure: goal_drift — The experiment achieved the target metric but violated a critical constraint (latency_under_100ms).
- Failure: goal_drift — Latency constraint violated despite achieving target accuracy.
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Hypothesis: Running the baseline experiment will provide initial evidence on the current performance of the default configuration, allowing us to understand the starting point and identify potential areas for improvement.
- Hypothesis: Given accuracy=0.961 already meets the target but latency=156.9ms exceeds 100ms, hidden_layers: 2 -> 1 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.97 already meets the target but latency=137.1ms exceeds 100ms, hidden_size: 128 -> 64 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Experiment baseline_experiment parent=None acc=0.961 lat=156.9
- Experiment baseline_experiment parent=baseline_experiment acc=0.97 lat=137.1
- Experiment latency_optimized_experiment parent=baseline_experiment acc=0.9575 lat=76.1
- Final: completed=True best_valid=0.9575/76.1

#### G02
- Initial: `baseline_experiment` accuracy=0.961 latency=156.9
- Failure: goal_drift — The experiment achieved the target accuracy but violated the latency constraint.
- Failure: goal_drift — Latency constraint violated despite achieving target accuracy.
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Hypothesis: Running the baseline experiment will provide initial evidence on both accuracy and latency, helping us understand the current state and identify potential areas for improvement.
- Hypothesis: Given accuracy=0.961 already meets the target but latency=156.9ms exceeds 100ms, hidden_layers: 2 -> 1 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.97 already meets the target but latency=137.1ms exceeds 100ms, hidden_size: 128 -> 64 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Experiment baseline_experiment parent=None acc=0.961 lat=156.9
- Experiment exp_02 parent=baseline_experiment acc=0.97 lat=137.1
- Experiment exp_03 parent=exp_02 acc=0.9575 lat=76.1
- Final: completed=True best_valid=0.9575/76.1

#### G03
- Initial: `exp_1788931875` accuracy=0.961 latency=156.9
- Failure: tool_schema_error — Tool response contains schema error indicating a failure.
- Failure: goal_drift — Goal achieved but constraint violated: latency_under_100ms
- Failure: goal_drift — The experiment achieved the target accuracy but violated the latency constraint, indicating goal drift.
- Recovery: schema_validate_and_repair success=True — Repaired tool arguments from schema/argparse feedback
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Hypothesis: The default/current evidence is accuracy=None, latency=Nonems. Testing a materially different configuration should produce a measurable change in the blocking metric. Falsified if metrics are unchanged.
- Hypothesis: Given accuracy=0.961 already meets the target but latency=156.9ms exceeds 100ms, hidden_layers: 2 -> 1 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.97 already meets the target but latency=137.1ms exceeds 100ms, hidden_size: 128 -> 64 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Experiment exp_1788931875 parent=None acc=0.961 lat=156.9
- Experiment exp_1788931875 parent=exp_1788931875 acc=0.97 lat=137.1
- Experiment exp_1788931876 parent=exp_1788931875 acc=0.9575 lat=76.1
- Final: completed=True best_valid=0.9575/76.1


## Real Execution Benchmark

- Execution mode: **real**
- Generated at: 2026-09-09T07:50:08.309240+00:00
- Goals: 10
- Denominator: average_steps = total steps across all runs / total runs (failed runs included)

### Self-Correcting vs Naive Results

| | Self-Correcting | Naive |
|---|---|---|
| Completion | 10/10 (100.0%) | 0/10 (0.0%) |
| Average steps (all runs) | 6.3 | 6.1 |
| Average steps (completed) | 6.3 | None |
| Self-corrections (total / mean) | 29 / 2.9 | 0 / 0 |
| Retries (total / mean) | 0 / 0 | 40 / 4.0 |
| Recovery attempts | 29 | 0 |
| Recovery failures | 0 | 0 |
| Recovery failure rate | 0.0 | n/a |
| Runs with recovery failure | 0 | 0 |

Naive retries are **not** counted as self-corrections.

### Per-Goal Results

| Goal | Self-Correcting | Naive | Self Steps | Naive Steps | Self reason | Naive reason |
|---|---|---|---|---|---|---|
| G01 | PASS | FAIL | 6 | 4 | completed | no_progress |
| G02 | PASS | FAIL | 3 | 4 | completed | no_progress |
| G03 | PASS | FAIL | 8 | 7 | completed | no_progress |
| G04 | PASS | FAIL | 3 | 4 | completed | no_progress |
| G05 | PASS | FAIL | 6 | 6 | completed | incomplete |
| G06 | PASS | FAIL | 8 | 7 | completed | no_progress |
| G07 | PASS | FAIL | 8 | 7 | completed | no_progress |
| G08 | PASS | FAIL | 7 | 7 | completed | no_progress |
| G09 | PASS | FAIL | 7 | 8 | completed | incomplete |
| G10 | PASS | FAIL | 7 | 7 | completed | incomplete |

### Failure Recovery Analysis

- Goal-drift recoveries: 21 (success rate 1.0)
- Timeout recoveries: 2 (success rate 0.5)
- Schema recoveries: 6 (success rate 1.0)
- Result-inconsistency recoveries: 0 (success rate None)
- Regression recoveries: 0 (success rate None)

### Correction traces from this execution

#### G01
- Initial: `experiment_001` accuracy=0.9555 latency=417.1
- Failure: goal_drift — The metric was achieved, but the latency constraint was violated.
- Failure: tool_schema_error — Tool response contains schema error indicating a failure.
- Failure: goal_drift — The experiment achieved the target accuracy but violated the latency constraint, indicating goal drift.
- Failure: goal_drift — The experiment achieved the target accuracy but violated the latency constraint.
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: schema_validate_and_repair success=True — Repaired tool arguments from schema/argparse feedback
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Hypothesis: Increasing the hidden size and number of hidden layers will improve the model's accuracy without significantly affecting the latency.
- Hypothesis: Given accuracy=0.9555 already meets the target but latency=417.1ms exceeds 100ms, hidden_size: 256 -> 128, hidden_layers: 3 -> 2 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.961 already meets the target but latency=156.9ms exceeds 100ms, hidden_layers: 2 -> 1 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.9515 already meets the target but latency=137.1ms exceeds 100ms, hidden_size: 128 -> 64 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Experiment experiment_001 parent=None acc=0.9555 lat=417.1
- Experiment experiment_002 parent=experiment_001 acc=0.961 lat=156.9
- Experiment experiment_003 parent=experiment_002 acc=0.9515 lat=137.1
- Experiment experiment_004 parent=experiment_003 acc=0.932 lat=76.1
- Experiment experiment_005 parent=experiment_004 acc=0.9575 lat=76.1
- Final: completed=True best_valid=0.9575/76.1

#### G02
- Initial: `experiment_001` accuracy=0.961 latency=156.9
- Failure: goal_drift — The experiment achieved the target metric but violated the latency constraint, indicating goal drift.
- Failure: goal_drift — The experiment achieved the target accuracy but violated the latency constraint, indicating goal drift.
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Hypothesis: Running a baseline experiment will provide initial evidence on the current performance of the default configuration. If the baseline experiment fails to meet the latency constraint, further interventions will be applied to reduce latency without compromising accuracy.
- Hypothesis: Given accuracy=0.961 already meets the target but latency=156.9ms exceeds 100ms, hidden_layers: 2 -> 1 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.9515 already meets the target but latency=137.1ms exceeds 100ms, hidden_size: 128 -> 64 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Experiment experiment_001 parent=None acc=0.961 lat=156.9
- Experiment experiment_002 parent=experiment_001 acc=0.9515 lat=137.1
- Experiment experiment_003 parent=experiment_002 acc=0.932 lat=76.1
- Final: completed=True best_valid=0.932/76.1

#### G03
- Initial: `experiment_001` accuracy=0.961 latency=156.9
- Failure: tool_schema_error — Tool response contains schema error indicating a failure.
- Failure: goal_drift — The experiment achieved the target metric but violated a critical constraint (latency_under_100ms).
- Failure: goal_drift — The experiment achieved the target accuracy but violated the latency constraint.
- Recovery: schema_validate_and_repair success=True — Repaired tool arguments from schema/argparse feedback
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Recovery: constraint_reject_and_rollback success=True — Rejected constraint-violating result
- Hypothesis: The default/current evidence is accuracy=None, latency=Nonems. Testing a materially different configuration should produce a measurable change in the blocking metric. Falsified if metrics are unchanged.
- Hypothesis: Given accuracy=0.961 already meets the target but latency=156.9ms exceeds 100ms, hidden_layers: 2 -> 1 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given accuracy=0.9515 already meets the target but latency=137.1ms exceeds 100ms, hidden_size: 128 -> 64 should reduce computational cost enough to satisfy the latency cap while preserving as much accuracy as possible. Falsified if latency remains above 100ms.
- Hypothesis: Given the previous run observed accuracy=0.932 (target 0.94) with legal latency=76.1ms, normalize: False -> True may recover accuracy while remaining under the latency cap of 100ms. Falsified if accuracy does not improve or latency exceeds 100ms.
- Experiment experiment_001 parent=None acc=0.961 lat=156.9
- Experiment experiment_002 parent=experiment_001 acc=0.9515 lat=137.1
- Experiment experiment_003 parent=experiment_002 acc=0.932 lat=76.1
- Experiment experiment_004 parent=experiment_002 acc=0.9575 lat=76.1
- Final: completed=True best_valid=0.9575/76.1


## Comparison

Stub tools exercise planner/evaluator/recovery logic without MNIST subprocesses.
Real execution runs `environments/image_classification/train.py` through the fault injector.
Headline percentages from the stub run must not be quoted as the real benchmark.

## 5. Failure Recovery Analysis

See per-mode sections above for recovery rates. The strongest evidence is the traces:
failure → diagnosis → changed action → new evidence.

## 6. Three Detailed Self-Correction Traces

### Trace 1 — Constraint / goal drift (real MNIST `run_1788916771`)

```text
0.961 / 156.9ms
→ latency violation
→ evaluator = goal_drift
→ reject as best-valid (retain as best_observed)
→ reduce hidden_layers 2→1
→ 0.9515 / 137.1ms still illegal
→ reduce hidden_size 128→64
→ 0.932 / 76.1ms legal, accuracy blocking
```

### Trace 2 — Accuracy failure (same campaign)

```text
0.932 / 76.1ms
→ latency PASS, accuracy FAIL
→ blocking criterion = accuracy
→ hypothesis: enable normalization, preserve latency cap
→ 64×1 + normalize=true
→ 0.9575 / 76.1ms criteria PASS
→ independent re-score 0.9575, discrepancy 0
```

### Trace 3 — Tool / resource failure (real `run_1788917138`)

```text
timeout
→ tool_crash
→ resource-aware repair
→ new experiment ID (failed directory not overwritten)
→ max_iter reduced, timeout increased
→ continue → legal candidate 0.9545 / 76.1ms
```

## 7. Clean MNIST Demonstration

Primary clean self-correction campaign (real subprocess, not overwritten):

```text
run_id: run_1788916771
goal: accuracy >= 0.95 AND latency_ms <= 100
default 128×2 → 0.961 / 156.9 ms → GOAL_DRIFT (illegal, retained as best_observed)
128×1 → 0.9515 / 137.1 ms → GOAL_DRIFT
64×1 → 0.932 / 76.1 ms → latency PASS, accuracy FAIL
64×1 + normalize=true → 0.9575 / 76.1 ms → GOAL_ACHIEVED
best_valid = exp_04, accuracy=0.9575, latency=76.1ms
independent accuracy = 0.9575, discrepancy = 0, consistent = true
completed = true
```

`best_observed_accuracy` remains 0.961 at 156.9 ms and is **illegal** under the AND constraint.

## 8. Timeout Demonstration

```text
run_id: run_1788917138
timeout → tool_crash → resource-aware recovery
new experiment ID (no overwrite of the failed directory)
max_iter 100 → 50, timeout 600 → 900
execution continues → legal candidate 0.9545 / 76.1 ms → success
independently_verified = true
```

## 9. Schema Repair Demonstration

```text
invalid --learning-rate
→ tool_schema_error
→ semantic repair maps to --lr
→ schema validation
→ valid execution

unknown/unrepairable flag → schema repair exhausted / honest terminate
```
Covered by unit tests (`test_learning_rate_alias_is_repaired_to_lr`, `test_schema_recovery_exhausts_honestly_when_unrepairable`).

## 10. Artifact / Reproducibility Checks

- Runner allocates `experiment_NNN` IDs; planner IDs are ignored if they would collide.
- Completed experiment directories cannot be overwritten.
- `config.json` stores canonical executed parameters (`learning_rate` → `lr`).
- `latency_ms` is an environment-defined computational-cost proxy based on model complexity. `inference_ms` is an informational wall-clock measurement and is not the benchmark constraint.
- Production runs never silently switch to synthetic MNIST; `--synthetic` is explicit/test-only.
- Stub independent evaluation is recorded as `independently_verified = null`.

## 11. Limitations

1. `latency_ms` is a computational-cost proxy, not measured serving latency.
2. Local 7B planner JSON can be malformed; evidence-based fallback selects an intervention class, not a hardcoded winner.
3. Real MNIST must be available or pre-cached.
4. LLM trajectories are not deterministic; metadata (goal, seed, budget, fault, mode) is recorded for reproduction of conditions, not of token-level planner output.
5. A `--stub-tools` scorecard is a controlled comparison, not a substitute for real subprocess campaigns.

