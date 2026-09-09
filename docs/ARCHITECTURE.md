# Architecture

This document describes the system that is implemented in `src/research_repro/`.
It is not a proposal.

## 1. System overview

```text
Goal
→ GoalContract
→ AgentLoop
→ Planner
→ Typed Tool
→ Execution (ExperimentRunner ± FaultInjectingRunner)
→ Observation
→ GoalChecker / SelfEvaluator
→ Recovery (if failed)
→ Working Memory
→ Next Action
```

The loop is a ReAct cycle with a **deterministic evaluator** between observation
and the next plan. The planner never declares success.

## 2. Design goals

1. Execution success is not goal success.
2. The objective cannot be rewritten mid-run.
3. Failure is classified, not merely retried.
4. Illegal high-accuracy results cannot become `best_valid`.
5. Experiment identity is owned by the runner, not the LLM.
6. Every step is inspectable (`run.jsonl`, `memory.json`, experiment dirs).
7. A naive retry baseline can be compared under identical conditions.

## 3. Component architecture

| Component | Module | Responsibility |
|---|---|---|
| GoalContract | `memory/models.py` | Immutable metric, constraints, budget |
| AgentLoop | `agent/loop.py` | Phase machine, step routing, report |
| ReAct | `agent/react.py` | reason → act → observe → evaluate |
| Planner | `agent/planner.py` | One structured tool call + reasoning trace |
| Planning context | `agent/planning_context.py` | Fingerprints, blocking, evidence-based fallback |
| Strategy | `agent/strategy.py` | Self-correcting vs naive post-evaluation policy |
| Tools | `tools/` | Pydantic-typed registry (stub or real) |
| Runner | `experiments/runner.py` | Subprocess, unique IDs, artifacts |
| Faults | `experiments/fault_injection.py` | First-call timeout / fail_once / corrupt / wrong_result |
| GoalChecker | `evaluation/goal_checker.py` | Metric + constraint + regression facts |
| Evaluator | `evaluation/evaluator.py` | Stage 1 facts + Stage 2 LLM rationale |
| Taxonomy | `evaluation/failure_taxonomy.py` | Failure type classification |
| Recovery | `recovery/strategies.py` | One strategy per `FailureType` |
| Memory store | `memory/` | Persist `memory.json` |
| Observability | `observability/` | JSONL events, viewer, benchmark report |
| CLI | `cli.py` | `run`, `show`, `benchmark`, … |
| Benchmark | `evaluations/run_benchmark.py` | 10-goal A/B harness |

## 4. GoalContract

**Input:** natural-language objective plus structured fields from CLI or
`evaluations/goals.yaml`.

**Output:** an immutable `GoalContract` (`memory/models.py`).

Fields actually stored:

- `objective`
- `primary_metric`
- `target_value`
- `success_threshold`
- `criterion` (`MetricCriterion`: direction, target, minimum, tolerance)
- `constraints` (e.g. `latency_under_100ms`)
- `max_experiments`
- `allowed_resources` (`ResourceBudget`)

`meets_contract` is true only when the primary metric **and** every constraint
hold. Latency > 100 ms fails `constraints_hold` regardless of accuracy.

Purpose: prevent the planner from redefining success after seeing 0.961 at
156.9 ms.

## 5. AgentLoop

`AgentLoop.run()` in `agent/loop.py`:

- logs `GOAL_SET`
- creates an initial plan
- loops until a terminal phase, `MAX_STEPS_SAFETY`, `no_progress`, or budget
- on `GOAL_ACHIEVED`, may run independent evaluation before completing
- on `DIAGNOSE_AND_RECOVER`, increments `self_corrections` or `retries`
  depending on whether the strategy is `NaiveStrategy`
- writes `FinalReport`

Naive policy skips recovery events: `_handle_failure` returns the naive
retry/terminate decision without `RECOVERY_STARTED`.

## 6. Planner

```text
LLM (default qwen2.5-coder:7b via Ollama)
 ↓
JSON
 ↓
PlannerOutput (Pydantic)
 ↓
tool name + arguments + ReasoningTrace
```

`ReasoningTrace` fields: `goal_relevance`, `evidence_basis`, `hypothesis`,
`intended_action`. `PlannerOutput` also has `selected_tool`, `tool_arguments`,
`confidence`.

The planner is **not** deterministic. If JSON is malformed, an evidence-based
fallback (`deterministic_next_parameters`) selects an intervention *class*
(reduce layers, then size, then enable normalize). It does not hardcode a
winning configuration.

The planner cannot allocate the authoritative experiment ID.

## 7. Tool interface

Tools subclass `Tool` with a Pydantic `request_model`. The registry validates
before `execute`. Unknown tools raise. Schema errors become
`tool_schema_error`.

Real `run_experiment` (`tools/execution.py`):

1. `allocate_experiment_id(requested)` — runner owns the ID
2. canonicalize parameters (`learning_rate` → `lr`)
3. `runner.run(ExperimentConfig)`
4. collect `metrics.json` and artifacts
5. return accuracy / latency / experiment_id

Stub tools (`tools/stub.py`) are for unit tests and the controlled benchmark.
Stub independent evaluation returns `independently_verified=null`.

## 8. Working memory

`ResearchMemory` is a typed snapshot, not a chat log.

Notable fields:

- `experiments` with `parent_id` lineage
- `best_experiment_id` — **best-valid** (contract satisfied)
- `best_observed_accuracy_id` — best metric even if illegal
- `tested_configs` / `rejected_configs` (fingerprints)
- `failures` / `recoveries`
- `hypotheses`, `evidence`, `discrepancies`
- `reproduction.independently_verified`
- `no_progress_count`, `terminate_reason`
- `budget`

Persisted as `memory.json` after each step.

### best_valid vs best_observed

```text
best_observed  = best metric seen, even if constraints are violated
best_valid     = best result that satisfies the complete GoalContract
```

Real G01: `best_observed` = 0.961 / 156.9 ms (illegal);
`best_valid` = 0.9575 / 76.1 ms (experiment_005).

## 9. Evaluator

Two stages (`evaluation/evaluator.py`):

1. **Deterministic:** GoalChecker, discrepancy analysis, failure taxonomy.
2. **LLM synthesis:** rationale only. It cannot contradict Stage 1 hard facts
   (cannot call a legal accuracy miss `goal_drift`; cannot declare success if
   a constraint failed).

Decisions: `continue`, `diagnose_and_recover`, `goal_achieved`, `terminate`.

Why the evaluator is separate from the planner:

```text
Planner:  "What should I do next?"
Evaluator: "Did what I just did satisfy the goal?"
```

## 10. Recovery engine

`recovery/strategies.py` maps `FailureType` to a named strategy:

| FailureType | Strategy name |
|---|---|
| `tool_crash` | `tool_crash_repair_and_retry` |
| `resource_failure` | `resource_budget_repair_and_retry` |
| `tool_schema_error` | `schema_validate_and_repair` |
| `result_inconsistency` | `independent_metric_reaudit` |
| `goal_drift` | `constraint_reject_and_rollback` |
| `regression` | `best_node_rollback_and_prune` |
| `non_determinism` | `multi_seed_variance_filter` |

Self-correcting strategy calls these. Naive strategy retries the same action
up to `max_retries` (default 2 inner retries; the loop still counts each
diagnose-and-recover as a retry) and never mutates the plan from evidence.

## 11. Experiment runner

`ExperimentRunner`:

- `allocate_experiment_id()` yields `experiment_001`, `experiment_002`, …
- refuses to overwrite a completed directory (`FileExistsError`)
- writes canonical `config.json` **after** the subprocess so aliases match
  what actually ran
- captures stdout/stderr to `stdout.log` / `stderr.log` / `run.log`

```text
Planner-requested ID
        │
        │ ignored if it would collide
        ▼
Runner allocates experiment_NNN
```

This prevents duplicate IDs, accidental overwrite, and ambiguous lineage.

## 12. Fault injection

`FaultInjectingRunner` wraps the real runner. Benchmark goals pass ordinal
keys (`"1"`) so unique IDs still match the first allocated experiment.

| Goal | Fault on first call | Observed in real run |
|---|---|---|
| G07 | `timeout` | `experiment_001` stderr: `Process timed out after 600s` |
| G08 | `fail_once` | `experiment_001` stderr: injected CUDA OOM |
| G09 | `corrupt_metric` | `metrics.json` accuracy 0.99, `_injected_fault: corrupt_metric` |
| G10 | `wrong_result` | `metrics.json` 0.968 / 143.7 ms, `_injected_fault: wrong_result` |

Faults execute at the runner/subprocess layer, not in the scoring script.

## 13. Observability

| Artifact | Purpose |
|---|---|
| `memory.json` | Persistent working state |
| `run.jsonl` | Chronological events (`EventType`) |
| `evidence.jsonl` | Evidence ledger |
| `config.json` | Canonical executed configuration |
| `hypothesis.json` | Why the next experiment was selected |
| `evaluation.json` | Objective evaluation |
| `metrics.json` | Training metrics |
| `stdout.log` / `stderr.log` | Raw process I/O |

CLI: `show`, `list`, `status`, `replay`, `report`, `benchmark-report`.

## 14. Budget and termination

Hard stops:

- `max_experiments` on the goal / `ResourceBudget`
- `max_total_recoveries` / `max_recoveries_per_subtask`
- `no_progress` streak (duplicate configs / inner replans count)
- `MAX_STEPS_SAFETY` in `AgentLoop`
- evaluator `terminate`
- naive retry budget exhausted

Termination is honest: `terminate_reason` is persisted (e.g. `no_progress`).

## 15. Data flow

```text
goals.yaml or CLI objective
        ↓
GoalContract
        ↓
PlannerOutput
        ↓
schema validate / alias repair
        ↓
allocate experiment_NNN
        ↓
train.py subprocess (± fault)
        ↓
metrics + artifacts
        ↓
GoalChecker facts
        ↓
decision
        ↓
recovery (self-correcting) or retry (naive)
        ↓
memory.json + run.jsonl
```

## 16. Design trade-offs

- **Deterministic GoalChecker over LLM-only judging.** Prevents the 7B model
  from calling a legal accuracy miss `goal_drift` or accepting illegal
  high-accuracy configs.
- **Cost-proxy latency instead of a serving harness.** Makes the AND contract
  testable without a production inference stack. Documented as a proxy.
- **Runner-owned IDs.** Breaks planner collisions; requires ordinal fault keys.
- **Stub vs real split.** Stub is fast and does not exercise subprocess faults.
  Real is slow and is the primary evidence.
- **No LangChain / swarm frameworks.** Keeps the loop, memory, and evaluator
  inspectable in this repository.
