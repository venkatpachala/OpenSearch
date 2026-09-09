# Agent flow

How one campaign executes, using the code in `src/research_repro/agent/`
and the real G01 artifacts as the concrete example.

## End-to-end

```text
1. User supplies an objective (CLI or evaluations/goals.yaml)
2. GoalContract freezes metric, constraints, and budget
3. AgentLoop logs GOAL_SET and creates an initial plan
4. Planner reads memory + goal and emits PlannerOutput
5. Tool schema validates the action
6. Duplicate-configuration guard may block the proposal
7. Tool executes (real: subprocess train.py)
8. Observation is written to memory, evidence, and run.jsonl
9. Stage 1 GoalChecker: execution, fields, metric, constraints, regression
10. Stage 2 evaluator adds rationale; cannot override hard facts
11. Decision: continue | diagnose_and_recover | goal_achieved | terminate
12. On recover: classify failure → strategy mutates plan/config → replan
13. Loop until success, no_progress, or budget exhaustion
14. Optional independent re-score of model.joblib + eval_split.npz
15. FinalReport is emitted
```

## 1. Goal initialization

CLI `research-repro run "..."` or `evaluations/run_benchmark.py` builds a
`GoalContract`. For G01 that is:

- objective: highest possible MNIST accuracy with latency below 100 ms
- `primary_metric`: accuracy, minimum 0.95
- constraint: `latency_under_100ms`
- `max_experiments`: 6

The contract is not updated later.

## 2. Planning

`Planner` produces:

```text
goal_relevance
evidence_basis
hypothesis
intended_action
selected_tool
tool_arguments
confidence
```

If the LLM JSON is invalid, `deterministic_next_parameters` proposes an
intervention class from observed evidence (complexity first, then normalize).

## 3. Schema validation

The registry instantiates the tool's Pydantic request model.

- Known alias: `learning_rate` → `lr` (`planning_context.alias_key`)
- Missing required fields → `tool_schema_error`
- Unrepairable flags → schema recovery returns failure; the loop terminates
  or replans rather than spinning

## 4. Blocked proposals

`assess_proposal` fingerprints the configuration. A previously tested
fingerprint is blocked unless `retry=true` with a justified reason
(`transient_timeout`, `schema_repair`, `corrupted_result`, …).

## 5. Tool execution

Real `run_experiment`:

1. Runner allocates `experiment_NNN`
2. Writes `config.json` / `hypothesis.json`
3. Runs `train.py` as a subprocess with timeout
4. Rewrites canonical `config.json` after the process
5. Returns metrics or a failure (`timed_out`, non-zero exit, missing metrics)

## 6. Observation and evaluation

Evaluation order implemented by GoalChecker + evaluator:

```text
Tool execution
      ↓
Execution success?
      ↓
Required result fields?
      ↓
Metric valid?
      ↓
Constraints satisfied?
      ↓
Progress vs best-valid?
      ↓
Independent verification? (on GOAL_ACHIEVED)
      ↓
Decision
```

## Branch: successful action, goal not yet met

Example: G01 `experiment_004` — 0.932 / 76.1 ms.

- execution success: yes
- constraints: all_met
- accuracy: FAIL vs 0.95
- `failure_type`: null
- `decision`: **continue** (not `goal_drift`)
- planner is told the blocking criterion is accuracy

## Branch: goal drift

Example: G01 `experiment_001` — 0.9555 / 417.1 ms.

- accuracy PASS, latency FAIL
- `failure_type`: `goal_drift`
- `decision`: `diagnose_and_recover`
- recovery: `constraint_reject_and_rollback`
- result is **not** written as `best_valid`
- planner reduces complexity

## Branch: tool / timeout failure

Example: G07 `experiment_001`.

- stderr: `Process timed out after 600s`
- classified `tool_crash`
- recovery: `tool_crash_repair_and_retry`
- new ID (`experiment_002`), `max_iter` 100 → 50
- failed directory is not overwritten

## Branch: schema error

Missing `paper_id` or `--learning-rate`:

- `tool_schema_error`
- `schema_validate_and_repair` maps known aliases
- if the repaired action equals the invalid original, recovery fails
- naive policy retries the same invalid args (G05/G09/G10 naive never trained)

## Branch: goal achieved

Example: G01 `experiment_005` — 0.9575 / 76.1 ms.

- both criteria PASS
- `decision`: `goal_achieved`
- independent eval reloads `model.joblib` + `eval_split.npz`
- G01: `eval_accuracy=0.9575`, `discrepancy=0.0`, `independently_verified=true`
- phase → `COMPLETED`

If independent eval disagrees, the loop treats it as inconsistency and does
not complete.

## Branch: budget / no-progress

Naive G01: four illegal 0.961 / 156.9 ms runs, `terminate_reason=no_progress`.

Self-correcting G01: five distinct configurations, completed in 6 steps.

## What is persisted after each step

- `memory.json` — full `ResearchMemory`
- `run.jsonl` — events (`reason`, `action`, `tool_call`, `self_evaluation`,
  `recovery_started`, …)
- experiment dir artifacts when `run_experiment` ran

The reviewer can replay this with `research-repro replay <run_dir>` or by
reading the JSONL directly.
