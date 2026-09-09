# FINAL IMPLEMENTATION SPECIFICATION

## Harden the Self-Correcting Agent and Build the Evaluation Benchmark

### Context

The core self-correcting agent is now functioning successfully on a real MNIST optimization campaign.

Successful clean run:

```text
run_1788916771
```

Goal:

```text
accuracy >= 0.95
latency_ms <= 100
```

Observed trajectory:

```text
default 128×2
0.961 accuracy / 156.9 ms
        ↓
GOAL_DRIFT
        ↓
128×1
0.9515 accuracy / 137.1 ms
        ↓
GOAL_DRIFT
        ↓
64×1
0.932 accuracy / 76.1 ms
        ↓
accuracy failure / latency success
        ↓
64×1 + normalize=true
0.9575 accuracy / 76.1 ms
        ↓
independent evaluation
        ↓
GOAL_ACHIEVED
```

The agent therefore demonstrated:

* real experiment execution
* constraint-aware evaluation
* goal-drift recovery
* evidence-driven replanning
* configuration memory
* duplicate prevention
* meaningful hypotheses
* schema-error recovery
* timeout recovery
* independent evaluation
* honest goal completion

Current test suite:

```text
181 passed
0 failed
```

The next task is NOT to redesign the architecture.

The next task is to:

1. Harden the remaining weak points.
2. Make experiment artifacts completely reliable.
3. Make the benchmark reproducible.
4. Compare self-correcting behavior against a naive baseline.
5. Produce the evidence required by the assignment.

Do not introduce LangChain Agents, AutoGen, CrewAI, LlamaIndex Agents, or a multi-agent swarm.

---

# PART A — HARDEN EXPERIMENT IDENTITY AND ARTIFACTS

## A1. Remove planner ownership of experiment IDs

Current limitation:

The LLM can reuse an experiment ID, which can cause artifact collisions/overwrites.

This must be fixed at the execution layer.

The planner may propose:

```text
configuration
hypothesis
reason
```

but it must NOT be trusted to determine the canonical experiment ID.

The experiment runner should allocate the ID.

For example:

```text
experiment_001
experiment_002
experiment_003
...
```

or another guaranteed-unique deterministic scheme.

The important property is:

```text
one execution = one unique immutable experiment artifact
```

If the planner requests:

```text
experiment_003
```

but the runner has already used it, the runner must ignore/replace the requested ID with the next generated ID.

Do not overwrite an existing experiment directory.

---

## A2. Artifact immutability

Once an experiment finishes, its artifacts must be immutable.

At minimum:

```text
runs/<run_id>/experiments/<experiment_id>/
    config.json
    hypothesis.json
    metrics.json
    evaluation.json
    stdout.log
    stderr.log
```

If the current structure differs, preserve the existing structure rather than creating unnecessary duplication.

The invariant should be:

> A completed experiment directory can never be silently overwritten by a later experiment.

---

## A3. Record canonical configuration

Every experiment must store the normalized configuration that was actually executed.

Example:

```json
{
  "hidden_size": 64,
  "hidden_layers": 1,
  "max_iter": 100,
  "lr": 0.001,
  "batch_size": 200,
  "normalize": true,
  "pca_components": 0,
  "solver": "adam"
}
```

Do not store only the original LLM arguments.

The canonical configuration must represent the actual execution.

This allows fingerprints to be independently recomputed later.

---

# PART B — STRENGTHEN CONFIGURATION FINGERPRINTING

## B1. Fingerprint must be canonical

The same semantic configuration must always generate the same fingerprint.

For example:

```text
{}
```

must resolve to the same canonical configuration as the explicit defaults.

Aliases must normalize:

```text
learning_rate → lr
```

before fingerprinting.

The fingerprint must include all experiment-affecting parameters.

Current expected fields:

```text
hidden_size
hidden_layers
max_iter
lr
batch_size
normalize
pca_components
solver
```

Sort keys deterministically before generating the fingerprint.

---

## B2. Tested configuration memory

Maintain explicit state:

```text
tested_configs
```

Each entry should contain at least:

```text
fingerprint
experiment_id
configuration
outcome
metrics
```

Possible outcomes:

```text
SUCCESS
CONSTRAINT_VIOLATION
OBJECTIVE_FAILURE
TOOL_FAILURE
INVALID_CONFIGURATION
NO_PROGRESS
```

The agent must not execute the same configuration again unless there is an explicit retry reason.

---

## B3. Intentional retries

There are cases where repeating a configuration is legitimate.

Examples:

* non-determinism verification
* independent evaluation
* transient timeout
* corrupted result
* explicitly requested replication

Therefore, duplicate blocking must distinguish:

```text
unjustified duplicate
```

from:

```text
justified verification/retry
```

Record the reason.

Example:

```json
{
  "retry": true,
  "retry_reason": "non_determinism_verification",
  "parent_experiment": "experiment_003"
}
```

---

# PART C — IMPROVE NO-PROGRESS ACCOUNTING

## C1. Count all unproductive replans

No-progress accounting must include:

* normal planner calls
* inner replans
* duplicate proposals
* repeated failed configurations
* repeated invalid tool calls
* repeated equivalent hypotheses

Current problem:

Inner replans can occur without increasing the no-progress counter.

Fix this.

---

## C2. Define meaningful progress

A step is meaningful if it does at least one of:

```text
NEW_CONFIGURATION
OBJECTIVE_IMPROVEMENT
CONSTRAINT_REPAIR
NEW_EVIDENCE
STRATEGY_CHANGE
GOAL_CRITERION_SATISFIED
```

A step is non-progress if:

```text
DUPLICATE_CONFIGURATION
IDENTICAL_FAILURE
IDENTICAL_PROPOSAL
INVALID_ACTION_REPEATED
NO_MEANINGFUL_METRIC_CHANGE
```

The exact enum names may follow the existing codebase.

---

## C3. Termination

When the no-progress budget is exhausted:

```text
completed = false
termination_reason = "no_progress"
```

The system must still produce the best available evidence.

Never convert no-progress exhaustion into success.

---

# PART D — TOOL SCHEMA SAFETY

## D1. Planner output must be validated before execution

Every proposed tool action must pass:

```text
LLM output
→ JSON parsing
→ Pydantic/tool schema validation
→ configuration normalization
→ fingerprint generation
→ duplicate check
→ execution
```

Only validated actions reach the real environment.

---

## D2. Schema-error repair

Maintain the existing semantic repair:

```text
--learning-rate
        ↓
--lr
```

But generalize the mechanism where practical.

The recovery system should inspect:

```text
tool name
argument name
error message
valid schema
```

and determine whether a deterministic repair is possible.

If repair is possible:

```text
invalid proposal
→ repair
→ validate repaired proposal
→ execute repaired proposal
```

If repair is impossible:

```text
invalid proposal
→ planner receives structured error
→ planner proposes correction
→ validate
→ execute
```

Do not repeatedly execute known-invalid actions.

---

# PART E — FAILURE-SPECIFIC RECOVERY

Preserve distinct recovery strategies.

## E1. Goal drift

Example:

```text
accuracy = 0.961
latency = 156.9
```

Required behavior:

```text
detect constraint violation
→ reject result as goal candidate
→ retain result as evidence
→ modify strategy/configuration
→ continue
```

Do not simply retry the same configuration.

---

## E2. Accuracy failure

Example:

```text
accuracy = 0.932
latency = 76.1
```

Required behavior:

```text
latency satisfied
accuracy blocking
→ preserve latency constraint
→ target accuracy-relevant intervention
→ create falsifiable hypothesis
→ execute materially different configuration
```

Do not repeatedly change irrelevant parameters.

---

## E3. Timeout

Required behavior:

```text
timeout
→ classify resource/tool failure
→ modify resource-related execution strategy
→ continue
```

The recovery must not blindly repeat the same payload.

---

## E4. Schema failure

Required behavior:

```text
invalid tool argument
→ diagnose
→ repair argument
→ validate
→ execute
```

---

## E5. Result inconsistency

If metrics are malformed, missing, or contradictory:

```text
invalid result
→ do not update best_valid
→ collect/re-run/verify
→ continue if possible
```

---

## E6. Regression

If a new legal experiment is worse than the current best:

```text
retain current best
→ record regression
→ use result as evidence
→ choose another search direction
```

Do not lose the previous best.

---

# PART F — WORKING MEMORY

Ensure memory clearly separates:

```text
goal
current phase
current plan
tested_configs
rejected_configs
best_valid_experiment
best_observed_accuracy
hypotheses
failures
recoveries
unresolved_questions
no_progress_count
budget
```

Important distinction:

### best_valid_experiment

Must satisfy all constraints.

### best_observed_accuracy

May violate constraints.

Example:

```text
best_observed_accuracy:
    baseline_experiment_1
    accuracy = 0.961
    latency = 156.9
    legal = false
```

while:

```text
best_valid_experiment:
    exp_04
    accuracy = 0.9575
    latency = 76.1
    legal = true
```

Never merge these concepts.

---

# PART G — HYPOTHESIS QUALITY

Every corrective experiment must have a falsifiable hypothesis.

Required structure:

```text
evidence
→ intervention
→ expected effect
→ preserved constraint
→ falsifier
```

Example:

```text
Observed accuracy=0.932 with latency=76.1ms.
Normalization is currently disabled.
Enabling normalization may improve accuracy while preserving the
current architecture and therefore remaining below the 100ms latency cap.
The hypothesis is falsified if accuracy remains below 0.95 or latency
exceeds 100ms.
```

Do not accept generic filler.

Reject examples such as:

```text
"The proposed experiment should be evaluated."
```

or:

```text
"This configuration may work."
```

---

# PART H — INDEPENDENT EVALUATION

The independent evaluation must use actual experiment artifacts.

Current desired flow:

```text
successful experiment
        ↓
load model.joblib
        ↓
load evaluation dataset
        ↓
re-score independently
        ↓
compare with training/evaluation metrics
        ↓
discrepancy analysis
        ↓
consistent?
        ↓
GOAL_ACHIEVED
```

Do not introduce synthetic noise.

Do not simply copy metrics.json into the independent evaluation result.

The independent evaluator must independently calculate the metric.

Record:

```text
eval_accuracy
reported_accuracy
discrepancy
consistent
independently_verified
```

---

# PART I — REAL DATA INTEGRITY

The real benchmark must use real MNIST.

There must be no silent synthetic fallback.

Required behavior:

```text
real MNIST unavailable
→ explicit data/environment failure
→ recovery or honest termination
```

Synthetic data may exist only behind an explicit test/stub switch.

For example:

```text
--synthetic
```

must be explicit.

A production benchmark without that flag must never silently switch datasets.

Add a regression test confirming this.

---

# PART J — LATENCY SEMANTICS

The current environment computes latency using a parameter-count-based formula.

Do NOT describe this as measured serving latency.

Document the distinction clearly.

If feasible without destabilizing the environment, add actual inference timing.

If not, retain the current latency proxy but rename/document it clearly.

For example:

```text
latency_ms:
    environment-defined computational-cost proxy
```

rather than:

```text
measured inference latency
```

The benchmark must consistently use the same latency definition for all configurations.

Do not change the semantics between self-correcting and baseline runs.

---

# PART K — BENCHMARK FRAMEWORK

Now implement the evaluation benchmark required by the assignment.

The benchmark must compare:

```text
SELF_CORRECTING
```

against:

```text
NAIVE_BASELINE
```

on the same goals.

---

## K1. Self-correcting condition

Use the complete current agent:

```text
planner
→ tool
→ evaluator
→ recovery
→ memory
→ replan
```

---

## K2. Naive baseline

The baseline must use the same:

* tools
* environment
* initial goal
* initial configuration
* experiment budget
* timeout policy where applicable
* evaluation criteria

But remove self-correction.

The simplest acceptable baseline:

```text
planner selects action
→ execute
→ if failure, retry same action
```

It must NOT use:

* recovery strategy
* evidence-driven intervention
* duplicate-aware search
* failure-specific planning
* corrective hypothesis generation

The baseline is intentionally weak.

Do not sabotage it artificially beyond removing self-correction behavior.

---

# PART L — AT LEAST 10 GOALS

Create or update:

```text
evaluations/goals.yaml
```

so it contains at least 10 meaningful goals.

The goals must cover multiple types of behavior.

Do not create ten cosmetic variations of the same goal.

Suggested categories:

```text
G01 — accuracy under latency constraint
G02 — latency optimization while preserving accuracy
G03 — maximize accuracy under fixed complexity
G04 — normalization/ablation-style experiment
G05 — parameter efficiency
G06 — training-budget tradeoff
G07 — injected timeout recovery
G08 — injected schema/tool failure
G09 — result inconsistency/corruption recovery
G10 — regression/non-determinism case
```

Only use goals that are actually supported by the current environment.

Do not claim a goal tests something that the environment cannot meaningfully exercise.

Each goal should specify:

```yaml
id:
description:
criteria:
budget:
faults:
```

Use the existing GoalContract format if already implemented.

---

# PART M — SAME INPUTS FOR BASELINE COMPARISON

For every benchmark goal:

```text
self-correcting(goal_i)
```

and:

```text
naive(goal_i)
```

must start from equivalent conditions.

Record:

```text
goal_id
seed
initial_configuration
fault_injection
budget
```

This allows a fair comparison.

If randomness affects the environment, use controlled seeds where supported.

---

# PART N — BENCHMARK METRICS

For every run record:

```text
completed
steps
experiments
tool_calls
self_corrections
recovery_attempts
recovery_failures
failures_detected
goal_drift_count
schema_error_count
timeout_count
duplicate_block_count
no_progress_count
final_accuracy
final_latency
best_valid_accuracy
best_valid_latency
termination_reason
independently_verified
```

Then aggregate across all goals.

---

# PART O — REQUIRED SUMMARY METRICS

Calculate at minimum:

## Completion rate

```text
completed_goals / total_goals
```

Report separately:

```text
self-correcting completion rate
naive completion rate
```

---

## Average steps

```text
total steps / completed or total runs
```

Use a clearly documented denominator.

Prefer reporting both if useful.

---

## Average self-corrections

```text
total recovery/self-correction events / total runs
```

---

## Recovery failures

Count runs where:

```text
recovery attempted
but recovery failed
```

---

## Failure-specific recovery success

Calculate:

```text
goal drift recovery success
timeout recovery success
schema recovery success
result inconsistency recovery success
```

where those cases exist.

---

# PART P — THREE FULL SELF-CORRECTION CASES

The assignment requires at least three detailed cases.

Produce machine-readable and human-readable traces.

Required examples:

## Case 1 — Goal drift

```text
0.961 / 156.9ms
→ latency violation
→ reject
→ reduce complexity
→ 0.9515 / 137.1ms
→ latency violation
→ reduce complexity
→ 0.932 / 76.1ms
```

---

## Case 2 — Accuracy failure

```text
0.932 / 76.1ms
→ latency satisfied
→ accuracy blocking
→ evidence-based hypothesis
→ normalization enabled
→ 0.9575 / 76.1ms
→ success
```

---

## Case 3 — Timeout/schema/tool failure

Use the actual timeout run or schema-repair test.

Example:

```text
timeout
→ tool_crash
→ resource-aware repair
→ continue
→ successful candidate
```

or:

```text
invalid --learning-rate
→ schema error
→ map to --lr
→ execute corrected action
```

Each case must contain:

```text
initial state
failure
failure classification
agent evidence
hypothesis
recovery
new action
result
final outcome
```

---

# PART Q — REPLAYABLE BENCHMARK RESULTS

Every benchmark run must remain replayable from JSONL logs.

At minimum the replay should answer:

```text
What goal was given?
What did the planner propose?
What evidence did it have?
What tool was called?
What actually happened?
How did the evaluator classify it?
What recovery was selected?
What changed?
Why did the next experiment happen?
Why did the agent stop?
```

Do not rely on terminal output alone.

---

# PART R — BENCHMARK REPORT

Create a benchmark reporting command using the existing observability/reporting architecture.

Example concept:

```text
research-repro benchmark-report
```

or adapt the existing CLI naming.

The report should produce:

```text
Benchmark Summary

Goals: 10

                         Self-Correcting    Naive
Completion Rate          X%                  Y%
Average Steps            X                   Y
Avg Self-Corrections     X                   Y
Recovery Failures        X                   Y
```

Then show per-goal results:

```text
G01  self: PASS   naive: FAIL
G02  self: PASS   naive: PASS
...
```

Then list the detailed correction traces.

---

# PART S — TEST REQUIREMENTS

All existing tests must continue passing.

Target:

```text
pytest
→ 0 failures
```

Add tests for:

### Artifact identity

* planner cannot overwrite experiment
* experiment IDs are unique
* artifacts are immutable

### Fingerprinting

* `{}` equals explicit defaults
* aliases normalize
* ordering does not affect fingerprint

### Duplicate handling

* previously failed config is blocked
* justified retry is allowed

### No-progress

* inner replans count
* repeated invalid proposals count
* termination is honest

### Recovery

* goal drift
* timeout
* schema error
* inconsistent result
* regression

### Independent evaluation

* independently calculated metric
* no noise injection
* discrepancy calculation

### Dataset integrity

* production run cannot silently switch to synthetic data

### Benchmark

* self-correcting and naive use same goal
* same initial state
* metrics aggregate correctly
* completion rate calculation is correct
* per-goal report is deterministic

---

# PART T — DO NOT HARD-CODE THE CURRENT WINNER

This is critical.

Do NOT add:

```text
if accuracy < 0.95:
    normalize = true
```

Do NOT add:

```text
if latency > 100:
    hidden_layers = 1
```

Do NOT hardcode:

```text
64 × 1
```

as the answer.

The existing successful run is evidence that the planner can discover an intervention.

The implementation must preserve the general mechanism:

```text
failure
→ identify blocking criterion
→ generate intervention class
→ choose candidate
→ execute
→ observe
→ evaluate
→ continue
```

The planner can use known valid intervention classes, but it must not bypass evidence-driven search with a single hardcoded winning configuration.

---

# PART U — README / DOCUMENTATION

After the code is complete, update README sections:

## Architecture

Explain:

```text
Goal
 ↓
Planner
 ↓
Typed Tool
 ↓
Real Environment
 ↓
Observation
 ↓
Evaluator
 ↓
Failure Taxonomy
 ↓
Recovery
 ↓
Working Memory
 ↓
Replan
```

---

## Failure taxonomy

Document at least:

```text
goal_drift
tool_schema_error
tool_crash
resource_failure
result_inconsistency
regression
non_determinism
```

For each:

```text
what it means
how detected
how recovered
```

---

## Evaluation

Document:

* number of goals
* self-correcting baseline
* naive baseline
* metrics
* benchmark command
* example results

---

## Limitations

Be explicit about:

1. latency is a parameter-based proxy if actual timing is not implemented
2. local 7B planner may occasionally produce malformed JSON
3. deterministic fallbacks exist for planner robustness
4. real MNIST requires dataset availability
5. benchmark scope is experimental optimization rather than full paper reproduction

Do not hide these limitations.

---

# PART V — FINAL ACCEPTANCE CRITERIA

Do not consider this task complete until:

### Core

* [ ] Existing 181+ tests still pass.
* [ ] Experiment IDs cannot collide.
* [ ] Experiment artifacts cannot be overwritten.
* [ ] Canonical configurations are stored.
* [ ] Duplicate configurations are blocked.
* [ ] Legitimate retries remain possible.
* [ ] No-progress counts inner replans.
* [ ] Schema recovery genuinely repairs actions.
* [ ] Timeout recovery genuinely changes execution strategy.
* [ ] Goal drift recovery changes configuration/search strategy.
* [ ] Best-valid and best-observed remain separate.
* [ ] Independent evaluation uses real artifacts.
* [ ] Synthetic data cannot silently enter production runs.

### Benchmark

* [ ] At least 10 meaningful goals.
* [ ] Self-correcting runner implemented.
* [ ] Naive baseline implemented.
* [ ] Same inputs/conditions used for comparison.
* [ ] Completion rate calculated.
* [ ] Average steps calculated.
* [ ] Self-correction count calculated.
* [ ] Recovery failures calculated.
* [ ] Per-goal comparison generated.
* [ ] At least 3 full correction traces available.
* [ ] Benchmark results replayable from logs.

### Real validation

Run at least:

1. Clean real-MNIST optimization.
2. Timeout-injected real run.
3. Schema-repair test.
4. Full 10-goal benchmark.

Do not fabricate results.

---

# FINAL OUTPUT FROM THE CODING AGENT

When finished, return a concise engineering report containing:

## 1. Implementation

List every changed file and what changed.

## 2. Tests

```text
pytest:
XXX passed
0 failed
```

## 3. Clean run

Include:

```text
run_id
goal
experiments
best_valid
final criteria
independent verification
completion
```

## 4. Failure demonstrations

Include:

```text
goal drift
timeout
schema error
```

with recovery outcomes.

## 5. Benchmark

Include:

```text
number of goals
self-correcting completion rate
naive completion rate
average steps
average self-corrections
recovery failures
```

## 6. Three detailed traces

Show:

```text
failure
→ diagnosis
→ hypothesis
→ recovery
→ next action
→ result
```

## 7. Remaining limitations

Be honest.

---

# Core engineering principle

The implementation must demonstrate:

> The agent does not merely retry until something works.

It must demonstrate:

> The agent observes what went wrong, identifies what criterion is blocking progress, changes its strategy or action accordingly, records the evidence, and uses the new result to decide what to do next.

The successful MNIST run is already evidence that this mechanism works.

The purpose of this phase is to make that behavior:

1. reliable,
2. reproducible,
3. measurable,
4. comparable against a naive baseline,
5. auditable from logs,
6. defensible in the final assignment.
