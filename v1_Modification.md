# Self-Correcting Agent — Planner & Recovery Implementation Specification

## Objective

Improve the existing self-correcting agent so that it can successfully complete the current MNIST optimization goal:

> Find a configuration for the existing MNIST image-classification experiment that achieves the highest possible test accuracy while keeping latency below 100 ms. Start from the default configuration, run real experiments, use observed results to guide subsequent experiments, and maintain experiment hypotheses and lineage. Do not assume a particular architecture or parameter configuration in advance. Stop when the experiment budget is exhausted or further experiments provide no meaningful progress, and report the best constraint-valid configuration and the evidence supporting it.

Do NOT change the goal to make the test easier.

The current implementation already has:

* GoalContract
* AgentLoop
* ReAct planner → typed tool → observation → evaluator
* deterministic GoalChecker
* failure taxonomy
* recovery strategies
* working memory
* experiment lineage
* proposal fingerprints
* constraint-aware planner context
* blocked proposal guard
* no-progress budget
* real MNIST subprocess experiments
* JSONL observability

Preserve this architecture. Do not introduce LangChain Agents, AutoGen, CrewAI, LlamaIndex Agents, or a multi-agent swarm.

The purpose of this change is to make the existing planner/recovery loop genuinely evidence-driven.

---

# 1. Required behavioral outcome

The agent should be able to reason through a run approximately like this:

1. Start with default configuration.

2. Run the real MNIST experiment.

3. Observe:

   * accuracy = 0.961
   * latency = 156.9 ms

4. Evaluator identifies:

   * accuracy criterion satisfied
   * latency criterion violated
   * overall goal NOT satisfied
   * failure type = goal_drift

5. Recovery rejects the configuration as constraint-illegal.

6. Planner must propose a materially different configuration that attacks the violated latency constraint.

7. Run a smaller/faster architecture.

8. Observe something like:

   * accuracy = 0.9125
   * latency = 76.1 ms

9. Evaluator identifies:

   * latency satisfied
   * accuracy violated

10. Planner must now switch optimization direction:

    * preserve latency
    * improve accuracy

11. It must NOT repeatedly execute the same failed configuration.

12. It should generate an explicit falsifiable hypothesis, e.g.:

    "Adding normalization may recover accuracy while preserving the current low-complexity architecture and therefore keeping latency below the cap."

13. Execute a materially different configuration.

14. Continue searching based on observed evidence.

15. If a configuration reaches both:

    * accuracy >= 0.95
    * latency <= 100 ms

    it must be considered goal-achieved only after the normal evaluator/verification path.

16. If a tool/schema error occurs, recovery must repair the action rather than simply retrying the identical invalid action.

17. If the recovery/search budget is exhausted, terminate honestly and report the best legal result.

---

# 2. Fix configuration-memory semantics

## Problem

The current fingerprint system prevents repetition of configurations rejected for constraints, but it does not prevent repeated execution of configurations that were valid with respect to constraints but failed the target metric.

Example:

* 64×1, lr=0.0005 → 0.9125 / 76.1 ms
* same configuration → repeated
* same configuration → repeated

This wastes real training runs.

## Required change

Introduce explicit configuration outcome tracking.

Add a concept such as:

```python
ConfigurationRecord
```

or an equivalent structure containing:

```text
fingerprint
config
experiment_id
accuracy
latency_ms
constraint_status
criterion_status
failure_type
hypothesis
outcome
timestamp
```

Possible outcome values:

```text
SUCCESS
CONSTRAINT_VIOLATION
OBJECTIVE_FAILURE
TOOL_FAILURE
INVALID_CONFIGURATION
NO_PROGRESS
```

Do not create unnecessary abstractions if an equivalent existing memory structure can be extended.

## Working-memory requirement

The agent's memory must distinguish:

```text
rejected_configs
```

from:

```text
tested_configs
```

and from:

```text
best_valid_experiment
```

For example:

```text
tested_configs:
    fingerprint -> experiment/outcome

rejected_configs:
    fingerprint -> reason

best_valid_experiment:
    experiment_id + metrics + config
```

A configuration that has already executed and failed the objective should not be proposed again unless the planner explicitly identifies a meaningful reason why the same configuration should be retried.

---

# 3. Define "meaningful change"

The planner must not fool the guard by changing irrelevant metadata.

A proposal is materially different only when at least one experiment-affecting parameter changes.

For the current MNIST environment, experiment-affecting parameters include:

* hidden_size
* hidden_layers
* max_iter
* lr
* batch_size
* normalize
* pca_components
* solver

Do not use experiment ID, ordering, formatting, or argument aliases as evidence of a new configuration.

The fingerprint must be deterministic.

Example:

```text
{}
```

must resolve to the same configuration as the actual default values:

```text
hidden_size=128
hidden_layers=2
max_iter=50
lr=0.001
batch_size=200
normalize=false
pca_components=0
solver=adam
```

Likewise, equivalent argument representations must resolve to the same fingerprint.

---

# 4. Planner must receive previous experiment evidence

Extend PlannerFailureContext / planning context so that after each experiment the planner receives structured evidence.

At minimum provide:

```text
goal
criteria
satisfied criteria
failed criteria
observed metrics
target metrics
constraint status
best valid experiment
previous experiment
previous configuration
previous hypothesis
tested configurations
rejected configurations
recent failures
required_change
```

For the current failure:

```text
accuracy:
    observed = 0.9125
    target = 0.95
    status = FAIL

latency:
    observed = 76.1
    target = 100
    status = PASS
```

The context should explicitly say:

```text
Latency is currently satisfied.
Accuracy is currently the blocking criterion.
The next proposal must primarily target accuracy improvement while preserving latency <= 100ms.
```

Do not make the LLM infer this solely from raw numbers.

---

# 5. Add criterion-aware intervention guidance

Create a deterministic mapping from failed criteria to classes of valid interventions.

Do not prescribe one exact configuration.

For example:

```python
CRITERION_INTERVENTIONS = {
    "latency": [
        "reduce hidden_size",
        "reduce hidden_layers",
        "reduce model complexity",
        "reduce dimensionality"
    ],
    "accuracy": [
        "enable normalization",
        "increase training capacity",
        "increase max_iter",
        "adjust learning rate",
        "increase useful feature representation"
    ]
}
```

The exact list should be derived from the existing environment capabilities.

The important distinction is:

### Constraint failure

If latency fails:

```text
required_change =
    reduce computational cost while preserving as much accuracy as possible
```

### Accuracy failure while latency passes

If accuracy fails:

```text
required_change =
    improve predictive performance without violating latency <= 100ms
```

### Both fail

The planner must consider tradeoffs and should not blindly optimize one metric while ignoring the other.

The planner must not be hardcoded to:

```text
latency failure -> 64x1
```

or:

```text
accuracy failure -> normalize=true
```

Those can be valid candidate interventions, but the planner must discover/select them using evidence.

---

# 6. Make hypotheses falsifiable

The current hypothesis fallback:

> "The proposed experiment configuration should be evaluated."

is not useful.

Every experiment proposal should contain a meaningful hypothesis.

Good:

```text
"Enabling normalization may recover accuracy lost from reducing the network to 64x1 while remaining under the latency cap."
```

Good:

```text
"Increasing max_iter may improve accuracy without increasing parameter-count-based latency because model size is unchanged."
```

Bad:

```text
"The proposed experiment configuration should be evaluated."
```

Bad:

```text
"This configuration may work."
```

## Hypothesis requirements

A hypothesis must:

1. Refer to observed evidence.
2. Identify the expected effect.
3. Identify the constraint that must remain satisfied.
4. Be falsifiable by the resulting experiment.

Represent it in the plan:

```json
{
  "hypothesis": "...",
  "intended_action": "...",
  ...
}
```

If the LLM returns a generic hypothesis, either reject it and request a replan or construct a deterministic evidence-based fallback.

Do not silently accept generic filler.

---

# 7. Planner proposal validation

Before executing a proposal, validate:

## A. Schema validity

All tool arguments must match the actual typed tool schema.

For the current experiment tool, valid CLI parameters include:

```text
--hidden-size
--hidden-layers
--max-iter
--lr
--batch-size
--normalize
--pca-components
--solver
```

The planner must NOT produce:

```text
--learning-rate
```

when the tool expects:

```text
--lr
```

Do not rely on the subprocess to repeatedly discover this.

## B. Configuration validity

Reject:

* impossible values
* unsupported enum values
* malformed types
* unknown experiment parameters

## C. Duplicate configuration

If the fingerprint is already in:

```text
tested_configs
```

and the prior result did not establish a reason to retry, block it.

Return structured feedback:

```text
proposal_blocked:
    reason = duplicate_tested_configuration
    fingerprint = ...
    previous_experiment = ...
    previous_outcome = ...
```

The planner should then replan within the same step.

---

# 8. Tool schema-error recovery must actually repair

Current behavior:

```text
invalid --learning-rate
→ crash
→ increment experiment ID
→ retry same invalid --learning-rate
→ crash
→ retry
→ budget exhausted
```

This is NOT self-correction.

Implement:

```text
tool crash
→ inspect structured error
→ classify error
→ determine repair
→ modify action
→ validate repaired action
→ execute repaired action
```

For an argparse error such as:

```text
unrecognized arguments: --learning-rate 0.001
```

the recovery should identify that the requested semantic parameter is learning rate and map it to the valid tool parameter:

```text
--lr
```

If deterministic repair is possible, do it without another LLM call.

If deterministic repair is not possible:

1. send the error back to planner
2. explicitly request a corrected action
3. validate it
4. execute only if valid

Record:

```text
original_action
error
repair
repaired_action
```

in the recovery event.

---

# 9. Recovery types must remain genuinely distinct

Preserve separate recovery semantics.

At minimum:

## Goal drift

Example:

```text
accuracy good
latency violates constraint
```

Recovery:

```text
reject illegal result
retain useful evidence
change strategy/configuration
```

Do NOT simply retry the same configuration.

## Tool/schema error

Example:

```text
--learning-rate invalid
```

Recovery:

```text
repair action/schema
```

Do NOT merely increment experiment ID.

## Resource/timeout failure

Recovery:

```text
reduce computational burden or adjust execution parameters
```

A timeout should not necessarily produce the same recovery as a schema error.

## Result inconsistency

Recovery should validate/recollect/re-run appropriately.

## Regression

If a new legal experiment is worse than the current best, retain the previous best and alter the search direction.

## Non-determinism

Use repeat/variance logic already present, but ensure repeated execution is intentional and recorded as such.

---

# 10. No-progress accounting

Current behavior allows inner replans to escape the no-progress counter.

Fix this.

Every blocked proposal or materially unproductive planning cycle must contribute to progress accounting.

Define a progress event.

Examples:

```text
NEW_VALID_CONFIGURATION
IMPROVED_OBJECTIVE
FIXED_FAILED_CONSTRAINT
NEW_EVIDENCE
MEANINGFUL_STRATEGY_CHANGE
```

Non-progress examples:

```text
DUPLICATE_CONFIGURATION
SAME_FAILURE_WITHOUT_CHANGE
INVALID_ACTION_REPEATED
IDENTICAL_REJECTED_PROPOSAL
NO_MEANINGFUL_METRIC_CHANGE
```

Maintain:

```text
no_progress_count
```

across both:

* normal planner steps
* inner replans

Reset only after meaningful progress.

When the configured no-progress threshold is reached:

```text
terminate_reason = no_progress
completed = false
```

Do not mark the goal achieved.

---

# 11. "Best" must remain constraint-aware

Keep the existing important behavior:

A high-accuracy but illegal experiment must never become the best goal result.

For example:

```text
0.961 accuracy
156.9ms
```

must NOT beat:

```text
0.9185 accuracy
76.1ms
```

for a goal requiring:

```text
accuracy >= 0.95
latency <= 100ms
```

However, retain the illegal experiment as evidence.

Memory should distinguish:

```text
best_valid_experiment
```

from:

```text
best_observed_accuracy
```

This lets the planner know:

```text
We have demonstrated that the model can reach 96.1%,
but that configuration violates latency.
```

That is useful evidence for future search.

---

# 12. Goal completion semantics

The goal is an AND condition.

Do not allow:

```text
accuracy PASS + latency FAIL
```

to become success.

Do not allow:

```text
latency PASS + accuracy FAIL
```

to become success.

Completion requires all criteria to pass.

The evaluator remains the authority.

The planner must never directly set:

```text
GOAL_ACHIEVED
```

---

# 13. Independent evaluation

Once a candidate satisfies all goal criteria, run the existing independent evaluation path if the architecture requires it.

The final success path should be:

```text
candidate experiment
→ deterministic evaluator
→ all criteria pass
→ independent evaluation
→ final evaluator/verification
→ goal_achieved
```

Do not manufacture independent metrics with random noise.

If the existing `run_independent_evaluation` currently adds Gaussian noise to the reported metric, replace that behavior with a genuine independent evaluation using the available environment artifacts.

If a real independent evaluation cannot be implemented safely in this change, do NOT pretend it is independent. Record the limitation explicitly and leave the campaign inconclusive rather than fabricating verification.

---

# 14. Preserve experiment lineage

Every experiment must record:

```text
experiment_id
parent_id
configuration
hypothesis
reason
trigger
metrics
evaluation
```

Example:

```text
experiment_1
parent = null
hypothesis = baseline

experiment_2
parent = experiment_1
hypothesis = reduce model complexity because latency violated cap

experiment_3
parent = experiment_2
hypothesis = enable normalization to recover accuracy while preserving latency
```

This should make the self-correction path visually obvious in the experiment graph.

---

# 15. Observability requirements

Emit structured events for:

```text
proposal_created
proposal_blocked
hypothesis_created
experiment_started
experiment_completed
criterion_evaluated
failure_detected
recovery_started
recovery_completed
action_repaired
replan
no_progress
goal_achieved
campaign_terminated
```

For every planner decision, retain:

```text
goal_relevance
evidence_basis
hypothesis
intended_action
selected_tool
args
confidence
```

For every recovery, retain:

```text
failure_type
observed_failure
recovery_strategy
original_action
repaired_action
result
```

This is required for later demonstration of the self-correction cases.

---

# 16. Do not fake the experiment environment

The optimization test must continue using the real:

```text
environments/image_classification/train.py
```

and real MNIST metrics.

Do not:

* hardcode 0.956
* hardcode a successful configuration
* inject fake accuracy improvements
* make the evaluator declare success
* modify metrics after execution
* use LLM-generated metrics

The planner may choose flags, but the environment must determine the result.

---

# 17. Remove synthetic-data fallback

Audit:

```text
environments/image_classification/train.py
```

The current implementation has a fallback to synthetic data when MNIST download fails.

For the production/research reproduction path, remove this silent fallback.

Preferred behavior:

```text
MNIST unavailable
→ explicit environment/data failure
→ structured failure
→ recovery or honest termination
```

If synthetic data is needed for unit tests, expose it only through an explicit test/stub mode.

Never silently switch from MNIST to synthetic data during a real benchmark.

This is important for the final README and evaluation credibility.

---

# 18. Tests to add/update

All existing tests must continue passing.

Add focused tests for:

## Duplicate configuration

Given:

```text
experiment A
config X
objective failure
```

planner proposes X again.

Expected:

```text
proposal_blocked
```

and no real experiment executes.

## Constraint-invalid configuration

Given:

```text
X = 0.961 accuracy / 156.9ms
```

for:

```text
accuracy >= .95
latency <= 100
```

Expected:

```text
goal_drift
best_valid != X
```

## Accuracy recovery

Given:

```text
accuracy fail
latency pass
```

ensure planner context says:

```text
accuracy is blocking
preserve latency
```

and the next proposal is materially different.

## Schema recovery

Given:

```text
--learning-rate
```

Expected recovery:

```text
--lr
```

or another valid semantically equivalent repair.

The invalid command must NOT be executed repeatedly.

## Schema recovery exhaustion

If repair is impossible:

```text
terminate honestly
```

rather than loop.

## No-progress

Repeated duplicate proposals must increment:

```text
no_progress_count
```

including inner replans.

## Hypothesis quality

Generic hypothesis:

```text
"The proposed experiment should be evaluated."
```

must not be accepted as a meaningful hypothesis.

## Best-valid semantics

A constraint-invalid high-accuracy experiment must not become:

```text
best_valid_experiment
```

## Goal completion

Test:

```text
accuracy pass + latency fail → not complete
accuracy fail + latency pass → not complete
both pass → candidate complete
```

## Lineage

Ensure every corrective experiment references the experiment that caused the correction.

---

# 19. Integration test

Add one deterministic integration test representing the exact failure sequence:

```text
default
→ accuracy 0.961
→ latency 156.9
→ goal_drift
→ smaller architecture
→ latency 76.1
→ accuracy failure
→ accuracy-focused intervention
→ successful/legal configuration
```

The test should not require a real MNIST download if that would make CI unreliable.

Use the existing test/stub infrastructure for deterministic unit/integration tests, while the actual manual showcase must use the real environment.

The test must verify behavior, not merely final output.

Verify:

```text
goal not falsely completed
illegal baseline rejected
duplicate configuration avoided
accuracy intervention occurs
hypotheses are meaningful
recovery is distinct
lineage exists
no-progress accounting works
```

---

# 20. Manual acceptance test

After implementation, run the real clean experiment.

Use the existing CLI.

Do not modify the goal to mention:

```text
normalize
64x1
17.4ms
```

The planner must discover interventions from evidence.

Expected successful trace should resemble:

```text
Run
  ↓
default 128×2
  ↓
0.961 accuracy / 156.9ms
  ↓
goal_drift
  ↓
constraint recovery
  ↓
different lower-complexity config
  ↓
~76ms latency / accuracy below target
  ↓
accuracy becomes blocking criterion
  ↓
planner forms accuracy-focused hypothesis
  ↓
different configuration
  ↓
accuracy improves while latency remains legal
  ↓
all criteria pass
  ↓
independent verification
  ↓
goal achieved
```

Exact numerical results do not have to match historical runs.

---

# 21. Second manual acceptance test: injected failure

Run the same goal with the existing fault-injection mechanism, for example:

```text
--fault exp_01:timeout
```

Verify that the agent:

1. detects timeout
2. classifies it correctly
3. uses timeout/resource recovery
4. changes execution strategy appropriately
5. continues toward the goal
6. does not consume the entire campaign by blindly retrying the same failed action

This should become one of the eventual self-correction showcase traces.

---

# 22. What NOT to implement

Do not solve this by:

* hardcoding the successful configuration
* adding `normalize=true` automatically whenever accuracy fails
* automatically selecting 64×1 whenever latency fails
* treating every failure as retry
* increasing budgets indefinitely
* allowing the planner to declare success
* hiding failures from logs
* changing evaluator thresholds
* fabricating independent evaluation
* using synthetic data silently
* adding another LLM solely to generate fake hypotheses
* replacing the existing architecture with an agent framework

The system should remain:

```text
one agent loop
+ planner
+ typed tools
+ deterministic evaluator
+ recovery strategy
+ working memory
+ real experiments
```

---

# 23. Success criteria for this implementation

Consider this implementation complete only when all of the following are true:

### Planner

* Uses evaluator evidence.
* Knows which criterion is currently blocking.
* Proposes materially different configurations.
* Does not repeat previously failed configurations.
* Generates falsifiable hypotheses.
* Respects tool schemas.

### Recovery

* Goal drift → strategy/configuration change.
* Schema error → action repair.
* Timeout/resource failure → resource-aware recovery.
* Recovery behavior differs by failure type.
* Recovery does not become blind retry.

### Memory

* Tracks tested configurations.
* Tracks rejected configurations.
* Tracks best legal result.
* Tracks best observed but illegal result.
* Tracks hypotheses.
* Tracks lineage.
* Tracks no-progress.

### Evaluation

* AND semantics remain correct.
* Illegal results never become successful.
* Independent verification is genuine or explicitly marked unavailable.
* Final status is honest.

### Observability

A replay of the run should allow an evaluator to answer:

> What did the agent know?

> What did it believe had failed?

> Why did it choose the next configuration?

> What changed after the failure?

> Why was the previous configuration rejected?

> Why did it stop?

---

# 24. Final deliverables

After implementation, provide:

1. Files changed.
2. Summary of each behavioral change.
3. Test count and result.
4. Clean real-MNIST run result.
5. Whether the goal was actually achieved.
6. Full recovery trace for at least:

   * goal drift
   * schema/tool failure
7. Example hypothesis generated by the planner.
8. Example of a blocked duplicate configuration.
9. Final best-valid experiment.
10. Remaining limitations.

Do not report "success" unless the evaluator actually reached the success state.

The most important principle:

> **The system should improve because it learned from the observed failure, not because the code already knew the answer.**
