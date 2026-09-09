# Trace examples

Three real self-correction traces from the 10-goal MNIST campaign.
Paths are under `evaluations/results_real/`. Nothing here is reconstructed
from stub tools.

## Trace 1 — Goal / constraint drift (G01)

Directory: `evaluations/results_real/G01/self_correcting/`

The planner first proposed a **larger** than default network. That is actual
planner behavior, not a hardcoded 128×2 start.

### experiment_001

`experiments/experiment_001/config.json`: `hidden_size=256`, `hidden_layers=3`

`evaluation.json`:

```json
"decision": "diagnose_and_recover",
"failure_type": "goal_drift",
"criterion_status": {
  "accuracy": {"observed": 0.9555, "target": 0.95, "status": "PASS"},
  "latency":  {"observed": 417.1, "target": 100.0, "status": "FAIL"}
}
```

**Why the next action changed:** latency failed the AND contract. Recovery
`constraint_reject_and_rollback` refused this node as `best_valid`. The next
hypothesis reduced `hidden_size` 256→128 and `hidden_layers` 3→2.

### experiment_002

128×2 → 0.961 / 156.9 ms → still `goal_drift`. This becomes
`best_observed_accuracy` (illegal).

### experiment_003

128×1 → 0.9515 / 137.1 ms → still `goal_drift`. Next change: `hidden_size` 128→64.

```text
256×3  0.9555 / 417.1ms  → goal_drift
128×2  0.961  / 156.9ms  → goal_drift  (best_observed, not best_valid)
128×1  0.9515 / 137.1ms  → goal_drift
```

Naive G01 (`G01/naive/`): experiment_001 was default 128×2, 0.961 / 156.9 ms,
same `goal_drift` diagnosis, then three more trains of that illegal config
and `terminate_reason=no_progress`.

## Trace 2 — Accuracy failure then normalize (same G01 campaign)

### experiment_004

`config.json`: 64×1, `normalize=false`

`evaluation.json`:

```json
"decision": "continue",
"failure_type": null,
"constraint_status": "all_met",
"criterion_status": {
  "accuracy": {"observed": 0.932, "target": 0.95, "status": "FAIL"},
  "latency":  {"observed": 76.1, "target": 100.0, "status": "PASS"}
},
"outcome": "OBJECTIVE_FAILURE"
```

This is **not** classified as `goal_drift`. The blocking criterion is accuracy.

### experiment_005

`hypothesis.json` (abridged):

> Given the previous run observed accuracy=0.932 (target 0.95) with legal
> latency=76.1ms, normalize: False -> True may recover accuracy while remaining
> under the latency cap of 100ms.

`metrics.json`: accuracy 0.9575, latency_ms 76.1, inference_ms 2.942

`evaluation.json`: `decision: goal_achieved`

`eval/eval_metrics.json`:

```json
"eval_accuracy": 0.9575,
"training_accuracy": 0.9575,
"discrepancy": 0.0,
"consistent": true,
"independently_verified": true,
"n_eval_samples": 2000
```

```text
0.932 / 76.1ms
 ↓
accuracy FAIL, latency PASS
 ↓
hypothesis: normalize=true (from observed miss, not a baked-in winner)
 ↓
0.9575 / 76.1ms
 ↓
independent rescore 0.9575, discrepancy 0
 ↓
GOAL_ACHIEVED
best_valid = experiment_005
best_observed remains 0.961 / 156.9ms
```

Historical campaign `runs/run_1788916771` showed the same qualitative path
(without the extra 256×3 first step) and is preserved on disk.

## Trace 3 — Timeout / resource recovery (G07)

Directory: `evaluations/results_real/G07/self_correcting/`

Goal G07 injects `faults: {"1": timeout}` on the first allocated experiment.

### experiment_001

Artifacts: `run.log`, `hypothesis.json` only (no `metrics.json` — the process
did not finish).

`run.log`:

```text
=== STDERR ===
Process timed out after 600s
```

The directory was **not** overwritten.

### Recovery

`memory.json` recovery record:

- strategy: `tool_crash_repair_and_retry`
- action_taken: `Adjusted timeout/iteration budget after process timeout; retry as exp_1788937645`

The runner ignored the planner retry id and allocated **`experiment_002`**.

### experiment_002

`config.json`: `max_iter=50`, `timeout_seconds=450`, 128×2

`evaluation.json`: 0.9575 / 156.9 ms → `goal_drift` (accuracy PASS, latency FAIL)

### experiment_004

Legal candidate: 0.9185 / 76.1 ms (target was 0.90). Independent eval:

```json
"eval_accuracy": 0.9185,
"discrepancy": 0.0,
"independently_verified": true
```

`decision`: `goal_achieved`

```text
timeout 600s
 ↓
tool_crash
 ↓
new experiment ID (experiment_001 preserved)
 ↓
max_iter 100 → 50
 ↓
continue through goal_drift
 ↓
legal 0.9185 / 76.1ms, independently verified
```

Historical timeout demo `runs/run_1788917138` is also preserved.

## Related real-fault traces (not the required three)

**G08 fail_once** — `experiment_001/run.log`: injected CUDA OOM; recovered as
`experiment_002` onward; best-valid `experiment_004` 0.9575 / 76.1 ms verified.

**G09 corrupt_metric** — `experiment_001/metrics.json` accuracy 0.99 with
`_injected_fault: corrupt_metric`; best-valid 0.932 / 76.1 ms verified.

**G10 wrong_result** — `experiment_001/metrics.json` 0.968 / 143.7 ms with
`_injected_fault: wrong_result`; best-valid 0.9575 / 76.1 ms.
