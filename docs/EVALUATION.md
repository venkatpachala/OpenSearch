# Evaluation

Scientific evaluation of the implemented agent. Numbers are copied from
generated artifacts, not rewritten by hand.

Primary source: `evaluations/results_real/benchmark_results.json`
(`execution_mode: "real"`, `stub_tools: false`,
`generated_at: 2026-09-09T07:50:08.309240+00:00`).

Controlled source: `evaluations/results/benchmark_results.json`
(`execution_mode: "stub"`, `stub_tools: true`,
`generated_at: 2026-09-09T06:02:03.119388+00:00`).

Do not quote the stub scorecard as the real MNIST result.

## 1. Evaluation questions

- Q1. Does the agent complete multi-step goals under a frozen contract?
- Q2. Does self-correction improve completion versus naive retry?
- Q3. Can it recover from distinct failure classes?
- Q4. Does it refuse invalid high-accuracy results as `best_valid`?
- Q5. Does it terminate under bounded recovery / no-progress?

## 2. Experimental setup

- Environment: `environments/image_classification/train.py` (scikit-learn MLP on MNIST)
- Planner/evaluator model: `qwen2.5-coder:7b` at `http://localhost:11434/v1`
- Goals: `evaluations/goals.yaml` G01–G10
- Seed: 42 for every goal
- Policies: `self_correcting` and `naive`, same goal/seed/budget/fault/tools
- Constraint metric: `latency_ms` computational-cost proxy (not serving latency)
- Independent eval: reload `model.joblib` + `eval_split.npz` (no Gaussian noise)

Average steps use **all runs** as the denominator, including failures
(`denominator_note` in the JSON).

## 3. Benchmark goals

| ID | What it tests | Fault |
|---|---|---|
| G01 | Accuracy ≥ 0.95 and latency < 100 ms from default | none |
| G02 | Latency < 100 ms while accuracy ≥ 0.93 | none |
| G03 | Maximize accuracy with a smaller start config | none |
| G04 | Normalization ablation, accuracy ≥ 0.95 | none |
| G05 | Accuracy ≥ 0.90 with PCA option | none |
| G06 | Accuracy ≥ 0.92 under `max_iter: 40` | none |
| G07 | Recover from first-call timeout | `timeout` |
| G08 | Recover from first-call crash | `fail_once` |
| G09 | Ignore corrupted metric | `corrupt_metric` |
| G10 | Ignore illegal high-accuracy result | `wrong_result` |

Goals were not relaxed to make the test easier.

## 4. Self-correcting policy

After `DIAGNOSE_AND_RECOVER`: classify failure, run the matching recovery
strategy, record evidence, replan a different configuration or repaired
action, allocate a new experiment ID.

## 5. Naive baseline

Same planner, tools, environment, goals, seed, budget, and fault schedule.

Different: post-evaluation policy is retry of the failed action, without
recovery strategies, evidence-driven replanning, or duplicate-aware
corrective search.

## 6. Metrics

Recorded per run in `metrics` (from `run.jsonl` + `memory.json` + `FinalReport`):

`run_id`, `goal_id`, `mode`, `seed`, `budget`, `completed`,
`termination_reason`, `steps`, `tool_calls`, `experiments`,
`self_corrections`, `retries`, `recovery_attempts`, `recovery_failures`,
failure counts, `best_valid_*`, `best_observed_*`, `independently_verified`.

Naive: `self_corrections = 0`, `recovery_attempts = 0`, `retries = N`.

## 7. Results — real MNIST (primary)

```text
10 goals

Self-correcting:
  10/10 completed
  6.3 average steps (all runs)
  29 self-corrections
  29 recovery attempts
  0 recovery failures

Naive:
  0/10 completed
  6.1 average steps (all runs)
  40 retries
  0 self-corrections
  0 recovery attempts
```

### Controlled stub (not the primary result)

Stub tools, no MNIST subprocess: self-correcting 10/10, naive 0/10,
self-corrections 21 vs 0, mean steps 5.7 vs 6.3. Stub naive `retries`
were 0 in that JSON because those runs predated retry accounting on
`FinalReport` for that file; they are still not counted as self-corrections.

## 8. Per-goal results (real)

| Goal | Self-correcting | Naive | Self steps | Naive steps | Self best-valid | Verified | Naive reason |
|---|---|---|---:|---:|---|---|---|
| G01 | PASS | FAIL | 6 | 4 | 0.9575 / 76.1 | true | no_progress |
| G02 | PASS | FAIL | 3 | 4 | 0.932 / 76.1 | null | no_progress |
| G03 | PASS | FAIL | 8 | 7 | 0.9575 / 76.1 | true | no_progress |
| G04 | PASS | FAIL | 3 | 4 | 0.9575 / 76.1 | true | no_progress |
| G05 | PASS | FAIL | 6 | 6 | 0.9575 / 76.1 | true | incomplete (schema retries, 0 trains) |
| G06 | PASS | FAIL | 8 | 7 | 0.932 / 76.1 | true | no_progress |
| G07 | PASS | FAIL | 8 | 7 | 0.9185 / 76.1 | true | no_progress |
| G08 | PASS | FAIL | 7 | 7 | 0.9575 / 76.1 | true | no_progress |
| G09 | PASS | FAIL | 7 | 8 | 0.932 / 76.1 | true | incomplete (schema retries) |
| G10 | PASS | FAIL | 7 | 7 | 0.9575 / 76.1 | null | incomplete (schema retries) |

## 9. Failure recovery results (real, self-correcting)

| Named recovery | Count | Success rate in summary |
|---|---:|---|
| Goal drift (`constraint_reject_and_rollback`) | 21 | 1.0 |
| Schema (`schema_validate_and_repair`) | 6 | 1.0 |
| Timeout/crash (`tool_crash_repair_and_retry`) | 2 | 0.5 in the counter (see note) |
| Result inconsistency (named strategy) | 0 | n/a |
| Regression (named strategy) | 0 | n/a |

Note: G07 timeout and G08 fail_once both used `tool_crash_repair_and_retry`
successfully. The 0.5 figure is a labeling artifact: success is counted only
when `action_taken` contains `"timeout"`. Artifact-level both recoveries
succeeded.

## 10. Interpretation

Self-correction substantially improved **task completion** under the tested
failure scenarios (10/10 vs 0/10), at a small **increase** in average step
count (6.3 vs 6.1). That is not an efficiency win.

The naive agent often observed the same illegal high-latency result the
self-correcting agent saw, then retried it. Completion required changing
configuration after diagnosis.

Q4 is supported by G01: `best_observed` 0.961 / 156.9 ms never became
`best_valid`. G09/G10 injected 0.99 and 0.968 illegal results were also
kept off `best_valid`.

Q5 is supported by naive `no_progress` terminations and by schema-repair
exhaustion tests.

## 11. Limitations

See README and `FINAL_EVALUATION.md`. In this evaluation specifically:

- G02 and G10 completed without `independently_verified=true` in memory.
- Stub tools do not inject real subprocess faults.
- The first real campaign hit disk-full (`[Errno 28]`) during G07 naive;
  `--resume` finished the remaining campaigns without rerunning completed ones.
- LLM trajectories are not token-level reproducible.

## 12. How to regenerate the scorecard (not the 20 campaigns)

```bash
python evaluations/reprocess_results.py \
  --real evaluations/results_real/benchmark_results.json \
  --stub evaluations/results/benchmark_results.json \
  --output evaluations/results/benchmark_report.md
```

Rerunning `python evaluations/run_benchmark.py` without `--stub-tools`
launches 20 real MNIST campaigns and is not required to inspect these results.
