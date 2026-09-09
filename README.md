# ResearchRepro

A self-correcting multi-step research and experimentation agent.

ResearchRepro is a goal-driven agent that runs multi-step research and
experimentation with typed tools, persistent working memory, deterministic
evaluation, and autonomous recovery. After every tool call it checks execution
success, result validity, and progress toward a frozen `GoalContract`. When a
failure or constraint violation is detected, it diagnoses the failure, changes
strategy, and continues within bounded recovery budgets. The evaluator — not
the planner — is the only component that can declare `GOAL_ACHIEVED`.

## Validation at a glance

The **primary** validation used real MNIST training subprocesses
(`environments/image_classification/train.py`), not stub tools.

Source: `evaluations/results_real/benchmark_results.json`
(`execution_mode: real`, `stub_tools: false`, generated 2026-09-09T07:50:08Z).

| Metric | Self-correcting | Naive |
|---|---:|---:|
| Goals completed | 10/10 (100%) | 0/10 (0%) |
| Average steps (all runs) | 6.3 | 6.1 |
| Self-corrections | 29 | 0 |
| Recovery attempts | 29 | 0 |
| Recovery failures | 0 | 0 |
| Retries | 0 | 40 |

The baseline uses the same planner, tools, goals, seeds, budgets, faults, and
environment. The only policy difference is that the baseline **retries** after
failure instead of diagnosing and recovering. Naive retries are **not** counted
as self-corrections.

A separate **controlled stub** scorecard (`--stub-tools`) exists under
`evaluations/results/` and must not be quoted as the real MNIST result.

Unit tests at documentation time: **196 passed, 0 failed**.

## Problem

Standard tool-using agents can execute multi-step workflows, but execution
success does not imply goal success.

An experiment can:

- execute successfully but violate a constraint
- return malformed or missing metrics
- time out
- produce an inconsistent or injected result
- regress relative to the current best-valid result
- drift from the original objective

ResearchRepro puts an explicit evaluation and recovery loop around every tool
invocation.

```text
Goal
 ↓
Plan
 ↓
Typed action
 ↓
Tool execution
 ↓
Observation
 ↓
Evaluation
 ↓
 ┌───────────────┐
 │               │
SUCCESS       FAILURE
 │               │
 ▼               ▼
DONE          DIAGNOSE
                 │
                 ▼
              RECOVER
                 │
                 ▼
              REPLAN
                 │
                 └──────→ next action
```

## Architecture

ResearchRepro consists of:

- **GoalContract** — immutable success criteria (metric, target, constraints, budget).
- **AgentLoop** — bounded ReAct loop; routes continue / recover / achieve / terminate.
- **LLM planner** — proposes one typed tool call plus a structured reasoning trace.
- **Typed tool layer** — Pydantic request schemas; invalid args never reach the runner.
- **Working memory** — `memory.json`: experiments, hypotheses, failures, recoveries, fingerprints.
- **Deterministic GoalChecker / evaluator** — Stage 1 facts override LLM interpretation.
- **Recovery engine** — one strategy per failure type (see `docs/FAILURE_RECOVERY.md`).
- **Experiment runner** — subprocess `train.py`; allocates `experiment_NNN` IDs.
- **Fault injector** — first-call timeout / fail_once / corrupt_metric / wrong_result.
- **Observability** — `run.jsonl`, `evidence.jsonl`, benchmark report.
- **Benchmark harness** — `evaluations/run_benchmark.py` (self-correcting vs naive).

The planner proposes. The evaluator decides objective status.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/AGENT_FLOW.md](docs/AGENT_FLOW.md)

## Core execution loop

```text
                    USER GOAL
                       │
                       ▼
                 GoalContract
                       │
                       ▼
                    Planner
                       │
                 structured plan
                       │
                       ▼
                Schema validation
                       │
                       ▼
                 Typed Tool
                       │
                       ▼
                  Execution
                       │
                       ▼
                   Observe
                       │
                       ▼
             Deterministic evaluator
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
       Continue     Recover       Achieved
          │            │             │
          │            ▼             │
          │         Diagnose         │
          │            │             │
          │            ▼             │
          │          Replan           │
          │            │             │
          └────────────┴─────────────┘
                       │
                       ▼
                  Working memory
```

`GoalContract` is never rewritten during a run. That is what stops the planner
from redefining success after it sees a high accuracy that violates latency.

Working memory is not chat history. It stores tested/rejected configuration
fingerprints, `best_experiment_id` (best-valid) vs `best_observed_accuracy_id`
(best metric even if illegal), hypotheses, and recovery records.

## Real experiment path

`environments/image_classification/train.py` trains a scikit-learn
`MLPClassifier` on MNIST (`fetch_openml("mnist_784")`). The agent changes
configuration; it does not edit the training source.

Parameters the planner may change: `hidden_size`, `hidden_layers`, `max_iter`,
`lr`, `batch_size`, `normalize`, `pca_components`, `solver`.

```text
Planner
   ↓
ExperimentConfig
   ↓
Pydantic validation + alias repair (learning_rate → lr)
   ↓
Runner allocates experiment_NNN (planner IDs are not trusted)
   ↓
train.py subprocess
   ↓
metrics.json, config.json, model.joblib, eval_split.npz, stdout/stderr
```

`latency_ms` is an environment-defined **computational-cost proxy**
(`n_params / 10000 * 12 + 15`). It is **not** measured serving latency.
`inference_ms` is an informational wall-clock timing and is not the constraint.

Synthetic digits are used only with explicit `--synthetic` (tests). A real run
with MNIST unavailable fails honestly.

## Self-correction example (real G01)

From `evaluations/results_real/G01/self_correcting/`:

```text
experiment_001  256×3     0.9555 / 417.1ms   → goal_drift (latency)
experiment_002  128×2     0.961  / 156.9ms   → goal_drift (latency)
experiment_003  128×1     0.9515 / 137.1ms   → goal_drift (latency)
experiment_004  64×1      0.932  /  76.1ms   → latency PASS, accuracy FAIL
experiment_005  64×1 + normalize=true
                          0.9575 /  76.1ms   → GOAL_ACHIEVED
independent rescore       0.9575, discrepancy 0
```

`best_observed` remains 0.961 / 156.9ms and is illegal under the AND contract.
`best_valid` is experiment_005.

Naive G01, same seed/budget: four retries of the illegal 128×2 0.961/156.9ms
config, then `no_progress`. No strategy change.

Full traces: [docs/TRACE_EXAMPLES.md](docs/TRACE_EXAMPLES.md)

## Failure taxonomy

| Failure | Detection | Recovery |
|---|---|---|
| Goal drift | Constraint evaluation (accuracy PASS, latency FAIL) | Reject as best-valid; change configuration |
| Accuracy miss | Metric evaluation while constraints hold | Evidence-based hypothesis (not classified as goal_drift) |
| Timeout | Process timeout / `tool_crash` | New experiment ID; reduce `max_iter` / adjust timeout |
| Schema error | Pydantic / argparse | Alias repair (`learning_rate` → `lr`); else terminate |
| Fail-once / tool crash | Tool execution failure | New experiment ID + retry |
| Corrupt metric | Injected / inconsistent metric | Reject as best-valid; continue search |
| Wrong result | Constraint / consistency check | Reject illegal result; continue |
| Regression | Comparison against best-valid | Keep previous best; change search direction |
| Non-determinism | Repeated-sample variance | Multi-seed aggregate / justified retry |

Real 10-goal recoveries (self-correcting): 21 goal-drift, 6 schema, 2
timeout/crash-class (`tool_crash_repair_and_retry`). Details:
[docs/FAILURE_RECOVERY.md](docs/FAILURE_RECOVERY.md)

## Experiment artifacts

Each executed experiment directory is immutable once completed:

| Artifact | Purpose |
|---|---|
| `memory.json` | Persistent working state for the campaign |
| `run.jsonl` | Chronological event trace |
| `evidence.jsonl` | Evidence collected during the run |
| `config.json` | Canonical executed parameters (`learning_rate` stored as `lr`) |
| `hypothesis.json` | Why this experiment was selected |
| `evaluation.json` | Objective evaluator decision |
| `metrics.json` | Training metrics from the subprocess |
| `stdout.log` / `stderr.log` | Raw process output |
| `model.joblib` + `eval_split.npz` | Independent re-score inputs |

The runner allocates `experiment_001`, `experiment_002`, … Planner-requested
IDs are ignored if they would collide. `parent_id` records lineage.

## Benchmark

`evaluations/goals.yaml` defines G01–G10 (latency-constrained accuracy, ablation,
training budget, injected timeout, fail-once, corrupt metric, illegal
high-accuracy drift). Each goal is run with both policies from the same seed,
budget, initial configuration, and fault schedule.

| | Controlled / stub | Real (primary) |
|---|---|---|
| Command | `python evaluations/run_benchmark.py --stub-tools` | `python evaluations/run_benchmark.py` |
| Output | `evaluations/results/` | `evaluations/results_real/` |
| Subprocess MNIST | no | yes |
| Headline | 10/10 vs 0/10 (stub tools) | **10/10 vs 0/10 (real MNIST)** |

Average steps were 6.3 (self-correcting) vs 6.1 (naive). That does **not**
show that self-correction is cheaper. It shows that self-correction
substantially improved **completion** under the tested failures, at a small
increase in average step count.

Methodology: [docs/EVALUATION.md](docs/EVALUATION.md)

## Assignment requirement coverage

| Requirement | Implementation | Evidence |
|---|---|---|
| Multi-step agent | `AgentLoop` + typed tools | Real 10-goal campaign |
| ReAct | reason → act → observe → evaluate | `run.jsonl` |
| Working memory | `ResearchMemory` / `memory.json` | run artifacts |
| Self-monitoring | `GoalChecker` Stage 1 | `evaluation.json` |
| Autonomous recovery | recovery strategies | G01, G07, G08, G09, G10 |
| ≥3 failure modes | taxonomy + `FaultInjectingRunner` | real G07–G10 |
| Bounded recovery | experiment / recovery / no-progress budgets | implementation + tests |
| Broken tool | fault injector on first call | G07 timeout, G08 fail_once |
| ≥10 goals | G01–G10 | `evaluations/goals.yaml` |
| Baseline | `NaiveStrategy` | real 0/10 |
| Evaluation metrics | completion, steps, recoveries, retries | `benchmark_results.json` |
| Inspectable traces | JSONL + experiment dirs | `docs/TRACE_EXAMPLES.md` |
| Typed tools | Pydantic request models | tests |
| Graceful schema failure | alias repair or honest terminate | tests + G03/G06 recoveries |
| Structured logging | `run.jsonl`, `evidence.jsonl` | artifacts |
| Viewer / report | CLI `show` / `report` / `benchmark-report` | generated Markdown |

## Installation and commands

Requires **Python ≥ 3.11** (documented run used 3.12.10), a local Ollama
endpoint with `qwen2.5-coder:7b` (defaults in `.env.example`), and MNIST via
scikit-learn OpenML for real runs.

```bash
pip install -e ".[dev]"
cp .env.example .env
# defaults: LLM_MODEL=qwen2.5-coder:7b, LLM_BASE_URL=http://localhost:11434/v1

pytest

# Single run (stub tools — no MNIST subprocess)
research-repro run "Find a MNIST configuration with accuracy at least 0.95 and latency below 100 ms" --stub-tools --max-experiments 3

# Parse the goal contract and exit
research-repro run "..." --dry-run

# Inspect a run
research-repro list
research-repro show <run_id>
research-repro status <run_id>
research-repro replay runs/<run_id>
research-repro report runs/<run_id> --output report.md

# Controlled stub benchmark
python evaluations/run_benchmark.py --stub-tools --output evaluations/results

# Real 10-goal benchmark (MNIST subprocesses; long-running)
python evaluations/run_benchmark.py --output evaluations/results_real

# Resume after interruption (skip finished campaigns)
python evaluations/run_benchmark.py --output evaluations/results_real --resume

# Combined Markdown scorecard from persisted JSON
python evaluations/reprocess_results.py
research-repro benchmark-report evaluations/results_real/benchmark_results.json -o evaluations/results/benchmark_report.md --stub evaluations/results/benchmark_results.json --real evaluations/results_real/benchmark_results.json
```

`research-repro` is the console script from `pyproject.toml`. Equivalent:
`python -m research_repro.cli`.

Do **not** rerun the real 10-goal campaign unless you intend to spend the
wall-clock of 20 MNIST campaigns. Existing artifacts are the source of truth.

## Limitations

1. `latency_ms` is a computational-cost proxy, not production serving latency.
2. The 7B planner can emit malformed JSON or unexpected initial configs (G01
   started at 256×3 rather than the default 128×2).
3. LLM trajectories are not deterministic. Token-level replay is not claimed.
4. G02 and G10 completed but `independently_verified` remained `null` in memory.
5. The RAG environment is a small authored evaluation, not a production retriever.
6. Literature / methodology extraction is less deeply validated than the MNIST
   experiment loop.
7. The “timeout recoveries 2 / success 0.5” aggregate mixes G07 timeout and G08
   crash into one `tool_crash_repair_and_retry` counter; both recoveries
   succeeded in the artifacts.

## Documentation map

| Doc | Question it answers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How is it designed? |
| [docs/AGENT_FLOW.md](docs/AGENT_FLOW.md) | How does one run execute? |
| [docs/FAILURE_RECOVERY.md](docs/FAILURE_RECOVERY.md) | How does self-correction work? |
| [docs/EVALUATION.md](docs/EVALUATION.md) | How did we evaluate it? |
| [docs/TRACE_EXAMPLES.md](docs/TRACE_EXAMPLES.md) | Show the actual self-correction. |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | How do I reproduce it? |
| [FINAL_EVALUATION.md](FINAL_EVALUATION.md) | Final submission evidence |
| `evaluations/results/benchmark_report.md` | Generated stub + real scorecard |

This project does not add LangChain, CrewAI, AutoGen, LlamaIndex Agents, swarm
orchestration, fake metrics, or synthetic showcase traces.
