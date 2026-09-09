# Reproducibility

What a reviewer can do after cloning this repository.

LLM token sequences are **not** claimed to be deterministic. Goal, seed,
budget, fault, mode, and artifacts **are** recorded so conditions can be
replayed even when the 7B planner takes a different first step.

## Environment

Verified while producing this documentation:

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.12.10 (`requires-python = ">=3.11"` in `pyproject.toml`) |
| Default model | `qwen2.5-coder:7b` via Ollama at `http://localhost:11434/v1` |
| ML stack | numpy, scikit-learn (see `pyproject.toml`) |
| MNIST | `sklearn.datasets.fetch_openml("mnist_784")` |

`.env.example`:

```text
LLM_MODEL=qwen2.5-coder:7b
LLM_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
MAX_EXPERIMENTS=8
MAX_RECOVERIES=6
MAX_RECOVERIES_PER_SUBTASK=2
```

## Installation

```bash
pip install -e ".[dev]"
cp .env.example .env
```

Ollama must serve `qwen2.5-coder:7b` for live planner/evaluator calls.
Stub-tool tests do not need a live model for every case; several tests still
construct agent objects.

Console script after install: `research-repro` → `research_repro.cli:main`.
Equivalent: `python -m research_repro.cli`.

## Unit tests

```bash
pytest
```

Last full run during this documentation pass is recorded in
`FINAL_EVALUATION.md`. Expected: **0 failed**. Count at the prior
implementation freeze: **196 passed**.

Do not use `--synthetic` except in tests. Production `train.py` refuses to
silently substitute digits if MNIST is missing.

## Single agent run

```bash
research-repro run "Find a MNIST configuration with accuracy at least 0.95 and latency below 100 ms" --dry-run

research-repro run "Find a MNIST configuration with accuracy at least 0.95 and latency below 100 ms" --stub-tools --max-experiments 3

research-repro run "Find a MNIST configuration with accuracy at least 0.95 and latency below 100 ms" --constraint latency_under_100ms --max-experiments 6
```

Without `--stub-tools` the default path uses real subprocess tools when the
environment is `image_classification`.

Inspect:

```bash
research-repro list
research-repro show <run_id>
research-repro status <run_id>
research-repro replay runs/<run_id>
research-repro report runs/<run_id> --output report.md
```

## Controlled (stub) benchmark

Fast planner/evaluator comparison. **Does not** run MNIST subprocesses and
**does not** inject real runner faults.

```bash
python evaluations/run_benchmark.py --stub-tools --output evaluations/results
```

Existing output: `evaluations/results/benchmark_results.json` (stub_tools true).

## Real benchmark

Launches 20 campaigns (10 goals × 2 policies), each with several MNIST trains.
Wall-clock on the documented machine was on the order of an hour plus a
resume after a disk-full interrupt; plan for longer.

```bash
python evaluations/run_benchmark.py --output evaluations/results_real
```

Do **not** pass `--stub-tools`.

Existing output (do not overwrite unless you intend a new campaign):

```text
evaluations/results_real/benchmark_results.json
evaluations/results_real/G01/…/G10/
```

## Resume behavior

If a campaign dies mid-run (the documented real run hit
`[Errno 28] No space left on device` during G07 naive):

```bash
python evaluations/run_benchmark.py --output evaluations/results_real --resume
```

`--resume` skips directories that already have `current_phase=completed`,
`terminate_reason`, or a `final_report` event. Incomplete dirs are rerun.

Results JSON is rewritten after each campaign so a crash cannot drop earlier
rows.

## Scorecard from persisted JSON

Does not train models:

```bash
python evaluations/reprocess_results.py \
  --real evaluations/results_real/benchmark_results.json \
  --stub evaluations/results/benchmark_results.json \
  --output evaluations/results/benchmark_report.md
```

Or:

```bash
research-repro benchmark-report evaluations/results_real/benchmark_results.json \
  --stub evaluations/results/benchmark_results.json \
  --real evaluations/results_real/benchmark_results.json \
  -o evaluations/results/benchmark_report.md
```

## Output locations

| Path | What it is |
|---|---|
| `evaluations/results/` | Stub benchmark JSON + combined Markdown report |
| `evaluations/results_real/` | Real 10-goal artifacts + JSON |
| `runs/` | Interactive CLI campaigns, including historical `run_1788916771` and `run_1788917138` |
| `environments/image_classification/` | Real train/eval scripts |

## Expected artifacts per real experiment

```text
config.json
hypothesis.json
evaluation.json
metrics.json
stdout.log
stderr.log
run.log
model.joblib          # when training finished
eval_split.npz        # when training finished
eval/eval_metrics.json  # when independent eval ran
```

Campaign-level: `memory.json`, `run.jsonl`, `evidence.jsonl`.

## Commands verified against the CLI

The following were executed while writing this file:

```text
python --version                         → Python 3.12.10
python -m research_repro.cli --help      → run, show, list, status, replay,
                                           report, showcase, benchmark,
                                           benchmark-report, chaos-benchmark
python -m research_repro.cli run --help  → flags including --stub-tools,
                                           --dry-run, --constraint, --fault
python evaluations/run_benchmark.py --help
  → --goals, --output, --goal, --stub-tools, --resume
python evaluations/reprocess_results.py --help
  → --real, --stub, --output
```

`research-repro` is the same Typer app after `pip install -e .`.
