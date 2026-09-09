# OpenSearch

> **Self-Correcting Scientific Experiment Agent**

Given a research objective, the agent discovers relevant literature, extracts an experimental method, implements it, executes experiments in a sandbox, compares observed results against expected/reported results, diagnoses discrepancies, and autonomously recovers/replans until it reaches a verified result or exhausts a bounded recovery budget.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# Test with stub tools (no real ML, no API key needed initially)
research-repro run "Reproduce MNIST experiment" --stub-tools --max-experiments 3

# View the run timeline
research-repro show <run_id>
```

## Architecture

```
USER GOAL → GOAL CONTRACT → REACT LOOP → SELF EVALUATOR
                                ↓               ↓
                          TYPED TOOLS      SUCCESS / FAILURE
                                ↓               ↓
                      EXPERIMENT SANDBOX   DIAGNOSER → RECOVERY → REPLANNER
                                                ↓
                                       EVIDENCE LEDGER → FINAL REPORT
```

See `docs/architecture.md` for the full design.

## Commands

```bash
research-repro run "<objective>"              # Start a new run
research-repro run "<objective>" --dry-run    # Parse goal and exit
research-repro run "<objective>" --stub-tools # Use stub tools for testing
research-repro show <run_id>                  # View event timeline
research-repro list                           # List all runs
research-repro status <run_id>                # Phase + budget status
research-repro replay runs/<run_id>           # Replay persisted JSONL events
research-repro replay runs/<run_id> --failures
research-repro report <run_id> --output report.html
python evaluations/run_benchmark.py           # Real 10-goal A/B benchmark
```

## Real execution requirement

Production runs use the real Ollama/LLM planner and evaluator, the shared
typed `ToolRegistry`, subprocess experiment environments, Git checkpoints,
and persisted `run.jsonl`, `memory.json`, and `evidence.jsonl` artifacts.
Benchmark results and reports are derived from those artifacts. No production
benchmark outcome, metric, recovery, or showcase trace is hard-coded.

The benchmark runs each goal with both `self_correcting` and `naive` policies
against isolated directories. Both policies use the same model, evaluator,
tools, sandbox, budgets, and explicit fault configuration. The only policy
difference is post-evaluation behavior: self-correction diagnoses and recovers;
the naive control retries the exact failed action without replanning.

## Why this is not retry

Retry repeats an action. ResearchRepro classifies the failure, records the
observed evidence, updates the hypothesis and plan, performs a targeted
recovery, and independently verifies the next result. Failed runs remain in
the evidence ledger and are reported honestly when budgets are exhausted.

## Scientific verdict

Metric proximity alone is not a confirmed reproduction. The persisted
assessment separates metric success, methodology alignment, and independent
verification, producing `CONFIRMED`, `INCONCLUSIVE`, or `REFUTED`.

## Scope limits

The project intentionally does not add LangChain, CrewAI, AutoGen, LlamaIndex
Agents, vector-database infrastructure, swarm orchestration, synthetic
benchmark generators, fake metrics, or synthetic showcase traces.
