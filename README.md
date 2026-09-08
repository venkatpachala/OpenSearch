# ResearchRepro

> **Self-Correcting Scientific Experiment Agent**

Given a research objective, the agent discovers relevant literature, extracts an experimental method, implements it, executes experiments in a sandbox, compares observed results against expected/reported results, diagnoses discrepancies, and autonomously recovers/replans until it reaches a verified result or exhausts a bounded recovery budget.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# Test with stub tools (no real ML, no API key needed initially)
research-repro run "Reproduce CIFAR-10 experiment" --stub-tools --max-experiments 3

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
```
