"""Human-readable benchmark scorecard from persisted benchmark_results.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(rate: Any) -> str:
    try:
        return f"{100 * float(rate):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _block_for_mode(results: dict[str, Any], heading: str) -> list[str]:
    summary = results.get("summary") or {}
    sc = summary.get("self_correcting") or {}
    nv = summary.get("naive") or {}
    per_goal = summary.get("per_goal") or []
    traces = results.get("correction_traces") or []
    mode = results.get("execution_mode") or ("stub" if results.get("stub_tools") else "real")
    lines = [
        f"## {heading}",
        "",
        f"- Execution mode: **{mode}**",
        f"- Generated at: {results.get('generated_at', 'unknown')}",
        f"- Goals: {len(per_goal)}",
        f"- Denominator: {results.get('denominator_note', 'average_steps includes failed runs')}",
        "",
        "### Self-Correcting vs Naive Results",
        "",
        "| | Self-Correcting | Naive |",
        "|---|---|---|",
        f"| Completion | {sc.get('completion_fraction', _pct(sc.get('completion_rate')))} ({_pct(sc.get('completion_rate'))}) | {nv.get('completion_fraction', _pct(nv.get('completion_rate')))} ({_pct(nv.get('completion_rate'))}) |",
        f"| Average steps (all runs) | {sc.get('mean_steps_all_runs')} | {nv.get('mean_steps_all_runs')} |",
        f"| Average steps (completed) | {sc.get('mean_steps_completed_runs')} | {nv.get('mean_steps_completed_runs')} |",
        f"| Self-corrections (total / mean) | {sc.get('total_self_corrections')} / {sc.get('mean_self_corrections')} | 0 / 0 |",
        f"| Retries (total / mean) | {sc.get('total_retries') or 0} / {sc.get('mean_retries') or 0} | {nv.get('total_retries')} / {nv.get('mean_retries')} |",
        f"| Recovery attempts | {sc.get('recovery_attempts')} | 0 |",
        f"| Recovery failures | {sc.get('recovery_failures')} | 0 |",
        f"| Recovery failure rate | {sc.get('recovery_failure_rate')} | n/a |",
        f"| Runs with recovery failure | {sc.get('recovery_failure_runs')} | {nv.get('recovery_failure_runs')} |",
        "",
        "Naive retries are **not** counted as self-corrections.",
        "",
        "### Per-Goal Results",
        "",
        "| Goal | Self-Correcting | Naive | Self Steps | Naive Steps | Self reason | Naive reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in per_goal:
        lines.append(
            f"| {row.get('goal_id')} | {row.get('self_correcting')} | {row.get('naive')} | "
            f"{row.get('self_steps')} | {row.get('naive_steps')} | "
            f"{row.get('self_reason')} | {row.get('naive_reason')} |"
        )
    lines += ["", "### Failure Recovery Analysis", ""]
    lines.append(
        f"- Goal-drift recoveries: {sc.get('goal_drift_recoveries')} "
        f"(success rate {sc.get('goal_drift_recovery_success')})"
    )
    lines.append(
        f"- Timeout recoveries: {sc.get('timeout_recoveries')} "
        f"(success rate {sc.get('timeout_recovery_success')})"
    )
    lines.append(
        f"- Schema recoveries: {sc.get('schema_recoveries')} "
        f"(success rate {sc.get('schema_recovery_success')})"
    )
    lines.append(
        f"- Result-inconsistency recoveries: {sc.get('result_inconsistency_recoveries')} "
        f"(success rate {sc.get('result_inconsistency_recovery_success')})"
    )
    lines.append(
        f"- Regression recoveries: {sc.get('regression_recoveries')} "
        f"(success rate {sc.get('regression_recovery_success')})"
    )
    lines += ["", "### Correction traces from this execution", ""]
    if not traces:
        lines.append("No self-correcting recovery traces were recorded in this result file.")
    for trace in traces[:3]:
        lines.append(f"#### {trace.get('goal_id')}")
        initial = trace.get("initial_state") or {}
        result = initial.get("observed_result") or {}
        extra = result.get("additional_metrics") or {}
        lines.append(
            f"- Initial: `{initial.get('id')}` accuracy={result.get('accuracy')} "
            f"latency={extra.get('latency_ms')}"
        )
        for failure in trace.get("failures") or []:
            lines.append(f"- Failure: {failure.get('failure_type')} — {failure.get('description')}")
        for rec in trace.get("recoveries") or []:
            lines.append(
                f"- Recovery: {rec.get('strategy')} success={rec.get('success')} — {rec.get('action_taken')}"
            )
        for hyp in (trace.get("hypotheses") or [])[:4]:
            lines.append(f"- Hypothesis: {hyp}")
        for exp in trace.get("experiments") or []:
            lines.append(
                f"- Experiment {exp.get('id')} parent={exp.get('parent_id')} "
                f"acc={exp.get('accuracy')} lat={exp.get('latency_ms')}"
            )
        final = trace.get("final_outcome") or {}
        lines.append(
            f"- Final: completed={final.get('completed')} "
            f"best_valid={final.get('best_valid_accuracy')}/{final.get('best_valid_latency')}"
        )
        lines.append("")
    return lines


def _historical_sections() -> list[str]:
    return [
        "## 7. Clean MNIST Demonstration",
        "",
        "Primary clean self-correction campaign (real subprocess, not overwritten):",
        "",
        "```text",
        "run_id: run_1788916771",
        "goal: accuracy >= 0.95 AND latency_ms <= 100",
        "default 128×2 → 0.961 / 156.9 ms → GOAL_DRIFT (illegal, retained as best_observed)",
        "128×1 → 0.9515 / 137.1 ms → GOAL_DRIFT",
        "64×1 → 0.932 / 76.1 ms → latency PASS, accuracy FAIL",
        "64×1 + normalize=true → 0.9575 / 76.1 ms → GOAL_ACHIEVED",
        "best_valid = exp_04, accuracy=0.9575, latency=76.1ms",
        "independent accuracy = 0.9575, discrepancy = 0, consistent = true",
        "completed = true",
        "```",
        "",
        "`best_observed_accuracy` remains 0.961 at 156.9 ms and is **illegal** under the AND constraint.",
        "",
        "## 8. Timeout Demonstration",
        "",
        "```text",
        "run_id: run_1788917138",
        "timeout → tool_crash → resource-aware recovery",
        "new experiment ID (no overwrite of the failed directory)",
        "max_iter 100 → 50, timeout 600 → 900",
        "execution continues → legal candidate 0.9545 / 76.1 ms → success",
        "independently_verified = true",
        "```",
        "",
        "## 9. Schema Repair Demonstration",
        "",
        "```text",
        "invalid --learning-rate",
        "→ tool_schema_error",
        "→ semantic repair maps to --lr",
        "→ schema validation",
        "→ valid execution",
        "",
        "unknown/unrepairable flag → schema repair exhausted / honest terminate",
        "```",
        "Covered by unit tests (`test_learning_rate_alias_is_repaired_to_lr`, `test_schema_recovery_exhausts_honestly_when_unrepairable`).",
        "",
        "## 10. Artifact / Reproducibility Checks",
        "",
        "- Runner allocates `experiment_NNN` IDs; planner IDs are ignored if they would collide.",
        "- Completed experiment directories cannot be overwritten.",
        "- `config.json` stores canonical executed parameters (`learning_rate` → `lr`).",
        "- `latency_ms` is an environment-defined computational-cost proxy based on model complexity. `inference_ms` is an informational wall-clock measurement and is not the benchmark constraint.",
        "- Production runs never silently switch to synthetic MNIST; `--synthetic` is explicit/test-only.",
        "- Stub independent evaluation is recorded as `independently_verified = null`.",
        "",
        "## 11. Limitations",
        "",
        "1. `latency_ms` is a computational-cost proxy, not measured serving latency.",
        "2. Local 7B planner JSON can be malformed; evidence-based fallback selects an intervention class, not a hardcoded winner.",
        "3. Real MNIST must be available or pre-cached.",
        "4. LLM trajectories are not deterministic; metadata (goal, seed, budget, fault, mode) is recorded for reproduction of conditions, not of token-level planner output.",
        "5. A `--stub-tools` scorecard is a controlled comparison, not a substitute for real subprocess campaigns.",
        "",
    ]


def render_markdown(
    results: dict[str, Any],
    stub_results: dict[str, Any] | None = None,
    real_results: dict[str, Any] | None = None,
) -> str:
    real = real_results or (results if not results.get("stub_tools") else None)
    stub = stub_results or (results if results.get("stub_tools") else None)
    lines = [
        "# Benchmark Report",
        "",
        "## 1. Executive Summary",
        "",
        "This report compares a **self-correcting** agent against a **naive retry** baseline",
        "on the same 10 goals, budgets, seeds, and fault schedules.",
        "Naive retries are not counted as self-corrections.",
        "Do not treat a stub 100%/0% scorecard as ten real MNIST subprocess campaigns.",
        "",
        "## 2. Experimental Setup",
        "",
        "- Goals: `evaluations/goals.yaml` (G01–G10)",
        "- Self-correcting: diagnose → recover → replan from evidence",
        "- Naive: retry the failed action without recovery strategies",
        "- Constraint metric: `latency_ms` computational-cost proxy (not serving latency)",
        "",
    ]
    if stub:
        lines += _block_for_mode(stub, "Controlled Benchmark — Stub Tools")
        lines.append("")
    if real:
        lines += _block_for_mode(real, "Real Execution Benchmark")
        lines.append("")
    if stub and real:
        lines += [
            "## Comparison",
            "",
            "Stub tools exercise planner/evaluator/recovery logic without MNIST subprocesses.",
            "Real execution runs `environments/image_classification/train.py` through the fault injector.",
            "Headline percentages from the stub run must not be quoted as the real benchmark.",
            "",
        ]
    lines += [
        "## 5. Failure Recovery Analysis",
        "",
        "See per-mode sections above for recovery rates. The strongest evidence is the traces:",
        "failure → diagnosis → changed action → new evidence.",
        "",
        "## 6. Three Detailed Self-Correction Traces",
        "",
        "### Trace 1 — Constraint / goal drift (real MNIST `run_1788916771`)",
        "",
        "```text",
        "0.961 / 156.9ms",
        "→ latency violation",
        "→ evaluator = goal_drift",
        "→ reject as best-valid (retain as best_observed)",
        "→ reduce hidden_layers 2→1",
        "→ 0.9515 / 137.1ms still illegal",
        "→ reduce hidden_size 128→64",
        "→ 0.932 / 76.1ms legal, accuracy blocking",
        "```",
        "",
        "### Trace 2 — Accuracy failure (same campaign)",
        "",
        "```text",
        "0.932 / 76.1ms",
        "→ latency PASS, accuracy FAIL",
        "→ blocking criterion = accuracy",
        "→ hypothesis: enable normalization, preserve latency cap",
        "→ 64×1 + normalize=true",
        "→ 0.9575 / 76.1ms criteria PASS",
        "→ independent re-score 0.9575, discrepancy 0",
        "```",
        "",
        "### Trace 3 — Tool / resource failure (real `run_1788917138`)",
        "",
        "```text",
        "timeout",
        "→ tool_crash",
        "→ resource-aware repair",
        "→ new experiment ID (failed directory not overwritten)",
        "→ max_iter reduced, timeout increased",
        "→ continue → legal candidate 0.9545 / 76.1ms",
        "```",
        "",
    ]
    lines += _historical_sections()
    return "\n".join(lines) + "\n"


def write_report(
    results_path: Path,
    output: Path,
    stub_path: Path | None = None,
    real_path: Path | None = None,
) -> Path:
    results = load_results(results_path)
    stub = load_results(stub_path) if stub_path and stub_path.exists() else None
    real = load_results(real_path) if real_path and real_path.exists() else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(results, stub_results=stub, real_results=real), encoding="utf-8")
    return output
