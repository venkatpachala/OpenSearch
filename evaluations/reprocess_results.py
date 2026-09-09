"""Recompute metrics/summary for a finished benchmark_results.json using current code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluations.run_benchmark import extract_traces, metrics_from_run, summarize
from research_repro.observability.benchmark_report import write_report


def reprocess(results_path: Path) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    for row in data.get("runs") or []:
        run_dir = Path(row["run_dir"])
        memory = row.get("memory") or {}
        report = row.get("report") or {}
        mets = metrics_from_run(run_dir, memory, report, row.get("agent_type", "self_correcting"))
        mets["goal_id"] = row.get("goal_id")
        mets["seed"] = row.get("seed")
        mets["budget"] = row.get("budget")
        row["metrics"] = mets
    data["summary"] = summarize(data.get("runs") or [])
    data["correction_traces"] = extract_traces(data.get("runs") or [])
    data["execution_mode"] = "stub" if data.get("stub_tools") else "real"
    data["denominator_note"] = (
        "average_steps = total steps across all runs / total runs (failed runs included)"
    )
    results_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, default=ROOT / "evaluations" / "results_real" / "benchmark_results.json")
    parser.add_argument("--stub", type=Path, default=ROOT / "evaluations" / "results" / "benchmark_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations" / "results" / "benchmark_report.md")
    args = parser.parse_args()
    if args.real.exists():
        reprocess(args.real)
    if args.stub.exists():
        reprocess(args.stub)
    primary = args.real if args.real.exists() else args.stub
    write_report(primary, args.output, stub_path=args.stub if args.stub.exists() else None, real_path=args.real if args.real.exists() else None)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
