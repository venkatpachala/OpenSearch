"""Select actual benchmark runs containing real recovery events."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def select_showcase(benchmark_results: Path, output_dir: Path, wanted: tuple[str, ...] = ("tool_crash", "result_inconsistency", "regression")) -> list[Path]:
    data = json.loads(benchmark_results.read_text(encoding="utf-8"))
    selected: list[Path] = []
    for failure_type in wanted:
        for row in data.get("runs", []):
            run_dir = Path(row["run_dir"])
            log = run_dir / "run.jsonl"
            if not log.exists(): continue
            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            found = any(e.get("event_type") == "failure_detected" and e.get("payload", {}).get("failure_type") == failure_type for e in events)
            recovered = any(e.get("event_type") == "recovery_completed" and e.get("payload", {}).get("success") for e in events)
            if found and recovered:
                destination = output_dir / f"trace_{failure_type}"
                if destination.exists(): shutil.rmtree(destination)
                shutil.copytree(run_dir, destination)
                selected.append(destination)
                break
    return selected

