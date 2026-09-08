import json
from pathlib import Path

from research_repro.experiments import ChaosFault, ChaosHarness
from research_repro.experiments.runner import ExperimentRunner


def test_chaos_benchmark_produces_six_case_scorecard(tmp_path: Path):
    script = tmp_path / "train.py"
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output-dir'); p.add_argument('--seed')\n"
        "a=p.parse_args(); Path(a.output_dir).mkdir(parents=True, exist_ok=True)\n"
        "Path(a.output_dir, 'metrics.json').write_text(json.dumps({'accuracy': .9}))\n"
    )
    runner = ExperimentRunner(tmp_path / "runs")
    scorecard = ChaosHarness(runner, script, tmp_path / "benchmark").benchmark()
    assert scorecard["total"] == 6
    assert len(scorecard["cases"]) == 6
    assert all("expected_strategy" in row for row in scorecard["cases"])
    assert scorecard["score"] == 1.0

