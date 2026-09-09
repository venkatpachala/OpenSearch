"""Launch the MNIST self-correction smoke test without shell-quoting issues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_repro.cli import run as cli_run  # type: ignore


OBJECTIVE = (
    "Improve the existing MNIST image-classification experiment to achieve at least "
    "96% test accuracy while keeping simulated latency below 100 ms. Start from the "
    "baseline configuration, run experiments, evaluate the observed results against "
    "the goal, identify why the current configuration is insufficient, and autonomously "
    "change the experiment configuration and rerun until the goal is achieved or the "
    "experiment/recovery budget is exhausted. Do not assume the required hyperparameters "
    "in advance."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", action="append", default=[])
    args = parser.parse_args()
    cli_run(
        objective=OBJECTIVE,
        max_experiments=8,
        target_metric="accuracy",
        target_value=0.96,
        success_threshold=0.0,
        constraint=["latency_under_100ms"],
        model=None,
        base_url=None,
        environment="image_classification",
        stub_tools=False,
        real_tools=True,
        live_papers=False,
        fault=args.fault,
        dry_run=False,
    )


if __name__ == "__main__":
    main()
