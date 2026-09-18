"""Heva recording demo: real MNIST, max_iter 20 -> 40 after an accuracy miss.

Measured locally (seed 42, hidden_size=72, hidden_layers=1, no normalize):
  max_iter=20 -> accuracy 0.9130  (< 0.92)
  max_iter=40 -> accuracy 0.9265  (>= 0.92)

Default 128x2 at max_iter=20 already scores 0.9365, so it cannot show this
correction. The starting architecture is only what makes under-training the
blocking factor. The second run must still change max_iter because of the
observed miss — metrics are not stubbed or hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_repro.cli import run as cli_run  # type: ignore


OBJECTIVE = (
    "Run the existing MNIST image-classification experiment starting from "
    "max_iter=20. Achieve test accuracy at least 0.92. If the observed accuracy "
    "is below 0.92, diagnose that failure, increase max_iter, and rerun the "
    "real training script. Do not assume the required hyperparameters in advance."
)


def main() -> None:
    cli_run(
        objective=OBJECTIVE,
        max_experiments=4,
        target_metric="accuracy",
        target_value=0.92,
        success_threshold=0.0,
        constraint=[],
        model=None,
        base_url=None,
        environment="image_classification",
        stub_tools=False,
        real_tools=True,
        live_papers=False,
        fault=[],
        dry_run=False,
        parameter=[
            "hidden_size=72",
            "hidden_layers=1",
            "max_iter=20",
            "normalize=false",
        ],
    )


if __name__ == "__main__":
    main()
