#!/usr/bin/env python3
"""
Independent Evaluator — re-scores persisted training artifacts.

Loads the saved model and holdout split written by train.py and computes
metrics in a separate process. Does not add noise and does not invent scores.
If artifacts are missing, fail honestly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent Evaluation")
    p.add_argument("--model-dir", required=True, help="Directory with model artifacts")
    p.add_argument("--output-dir", required=True, help="Directory to write eval metrics")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = model_dir / "metrics.json"
    model_path = model_dir / "model.joblib"
    split_path = model_dir / "eval_split.npz"

    if not metrics_path.exists():
        print("ERROR: metrics.json not found", file=sys.stderr)
        return 1
    try:
        training_metrics = json.loads(metrics_path.read_text())
    except Exception as e:
        print(f"ERROR reading metrics: {e}", file=sys.stderr)
        return 1

    reported_accuracy = training_metrics.get("accuracy")
    if not model_path.exists() or not split_path.exists():
        payload = {
            "eval_accuracy": None,
            "training_accuracy": reported_accuracy,
            "discrepancy": None,
            "consistent": None,
            "independently_verified": False,
            "limitation": (
                "Independent evaluation artifacts (model.joblib / eval_split.npz) "
                "are missing; refusing to fabricate verification metrics."
            ),
        }
        (output_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2))
        print(payload["limitation"], file=sys.stderr)
        return 1

    try:
        import joblib
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
    except Exception as e:
        print(f"ERROR importing evaluation dependencies: {e}", file=sys.stderr)
        return 1

    try:
        bundle = joblib.load(model_path)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        split = np.load(split_path, allow_pickle=True)
        X_test = split["X_test"]
        y_test = split["y_test"]
        y_pred = model.predict(X_test)
        eval_accuracy = float(accuracy_score(y_test, y_pred))
        eval_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    except Exception as e:
        print(f"ERROR during independent re-score: {e}", file=sys.stderr)
        payload = {
            "eval_accuracy": None,
            "training_accuracy": reported_accuracy,
            "discrepancy": None,
            "consistent": None,
            "independently_verified": False,
            "limitation": f"Independent evaluation failed: {e}",
        }
        (output_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2))
        return 1

    discrepancy = None if reported_accuracy is None else abs(eval_accuracy - float(reported_accuracy))
    eval_metrics = {
        "eval_accuracy": round(eval_accuracy, 4),
        "eval_f1": round(eval_f1, 4),
        "training_accuracy": reported_accuracy,
        "discrepancy": None if discrepancy is None else round(discrepancy, 4),
        "consistent": discrepancy is not None and discrepancy < 0.01,
        "independently_verified": True,
        "n_eval_samples": int(len(y_test)),
    }

    (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    print(f"Eval accuracy: {eval_accuracy:.4f}")
    print(f"Training reported: {reported_accuracy}")
    print(f"Discrepancy: {eval_metrics['discrepancy']}")
    print(f"Consistent: {eval_metrics['consistent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
