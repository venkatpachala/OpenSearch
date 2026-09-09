#!/usr/bin/env python3
"""
Image Classification Training Script — ResearchRepro Experiment Environment.

Dataset: MNIST (built into scikit-learn via fetch_openml — no GPU needed)
Model:   MLPClassifier (multi-layer perceptron)

This script is intentionally designed so that:
  - BASELINE (no normalize, step LR):       ~92-93% accuracy
  - WITH normalize:                          ~95-96% accuracy  
  - WITH normalize + cosine LR equivalent:  ~96-97% accuracy

The "paper" reports 97% — the agent must discover the fixes to close the gap.

Output (written to --output-dir):
  metrics.json   — all metrics
  run.log        — training progress
  config.json    — config used

Usage:
  python train.py --output-dir runs/exp_001 --seed 42
  python train.py --output-dir runs/exp_002 --normalize --seed 42
  python train.py --output-dir runs/exp_003 --normalize --hidden-size 256 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image Classification Experiment")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Architecture
    p.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    p.add_argument("--hidden-layers", type=int, default=2, help="Number of hidden layers")

    # Training
    p.add_argument("--max-iter", type=int, default=100, help="Max training iterations (default 100; use 200 for best results with normalize)")

    p.add_argument("--lr", type=float, default=0.001, help="Learning rate (Adam)")
    p.add_argument("--batch-size", type=int, default=200, help="Mini-batch size")

    # Preprocessing — these are the fixes the agent needs to discover
    p.add_argument("--normalize", action="store_true",
                   help="Normalize features to zero mean, unit variance")
    p.add_argument("--pca-components", type=int, default=0,
                   help="PCA dimensionality reduction (0=disabled)")

    # Optimizer variant that mimics cosine LR schedule effect
    p.add_argument("--solver", type=str, default="adam",
                   choices=["adam", "sgd", "lbfgs"],
                   help="Optimizer to use")

    # Constraint metric simulation
    p.add_argument("--latency-budget-ms", type=float, default=100.0,
                   help="Simulated latency budget in ms (affects model complexity penalty)")

    return p.parse_args()


def load_data(seed: int):
    """Load the real MNIST dataset; fail loudly if it cannot be retrieved."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X, y = mnist.data, mnist.target
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except Exception as e:
        raise RuntimeError(
            "Real MNIST retrieval failed; refusing to substitute synthetic data. "
            "Check network access or pre-cache MNIST before running the benchmark."
        ) from e


def run_experiment(args: argparse.Namespace) -> dict:
    """Run the full training + evaluation pipeline and return metrics."""
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    np.random.seed(args.seed)
    start = time.monotonic()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y = load_data(args.seed)
    X = X.astype("float32")

    # Subsample for speed (use 20k train, 2k test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=2000, train_size=min(20000, len(X) - 2000),
        random_state=args.seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    if args.normalize:
        print("Applying input normalization ([0, 1] scaling)...")
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    else:
        print("Using unnormalized raw [0, 255] pixel values (baseline)")

    if args.pca_components > 0:
        from sklearn.decomposition import PCA
        print(f"Applying PCA with {args.pca_components} components...")
        pca = PCA(n_components=args.pca_components, random_state=args.seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # ── 3. Build model ────────────────────────────────────────────────────────
    hidden_layer_sizes = tuple([args.hidden_size] * args.hidden_layers)

    # Map solver + lr to sklearn equivalents
    solver = args.solver
    learning_rate_init = args.lr
    # "cosine" effect: use adaptive lr which behaves similarly
    learning_rate = "adaptive" if solver == "adam" else "constant"

    print(f"Building MLP: layers={hidden_layer_sizes}, solver={solver}, lr={learning_rate_init}")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        solver=solver,
        learning_rate_init=learning_rate_init,
        learning_rate=learning_rate,
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"Training for up to {args.max_iter} iterations...")
    t_train_start = time.monotonic()
    model.fit(X_train, y_train)
    train_time = time.monotonic() - t_train_start
    actual_iters = model.n_iter_
    print(f"Training converged after {actual_iters} iterations ({train_time:.1f}s)")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    train_accuracy = float(model.score(X_train, y_train))
    final_loss = float(model.loss_) if hasattr(model, "loss_") else None

    total_time = time.monotonic() - start

    # Simulate latency (proportional to model complexity)
    n_params = sum(
        w.size for w in model.coefs_
    ) + sum(b.size for b in model.intercepts_)
    simulated_latency_ms = round(n_params / 10000 * 12.0 + 15.0, 1)

    print(f"\n{'='*50}")
    print(f"Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test F1:        {f1:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Runtime:        {total_time:.1f}s")
    print(f"Latency (sim):  {simulated_latency_ms}ms")
    print(f"{'='*50}")

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "train_accuracy": round(train_accuracy, 4),
        "loss": round(final_loss, 4) if final_loss is not None else None,
        "runtime_seconds": round(total_time, 2),
        "train_time_seconds": round(train_time, 2),
        "iterations": actual_iters,
        "n_params": n_params,
        "latency_ms": simulated_latency_ms,
        "config": {
            "normalize": args.normalize,
            "solver": args.solver,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "max_iter": args.max_iter,
            "lr": args.lr,
            "seed": args.seed,
            "pca_components": args.pca_components,
        }
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    try:
        metrics = run_experiment(args)

        # Write metrics
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        print(f"\nMetrics written to {output_dir / 'metrics.json'}")
        return 0

    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
