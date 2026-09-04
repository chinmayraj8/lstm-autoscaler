#!/usr/bin/env python3
"""
Step 28 CLI: run `train_residual_hybrid_model` (src/autoscaler/train_hybrid.py)
against the REAL cluster's Prometheus, from your own machine.

Prerequisites (do these first, in your own Terminal):

    kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring 9090:9090

Leave that running in its own terminal tab -- it's what makes
http://localhost:9090 (this script's default --prometheus-url) reach the
real in-cluster Prometheus that's been accumulating real node history
since Step 26/27. This script itself makes no cluster changes and is safe
to re-run or interrupt (Ctrl-C) at any point before it prints "saved to".

What this actually does, in order:
  1. Pulls `--fit-hours` of real cpu_util_percent history for
     `--machine-id` from Prometheus (one HTTP call per 5-minute sample --
     see metrics_source.PrometheusMetricsSource's own docstring for why;
     this is the real, already-known cost of this step, not a bug here).
  2. Fits a real ARIMA(2,0,1) on it and computes real walk-forward
     training residuals (arima_baseline._arima_train_walkforward).
  3. Trains a real LSTM (identical architecture to every other LSTM in
     this project) to predict those residuals.
  4. Saves the result to `--model-dir/<machine-id>.keras`.

After it finishes, copy the file into the observer pod's PVC (the exact
`kubectl cp` command is printed at the end) -- that's what makes the
already-existing shadow-mode mechanism (live_loop._load_hybrid_residual_model)
pick it up and start evaluating it for real, alongside the deployed model,
with zero code changes needed.

This script is never invoked automatically. Training is a deliberate,
human-triggered, offline action -- see train_hybrid.py's own module
docstring for why that posture matches every other "actually change
something real" step in this project (Step 26's actuation included).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.autoscaler.metrics_source import DEFAULT_PROMETHEUS_URL, PrometheusMetricsSource  # noqa: E402
from src.autoscaler.train_hybrid import InsufficientRealHistory, train_residual_hybrid_model  # noqa: E402

DEFAULT_MODEL_DIR = "models/hybrid_residual"
DEFAULT_LOCAL_PROMETHEUS_URL = "http://localhost:9090"  # via kubectl port-forward, see module docstring
# query_step_minutes defaults to 5 in PrometheusMetricsSource -- this mirrors
# that constant so the pre-flight estimate below stays accurate if it ever
# changes; it is not re-imported because it's a constructor default, not a
# module-level constant.
QUERY_STEP_MINUTES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a real residual-hybrid (ARIMA+LSTM) model from real Prometheus history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--machine-id", required=True,
        help="The node/instance id to train for, e.g. 172.18.0.3 -- must match a "
             "value Prometheus is actually reporting for; see "
             "resolve_tracked_machine_ids or /shadow/status on the running observer "
             "for currently-live ids.",
    )
    parser.add_argument(
        "--fit-hours", type=float, default=48.0,
        help="How many hours of real history (ending now) to train on. Default 48. "
             "Larger is better for the model but costs one HTTP call per 5 minutes "
             "of history (see pre-flight estimate printed before training starts).",
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help=f"Local directory to save <machine-id>.keras into. Default {DEFAULT_MODEL_DIR!r} "
             "(relative to wherever you run this script from).",
    )
    parser.add_argument(
        "--prometheus-url", default=DEFAULT_LOCAL_PROMETHEUS_URL,
        help=f"Default {DEFAULT_LOCAL_PROMETHEUS_URL!r} -- requires "
             "`kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring 9090:9090` "
             f"running in another terminal. (In-cluster default would be {DEFAULT_PROMETHEUS_URL!r}, "
             "which only resolves from inside the cluster, not from your Mac.)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for numpy/tensorflow, for a reproducible training run. Default 42.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the pre-flight confirmation prompt (e.g. for a non-interactive run).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    n_calls = int(round(args.fit_hours * 60 / QUERY_STEP_MINUTES))
    print("=" * 70)
    print("Step 28: real hybrid model training")
    print("=" * 70)
    print(f"  machine_id     = {args.machine_id}")
    print(f"  fit_hours      = {args.fit_hours}")
    print(f"  prometheus_url = {args.prometheus_url}")
    print(f"  model_dir      = {args.model_dir}")
    print(f"  seed           = {args.seed}")
    print()
    print(f"This will make ~{n_calls} sequential HTTP calls to Prometheus "
          f"(one per {QUERY_STEP_MINUTES}-minute sample over {args.fit_hours}h), "
          "then fit ARIMA + train an LSTM locally.")
    print("Rough wall-clock budget: a few minutes for the Prometheus pull, "
          "then a few more for LSTM training on a laptop CPU.")
    print()
    print("Make sure `kubectl port-forward svc/kube-prometheus-stack-prometheus "
          "-n monitoring 9090:9090` is running in another terminal first.")
    print()

    if not args.yes:
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted -- nothing was changed.")
            return 1

    source = PrometheusMetricsSource(prometheus_url=args.prometheus_url)
    now = datetime.now(timezone.utc)

    t0 = time.monotonic()
    try:
        result = train_residual_hybrid_model(
            machine_id=args.machine_id,
            source=source,
            now=now,
            model_dir=args.model_dir,
            fit_hours=args.fit_hours,
            seed=args.seed,
        )
    except InsufficientRealHistory as exc:
        print()
        print(f"Not enough real history yet: {exc}")
        print("This is expected while the cluster is still accumulating real uptime "
              "-- come back later (see progress/2026-09-04_step28-real-hybrid-model.md) "
              "and re-run this same command.")
        return 2
    except Exception:
        print()
        print("Training failed with an unexpected error (full traceback below). "
              "Common causes: the port-forward isn't running (connection refused), "
              "or --machine-id doesn't match a currently-live Prometheus target.")
        raise
    elapsed = time.monotonic() - t0

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)
    print(f"  n_train_points     = {result.n_train_points}")
    print(f"  n_sequences        = {result.n_sequences}")
    print(f"  epochs_trained     = {result.epochs_trained}")
    print(f"  final_train_loss   = {result.final_train_loss:.6f}")
    print(f"  train_residual_std = {result.train_residual_std:.6f}")
    print(f"  training wall time = {result.wall_clock_secs:.1f}s "
          f"(total incl. Prometheus pull: {elapsed:.1f}s)")
    print(f"  saved to           = {result.model_path}")
    print()
    print("Next step -- copy it into the observer pod's PVC so the running "
          "shadow-mode mechanism picks it up (adjust the pod name if it's different):")
    print()
    print("    OBSERVER_POD=$(kubectl get pod -n lstm-autoscaler -l app=lstm-observer "
          "-o jsonpath='{.items[0].metadata.name}')")
    print(f"    kubectl cp {result.model_path} "
          f"lstm-autoscaler/$OBSERVER_POD:/data/hybrid_residual/{args.machine_id}.keras")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
