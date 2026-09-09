#!/usr/bin/env python3
"""
Step 28 CLI, part 2: build ONE real ARIMA-vs-hybrid shadow window from the
live cluster's own real Prometheus history, and submit it to the running
observer's `POST /shadow/{machine_id}/window` endpoint (Steps 18/19).

Why this script exists (the gap `train_real_hybrid_model.py` doesn't
close): saving a trained `.keras` file into the observer pod's PVC does
NOT put a node on the hybrid forecaster. `live_loop.py`'s own module
docstring says this outright -- the live loop's hybrid path only
RE-VALIDATES a node ALREADY assigned to the hybrid; it "does not, and
structurally cannot, promote an ARIMA node to hybrid on its own." The
only real path to that first assignment is this manual one: submit real,
already-observed shadow windows by hand until `shadow.decide_assignment`
flips the assignment (needs `shadow.DEFAULT_MIN_WINDOWS` = 3 consecutive
wins that also clear a combined-+/-1sigma statistical bar -- see
shadow.py and src/api/main.py's submit_shadow_window docstring).

This script does NOT reimplement that comparison logic -- it reuses the
exact same real-data pipeline `live_loop._build_hybrid_window` already
runs internally (real ARIMA walk-forward forecast via
arima_baseline._arima_rolling_forecast, real hybrid forecast via the
trained LSTM's residual prediction, both inverse-scaled with
arima_baseline._inv_flat), stopping one step short: instead of calling
`shadow.run_shadow_window` in-process against a local ShadowStore (which
only the pod itself has -- its shadow_state.db lives on the pod's PVC,
not reachable from your Mac), it POSTs the same real arrays to the
already-deployed HTTP endpoint that exists for exactly this situation.

Prerequisites (do these first, in your own Terminal -- TWO port-forwards,
each in its own tab, both need to stay running):

    kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring 9090:9090
    kubectl port-forward svc/lstm-autoscaler-observer -n lstm-autoscaler 8000:80

The first reaches real Prometheus (same as train_real_hybrid_model.py).
The second reaches the observer's real FastAPI app -- this is what makes
http://localhost:8000 (this script's default --api-url) resolve to the
same pod whose shadow_state.db actually gets updated.

Leakage guard, real and enforced (not just documented): this script reads
the model file's own mtime and refuses to build a window whose data
(including the ARIMA fit portion, `--shadow-fit-hours` before the window)
starts before the model was trained -- the LSTM must never be scored, even
in shadow mode, against data it already saw as training input. Pass
--allow-overlap only for a deliberate, throwaway sanity check; a window
built that way is NOT a valid step toward the real 3-window promotion and
this script says so loudly if you use it.

Cadence that actually matters for a REAL 3-window promotion: each shadow
window is `--shadow-window-hours` (default 24) of real, already-elapsed
data. Windows must be genuinely independent, not the same 24h scored
three times with small shifts -- so re-running this command sooner than
~24h after the previous real (non---allow-overlap) run will mostly
re-cover the same hours and does not count as a second real window. Once
a day, for 3 days, is the real cadence this endpoint's 3-window statistical
bar assumes.

This script makes no cluster changes beyond the one intended write (a
banked shadow window + possibly a forecaster-assignment flip, both inside
the observer's own shadow_state.db, exactly what this endpoint is for).
It never touches the real fleet -- see submit_shadow_window's own
docstring: "Neither forecaster's targets are ever applied to the real
fleet through this endpoint."
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

from src.autoscaler import config
from src.autoscaler.arima_baseline import ARIMA_ORDER, _arima_rolling_forecast, _inv_flat  # noqa: E402
from src.autoscaler.data import _make_sequences  # noqa: E402
from src.autoscaler.live_loop import (  # noqa: E402
    DEFAULT_SHADOW_FIT_HOURS,
    DEFAULT_SHADOW_WINDOW_HOURS,
    MIN_FIT_POINTS,
    HybridModelUnavailable,
    _load_hybrid_residual_model,
)
from src.autoscaler.metrics_source import DEFAULT_PROMETHEUS_URL, PrometheusMetricsSource, resample_readings  # noqa: E402

DEFAULT_MODEL_DIR = "models/hybrid_residual"
DEFAULT_LOCAL_PROMETHEUS_URL = "http://localhost:9090"   # via port-forward, see module docstring
DEFAULT_API_URL = "http://localhost:8000"                 # via port-forward, see module docstring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one real shadow window from live Prometheus history and submit it "
                    "to the running observer's /shadow/{machine_id}/window endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--machine-id", required=True,
                        help="Must match a currently-live Prometheus target, and the id the "
                             "model was trained for (models/hybrid_residual/<machine-id>.keras).")
    parser.add_argument("--shadow-fit-hours", type=float, default=DEFAULT_SHADOW_FIT_HOURS,
                        help=f"History used only to fit ARIMA before rolling through the window. "
                             f"Default {DEFAULT_SHADOW_FIT_HOURS} (code default, matches the live "
                             "loop -- don't shrink this outside of a flagged demo).")
    parser.add_argument("--shadow-window-hours", type=float, default=DEFAULT_SHADOW_WINDOW_HOURS,
                        help=f"The real window being scored. Default {DEFAULT_SHADOW_WINDOW_HOURS} "
                             "(code default -- see module docstring on cadence).")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Local directory holding <machine-id>.keras. Default {DEFAULT_MODEL_DIR!r}. "
                             "Must be the SAME file already copied into the pod's PVC, or the shadow "
                             "score won't reflect what's actually deployed.")
    parser.add_argument("--prometheus-url", default=DEFAULT_LOCAL_PROMETHEUS_URL,
                        help=f"Default {DEFAULT_LOCAL_PROMETHEUS_URL!r} (in-cluster default would be "
                             f"{DEFAULT_PROMETHEUS_URL!r}, only resolves inside the cluster).")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help=f"Default {DEFAULT_API_URL!r} -- the observer's own FastAPI app, reached "
                             "via its own separate port-forward (see module docstring).")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="Skip the training-data-overlap leakage guard. The resulting window is "
                             "a real end-to-end sanity check ONLY, never a valid step toward the real "
                             "3-window promotion -- this script labels it as such in its output.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the window, but do not POST it. Safe to run at any time.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(timezone.utc)
    fit_start = now - timedelta(hours=args.shadow_fit_hours + args.shadow_window_hours)
    window_start = now - timedelta(hours=args.shadow_window_hours)

    model_path = os.path.join(args.model_dir, f"{args.machine_id}.keras")
    print("=" * 70)
    print("Step 28: real shadow-window bootstrap")
    print("=" * 70)
    print(f"  machine_id         = {args.machine_id}")
    print(f"  fit_start           = {fit_start.isoformat()}")
    print(f"  window_start        = {window_start.isoformat()}")
    print(f"  window_end (now)    = {now.isoformat()}")
    print(f"  model_path          = {model_path}")
    print(f"  api_url             = {args.api_url}")
    print(f"  dry_run             = {args.dry_run}")
    print()

    if not os.path.exists(model_path):
        print(f"No trained model at {model_path!r}. Run scripts/train_real_hybrid_model.py first.")
        return 2

    model_trained_at = datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc)
    print(f"  model file mtime    = {model_trained_at.isoformat()} (treated as its training cutoff)")
    if fit_start < model_trained_at and not args.allow_overlap:
        print()
        print("REFUSING: this window's data starts before the model's own training cutoff -- "
              "the LSTM would be scored (even in shadow mode) against data it may have already "
              "trained on. Wait until now - (shadow_fit_hours + shadow_window_hours) >= the model's "
              "training cutoff above, or pass --allow-overlap for a throwaway sanity check only "
              "(will NOT count toward the real 3-window promotion).")
        return 3
    overlap_flagged = fit_start < model_trained_at  # only reachable here if --allow-overlap was passed
    if overlap_flagged:
        print()
        print("WARNING: --allow-overlap set and this window DOES overlap the model's training data. "
              "Proceeding anyway -- this result is a wiring/sanity check only.")
    print()

    if not args.yes:
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted -- nothing was changed.")
            return 1

    # Cheap check first, same order train/live_loop already use.
    try:
        model = _load_hybrid_residual_model(args.machine_id, args.model_dir)
    except HybridModelUnavailable as exc:
        print(f"Model load failed: {exc}")
        return 2

    source = PrometheusMetricsSource(prometheus_url=args.prometheus_url)
    raw = source.fetch_readings(args.machine_id, fit_start, now)
    resampled = resample_readings(raw)
    fit_flat = resampled.loc[resampled.index < window_start, config.FEATURE_COL].values.astype(np.float64)
    target_flat = resampled.loc[resampled.index >= window_start, config.FEATURE_COL].values.astype(np.float64)

    min_needed = config.LOOKBACK_STEPS + config.HORIZON_STEPS
    if len(fit_flat) < MIN_FIT_POINTS or len(target_flat) < min_needed:
        print(f"Not enough real history yet: fit={len(fit_flat)} pts (need >={MIN_FIT_POINTS}), "
              f"target={len(target_flat)} pts (need >={min_needed}). Come back later.")
        return 2

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(fit_flat.reshape(-1, 1))
    fit_scaled = scaler.transform(fit_flat.reshape(-1, 1)).flatten()
    target_scaled = scaler.transform(target_flat.reshape(-1, 1)).flatten()

    print(f"Pulled {len(fit_flat) + len(target_flat)} real points "
          f"({len(fit_flat)} fit / {len(target_flat)} window). Fitting ARIMA{ARIMA_ORDER} "
          "and rolling it through the window (real walk-forward, one refit-free step at a time)...")
    arima_pred_scaled, y_true_scaled = _arima_rolling_forecast(
        fit_scaled, np.array([], dtype=fit_scaled.dtype), target_scaled,
        config.LOOKBACK_STEPS, config.HORIZON_STEPS, ARIMA_ORDER,
    )

    X, _ = _make_sequences(target_scaled.reshape(-1, 1), config.LOOKBACK_STEPS, config.HORIZON_STEPS)
    residual_scaled = np.asarray(model.predict(X, verbose=0))
    hybrid_pred_scaled = np.clip(arima_pred_scaled + residual_scaled, 0.0, 1.0)

    arima_pred_real = _inv_flat(arima_pred_scaled, scaler)
    hybrid_pred_real = _inv_flat(hybrid_pred_scaled, scaler)
    y_actual_real = _inv_flat(y_true_scaled, scaler)
    print(f"Built {arima_pred_real.shape[0]} real ({config.LOOKBACK_STEPS}-in / "
          f"{config.HORIZON_STEPS}-out) sequences for both forecasters.")
    print()

    payload = {
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
        "demand_scale": config.DEMAND_SCALE,
        "y_actual": y_actual_real.tolist(),
        "arima_forecast": arima_pred_real.tolist(),
        "arima_under_prov_weight": config.DEC_UNDER_WEIGHT,
        "arima_safety_margin": config.SAFETY_MARGIN,
        "hybrid_forecast": hybrid_pred_real.tolist(),
        "hybrid_under_prov_weight": config.DEC_UNDER_WEIGHT,
        "hybrid_safety_margin": config.SAFETY_MARGIN,
    }

    if args.dry_run:
        print("--dry-run set -- not submitting. Payload shapes: "
              f"y_actual={y_actual_real.shape}, arima_forecast={arima_pred_real.shape}, "
              f"hybrid_forecast={hybrid_pred_real.shape}")
        return 0

    url = f"{args.api_url}/shadow/{args.machine_id}/window"
    print(f"POSTing to {url} ...")
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Submission failed: {exc}")
        print("Common cause: `kubectl port-forward svc/lstm-autoscaler-observer "
              "-n lstm-autoscaler 8000:80` isn't running in another terminal.")
        return 2

    result = resp.json()
    print()
    print("=" * 70)
    print("Shadow window banked" + (" (--allow-overlap: sanity check only, does not count)" if overlap_flagged else ""))
    print("=" * 70)
    print(f"  window_result       = {result['window_result']}")
    print(f"  current_forecaster  = {result['current_forecaster']}")
    print(f"  n_banked_windows    = {result['n_banked_windows']}  (needs 3 consecutive real wins to promote)")
    print(f"  assignment_changed  = {result['assignment_changed']}")
    if result.get("verdict"):
        print(f"  verdict             = {result['verdict']}")
    print(f"  cumulative          = {result['cumulative']}")
    print()
    if result["current_forecaster"] == "hybrid":
        print(f"machine_id={args.machine_id} is now on the HYBRID forecaster for real.")
    else:
        remaining = max(0, 3 - result["n_banked_windows"])
        print(f"Still on ARIMA. Come back in ~{args.shadow_window_hours:.0f}h for the next real, "
              f"non-overlapping window (roughly {remaining} more consecutive wins needed, "
              "assuming this one wins too).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
