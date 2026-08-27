"""
Demo client: replays m_1933's real test-split data against the running API.

Each tick sends the previous 30 minutes of actual CPU readings to POST /forecast
and prints the model's prediction alongside the true next value, so you can
watch the forecast track (or diverge from) reality in real time.

Usage
-----
1. Start the server (from the project root):
       uvicorn src.api.main:app --reload

2. In a second terminal (also from the project root):
       python src/api/demo_client.py [--ticks N] [--delay S]

Options
-------
--ticks N    Stop after N ticks  (default: all test-window ticks)
--delay S    Sleep S seconds between ticks for a real-time feel  (default: 0)
--url URL    Server base URL  (default: http://127.0.0.1:8000)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warnings
warnings.filterwarnings("ignore")

from experiments.pipeline import (  # noqa: E402
    DEMAND_SCALE,
    FEATURE_COL,
    HORIZON_STEPS,
    LOOKBACK_STEPS,
    SAFETY_MARGIN,
    TEST_RATIO,
    _load_and_prepare,
    _split_three_way,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticks", type=int, default=None,
                   help="Max ticks to replay (default: all)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds between ticks (default: 0 = as fast as possible)")
    p.add_argument("--url", default="http://127.0.0.1:8000",
                   help="Server base URL")
    return p.parse_args()


def _check_server(base_url: str) -> dict:
    try:
        r = requests.get(f"{base_url}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot reach {base_url}/health")
        print("  Start the server with:  uvicorn src.api.main:app --reload")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: /health returned {e.response.status_code}: {e.response.text}")
        sys.exit(1)


def _load_test_cpu(base_url: str) -> tuple[str, np.ndarray]:
    """Return (machine_id, 1-D array of CPU% values for the test split)."""
    print("Loading m_1933 test data …", flush=True)
    ts, machine_id = _load_and_prepare()

    n         = len(ts)
    test_start = int(n * (1.0 - TEST_RATIO))
    cpu_vals  = ts[FEATURE_COL].values[test_start:]

    print(f"  Machine : {machine_id}")
    print(f"  Total ts: {n} points  |  test split: {len(cpu_vals)} points  "
          f"({len(cpu_vals) - LOOKBACK_STEPS} forecast windows)")
    return machine_id, cpu_vals


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # 1. Verify server is up
    health = _check_server(args.url)
    print(f"\nServer OK")
    print(f"  Model machine : {health['machine_id']}")
    print(f"  Lookback      : {health['lookback_steps']} steps "
          f"({health['lookback_minutes']} min)")
    print(f"  Horizon       : {health['horizon_steps']} steps "
          f"({health['horizon_minutes']} min)")
    print(f"  Started at    : {health['started_at']}")

    # 2. Load test data
    machine_id, cpu_vals = _load_test_cpu(args.url)

    # 3. Set up replay
    n_windows = len(cpu_vals) - LOOKBACK_STEPS
    if args.ticks is not None:
        n_windows = min(n_windows, args.ticks)

    # Track server count across ticks (stateful client)
    servers = 2

    # 4. Print table header
    COL = {
        "tick":   4,  "last":  9, "fc5":  7, "fc10": 7, "fc15": 7,
        "load":  14,  "now":   7, "rec":  7, "act": 18, "real": 10,
        "err":    8,
    }
    hdr = (
        f"{'Tick':>{COL['tick']}}  "
        f"{'LastCPU%':>{COL['last']}}  "
        f"{'Fc+5':>{COL['fc5']}}  {'Fc+10':>{COL['fc10']}}  {'Fc+15':>{COL['fc15']}}  "
        f"{'PlannedLoad%':>{COL['load']}}  "
        f"{'SrvNow':>{COL['now']}}  {'SrvRec':>{COL['rec']}}  "
        f"{'Action':<{COL['act']}}  "
        f"{'ActualCPU%':>{COL['real']}}  "
        f"{'Err(pp)':>{COL['err']}}"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    errors: list[float] = []

    for i in range(n_windows):
        window     = cpu_vals[i : i + LOOKBACK_STEPS].tolist()
        actual_t15 = float(cpu_vals[i + LOOKBACK_STEPS])  # the "real" next value

        payload = {
            "cpu_pct":          window,
            "current_servers":  servers,
            "demand_scale":     DEMAND_SCALE,
            "safety_margin":    SAFETY_MARGIN,
        }

        try:
            r = requests.post(f"{args.url}/forecast", json=payload, timeout=5)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"  {'':>{COL['tick']}}  request error — {exc}")
            continue

        resp    = r.json()
        fc      = resp["forecast_cpu_pct"]
        action  = resp["action"]
        servers = resp["recommended_servers"]

        # Error = actual t+5 CPU% vs first-step forecast
        err = fc[0] - actual_t15

        print(
            f"{i:>{COL['tick']}}  "
            f"{window[-1]:>{COL['last']}.3f}  "
            f"{fc[0]:>{COL['fc5']}.3f}  {fc[1]:>{COL['fc10']}.3f}  {fc[2]:>{COL['fc15']}.3f}  "
            f"{resp['planned_load_pct']:>{COL['load']}.1f}  "
            f"{resp['current_servers']:>{COL['now']}}  {resp['recommended_servers']:>{COL['rec']}}  "
            f"{action:<{COL['act']}}  "
            f"{actual_t15:>{COL['real']}.3f}  "
            f"{err:>{COL['err']:}.3f}"
        )
        errors.append(abs(err))

        if args.delay > 0:
            time.sleep(args.delay)

    # 5. Summary
    if errors:
        print(sep)
        print(f"\nSummary over {n_windows} ticks (1-step-ahead forecast vs actual t+5):")
        print(f"  MAE  = {np.mean(errors):.4f} CPU%")
        print(f"  RMSE = {np.sqrt(np.mean(np.array(errors)**2)):.4f} CPU%")
        print(f"  Max  = {np.max(errors):.4f} CPU%")


if __name__ == "__main__":
    main()
