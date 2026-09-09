"""Unit tests for scripts/bootstrap_shadow_window.py's leakage guard (Step 28).

Only `check_training_data_overlap` is under test -- the pure decision
function main() calls, extracted specifically so it's testable without a
real trained model file, real Prometheus/observer API, or CLI args. No
TensorFlow, no network -- matches this project's test-fast convention.

The rest of the script (fetching real data, real ARIMA/hybrid forecasting,
the actual HTTP POST) is exercised only against the real cluster by hand
(see progress/2026-09-04_step28-real-hybrid-model.md's "Real-cluster
verification" section for that script's own real training run, and the
--dry-run runs recorded there for this script) -- the same posture
train_hybrid.py's offline pipeline vs. its scripts/ CLI wrapper already
has: the pipeline logic is unit-tested, the wrapper's real-infrastructure
calls are verified by hand.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bootstrap_shadow_window import LeakageGuardResult, check_training_data_overlap

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_no_overlap_when_fit_start_is_after_training_cutoff():
    model_trained_at = T0
    fit_start = T0 + timedelta(hours=1)  # comfortably after training

    result = check_training_data_overlap(fit_start, model_trained_at, allow_overlap=False)

    assert result == LeakageGuardResult(overlaps=False, refuse=False)


def test_refuses_when_fit_start_is_before_training_cutoff():
    model_trained_at = T0
    fit_start = T0 - timedelta(hours=1)  # this window's data predates training -- real leakage risk

    result = check_training_data_overlap(fit_start, model_trained_at, allow_overlap=False)

    assert result == LeakageGuardResult(overlaps=True, refuse=True)


def test_allow_overlap_flag_permits_it_but_still_flags_the_overlap():
    # --allow-overlap is a deliberate, labeled escape hatch: it must let
    # the caller proceed (refuse=False) but must NOT hide the fact that
    # this run overlaps training data (overlaps stays True), since that's
    # what main() uses to label the result as a sanity-check-only run.
    model_trained_at = T0
    fit_start = T0 - timedelta(hours=1)

    result = check_training_data_overlap(fit_start, model_trained_at, allow_overlap=True)

    assert result == LeakageGuardResult(overlaps=True, refuse=False)


def test_exact_equality_boundary_does_not_count_as_overlap():
    # fit_start == model_trained_at exactly: the model's training data
    # ends exactly where this window's data begins -- not < , so this is
    # the boundary case, not an overlap. (A live run essentially never
    # hits this exactly, but the comparison itself -- strict `<` -- should
    # behave this way rather than off-by-one refusing a perfectly valid,
    # just-in-time window.)
    model_trained_at = T0
    fit_start = T0

    result = check_training_data_overlap(fit_start, model_trained_at, allow_overlap=False)

    assert result == LeakageGuardResult(overlaps=False, refuse=False)


def test_allow_overlap_is_a_no_op_when_there_is_no_overlap_to_allow():
    model_trained_at = T0
    fit_start = T0 + timedelta(hours=1)

    result = check_training_data_overlap(fit_start, model_trained_at, allow_overlap=True)

    assert result == LeakageGuardResult(overlaps=False, refuse=False)
