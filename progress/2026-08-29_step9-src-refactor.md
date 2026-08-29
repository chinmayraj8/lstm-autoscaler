# Step 9: src/ package extraction + pytest test suite
Date: 2026-08-29
Status: done (verified by real test execution + import-graph checks; full
regression run against the real dataset/TensorFlow still pending -- see
"Still open")

## What changed

1. **`src/autoscaler/` created**: a proper, importable package holding the
   logic that used to live entirely in `experiments/pipeline.py` (515
   lines, one file, no tests). Split by responsibility, not by size:
   - `config.py` — every frozen constant, copied byte-for-byte. The one
     deliberate change: `DATA_PATH` can now be overridden with the
     `LSTM_AUTOSCALER_DATA_PATH` environment variable; the default still
     resolves to `~/Desktop/machine_usage_bigger.csv`, so nothing changes
     unless the variable is set.
   - `data.py` — CSV loading, per-machine resampling, the chronological
     60/20/20 split, sliding-window sequence construction.
   - `decision.py` — `DecisionConfig`, the penalty function, and the
     greedy scale up/down/hold rule.
   - `simulation.py` — `SimConfig`/`SimMetrics`, the tick-by-tick fleet
     simulator (startup delay, SLA/over-provision detection), the cost
     score, and the Reactive policy.
   - `calibration.py` — `calibrate_demand_scale` and `check_feasibility`
     (the 115%-of-one-server target and the p99 feasibility check).
   - `forecasting.py` — LSTM build/train/evaluate plus the naive-baseline
     evaluator. **This is the only module that imports TensorFlow.**
   - `experiment.py` — `tune_on_validation` and `run_single_experiment`,
     composed from the modules above. Logic is unchanged; only the
     temp-model-file path calculation was adjusted for the new file
     depth (still writes to `experiments/_tmp_*.keras`, same as before).
   - `__init__.py` — re-exports the public API. Eagerly imports
     config/calibration/data/decision/simulation (numpy/pandas/sklearn
     only); lazily imports `run_single_experiment`, `tune_on_validation`,
     and the forecasting helpers on first access, via a module-level
     `__getattr__`. That means `from src.autoscaler import DecisionConfig`
     (or anything from the decision engine, simulator, or calibration)
     now works with **no TensorFlow install at all** — the point of doing
     this was to make the pure-logic pieces testable in a lightweight
     environment / CI, without needing the full ML stack.

2. **`experiments/pipeline.py` is now a thin backward-compatible shim.**
   It re-exports everything from `src.autoscaler` under the exact same
   names (including the underscored private helpers and the grid-search
   constants). Every existing script's
   `from experiments.pipeline import X` keeps working unchanged.

3. **`run_multiseed.py`, `run_multiseed_v2.py`, `run_multimachine.py`,
   `run_multimachine_v2.py`, `run_baselines_and_sweep.py`,
   `src/api/main.py`, `src/api/demo_client.py`,
   `src/dashboard/prepare_replay_data.py` repointed** to import from
   `src.autoscaler` directly instead of `experiments.pipeline`. In every
   file this is a **single-line change** (the import source line only —
   confirmed by diffing each rewritten file against its original); the
   imported names and all other code are untouched.

4. **`tests/` created** with a real pytest suite (25 tests, all passing)
   covering the parts of the pipeline that don't need TensorFlow or the
   real dataset:
   - `test_decision.py` — penalty function boundary cases, hold/scale-up/
     scale-down selection, clamping at min/max servers.
   - `test_simulation.py` — the startup-delay mechanic (scale-up is
     delayed one tick, scale-down is immediate), SLA-violation and
     over-provisioning detection, the cost-score formula, Reactive's
     threshold behavior and clamping.
   - `test_calibration.py` — the calibration formula, and specifically a
     regression test that `check_feasibility` uses **p99, not p95** (the
     exact mixup the earlier reconstruction pipeline made — see docs/ and
     the handoff's §16c).
   - `test_data.py` — split sizes are exactly 60/20/20, and
     `test_split_has_no_scaler_leakage` directly checks the Step 1 fix:
     the scaler's fitted range comes from the train split only, so val/
     test values outside that range transform to outside `[0, 1]`.

## Why

This was the project's own stated top priority (`progress/00_INDEX.md`,
"Phase 2B onward... not started"; the handoff doc's §21 NEXT STEP #1):
everything else on the roadmap — Docker, Kubernetes, a dashboard worth
showing outside the team, CI — is safer and faster once the core logic
isn't 515 lines in one script with zero tests. No numeric behavior was
supposed to change here, and the verification below is aimed specifically
at proving that.

## Verification (what was actually run, and what wasn't)

This was built and verified in a sandboxed environment without access to
the real TensorFlow install, the real dataset, or a GPU — so two
different kinds of check were used:

- **Real execution**: all 25 pytest tests in `tests/` actually ran (not
  just syntax-checked) against the real `src/autoscaler` code, using real
  numpy/pandas/scikit-learn. 25/25 passed.
- **Import-graph verification**: every touched file (`experiments/
  pipeline.py`, all five `run_*.py` scripts, `src/api/main.py`,
  `src/api/demo_client.py`, `src/dashboard/prepare_replay_data.py`) was
  imported end-to-end with a stubbed-out `tensorflow` module, to confirm
  every name referenced actually exists at the expected location with no
  typos or missing re-exports. All imports resolved cleanly.

## Still open

- **No regression run against the real dataset/TensorFlow yet.** The
  pure-logic tests above prove the decision engine, simulator, and
  calibration math didn't change. They do NOT prove the LSTM training/
  forecasting path still produces identical numbers, since that needs
  real TensorFlow and `~/Desktop/machine_usage_bigger.csv`, neither of
  which existed in the environment this refactor was built in. **Next:**
  run `venv/bin/python -m pytest tests/ -v` and then
  `venv/bin/python experiments/run_multiseed.py` (or any other `run_*.py`)
  for real, and confirm the numbers match the existing results CSVs.
- No tests yet for `run_multimachine.py`'s `_compute_machine_stats` /
  `_select_machines` (the K-means clustering helpers) — those weren't
  moved into the package this round, since they're specific to that one
  experiment script rather than core reusable logic. Could be split out
  into `src/autoscaler/clustering.py` later if that logic needs to be
  reused elsewhere (e.g. a future capacity-tier feature for cluster 1).
- `requirements-dev.txt` (just `pytest`) was added for running the new
  test suite; a full pinned `requirements.txt` for the production stack
  (tensorflow-macos, streamlit, fastapi, etc.) is still a separate,
  not-yet-done item.
