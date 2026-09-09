# Step 28: real hybrid model training pipeline

## Why this step exists

Every step from 16 through 26 flagged the same gap: no pretrained
residual-hybrid model exists for any real node. The offline experiments
(Step 16, `experiments/run_hybrid_arima_lstm.py`) proved the ARIMA+LSTM
residual-hybrid mechanism works, but only against the offline Alibaba
Cluster Trace 2018 CSV (`machine_usage_bigger.csv`, from Kaggle -- real
production data from a real company, released for research; see
README.md's Dataset section). The live loop (Step 20+), shadow evaluation
(Step 19), and now real actuation (Step 26) all run against the *actual*
Docker Desktop cluster, but nothing had ever trained a hybrid model from
that cluster's own real Prometheus history -- `_load_hybrid_residual_model`
in `live_loop.py` has always had nothing to load.

This step closes that gap: real Prometheus history -> a real fitted
ARIMA -> real walk-forward training residuals -> a real trained LSTM -> a
real `.keras` file, deployable into the same shadow-mode mechanism that's
already been running since Step 19.

## What idle data would have meant

An idle Docker Desktop cluster's node CPU is close to flat. Training a
"real" hybrid model against that would be real in name only -- there'd be
nothing for ARIMA or the residual LSTM to actually learn. So this step
also needed a way to put genuine, bounded, real CPU pressure on the
cluster's nodes, the same honesty posture already applied to
`demo-workload` (Step 26): a clearly-labeled synthetic signal, never
described as real production traffic.

### `k8s/load-generator.yaml` -- and a real revision mid-step

First version: a `DaemonSet` (`cpu-load-generator`, `polinux/stress-ng`)
running a fixed 8-phase, 15-minutes-per-phase repeating ramp (10% -> 25%
-> 45% -> 65% -> 45% -> 25% -> 10% -> 0%, exactly every 2 hours), bounded
to a single core with a hard 1-CPU resource limit.

Before applying it, the user asked directly whether this load was "good
enough," or whether a different/external dataset would be better. Worth
recording the reasoning plainly, not just the conclusion:

- **Not an external dataset.** Pulling in a public cluster trace (Azure,
  Google, Alibaba, etc.) here would undo the point of this step -- it
  would still be a canned, offline dataset, just a different one, not
  this project's own real cluster being exercised live. The project's own
  "not just based on a single dataset" bar is about exactly this: Step 16
  already used a real company's real data (Alibaba) offline; this step is
  about proving the *live* path for real, on infrastructure that has no
  real traffic source of its own.
- **The original load generator had a real weakness, though.** It was a
  real CPU signal on real infrastructure, but a *perfectly periodic* one
  -- an LSTM (or ARIMA) can memorize one exact repeating waveform without
  learning anything resembling real forecasting. That's the same
  "too clean, not representative" problem as a canned dataset, just
  self-inflicted this time.

Fix, applied before the manifest was ever deployed: both the target load%
of each phase (base +/-10 points, clamped to [0, 65]) and each phase's
duration (600-1200s instead of a fixed 900s) are now randomized every
cycle via `/dev/urandom` (`od -An -N2 -tu2 /dev/urandom`, POSIX `/bin/sh`
-- no bash `$RANDOM`, no `awk` dependency, since the stress-ng image's
shell can't be assumed to have either). Verified locally: extracted the
embedded shell script from the YAML with a small PyYAML script, checked
it with `sh -n` (syntax) and ran the `rand_range`/jitter/clamp logic
directly a few dozen iterations to confirm every value stays inside
[0, 65] for load and [600, 1200] for duration, and that two runs produce
different sequences. Still bounded, still single-core, still capped at
65% -- the safety properties didn't change, only the predictability did.

Also asked directly: would it carry more weight to say this was trained
on real company data? Answered plainly rather than reaching for that
framing where it doesn't apply -- the *core* forecasting model (Steps
1-16, the one actually deployed and actuating `demo-workload` right now)
already **is** trained on real company data (Alibaba's 2018 cluster
trace). This step's synthetic load generator is a separate, necessary
piece for a different reason: no real company routes real production
traffic to a personal Docker Desktop cluster, so a clearly-labeled
synthetic signal is the only honest option for the *live* half of the
system. The two are not in tension and neither should be described as
the other.

## `src/autoscaler/train_hybrid.py`

Deliberately reuses Step 16's already-audited residual-hybrid mechanism
unmodified rather than reinventing it:
- `arima_baseline.ARIMA_ORDER` (the frozen (2,0,1) order used everywhere
  else in this project).
- `arima_baseline._arima_train_walkforward` (honest walk-forward training
  residual labels from the deployed ARIMA's own fixed parameters -- no
  re-estimation leakage).
- `forecasting._build_lstm_model` / `_train_lstm` (identical architecture
  and training procedure as every other LSTM in this project).

The only genuinely new logic is *where the data comes from*: real
Prometheus history via a `MetricsSource`, instead of the offline CSV
`experiments/run_hybrid_arima_lstm.py` reads -- mirroring
`live_loop._build_hybrid_window`'s existing real-data-sourcing pattern,
applied to training instead of evaluation.

`fetch_and_scale_training_series(machine_id, source, now, fit_hours)`
pulls and resamples real history, scales it with a scaler fit on that
data only (this project's train-only-fit convention throughout), and
raises `InsufficientRealHistory` rather than silently training on too
little data (`MIN_TRAIN_POINTS = LOOKBACK_STEPS + HORIZON_STEPS + 100 =
109` -- a floor, not a target; the offline dataset gave each machine
roughly 1,300-2,300 points).

`train_residual_hybrid_model(...)` fits real ARIMA, computes real
walk-forward residuals, trains the residual LSTM, and saves the result to
`{model_dir}/{machine_id}.keras` via `model.save()` -- the same
full-save convention `lstm_model.keras` was built with, which
`live_loop._load_hybrid_residual_model` already knows how to load via
`model.load_weights()`. Output destination on the real cluster:
`k8s/observer.yaml` mounts a PVC at `/data` and already points
`LSTM_AUTOSCALER_HYBRID_MODEL_DIR` at `/data/hybrid_residual` -- no
manifest change needed, this step only had to produce a file that goes
there.

### A real bug this step's own tests caught

Original ordering: `import tensorflow as tf` happened before
`fetch_and_scale_training_series`, so calling
`train_residual_hybrid_model` with too little history raised
`ModuleNotFoundError: No module named 'tensorflow'` in a sandbox without
TensorFlow installed, instead of the intended `InsufficientRealHistory`.
Caught by `tests/test_train_hybrid.py`'s
`test_train_residual_hybrid_model_raises_insufficient_history_before_touching_tensorflow`,
which is exactly the point of writing that test -- this project's whole
TensorFlow-isolation convention (`forecasting.py`'s module docstring,
`tests/conftest.py`) exists so the TensorFlow-free paths stay TensorFlow-free
in practice, not just in theory. Fixed by moving the data-fetch-and-check
call above the `import tensorflow as tf` line; verified by re-running the
test (passes) and the full non-TensorFlow-full test suite (122 passed, 1
skipped -- the skip is the correct TensorFlow-requiring end-to-end test,
under a sandbox with no TensorFlow install).

## `scripts/train_real_hybrid_model.py`

The CLI entry point the user runs from their own Mac, using their
existing working TensorFlow venv, against Prometheus reached via
`kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring
9090:9090` (the exact in-cluster service/namespace `k8s/observer.yaml`
already uses for `LSTM_AUTOSCALER_PROMETHEUS_URL`, confirmed by reading
that file rather than assumed). Defaults: `--prometheus-url
http://localhost:9090` (the in-cluster DNS name only resolves from inside
the cluster), `--model-dir models/hybrid_residual`, `--fit-hours 48`,
`--seed 42`. Prints a pre-flight estimate of HTTP call volume (~12 calls
per hour of `--fit-hours`, matching `PrometheusMetricsSource`'s
one-call-per-5-minute-sample design, already documented in Steps 21/22)
and asks for confirmation before starting (skippable with `--yes`).
Surfaces `InsufficientRealHistory` as a normal, expected outcome (exit
code 2, not a crash) with a plain-language "come back later" message.
On success, prints the exact `kubectl cp` command needed to get the
resulting file into the observer pod's `/data/hybrid_residual/` PVC path.

## Tests: `tests/test_train_hybrid.py`

Split the same way `live_loop`'s hybrid path already is: everything that
does not require TensorFlow (`fetch_and_scale_training_series`'s
sufficiency check, its within-window scaling, its train-only-fit scaler)
runs unconditionally against `StaticMetricsSource` (Step 20), no real
cluster needed. The one test that calls `train_residual_hybrid_model`
end-to-end uses `pytest.importorskip("tensorflow")` so it's automatically
skipped where TensorFlow isn't installed (this sandbox, and CI's
`test-fast` job) and actually runs where it is (CI's `test-full` job) --
no separate `--ignore` entry needed in `ci.yml`. That end-to-end test
monkeypatches `config.MAX_EPOCHS`/`config.ES_PATIENCE` down so it stays a
fast wiring/smoke test (does the real pipeline produce a real, loadable
`.keras` file?) rather than a claim about forecast quality -- forecast
quality is what the real multi-day cluster run is for.

Locally verified in this sandbox (no TensorFlow installed here): 5
passed, 1 correctly skipped for `test_train_hybrid.py`; full suite
(`--ignore=tests/test_api_shadow.py`) 122 passed, 1 skipped, nothing
else broken. `ruff check` clean on all three new files. Not yet verified
against CI's real `test-full` job (which does have TensorFlow) -- that
happens on the next push.

## Before -> After

| | Before | After |
|---|---|---|
| Hybrid model for a real node | Does not exist -- `_load_hybrid_residual_model` has nothing to load | Trainable end-to-end from real Prometheus history via one CLI command |
| Live cluster CPU signal | Flat/idle (nothing to forecast) | Real, bounded, randomized synthetic load (not a repeating waveform) |
| Training data sufficiency check | N/A | `InsufficientRealHistory`, checked before any TensorFlow import |

## Real-cluster verification

Applied `k8s/load-generator.yaml` on the real Docker Desktop cluster
(single node, so one `cpu-load-generator` pod total). Confirmed via
`kubectl get pods` and `kubectl logs -f`:

    cpu-load-generator: starting the randomized ramp cycle on cpu-load-generator-wrvwf
    cpu-load-generator: 2026-09-05T07:48:59Z load=18% for 1109s
    stress-ng: info:  [23] dispatching hogs: 1 cpu

`load=18%` (base 10% + jitter) and `1109s` (inside the randomized
600-1200s window) confirm the `/dev/urandom`-based jitter logic verified
locally earlier is actually running correctly on the real cluster, not
just in the sandbox dry-run. Real history is now accumulating.

## Real training run -- completed

Fixed a real bug found the first time this was actually run:
`scripts/train_real_hybrid_model.py`'s `sys.path.insert` only went up one
directory (to `scripts/` itself) instead of two (to the repo root), so
`from src.autoscaler...` failed with `ModuleNotFoundError`. `scripts/`
sits at the same depth as `experiments/` under the repo root, which
already uses the correct two-`dirname` pattern -- this script now matches
it. Fixed and committed (`dde0833`).

After a real macOS restart (which reset the `kubectl port-forward`, an
expected and harmless gap), and reconnecting via the same command, the
training run itself completed successfully against a genuinely
gap-free 24-hour real Prometheus window for `172.18.0.3` (100% coverage,
confirmed via a direct `/api/v1/query_range` check before training --
see below):

    n_train_points     = 289
    n_sequences        = 281
    epochs_trained     = 46
    final_train_loss   = 0.000401
    train_residual_std = 0.052965
    training wall time = 4.3s (GPU-accelerated via tensorflow-metal)
    saved to           = models/hybrid_residual/172.18.0.3.keras

### Getting a genuinely clean window took real, honest troubleshooting

Worth recording plainly, since it's a real part of this step, not just the
happy path: getting to a gap-free 24h window took several rounds of real
diagnosis, not a straight wait-and-check.

- Pod `AGE` (calendar time since creation) is NOT the same thing as "hours
  of real data collected" -- it does not pause during sleep, so relying on
  it alone overstates real coverage whenever the Mac has slept.
- Directly querying Prometheus's `/api/v1/query_range` for the same
  metric the training pipeline uses, and comparing point count to the
  window's clock-time span, gives the real, honest coverage number instead.
- An initial overnight run showed only ~48-56% coverage even in windows
  believed to be "up the whole time." Root-caused with real evidence
  (`pmset -g log`, corrected after an initial too-broad grep pattern
  accidentally matched assertion bookkeeping instead of real state
  transitions): a closed lid combined with running on battery triggers
  repeated short "Maintenance Sleep" / Power Nap cycles that `caffeinate`
  cannot override, each one briefly pausing Docker Desktop's VM and
  Prometheus scraping. Not a pipeline bug -- a real macOS power-management
  interaction.
- Fix: plugged in + lid genuinely open (or external display) for a full
  24h stretch. Verified clean (100% coverage, 288/288 points) before
  training, rather than assumed.
- A subsequent macOS restart (to apply other changes) reset the
  `kubectl port-forward` process, producing a `ConnectionRefusedError` on
  the next training attempt -- expected, not a bug; fixed by simply
  restarting the port-forward.

## Still open

- **The real multi-day run itself.** Nothing above has touched the real
  cluster yet -- `k8s/load-generator.yaml` still needs `kubectl apply`,
  and real history needs to accumulate before `--fit-hours` worth of it
  exists. This is a genuine wait, not a code gap; see the commands below.
- Once enough real hours have accumulated: run
  `scripts/train_real_hybrid_model.py` for real, `kubectl cp` the result
  into the observer pod's PVC, and let the existing shadow-mode mechanism
  (already running since Step 19) evaluate/promote it for real -- no new
  code needed for that part, it already exists.
- Everything in Task #20 (read-before-write reconciliation before
  actuation writes, a rate-limit/circuit-breaker beyond +/-1/tick,
  retry/backoff on transient Prometheus failures, ARIMA
  refit-from-scratch-every-tick, PVC storageClassName/backup policy) is
  still untouched, deliberately deferred until this step's real training
  run is underway.

## Commands to actually start this

Apply the load generator (safe to leave running for days -- bounded to a
single core and 65% max per node, see `k8s/load-generator.yaml`'s own
comments for the full reasoning):

    kubectl apply -f k8s/load-generator.yaml

Then, once enough real hours have accumulated, from a second terminal:

    kubectl port-forward svc/kube-prometheus-stack-prometheus -n monitoring 9090:9090

And in a third:

    python3 scripts/train_real_hybrid_model.py --machine-id <live node id> --fit-hours 48
