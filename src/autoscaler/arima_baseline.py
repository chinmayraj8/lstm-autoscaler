"""
ARIMA baseline wired into the same decision engine + simulator as the LSTM
and Reactive policies (Phase 2, Step 10).

Step 6 (experiments/run_baselines_and_sweep.py) only ever compared ARIMA's
point-forecast accuracy (RMSE/MAE) against the LSTM. It never ran ARIMA's
forecasts through the autoscaler's decision engine or fleet simulator, so
there was no cost-score / SLA-rate comparison -- ARIMA's actual autoscaling
behavior was unknown. This module closes that gap by reusing:

- The exact same rolling-forecast procedure as Step 6's `_evaluate_arima`
  (fit on train only, advance state through val with `.append(refit=False)`,
  roll horizon-step-ahead through the test window) -- generalized here to
  also roll through the *validation* window, so ARIMA's decision-engine
  params (under_prov_weight, safety_margin) can be tuned on validation
  exactly like the LSTM's, never touching test until the final run.
- `_build_lstm_targets` (decision.py) unchanged -- it only consumes a
  (y_pred_real, y_actual_real) array pair of shape (n_sequences, horizon)
  and has no LSTM-specific logic, so ARIMA's forecasts plug into it
  directly. This is what guarantees ARIMA is judged on the identical
  yardstick as the LSTM: same DecisionConfig, same SimConfig, same
  `_run_simulation`, same `_compute_cost_score`.

Deliberately NOT imported eagerly from `src.autoscaler/__init__.py` --
statsmodels is a real dependency but there's no reason to pay its import
cost for callers who only need the decision engine / simulator / LSTM path.
Resolved lazily instead, same pattern as the TensorFlow-dependent names in
`forecasting.py` and `experiment.py`.

Determinism note (important for reporting)
--------------------------------------------
Unlike the LSTM (random weight init + internal validation split -> genuine
seed-to-seed variance) and like the Reactive policy (no randomness at all),
ARIMA's MLE fit via statsmodels is deterministic given the same data and
order: same starting parameters (Hannan-Rissanen), same optimizer (L-BFGS-B),
no random restarts. Running the same (machine, order) pair through
`run_arima_experiment` twice reproduces bit-identical numbers -- there is no
seed-driven distribution to report a mean±std over. `run_arima_experiment`
still accepts a `seed` argument (for CSV-schema symmetry with the LSTM
multi-seed harness) but does not use it for any randomness.
"""

import time
import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data import _load_and_prepare, _split_three_way
from .decision import DecisionConfig, _build_lstm_targets
from .simulation import SimConfig, _compute_cost_score, _run_simulation

ARIMA_ORDER = (2, 0, 1)   # same order selected a priori in Step 6 -- frozen
                          # unchanged through Steps 8-13; Step 14 adds
                          # per-machine order re-selection alongside it
                          # (select_arima_order, below), not a replacement.


def select_arima_order(train_flat, p_range=range(0, 5), d_range=(0, 1), q_range=range(0, 5),
                       criterion: str = "aic"):
    """Grid-search ARIMA(p,d,q) order on the TRAIN split only via AIC (default)
    or BIC -- standard Box-Jenkins order selection, no leakage (never touches
    val or test). Each candidate is a single `ARIMA(order).fit()` MLE call on
    `train_flat` (not the rolling forecast -- that happens later, once an
    order is chosen, inside `_arima_rolling_forecast`). Orders that fail to
    converge or hit non-stationarity/non-invertibility errors are skipped,
    not scored -- this is expected for some (p, d, q) combinations and not a
    bug.

    Returns `(best_order, results)` where `results` is a list of
    `(order, aic, bic)` for every order that converged, sorted by `criterion`
    ascending (best first). Falls back to `ARIMA_ORDER` if nothing converged
    (should not happen for reasonable ranges on real demand data).
    """
    from statsmodels.tsa.arima.model import ARIMA

    results = []
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                order = (p, d, q)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = ARIMA(train_flat, order=order).fit()
                    if np.isfinite(fit.aic) and np.isfinite(fit.bic):
                        results.append((order, float(fit.aic), float(fit.bic)))
                except Exception:
                    continue

    if not results:
        return ARIMA_ORDER, results

    key_idx = 1 if criterion == "aic" else 2
    results.sort(key=lambda r: r[key_idx])
    return results[0][0], results


def _inv_flat(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform a (n, horizon) scaled array back to real units.

    Identical formula to forecasting._inv, duplicated here (rather than
    imported) so this module never needs to import forecasting.py and, with
    it, TensorFlow -- see module docstring.
    """
    out = np.zeros_like(arr)
    for h in range(arr.shape[1]):
        out[:, h] = scaler.inverse_transform(arr[:, h].reshape(-1, 1)).flatten()
    return out


def _inv_residual(residual_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform a SCALED RESIDUAL (a difference of two scaled
    values) back to real units -- Step 16 (residual hybrid).

    NOT the same operation as `_inv_flat`/`scaler.inverse_transform` on a
    raw value: for an affine MinMaxScaler (`real = scaled * range + min`),
    `real_a - real_b == (scaled_a - scaled_b) * range` -- the additive
    offset (`min`) cancels out of any difference, so a residual must only
    be multiplied by the scaler's range, never shifted by its offset.
    Applying `inverse_transform` directly to a residual would incorrectly
    add that offset back in.
    """
    rng = float(scaler.data_max_[0] - scaler.data_min_[0])
    return residual_scaled * rng


def _arima_train_walkforward(fit_full, train_flat, lookback, horizon):
    """Honest walk-forward multi-step rolling forecast THROUGH train itself
    -- Step 16 (residual hybrid), used to generate training labels for the
    residual-predicting LSTM.

    Reuses `fit_full`'s ALREADY-ESTIMATED parameters (from the full train
    set via `.fit()` -- identical to the deployed model everywhere else in
    this project) but replays the observation history from scratch via
    `.apply()` (re-uses fixed parameters, does not re-estimate) followed by
    the same `.append(refit=False)` walk-forward loop `_arima_rolling_forecast`
    already uses for val/test, so each forecast at step `i` only reflects
    points revealed so far (indices `0..lookback+i-1`), never later points
    in train. This avoids "the model already saw this exact point while
    being fit" as a leakage concern for the residual labels specifically,
    while keeping parameters identical to the deployed ARIMA (no
    fit-mismatch between the label-generating and the actually-deployed
    model). Standard hybrid-ARIMA-ANN practice (e.g. Zhang 2003) uses
    plain in-sample fitted values for this; this is a stricter walk-forward
    variant of the same idea, made possible by `_arima_rolling_forecast`'s
    existing `.append(refit=False)` mechanism already being available.

    Returns `(y_pred_scaled, y_true_scaled)`, shape `(n_sequences, horizon)`,
    aligned with `_make_sequences(train_data, lookback, horizon)` exactly
    like `_arima_rolling_forecast`'s val/test outputs are.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_walk = fit_full.apply(train_flat[:lookback], refit=False)

    n_sequences = len(train_flat) - lookback - horizon + 1
    if n_sequences <= 0:
        raise ValueError(f"Not enough train data for lookback={lookback} horizon={horizon}")

    y_pred_sc, y_true_sc = [], []
    for i in range(n_sequences):
        fc = fit_walk.forecast(steps=horizon)
        y_pred_sc.append(np.clip(fc, 0.0, 1.0))
        y_true_sc.append(train_flat[i + lookback: i + lookback + horizon])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_walk = fit_walk.append([train_flat[i + lookback]], refit=False)

    return np.array(y_pred_sc), np.array(y_true_sc)


def _arima_rolling_forecast(train_flat, context_flat, target_flat,
                            lookback, horizon, order=ARIMA_ORDER):
    """Roll ARIMA forecasts across `target_flat`, aligned with `_make_sequences`.

    Fits on `train_flat` only (parameters estimated from train, matching the
    scaler-fits-on-train-only convention used throughout this project).
    Advances state through `context_flat` (if any) with `.append(refit=False)`
    -- data the model has "seen" but that did not inform its parameters, e.g.
    validation data when the target is test. Then advances through the first
    `lookback` points of `target_flat` (to mirror the LSTM's lookback window:
    those points are context, not something the model is scored on) and rolls
    an horizon-step-ahead forecast through the rest, appending each true value
    once revealed -- identical procedure to Step 6's `_evaluate_arima`.

    Returns (y_pred_scaled, y_true_scaled), each shape (n_sequences, horizon),
    in the same scaled [0, 1] space `_make_sequences` produces.
    """
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train_flat, order=order).fit()
        if len(context_flat) > 0:
            fit = fit.append(context_flat, refit=False)
        if lookback > 0 and len(target_flat) >= lookback:
            fit = fit.append(target_flat[:lookback], refit=False)

    n_sequences = len(target_flat) - lookback - horizon + 1
    if n_sequences <= 0:
        raise ValueError(f"Not enough data for lookback={lookback} horizon={horizon}")

    y_pred_sc, y_true_sc = [], []
    for i in range(n_sequences):
        fc = fit.forecast(steps=horizon)
        y_pred_sc.append(np.clip(fc, 0.0, 1.0))
        y_true_sc.append(target_flat[i + lookback: i + lookback + horizon])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit.append([target_flat[i + lookback]], refit=False)

    return np.array(y_pred_sc), np.array(y_true_sc)


def _arima_forecast_once(history_flat: np.ndarray, horizon: int, order=ARIMA_ORDER) -> np.ndarray:
    """Fit ARIMA on `history_flat` (scaled [0, 1], same convention as
    `_arima_rolling_forecast`) and forecast `horizon` steps past the end of
    it -- one shot, no walk-forward loop. Step 22 (Stage 4, live forecasting
    loop): unlike every other caller in this file, a live tick has no
    already-known future `target_flat` to roll through (that's what's being
    predicted) -- `_arima_rolling_forecast` is shaped for backtesting
    against already-observed data, not for forecasting into the genuine
    unobserved future, so it doesn't fit this call site. This function
    extracts exactly the fit-and-forecast primitive `_arima_rolling_forecast`
    already uses internally (same `ARIMA` class, same `order`, same
    `np.clip(..., 0.0, 1.0)` convention for scaled output) without its
    rolling/walk-forward machinery, rather than reimplementing the fit call
    from scratch.

    Returns shape `(horizon,)`, still in scaled [0, 1] space -- inverse-
    transform with the same `MinMaxScaler` `history_flat` was scaled with,
    e.g. via `_inv_flat(forecast.reshape(1, -1), scaler)`.
    """
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(history_flat, order=order).fit()
        fc = fit.forecast(steps=horizon)
    return np.clip(fc, 0.0, 1.0)


def _forecast_and_score(y_pred_scaled, y_true_scaled, scaler):
    """Inverse-transform + RMSE/MAE, mirroring forecasting._evaluate_lstm's return shape."""
    y_pred_real = _inv_flat(y_pred_scaled, scaler)
    y_true_real = _inv_flat(y_true_scaled, scaler)
    rmse = float(np.sqrt(mean_squared_error(y_true_real.flatten(), y_pred_real.flatten())))
    mae = float(mean_absolute_error(y_true_real.flatten(), y_pred_real.flatten()))
    return y_pred_real, y_true_real, rmse, mae


def tune_arima_on_validation(seed: int = 42, machine_id=None, nrows=config.NROWS,
                              df_raw=None, demand_scale: float = config.DEMAND_SCALE,
                              order=ARIMA_ORDER, target_builder=_build_lstm_targets,
                              horizon_weights_grid=None) -> dict:
    """Grid-search ARIMA's decision-engine params on validation only.

    Same grids, same objective (minimize validation cost score), same
    procedure as the LSTM half of `experiment.tune_on_validation` -- so
    neither policy is tuned harder than the other. `seed` is accepted only
    for call-signature symmetry with `tune_on_validation`; ARIMA's rolling
    forecast has no seed-dependent randomness (see module docstring).
    Never touches the test split.

    `target_builder` selects the decision engine (default: the original
    greedy `_build_lstm_targets`; pass `_build_multistep_targets` from
    decision.py, Step 12, to tune the multi-step-aware engine on ARIMA's
    forecasts instead) -- same option `tune_on_validation` exposes for the
    LSTM, so both forecasters can be compared under either engine fairly.

    `horizon_weights_grid` (Step 13): same optional third grid dimension
    `tune_on_validation` exposes -- see its docstring for the exact
    backward-compatibility contract (default `None` behaves identically to
    before this parameter existed; `arima_grid`'s tuple shape only grows
    from 3- to 4-tuples when 2+ candidates are actually passed).
    """
    ts, machine_id = _load_and_prepare(machine_id, nrows, df_raw)
    train_data, val_data, _, scaler = _split_three_way(ts, config.FEATURE_COL)

    train_flat = train_data.flatten()
    val_flat = val_data.flatten()

    y_pred_sc, y_true_sc = _arima_rolling_forecast(
        train_flat, np.array([], dtype=train_flat.dtype), val_flat,
        config.LOOKBACK_STEPS, config.HORIZON_STEPS, order,
    )
    y_pred_val, y_val_real, val_rmse, val_mae = _forecast_and_score(y_pred_sc, y_true_sc, scaler)

    sim_cfg = SimConfig()

    multi_hw = horizon_weights_grid is not None and len(horizon_weights_grid) > 1
    hw_candidates = horizon_weights_grid if horizon_weights_grid is not None else [None]

    best_cost = float("inf")
    best_upw = config.DEC_UNDER_WEIGHT
    best_sm = config.SAFETY_MARGIN
    best_hw = None
    grid_results = []

    for upw in config.LSTM_UPW_GRID:
        for sm in config.LSTM_SM_GRID:
            for hw in hw_candidates:
                dec_cfg = DecisionConfig(under_prov_weight=upw)
                hw_kwargs = {} if hw is None else {"horizon_weights": np.asarray(hw)}
                targets, demand = target_builder(
                    y_pred_val, y_val_real, dec_cfg, sim_cfg, demand_scale, sm, **hw_kwargs
                )
                metrics = _run_simulation(demand, targets, sim_cfg)
                cost = _compute_cost_score(metrics, dec_cfg)
                grid_results.append((upw, sm, hw, cost) if multi_hw else (upw, sm, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_upw, best_sm, best_hw = upw, sm, hw

    return {
        "machine_id": machine_id,
        "arima_order": order,
        "arima_under_prov_weight": best_upw,
        "arima_safety_margin": best_sm,
        "arima_horizon_weights": best_hw,
        "arima_val_cost": round(best_cost, 6),
        "arima_val_rmse": round(val_rmse, 6),
        "arima_val_mae": round(val_mae, 6),
        "arima_grid": grid_results,
    }


def run_arima_experiment(
    seed: int,
    under_prov_weight: float = config.DEC_UNDER_WEIGHT,
    safety_margin: float = config.SAFETY_MARGIN,
    machine_id=None,
    nrows=config.NROWS,
    df_raw=None,
    demand_scale: float = config.DEMAND_SCALE,
    order=ARIMA_ORDER,
    target_builder=_build_lstm_targets,
    horizon_weights=None,
) -> dict:
    """Run ARIMA through the full pipeline, evaluated on the TEST split only.

    Mirrors `experiment.run_single_experiment`'s shape and field names
    (prefixed `arima_` instead of `lstm_`) so results drop into the same
    per-machine comparison tables and CSV schema. `under_prov_weight` /
    `safety_margin` must come from `tune_arima_on_validation` so test is
    never involved in tuning -- identical discipline to the LSTM path.
    `target_builder` must match whatever was used during tuning.
    `horizon_weights` (Step 13): the winning candidate from
    `tune_arima_on_validation`'s `horizon_weights_grid` (or `None`,
    unchanged behavior).

    `seed` does not affect ARIMA's fit (see module docstring); it is kept
    in the signature and output only for schema symmetry with the LSTM
    multi-seed harness.
    """
    ts, machine_id = _load_and_prepare(machine_id, nrows, df_raw)
    train_data, val_data, test_data, scaler = _split_three_way(ts, config.FEATURE_COL)

    train_flat = train_data.flatten()
    val_flat = val_data.flatten()
    test_flat = test_data.flatten()

    t0 = time.time()
    y_pred_sc, y_true_sc = _arima_rolling_forecast(
        train_flat, val_flat, test_flat,
        config.LOOKBACK_STEPS, config.HORIZON_STEPS, order,
    )
    wall_clock_secs = time.time() - t0

    y_pred_real, y_test_real, arima_rmse, arima_mae = _forecast_and_score(
        y_pred_sc, y_true_sc, scaler
    )

    dec_cfg = DecisionConfig(under_prov_weight=under_prov_weight)
    sim_cfg = SimConfig()

    hw_kwargs = {} if horizon_weights is None else {"horizon_weights": np.asarray(horizon_weights)}
    arima_targets, demand_series = target_builder(
        y_pred_real, y_test_real, dec_cfg, sim_cfg, demand_scale, safety_margin, **hw_kwargs
    )
    arima_metrics = _run_simulation(demand_series, arima_targets, sim_cfg)
    arima_cost = _compute_cost_score(arima_metrics, dec_cfg)

    n = arima_metrics.total_steps
    return {
        "seed": seed,
        "arima_order": order,
        "arima_forecast_rmse": round(arima_rmse, 6),
        "arima_forecast_mae": round(arima_mae, 6),
        "arima_sla_violation_rate_pct": round(arima_metrics.sla_violations / n * 100, 4),
        "arima_over_prov_waste_pct": round(arima_metrics.over_prov_steps / n * 100, 4),
        "arima_cost_score": arima_cost,
        "arima_wall_clock_secs": round(wall_clock_secs, 1),
    }
