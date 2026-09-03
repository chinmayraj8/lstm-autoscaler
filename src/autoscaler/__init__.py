"""
LSTM autoscaler core package.

This is the tested, importable home for the pipeline logic that used to
live entirely in experiments/pipeline.py: data loading/preprocessing,
LSTM forecasting, the decision engine, the fleet simulator, and per-machine
demand-scale calibration.

Public API
----------
tune_on_validation(seed)          -> dict of best params for both policies
run_single_experiment(seed, ...)  -> dict of all metrics on the TEST split

experiments/pipeline.py now re-exports everything below for backward
compatibility -- existing scripts that do
`from experiments.pipeline import X` keep working unchanged.

Import cost, on purpose
------------------------
Everything eagerly imported below (config, calibration, data, decision,
simulation) needs only numpy/pandas/scikit-learn. The two names that
actually train or run the LSTM -- run_single_experiment and
tune_on_validation -- along with the low-level forecasting helpers, need
TensorFlow, so they're loaded lazily via module __getattr__ the first
time they're accessed, not at `import src.autoscaler` time. That means
`from src.autoscaler import DecisionConfig, calibrate_demand_scale, ...`
(or anything from the decision engine, simulator, or calibration) works
in a plain numpy/pandas environment with no TensorFlow install at all --
useful for fast unit tests and CI. Actually calling run_single_experiment
or tune_on_validation still needs TensorFlow installed, same as before.
"""

from . import config
from .config import (
    DATA_PATH,
    DEMAND_SCALE,
    TARGET_MEAN_LOAD_PCT,
    LOOKBACK_STEPS,
    HORIZON_STEPS,
    FEATURE_COL,
    VAL_RATIO,
    TEST_RATIO,
    NROWS,
    LSTM_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    MAX_EPOCHS,
    BATCH_SIZE,
    VAL_SPLIT,
    ES_PATIENCE,
    DEC_SERVER_CAPACITY,
    DEC_MIN_SERVERS,
    DEC_MAX_SERVERS,
    DEC_OVER_WEIGHT,
    DEC_UNDER_WEIGHT,
    DEC_SCALE_STEP,
    SAFETY_MARGIN,
    SIM_INITIAL_SERVERS,
    SIM_SERVER_CAPACITY,
    SIM_STARTUP_DELAY,
    REACTIVE_UP_THRESHOLD,
    REACTIVE_DOWN_THRESHOLD,
)
from .calibration import calibrate_demand_scale, check_feasibility
from .data import (
    _load_and_prepare,
    _make_multivariate_sequences,
    _make_sequences,
    _pick_best_machine,
    _prepare_multivariate,
    _prepare_timeseries,
    _split_three_way,
    _split_three_way_multivariate,
)
from .decision import (
    DecisionConfig,
    _build_lstm_targets,
    _build_multistep_targets,
    _compute_penalty,
    _compute_penalty_multistep,
    _decide_scaling,
    _decide_scaling_multistep,
)
from .simulation import (
    SimConfig,
    SimMetrics,
    _compute_cost_score,
    _reactive_autoscaler,
    _run_simulation,
    verdict,
)
from . import shadow
from . import metrics_source
from . import live_loop

# Names that require TensorFlow -- resolved lazily, see module docstring.
_LAZY = {
    "run_single_experiment": (".experiment", "run_single_experiment"),
    "tune_on_validation":    (".experiment", "tune_on_validation"),
    "_build_lstm_model":     (".forecasting", "_build_lstm_model"),
    "_evaluate_lstm":        (".forecasting", "_evaluate_lstm"),
    "_evaluate_naive":       (".forecasting", "_evaluate_naive"),
    "_inv":                  (".forecasting", "_inv"),
    "_train_lstm":           (".forecasting", "_train_lstm"),
    # Requires statsmodels (not TensorFlow), but resolved lazily for the same
    # reason: no import cost for callers who don't need it. See
    # arima_baseline.py's module docstring.
    "run_arima_experiment":      (".arima_baseline", "run_arima_experiment"),
    "tune_arima_on_validation":  (".arima_baseline", "tune_arima_on_validation"),
    "_arima_rolling_forecast":   (".arima_baseline", "_arima_rolling_forecast"),
    "ARIMA_ORDER":                (".arima_baseline", "ARIMA_ORDER"),
    "select_arima_order":        (".arima_baseline", "select_arima_order"),
    "_arima_train_walkforward":  (".arima_baseline", "_arima_train_walkforward"),
    "_inv_residual":             (".arima_baseline", "_inv_residual"),
    "_inv_flat":                 (".arima_baseline", "_inv_flat"),
}


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    if name in _LAZY:
        import importlib
        module_name, attr_name = _LAZY[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache on the module so it's fast next time
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "config",
    "DATA_PATH", "DEMAND_SCALE", "TARGET_MEAN_LOAD_PCT", "LOOKBACK_STEPS",
    "HORIZON_STEPS", "FEATURE_COL", "VAL_RATIO", "TEST_RATIO", "NROWS",
    "LSTM_UNITS", "DROPOUT_RATE", "LEARNING_RATE", "MAX_EPOCHS", "BATCH_SIZE",
    "VAL_SPLIT", "ES_PATIENCE", "DEC_SERVER_CAPACITY", "DEC_MIN_SERVERS",
    "DEC_MAX_SERVERS", "DEC_OVER_WEIGHT", "DEC_UNDER_WEIGHT", "DEC_SCALE_STEP",
    "SAFETY_MARGIN", "SIM_INITIAL_SERVERS", "SIM_SERVER_CAPACITY",
    "SIM_STARTUP_DELAY", "REACTIVE_UP_THRESHOLD", "REACTIVE_DOWN_THRESHOLD",
    "calibrate_demand_scale", "check_feasibility",
    "_load_and_prepare", "_make_sequences", "_pick_best_machine",
    "_prepare_timeseries", "_split_three_way",
    "_make_multivariate_sequences", "_prepare_multivariate", "_split_three_way_multivariate",
    "DecisionConfig", "_build_lstm_targets", "_compute_penalty", "_decide_scaling",
    "_build_multistep_targets", "_compute_penalty_multistep", "_decide_scaling_multistep",
    "run_single_experiment", "tune_on_validation",
    "_build_lstm_model", "_evaluate_lstm", "_evaluate_naive", "_inv", "_train_lstm",
    "SimConfig", "SimMetrics", "_compute_cost_score", "_reactive_autoscaler",
    "_run_simulation", "verdict", "shadow", "metrics_source", "live_loop",
    "run_arima_experiment", "tune_arima_on_validation", "_arima_rolling_forecast",
    "ARIMA_ORDER", "select_arima_order",
    "_arima_train_walkforward", "_inv_residual", "_inv_flat",
]
