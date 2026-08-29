"""
Pipeline for the LSTM autoscaler experiment harness.

As of the src/ refactor (see progress/ for the dated entry), the actual
logic lives in src/autoscaler/ as a tested package. This module is now a
thin backward-compatible re-export so every existing script and notebook
that does `from experiments.pipeline import X` keeps working unchanged,
with byte-for-byte identical behavior.

If you're writing new code, prefer `from src.autoscaler import X` (or
`import src.autoscaler as autoscaler`) directly — this shim exists for
compatibility, not as the preferred import path going forward.

Split: 60 % train / 20 % validation / 20 % test (chronological).
- Scaler fits on train only.
- Validation is used exclusively for hyperparameter tuning (both policies).
- Test is touched once, at the end, for the final reported numbers.

Public API
----------
tune_on_validation(seed)          -> dict of best params for both policies
run_single_experiment(seed, ...)  -> dict of all metrics on the TEST split
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autoscaler import (  # noqa: E402,F401
    # constants
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
    # calibration
    calibrate_demand_scale,
    check_feasibility,
    # data
    _load_and_prepare,
    _make_sequences,
    _pick_best_machine,
    _prepare_timeseries,
    _split_three_way,
    # decision
    DecisionConfig,
    _build_lstm_targets,
    _compute_penalty,
    _decide_scaling,
    # forecasting
    _build_lstm_model,
    _evaluate_lstm,
    _evaluate_naive,
    _inv,
    _train_lstm,
    # simulation
    SimConfig,
    SimMetrics,
    _compute_cost_score,
    _reactive_autoscaler,
    _run_simulation,
    # public orchestration
    run_single_experiment,
    tune_on_validation,
)

# Private grid-search constants — internal to tune_on_validation, kept here
# under their original underscored names purely for compatibility with
# anything that may have imported them directly.
from src.autoscaler.config import (  # noqa: E402,F401
    REACTIVE_UP_GRID as _REACTIVE_UP_GRID,
    REACTIVE_DOWN_GRID as _REACTIVE_DOWN_GRID,
    LSTM_UPW_GRID as _LSTM_UPW_GRID,
    LSTM_SM_GRID as _LSTM_SM_GRID,
)
