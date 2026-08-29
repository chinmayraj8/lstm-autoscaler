"""
Central configuration for the LSTM autoscaler pipeline.

Every value here is copied byte-for-byte from the original
experiments/pipeline.py's "Frozen constants" block — this module changes
*where* these values live, not what they are. No numeric behavior changes
as a result of this refactor.

The one deliberate improvement: DATA_PATH can now be overridden with the
LSTM_AUTOSCALER_DATA_PATH environment variable. The default still resolves
to ~/Desktop/machine_usage_bigger.csv, so nothing changes for anyone who
doesn't set the variable.
"""

import os

# ── Data path (overridable; see module docstring) ─────────────────────────────
DATA_PATH = os.environ.get(
    "LSTM_AUTOSCALER_DATA_PATH",
    os.path.expanduser("~/Desktop/machine_usage_bigger.csv"),
)

# ── Frozen constants (unchanged from experiments/pipeline.py) ─────────────────
DEMAND_SCALE = 20.0            # kept for backward compat with Steps 1-3 scripts
TARGET_MEAN_LOAD_PCT = 115.0   # per-machine calibration target: mean aggregate demand
LOOKBACK_STEPS = 6
HORIZON_STEPS = 3
FEATURE_COL = "cpu_util_percent"
VAL_RATIO = 0.2
TEST_RATIO = 0.2
NROWS = 500_000

LSTM_UNITS = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
MAX_EPOCHS = 60
BATCH_SIZE = 64
VAL_SPLIT = 0.1   # Keras internal val split used for early-stopping only
ES_PATIENCE = 10

DEC_SERVER_CAPACITY = 80.0
DEC_MIN_SERVERS = 1
DEC_MAX_SERVERS = 10
DEC_OVER_WEIGHT = 1.0
DEC_UNDER_WEIGHT = 20.0   # original notebook value; may be overridden by tuning
DEC_SCALE_STEP = 1
SAFETY_MARGIN = 0.25      # original notebook value; may be overridden by tuning

SIM_INITIAL_SERVERS = 2
SIM_SERVER_CAPACITY = 80.0
SIM_STARTUP_DELAY = 1

# Original (untuned) Reactive thresholds; may be overridden by tuning
REACTIVE_UP_THRESHOLD = 80.0
REACTIVE_DOWN_THRESHOLD = 30.0

# Tuning search spaces
REACTIVE_UP_GRID = [60, 65, 70, 75, 80, 85, 90]
REACTIVE_DOWN_GRID = [10, 15, 20, 25, 30, 35, 40]
LSTM_UPW_GRID = [5, 10, 15, 20, 25, 30, 40]
LSTM_SM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
