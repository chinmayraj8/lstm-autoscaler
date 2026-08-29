"""
Makes `src.autoscaler` importable when pytest is run from the repo root
(matches how experiments/pipeline.py and the run_*.py scripts already find
the project root).

Tests import the individual submodules (src.autoscaler.decision,
.simulation, .calibration, .data) rather than the top-level `src.autoscaler`
package on purpose: the top-level package's __init__ pulls in
experiment.py -> forecasting.py -> tensorflow, and these tests are meant
to run fast, in CI, without a TensorFlow install. See
src/autoscaler/forecasting.py's docstring for why TensorFlow is isolated
to that one module.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
