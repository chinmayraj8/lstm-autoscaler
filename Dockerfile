# Minimal in-cluster observer runner (Step 23, Stage 5).
#
# Scope, stated plainly: this is JUST enough to run Stage 4's scheduler
# (src/autoscaler/live_loop.py) + the FastAPI /shadow/* endpoints
# (src/api/main.py) continuously against the in-cluster Prometheus, instead
# of depending on a laptop staying on and port-forwarded. It is NOT the
# broader "Part 2" Docker/Kubernetes deployment work -- no /forecast model
# serving is expected to run from this image (see LSTM_AUTOSCALER_SKIP_LSTM_MODEL
# below; lstm_model.keras and the training CSV are both excluded via
# .dockerignore), no multi-stage build, no non-root user hardening, no
# health-check tuning beyond what k8s/observer.yaml's probes need. Still
# observe-only: nothing in this image calls a real scaling API -- see
# live_loop.py's and main.py's own module docstrings.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# LSTM_AUTOSCALER_SKIP_LSTM_MODEL and LSTM_AUTOSCALER_PROMETHEUS_URL are set
# in k8s/observer.yaml, not baked in here -- keeps this image reusable for
# local testing against a port-forwarded Prometheus too (see progress doc).
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
