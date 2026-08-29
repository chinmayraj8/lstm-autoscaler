"""
LSTM Autoscaler — Live Monitor (replay dashboard).

Replays a historical test-period window frame by frame, as if it were a
live monitoring screen: actual CPU demand, the LSTM's forecast, both
policies' scaling decisions, and a running SLA-violation / cost tally for
each. Data is precomputed by `prepare_replay_data.py` (no model training
happens in this process — see that file's docstring for why).

Run from the project root:
    venv/bin/streamlit run src/dashboard/app.py
"""

import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(
    page_title="LSTM Autoscaler — Live Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @keyframes blink { 50% { opacity: 0.25; } }
    .live-dot { color: #ff4b4b; font-weight: 700; animation: blink 1.4s infinite; }
    .console-line { font-family: "SF Mono", Monaco, monospace; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-family: "SF Mono", Monaco, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_meta() -> dict:
    with open(os.path.join(_DATA_DIR, "machines_meta.json")) as f:
        return json.load(f)


@st.cache_data
def load_trace(machine_id: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(_DATA_DIR, f"replay_{machine_id}.csv"),
        parse_dates=["timestamp"],
    )
    return df


meta = load_meta()
if not meta:
    st.error(
        "No replay data found. Run `venv/bin/python -m src.dashboard.prepare_replay_data` "
        "from the project root first."
    )
    st.stop()

machine_ids = list(meta.keys())

# URL deep-linking (e.g. ?machine=m_2189&step=120&autoplay=1) — lets the
# dashboard be driven to a specific state without clicks, for scripted
# screenshots/recordings as well as sharing a specific moment as a link.
qp = st.query_params
qp_machine = qp.get("machine")
qp_step = qp.get("step")
qp_autoplay = qp.get("autoplay") == "1"
_default_machine_idx = machine_ids.index(qp_machine) if qp_machine in machine_ids else 0

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Playback controls")
machine_id = st.sidebar.selectbox(
    "Machine", machine_ids, index=_default_machine_idx,
    format_func=lambda m: meta[m]["label"],
)
df = load_trace(machine_id)
n_steps = len(df)
window = st.sidebar.slider("Rolling window (steps shown on charts)", 20, 150, 60, step=10)
speed = st.sidebar.select_slider("Playback speed", options=[0.5, 1, 2, 4, 8], value=2)

first_load = "_last_machine" not in st.session_state
if st.session_state.get("_last_machine") != machine_id:
    if first_load and qp_step and qp_step.isdigit():
        st.session_state.frame_idx = min(int(qp_step), n_steps - 1)
    else:
        st.session_state.frame_idx = min(window, n_steps - 1)
    st.session_state.playing = first_load and qp_autoplay
    st.session_state._last_machine = machine_id

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Play", use_container_width=True):
    st.session_state.playing = True
if c2.button("⏸ Pause", use_container_width=True):
    st.session_state.playing = False
if c3.button("⏮ Reset", use_container_width=True):
    st.session_state.playing = False
    st.session_state.frame_idx = min(window, n_steps - 1)

# Auto-advance BEFORE the scrub widget is created, so the widget reflects it.
if st.session_state.playing and st.session_state.frame_idx < n_steps - 1:
    st.session_state.frame_idx += 1
if st.session_state.frame_idx >= n_steps - 1:
    st.session_state.playing = False

frame_idx = st.sidebar.slider("Scrub to step", 0, n_steps - 1, key="frame_idx")

with st.sidebar.expander("ℹ️ About this replay"):
    m = meta[machine_id]
    st.markdown(
        f"""
        **{m['cluster_note']}**

        - Demand scale: `{m['demand_scale']}x`
        - Reactive thresholds: up=`{m['reactive_up']}` down=`{m['reactive_down']}`
        - LSTM decision engine: upw=`{m['lstm_upw']}` sm=`{m['lstm_sm']}`
        - Forecast RMSE / MAE: `{m['lstm_forecast_rmse']}` / `{m['lstm_forecast_mae']}`
        - Trained {m['epochs_trained']} epochs in {m['wall_clock_secs']}s

        This is a **replay of the held-out test split** from the project's
        experiment pipeline (`experiments/pipeline.py`), tuned on a
        validation split that never touches this window. It is not live
        data — it is historical data played back step by step to look like
        a live monitor.
        """
    )

# ── Header / status bar ───────────────────────────────────────────────────────
current = df.iloc[frame_idx]
lo = max(0, frame_idx - window + 1)
win_df = df.iloc[lo: frame_idx + 1]

live_badge = '<span class="live-dot">●</span> REPLAYING' if st.session_state.playing else "⏸ PAUSED"
st.markdown(f"### 🖥️ LSTM Autoscaler — Live Monitor &nbsp;&nbsp; {live_badge}", unsafe_allow_html=True)
st.caption(meta[machine_id]["label"])

hcol1, hcol2, hcol3, hcol4 = st.columns(4)
hcol1.metric("Sim clock", current["timestamp"].strftime("%H:%M:%S"),
             help=current["timestamp"].strftime("%Y-%m-%d"))
hcol2.metric("Step", f"{frame_idx + 1} / {n_steps}")
hcol3.metric("Actual demand", f"{current['actual_demand']:.1f}%")
hcol4.metric("LSTM forecast (t+5m)", f"{current['lstm_forecast']:.1f}%")

if current["reactive_sla_violation"] or current["lstm_sla_violation"]:
    who = []
    if current["reactive_sla_violation"]:
        who.append("Reactive")
    if current["lstm_sla_violation"]:
        who.append("LSTM")
    st.error(f"⚠️ SLA violation this step: {' & '.join(who)} over capacity")
else:
    st.success("✅ Both policies within capacity this step")

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Actual demand vs LSTM forecast**")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=win_df["timestamp"], y=win_df["actual_demand"],
        name="Actual demand", mode="lines", line=dict(color="#4c9aff", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=win_df["timestamp"], y=win_df["lstm_forecast"],
        name="LSTM forecast (t+5m)", mode="lines",
        line=dict(color="#ffab00", width=2, dash="dash"),
    ))
    fig1.add_vline(x=current["timestamp"], line_width=1, line_dash="dot", line_color="gray")
    fig1.update_layout(
        height=340, margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)),
        yaxis_title="Aggregate demand (%)",
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with chart_col2:
    st.markdown("**Fleet capacity vs demand (both policies)**")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=win_df["timestamp"], y=win_df["actual_demand"],
        name="Actual demand", mode="lines", line=dict(color="#4c9aff", width=1.5),
    ))
    fig2.add_trace(go.Scatter(
        x=win_df["timestamp"], y=win_df["reactive_capacity"],
        name="Reactive capacity", mode="lines", line=dict(color="#36b37e", width=2, shape="hv"),
    ))
    fig2.add_trace(go.Scatter(
        x=win_df["timestamp"], y=win_df["lstm_capacity"],
        name="LSTM capacity", mode="lines", line=dict(color="#ff5630", width=2, shape="hv"),
    ))
    viol_r = win_df[win_df["reactive_sla_violation"] == 1]
    viol_l = win_df[win_df["lstm_sla_violation"] == 1]
    if len(viol_r):
        fig2.add_trace(go.Scatter(
            x=viol_r["timestamp"], y=viol_r["actual_demand"], mode="markers",
            marker=dict(color="#36b37e", size=8, symbol="x"), name="Reactive SLA miss",
        ))
    if len(viol_l):
        fig2.add_trace(go.Scatter(
            x=viol_l["timestamp"], y=viol_l["actual_demand"], mode="markers",
            marker=dict(color="#ff5630", size=8, symbol="x"), name="LSTM SLA miss",
        ))
    fig2.add_vline(x=current["timestamp"], line_width=1, line_dash="dot", line_color="gray")
    fig2.update_layout(
        height=340, margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
        yaxis_title="Capacity / demand (%)",
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Running tallies ────────────────────────────────────────────────────────────
st.markdown("#### Running tally (cumulative, since start of replay)")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Reactive SLA violations", f"{current['reactive_cum_sla_pct']:.2f}%")
t2.metric("LSTM SLA violations", f"{current['lstm_cum_sla_pct']:.2f}%")
t3.metric("Reactive cost score", f"{current['reactive_cum_cost']:.4f}")
t4.metric("LSTM cost score", f"{current['lstm_cum_cost']:.4f}")

hist_df = df.iloc[: frame_idx + 1]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["reactive_cum_cost"],
                           name="Reactive cost (cum.)", line=dict(color="#36b37e", width=2)))
fig3.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["lstm_cum_cost"],
                           name="LSTM cost (cum.)", line=dict(color="#ff5630", width=2)))
fig3.update_layout(
    height=220, margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)),
    yaxis_title="Cumulative cost score",
)
st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ── Decision logs ──────────────────────────────────────────────────────────────
log_col1, log_col2 = st.columns(2)


def render_log(container, sub_df: pd.DataFrame, action_col: str, title: str) -> None:
    container.markdown(f"**{title}**")
    events = sub_df[sub_df[action_col] != "hold"].tail(8).iloc[::-1]
    if events.empty:
        container.caption("No scaling actions yet in this window.")
        return
    lines = []
    for _, row in events.iterrows():
        act = row[action_col]
        icon = "🔺" if act.startswith("scale_up") else "🔻"
        lines.append(f'<div class="console-line">{icon} {row["timestamp"].strftime("%H:%M:%S")} — {act}</div>')
    container.markdown("".join(lines), unsafe_allow_html=True)


render_log(log_col1, win_df, "reactive_action", "🟢 Reactive — decision log")
render_log(log_col2, win_df, "lstm_action", "🔴 LSTM — decision log")

# ── Playback loop ─────────────────────────────────────────────────────────────
if st.session_state.playing:
    time.sleep(1.0 / speed)
    st.rerun()
