"""
Streamlit app for three-phase transmission line profile analysis.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure the app's directory is on the path (needed for Streamlit Cloud)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import linalg
import streamlit as st
import plotly.graph_objects as go
from line_parameters import compute_line_parameters

st.set_page_config(page_title="Transmission Line Profile", layout="wide")
st.title("Three-Phase Transmission Line Profile")

# ==========================================================================
# ALL INPUTS INSIDE A FORM — single "Calculate" button
# ==========================================================================

with st.form("inputs"):

    # --- Sidebar inputs ---
    st.sidebar.header("System")
    freq = st.sidebar.number_input("Frequency (Hz)", value=50.0, step=10.0)
    rho_earth = st.sidebar.number_input("Earth resistivity (ohm·m)", value=100.0, step=50.0)
    V_rated = st.sidebar.number_input("Rated voltage, line-to-line (kV)", value=220.0, step=10.0) * 1e3
    length = st.sidebar.number_input("Line length (km)", value=80.0, step=10.0)

    st.sidebar.header("Three-phase sending-end power")
    P_mw = st.sidebar.number_input("P (MW)", value=500.0, step=50.0)
    Q_mvar = st.sidebar.number_input("Q (MVAr)", value=60.0, step=10.0)

    # --- Conductor table ---
    st.header("Conductor table")
    st.markdown("Edit the table below to add/remove phase and ground conductors. "
                "Set **Type** to `phase` or `ground`.")

    default_data = pd.DataFrame([
        {"Type": "phase",  "R_dc (ohm/km)": 0.0958, "Diameter (mm)": 24.0, "x (m)": -7.0, "y (m)": 22.0},
        {"Type": "phase",  "R_dc (ohm/km)": 0.0958, "Diameter (mm)": 24.0, "x (m)":  0.0, "y (m)": 22.0},
        {"Type": "phase",  "R_dc (ohm/km)": 0.0958, "Diameter (mm)": 24.0, "x (m)":  7.0, "y (m)": 22.0},
        {"Type": "ground", "R_dc (ohm/km)": 4.0000, "Diameter (mm)": 11.0, "x (m)":  0.0, "y (m)": 27.0},
    ])

    conductor_df = st.data_editor(
        default_data,
        num_rows="dynamic",
        column_config={
            "Type": st.column_config.SelectboxColumn(
                options=["phase", "ground"],
                required=True,
            ),
            "R_dc (ohm/km)": st.column_config.NumberColumn(format="%.4f", required=True),
            "Diameter (mm)": st.column_config.NumberColumn(format="%.1f", required=True),
            "x (m)": st.column_config.NumberColumn(format="%.1f", required=True),
            "y (m)": st.column_config.NumberColumn(format="%.1f", required=True),
        },
        use_container_width=True,
    )

    # --- Submit button ---
    submitted = st.form_submit_button("Calculate", type="primary", use_container_width=True)

# ==========================================================================
# STOP HERE UNTIL BUTTON IS PRESSED (or on first load, run with defaults)
# ==========================================================================

P = P_mw * 1e6
Q = Q_mvar * 1e6

# --- Build conductor list ---
conductors = []
phase_idx = []
ground_idx = []

for i, row in conductor_df.iterrows():
    conductors.append({
        'r_dc': row["R_dc (ohm/km)"],
        'diameter': row["Diameter (mm)"],
        'x': row["x (m)"],
        'y': row["y (m)"],
    })
    if row["Type"] == "phase":
        phase_idx.append(i)
    else:
        ground_idx.append(i)

n_phase = len(phase_idx)

if n_phase < 3:
    st.error("At least 3 phase conductors are required.")
    st.stop()

if n_phase != 3:
    st.warning(f"Model expects exactly 3 phase conductors, got {n_phase}. "
               "Using the first 3 as phases A, B, C.")
    phase_idx = phase_idx[:3]

# ==========================================================================
# COMPUTATION
# ==========================================================================

R, L, C_mat, G = compute_line_parameters(conductors, phase_idx, ground_idx, freq, rho_earth)

omega = 2 * np.pi * freq
Z = R + 1j * omega * L * 1e-3
Y = (G * 1e-6) + 1j * omega * C_mat * 1e-9

zeros = np.zeros((3, 3))
F_mat = np.block([[zeros, -Z],
                  [-Y, zeros]])

M = linalg.expm(F_mat * length)
A_m = M[:3, :3]
B_m = M[:3, 3:]
C_m = M[3:, :3]
D_m = M[3:, 3:]

a = np.exp(1j * 2 * np.pi / 3)
u = np.array([1.0, a**2, a])
Vs = V_rated / np.sqrt(3)
V0 = Vs * u

B_inv = np.linalg.inv(B_m)
W = (P + 1j * Q + Vs**2 * (u @ np.conj(B_inv @ (A_m @ u)))) / \
    (Vs * (u @ np.conj(B_inv @ u)))

Vr = np.abs(W)
delta = -np.angle(W)

I0 = B_inv @ (Vr * np.exp(1j * delta) * u - Vs * A_m @ u)

phi0 = np.concatenate([V0, I0])
n_points = 501
x = np.linspace(0, length, n_points)
phi = np.zeros((6, n_points), dtype=complex)
for i, xi in enumerate(x):
    phi[:, i] = linalg.expm(F_mat * xi) @ phi0

V_profile = phi[:3, :]
I_profile = phi[3:, :]

Vl = V_profile[:, -1]
Il = I_profile[:, -1]
Sr = Vl @ np.conj(Il)

# ==========================================================================
# DISPLAY — RLCG MATRICES
# ==========================================================================

st.header("Distributed parameters (RLCG)")
col_r, col_l, col_c, col_g = st.columns(4)

def fmt_matrix(mat, fmt=".4f"):
    return "\n".join(["  ".join([f"{v:{fmt}}" for v in row]) for row in mat])

with col_r:
    st.markdown("**R** (ohm/km)")
    st.code(fmt_matrix(R))
with col_l:
    st.markdown("**L** (mH/km)")
    st.code(fmt_matrix(L))
with col_c:
    st.markdown("**C** (nF/km)")
    st.code(fmt_matrix(C_mat))
with col_g:
    st.markdown("**G** (uS/km)")
    st.code(fmt_matrix(G))

# ==========================================================================
# DISPLAY — SUMMARY
# ==========================================================================

st.header("Results")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Vr", f"{Vr:.0f} V", f"{Vr / Vs * 100 - 100:.2f}% from Vs")
with col_s2:
    st.metric("delta", f"{np.degrees(delta):.4f} deg")
with col_s3:
    st.metric("Losses", f"{(P - Sr.real)/1e6:.2f} MW")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown(f"**Sending (3-phase):** P = {P/1e6:.1f} MW, Q = {Q/1e6:.1f} MVAr")
with col_p2:
    st.markdown(f"**Receiving (3-phase):** P = {Sr.real/1e6:.1f} MW, Q = {Sr.imag/1e6:.1f} MVAr")

# ==========================================================================
# DISPLAY — CURRENT UNBALANCE
# ==========================================================================

st.header("Current unbalance")

def current_unbalance_table(I_vec):
    mag = np.abs(I_vec)
    ang = np.degrees(np.angle(I_vec))
    mag_mean = np.mean(mag)
    spacings = np.array([
        (ang[1] - ang[0] + 180) % 360 - 180,
        (ang[2] - ang[1] + 180) % 360 - 180,
        (ang[0] - ang[2] + 180) % 360 - 180,
    ])
    ang_mean = np.mean(np.abs(spacings))
    rows = []
    for k, name in enumerate(["A", "B", "C"]):
        mag_dev = (mag[k] - mag_mean) / mag_mean * 100
        spacing_dev = np.abs(spacings[k]) - ang_mean
        rows.append({
            "Phase": name,
            "|I| (A)": f"{mag[k]:.2f}",
            "Angle (deg)": f"{ang[k]:.2f}",
            "Mag. dev. (%)": f"{mag_dev:+.2f}",
            "Spacing dev. (deg)": f"{spacing_dev:+.2f}",
        })
    return rows, mag_mean, ang_mean

col_u1, col_u2 = st.columns(2)
with col_u1:
    rows_s, mag_mean_s, ang_mean_s = current_unbalance_table(I0)
    st.markdown(f"**Sending end** (mean |I| = {mag_mean_s:.2f} A, mean spacing = {ang_mean_s:.2f} deg)")
    st.table(rows_s)
with col_u2:
    rows_r, mag_mean_r, ang_mean_r = current_unbalance_table(Il)
    st.markdown(f"**Receiving end** (mean |I| = {mag_mean_r:.2f} A, mean spacing = {ang_mean_r:.2f} deg)")
    st.table(rows_r)

# ==========================================================================
# DISPLAY — INTERACTIVE PROFILES (Plotly)
# ==========================================================================

st.header("Profiles along the line")

phase_names = ["Phase A", "Phase B", "Phase C"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig_v = go.Figure()
for k in range(3):
    fig_v.add_trace(go.Scatter(
        x=x, y=np.abs(V_profile[k, :]) / Vs,
        name=phase_names[k],
        line=dict(color=colors[k]),
    ))
fig_v.update_layout(
    title="Voltage profile",
    xaxis_title="Distance (km)",
    yaxis_title="|V| (pu)",
    height=400,
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
)
st.plotly_chart(fig_v, use_container_width=True)

st.divider()

fig_i = go.Figure()
for k in range(3):
    I_mag = np.abs(I_profile[k, :])
    fig_i.add_trace(go.Scatter(
        x=x, y=I_mag,
        name=phase_names[k],
        line=dict(color=colors[k]),
    ))
fig_i.update_layout(
    title="Current profile",
    xaxis_title="Distance (km)",
    yaxis_title="|I| (A)",
    height=400,
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
)
st.plotly_chart(fig_i, use_container_width=True)

current_var_rows = []
for k in range(3):
    I_mag = np.abs(I_profile[k, :])
    I_max, I_min = I_mag.max(), I_mag.min()
    variation = (I_max - I_min) / I_min * 100
    current_var_rows.append({
        "Phase": phase_names[k],
        "Min (A)": f"{I_min:.2f}",
        "Max (A)": f"{I_max:.2f}",
        "Variation (%)": f"{variation:.2f}",
    })
st.table(current_var_rows)
