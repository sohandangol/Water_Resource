import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
 
st.set_page_config(layout="wide", page_title="Kinematic Wave Equation solver")
st.title("Kinematic Wave Equation solver")
 
st.markdown("""
This app simulates a wave moving down a rectangular channel using the **kinematic wave**
simplification of open-channel flow (continuity equation + Manning's equation,
assuming bed slope = friction slope). Adjust the channel and inflow hydrograph
on the left and see how the wave moves downstream.
""")
 
 
st.sidebar.header("Channel properties")
L = st.sidebar.slider("Channel length L (m)", 1000.0, 15000.0, 5000.0, 500.0)
W = st.sidebar.slider("Channel width W (m)", 5.0, 100.0, 10.0, 1.0)
S0 = st.sidebar.slider("Bed slope S0 (m/m)", 0.0005, 0.02, 0.001, 0.0005, format="%.4f")
manning_n = st.sidebar.slider("Manning's n", 0.012, 0.07, 0.03, 0.001, format="%.3f")
 

DX_TARGET = 100.0  # meters per segment
N = max(int(round(L / DX_TARGET)), 10)  # at least 10 segments, even for short channels
 
st.sidebar.header("Inflow hydrograph (boundary condition)")
Q_base = st.sidebar.slider("Baseflow Q₀ (m³/s)", 1.0, 50.0, 5.0, 1.0)
Q_peak = st.sidebar.slider("Peak flow Q_peak (m³/s)", 10.0, 300.0, 50.0, 5.0)
t_peak_hr = st.sidebar.slider("Time to peak (hr)", 0.5, 5.0, 2.0, 0.5)
t_end_hr = st.sidebar.slider("Time back to baseflow (hr)", 1.0, 8.0, 4.0, 0.5)
sim_duration_hr = st.sidebar.slider("Total simulation time (hr)", 3.0, 12.0, 6.0, 1.0)
 

def area_from_discharge(Q, W, n, S0):
    alpha = (n * W**(2/3)) / (S0**0.5)
    return (alpha * Q) ** (3/5)
 
def discharge_from_area(A, W, n, S0):
    return (1/n) * S0**0.5 * A**(5/3) / W**(2/3)
 
def inflow_hydrograph(t, Q_base, Q_peak, t_peak, t_end_rise):
    if t <= t_peak:
        return Q_base + (Q_peak - Q_base) * (t / t_peak)
    elif t <= t_end_rise:
        return Q_peak - (Q_peak - Q_base) * ((t - t_peak) / (t_end_rise - t_peak))
    else:
        return Q_base
 

@st.cache_data
def run_simulation(L, W, S0, manning_n, N, Q_base, Q_peak, t_peak_hr, t_end_hr, sim_duration_hr):
    dx = L / N
    t_peak = t_peak_hr * 3600
    t_end_rise = t_end_hr * 3600
    sim_duration = sim_duration_hr * 3600
 
    # Courant condition 
    # Working to figure out how to check and validating the numerical stability 
    # Is courant number < 1 ? check is left to do!
    # In future need to make it adaptive to the changing wave speed as the flood wave moves downstream
    A_peak_est = area_from_discharge(Q_peak, W, manning_n, S0)
    c_wave = (5/3) * discharge_from_area(A_peak_est, W, manning_n, S0) / A_peak_est
    dt = 0.8 * dx / c_wave
    n_steps = int(sim_duration / dt)
 
    Q = np.full(N + 1, Q_base)
    A = area_from_discharge(Q, W, manning_n, S0)
 
    time_axis = []
    Q_history = np.zeros((n_steps, N + 1))
 
    for step in range(n_steps):
        t = step * dt
        Q[0] = inflow_hydrograph(t, Q_base, Q_peak, t_peak, t_end_rise)
        A[0] = area_from_discharge(Q[0], W, manning_n, S0)
 
        A_new = A.copy()
        for i in range(1, N + 1):
            A_new[i] = A[i] - (dt / dx) * (Q[i] - Q[i - 1])
            A_new[i] = max(A_new[i], 1e-6)
        A = A_new
        Q = discharge_from_area(A, W, manning_n, S0)
 
        time_axis.append(t / 3600)
        Q_history[step, :] = Q
 
    return np.array(time_axis), Q_history, dt, c_wave, dx
 
time_axis, Q_history, dt, c_wave, dx = run_simulation(
    L, W, S0, manning_n, N, Q_base, Q_peak, t_peak_hr, t_end_hr, sim_duration_hr
)


fig, ax = plt.subplots(figsize=(9, 5))
mid = N // 2
ax.plot(time_axis, Q_history[:, 0], label=f"Upstream (x = 0 km)")
ax.plot(time_axis, Q_history[:, mid], label=f"Midpoint (x = {L/2/1000:.1f} km)")
ax.plot(time_axis, Q_history[:, -1], label=f"Downstream (x = {L/1000:.1f} km)")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Discharge Q (m³/s)")
ax.set_title("Kinematic Wave — Hydrograph Translation Downstream")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)