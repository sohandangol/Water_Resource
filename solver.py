import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide", page_title="Hydraulic Flood Simulator")

st.markdown("""
<style>
    .block-container { 
        padding-top: 3.2rem !important; 
        padding-bottom: 0.2rem !important; 
        padding-left: 2rem !important; 
        padding-right: 2rem !important; 
    }
    h1 { font-size: 1.35rem !important; margin-top: 0rem !important; margin-bottom: 0.2rem !important; }
    div[data-testid="stMetricValue"] { font-size: 0.95rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    section[data-testid="stSidebar"] div.stSlider { margin-bottom: -10px; }
</style>
""", unsafe_allow_html=True)

st.title("Kinematic & Diffusive Wave Open-Channel")


st.sidebar.header("1. Governing Equation")
routing_model = st.sidebar.selectbox(
    "Routing Formulation", 
    ["Kinematic Wave", "Diffusive Wave"]
)

st.sidebar.header("2. Numerical Discretization")
sb_col1, sb_col2 = st.sidebar.columns(2)
with sb_col1:
    dx = st.slider("Δx (m)", 100.0, 1500.0, 400.0, 50.0, format="%.0f")
with sb_col2:
    dt_min = st.slider("Δt (min)", 0.5, 10.0, 2.0, 0.5, format="%.1f")
dt = dt_min * 60.0

beta = st.sidebar.slider("Exponent β (0.60 for Manning)", 0.50, 0.80, 0.60, 0.01, format="%.2f")

# Placeholder for Cournat number validation message
courant_placeholder = st.sidebar.empty()


st.sidebar.header("3. Channel Hydraulics")
So = st.sidebar.slider("Bed Slope S0 (m/m)", 0.0005, 0.0200, 0.0040, 0.0005, format="%.4f")
manning_n = st.sidebar.slider("Manning's n", 0.012, 0.070, 0.030, 0.001, format="%.3f")
B = st.sidebar.slider("Bottom Width B (m)", 10.0, 150.0, 50.0, 5.0, format="%.1f")
L = st.sidebar.slider("Total Reach Length L (m)", 1000.0, 15000.0, 6000.0, 200.0, format="%.0f")

# Exact Manning alpha derivation: A = alpha * Q^beta
alpha = (manning_n * (B ** (2.0 / 3.0)) / np.sqrt(So)) ** beta


st.sidebar.header("4. Boundary Hydrograph")
Q_base = st.sidebar.slider("Baseflow Q₀ (m³/s)", 5.0, 80.0, 25.0, 1.0, format="%.1f")
Q_peak = st.sidebar.slider("Peak Flow Q_peak (m³/s)", 40.0, 600.0, 160.0, 5.0, format="%.1f")
peak_time_min = st.sidebar.slider("Time to Peak t_p (min)", 15.0, 120.0, 50.0, 5.0, format="%.1f")
sigma_t_min = st.sidebar.slider("Flood Spread σ_t (min)", 5.0, 45.0, 18.0, 1.0, format="%.1f")



c_k_max = 1.0 / (alpha * beta * (max(Q_peak, 1.0) ** (beta - 1.0)))
courant_number = c_k_max * dt / dx

with courant_placeholder.container():
    st.markdown(f"**Max Wave Celerity ($c_k$):** `{c_k_max:.2f} m/s`")
    
    if courant_number <= 1.0:
        st.success(f"Courant No. $C_r = {courant_number:.2f} \\le 1.0$ (Stable & Accurate)")
    else:
        st.error(
            f"Courant No. $C_r = {courant_number:.2f} > 1.0$\n\n"
            "**Issue:** Greater numerical dispersion.\n\n"
            "**Remedy:**\n"
            "* **Increase $\\Delta x$** (larger distance between nodes). or\n" 
            "* **Decrease $\\Delta t$** (finer time step)."
            
        )

# SOLVERS for (KINEMATIC & DIFFUSION WAVE)
def area_from_Q(Q, B_w, So_val, n):
    """Full-section Manning inversion via bisection."""
    if Q <= 0:
        return 1e-6
    A_lo, A_hi = 1e-5, 1e4
    for _ in range(40):
        A_mid = 0.5 * (A_lo + A_hi)
        h = A_mid / B_w
        P = B_w + 2.0 * h
        R = A_mid / P
        Q_mid = (1.0 / n) * A_mid * (R ** (2.0 / 3.0)) * np.sqrt(So_val)
        if Q_mid < Q:
            A_lo = A_mid
        else:
            A_hi = A_mid
    return A_mid

def solve_thomas(a, b, c, d):
    """O(N) Tridiagonal matrix solver."""
    n = len(d)
    c_p = np.zeros(n)
    d_p = np.zeros(n)
    c_p[0] = c[0] / b[0]
    d_p[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_p[i - 1]
        c_p[i] = c[i] / denom if i < n - 1 else 0.0
        d_p[i] = (d[i] - a[i] * d_p[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d_p[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_p[i] - c_p[i] * x[i + 1]
    return x

@st.cache_data
def solve_routing(model_choice, Q_up, alpha_val, beta_val, dx_val, dt_val, n_pts, Q_init, B_val, So_val):
    n_time = len(Q_up)
    Q = np.zeros((n_time, n_pts))
    Q[0, :] = Q_init
    Q[:, 0] = Q_up

    if model_choice == "Kinematic Wave":
        for j in range(n_time - 1):
            for i in range(n_pts - 1):
                Q_j_i1 = Q[j, i + 1]
                Q_j1_i = Q[j + 1, i]
                Q_avg = max((Q_j_i1 + Q_j1_i) / 2.0, 1e-4)
                term = alpha_val * beta_val * (Q_avg ** (beta_val - 1.0))
                Q[j + 1, i + 1] = max((dt_val / dx_val * Q_j1_i + term * Q_j_i1) / (dt_val / dx_val + term), 1e-4)
    else:
        Q_ref = 0.5 * (np.max(Q_up) + Q_init)
        c_k = 1.0 / (alpha_val * beta_val * (Q_ref ** (beta_val - 1.0)))
        D = Q_ref / (2.0 * B_val * So_val)

        r_adv = c_k * dt_val / (2.0 * dx_val)
        r_diff = D * dt_val / (dx_val ** 2)

        m = n_pts - 2
        a = np.full(m, -r_diff - r_adv)
        b = np.full(m, 1.0 + 2.0 * r_diff)
        c = np.full(m, -r_diff + r_adv)

        for j in range(n_time - 1):
            rhs = Q[j, 1:n_pts - 1].copy()
            rhs[0] -= a[0] * Q[j + 1, 0]
            b_modified = b.copy()
            b_modified[-1] += c[-1]
            sol = solve_thomas(a, b_modified, c, rhs)
            Q[j + 1, 1:n_pts - 1] = np.maximum(sol, Q_init)
            Q[j + 1, -1] = Q[j + 1, -2]

    return Q

# NUMERICAL COMPUTATION
n_nodes = int(np.round(L / dx)) + 1
x_coords = np.linspace(0.0, L, n_nodes)

total_min = 210.0
t_min = np.arange(0.0, total_min + dt_min, dt_min)
t_sec = t_min * 60.0
Q_upstream = Q_base + (Q_peak - Q_base) * np.exp(-((t_sec - peak_time_min * 60.0) ** 2) / (2.0 * ((sigma_t_min * 60.0) ** 2)))

Q_result = solve_routing(routing_model, tuple(Q_upstream), alpha, beta, dx, dt, n_nodes, Q_base, B, So)

# Precalculate depths
h_result = np.zeros_like(Q_result)
for j in range(Q_result.shape[0]):
    for i in range(Q_result.shape[1]):
        h_result[j, i] = area_from_Q(Q_result[j, i], B, So, manning_n) / B


# TIME PLAYBACK CONTROL ROW
ctrl_col, info_col1, info_col2 = st.columns([2.5, 1, 1])
with ctrl_col:
    time_idx = st.slider("Playback Time Frame", 0, len(t_min) - 1, min(25, len(t_min) - 1), format="Step %d", label_visibility="collapsed")
current_time = t_min[time_idx]
with info_col1:
    st.metric("Time Elapsed", f"{current_time:.1f} min")
with info_col2:
    st.metric("Peak Attenuation", f"{(Q_result[:, 0].max() - Q_result[:, -1].max()):.2f} m³/s")

# Displaying the hydrograph and depth profile
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), dpi=130)

# Discharge Hydrographs
for i in range(1, n_nodes - 1):
    ax1.plot(t_min, Q_result[:, i], color='gray', linewidth=0.45, alpha=0.22)

ax1.plot(t_min, Q_result[:, 0], color='blue', linewidth=1.5, label='Inflow (x = 0)')
ax1.plot(t_min, Q_result[:, -1], color='darkmagenta', linestyle='--', linewidth=1.5, label=f'Outflow (x = {L/1000:.1f} km)')

ax1.axvline(x=current_time, color='crimson', linestyle=':', linewidth=1.2)
ax1.scatter([current_time], [Q_result[time_idx, 0]], color='blue', s=22, zorder=5)
ax1.scatter([current_time], [Q_result[time_idx, -1]], color='darkmagenta', s=22, zorder=5)

ax1.set_ylabel('Discharge Q (m³/s)', fontsize=7.5)
ax1.set_ylim(0, max(Q_peak, Q_result.max()) * 1.15)
ax1.set_xlabel('Time (min)', fontsize=7.5)
ax1.tick_params(labelsize=7)
ax1.legend(loc='upper right', fontsize=6.5, framealpha=0.8)
ax1.grid(alpha=0.25)
ax1.set_title(f"Discharge Hydrograph [{routing_model}]", fontsize=8.0, pad=2)

# Water Depth Profile
h_now = h_result[time_idx, :]
ax2.plot(x_coords, h_now, color='dodgerblue', linewidth=1.5, label='Water Depth h(x)')
ax2.fill_between(x_coords, 0, h_now, color='dodgerblue', alpha=0.35)
h_base = area_from_Q(Q_base, B, So, manning_n) / B
ax2.axhline(y=h_base, color='black', linestyle=':', linewidth=0.8, label='Baseflow Depth')

ax2.set_xlabel('Reach Distance X (m)', fontsize=7.5)
ax2.set_ylabel('Depth h (m)', fontsize=7.5)
ax2.set_ylim(0, max(h_result.max() * 1.25, 0.5))
ax2.tick_params(labelsize=7)
ax2.legend(loc='upper right', fontsize=6.5, framealpha=0.8)
ax2.grid(alpha=0.25)
ax2.set_title(f"Water Depth Profile h(x) at t = {current_time:.0f} min", fontsize=8.0, pad=2)

fig.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=True)

# Metrics Strip
m1, m2, m3, m4 = st.columns(4)
m1.metric("Courant No. (Cr)", f"{courant_number:.2f}")
m2.metric("Manning α", f"{alpha:.3f}")
m3.metric("Peak Inflow", f"{Q_result[:, 0].max():.1f} m³/s")
m4.metric("Peak Outflow", f"{Q_result[:, -1].max():.1f} m³/s")