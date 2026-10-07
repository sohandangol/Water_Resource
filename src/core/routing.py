"""Dynamic 2D Kinematic Wave Numerical Solver."""
import numpy as np

M_OVER = 5.0 / 3.0


def step_series(t: np.ndarray, starts: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Piecewise-constant interpolation for rainfall hyetographs."""
    k = np.searchsorted(starts, t, side="right") - 1
    return np.where(k >= 0, np.asarray(vals)[np.clip(k, 0, None)], 0.0)


def run_raster(
    prep: dict,
    hyet_t: np.ndarray,
    hyet_i: np.ndarray,
    t_end: float,
    dt_out: float,
    n_over: float,
    n_ch: float,
    ch_width: float,
    ch_area_min: float = 0.0,
    cr: float = 0.8
) -> dict:
    """Explicit cell-by-cell 2D Kinematic Wave solver."""
    dx, ny_nx = prep["dx"], prep["shape"]
    idx = np.flatnonzero(prep["valid"])
    cmap = np.full(prep["valid"].size, -1)
    cmap[idx] = np.arange(len(idx))

    rec = prep["rec"][idx]
    rec = np.where(rec >= 0, cmap[np.clip(rec, 0, None)], -1)
    S, L = prep["S"][idx], prep["L"][idx]

    chan = (prep["acc"][idx] * dx * dx >= ch_area_min) if ch_area_min > 0 else np.zeros(len(idx), bool)
    W = np.where(chan, ch_width, dx)
    nn = np.where(chan, n_ch, n_over)

    alpha = np.sqrt(S) / (nn * W ** (2 / 3))
    m = M_OVER
    has = rec >= 0
    src, dst, outlet = np.flatnonzero(has), rec[has], ~has
    n = len(idx)

    t_out = np.arange(0, t_end + 1e-9, dt_out)
    marks = np.unique(np.concatenate([t_out, hyet_t[(hyet_t > 0) & (hyet_t < t_end)]]))

    A, hmax = np.zeros(n), np.zeros(n)
    Q_out = np.zeros(len(t_out))

    t, nxt = 0.0, 1

    while nxt < len(t_out):
        Q = alpha * A ** m
        wet = A > 1e-9
        dtc = cr * np.min(L[wet] / (alpha[wet] * m * A[wet] ** (m - 1))) if wet.any() else np.inf
        t_mark = marks[np.searchsorted(marks, t + 1e-9, side="right")]
        dt = min(dtc, dt_out, t_mark - t)

        i = float(step_series(np.array([t]), hyet_t, hyet_i)[0])
        inflow = np.bincount(dst, weights=Q[src], minlength=n)

        # Explicit finite difference cell update
        A = np.maximum(A + dt * (inflow - Q + i * dx * dx) / L, 0.0)
        t += dt
        np.maximum(hmax, A / W, out=hmax)

        if t >= t_out[nxt] - 1e-6:
            Qn = alpha * A ** m
            Q_out[nxt] = Qn[outlet].sum()
            nxt += 1

    full = np.full(prep["valid"].size, np.nan)
    full[idx] = hmax

    return dict(t=t_out, Q_out=Q_out, hmax=full.reshape(ny_nx))