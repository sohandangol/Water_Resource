"""HEC-HMS Hydrology Engine: IDF Curves & SCS Type II Synthetic Hyetographs."""
import numpy as np


def compute_idf_intensity(return_period: int, duration_min: float, c_factors: dict = None) -> float:
    """Compute rainfall intensity (mm/hr) using standard Sherman IDF equation."""
    if c_factors is None:
        c_factors = {"c1": 45.0, "c2": 0.22, "c3": 12.0, "c4": 0.75}

    c1, c2, c3, c4 = c_factors["c1"], c_factors["c2"], c_factors["c3"], c_factors["c4"]
    intensity = (c1 * (return_period ** c2)) / ((duration_min + c3) ** c4)
    return float(intensity)


def generate_scs_hyetograph(return_period: int, duration_hr: float = 6.0, dt_min: float = 5.0) -> tuple:
    """Generate alternating block hyetograph (SCS Type II storm distribution)."""
    t_min = np.arange(dt_min, duration_hr * 60.0 + dt_min, dt_min)
    intensities = [compute_idf_intensity(return_period, t) for t in t_min]
    cum_depths = intensities * (t_min / 60.0)

    inc_depths = np.diff(np.insert(cum_depths, 0, 0.0))
    sorted_depths = np.sort(inc_depths)[::-1]
    hyetograph = np.zeros_like(sorted_depths)

    mid = len(sorted_depths) // 2
    hyetograph[mid] = sorted_depths[0]

    for k in range(1, len(sorted_depths)):
        if k % 2 == 1:
            hyetograph[mid - (k + 1) // 2] = sorted_depths[k]
        else:
            hyetograph[mid + k // 2] = sorted_depths[k]

    t_sec = np.arange(0, duration_hr * 3600.0, dt_min * 60.0)
    i_m_per_s = (hyetograph / (dt_min * 60.0)) / 1000.0

    return t_sec, i_m_per_s