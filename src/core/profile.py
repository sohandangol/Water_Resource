"""HEC-RAS Geometry Engine: Reach Profile & Cross-Section (XS) Slicing."""
import numpy as np


def extract_main_channel_profile(prep: dict, hmax_dict: dict) -> dict:
    """Trace main channel along D8 path and extract bed elevation + water surface profiles."""
    outlet_idx = np.argmax(prep["acc"])
    rec = prep["rec"]
    zf, L = prep["filled"].ravel(), prep["L"]

    path = [outlet_idx]
    current = outlet_idx
    while True:
        upstream = np.where(rec == current)[0]
        if len(upstream) == 0:
            break
        current = upstream[np.argmax(prep["acc"][upstream])]
        path.append(current)

    path = path[::-1]  # Upstream -> Downstream
    station_dist = np.cumsum([0.0] + [L[idx] for idx in path[:-1]])
    bed_elevation = zf[path]

    ws_profiles = {}
    for T, hmax_arr in hmax_dict.items():
        h_flat = hmax_arr.ravel()
        ws_profiles[T] = bed_elevation + np.nan_to_num(h_flat[path])

    return {
        "station": station_dist,
        "bed": bed_elevation,
        "ws_profiles": ws_profiles,
        "path_indices": path
    }


def extract_cross_section(
    prep: dict,
    hmax_dict: dict,
    center_idx: int,
    xs_length_m: float = 200.0,
    n_points: int = 50
) -> dict:
    """Extract a 2D perpendicular Cross-Section (XS cutline) across channel bank stations."""
    ny, nx = prep["shape"]
    dx = prep["dx"]
    i_c, j_c = divmod(center_idx, nx)

    # Determine flow vector angle from D8 downstream receiver cell
    rec_idx = prep["rec"][center_idx]
    if rec_idx >= 0:
        i_r, j_r = divmod(rec_idx, nx)
        flow_angle = np.arctan2(i_r - i_c, j_r - j_c)
    else:
        flow_angle = 0.0

    # Perpendicular cutline angle
    xs_angle = flow_angle + np.pi / 2.0

    # Sample points along cutline
    offsets = np.linspace(-xs_length_m / 2.0, xs_length_m / 2.0, n_points)
    station_x = offsets + xs_length_m / 2.0

    grid_i = np.clip(np.round(i_c + (offsets / dx) * np.sin(xs_angle)).astype(int), 0, ny - 1)
    grid_j = np.clip(np.round(j_c + (offsets / dx) * np.cos(xs_angle)).astype(int), 0, nx - 1)

    bed_xs = prep["filled"][grid_i, grid_j]

    ws_xs = {}
    for T, hmax_arr in hmax_dict.items():
        h_vals = hmax_arr[grid_i, grid_j]
        ws_xs[T] = bed_xs + np.nan_to_num(h_vals)

    return {"station": station_x, "bed": bed_xs, "ws_xs": ws_xs}