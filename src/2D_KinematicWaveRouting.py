import heapq
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def print_progress(current: int, total: int, start_time: float, prefix: str = "", extra: str = "", bar_len: int = 30):
       
    fraction = current / total
    filled_len = int(bar_len * fraction)
    bar = "█" * filled_len + "░" * (bar_len - filled_len)
    percent = fraction * 100.0

    elapsed = time.time() - start_time
    if current > 0:
        eta_sec = (elapsed / current) * (total - current)
        time_str = f"Elapsed: {elapsed:.0f}s | ETA: {eta_sec:.0f}s"
    else:
        time_str = f"Elapsed: {elapsed:.0f}s | ETA: --"

    msg = f"\r{prefix} [{bar}] {percent:5.1f}% | {current}/{total} | {time_str} | {extra}"
    sys.stdout.write(msg)
    sys.stdout.flush()

    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

def load_dem_raster(dem_filepath: str = None, max_dimension: int = 400):
  
    if dem_filepath and HAS_RASTERIO:
        print(f"Loading real DEM from: {dem_filepath}")
        with rasterio.open(dem_filepath) as src:
            orig_h, orig_w = src.height, src.width
            print(f"Original DEM Size: {orig_w} x {orig_h} pixels")

            max_dim = max(orig_h, orig_w)
            if max_dim > max_dimension:
                scale = max_dimension / max_dim
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                print(f"--> Downsampling DEM for fast 2D routing to: {new_w} x {new_h} pixels...")
                
                z = src.read(
                    1,
                    out_shape=(new_h, new_w),
                    resampling=rasterio.enums.Resampling.bilinear
                ).astype(np.float64)
                
                dx_orig = abs(src.transform.a)
                dx = dx_orig / scale
            else:
                z = src.read(1).astype(np.float64)
                dx = abs(src.transform.a)

            if src.nodata is not None:
                z[z == src.nodata] = np.nan

            if src.crs and src.crs.is_geographic:
                dx_meters = dx * 111320.0 * np.cos(np.radians(27.7))
                print(f"Converted geographic resolution {dx:.6f}° -> {dx_meters:.2f} meters")
                dx = dx_meters
            else:
                print(f"Projected CRS detected. Cell resolution: {dx:.2f} meters")

        return z, dx
    else:
        print("DEM not-found in the provided local file path! Please ensure a valid GeoTIFF DEM esists in the file path.")
        return None, None

def fill_depressions(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    
    sys.stdout.write("  [1/2] Priority-flood depression filling...")
    sys.stdout.flush()
    t0 = time.time()

    # ny = number of rows (think as latitude)
    # nx = number of columns (think as longitude)
    # Hence, z.shape gets dimension of the array
    ny, nx = z.shape    
    
    # To compute how (row, column) ie center (0,0) relates to its 8 neighbors in a 2D grid
    offs = [(-1,-1), (-1,0), (-1,1),
            (0,-1),         (0,1), 
            (1,-1), (1,0), (1,1)]
    
    valid = ~np.isnan(z)
    pad = np.pad(valid, 1, constant_values=False)
    edge = np.zeros(z.shape, dtype=bool)
    
    # Instead of complex scipy packages, I resolved to this code since it also checks for the NaN vlaues inside the DEM raster and assigns it as sinks, alongside determining the edges of rasters.  
    
    # The directions assumed are
    # di = row (vertical y-axis/longitudes)
    # dj = column (horizontal x-axis/latitudes)
    # in numpy the first indes is row ie (y, x)
    for di, dj in offs:
        edge |= ~pad[1 + di : 1 + di + ny,
                     1 + dj : 1 + dj + nx]

    filled = z.copy() # Clone the DEM raster instead of modifying the original DEM raster, to avoid any data loss or corruption.
    
    closed = np.zeros(z.shape, dtype=bool)
    heap = []

    for i, j in np.argwhere(valid & edge):
        closed[i, j] = True
        heap.append((z[i, j], i, j))    # tupels of elevations of each raster cells
    heapq.heapify(heap) 
    # The heap picks the lowest elevation cell first ie making sure the water flows towrards sinks (depressions).

    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in offs:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and valid[a, b] and not closed[a, b]:
                closed[a, b] = True
                filled[a, b] = max(z[a, b], h + eps)
                heapq.heappush(heap, (filled[a, b], a, b))

    sys.stdout.write(f" done ({time.time() - t0:.2f}s)\n")
    sys.stdout.flush()
    return filled


def preprocess_d8_topology(z: np.ndarray, dx: float, min_slope: float = 1e-3) -> dict:
    """Computes flow direction of water to surrounding cells, continuous slope, and flow accumulation from DEM grid."""
    
    
    print("\n[Topology Preprocessing]")
    
    # Filling the sinks
    zf = fill_depressions(z)    # It is required to fill the sinks' value to closest minimum cell's elevation to obtain continuous downhill flow of water stream and prevent abrupt water accumulation streams in the DEM raster. 
    
    ny, nx = z.shape
    n_cells = ny * nx

    sys.stdout.write("  [2/2] Calculating D8 slopes & accumulation...")
    sys.stdout.flush()
    t0 = time.time()

    offs = [(-1,-1), (-1,0), (-1,1), 
            (0,-1),         (0,1), 
            (1,-1), (1,0), (1,1)]
    
    f = np.pad(zf, 1, constant_values=np.inf)
    ii, jj = np.indices(z.shape)
    
    best_slope = np.zeros(z.shape)  # starting from zero slope ie flat
    
    receiver = np.full(z.shape, -1, dtype=int)  # If water has finally no where to flow, it is assigned as -1 (sink) in the receiver array, eg. an outlet of basin

    for di, dj in offs:
        dist = dx * np.hypot(di, dj)
        neighbor = f[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        slope = (zf - neighbor) / dist
        
        # Check condition if diagonal slope greater than linears (along rows or columns)
        better = slope > best_slope
        
        best_slope = np.where(better, slope, best_slope)
        receiver = np.where(better, (ii + di) * nx + (jj + dj), receiver)

    flat_order = np.argsort(-zf.ravel())
    acc = np.ones(n_cells, dtype=float)
    rec_flat = receiver.ravel()
    
    for idx in flat_order:
        target = rec_flat[idx]
        if target >= 0:
            acc[target] += acc[idx]

    sys.stdout.write(f" done ({time.time() - t0:.2f}s)\n\n")
    sys.stdout.flush()

    return {
        "filled_z": zf,
        "slope": np.maximum(best_slope.ravel(), min_slope),
        "receiver": rec_flat,
        "order": flat_order,
        "acc": acc.reshape((ny, nx)),
        "dx": dx,
        "ny": ny,
        "nx": nx
    }


def run_kinematic_wave_2d(
    topo: dict, # dictionary to hold the return value of preporcessed function
    rainfall_t_sec: np.ndarray,
    rainfall_rate_m_s: np.ndarray,
    sim_time_sec: float = 7200.0,
    dt_sec: float = 5.0,
    n_overland: float = 0.040, 
    n_channel: float = 0.025,
    channel_acc_thresh: int = 50
    
) -> dict:
    """Executes 2D Kinematic Wave routing on preprocessed raster topology."""
    ny, nx = topo["ny"], topo["nx"]
    n_cells = ny * nx
    dx = topo["dx"]
    cell_area = dx * dx
    rec = topo["receiver"]
    S0 = topo["slope"]
    order = topo["order"]
    acc_flat = topo["acc"].ravel()  # ravel flattens 2D array to 1D array

    is_channel = acc_flat >= channel_acc_thresh
    n_eff = np.where(is_channel, n_channel, n_overland)
    width_eff = np.where(is_channel, dx * 0.2, dx)
    conductance = (1.0 / n_eff) * width_eff * np.sqrt(S0)

    num_steps = int(sim_time_sec / dt_sec)
    t_series = np.linspace(0, sim_time_sec, num_steps)
    rain_interp = np.interp(t_series, rainfall_t_sec, rainfall_rate_m_s)

    h = np.zeros(n_cells, dtype=np.float64)
    h_max = np.zeros(n_cells, dtype=np.float64)
    outlet_idx = topo["order"][-1]
    q_outlet = []

    # Frequency divider: update terminal every N steps or 0.25 seconds to eliminate I/O overhead
    update_interval = max(1, num_steps // 200)
    sim_start_time = time.time()

    print("Running 2D Kinematic Wave Engine:")
    for step in range(num_steps):
        i_t = rain_interp[step]
        h += i_t * dt_sec
        q_out = conductance * np.maximum(0.0, h) ** (5.0 / 3.0)
        q_in = np.zeros(n_cells, dtype=np.float64)

        for idx in order:
            vol_out = q_out[idx] * dt_sec
            max_vol = h[idx] * cell_area
            if vol_out > max_vol:
                vol_out = max_vol
                q_out[idx] = vol_out / dt_sec

            target = rec[idx]
            if target >= 0:
                q_in[target] += q_out[idx]

        h += (q_in - q_out) * (dt_sec / cell_area)
        h = np.maximum(0.0, h)

        h_max = np.maximum(h_max, h)
        q_outlet.append(q_out[outlet_idx])

        # Status update check
        if (step + 1) % update_interval == 0 or (step + 1) == num_steps:
            stats = f"Peak h: {h_max.max():.3f}m | Q_out: {q_outlet[-1]:.2f}m³/s"
            print_progress(
                current=step + 1,
                total=num_steps,
                start_time=sim_start_time,
                prefix="Routing",
                extra=stats
            )

    return {
        "h_max": h_max.reshape((ny, nx)),
        "t": t_series,
        "Q_outlet": np.array(q_outlet)
    }

if __name__ == "__main__":
    # 1. Load DEM
    z_dem, grid_dx = load_dem_raster(dem_filepath=r"F:\GIS_Elective\Data_Teams\DEM_KTM1.tif")

    # 2. Preprocess DEM
    topo = preprocess_d8_topology(z_dem, grid_dx)

    # 3. Setup some random Rainfall data (50 mm/hr for 30 minutes straight and ends abruptly) to test the 2D Kinematic Wave routing engine.
    rain_t = np.array([0.0, 1800.0, 1801.0, 3600.0])
    rain_i = np.array([50.0, 50.0, 0.0, 0.0]) / (1000.0 * 3600.0)
    # But in future I plan to call the rainfall durations and rainfall intensities inside from the CSV file of actual rainfall data obtained from DHM. 
    # OR even better, I am thinking to fetch the live forecasted weather api data?- will explore in future final product.

    # 4. Run Kinematic Routing
    results = run_kinematic_wave_2d(
        topo=topo,
        rainfall_t_sec=rain_t,
        rainfall_rate_m_s=rain_i,
        sim_time_sec=7200.0,
        dt_sec=5.0
    )

    print("Rendering diagnostic plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    im1 = ax1.imshow(topo["filled_z"], cmap="terrain", origin="upper")
    ax1.set_title("Input Elevation DEM Z (m)")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")

    im2 = ax2.imshow(results["h_max"], cmap="Blues", origin="upper")
    ax2.set_title("Max Water Depth h (m)")
    plt.colorbar(im2, ax=ax2, label="Depth (m)")

    ax3.plot(results["t"] / 3600.0, results["Q_outlet"], color="navy", lw=2)
    ax3.set_title("Outlet Hydrograph Q (m³/s)")
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Discharge Q (m³/s)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()