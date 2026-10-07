"""GIS Preprocessing, Direct Pixel Overlay Mapping, and Spatial Bounds Routing."""
import heapq
import io
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import scipy.ndimage
from affine import Affine

OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def read_dem(file_bytes: bytes) -> dict:
    """Reads a GeoTIFF byte stream into a numpy elevation matrix and raster profile."""
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        z = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            z[z == nodata] = np.nan

        transform = src.transform
        dx = abs(transform.a)
        dy = abs(transform.e)

        profile = src.profile.copy()
        crs = src.crs if src.crs else pyproj.CRS.from_epsg(4326)

    return {
        "z": z,
        "dx": dx,
        "dy": dy,
        "crs": crs,
        "profile": profile
    }


def generate_synthetic_dem(bbox_wgs84: list, resolution_m: float = 30.0) -> dict:
    """Generates a realistic synthetic DEM for WGS84 BBox [[south, west], [north, east]]."""
    south, west = bbox_wgs84[0]
    north, east = bbox_wgs84[1]

    lat_center = (south + north) / 2.0
    lat_meters = max(100.0, (north - south) * 111000.0)
    lon_meters = max(100.0, (east - west) * 111000.0 * np.cos(np.radians(lat_center)))

    ny = max(20, int(lat_meters / resolution_m))
    nx = max(20, int(lon_meters / resolution_m))

    y = np.linspace(0, 1, ny)
    x = np.linspace(0, 1, nx)
    xx, yy = np.meshgrid(x, y)

    # Elevation slope with synthetic channel
    z = 1200.0 - (yy * 300.0) - (xx * 150.0) + (np.sin(xx * np.pi * 3) * 20.0)
    channel_path = 0.5 + 0.2 * np.sin(yy * np.pi * 2)
    dist_to_channel = np.abs(xx - channel_path)
    z -= np.maximum(0.0, (0.08 - dist_to_channel) * 250.0)

    dx_deg = (east - west) / nx
    dy_deg = (north - south) / ny
    transform = Affine.translation(west, north) * Affine.scale(dx_deg, -dy_deg)

    profile = {
        "driver": "GTiff",
        "dtype": "float64",
        "nodata": None,
        "width": nx,
        "height": ny,
        "count": 1,
        "crs": pyproj.CRS.from_epsg(4326),
        "transform": transform
    }

    return {
        "z": z,
        "dx": resolution_m,
        "dy": resolution_m,
        "crs": pyproj.CRS.from_epsg(4326),
        "profile": profile,
        "bounds_wgs84": [[south, west], [north, east]]
    }


def generate_dem_overlay_png(z: np.ndarray) -> bytes:
    """Generates hillshaded DEM PNG overlay aligned properly for Folium ImageOverlay."""
    from PIL import Image

    z_norm = (z - np.nanmin(z)) / max(1e-5, (np.nanmax(z) - np.nanmin(z)))
    cmap = plt.colormaps["terrain"]
    rgba = (cmap(z_norm) * 255).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_raster_overlay_png(hmax: np.ndarray, vmin: float = 0.05, vmax: float = 3.0) -> bytes:
    """Converts 2D water depth matrix directly to transparent PNG byte stream."""
    from PIL import Image

    cmap = plt.colormaps["YlOrRd"]
    norm_h = np.clip((hmax - vmin) / max(1e-5, (vmax - vmin)), 0.0, 1.0)
    rgba = (cmap(norm_h) * 255).astype(np.uint8)

    # Set dry cells (< vmin) to zero alpha (fully transparent)
    rgba[hmax < vmin, 3] = 0

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_colorbar_image(
    cmap_name: str = "YlOrRd", 
    vmin: float = 0.0, 
    vmax: float = 3.0, 
    label: str = "Water Depth (m)"
) -> bytes:
    """Generates a vertical, clean colorbar legend byte stream for Streamlit UI columns."""
    fig, ax = plt.subplots(figsize=(1.4, 4.5))
    fig.subplots_adjust(left=0.25, right=0.55, top=0.92, bottom=0.08)

    cmap = plt.colormaps[cmap_name]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="vertical"
    )
    cb.set_label(label, fontsize=10, labelpad=10, fontweight="bold")
    cb.ax.tick_params(labelsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_dem_bounds_wgs84(profile: dict, crs) -> list:
    """Retrieves exact [[south, west], [north, east]] bounding coordinates."""
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]

    west = transform.c
    north = transform.f
    east = west + transform.a * width
    south = north + transform.e * height

    min_lat, max_lat = min(south, north), max(south, north)
    min_lon, max_lon = min(west, east), max(west, east)

    if crs != pyproj.CRS.from_epsg(4326):
        try:
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(min_lon, min_lat)
            max_lon, max_lat = transformer.transform(max_lon, max_lat)
        except Exception:
            pass

    return [[min_lat, min_lon], [max_lat, max_lon]]


def crop_dem_by_bbox_wgs84(dem_dict: dict, bbox_wgs84: list) -> dict:
    """Crops DEM matrix using WGS84 bounding coordinates [[south, west], [north, east]]."""
    from rasterio.windows import from_bounds

    south, west = bbox_wgs84[0]
    north, east = bbox_wgs84[1]

    transformer = pyproj.Transformer.from_crs("EPSG:4326", dem_dict["crs"], always_xy=True)
    minx, miny = transformer.transform(west, south)
    maxx, maxy = transformer.transform(east, north)

    profile = dem_dict["profile"]
    transform = profile["transform"]

    window = from_bounds(minx, miny, maxx, maxy, transform=transform)
    col_off, row_off = int(window.col_off), int(window.row_off)
    width, height = int(window.width), int(window.height)

    r_start = max(0, row_off)
    r_end = min(profile["height"], row_off + height)
    c_start = max(0, col_off)
    c_end = min(profile["width"], col_off + width)

    cropped_z = dem_dict["z"][r_start:r_end, c_start:c_end]
    new_transform = transform * Affine.translation(c_start, r_start)

    new_profile = dict(profile)
    new_profile.update({
        "height": cropped_z.shape[0],
        "width": cropped_z.shape[1],
        "transform": new_transform
    })

    return {
        "z": cropped_z,
        "dx": dem_dict["dx"],
        "dy": dem_dict["dy"],
        "crs": dem_dict["crs"],
        "profile": new_profile
    }


def resample_dem_grid(dem_dict: dict, scale_factor: int) -> dict:
    """Resamples DEM raster to accelerate solver computation."""
    if scale_factor == 1:
        return dem_dict

    z = dem_dict["z"]
    orig_h, orig_w = z.shape
    new_h, new_w = int(orig_h / scale_factor), int(orig_w / scale_factor)
    new_dx = dem_dict["dx"] * scale_factor

    resampled_z = scipy.ndimage.zoom(z, (new_h / orig_h, new_w / orig_w), order=1)

    new_profile = dict(dem_dict["profile"])
    new_transform = dem_dict["profile"]["transform"] * Affine.scale(scale_factor, scale_factor)
    new_profile.update({
        "height": new_h,
        "width": new_w,
        "transform": new_transform
    })

    return {
        "z": resampled_z,
        "dx": new_dx,
        "dy": new_dx,
        "profile": new_profile,
        "crs": dem_dict["crs"]
    }


def fill_depressions(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-flood sink removal."""
    ny, nx = z.shape
    valid = ~np.isnan(z)
    pad = np.pad(valid, 1, constant_values=False)
    edge = np.zeros(z.shape, dtype=bool)
    for di, dj in OFFS:
        edge |= ~pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]

    filled = z.copy()
    closed = np.zeros(z.shape, dtype=bool)
    heap = []

    for i, j in np.argwhere(valid & edge):
        closed[i, j] = True
        heap.append((z[i, j], i, j))
    heapq.heapify(heap)

    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in OFFS:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and valid[a, b] and not closed[a, b]:
                closed[a, b] = True
                filled[a, b] = max(z[a, b], h + eps)
                heapq.heappush(heap, (filled[a, b], a, b))

    return filled


def prepare(z: np.ndarray, dx: float, min_slope: float = 1e-3, eps: float = 1e-4) -> dict:
    """D8 Flow direction and accumulation setup."""
    ny, nx = z.shape
    valid = ~np.isnan(z)
    zf = fill_depressions(z, eps)
    f = np.where(valid, zf, np.inf)
    pad = np.pad(f, 1, constant_values=np.inf)
    ii, jj = np.indices(z.shape)
    best = np.zeros(z.shape)
    rec = np.full(z.shape, -1)
    length = np.full(z.shape, float(dx))

    for di, dj in OFFS:
        d = dx * np.hypot(di, dj)
        nb = pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        with np.errstate(invalid="ignore"):
            s = (f - nb) / d
        better = valid & (s > best)
        best = np.where(better, s, best)
        rec = np.where(better, (ii + di) * nx + (jj + dj), rec)
        length = np.where(better, d, length)

    rec, valid_f = rec.ravel(), valid.ravel()
    order = np.argsort(-np.where(valid_f, zf.ravel(), -np.inf))
    acc = valid_f.astype(float).tolist()
    rl = rec.tolist()

    for k in order.tolist():
        if not valid_f[k]:
            break
        if rl[k] >= 0:
            acc[rl[k]] += acc[k]

    return dict(
        shape=z.shape,
        dx=float(dx),
        valid=valid_f,
        rec=rec,
        S=np.maximum(best.ravel(), min_slope),
        L=length.ravel(),
        acc=np.array(acc),
        filled=zf,
        z_orig=z
    )


def extract_inundation_geojson(hmax: np.ndarray, transform, crs) -> dict:
    """Extracts inundation vector contours with aligned affine transforms."""
    import rasterio.features
    from shapely.geometry import shape, mapping
    from shapely.ops import transform as reproject_geom

    mask = (hmax > 0.05).astype("uint8")
    shapes = rasterio.features.shapes(mask, mask=mask, transform=transform)

    try:
        project = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        features = []

        for geom, val in shapes:
            if val == 1:
                s = shape(geom)
                s_wgs84 = reproject_geom(project, s)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(s_wgs84),
                    "properties": {"status": "inundated"}
                })

        return {"type": "FeatureCollection", "features": features}
    except Exception:
        return {"type": "FeatureCollection", "features": []}