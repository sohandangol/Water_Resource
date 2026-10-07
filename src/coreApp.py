"""Streamlit Hydrodynamics Suite with Aligned GIS Overlays and Vertical Colorbars."""
import base64
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import folium
from folium.plugins import Draw
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from core.gis import (
    crop_dem_by_bbox_wgs84,
    extract_inundation_geojson,
    generate_colorbar_image,
    generate_dem_overlay_png,
    generate_raster_overlay_png,
    generate_synthetic_dem,
    get_dem_bounds_wgs84,
    prepare,
    read_dem,
    resample_dem_grid,
)
from core.hydrology import generate_scs_hyetograph
from core.profile import extract_cross_section, extract_main_channel_profile
from core.routing import run_raster

st.set_page_config(page_title="HEC Hydrodynamics Modeling Suite", layout="wide")
st.title("HEC-HMS & RAS Integrated Hydrodynamics Modeling Suite")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Computational Mesh Control")
mesh_mode = st.sidebar.selectbox(
    "Mesh Resolution Scale",
    options=["Native DEM (1x)", "Downsampled Fast Mesh (2x)", "Coarse Fast Mesh (3x)"],
    index=0
)
SCALE_MAP = {"Native DEM (1x)": 1, "Downsampled Fast Mesh (2x)": 2, "Coarse Fast Mesh (3x)": 3}
scale_factor = SCALE_MAP[mesh_mode]

st.sidebar.header("2. Return Periods (HEC-HMS)")
return_periods = st.sidebar.multiselect(
    "Select Design Return Periods (Years)",
    options=[5, 20, 50, 100, 500],
    default=[5, 20, 100]
)

st.sidebar.header("3. Roughness & Channel Specs")
n_over = st.sidebar.number_input("Overland Manning's n", value=0.050, step=0.005, format="%.3f")
n_ch = st.sidebar.number_input("Channel Manning's n", value=0.035, step=0.005, format="%.3f")
ch_width = st.sidebar.number_input("Channel Width (m)", value=10.0, step=1.0)
ch_area_min = st.sidebar.number_input("Min Channel Threshold Area (m²)", value=50000.0, step=10000.0)

st.sidebar.header("4. Cross-Section (XS) Cutline Control")
xs_length = st.sidebar.number_input("XS Cutline Width (m)", value=250.0, step=25.0)


# --- DEM INPUT METHOD SELECTION ---
st.subheader("DEM Domain Setup")
input_method = st.radio(
    "Select DEM Input Method:",
    options=["Select Bounding Box on Map (Interactive)", "Upload Local GeoTIFF File"],
    horizontal=True
)

raw_dem = None
active_bbox = None

if input_method == "Select Bounding Box on Map (Interactive)":
    st.markdown("Use the toolbar (top-left of map) to draw a rectangle over your study area.")

    m_bbox = folium.Map(
        location=[27.575, 85.485],
        zoom_start=12,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True
        }
    ).add_to(m_bbox)

    map_draw_data = st_folium(m_bbox, width=900, height=450, key="interactive_bbox_picker")

    if map_draw_data and map_draw_data.get("all_drawings"):
        last_draw = map_draw_data["all_drawings"][-1]
        if last_draw.get("geometry", {}).get("type") == "Polygon":
            coords = last_draw["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]

            s_lat, n_lat = min(lats), max(lats)
            w_lon, e_lon = min(lons), max(lons)
            active_bbox = [[s_lat, w_lon], [n_lat, e_lon]]

            st.info(f"**Selected Extent Bounds:**\n"
                    f"* **South Lat:** `{s_lat:.4f}` | **North Lat:** `{n_lat:.4f}`\n"
                    f"* **West Lon:** `{w_lon:.4f}` | **East Lon:** `{e_lon:.4f}`")

    if active_bbox is not None:
        if st.button("Generate Synthetic DEM for Selected Area", type="primary"):
            raw_dem = generate_synthetic_dem(active_bbox)
            st.session_state["raw_dem"] = raw_dem
            st.session_state["active_bbox"] = active_bbox
            st.success("DEM spatial domain initialized!")
    else:
        st.caption("Draw a rectangle on the map to specify the bounding box.")

    if "raw_dem" in st.session_state and input_method == "Select Bounding Box on Map (Interactive)":
        raw_dem = st.session_state["raw_dem"]

        # --- DEM PREVIEW MAP WITH COLORBAR ---
        st.markdown("**Generated DEM Surface Relief Preview**")
        dem_bounds = get_dem_bounds_wgs84(raw_dem["profile"], raw_dem["crs"])
        lat_c = (dem_bounds[0][0] + dem_bounds[1][0]) / 2.0
        lon_c = (dem_bounds[0][1] + dem_bounds[1][1]) / 2.0

        m_dem_prev = folium.Map(
            location=[lat_c, lon_c],
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )

        dem_png = generate_dem_overlay_png(raw_dem["z"])
        enc_dem = base64.b64encode(dem_png).decode("utf-8")
        ImageOverlay(
            image=f"data:image/png;base64,{enc_dem}",
            bounds=dem_bounds,
            opacity=0.8,
            name="Elevation DEM Surface"
        ).add_to(m_dem_prev)

        m_dem_prev.fit_bounds(dem_bounds)
        
        c1, c2 = st.columns([4, 1])
        with c1:
            st_folium(m_dem_prev, width=720, height=400, key="dem_surface_preview_map")
        with c2:
            st.markdown("**Elevation Scale (m)**")
            z_min, z_max = np.nanmin(raw_dem["z"]), np.nanmax(raw_dem["z"])
            cbar_bytes = generate_colorbar_image(cmap_name="terrain", vmin=z_min, vmax=z_max, label="Elevation (m)")
            st.image(cbar_bytes, use_container_width=True)

else:
    uploaded_dem = st.file_uploader("Upload Catchment DEM GeoTIFF", type=["tif", "tiff"])
    if uploaded_dem:
        raw_dem = read_dem(uploaded_dem.getvalue())
        st.session_state["raw_dem"] = raw_dem

        full_bounds_wgs84 = get_dem_bounds_wgs84(raw_dem["profile"], raw_dem["crs"])
        st.markdown("**Uploaded GeoTIFF Coverage Map**")

        lat_center = (full_bounds_wgs84[0][0] + full_bounds_wgs84[1][0]) / 2.0
        lon_center = (full_bounds_wgs84[0][1] + full_bounds_wgs84[1][1]) / 2.0

        m_crop = folium.Map(
            location=[lat_center, lon_center],
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
        m_crop.fit_bounds(full_bounds_wgs84)

        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "rectangle": True
            }
        ).add_to(m_crop)

        crop_map_data = st_folium(m_crop, width=900, height=450, key="uploaded_dem_crop_picker")

        if crop_map_data and crop_map_data.get("all_drawings"):
            last_draw = crop_map_data["all_drawings"][-1]
            if last_draw.get("geometry", {}).get("type") == "Polygon":
                coords = last_draw["geometry"]["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                active_bbox = [[min(lats), min(lons)], [max(lats), max(lons)]]


# --- COMPUTATIONAL EXECUTION ---
if raw_dem is not None:
    if not return_periods:
        st.warning("Please select at least one return period in the sidebar before running simulation.")
    else:
        active_dem = crop_dem_by_bbox_wgs84(raw_dem, active_bbox) if active_bbox and input_method != "Select Bounding Box on Map (Interactive)" else raw_dem
        dem = resample_dem_grid(active_dem, scale_factor)

        @st.cache_data(show_spinner="Processing DEM priority-flood filling & D8 flow directions...")
        def process_gis_cache(z, dx):
            return prepare(z, dx)

        prep = process_gis_cache(dem["z"], dem["dx"])

        st.info(f"Active Mesh Resolution: **{dem['dx']:.1f} m** | Grid Dimensions: **{dem['z'].shape[0]} × {dem['z'].shape[1]} cells**")

        if st.button("Run Hydrodynamic Simulation Suite", type="primary"):
            hmax_dict = {}
            q_out_dict = {}
            geojson_dict = {}

            pbar = st.progress(0.0)
            status = st.empty()

            for idx, T in enumerate(return_periods):
                status.text(f"Computing SCS Storm & Hydraulic Routing for T = {T} Years...")

                hyet_t, hyet_i = generate_scs_hyetograph(return_period=T, duration_hr=6.0)
                res = run_raster(
                    prep=prep,
                    hyet_t=hyet_t,
                    hyet_i=hyet_i,
                    t_end=6.0 * 3600.0,
                    dt_out=300.0,
                    n_over=n_over,
                    n_ch=n_ch,
                    ch_width=ch_width,
                    ch_area_min=ch_area_min
                )

                hmax_dict[T] = res["hmax"]
                q_out_dict[T] = (res["t"], res["Q_out"])
                geojson_dict[T] = extract_inundation_geojson(res["hmax"], dem["profile"]["transform"], dem["crs"])

                pbar.progress((idx + 1) / len(return_periods))

            status.text("Simulation Suite Completed!")
            pbar.empty()

            st.session_state["prep"] = prep
            st.session_state["hmax_dict"] = hmax_dict
            st.session_state["q_out_dict"] = q_out_dict
            st.session_state["geojson_dict"] = geojson_dict
            st.session_state["dem"] = dem


# --- RESULTS PRESENTATION ---
if "hmax_dict" in st.session_state and st.session_state["hmax_dict"]:
    prep = st.session_state["prep"]
    hmax_dict = st.session_state["hmax_dict"]
    geojson_dict = st.session_state["geojson_dict"]
    dem = st.session_state["dem"]

    st.divider()
    st.subheader("1. HEC-RAS Style Reach Profile Plot")
    prof_data = extract_main_channel_profile(prep, hmax_dict)

    fig_prof, ax_p = plt.subplots(figsize=(12, 4.5))
    ax_p.plot(prof_data["station"], prof_data["bed"], color="black", lw=1.8, label="Channel Bed Elevation")
    ax_p.fill_between(prof_data["station"], prof_data["bed"], color="lightgray", alpha=0.5)

    colors = ["blue", "navy", "purple", "darkred", "red"]
    for idx, (T, ws) in enumerate(prof_data["ws_profiles"].items()):
        c = colors[idx % len(colors)]
        ax_p.plot(prof_data["station"], ws, label=f"WS {T} Year", color=c, lw=1.8, linestyle="--")

    ax_p.set_xlabel("Main Channel Distance (m)")
    ax_p.set_ylabel("Elevation (m)")
    ax_p.legend(loc="upper right")
    ax_p.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_prof)

    st.subheader("2. Cross-Section (XS) Cutline Profile")
    path_len = len(prof_data["path_indices"])
    xs_station_idx = st.slider(
        "Select Channel Reach Station Node (Upstream -> Downstream)",
        min_value=0,
        max_value=path_len - 1,
        value=path_len // 2
    )

    selected_node_idx = prof_data["path_indices"][xs_station_idx]
    xs_data = extract_cross_section(prep, hmax_dict, center_idx=selected_node_idx, xs_length_m=xs_length)

    fig_xs, ax_xs = plt.subplots(figsize=(10, 4))
    ax_xs.plot(xs_data["station"], xs_data["bed"], color="brown", lw=2, label="Ground Cross-Section")
    ax_xs.fill_between(xs_data["station"], xs_data["bed"], color="navajowhite", alpha=0.5)

    for idx, (T, ws) in enumerate(xs_data["ws_xs"].items()):
        c = colors[idx % len(colors)]
        ax_xs.plot(xs_data["station"], ws, label=f"WS {T} Year", color=c, lw=1.8, linestyle="--")

    ax_xs.set_xlabel("Cross-Section Cutline Distance (m)")
    ax_xs.set_ylabel("Elevation (m)")
    ax_xs.set_title(f"Channel Cross-Section at Station Distance: {prof_data['station'][xs_station_idx]:.1f} m")
    ax_xs.legend(loc="upper right")
    ax_xs.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_xs)

    st.subheader("3. Spatial Inundation Map Overlay")
    sel_T = st.selectbox("Select Return Period Overlay", options=list(hmax_dict.keys()))

    bounds_wgs84 = get_dem_bounds_wgs84(dem["profile"], dem["crs"])
    lat_center = (bounds_wgs84[0][0] + bounds_wgs84[1][0]) / 2.0
    lon_center = (bounds_wgs84[0][1] + bounds_wgs84[1][1]) / 2.0

    m_res = folium.Map(
        location=[lat_center, lon_center],
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        zoom_start=13
    )

    max_val = max(0.2, float(np.nanmax(hmax_dict[sel_T])))

    # Low threshold (vmin=0.05 m) ensures the complete spatial flood footprint is preserved
    png_bytes = generate_raster_overlay_png(hmax_dict[sel_T], vmin=0.05, vmax=max_val)
    encoded_png = base64.b64encode(png_bytes).decode("utf-8")
    png_url = f"data:image/png;base64,{encoded_png}"

    ImageOverlay(
        image=png_url,
        bounds=bounds_wgs84,
        opacity=0.75,
        name=f"Inundation Heatmap ({sel_T}-Yr)"
    ).add_to(m_res)

    m_res.fit_bounds(bounds_wgs84)
    folium.LayerControl().add_to(m_res)

    res_c1, res_c2 = st.columns([5, 1])

    with res_c1:
        st_folium(m_res, width=850, height=580, key=f"folium_map_results_{sel_T}")

    with res_c2:
        st.markdown("### Legend")
        cbar_res = generate_colorbar_image(
            cmap_name="YlOrRd", 
            vmin=0.0, 
            vmax=max_val, 
            label="Water Depth (m)"
        )
        st.image(cbar_res, use_container_width=True)