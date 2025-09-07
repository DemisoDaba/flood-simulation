# delineate_watershed_local.py
"""
Streamlit app: Delineate watershed (upstream area) from an outlet point using a local DEM (GeoTIFF).
No GEE required.

How it works:
- Upload a DEM GeoTIFF (must have CRS and geotransform).
- Click on the folium map or enter lat,lon for the outlet.
- DEM is filled, D8 flow directions and accumulation are computed with richdem.
- The app finds all cells that drain to the outlet (upstream mask).
- Converts the mask to a polygon (GeoJSON) and displays it.
- Download option for the basin GeoJSON.

Dependencies (add to requirements.txt):
streamlit
rasterio
richdem
numpy
geopandas
shapely
folium
streamlit-folium
pyproj
matplotlib
"""

import streamlit as st
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform
import numpy as np
import richdem as rd
import geopandas as gpd
from shapely.geometry import shape, mapping, Point
import folium
from streamlit_folium import st_folium
import json
import tempfile
import os
from pyproj import Transformer

st.set_page_config(page_title="Local Watershed Delineation", layout="wide")
st.title("Watershed Delineation from Outlet (Local DEM) — No GEE")

st.sidebar.header("1) Upload DEM (GeoTIFF)")
dem_file = st.sidebar.file_uploader("Upload DEM GeoTIFF", type=["tif", "tiff"])

st.sidebar.markdown("""
**Notes**
- DEM must have a CRS (e.g., EPSG:4326 or a projected CRS).
- For speed, clip your DEM to the general study area before upload.
""")

if dem_file is None:
    st.info("Please upload a DEM GeoTIFF to proceed.")
    st.stop()

# Save uploaded DEM to temporary file so rasterio can open it
temp_dem_path = os.path.join(tempfile.gettempdir(), "uploaded_dem.tif")
with open(temp_dem_path, "wb") as f:
    f.write(dem_file.getbuffer())

# Read DEM metadata
with rasterio.open(temp_dem_path) as src:
    dem_crs = src.crs
    dem_transform = src.transform
    dem_bounds = src.bounds
    dem_arr = src.read(1).astype('float64')
    dem_nodata = src.nodatavals[0]
    dem_profile = src.profile

# Replace nodata values with np.nan
if dem_nodata is not None:
    dem_arr[dem_arr == dem_nodata] = np.nan

st.sidebar.write(f"DEM CRS: {dem_crs}")
st.sidebar.write(f"DEM bounds: {dem_bounds}")

# Provide a small preview map and let user click to pick outlet (in lat/lon)
st.write("## 2) Choose outlet point")
st.write("You can click on the map to place an outlet, or enter lat,lon manually in the sidebar.")

# Create a Folium map centered at DEM centroid
cent_y = (dem_bounds.top + dem_bounds.bottom) / 2.0
cent_x = (dem_bounds.left + dem_bounds.right) / 2.0
m = folium.Map(location=[cent_y, cent_x], zoom_start=9)
# Add DEM bounds rectangle
folium.Rectangle(
    bounds=[[dem_bounds.bottom, dem_bounds.left], [dem_bounds.top, dem_bounds.right]],
    color="blue", weight=1, fill=False, popup="DEM bounds"
).add_to(m)

st_map = st_folium(m, height=450, returned_objects=["last_clicked"])

# Allow manual input
manual_coords = st.sidebar.text_input("Manual outlet lat,lon (e.g. 9.0,38.7)")
clicked = None
if st_map and st_map.get("last_clicked"):
    clicked = st_map["last_clicked"]
    st.sidebar.success(f"Clicked: {clicked['lat']:.6f}, {clicked['lng']:.6f}")

outlet_lat = None
outlet_lon = None
if clicked:
    outlet_lat = clicked["lat"]
    outlet_lon = clicked["lng"]
elif manual_coords:
    try:
        lat, lon = [float(x.strip()) for x in manual_coords.split(",")]
        outlet_lat, outlet_lon = lat, lon
        st.sidebar.success(f"Using manual coords: {lat:.6f}, {lon:.6f}")
    except Exception as e:
        st.sidebar.error("Could not parse manual coords. Use format: lat,lon")
        st.stop()
else:
    st.info("Click map or input coordinates and press 'Delineate watershed' to continue.")
    st.stop()

# Convert outlet lat/lon (EPSG:4326) -> DEM CRS coordinates
if dem_crs.to_string() != "EPSG:4326":
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    outlet_x_dem, outlet_y_dem = transformer.transform(outlet_lon, outlet_lat)
else:
    outlet_x_dem, outlet_y_dem = outlet_lon, outlet_lat

# Convert DEM coordinates to row, col
inv_transform = ~dem_transform
col_f, row_f = inv_transform * (outlet_x_dem, outlet_y_dem)
row = int(np.floor(row_f))
col = int(np.floor(col_f))

# Check bounds
nrows, ncols = dem_arr.shape
if not (0 <= row < nrows and 0 <= col < ncols):
    st.error("Outlet is outside DEM bounds. Choose a point inside the DEM.")
    st.stop()

st.write(f"Outlet mapped to DEM pixel row={row}, col={col}")

# ---------------------------
# Hydrological processing with richdem
# ---------------------------
st.write("## 3) Running hydrological preprocessing (filling, D8 and flow accumulation). This may take a few seconds to minutes depending on DEM size.")

# Copy array (richdem expects its own RDArray)
dem_work = dem_arr.copy()
# Replace nan with a high value? richdem has fill depressions function that handles np.nan poorly.
# We'll temporarily fill nan with a large positive value and mask later.
nan_mask = np.isnan(dem_work)
if np.any(nan_mask):
    # Fill NaNs with local mean or high value - better to set to max elevation + 1
    max_elev = np.nanmax(dem_work)
    dem_work[nan_mask] = max_elev + 1000

rd_dem = rd.rdarray(dem_work, no_data=np.nan)
rd_dem.geotransform = (dem_transform.a, dem_transform.b, dem_transform.c, dem_transform.d, dem_transform.e, dem_transform.f) if hasattr(dem_transform, 'a') else (dem_transform[2], dem_transform[0], dem_transform[1], dem_transform[5], dem_transform[3], dem_transform[4])

# Fill depressions
try:
    rd_filled = rd.FillDepressions(rd_dem, in_place=False)
except Exception as e:
    st.warning(f"FillDepressions failed: {e}. Attempting alternate fill using FillDepressions(rd_dem, in_place=True).")
    rd_filled = rd.FillDepressions(rd_dem, in_place=True)

# Flow direction (D8)
try:
    flow_dir = rd.FlowDirectionD8(rd_filled)
except Exception as e:
    st.error(f"FlowDirectionD8 failed: {e}")
    st.stop()

# Flow accumulation
try:
    acc = rd.FlowAccumulation(flow_dir, method='D8')
except Exception as e:
    st.error(f"FlowAccumulation failed: {e}")
    st.stop()

acc_arr = np.array(acc)

st.write("Flow accumulation computed. Now computing upstream mask for the chosen outlet...")

# ---------------------------
# Upstream mask: mark all cells that drain to outlet
# ---------------------------
# D8 coding in richdem: values 1..8 indicating direction to neighbor
# We'll implement a fast upstream search by creating a "downstream index" mapping and then tracing each cell
# But simpler approach: starting from outlet cell, walk upstream by finding neighbors that point to current cell, iteratively expand.

# Create padded indices for ease
rows, cols = acc_arr.shape

# Flow direction array (int)
fdir = np.array(flow_dir, dtype=np.int32)

# Function to get neighbor indices that flow into a given cell (i,j)
# D8 encoding used by richdem: 1=E, 2=SE, 3=S, 4=SW, 5=W, 6=NW, 7=N, 8=NE
# When a cell has flow_dir value d, it flows from that cell to the neighbor indicated by d.
# So neighbors that flow *into* cell (i,j) are those whose flow_dir points to (i,j).
dir_to_delta = {
    1: (0, 1),   # E
    2: (1, 1),   # SE
    3: (1, 0),   # S
    4: (1, -1),  # SW
    5: (0, -1),  # W
    6: (-1, -1), # NW
    7: (-1, 0),  # N
    8: (-1, 1)   # NE
}
# Reverse mapping: for a neighbor at (ni,nj) to flow to (i,j), its flow_dir must equal the direction from neighbor to (i,j)
# But we can test neighbors explicitly.

from collections import deque

visited = np.zeros_like(fdir, dtype=np.uint8)
q = deque()
# seed with outlet cell if outlet cell is valid
visited[row, col] = 1
q.append((row, col))

# offsets to iterate 8 neighbors
neigh_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

while q:
    i,j = q.popleft()
    # check all neighbors: if neighbor flows to (i,j), mark neighbor visited and add to queue
    for di, dj in neigh_offsets:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            if visited[ni, nj]:
                continue
            # neighbor flow direction value
            val = int(fdir[ni, nj])
            if val == 0:
                continue
            # compute where neighbor flows to
            d = dir_to_delta.get(val, None)
            if d is None:
                continue
            dest_i = ni + d[0]
            dest_j = nj + d[1]
            if dest_i == i and dest_j == j:
                visited[ni, nj] = 1
                q.append((ni, nj))

upstream_mask = visited.astype(bool)

# Optionally mask out places where original DEM had nan
if np.any(nan_mask):
    upstream_mask[nan_mask] = False

st.write(f"Computed upstream mask — {upstream_mask.sum()} cells belong to upstream catchment.")

# ---------------------------
# Convert mask to vector polygon
# ---------------------------

# Create shapes (polygons) from mask; shapes yields (geom, value)
transform = dem_transform
mask_uint8 = upstream_mask.astype(np.uint8)

shapes_gen = shapes(mask_uint8, mask=mask_uint8==1, transform=transform)

polys = []
for geom, val in shapes_gen:
    if val == 1:
        polys.append(shape(geom))

if not polys:
    st.error("No polygon generated from mask. Perhaps the outlet is on boundary or DEM extents too small.")
    st.stop()

# Merge polygons into single geometry
from shapely.ops import unary_union
basin_geom = unary_union(polys)

gdf = gpd.GeoDataFrame({"id":[1]}, geometry=[basin_geom], crs=dem_crs.to_string())

st.write("## 4) Result: Delineated Basin / Study Area")
st.write(gdf)

# Convert to WGS84 for display on folium if needed
if dem_crs.to_string() != "EPSG:4326":
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
else:
    gdf_wgs84 = gdf.copy()

# Show map with basin polygon and outlet marker
m2 = folium.Map(location=[outlet_lat, outlet_lon], zoom_start=11)
folium.GeoJson(data=json.loads(gdf_wgs84.to_json()), name="Basin", style_function=lambda x: {"fillColor":"#00AAFF","color":"#0066cc","weight":2,"fillOpacity":0.4}).add_to(m2)
folium.Marker(location=[outlet_lat, outlet_lon], popup="Outlet", icon=folium.Icon(color="red")).add_to(m2)
st_folium(m2, height=600)

# Download GeoJSON
st.write("### Download basin polygon")
geojson_str = gdf_wgs84.to_json()
st.download_button("Download basin GeoJSON", data=geojson_str, file_name="basin.geojson", mime="application/geo+json")

st.success("Delineation complete. If the area looks wrong, try a higher-resolution DEM, or ensure the outlet is snapped to a stream (place it on a low accumulation pixel).")
