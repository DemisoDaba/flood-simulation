# delineate_watershed_local.py
"""
Streamlit app: Delineate watershed (upstream area) from an outlet point using a local DEM (GeoTIFF).
No GEE required.

Dependencies (requirements.txt):
streamlit
streamlit-autorefresh
tenacity
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
import numpy as np
import richdem as rd
import geopandas as gpd
from shapely.geometry import shape
import folium
from streamlit_folium import st_folium
import json
import tempfile
import os
from pyproj import Transformer
from collections import deque
from shapely.ops import unary_union

st.set_page_config(page_title="Local Watershed Delineation", layout="wide")
st.title("Watershed Delineation from Outlet (Local DEM) — No GEE")

# --------------------------
# 1) Upload DEM
# --------------------------
st.sidebar.header("1) Upload DEM (GeoTIFF)")
dem_file = st.sidebar.file_uploader("Upload DEM GeoTIFF", type=["tif","tiff"])
st.sidebar.markdown("""
**Notes**
- DEM must have a CRS (e.g., EPSG:4326 or projected CRS).
- For speed, clip your DEM to the study area before upload.
""")
if dem_file is None:
    st.info("Please upload a DEM GeoTIFF to proceed.")
    st.stop()

# Save temporary DEM
temp_dem_path = os.path.join(tempfile.gettempdir(), "uploaded_dem.tif")
with open(temp_dem_path, "wb") as f:
    f.write(dem_file.getbuffer())

# Read DEM
with rasterio.open(temp_dem_path) as src:
    dem_crs = src.crs
    dem_transform = src.transform
    dem_bounds = src.bounds
    dem_arr = src.read(1).astype('float64')
    dem_nodata = src.nodatavals[0]
    dem_profile = src.profile

# Handle nodata
if dem_nodata is not None:
    dem_arr[dem_arr==dem_nodata] = np.nan

st.sidebar.write(f"DEM CRS: {dem_crs}")
st.sidebar.write(f"DEM bounds: {dem_bounds}")

# --------------------------
# 2) Select outlet
# --------------------------
st.write("## 2) Choose outlet point")
st.write("Click on the map or enter lat,lon manually in the sidebar.")

# Folium map
cent_y = (dem_bounds.top + dem_bounds.bottom)/2
cent_x = (dem_bounds.left + dem_bounds.right)/2
m = folium.Map(location=[cent_y, cent_x], zoom_start=9)
folium.Rectangle(
    bounds=[[dem_bounds.bottom, dem_bounds.left],[dem_bounds.top, dem_bounds.right]],
    color="blue", weight=1, fill=False, popup="DEM bounds"
).add_to(m)

st_map = st_folium(m, height=450, returned_objects=["last_clicked"]) or {}
clicked = st_map.get("last_clicked") if st_map else None

# Manual coords
manual_coords = st.sidebar.text_input("Manual outlet lat,lon (e.g., 9.0,38.7)")
outlet_lat, outlet_lon = None, None

if clicked:
    outlet_lat, outlet_lon = clicked["lat"], clicked["lng"]
    st.sidebar.success(f"Clicked: {outlet_lat:.6f}, {outlet_lon:.6f}")
elif manual_coords:
    try:
        outlet_lat, outlet_lon = [float(x.strip()) for x in manual_coords.split(",")]
        st.sidebar.success(f"Using manual coords: {outlet_lat:.6f}, {outlet_lon:.6f}")
    except:
        st.sidebar.error("Could not parse manual coords. Use format: lat,lon")
        st.stop()
else:
    st.info("Click map or input coordinates to continue.")
    st.stop()

# Convert lat/lon -> DEM CRS
if dem_crs.to_string() != "EPSG:4326":
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    outlet_x_dem, outlet_y_dem = transformer.transform(outlet_lon, outlet_lat)
else:
    outlet_x_dem, outlet_y_dem = outlet_lon, outlet_lat

# Convert DEM coordinates -> row,col
inv_transform = ~dem_transform
col_f, row_f = inv_transform * (outlet_x_dem, outlet_y_dem)
row, col = int(np.floor(row_f)), int(np.floor(col_f))

nrows, ncols = dem_arr.shape
if not (0<=row<nrows and 0<=col<ncols):
    st.error("Outlet outside DEM bounds.")
    st.stop()

st.write(f"Outlet mapped to DEM pixel row={row}, col={col}")

# --------------------------
# 3) Hydrological processing
# --------------------------
st.write("## 3) Running hydrological preprocessing...")

dem_work = dem_arr.copy()
nan_mask = np.isnan(dem_work)
if np.any(nan_mask):
    dem_work[nan_mask] = np.nanmax(dem_work)+1000

rd_dem = rd.rdarray(dem_work, no_data=np.nan)
rd_dem.geotransform = dem_transform[:6] if hasattr(dem_transform, '__getitem__') else (
    dem_transform.a, dem_transform.b, dem_transform.c, dem_transform.d, dem_transform.e, dem_transform.f
)

# Fill depressions
try:
    rd_filled = rd.FillDepressions(rd_dem, in_place=False)
except:
    rd_filled = rd.FillDepressions(rd_dem, in_place=True)

# Flow direction
flow_dir = rd.FlowDirectionD8(rd_filled)

# Flow accumulation
acc = rd.FlowAccumulation(flow_dir, method='D8')
acc_arr = np.array(acc)
st.write("Flow accumulation computed.")

# --------------------------
# 4) Compute upstream mask
# --------------------------
fdir = np.array(flow_dir, dtype=np.int32)
dir_to_delta = {1:(0,1),2:(1,1),3:(1,0),4:(1,-1),5:(0,-1),6:(-1,-1),7:(-1,0),8:(-1,1)}
visited = np.zeros_like(fdir,dtype=np.uint8)
q = deque()
visited[row,col]=1
q.append((row,col))

neigh_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
while q:
    i,j = q.popleft()
    for di,dj in neigh_offsets:
        ni,nj=i+di,j+dj
        if 0<=ni<nrows and 0<=nj<ncols and visited[ni,nj]==0:
            val=int(fdir[ni,nj])
            if val==0: continue
            d=dir_to_delta.get(val)
            if d is None: continue
            dest_i,dest_j=ni+d[0], nj+d[1]
            if dest_i==i and dest_j==j:
                visited[ni,nj]=1
                q.append((ni,nj))

upstream_mask = visited.astype(bool)
if np.any(nan_mask):
    upstream_mask[nan_mask]=False

st.write(f"Computed upstream mask — {upstream_mask.sum()} cells.")

# --------------------------
# 5) Convert mask -> polygon
# --------------------------
mask_uint8 = upstream_mask.astype(np.uint8)
shapes_gen = shapes(mask_uint8, mask=mask_uint8==1, transform=dem_transform)

polys=[]
for geom,val in shapes_gen:
    if val==1:
        polys.append(shape(geom))

if not polys:
    st.error("No polygon generated. Try another outlet or DEM.")
    st.stop()

basin_geom = unary_union(polys)
gdf = gpd.GeoDataFrame({"id":[1]}, geometry=[basin_geom], crs=dem_crs.to_string())

st.write("## 5) Delineated Basin")
st.write(gdf)

# Convert to WGS84 for folium
gdf_wgs84 = gdf.to_crs("EPSG:4326") if dem_crs.to_string()!="EPSG:4326" else gdf.copy()

m2 = folium.Map(location=[outlet_lat, outlet_lon], zoom_start=11)
folium.GeoJson(data=json.loads(gdf_wgs84.to_json()),
               name="Basin",
               style_function=lambda x: {"fillColor":"#00AAFF","color":"#0066cc","weight":2,"fillOpacity":0.4}).add_to(m2)
folium.Marker(location=[outlet_lat, outlet_lon],
              popup="Outlet", icon=folium.Icon(color="red")).add_to(m2)
st_folium(m2, height=600)

# --------------------------
# 6) Download GeoJSON
# --------------------------
geojson_str = gdf_wgs84.to_json()
st.download_button("Download basin GeoJSON", data=geojson_str, file_name="basin.geojson", mime="application/geo+json")

st.success("Delineation complete. Ensure outlet is on a low-accumulation pixel for realistic watershed.")
