"""
Streamlit app: Delineate watershed (upstream area) from an outlet using a local DEM (GeoTIFF) with pysheds.
No GEE required.

Dependencies:
streamlit
numpy
rasterio
geopandas
shapely
folium
streamlit-folium
pyproj
pysheds
"""

import streamlit as st
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
from pysheds.grid import Grid
import os
import tempfile
import json
from shapely.ops import unary_union

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Local Watershed Delineation", layout="wide")
st.title("Watershed Delineation from Outlet (Local DEM) — pysheds version")

# -----------------------------
# 1) Upload DEM
# -----------------------------
st.sidebar.header("1) Upload DEM (GeoTIFF)")
dem_file = st.sidebar.file_uploader("Upload DEM GeoTIFF", type=["tif", "tiff"])

if dem_file is None:
    st.info("Please upload a DEM GeoTIFF to proceed.")
    st.stop()

# Save uploaded DEM to temporary file
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

# Replace nodata with np.nan
if dem_nodata is not None:
    dem_arr[dem_arr == dem_nodata] = np.nan

st.sidebar.write(f"DEM CRS: {dem_crs}")
st.sidebar.write(f"DEM bounds: {dem_bounds}")

# -----------------------------
# 2) Select outlet
# -----------------------------
st.write("## 2) Choose outlet point")
st.write("Click on map to place outlet, or enter lat,lon manually in sidebar.")

# Center map on DEM
cent_y = (dem_bounds.top + dem_bounds.bottom) / 2.0
cent_x = (dem_bounds.left + dem_bounds.right) / 2.0
m = folium.Map(location=[cent_y, cent_x], zoom_start=9)

# Add DEM bounds rectangle
folium.Rectangle(
    bounds=[[dem_bounds.bottom, dem_bounds.left], [dem_bounds.top, dem_bounds.right]],
    color="blue", weight=1, fill=False
).add_to(m)

st_map = st_folium(m, height=450, returned_objects=["last_clicked"])
manual_coords = st.sidebar.text_input("Manual outlet lat,lon (e.g. 9.0,38.7)")

clicked = st_map.get("last_clicked") if st_map else None
outlet_lat, outlet_lon = None, None

# Use clicked point or manual input
if clicked:
    outlet_lat = clicked["lat"]
    outlet_lon = clicked["lng"]
    st.sidebar.success(f"Clicked: {outlet_lat:.6f}, {outlet_lon:.6f}")
elif manual_coords:
    try:
        lat, lon = [float(x.strip()) for x in manual_coords.split(",")]
        outlet_lat, outlet_lon = lat, lon
        st.sidebar.success(f"Using manual coords: {lat:.6f}, {lon:.6f}")
    except:
        st.sidebar.error("Use format: lat,lon")
        st.stop()
else:
    # Provide button to auto-select DEM center
    if st.sidebar.button("Use DEM center as outlet"):
        if dem_crs.to_string() == "EPSG:4326":
            outlet_lat, outlet_lon = cent_y, cent_x
        else:
            transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
            outlet_lon, outlet_lat = transformer.transform(cent_x, cent_y)
        st.sidebar.info(f"Using DEM center: {outlet_lat:.6f}, {outlet_lon:.6f}")
    else:
        st.info("Click map, input coordinates, or press 'Use DEM center as outlet'.")
        st.stop()

# -----------------------------
# 3) Convert outlet to DEM CRS & row/col safely
# -----------------------------
if dem_crs.to_string() != "EPSG:4326":
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    outlet_x, outlet_y = transformer.transform(outlet_lon, outlet_lat)
else:
    outlet_x, outlet_y = outlet_lon, outlet_lat

inv_transform = ~dem_transform
col_f, row_f = inv_transform * (outlet_x, outlet_y)

# Check for NaN
if np.isnan(row_f) or np.isnan(col_f):
    st.error("Outlet coordinates are invalid or outside DEM bounds.")
    st.stop()

row = int(np.clip(np.floor(row_f), 0, dem_arr.shape[0]-1))
col = int(np.clip(np.floor(col_f), 0, dem_arr.shape[1]-1))
st.write(f"Outlet mapped to DEM pixel row={row}, col={col} (clamped to DEM bounds)")

# -----------------------------
# 4) Hydrological processing with pysheds
# -----------------------------
st.write("## 3) Running hydrological preprocessing with pysheds...")

# 1) Replace NaN with nodata and cast
dem_cleaned = np.where(np.isnan(dem_arr), -9999, dem_arr).astype('float32')

# 2) Save cleaned DEM temporarily
safe_dem_path = os.path.join(tempfile.gettempdir(), "safe_dem.tif")
with rasterio.open(
    safe_dem_path,
    'w',
    driver='GTiff',
    height=dem_cleaned.shape[0],
    width=dem_cleaned.shape[1],
    count=1,
    dtype='float32',
    crs=dem_crs,
    transform=dem_transform,
    nodata=-9999
) as dst:
    dst.write(dem_cleaned, 1)

# 3) Create a Grid and read raster properly
grid = Grid()
grid.read_raster(safe_dem_path, data_name='dem', dtype='float32', nodata=-9999)

# 4) Hydrological preprocessing using the raster name 'dem'
grid.fill_depressions(data='dem', out_name='flooded_dem', nodata=-9999)
grid.resolve_flats('flooded_dem', out_name='inflated_dem')
grid.flowdir('inflated_dem', out_name='dir', dirmap=Grid.D8)
grid.accumulation('dir', out_name='acc')

st.success("Hydrological preprocessing complete!")


# -----------------------------
# 5) Delineate upstream basin
# -----------------------------
st.write("Computing upstream basin...")

basin_mask = grid.catchment(x=outlet_x, y=outlet_y, data='dir', dirmap=Grid.D8)
upstream_mask = basin_mask.astype(bool)
st.write(f"Upstream mask has {upstream_mask.sum()} cells.")

# -----------------------------
# 6) Convert mask to polygon
# -----------------------------
mask_uint8 = upstream_mask.astype(np.uint8)
shapes_gen = shapes(mask_uint8, mask=mask_uint8==1, transform=dem_transform)
polys = [shape(geom) for geom, val in shapes_gen if val==1]

if not polys:
    st.error("No polygon generated from mask. Check outlet location.")
    st.stop()

basin_geom = unary_union(polys)
gdf = gpd.GeoDataFrame({"id":[1]}, geometry=[basin_geom], crs=dem_crs.to_string())
gdf_wgs84 = gdf.to_crs("EPSG:4326") if dem_crs.to_string() != "EPSG:4326" else gdf.copy()

# -----------------------------
# 7) Display basin
# -----------------------------
st.write("## 4) Result: Delineated Basin")
st.write(gdf)

m2 = folium.Map(location=[outlet_lat, outlet_lon], zoom_start=11)
folium.GeoJson(
    data=json.loads(gdf_wgs84.to_json()),
    name="Basin",
    style_function=lambda x: {"fillColor":"#00AAFF","color":"#0066cc","weight":2,"fillOpacity":0.4}
).add_to(m2)
folium.Marker(location=[outlet_lat, outlet_lon], popup="Outlet", icon=folium.Icon(color="red")).add_to(m2)
st_folium(m2, height=600)

# -----------------------------
# 8) Download
# -----------------------------
st.write("### Download basin polygon")
geojson_str = gdf_wgs84.to_json()
st.download_button("Download basin GeoJSON", data=geojson_str, file_name="basin.geojson", mime="application/geo+json")

st.success("Delineation complete!")
