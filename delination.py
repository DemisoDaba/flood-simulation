# delineate_watershed.py
"""
Streamlit app: Delineate watershed/study-area from an outlet point using GEE + HydroSHEDS (if available).
- Click map or enter lat,lon.
- Finds HydroSHEDS basin containing the outlet (fast).
- Displays basin polygon + DEM hillshade for context.
- Export basin as GeoJSON.
Notes:
- Requires earthengine-api and geemap.
- For local use: run ee.Authenticate(); ee.Initialize()
- For deployed use: initialize with service account credentials (see comments below).
"""

import streamlit as st
import ee
import geemap.foliumap as geemap
import geopandas as gpd
import json
from shapely.geometry import shape
import tempfile
from streamlit_folium import st_folium

st.set_page_config(page_title="Watershed Delineation (Outlet -> Basin)", layout="wide")
st.title("Watershed / Study-area Delineation from Outlet Point 🌊")

# ---------------------------
# EARTH ENGINE AUTH / INITIALIZE
# ---------------------------
# Local testing (interactive):
#   1) uncomment ee.Authenticate() then run locally to authenticate.
#   2) ee.Initialize()
#
# For headless deployment (Streamlit Cloud), use a service account:
#   - Create Google Cloud service account with Earth Engine access.
#   - Put the service account JSON into Streamlit secrets (base64 or raw).
#   - Use ee.ServiceAccountCredentials(...) or ee.Initialize(...) with credentials.
#
# Example (service account stored as base64 in st.secrets["GEE_SA_B64"]):
#
# import base64, json, os
# sa_json = json.loads(base64.b64decode(st.secrets["GEE_SA_B64"]))
# key_str = json.dumps(sa_json)
# with open('/tmp/sa_key.json', 'w') as f:
#     f.write(key_str)
# creds = ee.ServiceAccountCredentials(sa_json['client_email'], '/tmp/sa_key.json')
# ee.Initialize(credentials=creds)
#
# For local quick test use:
try:
    ee.Initialize()
except Exception as e:
    st.sidebar.info("GEE not initialized. If running locally, Streamlit will attempt interactive auth.")
    if st.sidebar.button("Authenticate GEE (local)"):
        try:
            ee.Authenticate()
            ee.Initialize()
            st.experimental_rerun()
        except Exception as e2:
            st.error(f"GEE auth failed: {e2}")
            st.stop()
    else:
        st.warning("Click 'Authenticate GEE (local)' for interactive login (local use).")
        st.stop()

# ---------------------------
# Parameters / UI
# ---------------------------
st.sidebar.header("Controls")
st.sidebar.markdown("Click the map to choose an outlet, or enter coordinates manually.")
latlon_input = st.sidebar.text_input("lat,lon (e.g. 9.0,38.7)", "")
search_button = st.sidebar.button("Find basin from coords")

# Choose dataset for basin lookup: HydroSHEDS basins (coarse) is a fast option.
# If HydroSHEDS isn't available in your GEE account, replace with another administrative or basin layer.
BASIN_COLLECTION = "WWF/HydroSHEDS/v1/Basins/hybas_7"  # <-- replace if needed

# DEM for background/context
DEM = ee.Image("USGS/SRTMGL1_003")  # global 30m SRTM DEM

# Build map with geemap
m = geemap.Map(center=[9, 38], zoom=6)
m.add_basemap("HYBRID")

# add DEM hillshade for context
hillshade = ee.Terrain.hillshade(DEM)
m.addLayer(hillshade.visualize(min=0, max=255), {}, "DEM hillshade", shown=False)

# Add basin collection (visual)
try:
    basins_fc = ee.FeatureCollection(BASIN_COLLECTION)
    # add basins layer with light outline (this might be heavy - shown off by default)
    m.addLayer(basins_fc.style(**{'color':'#0066cc','fillColor':'00000000','width':1}), {}, "HydroSHEDS basins")
except Exception as e:
    st.warning(f"Could not load basin collection '{BASIN_COLLECTION}'. Replace collection id if needed.\n{e}")

# Map click handling
st.write("## Click map to select the outlet point (or enter lat,lon and press button).")
map_state = st_folium(m.to_streamlit(height=500), returned_objects=["last_clicked"], key="map1")

selected_point = None
if map_state and map_state.get("last_clicked"):
    pt = map_state["last_clicked"]
    lat_clicked = pt["lat"]
    lon_clicked = pt["lng"]
    selected_point = (lat_clicked, lon_clicked)
    st.sidebar.success(f"Clicked: {lat_clicked:.6f}, {lon_clicked:.6f}")

# If manual input provided
if latlon_input and search_button:
    try:
        lat, lon = [float(x.strip()) for x in latlon_input.split(",")]
        selected_point = (lat, lon)
        st.sidebar.success(f"Using coords: {lat:.6f}, {lon:.6f}")
    except Exception as ex:
        st.sidebar.error("Could not parse coords. Use format: lat,lon")

if not selected_point:
    st.info("Awaiting map click or manual coordinates.")
    st.stop()

lat, lon = selected_point
# create EE point
pt_geom = ee.Geometry.Point([lon, lat])

st.write(f"### Outlet chosen: {lat:.6f}, {lon:.6f}")

# ---------------------------
# Find basin containing the point
# ---------------------------
st.write("Finding basin that contains the outlet point...")

# Query basins that intersect the point
try:
    candidate = basins_fc.filterBounds(pt_geom).first()
    if candidate is None:
        st.warning("No basin found in HydroSHEDS collection for this point. Try larger basin collection or different dataset.")
        st.stop()
    # get feature geometry and properties
    feat_json = candidate.getInfo()  # small feature, ok to getInfo
    geom = feat_json["geometry"]
    props = feat_json.get("properties", {})
    st.write("Basin properties (if available):")
    st.json(props)
except Exception as e:
    st.error(f"Error querying basins collection: {e}")
    st.stop()

# Convert to GeoDataFrame for display / export
try:
    basin_shape = shape(geom)
    gdf = gpd.GeoDataFrame([props], geometry=[basin_shape], crs="EPSG:4326")
    st.write("## Delineated Study Area (Basin polygon):")
    st.write(gdf)
    st.map(gdf)  # quick map preview
except Exception as e:
    st.error(f"Error converting basin to GeoDataFrame: {e}")

# Show basin on the geemap map (add a highlighted layer)
m2 = geemap.Map(center=[lat, lon], zoom=9)
m2.add_basemap("ROADMAP")
# add hillshade/context
m2.addLayer(hillshade.visualize(min=0, max=255), {}, "hillshade", shown=False)
# Add the basin feature
m2.addLayer(ee.FeatureCollection([candidate]), {'color':'FF0000'}, "Selected basin")
# Add the outlet marker
m2.add_marker(location=(lat, lon), popup="Outlet", draggable=False)
st.write("### Map (click to interact):")
st_folium(m2.to_streamlit(height=500), returned_objects=["last_clicked"])

# ---------------------------
# Export GeoJSON option
# ---------------------------
st.write("### Export / Download")
if st.button("Download basin GeoJSON"):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
        gdf.to_file(tmp.name, driver="GeoJSON")
        with open(tmp.name, "r", encoding="utf-8") as f:
            geojson_data = f.read()
        st.download_button("Download GeoJSON", geojson_data, file_name="basin.geojson", mime="application/geo+json")
    except Exception as e:
        st.error(f"Failed to prepare GeoJSON: {e}")

st.success("Delineation finished. Use the download button to export the basin polygon.")
