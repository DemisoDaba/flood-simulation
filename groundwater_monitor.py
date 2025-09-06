# groundwater_monitor_gee.py
import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.pyplot as plt

# Initialize GEE
ee.Initialize()

st.set_page_config(page_title="Global Groundwater Monitor", layout="wide")
st.title("Global Groundwater Monitoring via GEE 🌍💧")

# Sidebar instructions
st.sidebar.write("Click on the map to view groundwater storage anomaly time series")

# Create interactive map
Map = geemap.Map(center=[0, 0], zoom=2)
Map.add_basemap("HYBRID")

# Example: GRACE TWS anomaly dataset from GEE
# Dataset: NASA/GRACE/MASS_GRIDS
tws = ee.ImageCollection("NASA/GRACE/MASS_GRIDS") \
        .select("lwe_thickness")  # cm equivalent water thickness

# Add a median layer for visualization
median = tws.median()
Map.addLayer(median, {'min': -10, 'max': 10, 'palette': ['red','white','blue']}, "Median TWS Anomaly")

# Enable click for point
clicked_coords = st.text_input("Clicked coordinates (lat, lon)")

def get_point_timeseries(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    # Reduce each image to point
    def extract_val(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        val = img.reduceRegion(ee.Reducer.mean(), point).get("lwe_thickness")
        return ee.Feature(None, {"date": date, "value": val})
    feats = tws.map(extract_val).filter(ee.Filter.notNull(["value"]))
    df = geemap.ee_to_pandas(feats)
    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df

# Map click (dummy for now) - manually enter lat/lon
if clicked_coords:
    try:
        lat, lon = map(float, clicked_coords.split(","))
        ts_df = get_point_timeseries(lat, lon)
        st.write(f"Time series at ({lat}, {lon}):")
        st.line_chart(ts_df.set_index("date")["value"])
    except:
        st.error("Enter coordinates as lat,lon e.g., 28.5,84.3")

# Display map
st_data = Map.to_streamlit(height=500)
