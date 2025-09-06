# groundwater_monitor.py
import streamlit as st
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Groundwater Monitoring", layout="wide")
st.title("Groundwater Monitoring Dashboard 🌍💧")

# 1. Upload study area shapefile (optional)
st.sidebar.header("Upload Study Area")
shapefile = st.sidebar.file_uploader("Upload shapefile (.shp, .shx, .dbf, .prj)", type=['zip'])

gdf = None
if shapefile:
    import zipfile, os
    with zipfile.ZipFile(shapefile, "r") as zip_ref:
        zip_ref.extractall("temp_shapefile")
    gdf = gpd.read_file("temp_shapefile")
    st.write("Shapefile preview:")
    st.dataframe(gdf.head())

# 2. Upload GRACE NetCDF file
st.sidebar.header("Upload GRACE Data (NetCDF)")
grace_file = st.sidebar.file_uploader("Upload GRACE NetCDF", type=['nc'])

if grace_file:
    ds = xr.open_dataset(grace_file)
    st.write("GRACE dataset variables:", list(ds.data_vars))
    
    # Select a variable (usually 'lwe_thickness')
    var_name = st.selectbox("Select variable", list(ds.data_vars))
    
    # Select a time range
    time_values = pd.to_datetime(ds['time'].values)
    start_date = st.date_input("Start date", time_values[0].date())
    end_date = st.date_input("End date", time_values[-1].date())
    
    ds_time = ds.sel(time=slice(str(start_date), str(end_date)))
    
    # Calculate mean over selected time
    mean_data = ds_time[var_name].mean(dim='time')
    
    st.write("Mean Groundwater Storage Anomaly (GRACE)")
    
    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(8,5))
    im = ax.imshow(mean_data.values, cmap='Blues')
    plt.colorbar(im, ax=ax, label="cm Equivalent Water Thickness")
    st.pyplot(fig)
    
    # Optional: overlay shapefile
    if gdf is not None:
        fig2, ax2 = plt.subplots(figsize=(8,5))
        im2 = ax2.imshow(mean_data.values, cmap='Blues')
        gdf.boundary.plot(ax=ax2, edgecolor='red')
        plt.colorbar(im2, ax=ax2, label="cm Equivalent Water Thickness")
        st.pyplot(fig2)
    
# 3. Time series at a point
st.sidebar.header("Groundwater Time Series")
if grace_file:
    st.write("Click to select a grid cell (row, col) to see time series")
    row = st.number_input("Row index", min_value=0, max_value=mean_data.shape[0]-1, value=mean_data.shape[0]//2)
    col = st.number_input("Column index", min_value=0, max_value=mean_data.shape[1]-1, value=mean_data.shape[1]//2)
    
    ts = ds_time[var_name][:, row, col].values
    dates = pd.to_datetime(ds_time['time'].values)
    df = pd.DataFrame({'Date': dates, 'Groundwater Anomaly (cm)': ts})
    st.line_chart(df.set_index('Date'))
    
st.info("This is a starter app for GRACE-based groundwater monitoring. You can extend it with soil moisture data, well observations, or regional masking.")

