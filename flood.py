import streamlit as st
import os
import tempfile
import shutil
import zipfile
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio import features
from shapely.geometry import shape
from scipy.ndimage import distance_transform_edt
from rasterio.features import shapes

# --- Your Original Flood Simulation Functions --- #

def load_data(satellite_path, dem_path, houses_path, roads_path, river_path):
    for path in [satellite_path, dem_path, houses_path, roads_path, river_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    with rasterio.open(satellite_path) as sat:
        satellite = sat.read()
        sat_transform = sat.transform
        sat_bounds = sat.bounds
        sat_profile = sat.profile
    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)
        dem_transform = dem.transform
        dem_bounds = dem.bounds
        dem_profile = dem.profile
    houses = gpd.read_file(houses_path)
    roads = gpd.read_file(roads_path)
    river = gpd.read_file(river_path)
    if sat_profile['crs'] != dem_profile['crs'] or sat_profile['crs'] != river.crs:
        st.warning("CRS mismatch detected. Reprojecting all vector data to river CRS.")
        houses = houses.to_crs(river.crs)
        roads = roads.to_crs(river.crs)
        # Note: Reprojecting raster is complex, skipping here for simplicity
    return satellite, sat_transform, sat_bounds, sat_profile, dem_data, dem_transform, dem_bounds, dem_profile, houses, roads, river


def simulate_flood(dem_data, dem_transform, dem_profile, river_gdf, flood_height, output_dir):
    if river_gdf.geometry.type.isin(['MultiLineString']).any():
        river_gdf = river_gdf.buffer(0)
        river_gdf = river_gdf.dissolve(by=None)

    # Rasterize river geometry to DEM grid
    river_mask = features.rasterize(
        [(geom, 1) for geom in river_gdf.geometry],
        out_shape=dem_data.shape,
        transform=dem_transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    # Create flood threshold elevation based on river elevation + flood height
    flood_threshold = np.full_like(dem_data, fill_value=np.inf, dtype=np.float32)
    flood_threshold[river_mask] = dem_data[river_mask] + flood_height

    # Distance transform from river pixels for propagation
    distance, indices = distance_transform_edt(~river_mask, return_indices=True)

    # Propagate flood threshold to entire raster based on nearest river pixel
    propagated_threshold = flood_threshold[indices[0], indices[1]]

    # Determine flood extent by comparing DEM with propagated flood threshold
    flood_mask = (dem_data <= propagated_threshold).astype(np.uint8)

    # Vectorize flood extent raster to polygons
    shapes_gen = shapes(flood_mask, mask=flood_mask, transform=dem_transform)
    geoms = [shape(geom) for geom, value in shapes_gen if value == 1]
    flood_extent_gdf = gpd.GeoDataFrame(geometry=geoms, crs=river_gdf.crs)

    # Save flood extent shapefile
    buffer_output_path = os.path.join(output_dir, f"flood_extent_{flood_height}m.shp")
    flood_extent_gdf.to_file(buffer_output_path)

    return flood_mask, flood_extent_gdf


def identify_affected_features(gdf, flood_extent_gdf, feature_type, flood_height):
    if flood_extent_gdf.empty:
        return gpd.GeoDataFrame()
    
    flood_union = flood_extent_gdf.unary_union

    if feature_type == "roads":
        affected_segments = []
        for idx, road in gdf.iterrows():
            intersection = road.geometry.intersection(flood_union)
            if not intersection.is_empty:
                if intersection.geom_type == "LineString":
                    affected_segments.append(intersection)
                elif intersection.geom_type == "MultiLineString":
                    affected_segments.extend(list(intersection.geoms))
# //Return for Roads
        if affected_segments:
            affected_gdf = gpd.GeoDataFrame(geometry=affected_segments, crs=gdf.crs)
            affected_gdf['fld_hgt'] = flood_height
            return affected_gdf
        else:
            return gpd.GeoDataFrame()
    else:
        affected_features = gdf[gdf.intersects(flood_union)].copy()
        if not affected_features.empty:
            affected_features['fld_hgt'] = flood_height
        return affected_features


def plot_results(satellite, sat_transform, sat_bounds, dem_data, flood_mask, houses, affected_houses, roads, affected_roads, river, flood_height, output_dir, flood_extent_gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    satellite_rgb = np.transpose(satellite[:3], (1, 2, 0))
    ax.imshow(satellite_rgb, extent=[sat_bounds.left, sat_bounds.right, sat_bounds.bottom, sat_bounds.top])
    flood_cmap = ListedColormap(['none', 'blue'])
    ax.imshow(flood_mask, cmap=flood_cmap, alpha=0.5, extent=[sat_bounds.left, sat_bounds.right, sat_bounds.bottom, sat_bounds.top])
    if not flood_extent_gdf.empty:
        flood_extent_gdf.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=1.5, alpha=0.7, label='Flood Extent')
    houses.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
    if not affected_houses.empty:
        affected_houses.plot(ax=ax, facecolor='yellow', edgecolor='black', linewidth=1)
    roads.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=1)
    if not affected_roads.empty:
        affected_roads.plot(ax=ax, facecolor='none', edgecolor='orange', linewidth=2)
    river.plot(ax=ax, facecolor='none', edgecolor='cyan', linewidth=2)
    legend_elements = [
        Patch(facecolor='blue', edgecolor='none', alpha=0.5, label='Flooded Area'),
        Patch(facecolor='none', edgecolor='purple', linewidth=1.5, alpha=0.7, label='Flood Extent Boundary'),
        Patch(facecolor='none', edgecolor='red', linewidth=1, label='Houses'),
        Patch(facecolor='yellow', edgecolor='black', linewidth=1, label=f'Affected Houses ({len(affected_houses)})'),
        Patch(facecolor='none', edgecolor='gray', linewidth=1, label='Roads'),
        Patch(facecolor='none', edgecolor='orange', linewidth=2, label=f'Affected Roads ({affected_roads.geometry.length.sum()/1000:.2f} km)'),
        Patch(facecolor='none', edgecolor='cyan', linewidth=2, label='River')
    ]
    ax.legend(handles=legend_elements)
    plt.title(f"Flood Simulation at {flood_height}m Above Local River Elevation")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    output_path = os.path.join(output_dir, f"flood_map_{flood_height}m.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_flood_layer(flood_mask, dem_profile, output_path):
    profile = dem_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(flood_mask.astype(np.uint8), 1)


# --- Streamlit App Logic --- #

def save_uploaded_file(uploaded_file, tmp_dir):
    file_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_shapefile_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    for f in os.listdir(extract_dir):
        if f.endswith('.shp'):
            return os.path.join(extract_dir, f)
    raise FileNotFoundError("No .shp file found in the uploaded zip.")

def run_flood_simulation_streamlit(sat_path, dem_path, houses_path, roads_path, river_path, flood_height, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    satellite, sat_transform, sat_bounds, sat_profile, dem_data, dem_transform, dem_bounds, dem_profile, houses, roads, river = load_data(
        sat_path, dem_path, houses_path, roads_path, river_path
    )

    if river.geometry.type.isin(['MultiLineString']).any():
        river.geometry = river.geometry.buffer(0)
        river = river.dissolve(by=None)

    flood_mask, flood_extent_gdf = simulate_flood(dem_data, dem_transform, dem_profile, river, flood_height, output_dir)
    affected_houses = identify_affected_features(houses, flood_extent_gdf, "houses", flood_height)
    affected_roads = identify_affected_features(roads, flood_extent_gdf, "roads", flood_height)

    flood_output_path = os.path.join(output_dir, f"flood_extent_{flood_height}m.tif")
    save_flood_layer(flood_mask, dem_profile, flood_output_path)

    if not affected_houses.empty:
        houses_output_path = os.path.join(output_dir, f"affected_houses_{flood_height}m.shp")
        affected_houses.to_file(houses_output_path)
    if not affected_roads.empty:
        roads_output_path = os.path.join(output_dir, f"affected_roads_{flood_height}m.shp")
        affected_roads.to_file(roads_output_path)

    plot_results(satellite, sat_transform, sat_bounds, dem_data, flood_mask, houses, affected_houses, roads, affected_roads, river, flood_height, output_dir, flood_extent_gdf)


def main():
    st.set_page_config(page_title="Flood Simulation & Impact Analysis", layout="wide")
    st.title("🌊 Flood Simulation & Impact Analysis")

    st.sidebar.header("Upload Input Files")

    sat_file = st.sidebar.file_uploader("Satellite Raster (RGB) (.tif)", type=['tif', 'tiff'])
    dem_file = st.sidebar.file_uploader("DEM Raster (.tif)", type=['tif', 'tiff'])

    houses_zip = st.sidebar.file_uploader("Houses Shapefile (.zip)", type=['zip'])
    roads_zip = st.sidebar.file_uploader("Roads Shapefile (.zip)", type=['zip'])
    river_zip = st.sidebar.file_uploader("River Shapefile (.zip)", type=['zip'])

    flood_height = st.sidebar.number_input("Flood Height (meters)", min_value=0.0, value=10.0, step=0.5)
    output_dir = st.sidebar.text_input("Output Directory", value="output")

    run = st.sidebar.button("Run Flood Simulation")

    # Show satellite image preview immediately after upload
    if sat_file:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sat_path_tmp = save_uploaded_file(sat_file, tmpdir)
                with rasterio.open(sat_path_tmp) as src:
                    sat_img = src.read([1, 2, 3])
                    sat_img = np.transpose(sat_img, (1, 2, 0))
                    sat_img = np.clip(sat_img / sat_img.max(), 0, 1)  # Normalize for display
                st.subheader("Satellite Image Preview")
                st.image(sat_img, caption="Uploaded Satellite RGB", use_container_width=True)
        except Exception as e:
            st.error(f"Could not display satellite image preview: {e}")

    if run:
        if not all([sat_file, dem_file, houses_zip, roads_zip, river_zip]):
            st.error("Please upload all input files (satellite, DEM, and zipped shapefiles).")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                sat_path = save_uploaded_file(sat_file, tmpdir)
                dem_path = save_uploaded_file(dem_file, tmpdir)

                houses_zip_path = save_uploaded_file(houses_zip, tmpdir)
                roads_zip_path = save_uploaded_file(roads_zip, tmpdir)
                river_zip_path = save_uploaded_file(river_zip, tmpdir)

                houses_dir = os.path.join(tmpdir, 'houses')
                roads_dir = os.path.join(tmpdir, 'roads')
                river_dir = os.path.join(tmpdir, 'river')

                os.makedirs(houses_dir, exist_ok=True)
                os.makedirs(roads_dir, exist_ok=True)
                os.makedirs(river_dir, exist_ok=True)

                houses_path = extract_shapefile_zip(houses_zip_path, houses_dir)
                roads_path = extract_shapefile_zip(roads_zip_path, roads_dir)
                river_path = extract_shapefile_zip(river_zip_path, river_dir)

                st.info("Running flood simulation, please wait...")
                run_flood_simulation_streamlit(sat_path, dem_path, houses_path, roads_path, river_path, flood_height, output_dir)
                st.success("Flood simulation completed!")

                # Show the flood map PNG after simulation
                png_path = os.path.join(output_dir, f"flood_map_{flood_height}m.png")
                if os.path.exists(png_path):
                    st.subheader("Flood Map Output")
                    st.image(png_path, caption=f"Flood Map at {flood_height}m flood height", use_container_width=True)
                else:
                    st.warning("Flood map PNG not found.")

                # Show summary info about affected features
                affected_houses_shp = os.path.join(output_dir, f"affected_houses_{flood_height}m.shp")
                affected_roads_shp = os.path.join(output_dir, f"affected_roads_{flood_height}m.shp")

                if os.path.exists(affected_houses_shp):
                    affected_houses = gpd.read_file(affected_houses_shp)
                    st.write(f"**Number of houses affected:** {len(affected_houses)}")
                else:
                    st.write("No houses affected.")

                if os.path.exists(affected_roads_shp):
                    affected_roads = gpd.read_file(affected_roads_shp)
                    length_km = affected_roads.geometry.length.sum() / 1000
                    st.write(f"**Length of roads affected:** {length_km:.2f} km")
                else:
                    st.write("No roads affected.")

            except Exception as e:
                st.error(f"Error during simulation: {e}")


if __name__ == "__main__":
    main()
