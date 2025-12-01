import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
import rioxarray as rxr
import os

# Define a global PROJECTED_CRS for consistency in calculations
PROJECTED_CRS = "EPSG:32617" # WGS 84 / UTM Zone 17N (for Montgomery County, VA)

def initialize_feature_dataframe(block_groups_path="aligned_data/block_groups_clean_4326.geojson"): 
    """
    Loads cleaned block group data, reprojects it to a local UTM CRS,
    and sets geometries to centroids for feature calculation.

    Args:
        block_groups_path (str): Path to the cleaned block groups GeoJSON file.

    Returns:
        gpd.GeoDataFrame: Initialized GeoDataFrame with centroids as geometries and projected CRS.
    """
    block_groups_clean = gpd.read_file(block_groups_path)
    features_gdf = block_groups_clean.to_crs(PROJECTED_CRS)
    features_gdf['geometry'] = features_gdf.geometry.centroid
    return features_gdf

def calculate_nearest_facility_distances(features_gdf, facility_gdfs_dict):
    """
    Calculates the Euclidean distance from each block group centroid to the nearest
    facility of specified types.

    Args:
        features_gdf (gpd.GeoDataFrame): GeoDataFrame of block group centroids (in PROJECTED_CRS).
        facility_gdfs_dict (dict): Dictionary where keys are facility names (e.g., 'healthcare')
                                   and values are GeoDataFrames of facilities (in COMMON_CRS).

    Returns:
        gpd.GeoDataFrame: Updated features_gdf with 'nearest_{facility_name}_dist' columns (in meters).
    """
    print("Calculating nearest facility distances...")
    for facility_name, facility_gdf in facility_gdfs_dict.items():
        if facility_gdf.empty:
            features_gdf[f'nearest_{facility_name}_dist'] = np.nan
            print(f"  No {facility_name} facilities found. Skipping distance calculation.")
            continue

        # Reproject facilities to the projected CRS for accurate distance calculation
        facility_gdf_proj = facility_gdf.to_crs(PROJECTED_CRS)

        # Convert facilities' geometries to points (centroids) for cKDTree
        facility_coords = list(zip(facility_gdf_proj.geometry.centroid.x, facility_gdf_proj.geometry.centroid.y))
        tree = cKDTree(facility_coords)

        # Query the tree for each block group centroid
        distances, _ = tree.query(list(zip(features_gdf.geometry.x, features_gdf.geometry.y)))
        features_gdf[f'nearest_{facility_name}_dist'] = distances
        print(f"  Calculated nearest {facility_name} distances.")
    return features_gdf

def count_facilities_within_buffers(features_gdf, facility_gdfs_dict, buffer_distances=[1000, 2000, 5000]):
    """
    Counts the number of facilities within specified buffer distances around each
    block group centroid.

    Args:
        features_gdf (gpd.GeoDataFrame): GeoDataFrame of block group centroids (in PROJECTED_CRS).
        facility_gdfs_dict (dict): Dictionary where keys are facility names (e.g., 'healthcare')
                                   and values are GeoDataFrames of facilities (in COMMON_CRS).
        buffer_distances (list): List of buffer radii in meters.

    Returns:
        gpd.GeoDataFrame: Updated features_gdf with '{facility_name}_count_{dist_km}km' columns.
    """
    print("Counting facilities within buffer zones...")
    for dist in buffer_distances:
        print(f"  Processing buffer distance: {dist} meters")
        # Create buffer polygons around the block group centroids
        buffers = features_gdf.geometry.buffer(dist)
        buffers_gdf = gpd.GeoDataFrame(features_gdf.drop(columns=['geometry']), geometry=buffers, crs=PROJECTED_CRS)

        for facility_name, facility_gdf in facility_gdfs_dict.items():
            if facility_gdf.empty:
                features_gdf[f'{facility_name}_count_{dist // 1000}km'] = 0
                continue

            # Reproject facilities to the projected CRS
            facility_gdf_proj = facility_gdf.to_crs(PROJECTED_CRS)

            # Perform spatial join and count facilities
            joined = gpd.sjoin(buffers_gdf, facility_gdf_proj, how='left', predicate='intersects')
            facility_counts_per_buffer = joined.groupby(joined.index)['index_right'].count().reindex(buffers_gdf.index, fill_value=0)
            features_gdf[f'{facility_name}_count_{dist // 1000}km'] = facility_counts_per_buffer
    print("  Finished counting facilities.")
    return features_gdf

def calculate_road_density(features_gdf, roads_path="aligned_data/roads_4326.geojson", block_groups_polygons_path="aligned_data/block_groups_clean_4326.geojson"):
    """
    Calculates road density (length per unit area) for each block group.

    Args:
        features_gdf (gpd.GeoDataFrame): GeoDataFrame with block group centroids and other features.
        roads_path (str): Path to the aligned OSM roads GeoJSON.
        block_groups_polygons_path (str): Path to the aligned block group polygons GeoJSON.

    Returns:
        gpd.GeoDataFrame: Updated features_gdf with 'total_road_length_m', 'area_sqm', and 'road_density_km_per_sqkm'.
    """
    print("Calculating road density...")
    roads_gdf_loaded = gpd.read_file(roads_path).to_crs(PROJECTED_CRS)
    block_groups_polygons_loaded = gpd.read_file(block_groups_polygons_path).to_crs(PROJECTED_CRS)

    # Explode MultiLineString geometries for accurate length calculation
    roads_gdf_loaded_exploded = roads_gdf_loaded.explode(index_parts=True).reset_index(drop=True)

    # Perform spatial intersection
    roads_in_block_groups = gpd.sjoin(block_groups_polygons_loaded, roads_gdf_loaded_exploded, how='inner', predicate='intersects')

    # Calculate total road length per block group
    road_length_per_bg = roads_in_block_groups.groupby('GEOID')['geometry'].apply(lambda x: x.length.sum()).reset_index()
    road_length_per_bg.rename(columns={'geometry': 'total_road_length_m'}, inplace=True)

    # Calculate area of each block group
    block_groups_polygons_loaded['area_sqm'] = block_groups_polygons_loaded.geometry.area
    block_group_areas = block_groups_polygons_loaded[['GEOID', 'area_sqm']]

    # Merge road lengths and areas with features_gdf
    features_gdf = features_gdf.merge(road_length_per_bg, on='GEOID', how='left')
    features_gdf = features_gdf.merge(block_group_areas, on='GEOID', how='left')

    # Fill NaN values for road length with 0
    features_gdf['total_road_length_m'] = features_gdf['total_road_length_m'].fillna(0)

    # Ensure 'area_sqm' is not zero to avoid division by zero
    features_gdf['area_sqm'] = features_gdf['area_sqm'].replace(0, np.nan) # Replace 0 with NaN for division

    # Compute road density (length in km, area in sq km)
    features_gdf['road_density_km_per_sqkm'] = (features_gdf['total_road_length_m'] / 1000) / (features_gdf['area_sqm'] / 1_000_000)
    print("  Finished calculating road density.")
    return features_gdf

def calculate_population_density(features_gdf):
    """
    Calculates population density (people per square kilometer) for each block group.

    Args:
        features_gdf (gpd.GeoDataFrame): GeoDataFrame containing 'population' and 'area_sqm' columns.

    Returns:
        gpd.GeoDataFrame: Updated features_gdf with 'population_density_per_sqkm'.
    """
    print("Calculating population density...")
    if 'population' in features_gdf.columns and 'area_sqm' in features_gdf.columns:
        # Replace zero area values with NaN to prevent division by zero
        area_sqkm = features_gdf['area_sqm'].replace(0, np.nan) / 1_000_000
        features_gdf['population_density_per_sqkm'] = features_gdf['population'] / area_sqkm
    else:
        print("Warning: 'population' or 'area_sqm' column not found. Cannot calculate population density.")
        features_gdf['population_density_per_sqkm'] = np.nan
    print("  Finished calculating population density.")
    return features_gdf

def calculate_viirs_stats(features_gdf, viirs_raster_path="aligned_data/VIIRS_2022_aligned.tif", block_groups_polygons_path="aligned_data/block_groups_clean_4326.geojson"):
    """
    Calculates mean, std, max, and sum of VIIRS nightlight intensity for each block group.

    Args:
        features_gdf (gpd.GeoDataFrame): GeoDataFrame to merge statistics into.
        viirs_raster_path (str): Path to the aligned VIIRS raster.
        block_groups_polygons_path (str): Path to the aligned block group polygons GeoJSON.

    Returns:
        gpd.GeoDataFrame: Updated features_gdf with VIIRS statistics.
    """
    print("Calculating VIIRS nightlight statistics...")
    block_groups_polygons = gpd.read_file(block_groups_polygons_path)
    viirs_raster = rxr.open_rasterio(viirs_raster_path, masked=True)

    viirs_stats = []
    for index, row in block_groups_polygons.iterrows():
        geom = row.geometry
        geoid = row['GEOID']
        try:
            # Clip the VIIRS raster to the current block group's geometry
            clipped_viirs = viirs_raster.rio.clip([geom], block_groups_polygons.crs, drop=True)
            values = clipped_viirs.values.flatten()
            values = values[~np.isnan(values)] # Remove NaN values

            if values.size > 0:
                viirs_stats.append({
                    'GEOID': geoid,
                    'viirs_mean': np.nanmean(values),
                    'viirs_std': np.nanstd(values),
                    'viirs_max': np.nanmax(values),
                    'viirs_sum': np.nansum(values)
                })
            else:
                viirs_stats.append({
                    'GEOID': geoid,
                    'viirs_mean': np.nan, 'viirs_std': np.nan, 'viirs_max': np.nan, 'viirs_sum': np.nan
                })
        except Exception as e:
            print(f"Error processing VIIRS for GEOID {geoid}: {e}")
            viirs_stats.append({
                'GEOID': geoid,
                'viirs_mean': np.nan, 'viirs_std': np.nan, 'viirs_max': np.nan, 'viirs_sum': np.nan
            })

    viirs_df = pd.DataFrame(viirs_stats)
    features_gdf = features_gdf.merge(viirs_df, on='GEOID', how='left')
    print("  Finished calculating VIIRS statistics.")
    return features_gdf

def calculate_sentinel_indices_and_bands(features_gdf, sentinel_raster_path="aligned_data/Sentinel2_2022_aligned.tif", block_groups_polygons_path="aligned_data/block_groups_clean_4326.geojson"):
    """
    Calculates mean NDVI, NDBI, and mean optical band values (Red, NIR, SWIR) for each block group.

    Args:
        features_gdf (gpd.GeoDataFrame): GeoDataFrame to merge statistics into.
        sentinel_raster_path (str): Path to the aligned Sentinel-2 raster.
        block_groups_polygons_path (str): Path to the aligned block group polygons GeoJSON.

    Returns:
        gpd.GeoDataFrame: Updated features_gdf with Sentinel-2 statistics.
    """
    print("Calculating Sentinel-2 derived features...")
    block_groups_polygons = gpd.read_file(block_groups_polygons_path)
    sentinel_raster = rxr.open_rasterio(sentinel_raster_path, masked=True)

    # Sentinel-2 band mapping: B8 (NIR), B4 (Red), B11 (SWIR-1)
    NIR_BAND_INDEX = 0
    RED_BAND_INDEX = 1
    SWIR_BAND_INDEX = 2

    sentinel_stats = []
    for index, row in block_groups_polygons.iterrows():
        geom = row.geometry
        geoid = row['GEOID']
        try:
            clipped_sentinel = sentinel_raster.rio.clip([geom], block_groups_polygons.crs, drop=True)

            if clipped_sentinel.size == 0 or np.all(np.isnan(clipped_sentinel.values)):
                sentinel_stats.append({
                    'GEOID': geoid,
                    's2_red_mean': np.nan, 's2_nir_mean': np.nan, 's2_swir_mean': np.nan,
                    's2_ndvi_mean': np.nan, 's2_ndvi_std': np.nan, 's2_ndbi_mean': np.nan
                })
                continue

            # Extract values for each band
            nir_values = clipped_sentinel.isel(band=NIR_BAND_INDEX).values.flatten()
            red_values = clipped_sentinel.isel(band=RED_BAND_INDEX).values.flatten()
            swir_values = clipped_sentinel.isel(band=SWIR_BAND_INDEX).values.flatten()

            # Filter out NaN values from all bands simultaneously for consistent calculations
            valid_indices = ~np.isnan(nir_values) & ~np.isnan(red_values) & ~np.isnan(swir_values)
            nir_valid = nir_values[valid_indices]
            red_valid = red_values[valid_indices]
            swir_valid = swir_values[valid_indices]

            if len(nir_valid) == 0: # No valid pixels after filtering NaNs
                sentinel_stats.append({
                    'GEOID': geoid,
                    's2_red_mean': np.nan, 's2_nir_mean': np.nan, 's2_swir_mean': np.nan,
                    's2_ndvi_mean': np.nan, 's2_ndvi_std': np.nan, 's2_ndbi_mean': np.nan
                })
                continue

            # Calculate NDVI
            ndvi_numerator = nir_valid - red_valid
            ndvi_denominator = nir_valid + red_valid
            ndvi = np.where(ndvi_denominator != 0, ndvi_numerator / ndvi_denominator, np.nan)

            # Calculate NDBI
            ndbi_numerator = swir_valid - nir_valid
            ndbi_denominator = swir_valid + nir_valid
            ndbi = np.where(ndbi_denominator != 0, ndbi_numerator / ndbi_denominator, np.nan)

            sentinel_stats.append({
                'GEOID': geoid,
                's2_red_mean': np.nanmean(red_valid),
                's2_nir_mean': np.nanmean(nir_valid),
                's2_swir_mean': np.nanmean(swir_valid),
                's2_ndvi_mean': np.nanmean(ndvi),
                's2_ndvi_std': np.nanstd(ndvi), # Also include std dev for NDVI
                's2_ndbi_mean': np.nanmean(ndbi)
            })

        except Exception as e:
            print(f"Error processing Sentinel-2 for GEOID {geoid}: {e}")
            sentinel_stats.append({
                'GEOID': geoid,
                's2_red_mean': np.nan, 's2_nir_mean': np.nan, 's2_swir_mean': np.nan,
                's2_ndvi_mean': np.nan, 's2_ndvi_std': np.nan, 's2_ndbi_mean': np.nan
            })

    sentinel_df = pd.DataFrame(sentinel_stats)
    features_gdf = features_gdf.merge(sentinel_df, on='GEOID', how='left')
    print("  Finished calculating Sentinel-2 derived features.")
    return features_gdf
