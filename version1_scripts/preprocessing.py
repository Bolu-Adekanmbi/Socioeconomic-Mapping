import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask
import fiona
import numpy as np
import json
import os

COMMON_CRS = "EPSG:4326" # Common CRS for initial alignment

def ensure_common_crs_and_save(gdf, name, output_dir="aligned_data"):
    """
    Ensures a GeoDataFrame is in the COMMON_CRS and saves it.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to process.
        name (str): A descriptive name for the GeoDataFrame.
        output_dir (str): Directory to save the aligned data.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame in COMMON_CRS.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checking {name} CRS: {gdf.crs}")
    if gdf.crs != COMMON_CRS:
        gdf = gdf.to_crs(COMMON_CRS)
        print(f"Reprojected {name} to {COMMON_CRS}")
    output_path = f"{output_dir}/{name}_4326.geojson"
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved {name} to {output_path}")
    return gdf

def safe_clip(gdf, boundary_gdf):
    """
    Clips any geometry type GeoDataFrame safely to a polygon boundary GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to clip.
        boundary_gdf (gpd.GeoDataFrame): The boundary GeoDataFrame (must be a single polygon).

    Returns:
        gpd.GeoDataFrame: The clipped GeoDataFrame.
    """
    # New method to avoid deprecation warning in newer GeoPandas versions
    # Use unary_union for a single polygon boundary, or union_all() for multi-polygon
    # Assuming boundary_gdf contains a single polygon feature for clipping
    if len(boundary_gdf) > 1:
        poly = boundary_gdf.geometry.unary_union # For multiple features in boundary_gdf
    else:
        poly = boundary_gdf.geometry.iloc[0]

    # Ensure both GDFs are in the same CRS before intersection
    if gdf.crs != boundary_gdf.crs:
        gdf = gdf.to_crs(boundary_gdf.crs)

    # Intersect all geometries. It's safer to use overlay or spatial join for complex clipping,
    # but for simple point/line clipping by a single polygon, direct intersection can work.
    # Using a buffer(0) to fix invalid geometries before intersection
    gdf['geometry'] = gdf.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    gdf['geometry'] = gdf.geometry.intersection(poly)

    # Remove empty geometries resulting from clipping
    gdf = gdf[~gdf.geometry.is_empty].copy() # .copy() to avoid SettingWithCopyWarning

    return gdf

def load_and_align_vector_data(path, name, boundary_gdf, layer=None, output_dir="aligned_data"):
    """
    Loads vector data, fixes CRS, clips to boundary, and saves aligned data.

    Args:
        path (str): Path to the vector file.
        name (str): Descriptive name for the data.
        boundary_gdf (gpd.GeoDataFrame): GeoDataFrame of the boundary to clip to.
        layer (str, optional): Specific layer to load from GeoPackage. Defaults to None.
        output_dir (str): Directory to save the aligned data.

    Returns:
        gpd.GeoDataFrame: The aligned and clipped GeoDataFrame.
    """
    if path.endswith(".gpkg"):
        if layer is None:
            layers = fiona.listlayers(path)
            print(f"Layers in {path}: {layers}. Defaulting to first layer.")
            layer = layers[0]
        gdf = gpd.read_file(path, layer=layer)
    else:
        gdf = gpd.read_file(path)

    print(f"{name} original CRS: {gdf.crs}")

    # Fix missing CRS if needed
    if gdf.crs is None:
        print(f"\u26A0 {name} missing CRS! Assigning {COMMON_CRS}.")
        gdf = gdf.set_crs(COMMON_CRS, allow_override=True)

    # Reproject to common CRS
    if gdf.crs != COMMON_CRS:
        gdf = gdf.to_crs(COMMON_CRS)
        print(f"Reprojected {name} to {COMMON_CRS}")

    # Clip to boundary
    gdf = safe_clip(gdf, boundary_gdf)

    # Save aligned data
    output_path = f"{output_dir}/{name}_4326.geojson"
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved aligned {name} to {output_path}")

    return gdf

def load_and_align_raster_data(path, name, boundary_gdf, output_dir="aligned_data"):
    """
    Loads raster data, reprojects to COMMON_CRS if necessary, clips to boundary, and saves aligned data.
    This function handles large rasters by processing them in chunks (using rasterio.mask.mask).

    Args:
        path (str): Path to the raster file.
        name (str): Descriptive name for the raster data.
        boundary_gdf (gpd.GeoDataFrame): GeoDataFrame of the boundary to clip to.
        output_dir (str): Directory to save the aligned data.

    Returns:
        str: Path to the saved aligned TIFF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing raster: {name}")

    # Load boundary polygon for masking
    # Ensure boundary_gdf is in a geographic CRS (like 4326) for GEE sourced rasters
    boundary_geom_proj = boundary_gdf.to_crs(COMMON_CRS).geometry.unary_union
    # It's better to pass the geometry directly to rasterio.mask.mask
    boundary_json = [json.loads(gpd.GeoSeries([boundary_geom_proj]).to_json())['features'][0]['geometry']]

    with rasterio.open(path) as src:
        src_crs = src.crs
        print(f"Original raster CRS: {src_crs}")

        # Prepare for reprojection if CRS differs from COMMON_CRS
        if src_crs != COMMON_CRS:
            transform, width, height = calculate_default_transform(
                src_crs, COMMON_CRS, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": COMMON_CRS,
                "transform": transform,
                "width": width,
                "height": height
            })
            temp_reprojected_path = os.path.join(output_dir, f"{name}_temp_reprojected.tif")
            print(f"Reprojecting to {temp_reprojected_path}")

            with rasterio.open(temp_reprojected_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=COMMON_CRS,
                        resampling=Resampling.bilinear,
                    )
            src_to_clip = rasterio.open(temp_reprojected_path)
        else:
            src_to_clip = src

        # Clip using mask
        print("Clipping raster to boundary...")
        out_image, out_transform = mask(src_to_clip, boundary_json, crop=True)

        # Update metadata for clipped raster
        out_meta = src_to_clip.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save final clipped raster
        out_path = os.path.join(output_dir, f"{name}_aligned.tif")
        print(f"Saving aligned raster to: {out_path}")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)

        if src_to_clip != src: # Close temp file if it was created
            src_to_clip.close()
            os.remove(temp_reprojected_path)

    return out_path

def clean_and_impute_feature_matrix(df):
    """
    Identifies numerical columns with missing values and performs median imputation.

    Args:
        df (pd.DataFrame): The DataFrame to clean and impute.

    Returns:
        pd.DataFrame: The DataFrame with missing numerical values imputed.
    """
    # Identify numerical columns with missing values
    numerical_cols_with_missing = df.select_dtypes(include=np.number).columns[
        df.select_dtypes(include=np.number).isnull().any()
    ].tolist()

    if numerical_cols_with_missing:
        print("Applying median imputation for the following columns:")
        for col in numerical_cols_with_missing:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"  - '{col}' imputed with median value: {median_value}")
        print("Successfully imputed all missing numerical values.")
    else:
        print("No numerical columns with missing values found, no imputation needed.")

    return df

def filter_osm_features(gdf, name, relevant_tags, output_dir="aligned_data"):
    """
    Filters OSM GeoDataFrame to include only specified relevant features and saves it.

    Args:
        gdf (gpd.GeoDataFrame): The OSM GeoDataFrame to filter.
        name (str): Descriptive name for the data.
        relevant_tags (list): List of OSM tag values to keep (e.g., ['school', 'hospital']).
        output_dir (str): Directory to save the filtered data.

    Returns:
        gpd.GeoDataFrame: The filtered GeoDataFrame.
    """
    # Detect category column
    field = None
    for col in ["amenity", "fclass", "type", "category", "shop"]:
        if col in gdf.columns:
            field = col
            break

    if field is None:
        print(f"\u26A0 No OSM category field detected for {name}. Returning original GDF.")
        return gdf

    filtered_gdf = gdf[gdf[field].astype(str).str.lower().isin(relevant_tags)].copy()
    output_path = f"{output_dir}/{name}_filtered_4326.geojson"
    filtered_gdf.to_file(output_path, driver="GeoJSON")
    print(f"Filtered {name} features saved to {output_path}")
    return filtered_gdf

