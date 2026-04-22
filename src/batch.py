#!/usr/bin/env python3

import rasterio
import requests
from rasterio.features import shapes, sieve, rasterize
import geopandas as gpd
import os
import numpy as np
from tqdm.auto import tqdm
from shapely.geometry import shape, MultiPolygon, Polygon
from rasterio.mask import mask
import xdem
import time
import pandas as pd
from glob import glob
import whitebox
import argparse
tqdm.pandas()

# Configuration
DATA_DIR = "data"
TILE_RESULTS_DIR = "tile_results"
OUTPUT_DIR = "data"
WBT_DIR = "wbt_outputs"

def load_global_data():
    """Load global datasets needed for feature extraction."""
    start = time.time()
    print("Loading REC2 and LCDB datasets...")

    # Extract REC2 if needed
    if not os.path.exists("nzRec2_v5.gdb"):
        import zipfile
        print("Extracting REC2_geodata_version_5.zip...")
        with zipfile.ZipFile(f"{DATA_DIR}/REC2_geodata_version_5.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    # Extract LCDB if needed
    if not os.path.exists("lcdb-v50-land-cover-database-version-50-mainland-new-zealand.gpkg"):
        import zipfile
        print("Extracting LCDB zip...")
        with zipfile.ZipFile(f"{DATA_DIR}/lris-lcdb-v50-land-cover-database-version-50-mainland-new-zealand-GPKG.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    REC = gpd.read_file("nzRec2_v5.gdb", engine='pyogrio', use_arrow=True)
    all_streams = REC.union_all()
    LCDB = gpd.read_file("lcdb-v50-land-cover-database-version-50-mainland-new-zealand.gpkg", engine='pyogrio', use_arrow=True)

    # Load river polygons if needed
    river_polygons = None
    if os.path.exists(f"{DATA_DIR}/lds-nz-river-polygons-topo-150k-GPKG.zip"):
        import zipfile
        if not os.path.exists("nz-river-polygons-topo-150k.gpkg"):
            print("Extracting river polygons...")
            with zipfile.ZipFile(f"{DATA_DIR}/lds-nz-river-polygons-topo-150k-GPKG.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
        # Find the actual gpkg file
        gpkg_files = glob("*.gpkg")
        river_gpkg = [f for f in gpkg_files if "river" in f.lower()]
        if river_gpkg:
            river_polygons = gpd.read_file(river_gpkg[0], engine='pyogrio', use_arrow=True)
            print(f"Loaded river polygons from {river_gpkg[0]}")

    print(f"Loaded in {time.time() - start:.2f} seconds")
    return REC, all_streams, LCDB, river_polygons


def compute_hydrological_indices(dem_path, streams_path, tilename):
    """Compute hydrological indices using WhiteboxTools.

    Parameters
    ----------
    dem_path : str
        Path to local DEM raster file.
    streams_path : str
        Path to streams vector file (e.g. GeoPackage).
    tilename : str
        Tile identifier used to name output files.

    Returns
    -------
    tuple of rasterio.DatasetReader or None
        Opened datasets for (downslope_dist, elev_above_stream, depth_to_water).
    """
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    os.makedirs(WBT_DIR, exist_ok=True)

    # Output paths
    streams_raster = f"{WBT_DIR}/{tilename}_streams.tif"
    downslope_dist = f"{WBT_DIR}/{tilename}_downslope_dist.tif"
    elev_above_stream = f"{WBT_DIR}/{tilename}_elev_above_stream.tif"
    depth_to_water = f"{WBT_DIR}/{tilename}_depth_to_water.tif"

    try:
        # Rasterize streams to match DEM extent/resolution
        if not os.path.exists(streams_raster):
            wbt.rasterize_streams(
                streams=streams_path,
                base=dem_path,
                output=streams_raster
            )

        # DownslopeDistanceToStream
        if not os.path.exists(downslope_dist):
            wbt.downslope_distance_to_stream(
                dem=dem_path,
                streams=streams_raster,
                output=downslope_dist
            )

        # ElevationAboveStreamEuclidean
        if not os.path.exists(elev_above_stream):
            wbt.elevation_above_stream_euclidean(
                dem=dem_path,
                streams=streams_raster,
                output=elev_above_stream
            )

        # DepthToWater
        if not os.path.exists(depth_to_water):
            wbt.depth_to_water(
                dem=dem_path,
                output=depth_to_water
            )

        return (
            rasterio.open(downslope_dist) if os.path.exists(downslope_dist) else None,
            rasterio.open(elev_above_stream) if os.path.exists(elev_above_stream) else None,
            rasterio.open(depth_to_water) if os.path.exists(depth_to_water) else None,
        )
    except Exception as e:
        print(f"WhiteboxTools error: {e}")
        return None, None, None


def split_by_river_polygons(areas, river_polygons):
    """Split polygons based on river polygon boundaries."""
    if river_polygons is None:
        print("No river polygons available, skipping split")
        return areas

    print("Splitting polygons by river boundaries...")
    split_areas = []
    river_sindex = river_polygons.sindex

    for idx, row in tqdm(areas.iterrows(), total=len(areas), desc="Splitting polygons"):
        geom = row.geometry

        # Use spatial index to preselect candidate river polygons
        candidate_idx = river_sindex.query(geom, predicate="intersects")
        if len(candidate_idx) == 0:
            intersecting_rivers = river_polygons.iloc[0:0]
        else:
            intersecting_rivers = river_polygons.iloc[candidate_idx]

        if len(intersecting_rivers) == 0:
            # No intersection, keep original
            row_copy = row.copy()
            row_copy['split_by_river'] = False
            split_areas.append(row_copy)
        else:
            # Split by river polygons
            try:
                # Create difference with all intersecting rivers
                remaining = geom
                for _, river in intersecting_rivers.iterrows():
                    remaining = remaining.difference(river.geometry)

                # Add the split parts
                if not remaining.is_empty:
                    if isinstance(remaining, MultiPolygon):
                        for poly in remaining.geoms:
                            if poly.area > 4000:  # Keep minimum area threshold
                                new_row = row.copy()
                                new_row.geometry = poly
                                new_row['area'] = poly.area
                                new_row['split_by_river'] = True
                                split_areas.append(new_row)
                    elif isinstance(remaining, Polygon) and remaining.area > 4000:
                        new_row = row.copy()
                        new_row.geometry = remaining
                        new_row['area'] = remaining.area
                        new_row['split_by_river'] = True
                        split_areas.append(new_row)
            except Exception as e:
                # If splitting fails, keep original
                print(f"Failed to split polygon {idx}: {e}")
                row_copy = row.copy()
                row_copy['split_by_river'] = False
                split_areas.append(row_copy)

    result = gpd.GeoDataFrame(split_areas, crs=areas.crs)
    print(f"Split {len(areas)} polygons into {len(result)} polygons")
    return result


def get_features(row, old, new, diff, result, all_streams, LCDB, hydro_rasters=None):
    """Extract features for a single polygon."""
    geom = row.geometry
    masked_old, transform = mask(old, [geom], nodata=np.nan)
    nonan = np.argwhere(~np.isnan(masked_old[0]))

    if nonan.size == 0:
        masked_old = np.array([[np.nan]])
        masked_new = np.array([[np.nan]])
        masked_diff = np.array([[np.nan]])
    else:
        top_left = nonan.min(axis=0)
        bottom_right = nonan.max(axis=0)
        masked_old = masked_old[0][top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        masked_new, transform = mask(new, [geom], nodata=np.nan)
        masked_new = masked_new[0][top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        raster = rasterize([geom], out_shape=result.shape, transform=new.transform)
        masked_diff = np.where(raster, diff, np.nan)
        masked_diff = masked_diff[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

    row = row.to_dict()
    row.update({
        "old_min": np.nanmin(masked_old),
        "old_max": np.nanmax(masked_old),
        "old_mean": np.nanmean(masked_old),
        "old_median": np.nanmedian(masked_old),
        "old_std": np.nanstd(masked_old),
        "new_min": np.nanmin(masked_new),
        "new_max": np.nanmax(masked_new),
        "new_mean": np.nanmean(masked_new),
        "new_median": np.nanmedian(masked_new),
        "new_std": np.nanstd(masked_new),
        "diff_min": np.nanmin(masked_diff),
        "diff_max": np.nanmax(masked_diff),
        "diff_mean": np.nanmean(masked_diff),
        "diff_median": np.nanmedian(masked_diff),
        "diff_std": np.nanstd(masked_diff),
        "distance_to_river": geom.distance(all_streams)
    })

    # Add hydrological indices from WhiteboxTools
    if hydro_rasters is not None:
        downslope_dist_raster, elev_above_stream_raster, depth_to_water_raster = hydro_rasters

        try:
            if downslope_dist_raster is not None:
                masked_downslope, _ = mask(downslope_dist_raster, [geom], nodata=np.nan, crop=True)
                row.update({
                    "downslope_dist_mean": np.nanmean(masked_downslope),
                    "downslope_dist_median": np.nanmedian(masked_downslope),
                    "downslope_dist_std": np.nanstd(masked_downslope),
                })

            if elev_above_stream_raster is not None:
                masked_elev_stream, _ = mask(elev_above_stream_raster, [geom], nodata=np.nan, crop=True)
                row.update({
                    "elevation_above_stream_mean": np.nanmean(masked_elev_stream),
                    "elevation_above_stream_median": np.nanmedian(masked_elev_stream),
                    "elevation_above_stream_std": np.nanstd(masked_elev_stream),
                })

            if depth_to_water_raster is not None:
                masked_depth, _ = mask(depth_to_water_raster, [geom], nodata=np.nan, crop=True)
                row.update({
                    "depth_to_water_mean": np.nanmean(masked_depth),
                    "depth_to_water_median": np.nanmedian(masked_depth),
                    "depth_to_water_std": np.nanstd(masked_depth),
                })
        except Exception as e:
            print(f"Error processing hydrological indices: {e}")

    lcdb_candidate_idx = LCDB.sindex.query(geom, predicate="intersects")
    if len(lcdb_candidate_idx) > 0:
        lcdb_candidates = LCDB.iloc[lcdb_candidate_idx]
        lcdb_intersections = lcdb_candidates.intersects(geom)
        if lcdb_intersections.any():
            row.update(
                lcdb_candidates.loc[
                    lcdb_intersections, ["Class_2018", "Wetland_18", "Onshore_18"]
                ].mode().iloc[0].replace({"no": False, "yes": True})
            )

    attribute_names = ["roughness", "slope", "aspect", "curvature", "terrain_ruggedness_index", "rugosity", "profile_curvature", "planform_curvature"]
    diff_attributes = xdem.terrain.get_terrain_attribute(
        masked_diff,
        resolution=old.res,
        attribute=attribute_names
    )
    old_attributes = xdem.terrain.get_terrain_attribute(
        masked_old,
        resolution=old.res,
        attribute=attribute_names
    )
    new_attributes = xdem.terrain.get_terrain_attribute(
        masked_new,
        resolution=old.res,
        attribute=attribute_names
    )

    for i, name in enumerate(attribute_names):
        row.update({
            f"old_{name}_min": np.nanmin(old_attributes[i]),
            f"old_{name}_max": np.nanmax(old_attributes[i]),
            f"old_{name}_mean": np.nanmean(old_attributes[i]),
            f"old_{name}_median": np.nanmedian(old_attributes[i]),
            f"old_{name}_std": np.nanstd(old_attributes[i]),
            f"new_{name}_min": np.nanmin(new_attributes[i]),
            f"new_{name}_max": np.nanmax(new_attributes[i]),
            f"new_{name}_mean": np.nanmean(new_attributes[i]),
            f"new_{name}_median": np.nanmedian(new_attributes[i]),
            f"new_{name}_std": np.nanstd(new_attributes[i]),
            f"diff_{name}_min": np.nanmin(diff_attributes[i]),
            f"diff_{name}_max": np.nanmax(diff_attributes[i]),
            f"diff_{name}_mean": np.nanmean(diff_attributes[i]),
            f"diff_{name}_median": np.nanmedian(diff_attributes[i]),
            f"diff_{name}_std": np.nanstd(diff_attributes[i]),
        })

    return pd.Series(row)


def add_ndvi_features(areas):
    """Add NDVI features to the areas dataframe."""
    print("Adding NDVI features...")

    if not os.path.exists(f"{DATA_DIR}/NDVI_2018.tif") or not os.path.exists(f"{DATA_DIR}/NDVI_2023.tif"):
        print("NDVI rasters not found, skipping NDVI features")
        return areas

    old_ndvi = rasterio.open(f"{DATA_DIR}/NDVI_2018.tif")
    new_ndvi = rasterio.open(f"{DATA_DIR}/NDVI_2023.tif")
    original_crs = areas.crs

    def get_ndvi_features(row):
        geom = row.geometry
        masked_old, transform = mask(old_ndvi, [geom], nodata=np.nan, crop=True)
        masked_new, transform = mask(new_ndvi, [geom], nodata=np.nan, crop=True)
        masked_diff = masked_new - masked_old
        row = row.to_dict()
        row.update({
            "NDVI_old_min": np.nanmin(masked_old),
            "NDVI_old_max": np.nanmax(masked_old),
            "NDVI_old_mean": np.nanmean(masked_old),
            "NDVI_old_median": np.nanmedian(masked_old),
            "NDVI_old_std": np.nanstd(masked_old),
            "NDVI_new_min": np.nanmin(masked_new),
            "NDVI_new_max": np.nanmax(masked_new),
            "NDVI_new_mean": np.nanmean(masked_new),
            "NDVI_new_median": np.nanmedian(masked_new),
            "NDVI_new_std": np.nanstd(masked_new),
            "NDVI_diff_min": np.nanmin(masked_diff),
            "NDVI_diff_max": np.nanmax(masked_diff),
            "NDVI_diff_mean": np.nanmean(masked_diff),
            "NDVI_diff_median": np.nanmedian(masked_diff),
            "NDVI_diff_std": np.nanstd(masked_diff)
        })
        return pd.Series(row)

    areas = areas.progress_apply(get_ndvi_features, axis=1)
    areas = gpd.GeoDataFrame(areas, geometry="geometry", crs=original_crs)
    return areas


def join_parquet_files():
    """Join all individual tile parquet files into a single file."""
    print("Joining parquet files from tile_results/...")

    if not os.path.exists(TILE_RESULTS_DIR):
        print(f"No {TILE_RESULTS_DIR} directory found, skipping join")
        return None

    files = glob(f"{TILE_RESULTS_DIR}/*.parquet")
    if not files:
        print(f"No parquet files found in {TILE_RESULTS_DIR}")
        return None

    print(f"Found {len(files)} parquet files to join")
    areas = pd.concat(gpd.read_parquet(f) for f in tqdm(files))
    print(f"Combined shape: {areas.shape}")

    # Save the combined file
    areas.to_parquet(f"{OUTPUT_DIR}/areas.parquet")
    print(f"Saved combined areas to {OUTPUT_DIR}/areas.parquet")

    return areas


def process_tiles(REC, all_streams, LCDB, river_polygons):
    """Process DEM tiles to extract areas of change."""
    from shapely.geometry import box as shapely_box

    manifest = pd.json_normalize(requests.get("https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2023/dem_1m/2193/collection.json").json()["links"])
    manifest = manifest[manifest["rel"] == "item"]
    manifest["tilename"] = manifest.href.str.replace(".json", "").str.strip("./")

    # Create tile results directory
    os.makedirs(TILE_RESULTS_DIR, exist_ok=True)
    os.makedirs(WBT_DIR, exist_ok=True)

    for tilename in tqdm(manifest.tilename, desc="Processing tiles"):
        output_path = f"{TILE_RESULTS_DIR}/{tilename}.parquet"
        if os.path.exists(output_path):
            continue
        try:
            old = rasterio.open(f"https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2018-2020/dem_1m/2193/{tilename}.tiff")
            new = rasterio.open(f"https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2023/dem_1m/2193/{tilename}.tiff")
            diff = new.read(1) - old.read(1)
            result = diff.round().clip(min=-1, max=1).astype(np.int16)
            result = sieve(result, 4000)
            result = np.where(result != 0, result, np.nan)

            # Create initial polygons
            areas = gpd.GeoDataFrame(geometry=[shape(s) for s, v in shapes(result, transform=new.transform) if not np.isnan(v)])
            areas["area"] = areas.area
            areas = areas[areas.area > 4000]
            areas.sort_values("area", ascending=False, inplace=True)
            areas.crs = 2193

            # SPLIT POLYGONS BEFORE COMPUTING FEATURES
            if river_polygons is not None:
                areas = split_by_river_polygons(areas, river_polygons)

            # Compute hydrological indices using WhiteboxTools
            # Save DEM to local file so WhiteboxTools can process it
            tile_dem_path = f"{WBT_DIR}/{tilename}_dem.tif"
            tile_streams_path = f"{WBT_DIR}/{tilename}_streams.gpkg"
            hydro_rasters = None, None, None

            if not os.path.exists(tile_dem_path):
                with rasterio.open(tile_dem_path, 'w', **new.meta) as dst:
                    dst.write(new.read())

            # Clip REC streams to tile extent and save for rasterization
            if not os.path.exists(tile_streams_path):
                bounds = new.bounds
                tile_bbox = gpd.GeoDataFrame(
                    geometry=[shapely_box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
                    crs=new.crs
                )
                streams_clipped = gpd.clip(REC, tile_bbox)
                if not streams_clipped.empty:
                    streams_clipped.to_file(tile_streams_path, driver='GPKG')

            if os.path.exists(tile_streams_path):
                hydro_rasters = compute_hydrological_indices(tile_dem_path, tile_streams_path, tilename)

            # Now compute features on split polygons
            features = areas.progress_apply(
                lambda row: get_features(row, old, new, diff, result, all_streams, LCDB, hydro_rasters),
                axis=1
            )
            features = gpd.GeoDataFrame(features, geometry="geometry", crs=areas.crs)
            features.to_parquet(output_path)
        except Exception as e:
            print(f"Failed on {tilename}: {e}")
            continue


def main(force_rebuild=False):
    """
    Main processing pipeline.

    Args:
        force_rebuild: If True, rebuild areas.parquet from scratch even if it exists.
                      If False, load existing areas.parquet if available.
    """
    print("Starting batch processing...")

    # Load global datasets
    REC, all_streams, LCDB, river_polygons = load_global_data()

    # Process tiles if needed
    if force_rebuild or not os.path.exists(f"{OUTPUT_DIR}/areas.parquet"):
        if force_rebuild and os.path.exists(f"{OUTPUT_DIR}/areas.parquet"):
            print(f"\nForce rebuild requested - ignoring existing {OUTPUT_DIR}/areas.parquet")
        print("\nProcessing DEM tiles...")
        process_tiles(REC, all_streams, LCDB, river_polygons)

        # Join results
        areas = join_parquet_files()
    else:
        print(f"Loading existing areas from {OUTPUT_DIR}/areas.parquet")
        areas = gpd.read_parquet(f"{OUTPUT_DIR}/areas.parquet")

    if areas is not None:
        # Add NDVI features
        areas_with_ndvi = add_ndvi_features(areas)
        areas_with_ndvi.to_parquet(f"{OUTPUT_DIR}/areas+NDVI.parquet")
        print(f"Saved areas with NDVI to {OUTPUT_DIR}/areas+NDVI.parquet")

    print("\nBatch processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process river data from DEM tiles.')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of areas.parquet from scratch (default: load existing if available)')
    args = parser.parse_args()
    main(force_rebuild=args.force_rebuild)
