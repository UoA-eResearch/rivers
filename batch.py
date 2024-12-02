#!/usr/bin/env python3

import rasterio
import requests
from rasterio.features import shapes, sieve, rasterize
from rasterio.plot import show
import geopandas as gpd
import os
import numpy as np
from tqdm.auto import tqdm
from shapely.geometry import box, shape
from rasterio.mask import mask
from matplotlib import pyplot as plt
import xdem
import time
import pandas as pd
tqdm.pandas()

start = time.time()
REC = gpd.read_file("nzRec2_v5.gdb", engine='pyogrio', use_arrow=True)
all_streams = REC.union_all()
LCDB = gpd.read_file("lcdb-v50-land-cover-database-version-50-mainland-new-zealand.gpkg", engine='pyogrio', use_arrow=True)
print(f"Loaded in {time.time() - start:.2f} seconds")

manifest = pd.json_normalize(requests.get("https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2023/dem_1m/2193/collection.json").json()["links"])
manifest = manifest[manifest["rel"] == "item"]
manifest["tilename"] = manifest.href.str.replace(".json", "").str.strip("./")

def get_features(row):
    geom = row.geometry
    masked_old, transform = mask(old, [geom], nodata=np.nan)
    nonan = np.argwhere(~np.isnan(masked_old[0]))
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

    if any(LCDB.intersects(geom)):
        row.update(LCDB.loc[LCDB.intersects(geom), ["Class_2018", "Wetland_18", "Onshore_18"]].mode().iloc[0].replace({"no": False, "yes": True}))

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

for tilename in tqdm(manifest.tilename):
    if os.path.exists(f"{tilename}.parquet"):
        continue
    try:
        old = rasterio.open(f"https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2018-2020/dem_1m/2193/{tilename}.tiff")
        new = rasterio.open(f"https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2023/dem_1m/2193/{tilename}.tiff")
        diff = new.read(1) - old.read(1)
        result = diff.round().clip(min=-1, max=1).astype(np.int16)
        result = sieve(result, 4000)
        result = np.where(result != 0, result, np.nan)
        areas = gpd.GeoDataFrame(geometry=[shape(s) for s, v in shapes(result, transform=new.transform) if not np.isnan(v)])
        areas["area"] = areas.area
        # Not sure why I have to do this again, the sieve above should have done it. Had some very small polygons somehow anyway.
        areas = areas[areas.area > 4000]
        areas.sort_values("area", ascending=False, inplace=True)
        features = areas.progress_apply(get_features, axis=1)
        features.crs = 2193
        features.to_parquet(f"{tilename}.parquet")
    except Exception as e:
        print(f"Failed on {tilename}: {e}")
        continue