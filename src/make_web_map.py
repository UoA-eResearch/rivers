#!/usr/bin/env python3
"""Build the data for the interactive cluster map (``index.html``, assets in ``map/``).

Reads ``data/clusters.parquet`` (written by ``notebooks/clustering.ipynb``) and writes:

``map/clusters.pmtiles``
    Vector tiles (made with tippecanoe) with two layers: ``points`` (polygon
    centroids, for zoomed-out views) and ``polygons``. Each feature carries the
    cluster number and a few summary attributes.
``map/clusters.json``
    Field descriptions, value ranges and per-cluster summaries used by the map's
    legend and cluster profiles.

Requires tippecanoe (``apt install tippecanoe``). Usage::

    python src/make_web_map.py
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd

# key, source column, label, unit, decimals, colour scale (None = popup only), description
FIELDS = [
    ("dz", "diff_mean", "Elevation change", "m", 2, "diverging",
     "Mean elevation change from 2018-2020 to 2023 (negative = erosion, positive = deposition)"),
    ("dz_min", "diff_min", "Largest lowering", "m", 2, None, "Most negative elevation change in the area"),
    ("dz_max", "diff_max", "Largest raising", "m", 2, None, "Most positive elevation change in the area"),
    ("area", "area", "Area", "m²", 0, "sequential", "Area of the polygon"),
    ("elev", "old_mean", "Elevation", "m", 1, "sequential", "Mean elevation in 2018-2020"),
    ("slope", "old_slope_mean", "Slope", "°", 1, "sequential", "Mean slope in 2018-2020"),
    ("prof_curv", "old_profile_curvature_mean", "Profile curvature", "", 2, "diverging",
     "Mean curvature in the direction of the slope, 2018-2020"),
    ("plan_curv", "old_planform_curvature_mean", "Planform curvature", "", 2, "diverging",
     "Mean curvature across the slope, 2018-2020"),
    ("dist_river", "distance_to_river", "Distance to river", "m", 0, "sequential",
     "Distance to the nearest REC2 river line"),
    ("hand", "elevation_above_stream_mean", "Height above stream", "m", 1, "sequential",
     "Mean height above the nearest stream cell"),
    ("dsd", "downslope_dist_mean", "Flow distance to stream", "m", 0, "sequential",
     "Mean distance along the flow path to the nearest stream"),
    ("dtw", "depth_to_water_mean", "Depth to water", "m", 1, "sequential", "Mean cartographic depth to water"),
    ("twi", "twi_mean", "Wetness index (TWI)", "", 2, "sequential", "Mean topographic wetness index"),
    ("spi", "spi_median", "Stream power index", "", 1, "sequential", "Median stream power index"),
    ("ic", "connectivity_index_mean", "Connectivity index", "", 2, "sequential",
     "Mean Borselli index of sediment connectivity (higher = better connected to streams)"),
    ("ndvi", "NDVI_old_mean", "NDVI 2018", "", 2, "sequential", "Mean NDVI (vegetation greenness) in 2018"),
    ("dndvi", "NDVI_diff_mean", "NDVI change", "", 2, "diverging", "Mean change in NDVI from 2018 to 2023"),
]

LCDB_CLASSES = {
    1: "Built-up Area (settlement)", 2: "Urban Parkland/Open Space", 5: "Transport Infrastructure",
    6: "Surface Mine or Dump", 10: "Sand or Gravel", 12: "Landslide", 14: "Permanent Snow and Ice",
    15: "Alpine Grass/Herbfield", 16: "Gravel or Rock", 20: "Lake or Pond", 21: "River",
    22: "Estuarine Open Water", 30: "Short-rotation Cropland", 33: "Orchard, Vineyard or Other Perennial Crop",
    40: "High Producing Exotic Grassland", 41: "Low Producing Grassland", 43: "Tall Tussock Grassland",
    44: "Depleted Grassland", 45: "Herbaceous Freshwater Vegetation", 46: "Herbaceous Saline Vegetation",
    47: "Flaxland", 50: "Fernland", 51: "Gorse and/or Broom", 52: "Manuka and/or Kanuka",
    54: "Broadleaved Indigenous Hardwoods", 55: "Sub Alpine Shrubland", 56: "Mixed Exotic Shrubland",
    58: "Matagouri or Grey Scrub", 64: "Forest - Harvested", 68: "Deciduous Hardwoods",
    69: "Indigenous Forest", 70: "Mangrove", 71: "Exotic Forest",
}

SIMPLIFY_TOLERANCE = 2.0  # m; removes the 1 m pixel staircase (tippecanoe simplifies further per zoom)
# Centroids are tiled up to POINTS_MAX_ZOOM and polygons from POLYGONS_MIN_ZOOM to MAX_ZOOM;
# the map over-zooms beyond MAX_ZOOM (z12 tiles have ~2.4 m precision, ample for polygons of
# at least 4000 m2) which keeps the tiles to ~18 MB.
POINTS_MAX_ZOOM = 11
POLYGONS_MIN_ZOOM = 11
MAX_ZOOM = 12


def clean(value):
    """JSON-safe scalar (NaN -> None, numpy -> Python)."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_attributes(df, fields):
    attrs = pd.DataFrame({"id": np.arange(len(df)), "cluster": df["cluster"].astype(int).values})
    for key, column, _, _, decimals, _, _ in fields:
        attrs[key] = df[column].astype(float).round(decimals).values
    if "Class_2018" in df:
        attrs["lcdb"] = pd.to_numeric(df["Class_2018"], errors="coerce").values
    if "in_river" in df:
        attrs["in_river"] = df["in_river"].astype("boolean").fillna(False).astype(int).values
    return attrs


def write_features(path, geometries, attrs):
    """Write newline-delimited GeoJSON."""
    records = attrs.to_dict(orient="records")
    with open(path, "w") as f:
        for geom, props in zip(geometries, records):
            props = {k: clean(v) for k, v in props.items() if clean(v) is not None}
            feature = {"type": "Feature", "properties": props, "geometry": geom}
            f.write(json.dumps(feature, separators=(",", ":")) + "\n")


def summarise(df, attrs, fields, wgs84):
    """Value ranges per field and summaries per cluster for the map's legend and profiles."""
    field_meta = []
    for key, column, label, unit, decimals, scale, description in fields:
        values = attrs[key].dropna()
        q = np.percentile(values, [2, 25, 50, 75, 98]) if len(values) else [None] * 5
        field_meta.append(dict(
            key=key, column=column, label=label, unit=unit, decimals=decimals, scale=scale,
            description=description, min=clean(values.min()), max=clean(values.max()),
            p02=clean(q[0]), p25=clean(q[1]), p50=clean(q[2]), p75=clean(q[3]), p98=clean(q[4]),
        ))

    clusters = []
    for c, group in attrs.groupby("cluster"):
        idx = group.index
        bounds = wgs84.iloc[idx].total_bounds
        summary = dict(
            id=int(c), count=int(len(group)), area_ha=round(float(group["area"].sum()) / 1e4, 1),
            erosion_share=round(float((group["dz"] < 0).mean()), 3),
            bounds=[round(float(b), 5) for b in bounds],
            median={}, percentile={},
        )
        for key, *_ in fields:
            all_values = attrs[key].dropna().values
            med = group[key].median()
            summary["median"][key] = clean(med)
            # Where the cluster's median sits among all polygons (0-100)
            summary["percentile"][key] = (
                round(100 * float(np.mean(all_values <= med)), 1) if np.isfinite(med) and len(all_values) else None
            )
        if "lcdb" in group:
            shares = group["lcdb"].value_counts(normalize=True).head(3)
            summary["land_cover"] = [[int(k), round(float(v), 3)] for k, v in shares.items()]
        if "in_river" in group:
            summary["in_river_share"] = clean(group["in_river"].mean())
        clusters.append(summary)

    return dict(
        generated=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        count=int(len(attrs)),
        bounds=[round(float(b), 5) for b in wgs84.total_bounds],
        fields=field_meta,
        clusters=clusters,
        lcdb_classes={str(k): v for k, v in LCDB_CLASSES.items()},
    )


def main(source, out_dir):
    if shutil.which("tippecanoe") is None or shutil.which("tile-join") is None:
        raise SystemExit("tippecanoe not found (install with `apt install tippecanoe`)")
    df = gpd.read_parquet(source).reset_index(drop=True)
    fields = [f for f in FIELDS if f[1] in df.columns]
    missing = [f[1] for f in FIELDS if f[1] not in df.columns]
    if missing:
        print(f"Columns not in {source}, left out of the map: {missing}")
    print(f"{len(df)} polygons, {df['cluster'].nunique()} clusters, {len(fields)} fields")

    attrs = build_attributes(df, fields)
    polygons = df.geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True).to_crs(4326)
    points = df.geometry.representative_point().to_crs(4326)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clusters.pmtiles")
    # The two layers cover different zoom ranges, so they are tiled separately and
    # joined. (Per-feature "tippecanoe" zoom ranges made tippecanoe 2.49 drop most
    # polygons.)
    with tempfile.TemporaryDirectory() as tmp:
        poly_path, point_path = os.path.join(tmp, "polygons.geojsonl"), os.path.join(tmp, "points.geojsonl")
        write_features(poly_path, (g.__geo_interface__ for g in polygons), attrs)
        write_features(point_path, (g.__geo_interface__ for g in points), attrs)
        common = ["--force", "--quiet", "--read-parallel", "--no-tile-size-limit", "--no-feature-limit"]
        subprocess.run([
            "tippecanoe", *common, "-o", os.path.join(tmp, "points.pmtiles"), "-l", "points",
            "-Z", "5", "-z", str(POINTS_MAX_ZOOM),
            "--drop-rate=1",  # keep every centroid at every zoom
            point_path,
        ], check=True)
        subprocess.run([
            "tippecanoe", *common, "-o", os.path.join(tmp, "polygons.pmtiles"), "-l", "polygons",
            "-Z", str(POLYGONS_MIN_ZOOM), "-z", str(MAX_ZOOM),
            "--detect-shared-borders",  # adjacent erosion/deposition polygons stay gap-free
            "--no-tiny-polygon-reduction",
            poly_path,
        ], check=True)
        subprocess.run([
            "tile-join", "--force", "--quiet", "--no-tile-size-limit", "-o", out_path,
            "--name", "Gisborne erosion and deposition clusters",
            "--attribution", "Elevation change: LINZ Gisborne LiDAR 2018-2020 and 2023",
            os.path.join(tmp, "points.pmtiles"), os.path.join(tmp, "polygons.pmtiles"),
        ], check=True)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    summary = summarise(df, attrs, fields, polygons)
    with open(os.path.join(out_dir, "clusters.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    print(f"Wrote {os.path.join(out_dir, 'clusters.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="data/clusters.parquet")
    parser.add_argument("--out-dir", default="map")
    args = parser.parse_args()
    main(args.source, args.out_dir)
