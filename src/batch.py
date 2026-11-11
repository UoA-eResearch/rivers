#!/usr/bin/env python3
"""Feature extraction for areas of elevation change in the Gisborne 1 m DEMs.

The pipeline has three stages, which can be run together (the default) or
separately so that the tile stage can be spread over several machines:

``prepare``
    Index the DEM tiles of both epochs (2018-2020 and 2023), write a GDAL VRT
    for each epoch, and clip REC2 streams, LCDB land cover and LINZ river
    polygons to the study region (written to ``prepared/``).
``tiles``
    For each DEM tile: detect polygons of elevation change, split them at river
    polygon boundaries, then compute per-polygon features (elevation, terrain,
    hydrological and land-cover attributes). One GeoParquet file per tile is
    written to ``tile_results/``.
``finalize``
    Join the tile results into ``data/areas.parquet`` and add NDVI features
    (``data/areas+NDVI.parquet``).

Usage::

    python src/batch.py                         # all stages
    python src/batch.py --force-rebuild         # rebuild areas.parquet even if it exists
    python src/batch.py --stage tiles --shard 0 --num-shards 20
    python src/batch.py --stage tiles --tiles BD44_10000_0503
"""

import argparse
import math
import os
import shutil
import sys
import time
import traceback
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob

# GDAL settings for reading cloud-optimised GeoTIFFs over HTTP. Set as
# environment variables so they also apply in worker threads.
for _key, _value in {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff,.tif,.vrt",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}.items():
    os.environ.setdefault(_key, _value)

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import requests  # noqa: E402
import whitebox  # noqa: E402
import xdem  # noqa: E402
from rasterio.features import rasterize, shapes, sieve  # noqa: E402
from rasterio.mask import mask  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from scipy import ndimage  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

tqdm.pandas()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "data"
TILE_RESULTS_DIR = "tile_results"
PREPARED_DIR = "prepared"
WBT_DIR = "wbt_outputs"

DEM_COLLECTIONS = {
    "old": "https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2018-2020/dem_1m/2193",
    "new": "https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2023/dem_1m/2193",
}
CRS = "EPSG:2193"

REC_ZIP = f"{DATA_DIR}/REC2_geodata_version_5.zip"
REC_GDB = "nzRec2_v5.gdb"
LCDB_ZIP = f"{DATA_DIR}/lris-lcdb-v50-land-cover-database-version-50-mainland-new-zealand-GPKG.zip"
LCDB_GPKG = "lcdb-v50-land-cover-database-version-50-mainland-new-zealand.gpkg"
RIVERS_ZIP = f"{DATA_DIR}/lds-nz-river-polygons-topo-150k-GPKG.zip"
RIVERS_GPKG = "nz-river-polygons-topo-150k.gpkg"
LCDB_COLUMNS = ["Class_2018", "Wetland_18", "Onshore_18"]

# Change detection
MIN_AREA = 4000  # m^2; minimum polygon area, also the sieve size (in 1 m pixels)
STATS = ("min", "max", "mean", "median", "std")

# Terrain attributes (xdem), computed for the old DEM, new DEM and their difference
TERRAIN_ATTRIBUTES = [
    "roughness", "slope", "aspect", "curvature", "terrain_ruggedness_index",
    "rugosity", "profile_curvature", "planform_curvature",
]
TERRAIN_MARGIN = 3  # pixels read around each tile so moving-window attributes have no edge NaNs

# Hydrology (WhiteboxTools)
HYDRO_EPOCH = "old"  # DEM used for flow routing: the pre-change (2018-2020) terrain
HYDRO_BUFFER = 500  # m of neighbouring tiles loaded so flow paths are not cut at tile edges
STREAM_MIN_CATCHMENT = 1e4  # m^2; D8 contributing area for a DEM cell to count as a channel
STREAM_SNAP_DISTANCE = 50  # m; DEM channels within this distance of a REC2 line are streams
COAST_MAX_ELEVATION = 5  # m; DEM cells this low next to NoData (the sea) count as shoreline
SEA_MAX_ELEVATION = 0  # m; areas this low connected to NoData are sea (some DEMs have a flattened sea surface)
HYDRO_FEATURES = [
    "downslope_dist", "elevation_above_stream", "depth_to_water",
    "twi", "spi", "connectivity_index",
]
NODATA = -9999.0

# Borselli et al. (2008) connectivity index. The weighting factor W is the
# RUSLE cover-management factor (C) of the LCDB 2018 land-cover class. Values
# are indicative, loosely based on published C-factor ranges for comparable
# land covers (e.g. Panagos et al. 2015); adjust them for your own study.
# Sealed surfaces are given a moderate value because they transmit runoff
# and sediment efficiently even though they do not erode, and open water is
# given 1 (no impedance).
LCDB_C_FACTOR = {
    1: 0.1,     # Built-up Area (settlement)
    2: 0.05,    # Urban Parkland/Open Space
    5: 0.1,     # Transport Infrastructure
    6: 0.5,     # Surface Mine or Dump
    10: 0.5,    # Sand or Gravel
    12: 0.5,    # Landslide
    14: 0.001,  # Permanent Snow and Ice
    15: 0.04,   # Alpine Grass/Herbfield
    16: 0.1,    # Gravel or Rock
    20: 1.0,    # Lake or Pond
    21: 1.0,    # River
    22: 1.0,    # Estuarine Open Water
    30: 0.25,   # Short-rotation Cropland
    33: 0.2,    # Orchard, Vineyard or Other Perennial Crop
    40: 0.05,   # High Producing Exotic Grassland
    41: 0.08,   # Low Producing Grassland
    43: 0.03,   # Tall Tussock Grassland
    44: 0.2,    # Depleted Grassland
    45: 0.01,   # Herbaceous Freshwater Vegetation
    46: 0.01,   # Herbaceous Saline Vegetation
    47: 0.02,   # Flaxland
    50: 0.02,   # Fernland
    51: 0.02,   # Gorse and/or Broom
    52: 0.01,   # Manuka and/or Kanuka
    54: 0.005,  # Broadleaved Indigenous Hardwoods
    55: 0.02,   # Sub Alpine Shrubland
    56: 0.02,   # Mixed Exotic Shrubland
    58: 0.03,   # Matagouri or Grey Scrub
    64: 0.15,   # Forest - Harvested
    68: 0.003,  # Deciduous Hardwoods
    69: 0.001,  # Indigenous Forest
    70: 0.001,  # Mangrove
    71: 0.003,  # Exotic Forest
}
DEFAULT_C_FACTOR = 0.05  # cells not covered by LCDB
IC_SLOPE_BOUNDS = (0.005, 1.0)  # m/m; slope limits for the IC (Cavalli et al. 2013)

# WhiteboxTools D8 pointer codes -> (row, col) offsets
D8_OFFSETS = {
    1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
    16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def extract(zip_path, member):
    """Extract ``zip_path`` into the working directory unless ``member`` exists."""
    if not os.path.exists(member):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(".")
    return member


def grid_bounds(array_shape, transform):
    """(left, bottom, right, top) of a north-up raster grid."""
    left, top = transform.c, transform.f
    return (left, top + array_shape[0] * transform.e, left + array_shape[1] * transform.a, top)


def expand(bounds, distance):
    left, bottom, right, top = bounds
    return (left - distance, bottom - distance, right + distance, top + distance)


def write_raster(path, array, transform, nodata=NODATA):
    """Write a float32 GeoTIFF that WhiteboxTools can read (uncompressed)."""
    array = np.where(np.isfinite(array), array, nodata).astype("float32")
    profile = dict(driver="GTiff", width=array.shape[1], height=array.shape[0], count=1,
                   dtype="float32", crs=CRS, transform=transform, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def read_raster(path):
    """Read band 1 as float32 with NoData converted to NaN."""
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan)


_WBT = None


def wbt(tool, **kwargs):
    """Run a WhiteboxTools tool and raise if it did not produce its output.

    The ``whitebox`` Python wrapper returns 0 even when the tool fails, so
    failures are detected by checking that the output file was written.
    """
    global _WBT
    if _WBT is None:
        _WBT = whitebox.WhiteboxTools()
        _WBT.set_verbose_mode(True)  # messages go to the callback below, not stdout
        _WBT.set_compress_rasters(False)
    output = kwargs.get("output") or kwargs.get("out_accum")
    if os.path.exists(output):
        os.remove(output)
    messages = []
    getattr(_WBT, tool)(callback=messages.append, **kwargs)
    if not os.path.exists(output):
        raise RuntimeError(f"WhiteboxTools {tool} failed:\n" + "\n".join(messages[-15:]))
    return output


class Zones:
    """Per-polygon statistics of rasters, given a raster of polygon labels (1..n)."""

    def __init__(self, labels, n):
        self.n = n
        self.index = np.flatnonzero(labels)
        self.labels = labels.ravel()[self.index].astype(np.int64)

    def stats(self, values, prefix):
        out = {f"{prefix}_{s}": np.full(self.n, np.nan) for s in STATS}
        vals = values.ravel()[self.index].astype(np.float64)
        ok = np.isfinite(vals)
        vals, labs = vals[ok], self.labels[ok]
        if vals.size == 0:
            return out
        order = np.lexsort((vals, labs))
        vals, labs = vals[order], labs[order]
        counts = np.bincount(labs, minlength=self.n + 1)[1:]
        has = counts > 0
        ends = np.cumsum(counts)
        starts = ends - counts
        mean = np.bincount(labs, weights=vals, minlength=self.n + 1)[1:] / np.maximum(counts, 1)
        dev = vals - mean[labs - 1]
        var = np.bincount(labs, weights=dev * dev, minlength=self.n + 1)[1:] / np.maximum(counts, 1)
        c, s = counts[has], starts[has]
        out[f"{prefix}_min"][has] = vals[s]
        out[f"{prefix}_max"][has] = vals[s + c - 1]
        out[f"{prefix}_mean"][has] = mean[has]
        out[f"{prefix}_median"][has] = (vals[s + (c - 1) // 2] + vals[s + c // 2]) / 2
        out[f"{prefix}_std"][has] = np.sqrt(var[has])
        return out


# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------
def list_tiles(epoch):
    url = DEM_COLLECTIONS[epoch]
    links = requests.get(f"{url}/collection.json", timeout=60).json()["links"]
    return sorted(link["href"].replace(".json", "").strip("./") for link in links if link["rel"] == "item")


def build_tile_index(epoch):
    """Read the header of every DEM tile of an epoch to get its exact grid."""
    url = DEM_COLLECTIONS[epoch]

    def info(name):
        with rasterio.open(f"{url}/{name}.tiff") as src:
            b = src.bounds
            return dict(tile=name, url=f"{url}/{name}.tiff", left=b.left, bottom=b.bottom,
                        right=b.right, top=b.top, width=src.width, height=src.height,
                        res=src.res[0], nodata=src.nodata)

    names = list_tiles(epoch)
    with ThreadPoolExecutor(16) as ex:
        rows = list(tqdm(ex.map(info, names), total=len(names), desc=f"Indexing {epoch} DEM tiles"))
    return pd.DataFrame(rows)


def write_vrt(index, path):
    """Mosaic all tiles of an epoch into one VRT so any window can be read."""
    res = float(index.res.iloc[0])
    nodata = index.nodata.iloc[0]
    minx, maxy = index.left.min(), index.top.max()
    width = int(round((index.right.max() - minx) / res))
    height = int(round((maxy - index.bottom.min()) / res))
    sources = []
    for t in index.itertuples():
        xoff = int(round((t.left - minx) / res))
        yoff = int(round((maxy - t.top) / res))
        sources.append(
            f'<ComplexSource><SourceFilename relativeToVRT="0">/vsicurl/{t.url}</SourceFilename>'
            f'<SourceBand>1</SourceBand>'
            f'<SourceProperties RasterXSize="{t.width}" RasterYSize="{t.height}" DataType="Float32" '
            f'BlockXSize="512" BlockYSize="512"/>'
            f'<SrcRect xOff="0" yOff="0" xSize="{t.width}" ySize="{t.height}"/>'
            f'<DstRect xOff="{xoff}" yOff="{yoff}" xSize="{t.width}" ySize="{t.height}"/>'
            f'<NODATA>{nodata}</NODATA></ComplexSource>'
        )
    xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f'<SRS>{CRS}</SRS><GeoTransform>{minx}, {res}, 0, {maxy}, 0, {-res}</GeoTransform>'
        f'<VRTRasterBand dataType="Float32" band="1"><NoDataValue>{nodata}</NoDataValue>'
        + "\n".join(sources)
        + "</VRTRasterBand></VRTDataset>"
    )
    with open(path, "w") as f:
        f.write(xml)


def prepare(force=False):
    """Build the tile indexes/VRTs and region-clipped reference layers."""
    os.makedirs(PREPARED_DIR, exist_ok=True)
    indexes = {}
    for epoch in DEM_COLLECTIONS:
        index_path = f"{PREPARED_DIR}/tiles_{epoch}.csv"
        if force or not os.path.exists(index_path):
            build_tile_index(epoch).to_csv(index_path, index=False)
        indexes[epoch] = pd.read_csv(index_path)
        write_vrt(indexes[epoch], f"{PREPARED_DIR}/dem_{epoch}.vrt")

    tiles = pd.concat(indexes.values())
    region = expand((tiles.left.min(), tiles.bottom.min(), tiles.right.max(), tiles.top.max()),
                    HYDRO_BUFFER + STREAM_SNAP_DISTANCE + 1000)

    layers = {
        "rec2": (REC_ZIP, REC_GDB, dict(layer="riverlines", columns=["nzsegment", "StreamOrde", "CUM_AREA"])),
        "lcdb": (LCDB_ZIP, LCDB_GPKG, dict(columns=LCDB_COLUMNS)),
        "river_polygons": (RIVERS_ZIP, RIVERS_GPKG, dict(columns=[])),
    }
    for name, (zip_path, member, kwargs) in layers.items():
        out_path = f"{PREPARED_DIR}/{name}.parquet"
        if os.path.exists(out_path) and not force:
            continue
        if not os.path.exists(zip_path):
            print(f"{zip_path} not found, skipping {name}")
            continue
        start = time.time()
        gdf = gpd.read_file(extract(zip_path, member), bbox=region, engine="pyogrio", **kwargs)
        gdf = gdf.to_crs(CRS)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
        gdf["geometry"] = gdf.geometry.make_valid()
        gdf.to_parquet(out_path)
        print(f"Prepared {name}: {len(gdf)} features in {time.time() - start:.1f}s")


def load_references():
    refs = {}
    for name in ("rec2", "lcdb", "river_polygons"):
        path = f"{PREPARED_DIR}/{name}.parquet"
        refs[name] = gpd.read_parquet(path) if os.path.exists(path) else None
        if refs[name] is None:
            print(f"Warning: {path} not found; features depending on it will be missing")
    if refs["lcdb"] is not None:
        refs["lcdb"]["c_factor"] = (
            pd.to_numeric(refs["lcdb"]["Class_2018"], errors="coerce").map(LCDB_C_FACTOR).fillna(DEFAULT_C_FACTOR)
        )
    for epoch in DEM_COLLECTIONS:
        refs[f"tiles_{epoch}"] = pd.read_csv(f"{PREPARED_DIR}/tiles_{epoch}.csv").set_index("tile")
        refs[f"vrt_{epoch}"] = f"{PREPARED_DIR}/dem_{epoch}.vrt"
    return refs


# ---------------------------------------------------------------------------
# Stage 2: tiles
# ---------------------------------------------------------------------------
def read_dem_window(vrt_path, bounds, res):
    """Read a window of a DEM mosaic; cells outside the data are NaN."""
    with rasterio.open(vrt_path) as src:
        col = int(round((bounds[0] - src.transform.c) / res))
        row = int(round((src.transform.f - bounds[3]) / res))
        width = int(round((bounds[2] - bounds[0]) / res))
        height = int(round((bounds[3] - bounds[1]) / res))
        window = Window(col, row, width, height)
        data = src.read(1, window=window, boundless=True, masked=True)
    transform = from_origin(bounds[0], bounds[3], res, res)
    return data.astype("float32").filled(np.nan), transform


def detect_change_polygons(diff, transform, res):
    """Polygons where the DEM difference rounds to a non-zero number of metres."""
    change = np.nan_to_num(diff, nan=0.0).round().clip(-1, 1).astype(np.int16)
    change = sieve(change, int(MIN_AREA / res ** 2))
    geoms = [shape(s) for s, _ in shapes(change, mask=change != 0, transform=transform)]
    areas = gpd.GeoDataFrame(geometry=geoms, crs=CRS)
    areas["area"] = areas.area
    # Sieving should already have removed small regions, but a few slip through.
    areas = areas[areas.area > MIN_AREA]
    return areas.sort_values("area", ascending=False).reset_index(drop=True)


def split_by_river_polygons(areas, river_polygons):
    """Split change polygons at river polygon boundaries.

    Both the parts inside and outside river polygons are kept (``in_river``
    tells them apart); ``split_by_river`` is True for parts of polygons that
    straddled a river boundary. Parts smaller than ``MIN_AREA`` are dropped.
    """
    areas = areas.copy()
    areas["in_river"] = False
    areas["split_by_river"] = False
    if river_polygons is None or areas.empty:
        return areas
    candidates = river_polygons.iloc[
        river_polygons.sindex.query(box(*areas.total_bounds), predicate="intersects")
    ]
    if candidates.empty:
        return areas
    rivers = candidates.union_all()
    inside = areas.geometry.intersection(rivers)
    outside = areas.geometry.difference(rivers)
    straddles = (inside.area > 0) & (outside.area > 0)

    parts = []
    for in_river, geoms in ((True, inside), (False, outside)):
        part = areas.copy()
        part["geometry"] = geoms
        part["in_river"] = in_river
        part["split_by_river"] = straddles
        parts.append(part)
    result = pd.concat(parts).explode(index_parts=False)
    result = result[result.geom_type == "Polygon"]
    result["area"] = result.area
    result = result[result.area > MIN_AREA]
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=areas.crs)
    return result.sort_values("area", ascending=False).reset_index(drop=True)


def stream_mask(dem, transform, refs, workdir, acc):
    """Stream/water cells: REC2 streams snapped to DEM channels, river polygons and the shoreline.

    REC2 lines are offset from the channels in the 1 m lidar DEM (median ~10 m
    in testing), so the REC2 lines are rasterised with ``RasterizeStreams`` and
    then replaced by the DEM channel cells (D8 contributing area of at least
    ``STREAM_MIN_CATCHMENT``) that lie within ``STREAM_SNAP_DISTANCE`` of them.
    The sea is NoData in the DEMs, or a flattened surface just below 0 m
    between the shore and NoData, so low-lying cells next to NoData and low
    areas connected to it are added as sea; otherwise land draining straight
    to the sea has no target.
    """
    res = transform.a
    bounds = grid_bounds(dem.shape, transform)
    water = np.zeros(dem.shape, dtype=bool)

    rec = refs["rec2"]
    if rec is not None:
        rec = rec.iloc[rec.sindex.query(box(*bounds), predicate="intersects")]
    if rec is not None and not rec.empty:
        shp = os.path.join(workdir, "rec2.shp")
        rec[["geometry"]].explode(index_parts=False).to_file(shp)
        wbt("rasterize_streams", streams=shp, base=os.path.join(workdir, "dem.tif"),
            output=os.path.join(workdir, "rec2.tif"), nodata=False)
        wbt("euclidean_distance", i=os.path.join(workdir, "rec2.tif"),
            output=os.path.join(workdir, "rec2_distance.tif"))
        near_rec = read_raster(os.path.join(workdir, "rec2_distance.tif")) <= STREAM_SNAP_DISTANCE
        water |= near_rec & (acc * res * res >= STREAM_MIN_CATCHMENT)

    rivers = refs["river_polygons"]
    if rivers is not None:
        rivers = rivers.iloc[rivers.sindex.query(box(*bounds), predicate="intersects")]
        if not rivers.empty:
            water |= rasterize(rivers.geometry, out_shape=dem.shape, transform=transform,
                               fill=0, default_value=1, dtype="uint8").astype(bool)

    nodata = ~np.isfinite(dem)
    if nodata.any():
        eight = np.ones((3, 3), dtype=bool)
        shore = ndimage.binary_dilation(nodata, structure=eight) & ~nodata
        with np.errstate(invalid="ignore"):
            water |= shore & (dem <= COAST_MAX_ELEVATION)
            low = dem <= SEA_MAX_ELEVATION
        regions, n = ndimage.label(low, structure=eight)
        is_sea = np.zeros(n + 1, dtype=bool)
        is_sea[regions[shore & low]] = True
        is_sea[0] = False
        water |= is_sea[regions]
    return water & ~nodata


def index_dtype(n):
    return np.int32 if n < 2 ** 31 else np.int64


def d8_receivers(pointer, valid, res):
    """Flat index of each cell's D8 receiver (-1 if none) and its flow step length."""
    nrows, ncols = pointer.shape
    receivers = np.full(pointer.size, -1, dtype=index_dtype(pointer.size))
    step = np.full(pointer.size, res, dtype=np.float32)
    for code, (dr, dc) in D8_OFFSETS.items():
        r, c = np.nonzero(pointer == code)
        rr, cc = r + dr, c + dc
        ok = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
        r, c, rr, cc = r[ok], c[ok], rr[ok], cc[ok]
        ok = valid[rr, cc]
        src = r[ok] * ncols + c[ok]
        receivers[src] = rr[ok] * ncols + cc[ok]
        if dr and dc:
            step[src] = res * math.sqrt(2)
    return receivers, step


def downstream_sum(receivers, cost, target):
    """Sum ``cost`` along each cell's D8 flow path, stopping at the first target cell.

    Returns the sum including the starting cell and the target cell's own cost
    (so targets get their own cost), or NaN where the flow path leaves the grid
    or ends in a pit before reaching a target. Uses pointer jumping, so it
    needs only O(log path length) vectorised passes.
    """
    dtype = index_dtype(receivers.size)
    nxt = np.where(target, -1, receivers).astype(dtype)
    total = cost.astype(np.float64)
    end = np.arange(receivers.size, dtype=dtype)
    active = np.flatnonzero(nxt >= 0).astype(dtype)
    for _ in range(64):
        if active.size == 0:
            break
        jump = nxt[active]
        total[active] += total[jump]
        end[active] = end[jump]
        nxt[active] = nxt[jump]
        active = active[nxt[active] >= 0]
    else:
        raise RuntimeError("D8 flow paths did not terminate (cycle in pointer raster?)")
    return np.where(target[end], total, np.nan)


def connectivity_index(workdir, transform, slope_deg, water, c_factor, valid, crop):
    """Borselli et al. (2008) index of sediment connectivity, IC = log10(D_up / D_dn).

    D_up = W_mean * S_mean * sqrt(A), with W_mean and S_mean averaged over the
    D8 upslope contributing area A; D_dn = sum(d_i / (W_i * S_i)) along the D8
    flow path to the nearest stream/water cell. W is the land-cover C-factor
    and S the slope gradient (m/m), bounded as in Cavalli et al. (2013).
    """
    res = transform.a
    p = lambda name: os.path.join(workdir, f"{name}.tif")  # noqa: E731
    S = np.clip(np.tan(np.radians(slope_deg)), *IC_SLOPE_BOUNDS).astype("float32")
    W = c_factor.astype("float32")
    write_raster(p("ic_w"), np.where(valid, W, np.nan), transform)
    write_raster(p("ic_s"), np.where(valid, S, np.nan), transform)
    write_raster(p("ones"), np.where(valid, 1.0, np.nan), transform)
    write_raster(p("zeros"), np.where(valid, 0.0, np.nan), transform)
    for name in ("w", "s"):
        wbt("d8_mass_flux", dem=p("dem_cond"), loading=p(f"ic_{name}"), efficiency=p("ones"),
            absorption=p("zeros"), output=p(f"ic_sum_{name}"))
    cells = read_raster(p("acc"))[crop]
    w_mean = read_raster(p("ic_sum_w"))[crop] / cells
    s_mean = read_raster(p("ic_sum_s"))[crop] / cells
    d_up = w_mean * s_mean * np.sqrt(cells * res * res)
    del w_mean, s_mean, cells

    receivers, step = d8_receivers(read_raster(p("d8")), valid, res)
    cost = step / (W.ravel() * S.ravel())
    del step, W, S
    d_dn = downstream_sum(receivers, cost, water.ravel()).reshape(water.shape)[crop]
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.log10(d_up / d_dn)
    return np.where(valid[crop], ic, np.nan).astype("float32")


def compute_hydrology(dem, transform, refs, workdir, crop):
    """Hydrological indices on a (buffered) DEM, using WhiteboxTools where possible.

    Returns a dict of float32 arrays, cropped to ``crop`` (a pair of slices
    selecting the tile from the buffered grid).
    """
    os.makedirs(workdir, exist_ok=True)
    p = lambda name: os.path.join(workdir, f"{name}.tif")  # noqa: E731
    valid = np.isfinite(dem)
    write_raster(p("dem"), dem, transform)

    # Hydrologically condition the DEM (required by flow-routing tools).
    # BreachDepressionsLeastCost was tried but took >5 minutes per tile on the
    # flat Poverty Bay floodplain (and is non-deterministic when multi-threaded);
    # BreachDepressions takes ~20 s, is deterministic and drains every cell.
    wbt("breach_depressions", dem=p("dem"), output=p("dem_cond"), fill_pits=True)
    wbt("d8_pointer", dem=p("dem_cond"), output=p("d8"))
    wbt("d8_flow_accumulation", i=p("dem_cond"), output=p("acc"), out_type="cells")
    wbt("d_inf_flow_accumulation", i=p("dem_cond"), output=p("sca"), out_type="sca")
    wbt("slope", dem=p("dem"), output=p("slope"), units="degrees")

    # Topographic wetness index ln(SCA / tan(slope)) and stream power index SCA * tan(slope)
    wbt("wetness_index", sca=p("sca"), slope=p("slope"), output=p("twi"))
    wbt("stream_power_index", sca=p("sca"), slope=p("slope"), output=p("spi"))
    out = {"twi": read_raster(p("twi"))[crop], "spi": read_raster(p("spi"))[crop]}

    slope_deg = read_raster(p("slope"))
    water = stream_mask(dem, transform, refs, workdir, read_raster(p("acc")))
    if not water.any():
        print("  no streams or river polygons near this tile; stream-based indices are NaN")
        for name in ("downslope_dist", "elevation_above_stream", "depth_to_water", "connectivity_index"):
            out[name] = np.full(out["twi"].shape, np.nan, dtype="float32")
        return out
    write_raster(p("water"), np.where(valid, water, np.nan), transform)

    wbt("downslope_distance_to_stream", dem=p("dem_cond"), streams=p("water"), output=p("downslope_dist"))
    wbt("elevation_above_stream_euclidean", dem=p("dem"), streams=p("water"), output=p("elevation_above_stream"))
    out["downslope_dist"] = read_raster(p("downslope_dist"))[crop]
    out["elevation_above_stream"] = read_raster(p("elevation_above_stream"))[crop]

    # Cartographic depth-to-water (Murphy et al. 2009): least-cost accumulation of
    # slope (m/m) x distance from the nearest water cell. WhiteboxTools' own
    # DepthToWater tool is part of the paid extension, so CostDistance is used.
    write_raster(p("tan_slope"), np.tan(np.radians(slope_deg)), transform)
    wbt("cost_distance", source=p("water"), cost=p("tan_slope"), out_accum=p("depth_to_water"),
        out_backlink=p("backlink"))
    out["depth_to_water"] = read_raster(p("depth_to_water"))[crop]

    c_factor = np.full(dem.shape, DEFAULT_C_FACTOR, dtype="float32")
    lcdb = refs["lcdb"]
    if lcdb is not None:
        lcdb = lcdb.iloc[lcdb.sindex.query(box(*grid_bounds(dem.shape, transform)), predicate="intersects")]
        if not lcdb.empty:
            c_factor = rasterize(zip(lcdb.geometry, lcdb.c_factor), out_shape=dem.shape, transform=transform,
                                 fill=DEFAULT_C_FACTOR, dtype="float32")
    out["connectivity_index"] = connectivity_index(workdir, transform, slope_deg, water, c_factor, valid, crop)
    return out


def terrain_attribute(surface, attribute, res, needed, margin, block=128):
    """An xdem terrain attribute, computed in blocks and only where ``needed``.

    ``surface`` has ``margin`` extra cells on every side of the ``needed`` grid,
    which must cover the attribute's moving window. Working in blocks bounds
    memory (xdem's rugosity needs ~400 bytes per cell) and skips blocks without
    polygons. Each attribute is requested on its own: when xdem 0.2 is given a
    list of attributes it returns them in a different order from the request.
    """
    out = np.full(needed.shape, np.nan, dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r0 in range(0, needed.shape[0], block):
            for c0 in range(0, needed.shape[1], block):
                r1, c1 = min(r0 + block, needed.shape[0]), min(c0 + block, needed.shape[1])
                if not needed[r0:r1, c0:c1].any():
                    continue
                window = surface[r0:r1 + 2 * margin, c0:c1 + 2 * margin]
                values = np.asarray(xdem.terrain.get_terrain_attribute(window, attribute=attribute, resolution=res))
                out[r0:r1, c0:c1] = values[margin:margin + r1 - r0, margin:margin + c1 - c0]
    return out


def distance_to_river(areas, rec):
    """Distance from each polygon to the nearest REC2 stream line (m)."""
    out = np.full(len(areas), np.nan)
    if rec is None or rec.empty:
        return out
    (src, _), dist = rec.sindex.nearest(areas.geometry.values, return_distance=True, return_all=False)
    out[src] = dist
    return out


def dominant_land_cover(areas, lcdb):
    """LCDB attributes of the land-cover class covering most of each polygon."""
    out = pd.DataFrame(index=areas.index, columns=LCDB_COLUMNS, dtype=object)
    if lcdb is None:
        return out
    lcdb = lcdb.iloc[lcdb.sindex.query(box(*areas.total_bounds), predicate="intersects")]
    if lcdb.empty:
        return out
    pieces = gpd.overlay(
        gpd.GeoDataFrame({"_pid": areas.index}, geometry=areas.geometry.values, crs=areas.crs),
        lcdb[LCDB_COLUMNS + ["geometry"]], how="intersection", keep_geom_type=True,
    )
    if pieces.empty:
        return out
    pieces["_area"] = pieces.area
    by_class = pieces.groupby(["_pid"] + LCDB_COLUMNS, dropna=False)["_area"].sum().reset_index()
    best = by_class.loc[by_class.groupby("_pid")["_area"].idxmax()].set_index("_pid")
    out.loc[best.index, LCDB_COLUMNS] = best[LCDB_COLUMNS].values
    for col in ("Wetland_18", "Onshore_18"):
        out[col] = out[col].map({"yes": True, "no": False, True: True, False: False})
    return out


def process_tile(tile, refs, keep_intermediate=False):
    """Detect change polygons in one DEM tile and compute their features."""
    t = refs["tiles_new"].loc[tile]
    res = float(t.res)
    core = (t.left, t.bottom, t.right, t.top)
    m = TERRAIN_MARGIN
    hb = int(round(HYDRO_BUFFER / res))
    other = "new" if HYDRO_EPOCH == "old" else "old"

    # The hydrology DEM is read with a wide buffer; the terrain margin is a sub-window of it.
    hydro_dem, hydro_transform = read_dem_window(refs[f"vrt_{HYDRO_EPOCH}"], expand(core, hb * res), res)
    surfaces = {HYDRO_EPOCH: hydro_dem[hb - m: hydro_dem.shape[0] - hb + m, hb - m: hydro_dem.shape[1] - hb + m]}
    surfaces[other], _ = read_dem_window(refs[f"vrt_{other}"], expand(core, m * res), res)
    surfaces["diff"] = surfaces["new"] - surfaces["old"]
    inner = (slice(m, -m), slice(m, -m))
    core_transform = from_origin(t.left, t.top, res, res)

    areas = detect_change_polygons(surfaces["diff"][inner], core_transform, res)
    areas = split_by_river_polygons(areas, refs["river_polygons"])
    if areas.empty:
        return areas
    areas.insert(0, "tile", tile)

    labels = rasterize(((geom, i + 1) for i, geom in enumerate(areas.geometry)),
                       out_shape=surfaces["diff"][inner].shape, transform=core_transform,
                       fill=0, dtype="int32")
    zones = Zones(labels, len(areas))
    features = {}

    # Elevation statistics
    for name in ("old", "new", "diff"):
        features.update(zones.stats(surfaces[name][inner], name))
    features["distance_to_river"] = distance_to_river(areas, refs["rec2"])

    # Hydrological indices
    workdir = os.path.abspath(os.path.join(WBT_DIR, tile))
    try:
        hydro = compute_hydrology(hydro_dem, hydro_transform, refs, workdir, (slice(hb, -hb), slice(hb, -hb)))
    finally:
        if not keep_intermediate:
            shutil.rmtree(workdir, ignore_errors=True)
    del hydro_dem
    for name in HYDRO_FEATURES:
        features.update(zones.stats(hydro[name], name))
    del hydro

    # Terrain attributes
    for name in ("old", "new", "diff"):
        for attribute in TERRAIN_ATTRIBUTES:
            values = terrain_attribute(surfaces[name], attribute, res, labels > 0, m)
            features.update(zones.stats(values, f"{name}_{attribute}"))

    result = pd.concat([areas, pd.DataFrame(features, index=areas.index),
                        dominant_land_cover(areas, refs["lcdb"])], axis=1)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=CRS)


def process_tiles(refs, tiles=None, shard=0, num_shards=1, keep_intermediate=False, retries=2):
    """Process DEM tiles, writing one parquet file per tile to ``TILE_RESULTS_DIR``."""
    os.makedirs(TILE_RESULTS_DIR, exist_ok=True)
    all_tiles = sorted(refs["tiles_new"].index)
    if tiles:
        unknown = set(tiles) - set(all_tiles)
        if unknown:
            raise SystemExit(f"Unknown tiles: {sorted(unknown)}")
        all_tiles = [t for t in all_tiles if t in tiles]
    my_tiles = all_tiles[shard::num_shards]
    print(f"Processing {len(my_tiles)} of {len(all_tiles)} tiles (shard {shard + 1}/{num_shards})")

    failed = []
    for tile in tqdm(my_tiles, desc="Processing tiles"):
        output_path = f"{TILE_RESULTS_DIR}/{tile}.parquet"
        if os.path.exists(output_path):
            continue
        if tile not in refs["tiles_old"].index:
            print(f"Skipping {tile}: not in the 2018-2020 collection")
            continue
        for attempt in range(1, retries + 2):
            start = time.time()
            try:
                features = process_tile(tile, refs, keep_intermediate)
                features.to_parquet(output_path)
                tqdm.write(f"{tile}: {len(features)} polygons in {time.time() - start:.0f}s")
                break
            except Exception:
                tqdm.write(f"Failed on {tile} (attempt {attempt}):\n{traceback.format_exc()}")
        else:
            failed.append(tile)
    if failed:
        print(f"{len(failed)} tiles failed: {failed}")
    return failed


# ---------------------------------------------------------------------------
# Stage 3: finalize
# ---------------------------------------------------------------------------
def join_parquet_files():
    """Join all individual tile parquet files into a single file."""
    print(f"Joining parquet files from {TILE_RESULTS_DIR}/...")
    files = sorted(glob(f"{TILE_RESULTS_DIR}/*.parquet"))
    if not files:
        print(f"No parquet files found in {TILE_RESULTS_DIR}")
        return None
    print(f"Found {len(files)} parquet files to join")
    areas = pd.concat((gpd.read_parquet(f) for f in tqdm(files)), ignore_index=True)
    print(f"Combined shape: {areas.shape}")
    areas.to_parquet(f"{OUTPUT_DIR}/areas.parquet")
    print(f"Saved combined areas to {OUTPUT_DIR}/areas.parquet")
    return areas


def add_ndvi_features(areas):
    """Add NDVI statistics (2018, 2023 and their difference) for each polygon."""
    print("Adding NDVI features...")
    paths = [f"{DATA_DIR}/NDVI_2018.tif", f"{DATA_DIR}/NDVI_2023.tif"]
    if not all(os.path.exists(p) for p in paths):
        print("NDVI rasters not found, skipping NDVI features")
        return areas

    with rasterio.open(paths[0]) as old_ndvi, rasterio.open(paths[1]) as new_ndvi:
        def ndvi_stats(geom):
            row = {}
            try:
                masked_old, _ = mask(old_ndvi, [geom], nodata=np.nan, crop=True)
                masked_new, _ = mask(new_ndvi, [geom], nodata=np.nan, crop=True)
            except ValueError:  # polygon outside the NDVI rasters
                masked_old = masked_new = np.array([np.nan])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
                for name, values in (("old", masked_old), ("new", masked_new), ("diff", masked_new - masked_old)):
                    for stat in STATS:
                        row[f"NDVI_{name}_{stat}"] = getattr(np, f"nan{stat}")(values)
            return row

        ndvi = pd.DataFrame([ndvi_stats(g) for g in tqdm(areas.geometry)], index=areas.index)
    areas = areas.drop(columns=ndvi.columns, errors="ignore")
    return gpd.GeoDataFrame(pd.concat([areas, ndvi], axis=1), geometry="geometry", crs=areas.crs)


def finalize(force_rebuild=False):
    if force_rebuild or not os.path.exists(f"{OUTPUT_DIR}/areas.parquet"):
        areas = join_parquet_files()
    else:
        print(f"Loading existing areas from {OUTPUT_DIR}/areas.parquet")
        areas = gpd.read_parquet(f"{OUTPUT_DIR}/areas.parquet")
    if areas is None:
        return
    areas = add_ndvi_features(areas)
    areas.to_parquet(f"{OUTPUT_DIR}/areas+NDVI.parquet")
    print(f"Saved areas with NDVI to {OUTPUT_DIR}/areas+NDVI.parquet")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(force_rebuild=False, stage="all", tiles=None, shard=0, num_shards=1, keep_intermediate=False):
    """Run the pipeline.

    Args:
        force_rebuild: rebuild data/areas.parquet from tile results even if it
            exists (tiles already in tile_results/ are reused; delete that
            directory to recompute them).
        stage: "all", "prepare", "tiles" or "finalize".
        tiles: optional list of tile names to process.
        shard, num_shards: process every ``num_shards``-th tile starting at ``shard``.
        keep_intermediate: keep the WhiteboxTools rasters in wbt_outputs/.
    """
    print(f"Starting batch processing (stage: {stage})...")
    failed = []
    if stage in ("all", "prepare"):
        prepare()
    if stage == "tiles" or (stage == "all" and (force_rebuild or not os.path.exists(f"{OUTPUT_DIR}/areas.parquet"))):
        refs = load_references()
        failed = process_tiles(refs, tiles, shard, num_shards, keep_intermediate)
    if stage in ("all", "finalize"):
        finalize(force_rebuild=force_rebuild)
    print("\nBatch processing complete!")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process river data from DEM tiles.")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild data/areas.parquet from tile results even if it exists")
    parser.add_argument("--stage", choices=["all", "prepare", "tiles", "finalize"], default="all")
    parser.add_argument("--tiles", nargs="+", help="Only process these tiles (e.g. BD44_10000_0503)")
    parser.add_argument("--shard", type=int, default=0, help="Index of this shard (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep WhiteboxTools intermediate rasters in wbt_outputs/")
    args = parser.parse_args()
    failed_tiles = main(args.force_rebuild, args.stage, args.tiles, args.shard, args.num_shards,
                        args.keep_intermediate)
    sys.exit(1 if failed_tiles else 0)
