#!/usr/bin/env python3

import geopandas as gpd
import ee
import geemap
from tqdm.auto import tqdm

service_account = 'service-account@iron-dynamics-294100.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
ee.Initialize(credentials)

extent = gpd.read_file("https://nz-elevation.s3.ap-southeast-2.amazonaws.com/gisborne/gisborne_2018-2020/dem_1m/2193/capture-area.geojson").geometry[0].__geo_interface__

def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')  # B8 (NIR), B4 (Red)
    return image.addBands(ndvi)

for year in tqdm(range(2018, 2024)):
  print(year)
  ndvi = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
      .filterDate(f'{year}-01-01', f'{year}-12-31') \
      .filterBounds(extent) \
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
      .map(calculate_ndvi) \
      .select('NDVI') \
      .median()

  geemap.download_ee_image(
      ndvi,
      filename=f"NDVI_{year}.tif",
      #format="GEO_TIFF",
      region=extent,
      scale=10,
      crs='EPSG:2193',
      max_tile_size=16
  )