# Study-area extracts

These files are the reference layers used by `src/batch.py`, clipped to the study
region (the Gisborne DEM tiles plus a 1.55 km margin: NZTM 1956050, 5670850 –
2098350, 5839550) and stored directly in git rather than Git LFS. Using them means
a pipeline run doesn't need the national archives in `data/` (about 1.2 GB of LFS
downloads) or the GNS geology package (400 MB).

| File | Contents | Source | Licence |
|---|---|---|---|
| `rec2.parquet` | River network lines (`nzsegment`, `StreamOrde`, `CUM_AREA`) | NIWA River Environment Classification (REC2) v5, `data/REC2_geodata_version_5.zip` | No licence stated; freely downloadable from NIWA, and the full archive is already in this repository |
| `lcdb.parquet` | Land cover polygons (`Class_2018`, `Wetland_18`, `Onshore_18`, `Class_2023`) | Manaaki Whenua – Landcare Research, LCDB v6.0 ([LRIS layer 123148](https://lris.scinfo.org.nz/layer/123148)), fetched from the LRIS WFS | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `river_polygons.parquet` | River polygons | LINZ NZ River Polygons (Topo, 1:50k), `data/lds-nz-river-polygons-*.zip` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `geology.parquet` | Geological units (`ROCKGROUP`, `KEYGRPNAME`, `SIMPLENAME`, `ABSMIN_MA`, `ABSMAX_MA`) | GNS Science Geological Map of New Zealand 1:250 000, 4th edition (2023) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `landslides.parquet` | Mapped landslide units | As above | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Attribution:

- Land cover: LCDB v6.0, Manaaki Whenua – Landcare Research, licensed under CC BY 4.0.
- River polygons: sourced from the LINZ Data Service, licensed under CC BY 4.0.
- Geology: © GNS Science 2023. Heron, D.W. (custodian) 2023: Geological Map of New
  Zealand 1:250 000 (4th ed.) [digital data]. Lower Hutt (NZ): GNS Science. (GNS
  Science geological map; 1). https://doi.org/10.21420/5XTJ-5718

## Rebuilding

The extracts are clipped to the study region, made valid and saved as zstd-compressed
GeoParquet by `build_extracts()` in `src/batch.py`. To rebuild all of them, fetch the
REC2 and river polygon archives with
`git lfs pull --include="data/REC2_geodata_version_5.zip,data/lds-nz-river-polygons-*.zip"`
and run:

```bash
LRIS_API_KEY=<your LRIS API key> python src/batch.py --stage prepare --rebuild-extracts
```

Name layers to rebuild only those, e.g. `--rebuild-extracts geology landslides`. The
GNS geology package is downloaded automatically from GNS Science, and LCDB is fetched
from the LRIS WFS, which needs an API key (create one in your LRIS account settings,
with access to query layer data).

## Soil and land-use data

Soil (S-map, the Fundamental Soil Layers) and NZLRI land use capability, rock and
erosion data are not included. The NZLRI and FSL layers on the Manaaki Whenua LRIS
portal are under the Landcare Data Use Licence, which doesn't allow the data or
derivatives to be provided to third parties, and S-map is CC BY-NC-ND. These can't be
published in this public repository.
