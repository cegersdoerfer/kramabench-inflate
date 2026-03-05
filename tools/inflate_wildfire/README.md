# Wildfire Data Inflation Tool

Synthetically inflates the Kramabench wildfire dataset (~30 MB, 22 files) to model realistic HPC-scale data volumes, while preserving the correctness of all 21 benchmark tasks.

## Prerequisites

- Python 3.9+
- `numpy`, `pandas`
- Optional: `geopandas`, `shapely` (for GeoPackage generation)

```bash
pip install numpy pandas geopandas shapely
```

## Quick Start

```bash
# From the repo root
python tools/inflate_wildfire/inflate_wildfire_data.py \
  data/wildfire/input/ \
  data/wildfire/inflated_input/ \
  --scale-factor 2
```

### Arguments

| Argument | Description |
|---|---|
| `input_dir` | Path to original `data/wildfire/input/` |
| `output_dir` | Path for inflated output (new directory; preserves structure) |
| `--scale-factor N` | Target size multiplier (default: 10) |

### Approximate Output Sizes

| Scale factor | Size | File count |
|---|---|---|
| 2x | ~42 MB | ~37 |
| 5x | ~500 MB | ~80 |
| 10x | ~1 GB | ~120 |
| 50x | ~5 GB | ~300+ |

## Running KramaBench with Inflated Data

The benchmark resolves the data path from `--project_root` using the pattern `data/<workload>/input/`. To use inflated data, create a directory tree where that path points to the inflated output, then pass it via `--project_root`.

### Option A: Symlink approach (recommended)

```bash
# Create a project root that maps to inflated data
mkdir -p /tmp/kramabench_inflated/data/wildfire
ln -s "$(pwd)/data/wildfire/inflated_input" /tmp/kramabench_inflated/data/wildfire/input

# Symlink the rest of the project structure
ln -s "$(pwd)/workload" /tmp/kramabench_inflated/workload
ln -s "$(pwd)/benchmark" /tmp/kramabench_inflated/benchmark
ln -s "$(pwd)/results"   /tmp/kramabench_inflated/results

# Run the benchmark
python evaluate.py \
  --sut YourSystem \
  --workload wildfire \
  --project_root /tmp/kramabench_inflated \
  --verbose
```

### Option B: In-place swap

```bash
# Back up original data
mv data/wildfire/input data/wildfire/input_original

# Point input at inflated data
ln -s inflated_input data/wildfire/input

# Run normally (no --project_root needed)
python evaluate.py --sut YourSystem --workload wildfire --verbose

# Restore when done
rm data/wildfire/input
mv data/wildfire/input_original data/wildfire/input
```

## What the Script Does

All original data is copied verbatim. The script only augments the copy.

### Phase 1 -- Column Inflation

Adds 15-34 derived columns to existing CSV files:

- **noaa_wildfires.csv** (37 → 71 cols): `hec_log10`, `hec_sqrt`, `wind_speed_squared`, `erc_wind_interaction`, `fire_intensity_proxy`, coordinate radians, rolling means (3/5/10-row) of `avrh_mean`/`wind_med`/`erc_med`/`rain_sum`, lagged variants (1/3/5-row) of `wind_med`/`erc_med`, binary flags (`is_large_fire`, `is_high_wind`).
- **Fire_Weather_Data_2002-2014_2016.csv** (37 → 70 cols): Same derived columns as noaa_wildfires, plus `fatalities_per_day`.
- **PublicView_RAWS_*.csv** (34 → 49 cols): Unit conversions (`Elevation_m`, `Wind_Speed_ms`, `Temp_C`, `Temp_K`), `Heat_Index_proxy`, `Wind_Chill_proxy`, coordinate radians, fuel moisture transforms.
- **annual_aqi_by_county_2024.csv** (18 → 30 cols): `AQI_range`, day-type ratios (`Good_ratio`, `Unhealthy_ratio`), `AQI_skew_proxy`, pollutant ratios.
- **Other CSVs**: Squared, log, and normalized variants of numeric columns.

Special-format files are left untouched: `noaa_wildfires_monthly_stats.csv` (3 header rows), `noaa_wildfires_variabledescrip.csv` (metadata dictionary), and all tab-separated NIFC files.

### Phase 2 -- New Synthetic CSVs

Generates new files with the `synthetic_` prefix (no task references these names):

- **RAWS station data**: `synthetic_RAWS_station_data_2020.csv`, etc. -- same schema as PublicView_RAWS with synthetic stations.
- **Fire weather observations**: `synthetic_fire_weather_observations_2015_2017.csv`, etc. -- same schema as noaa_wildfires/Fire_Weather_Data.
- **AQI files**: `synthetic_annual_aqi_by_county_2020.csv`, etc. -- same schema as the 2024 AQI file.
- **NIFC-style files**: `synthetic_nifc_prescribed_fire_acres.csv`, `synthetic_nifc_wildland_fire_use_acres.csv`.
- **State-level files**: Injuries, structures destroyed, and duration by state.
- **Summary files**: Seasonal stats and annual summaries.

### Phase 3 -- Large Observation Datasets

Generates bulk observation CSVs that scale with `--scale-factor`:

- **Hourly weather station data** (scale >= 3): Multi-year hourly readings across 10+ stations with realistic seasonal/diurnal patterns.
- **Satellite fire detections**: MODIS/VIIRS-style fire detection records with lat/lon, confidence, FRP, and brightness temperatures.

### Phase 4 -- GeoPackage Expansion

Generates additional `.gpkg` files (requires `geopandas`/`shapely`; skipped if not installed):

- **synthetic_usa_counties.gpkg**: County-level bounding boxes with fire risk scores.
- **synthetic_usa_fire_districts.gpkg**: Fire district boundaries.
- **synthetic_nifc_geographic_areas_detailed.gpkg**: Sub-region polygons.
- **synthetic_fire_locations.gpkg** (scale >= 5): Point-based fire location data.

## Safety Guarantees

The script is designed so that all 21 wildfire benchmark tasks produce identical answers against inflated data. The key constraints:

1. **Original file content is never modified** -- only copied, then columns are appended.
2. **Original row counts are preserved** -- no rows added or removed from existing files.
3. **Tasks reference columns by name**, so added columns are invisible to task queries.
4. **New files use the `synthetic_` prefix**, which no task's `data_sources` references.
5. **Tab-separated NIFC files** (with `$` and comma-formatted numbers) are byte-identical copies.
6. **Special-format files** (`noaa_wildfires_monthly_stats.csv` with 3 header rows, `noaa_wildfires_variabledescrip.csv`, `state_abbreviation_to_state.json`) are byte-identical copies.
7. **GeoPackage files** (`usa.gpkg`, `nifc_geographic_areas.gpkg`) are byte-identical copies.
8. **Deterministic**: `np.random.seed(42)` ensures reproducible output.

## Verification

After generating inflated data, verify correctness:

```bash
# 1. Spot-check that original columns are preserved (ignoring added columns)
python3 -c "
import pandas as pd
orig = pd.read_csv('data/wildfire/input/noaa_wildfires.csv')
infl = pd.read_csv('data/wildfire/inflated_input/noaa_wildfires.csv')
for col in orig.columns:
    assert orig[col].equals(infl[col]), f'MISMATCH: {col}'
print('All original columns preserved.')
"

# 2. Verify special-format files are byte-identical
diff data/wildfire/input/noaa_wildfires_monthly_stats.csv \
     data/wildfire/inflated_input/noaa_wildfires_monthly_stats.csv
diff data/wildfire/input/nifc_suppression_costs.csv \
     data/wildfire/inflated_input/nifc_suppression_costs.csv

# 3. Run all 21 wildfire tasks and compare answers to ground truth
python evaluate.py --sut YourSystem --workload wildfire --project_root /tmp/kramabench_inflated --verbose
```
