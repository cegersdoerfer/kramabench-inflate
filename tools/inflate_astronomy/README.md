# Astronomy Data Inflation Tool

Synthetically inflates the Kramabench astronomy dataset to model realistic HPC-scale scientific data volumes, while preserving the correctness of all 12 benchmark tasks.

## Prerequisites

- Python 3.9+
- `numpy`, `pandas`

```bash
pip install numpy pandas
```

## Quick Start

```bash
# From the repo root
python tools/inflate_astronomy/inflate_astronomy_data.py \
  data/astronomy/input/ \
  data/astronomy/inflated_input/ \
  --scale-factor 2
```

### Arguments

| Argument | Description |
|---|---|
| `input_dir` | Path to original `data/astronomy/input/` |
| `output_dir` | Path for inflated output (new directory; preserves structure) |
| `--scale-factor N` | Target size multiplier (default: 10) |

### Approximate Output Sizes

| Scale factor | Size | File count |
|---|---|---|
| 2x | ~2.4 GB | ~3,000 |
| 5x | ~6 GB | ~7,500 |
| 10x | ~12 GB | ~15,000 |
| 50x | ~60 GB | ~72,000 |

## Running KramaBench with Inflated Data

The benchmark resolves the data path from `--project_root` using the pattern `data/<workload>/input/`. To use inflated data, create a directory tree where that path points to the inflated output, then pass it via `--project_root`.

### Option A: Symlink approach (recommended)

```bash
# Create a project root that maps to inflated data
mkdir -p /tmp/kramabench_inflated/data/astronomy
ln -s "$(pwd)/data/astronomy/inflated_input" /tmp/kramabench_inflated/data/astronomy/input

# Symlink the rest of the project structure
ln -s "$(pwd)/workload" /tmp/kramabench_inflated/workload
ln -s "$(pwd)/benchmark" /tmp/kramabench_inflated/benchmark
ln -s "$(pwd)/results"   /tmp/kramabench_inflated/results

# Run the benchmark
python evaluate.py \
  --sut YourSystem \
  --workload astronomy \
  --project_root /tmp/kramabench_inflated \
  --verbose
```

### Option B: In-place swap

```bash
# Back up original data
mv data/astronomy/input data/astronomy/input_original

# Point input at inflated data
ln -s inflated_input data/astronomy/input

# Run normally (no --project_root needed)
python evaluate.py --sut YourSystem --workload astronomy --verbose

# Restore when done
rm data/astronomy/input
mv data/astronomy/input_original data/astronomy/input
```

## What the Script Does

All original data is copied verbatim. The script only augments the copy.

### Phase 1 -- Column Inflation

Adds 20-30 deterministically-computed columns to existing CSV files:

- **OMNI2 CSVs** (715 files): Derived physics (IMF clock angle, dynamic pressure, Alfven speed, cone angle, etc.), rolling means (3h/6h/12h/24h) of Dst/Kp/f10.7/BZ, and lagged variants (1h/3h/6h) of Dst/Kp.
- **Sat_Density CSVs** (715 files): log10 density, rolling mean/std/min/max (1h window), detrended density, z-score normalized density, rate of change.
- **GOES CSVs**: log10 flux, rolling stats, flux ratios.
- **Space-track CSVs**: Derived orbital period, perigee/apogee altitude.

Files referenced by hard-10's all-column correlation are excluded (`omni2-wu590`, `swarma-wu570` through `wu579`).

### Phase 2 -- New Warmup Files

Generates `(scale_factor - 1) * 715` new OMNI2 + Sat_Density file pairs:

- Warmup IDs start at wu716
- All dates in 2020-2021 (safe range for all tasks)
- Content sampled from per-column statistics of existing files
- `initial_states.csv` is extended with new rows (altitudes outside 450-500 km to avoid easy-3's filter)

### Phase 3 -- Ancillary Files

Generates additional files that add bulk without affecting any task:

- **TLE files**: SATCAT numbers 99001+ (tasks only read 48445.tle and 43180.tle)
- **Geomag forecasts**: Dates outside 0309-0313 (tasks only read those 5 dates)
- **SP3 directories**: 2020-2021 dates (tasks only glob Oct 2018 and Sep 2019)
- **OMNI2 .dat files**: 2020 and 2021 (tasks only read 2023/2024)
- **GOES CSVs**: New wu716+ files (tasks only reference wu334/wu335)
- **SILSO files**: Monthly/daily sunspot data (easy-4 hardcodes `SN_y_tot_V2.0.csv`)
- **Swarm-B density files**: 2020-2021 months (hard-11 only globs `2024_*`)

## Safety Guarantees

The script is designed so that all 12 astronomy benchmark tasks produce identical answers against inflated data. The key constraints:

1. **Original file content is never modified** -- only copied, then columns are appended.
2. **New filenames avoid all task glob patterns**:
   - No `2015` in Sat_Density filenames (easy-3)
   - No Oct 2018 dates in Sat_Density or SP3 names (hard-10)
   - No `2024_*` in Swarm-B filenames (hard-11)
   - No Sep 2019 dates in SP3 names (hard-12)
3. **hard-10 exclusion files** are copied without column changes.
4. **New initial_states.csv rows** use altitudes outside 450-500 km (easy-3 filters to that range).
5. **Deterministic**: `np.random.seed(42)` ensures reproducible output.

## Verification

After generating inflated data, verify correctness:

```bash
# 1. Spot-check that original files are identical (ignoring added columns)
diff <(cut -d',' -f1-57 data/astronomy/inflated_input/STORM-AI/warmup/v2/OMNI2/omni2-wu001-20131129_to_20140128.csv) \
     <(cut -d',' -f1-57 data/astronomy/input/STORM-AI/warmup/v2/OMNI2/omni2-wu001-20131129_to_20140128.csv)

# 2. Verify excluded files are untouched
diff data/astronomy/inflated_input/STORM-AI/warmup/v2/OMNI2/omni2-wu590-20181001_to_20181130.csv \
     data/astronomy/input/STORM-AI/warmup/v2/OMNI2/omni2-wu590-20181001_to_20181130.csv

# 3. Run all 12 astronomy tasks and compare answers to ground truth
python evaluate.py --sut YourSystem --workload astronomy --project_root /tmp/kramabench_inflated --verbose
```
