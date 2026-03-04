#!/usr/bin/env python3
"""
inflate_astronomy_data.py — Synthetic Dataset Inflation for Astronomy Domain

Inflates the Kramabench astronomy dataset by:
  1. Adding synthetic derived columns to existing CSVs
  2. Generating new warmup file pairs (OMNI2 + Sat_Density)
  3. Generating new ancillary files (TLE, geomag, SP3, OMNI2 .dat, GOES, SILSO)

All original data is copied verbatim; only augmented copies and new files are added.
New files use 2020–2021 dates (safe range — no task references this period).

Usage:
    python inflate_astronomy_data.py <input_dir> <output_dir> --scale-factor N
"""

import argparse
import math
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RNG_SEED = 42

# Files that must NOT have columns added (hard-10 all-column correlation)
HARD10_OMNI2_EXCLUSION = "omni2-wu590-20181001_to_20181130.csv"
HARD10_SATDENSITY_EXCLUSIONS = {
    f"swarma-wu{i}-" for i in range(570, 580)
}

# Safe date range for new files
SAFE_START = datetime(2020, 1, 3)
SAFE_END = datetime(2021, 12, 28)

# Physical constants
MU0 = 4 * math.pi * 1e-7  # Permeability of free space (H/m)
MP_KG = 1.6726e-27  # Proton mass (kg)

# Sentinel values used in OMNI2 data
OMNI2_SENTINELS = {
    "Scalar_B_nT": 999.9,
    "Vector_B_Magnitude_nT": 999.9,
    "BX_nT_GSE_GSM": 999.9,
    "BY_nT_GSE": 999.9,
    "BZ_nT_GSE": 999.9,
    "BY_nT_GSM": 999.9,
    "BZ_nT_GSM": 999.9,
    "SW_Plasma_Temperature_K": 9999999.0,
    "SW_Proton_Density_N_cm3": 999.9,
    "SW_Plasma_Speed_km_s": 9999.0,
    "Flow_pressure": 99.99,
    "Plasma_Beta": 999.99,
    "Alfen_mach_number": 999.9,
    "Magnetosonic_Mach_number": 99.9,
    "Kp_index": 99,
    "Dst_index_nT": 99999,
    "f10.7_index": 999.9,
}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Column Inflation
# ──────────────────────────────────────────────────────────────────────────────

def add_omni2_columns(df):
    """Add derived physics and rolling/lagged columns to an OMNI2 DataFrame."""
    by = df.get("BY_nT_GSE")
    bz = df.get("BZ_nT_GSE")
    if by is not None and bz is not None:
        df["IMF_clock_angle_rad"] = np.arctan2(by, bz)

    b = df.get("Scalar_B_nT")
    n = df.get("SW_Proton_Density_N_cm3")
    v = df.get("SW_Plasma_Speed_km_s")

    if n is not None and v is not None:
        # Dynamic pressure: P = 1.67e-6 * n * v^2 (nPa, with n in cm^-3, v in km/s)
        df["SW_dynamic_pressure_nPa"] = 1.67e-6 * n * v ** 2

    if b is not None and n is not None:
        # Alfven speed: Va = 20 * B / sqrt(n) (km/s, with B in nT, n in cm^-3)
        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_n = np.sqrt(n)
            df["Alfven_speed_km_s"] = np.where(sqrt_n > 0, 20.0 * b / sqrt_n, np.nan)

    if b is not None and n is not None:
        temp = df.get("SW_Plasma_Temperature_K")
        if temp is not None:
            # Beta derived: Beta = [(T*4.16e-5) + 5.34] * n / B^2
            with np.errstate(divide="ignore", invalid="ignore"):
                df["Beta_derived"] = np.where(
                    b > 0,
                    ((temp * 4.16e-5) + 5.34) * n / (b ** 2),
                    np.nan,
                )

    if b is not None and n is not None:
        temp = df.get("SW_Plasma_Temperature_K")
        if temp is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                alfven = np.where(np.sqrt(n) > 0, 20.0 * b / np.sqrt(n), np.nan)
                sound = 0.12 * np.sqrt(temp + 1.28e5)
                df["Magnetosonic_speed_derived"] = np.sqrt(alfven ** 2 + sound ** 2)

    # Electric field magnitude
    bz_gsm = df.get("BZ_nT_GSM")
    if v is not None and bz_gsm is not None:
        df["E_field_magnitude_mV_m"] = np.abs(v * bz_gsm * 1e-3)

    # Proton gyrofrequency
    if b is not None:
        df["Proton_gyrofreq_Hz"] = 1.5241e-2 * b  # approximate: eB/(2*pi*mp) with B in nT

    # IMF cone angle
    bx = df.get("BX_nT_GSE_GSM")
    if bx is not None and b is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["IMF_cone_angle_deg"] = np.degrees(np.arccos(np.clip(np.abs(bx) / np.where(b > 0, b, np.nan), -1, 1)))

    # Rolling means for key parameters
    for col, windows in [
        ("Dst_index_nT", [3, 6, 12, 24]),
        ("Kp_index", [3, 6, 12]),
        ("f10.7_index", [6, 12, 24]),
        ("BZ_nT_GSE", [3, 6, 12]),
    ]:
        if col in df.columns:
            for w in windows:
                df[f"{col}_rolling_mean_{w}h"] = (
                    df[col].rolling(window=w, min_periods=1).mean()
                )

    # Lagged variants
    for col, lags in [
        ("Dst_index_nT", [1, 3, 6]),
        ("Kp_index", [1, 3, 6]),
    ]:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    return df


def add_satdensity_columns(df):
    """Add derived columns to a Sat_Density DataFrame."""
    density_col = "Orbit Mean Density (kg/m^3)"
    if density_col not in df.columns:
        return df

    density = df[density_col]

    # Log10 density
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Orbit_Mean_Density_log10"] = np.where(
            density > 0, np.log10(density), np.nan
        )

    # Rolling statistics (1h = 6 points at 10-min cadence)
    df["Density_rolling_mean_1h"] = density.rolling(window=6, min_periods=1).mean()
    df["Density_rolling_std_1h"] = density.rolling(window=6, min_periods=1).std()

    # Detrended (density minus 6h rolling mean; 6h = 36 points)
    rolling_6h = density.rolling(window=36, min_periods=1).mean()
    df["Density_detrended"] = density - rolling_6h

    # Normalized (z-score over entire file)
    file_mean = density.mean()
    file_std = density.std()
    if file_std > 0 and not np.isnan(file_std):
        df["Density_normalized"] = (density - file_mean) / file_std
    else:
        df["Density_normalized"] = 0.0

    # Rolling min/max
    df["Density_rolling_min_1h"] = density.rolling(window=6, min_periods=1).min()
    df["Density_rolling_max_1h"] = density.rolling(window=6, min_periods=1).max()

    # Rate of change (finite difference)
    df["Density_rate_of_change"] = density.diff()

    # Cumulative sum (normalized)
    if file_mean != 0 and not np.isnan(file_mean):
        df["Density_cumsum_norm"] = (density / file_mean).cumsum()
    else:
        df["Density_cumsum_norm"] = density.cumsum()

    return df


def add_goes_columns(df):
    """Add derived columns to a GOES DataFrame."""
    for prefix in ["xrsa", "xrsb"]:
        flux_col = f"{prefix}_flux"
        if flux_col in df.columns:
            flux = df[flux_col]
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"{prefix}_flux_log10"] = np.where(flux > 0, np.log10(flux), np.nan)
            df[f"{prefix}_flux_rolling_mean_5m"] = flux.rolling(window=5, min_periods=1).mean()
            df[f"{prefix}_flux_rolling_std_5m"] = flux.rolling(window=5, min_periods=1).std()
            df[f"{prefix}_flux_rate"] = flux.diff()

    # Flux ratio
    if "xrsa_flux" in df.columns and "xrsb_flux" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["xrsa_xrsb_flux_ratio"] = np.where(
                df["xrsb_flux"] > 0,
                df["xrsa_flux"] / df["xrsb_flux"],
                np.nan,
            )

    return df


def is_hard10_excluded(filename):
    """Check if a file should be excluded from column inflation."""
    basename = os.path.basename(filename)
    if basename == HARD10_OMNI2_EXCLUSION:
        return True
    for prefix in HARD10_SATDENSITY_EXCLUSIONS:
        if basename.startswith(prefix):
            return True
    return False


def inflate_columns(output_dir):
    """Phase 1: Add synthetic columns to all eligible CSVs."""
    storm_ai = os.path.join(output_dir, "STORM-AI", "warmup", "v2")

    # OMNI2 files
    omni2_dir = os.path.join(storm_ai, "OMNI2")
    if os.path.isdir(omni2_dir):
        files = sorted(f for f in os.listdir(omni2_dir) if f.endswith(".csv"))
        for i, fname in enumerate(files):
            fpath = os.path.join(omni2_dir, fname)
            if is_hard10_excluded(fname):
                continue
            try:
                df = pd.read_csv(fpath)
                df = add_omni2_columns(df)
                df.to_csv(fpath, index=False)
            except Exception as e:
                print(f"  Warning: could not inflate {fname}: {e}", file=sys.stderr)
            if (i + 1) % 100 == 0:
                print(f"    OMNI2 column inflation: {i + 1}/{len(files)}")

    # Sat_Density files
    sat_dir = os.path.join(storm_ai, "Sat_Density")
    if os.path.isdir(sat_dir):
        files = sorted(f for f in os.listdir(sat_dir) if f.endswith(".csv"))
        for i, fname in enumerate(files):
            fpath = os.path.join(sat_dir, fname)
            if is_hard10_excluded(fname):
                continue
            try:
                df = pd.read_csv(fpath)
                df = add_satdensity_columns(df)
                df.to_csv(fpath, index=False)
            except Exception as e:
                print(f"  Warning: could not inflate {fname}: {e}", file=sys.stderr)
            if (i + 1) % 100 == 0:
                print(f"    Sat_Density column inflation: {i + 1}/{len(files)}")

    # GOES files
    goes_dir = os.path.join(storm_ai, "GOES")
    if os.path.isdir(goes_dir):
        for fname in os.listdir(goes_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(goes_dir, fname)
            try:
                df = pd.read_csv(fpath)
                df = add_goes_columns(df)
                df.to_csv(fpath, index=False)
            except Exception as e:
                print(f"  Warning: could not inflate {fname}: {e}", file=sys.stderr)

    # Space-track files
    spacetrack_dir = os.path.join(output_dir, "space-track")
    if os.path.isdir(spacetrack_dir):
        for fname in os.listdir(spacetrack_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(spacetrack_dir, fname)
            try:
                df = pd.read_csv(fpath)
                # Add derived columns
                if "MEAN_MOTION" in df.columns:
                    df["PERIOD_DERIVED_MIN"] = 1440.0 / df["MEAN_MOTION"]
                if "ECCENTRICITY" in df.columns and "SEMIMAJOR_AXIS" in df.columns:
                    df["PERIGEE_ALT_DERIVED"] = (
                        df["SEMIMAJOR_AXIS"] * (1 - df["ECCENTRICITY"]) - 6371.0
                    )
                    df["APOGEE_ALT_DERIVED"] = (
                        df["SEMIMAJOR_AXIS"] * (1 + df["ECCENTRICITY"]) - 6371.0
                    )
                df.to_csv(fpath, index=False)
            except Exception as e:
                print(f"  Warning: could not inflate {fname}: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: New Warmup Files
# ──────────────────────────────────────────────────────────────────────────────

def compute_omni2_stats(omni2_dir, sample_size=50):
    """Compute per-column mean and std from a sample of OMNI2 files."""
    rng = np.random.RandomState(RNG_SEED)
    files = sorted(f for f in os.listdir(omni2_dir) if f.endswith(".csv"))
    sample_files = rng.choice(files, size=min(sample_size, len(files)), replace=False)

    all_dfs = []
    for fname in sample_files:
        try:
            df = pd.read_csv(os.path.join(omni2_dir, fname))
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        return None, None, None

    combined = pd.concat(all_dfs, ignore_index=True)
    # Get numeric columns only (excluding Timestamp)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    means = combined[numeric_cols].mean()
    stds = combined[numeric_cols].std()
    cols = combined.columns.tolist()
    return cols, means, stds


def compute_satdensity_stats(sat_dir, sample_size=50):
    """Compute density statistics from a sample of Sat_Density files."""
    rng = np.random.RandomState(RNG_SEED + 1)
    files = sorted(f for f in os.listdir(sat_dir) if f.endswith(".csv"))
    sample_files = rng.choice(files, size=min(sample_size, len(files)), replace=False)

    densities = []
    for fname in sample_files:
        try:
            df = pd.read_csv(os.path.join(sat_dir, fname))
            col = "Orbit Mean Density (kg/m^3)"
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    densities.extend(valid.values)
        except Exception:
            continue

    if densities:
        arr = np.array(densities)
        return float(np.nanmean(arr)), float(np.nanstd(arr))
    return 1.5e-13, 5e-14  # Fallback


def generate_omni2_csv(rng, cols, means, stds, start_date, end_date, filepath):
    """Generate a synthetic OMNI2 CSV file."""
    hours = int((end_date - start_date).total_seconds() / 3600)
    timestamps = [start_date + timedelta(hours=h) for h in range(hours)]

    data = {"Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]}

    for col in cols:
        if col == "Timestamp":
            continue
        if col == "YEAR":
            data[col] = [t.year for t in timestamps]
        elif col == "DOY":
            data[col] = [t.timetuple().tm_yday for t in timestamps]
        elif col in means.index and col in stds.index:
            mean_val = means[col]
            std_val = stds[col]
            if np.isnan(mean_val):
                mean_val = 0.0
            if np.isnan(std_val) or std_val == 0:
                std_val = 1.0
            values = rng.normal(mean_val, std_val, size=hours)
            # Integer columns
            if col in (
                "Bartels_rotation_number", "ID_for_IMF_spacecraft",
                "ID_for_SW_Plasma_spacecraft", "num_points_IMF_averages",
                "num_points_Plasma_averages", "Kp_index", "R_Sunspot_No",
                "Dst_index_nT", "ap_index_nT", "AE_index_nT",
                "AL_index_nT", "AU_index_nT", "Flux_FLAG",
            ):
                values = np.round(values).astype(int)
            # Sprinkle sentinel values (~2% of rows)
            if col in OMNI2_SENTINELS:
                sentinel_mask = rng.random(hours) < 0.02
                values = np.where(sentinel_mask, OMNI2_SENTINELS[col], values)
            data[col] = values
        else:
            data[col] = [0] * hours

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def generate_satdensity_csv(rng, density_mean, density_std, start_date, end_date, filepath):
    """Generate a synthetic Sat_Density CSV file."""
    # 10-minute cadence
    minutes = int((end_date - start_date).total_seconds() / 60)
    n_points = minutes // 10
    timestamps = [start_date + timedelta(minutes=10 * i) for i in range(n_points)]

    densities = rng.normal(density_mean, density_std, size=n_points)
    densities = np.abs(densities)  # Density must be positive

    # ~5% missing values (empty strings)
    missing_mask = rng.random(n_points) < 0.05

    data = {
        "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "Orbit Mean Density (kg/m^3)": [
            "" if missing_mask[i] else f"{densities[i]:.15e}"
            for i in range(n_points)
        ],
    }

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def generate_warmup_files(output_dir, scale_factor):
    """Phase 2: Generate new OMNI2 + Sat_Density warmup file pairs."""
    storm_ai = os.path.join(output_dir, "STORM-AI", "warmup", "v2")
    omni2_dir = os.path.join(storm_ai, "OMNI2")
    sat_dir = os.path.join(storm_ai, "Sat_Density")
    initial_states_path = os.path.join(storm_ai, "wu001_to_wu715-initial_states.csv")

    n_new = int((scale_factor - 1) * 715)
    if n_new <= 0:
        return 0

    print(f"  Computing statistics from existing files...")
    cols, means, stds = compute_omni2_stats(omni2_dir)
    density_mean, density_std = compute_satdensity_stats(sat_dir)

    if cols is None:
        print("  Warning: could not compute OMNI2 stats, skipping warmup generation")
        return 0

    rng = np.random.RandomState(RNG_SEED + 100)

    # Generate date ranges within 2020–2021
    total_days = (SAFE_END - SAFE_START).days
    new_initial_rows = []

    print(f"  Generating {n_new} new warmup file pairs...")
    for i in range(n_new):
        wu_id = 716 + i
        wu_str = f"wu{wu_id:03d}" if wu_id < 1000 else f"wu{wu_id}"

        # Random start date within safe range
        offset_days = rng.randint(0, max(1, total_days - 90))
        start = SAFE_START + timedelta(days=int(offset_days))

        # OMNI2: ~60 day span
        omni2_end = start + timedelta(days=60)
        if omni2_end > SAFE_END:
            omni2_end = SAFE_END

        omni2_start_str = start.strftime("%Y%m%d")
        omni2_end_str = omni2_end.strftime("%Y%m%d")
        omni2_fname = f"omni2-{wu_str}-{omni2_start_str}_to_{omni2_end_str}.csv"
        generate_omni2_csv(
            rng, cols, means, stds, start, omni2_end,
            os.path.join(omni2_dir, omni2_fname),
        )

        # Sat_Density: ~3 day span at end of OMNI2 period
        sat_start = omni2_end - timedelta(days=3)
        sat_end = omni2_end
        sat_start_str = sat_start.strftime("%Y%m%d")
        sat_end_str = sat_end.strftime("%Y%m%d")
        sat_fname = f"swarma-{wu_str}-{sat_start_str}_to_{sat_end_str}.csv"
        generate_satdensity_csv(
            rng, density_mean, density_std, sat_start, sat_end,
            os.path.join(sat_dir, sat_fname),
        )

        # Initial state row — altitude outside 450–500 km (use 350–440 or 510–550)
        if rng.random() < 0.5:
            altitude = rng.uniform(350, 440)
        else:
            altitude = rng.uniform(510, 550)

        sma = 6371.0 + altitude + rng.uniform(-5, 5)
        ecc = rng.uniform(0.0001, 0.01)
        inc = rng.uniform(87, 91)
        raan = rng.uniform(0, 360)
        aop = rng.uniform(0, 360)
        ta = rng.uniform(0, 360)
        lat = rng.uniform(-90, 90)
        lon = rng.uniform(-180, 180)

        new_initial_rows.append({
            "File ID": wu_str,
            "Timestamp": sat_start.strftime("%Y-%m-%d %H:%M:%S"),
            "Semi-major Axis (km)": sma,
            "Eccentricity": ecc,
            "Inclination (deg)": inc,
            "RAAN (deg)": raan,
            "Argument of Perigee (deg)": aop,
            "True Anomaly (deg)": ta,
            "Latitude (deg)": lat,
            "Longitude (deg)": lon,
            "Altitude (km)": altitude,
        })

        if (i + 1) % 500 == 0:
            print(f"    Generated {i + 1}/{n_new} warmup pairs")

    # Append to initial_states.csv
    if new_initial_rows and os.path.isfile(initial_states_path):
        existing = pd.read_csv(initial_states_path)
        new_df = pd.DataFrame(new_initial_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(initial_states_path, index=False)
        print(f"  Appended {len(new_initial_rows)} rows to initial_states.csv")

    return n_new


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Ancillary Files
# ──────────────────────────────────────────────────────────────────────────────

def generate_tle_files(output_dir, scale_factor):
    """Generate new TLE files with unique SATCAT numbers."""
    tle_dir = os.path.join(output_dir, "TLE")
    os.makedirs(tle_dir, exist_ok=True)

    n_new = max(5, int(scale_factor * 2))
    rng = np.random.RandomState(RNG_SEED + 200)

    count = 0
    for i in range(n_new):
        satcat = 99001 + i
        fname = os.path.join(tle_dir, f"{satcat}.tle")

        lines = []
        # Generate ~10 TLE sets per file
        epoch_day = 120.0
        for j in range(10):
            epoch_day += rng.uniform(0.3, 0.5)
            mean_motion = rng.uniform(15.0, 15.5)
            inc = rng.uniform(87.0, 98.0)
            raan = rng.uniform(0, 360)
            ecc = rng.uniform(0.0001, 0.005)
            aop = rng.uniform(0, 360)
            ma = rng.uniform(0, 360)
            ndot = rng.uniform(0.00001, 0.0001)
            bstar = rng.uniform(1e-4, 5e-4)

            # Format TLE lines (simplified but structurally valid)
            ecc_str = f"{ecc:.7f}"[2:]  # Remove "0."
            bstar_exp = int(math.floor(math.log10(abs(bstar)))) if bstar != 0 else 0
            bstar_man = bstar / (10 ** bstar_exp)
            bstar_str = f"{bstar_man * 10000:5.0f}{bstar_exp:+d}".replace("+", "+").replace("-0", "-0")

            line1 = (
                f"1 {satcat:05d}U 20001A   20{epoch_day:012.8f}  "
                f".{ndot * 1e8:08.0f}  00000-0  {bstar_str} 0  999{j % 10}"
            )
            line2 = (
                f"2 {satcat:05d} {inc:8.4f} {raan:8.4f} {ecc_str} "
                f"{aop:8.4f} {ma:8.4f} {mean_motion:11.8f}{(10000 + j):05d}{j % 10}"
            )
            lines.append(line1)
            lines.append(line2)

        with open(fname, "w") as f:
            f.write("\n".join(lines) + "\n")
        count += 1

    return count


def generate_geomag_files(output_dir, scale_factor):
    """Generate new geomag forecast files for dates outside 0309–0313."""
    geomag_dir = os.path.join(output_dir, "geomag_forecast")
    os.makedirs(geomag_dir, exist_ok=True)

    rng = np.random.RandomState(RNG_SEED + 300)

    # Safe dates: 0301–0308 and 0314–0331
    safe_dates = list(range(301, 309)) + list(range(314, 332))
    n_new = min(len(safe_dates), max(5, int(scale_factor)))

    count = 0
    for day_num in safe_dates[:n_new]:
        mmdd = f"{day_num:04d}"
        fname = os.path.join(geomag_dir, f"{mmdd}geomag_forecast.txt")

        month = day_num // 100
        day = day_num % 100
        month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]

        ap_obs = rng.randint(5, 50)
        ap_est = rng.randint(5, 50)
        ap_pred = [rng.randint(5, 40) for _ in range(3)]

        kp_values = [[rng.uniform(1, 5) for _ in range(3)] for _ in range(8)]
        time_slots = [
            "00-03UT", "03-06UT", "06-09UT", "09-12UT",
            "12-15UT", "15-18UT", "18-21UT", "21-00UT",
        ]

        next_day1 = day + 1
        next_day2 = day + 2
        next_day3 = day + 3

        content = f""":Product: {mmdd}geomag_forecast.txt
:Issued: 2025 {month_name} {day:02d} 2205 UTC
# Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction Center
#
NOAA Ap Index Forecast
Observed Ap {day - 1:02d} {month_name} {ap_obs:03d}
Estimated Ap {day:02d} {month_name} {ap_est:03d}
Predicted Ap {next_day1:02d} {month_name}-{next_day3:02d} {month_name} {ap_pred[0]:03d}-{ap_pred[1]:03d}-{ap_pred[2]:03d}

NOAA Geomagnetic Activity Probabilities {next_day1:02d} {month_name}-{next_day3:02d} {month_name}
Active                {rng.randint(10, 50):02d}/{rng.randint(10, 50):02d}/{rng.randint(10, 50):02d}
Minor storm           {rng.randint(1, 30):02d}/{rng.randint(1, 20):02d}/{rng.randint(1, 15):02d}
Moderate storm        {rng.randint(1, 15):02d}/{rng.randint(1, 10):02d}/{rng.randint(1, 5):02d}
Strong-Extreme storm  {rng.randint(1, 5):02d}/{rng.randint(1, 3):02d}/{rng.randint(1, 3):02d}

NOAA Kp index forecast {next_day1:02d} {month_name} - {next_day3:02d} {month_name}
             {month_name} {next_day1:02d}    {month_name} {next_day2:02d}    {month_name} {next_day3:02d}
"""
        for slot_idx, slot in enumerate(time_slots):
            vals = kp_values[slot_idx]
            content += f"{slot}        {vals[0]:4.2f}      {vals[1]:4.2f}      {vals[2]:4.2f}\n"

        with open(fname, "w") as f:
            f.write(content)
        count += 1

    return count


def generate_sp3_directories(output_dir, scale_factor):
    """Generate new SP3 directories with 2020–2021 dates."""
    pod_dir = os.path.join(output_dir, "swarm", "POD")
    os.makedirs(pod_dir, exist_ok=True)

    rng = np.random.RandomState(RNG_SEED + 400)
    n_new = max(5, int(scale_factor * 3))

    # Generate dates in 2020–2021
    start = datetime(2020, 1, 1)
    count = 0

    for i in range(n_new):
        day_offset = int(i * (730 / n_new))  # Spread across 2 years
        dt = start + timedelta(days=day_offset)
        next_dt = dt + timedelta(days=1)

        t1 = dt.strftime("%Y%m%dT235942")
        t2 = next_dt.strftime("%Y%m%dT235942")
        dirname = f"SW_OPER_SP3ACOM_2__{t1}_{t2}_0201"
        dirpath = os.path.join(pod_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)

        # Generate SP3 file
        sp3_path = os.path.join(dirpath, f"{dirname}.sp3")
        year = dt.year
        month = dt.month
        day = dt.day

        # SP3 header
        lines = []
        lines.append(f"#dV{year} {month:2d} {day:2d}  0  0  0.00000000    8640 U+u   IGS14 FIT TUD ")
        lines.append(f"## 2022  86400.00000000    10.00000000 58399 0.0000000000000")
        lines.append("+    1   L47  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0")
        for _ in range(4):
            lines.append("+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0")
        for _ in range(5):
            lines.append("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0")
        lines.append("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc")
        lines.append("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc")
        lines.append("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000")
        lines.append("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000")
        lines.append("%i    0    0    0    0      0      0      0      0         0")
        lines.append("%i    0    0    0    0      0      0      0      0         0")
        lines.append(f"/* Created by: inflate_astronomy_data.py (synthetic)")
        lines.append(f"/* Date: {dt.strftime('%Y-%m-%d')}")
        lines.append("/*")
        lines.append("/*")

        # Data records: every 10 seconds for 24 hours = 8640 epochs
        # Generate a subset to keep file size manageable
        n_epochs = 8640
        # Generate position data with orbital motion
        for epoch in range(n_epochs):
            t_sec = epoch * 10
            h = t_sec // 3600
            m = (t_sec % 3600) // 60
            s = t_sec % 60

            lines.append(f"*  {year} {month:2d} {day:2d} {h:2d} {m:2d} {s:2d}.00000000")

            # Simplified orbital position (LEO circular orbit)
            omega = 2 * math.pi / 5400  # ~90 min period
            angle = omega * t_sec
            r = 6800  # km
            x = r * math.cos(angle) + rng.normal(0, 0.1)
            y = r * math.sin(angle) * 0.9 + rng.normal(0, 0.1)
            z = r * math.sin(angle) * 0.4 + rng.normal(0, 0.1)

            vx = -r * omega * math.sin(angle) * 1e4 + rng.normal(0, 1)
            vy = r * omega * math.cos(angle) * 0.9 * 1e4 + rng.normal(0, 1)
            vz = r * omega * math.cos(angle) * 0.4 * 1e4 + rng.normal(0, 1)

            lines.append(f"PL47 {x:13.7f} {y:13.7f} {z:13.7f} 999999.999999")
            lines.append(f"VL47{vx:14.7f}{vy:14.7f}{vz:14.7f} 999999.999999")

        lines.append("EOF")

        with open(sp3_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        # Generate HDR file (XML skeleton)
        hdr_path = os.path.join(dirpath, f"{dirname}.HDR")
        hdr_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Earth_Explorer_Header xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Fixed_Header>
    <File_Name>{dirname}</File_Name>
    <File_Description>Swarm-A SP3 Precise Orbit (Synthetic)</File_Description>
    <Mission>Swarm</Mission>
    <File_Type>SP3ACOM_2_</File_Type>
    <Validity_Period>
      <Validity_Start>UTC={dt.strftime("%Y-%m-%dT23:59:42")}</Validity_Start>
      <Validity_Stop>UTC={next_dt.strftime("%Y-%m-%dT23:59:42")}</Validity_Stop>
    </Validity_Period>
    <Source>
      <System>SYNTHETIC</System>
      <Creator>inflate_astronomy_data.py</Creator>
      <Creator_Version>1.0</Creator_Version>
      <Creation_Date>UTC=2025-01-01T00:00:00</Creation_Date>
    </Source>
  </Fixed_Header>
</Earth_Explorer_Header>
"""
        with open(hdr_path, "w") as f:
            f.write(hdr_content)

        count += 1
        if count % 10 == 0:
            print(f"    SP3 directories: {count}/{n_new}")

    return count


def generate_omni2_dat_files(output_dir, scale_factor):
    """Generate new OMNI2 .dat files for 2020 and 2021."""
    dat_dir = os.path.join(output_dir, "omni2_low_res")
    os.makedirs(dat_dir, exist_ok=True)

    rng = np.random.RandomState(RNG_SEED + 500)
    count = 0

    for year in [2020, 2021]:
        fname = os.path.join(dat_dir, f"omni2_{year}.dat")
        if os.path.exists(fname):
            continue

        # Determine number of hours in year
        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        n_days = 366 if is_leap else 365
        n_hours = n_days * 24

        lines = []
        for h in range(n_hours):
            doy = h // 24 + 1
            hour = h % 24

            bartels = 2500 + (h // (27 * 24))
            isc_imf = rng.choice([51, 52])
            isc_sw = 52
            n_imf = rng.randint(30, 65)
            n_sw = rng.randint(20, 40)

            b_mag = rng.uniform(3.0, 10.0)
            b_vec = b_mag * rng.uniform(0.8, 1.0)
            lat_b = rng.uniform(-30, 30)
            lon_b = rng.uniform(0, 360)
            bx = rng.uniform(-5, 5)
            by_gse = rng.uniform(-5, 5)
            bz_gse = rng.uniform(-5, 5)
            by_gsm = by_gse + rng.uniform(-0.5, 0.5)
            bz_gsm = bz_gse + rng.uniform(-0.5, 0.5)

            sigma_b = rng.uniform(0.1, 2.0)
            sigma_bv = rng.uniform(0.1, 2.0)
            sigma_bx = rng.uniform(0.1, 2.0)
            sigma_by = rng.uniform(0.1, 2.0)
            sigma_bz = rng.uniform(0.1, 2.0)

            temp = rng.uniform(10000, 200000)
            density = rng.uniform(1.0, 20.0)
            speed = rng.uniform(250, 700)
            flow_lon = rng.uniform(-5, 5)
            flow_lat = rng.uniform(-5, 5)
            na_np = rng.uniform(0.001, 0.05)

            sigma_t = rng.uniform(1000, 50000)
            sigma_n = rng.uniform(0.1, 3.0)
            sigma_v = rng.uniform(1, 20)
            sigma_phi = rng.uniform(0.1, 3.0)
            sigma_theta = rng.uniform(0.1, 3.0)
            sigma_ratio = rng.uniform(0.001, 0.01)

            flow_p = 1.67e-6 * density * speed ** 2
            e_field = -speed * bz_gsm * 1e-3
            plasma_beta = rng.uniform(0.3, 5.0)
            alfven_mach = rng.uniform(3, 15)

            kp = rng.choice([0, 3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40])
            sunspot = rng.randint(0, 200)
            dst = rng.randint(-100, 50)
            ae = rng.randint(10, 500)

            # Proton fluxes (use sentinel for simplicity)
            pflux = "999999.99 99999.99 99999.99 99999.99 99999.99 99999.99"
            flag = 0

            ap = rng.randint(2, 50)
            f107 = rng.uniform(65, 200)
            pc = rng.uniform(0, 3)
            al = rng.randint(-500, 0)
            au = rng.randint(0, 500)
            ms_mach = rng.uniform(3, 10)

            line = (
                f"{year:4d}{doy:4d}{hour:3d}{bartels:5d}{isc_imf:3d}{isc_sw:3d}"
                f"{n_imf:4d}{n_sw:4d}"
                f"{b_mag:6.1f}{b_vec:6.1f}{lat_b:6.1f}{lon_b:6.1f}"
                f"{bx:6.1f}{by_gse:6.1f}{bz_gse:6.1f}{by_gsm:6.1f}{bz_gsm:6.1f}"
                f"{sigma_b:6.1f}{sigma_bv:6.1f}{sigma_bx:6.1f}{sigma_by:6.1f}{sigma_bz:6.1f}"
                f"{temp:9.0f}{density:6.1f}{speed:6.0f}{flow_lon:6.1f}{flow_lat:6.1f}"
                f"{na_np:6.3f}"
                f"{sigma_t:9.0f}{sigma_n:6.1f}{sigma_v:6.0f}"
                f"{sigma_phi:6.1f}{sigma_theta:6.1f}{sigma_ratio:6.3f}"
                f"{flow_p:7.2f}{e_field:7.2f}{plasma_beta:6.1f}{alfven_mach:6.1f}"
                f"{kp:3d}{sunspot:4d}{dst:6d}{ae:5d}"
                f" {pflux}"
                f"{flag:3d}{ap:4d}{f107:6.1f}{pc:6.1f}{al:6d}{au:6d}{ms_mach:5.1f}"
            )
            lines.append(line)

        with open(fname, "w") as f:
            f.write("\n".join(lines) + "\n")
        count += 1

    return count


def generate_goes_files(output_dir, scale_factor):
    """Generate new GOES CSV files."""
    goes_dir = os.path.join(output_dir, "STORM-AI", "warmup", "v2", "GOES")
    os.makedirs(goes_dir, exist_ok=True)

    rng = np.random.RandomState(RNG_SEED + 600)
    n_new = max(2, int(scale_factor))

    count = 0
    for i in range(n_new):
        wu_id = 716 + i
        wu_str = f"wu{wu_id:03d}" if wu_id < 1000 else f"wu{wu_id}"

        # Random 2020–2021 date range (~60 days)
        offset_days = rng.randint(0, 600)
        start = SAFE_START + timedelta(days=int(offset_days))
        end = start + timedelta(days=60)
        if end > SAFE_END:
            end = SAFE_END

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        fname = f"goes-{wu_str}-{start_str}_to_{end_str}.csv"
        fpath = os.path.join(goes_dir, fname)

        # 1-minute cadence
        n_minutes = int((end - start).total_seconds() / 60)
        timestamps = [start + timedelta(minutes=m) for m in range(n_minutes)]

        data = {
            "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        }

        for prefix in ["xrsa", "xrsb"]:
            base_flux = 1e-9 if prefix == "xrsa" else 8e-8
            flux = np.abs(rng.normal(base_flux, base_flux * 0.3, size=n_minutes))
            data[f"{prefix}_flux"] = flux
            data[f"{prefix}_flux_observed"] = flux
            data[f"{prefix}_flux_electrons"] = np.zeros(n_minutes)
            data[f"{prefix}_flag"] = np.full(n_minutes, 16.0)
            data[f"{prefix}_num"] = rng.randint(28, 31, size=n_minutes).astype(float)
            data[f"{prefix}_flag_excluded"] = np.zeros(n_minutes)

        df = pd.DataFrame(data)
        df.to_csv(fpath, index=False)
        count += 1

    return count


def generate_silso_files(output_dir, scale_factor):
    """Generate additional SILSO-style sunspot data files."""
    silso_dir = os.path.join(output_dir, "SILSO")
    os.makedirs(silso_dir, exist_ok=True)

    rng = np.random.RandomState(RNG_SEED + 700)

    # Monthly sunspot file
    fname = os.path.join(silso_dir, "SN_m_tot_V2.0.csv")
    if os.path.exists(fname):
        return 0

    lines = []
    for year in range(1749, 2025):
        for month in range(1, 13):
            year_frac = year + (month - 0.5) / 12.0
            # Simple solar cycle model: ~11 year period
            phase = 2 * math.pi * (year - 1749) / 11.0
            base = 60 + 80 * max(0, math.sin(phase))
            ssn = max(0, base + rng.normal(0, 20))
            std = rng.uniform(0.5, 15.0)
            obs = rng.randint(50, 300)
            prov = 1 if year < 2024 else 0
            lines.append(f"{year_frac:8.3f};{ssn:6.1f};{std:5.1f};{obs:6d};{prov}")

    with open(fname, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Daily sunspot file (larger)
    if scale_factor >= 5:
        fname2 = os.path.join(silso_dir, "SN_d_tot_V2.0.csv")
        lines2 = []
        dt = datetime(1818, 1, 1)
        end_dt = datetime(2025, 1, 1)
        while dt < end_dt:
            year_frac = dt.year + dt.timetuple().tm_yday / 365.25
            phase = 2 * math.pi * (dt.year - 1818) / 11.0
            base = 60 + 80 * max(0, math.sin(phase))
            ssn = max(0, base + rng.normal(0, 30))
            std = rng.uniform(0.5, 10.0)
            obs = rng.randint(10, 100)
            prov = 1 if dt.year < 2024 else 0
            lines2.append(
                f"{dt.year:4d};{dt.month:2d};{dt.day:2d};{year_frac:10.4f};"
                f"{ssn:6.1f};{std:5.1f};{obs:4d};{prov}"
            )
            dt += timedelta(days=1)
        with open(fname2, "w") as f:
            f.write("\n".join(lines2) + "\n")

    return 1


def generate_swarmb_files(output_dir, scale_factor):
    """Generate additional Swarm-B density files for 2020–2021 (safe range)."""
    swarmb_dir = os.path.join(output_dir, "swarmb")
    os.makedirs(swarmb_dir, exist_ok=True)

    rng = np.random.RandomState(RNG_SEED + 800)
    count = 0

    for year in [2020, 2021]:
        for month in range(1, 13):
            fname = f"SB_DNS_POD_{year}_{month:02d}_v02.txt"
            fpath = os.path.join(swarmb_dir, fname)
            if os.path.exists(fpath):
                continue

            header = (
                "# Column  1:  a27     Date/time (UTC)\n"
                "# Column  2:   f10.3  Altitude (m)\n"
                "# Column  3:  f13.8  Longitude (deg)\n"
                "# Column  4:  f13.8  Latitude (deg)\n"
                "# Column  5:   f6.3  Local solar time (h)\n"
                "# Column  6:  f13.8  Argument of latitude (deg)\n"
                "# Column  7:  e15.8  Density derived from GPS accelerations (kg/m3)\n"
                "# Fortran style format string: (a27,1x,f10.3,1X,f13.8,1X,f13.8,1X,f6.3,1X,f13.8,1X,e15.8)\n"
                "# Date/time                 alt        lon           lat           lst    arglat        dens_x         \n"
                "#                           m          deg           deg           h      deg           kg/m3          \n"
            )

            # Days in month
            if month == 12:
                next_month_start = datetime(year + 1, 1, 1)
            else:
                next_month_start = datetime(year, month + 1, 1)
            month_start = datetime(year, month, 1)
            n_days = (next_month_start - month_start).days

            lines = [header]
            # 30-second cadence
            for day in range(1, n_days + 1):
                for h in range(24):
                    for m in range(0, 60, 1):
                        for s in [0, 30]:
                            dt = datetime(year, month, day, h, m, s)
                            alt = rng.uniform(500000, 520000)
                            lon = rng.uniform(-180, 180)
                            lat = rng.uniform(-90, 90)
                            lst = (h + lon / 15.0) % 24.0
                            arglat = rng.uniform(0, 360)
                            density = rng.uniform(1e-13, 5e-13)

                            line = (
                                f"{dt.strftime('%Y-%m-%d %H:%M:%S.000')} UTC "
                                f"{alt:10.3f} {lon:13.8f} {lat:13.8f} "
                                f"{lst:6.3f} {arglat:13.8f}  {density:.8E}"
                            )
                            lines.append(line)

            with open(fpath, "w") as f:
                f.write("\n".join(lines) + "\n")
            count += 1

    return count


def generate_ancillary_files(output_dir, scale_factor):
    """Phase 3: Generate all ancillary files."""
    results = {}

    print("  Generating TLE files...")
    results["TLE"] = generate_tle_files(output_dir, scale_factor)

    print("  Generating geomag forecast files...")
    results["geomag"] = generate_geomag_files(output_dir, scale_factor)

    print("  Generating SP3 directories...")
    results["SP3"] = generate_sp3_directories(output_dir, scale_factor)

    print("  Generating OMNI2 .dat files...")
    results["OMNI2_dat"] = generate_omni2_dat_files(output_dir, scale_factor)

    print("  Generating GOES files...")
    results["GOES"] = generate_goes_files(output_dir, scale_factor)

    print("  Generating SILSO files...")
    results["SILSO"] = generate_silso_files(output_dir, scale_factor)

    print("  Generating Swarm-B density files...")
    results["SwarmB"] = generate_swarmb_files(output_dir, scale_factor)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Size Summary
# ──────────────────────────────────────────────────────────────────────────────

def get_dir_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def count_files(path):
    """Count total files in a directory."""
    total = 0
    for _dirpath, _dirnames, filenames in os.walk(path):
        total += len(filenames)
    return total


def format_size(size_bytes):
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inflate Kramabench astronomy dataset with synthetic data"
    )
    parser.add_argument("input_dir", help="Path to original data/astronomy/input/")
    parser.add_argument("output_dir", help="Path for inflated output")
    parser.add_argument(
        "--scale-factor", type=float, default=10,
        help="Target size multiplier (default: 10)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    scale_factor = args.scale_factor

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if scale_factor < 1:
        print("Error: scale-factor must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Measure original size
    print(f"Input directory: {input_dir}")
    orig_size = get_dir_size(input_dir)
    orig_files = count_files(input_dir)
    print(f"Original size: {format_size(orig_size)} ({orig_files} files)")
    print(f"Scale factor: {scale_factor}x")
    print()

    # Step 1: Copy input to output
    print("Step 1: Copying input directory to output...")
    if os.path.exists(output_dir):
        print(f"  Output directory already exists, removing: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    print(f"  Copied to: {output_dir}")
    print()

    # Step 2: Column inflation
    print("Step 2: Adding synthetic columns to existing CSVs...")
    inflate_columns(output_dir)
    col_size = get_dir_size(output_dir)
    print(f"  Size after column inflation: {format_size(col_size)}")
    print()

    # Step 3: New warmup files
    print("Step 3: Generating new warmup file pairs...")
    n_warmup = generate_warmup_files(output_dir, scale_factor)
    warmup_size = get_dir_size(output_dir)
    print(f"  Generated {n_warmup} new warmup pairs")
    print(f"  Size after warmup generation: {format_size(warmup_size)}")
    print()

    # Step 4: Ancillary files
    print("Step 4: Generating ancillary files...")
    ancillary_results = generate_ancillary_files(output_dir, scale_factor)
    print(f"  Generated: {ancillary_results}")
    print()

    # Summary
    final_size = get_dir_size(output_dir)
    final_files = count_files(output_dir)
    print("=" * 60)
    print("INFLATION SUMMARY")
    print("=" * 60)
    print(f"Original:  {format_size(orig_size):>12s}  ({orig_files:,d} files)")
    print(f"Inflated:  {format_size(final_size):>12s}  ({final_files:,d} files)")
    print(f"Ratio:     {final_size / orig_size:.1f}x size, {final_files / orig_files:.1f}x files")
    print("=" * 60)


if __name__ == "__main__":
    main()
