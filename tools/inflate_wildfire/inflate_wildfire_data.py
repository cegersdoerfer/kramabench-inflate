#!/usr/bin/env python3
"""
inflate_wildfire_data.py — Synthetic Dataset Inflation for Wildfire Domain

Inflates the Kramabench wildfire dataset by:
  1. Adding synthetic derived columns to existing CSVs (tasks reference columns
     by name, so extra columns are ignored)
  2. Generating new synthetic CSV files (distinct names, no task references them)
  3. Generating large observation datasets (scale-dependent)
  4. Generating additional GeoPackage files (optional, requires geopandas)

All original data is copied verbatim; only augmented copies and new files are added.
Original rows and column values are NEVER modified.

Usage:
    python inflate_wildfire_data.py <input_dir> <output_dir> --scale-factor N
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RNG_SEED = 42

# NIFC geographic area regions (used for synthetic data generation)
NIFC_REGIONS = [
    "Alaska", "Northwest", "Northern California", "Southern California",
    "Northern Rockies", "Great Basin", "Southwest", "Rocky Mountains",
    "Eastern Area", "Southern Area",
]

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

RAWS_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

RAWS_AGENCIES = ["BLM", "NPS", "USFS", "FWS", "BIA", "STATE", "DOD"]
RAWS_REGIONS = [
    "PACIFIC WEST", "INTERMOUNTAIN", "SOUTHEAST", "NORTHEAST",
    "NORTHERN ROCKIES", "GREAT BASIN", "SOUTHWEST", "ROCKY MOUNTAIN",
]


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_log10(series):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(series > 0, np.log10(series), np.nan)


def safe_sqrt(series):
    with np.errstate(invalid="ignore"):
        return np.where(series >= 0, np.sqrt(series), np.nan)


def safe_log(series):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(series > 0, np.log(series), np.nan)


def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series, window):
    return series.rolling(window=window, min_periods=1).std()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Column Inflation
# ──────────────────────────────────────────────────────────────────────────────

def inflate_noaa_wildfires(filepath):
    """Add derived columns to noaa_wildfires.csv."""
    df = pd.read_csv(filepath)

    # Math transforms
    if "hec" in df.columns:
        df["hec_log10"] = safe_log10(df["hec"])
        df["hec_sqrt"] = safe_sqrt(df["hec"])
        df["hec_squared"] = df["hec"] ** 2
        df["is_large_fire"] = (df["hec"] > 1000).astype(int)

    if "wind_med" in df.columns:
        df["wind_speed_squared"] = df["wind_med"] ** 2
        df["is_high_wind"] = (df["wind_med"] > 20).astype(int)

    if "duration" in df.columns:
        df["duration_log"] = safe_log(df["duration"])
        df["duration_squared"] = df["duration"] ** 2

    # Interactions
    if "erc_med" in df.columns and "wind_med" in df.columns:
        df["erc_wind_interaction"] = df["erc_med"] * df["wind_med"]

    if "wind_med" in df.columns and "erc_med" in df.columns and "avrh_mean" in df.columns:
        df["fire_intensity_proxy"] = df["wind_med"] * df["erc_med"] / (df["avrh_mean"] + 1)

    # Coordinate transforms
    if "latitude" in df.columns:
        df["latitude_rad"] = np.radians(df["latitude"])
    if "longitude" in df.columns:
        df["longitude_rad"] = np.radians(df["longitude"])

    # Rolling means
    for col in ["avrh_mean", "wind_med", "erc_med", "rain_sum"]:
        if col in df.columns:
            for w in [3, 5, 10]:
                df[f"{col}_rolling_mean_{w}"] = rolling_mean(df[col], w)

    # Lagged variants
    for col in ["wind_med", "erc_med"]:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Rolling std
    for col in ["avrh_mean", "wind_med"]:
        if col in df.columns:
            df[f"{col}_rolling_std_5"] = rolling_std(df[col], 5)

    # Derived ratios
    if "prim_threatened_aggregate" in df.columns and "hec" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["threatened_per_hectare"] = np.where(
                df["hec"] > 0,
                df["prim_threatened_aggregate"] / df["hec"],
                np.nan,
            )

    if "injuries_to_date_last" in df.columns and "duration" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["injuries_per_day"] = np.where(
                df["duration"] > 0,
                df["injuries_to_date_last"] / df["duration"],
                np.nan,
            )

    df.to_csv(filepath, index=False)


def inflate_fire_weather(filepath):
    """Add derived columns to Fire_Weather_Data_2002-2014_2016.csv."""
    df = pd.read_csv(filepath)

    if "hec" in df.columns:
        df["hec_log10"] = safe_log10(df["hec"])
        df["hec_sqrt"] = safe_sqrt(df["hec"])
        df["is_large_fire"] = (df["hec"] > 1000).astype(int)

    if "wind_med" in df.columns:
        df["wind_speed_squared"] = df["wind_med"] ** 2
        df["is_high_wind"] = (df["wind_med"] > 20).astype(int)

    if "duration" in df.columns:
        df["duration_log"] = safe_log(df["duration"])
        df["duration_squared"] = df["duration"] ** 2

    if "erc_med" in df.columns and "wind_med" in df.columns:
        df["erc_wind_interaction"] = df["erc_med"] * df["wind_med"]

    if "wind_med" in df.columns and "erc_med" in df.columns and "avrh_mean" in df.columns:
        df["fire_intensity_proxy"] = df["wind_med"] * df["erc_med"] / (df["avrh_mean"] + 1)

    if "latitude" in df.columns:
        df["latitude_rad"] = np.radians(df["latitude"])
    if "longitude" in df.columns:
        df["longitude_rad"] = np.radians(df["longitude"])

    for col in ["avrh_mean", "wind_med", "erc_med", "rain_sum"]:
        if col in df.columns:
            for w in [3, 5, 10]:
                df[f"{col}_rolling_mean_{w}"] = rolling_mean(df[col], w)

    for col in ["wind_med", "erc_med"]:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for col in ["avrh_mean", "wind_med"]:
        if col in df.columns:
            df[f"{col}_rolling_std_5"] = rolling_std(df[col], 5)

    if "injuries_to_date_last" in df.columns and "duration" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["injuries_per_day"] = np.where(
                df["duration"] > 0,
                df["injuries_to_date_last"] / df["duration"],
                np.nan,
            )

    if "fatalities_last" in df.columns and "duration" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["fatalities_per_day"] = np.where(
                df["duration"] > 0,
                df["fatalities_last"] / df["duration"],
                np.nan,
            )

    df.to_csv(filepath, index=False)


def inflate_raws(filepath):
    """Add derived columns to PublicView_RAWS_*.csv."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # Extract numeric values from string columns
    def extract_numeric(series):
        return pd.to_numeric(series.astype(str).str.extract(r"([\d.]+)", expand=False), errors="coerce")

    if "Elevation" in df.columns:
        elev = pd.to_numeric(df["Elevation"], errors="coerce")
        df["Elevation_m"] = elev * 0.3048

    wind_col = [c for c in df.columns if "Wind Speed MPH" in c]
    if wind_col:
        wind_mph = extract_numeric(df[wind_col[0]])
        df["Wind_Speed_ms"] = wind_mph * 0.44704
        df["Wind_Speed_kph"] = wind_mph * 1.60934
        df["Wind_Speed_knots"] = wind_mph * 0.868976

    temp_col = [c for c in df.columns if "Air Temperature" in c]
    if temp_col:
        temp_f = extract_numeric(df[temp_col[0]])
        df["Temp_C"] = (temp_f - 32) * 5 / 9
        df["Temp_K"] = (temp_f - 32) * 5 / 9 + 273.15

    rh_col = [c for c in df.columns if "Relative Humidity" in c]
    if rh_col:
        rh = extract_numeric(df[rh_col[0]])
        df["RH_decimal"] = rh / 100.0

    # Heat index proxy (simplified)
    if temp_col and rh_col:
        temp_f_vals = extract_numeric(df[temp_col[0]])
        rh_vals = extract_numeric(df[rh_col[0]])
        df["Heat_Index_proxy"] = 0.5 * (temp_f_vals + 61.0 + (temp_f_vals - 68.0) * 1.2 + rh_vals * 0.094)

    # Wind chill proxy (simplified)
    if temp_col and wind_col:
        temp_f_vals = extract_numeric(df[temp_col[0]])
        wind_mph_vals = extract_numeric(df[wind_col[0]])
        df["Wind_Chill_proxy"] = (
            35.74 + 0.6215 * temp_f_vals
            - 35.75 * (wind_mph_vals ** 0.16)
            + 0.4275 * temp_f_vals * (wind_mph_vals ** 0.16)
        )

    if "Latitude" in df.columns:
        df["Latitude_rad"] = np.radians(pd.to_numeric(df["Latitude"], errors="coerce"))
    if "Longitude" in df.columns:
        df["Longitude_rad"] = np.radians(pd.to_numeric(df["Longitude"], errors="coerce"))

    # Solar radiation derived
    solar_col = [c for c in df.columns if "Solar Radiation" in c]
    if solar_col:
        solar = extract_numeric(df[solar_col[0]])
        df["Solar_Radiation_kJ_m2"] = solar * 3.6  # W/m2 to kJ/m2 (per hour approx)

    # Fuel moisture transforms
    fuel_col = [c for c in df.columns if "Fuel Moisture" in c and "Fuel Temperature" not in c]
    if fuel_col:
        fm = extract_numeric(df[fuel_col[0]])
        df["Fuel_Moisture_log"] = safe_log(fm)
        df["Fuel_Moisture_squared"] = fm ** 2

    # Battery health indicator
    batt_col = [c for c in df.columns if "Battery" in c]
    if batt_col:
        batt = extract_numeric(df[batt_col[0]])
        df["Battery_Low"] = (batt < 12.0).astype(int)

    df.to_csv(filepath, index=False)


def inflate_aqi(filepath):
    """Add derived columns to annual_aqi_by_county_2024.csv."""
    df = pd.read_csv(filepath)

    if "Max AQI" in df.columns and "Median AQI" in df.columns:
        df["AQI_range"] = df["Max AQI"] - df["Median AQI"]
        df["AQI_range_ratio"] = df["AQI_range"] / (df["Median AQI"] + 1)

    if "Days with AQI" in df.columns:
        days = df["Days with AQI"]
        for col in ["Good Days", "Moderate Days", "Unhealthy for Sensitive Groups Days",
                     "Unhealthy Days", "Very Unhealthy Days", "Hazardous Days"]:
            if col in df.columns:
                short = col.replace(" Days", "").replace(" ", "_").replace("for_Sensitive_Groups_", "SG_")
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[f"{short}_ratio"] = np.where(days > 0, df[col] / days, np.nan)

        unhealthy_cols = ["Unhealthy Days", "Very Unhealthy Days", "Hazardous Days"]
        present = [c for c in unhealthy_cols if c in df.columns]
        if present:
            df["Total_Unhealthy_Days"] = df[present].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["Unhealthy_ratio"] = np.where(days > 0, df["Total_Unhealthy_Days"] / days, np.nan)

    if "90th Percentile AQI" in df.columns and "Median AQI" in df.columns:
        df["AQI_skew_proxy"] = df["90th Percentile AQI"] - df["Median AQI"]

    if "Days CO" in df.columns and "Days NO2" in df.columns:
        df["CO_NO2_ratio"] = np.where(
            df["Days NO2"] > 0, df["Days CO"] / df["Days NO2"], np.nan
        )

    if "Days Ozone" in df.columns and "Days PM2.5" in df.columns:
        df["Ozone_PM25_ratio"] = np.where(
            df["Days PM2.5"] > 0, df["Days Ozone"] / df["Days PM2.5"], np.nan
        )

    df.to_csv(filepath, index=False)


def inflate_generic_csv(filepath):
    """Add derived columns to any CSV with numeric data."""
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return

    added = False
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        vals = df[col]
        if vals.nunique() > 2:  # Skip near-constant columns
            df[f"{col}_squared"] = vals ** 2
            df[f"{col}_log"] = safe_log(vals.clip(lower=0.001))
            df[f"{col}_normalized"] = (vals - vals.mean()) / (vals.std() + 1e-10)
            added = True

    if added:
        df.to_csv(filepath, index=False)


def inflate_columns(output_dir):
    """Phase 1: Add synthetic columns to eligible CSVs."""
    print("  Processing noaa_wildfires.csv...")
    noaa_path = os.path.join(output_dir, "noaa_wildfires.csv")
    if os.path.isfile(noaa_path):
        inflate_noaa_wildfires(noaa_path)

    print("  Processing Fire_Weather_Data_2002-2014_2016.csv...")
    fwd_path = os.path.join(output_dir, "Fire_Weather_Data_2002-2014_2016.csv")
    if os.path.isfile(fwd_path):
        inflate_fire_weather(fwd_path)

    print("  Processing PublicView_RAWS_*.csv...")
    for fname in os.listdir(output_dir):
        if fname.startswith("PublicView_RAWS_") and fname.endswith(".csv"):
            inflate_raws(os.path.join(output_dir, fname))

    print("  Processing annual_aqi_by_county_2024.csv...")
    aqi_path = os.path.join(output_dir, "annual_aqi_by_county_2024.csv")
    if os.path.isfile(aqi_path):
        inflate_aqi(aqi_path)

    # Generic inflation for remaining CSVs that are safe to modify
    # (we skip noaa_wildfires_monthly_stats.csv because of its 3-row header)
    skip_files = {
        "noaa_wildfires.csv",
        "Fire_Weather_Data_2002-2014_2016.csv",
        "annual_aqi_by_county_2024.csv",
        "noaa_wildfires_monthly_stats.csv",  # 3 header rows, special format
        "noaa_wildfires_variabledescrip.csv",  # metadata dictionary
    }
    skip_prefixes = ("PublicView_RAWS_",)

    print("  Processing other CSVs...")
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".csv"):
            continue
        if fname in skip_files:
            continue
        if any(fname.startswith(p) for p in skip_prefixes):
            continue
        fpath = os.path.join(output_dir, fname)
        # Tab-separated files need special handling
        try:
            # Try to detect delimiter
            with open(fpath, "r") as f:
                first_line = f.readline()
            if "\t" in first_line:
                # Tab-separated NIFC files — skip generic inflation
                # (they have formatted numbers with commas and $)
                continue
            inflate_generic_csv(fpath)
        except Exception as e:
            print(f"    Warning: could not inflate {fname}: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: New Synthetic CSV Files
# ──────────────────────────────────────────────────────────────────────────────

def generate_raws_station_data(output_dir, scale_factor):
    """Generate synthetic RAWS station data files for additional years."""
    rng = np.random.RandomState(RNG_SEED + 100)
    n_years = max(1, int(scale_factor))
    count = 0

    for year_offset in range(n_years):
        year = 2020 + year_offset
        fname = f"synthetic_RAWS_station_data_{year}.csv"
        fpath = os.path.join(output_dir, fname)

        n_stations = rng.randint(200, 500)
        rows = []
        for i in range(n_stations):
            station_name = f"SYNTH_STATION_{year}_{i:04d}"
            wx_id = rng.randint(10000000, 99999999)
            obj_id = 100000 + year_offset * 1000 + i
            elev = rng.randint(100, 12000)
            lat = rng.uniform(25.0, 49.0)
            lon = rng.uniform(-125.0, -67.0)
            state = rng.choice(RAWS_STATES)
            agency = rng.choice(RAWS_AGENCIES)
            region = rng.choice(RAWS_REGIONS)
            wind_mph = rng.uniform(0, 40)
            wind_dir = rng.randint(0, 360)
            temp_f = rng.uniform(10, 110)
            fuel_temp = temp_f + rng.uniform(-5, 10)
            rh = rng.uniform(5, 100)
            batt = rng.uniform(11.5, 14.0)
            fuel_moist = rng.uniform(3, 30)
            rain = rng.uniform(0, 5)
            solar = rng.uniform(0, 1200)
            peak_dir = rng.randint(0, 360)
            peak_mph = wind_mph + rng.uniform(0, 15)

            rows.append({
                "OBJECTID": obj_id,
                "Station Name": station_name,
                "WX ID": wx_id,
                "Ob Time": f"6/15/{year} 3:00:00 PM",
                "NESS ID": "",
                "NWS ID": "",
                "Elevation": elev,
                "Site Description": "",
                "Latitude": round(lat, 5),
                "Longitude": round(lon, 5),
                "State": state,
                "County": "",
                "Agency": agency,
                "Region": region,
                "Unit": "",
                "Sub-Unit": "",
                "Status": "A",
                "Rain Accumulation": f"{rain:.1f} inches",
                "Wind Speed MPH": f"{wind_mph:.0f} mph",
                "Wind Direction Degrees": f"{wind_dir} degrees",
                "Air Temperaturee, Standard Placement": f"{temp_f:.0f} deg. F",
                "Fuel Temperature": f"{fuel_temp:.0f} deg. F",
                "Relative Humidity": f"{rh:.0f} %",
                "Battery Voltage": f"{batt:.1f} volts",
                "Fuel Moisture": f"{fuel_moist:.1f} (unk)",
                "Wind Direction, Peak": f"{peak_dir} degrees",
                "Wind Speed Peak MPH": f"{peak_mph:.0f} mph",
                "Solar Radiation": f"{solar:.0f} w/m2",
                "Station ID": wx_id - 2,
                "MesoWest Station ID": f"SYN{state}{i:03d}",
                "MesoWest Detailed Weather Link": "",
                "NOAA Detailed Weather Link": "",
                "x": round(lon, 5),
                "y": round(lat, 5),
            })

        df = pd.DataFrame(rows)
        df.to_csv(fpath, index=False)
        count += 1

    return count


def generate_fire_weather_observations(output_dir, scale_factor):
    """Generate synthetic fire weather observation files."""
    rng = np.random.RandomState(RNG_SEED + 200)
    n_files = max(1, int(scale_factor / 2))
    count = 0

    regions = [
        "California", "Oregon", "Washington", "Idaho", "Montana",
        "Colorado", "Arizona", "New Mexico", "Nevada", "Utah",
    ]
    states = ["CA", "OR", "WA", "ID", "MT", "CO", "AZ", "NM", "NV", "UT"]
    causes = ["H", "L", "U", "N"]
    strategies = ["Full Suppression", "Point/Zone Protection", "Monitor/Modify"]

    for file_idx in range(n_files):
        year_start = 2015 + file_idx
        year_end = min(year_start + 2, 2024)
        fname = f"synthetic_fire_weather_observations_{year_start}_{year_end}.csv"
        fpath = os.path.join(output_dir, fname)

        n_records = rng.randint(500, 2000)
        rows = []
        for i in range(n_records):
            year = rng.randint(year_start, year_end + 1)
            reg_idx = rng.randint(0, len(regions))
            region = regions[reg_idx]
            state = states[reg_idx]
            avrh = rng.uniform(10, 80)
            wind = rng.uniform(0, 35)
            erc = rng.uniform(10, 95)
            rain = rng.uniform(0, 2)
            duration = max(1, int(rng.exponential(15)))
            hec = max(0, rng.exponential(5000))
            lat = rng.uniform(30, 49)
            lon = rng.uniform(-125, -100)
            start_doy = rng.randint(1, 366)
            cause = rng.choice(causes)

            rows.append({
                "start_year": year,
                "region_ind": reg_idx + 1,
                "incident_number": f"SYN-{state}-{year}-{i:05d}",
                "avrh_mean": round(avrh, 1),
                "wind_med": round(wind, 1),
                "erc_med": round(erc, 1),
                "rain_sum": round(rain, 2),
                "region": region,
                "state": state,
                "incident_name": f"synthetic_fire_{i}",
                "cause": cause,
                "dominant_strategy_25_s": rng.choice(strategies),
                "dominant_strategy_50_s": rng.choice(strategies),
                "dominant_strategy_75_s": rng.choice(strategies),
                "subdom_strategy": "",
                "start_date": f"{rng.randint(1,13)}/{rng.randint(1,29)}/{year}",
                "controlled_date": f"{rng.randint(1,13)}/{rng.randint(1,29)}/{year}",
                "duration": duration,
                "prim_threatened_aggregate": int(rng.exponential(500)),
                "comm_threatened_aggregate": int(rng.exponential(100)),
                "outb_threatened_aggregate": int(rng.exponential(200)),
                "injuries_to_date_last": rng.randint(0, 5),
                "fatalities_last": rng.randint(0, 2),
                "latitude": round(lat, 0),
                "longitude": round(lon, 0),
                "start_day_of_year": start_doy,
                "control_year": year,
                "control_day_of_year": min(365, start_doy + duration),
                "gt_100": int(hec > 100),
                "dom_strat_ind_75": rng.randint(0, 3),
                "dom_strat_ind_25": rng.randint(0, 3),
                "dom_strat_ind_50": rng.randint(0, 3),
                "station_verified_in_psa": rng.randint(40000, 50000),
                "hec": round(hec, 0),
                "cause_ind": rng.randint(0, 2),
                "total_fire_region": rng.randint(50, 500),
                "total_fire_west": rng.randint(200, 800),
            })

        df = pd.DataFrame(rows)
        df.to_csv(fpath, index=False)
        count += 1

    return count


def generate_aqi_files(output_dir, scale_factor):
    """Generate synthetic AQI files for additional years."""
    rng = np.random.RandomState(RNG_SEED + 300)

    # Read existing 2024 AQI to get county list
    aqi_2024 = os.path.join(output_dir, "annual_aqi_by_county_2024.csv")
    if not os.path.isfile(aqi_2024):
        return 0

    df_template = pd.read_csv(aqi_2024)
    counties = df_template[["State", "County"]].values.tolist()

    n_years = min(4, max(1, int(scale_factor / 2)))
    count = 0

    for year_offset in range(n_years):
        year = 2020 + year_offset
        fname = f"synthetic_annual_aqi_by_county_{year}.csv"
        fpath = os.path.join(output_dir, fname)

        rows = []
        for state, county in counties:
            days = rng.randint(180, 366)
            good = int(days * rng.uniform(0.5, 0.9))
            moderate = int(days * rng.uniform(0.05, 0.3))
            usg = max(0, days - good - moderate - rng.randint(0, 10))
            unhealthy = rng.randint(0, 5)
            very_unhealthy = rng.randint(0, 2)
            hazardous = rng.randint(0, 1)
            max_aqi = rng.randint(50, 300)
            p90 = rng.randint(30, 150)
            median = rng.randint(20, 70)

            rows.append({
                "State": state,
                "County": county,
                "Year": year,
                "Days with AQI": days,
                "Good Days": good,
                "Moderate Days": moderate,
                "Unhealthy for Sensitive Groups Days": usg,
                "Unhealthy Days": unhealthy,
                "Very Unhealthy Days": very_unhealthy,
                "Hazardous Days": hazardous,
                "Max AQI": max_aqi,
                "90th Percentile AQI": p90,
                "Median AQI": median,
                "Days CO": rng.randint(0, 5),
                "Days NO2": rng.randint(0, 20),
                "Days Ozone": rng.randint(0, 200),
                "Days PM2.5": rng.randint(0, 300),
                "Days PM10": rng.randint(0, 50),
            })

        df = pd.DataFrame(rows)
        df.to_csv(fpath, index=False, quoting=1)  # Quote all strings like original
        count += 1

    return count


def generate_nifc_style_files(output_dir, scale_factor):
    """Generate new NIFC-style tab-separated files."""
    rng = np.random.RandomState(RNG_SEED + 400)
    count = 0

    regions_with_wgb = NIFC_REGIONS[:6] + ["Western Great Basin"] + NIFC_REGIONS[6:]

    # Prescribed fire acres
    fname = os.path.join(output_dir, "synthetic_nifc_prescribed_fire_acres.csv")
    header = "Year\t" + "\t".join(regions_with_wgb) + "\tTotal\n"
    lines = [header]
    for year in range(2024, 2000, -1):
        vals = []
        for _ in regions_with_wgb:
            v = int(rng.exponential(50000))
            vals.append(f"{v:,}")
        total = sum(int(v.replace(",", "")) for v in vals)
        lines.append(f"{year}\t" + "\t".join(vals) + f"\t{total:,}\n")
    with open(fname, "w") as f:
        f.writelines(lines)
    count += 1

    # Wildland fire use acres
    fname = os.path.join(output_dir, "synthetic_nifc_wildland_fire_use_acres.csv")
    lines = [header]
    for year in range(2024, 2000, -1):
        vals = []
        for _ in regions_with_wgb:
            v = int(rng.exponential(30000))
            vals.append(f"{v:,}")
        total = sum(int(v.replace(",", "")) for v in vals)
        lines.append(f"{year}\t" + "\t".join(vals) + f"\t{total:,}\n")
    with open(fname, "w") as f:
        f.writelines(lines)
    count += 1

    return count


def generate_state_level_files(output_dir, scale_factor):
    """Generate synthetic state-level wildfire data files."""
    rng = np.random.RandomState(RNG_SEED + 500)
    count = 0

    # Wildfire Injuries by State
    rows = []
    for state in US_STATES:
        rows.append({
            "State": state,
            "Total Injuries": int(rng.exponential(50)),
            "Population": int(rng.uniform(500000, 40000000)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "synthetic_Wildfire_Injuries_by_State.csv"), index=False)
    count += 1

    # Wildfire Structures Destroyed by State
    rows = []
    for state in US_STATES:
        rows.append({
            "State": state,
            "Total Structures Destroyed": int(rng.exponential(200)),
            "Residential": int(rng.exponential(150)),
            "Commercial": int(rng.exponential(30)),
            "Other": int(rng.exponential(20)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "synthetic_Wildfire_Structures_by_State.csv"), index=False)
    count += 1

    # Wildfire Duration by State
    rows = []
    for state in US_STATES:
        rows.append({
            "State": state,
            "Avg Duration Days": round(rng.uniform(3, 45), 1),
            "Max Duration Days": int(rng.uniform(30, 365)),
            "Median Duration Days": round(rng.uniform(2, 20), 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "synthetic_Wildfire_Duration_by_State.csv"), index=False)
    count += 1

    return count


def generate_monthly_seasonal_files(output_dir, scale_factor):
    """Generate synthetic monthly/seasonal summary files."""
    rng = np.random.RandomState(RNG_SEED + 600)
    count = 0

    # Seasonal stats
    rows = []
    for year in range(2000, 2025):
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            rows.append({
                "Year": year,
                "Season": season,
                "Total_Acres": int(rng.exponential(500000)),
                "Total_Fires": int(rng.exponential(5000)),
                "Avg_Duration": round(rng.uniform(5, 30), 1),
                "Max_Fire_Size_Acres": int(rng.exponential(100000)),
                "Avg_Wind_Speed": round(rng.uniform(3, 15), 1),
                "Avg_Humidity": round(rng.uniform(20, 70), 1),
                "Total_Rainfall": round(rng.exponential(3), 2),
                "Avg_ERC": round(rng.uniform(20, 80), 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "synthetic_noaa_wildfires_seasonal_stats.csv"), index=False)
    count += 1

    # Annual summary
    rows = []
    for year in range(2000, 2025):
        rows.append({
            "Year": year,
            "Total_Acres": int(rng.exponential(3000000)),
            "Total_Fires": int(rng.exponential(50000)),
            "Avg_Fire_Size": round(rng.uniform(10, 200), 1),
            "Median_Fire_Size": round(rng.uniform(5, 50), 1),
            "Largest_Fire_Acres": int(rng.exponential(500000)),
            "Total_Suppression_Cost": int(rng.exponential(1500000000)),
            "Human_Caused_Pct": round(rng.uniform(0.3, 0.7), 3),
            "Lightning_Caused_Pct": round(rng.uniform(0.2, 0.5), 3),
            "Avg_Duration_Days": round(rng.uniform(8, 25), 1),
            "Peak_Month": rng.choice(["June", "July", "August", "September"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "synthetic_noaa_wildfires_annual_summary.csv"), index=False)
    count += 1

    return count


def generate_synthetic_csvs(output_dir, scale_factor):
    """Phase 2: Generate all new synthetic CSV files."""
    results = {}

    print("  Generating synthetic RAWS station data...")
    results["RAWS_stations"] = generate_raws_station_data(output_dir, scale_factor)

    print("  Generating synthetic fire weather observations...")
    results["fire_weather_obs"] = generate_fire_weather_observations(output_dir, scale_factor)

    print("  Generating synthetic AQI files...")
    results["AQI"] = generate_aqi_files(output_dir, scale_factor)

    print("  Generating synthetic NIFC files...")
    results["NIFC"] = generate_nifc_style_files(output_dir, scale_factor)

    print("  Generating synthetic state-level files...")
    results["state_level"] = generate_state_level_files(output_dir, scale_factor)

    print("  Generating synthetic monthly/seasonal files...")
    results["monthly_seasonal"] = generate_monthly_seasonal_files(output_dir, scale_factor)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Large Observation Datasets
# ──────────────────────────────────────────────────────────────────────────────

def generate_hourly_weather_station_data(output_dir, scale_factor):
    """Generate large hourly weather station observation files."""
    rng = np.random.RandomState(RNG_SEED + 700)

    n_stations = max(10, int(scale_factor * 10))
    n_years = max(1, min(10, int(scale_factor / 2)))
    start_year = 2015
    end_year = start_year + n_years

    fname = f"synthetic_weather_station_hourly_{start_year}_{end_year - 1}.csv"
    fpath = os.path.join(output_dir, fname)

    print(f"    Generating {n_stations} stations x {n_years} years of hourly data...")

    # Generate station metadata
    station_ids = [f"WS{i:04d}" for i in range(n_stations)]
    station_lats = rng.uniform(25.0, 49.0, size=n_stations)
    station_lons = rng.uniform(-125.0, -67.0, size=n_stations)
    station_elevs = rng.uniform(50, 10000, size=n_stations)
    station_states = [rng.choice(RAWS_STATES) for _ in range(n_stations)]

    # Write in chunks to avoid memory issues
    chunk_size = 100000
    first_chunk = True
    total_rows = 0

    for year in range(start_year, end_year):
        n_hours = 366 * 24 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365 * 24

        for station_idx in range(n_stations):
            # Generate data for this station-year
            hours = np.arange(n_hours)
            doy = hours // 24 + 1
            hour_of_day = hours % 24

            # Seasonal temperature pattern
            temp_base = 50 + 30 * np.sin(2 * np.pi * (doy - 80) / 365)
            temp_diurnal = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            temp_f = temp_base + temp_diurnal + rng.normal(0, 5, size=n_hours)
            temp_f += (station_elevs[station_idx] - 5000) * (-0.003)  # Lapse rate

            # Wind
            wind_mph = np.abs(rng.normal(8, 5, size=n_hours))

            # Humidity (inverse correlation with temperature)
            rh = np.clip(80 - 0.5 * temp_f + rng.normal(0, 10, size=n_hours), 5, 100)

            # ERC
            erc = np.clip(40 + 0.3 * temp_f - 0.5 * rh + rng.normal(0, 10, size=n_hours), 0, 100)

            # Rain (sparse)
            rain = np.where(rng.random(n_hours) < 0.05, rng.exponential(0.1, size=n_hours), 0)

            # Solar radiation
            solar = np.clip(
                800 * np.sin(np.pi * hour_of_day / 24) * np.where(
                    (hour_of_day >= 6) & (hour_of_day <= 18), 1, 0
                ) + rng.normal(0, 50, size=n_hours),
                0, 1400,
            )

            chunk_data = {
                "station_id": station_ids[station_idx],
                "year": year,
                "day_of_year": doy,
                "hour": hour_of_day,
                "latitude": round(station_lats[station_idx], 5),
                "longitude": round(station_lons[station_idx], 5),
                "elevation_ft": round(station_elevs[station_idx], 0),
                "state": station_states[station_idx],
                "temperature_f": np.round(temp_f, 1),
                "temperature_c": np.round((temp_f - 32) * 5 / 9, 1),
                "relative_humidity_pct": np.round(rh, 1),
                "wind_speed_mph": np.round(wind_mph, 1),
                "wind_direction_deg": rng.randint(0, 360, size=n_hours),
                "erc": np.round(erc, 1),
                "rainfall_inches": np.round(rain, 3),
                "solar_radiation_wm2": np.round(solar, 1),
                "fuel_moisture_pct": np.round(np.clip(rh * 0.3 + rng.normal(0, 3, size=n_hours), 2, 40), 1),
                "fire_danger_index": np.round(np.clip(erc * (1 - rh / 100) * (wind_mph / 10 + 1), 0, 200), 1),
            }
            chunk_df = pd.DataFrame(chunk_data)
            chunk_df.to_csv(fpath, mode="a", header=first_chunk, index=False)
            first_chunk = False
            total_rows += len(chunk_df)

        print(f"      Year {year} complete ({total_rows:,} rows so far)")

    return total_rows


def generate_satellite_fire_detections(output_dir, scale_factor):
    """Generate synthetic satellite fire detection data (MODIS/VIIRS style)."""
    rng = np.random.RandomState(RNG_SEED + 800)

    n_years = max(1, min(5, int(scale_factor / 2)))
    detections_per_year = max(10000, int(scale_factor * 50000))

    fname = f"synthetic_satellite_fire_detections_2020_{2019 + n_years}.csv"
    fpath = os.path.join(output_dir, fname)

    print(f"    Generating {detections_per_year * n_years:,} fire detections...")

    first_chunk = True
    total_rows = 0

    for year_offset in range(n_years):
        year = 2020 + year_offset

        # Fire season peaks in summer
        doy = rng.choice(
            np.arange(1, 366),
            size=detections_per_year,
            p=np.array([
                max(0.001, 0.003 * np.exp(-0.5 * ((d - 210) / 50) ** 2))
                for d in range(1, 366)
            ]) / sum(
                max(0.001, 0.003 * np.exp(-0.5 * ((d - 210) / 50) ** 2))
                for d in range(1, 366)
            ),
        )

        lat = rng.uniform(25.0, 49.0, size=detections_per_year)
        lon = rng.uniform(-125.0, -67.0, size=detections_per_year)

        # Higher confidence in summer, western US
        base_conf = 50 + 30 * np.exp(-0.5 * ((doy - 210) / 50) ** 2)
        confidence = np.clip(base_conf + rng.normal(0, 15, size=detections_per_year), 0, 100).astype(int)

        # Fire radiative power
        frp = np.abs(rng.exponential(30, size=detections_per_year))

        brightness = 300 + frp * 2 + rng.normal(0, 10, size=detections_per_year)
        bright_t31 = brightness - rng.uniform(5, 30, size=detections_per_year)

        scan = rng.uniform(1.0, 2.5, size=detections_per_year)
        track = rng.uniform(1.0, 2.5, size=detections_per_year)

        data = {
            "year": year,
            "day_of_year": doy,
            "latitude": np.round(lat, 4),
            "longitude": np.round(lon, 4),
            "brightness_K": np.round(brightness, 1),
            "brightness_T31_K": np.round(bright_t31, 1),
            "scan_pixel_size": np.round(scan, 1),
            "track_pixel_size": np.round(track, 1),
            "confidence_pct": confidence,
            "frp_MW": np.round(frp, 1),
            "satellite": rng.choice(["TERRA", "AQUA", "NOAA20", "SNPP"], size=detections_per_year),
            "instrument": rng.choice(["MODIS", "VIIRS"], size=detections_per_year),
            "day_night": rng.choice(["D", "N"], size=detections_per_year, p=[0.65, 0.35]),
            "type": rng.choice([0, 1, 2, 3], size=detections_per_year, p=[0.7, 0.15, 0.1, 0.05]),
        }

        chunk_df = pd.DataFrame(data)
        chunk_df.to_csv(fpath, mode="a", header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(chunk_df)

    return total_rows


def generate_large_observations(output_dir, scale_factor):
    """Phase 3: Generate large observation datasets."""
    results = {}

    if scale_factor >= 3:
        print("  Generating hourly weather station data...")
        results["hourly_weather"] = generate_hourly_weather_station_data(output_dir, scale_factor)

    print("  Generating satellite fire detections...")
    results["satellite_detections"] = generate_satellite_fire_detections(output_dir, scale_factor)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: GeoPackage Expansion
# ──────────────────────────────────────────────────────────────────────────────

def generate_geopackages(output_dir, scale_factor):
    """Phase 4: Generate additional GeoPackage files with synthetic geometry."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point, Polygon, box
    except ImportError:
        print("  geopandas/shapely not installed, skipping GeoPackage generation")
        return {}

    rng = np.random.RandomState(RNG_SEED + 900)
    results = {}

    # Generate synthetic county-level GeoPackage
    print("  Generating synthetic county GeoPackage...")
    n_counties = max(100, int(scale_factor * 50))
    county_rows = []
    for i in range(n_counties):
        lat = rng.uniform(25, 49)
        lon = rng.uniform(-125, -67)
        size = rng.uniform(0.1, 0.5)
        geom = box(lon, lat, lon + size, lat + size * 0.7)
        county_rows.append({
            "geometry": geom,
            "county_id": i,
            "county_name": f"Synthetic County {i}",
            "state": rng.choice(US_STATES),
            "area_sq_mi": round(rng.uniform(100, 5000), 1),
            "population": int(rng.exponential(50000)),
            "fire_risk_score": round(rng.uniform(0, 100), 1),
        })

    gdf = gpd.GeoDataFrame(county_rows, crs="EPSG:4326")
    gdf.to_file(os.path.join(output_dir, "synthetic_usa_counties.gpkg"), driver="GPKG")
    results["counties"] = n_counties

    # Generate synthetic fire district GeoPackage
    print("  Generating synthetic fire district GeoPackage...")
    n_districts = max(50, int(scale_factor * 20))
    district_rows = []
    for i in range(n_districts):
        lat = rng.uniform(25, 49)
        lon = rng.uniform(-125, -67)
        size = rng.uniform(0.2, 1.0)
        geom = box(lon, lat, lon + size, lat + size * 0.6)
        district_rows.append({
            "geometry": geom,
            "district_id": i,
            "district_name": f"Fire District {i}",
            "region": rng.choice(NIFC_REGIONS),
            "station_count": int(rng.uniform(1, 20)),
            "coverage_sq_mi": round(rng.uniform(500, 20000), 1),
            "avg_response_time_min": round(rng.uniform(5, 60), 1),
        })

    gdf = gpd.GeoDataFrame(district_rows, crs="EPSG:4326")
    gdf.to_file(os.path.join(output_dir, "synthetic_usa_fire_districts.gpkg"), driver="GPKG")
    results["districts"] = n_districts

    # Generate detailed sub-region GeoPackage
    print("  Generating synthetic NIFC sub-region GeoPackage...")
    n_subregions = max(30, int(scale_factor * 15))
    subregion_rows = []
    for i in range(n_subregions):
        lat = rng.uniform(25, 49)
        lon = rng.uniform(-125, -67)
        size = rng.uniform(0.5, 2.0)
        # Create irregular polygon
        n_points = rng.randint(5, 10)
        angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_points))
        radii = size * (0.5 + 0.5 * rng.random(n_points))
        coords = [
            (lon + r * np.cos(a), lat + r * np.sin(a))
            for a, r in zip(angles, radii)
        ]
        coords.append(coords[0])  # Close polygon
        try:
            geom = Polygon(coords)
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            geom = box(lon, lat, lon + size, lat + size)

        subregion_rows.append({
            "geometry": geom,
            "subregion_id": i,
            "subregion_name": f"Sub-Region {i}",
            "parent_region": rng.choice(NIFC_REGIONS),
            "terrain_type": rng.choice(["Mountain", "Forest", "Grassland", "Desert", "Coastal"]),
            "avg_annual_fires": int(rng.exponential(100)),
            "avg_annual_acres": int(rng.exponential(50000)),
            "fire_weather_zone": f"FWZ{rng.randint(100, 999)}",
        })

    gdf = gpd.GeoDataFrame(subregion_rows, crs="EPSG:4326")
    gdf.to_file(os.path.join(output_dir, "synthetic_nifc_geographic_areas_detailed.gpkg"), driver="GPKG")
    results["subregions"] = n_subregions

    # For higher scale factors, generate point-based fire location GeoPackage
    if scale_factor >= 5:
        print("  Generating synthetic fire locations GeoPackage...")
        n_fires = max(1000, int(scale_factor * 500))
        fire_rows = []
        for i in range(n_fires):
            lat = rng.uniform(25, 49)
            lon = rng.uniform(-125, -67)
            fire_rows.append({
                "geometry": Point(lon, lat),
                "fire_id": i,
                "fire_name": f"Synthetic Fire {i}",
                "year": rng.randint(2000, 2025),
                "acres": round(rng.exponential(500), 1),
                "cause": rng.choice(["Human", "Lightning", "Unknown"]),
                "duration_days": max(1, int(rng.exponential(10))),
                "region": rng.choice(NIFC_REGIONS),
                "state": rng.choice(US_STATES),
            })

        gdf = gpd.GeoDataFrame(fire_rows, crs="EPSG:4326")
        gdf.to_file(os.path.join(output_dir, "synthetic_fire_locations.gpkg"), driver="GPKG")
        results["fire_locations"] = n_fires

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
        description="Inflate Kramabench wildfire dataset with synthetic data"
    )
    parser.add_argument("input_dir", help="Path to original data/wildfire/input/")
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

    # Step 3: New synthetic CSV files
    print("Step 3: Generating new synthetic CSV files...")
    csv_results = generate_synthetic_csvs(output_dir, scale_factor)
    csv_size = get_dir_size(output_dir)
    print(f"  Generated: {csv_results}")
    print(f"  Size after synthetic CSVs: {format_size(csv_size)}")
    print()

    # Step 4: Large observation datasets
    print("Step 4: Generating large observation datasets...")
    obs_results = generate_large_observations(output_dir, scale_factor)
    obs_size = get_dir_size(output_dir)
    print(f"  Generated: {obs_results}")
    print(f"  Size after observations: {format_size(obs_size)}")
    print()

    # Step 5: GeoPackage expansion
    print("Step 5: Generating additional GeoPackage files...")
    gpkg_results = generate_geopackages(output_dir, scale_factor)
    print(f"  Generated: {gpkg_results}")
    print()

    # Summary
    final_size = get_dir_size(output_dir)
    final_files = count_files(output_dir)
    print("=" * 60)
    print("INFLATION SUMMARY")
    print("=" * 60)
    print(f"Original:  {format_size(orig_size):>12s}  ({orig_files:,d} files)")
    print(f"Inflated:  {format_size(final_size):>12s}  ({final_files:,d} files)")
    if orig_size > 0:
        print(f"Ratio:     {final_size / orig_size:.1f}x size, {final_files / orig_files:.1f}x files")
    print("=" * 60)


if __name__ == "__main__":
    main()
