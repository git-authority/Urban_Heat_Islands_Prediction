#!/usr/bin/env python3
"""
Daily MIN (per-timepoint mask min → daily mean/std) plots per year for Rural & Urban.
Produces 1 plot per year (Jan–Dec) & CSV files.
"""

import os, glob
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates

# -------- CONFIG ----------
DATA_ROOT = os.path.join("..", "..", "Dataset")
YEARS = ["2020", "2021", "2022", "2023", "2024"]
HOURS_KEEP = {0, 3, 6, 9, 12, 15, 18, 21}

# Big rural box
R_lat_min, R_lat_max = 18.6, 19.6
R_lon_min, R_lon_max = 72.71, 73.71

# Urban = union of 3 boxes
URBAN_AREAS = [
    {
        "lat_min": 18.895000,
        "lat_max": 19.318333,
        "lon_min": 72.792500,
        "lon_max": 73.015833,
    },
    {
        "lat_min": 19.164167,
        "lat_max": 19.265000,
        "lon_min": 73.070833,
        "lon_max": 73.215833,
    },
    {
        "lat_min": 19.363333,
        "lat_max": 19.477500,
        "lon_min": 72.790000,
        "lon_max": 72.880833,
    },
]

LAT_CANDS = ["lat", "latitude", "y", "nav_lat", "latitude_0"]
LON_CANDS = ["lon", "longitude", "x", "nav_lon", "longitude_0"]
TIME_CANDS = ["time", "times", "Time", "datetime", "valid_time"]
TEMP_CANDS = ["t2m", "tas", "temperature", "temp", "air_temperature", "T2", "air_temp"]


# -------- helpers ----------
def find_variable(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None


def infer_temp_var(ds):
    v = find_variable(ds, TEMP_CANDS)
    if v:
        return v
    for name, var in ds.variables.items():
        if "time" in var.dimensions and len(var.dimensions) >= 2:
            return name
    latn = find_variable(ds, LAT_CANDS) or ""
    lonn = find_variable(ds, LON_CANDS) or ""
    for name, var in ds.variables.items():
        if len(var.dimensions) >= 2 and name not in (latn, lonn):
            return name
    raise RuntimeError("Could not infer temperature variable name")


def open_nc_try_vars(path):
    ds = Dataset(path)
    lat_name = find_variable(ds, LAT_CANDS)
    lon_name = find_variable(ds, LON_CANDS)
    time_name = find_variable(ds, TIME_CANDS)
    temp_name = infer_temp_var(ds)

    if time_name is None:
        temp_var = ds.variables[temp_name]
        for d in temp_var.dimensions:
            if d in ds.variables:
                if "since" in str(getattr(ds.variables[d], "units", "")):
                    time_name = d
                    break
        if time_name is None:
            time_name = temp_var.dimensions[0]

    if lat_name is None or lon_name is None:
        for name, var in ds.variables.items():
            if getattr(var, "ndim", 0) == 1:
                try:
                    arr = np.array(var[:])
                    if (
                        lat_name is None
                        and -90 <= np.nanmin(arr) <= np.nanmax(arr) <= 90
                    ):
                        lat_name = name
                    if (
                        lon_name is None
                        and -180 <= np.nanmin(arr) <= np.nanmax(arr) <= 180
                    ):
                        lon_name = name
                except:
                    pass

    if lat_name is None:
        for name in ds.variables:
            if "lat" in name.lower():
                lat_name = name
                break

    if lon_name is None:
        for name in ds.variables:
            if "lon" in name.lower():
                lon_name = name
                break

    return ds, lat_name, lon_name, time_name, temp_name


def extract_lat_lon(ds, lat_name, lon_name):
    lat = ds.variables[lat_name][:]
    lon = ds.variables[lon_name][:]
    if getattr(lat, "ndim", 1) == 2:
        return lat, lon
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def make_mask_from_bbox(lat2, lon2, lat_min, lat_max, lon_min, lon_max):
    return (lat2 >= lat_min) & (lat2 <= lat_max) & (lon2 >= lon_min) & (lon2 <= lon_max)


def _union_mask_from_boxes(lat2, lon2, boxes):
    mask = np.zeros_like(lat2, dtype=bool)
    for b in boxes:
        mask |= make_mask_from_bbox(
            lat2, lon2, b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"]
        )
    return mask


def parse_times_for_var(ds, time_name):
    tvar = ds.variables[time_name]
    vals = tvar[:]
    units = getattr(tvar, "units", None)
    cal = getattr(tvar, "calendar", None)

    if (
        (
            np.issubdtype(vals.dtype, np.integer)
            or np.issubdtype(vals.dtype, np.floating)
        )
        and units
        and "since 1970" in str(units)
    ):
        return pd.to_datetime(vals, unit="s")

    try:
        cft = num2date(vals, units=units, calendar=cal or "standard")
    except:
        return pd.to_datetime(vals, errors="coerce")

    if hasattr(cft[0], "year") and cft[0].__class__.__module__.startswith("cftime"):
        py = []
        for t in cft:
            try:
                py.append(
                    datetime(
                        t.year,
                        t.month,
                        t.day,
                        getattr(t, "hour", 0),
                        getattr(t, "minute", 0),
                        int(getattr(t, "second", 0)),
                    )
                )
            except:
                py.append(pd.NaT)
        return pd.to_datetime(py)

    return pd.to_datetime(cft, errors="coerce")


# -------- compute per-timestep MIN for a mask ----------
def compute_min_timeseries_for_mask(files, mask):
    times = []
    vals = []

    for f in tqdm(sorted(files), desc="Reading", leave=False):
        ds, latn, lonn, timen, tempn = open_nc_try_vars(f)
        dtimes = parse_times_for_var(ds, timen)

        temp = ds.variables[tempn]
        dims = temp.dimensions
        t_axis = dims.index(timen) if timen in dims else 0

        nt = temp.shape[t_axis]
        use_nt = min(nt, len(dtimes))

        for ti in range(use_nt):
            dt = dtimes[ti]
            if pd.isna(dt) or dt.hour not in HOURS_KEEP:
                continue

            slicer = [slice(None)] * temp.ndim
            slicer[t_axis] = ti

            try:
                arr = temp[tuple(slicer)]
            except:
                arr = np.take(temp[:], ti, axis=t_axis)

            arr = np.array(arr)

            if arr.ndim > 2:
                arr = arr.mean(axis=tuple(range(1, arr.ndim)))

            if arr.shape != mask.shape:
                try:
                    arr = arr.reshape(mask.shape)
                except:
                    arr = arr.flatten()
                    m = mask.flatten()
                    vals.append(np.nanmin(arr[m]))
                    times.append(dt)
                    continue

            vals.append(np.nanmin(arr[mask]))
            times.append(dt)

        ds.close()

    return pd.DataFrame({"min": vals}, index=pd.to_datetime(times)).sort_index()


# -------- main ----------
def main():
    out_dir = "Daily_Per_Year_Min"
    os.makedirs(out_dir, exist_ok=True)

    month_locator = mdates.MonthLocator()
    month_fmt = mdates.DateFormatter("%b")

    for year in YEARS:
        year_dir = os.path.join(DATA_ROOT, year)
        files = sorted(glob.glob(os.path.join(year_dir, "*.nc")))
        if not files:
            print("No files for", year)
            continue

        # build masks
        ds0, latn, lonn, timen, tempn = open_nc_try_vars(files[0])
        lat2, lon2 = extract_lat_lon(ds0, latn, lonn)
        ds0.close()

        urban_mask = _union_mask_from_boxes(lat2, lon2, URBAN_AREAS)
        rural_mask = make_mask_from_bbox(
            lat2, lon2, R_lat_min, R_lat_max, R_lon_min, R_lon_max
        ) & (~urban_mask)

        rural_ts = compute_min_timeseries_for_mask(files, rural_mask)
        urban_ts = compute_min_timeseries_for_mask(files, urban_mask)

        # Daily mean of per-timepoint MIN, and daily STD
        rural_daily_min = rural_ts["min"].resample("D").mean()
        rural_daily_std = rural_ts["min"].resample("D").std(ddof=0)

        urban_daily_min = urban_ts["min"].resample("D").mean()
        urban_daily_std = urban_ts["min"].resample("D").std(ddof=0)

        day_idx = rural_daily_min.index.union(urban_daily_min.index).sort_values()

        rms = (
            float(np.nanmean(rural_daily_std.dropna()))
            if rural_daily_std.dropna().size
            else np.nan
        )
        ums = (
            float(np.nanmean(urban_daily_std.dropna()))
            if urban_daily_std.dropna().size
            else np.nan
        )

        fig, ax = plt.subplots(figsize=(14, 4))

        # Rural shading
        ax.fill_between(
            day_idx,
            rural_daily_min.reindex(day_idx) - rural_daily_std.reindex(day_idx),
            rural_daily_min.reindex(day_idx) + rural_daily_std.reindex(day_idx),
            alpha=0.22,
            color="tab:blue",
            label="Rural daily min ±STD",
        )
        ax.plot(
            day_idx,
            rural_daily_min.reindex(day_idx),
            lw=1.8,
            color="tab:blue",
            label="Rural daily min",
        )

        # Urban shading
        ax.fill_between(
            day_idx,
            urban_daily_min.reindex(day_idx) - urban_daily_std.reindex(day_idx),
            urban_daily_min.reindex(day_idx) + urban_daily_std.reindex(day_idx),
            alpha=0.2,
            color="tab:red",
            label="Urban daily min ±STD",
        )
        ax.plot(
            day_idx,
            urban_daily_min.reindex(day_idx),
            lw=1.8,
            color="tab:red",
            label="Urban daily min",
        )

        ax.text(
            0.02,
            0.88,
            f"Mean of Rural Daily Min STD: {rms:.2f} K\nMean of Urban Daily Min STD: {ums:.2f} K",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(
            f"Mumbai ({year}) — Daily Average of Hourly Minima for t2m (Jan-Dec)",
            fontsize=14,
            weight="bold",
        )
        ax.set_ylabel("Daily Min Temperature (K)")
        ax.set_xlabel("Months →")
        ax.xaxis.set_major_locator(month_locator)
        ax.xaxis.set_major_formatter(month_fmt)
        ax.grid(alpha=0.25)

        # place legend top-right (match mean plot)
        ax.legend(loc="upper right", fontsize=9)

        if len(day_idx):
            ax.set_xlim([day_idx.min(), day_idx.max()])

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{year}_daily_min.png"), dpi=180)
        plt.close(fig)

        # Save CSVs
        pd.DataFrame(
            {"daily_min": rural_daily_min, "daily_std": rural_daily_std}
        ).to_csv(os.path.join(out_dir, f"{year}_rural_daily_min.csv"))

        pd.DataFrame(
            {"daily_min": urban_daily_min, "daily_std": urban_daily_std}
        ).to_csv(os.path.join(out_dir, f"{year}_urban_daily_min.csv"))

        print("Saved:", os.path.join(out_dir, f"{year}_daily_min.png"))


if __name__ == "__main__":
    main()
