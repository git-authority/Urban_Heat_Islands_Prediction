#!/usr/bin/env python3
"""
Daily mean (Jan-Dec) plots per year for Rural and Urban (union of 3 boxes).
Saves one PNG per year named <YEAR>_daily_mean.png in output dir.
"""
import os, glob
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates

# -------- CONFIG (unchanged masks) ----------
DATA_ROOT = os.path.join("..", "..", "Dataset")
YEARS = ["2020", "2021", "2022", "2023", "2024"]
HOURS_KEEP = {0, 3, 6, 9, 12, 15, 18, 21}

R_lat_min, R_lat_max = 18.6, 19.6
R_lon_min, R_lon_max = 72.71, 73.71

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


# -------- helpers (same strategy as earlier scripts) ----------
def find_variable(ds, cands):
    for c in cands:
        if c in ds.variables:
            return c
    return None


def infer_temp_var(ds):
    v = find_variable(ds, TEMP_CANDS)
    if v:
        return v
    for name, var in ds.variables.items():
        dims = var.dimensions
        if "time" in dims or "valid_time" in dims:
            if len(dims) >= 2:
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
                units = getattr(ds.variables[d], "units", "")
                if isinstance(units, str) and "since" in units:
                    time_name = d
                    break
        if time_name is None and len(temp_var.dimensions) > 0:
            time_name = temp_var.dimensions[0]

    if lat_name is None or lon_name is None:
        for name, var in ds.variables.items():
            if getattr(var, "ndim", None) == 1:
                try:
                    arr = np.array(var[:])
                    if (
                        lat_name is None
                        and np.nanmin(arr) >= -90
                        and np.nanmax(arr) <= 90
                    ):
                        lat_name = name
                    if (
                        lon_name is None
                        and np.nanmin(arr) >= -180
                        and np.nanmax(arr) <= 180
                    ):
                        lon_name = name
                except Exception:
                    continue
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
        lat2, lon2 = lat, lon
    else:
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
    time_var = ds.variables[time_name]
    times_nc = time_var[:]
    time_units = getattr(time_var, "units", None)
    time_cal = getattr(time_var, "calendar", None)
    if (
        (
            np.issubdtype(np.array(times_nc).dtype, np.integer)
            or np.issubdtype(np.array(times_nc).dtype, np.floating)
        )
        and time_units
        and "since 1970" in str(time_units)
    ):
        return pd.to_datetime(times_nc, unit="s")
    try:
        dtimes = num2date(times_nc, units=time_units, calendar=time_cal or "standard")
    except Exception:
        dtimes = np.array(times_nc)
    if (
        len(dtimes) > 0
        and hasattr(dtimes[0], "year")
        and dtimes[0].__class__.__module__.startswith("cftime")
    ):
        tmp = []
        for t in dtimes:
            try:
                tmp.append(
                    datetime(
                        t.year,
                        t.month,
                        t.day,
                        getattr(t, "hour", 0),
                        getattr(t, "minute", 0),
                        int(getattr(t, "second", 0)),
                    )
                )
            except Exception:
                tmp.append(pd.NaT)
        return pd.to_datetime(tmp)
    return pd.to_datetime(dtimes, errors="coerce")


# compute per-timestep mean (8 times/day) for an arbitrary mask
def compute_mean_timeseries_for_mask(files, mask):
    times = []
    vals = []
    for f in tqdm(sorted(files), desc="reading files", leave=False):
        ds, latn, lonn, timen, tempn = open_nc_try_vars(f)
        if timen is None:
            for n, v in ds.variables.items():
                if "units" in getattr(v, "ncattrs", lambda: [])() and "since" in str(
                    getattr(v, "units", "")
                ):
                    timen = n
                    break
        if timen is None:
            ds.close()
            raise RuntimeError(f"No time variable in {f}")
        time_var = ds.variables[timen]
        dtimes = parse_times_for_var(ds, timen)
        temp_var = ds.variables[tempn]
        temp_dims = temp_var.dimensions
        t_axis = temp_dims.index(timen) if timen in temp_dims else 0
        nt = temp_var.shape[t_axis]
        use_nt = min(nt, len(dtimes))
        for ti in range(use_nt):
            dt = dtimes[ti]
            if pd.isna(dt):
                continue
            if dt.hour not in HOURS_KEEP:
                continue
            slicer = [slice(None)] * temp_var.ndim
            slicer[t_axis] = ti
            try:
                arr = temp_var[tuple(slicer)]
            except Exception:
                full = temp_var[:]
                arr = np.take(full, ti, axis=t_axis)
            arr = np.array(arr)
            if arr.ndim > 2:
                arr = arr.mean(axis=tuple(range(1, arr.ndim)))
            if arr.shape != mask.shape:
                try:
                    arr = arr.reshape(mask.shape)
                except Exception:
                    arr = arr.flatten()
                    mflat = mask.flatten()
                    sel = arr[mflat]
                    times.append(pd.to_datetime(dt))
                    vals.append(np.nanmean(sel))
                    continue
            sel = arr[mask]
            times.append(pd.to_datetime(dt))
            vals.append(np.nanmean(sel))
        ds.close()
    return pd.DataFrame(
        {"mean": np.array(vals)}, index=pd.to_datetime(times)
    ).sort_index()


# -------- main (per-year daily plots)
def main():
    month_locator = mdates.MonthLocator()
    month_fmt = mdates.DateFormatter("%b")
    out_dir = os.path.join(os.getcwd(), "Daily_Per_Year_Mean")
    os.makedirs(out_dir, exist_ok=True)

    # precollect files per year and validate
    for year in YEARS:
        year_dir = os.path.join(DATA_ROOT, year)
        if not os.path.isdir(year_dir):
            print(f"Warning: missing year dir {year_dir}, skipping {year}")
            continue
        files = sorted(glob.glob(os.path.join(year_dir, "*.nc")))
        if not files:
            print(f"Warning: no nc files for {year}, skipping")
            continue

        # build masks from first file in that year
        sample_ds, latn, lonn, timen, tempn = open_nc_try_vars(files[0])
        lat2, lon2 = extract_lat_lon(sample_ds, latn, lonn)
        sample_ds.close()
        urban_mask = _union_mask_from_boxes(lat2, lon2, URBAN_AREAS)
        rural_mask = make_mask_from_bbox(
            lat2, lon2, R_lat_min, R_lat_max, R_lon_min, R_lon_max
        ) & (~urban_mask)

        # compute timeseries (8 timepoints per day) for that year's files
        rural_ts = compute_mean_timeseries_for_mask(files, rural_mask)
        urban_ts = compute_mean_timeseries_for_mask(files, urban_mask)

        # resample to daily mean and compute daily std (from the per-timepoint means)
        rural_daily_mean = rural_ts["mean"].resample("D").mean()
        rural_daily_std = rural_ts["mean"].resample("D").std(ddof=0)
        urban_daily_mean = urban_ts["mean"].resample("D").mean()
        urban_daily_std = urban_ts["mean"].resample("D").std(ddof=0)

        day_idx = rural_daily_mean.index.union(urban_daily_mean.index).sort_values()

        # compute mean of daily stds for textbox
        mean_std_rural = (
            float(np.nanmean(rural_daily_std.dropna().values))
            if len(rural_daily_std.dropna()) > 0
            else np.nan
        )
        mean_std_urban = (
            float(np.nanmean(urban_daily_std.dropna().values))
            if len(urban_daily_std.dropna()) > 0
            else np.nan
        )

        # plotting style similar to your example image
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(
            day_idx,
            rural_daily_mean.reindex(day_idx) - rural_daily_std.reindex(day_idx),
            rural_daily_mean.reindex(day_idx) + rural_daily_std.reindex(day_idx),
            alpha=0.22,
            label="Rural daily ±STD",
            color="tab:blue",
        )
        ax.plot(
            day_idx,
            rural_daily_mean.reindex(day_idx),
            lw=1.8,
            label="Rural daily mean",
            color="tab:blue",
        )

        ax.fill_between(
            day_idx,
            urban_daily_mean.reindex(day_idx) - urban_daily_std.reindex(day_idx),
            urban_daily_mean.reindex(day_idx) + urban_daily_std.reindex(day_idx),
            alpha=0.2,
            label="Urban daily ±STD",
            color="tab:red",
        )
        ax.plot(
            day_idx,
            urban_daily_mean.reindex(day_idx),
            lw=1.8,
            label="Urban daily mean",
            color="tab:red",
        )

        txt = f"Mean of Rural Daily STD: {mean_std_rural:.2f} K\n Mean of Urban Daily STD: {mean_std_urban:.2f} K"
        ax.text(
            0.02,
            0.95,
            txt,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.6"),
        )

        ax.set_title(
            f"Mumbai ({year}) — Daily Average of Hourly t2m (Jan-Dec)",
            fontsize=14,
            weight="bold",
        )
        ax.set_ylabel("Daily Mean Temperature (K) →", fontsize=10)
        ax.set_xlabel("Months →", fontsize=10)
        ax.xaxis.set_major_locator(month_locator)
        ax.xaxis.set_major_formatter(month_fmt)
        if len(day_idx) > 0:
            ax.set_xlim([day_idx.min(), day_idx.max()])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{year}_daily_mean.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        # also save CSVs (optional)
        pd.DataFrame(
            {"rural_daily_mean": rural_daily_mean, "rural_daily_std": rural_daily_std}
        ).to_csv(os.path.join(out_dir, f"{year}_rural_daily_mean.csv"))
        pd.DataFrame(
            {"urban_daily_mean": urban_daily_mean, "urban_daily_std": urban_daily_std}
        ).to_csv(os.path.join(out_dir, f"{year}_urban_daily_mean.csv"))

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
