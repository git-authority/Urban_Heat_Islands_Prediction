"""
new.py

Generates 10 separate PNGs (2 per year for years 2020..2024):
 - {year}_hourly.png  : hourly series (8 points/day) with per-region mean lines only (no std)
 - {year}_daily.png   : daily-average series with per-region spatial ±STD bands and a small annotation
                       in the figure that shows the mean daily STD value for each region.

Keeps labels, titles and output filenames similar to the original script.
"""

import os
import glob
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates

# --------- CONFIG ----------
DATA_ROOT = os.path.join("..", "..", "Dataset")  # adjust if needed
YEARS = ["2020", "2021", "2022", "2023", "2024"]
MONTH_ORDER = ["March", "April", "May", "June"]
HOURS_KEEP = {0, 3, 6, 9, 12, 15, 18, 21}
# BBoxes
R_lat_min, R_lat_max = 18.6, 19.6
R_lon_min, R_lon_max = 72.71, 73.71
U_lat_min, U_lat_max = 18.89215, 19.27
U_lon_min, U_lon_max = 72.776, 72.98

# robust candidate names
LAT_CANDS = ["lat", "latitude", "y", "nav_lat", "latitude_0"]
LON_CANDS = ["lon", "longitude", "x", "nav_lon", "longitude_0"]
TIME_CANDS = ["time", "times", "Time", "datetime", "valid_time"]
TEMP_CANDS = ["t2m", "tas", "temperature", "temp", "air_temperature", "T2", "air_temp"]


# --------- helpers ----------
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

    # infer time from temp dims/units if necessary
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

    # infer lat/lon if needed
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
        lat2 = lat
        lon2 = lon
    else:
        lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def make_mask_from_bbox(lat2, lon2, lat_min, lat_max, lon_min, lon_max):
    return (lat2 >= lat_min) & (lat2 <= lat_max) & (lon2 >= lon_min) & (lon2 <= lon_max)


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


# compute per-timestep spatial mean+std for a bounding box
def compute_mean_std_per_timestep(files, lat_min, lat_max, lon_min, lon_max):
    times = []
    means = []
    stds = []
    mask = None

    for f in tqdm(sorted(files), desc="reading files", leave=False):
        ds, lat_name, lon_name, time_name, temp_name = open_nc_try_vars(f)
        if time_name is None:
            # attempt to find a units-with-since variable
            for n, v in ds.variables.items():
                if "units" in getattr(v, "ncattrs", lambda: [])() and "since" in str(
                    getattr(v, "units", "")
                ):
                    time_name = n
                    break
        if time_name is None:
            ds.close()
            raise RuntimeError(f"Could not find time variable in {f}")

        lat2, lon2 = extract_lat_lon(ds, lat_name, lon_name)
        if mask is None:
            mask = make_mask_from_bbox(lat2, lon2, lat_min, lat_max, lon_min, lon_max)
            if not mask.any():
                ds.close()
                raise RuntimeError(f"No gridpoints found in bbox for file {f}")

        dtimes = parse_times_for_var(ds, time_name)
        temp_var = ds.variables[temp_name]
        temp_dims = temp_var.dimensions
        t_axis = temp_dims.index(time_name) if time_name in temp_dims else 0
        nt = temp_var.shape[t_axis]
        use_nt = min(nt, len(dtimes))

        for ti in range(use_nt):
            dt = dtimes[ti]
            if pd.isna(dt):
                continue
            month = dt.month
            hour = dt.hour
            if month not in {3, 4, 5, 6}:
                continue
            if hour not in HOURS_KEEP:
                continue

            # slice temp
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
            # align shapes
            if arr.shape != mask.shape:
                try:
                    arr = arr.reshape(mask.shape)
                except Exception:
                    arr = arr.flatten()
                    mask_flat = mask.flatten()
                    sel = arr[mask_flat]
                    times.append(pd.to_datetime(dt))
                    means.append(np.nanmean(sel))
                    stds.append(np.nanstd(sel, ddof=0))
                    continue
            sel = arr[mask]
            times.append(pd.to_datetime(dt))
            means.append(np.nanmean(sel))
            stds.append(np.nanstd(sel, ddof=0))

        ds.close()

    df = pd.DataFrame(
        {"mean": np.array(means), "std": np.array(stds)}, index=pd.to_datetime(times)
    ).sort_index()
    return df


# --------- main: produce 2 separate figures per year ----------
def make_and_save_separate_figures(data_root, years):
    month_locator = mdates.MonthLocator()
    month_fmt = mdates.DateFormatter("%b")

    for year in years:
        year_dir = os.path.join(data_root, year)
        if not os.path.isdir(year_dir):
            print(f"Warning: year folder not found: {year_dir}. Skipping.")
            continue

        files = sorted(glob.glob(os.path.join(year_dir, "*.nc")))
        if not files:
            print(f"Warning: no .nc files found in {year_dir}. Skipping.")
            continue

        # prefer files that contain month names, else use all
        month_files = []
        for m in MONTH_ORDER:
            found = [f for f in files if m.lower() in os.path.basename(f).lower()]
            if found:
                month_files.extend(found)
        use_files = month_files if month_files else files

        # compute per-timestep mean/std for both regions
        rural_df = compute_mean_std_per_timestep(
            use_files, R_lat_min, R_lat_max, R_lon_min, R_lon_max
        )
        urban_df = compute_mean_std_per_timestep(
            use_files, U_lat_min, U_lat_max, U_lon_min, U_lon_max
        )

        # common index
        idx = rural_df.index.union(urban_df.index).sort_values()

        # ---------------- Hourly figure: NO std (means only) ----------------
        fig_h, ax_h = plt.subplots(figsize=(14, 4))
        # rural mean line
        ax_h.plot(
            idx,
            rural_df.reindex(idx)["mean"],
            color="tab:blue",
            lw=1.0,
            label="Rural mean",
        )
        # urban mean line
        ax_h.plot(
            idx,
            urban_df.reindex(idx)["mean"],
            color="tab:red",
            lw=1.0,
            label="Urban mean",
        )

        ax_h.set_title(
            f"{year} — Hourly t2m at 8 timepoints/day (Mar–Jun)",
            fontsize=12,
            weight="bold",
        )
        ax_h.set_ylabel("Temperature (K) →", fontsize=10)
        ax_h.set_xlabel("Months (Mar — Apr — May — Jun)", fontsize=10)
        ax_h.xaxis.set_major_locator(month_locator)
        ax_h.xaxis.set_major_formatter(month_fmt)
        if len(idx) > 0:
            ax_h.set_xlim([idx.min(), idx.max()])
        ax_h.grid(alpha=0.25)
        ax_h.legend(loc="upper right", fontsize=9)

        out_h = f"{year}_hourly.png"
        fig_h.tight_layout()
        fig_h.savefig(out_h, dpi=180)
        plt.close(fig_h)

        # ---------------- Daily figure: keep spatial ±STD and show mean daily STD numbers in-plot ----------------
        rural_daily_mean = rural_df["mean"].resample("D").mean()
        rural_daily_stdtime = rural_df["mean"].resample("D").std(ddof=0)
        urban_daily_mean = urban_df["mean"].resample("D").mean()
        urban_daily_stdtime = urban_df["mean"].resample("D").std(ddof=0)
        day_idx = rural_daily_mean.index.union(urban_daily_mean.index).sort_values()

        fig_d, ax_d = plt.subplots(figsize=(14, 4))
        ax_d.fill_between(
            day_idx,
            rural_daily_mean.reindex(day_idx) - rural_daily_stdtime.reindex(day_idx),
            rural_daily_mean.reindex(day_idx) + rural_daily_stdtime.reindex(day_idx),
            color="tab:blue",
            alpha=0.22,
            label="Rural daily ±STD",
        )
        ax_d.plot(
            day_idx,
            rural_daily_mean.reindex(day_idx),
            color="tab:blue",
            lw=1.8,
            label="Rural daily mean",
        )

        ax_d.fill_between(
            day_idx,
            urban_daily_mean.reindex(day_idx) - urban_daily_stdtime.reindex(day_idx),
            urban_daily_mean.reindex(day_idx) + urban_daily_stdtime.reindex(day_idx),
            color="tab:red",
            alpha=0.2,
            label="Urban daily ±STD",
        )
        ax_d.plot(
            day_idx,
            urban_daily_mean.reindex(day_idx),
            color="tab:red",
            lw=1.8,
            label="Urban daily mean",
        )

        # compute a simple summary (mean of daily std values) and display it on the plot
        mean_std_rural = (
            float(np.nanmean(rural_daily_stdtime.values))
            if len(rural_daily_stdtime.dropna()) > 0
            else np.nan
        )
        mean_std_urban = (
            float(np.nanmean(urban_daily_stdtime.values))
            if len(urban_daily_stdtime.dropna()) > 0
            else np.nan
        )
        txt = f"Rural mean daily STD: {mean_std_rural:.2f} K\nUrban mean daily STD: {mean_std_urban:.2f} K"
        ax_d.text(
            0.02,
            0.95,
            txt,
            transform=ax_d.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.6"),
        )

        ax_d.set_title(
            f"{year} — Daily average t2m (Mar–Jun)", fontsize=12, weight="bold"
        )
        ax_d.set_ylabel("Daily mean Temperature (K) →", fontsize=10)
        ax_d.set_xlabel("Months (Mar — Apr — May — Jun)", fontsize=10)
        ax_d.xaxis.set_major_locator(month_locator)
        ax_d.xaxis.set_major_formatter(month_fmt)
        if len(day_idx) > 0:
            ax_d.set_xlim([day_idx.min(), day_idx.max()])
        ax_d.grid(alpha=0.25)
        ax_d.legend(loc="upper right", fontsize=9)

        out_d = f"{year}_daily.png"
        fig_d.tight_layout()
        fig_d.savefig(out_d, dpi=180)
        plt.close(fig_d)

        print(f"Saved: {out_h}  and  {out_d}")


# --------- run ----------
if __name__ == "__main__":
    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")
    make_and_save_separate_figures(DATA_ROOT, YEARS)
