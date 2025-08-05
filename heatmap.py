import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the ERA5 dataset
file_path = "t2m_2025_May_1-5/data_mumbai.nc"
ds = xr.open_dataset(file_path)

# Extract coordinates and temperature
lats = ds.latitude.values
lons = ds.longitude.values
temps = ds["t2m"] - 273.15  # Convert Kelvin to Celsius
times = ds["valid_time"].values
all_days = np.datetime_as_string(times, unit="D")

# Loop through each day: 1st to 5th May
for day in range(1, 6):
    selected_day = f"2025-05-{day:02d}"
    day_indices_all = np.where(all_days == selected_day)[0]
    day_indices = day_indices_all[::3][:8]  # 3-hour interval, first 8 time steps

    # Plot settings
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.25)

    cmap = "Greys"
    vmin = np.min(temps[day_indices])
    vmax = np.max(temps[day_indices])

    for i, ax in enumerate(axs.flat):
        idx = day_indices[i]
        temp_slice = temps.isel(valid_time=idx).values

        mesh = ax.pcolormesh(lons, lats, temp_slice, cmap=cmap, vmin=vmin, vmax=vmax)

        ts = times[idx]
        hour = int(str(ts)[11:13])
        meridiem = "AM" if hour < 12 else "PM"
        hour_formatted = f"{hour%12 if hour%12 != 0 else 12}{meridiem}"

        ax.set_title(f"{day}st May 2025, time - {hour_formatted}", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Add horizontal colorbar below all plots
    cbar_ax = fig.add_axes([0.35, 0.10, 0.3, 0.02])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("2m Temperature (°C)")
    cbar.ax.tick_params(labelsize=8)

    # Save the figure
    plt.savefig(f"may{day}_bw_8plots_3hrinterval.png", dpi=300)
    plt.close()
