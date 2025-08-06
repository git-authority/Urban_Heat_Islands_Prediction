import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np

# Set a clean, professional font
plt.rcParams["font.family"] = "Times New Roman"


# Load dataset
nc_path = "t2m_2025_May_1-5/data_mumbai.nc"
ds = Dataset(nc_path)

# Read variables
t2m = ds.variables["t2m"][:]  # (time, lat, lon)
lats = ds.variables["latitude"][:]
lons = ds.variables["longitude"][:]
times = ds.variables["valid_time"]

# Convert time to datetime objects
time_units = times.units
time_calendar = times.calendar if hasattr(times, "calendar") else "standard"
time_values = num2date(times[:], units=time_units, calendar=time_calendar)

# Convert temperature from K to °C
t2m_celsius = t2m - 273.15

# Flip latitude if needed (north at top)
if lats[0] > lats[-1]:
    lats = lats[::-1]
    t2m_celsius = t2m_celsius[:, ::-1, :]


# Function to get correct day suffix
def get_day_suffix(day):
    if 11 <= day <= 13:
        return "th"
    last_digit = day % 10
    return {1: "st", 2: "nd", 3: "rd"}.get(last_digit, "th")


# Plotting
for day in range(5):
    fig, axes = plt.subplots(2, 4, figsize=(14, 9))  # Reduced heatmap size
    axes = axes.flatten()

    start_idx = day * 24
    end_idx = start_idx + 21
    start_time_str = time_values[start_idx].strftime("%I%p").lstrip("0").upper()
    end_time_str = time_values[end_idx].strftime("%I%p").lstrip("0").upper()

    for i in range(8):
        idx = start_idx + i * 3
        ax = axes[i]

        im = ax.imshow(
            t2m_celsius[idx],
            cmap="binary",
            origin="lower",
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        )

        # Subplot title with extra padding
        dt = time_values[idx]
        hour_12 = dt.strftime("%I%p").lstrip("0").upper()
        day_num = dt.day
        suffix = get_day_suffix(day_num)
        title = f"{day_num}{suffix} May 2025, Time - {hour_12}"

        ax.set_title(
            title, fontsize=10, pad=10, fontweight="bold"
        )  # Added vertical padding
        ax.set_xlabel("Longitude   → ", fontsize=12, labelpad=10, fontweight="bold")
        ax.set_ylabel("Latitude   → ", fontsize=12, labelpad=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)

    # Main title (single line, minimal)
    day_num = time_values[start_idx].day
    suffix = get_day_suffix(day_num)
    fig.subplots_adjust(top=0.9)  # Try values between 0.8 and 0.9

    fig.suptitle(
        f"\n\nMumbai | {day_num}{suffix} May 2025 | {start_time_str} - {end_time_str}",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )

    # Layout adjustments
    plt.subplots_adjust(wspace=0.6, hspace=0.1, bottom=0.18)

    # Colorbar
    cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("2m Temperature (°C)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(
        f"{day_num}{suffix}_May_color_8plots_3hrinterval.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()
