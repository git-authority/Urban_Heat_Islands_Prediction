import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import os

# Set font
plt.rcParams["font.family"] = "Times New Roman"

# Path to folder containing datasets
folder_path = "Dataset/2024"

# Folder to save plots
save_folder = "plots_may_1_to_5_color"
os.makedirs(save_folder, exist_ok=True)


# Function to get correct day suffix
def get_day_suffix(day):
    if 11 <= day <= 13:
        return "th"
    last_digit = day % 10
    return {1: "st", 2: "nd", 3: "rd"}.get(last_digit, "th")


# Loop over all .nc files in folder
for file_name in sorted(os.listdir(folder_path)):
    if not file_name.endswith(".nc"):
        continue

    file_path = os.path.join(folder_path, file_name)
    print(f"Processing {file_name}...")

    # Load dataset
    ds = Dataset(file_path)

    # Read variables
    t2m = ds.variables["t2m"][:]  # (time, lat, lon)
    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]
    times = ds.variables["valid_time"]

    # Convert time to datetime objects
    time_units = times.units
    time_calendar = getattr(times, "calendar", "standard")
    time_values = num2date(times[:], units=time_units, calendar=time_calendar)

    # Flip latitude if needed
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        t2m = t2m[:, ::-1, :]

    # Loop through days in this file
    total_days = len(time_values) // 24
    for day in range(total_days):
        day_num = time_values[day * 24].day
        month_num = time_values[day * 24].month

        # Only process 1st May to 5th May
        if month_num != 5 or not (1 <= day_num <= 5):
            continue

        fig, axes = plt.subplots(2, 4, figsize=(14, 9))
        axes = axes.flatten()

        start_idx = day * 24
        end_idx = min(start_idx + 21, len(time_values) - 1)
        start_time_str = time_values[start_idx].strftime("%I%p").lstrip("0").upper()
        end_time_str = time_values[end_idx].strftime("%I%p").lstrip("0").upper()

        for i in range(8):
            idx = start_idx + i * 3
            if idx >= len(time_values):
                break
            ax = axes[i]

            im = ax.imshow(
                t2m[idx],
                cmap="jet",
                origin="lower",
                extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            )

            dt = time_values[idx]
            hour_12 = dt.strftime("%I%p").lstrip("0").upper()
            suffix = get_day_suffix(dt.day)
            title = f"{dt.day}{suffix} {dt.strftime('%B %Y')}, Time - {hour_12}"

            ax.set_title(title, fontsize=10, pad=10, fontweight="bold")
            ax.set_xlabel("Longitude   → ", fontsize=12, labelpad=10, fontweight="bold")
            ax.set_ylabel("Latitude   → ", fontsize=12, labelpad=10, fontweight="bold")
            ax.tick_params(axis="both", labelsize=9)

        # Main title
        suffix = get_day_suffix(day_num)
        month_name = time_values[start_idx].strftime("%B")
        year = time_values[start_idx].year

        fig.subplots_adjust(top=0.9)
        fig.suptitle(
            f"\n\nMumbai | {day_num}{suffix} {month_name} {year} | {start_time_str} - {end_time_str}",
            fontsize=13,
            y=0.98,
            fontweight="bold",
        )

        # Layout adjustments
        plt.subplots_adjust(wspace=0.6, hspace=0.1, bottom=0.18)

        # Colorbar
        cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("2m Temperature (K)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        # Save plot
        save_path = os.path.join(save_folder, f"mumbai_{year}_may_{day_num:02d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    ds.close()

print(f"Plots saved in '{save_folder}'")
