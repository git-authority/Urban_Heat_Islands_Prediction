import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

file_path = '../../Dataset/Population_2020.tif'

mumbai_bbox = {
    'lat_min': 18.89215,
    'lat_max': 19.27,
    'lon_min': 72.776,
    'lon_max': 72.98
}

downloaded_bbox = {
    'lat_min': 18.6,
    'lat_max': 19.6,
    'lon_min': 72.71,
    'lon_max': 73.71
}

colors = [
    "#ffff66",
    "#b2ff66",
    "#66ff66",
    "#66b2b2",
    "#000066"
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

def read_and_mask(bbox):
    with rasterio.open(file_path) as src:
        window = from_bounds(
            bbox['lon_min'], bbox['lat_min'],
            bbox['lon_max'], bbox['lat_max'],
            src.transform
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
    data_masked = np.ma.masked_where(data <= 0, data)
    return data_masked, transform, data.shape, bbox

def create_ticks(data_shape, bbox):
    height, width = data_shape
    xticks = np.linspace(0, width - 1, 5)
    yticks = np.linspace(0, height - 1, 5)
    xlabels = [f"{bbox['lon_min'] + (bbox['lon_max'] - bbox['lon_min']) * (x / (width - 1)):.2f}°E" for x in xticks]
    ylabels = [f"{bbox['lat_max'] - (bbox['lat_max'] - bbox['lat_min']) * (y / (height - 1)):.2f}°N" for y in yticks]
    return xticks, xlabels, yticks, ylabels

# Font properties
font_family = 'Times New Roman'
label_fontsize = 10
label_fontweight = 'bold'
title_fontsize = 12

mumbai_data, mumbai_transform, mumbai_shape, mumbai_bbox = read_and_mask(mumbai_bbox)
downloaded_data, downloaded_transform, downloaded_shape, downloaded_bbox = read_and_mask(downloaded_bbox)

mumbai_height, mumbai_width = mumbai_shape
downloaded_height, downloaded_width = downloaded_shape

scale_factor = downloaded_height / mumbai_height
scaled_mumbai_width = mumbai_width * scale_factor

fig_width = scaled_mumbai_width + downloaded_width
fig_height = downloaded_height

dpi = 100
figsize = (fig_width / dpi, fig_height / dpi + 0.5)

fig, axs = plt.subplots(1, 2, figsize=figsize,
                        gridspec_kw={'width_ratios': [scaled_mumbai_width, downloaded_width]})

# Increased horizontal space between plots
plt.subplots_adjust(wspace=0.5, top=0.85, left=0.08, right=0.92)

# Main heading
fig.suptitle('Population Density', fontsize=16, fontweight='bold', fontfamily=font_family, y=0.98)

# Plot Mumbai region
img1 = axs[0].imshow(mumbai_data, cmap=cmap)

# Plot downloaded dataset region
img2 = axs[1].imshow(downloaded_data, cmap=cmap)
cbar2 = fig.colorbar(img2, ax=axs[1], fraction=0.045, pad=0.08, aspect=30)
cbar2.set_label('Population Density (people per pixel) →', fontsize=label_fontsize, fontweight=label_fontweight,
                fontfamily=font_family, labelpad=15)

# --- UPDATED COLORBAR SECTION (left colorbar thinner than right) ---
cbar2_pos = cbar2.ax.get_position()
left_cbar_width = cbar2_pos.width * 0.7  # 70% of right colorbar width

cbar_ax_left = fig.add_axes([
    axs[0].get_position().x1 + (axs[1].get_position().x0 - axs[0].get_position().x1) / 10,
    cbar2_pos.y0,
    left_cbar_width,
    cbar2_pos.height
])

cbar1 = fig.colorbar(img1, cax=cbar_ax_left)
cbar1.set_label('Population Density (people per pixel) →', fontsize=label_fontsize,
                fontweight=label_fontweight, fontfamily=font_family, labelpad=15)
# -------------------------------------------------------------------

xticks, xlabels, yticks, ylabels = create_ticks(mumbai_shape, mumbai_bbox)
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xlabels, fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels(ylabels, fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family)

axs[0].set_xlabel('Longitude →', fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family, labelpad=15)
axs[0].set_ylabel('Latitude →', fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family, labelpad=15)
axs[0].set_title('Mumbai Region (2020)', fontsize=title_fontsize, fontweight='bold', fontfamily=font_family, pad=25)

xticks, xlabels, yticks, ylabels = create_ticks(downloaded_shape, downloaded_bbox)
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xlabels, fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family)
axs[1].set_yticks(yticks)
axs[1].set_yticklabels(ylabels, fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family)

axs[1].set_xlabel('Longitude →', fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family, labelpad=15)
axs[1].set_ylabel('Latitude →', fontsize=label_fontsize, fontweight=label_fontweight, fontfamily=font_family, labelpad=15)
axs[1].set_title('Downloaded Dataset (2020)', fontsize=title_fontsize, fontweight='bold', fontfamily=font_family, pad=25)

plt.show()
