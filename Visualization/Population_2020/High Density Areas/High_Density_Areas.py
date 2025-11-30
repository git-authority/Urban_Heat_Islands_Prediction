"""
Demagnified plot of the downloaded window with urban bounding boxes and proper lat/lon ticks.
Run as-is. Requires: rasterio, numpy, scipy, shapely, matplotlib.
"""

import rasterio
from rasterio.windows import from_bounds
from rasterio.features import shapes
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------
# Parameters (edit if needed)
# -------------------------------
file_path = "../../Dataset/Population_2020.tif"

downloaded_bbox = {"lat_min": 18.6, "lat_max": 19.6, "lon_min": 72.71, "lon_max": 73.71}

percentile_for_urban = 90  # top 10% pixels = urban
min_area_pixels = 500
morph_open_radius = 2
top_n = 8

# plotting appearance
font_family = "Times New Roman"
label_fontsize = 10
label_fontweight = "bold"
title_fontsize = 14
figsize = (10, 9)  # demagnified overall — change to make plot larger/smaller
top_margin = 0.92  # controls space for title
right_margin = 0.88

# colormap (your custom)
colors = ["#ffff66", "#b2ff66", "#66ff66", "#66b2b2", "#000066"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
eps = 1e-6


# -------------------------------
# Helper functions
# -------------------------------
def read_window(src_path, bbox):
    with rasterio.open(src_path) as src:
        window = from_bounds(
            bbox["lon_min"],
            bbox["lat_min"],
            bbox["lon_max"],
            bbox["lat_max"],
            src.transform,
        )
        data = src.read(1, window=window)
        transform = src.window_transform(window)
    return data, transform, window


def compute_threshold(data, percentile):
    mask_pos = data > 0
    if not np.any(mask_pos):
        raise ValueError("No positive population pixels found in the window.")
    vals = data[mask_pos]
    return np.percentile(vals, percentile)


def clean_mask(binary_mask, radius=2):
    if radius <= 0:
        return binary_mask
    struct = ndi.generate_binary_structure(2, 1)
    for _ in range(radius - 1):
        struct = ndi.binary_dilation(struct)
    cleaned = ndi.binary_opening(binary_mask, structure=struct)
    cleaned = ndi.binary_closing(cleaned, structure=struct)
    return cleaned


def shapes_from_mask(mask, transform):
    yield from shapes(mask.astype(np.uint8), mask=mask, transform=transform)


def bbox_from_geom(geom):
    minx, miny, maxx, maxy = geom.bounds
    return (minx, maxx, miny, maxy)  # lon_min, lon_max, lat_min, lat_max


def create_ticks_for_window(data_shape, bbox, n_ticks=5):
    """Return xticks (pixel positions) and formatted xlabels (lon) and same for y (lat).
    data_shape : (height, width)
    bbox : dict with lon_min, lon_max, lat_min, lat_max
    """
    height, width = data_shape
    xticks = np.linspace(0, width - 1, n_ticks)
    yticks = np.linspace(0, height - 1, n_ticks)
    xlabels = [
        f"{bbox['lon_min'] + (bbox['lon_max'] - bbox['lon_min']) * (x / (width - 1)):.2f}°E"
        for x in xticks
    ]
    # note: y is top-to-bottom in image; label should show decreasing lat from top
    ylabels = [
        f"{bbox['lat_max'] - (bbox['lat_max'] - bbox['lat_min']) * (y / (height - 1)):.2f}°N"
        for y in yticks
    ]
    return xticks, xlabels, yticks, ylabels


# -------------------------------
# Main detection + plotting
# -------------------------------
def find_urban_bboxes_in_window(
    src_path, bbox, percentile=90, min_area_pixels=500, morph_open_radius=2, top_n=8
):
    data, transform, window = read_window(src_path, bbox)
    data_pos = np.ma.masked_where(data <= 0, data)
    print("Window linear stats (positive pixels only):")
    print(
        f"Min: {data_pos.min():.4f} people/pixel  |  Max: {data_pos.max():.4f} people/pixel"
    )

    thr = compute_threshold(data, percentile)
    print(f"Using {percentile}th percentile as urban threshold: {thr:.6f} people/pixel")

    urban_mask = (data > 0) & (data >= thr)
    urban_mask_clean = clean_mask(urban_mask, radius=morph_open_radius)

    labeled, ncomp = ndi.label(urban_mask_clean)
    print(f"Found {ncomp} connected components in cleaned urban mask.")

    # label counts map for pixel-accurate counts
    if ncomp > 0:
        label_ids, counts = np.unique(labeled[labeled > 0], return_counts=True)
        label_count_map = dict(zip(label_ids.tolist(), counts.tolist()))
    else:
        label_count_map = {}

    candidates = []
    for geom_geojson, value in shapes_from_mask(urban_mask_clean, transform):
        geom = shape(geom_geojson)
        lon_min, lon_max, lat_min, lat_max = bbox_from_geom(geom)
        # convert lon/lat bbox to pixel indices relative to the window transform
        inv_transform = ~transform
        col0, row0 = inv_transform * (lon_min, lat_max)  # top-left
        col1, row1 = inv_transform * (lon_max, lat_min)  # bottom-right

        # clip and slice
        c0 = int(max(0, min(data.shape[1] - 1, np.floor(min(col0, col1)))))
        c1 = int(max(0, min(data.shape[1] - 1, np.ceil(max(col0, col1)))))
        r0 = int(max(0, min(data.shape[0] - 1, np.floor(min(row0, row1)))))
        r1 = int(max(0, min(data.shape[0] - 1, np.ceil(max(row0, row1)))))

        pixel_count = 0
        if r1 >= r0 and c1 >= c0:
            sub = labeled[r0 : r1 + 1, c0 : c1 + 1]
            labels_in_sub, cnts = np.unique(sub[sub > 0], return_counts=True)
            if labels_in_sub.size > 0:
                best_label = labels_in_sub[np.argmax(cnts)]
                pixel_count = int(label_count_map.get(int(best_label), int(cnts.max())))
        candidates.append(
            {
                "geom": geom,
                "bounds": (lon_min, lon_max, lat_min, lat_max),
                "pixel_count": pixel_count,
            }
        )

    retained = [c for c in candidates if c["pixel_count"] >= min_area_pixels]
    print(
        f"Retained {len(retained)} components after filtering by min_area_pixels={min_area_pixels}."
    )

    retained.sort(key=lambda x: x["pixel_count"], reverse=True)
    selected = retained[:top_n]
    bboxes = [
        (
            r["bounds"][0],
            r["bounds"][1],
            r["bounds"][2],
            r["bounds"][3],
            r["pixel_count"],
        )
        for r in selected
    ]

    return {
        "bboxes": bboxes,
        "threshold": thr,
        "urban_mask_clean": urban_mask_clean,
        "data_window": data,
        "transform": transform,
    }


# -------------------------------
# Run and render demagnified figure
# -------------------------------
if __name__ == "__main__":
    res = find_urban_bboxes_in_window(
        file_path,
        downloaded_bbox,
        percentile=percentile_for_urban,
        min_area_pixels=min_area_pixels,
        morph_open_radius=morph_open_radius,
        top_n=top_n,
    )

    bboxes = res["bboxes"]
    threshold_used = res["threshold"]
    data_window = res["data_window"]
    transform = res["transform"]
    mask = res["urban_mask_clean"]

    # prepare log image safely (silence invalid log warnings by avoiding zeros)
    safe = np.where(data_window > 0, data_window, eps)
    data_log = np.log10(safe)
    # mask zeros so colorbar and plot ignore them
    data_log_masked = np.ma.masked_where(data_window <= 0, data_log)

    # Create ticks and labels using data shape + provided bbox
    data_shape = data_window.shape  # (height, width)
    xticks, xlabels, yticks, ylabels = create_ticks_for_window(
        data_shape, downloaded_bbox, n_ticks=5
    )

    # Plot demagnified full window with title and colorbar visible
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    img = ax.imshow(data_log_masked, cmap=cmap, origin="upper")
    title = f"High-Density Urban Areas Identified in the Mumbai Region (2020)"
    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        fontfamily=font_family,
        y=1.03,
    )

    # set proper ticks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xlabels,
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        fontfamily=font_family,
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        ylabels,
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        fontfamily=font_family,
    )

    ax.set_xlabel(
        "Longitude →",
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        fontfamily=font_family,
        labelpad=8,
    )
    ax.set_ylabel(
        "Latitude →",
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        fontfamily=font_family,
        labelpad=8,
    )

    # Overlay bounding boxes (convert lon/lat -> pixel coords using inverse transform)
    inv_transform = ~transform
    for lon_min, lon_max, lat_min, lat_max, pix_count in bboxes:
        col0, row0 = inv_transform * (lon_min, lat_max)
        col1, row1 = inv_transform * (lon_max, lat_min)
        x = min(col0, col1)
        y = min(row0, row1)
        width = abs(col1 - col0)
        height = abs(row1 - row0)
        rect = Rectangle(
            (x, y), width, height, linewidth=2.0, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        # label above box
        ax.text(
            x + width / 2,
            max(0, y - 6),
            f"{pix_count} px",
            color="red",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            family=font_family,
        )

    # colorbar to the right and adjust layout so title is visible
    cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(
        "log10(Population density) [people/pixel]",
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        fontfamily=font_family,
        labelpad=10,
    )

    plt.subplots_adjust(top=top_margin, right=right_margin, left=0.08, bottom=0.08)
    plt.show()
