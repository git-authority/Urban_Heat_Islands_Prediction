"""
Centered plotting version (full script)
- masks nodata properly
- uses area-based ranking (top N by area)
- prints lat/lon bbox for each selected box
- draws half-pixel-corrected rectangles + population-weighted centroids
- uses GridSpec so the image is visually centered while the colorbar sits on the right
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
import matplotlib.gridspec as gridspec

# -------------------------------
# Parameters (same as original)
# -------------------------------
file_path = "../../Dataset/Population_2020.tif"
downloaded_bbox = {"lat_min": 18.6, "lat_max": 19.6, "lon_min": 72.71, "lon_max": 73.71}

percentile_for_urban = 93
min_area_pixels = 500
morph_open_radius = 2
top_n = 3  # original: top 3 by area

# plotting appearance (match your earlier settings)
font_family = "Times New Roman"
label_fontsize = 10
label_fontweight = "bold"
title_fontsize = 14
figsize = (10, 9)

# colormap (your custom)
colors = ["#ffff66", "#b2ff66", "#66ff66", "#66b2b2", "#000066"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
eps = 1e-12

PLOT_MASK_OVERLAY = True
PLOT_CENTROIDS = True


# -------------------------------
# Helpers (unchanged behavior)
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
        nodata = src.nodatavals[0]
    return data, transform, nodata, window


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
    height, width = data_shape
    xticks = np.linspace(0, width - 1, n_ticks)
    yticks = np.linspace(0, height - 1, n_ticks)
    xlabels = [
        f"{bbox['lon_min'] + (bbox['lon_max'] - bbox['lon_min']) * (x / (width - 1)):.2f}°E"
        for x in xticks
    ]
    ylabels = [
        f"{bbox['lat_max'] - (bbox['lat_max'] - bbox['lat_min']) * (y / (height - 1)):.2f}°N"
        for y in yticks
    ]
    return xticks, xlabels, yticks, ylabels


# -------------------------------
# Main (area-based selection, centered plotting)
# -------------------------------
if __name__ == "__main__":
    data, transform, nodata, window = read_window(file_path, downloaded_bbox)

    # Mask nodata explicitly
    if nodata is not None:
        nodata_mask = data == nodata
    else:
        nodata_mask = ~np.isfinite(data)
    valid_mask = (~nodata_mask) & np.isfinite(data)
    pos_mask = valid_mask & (data > 0)

    # Prepare people-per-pixel array for internal use (mask nodata & zeros)
    data_pp = np.where(pos_mask, data, 0.0)

    # compute threshold on positive values only (people-per-pixel)
    pos_vals = data_pp[data_pp > 0]
    if pos_vals.size == 0:
        raise RuntimeError("No positive population pixels found in window.")
    thr = np.percentile(pos_vals, percentile_for_urban)
    print(
        f"Using {percentile_for_urban}th percentile as urban threshold: {thr:.6f} people/pixel"
    )

    urban_mask = (data_pp > 0) & (data_pp >= thr)
    urban_mask_clean = clean_mask(urban_mask, radius=morph_open_radius)

    labeled, ncomp = ndi.label(urban_mask_clean)
    print(f"Found {ncomp} connected components in cleaned urban mask.")

    label_ids, counts = ([], [])
    if ncomp > 0:
        label_ids, counts = np.unique(labeled[labeled > 0], return_counts=True)
        label_count_map = dict(zip(label_ids.tolist(), counts.tolist()))
    else:
        label_count_map = {}

    # filter by min_area_pixels and sort by area (this restores original selection)
    filtered_labels = [
        (lbl, cnt) for lbl, cnt in label_count_map.items() if cnt >= min_area_pixels
    ]
    filtered_labels.sort(key=lambda x: x[1], reverse=True)

    print(
        f"{len(filtered_labels)} labels remain after applying min_area_pixels={min_area_pixels}."
    )
    top_labels = filtered_labels[:top_n]

    # compute full candidate entries (area-based)
    candidates = []
    for lbl, cnt in top_labels:
        single_mask = labeled == lbl
        # compute total population inside component (helpful for debug/annotation)
        total_pop = float(np.sum(data_pp[single_mask]))
        # geometry as before
        geoms = list(shapes_from_mask(single_mask, transform))
        if not geoms:
            continue
        geom_json, _ = geoms[0]
        geom_obj = shape(geom_json)
        lon_min, lon_max, lat_min, lat_max = bbox_from_geom(geom_obj)
        candidates.append(
            {
                "label": int(lbl),
                "area_px": int(cnt),
                "sum_pop": total_pop,
                "bounds": (lon_min, lon_max, lat_min, lat_max),
            }
        )

    # Prepare masked log10 image (same visual as original)
    safe = np.where(data_pp > 0, data_pp, eps)
    data_log = np.log10(safe)
    data_log_masked = np.ma.masked_where(data_pp <= 0, data_log)

    # ticks/labels
    data_shape = data_pp.shape
    xticks, xlabels, yticks, ylabels = create_ticks_for_window(
        data_shape, downloaded_bbox, n_ticks=5
    )

    # --- Centered layout using GridSpec: left cell=image, right cell=colorbar ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[40, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])  # main image axis
    cax = fig.add_subplot(gs[0, 1])  # colorbar axis

    # show image
    img = ax.imshow(data_log_masked, cmap=cmap, origin="upper", aspect="auto")

    # decorate axes (ticks, labels, title)
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

    ax.set_title(
        "High-Density Urban Areas Identified in the Mumbai Region (2020)",
        fontsize=title_fontsize,
        fontweight="bold",
        fontfamily=font_family,
        y=1.02,
    )
    ax.set_xlabel(
        "Longitude →",
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        family=font_family,
    )
    ax.set_ylabel(
        "Latitude →",
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        family=font_family,
    )

    # optional overlay of the cleaned mask
    if PLOT_MASK_OVERLAY:
        ax.imshow(
            np.where(urban_mask_clean, 1.0, np.nan),
            cmap="gray",
            origin="upper",
            alpha=0.25,
            interpolation="nearest",
        )

    inv_transform = ~transform
    for c in candidates:
        lon_min, lon_max, lat_min, lat_max = c["bounds"]
        col_tl, row_tl = inv_transform * (lon_min, lat_max)
        col_br, row_br = inv_transform * (lon_max, lat_min)
        x_center = min(col_tl, col_br)
        y_center = min(row_tl, row_br)
        width = abs(col_br - col_tl)
        height = abs(row_br - row_tl)

        x_edge = x_center - 0.5
        y_edge = y_center - 0.5

        rect = Rectangle(
            (x_edge, y_edge),
            width,
            height,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # annotate with pixel-count (same as original)
        ax.text(
            x_edge + width / 2,
            max(0, y_edge - 6),
            f"{c['area_px']} px (lbl {c['label']})",
            color="red",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            family=font_family,
        )

        # print area, total population and lat/lon bounding box
        lon_min_f, lon_max_f, lat_min_f, lat_max_f = lon_min, lon_max, lat_min, lat_max
        print(
            f"Label {c['label']}: area={c['area_px']} px, sum_pop={c['sum_pop']:.1f}, "
            f"lon_min={lon_min_f:.6f}, lon_max={lon_max_f:.6f}, lat_min={lat_min_f:.6f}, lat_max={lat_max_f:.6f}"
        )

        # centroid marker (population-weighted)
        mask_comp = labeled == c["label"]
        r, cc = np.where(mask_comp)
        vals = data_pp[mask_comp]
        if vals.size:
            wcol = np.sum(cc.astype(float) * vals) / np.sum(vals)
            wrow = np.sum(r.astype(float) * vals) / np.sum(vals)
            ax.plot(wcol, wrow, "x", color="red", markersize=6)
        elif PLOT_CENTROIDS:
            lon_c = 0.5 * (lon_min + lon_max)
            lat_c = 0.5 * (lat_min + lat_max)
            col_c, row_c = inv_transform * (lon_c, lat_c)
            ax.plot(col_c, row_c, "x", color="red", markersize=6)

    # colorbar in its dedicated axis (so it doesn't push the image)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(
        "log10(People per pixel)",
        fontsize=label_fontsize,
        fontweight=label_fontweight,
        family=font_family,
    )

    # tidy up and show
    plt.tight_layout()
    plt.show()
