import folium
import numpy as np
import webbrowser

# -------- Bounding Box 1: Downloaded Dataset (1° x 1°) --------
lat_min_ds, lat_max_ds = 18.6, 19.6
lon_min_ds, lon_max_ds = 72.71, 73.71

# -------- Bounding Box 2: Mumbai Extremes --------
lat_max_mum = 19.27
lat_min_mum = 18.89215
lon_max_mum = 72.98
lon_min_mum = 72.776

# Create grids (11x11 each)
latitudes_ds = np.linspace(lat_max_ds, lat_min_ds, 11)
longitudes_ds = np.linspace(lon_min_ds, lon_max_ds, 11)

latitudes_mum = np.linspace(lat_max_mum, lat_min_mum, 11)
longitudes_mum = np.linspace(lon_min_mum, lon_max_mum, 11)

# Center of overall map
center_lat = (max(lat_max_ds, lat_max_mum) + min(lat_min_ds, lat_min_mum)) / 2
center_lon = (max(lon_max_ds, lon_max_mum) + min(lon_min_ds, lon_min_mum)) / 2

# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)

# Draw rectangles for both bounding boxes
folium.Rectangle(
    bounds=[[lat_min_ds, lon_min_ds], [lat_max_ds, lon_max_ds]],
    color="red",
    weight=2,
    fill=False,
    tooltip="Dataset Grid (1°x1°)"
).add_to(m)

folium.Rectangle(
    bounds=[[lat_min_mum, lon_min_mum], [lat_max_mum, lon_max_mum]],
    color="blue",
    weight=2,
    fill=False,
    tooltip="Mumbai Grid (extremes)"
).add_to(m)

# Plot dataset grid points (red)
for lat in latitudes_ds:
    for lon in longitudes_ds:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

# Plot Mumbai grid points (blue)
for lat in latitudes_mum:
    for lon in longitudes_mum:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

# Save and open
map_file = "mumbai_comparison.html"
m.save(map_file)
webbrowser.open(map_file)
