import folium
import webbrowser
import numpy as np

# Mumbai extreme endpoints in decimal degrees
lat_max = 19.27   # northernmost
lat_min = 18.89215   # southernmost
lon_max = 72.98   # easternmost
lon_min = 72.776   # westernmost

# Create 11x11 grid
latitudes = np.linspace(lat_max, lat_min, 11)  # descending
longitudes = np.linspace(lon_min, lon_max, 11)

# Bounding box
north, south = lat_max, lat_min
west, east = lon_min, lon_max

# Center point
center_lat = (north + south) / 2
center_lon = (west + east) / 2

# Create map
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True)

# Draw bounding rectangle
folium.Rectangle(
    bounds=[[south, west], [north, east]],
    color="red",
    weight=2,
    fill=True,
    fill_opacity=0.1,
).add_to(m)

# Center marker
folium.Marker(
    location=[center_lat, center_lon],
    popup="Center of Grid",
    icon=folium.Icon(color="blue"),
).add_to(m)

# Plot grid points
for lat in latitudes:
    for lon in longitudes:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="black",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Lat: {lat:.4f}, Lon: {lon:.4f}",
        ).add_to(m)

# Save and open map
map_file = "mumbai_map.html"
m.save(map_file)
webbrowser.open(map_file)
