"""
demo_visualization.py

Member 1 — Satellite Data & Geospatial

Builds the presentation-round demo visual:
  - Real Sentinel-1 SAR image (downsampled for display — real scenes are
    huge, e.g. 25526x15203 pixels)
  - A MOCK spill polygon overlaid (placeholder until Member 2's real
    detect_spill() model exists)
  - MOCK vessel markers near the spill (placeholder until Member 4's
    real AIS matching exists)

Produces two outputs:
  1. demo_static.png   — static image for slides (matplotlib)
  2. demo_map.html      — interactive web map for a live demo (folium)

Both mock elements are clearly labeled as mock in the output, so it's
obvious to judges/teammates that this is illustrative, not a real
detection result.
"""

import numpy as np
import rasterio
import folium
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy import ndimage

from services.geo_service import extract_image_metadata, mask_to_polygons


# ---------------------------------------------------------------------
# Config — edit these for your actual file
# ---------------------------------------------------------------------
IMAGE_PATH = "s1d-iw-grd-vh-20260626t010259-20260626t010322-003400-005fc3-002.tiff"   # <-- swap this for your real Sentinel-1 .tiff
DOWNSAMPLE_FACTOR = 20      # real S1 scenes are huge; shrink for display


def load_downsampled_image(path: str, factor: int):
    """Reads a downsampled overview of a large raster for fast display."""
    with rasterio.open(path) as src:
        out_shape = (src.count, src.height // factor, src.width // factor)
        data = src.read(out_shape=out_shape)
        # Use band 1 for display (SAR is often single-band; if multi-band,
        # this just shows the first band, which is fine for a demo)
        band = data[0].astype("float32")
        # Simple contrast stretch so SAR imagery is actually visible
        p2, p98 = np.percentile(band, (2, 98))
        band = np.clip((band - p2) / (p98 - p2 + 1e-9), 0, 1)
        return band, src.height, src.width


def find_water_center(band: np.ndarray, blob_radius_frac: float = 0.05):
    """
    Finds a point that's safely inside a water region, using the SAR
    image itself: calm water has low backscatter (dark pixels), while
    land/ships/waves are brighter and noisier. This avoids placing the
    mock spill blob on land.

    Returns (row, col) in DOWNSAMPLED pixel coordinates, or None if no
    suitable water region is found.
    """
    # Water = darker than average. Threshold picked from the image's own
    # distribution rather than a fixed number, so it adapts per-scene.
    water_thresh = np.percentile(band, 35)
    water_mask = band < water_thresh

    # Keep only the largest connected water region — avoids scattered
    # dark noise pixels being mistaken for a usable water area.
    labeled, num_features = ndimage.label(water_mask)
    if num_features == 0:
        return None
    sizes = ndimage.sum(water_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    largest_region = labeled == largest_label

    # Erode the region inward so the blob (once drawn) can't reach the
    # region's edge and bleed onto land.
    blob_radius_px = max(int(band.shape[0] * blob_radius_frac), 3)
    eroded = ndimage.binary_erosion(largest_region, iterations=blob_radius_px)

    if not eroded.any():
        # Region too small/thin for this blob size — fall back to the
        # un-eroded region's centroid instead of failing outright.
        rows, cols = np.where(largest_region)
    else:
        rows, cols = np.where(eroded)

    # Pick a random valid point (fixed seed for a reproducible demo)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(rows))
    return rows[idx], cols[idx]


def build_mock_mask(full_height: int, full_width: int, band: np.ndarray, downsample_factor: int) -> np.ndarray:
    """
    Creates a fake spill-shaped blob, placed at a point verified to be
    over water (using find_water_center), in full-resolution pixel
    space (matches mask_to_polygons' expectations).
    Replace this with Member 2's real mask once available.
    """
    water_point = find_water_center(band)
    if water_point is None:
        # No clear water region detected — fall back to image center
        # and warn, rather than silently producing a bad demo image.
        print("WARNING: couldn't confidently detect a water region — "
              "falling back to image center. Check the output image.")
        cy, cx = full_height // 2, full_width // 2
    else:
        row_ds, col_ds = water_point
        cy, cx = row_ds * downsample_factor, col_ds * downsample_factor

    mask = np.zeros((full_height, full_width), dtype="uint8")
    yy, xx = np.ogrid[:full_height, :full_width]
    blob = ((yy - cy) ** 2) / (full_height * 0.025) ** 2 + \
           ((xx - cx) ** 2) / (full_width * 0.04) ** 2 <= 1
    mask[blob] = 1
    return mask


def build_mock_vessels(spill_centroid_latlon: list, n: int = 3) -> list:
    """
    Places n mock vessels at small random offsets from the spill
    centroid, with fake MMSI/speed/heading, standing in for Member 4's
    real AIS matching output.
    """
    lat0, lon0 = spill_centroid_latlon
    rng = np.random.default_rng(42)  # fixed seed = reproducible demo
    vessels = []
    for i in range(n):
        offset_lat = rng.uniform(-0.08, 0.08)
        offset_lon = rng.uniform(-0.08, 0.08)
        vessels.append({
            "mmsi": f"MOCK-{100000 + i}",
            "latitude": lat0 + offset_lat,
            "longitude": lon0 + offset_lon,
            "distance_km": round(np.hypot(offset_lat, offset_lon) * 111, 1),  # rough deg->km
        })
    # Sort nearest-first — the nearest is your "top suspect" for demo purposes
    vessels.sort(key=lambda v: v["distance_km"])
    return vessels


def render_static(image_band, polygons, vessels, bbox, out_path="demo_static.png"):
    """Static matplotlib figure for slide decks."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_band, cmap="gray", extent=[
        bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"]
    ])

    for p in polygons:
        coords = [(lon, lat) for lat, lon in p["polygon_latlon"]]
        patch = MplPolygon(coords, closed=True, facecolor="red", edgecolor="darkred",
                            alpha=0.4, linewidth=2, label="Detected spill (MOCK)")
        ax.add_patch(patch)
        ax.plot(p["centroid_latlon"][1], p["centroid_latlon"][0], "r+", markersize=12)

    for v in vessels:
        ax.plot(v["longitude"], v["latitude"], "^", color="cyan", markersize=10,
                 markeredgecolor="black")
        ax.annotate(f"{v['mmsi']}\n{v['distance_km']} km", (v["longitude"], v["latitude"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="cyan")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Oil Spill Investigation — Demo (MOCK spill + MOCK vessels)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved static demo image -> {out_path}")


def render_interactive(polygons, vessels, bbox, out_path="demo_map.html"):
    """Interactive folium map for a live click-through demo."""
    center_lat = (bbox["lat_min"] + bbox["lat_max"]) / 2
    center_lon = (bbox["lon_min"] + bbox["lon_max"]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

    folium.Rectangle(
        bounds=[[bbox["lat_min"], bbox["lon_min"]], [bbox["lat_max"], bbox["lon_max"]]],
        color="gray", weight=1, fill=False,
        popup="SAR scene footprint",
    ).add_to(m)

    for p in polygons:
        folium.Polygon(
            locations=p["polygon_latlon"],
            color="darkred", fill=True, fill_color="red", fill_opacity=0.4,
            popup="Detected spill (MOCK — placeholder for Member 2's model)",
        ).add_to(m)

    for v in vessels:
        folium.Marker(
            location=[v["latitude"], v["longitude"]],
            popup=f"{v['mmsi']} — {v['distance_km']} km from spill (MOCK)",
            icon=folium.Icon(color="blue", icon="ship", prefix="fa"),
        ).add_to(m)

    m.save(out_path)
    print(f"Saved interactive demo map -> {out_path}")


if __name__ == "__main__":
    print(f"Loading {IMAGE_PATH} ...")
    band, full_h, full_w = load_downsampled_image(IMAGE_PATH, DOWNSAMPLE_FACTOR)

    meta = extract_image_metadata(IMAGE_PATH)
    bbox = meta["bounding_box"]
    print(f"Bounding box: {bbox}")
    print(f"Acquisition time: {meta['acquisition_time']}")

    mock_mask = build_mock_mask(full_h, full_w, band, DOWNSAMPLE_FACTOR)
    polygons = mask_to_polygons(mock_mask, IMAGE_PATH)
    print(f"Mock spill polygon centroid: {polygons[0]['centroid_latlon']}")

    vessels = build_mock_vessels(polygons[0]["centroid_latlon"], n=3)
    print(f"Mock vessels: {vessels}")

    render_static(band, polygons, vessels, bbox)
    render_interactive(polygons, vessels, bbox)
